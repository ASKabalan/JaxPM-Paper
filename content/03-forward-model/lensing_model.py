import jax
import jax.numpy as jnp
import jax_cosmo as jc

from diffrax import (
    SaveAt,
    ODETerm,
    diffeqsolve,
    RecursiveCheckpointAdjoint,
)

import numpyro
import numpyro.distributions as dist

from jax_cosmo.scipy.integrate import simps
from jax.scipy.ndimage import map_coordinates
import jax_cosmo.constants as constants

from jaxpm.pm import pm_forces, growth_factor, growth_rate
from jaxpm.kernels import fftk
from jaxpm.painting import cic_paint_2d
from jaxpm.utils import gaussian_smoothing
from jaxpm.distributed import fft3d, ifft3d, uniform_particles
import sys
import os
from typing import NamedTuple

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

from tools.integrate import integrate as reverse_adjoint_integrate  # noqa : E402
from tools.semi_implicite_euler import SemiImplicitEuler  # noqa : E402
from tools.ode import symplectic_fpm_ode  # noqa : E402


def convergence_Born(cosmo, density_planes, r, a, dx, dz, coords, z_source):
    """
    Compute the Born convergence
    Args:
      cosmo: `Cosmology`, cosmology object.
      density_planes: list of dictionaries (r, a, density_plane, dx, dz), lens planes to use
      coords: a 3-D array of angular coordinates in radians of N points with shape [batch, N, 2].
      z_source: 1-D `Tensor` of source redshifts with shape [Nz] .
      name: `string`, name of the operation.
    Returns:
      `Tensor` of shape [batch_size, N, Nz], of convergence values.
    """
    # Compute constant prefactor:
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c) ** 2
    # Compute comoving distance of source galaxies
    r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

    convergence = 0
    n_planes = len(r)

    def scan_fn(carry, i):
        density_planes, a, r = carry

        p = density_planes[:, :, i]
        density_normalization = dz * r[i] / a[i]
        p = (p - p.mean()) * constant_factor * density_normalization

        # Interpolate at the density plane coordinates
        im = map_coordinates(p, coords * r[i] / dx - 0.5, order=1, mode="wrap")

        return carry, im * jnp.clip(1.0 - (r[i] / r_s), 0, 1000).reshape([-1, 1, 1])

    # Similar to for loops but using a jaxified approach
    _, convergence = jax.lax.scan(scan_fn, (density_planes, a, r), jnp.arange(n_planes))

    return convergence.sum(axis=0)


# field = normal_field(mesh_shape, seed=seed, sharding=sharding)
def E(cosmo, a):
    return jnp.sqrt(jc.background.Esqr(cosmo, a))


def linear_field(mesh_shape, box_size, pk, field):
    """
    Generate initial conditions.
    """
    # Initialize a random field with one slice on each gpu
    field = fft3d(field)
    kvec = fftk(field)
    kmesh = (
        sum((kk / box_size[i] * mesh_shape[i]) ** 2 for i, kk in enumerate(kvec)) ** 0.5
    )
    pkmesh = (
        pk(kmesh)
        * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2])
        / (box_size[0] * box_size[1] * box_size[2])
    )

    field = field * (pkmesh) ** 0.5
    field = ifft3d(field)
    return field


def lpt_lightcone(cosmo, initial_conditions, a, mesh_shape):
    """Computes first order LPT displacement"""
    particles = jax.tree.map(
        lambda ic: jnp.zeros_like(ic, shape=(*ic.shape, 3)), initial_conditions
    )

    initial_force = pm_forces(
        particles, delta=initial_conditions, paint_absolute_pos=False
    )
    a = jnp.atleast_1d(a)
    dx = growth_factor(cosmo, a).reshape([1, 1, -1, 1]) * initial_force
    p = (a**2 * growth_rate(cosmo, a) * E(cosmo, a) * growth_factor(cosmo, a)).reshape(
        [1, 1, -1, 1]
    ) * initial_force
    return dx, p


def integrate(terms, solver, t0, t1, dt0, y0, args, saveat, adjoint):
    if isinstance(adjoint, RecursiveCheckpointAdjoint):
        solution =  diffeqsolve(
            terms, solver, t0, t1, dt0, y0, args, saveat=saveat, adjoint=adjoint
        )
        return solution.ys , saveat.subs.ts
    else:
        solution =  reverse_adjoint_integrate(terms, solver, t0, t1, dt0, y0, args, saveat)
        return solution, saveat.subs.ts


def make_full_field_model(
    field_size,
    field_npix,
    box_shape,
    box_size,
    density_plane_width=None,
    density_plane_npix=None,
    density_plane_smoothing=None,
    adjoint=RecursiveCheckpointAdjoint(5),
):
    def density_plane_fn(t, y, args):
        cosmo, = args
        positions = y[0]
        nx, ny, nz = box_shape

        # Converts time t to comoving distance in voxel coordinates
        w = density_plane_width / box_size[2] * box_shape[2]
        center = (
            jc.background.radial_comoving_distance(cosmo, t)
            / box_size[2]
            * box_shape[2]
        )
        positions = uniform_particles(box_shape) + positions
        xy = positions[..., :2]
        d = positions[..., 2]

        # Apply 2d periodic conditions
        xy = jax.tree.map(lambda xy: jnp.mod(xy, nx), xy)

        # Rescaling positions to target grid
        xy = xy / nx * density_plane_npix
        # Selecting only particles that fall inside the volume of interest
        weight = jax.tree.map(
            lambda x: jnp.where(
                (d > (center - w / 2)) & (d <= (center + w / 2)), 1.0, 0.0
            ),
            d,
        )
        # Painting density plane
        zero_mesh = jax.tree.map(
            lambda _: jnp.zeros([density_plane_npix, density_plane_npix]), xy
        )
        density_plane = cic_paint_2d(zero_mesh, xy, weight)

        # Apply density normalization
        density_plane = density_plane / (
            (nx / density_plane_npix) * (ny / density_plane_npix) * w
        )
        return density_plane

    def forward_model(cosmo, nz_shear, initial_conditions):
        # Create a small function to generate the matter power spectrum
        k = jnp.logspace(-4, 1, 128)
        pk = jc.power.linear_matter_power(cosmo, k)

        def pk_fn(x):
            return jax.tree.map(
                lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(
                    x.shape
                ),
                x,
            )

        # Create initial conditions
        lin_field = linear_field(box_shape, box_size, pk_fn, initial_conditions)

        cosmo = jc.Cosmology(
            Omega_c=cosmo.Omega_c,
            sigma8=cosmo.sigma8,
            Omega_b=cosmo.Omega_b,
            h=cosmo.h,
            n_s=cosmo.n_s,
            w0=cosmo.w0,
            Omega_k=0.0,
            wa=0.0,
        )
        # Temporary fix
        cosmo._workspace = {}

        # Initial displacement

        assert density_plane_width is not None
        assert density_plane_npix is not None

        density_plane_smoothing = 0.1
        drift, kick, first_kick = symplectic_fpm_ode(
            box_shape, dt0=0.05, paint_absolute_pos=False
        )
        first_term = ODETerm(first_kick)
        ode_terms = ODETerm(drift), ODETerm(kick)

        a_init = 0.01
        n_lens = int(box_size[-1] // density_plane_width)
        r = jnp.linspace(0.0, box_size[-1], n_lens + 1)
        r_center = 0.5 * (r[1:] + r[:-1])
        a_center = jc.background.a_of_chi(cosmo, r_center)

        eps, p = lpt_lightcone(cosmo, lin_field, a_init, box_shape)
        solver = SemiImplicitEuler()
        saveat = SaveAt(ts=a_center[::-1], fn=density_plane_fn)
        y0 = (eps, p)
        args = cosmo,

        y0 = solver.first_step(first_term, 0.01, dt0=0.05, y0=y0, args=args)

        solution , ts = integrate(
            ode_terms,
            solver,
            t0=0.01,
            t1=1.0,
            dt0=0.05,
            y0=y0,
            args=args,
            saveat=saveat,
            adjoint=adjoint,
        )

        dx = box_size[0] / density_plane_npix
        dz = density_plane_width

        lightcone = jax.vmap(
            lambda x: gaussian_smoothing(x, density_plane_smoothing / dx)
        )(solution)
        lightcone = lightcone[::-1]
        a = ts[::-1]
        lightcone = jax.tree.map(
            lambda lc: jnp.transpose(lc, axes=(1, 2, 0)), lightcone
        )

        # Defining the coordinate grid for lensing map
        xgrid, ygrid = jnp.meshgrid(
            jnp.linspace(
                0, field_size, box_shape[0], endpoint=False
            ),  # range of X coordinates
            jnp.linspace(0, field_size, box_shape[1], endpoint=False),
        )  # range of Y coordinates

        # coords       = jnp.array((jnp.stack([xgrid, ygrid], axis=0)*u.deg).to(u.rad))
        coords = jnp.array(
            (jnp.stack([xgrid, ygrid], axis=0)) * 0.017453292519943295
        )  # deg->rad

        # Generate convergence maps by integrating over nz and source planes
        convergence_maps = [
            simps(
                lambda z: nz(z).reshape([-1, 1, 1])
                * convergence_Born(
                    cosmo, lightcone.data, r_center, a, dx, dz, coords, z
                ),
                0.01,
                3.0,
                N=32,
            )
            for nz in nz_shear
        ]

        # Reshape the maps to desired resoluton
        convergence_maps = [
            kmap.reshape(
                [
                    field_npix,
                    box_shape[0] // field_npix,
                    field_npix,
                    box_shape[1] // field_npix,
                ]
            )
            .mean(axis=1)
            .mean(axis=-1)
            for kmap in convergence_maps
        ]

        return convergence_maps, lightcone

    return forward_model


class LensingConfig(NamedTuple):
    field_size: float
    field_npix: int
    box_shape: tuple
    box_size: tuple
    nz_shear: list
    sigma_e: float
    priors: dict
    fiducial_cosmology: jc.Cosmology

# Build the probabilistic model
def full_field_probmodel(config):
    forward_model = make_full_field_model(
        config.field_size, config.field_npix, config.box_shape, config.box_size
    )

    # Sampling the cosmological parameters
    cosmo = config.fiducial_cosmology(
        **{k: numpyro.sample(k, v) for k, v in config.priors.items()}
    )

    # Sampling the initial conditions
    initial_conditions = numpyro.sample(
        "initial_conditions",
        dist.Normal(jnp.zeros(config.box_shape), jnp.ones(config.box_shape)),
    )

    # Apply the forward model
    convergence_maps, _ = forward_model(cosmo, config.nz_shear, initial_conditions)

    # Define the likelihood of observations
    observed_maps = [
        numpyro.sample(
            "kappa_%d" % i,
            dist.Normal(
                k,
                config.sigma_e
                / jnp.sqrt(
                    config.nz_shear[i].gals_per_arcmin2
                    * (config.field_size * 60 / config.field_npix) ** 2
                ),
            ),
        )
        for i, k in enumerate(convergence_maps)
    ]

    return observed_maps


def pixel_window_function(L, pixel_size_arcmin):
    """
    Calculate the pixel window function W_l for a given angular wave number l and pixel size.

    Parameters:
    - L: Angular wave number (can be a numpy array or a single value).
    - pixel_size_arcmin: Pixel size in arcminutes.

    Returns:
    - W_l: Pixel window function for the given L and pixel size.
    """
    # Convert pixel size from arcminutes to radians
    pixel_size_rad = pixel_size_arcmin * (jnp.pi / (180.0 * 60.0))

    # Calculate the Fourier transform of the square pixel (sinc function)
    # Note: l should be the magnitude of the angular wave number vector, |l| = sqrt(lx^2 + ly^2) for a general l
    # For simplicity, we assume l is already provided as |l|
    W_l = (jnp.sinc(L * pixel_size_rad / (2 * jnp.pi))) ** 2

    return W_l


def make_2pt_model(pixel_scale, ell, sigma_e=0.3):
    """
    Create a function that computes the theoretical 2-point correlation function for a given cosmology and redshift distribution.

    Parameters:
    - pixel_scale: Pixel scale in arcminutes.
    - ell: Angular wave number (numpy array).

    Returns:
    - forward_model: Function that computes the theoretical 2-point correlation function for a given cosmology and redshift distribution.
    """

    def forward_model(cosmo, nz_shear):
        tracer = jc.probes.WeakLensing(nz_shear, sigma_e=sigma_e)
        cell_theory = jc.angular_cl.angular_cl(
            cosmo, ell, [tracer], nonlinear_fn=jc.power.linear
        )
        cell_theory = cell_theory * pixel_window_function(ell, pixel_scale)
        cell_noise = jc.angular_cl.noise_cl(ell, [tracer])
        return cell_theory, cell_noise

    return forward_model
