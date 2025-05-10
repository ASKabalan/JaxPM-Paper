import argparse
import os
from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpyro
import numpyro.distributions as dist
from diffrax import RecursiveCheckpointAdjoint
from numpyro.handlers import condition, seed, trace
from numpyro.infer import NUTS
from scipy.stats import norm
from tools.lensing_model import (
    Configurations,
    Planck18,
    full_field_probmodel,
)
from tools.sampling import batched_sampling


def parse_args():
    parser = argparse.ArgumentParser(description="Run lensing full field inference.")
    parser.add_argument("--output", type=str, default="samples", help="Directory to save samples")

    parser.add_argument("--box_shape", nargs=3, type=int, default=[64, 64, 128])
    parser.add_argument("--box_size", nargs=3, type=float, default=[400.0, 400.0, 800.0])
    parser.add_argument("--field_size", type=float, default=16.0)
    parser.add_argument("--field_npix", type=int, default=64)
    parser.add_argument("--density_plane_width", type=float, default=100.0)
    parser.add_argument("--density_plane_npix", type=int, default=64)
    parser.add_argument("--density_plane_smoothing", type=float, default=0.1)

    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    box_shape = tuple(args.box_shape)
    box_size = list(args.box_size)
    field_size = args.field_size
    field_npix = args.field_npix
    density_plane_width = args.density_plane_width
    density_plane_npix = args.density_plane_npix
    density_plane_smoothing = args.density_plane_smoothing
    sigma_e = 0.3
    t0 = 0.1
    t1 = 1.0
    dt0 = 0.1

    print("Pixel size in arcmin: ", field_size * 60 / field_npix)

    # Setup shear redshift bins
    z = jnp.linspace(0, 2.5, 1000)

    nz_shear = [
        jc.redshift.kde_nz(
            z, norm.pdf(z, loc=z_center, scale=0.12), bw=0.01, zmax=2.5, gals_per_arcmin2=g
        )
        for z_center, g in zip([0.5, 1.0, 1.5, 2.0], [7, 8.5, 7.5, 7])
    ]

    # Cosmology
    fiducial_cosmology = Planck18()

    # Configuration
    config = Configurations(
        field_size=field_size,
        field_npix=field_npix,
        box_shape=box_shape,
        box_size=box_size,
        density_plane_width=density_plane_width,
        density_plane_npix=density_plane_npix,
        density_plane_smoothing=density_plane_smoothing,
        nz_shear=nz_shear,
        fiducial_cosmology=Planck18,
        sigma_e=sigma_e,
        priors={
            "Omega_c": dist.Uniform(0.2, 0.4),
            "sigma8": dist.Uniform(0.6, 1.0),
            "h": dist.Uniform(0.5, 0.9),
        },
        t0=t0,
        t1=t1,
        dt0=dt0,
        adjoint=RecursiveCheckpointAdjoint(checkpoints=5),
    )

    # Build forward model and trace fiducial simulation
    full_field_model = full_field_probmodel(config)
    fiducial_model = condition(
        full_field_model,
        {"Omega_c": fiducial_cosmology.Omega_c, "sigma8": fiducial_cosmology.sigma8},
    )

    print("Tracing fiducial simulation...")
    model_trace = trace(seed(fiducial_model, jax.random.PRNGKey(1234))).get_trace()

    observed_model = condition(
        full_field_model,
        {
            "kappa_0": model_trace["kappa_0"]["value"],
            "kappa_1": model_trace["kappa_1"]["value"],
            "kappa_2": model_trace["kappa_2"]["value"],
            "kappa_3": model_trace["kappa_3"]["value"],
        },
    )

    print("Model Traced...")

    init_strategy = (
        partial(
            numpyro.infer.init_to_value,
            values={
                "Omega_c": fiducial_model.Omega_c,
                "sigma8": fiducial_model.sigma8,
                "initial_conditions": model_trace["initial_conditions"]["value"],
            },
        ),
    )

    kernel = NUTS(
        model=observed_model,
        init_strategy=init_strategy,
        max_tree_depth=3,
        step_size=0.05,
    )

    print("Starting MCMC sampling...")
    last_state, mcmc = batched_sampling(
        kernel,
        args.output,
        rng_key=jax.random.key(1234),
        num_chains=1,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        thinning=args.thinning,
    )


if __name__ == "__main__":
    main()
