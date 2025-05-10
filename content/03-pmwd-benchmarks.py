import argparse
import os
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from jax_hpc_profiler import Timer
from pmesh.pm import ParticleMesh
from pmwd import (
    Configuration,
    Cosmology,
    boltzmann,
    lpt,
    nbody,
    scatter,
)

jax.config.update("jax_enable_x64", True)
os.environ["EQX_ON_ERROR"] = "nan"

def parse_args():
    """Parse CLI arguments for PMWD simulation."""
    parser = argparse.ArgumentParser(description="Run PMWD N-body simulation.")
    parser.add_argument(
        "-m",
        "--mesh-sizes",
        type=int,
        nargs="+",
        required=True,
        help="List of mesh sizes (e.g., 64 128 256).",
    )
    parser.add_argument(
        "-b",
        "--box-sizes",
        type=float,
        nargs="+",
        required=True,
        help="List of box sizes (must match mesh sizes).",
    )
    parser.add_argument(
        "-n", "--steps", type=int, default=10, help="Number of time steps (e.g., 10)"
    )

    return parser.parse_args()


class Params(NamedTuple):
    Omega_c: float
    sigma8: float

@jax.jit
def run_nbody(params, ic, conf):
    """Run LPT + N-body and return final density field."""
    cosmo = Cosmology(
        conf,
        A_s_1e9=2.0,
        n_s=0.96,
        Omega_m=params.Omega_c,
        Omega_b=0.05,
        h=0.7,
    )
    # Compute transfer functions
    cosmo = boltzmann(cosmo, conf)

    ic = jnp.fft.rfftn(ic)
    ptcl, obsvbl = lpt(ic, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    print(f"ptcl dtype: {ptcl.pmid.dtype}")
    dens = scatter(ptcl, conf)
    print(f"dtype of dens: {dens.dtype}")
    return dens

def MSE(x, y):
    return jnp.mean((x - y) ** 2)


@jax.jit
def model(params, ic, obs, conf):
    dens = run_nbody(params, ic, conf)
    return MSE(dens, obs)

nbody_ic = jax.jit(jax.grad(model, argnums=2))

def main():
    args = parse_args()

    # Setup time and grid
    t0 = 0.1
    t1 = 1.0
    dt0 = (t1 - t0) / args.steps

    for mesh_size, box_size in zip(args.mesh_sizes, args.box_sizes):

        ptcl_spacing = box_size / mesh_size
        ptcl_grid_shape = (mesh_size,) * 3
        mesh_shape = (mesh_size,) * 3
        box_shape = (box_size,) * 3
        # Create configuration matching target behavior
        conf = Configuration(
            ptcl_spacing=ptcl_spacing,
            ptcl_grid_shape=ptcl_grid_shape,
            mesh_shape=1,
            a_start=t0,
            a_stop=t1,
            a_nbody_maxstep=dt0,
            a_lpt_maxstep=t0,  # LPT integrates 0 → 0.1 in one step
            lpt_order=2,
        )

        print(conf)  # with other default parameters
        print(f'Simulating {conf.ptcl_num} particles with a {conf.mesh_shape} mesh for {conf.a_nbody_num} time steps.')

        params = Params(Omega_c=0.25, sigma8=0.8)
        # Set up cosmology
        cosmo = jc.Planck15(Omega_c=params.Omega_c, sigma8=params.sigma8)
        # Generate initial particle positions
        pm = ParticleMesh(BoxSize=box_shape, Nmesh=mesh_shape, dtype="f8")
        grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float64)
        # Interpolate with linear_matter spectrum to get initial density field
        k = jnp.logspace(-4, 1, 128)
        pk = jc.power.linear_matter_power(cosmo, k)

        whitec = pm.generate_whitenoise(42, type="complex", unitary=False)

        def pk_fn(x):
            return jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

        lineark = whitec.apply(
            lambda k, v: pk_fn(sum(ki**2 for ki in k) ** 0.5) ** 0.5
            * v
            * (1 / v.BoxSize).prod() ** 0.5
        )
        init_mesh = lineark.c2r().value

        # Make Guess IC
        guess_params = Params(Omega_c=0.8, sigma8=0.8)
        guess_cosmo = jc.Planck15(Omega_c=guess_params.Omega_c, sigma8=guess_params.sigma8)
        pk = jc.power.linear_matter_power(guess_cosmo, k)

        def pk_fn_2(x):
            return jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

        lineark = whitec.apply(
            lambda k, v: pk_fn_2(sum(ki**2 for ki in k) ** 0.5) ** 0.5
            * v
            * (1 / v.BoxSize).prod() ** 0.5
        )
        init_mesh = lineark.c2r().value
        guess_ic = jnp.asarray(init_mesh)

        print(f"dtype of guess_ic: {guess_ic.dtype} and initial mesh: {init_mesh.dtype}")


        jax_timer = Timer(save_jaxpr=False, jax_fn=True)
        run_body = partial(run_nbody, conf=conf)
        # FORWARD
        print("Running Forward Pass")
        observable = jax_timer.chrono_jit(run_body, params, init_mesh)
        for _ in range(5):
            observable = jax_timer.chrono_fun(run_body, params, init_mesh)

        data = {"observable": observable}
        print(f"DTYPE: {observable.dtype}")
        kwargs = {
            "function": "Forward",
            "precision": "float64",
            "x": mesh_shape[0],
            "y": mesh_shape[1],
            "z": mesh_shape[2],
            "npz_data": data,
        }
        extra_info = {
            "solver": "PMWD",
            "adjoint": "REVERSE",
            "steps": args.steps,
        }
        jax_timer.report("runs/PMWD.csv", **kwargs, extra_info=extra_info)

        # BACKWARD
        print("Running Backward Pass")
        grads = jax_timer.chrono_jit(nbody_ic, params, guess_ic, observable, conf)
        for _ in range(5):
            grads = jax_timer.chrono_fun(nbody_ic, params, guess_ic, observable, conf)

        data = {"grads": grads}
        kwargs = {
            "function": "Backward",
            "precision": "float64",
            "x": mesh_shape[0],
            "y": mesh_shape[1],
            "z": mesh_shape[2],
            "npz_data": data,
        }
        extra_info = {
            "solver": "PMWD",
            "adjoint": "REVERSE",
            "steps": args.steps,
        }
        jax_timer.report("runs/PMWD.csv", **kwargs, extra_info=extra_info)


if __name__ == "__main__":
    main()
