import argparse
import os

os.environ["EQX_ON_ERROR"] = "nan"
os.environ["JC_CACHE"] = "off"

DISTRIBUTED = False
if os.environ.get("FAKE_DIST", "0") == "1":
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    DISTRIBUTED = True

import jax

# =============================================================================
# 1. If running on a distributed system, initialize JAX distributed
# =============================================================================
if (
    int(os.environ.get("SLURM_NTASKS", 0)) > 1
    or int(os.environ.get("SLURM_NTASKS_PER_NODE", 0)) > 1
):
    os.environ["VSCODE_PROXY_URI"] = ""
    os.environ["no_proxy"] = ""
    os.environ["NO_PROXY"] = ""
    del os.environ["VSCODE_PROXY_URI"]
    del os.environ["no_proxy"]
    del os.environ["NO_PROXY"]
    jax.distributed.initialize()

# =============================================================================
if jax.device_count() > 1:
    DISTRIBUTED = True

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from diffrax import (
    BacksolveAdjoint,
    ConstantStepSize,
    Dopri5,
    ODETerm,
    PIDController,
    RecursiveCheckpointAdjoint,
    Tsit5,
    diffeqsolve,
)
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax_hpc_profiler import Timer
from jaxpm.painting import cic_paint_dx
from jaxpm.pm import lpt, make_diffrax_ode
from pmesh.pm import ParticleMesh
from tools.integrate import integrate
from tools.ode import symplectic_fpm_ode
from tools.semi_implicite_euler import SemiImplicitEuler

all_gather = partial(process_allgather, tiled=True)

jax.config.update("jax_enable_x64", True)


# Define available solvers and adjoint methods
ADAPTIVE_SOLVERS = {
    "TSIT": Tsit5(),
    "DOPRI": Dopri5(),
}

SYMPLECTIC_SOLVERS = {
    "FASTPM": SemiImplicitEuler(),
}

SOLVERS = [*ADAPTIVE_SOLVERS.keys(), *SYMPLECTIC_SOLVERS.keys()]

ADJOINTS_ADAPTIVE = ["RECURSIVE", "BACKSOLVE"]
ADJOINTS_CONSTANT = ["RECURSIVE", "REVERSE"]

ADJOINTS = ["RECURSIVE", "BACKSOLVE", "REVERSE"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run N-body simulation with multiple configurations."
    )
    parser.add_argument(
        "-m",
        "--mesh_sizes",
        type=int,
        nargs="+",
        required=True,
        help="List of mesh sizes (e.g., 64 128 256).",
    )
    parser.add_argument(
        "-b",
        "--box_sizes",
        type=float,
        nargs="+",
        required=True,
        help="List of box sizes (must match mesh sizes).",
    )
    parser.add_argument(
        "-s", "--solver", type=str, choices=SOLVERS, default="FASTPM", help="Solver to use."
    )
    parser.add_argument(
        "-a",
        "--adjoints",
        type=str,
        nargs="+",
        choices=ADJOINTS,
        required=True,
        help="List of adjoint methods to use.",
    )
    parser.add_argument(
        "-r", "--rtol", type=float, default=None, help="Relative tolerance (adaptive solvers only)."
    )
    parser.add_argument(
        "-n", "--steps", type=int, default=10, help="Number of time steps (e.g., 10)"
    )
    parser.add_argument(
        "-p",
        "--pdims",
        type=int,
        nargs=2,
        default=[8, 1],
        help="Partition dimensions for distributed JAX (e.g., 8 1).",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=2,
        help="Number of iterations to run for each configuration.",
    )
    return parser.parse_args()


class Params(NamedTuple):
    Omega_c: float
    sigma8: float


def run_lpt(params, ic, halo_size, sharding):
    cosmo = jc.Planck15(Omega_c=params.Omega_c, sigma8=params.sigma8)
    dx, p, _ = lpt(cosmo, ic, a=0.1, order=1, halo_size=halo_size, sharding=sharding)
    return dx, p


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def run_nbody(
    params,
    ic,
    terms,
    step_size=0.01,
    stepsize_controller=ConstantStepSize(),
    solver=Tsit5(),
    adjoint="RECURSIVE",
    halo_size=0,
    sharding=None,
):
    ic = jax.lax.with_sharding_constraint(ic, sharding) if DISTRIBUTED else ic
    dx, p = run_lpt(params, ic, halo_size=halo_size, sharding=sharding)
    cosmo = jc.Planck15(Omega_c=params.Omega_c, sigma8=params.sigma8)
    if isinstance(terms, ODETerm) or len(terms) == 1:
        y0 = jax.tree.map(lambda dx, p: jnp.stack([dx, p]), dx, p)
    elif len(terms) == 2:
        y0 = (dx, p)
    elif len(terms) == 3:
        *terms, first_kick_term = terms
        y0 = (dx, p)
        y0 = solver.first_step(first_kick_term, t0=0.1, dt0=step_size, y0=y0, args=(cosmo,))
        terms = tuple(terms)
    else:
        raise ValueError("Invalid number of terms.")

    t0, t1 = 0.1, 1.0
    num_steps = int((t1 - t0) // step_size)
    if adjoint == "REVERSE":
        ode_solutions = integrate(
            terms, solver=solver, t0=t0, t1=t1, dt0=step_size, y0=y0, args=(cosmo,)
        )
        last_y = jax.tree.map(lambda x: x[-1], ode_solutions)
        observable = cic_paint_dx(last_y[0], halo_size=halo_size, sharding=sharding)
    else:
        if adjoint == "RECURSIVE":
            checkpoints = int(np.ceil(np.log(num_steps)))
            adjoint = RecursiveCheckpointAdjoint(checkpoints=checkpoints)
        elif adjoint == "BACKSOLVE":
            adjoint = BacksolveAdjoint(solver=solver)
        else:
            raise ValueError("Invalid adjoint method.")

        ode_solutions = diffeqsolve(
            terms=terms,
            solver=solver,
            t0=0.1,
            t1=1.0,
            dt0=step_size,
            y0=y0,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
            args=(cosmo,),
            max_steps=num_steps,
        )
        last_y = jax.tree.map(lambda x: x[-1], ode_solutions.ys)
        num_steps = ode_solutions.stats["num_steps"]
        observable = cic_paint_dx(last_y[0], halo_size=halo_size, sharding=sharding)

    return observable, num_steps


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def model(
    params,
    ic,
    obs,
    term,
    step_size=0.01,
    stepsize_controller=ConstantStepSize(),
    solver=Tsit5(),
    adjoint="RECURSIVE",
    halo_size=0,
    sharding=None,
):
    ic = jax.lax.with_sharding_constraint(ic, sharding) if DISTRIBUTED else ic
    obs = jax.lax.with_sharding_constraint(obs, sharding) if DISTRIBUTED else obs

    y_hat_field, _ = run_nbody(
        params,
        ic,
        term,
        step_size,
        stepsize_controller,
        solver,
        adjoint,
        halo_size,
        sharding,
    )
    y_hat_field = (
        jax.lax.with_sharding_constraint(y_hat_field, sharding) if DISTRIBUTED else y_hat_field
    )

    return ((y_hat_field - obs) ** 2).mean()


nbody = jax.jit(model, static_argnums=(3, 4, 5, 6, 7, 8, 9))
nbody_ic = jax.jit(jax.grad(model, argnums=1), static_argnums=(3, 4, 5, 6, 7, 8, 9))

if __name__ == "__main__":
    args = parse_args()
    solver = args.solver.upper()
    assert solver in SOLVERS
    solver = args.solver
    adaptive_step_solver = solver in ADAPTIVE_SOLVERS

    if adaptive_step_solver:
        if args.rtol is None:
            raise ValueError("Relative tolerance is required for adaptive solvers.")
        for adjoint in args.adjoints:
            assert adjoint in ADJOINTS_ADAPTIVE
        stepsize_controller = PIDController(rtol=args.rtol, atol=args.rtol)
        step_size = 0.01
        solver = ADAPTIVE_SOLVERS[solver]
    else:
        if args.steps is None:
            raise ValueError("Number of steps is required for constant step solvers.")
        stepsize_controller = ConstantStepSize()
        step_size = (1.0 - 0.1) / args.steps
        for adjoint in args.adjoints:
            assert adjoint in ADJOINTS_CONSTANT
        solver = SYMPLECTIC_SOLVERS[solver]

    if DISTRIBUTED:
        pdims = tuple(args.pdims)
        gpu_mesh = jax.make_mesh(pdims, ("x", "y"))
        sharding = NamedSharding(gpu_mesh, P("x", "y"))

    else:
        pdims = (1, 1)
        sharding = None
        gpu_mesh = jax.make_mesh(pdims, ("x", "y"))

    for mesh_size, box_size in zip(args.mesh_sizes, args.box_sizes):
        mesh_shape = (mesh_size,) * 3
        box_shape = (box_size,) * 3
        for adjoint in args.adjoints:
            print(
                f"Running simulation with {args.solver}, adjoint {adjoint}, rtol={args.rtol}, steps={args.steps}"
            )
            print(f" -> mesh_shape={mesh_shape}, box_size={box_shape}, sharding={sharding}")

            omega_c = 0.25
            sigma8 = 0.8
            params = Params(Omega_c=omega_c, sigma8=sigma8)
            cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
            halo_size = mesh_shape[0] // 4 if DISTRIBUTED else 0
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
            init_mesh = (
                jax.lax.with_sharding_constraint(jnp.asarray(init_mesh), sharding)
                if DISTRIBUTED
                else jnp.asarray(init_mesh)
            )

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
            guess_ic = (
                jax.lax.with_sharding_constraint(guess_ic, sharding) if DISTRIBUTED else guess_ic
            )
            # if adaptive solver
            if adaptive_step_solver:
                ode_terms = ODETerm(
                    make_diffrax_ode(
                        mesh_shape, paint_absolute_pos=False, halo_size=halo_size, sharding=sharding
                    )
                )
            elif args.solver == "FASTPM":
                drift, kick, first_kick = symplectic_fpm_ode(
                    mesh_shape,
                    step_size,
                    paint_absolute_pos=False,
                    halo_size=halo_size,
                    sharding=sharding,
                )
                print(f"Creating terms with halo_size={halo_size} and sharding={sharding}")
                ode_terms = ODETerm(drift), ODETerm(kick), ODETerm(first_kick)
            else:
                raise ValueError("Invalid solver.")

            jax_timer = Timer(save_jaxpr=False, jax_fn=True, static_argnums=(2, 3, 4, 5, 6, 7, 8))
            model_timer = Timer(save_jaxpr=False, jax_fn=True, static_argnums=(3, 4, 5, 6, 7, 8, 9))

            # FORWARD
            with gpu_mesh:
                print("Running Forward Pass")
                observable, num_steps = jax_timer.chrono_jit(
                    run_nbody,
                    params,
                    init_mesh,
                    ode_terms,
                    step_size=step_size,
                    stepsize_controller=stepsize_controller,
                    solver=solver,
                    adjoint=adjoint,
                    halo_size=halo_size,
                    sharding=sharding,
                )
                for _ in range(args.iterations):
                    observable, num_steps = jax_timer.chrono_fun(
                        run_nbody,
                        params,
                        init_mesh,
                        ode_terms,
                        step_size=step_size,
                        stepsize_controller=stepsize_controller,
                        solver=solver,
                        adjoint=adjoint,
                        halo_size=halo_size,
                        sharding=sharding,
                    )

            print(f" -> Sharding of Observable: {observable.sharding}")
            data = {"observable": all_gather(observable)}
            kwargs = {
                "function": f"Forward {adjoint}",
                "precision": "float64",
                "x": mesh_shape[0],
                "y": mesh_shape[1],
                "z": mesh_shape[2],
                "npz_data": data,
                "px": pdims[0],
                "py": pdims[1],
            }
            extra_info = {
                "solver": solver,
                "adjoint": adjoint,
                "rtol": args.rtol,
                "steps": args.steps,
            }
            jax_timer.report(f"runs/{args.solver}.csv", **kwargs, extra_info=extra_info)

            # BACKWARD
            with gpu_mesh:
                print("Running Backward Pass")
                grads = model_timer.chrono_jit(
                    nbody_ic,
                    guess_params,
                    guess_ic,
                    observable,
                    ode_terms,
                    step_size=step_size,
                    stepsize_controller=stepsize_controller,
                    solver=solver,
                    adjoint=adjoint,
                    halo_size=halo_size,
                    sharding=sharding,
                )
                for _ in range(args.iterations):
                    grads = model_timer.chrono_fun(
                        nbody_ic,
                        guess_params,
                        guess_ic,
                        observable,
                        ode_terms,
                        step_size=step_size,
                        stepsize_controller=stepsize_controller,
                        solver=solver,
                        adjoint=adjoint,
                        halo_size=halo_size,
                        sharding=sharding,
                    )
            print(f" -> Sharding of Gradients: {grads.sharding}")
            data = {"grads": all_gather(grads)}
            kwargs = {
                "function": f"Backward {adjoint}",
                "precision": "float64",
                "x": mesh_shape[0],
                "y": mesh_shape[1],
                "z": mesh_shape[2],
                "npz_data": data,
                "px": pdims[0],
                "py": pdims[1],
            }
            extra_info = {
                "solver": solver,
                "adjoint": adjoint,
                "rtol": args.rtol,
                "steps": args.steps,
            }
            model_timer.report(f"runs/{args.solver}.csv", **kwargs, extra_info=extra_info)
