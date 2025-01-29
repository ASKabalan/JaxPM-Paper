import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
os.environ["EQX_ON_ERROR"] = "nan"
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from functools import partial
from jaxdecomp import ShardedArray

from jaxpm.painting import cic_paint_dx
from jaxpm.pm import lpt, make_diffrax_ode
from diffrax import (
    ConstantStepSize,
    ODETerm,
    diffeqsolve,
    Dopri5,
    Tsit5,
    PIDController,
    RecursiveCheckpointAdjoint,
    BacksolveAdjoint,
)

from typing import NamedTuple

from tools.ode import symplectic_ode, DriftODETerm, DoubleKickODETerm, KickODETerm
from tools.integrate import integrate
from tools.semi_implicite_euler import SemiImplicitEuler
from tools.fast_pm import EfficientLeapFrog


from pmesh.pm import ParticleMesh

import numpy as np

import argparse
from jax_hpc_profiler import Timer

jax.config.update("jax_enable_x64", True)

# Define available solvers and adjoint methods
ADAPTIVE_SOLVERS = {
    "TSIT": Tsit5(),
    "DOPRI": Dopri5(),
}

SYMPLECTIC_SOLVERS = {
    "EULER": SemiImplicitEuler(),
    "LEAPFROG": EfficientLeapFrog(),
}

SOLVERS = [*ADAPTIVE_SOLVERS.keys(), *SYMPLECTIC_SOLVERS.keys()]

ADJOINTS_ADAPTIVE = ["RECURSIVE", "BACKSOLVE"]
ADJOINTS_CONSTANT = ["RECURSIVE", "REVERSE"]

ADJOINTS = ["RECURSIVE", "BACKSOLVE", "REVERSE"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run N-body simulation with given parameters."
    )
    parser.add_argument(
        "-m",
        "--mesh_shape",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        help="Mesh shape of the simulation.",
    )
    parser.add_argument(
        "-b",
        "--box_size",
        type=float,
        nargs=3,
        default=[512.0, 512.0, 512.0],
        help="Box size of the simulation.",
    )
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        choices=SOLVERS,
        required=True,
        help="Solver to use.",
    )
    parser.add_argument(
        "-a",
        "--adjoint",
        type=str,
        choices=ADJOINTS,
        required=True,
        help="Adjoint method to use.",
    )
    parser.add_argument(
        "-r",
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance (only for adaptive solvers).",
    )
    parser.add_argument(
        "-n",
        "--steps",
        type=int,
        default=None,
        help="Number of steps (only for constant step solvers).",
    )
    return parser.parse_args()


class Params(NamedTuple):
    Omega_c: float
    sigma8: float


def run_lpt(params, ic):
    cosmo = jc.Planck15(Omega_c=params.Omega_c, sigma8=params.sigma8)
    dx, p, _ = lpt(cosmo, ic, a=0.1, order=1)
    return dx, p


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def run_nbody(
    params,
    ic,
    terms,
    step_size=0.01,
    stepsize_controller=ConstantStepSize(),
    solver=Tsit5(),
    adjoint="RECURSIVE",
    checkpoints=20,
):
    dx, p = run_lpt(params, ic)
    cosmo = jc.Planck15(Omega_c=params.Omega_c, sigma8=params.sigma8)
    if isinstance(terms, ODETerm) or len(terms) == 1:
        y0 = jax.tree.map(lambda dx, p: jnp.stack([dx, p]), dx, p)
    elif len(terms) == 2:
        y0 = (dx, p)
    elif len(terms) == 3:
        *terms, first_kick_term = terms
        y0 = solver.first_step(
            first_kick_term, t0=0.1, dt0=step_size, y0=y0, args=cosmo
        )
        terms = tuple(terms)
    else:
        raise ValueError("Invalid number of terms.")

    if adjoint == "REVERSE":
        t0, t1 = 0.1, 1.0
        ode_solutions = integrate(
            y0, cosmo, terms, solver=solver, t0=t0, t1=t1, dt0=step_size
        )
        return ode_solutions[0], (t1 - t0) / step_size
    else:
        if adjoint == "RECURSIVE":
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
            args=cosmo,
        )
        last_y = jax.tree.map(lambda x: x[-1], ode_solutions.ys)
        num_steps = ode_solutions.stats["num_steps"]
        return last_y[0], num_steps


def MSE(x, y):
    return jnp.mean((x.data - y.data) ** 2)


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def model(
    params,
    ic,
    obs,
    term,
    step_size=0.01,
    stepsize_controller=ConstantStepSize(),
    solver=Tsit5(),
    adjoint="RECURSIVE",
    checkpoints=20,
):
    y_hat, num_steps = run_nbody(
        params, ic, term, step_size, stepsize_controller, solver, adjoint, checkpoints
    )
    y_hat_field = cic_paint_dx(y_hat)
    return MSE(y_hat_field, obs), num_steps


nbody = jax.jit(model, static_argnums=(3, 4, 5, 6, 7, 8))
nbody_ic = jax.jit(
    jax.grad(model, argnums=1, has_aux=True), static_argnums=(3, 4, 5, 6, 7, 8)
)

if __name__ == "__main__":
    args = parse_args()
    solver = args.solver.upper()
    adjoint = args.adjoint.upper()
    assert solver in SOLVERS
    solver = args.solver
    adaptive_step_solver = solver in ADAPTIVE_SOLVERS
    mesh_shape = args.mesh_shape
    box_shape = args.box_size

    if adaptive_step_solver:
        if args.rtol is None:
            raise ValueError("Relative tolerance is required for adaptive solvers.")
        assert args.adjoint in ADJOINTS_ADAPTIVE
        stepsize_controller = PIDController(rtol=args.rtol, atol=args.rtol)
        step_size = 0.01
        solver = ADAPTIVE_SOLVERS[solver]
    else:
        if args.steps is None:
            raise ValueError("Number of steps is required for constant step solvers.")
        stepsize_controller = ConstantStepSize()
        step_size = (1.0 - 0.1) / args.steps
        assert args.adjoint in ADJOINTS_CONSTANT
        solver = SYMPLECTIC_SOLVERS[solver]

    print(
        f"Running simulation with {args.solver}, adjoint {args.adjoint}, rtol={args.rtol}, steps={args.steps}"
    )
    print(f" -> mesh_shape={args.mesh_shape}, box_size={args.box_size}")

    omega_c = 0.25
    sigma8 = 0.8
    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
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

    initial_conditions = ShardedArray(jnp.asarray(init_mesh))

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
    guess_ic = ShardedArray(jnp.asarray(init_mesh))
    # if adaptive solver
    if adaptive_step_solver:
        ode_terms = ODETerm(make_diffrax_ode(mesh_shape, paint_absolute_pos=False))
    elif args.solver == "LEAPFROG":
        kick, drift = symplectic_ode(mesh_shape, paint_absolute_pos=False)
        ode_terms = DoubleKickODETerm(kick), DriftODETerm(drift), KickODETerm(kick)
    else:
        ode_terms = jax.tree.map(
            lambda x: ODETerm(x), symplectic_ode(mesh_shape, paint_absolute_pos=False)
        )

    jax_timer = Timer(
        save_jaxpr=False, jax_fn=True, ndarray_arg=0, static_argnums=(2, 3, 4, 5, 6, 7)
    )
    model_timer = Timer(
        save_jaxpr=False, jax_fn=True, ndarray_arg=0, static_argnums=(3, 4, 5, 6, 7, 8)
    )

    # FORWARD
    print("Running Forward Pass")
    observable, num_steps = jax_timer.chrono_jit(
        run_nbody,
        guess_params,
        guess_ic,
        ode_terms,
        step_size=step_size,
        stepsize_controller=stepsize_controller,
        solver=solver,
        adjoint=adjoint,
        checkpoints=20,
    )
    for _ in range(5):
        observable, num_steps = jax_timer.chrono_fun(
            run_nbody,
            guess_params,
            guess_ic,
            ode_terms,
            step_size=step_size,
            solver=solver,
            adjoint=adjoint,
            checkpoints=20,
        )

    observable = cic_paint_dx(observable)

    data = {"observable": observable}
    kwargs = {
        "function": f"Forward {adjoint}",
        "precision": "float64",
        "x": mesh_shape[0],
        "y": mesh_shape[1],
        "z": mesh_shape[2],
        "npz_data": data,
    }
    extra_info = {
        "solver": solver,
        "adjoint": adjoint,
        "rtol": args.rtol,
        "steps": args.steps,
    }
    jax_timer.report(f"runs/{args.solver}.csv", **kwargs, extra_info=extra_info)

    # BACKWARD
    print("Running Backward Pass")

    grads, num_steps = model_timer.chrono_jit(
        nbody_ic,
        guess_params,
        guess_ic,
        observable,
        ode_terms,
        step_size=step_size,
        stepsize_controller=stepsize_controller,
        solver=solver,
        adjoint=adjoint,
        checkpoints=20,
    )
    for _ in range(5):
        grads, num_steps = model_timer.chrono_fun(
            nbody_ic,
            guess_params,
            guess_ic,
            observable,
            ode_terms,
            step_size=step_size,
            stepsize_controller=stepsize_controller,
            solver=solver,
            adjoint=adjoint,
            checkpoints=20,
        )

    data = {"grads": grads}
    kwargs = {
        "function": f"Backward {adjoint}",
        "precision": "float64",
        "x": mesh_shape[0],
        "y": mesh_shape[1],
        "z": mesh_shape[2],
        "npz_data": data,
    }
    extra_info = {
        "solver": solver,
        "adjoint": adjoint,
        "rtol": args.rtol,
        "steps": args.steps,
    }
    model_timer.report(f"runs/{args.solver}.csv", **kwargs, extra_info=extra_info)
