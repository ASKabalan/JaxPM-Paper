# --- Imports ---
import argparse
import os
from enum import Enum
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from diffrax import (
    ConstantStepSize,
    ODETerm,
    RecursiveCheckpointAdjoint,
    diffeqsolve,
)
from jaxpm.painting import cic_paint_dx
from jaxpm.pm import linear_field, lpt
from pmesh.pm import ParticleMesh
from tools.finite_difference import numerical_jvp
from tools.integrate import integrate
from tools.ode import symplectic_fpm_ode
from tools.plotting import plot_gradient_errors, plot_memory_runs
from tools.semi_implicite_euler import SemiImplicitEuler
from tqdm import tqdm

# --- JAX and Env Setup ---
jax.config.update("jax_enable_x64", True)
os.environ["EQX_ON_ERROR"] = "nan"
os.makedirs("data", exist_ok=True)


# --- Enum and Configs ---
class GradientType(Enum):
    Checkpoint = "Checkpoint"
    Reverse = "Reverse"


CHECKPOINT = GradientType.Checkpoint
REVERSE = GradientType.Reverse


# --- Parameter Tuple ---
class Params(NamedTuple):
    Omega_c: float
    sigma8: float


# --- LPT Function ---
@jax.jit
def run_lpt(params, ic):
    cosmo = jc.Planck15(Omega_c=params.Omega_c, sigma8=params.sigma8)
    dx, p, _ = lpt(cosmo, ic, a=0.1, order=1)
    return dx, p


# --- N-body Simulation ---
@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def run_nbody(
    params,
    ic,
    terms,
    step_size=0.01,
    stepsize_controller=ConstantStepSize(),
    solver=SemiImplicitEuler(),
    adjoint=CHECKPOINT,
    checkpoints=100,
):
    dx, p = run_lpt(params, ic)
    cosmo = jc.Planck15(Omega_c=params.Omega_c, sigma8=params.sigma8)
    y0 = (dx, p)

    # Handle first kick term if included (first kick is used for FastPM solver)
    if len(terms) == 3:
        *terms, first_kick_term = terms
        y0 = solver.first_step(first_kick_term, t0=0.1, dt0=step_size, y0=y0, args=(cosmo,))
        terms = tuple(terms)

    # Choose adjoint method
    if adjoint == CHECKPOINT:
        ode_solutions = diffeqsolve(
            terms,
            solver=solver,
            t0=0.1,
            t1=1.0,
            dt0=step_size,
            y0=y0,
            args=(cosmo,),
            stepsize_controller=stepsize_controller,
            adjoint=RecursiveCheckpointAdjoint(checkpoints=checkpoints),
        )
        last_y = jax.tree.map(lambda x: x[-1], ode_solutions.ys)
        num_steps = ode_solutions.stats["num_steps"]
        return last_y[0], num_steps

    elif adjoint == REVERSE:
        t0, t1 = 0.1, 1.0
        ode_solutions = integrate(
            terms, solver=solver, t0=t0, t1=t1, dt0=step_size, y0=y0, args=(cosmo,)
        )
        last_y = jax.tree.map(lambda x: x[-1], ode_solutions)
        return last_y[0], (t1 - t0) / step_size

    else:
        raise ValueError(f"Unknown adjoint method {adjoint}")


# --- Utility ---
def MSE(x, y):
    return jnp.mean((x - y) ** 2)


# --- LPT Model for Loss ---
@jax.jit
def lpt_model(params, ic, obs):
    dx, p = run_lpt(params, ic)
    return MSE(cic_paint_dx(dx), obs)


# --- Full Model Wrapper ---
@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def model(
    params,
    ic,
    obs,
    term,
    step_size=0.01,
    stepsize_controller=ConstantStepSize(),
    solver=SemiImplicitEuler(),
    adjoint=CHECKPOINT,
    checkpoints=100,
):
    y_hat, num_steps = run_nbody(
        params, ic, term, step_size, stepsize_controller, solver, adjoint, checkpoints
    )
    y_hat_field = cic_paint_dx(y_hat)
    return MSE(y_hat_field, obs), num_steps


# --- Gradients ---
nbody_cosmo = jax.jit(jax.grad(model, has_aux=True), static_argnums=(3, 4, 5, 6, 7, 8))
nbody_ic = jax.jit(jax.grad(model, argnums=1, has_aux=True), static_argnums=(3, 4, 5, 6, 7, 8))


# --- CLI Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Gradient Stability: run or plot.")
    parser.add_argument("-m", "--mesh_size", type=int, default=64, help="Mesh size (e.g., 64).")
    parser.add_argument("-b", "--box_size", type=float, default=512.0, help="Box size.")
    parser.add_argument(
        "-t", "--run-type", type=str, choices=["grad", "mem"], required=True, help="Run type."
    )
    parser.add_argument("-p", "--plot", action="store_true", help="Plot existing results.")
    return parser.parse_args()


# --- Gradient Pipeline ---
def run_gradient_pipeline(solver, mesh_shape, guess_ic, guess_params, t0, t1, observable):
    def constant_steps_model(observable, adjoint, num_steps):
        step_size = (t1 - t0) / num_steps
        drift, kick, first_kick = symplectic_fpm_ode(
            mesh_shape, step_size, paint_absolute_pos=False
        )
        ode_terms = ODETerm(drift), ODETerm(kick), ODETerm(first_kick)
        tangents = jax.random.normal(jax.random.PRNGKey(42), guess_ic.shape)
        tangents = jax.tree.unflatten(jax.tree.structure(guess_ic), (tangents,))

        if adjoint == "finite_diff":

            def fn(guess_ic):
                val, _ = model(
                    guess_params,
                    guess_ic,
                    observable,
                    ode_terms,
                    step_size=step_size,
                    solver=solver,
                )
                return val

            grads = numerical_jvp(fn, (guess_ic,), (tangents,), eps=1e-12)
            return grads, num_steps

        grads, steps = nbody_ic(
            guess_params,
            guess_ic,
            observable,
            ode_terms,
            step_size=step_size,
            solver=solver,
            adjoint=adjoint,
        )
        grads = (grads * tangents).sum()
        return grads, steps

    def generate_gradient_data(observable, adjoint, base_line_tol=90):
        gradients, steps = [], []
        print(
            f"Running for EfficientLeapFrog with adjoint {adjoint} and baseline tol {base_line_tol}"
        )
        for num_steps in tqdm(jnp.arange(10, 100, 10).tolist()):
            grad, num_steps = constant_steps_model(observable, adjoint, num_steps)
            gradients.append(grad)
            steps.append(num_steps)
        return gradients, steps

    base_fpm_grad, base_fpm_steps = constant_steps_model(observable, CHECKPOINT, num_steps=90)
    fpm_grads_DTO, fpm_steps_DTO = generate_gradient_data(observable, CHECKPOINT)
    fpm_grads_REV, fpm_steps_REV = generate_gradient_data(observable, REVERSE)
    fpm_grads_fd, _ = generate_gradient_data(observable, "finite_diff")

    return {
        "base_fpm_grad": base_fpm_grad,
        "base_fpm_steps": base_fpm_steps,
        "fpm_grads_DTO": fpm_grads_DTO,
        "fpm_steps_DTO": fpm_steps_DTO,
        "fpm_grads_REV": fpm_grads_REV,
        "fpm_steps_REV": fpm_steps_REV,
        "fpm_grads_fd": fpm_grads_fd,
    }


# --- Memory Pipeline ---
def run_memory_pipeline(solver, mesh_shape, guess_ic, guess_params, t0, t1, observable):
    def constant_steps_memory(solver, observable, adjoint, num_steps, checkpoint=100):
        step_size = (t1 - t0) / num_steps
        drift, kick, first_kick = symplectic_fpm_ode(
            mesh_shape, step_size, paint_absolute_pos=False
        )
        ode_terms = ODETerm(drift), ODETerm(kick), ODETerm(first_kick)

        @jax.jit
        def fn(guess_params):
            return nbody_cosmo(
                guess_params,
                guess_ic,
                observable,
                ode_terms,
                solver=solver,
                step_size=step_size,
                adjoint=adjoint,
                checkpoints=checkpoint,
            )

        grad, steps = fn(guess_params)
        memory = fn.lower(guess_params).compile().memory_analysis().temp_size_in_bytes
        return grad.Omega_c, num_steps, memory

    def generate_memory_data(solver, observable, base_line_steps=90):
        gradients, steps, checkpoints, memories = [], [], [], []
        for checkpoint in tqdm(jnp.arange(20, 90, 10).tolist()):
            grad, num_steps, memory = constant_steps_memory(
                solver, observable, CHECKPOINT, num_steps=90, checkpoint=checkpoint
            )
            gradients.append(grad)
            steps.append(num_steps)
            checkpoints.append(checkpoint)
            memories.append(memory)
        return gradients, steps, checkpoints, memories

    fpm_grad_base_DTO, fpm_steps_base_DTO, fpm_memories_base_DTO = constant_steps_memory(
        solver, observable, CHECKPOINT, num_steps=90
    )
    fpm_grad_base_REV, fpm_steps_base_REV, fpm_memories_base_REV = constant_steps_memory(
        solver, observable, REVERSE, num_steps=90
    )
    fpm_grads_DTO, fpm_steps_DTO, fpm_checkpoints_DTO, fpm_memories_DTO = generate_memory_data(
        solver, observable
    )

    return {
        "fpm_grad_base_DTO": fpm_grad_base_DTO,
        "fpm_steps_base_DTO": fpm_steps_base_DTO,
        "fpm_memories_base_DTO": fpm_memories_base_DTO,
        "fpm_grad_base_REV": fpm_grad_base_REV,
        "fpm_steps_base_REV": fpm_steps_base_REV,
        "fpm_memories_base_REV": fpm_memories_base_REV,
        "fpm_grads_DTO": fpm_grads_DTO,
        "fpm_steps_DTO": fpm_steps_DTO,
        "fpm_checkpoints_DTO": fpm_checkpoints_DTO,
        "fpm_memories_DTO": fpm_memories_DTO,
    }


# --- Main Execution ---
def main():
    args = parse_args()
    mesh_shape = [args.mesh_size] * 3
    box_size = [args.box_size] * 3

    # Cosmology setup
    omega_c = 0.25
    sigma8 = 0.8
    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
    t1 = 1.0
    t0 = 0.1

    out_file = (
        "data/GS_FPM_initial_field_gradient_error.npz"
        if args.run_type == "grad"
        else "data/GS_FPM_memory_usage_and_error.npz"
    )

    if args.plot:
        if not os.path.exists(out_file):
            raise FileNotFoundError(
                f"{out_file} not found. Run script without --plot to generate it."
            )
        if args.run_type == "grad":
            plot_gradient_errors(out_file)
        else:
            plot_memory_runs(out_file)

        return

    # Generate initial field
    pm = ParticleMesh(BoxSize=box_size, Nmesh=mesh_shape, dtype="f8")
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)
    whitec = pm.generate_whitenoise(42, type="complex", unitary=False)

    def pk_fn(x):
        return jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    lineark = whitec.apply(
        lambda k, v: pk_fn(sum(ki**2 for ki in k) ** 0.5) ** 0.5 * v * (1 / v.BoxSize).prod() ** 0.5
    )
    init_mesh = lineark.c2r().value
    initial_conditions = jnp.asarray(init_mesh)

    # Setup solver and observation
    drift, kick, first_kick = symplectic_fpm_ode(mesh_shape, 0.01, paint_absolute_pos=False)
    ode_terms = ODETerm(drift), ODETerm(kick), ODETerm(first_kick)
    solver = SemiImplicitEuler()
    params = Params(omega_c, sigma8)

    guess_ic = (
        initial_conditions
        + jax.random.normal(jax.random.PRNGKey(42), initial_conditions.shape) * 0.01
    )
    # Create initial conditions
    guess_ic = linear_field(mesh_shape, box_size, pk_fn, seed=jax.random.PRNGKey(42))

    guess_params = Params(Omega_c=0.8, sigma8=0.8)
    ode_solution_fpm, _ = run_nbody(
        params, initial_conditions, ode_terms, step_size=0.01, solver=solver
    )
    obs_fpm_field = cic_paint_dx(ode_solution_fpm)

    # Run pipeline
    if args.run_type == "grad":
        data_to_save = run_gradient_pipeline(
            solver, mesh_shape, guess_ic, guess_params, t0, t1, obs_fpm_field
        )
    elif args.run_type == "mem":
        data_to_save = run_memory_pipeline(
            solver, mesh_shape, guess_ic, guess_params, t0, t1, obs_fpm_field
        )

    np.savez(out_file, **data_to_save)


# --- Entry ---
if __name__ == "__main__":
    main()
