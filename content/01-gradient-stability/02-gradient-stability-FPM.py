#!/usr/bin/env python
# coding: utf-8

# # Gradient Accuracy for Adaptive ODE Solvers
# 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os
import sys

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)
os.environ["EQX_ON_ERROR"] = "nan"
from enum import Enum
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
from diffrax import (
    ConstantStepSize,
    ODETerm,
    RecursiveCheckpointAdjoint,
    Tsit5,
    diffeqsolve,
)
from fastpm.core import Cosmology as FastPMCosmology
from fastpm.core import Solver, leapfrog
from jaxpm.painting import cic_paint_dx
from jaxpm.pm import linear_field, lpt
from pmesh.pm import ParticleMesh
from tools.finite_difference import numerical_jvp
from tools.integrate import integrate
from tools.ode import symplectic_ode
from tools.semi_implicite_euler import SemiImplicitEuler
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


# Create `plot` and data `folders` 

# In[2]:


import os

os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ## Initialisation

# ### Generate Initial Conditions
# 
# This section initializes the simulation by defining the mesh resolution, box size, and cosmological parameters.  
# We generate a uniform particle grid and use the linear matter power spectrum to compute initial density fluctuations.  
# A white noise field is created and filtered to match the expected power spectrum, yielding the starting conditions for the simulation.
# 

# In[3]:


mesh_shape = [64, 64, 64]
box_size = [512.0, 512.0, 512.0]

omega_c = 0.25
sigma8 = 0.8
cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)

# Generate initial particle positions
pm = ParticleMesh(BoxSize=box_size, Nmesh=mesh_shape, dtype="f8")
grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float64)
# Interpolate with linear_matter spectrum to get initial density field
k = jnp.logspace(-4, 1, 128)
pk = jc.power.linear_matter_power(cosmo, k)

whitec = pm.generate_whitenoise(42, type="complex", unitary=False)


def pk_fn(x):
    return jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)


lineark = whitec.apply(
    lambda k, v: pk_fn(sum(ki**2 for ki in k) ** 0.5) ** 0.5 * v * (1 / v.BoxSize).prod() ** 0.5
)
init_mesh = lineark.c2r().value
ref_cosmo = FastPMCosmology(cosmo)
solver = Solver(pm, ref_cosmo, B=1)


initial_conditions = jnp.asarray(init_mesh)


# ### Gradient Computation for Symplectic Solvers
# 
# This section defines the gradient computation approach for symplectic solvers such as SemiImplicitEuler and FastPM LeapFrog.  
# 
# - **LPT Initialization**: The `run_lpt` function initializes particle displacements using first-order Lagrangian Perturbation Theory (LPT).  
# - **N-body Evolution**: The `run_nbody` function advances the system using a symplectic solver. It supports different adjoint methods for computing gradients:  
#   - **Checkpoint Adjoint**: Uses recursive checkpointing to balance memory and computational cost.  
#   - **Reverse Adjoint**: Uses reverse-mode differentiation to propagate gradients efficiently.  
# 

# In[4]:


class GradientType(Enum):
    Checkpoint = "Checkpoint"
    Reverse = "Reverse"


CHECKPOINT = GradientType.Checkpoint
REVERSE = GradientType.Reverse


class Params(NamedTuple):
    Omega_c: float
    sigma8: float


@jax.jit
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
    adjoint=CHECKPOINT,
    checkpoints=100,
):
    dx, p = run_lpt(params, ic)
    cosmo = jc.Planck15(Omega_c=params.Omega_c, sigma8=params.sigma8)
    y0 = (dx, p)
    if len(terms) == 3:
        *terms, first_kick_term = terms
        y0 = solver.first_step(first_kick_term, t0=0.1, dt0=step_size, y0=y0, args=(cosmo,))
        terms = tuple(terms)

    # Evolve the simulation forward
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

    return ode_solutions


# ### Forward Model for MSE Computation
# 
# We define a forward model that computes the Mean Squared Error (MSE) between the observed field and the generated field, which is derived from initial conditions and cosmological parameters.  
# 
# - **lpt_model**: Uses Lagrangian Perturbation Theory (LPT) to generate an approximate field and compare it with observations.  
# - **model**: Runs a full N-body simulation to evolve the system and computes the MSE against the observed field.  
# - **Gradient Computation**: The model is JIT-compiled for efficiency, allowing differentiation with respect to both cosmological parameters and initial conditions.  
# 

# In[ ]:


def MSE(x, y):
    return jnp.mean((x - y) ** 2)


@jax.jit
def lpt_model(params, ic, obs):
    dx, p = run_lpt(params, ic)
    return MSE(cic_paint_dx(dx), obs)


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def model(
    params,
    ic,
    obs,
    term,
    step_size=0.01,
    stepsize_controller=ConstantStepSize(),
    solver=Tsit5(),
    adjoint=CHECKPOINT,
    checkpoints=100,
):
    y_hat, num_steps = run_nbody(
        params, ic, term, step_size, stepsize_controller, solver, adjoint, checkpoints
    )
    y_hat = y_hat
    y_hat_field = cic_paint_dx(y_hat)

    return MSE(y_hat_field, obs), num_steps


nbody = jax.jit(model, static_argnums=(3, 4, 5, 6, 7, 8))

nbody_cosmo = jax.jit(jax.grad(model, has_aux=True), static_argnums=(3, 4, 5, 6, 7, 8))
nbody_ic = jax.jit(jax.grad(model, argnums=1, has_aux=True), static_argnums=(3, 4, 5, 6, 7, 8))


# #### Visualization
# 
# We visualize the projected density fields from both LPT and N-body simulations.  
# 
# - **Left (LPT)**: The field generated using first-order Lagrangian Perturbation Theory.  
# - **Right (N-body)**: The evolved field using the Semi-Implicit Euler solver.  
# 
# This comparison illustrates the differences in structure formation bet
# 

# In[7]:


params = Params(0.25, 0.8)

ode_terms = jax.tree.map(lambda x: ODETerm(x), symplectic_ode(mesh_shape, paint_absolute_pos=False))
final_field, _ = run_nbody(
    params,
    initial_conditions,
    ode_terms,
    step_size=0.1,
    solver=SemiImplicitEuler(),
)
lpt_dx, _ = run_lpt(params, initial_conditions)

sie_field = cic_paint_dx(final_field)
lpt_field = cic_paint_dx(lpt_dx)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(lpt_field[10:].sum(axis=0), cmap="magma")
plt.title("LPT")
plt.subplot(122)
plt.imshow(sie_field[10:].sum(axis=0), cmap="magma")
plt.title("N-body")
plt.show()


# ### Gradient Evaluation  
# 
# We compute gradients of the N-body simulation with respect to cosmological parameters and analyze their sensitivity when moving away from the correct values.  
# 

# In[8]:


best_params = params
guess_params = Params(Omega_c=0.8, sigma8=0.8)

best_grad, _ = nbody_cosmo(
    best_params,
    initial_conditions,
    sie_field,
    ode_terms,
    step_size=0.1,
    solver=SemiImplicitEuler(),
)
guess_grad, _ = nbody_cosmo(
    guess_params,
    initial_conditions,
    sie_field,
    ode_terms,
    step_size=0.1,
    solver=SemiImplicitEuler(),
)

print(f"Gradient at the correct cosmology: {best_grad}")
print(f"Evaluating the gradient at the Correct Cosmology: {guess_grad}")


# #### Sensitivity to Initial Conditions  
# 
# We compute gradients of the N-body simulation with respect to initial conditions and analyze how they change when perturbing the initial field.  
# 

# In[ ]:


best_ic = initial_conditions

cosmo = jc.Planck15(Omega_c=best_params.Omega_c, sigma8=best_params.sigma8)
pk = jc.power.linear_matter_power(cosmo, k)


def pk_fn_2(x):
    return jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)


# Create initial conditions
guess_ic = linear_field(mesh_shape, box_size, pk_fn_2, seed=jax.random.PRNGKey(42))


best_grad, _ = nbody_ic(
    best_params,
    best_ic,
    sie_field,
    ode_terms,
    step_size=0.1,
    solver=SemiImplicitEuler(),
)
guess_grad, _ = nbody_ic(
    best_params,
    guess_ic,
    sie_field,
    ode_terms,
    step_size=0.1,
    solver=SemiImplicitEuler(),
)

print(f"Gradient at the correct cosmology: Max {best_grad.max()} min {best_grad.min()}")
print(
    f"Evaluating the gradient at the Correct Cosmology: Max {guess_grad.max()} min {guess_grad.min()}"
)


# ## FastPM LeapFrog in JAX  
# 
# We visualize the projected density field obtained using the FastPM LeapFrog integrator, highlighting the evolved structure from the N-body simulation.  
# 

# In[ ]:


from tools.ode import symplectic_fpm_ode

drift, kick, first_kick = symplectic_fpm_ode(mesh_shape, 0.1, paint_absolute_pos=False)
ode_terms = ODETerm(drift), ODETerm(kick), ODETerm(first_kick)

ode_solution_fpm, _ = run_nbody(
    params,
    initial_conditions,
    ode_terms,
    step_size=0.1,
    solver=SemiImplicitEuler(),
)
fpm_field = cic_paint_dx(ode_solution_fpm)


plt.figure(figsize=(12, 6))
plt.imshow(fpm_field[10:].sum(axis=0), cmap="magma")
plt.title("FastPM LeapFrog")


plt.show()


# ### Running FastPM
# 
# We execute the CPU version of FastPM to evolve the particle state using LeapFrog integration. The simulation progresses from LPT-initialized positions through multiple time steps, producing the final density field.  
# 

# In[20]:


statelpt = solver.lpt(lineark, grid, 0.1, order=1)
stages = np.linspace(0.1, 1.0, 10, endpoint=True)

leapfrog_stages = leapfrog(stages)
finalstate = solver.nbody(statelpt, leapfrog_stages)
fpm_mesh = pm.paint(finalstate.X).value


# ### Validating the LeapFrog Integrator  
# 
# This comparison verifies the correctness of the LeapFrog implementation by measuring its difference from FastPM. The small error between Efficient LeapFrog and FastPM confirms that the integrator is accurate, while the larger error for Semi-Implicit Euler highlights its deviation from the expected solution.  
# 

# In[ ]:


def jax_mse(x, y):
    return ((x - y) ** 2).mean()


print(
    f"Difference between FPM and EfficientLeapFrog with 10 steps is {jax_mse(fpm_field, fpm_mesh)} "
)
print(
    f"Difference between FPM and SemiImpliciteEuler with 10 steps is {jax_mse(sie_field, fpm_mesh)} "
)


# ### Validating Reverse-Mode Differentiation  
# 
# This test checks that the reverse-mode differentiation of the integrator produces exact gradients matching those obtained with checkpointing. The near-zero difference confirms the correctness of the reverse-mode gradient computation.  
# 

# In[ ]:


drift, kick, first_kick = symplectic_fpm_ode(mesh_shape, 0.01, paint_absolute_pos=False)
ode_terms = ODETerm(drift), ODETerm(kick), ODETerm(first_kick)


gradients_rev, _ = nbody_ic(
    best_params,
    best_ic,
    fpm_field,
    ode_terms,
    step_size=0.01,
    solver=SemiImplicitEuler(),
    adjoint=REVERSE,
)
gradiennts_check, _ = nbody_ic(
    best_params,
    best_ic,
    fpm_field,
    ode_terms,
    step_size=0.01,
    solver=SemiImplicitEuler(),
    adjoint=CHECKPOINT,
)

print(
    f"Maximum difference between reverse and checkpoint gradients is {jnp.abs(gradients_rev - gradiennts_check).max()}"
)


# ## First Study : Gradient Accuracy for Cosmology  
# 
# We generate an observed field using the FastPM-inspired Efficient LeapFrog integrator. This serves as a reference to analyze the accuracy of cosmological gradients in the simulation.  
# 

# In[24]:


drift, kick, first_kick = symplectic_fpm_ode(mesh_shape, 0.01, paint_absolute_pos=False)
ode_terms = ODETerm(drift), ODETerm(kick), ODETerm(first_kick)


ode_solution_fpm, _ = run_nbody(
    params,
    initial_conditions,
    ode_terms,
    step_size=0.01,
    solver=SemiImplicitEuler(),
)
obs_fpm_field = cic_paint_dx(ode_solution_fpm)


# ### Gradient Accuracy Analysis  
# 
# We evaluate the accuracy of cosmological gradients by varying the number of integration steps and comparing different differentiation methods.  
# 
# A baseline gradient is computed using the Efficient LeapFrog solver with checkpointing at 90 steps. Gradients are then measured for different step counts using:  
# 
# - **Checkpoint Adjoint (DTO - Discretize Then Optimize)**  
# - **Reverse-Mode Differentiation (REV)**  
# - **Finite Differences (FD)**  
# 
# This analysis helps assess how gradient accuracy depends on the integration method and step resolution.  
# 

# In[25]:


t1 = 1.0
t0 = 0.1


def constant_steps_model(solver, observable, adjoint, num_steps):
    step_size = (t1 - t0) / num_steps
    kick, drift, first_kick = symplectic_fpm_ode(mesh_shape, step_size, paint_absolute_pos=False)
    ode_terms = ODETerm(drift), ODETerm(kick), ODETerm(first_kick)
    if adjoint == "finite_diff":

        def fn(guess_params):
            val, _ = nbody(
                guess_params,
                guess_ic,
                observable,
                ode_terms,
                step_size=step_size,
                solver=solver,
            )
            return val

        tangents = Params(jnp.array(1.0), jnp.array(0.0))
        grads = numerical_jvp(fn, (guess_params,), (tangents,), eps=1e-12)
        return grads, num_steps

    grads, steps = nbody_cosmo(
        guess_params,
        guess_ic,
        observable,
        ode_terms,
        step_size=step_size,
        solver=solver,
        adjoint=adjoint,
    )
    return grads.Omega_c, steps


def generate_gradient_data(solver, observable, adjoint, base_line_tol=90):
    gradients, steps = [], []
    print(f"running for {solver} with adjoint {adjoint} with base line tol {base_line_tol}")
    for num_steps in tqdm(jnp.arange(10, 100, 10).tolist()):
        grad, num_steps = constant_steps_model(solver, observable, adjoint, num_steps)
        gradients.append(grad)
        steps.append(num_steps)

    return gradients, steps


base_fpm_grad, base_fpm_steps = constant_steps_model(
    SemiImplicitEuler(), obs_fpm_field, adjoint=CHECKPOINT, num_steps=90
)

fpm_grads_DTO, fpm_steps_DTO = generate_gradient_data(
    SemiImplicitEuler(), obs_fpm_field, adjoint=CHECKPOINT
)
fpm_grads_REV, fpm_steps_REV = generate_gradient_data(
    SemiImplicitEuler(), obs_fpm_field, adjoint=REVERSE
)
fpm_grads_fd, _ = generate_gradient_data(SemiImplicitEuler(), obs_fpm_field, "finite_diff")


# In[32]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_context("paper")

# Compute absolute errors
fpm_error_DTO = [abs(base_fpm_grad - grad) for grad in fpm_grads_DTO]
fpm_error_REV = [abs(base_fpm_grad - grad) for grad in fpm_grads_REV]

# Compute errors relative to finite differences
fpm_error_FD_DTO = [
    abs(fd_grad - dto_grad) for fd_grad, dto_grad in zip(fpm_grads_fd, fpm_grads_DTO)
]
fpm_error_FD_REV = [
    abs(fd_grad - otd_grad) for fd_grad, otd_grad in zip(fpm_grads_fd, fpm_grads_REV)
]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot Gradient Error vs Steps in Subplot 1
ax1.plot(
    fpm_steps_DTO,
    fpm_error_DTO,
    marker="o",
    linestyle="-",
    label="EfficientLeapFrog DTO",
)
ax1.plot(
    fpm_steps_REV,
    fpm_error_REV,
    marker="s",
    linestyle="--",
    label="EfficientLeapFrog Reverse",
)

# Customize Subplot 1
ax1.set_yscale("log")  # Log scale for gradient errors
ax1.set_xlabel("Steps")
ax1.set_ylabel("OmegaC Gradient Error (|base - obtained|)")
ax1.set_title("OmegaC Gradient Error vs Steps")
ax1.legend()
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

# Plot Finite Difference Comparison in Subplot 2
ax2.plot(fpm_steps_DTO, fpm_error_FD_DTO, marker="o", linestyle="-", label="DTO vs FD")
ax2.plot(fpm_steps_REV, fpm_error_FD_REV, marker="s", linestyle="--", label="Reverse vs FD")

# Customize Subplot 2
ax2.set_yscale("log")  # Log scale for gradient errors
ax2.set_xlabel("Steps")
ax2.set_ylabel("OmegaC Gradient Error (|finite difference - obtained|)")
ax2.set_title("Comparison with Finite Difference")
ax2.legend()
ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

# Adjust layout and save
plt.tight_layout()
plt.savefig("plots/GS_FPM_OmegaC_gradient_error.pdf", dpi=600, transparent=True)
plt.show()


data_to_save = {
    "base_fpm_grad": base_fpm_grad,
    "fpm_error_DTO": fpm_error_DTO,
    "fpm_steps_DTO": fpm_steps_DTO,
    "fpm_steps_REV": fpm_steps_REV,
    "fpm_grads_DTO": fpm_grads_DTO,
    "fpm_grads_REV": fpm_grads_REV,
    "fpm_grads_fd": fpm_error_DTO,
}

np.savez("data/GS_FPM_OmegaC_gradient_error.npz", **data_to_save)


# ## Second Study: Gradient Accuracy for Initial Conditions  
# 
# We analyze the accuracy of gradients with respect to initial conditions by varying the number of integration steps and comparing different differentiation methods.  
# 
# A baseline gradient is computed using the Efficient LeapFrog solver with checkpointing at 90 steps. Gradients are then measured for different step counts using:  
# 
# - **Checkpoint Adjoint (DTO - Discretize Then Optimize)**  
# - **Reverse-Mode Differentiation (REV)**  
# - **Finite Differences (FD)**  
# 
# This study helps evaluate how integration step resolution impacts gradient accuracy when differentiating with respect to initial conditions.  
# 

# In[ ]:


t1, t0 = 1.0, 0.1
solver = SemiImplicitEuler()


def constant_steps_model(observable, adjoint, num_steps):
    step_size = (t1 - t0) / num_steps
    drift, kick, first_kick = symplectic_fpm_ode(mesh_shape, step_size, paint_absolute_pos=False)
    ode_terms = ODETerm(drift), ODETerm(kick), ODETerm(first_kick)
    tangents = jax.random.normal(jax.random.PRNGKey(42), guess_ic.shape)
    tangents = jax.tree.unflatten(jax.tree.structure(guess_ic), (tangents,))
    if adjoint == "finite_diff":

        def fn(guess_ic):
            val, _ = nbody(
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
    print(f"Running for EfficientLeapFrog with adjoint {adjoint} and baseline tol {base_line_tol}")
    for num_steps in tqdm(jnp.arange(10, 100, 10).tolist()):
        grad, num_steps = constant_steps_model(observable, adjoint, num_steps)
        gradients.append(grad)
        steps.append(num_steps)
    return gradients, steps


base_fpm_grad, base_fpm_steps = constant_steps_model(obs_fpm_field, CHECKPOINT, num_steps=90)
fpm_grads_DTO, fpm_steps_DTO = generate_gradient_data(obs_fpm_field, CHECKPOINT)
fpm_grads_REV, fpm_steps_REV = generate_gradient_data(obs_fpm_field, REVERSE)
fpm_grads_fd, _ = generate_gradient_data(obs_fpm_field, "finite_diff")


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_context("paper")

# Compute absolute errors
fpm_error_DTO = [abs(base_fpm_grad - grad) for grad in fpm_grads_DTO]
fpm_error_REV = [abs(base_fpm_grad - grad) for grad in fpm_grads_REV]

# Compute errors relative to finite differences
fpm_error_FD_DTO = [
    abs(fd_grad - dto_grad) for fd_grad, dto_grad in zip(fpm_grads_fd, fpm_grads_DTO)
]
fpm_error_FD_REV = [
    abs(fd_grad - otd_grad) for fd_grad, otd_grad in zip(fpm_grads_fd, fpm_grads_REV)
]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot Gradient Error vs Steps in Subplot 1
ax1.plot(
    fpm_steps_DTO,
    fpm_error_DTO,
    marker="o",
    linestyle="-",
    label="EfficientLeapFrog DTO",
)
ax1.plot(
    fpm_steps_REV,
    fpm_error_REV,
    marker="s",
    linestyle="--",
    label="EfficientLeapFrog Reverse",
)

# Customize Subplot 1
ax1.set_xlabel("Steps")
ax1.set_ylabel("Initial Field Gradient Error Gradient Error (|base - obtained|)")
ax1.set_title("Initial Field Gradient Error Gradient Error vs Steps")
ax1.legend()
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

# Plot Finite Difference Comparison in Subplot 2
ax2.plot(fpm_steps_DTO, fpm_error_FD_DTO, marker="o", linestyle="-", label="DTO vs FD")
ax2.plot(fpm_steps_REV, fpm_error_FD_REV, marker="s", linestyle="--", label="Reverse vs FD")

# Customize Subplot 2
# ax2.set_yscale("log")  # Log scale for gradient errors
# ax2.set_ylim(1e-8, 1e-3)
ax2.set_xlabel("Steps")
ax2.set_ylabel("Initial Field Gradient Error (|finite difference - obtained|)")
ax2.set_title("Comparison with Finite Difference")
ax2.legend()
ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

# Adjust layout and save
plt.tight_layout()
plt.savefig("plots/GS_FPM_initial_field_gradient_error.pdf", dpi=600, transparent=True)
plt.show()

data_to_save = {
    "base_fpm_grad": base_fpm_grad,
    "fpm_grads_DTO": fpm_grads_DTO,
    "fpm_grads_REV": fpm_grads_REV,
    "fpm_grads_fd": fpm_grads_fd,
    "fpm_steps_DTO": fpm_steps_DTO,
    "fpm_steps_REV": fpm_steps_REV,
}

np.savez("GS_FPM_initial_field_gradient_error.npz", **data_to_save)


# ## Third Study: Memory Usage in Gradient Computation  
# 
# We analyze the memory footprint of gradient computation by varying the number of checkpoints in the adjoint method.  
# 
# A baseline memory measurement is taken for both Checkpoint Adjoint (DTO) and Reverse-Mode Differentiation (REV) using the Efficient LeapFrog solver with 90 steps. We then evaluate memory usage across different checkpointing strategies.  
# 
# This study helps assess the trade-offs between memory consumption and gradient accuracy in large-scale simulations.  
# 

# In[33]:


t1, t0 = 1.0, 0.1
solver = SemiImplicitEuler()


def constant_steps_memory(solver, observable, adjoint, num_steps, checkpoint=100):
    step_size = (t1 - t0) / num_steps
    drift, kick, first_kick = symplectic_fpm_ode(mesh_shape, step_size, paint_absolute_pos=False)
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
    EfficientLeapFrog(), obs_fpm_field, CHECKPOINT, num_steps=90
)
fpm_grad_base_REV, fpm_steps_base_REV, fpm_memories_base_REV = constant_steps_memory(
    EfficientLeapFrog(), obs_fpm_field, REVERSE, num_steps=90
)

fpm_grads_DTO, fpm_steps_DTO, fpm_checkpoints_DTO, fpm_memories_DTO = generate_memory_data(
    EfficientLeapFrog(), obs_fpm_field
)


# In[ ]:


# Data Preparation
checkpoints = fpm_checkpoints_DTO

# Error baselines
fpm_grad_base_REV_err = abs(
    fpm_grad_base_DTO - fpm_grad_base_REV
)  # Baseline absolute error for OTD

# Errors for DTO and OTD
fpm_error_DTO = [abs(fpm_grad_base_DTO - grad) for grad in fpm_grads_DTO]
mean_err = sum(fpm_error_DTO) / len(fpm_error_DTO)
fpm_error_DTO = [mean_err if err == 0 else err for err in fpm_error_DTO]

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Define colors
error_color = "tab:blue"
memory_color = "tab:orange"

# Primary Y-axis: Absolute Error
ax1.set_xlabel("Checkpoints")
ax1.set_ylabel("Absolute Error (log scale)", color=error_color)

ax1.axhline(
    fpm_grad_base_REV_err,
    color=error_color,
    linestyle="-.",
    label=f"Reverse Baseline steps {fpm_steps_base_DTO}",
)
# Plot errors
ax1.plot(
    checkpoints,
    fpm_error_DTO,
    marker="o",
    color=error_color,
    label=f"FPM Error (DTO) steps {fpm_steps_base_DTO}",
)


# Add horizontal line for baseline errors
ax1.tick_params(axis="y", labelcolor=error_color)

ax1.legend(loc="upper left")
ax1.set_yscale("log")
# Secondary Y-axis: Memory Usage
ax2 = ax1.twinx()  # Create a secondary y-axis
ax2.set_ylabel("Memory Usage (bytes)", color=memory_color)

# Plot memory usage
ax2.plot(
    checkpoints,
    fpm_memories_DTO,
    marker="o",
    linestyle="-",
    color=memory_color,
    label="Memory (DTO)",
)

# Add horizontal line for memory baselines
ax2.axhline(fpm_memories_base_REV, color=memory_color, linestyle=":", label="Memory (Reverse)")

ax2.tick_params(axis="y", labelcolor=memory_color)
ax2.legend(loc="upper right")

# Title and grid
plt.title("Memory Usage and Absolute Error vs Checkpoints")
plt.grid(which="both", linestyle="--", linewidth=0.5)  # Add x-y grid
plt.minorticks_on()  # Enable minor ticks for finer grid
plt.grid(which="minor", linestyle=":", linewidth=0.5)  # Minor grid lines

# Show or Save
plt.tight_layout()
plt.savefig("plots/GS_FPM_memory_usage_and_error.pdf", dpi=600, transparent=True)
plt.show()

data_to_save = {
    "checkpoints": fpm_checkpoints_DTO,
    "fpm_grad_base_REV_err": fpm_grad_base_REV_err,
    "fpm_error_DTO": pm_error_DTO,
    "fpm_steps_base_DTO": m_steps_base_DTO,
    "fpm_memories_DTO": pm_memories_DTO,
    "fpm_memories_base_REV": fpm_memories_base_REV,
}

np.savez("GS_FPM_memory_usage_and_error.npz", **data_to_save)

