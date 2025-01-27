import jax
import jax.numpy as jnp
from jax import custom_vjp
from diffrax import ODETerm, SemiImplicitEuler, SaveAt
from functools import partial
from typing import Any, Tuple
from functools import partial
from diffrax import AbstractSolver


@partial(custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6))
def integrate(
    y0: Any,
    args: Any,
    terms: Tuple[ODETerm, ...],
    solver: AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
) -> Any:
    """
    Integrate an ODE system from time t0 to t1 using Diffrax's solver.

    Args:
        y0: Initial state of the system. Can be a scalar, array, or PyTree.
        args: Parameters for the ODE system. Can be any structure compatible with the ODE terms.
        terms: A tuple of ODETerm instances defining the ODE system.
        solver: A Diffrax solver instance (e.g., SemiImplicitEuler()).
        t0: Initial time.
        t1: Final time.
        dt0: Time step size.
        saveat: SaveAt instance specifying when to save intermediate results.

    Returns:
        y_final: Final state after integration from t0 to t1.
    """

    def forward_step(carry, t):
        y, args = carry
        t_next = t + dt0
        # Perform a single integration step
        y_next, _, _, _, _ = solver.step(terms,
                                         t,
                                         t_next,
                                         y,
                                         args,
                                         solver_state=None,
                                         made_jump=False)
        # Store current state and time for backward pass
        return (y_next, args), None

    # Initialize carry with initial state and parameters
    init_carry = (y0, args)
    # Create an array of time steps
    t_steps = jnp.arange(t0, t1, dt0)
    # Perform the integration using jax.lax.scan
    (y_final , _), _ = jax.lax.scan(forward_step, init_carry, t_steps)
    return y_final


def integrate_fwd(
    y0: Any,
    args: Any,
    terms: Tuple[ODETerm, ...],
    solver: AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
) -> Tuple[Any, Tuple[Any, Any]]:
    """
    Forward pass for the custom VJP of the integrate function.

    Stores all intermediate states required for the backward pass.

    Args:
        y0: Initial state of the system.
        args: Parameters for the ODE system.
        terms: A tuple of ODETerm instances defining the ODE system.
        solver: A Diffrax solver instance.
        t0: Initial time.
        t1: Final time.
        dt0: Time step size.
        saveat: SaveAt instance.

    Returns:
        y_final: Final state after integration.
        intermediates: Tuple containing all intermediate states and times for backward pass.
    """
    y_final = integrate(y0, args, terms, solver, t0, t1, dt0)

    return y_final, (y_final, args)


def integrate_bwd(terms: Tuple[ODETerm, ...], 
                  solver: Any, 
                  t0: float,
                   t1: float,
                  dt0: float, 
                res: Tuple[Any, Tuple[Any, Any]],
                  ct: Any) -> Tuple[Any, Any, None, None, None, None, None]:
    """
    Backward pass for the custom VJP of the integrate function.

    Reconstructs the computational graph to compute gradients.

    Args:
        res: Tuple containing all intermediates from the forward pass.
             Format: (ys, ts), where ys are the states and ts are the corresponding times.
        ct: Cotangent (gradient) with respect to the output y_final.

    Returns:
        Gradients with respect to the inputs y0 and args.
        Gradients with respect to terms, solver, t0, t1, dt0, saveat are None as they are treated as constants.
    """
    y1, args = res
    dy_final = ct  # Gradient w.r.t y_final
    d_args = jax.tree_map(lambda x: jnp.zeros_like(x), args)

    def step(carry, t1):

        y, dy, d_args = carry
        t0 = t1 - dt0
        y_prev = solver.reverse(terms,
                                        t0,
                                        t1,
                                        y,
                                        args,
                                        solver_state=None,
                                        made_jump=False)

        def _to_vjp(y, z_args):
            y_next, _, _, _, _ = solver.step(terms,
                                                     t0,
                                                     t1,
                                                     y,
                                                     z_args,
                                                     solver_state=None,
                                                     made_jump=False)
            return y_next

        _, f_vjp = jax.vjp(_to_vjp, y_prev, args)

        dy, dargs = f_vjp(dy)

        new_d_args = jax.tree.map(jnp.add, d_args, dargs)

        return (y_prev, dy, new_d_args), None

    init_carry = (y1, dy_final, d_args)
    (y, d_y, d_args), _ = jax.lax.scan(step, init_carry,
                                      jnp.arange(t1, t0, -dt0))

    return d_y, d_args


# Register the forward and backward functions with the custom VJP
integrate.defvjp(integrate_fwd, integrate_bwd)
