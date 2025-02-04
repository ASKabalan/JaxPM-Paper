import jax
import jax.numpy as jnp
from jax import custom_vjp
from diffrax import ODETerm
from functools import partial
from typing import Any, Tuple, Optional
from diffrax import AbstractSolver, SaveAt
import equinox as eqx
from jax._src.numpy.util import promote_dtypes_inexact


def handle_saveat(save_at: SaveAt, t0: float, t1: float) -> SaveAt:
    """
    Ensures that `save_at.subs.ts` is a valid JAX array of times,
    optionally including `t0` and/or `t1` if they are requested.

    This function:
      - Verifies that there is at least one kind of save time (t0, t1, or ts).
      - Replaces `None` for `ts` with an empty array.
      - If `save_at.subs.t0` is True, prepends `t0` to `ts`.
      - If `save_at.subs.t1` is True, appends `t1` to `ts`.

    Args:
        save_at: A diffrax SaveAt object that may have subs.t0, subs.t1, subs.ts.
        t0: Initial time of integration.
        t1: Final time of integration.

    Returns:
        A `SaveAt` object whose `subs.ts` is guaranteed to be a JAX array,
        possibly updated with t0, t1.
    """
    assert save_at.subs is not None, "You must set at least one of t0, t1, or ts."

    def _where_subs_ts(s):
        return s.subs.ts

    # Replace None with an empty array to ensure concatenation is valid
    if save_at.subs.ts is None:
        save_at = eqx.tree_at(_where_subs_ts, save_at, replace=jnp.array([]))

    # If t0==True and t1==True, prepend t0 and append t1
    if save_at.subs.t0 and save_at.subs.t1:
        save_at = eqx.tree_at(
            _where_subs_ts,
            save_at,
            replace_fn=lambda x: jnp.concatenate((jnp.array([t0]), x, jnp.array([t1]))),
        )
    # If only t0==True, prepend t0
    elif save_at.subs.t0:
        save_at = eqx.tree_at(
            _where_subs_ts,
            save_at,
            replace_fn=lambda x: jnp.concatenate((jnp.array([t0]), x)),
        )
    # If only t1==True, append t1
    elif save_at.subs.t1:
        save_at = eqx.tree_at(
            _where_subs_ts,
            save_at,
            replace_fn=lambda x: jnp.concatenate((x, jnp.array([t1]))),
        )

    return save_at

def integrate(
    terms: Tuple[ODETerm, ...],
    solver: AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
    y0: Any,
    args: Any,
    saveat: Optional[SaveAt] = SaveAt(t1=True),
) -> Any:
    saveat = handle_saveat(saveat, t0, t1)
    save_y = saveat.subs.fn
    ts = saveat.subs.ts
    ts, = promote_dtypes_inexact(ts)
    y0_args_ts = (y0, args, ts)
    return integrate_impl(y0_args_ts, terms=terms, solver=solver, t0=t0, t1=t1, dt0=dt0, save_y=save_y)

@eqx.filter_custom_vjp
def integrate_impl(
    y0_args_ts: Any,
    *,
    terms: Tuple[ODETerm, ...],
    solver: AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
    save_y: Any,
) -> Any:
    """
    Integrate an ODE system from time t0 to t1 using Diffrax's solver,
    then return the solution snapshots computed by `saveat`.

    Args:
        y0: Initial state of the system. Can be a scalar, array, or a PyTree.
        args: Parameters for the ODE system. Can be any PyTree.
        terms: A tuple of `ODETerm`s defining the system dynamics.
        solver: A `diffrax.AbstractSolver` specifying the integration method.
        t0: Initial integration time.
        t1: Final integration time.
        dt0: Step size for each integration step inside a loop.
        saveat: A `diffrax.SaveAt` specifying when/how the solution is saved.

    Returns:
        The collection of solution snapshots at the times specified by `saveat`.
    """
    y0 , args , ts = y0_args_ts
    args = jax.tree.map(jnp.asarray, args)

    def inner_forward_step(carry):
        y, args_, tc, t1 = carry
        t_next = tc + dt0
        # Perform a single integration step
        # The solver returns (y_next, solver_state, new_t, result, made_jump)
        y_next, _, _, _, _ = solver.step(
            terms, tc, t_next, y, args_, solver_state=None, made_jump=False
        )
        return (y_next, args_, t_next, t1)

    def inner_forward_cond(carry):
        # Condition for continuing the while_loop: current time < final time
        _, _, tc, t1 = carry
        return tc < t1

    def outer_forward_step(outer_carry, t1):
        """
        For each time in `fwd_save_at.subs.ts`, we integrate from t0 up to that time,
        then apply `fwd_save_at.subs.fn(...)`.
        """
        y, args_, t0 = outer_carry
        inner_carry = (y, args_, t0, t1)
        y, _, _, _ = jax.lax.while_loop(inner_forward_cond, inner_forward_step, inner_carry)
        outer_carry = (y, args_, t1)
        # Apply the user-defined function at this "snapshot" time
        return outer_carry, save_y(t1, y, args_)

    # Initialize carry
    init_carry = (y0, args, t0)

    # The outer scan runs over each requested snapshot time
    _, ys_final = jax.lax.scan(outer_forward_step, init_carry, ts)

    # Return snapshots plus final state+args
    return ys_final
   

@integrate_impl.def_fwd
def integrate_fwd(
    perturbed : Tuple[Any, Any],
    y0_args_ts: Any,
    *,
    terms: Tuple[ODETerm, ...],
    solver: AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
    save_y: Any,
) -> Tuple[Any, Tuple[Any, Any]]:
    """
    Forward pass for `integrate`. Computes the solution at the times requested
    by `saveat` in a piecewise manner using a fixed step size dt0.

    Internally, it uses a two-level structure:
      - an "inner" `while_loop` that steps from t0 to t1 in increments of dt0,
      - an "outer" `scan` loop that triggers at each time in `saveat.subs.ts`.

    Args:
        y0: Initial state of the system.
        args: Parameters for the ODE.
        terms: A tuple of `ODETerm`s describing the system's dynamics.
        solver: The chosen solver implementing `.step` and `.reverse`.
        t0: Initial time.
        t1: Final time.
        dt0: Step size for integration increments.
        saveat: SaveAt object with times, t0/t1 flags, etc.

    Returns:
        ys_final: The values of `saveat.subs.fn(t, y, args)` at each requested time.
        (y_final, args): Final state and arguments (needed for the backward pass).
    """
    ys_final = integrate_impl(y0_args_ts, terms=terms, solver=solver, t0=t0, t1=t1, dt0=dt0, save_y=save_y)
    y_final = jax.tree.map(lambda x: x[-1], ys_final)

    # Return snapshots plus final state+args
    return ys_final, y_final

@integrate_impl.def_bwd
def integrate_bwd(
    residuals: Tuple[Any, Tuple[Any, Any]],
    cotangents: Any,
    perturbed: Tuple[Any, Any],
    y0_args_ts: Any,
    *,
    terms: Tuple[ODETerm, ...],
    solver: AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
    save_y: Any,
) -> Tuple[Any, Any]:
    """
    Backward pass for `integrate`. Computes adjoints for y0 and args only.

    We declared `nondiff_argnums=(2,3,4,5,6,7)`, so `terms, solver, t0, t1, dt0, saveat`
    are not differentiable. Consequently, this backward function only needs to return
    partial derivatives w.r.t. the first two arguments (y0 and args).

    Args:
        terms: The tuple of ODETerm objects defining the system (non-differentiable).
        solver: The solver used for forward steps (non-differentiable).
        t0: Initial time (non-differentiable).
        t1: Final time (non-differentiable).
        dt0: Step size (non-differentiable).
        saveat: The SaveAt object used for the forward pass (non-differentiable).
        residuals: The (y_final, args) from the forward pass.
        cotangents: The cotangent (adjoint) wrt. the output of `integrate_fwd`.

    Returns:
        A 2-tuple (adj_y, adj_args), i.e. partial derivatives wrt. y0 and args.
    """

    _ , args , ts = y0_args_ts
    y_final = residuals
    ys_ct = cotangents  # Gradient w.r.t. the forward pass snapshots

    # Initialize adjoint for args and final y
    args = jax.tree.map(jnp.asarray, args)

    diff_args , nondiff_args = eqx.partition(args , eqx.is_inexact_array_like)

    adj_y = jax.tree_map(lambda x: jnp.zeros_like(x), y_final)
    adj_args = jax.tree_map(lambda x: jnp.zeros_like(x), diff_args)
    adj_ts = jax.tree_map(lambda x: jnp.zeros_like(x), ts)

    def inner_backward_step(carry):
        """
        Reverses a single forward integration step from `tc` down to `tc - dt0`,
        and updates the adjoints accordingly.
        """
        y, diff_args, adj_y, adj_args, t0, tc = carry
    
        t_prev = tc - dt0
        # Reverse the forward step
        y_prev = solver.reverse(
            terms, t_prev, tc, y, args, solver_state=None, made_jump=False
        )

        # We'll differentiate w.r.t. "forward step" to get partial derivatives.
        def _to_vjp(y, diff_args):
            args = eqx.combine(diff_args , nondiff_args)
            y_next, _, _, _, _ = solver.step(
                terms, t_prev, tc, y, args, solver_state=None, made_jump=False
            )
            return y_next

        _, f_vjp = jax.vjp(_to_vjp, y_prev, diff_args)
        adj_y, new_adj_args = f_vjp(adj_y)

        # Accumulate into existing adjoints
        new_d_args = jax.tree.map(jnp.add, new_adj_args, adj_args)

        return (y_prev, diff_args, adj_y, new_d_args, t0, t_prev)

    def inner_backward_cond(carry):
        """
        We continue stepping backward while `tc` remains above t0.
        """
        _, _, _, _, t0, tc = carry
        return tc > t0

    def outer_backward_step(outer_carry, vals):
        """
        For each snapshot time in reverse, add the snapshot cotangent to the
        current adjoint, then step backward until we reach the previous snapshot.
        """
        y_ct, t0 = vals
        y, diff_args, adj_y, adj_args,adj_ts, tc = outer_carry

        # Differentiate the "save function" at snapshot time `tc_`:
        def _to_vjp(tc , y, diff_args):
            args = eqx.combine(diff_args , nondiff_args)
            return save_y(tc, y, args)

        _, f_vjp = jax.vjp(_to_vjp, tc , y, diff_args)
        new_adj_ts , new_adj_y, new_adj_args = f_vjp(y_ct)

        # Accumulate
        adj_y = jax.tree.map(jnp.add, adj_y, new_adj_y)
        adj_args = jax.tree.map(jnp.add, adj_args, new_adj_args)
        adj_ts = jax.tree.map(jnp.add, adj_ts, new_adj_ts)

        # Now step backward in increments of dt0 from the current snapshot time down to snap_t0_
        inner_carry = (y, diff_args, adj_y, adj_args, t0, tc)
        y_prev, diff_args, adj_y, adj_args, tc, _ = jax.lax.while_loop(
            inner_backward_cond, inner_backward_step, inner_carry
        )

        outer_carry = (y_prev, diff_args, adj_y, adj_args, adj_ts ,  tc)
        return outer_carry, None

    # Reverse through the snapshot times

    # Define t1 as the last snapshot if available
    t1 = ts[-1]

    # Shift the array of snapshot times to incorporate the initial time
    t_steps = jnp.concatenate((jnp.asarray([t0]), ts[:-1]))

    # Initial carry is the final state and final adjoint
    init_carry = (y_final, diff_args, adj_y, adj_args ,adj_ts, t1)

    # We'll pair up the cotangents with the times
    vals = (ys_ct, t_steps)

    # Perform the reverse scan over the snapshots
    (_, _, adj_y, adj_args,adj_ts, _), _ = jax.lax.scan(
        outer_backward_step, init_carry, vals, reverse=True
    )
    zero_nondiff = jax.tree_map(jnp.zeros_like, nondiff_args)
    adj_args = eqx.combine(adj_args , zero_nondiff)
    jax.debug.print("adj_ts {a} " , a=adj_ts)

    # Return the adjoints for y0 and args. The rest are placeholders (None)
    # matching the custom_vjp signature convention.
    return (adj_y, adj_args, adj_ts)



def scan_integrate(
    terms: Tuple[ODETerm, ...],
    solver: AbstractSolver,
    t0: float,
    t1: float,
    dt0: float,
    y0: Any,
    args: Any,
    saveat: Optional[SaveAt] = None,
) -> Any:
    """
    An alternative "vanilla" scanning integrator that steps from t0 to t1 in equal
    increments of dt0, then picks out the solution at the times requested by `saveat`.

    This is primarily meant for comparison or debugging against the main integrator.
    It stores the solution at every single step, which can be memory-intensive
    for long or fine-grained integrations.

    Args:
        y0: Initial state of the system.
        args: Parameters for the ODE.
        terms: A tuple of `ODETerm`s describing the system's dynamics.
        solver: The chosen solver with a `.step` implementation.
        t0: Start time.
        t1: End time.
        dt0: Step size for each forward integration step.
        saveat: SaveAt object specifying snapshot times.

    Returns:
        A PyTree of solutions at the requested snapshot times.
    """
    def forward_step(carry, t):
        y, args = carry
        t_next = t + dt0
        # Perform a single integration step
        y_next, _, _, _, _ = solver.step(
            terms, t, t_next, y, args, solver_state=None, made_jump=False
        )
        # Store current state and time for backward pass
        return (y_next, args), y_next

    if saveat is None:
      saveat = SaveAt(ts=[t1])

    # Prepare the main scan
    init_carry = (y0, args)
    t_steps = jnp.arange(t0, t1, dt0)

    # Perform the integration using jax.lax.scan
    (_, _), ys = jax.lax.scan(forward_step, init_carry, t_steps)

    # Prepend the initial state to the scanned states
    ys = jax.tree.map(lambda y0 , ys :jnp.concatenate((jnp.asarray(y0)[None , ...] , ys) , axis=0) , y0 , ys)


    # Extract the requested snapshot times from the integrator results
    saveat = handle_saveat(saveat , t0 , t1)
    snapshots = saveat.subs.ts
    snapshots = ((snapshots - t0) / dt0).astype(jnp.int32)
    return jax.tree.map(lambda x : x[snapshots] , ys)
