import jax
import jax.numpy as jnp
import pytest
from diffrax import (
    ODETerm,
    RecursiveCheckpointAdjoint,
    SaveAt,
    diffeqsolve,
)

from .integrate import integrate, scan_integrate
from .semi_implicite_euler import SemiImplicitEuler

jax.config.update("jax_enable_x64", True)


def check_tree(x, y):
    return jax.tree.all(jax.tree.map(lambda x, y: jnp.allclose(x, y), x, y))


def mse(x, y):
    return jax.tree.reduce(
        lambda x, y: x + y, jax.tree.map(lambda x, y: jnp.mean((x - y) ** 2), x, y)
    )


def f(t, x, z):
    return x + z[0] * z[1]


def g(t, y, z):
    return y + z[0] * z[1]


def fg(t, x_y, z):
    x, y = x_y
    out = f(t, x, z), g(t, y, z)
    return jnp.stack(out, axis=0)


def save_ys(t, y, z):
    y_red = jax.tree.reduce(lambda x, y: x + y, y)
    z_red = jax.tree.reduce(lambda x, y: x + y, z)
    return jnp.sum(y_red * t**2 * z_red)


def reverse_integrate(ode_terms, solver, t0, t1, dt0, y0, args, ts, save_t0, save_t1):
    saveat = SaveAt(ts=ts, t0=save_t0, t1=save_t1, fn=save_ys)
    return integrate(ode_terms, solver, t0, t1, dt0, y0, args, saveat)


def diffrax_integrate(
    ode_terms, solver, t0, t1, dt0, y0, args, ts, save_t0, save_t1, checkpoints=10
):
    saveat = SaveAt(ts=ts, t0=save_t0, t1=save_t1, fn=save_ys)
    sol = diffeqsolve(
        ode_terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args,
        saveat=saveat,
        adjoint=RecursiveCheckpointAdjoint(checkpoints=checkpoints),
    )
    return sol.ys


def jax_integarate(ode_terms, solver, t0, t1, dt0, y0, args, ts, save_t0, save_t1):
    saveat = SaveAt(ts=ts, t0=save_t0, t1=save_t1, fn=save_ys)
    return scan_integrate(ode_terms, solver, t0, t1, dt0, y0, args, saveat=saveat)


t0 = 0.0
t1 = 10.0
dt0 = 2.0
ode_terms = (ODETerm(g), ODETerm(f))
solver = SemiImplicitEuler()

# Starting t0 and ending at t1
t0_t1 = jnp.arange(t0, t1 + dt0, dt0)
# Starting t0 and ending before t1
t0_tx = jnp.arange(t0, t1, dt0)
# Starting after t0 and ending at t1
tx_t1 = jnp.arange(t0 + dt0, t1 + dt0, dt0)
# Starting after t0 and ending before t1
tx_tx = jnp.arange(t0 + dt0, t1 - dt0, dt0)
# Getting steps starting from t0 and jumping steps by 2 * dt0 before t1
t0_tx_2 = jnp.arange(t0, t1 + dt0, 2 * dt0)
# Gettings steps starting after t0 and jumping steps by 2 * dt0 before t1
tx_tx_2 = jnp.arange(t0 + dt0, t1 + dt0, 2 * dt0)
# Getting steps starting from t0 and jumping steps by 2 * dt0 at t1
t0_t1_2 = jnp.arange(t0, t1 + dt0, 2 * dt0)
# Getting steps starting after t0 and jumping steps by 2 * dt0 at t1
tx_t1_2 = jnp.arange(t0 + dt0, t1 + dt0, 2 * dt0)


list_of_ts = [t0_t1, t0_tx, tx_t1, tx_tx, t0_tx_2, tx_tx_2, t0_t1_2, tx_t1_2]


@pytest.mark.parametrize("ts", list_of_ts)
@pytest.mark.parametrize("save_t0", [True, False])
@pytest.mark.parametrize("save_t1", [True, False])
def test_fwd(ts, save_t0, save_t1):
    y0 = (
        jax.random.normal(jax.random.PRNGKey(0), (3, 3, 3)),
        jax.random.normal(jax.random.PRNGKey(1), (3, 3, 3)),
    )
    args = (2.0, 5.0)

    diffrax_fwd = diffrax_integrate(ode_terms, solver, t0, t1, dt0, y0, args, ts, save_t0, save_t1)
    my_fwd = reverse_integrate(ode_terms, solver, t0, t1, dt0, y0, args, ts, save_t0, save_t1)
    jax_fwd = jax_integarate(ode_terms, solver, t0, t1, dt0, y0, args, ts, save_t0, save_t1)

    assert check_tree(my_fwd, jax_fwd), f"diffrax and my fwd are not equal {diffrax_fwd} {my_fwd}"
    assert check_tree(diffrax_fwd, jax_fwd), (
        f"diffrax and jax fwd are not equal {diffrax_fwd} {jax_fwd}"
    )


@pytest.mark.parametrize("ts", list_of_ts)
@pytest.mark.parametrize("save_t0", [True, False])
@pytest.mark.parametrize("save_t1", [True, False])
def test_y_arg_diff(ts, save_t0, save_t1):
    y0 = (
        jax.random.normal(jax.random.PRNGKey(0), (3, 3, 3)),
        jax.random.normal(jax.random.PRNGKey(1), (3, 3, 3)),
    )
    args = (2.0, 5.0)

    diffrax_fwd = jax.jacrev(diffrax_integrate, argnums=(5, 6), allow_int=True)(
        ode_terms, solver, t0, t1, dt0, y0, args, ts, save_t0, save_t1
    )
    my_fwd = jax.jacrev(reverse_integrate, argnums=(5, 6), allow_int=True)(
        ode_terms, solver, t0, t1, dt0, y0, args, ts, save_t0, save_t1
    )
    jax_fwd = jax.jacfwd(jax_integarate, argnums=(5, 6))(
        ode_terms, solver, t0, t1, dt0, y0, args, ts, save_t0, save_t1
    )

    assert check_tree(my_fwd, jax_fwd)
    assert check_tree(diffrax_fwd, jax_fwd)


@pytest.mark.parametrize("ts", list_of_ts)
@pytest.mark.parametrize("save_t0", [True, False])
@pytest.mark.parametrize("save_t1", [True, False])
def test_ts_diff(ts, save_t0, save_t1):
    if t1 in ts and save_t1:
        pytest.skip("t1 is in ts  with save_t1 flag True ... skipping test")

    y0 = (
        jax.random.normal(jax.random.PRNGKey(0), (3, 3, 3)),
        jax.random.normal(jax.random.PRNGKey(1), (3, 3, 3)),
    )
    args = (2.0, 5.0)

    my_fwd = jax.jacrev(reverse_integrate, argnums=(7), allow_int=True)(
        ode_terms, solver, t0, t1, dt0, y0, args, ts, save_t0, save_t1
    )
    jax_fwd = jax.jacfwd(jax_integarate, argnums=(7))(
        ode_terms, solver, t0, t1, dt0, y0, args, ts, save_t0, save_t1
    )

    assert check_tree(my_fwd, jax_fwd)
