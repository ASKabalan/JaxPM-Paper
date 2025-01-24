# Imports
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from jax import lax
import jax.numpy as jnp

from jax.experimental.shard_map import shard_map
from functools import partial
import jax

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

mesh = Mesh(jax.devices(), ("x"))
sharding = NamedSharding(mesh, P("x"))

x = jnp.arange(8).astype(jnp.float32)
x = lax.with_sharding_constraint(x, sharding)
batched_x = jnp.stack([x, 2 * x])

def scaler_fn_shardmap(x):
    @partial(shard_map, mesh=mesh, in_specs=P("x"), out_specs=P("x"))
    def inner(x):
        return lax.pmean(x, "x")

    return jnp.mean(inner(x))

def scaler_fn_with_constraint(x):
    x = lax.with_sharding_constraint(x, sharding)
    return jnp.mean(x)

def scaler_fn(x):
    return jnp.mean(x)

def fn_shardmap(x):
    @partial(shard_map, mesh=mesh, in_specs=P("x"), out_specs=P("x"))
    def inner(x):
        return 2 * x

    return inner(x)

def fn_with_constraint(x):
    return lax.with_sharding_constraint(2 * x, sharding)

def fn(x):
    return 2 * x


def trace(fn, name, scaler=False):
    print("=" * 50)
    f_pass = fn(x)
    print(f"{name} Forward pass sharding {fn(x).sharding}")
    print(f"{name} Vmapped sharding {jax.vmap(fn)(batched_x).sharding}")
    if scaler:
        print(f"{name} Gradient sharding {jax.grad(fn)(x).sharding}")
    else:
        print(
            f"{name} Gradient sharding {jax.grad(lambda x:jnp.sum(fn(x)))(x).sharding}"
        )
    print(f"{name} Jacrev sharding {jax.jacrev(fn)(x).sharding}")
    print(f"{name} Jacfwd sharding {jax.jacfwd(fn)(x).sharding}")
    print(f"{name} Jvp sharding {jax.jvp(fn , (x,) , (jnp.ones_like(x),))[1].sharding}")
    print(f"{name} Vjp sharding {jax.vjp(fn , x)[1](f_pass)[0].sharding}")


trace(scaler_fn_shardmap, "Scaler Shardmapped" , scaler=True)
trace(scaler_fn_with_constraint, "Scaler With Constraint" , scaler=True)
trace(scaler_fn, "Scaler No Constraint" , scaler=True)

trace(fn_shardmap, "Shardmapped")
trace(fn_with_constraint, "With Constraint")
trace(fn, "No Constraint")

