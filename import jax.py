import jax
import jax.numpy as jnp
from inspect import signature
from functools import partial
jax.clear_caches()

def f(a , b , c):
    print(f"a = {a}\nb = {b}\nc = {c}")
    return a * b + c

static_argnums = (1, 2)

fj = jax.jit(f, static_argnums=static_argnums)

def fv_impl(d , *args , **kwargs):
    print(f"d = {d}")
    return d * f(*args, **kwargs)


def fv(d , *args , **kwargs):
    print(f"d = {d}\nargs = {args}\nkwargs = {kwargs}")
    n_static_argnums = jax.tree.map(lambda x : x+1 , static_argnums)
    n_static_argnums = (0 , *n_static_argnums)

    return jax.jit(fv_impl, static_argnums=n_static_argnums)(d , *args, **kwargs)


fj(2, 3, 4)
print(f"===")
fv(1 , 2 , 3 , 4)

# Example call
from inspect import signature

def strip_static_args(func, static_argnums, *args, **kwargs):

    # Get the function's signature
    sig = signature(func)
    bound_args = sig.bind_partial(*args, **kwargs)
    
    # Identify static and non-static arguments
    static_args = {k: bound_args.arguments[k] for i, k in enumerate(sig.parameters) if i in static_argnums}
    non_static_args = {k: bound_args.arguments[k] for i, k in enumerate(sig.parameters) if i not in static_argnums and k in bound_args.arguments}
    
    # Bind static arguments using partial
    stripped_func = partial(func, **static_args)
    
    return stripped_func, non_static_args

# Example function
def f(a, b, c):
    print(f"a = {a}, b = {b}, c = {c}")
    return a * b + c

# Original function
static_argnums = (0, 2)

def fn(*args, **kwargs):
    stripped_func, remaining_args = strip_static_args(f, static_argnums, *args, **kwargs)
    return stripped_func, remaining_args

# Example call
stripped_function, non_static_args = fn(1, 2, 3)
print("Non-static arguments:", non_static_args)
print("Stripped function result:", stripped_function(**non_static_args))
