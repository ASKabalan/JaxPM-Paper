import jax.numpy as jnp
import jax
from functools import partial
import operator
import numpy as np
from jax._src import dtypes as _dtypes
from tqdm import tqdm


EPS = 1e-8


def _dtype(x):
  if hasattr(x, 'dtype'):
    return x.dtype
  elif type(x) in _dtypes.python_scalar_dtypes:
    return np.dtype(_dtypes.python_scalar_dtypes[type(x)])
  else:
    return np.asarray(x).dtype


def inner_prod(xs, ys):
  def contract(x, y):
    return np.real(np.dot(np.conj(x).reshape(-1), y.reshape(-1)))
  return tree_reduce(np.add, tree_map(contract, xs, ys))


def is_python_scalar(val):
  return not isinstance(val, np.generic) and isinstance(val, (bool, int, float, complex))


def _safe_subtract(x, y, *, dtype):
  """Subtraction that with `inf - inf == 0` semantics."""
  with np.errstate(invalid='ignore'):
    return np.where(np.equal(x, y), np.array(0, dtype),
                    np.subtract(x, y, dtype=dtype))


def _preserve_input_types(f):
  def wrapped(*args):
    dtype = _dtype(args[0])
    result = np.array(f(*args), dtype=dtype)
    if all(is_python_scalar(arg) for arg in args):
      result = result.item()
    return result
  return wrapped


# Partial functions for arithmetic
add = partial(jax.tree_map, _preserve_input_types(operator.add))
sub = partial(jax.tree_map, _preserve_input_types(operator.sub))
safe_sub = partial(jax.tree_map, lambda x, y: _safe_subtract(x, y, dtype=_dtype(x)))


def scalar_mul(xs, a):
  def mul(x):
    dtype = _dtype(x)
    result = np.multiply(x, np.array(a, dtype=dtype), dtype=dtype)
    return result.item() if is_python_scalar(x) else result
  return jax.tree_map(mul, xs)


def numerical_jvp(f, primals, tangents, eps=EPS):
    """Compute the numerical Jacobian-vector product."""
    delta = scalar_mul(tangents, eps)
    f_pos = f(*add(primals, delta))
    f_neg = f(*sub(primals, delta))
    return scalar_mul(safe_sub(f_pos, f_neg), 0.5 / eps)


def finite_diff(f , x , eps = EPS):
    grads = jax.tree.map(jnp.zeros_like, x)

    indices = jnp.moveaxis(jnp.indices(x.shape) , 0, -1).reshape(-1, x.ndim)

    for index in tqdm(indices):
      tangets = jax.tree.map(jnp.zeros_like, x)
      index = tuple(index)

      tangets = jax.tree.map(lambda t: t.at[index].set(1.0) , tangets)

      jvp_res = numerical_jvp(f , (x, ) , (tangets, ) , eps = eps)
      grads = jax.tree.map(lambda g: g.at[index].set(jvp_res) , grads)

    return grads



