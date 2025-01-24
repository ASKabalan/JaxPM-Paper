# Imports
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from functools import wraps
from jax import lax
from jax.interpreters import mlir, ad, batching
import jax.numpy as jnp
import jax
from jax.experimental.custom_partitioning import custom_partitioning
from jax._src import dispatch
import jax.extend as jex

from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P, Mesh
from functools import partial


mesh = Mesh(jax.devices(), ("x"))
sharding = NamedSharding(mesh, P("x"))
from functools import partial
import re
from jax import core, lax
from jax.interpreters import mlir, ad, batching
import jax
from jax._src import dispatch, mesh
from jax._src import custom_api_util
from jax import extend as jex
from jax.experimental.custom_partitioning import custom_partitioning
from inspect import signature
from jax.sharding import NamedSharding, PartitionSpec as P
import jax.numpy as jnp
from jax import jvp, vjp, jacfwd, jacrev


def strip_shaped_dtype_struct(arg_info):
    new_shape = arg_info.shape[1:]  # Remove the batch dimension
    new_spec = arg_info.sharding.spec[1:]  # Adjust the PartitionSpec
    new_sharding = NamedSharding(arg_info.sharding.mesh, P(*new_spec))
    return jax.ShapeDtypeStruct(
        new_shape, arg_info.dtype, sharding=new_sharding, weak_type=arg_info.weak_type
    )


def strip_shaped_array(shaped_array):
    return shaped_array.update(shape=shaped_array.shape[1:])


def add_batch_dimension_to_sharding(arg_info):
    # Add the batch dimension back to the shape and PartitionSpec
    new_spec = (None, *arg_info.spec)  # Add `None` to PartitionSpec
    new_sharding = NamedSharding(arg_info.mesh, P(*new_spec))
    return new_sharding


def argnames_from_argnums(signature, argnums):
    signature_parameters: list[str] = list(signature.parameters.keys())
    argnames = [signature_parameters[i] for i in argnums]

    return argnames


def strip_static_args(func, static_argnums, *args):
    # Get the function's signature
    sig = signature(func)
    bound_args = sig.bind_partial(*args)

    # Identify static and non-static arguments
    static_args = {
        k: bound_args.arguments[k]
        for i, k in enumerate(sig.parameters)
        if i in static_argnums
    }
    non_static_args = {
        k: bound_args.arguments[k]
        for i, k in enumerate(sig.parameters)
        if i not in static_argnums and k in bound_args.arguments
    }

    # Bind static arguments using partial
    stripped_func = partial(func, **static_args)

    return stripped_func, non_static_args


@custom_api_util.register_custom_decorator_type
class custom_spmd_rule:
    def __init__(self, fun, static_argnums=(), multiple_results=False):
        self.fun = fun
        self.static_argnums = static_argnums
        self.multiple_results = multiple_results

        # ============== PRIMITIVE ==============
        #       Declare primitive
        # ======================================
        self.primitive = jex.core.Primitive(fun.__name__)
        # This is needed for lowering custom spmd rule
        dispatch.prim_requires_devices_during_lowering.add(self.primitive)
        # Step 1: Define the Implementation and Abstract Evaluation
        self.primitive.def_impl(fun)

        def abstract_eval(*args, **kwargs):
            return jax.make_jaxpr(self.fun, static_argnums=self.static_argnums)(
                *args, **kwargs
            ).out_avals[0]

        self.primitive.def_abstract_eval(abstract_eval)

        # Functions to be registered
        self.partition = None
        self.infer_sharding_from_operands = None
        self.jvp_rule = None
        self.transpose_rule = None

    def def_partition(self, partition):
        self.partition = partition
        if self.infer_sharding_from_operands is not None:
            self.def_spmd_rule(partition, self.infer_sharding_from_operands)

    def def_infer_sharding(self, infer_sharding_from_operands):
        self.infer_sharding_from_operands = infer_sharding_from_operands
        if self.partition is not None:
            self.def_spmd_rule(self.partition, infer_sharding_from_operands)


    def def_spmd_rule(self, partition_rule, infer_sharding_rule):
        assert partition_rule is not None , "Partition rule is required"
        assert infer_sharding_rule is not None , "Infer sharding rule is required"

        paritioned_fn = custom_partitioning(
            self.fun, static_argnums=self.static_argnums
        )
        paritioned_fn.def_partition(
            infer_sharding_from_operands=infer_sharding_rule,
            partition=partition_rule,
        )

        # Prepare the vmap rule
        def v_fun(__batch_axis, *args):
            # Strip static arguments
            f, non_static_args = strip_static_args(self.fun, self.static_argnums, *args)

            if isinstance(__batch_axis, tuple) and len(__batch_axis) == 1:
                __batch_axis = __batch_axis[0]
            return jax.vmap(f, in_axes=__batch_axis)(**non_static_args)

        v_static_argnums = jax.tree.map(lambda x: x + 1, self.static_argnums)
        v_static_argnums = (0,) + v_static_argnums
        v_partitioned_fn = custom_partitioning(v_fun, static_argnums=v_static_argnums)

        def v_infer_sharding(*args):
            mesh, args_infos, result_infos = args[-3:]
            __batch_axis, *static_args = args[:-3]

            unbatch_arg_infos = jax.tree.map(strip_shaped_dtype_struct, args_infos)
            unbatch_result_infos = jax.tree.map(strip_shaped_array, result_infos)
            unbatch_output_sharding = infer_sharding_rule(
                *static_args, mesh, unbatch_arg_infos, unbatch_result_infos
            )
            batched_results = add_batch_dimension_to_sharding(unbatch_output_sharding)

            return batched_results

        def v_partition(*args):
            mesh, args_infos, result_infos = args[-3:]
            __batch_axis, *static_args = args[:-3]

            unbatch_arg_infos = jax.tree.map(strip_shaped_dtype_struct, args_infos)
            unbatch_result_infos = jax.tree.map(strip_shaped_dtype_struct, result_infos)
            input_mesh, impl, output_sharding, input_shardings = partition_rule(
                *static_args, mesh, unbatch_arg_infos, unbatch_result_infos
            )

            output_sharding = add_batch_dimension_to_sharding(output_sharding)
            input_shardings = jax.tree.map(
                add_batch_dimension_to_sharding, input_shardings
            )
            v_impl = jax.vmap(impl, in_axes=(__batch_axis))

            return input_mesh, v_impl, output_sharding, input_shardings

        v_partitioned_fn.def_partition(
            infer_sharding_from_operands=v_infer_sharding, partition=v_partition
        )

        def batching_rule(batched_args, batch_dims, *args, **kwargs):
            kwargs_flat = jax.tree.leaves(kwargs)

            return v_partitioned_fn(batch_dims, *batched_args, *args, *kwargs_flat), 0

        # ============== PRIMITIVE ==============
        #       Declare custom SPMD and batching rule
        # ======================================
        # Step 2: Register the Partitioned lowering and the batching rule
        mlir.register_lowering(
            self.primitive,
            mlir.lower_fun(paritioned_fn, multiple_results=self.multiple_results),
        )
        batching.primitive_batchers[self.primitive] = batching_rule

    def def_jvp_rule(self, jvp_rule):
        self.jvp_rule = jvp_rule
        ad.primitive_jvps[self.primitive] = jvp_rule

    def def_transpose_rule(self, transpose_rule):
        self.transpose_rule = transpose_rule
        ad.primitive_transposes[self.primitive] = transpose_rule

    def __call__(self, *args, **kwargs):
        def internal_call(*args, **kwargs):
            return self.primitive.bind(*args, **kwargs)

        return internal_call(*args, **kwargs)


@partial(custom_spmd_rule, static_argnums=(1,))
def double_fn(x, scalar):
    return x * scalar

@double_fn.def_infer_sharding
def infer_sharding_from_operands(scalar, mesh, arg_infos, result_infos):
    return arg_infos[0].sharding

@double_fn.def_partition
def partition(scaler, mesh, arg_infos, result_infos):
    input_sharding = arg_infos[0].sharding
    output_sharding = result_infos.sharding
    input_mesh = input_sharding.mesh

    def impl(operand):
        return scaler * operand

    return input_mesh, impl, output_sharding, (input_sharding,)

# Step 4: Define JVP Rule
@double_fn.def_jvp_rule
def double_prim_jvp_rule(primals, tangents, scalar):
    (x,) = primals
    (t,) = tangents
    # Forward computation
    primal_out = double_fn(x, scalar=scalar)

    # Tangent computation (reuse the primitive itself)
    tangent_out = double_fn(t, scalar=scalar)
    return primal_out, tangent_out


@double_fn.def_transpose_rule
def double_prim_transpose_rule(ct_out, x, scalar):
    ct_x = 2 * ct_out if ad.is_undefined_primal(x) else None
    return (ct_x,)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def double_fn_j(x, scalar=2):
    return double_fn(x, scalar=scalar)


@double_fn_j.defjvp
def double_fn_jvp(scalar, primals, tangents):
    (x,) = primals
    (t,) = tangents
    # Forward computation
    primal_out = double_fn(x, scalar=scalar)

    # Tangent computation (reuse the primitive itself)
    tangent_out = double_fn(t, scalar=scalar)
    return primal_out, tangent_out


x = jnp.arange(8).astype(jnp.float32)
x = lax.with_sharding_constraint(x, sharding)


batched_x = jnp.stack([x, 2 * x])




#trace(lambda x: double_fn(x, scalar=2), "custom_partial_double_fn")


a = jnp.arange(8*8*8).reshape(8,8,8).astype(jnp.float32)
a = lax.with_sharding_constraint(a, sharding)


from jaxdecomp import pfft3d,pifft3d , ShardedArray

def fn(a):
    return pfft3d(a).real
def ifn(a):
    return pifft3d(a).real

x = a
batched_x = jnp.stack([x, 2 * x])

@jax.jit
def s(a):
    y = pfft3d(a)
    y = (y * jax.tree.map(jnp.conjugate , y)).real.sum()
    return y.data

@jax.jit
def spmd_grad(a):
    y = pfft3d(a)
    y = (y * jax.tree.map(jnp.conjugate , y)).real.sum()
    return y


@jax.jit
def f_spmd_grad(a):
    y = jnp.fft.fftn(a)
    y = (y * jax.tree.map(jnp.conjugate , y)).real.sum()
    return y

def trace(fn, x , batched_x , name, scaler=False):
    print(f"=" * 50)
    f_pass = fn(x)
    print(f"{name} Forward pass sharding {fn(x).sharding}")
    #print(f"{name} Vmapped sharding {jax.vmap(fn)(batched_x).sharding}")
    if scaler:
        print(f"{name} Gradient sharding {jax.grad(spmd_grad)(x).sharding}")
    else:
        print(
            f"{name} Gradient sharding {jax.grad(lambda x:fn(x).sum())(x).sharding}"
        )
    #print(f"{name} Jacrev sharding {jax.jacrev(fn)(x).sharding}")
    #print(f"{name} Jacfwd sharding {jax.jacfwd(fn)(x).sharding}")
    #print(f"{name} Jvp sharding {jax.jvp(fn , (x,) , (jnp.ones_like(x),))[1].sharding}")
    print(f"{name} Vjp sharding {jax.vjp(fn , x)[1](f_pass)[0].sharding}")


trace(fn, x , batched_x, "pfft3d", scaler=True)
#trace(ifn, pfft3d(x) , batched_x,"pifft3d", scaler=True)



s = ShardedArray(x, sharding)
batched_s = jax.tree.map(lambda x , y , z : jnp.stack([x, y, z]) , s , 2*s , 3*s)

trace(fn, s, batched_s , "pfft3d", scaler=True)
s_k = pfft3d(s)
trace(ifn, s_k, batched_s , "pifft3d", scaler=True)