~~## Can we merge jaxPM~~

~~Je fais le solver FastPM sans l'assert après~~

~~- remove ODE Diffrax~~
~~- Workflow PyPi~~

## Leap Frog

Made actual Leap Frog solver in JAXPM
Works good

## Grad stability

Started working with grad stability

1 - JVP is needed

```py
@partial(jax.custom_jvp , non_diff_argnums=(1, 2))
def pfft(a , fft_type, adjoint=False):
    return pfft_impl(a, fft_type, adjoint)

@pfft.defjvp
def pfft_jvp(primals, tangents):
    x, fft_type, adjoint = primals
    x_tangent, _, _ = tangents

    # Compute the primal output
    primal_out = pfft_impl(x, fft_type=fft_type, adjoint=adjoint)

    # Non-linear tangent computation (hypothetical example)
    tangent_out =  pfft_impl(x_tangent, fft_type=fft_type, adjoint=not adjoint)

    return primal_out, tangent_out

```


2 - How do I check ? is my method good? Should I do like [callum](https://github.com/sammccallum/reversible/blob/master/experiments/stability/plot.py)

initial cond erreur L2 haut resolution et autre

SBI gradient bad not good MCMC has accept reject

pas daccept reject

importance sampling ()

plot different solver 
ajout un FastPM (avec des gradients) juste plot

dx * y = dy * x

Spectre Cross Correlation avec L2

descente de gradient avec bad grads

check 

dy = f(x) =D dy aganst df and dy agsainst dy10

send overleaf to FL

3 - Will do Recursive Reversible and Backsolve

## JAX COSMO

1 - Problem avec Cosmology Pytree

```py
import jax_cosmo as jc
from jaxpm.growth import growth_factor
import jax
# Define the cosmology

cosmo = jc.Planck15()

print(f"work space is {cosmo._workspace}")
g = growth_factor(cosmo, 0.5)
print(f"workspace is {cosmo._workspace}")

def non_jit_fn(cosmo, z):
    jax.debug.print("workspace is {a}" , a=cosmo._workspace)
    return growth_factor(cosmo, z)

non_jit_fn(cosmo, 0.5)

jax.jit(non_jit_fn)(cosmo, 0.5)
```

2 - use jnp.interp instead of custom one

## Will do reversable LF



## Please point to Intrinsic Alignement NB



## PhD Deblending + Shear maps

1 - MM VAE
2 - Probabilistic Shear
3 - might go to shear maps



# TODOS

LSS

1 - First PR implement deco with jacfwd jacrev vmap /home/wassim/Projects/NBody/JaxPM-Paper/proto_deco.py
2 - Second PR implement using shardmap and custom object and deprecate
3 - Back to gradient stability 
4 - Memory usage per solver and per adjoint method
5 - GPU need table
6 - bench against pmwd
7 - Weak/Strong Scaling perf
8 - Conditioning, tracing a model, and running NUTS to infer parameters LPT
Benchmarks
9 - ASK EIFFL HOW TO DO PM lightcone

check PM lignt code from Eiffl https://github.com/EiffL/LPTLensingComparison/blob/346020eaaba0c72f96412ef6d7d8f57d84bdb4d1/jax_lensing/model.py

CMB

1 - Bench against fgbester evaluation + solving
2 - Optimize cluster ray tuner + gridding
3 - PTEP
