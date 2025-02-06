# TOSHOW 06/02/25

 - jaxDecomp ShardMap
 - jaxPM shardMap
 - Gradient stability
 - Reversible Solvers
 - Full Field run with 20 step


# TODOS

## LSS

###  Deblending + Shear maps

BTK download images for Binh

### jaxDecomp

 - ~~Implement Shardmapped variant~~
 - ~~Tests~~
 - update paper

### jaxPM 

 - ~~PR to use new ShardMappedAPI~~
 - ~~tests~~
 - update notebooks

### jaxPM Paper

 - ~~First PR implement deco with jacfwd jacrev vmap /home/wassim/Projects/NBody/JaxPM-Paper/proto_deco.py~~
 - ~~Second PR implement using shardmap and custom object and deprecate~~
 - ~~Back to gradient stability~~
 - ~~Memory usage per solver and per adjoint method~~
 - GPU need table
 - bench against pmwd (scripts made)
 - Weak/Strong Scaling perf
 - ~~Conditioning, tracing a model, and running NUTS to infer parameters LPT~~
 - Benchmarks
 - ~~ASK EIFFL HOW TO DO PM lightcone~~
 - ~~check PM lignt code from Eiffl https://github.com/EiffL/LPTLensingComparison/blob/346020eaaba0c72f96412ef6d7d8f57d84bdb4d1/jax_lensing/model.py~~

__Note__ final todos are cleanup and run distributed Full Field with 100 steps

## CMB

 - ~~Bench against fgbester evaluation + solving~~
 - ~~Optimize cluster ray tuner + gridding~~
 - run validation GAL_20
 - run d0s0 no clusters entire map
 - run d1s1 with 10 000 grid steps
 - PTEP run with table
 - 
|                     | **Galactic latitude** |         |        |
|---------------------|-----------------------|---------|--------|
|                     | **Low**               | **Medium** | **High** |
| **βₙ**              | 64                    | 64      | 64     |
| **Tₙ**              | 8                     | 4       | 0      |
| **βₛ**              | 4                     | 2       | 2      |


 - Automatize selection of Planck Masks
