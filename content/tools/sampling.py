import os
import pickle
from functools import partial
from glob import glob

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather
from numpyro.infer import MCMC

all_gather = partial(process_allgather, tiled=True)


def batched_sampling(
    mcmc_kernel,
    path: str,
    rng_key: jax.random.PRNGKey,
    num_chains: int = 1,
    num_warmup: int = 500,
    num_samples: int = 1000,
    thinning: int = 1,
    batch_count: int = 5,
    save: bool = True,
    extra_fields=(),
    init_params=None,
    *model_args,
    **model_kwargs,
):
    """
    Run or resume MCMC sampling in batches with optional warmup.

    This function runs NumPyro MCMC sampling in batches, supporting multi-host distributed setups.
    It performs initial warmup if no previous state is found, then resumes from the saved sampler
    state and continues sampling across multiple batches.

    Samples are gathered from all hosts using `jax.experimental.multihost_utils.process_allgather`
    and saved to disk as `.npz` files. The sampler state is saved using `pickle` to enable resumption.

    Parameters
    ----------
    mcmc_kernel : numpyro.infer.mcmc.HMCKernel
        The kernel (e.g., NUTS(model)) used for inference.
    path : str
        Directory where samples and state will be saved.
    rng_key : jax.random.PRNGKey
        JAX random key for reproducibility.
    num_chains : int, optional
        Number of chains to run in parallel.
    num_warmup : int, optional
        Number of warmup steps before collecting samples.
    num_samples : int, optional
        Number of samples to collect in each batch.
    thinning : int, optional
        Thinning factor to reduce autocorrelation in samples.
    batch_count : int, optional
        Number of post-warmup sampling batches to run.
    save : bool, optional
        Whether to save samples and state to disk.
    extra_fields : tuple, optional
        Extra diagnostics to store during sampling.
    init_params : dict, optional
        Optional initial parameters in unconstrained space.
    *model_args : tuple
        Positional arguments passed to the model.
    **model_kwargs : dict
        Keyword arguments passed to the model.

    Returns
    -------
    last_state : HMCState
        Final MCMC sampler state.
    mcmc : numpyro.infer.MCMC
        The last MCMC object used.

    Note
    ----
    This function is compatible with multi-host distributed training via `pjit` or `xmap`.
    """
    state_path = f"{path}/sampling_state.pkl"
    samples_path = f"{path}/samples_0.npz"
    os.makedirs(path, exist_ok=True)
    mcmc = MCMC(
        mcmc_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        thinning=thinning,
        num_chains=num_chains,
        progress_bar=True,
    )

    if not os.path.exists(state_path):
        print("🔁 Starting fresh with warmup...")
        mcmc.run(rng_key, *model_args, extra_fields=extra_fields, **model_kwargs)
        if save:
            with open(state_path, "wb") as f:
                pickle.dump(mcmc.last_state, f)
            jnp.savez(samples_path, **mcmc.get_samples())
        last_state = mcmc.last_state
        rng_key = last_state.rng_key
    else:
        print("▶️ Resuming from saved warmup state...")
        with open(state_path, "rb") as f:
            last_state = pickle.load(f)
        rng_key = last_state.rng_key

    mcmc = None
    for i in range(2, batch_count + 2):
        if last_state.i >= num_warmup + num_samples * batch_count:
            print(
                f"✅ {num_warmup + num_samples * batch_count} samples already collected. Stopping."
            )
            break

        print(f"📦 Sampling batch {i}/{batch_count} ...")
        mcmc = MCMC(
            mcmc_kernel,
            num_warmup=0,
            num_samples=num_samples,
            thinning=thinning,
            num_chains=num_chains,
            progress_bar=True,
        )
        mcmc.post_warmup_state = last_state
        mcmc.run(rng_key, *model_args, **model_kwargs)

        samples = mcmc.get_samples()
        host_samples = all_gather(samples)
        del samples

        if save:
            jnp.savez(f"{path}/samples_{i}.npz", **host_samples)
            with open(state_path, "wb") as f:
                pickle.dump(mcmc.last_state, f)

        last_state = mcmc.last_state
        with open(state_path, "wb") as f:
            pickle.dump(last_state, f)
        rng_key = last_state.rng_key

    return last_state, mcmc


def load_samples(path: str, param_names: list[str]) -> dict:
    """
    Efficiently load and concatenate specified parameter samples from saved batches.

    This function searches the specified directory for all `.npz` sample files and extracts
    the requested parameter arrays, concatenating them across all saved batches.

    Parameters
    ----------
    path : str
        Directory where samples are saved (e.g., "output/mcmc_run").
    param_names : list of str
        List of parameter names to load and concatenate.

    Returns
    -------
    concatenated : dict
        Dictionary mapping parameter names to concatenated JAX arrays.

    Note
    ----
    This function assumes all samples were saved using `jnp.savez(...)` with consistent shapes
    and parameter names.
    """
    collected = {name: [] for name in param_names}
    files = glob(os.path.join(path, "*samples_*.npz"))

    for file in sorted(files):
        data = np.load(file)
        for name in param_names:
            if name in data:
                collected[name].append(jnp.array(data[name]))

    return {k: jnp.concatenate(v, axis=0) for k, v in collected.items() if v}
