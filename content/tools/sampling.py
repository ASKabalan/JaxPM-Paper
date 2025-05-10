import os
import pickle
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC
import numpy as np
import jax.numpy as jnp
import os
from glob import glob

def batched_sampling(
    mcmc_kernel,
    path: str,
    rng_key: jax.random.PRNGKey,
    num_chains: int = 1,
    num_warmup: int = 500,
    num_samples: int = 1000,
    batch_size: int = 1000,
    thinning: int = 1,
    batch_count: int = 5,
    save: bool = True,
    extra_fields = (),
    init_params = None,
    *model_args,
    **model_kwargs,
):
    """
    Run or resume MCMC sampling in batches with optional warmup.

    See docstring improvements in earlier messages.

    This version does not return samples — it only saves them to disk.
    """
    state_path = f"{path}/sampling_state.pkl"
    samples_path = f"{path}/samples_0.npz"
    os.makedirs(path, exist_ok=True)
    mcmc = MCMC(mcmc_kernel, num_warmup=num_warmup, num_samples=num_samples, thinning=thinning, num_chains=num_chains , progress_bar=True)

    if not os.path.exists(state_path):
        print("🔁 Starting fresh with warmup...")
        mcmc.run(rng_key, *model_args , extra_fields=extra_fields, init_params=init_params , **model_kwargs)
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

    for i in range(2, batch_count + 2):
        mcmc = MCMC(mcmc_kernel, num_warmup=0, num_samples=batch_size)
        mcmc.post_warmup_state = last_state
        if last_state.i >= num_warmup + num_samples * batch_count:
            print(f"✅ {num_warmup + num_samples * batch_count} samples already collected. Stopping.")
            break

        print(f"📦 Sampling batch {i}/{batch_count} ...")
        mcmc.run(rng_key, *model_args, **model_kwargs)

        if save:
            jnp.savez(f"{path}/samples_{i}.npz", **mcmc.get_samples())
            with open(state_path, "wb") as f:
                pickle.dump(mcmc.last_state, f)

        last_state = mcmc.last_state
        with open(state_path, "wb") as f:
                pickle.dump(last_state, f)
        rng_key = last_state.rng_key

    return last_state , mcmc

def load_samples(path: str, param_names: list[str]) -> dict:
    """
    Efficiently load and concatenate specified parameter samples from saved batches.

    Parameters
    ----------
    path : str
        Base path prefix used when saving (e.g. 'output/mcmc_run').
    param_names : list of str
        List of parameter names to extract and concatenate.

    Returns
    -------
    concatenated : dict
        Dictionary mapping each param name to a concatenated jnp.ndarray.
    """
    collected = {name: [] for name in param_names}
    files = glob(os.path.join(path, "*samples_*.npz"))

    for file in sorted(files):
            data = np.load(file)
            for name in param_names:
                if name in data:
                    collected[name].append(jnp.array(data[name]))

    return {k: jnp.concatenate(v, axis=0) for k, v in collected.items() if v}
