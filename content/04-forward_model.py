import argparse
import os

import jax

# =============================================================================
# 1. If running on a distributed system, initialize JAX distributed
# =============================================================================
if (
    int(os.environ.get("SLURM_NTASKS", 0)) > 1
    or int(os.environ.get("SLURM_NTASKS_PER_NODE", 0)) > 1
):
    os.environ["VSCODE_PROXY_URI"] = ""
    os.environ["no_proxy"] = ""
    os.environ["NO_PROXY"] = ""
    del os.environ["VSCODE_PROXY_URI"]
    del os.environ["no_proxy"]
    del os.environ["NO_PROXY"]
    jax.distributed.initialize()

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    pdims = (8, 1)
    mesh = jax.make_mesh(pdims, ("x", "y"))
    sharding = NamedSharding(mesh, P("x", "y"))
else:
    sharding = None
# =============================================================================


import arviz as az
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
from diffrax import RecursiveCheckpointAdjoint
from numpyro.handlers import condition, seed, trace
from scipy.stats import norm
from tools.lensing_model import (
    Configurations,
    Planck18,
    full_field_probmodel,
)
from tools.sampling import batched_sampling, load_samples

SAMPLERS = ["HMC", "NUTS", "MCLMC", "ADJ_MCLMC"]

os.environ["JC_CACHE"] = "off"
os.environ["EQX_ON_ERROR"] = "nan"
jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run lensing full field inference.")
    parser.add_argument("--output", type=str, default="samples", help="Directory to save samples")

    parser.add_argument("--box_shape", nargs=3, type=int, default=[16, 16, 32])
    parser.add_argument("--box_size", nargs=3, type=float, default=[200.0, 200.0, 400.0])
    parser.add_argument("--field_size", type=float, default=16.0)
    parser.add_argument("--field_npix", type=int, default=16)
    parser.add_argument("--density_plane_width", type=float, default=50.0)
    parser.add_argument("--density_plane_npix", type=int, default=16)
    parser.add_argument("--density_plane_smoothing", type=float, default=0.1)
    parser.add_argument(
        "--obs_file", type=str, default="obs.npz", help="Path to saved observed data file (npz)"
    )
    parser.add_argument("--plot", action="store_true", help="If set, plot results and exit")

    parser.add_argument("--rng_key", type=int, default=1234, help="Random seed for reproducibility")
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--batch_count", type=int, default=10)
    parser.add_argument(
        "--sampler",
        type=str,
        default="NUTS",
        choices=SAMPLERS,
        help=f"Sampler to use for inference. Choices: {SAMPLERS}",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="numpyro",
        choices=["numpyro", "blackjax"],
        help="Inference backend to use. Choices: numpyro, blackjax",
    )

    return parser.parse_args()


def plot_results(folder):
    samples = load_samples(folder)
    true_data = np.load(os.path.join(folder, "obs.npz"))
    true_ic = true_data["initial_conditions"]
    kappa_keys = [f"kappa_{i}" for i in range(4)]

    # Pair plot for scalar parameters
    scalar_keys = [k for k in samples.keys() if samples[k].ndim == 1]
    scalar_samples = {k: samples[k] for k in scalar_keys}
    if scalar_keys:
        print("Plotting posterior pairplot for scalar parameters...")
        az_data = az.from_dict(posterior=scalar_samples)
        az.plot_pair(az_data, kind="kde", marginals=True)
        plt.suptitle("Posterior Pairplot")
        plt.show()

    # Plot Initial Conditions (IC)
    print("Plotting Initial Conditions (IC)...")
    slice_true_ic = true_ic[1]
    mean_ic = jnp.mean(samples["ic"], axis=0)[1]
    std_ic = jnp.std(samples["ic"], axis=0)[1]
    residual_ic = mean_ic - slice_true_ic

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, img, title in zip(
        axes,
        [slice_true_ic, mean_ic, std_ic, residual_ic],
        ["True IC", "Mean IC", "Std Dev IC", "Residuals IC"],
    ):
        im = ax.imshow(img, cmap="viridis")
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    # Plot kappa maps (each redshift bin)
    print("Plotting Kappa Maps...")
    for i, key in enumerate(kappa_keys):
        true_kappa = true_data[key]
        mean_kappa = jnp.mean(samples[key], axis=0)
        std_kappa = jnp.std(samples[key], axis=0)
        residual_kappa = mean_kappa - true_kappa

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, img, title in zip(
            axes,
            [true_kappa, mean_kappa, std_kappa, residual_kappa],
            [f"True Kappa {i}", "Mean", "Std Dev", "Residual"],
        ):
            im = ax.imshow(img, cmap="viridis")
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    # --- Skip sampling if in plot mode ---
    if args.plot:
        plot_results(args.output)
        return

    box_shape = tuple(args.box_shape)
    box_size = list(args.box_size)
    field_size = args.field_size
    field_npix = args.field_npix
    density_plane_width = args.density_plane_width
    density_plane_npix = args.density_plane_npix
    density_plane_smoothing = args.density_plane_smoothing
    sigma_e = 0.3
    t0 = 0.1
    t1 = 1.0
    dt0 = 0.1

    # Cosmology
    fiducial_cosmology = Planck18()
    print("Pixel size in arcmin: ", field_size * 60 / field_npix)

    max_comoving_distance = box_size[2]  # in Mpc/h
    max_redshift = (1 / jc.background.a_of_chi(fiducial_cosmology, max_comoving_distance) - 1).squeeze()
    # Setup shear redshift bins
    z = jnp.linspace(0, max_redshift, 1000)

    nz_shear = [
        jc.redshift.kde_nz(
            z, norm.pdf(z, loc=z_center, scale=0.12), bw=0.01, zmax=max_redshift, gals_per_arcmin2=g
        )
        for z_center, g in zip([0.5, 1.0, 1.5, 2.0], [7, 8.5, 7.5, 7])
    ]


    # Configuration
    config = Configurations(
        field_size=field_size,
        field_npix=field_npix,
        box_shape=box_shape,
        box_size=box_size,
        density_plane_width=density_plane_width,
        density_plane_npix=density_plane_npix,
        density_plane_smoothing=density_plane_smoothing,
        nz_shear=nz_shear,
        fiducial_cosmology=Planck18,
        sigma_e=sigma_e,
        priors={
            "Omega_c": dist.Uniform(0.2, 0.4),
            "sigma8": dist.Uniform(0.6, 1.0),
        },
        t0=t0,
        t1=t1,
        dt0=dt0,
        adjoint=RecursiveCheckpointAdjoint(checkpoints=5),
        sharding=sharding,
        max_redshift=max_redshift,
    )

    # Build forward model and trace fiducial simulation
    full_field_model = full_field_probmodel(config)
    fiducial_model = condition(
        full_field_model,
        {"Omega_c": fiducial_cosmology.Omega_c, "sigma8": fiducial_cosmology.sigma8},
    )
    # --- Handle observations ---
    obs_file = f"{args.output}/{args.obs_file}"
    if obs_file is not None and os.path.exists(obs_file):
        print(f"Loading observations from {obs_file}")
        obs_npz = np.load(obs_file)
        obs = {k: obs_npz[k] for k in obs_npz.files}
        true_ic = obs["initial_conditions"]
    else:
        print("Generating synthetic observations...")
        model_trace = trace(seed(fiducial_model, jax.random.key(1234))).get_trace()
        obs = {f"kappa_{i}": model_trace[f"kappa_{i}"]["value"] for i in range(4)}
        true_ic = model_trace["initial_conditions"]["value"]
        print(f"Saving observations to {obs_file}")
        np.savez(obs_file, **obs, initial_conditions=true_ic)

    print("Model Traced...")

    observed_model = condition(full_field_model, obs)
    init_params = {
        "Omega_c": fiducial_cosmology.Omega_c,
        "sigma8": fiducial_cosmology.sigma8,
        "initial_conditions": true_ic,
    }

    print("Starting MCMC sampling...")
    batched_sampling(
        observed_model,
        args.output,
        rng_key=jax.random.key(args.rng_key),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        batch_count=args.batch_count,
        save=True,
        sampler=args.sampler,
        backend=args.backend,
        init_params=init_params,
    )


if __name__ == "__main__":
    main()
