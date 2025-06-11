#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt
import numpy as np
from diffrax import RecursiveCheckpointAdjoint
from jaxpm.distributed import normal_field

parent_dir = os.path.abspath("../content")
sys.path.append(parent_dir)

from tools.lensing_model import (
    Configurations,
    Planck18,
    full_field_probmodel,
    make_full_field_model,
)

os.environ["EQX_ON_ERROR"] = "nan"
os.environ["JC_CACHE"] = "off"
jax.config.update("jax_enable_x64", True)


# In[ ]:


box_shape = (64, 64, 64)
box_size = [400.0, 400.0, 800.0]
field_size = 16.0  # transverse size in degrees
field_npix = 64  # number of pixels per side
density_plane_width = 100
density_plane_npix = 64  # updated number of pixels for density plane
density_plane_smoothing = 0.1

forward_model = jax.jit(
    make_full_field_model(
        field_size=field_size,
        field_npix=field_npix,
        box_shape=box_shape,
        box_size=box_size,
        density_plane_width=density_plane_width,
        density_plane_npix=density_plane_npix,
        density_plane_smoothing=density_plane_smoothing,
        adjoint=RecursiveCheckpointAdjoint(checkpoints=5),
        t0=0.1,
        t1=1.0,
        dt0=0.1,
    )
)


# In[10]:


from scipy.stats import norm

z = jnp.linspace(0, 2.5, 1000)

nz_shear = [
    jc.redshift.kde_nz(
        z, norm.pdf(z, loc=z_center, scale=0.12), bw=0.01, zmax=2.5, gals_per_arcmin2=g
    )
    for z_center, g in zip([0.5, 1.0, 1.5, 2.0], [7, 8.5, 7.5, 7])
]
nbins = len(nz_shear)
# Define the fiducial cosmology
cosmo = Planck18()

# Specify the size and resolution of the patch to simulate
sigma_e = 0.3
print("Pixel size in arcmin: ", field_size * 60 / field_npix)


# In[11]:


# Plotting the redshift distribution
z = np.linspace(0, 3.0, 128)

for i in range(nbins):
    plt.plot(
        z,
        nz_shear[i](z) * nz_shear[i].gals_per_arcmin2,
        color=f"C{i}",
        label=f"Bin {i}",
    )
plt.legend()
plt.xlim(0, 3)
plt.title("Redshift distribution")


# In[13]:


initial_conditions = normal_field(jax.random.key(0) , box_shape)
kappas, lc = forward_model(cosmo, nz_shear, initial_conditions)


# In[14]:


import matplotlib.pyplot as plt

plt.figure(figsize=(4, 5))  # Taller figure

# Show image
img = plt.imshow(lc[..., 0], cmap='viridis', origin='lower')

# Get current Axes and modify ticks
ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])

# Layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top
plt.show()


# In[15]:


plt.figure(figsize=(4, 12))  # Taller figure

for i in range(nbins):
    ax = plt.subplot(nbins, 1, i + 1)  # Change to vertical layout
    im = ax.imshow(lc[..., i], cmap='viridis', origin='lower')

    # Bigger title text
    ax.set_title(f'Redshift Bin {i + 1}', fontsize=25)

    # Only show x-ticks on the last plot
    ax.set_xticks([])

    # Hide y-ticks for all but first
    ax.set_yticks([])

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top
plt.show()


# In[ ]:


import numpyro
import numpyro.distributions as dist
from numpyro.handlers import condition, seed, trace
from numpyro.infer import MCMC, NUTS

# condition the model on a given set of parameters
fiducial_cosmology = Planck18()

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
        "h": dist.Uniform(0.5, 0.9),
    },
    t0=0.1,
    t1=1.0,
    dt0=0.1,
)

full_field_basemodel = full_field_probmodel(config)

fiducial_model = condition(
    full_field_basemodel,
    {"Omega_c": fiducial_cosmology.Omega_c, "sigma8": fiducial_cosmology.sigma8},
)


# sample a mass map and save corresponding true parameters
model_trace = trace(seed(fiducial_model, 5)).get_trace()


# In[10]:


# plotting the trace
plt.figure(figsize=[16, 4])
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.title(f"redshift bin {i}")
    plt.imshow(model_trace[f"kappa_{i}"]["fn"].mean)
    plt.axis("off")


# In[ ]:


# Let's condition the model on the observed maps
observed_model = condition(
    full_field_basemodel,
    {
        "kappa_0": model_trace["kappa_0"]["value"],
        "kappa_1": model_trace["kappa_1"]["value"],
        "kappa_2": model_trace["kappa_2"]["value"],
        "kappa_3": model_trace["kappa_3"]["value"],
    },
)


# In[ ]:


nuts_kernel = NUTS(
    model=observed_model,
    init_strategy=partial(
        numpyro.infer.init_to_value,
        values={
            "Omega_c": cosmo.Omega_c,
            "sigma8": cosmo.sigma8,
            "initial_conditions": model_trace["initial_conditions"]["value"],
        },
    ),
    max_tree_depth=3,
    step_size=0.05,
)


# In[ ]:


mcmc = MCMC(
    nuts_kernel,
    num_warmup=10,
    num_samples=10,
    # num_chains=5,
    # chain_method='vectorized',
    thinning=1,
    progress_bar=True,
)

mcmc.run(jax.random.PRNGKey(0))


# In[ ]:


mcmc.print_summary()


# In[ ]:


import arviz as az
import corner

inf_data = az.from_numpyro(mcmc)
az.summary(inf_data)


corner.corner(inf_data, var_names=["Omega_c", "sigma8"], truths=[cosmo.Omega_c, cosmo.sigma8])

