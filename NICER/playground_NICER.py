"""
Playground for NICER inference: we sample some individual parameters, then solve the TOV equations and compute the NICER log likelihood. The results are saved and plotted to visually sanity-check the results. Meant for low number of samples and playing around.
"""

################
### PREAMBLE ###
################

import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"

import os
import tqdm
import time
import numpy as np
np.random.seed(42) # for reproducibility
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jimgw.prior import UniformPrior, CombinePrior

import utils as NICER_utils
plt.rcParams.update(NICER_utils.mpl_params)

start = time.time()

##############
### PRIORS ###
##############

L_sym_prior = UniformPrior(20.0, 150.0, parameter_names=["L_sym"])
K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
K_sat_prior = UniformPrior(200.0, 300.0, parameter_names=["K_sat"])

my_eps = 1e-3 # small nudge to avoid overlap in CSE gridpoints
NBREAK_NSAT = 2.0
nsat = 0.16

# TODO: this is a first attempt so pretty cumbersome, improve in the future
n_CSE_0_prior = UniformPrior((NBREAK_NSAT + my_eps) * nsat, (3.0 - my_eps) * nsat, parameter_names=["n_CSE_0"])
n_CSE_1_prior = UniformPrior(3.0 * nsat, (4.0 - my_eps) * nsat, parameter_names=["n_CSE_1"])
n_CSE_2_prior = UniformPrior(4.0 * nsat, (5.0 - my_eps) * nsat, parameter_names=["n_CSE_2"])
n_CSE_3_prior = UniformPrior(5.0 * nsat, (6.0 - my_eps) * nsat, parameter_names=["n_CSE_3"])
n_CSE_4_prior = UniformPrior(6.0 * nsat, (7.0 - my_eps) * nsat, parameter_names=["n_CSE_4"])
n_CSE_5_prior = UniformPrior(7.0 * nsat, (8.0 - my_eps) * nsat, parameter_names=["n_CSE_5"])
n_CSE_6_prior = UniformPrior(8.0 * nsat, (9.0 - my_eps) * nsat, parameter_names=["n_CSE_6"])
n_CSE_7_prior = UniformPrior(9.0 * nsat, (10.0 - my_eps) * nsat, parameter_names=["n_CSE_7"])

# TODO: I am a bit scared about the bounds
my_eps = 1e-2
cs2_CSE_0_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_0"])
cs2_CSE_1_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_1"])
cs2_CSE_2_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_2"])
cs2_CSE_3_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_3"])
cs2_CSE_4_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_4"])
cs2_CSE_5_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_5"])
cs2_CSE_6_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_6"])
cs2_CSE_7_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_7"])


prior_list = [L_sym_prior, 
            #   K_sym_prior, 
            #   K_sat_prior,
            #   n_CSE_0_prior,
            #   n_CSE_1_prior,
            #   n_CSE_2_prior,
            #   n_CSE_3_prior,
            #   n_CSE_4_prior,
            #   n_CSE_5_prior,
            #   n_CSE_6_prior,
            #   n_CSE_7_prior,
            #   cs2_CSE_0_prior,
            #   cs2_CSE_1_prior,
            #   cs2_CSE_2_prior,
            #   cs2_CSE_3_prior,
            #   cs2_CSE_4_prior,
            #   cs2_CSE_5_prior,
            #   cs2_CSE_6_prior,
            #   cs2_CSE_7_prior
]

prior = CombinePrior(prior_list)
sampled_param_names = prior.parameter_names

##################
### LIKELIHOOD ###
##################

likelihood = NICER_utils.NICERLikelihood(sampled_param_names, NBREAK_NSAT)

##############
### Sample ###
##############

N = 100
### Without vmap
jax_key = jax.random.PRNGKey(43)
for i in tqdm.tqdm(range(N)):
    
    jax_key, jax_subkey = jax.random.split(jax_key)
    
    params = prior.sample(jax_subkey, 1)
    for key, value in params.items():
        if isinstance(value, jnp.ndarray):
            params[key] = value.at[0].get()
    
    L = likelihood.evaluate(params, None)

# ### With vmap
# jax_key = jax.random.PRNGKey(43)
# jax_key, jax_subkey = jax.random.split(jax_key)
# params = prior.sample(jax_subkey, N)

# print(params)

# likelihood_evaluate_vmap = jax.vmap(likelihood.evaluate, in_axes=(0, None))
    
# print("Computing likelihood with vmap")
# my_time_start = time.time()
# L_array = likelihood_evaluate_vmap(params, None)
# my_time_end = time.time()
# print(f"Time taken: {my_time_end - my_time_start} s")
# print("Computing likelihood with vmap DONE")

# print("Results:")
# print(L_array)
# print(np.min(L_array), np.max(L_array))
    
# ##############
# ### CORNER ###
# ##############

# ### Plot a cornerplot of the data

# # TODO: do an inference problem and plot all the M, R curves on top coloured by their log-likelihood as per computed above

# # Prepare general corner kwargs first
# corner_kwargs = default_corner_kwargs.copy()
# hist_kwargs = {"density": True}
# corner_kwargs["labels"] = ["$R$ [km]", "$M$ [$M_{\odot}$]"]
# corner_kwargs["fill_contours"] = True
# corner_kwargs["alpha"] = 1.0
# corner_kwargs["zorder"] = 1e9
# corner_kwargs["no_fill_contours"] = True
# corner_kwargs["plot_contours"] = True

# corner_kwargs["color"] = AMSTERDAM_COLOR
# hist_kwargs["color"] = AMSTERDAM_COLOR

# corner_kwargs["hist_kwargs"] = hist_kwargs
# corner_kwargs["range"] = [(8, 17), (0.8, 2.25)]
# fig = corner.corner(amsterdam_samples[["R", "M"]], weights=amsterdam_samples["weight"], **corner_kwargs)

# corner_kwargs["color"] = MARYLAND_COLOR
# hist_kwargs["color"] = MARYLAND_COLOR
# corner_kwargs["hist_kwargs"] = hist_kwargs
# fig = corner.corner(maryland_samples[["R", "M"]], fig=fig, weights=maryland_samples["weight"], **corner_kwargs)

# fs = 24
# plt.text(0.65, 0.8, "Maryland", color=MARYLAND_COLOR, fontsize=fs, transform=plt.gcf().transFigure)
# plt.text(0.65, 0.7, "Amsterdam", color=AMSTERDAM_COLOR, fontsize=fs, transform=plt.gcf().transFigure)
# plt.savefig(f"./figures/corner_{PSR_NAME}.png")
# plt.close() 

################
### CONTOURS ###
################

fig, ax = plt.subplots(figsize = (12, 6))

# First the data TODO: improve the plotting, the contours are ugly but matplotlib is annoying...
for dataset, cmap in zip([NICER_utils.maryland_data_2d, NICER_utils.amsterdam_data_2d], [NICER_utils.MARYLAND_CMAP, NICER_utils.AMSTERDAM_CMAP]):

    data = dataset.T
    hist, xedges, yedges = np.histogram2d(data[:, 1], data[:, 0], bins=50)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters)
    plt.contour(X, Y, hist.T, levels=10, cmap = cmap)

# Read the EOS data
all_masses_EOS = []
all_radii_EOS = []
all_L = []

for i in range(N):
    data = np.load(f"./computed_data/{i}.npz")
    
    masses_EOS = data["masses_EOS"]
    radii_EOS = data["radii_EOS"]
    L = data["L"]
    
    # Check for any NaNs:
    if np.isnan(L) or np.isnan(masses_EOS).any() or np.isnan(radii_EOS).any():
        print(f"Skipping {i} due to NaNs")
        continue
    
    all_masses_EOS.append(masses_EOS)
    all_radii_EOS.append(radii_EOS)
    all_L.append(L)
    
# DEBUG
print("Before normalizing this is the log L range:")
print(np.min(all_L), np.max(all_L))
    
# Then plot all the EOS data
all_L = np.array(all_L)
if np.max(all_L) == np.min(all_L):
    all_L = np.ones_like(all_L)
else:
    all_L = (all_L - np.min(all_L))/(np.max(all_L) - np.min(all_L))

norm = mpl.colors.Normalize(vmin=np.min(all_L), vmax=np.max(all_L))
# cmap = mpl.cm.Greens
cmap = sns.color_palette("rocket_r", as_cmap=True)

for i in range(N):
    color = cmap(norm(all_L[i]))  # Get the color from the colormap
    plt.plot(all_radii_EOS[i], all_masses_EOS[i], color=color, alpha=all_L[i], linewidth = 2.0, zorder=1e10)
    
plt.xlim(8, 17)
plt.ylim(0.25, 2.75)
plt.xlabel(r"$R$ [km]")
plt.ylabel(r"$M$ [$M_{\odot}$]")

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r'Normalized $\log \mathcal{L}_{\rm{NICER}}$', fontsize = 22)

plt.tight_layout()
plt.savefig(f"./figures/contours_{NICER_utils.PSR_NAME}.png", bbox_inches = "tight")
plt.close()

print("DONE")
end = time.time()
print(f"Time taken: {end - start} s")