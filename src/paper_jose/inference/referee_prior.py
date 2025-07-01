"""
Plots of the prior distributions for the referee report of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
import sys
import corner
import tqdm
import argparse
import arviz

from scipy.stats import gaussian_kde

np.random.seed(2)
import joseTOV.utils as jose_utils

HAUKE_COLOR = "#83a1f0"
JESTER_COLOR = "#783fee"

fs = 16
mpl_params = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "axes.labelsize": fs,
        "legend.fontsize": fs,
        "legend.title_fontsize": fs,
        "figure.titlesize": fs}
plt.rcParams.update(mpl_params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)


###########################
### Load the Hauke data ###
###########################

data_hauke = np.load("./results_hauke_eos.npz")
MTOV_prior_hauke = data_hauke["MTOV"]
r14_prior_hauke = data_hauke["R14"]
p3nsat_prior_hauke = data_hauke["p3nsat"]
nTOV_prior_hauke = data_hauke["nTOV"] # TODO: get this data from the Hauke paper

bad_idx = data_hauke["bad_idx"]
MTOV_prior_hauke = np.delete(MTOV_prior_hauke, bad_idx)
r14_prior_hauke = np.delete(r14_prior_hauke, bad_idx)
p3nsat_prior_hauke = np.delete(p3nsat_prior_hauke, bad_idx)
nTOV_prior_hauke = np.delete(nTOV_prior_hauke, bad_idx)

hauke_dict = {
    "MTOV": MTOV_prior_hauke,
    "R14": r14_prior_hauke,
    "p3nsat": p3nsat_prior_hauke,
    "nTOV": nTOV_prior_hauke
}

###########################
### Load the Hauke data ###
###########################

# Load the prior samples:
prior_filename = "./outdir_prior/eos_samples.npz"
data_jester = np.load(prior_filename)
m, r = data_jester["masses_EOS"], data_jester["radii_EOS"]
r14_prior_jester = np.array([np.interp(1.4, m[i], r[i]) for i in range(len(m))])
mtov_jester = np.array([np.max(m[i]) for i in range(len(m))])

n, p, e = data_jester["n"], data_jester["p"], data_jester["e"]

n = n / jose_utils.fm_inv3_to_geometric / 0.16
p = p / jose_utils.MeV_fm_inv3_to_geometric
e = e / jose_utils.MeV_fm_inv3_to_geometric

p3nsat_jester = np.array([np.interp(3.0, n[i], p[i]) for i in range(len(n))])

# TODO: get this
logpc = data_jester["logpc_EOS"]
pc_EOS = np.exp(logpc) / jose_utils.MeV_fm_inv3_to_geometric
ntov_jester = []
for i in range(len(m)):
    # Get the maximum pressure for each EOS
    _m, _p = m[i], pc_EOS[i]
    # Interpolate to find the pressure at the maximum mass
    pc_max = np.interp(mtov_jester[i], _m, _p)
    # Interpolate to find nTOV
    nTOV = np.interp(pc_max, p[i], n[i])
    ntov_jester.append(nTOV)

# _n, _p,  = n[i], p[i]
# pc_TOV = np.interp(mtov_jester, _m, _pc)
# n_TOV = np.interp(pc_TOV, _p, _n)
# ntov_list.append(n_TOV)
# ntov_list = np.random.uniform(4.0, 8.0, size=len(r14_prior_jester))  # Placeholder for nTOV jester

jester_dict = {
    "MTOV": mtov_jester,
    "R14": r14_prior_jester,
    "p3nsat": p3nsat_jester,
    "nTOV": ntov_jester
}

labels_dict = {"MTOV": r"$M_{\rm TOV}$ [M$_\odot$]",
               "R14": r"$R_{1.4}$ [km]",
               "p3nsat": r"$p(3 n_{\rm sat})$ [MeV fm${}^{-3}$]",
               "nTOV": r"$n_{\rm TOV}$ [$n_{\rm sat}$]"}

# Make plot of the KDEs, save it to outdir_prior
bounds_dict = {"MTOV": (0.5, 4.0),
               "R14": (4.0, 18.0),
               "p3nsat": (-0.1, 400.0),
               "nTOV": (1.0, 16.0)}

# Iterate over the keys, fetch data, get KDE, plot
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for i, key in enumerate(hauke_dict.keys()):
    # Get the data for Hauke and Jester, fetch the ax
    jester_data = jester_dict[key]
    hauke_data = hauke_dict[key]
    bounds = bounds_dict[key]
    ax = axs[i // 2, i % 2]
    
    label = labels_dict[key]
    
    # Get the KDEs
    kde_hauke = gaussian_kde(hauke_data)
    kde_jester = gaussian_kde(jester_data)
    
    # Create x values for plotting
    x = np.linspace(bounds[0], bounds[1], 1_000)
    
    # Plot the KDEs
    lw = 3
    ax.plot(x, kde_hauke(x), color=HAUKE_COLOR, label="Koehn+2024", lw=lw)
    ax.plot(x, kde_jester(x), color=JESTER_COLOR, label="This work", lw=lw)
    
    # Set labels and title
    label_fontsize = 18
    ax.set_xlabel(label, fontsize = label_fontsize)
    ax.set_ylabel("Probability density", fontsize = 16)
    if key == "p3nsat":
        ax.legend(fontsize = label_fontsize)
        
    ax.set_ylim(bottom = 0.0)
    ax.set_xlim(bounds)
    
plt.tight_layout()
plt.savefig("./figures/prior_distributions.pdf", dpi=300, bbox_inches='tight')
plt.close()