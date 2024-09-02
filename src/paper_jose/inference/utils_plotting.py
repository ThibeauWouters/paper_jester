import numpy as np
import matplotlib.pyplot as plt
import jax
import os
import json
import corner
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
from jax.scipy.stats import gaussian_kde
import pandas as pd

from joseTOV import utils

mpl_params = {"axes.grid": True,
              "text.usetex" : True,
              "font.family" : "serif",
              "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(mpl_params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        levels=[0.68, 0.95],
                        plot_density=False,
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False)

AMSTERDAM_COLOR = "green"
AMSTERDAM_CMAP = "Greens"
MARYLAND_COLOR = "blue"
MARYLAND_CMAP = "Blues"
EOS_CURVE_COLOR = "darkgreen"

CREX_CMAP = "Oranges"
PREX_CMAP = "Purples"
REX_CMAP_DICT = {"CREX": CREX_CMAP, "PREX": PREX_CMAP}

def plot_corner(outdir,
                samples,
                keys):
    
    samples = np.reshape(samples, (len(keys), -1))
    corner.corner(samples.T, labels = keys, **default_corner_kwargs)
    plt.savefig(outdir + "corner.png", bbox_inches = "tight")
    plt.close()

def plot_eos(outdir, 
             transformed_max_log_prob: dict, 
             transformed_samples: dict,
             samples_kwargs = {"color": "black", "alpha": 0.1}):
    
    ### Micro
    
    n_max = transformed_max_log_prob["n"] / utils.fm_inv3_to_geometric / 0.16
    p_max = transformed_max_log_prob["p"] / utils.MeV_fm_inv3_to_geometric
    e_max = transformed_max_log_prob["e"] / utils.MeV_fm_inv3_to_geometric
    h_max = transformed_max_log_prob["h"]
    cs2_max = transformed_max_log_prob["cs2"]
    cs2_max = jnp.gradient(p_max, e_max)
    
    plt.subplots(nrows = 2, ncols = 2, figsize = (17, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(n_max, p_max, color = "red")
    plt.xlabel("n [$n_{\rm{sat}}$]")
    plt.ylabel("p [MeV/fm$^3$]")
    
    plt.subplot(2, 2, 2)
    plt.plot(n_max, e_max, color = "red")
    plt.xlabel("n [$n_{\rm{sat}}$]")
    plt.ylabel("e [MeV/fm$^3$]")
    
    plt.subplot(2, 2, 3)
    plt.plot(n_max, h_max, color = "red")
    plt.xlabel("n [$n_{\rm{sat}}$]")
    plt.ylabel("h [MeV/fm$^3$]")
    
    plt.subplot(2, 2, 4)
    plt.plot(n_max, cs2_max, color = "red")
    plt.xlabel("n [$n_{\rm{sat}}$]")
    plt.ylabel("$c_s^2$")
    
    for i in range(len(transformed_samples.values())):
        n = transformed_samples["n"][i] / utils.fm_inv3_to_geometric / 0.16
        p = transformed_samples["p"][i] / utils.MeV_fm_inv3_to_geometric
        e = transformed_samples["e"][i] / utils.MeV_fm_inv3_to_geometric
        h = transformed_samples["h"][i]
        cs2 = transformed_samples["cs2"][i]
        
        
        plt.subplot(2, 2, 1)
        plt.plot(n, p, **samples_kwargs)
        
        plt.subplot(2, 2, 2)
        plt.plot(n, e, **samples_kwargs)
        
        plt.subplot(2, 2, 3)
        plt.plot(n, h, **samples_kwargs)
        
        plt.subplot(2, 2, 4)
        plt.plot(n, cs2, **samples_kwargs)
        
    plt.savefig(outdir + "eos.png", bbox_inches = "tight")
    plt.close()
    
    ### Macro
    plt.subplots(nrows = 1, ncols = 2, figsize = (15, 7))
    
    m, r, l = transformed_max_log_prob["masses_EOS"], transformed_max_log_prob["radii_EOS"], transformed_max_log_prob["Lambdas_EOS"]
    plt.subplot(1, 2, 1)
    plt.plot(r, m, color = "red")
    plt.xlabel("R [km]")
    
    plt.subplot(1, 2, 2)
    plt.plot(m, l, color = "red")
    plt.xlabel("M [$M_{\odot}$]")
    plt.ylabel("$\Lambda$")
    
    for i in range(len(transformed_samples.values())):
        m, r, l = transformed_samples["masses_EOS"][i], transformed_samples["radii_EOS"][i], transformed_samples["Lambdas_EOS"][i]
        
        plt.subplot(1, 2, 1)
        plt.plot(r, m, **samples_kwargs)
        
        plt.subplot(1, 2, 2)
        plt.plot(m, l, **samples_kwargs)
        
    plt.savefig(outdir + "MRL.png", bbox_inches = "tight")
    plt.close()
    

###############
### TESTING ###
###############

def main():
    pass
        
        
if __name__ == "__main__":
	main()