from paper_jose.utils import PSR_PATHS_DICT
import corner
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt

params = {"axes.grid": True,
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

plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        # color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        hist_kwargs=dict(density=True),
                        save=False)

# Generate samples from data_samples_dict
N_samples = 20_000
N_samples_plot = 10_000

for psr in ["J0030", "J0740"]:
    for group in ["amsterdam", "maryland"]:
        
        print(f"\n\n\n Checking for {psr} and {group} \n\n\n")
        
        # Get the paths
        path = PSR_PATHS_DICT[psr][group]


        if group == "maryland":
            samples = pd.read_csv(path, sep=" ", names=["R", "M", "weight"] , skiprows = 6)
        else:
            if psr == "J0030":
                samples = pd.read_csv(path, sep=" ", names=["weight", "M", "R"])
            else:
                samples = pd.read_csv(path, sep=" ", names=["M", "R"])
                samples["weight"] = np.ones_like(samples["M"])
        
        print("samples")
        print(samples)
        
        if pd.isna(samples["weight"]).any():
            print("Warning: weights not properly specified, assuming constant weights instead.")
            samples["weight"] = np.ones_like(samples["weight"])
            
        # Get as samples and as KDE
        m, r, w = samples["M"].values, samples["R"].values, samples["weight"].values
        
        # Generate N_samples samples for the KDE:
        idx = np.random.choice(len(samples), size = N_samples)
        m, r, w = m[idx], r[idx], w[idx]
        
        # Generate the KDEs
        data_2d = jnp.array([m, r])

        posterior = gaussian_kde(data_2d, weights = w)
        
        # Generate N_samples_plot samples for the plot:
        jax_key = jax.random.PRNGKey(0)
        jax_key, jax_subkey = jax.random.split(jax_key)
        m_KDE, r_KDE = posterior.resample(jax_subkey, (N_samples_plot,))
        
        default_corner_kwargs["hist_kwargs"] = {"color": "blue", "density": True}
        fig = corner.corner(np.array([m, r]).T, labels=["M", "R"], color="blue", weights=w, hist_bin_factor=2, **default_corner_kwargs)
        default_corner_kwargs["hist_kwargs"] = {"color": "red", "density": True}
        corner.corner(np.array([m_KDE, r_KDE]).T, fig=fig, color="red", hist_bin_factor=2, **default_corner_kwargs)
        plt.savefig(f"figures/test_kde_corner_{psr}_{group}.png")
        plt.close()