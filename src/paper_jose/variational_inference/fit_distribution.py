"""
Can we fit simple distributions in a variational inference like manner and learn something from that?
"""

################
### PREAMBLE ###
################

import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
import corner
import shutil

import os
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns
from typing import Union, Callable
from collections import defaultdict

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
print(jax.devices())

import jax.numpy as jnp
from jimgw.prior import UniformPrior, CombinePrior
from jaxtyping import Array
import joseTOV.utils as jose_utils

from paper_jose.universal_relations.universal_relations import UniversalRelationBreaker
import paper_jose.utils as utils
import paper_jose.inference.utils_plotting as utils_plotting
plt.rcParams.update(utils_plotting.mpl_params)
import seaborn as sns

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

class GaussianPrior:
    
    """Multivariate Gaussian distribution of which covariance matrix entries are parameters"""
    
    def __init__(self, 
                 mean: Array,
                 param_names: list[str]):
        
        self.mean = mean
        self.param_names = param_names
        self.n_dim = len(mean)
        
    def get_covariance(self,
                       covariance_parameters: dict):
        
        cov = jnp.zeros((self.n_dim, self.n_dim))
        
        # Fill with the appropriate entries:
        for i, param_name in enumerate(self.param_names):
            cov = cov.at[i, i].set(covariance_parameters[f"sigma_{param_name}_{param_name}"])
            for j in range(i+1, self.n_dim):
                other_param_name = self.param_names[j]
                cov = cov.at[i, j].set(covariance_parameters[f"sigma_{param_name}_{other_param_name}"])
                cov = cov.at[j, i].set(cov[i, j])
        
        print("cov")
        print(cov)
        
        return cov
        
    def sample(self, 
               covariance_parameters: dict, 
               key: jax.random.PRNGKey,
               nb_samples: int = 100) -> Array:
        
        z = jax.random.normal(key, shape=(nb_samples, len(self.mean)))
        cov = self.get_covariance(covariance_parameters)
        L = jnp.linalg.cholesky(cov)
        x = self.mean + z @ L.T
        
        return x
    
    def add_name(self, x: Array):
        x_named = {name: x[i] for i, name in enumerate(self.param_names)}
        return x_named


def do_run(eos_dir: str = "../doppelgangers/real_doppelgangers/7945/data/"):
    
    # Load the target EOS
    target_eos = load_target_eos(eos_dir = eos_dir)
    
    # Prior
    my_nbreak = 2.0 * 0.16
    NMAX_NSAT = 25
    NB_CSE = 8
    NMAX = NMAX_NSAT * 0.16
    width = (NMAX - my_nbreak) / (NB_CSE + 1)

    # NEP priors
    K_sat_prior = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
    Q_sat_prior = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
    Z_sat_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])

    E_sym_prior = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
    L_sym_prior = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
    K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
    Q_sym_prior = UniformPrior(-800.0, 800.0, parameter_names=["Q_sym"])
    Z_sym_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sym"])

    prior_list: list[UniformPrior] = [
        E_sym_prior,
        L_sym_prior, 
        # K_sym_prior,
        # Q_sym_prior,
        # Z_sym_prior,

        # K_sat_prior,
        # Q_sat_prior,
        # Z_sat_prior,
    ]

    # Combine the prior
    prior = CombinePrior(prior_list)
    
    # Define the transform for EOS code and TOV solver
    sampled_param_names = prior.parameter_names
    name_mapping = (sampled_param_names, ["masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])
    transform = utils.MicroToMacroTransform(name_mapping, nmax_nsat=NMAX_NSAT, nb_CSE=NB_CSE)
    
    ### Now we use it to define a Gaussian distribution on the NEPs
    
    # Get the standard deviations
    widths = []
    for prior in prior_list:
        widths.append(prior.xmax - prior.xmin)
        
    scale_factor = 0.1
    widths = scale_factor * np.array(widths)
    
    print(f"Standard deviations of the Gaussian (rescaled widths) are: {widths}")
    
    # Get the mean
    param_names = [prior.parameter_names[0] for prior in prior_list]
    mean = np.array([target_eos[param_name] for param_name in param_names])
    
    print(f"Mean of the Gaussian is: {mean}")
    
    # Now define a multivariate Gaussian
    gaussian = GaussianPrior(mean, param_names)
    
    covariance_parameters = {}
    nb_params = len(param_names)
    for i, param_name in enumerate(param_names):
        covariance_parameters[f"sigma_{param_name}_{param_name}"] = widths[i]
    for i, param_name in enumerate(param_names):
        for j in range(i+1, nb_params):
            other_param_name = param_names[j]
            covariance_parameters[f"sigma_{param_name}_{other_param_name}"] = 0.0
            
    print(f"Starting with covariance matrix parameters:")
    print(covariance_parameters)
    
    # Try out sample:
    samples = gaussian.sample(covariance_parameters, jax.random.PRNGKey(0), nb_samples=1_000)
    samples = np.array(samples)
    
    fig = corner.corner(samples, truths = np.array(mean), labels=param_names, **default_corner_kwargs)
    plt.savefig("./figures/test.png", bbox_inches="tight")
    plt.close()
    
    
def load_target_eos(eos_dir: str = "../doppelgangers/real_doppelgangers/7945/data/") -> dict:
    """
    In the specified eos_dir, find and load the file with highest counter, which was the final file found by the doppelganger optimizer.
    Return this info, which contains all we need to know about the target EOS.
    """
    files = os.listdir(eos_dir)
    numbers = [f.split('.')[0] for f in files]
    
    max_nb = max([int(nb) for nb in numbers])
    filename = os.path.join(eos_dir, f"{max_nb}.npz")
    data = np.load(filename)
    
    return data  
    
    
def main():
    do_run()
    
if __name__ == "__main__":
    main()