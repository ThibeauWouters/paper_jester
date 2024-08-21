"""
Analyze NICER data with Jose sampling
"""
### On CIT:
# import psutil
# p = psutil.Process()
# p.cpu_affinity([0])
###

################
### PREAMBLE ###
################

import os
import tqdm
import time
import copy
import numpy as np
import pandas as pd
np.random.seed(43) # for reproducibility
import matplotlib.pyplot as plt
import corner

import jax
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from joseTOV.eos import MetaModel_with_CSE_EOS_model, construct_family
from joseTOV import utils

from jimgw.base import LikelihoodBase
from jimgw.prior import UniformPrior, CombinePrior

import utils as NICER_utils

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
                        levels=[0.68, 0.95],
                        plot_density=False,
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False)

############
### DATA ###
############

PATHS_DICT = {"J0030": {"maryland": "./data/J0030/J0030_RM_maryland.txt",
                        "amsterdam": "./data/J0030/ST_PST__M_R.txt"}}

# TODO: to generalize
PSR_NAME = "J0030"
maryland_path = PATHS_DICT[PSR_NAME]["maryland"]
amsterdam_path = PATHS_DICT[PSR_NAME]["amsterdam"]

# Load the radius-mass posterior samples from the data
maryland_samples = pd.read_csv(maryland_path, sep=" ", names=["R", "M", "weight"] , skiprows = 6)
if pd.isna(maryland_samples["weight"]).any():
	print("Warning: weights not properly specified, assuming constant weights instead.")
	maryland_samples["weight"] = np.ones_like(maryland_samples["weight"])
amsterdam_samples = pd.read_csv(amsterdam_path, sep=" ", names=["weight", "M", "R"])

# Construct KDE # TODO: Hauke takes only a subset of the samples, why?
maryland_posterior = gaussian_kde([maryland_samples["M"], maryland_samples["R"]], weights = maryland_samples["weight"])
amsterdam_posterior = gaussian_kde([amsterdam_samples["M"], amsterdam_samples["R"]], weights = amsterdam_samples["weight"])

### PRIORS ###

L_sym_prior = UniformPrior(20.0, 150.0, parameter_names=["L_sym"])
K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
K_sat_prior = UniformPrior(200.0, 300.0, parameter_names=["K_sat"])

prior_list = [L_sym_prior, 
              K_sym_prior, 
              K_sat_prior
]

prior = CombinePrior(prior_list)
sampled_param_names = prior.parameter_names

### LIKELIHOOD ###

class NICERLikelihood():
    
    def __init__(self,
                 sampled_NEP_param_names: list[str],
                 # metamodel kwargs:
                 nmin_nsat: float = 0.1, # TODO: check this value? Spikes?
                 nbreak_nsat: float = 2,
                 ndat_metamodel: int = 100,
                 # CSE kwargs
                 nmax_nsat: float = 15,
                 nb_CSE: int = 7,
                 fixed_CSE_grid: bool = True,
                 # TOV kwargs
                 min_nsat_TOV: float = 1.0,
                 ndat_TOV: int = 50,
                 ndat_CSE: int = 50,
                 # likelihood calculation kwargs
                 delta_m: float = 0.02,
                 ):
        
        self.delta_m = delta_m
        self.nmin_nsat = nmin_nsat
        self.nbreak_nsat = nbreak_nsat
        self.ndat_metamodel = ndat_metamodel
        self.nmax_nsat = nmax_nsat
        self.nb_CSE = nb_CSE
        self.fixed_CSE_grid = fixed_CSE_grid
        self.min_nsat_TOV = min_nsat_TOV
        self.ndat_TOV = ndat_TOV
        self.ndat_CSE = ndat_CSE
        
        # Remove those NEPs from the fixed values that we sample over
        self.fixed_NEP = copy.deepcopy(NICER_utils.NEP_CONSTANTS_DICT)
        for name in sampled_NEP_param_names:
            if name in list(self.fixed_NEP.keys()):
                self.fixed_NEP.pop(name)
            
        # Construct a jitted lambda function for solving the TOV equations
        self.construct_family_jit = jax.jit(lambda x: construct_family(x,
                                                                       ndat = self.ndat_TOV, 
                                                                       min_nsat = self.min_nsat_TOV))
                
        # TODO: add some tests to check if nb_CSE matches
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        
        params.update(self.fixed_NEP)
        
        print("params")
        print(params)
        
        # Metamodel part
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        
        ngrids = jnp.array([params[f"n_CSE_{i}"] for i in range(self.nb_CSE)])
        cs2grids = jnp.array([params[f"cs2_CSE_{i}"] for i in range(self.nb_CSE)])
        
        print("ngrids")
        print(ngrids)
        
        print("cs2grids")
        print(cs2grids)
        
        # Create the EOS
        eos = MetaModel_with_CSE_EOS_model(
                    NEP,
                    self.nbreak_nsat,
                    ngrids,
                    cs2grids,
                    nmin_nsat=self.nmin_nsat,
                    nmax_nsat=self.nmax_nsat,
                    ndat_metamodel=self.ndat_metamodel,
                    ndat_CSE=self.ndat_CSE,
                )
        
        # TODO: might want to check for "good indices" here?
        
        eos_tuple = (
            eos.n,
            eos.p,
            eos.h,
            eos.e,
            eos.dloge_dlogp
        )
        
        # Solve the TOV equations
        _, masses_EOS, radii_EOS, _ = self.construct_family_jit(eos_tuple)
        M_TOV = np.max(masses_EOS)
        
        # Create a grid of masses for the likelihood calculation
        m_array = np.arange(0, M_TOV, self.delta_m)
        r_array = np.interp(m_array, masses_EOS, radii_EOS)
        
        # Evaluate for Maryland
        mr_grid = jnp.vstack([m_array, r_array])
        logy_maryland = maryland_posterior.logpdf(mr_grid)
        logL_maryland = logsumexp(logy_maryland) - np.log(len(logy_maryland))
        
        # Evaluate for Amsterdam
        logy_amsterdam = amsterdam_posterior.logpdf(mr_grid)
        logL_amsterdam = logsumexp(logy_amsterdam) - np.log(len(logy_amsterdam))
        
        L_maryland = np.exp(logL_maryland)
        L_amsterdam = np.exp(logL_amsterdam)
        L = 1/2 * (L_maryland + L_amsterdam)
        
        return L

##############
### CORNER ###
##############

### Plot a cornerplot of the data

# TODO: do an inference problem and plot all the M, R curves on top coloured by their log-likelihood as per computed above

# Prepare general corner kwargs first
corner_kwargs = default_corner_kwargs.copy()
hist_kwargs = {"density": True}
corner_kwargs["labels"] = ["$R$ [km]", "$M$ [$M_{\odot}$]"]
corner_kwargs["fill_contours"] = True
corner_kwargs["alpha"] = 1.0
corner_kwargs["zorder"] = 1e9
corner_kwargs["no_fill_contours"] = True
corner_kwargs["plot_contours"] = True

corner_kwargs["color"] = "red"
hist_kwargs["color"] = "red"

corner_kwargs["hist_kwargs"] = hist_kwargs
corner_kwargs["range"] = [(8, 17), (0.8, 2.25)]
fig = corner.corner(amsterdam_samples[["R", "M"]], weights=amsterdam_samples["weight"], **corner_kwargs)

corner_kwargs["color"] = "blue"
hist_kwargs["color"] = "blue"
corner_kwargs["hist_kwargs"] = hist_kwargs
fig = corner.corner(maryland_samples[["R", "M"]], fig=fig, weights=maryland_samples["weight"], **corner_kwargs)

fs = 24
plt.text(0.65, 0.8, "Maryland", color="blue", fontsize=fs, transform=plt.gcf().transFigure)
plt.text(0.65, 0.7, "Amsterdam", color="red", fontsize=fs, transform=plt.gcf().transFigure)

plt.savefig(f"./figures/corner_{PSR_NAME}.png")
plt.close()

print("DONE")