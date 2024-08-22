"""
Playground for NICER inference: we sample some individual parameters, then solve the TOV equations and compute the NICER log likelihood. The results are saved and plotted to visually sanity-check the results. Meant for low number of samples and playing around.
"""

################
### PREAMBLE ###
################

import os
import tqdm
import time
import copy
import numpy as np
np.random.seed(43) # for reproducibility
import matplotlib.pyplot as plt
import matplotlib as mpl
import corner

import jax
import jax.numpy as jnp
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

AMSTERDAM_COLOR = "red"
AMSTERDAM_CMAP = "Reds"
MARYLAND_COLOR = "blue"
MARYLAND_CMAP = "Blues"
EOS_CURVE_COLOR = "darkgreen"

start = time.time()

##############
### PRIORS ###
##############

L_sym_prior = UniformPrior(20.0, 150.0, parameter_names=["L_sym"])
K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
K_sat_prior = UniformPrior(200.0, 300.0, parameter_names=["K_sat"])

my_eps = 1e-3 # small nudge to avoid overlap in CSE gridpoints
NBREAK_NSAT = 2.0

# # TODO: this is a first attempt so pretty cumbersome, improve in the future
# n_CSE_0_prior = UniformPrior(NBREAK_NSAT + my_eps, 3.0 - my_eps, parameter_names=["n_CSE_0"])
# n_CSE_1_prior = UniformPrior(3.0, 4.0 - my_eps, parameter_names=["n_CSE_1"])
# n_CSE_2_prior = UniformPrior(4.0, 5.0 - my_eps, parameter_names=["n_CSE_2"])
# n_CSE_3_prior = UniformPrior(5.0, 6.0 - my_eps, parameter_names=["n_CSE_3"])
# n_CSE_4_prior = UniformPrior(6.0, 7.0 - my_eps, parameter_names=["n_CSE_4"])
# n_CSE_5_prior = UniformPrior(7.0, 8.0 - my_eps, parameter_names=["n_CSE_5"])
# n_CSE_6_prior = UniformPrior(8.0, 9.0 - my_eps, parameter_names=["n_CSE_6"])
# n_CSE_7_prior = UniformPrior(9.0, 10.0 - my_eps, parameter_names=["n_CSE_7"])

# cs2_CSE_0_prior = UniformPrior(0.0, 1.0, parameter_names=["cs2_CSE_0"])
# cs2_CSE_1_prior = UniformPrior(0.0, 1.0, parameter_names=["cs2_CSE_1"])
# cs2_CSE_2_prior = UniformPrior(0.0, 1.0, parameter_names=["cs2_CSE_2"])
# cs2_CSE_3_prior = UniformPrior(0.0, 1.0, parameter_names=["cs2_CSE_3"])
# cs2_CSE_4_prior = UniformPrior(0.0, 1.0, parameter_names=["cs2_CSE_4"])
# cs2_CSE_5_prior = UniformPrior(0.0, 1.0, parameter_names=["cs2_CSE_5"])
# cs2_CSE_6_prior = UniformPrior(0.0, 1.0, parameter_names=["cs2_CSE_6"])
# cs2_CSE_7_prior = UniformPrior(0.0, 1.0, parameter_names=["cs2_CSE_7"])


prior_list = [L_sym_prior, 
              K_sym_prior, 
              K_sat_prior
]

prior = CombinePrior(prior_list)
sampled_param_names = prior.parameter_names

##################
### LIKELIHOOD ###
##################

class NICERLikelihood():
    
    def __init__(self,
                 sampled_NEP_param_names: list[str],
                 # metamodel kwargs:
                 nmin_nsat: float = 0.1, # TODO: check this value? Spikes?
                 nbreak_nsat: float = NBREAK_NSAT,
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
        construct_family_lambda = lambda x: construct_family(x, ndat = self.ndat_TOV, min_nsat = self.min_nsat_TOV)
        self.construct_family_jit = jax.jit(construct_family_lambda)
                
        # TODO: add some tests to check if nb_CSE matches
        
        # TODO: remove me, this is for initial testing/exploration + might not work when jitted
        self.counter = 0
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        
        params.update(self.fixed_NEP)
        
        # Metamodel part
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        
        ngrids = jnp.array([params[f"n_CSE_{i}"] for i in range(self.nb_CSE)])
        cs2grids = jnp.array([params[f"cs2_CSE_{i}"] for i in range(self.nb_CSE)])
        
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
        logy_maryland = NICER_utils.maryland_posterior.logpdf(mr_grid)
        logL_maryland = logsumexp(logy_maryland) - np.log(len(logy_maryland))
        
        # Evaluate for Amsterdam
        logy_amsterdam = NICER_utils.amsterdam_posterior.logpdf(mr_grid)
        logL_amsterdam = logsumexp(logy_amsterdam) - np.log(len(logy_amsterdam))
        
        L_maryland = np.exp(logL_maryland)
        L_amsterdam = np.exp(logL_amsterdam)
        L = 1/2 * (L_maryland + L_amsterdam)
        
        # TODO: for initial testing, save data
        
        np.savez(f"./data/{self.counter}.npz", masses_EOS = masses_EOS, radii_EOS = radii_EOS, logy_maryland = logy_maryland, logy_amsterdam = logy_amsterdam, L=L)
        self.counter += 1
        
        return L

likelihood = NICERLikelihood(sampled_param_names)

##############
### Sample ###
##############

N = 100
jax_key = jax.random.PRNGKey(42)

# for i in tqdm.tqdm(range(N)):
    
#     jax_key, jax_subkey = jax.random.split(jax_key)
    
#     params = prior.sample(jax_subkey, 1)
#     for key, value in params.items():
#         if isinstance(value, jnp.ndarray):
#             params[key] = value.at[0].get()
    
#     L = likelihood.evaluate(params, None)
    
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
for dataset, cmap in zip([NICER_utils.maryland_data_2d, NICER_utils.amsterdam_data_2d], [MARYLAND_CMAP, AMSTERDAM_CMAP]):

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
    data = np.load(f"./data/{i}.npz")
    
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
    
# Then plot all the EOS data
all_L = np.array(all_L)
if np.max(all_L) == np.min(all_L):
    all_L = np.ones_like(all_L)
else:
    all_L = (all_L - np.min(all_L))/(np.max(all_L) - np.min(all_L))

norm = mpl.colors.Normalize(vmin=np.min(all_L), vmax=np.max(all_L))
cmap = mpl.cm.Greens

for i in range(N):
    color = cmap(norm(all_L[i]))  # Get the color from the colormap
    plt.plot(all_radii_EOS[i], all_masses_EOS[i], color=color, alpha=all_L[i], linewidth = 2.0, zorder=1e10)
    
plt.xlim(8, 17)
plt.ylim(0.8, 2.75)
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