import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
from jax.scipy.stats import gaussian_kde
import pandas as pd
import copy
from functools import partial

from joseTOV.eos import MetaModel_with_CSE_EOS_model, construct_family
from joseTOV import utils
from jimgw.base import LikelihoodBase
from jimgw.transforms import NtoMTransform


################
### PLOTTING ###
################

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

#################
### CONSTANTS ###
#################

# Taken from emulators paper Ingo and Rahul
NEP_CONSTANTS_DICT = {
    "E_sym": 32,
    "L_sym": 50,
    "K_sym": 0,
    "Q_sym": 0,
    "Z_sym": 0,
    
    "E_sat": -16,
    "K_sat": 230,
    "Q_sat": 0,
    "Z_sat": 0,
    
    "nbreak": 1.5,
    
    "n_CSE_0": 3 * 0.16,
    "n_CSE_1": 4 * 0.16,
    "n_CSE_2": 5 * 0.16,
    "n_CSE_3": 6 * 0.16,
    "n_CSE_4": 7 * 0.16,
    "n_CSE_5": 8 * 0.16,
    "n_CSE_6": 9 * 0.16,
    "n_CSE_7": 10 * 0.16,
    
    "cs2_CSE_0": 0.5,
    "cs2_CSE_1": 0.7,
    "cs2_CSE_2": 0.5,
    "cs2_CSE_3": 0.4,
    "cs2_CSE_4": 0.8,
    "cs2_CSE_5": 0.6,
    "cs2_CSE_6": 0.9,
    "cs2_CSE_7": 0.8,
}

def merge_dicts(dict1: dict, dict2: dict):
    """
    Merges 2 dicts, but if the key is already in dict1, it will not be overwritten by dict2.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary. Do not use its values if keys are in dict1
    """
    
    result = {}
    for key, value in dict1.items():
        result[key] = value
        
    for key, value in dict2.items():
        if key not in result.keys():
            result[key] = value
            
    return result

############
### DATA ###
############

PRS_PATHS_DICT = {"J0030": {"maryland": "../data/J0030/J0030_RM_maryland.txt",
                            "amsterdam": "../data/J0030/ST_PST__M_R.txt"},
                  "J0740": {"maryland": "../data/J0740/J0740_NICERXMM_full_mr.txt",
                            "amsterdam": "../data/J0740/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat"}}
SUPPORTED_PSR_NAMES = list(PRS_PATHS_DICT.keys()) # we do not include the most recent PSR for now

data_samples_dict: dict[str, dict[str, pd.Series]] = {}
kde_dict: dict[str, dict[str, gaussian_kde]] = {}

for psr_name in PRS_PATHS_DICT.keys():

    # Get the paths
    maryland_path = PRS_PATHS_DICT[psr_name]["maryland"]
    amsterdam_path = PRS_PATHS_DICT[psr_name]["amsterdam"]

    # Load the radius-mass posterior samples from the data
    maryland_samples = pd.read_csv(maryland_path, sep=" ", names=["R", "M", "weight"] , skiprows = 6)
    if pd.isna(maryland_samples["weight"]).any():
        print("Warning: weights not properly specified, assuming constant weights instead.")
        maryland_samples["weight"] = np.ones_like(maryland_samples["weight"])
        
    if psr_name == "J0030":
        amsterdam_samples = pd.read_csv(amsterdam_path, sep=" ", names=["weight", "M", "R"])
    else:
        amsterdam_samples = pd.read_csv(amsterdam_path, sep=" ", names=["M", "R"])
        amsterdam_samples["weight"] = np.ones_like(amsterdam_samples["M"])

    # Get as samples and as KDE
    maryland_data_2d = jnp.array([maryland_samples["M"].values, maryland_samples["R"].values])
    amsterdam_data_2d = jnp.array([amsterdam_samples["M"].values, amsterdam_samples["R"].values])

    maryland_posterior = gaussian_kde(maryland_data_2d, weights = maryland_samples["weight"].values)
    amsterdam_posterior = gaussian_kde(amsterdam_data_2d, weights = amsterdam_samples["weight"].values)
    
    data_samples_dict[psr_name] = {"maryland": maryland_samples, "amsterdam": amsterdam_samples}
    kde_dict[psr_name] = {"maryland": maryland_posterior, "amsterdam": amsterdam_posterior}

prex_posterior = gaussian_kde(np.loadtxt("../data/PREX/PREX_samples.txt", skiprows = 1).T)
crex_posterior = gaussian_kde(np.loadtxt("../data/CREX/CREX_samples.txt", skiprows = 1).T)

kde_dict["PREX"] = prex_posterior
kde_dict["CREX"] = crex_posterior

#################
### TRANSFORM ###
#################

class MicroToMacroTransform(NtoMTransform):
    
    def __init__(self,
                 name_mapping: tuple[list[str], list[str]],
                 keep_names: list[str] = [],
                 # metamodel kwargs:
                 nmin_nsat: float = 0.1, # TODO: check this value? Spikes?
                 ndat_metamodel: int = 100,
                 # CSE kwargs
                 nmax_nsat: float = 25,
                 nb_CSE: int = 8,
                 # TOV kwargs
                 min_nsat_TOV: float = 1.0,
                 ndat_TOV: int = 100,
                 ndat_CSE: int = 100,
                 nb_masses: int = 100,
                ):
    
        super().__init__(name_mapping, keep_names=keep_names)
    
        # Save as attributes
        self.nmin_nsat = nmin_nsat
        self.ndat_metamodel = ndat_metamodel
        self.nmax_nsat = nmax_nsat
        self.nmax = nmax_nsat * 0.16
        self.nb_CSE = nb_CSE
        self.min_nsat_TOV = min_nsat_TOV
        self.ndat_TOV = ndat_TOV
        self.ndat_CSE = ndat_CSE
        self.nb_masses = nb_masses
        
        # Create the EOS object
        eos = MetaModel_with_CSE_EOS_model(nmin_nsat=self.nmin_nsat,
                                           nmax_nsat=self.nmax_nsat,
                                           ndat_metamodel=self.ndat_metamodel,
                                           ndat_CSE=self.ndat_CSE,
                )
        self.eos = eos
        
        # Remove those NEPs from the fixed values that we sample over
        self.fixed_params = copy.deepcopy(NEP_CONSTANTS_DICT)
        for name in self.name_mapping[0]:
            if name in list(self.fixed_params.keys()):
                self.fixed_params.pop(name)
            
        # Construct a lambda function for solving the TOV equations, fix the given parameters
        self.construct_family_lambda = lambda x: construct_family(x, ndat = self.ndat_TOV, min_nsat = self.min_nsat_TOV)
        
    def transform_func(self, params: dict[str, Float]) -> dict[str, Float]:
        
        params.update(self.fixed_params)
        
        # Separate the MM and CSE parameters
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        NEP["nbreak"] = params["nbreak"]
        
        ngrids = jnp.array([params[f"n_CSE_{i}"] for i in range(self.nb_CSE)])
        cs2grids = jnp.array([params[f"cs2_CSE_{i}"] for i in range(self.nb_CSE)])
        
        # Append the final cs2 value, which is fixed at nmax 
        ngrids = jnp.append(ngrids, jnp.array([self.nmax]))
        cs2grids = jnp.append(cs2grids, jnp.array([params[f"cs2_CSE_{self.nb_CSE}"]]))
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, _ = self.eos.construct_eos(NEP, ngrids, cs2grids)
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        
        # Solve the TOV equations
        _, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
        
        # M_TOV = jnp.max(masses_EOS)
        # # Create a grid of masses for the likelihood calculation
        # m_array = jnp.linspace(0, M_TOV, self.nb_masses)
        # r_array = jnp.interp(m_array, masses_EOS, radii_EOS)
        # Lambdas_array = jnp.interp(m_array, masses_EOS, Lambdas_EOS)
        
        return_dict = {"masses_EOS": masses_EOS, "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS}
        
        return return_dict
        
##################
### LIKELIHOOD ###
##################

class NICERLikelihood(LikelihoodBase):
    
    def __init__(self,
                 psr_name: str,
                 transform: MicroToMacroTransform = None,
                 # likelihood calculation kwargs
                 nb_masses: int = 100):
        
        # TODO: remove me
        self.psr_name = psr_name
        self.transform = transform
        self.counter = 0
        self.nb_masses = nb_masses
        
        # Load the data
        self.amsterdam_posterior = kde_dict[psr_name]["amsterdam"]
        self.maryland_posterior = kde_dict[psr_name]["maryland"]
        
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        m, r, Lambdas = params["masses_EOS"], params["radii_EOS"], params["Lambdas_EOS"]
        
        mr_grid = jnp.vstack([m, r])
        logy_maryland = self.maryland_posterior.logpdf(mr_grid)
        logL_maryland = logsumexp(logy_maryland) - jnp.log(len(logy_maryland))
        
        # Evaluate for Amsterdam
        logy_amsterdam = self.amsterdam_posterior.logpdf(mr_grid)
        logL_amsterdam = logsumexp(logy_amsterdam) - jnp.log(len(logy_amsterdam))
        
        L_maryland = jnp.exp(logL_maryland)
        L_amsterdam = jnp.exp(logL_amsterdam)
        L = 1/2 * (L_maryland + L_amsterdam)
        log_likelihood = jnp.log(L)
        
        # Save: # NOTE: this can only be used if we are not jitting/vmapping over the likelihood
        np.savez(f"./computed_data/{self.counter}.npz", masses_EOS = m, radii_EOS = r, logy_maryland = logy_maryland, logy_amsterdam = logy_amsterdam, L=L)
        self.counter += 1
        
        return log_likelihood
    
class REXLikelihood(LikelihoodBase):
    
    def __init__(self,
                 experiment_name: str,
                 # likelihood calculation kwargs
                 nb_masses: int = 100):
        
        # TODO: remove me
        assert experiment_name in ["PREX", "CREX"], "Only PREX and CREX are supported as experiment name arguments."
        self.experiment_name = experiment_name
        self.counter = 0
        self.nb_masses = nb_masses
        
        # Load the data
        self.posterior: gaussian_kde = kde_dict[experiment_name]
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        log_likelihood_array = self.posterior.logpdf(jnp.array([params["E_sym"], params["L_sym"]]))
        log_likelihood = log_likelihood_array.at[0].get()
        
        ## For testing/debugging:
        try:
            m, r = params["masses_EOS"], params["radii_EOS"]
            # Save: # NOTE: this can only be used if we are not jitting/vmapping over the likelihood
            np.savez(f"./computed_data/{self.counter}.npz", masses_EOS = m, radii_EOS = r, L=log_likelihood)
            self.counter += 1
            
        except Exception as e:
            print(e)
        
        return log_likelihood
    
class CombinedLikelihood(LikelihoodBase):
    
    def __init__(self,
                 likelihoods_list: list[LikelihoodBase],
                 transform: MicroToMacroTransform = None):
        
        # TODO: remove transform input?
        
        super().__init__()
        self.likelihoods_list = likelihoods_list
        self.transform = transform
        self.counter = 0
        
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        
        all_log_likelihoods = jnp.array([likelihood.evaluate(params, data) for likelihood in self.likelihoods_list])
        return jnp.sum(all_log_likelihoods)
    
class ZeroLikelihood(LikelihoodBase):
    def __init__(self,
                 transform: MicroToMacroTransform = None):
        
        # TODO: remove transform input?
        
        super().__init__()
        self.transform = transform
        self.counter = 0
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        m, r = params["masses_EOS"], params["radii_EOS"]
        # Save: # NOTE: this can only be used if we are not jitting/vmapping over the likelihood
        np.savez(f"./computed_data/{self.counter}.npz", masses_EOS = m, radii_EOS = r, L=0.0)
        self.counter += 1
        return 0.0