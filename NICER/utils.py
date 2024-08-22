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
    
    "n_CSE_0": 3 * 0.16,
    "n_CSE_1": 4 * 0.16,
    "n_CSE_2": 5 * 0.16,
    "n_CSE_3": 6 * 0.16,
    "n_CSE_4": 7 * 0.16,
    "n_CSE_5": 8 * 0.16,
    "n_CSE_6": 9 * 0.16,
    "n_CSE_7": 10 * 0.16,
    
    "cs2_CSE_0": 0.5, # TODO: choosing something random here but not sure if smart...
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

##################
### NICER DATA ###
##################

PATHS_DICT = {"J0030": {"maryland": "./data/J0030/J0030_RM_maryland.txt",
                        "amsterdam": "./data/J0030/ST_PST__M_R.txt"}}

# TODO: add support and test for the other pulsars
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
maryland_data_2d = jnp.array([maryland_samples["M"].values, maryland_samples["R"].values])
maryland_posterior = gaussian_kde(maryland_data_2d, weights = maryland_samples["weight"].values)

amsterdam_data_2d = jnp.array([amsterdam_samples["M"].values, amsterdam_samples["R"].values])
amsterdam_posterior = gaussian_kde(amsterdam_data_2d, weights = amsterdam_samples["weight"].values)


###
### LIKELIHOOD ###
###

class NICERLikelihood(LikelihoodBase):
    
    def __init__(self,
                 sampled_NEP_param_names: list[str],
                 nbreak_nsat: float,
                 # metamodel kwargs:
                 nmin_nsat: float = 0.1, # TODO: check this value? Spikes?
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
                 nb_masses: int = 100,
                 ):
        
        self.nmin_nsat = nmin_nsat
        self.nbreak_nsat = nbreak_nsat
        self.ndat_metamodel = ndat_metamodel
        self.nmax_nsat = nmax_nsat
        self.nb_CSE = nb_CSE
        self.fixed_CSE_grid = fixed_CSE_grid
        self.min_nsat_TOV = min_nsat_TOV
        self.ndat_TOV = ndat_TOV
        self.ndat_CSE = ndat_CSE
        self.nb_masses = nb_masses
        
        # Remove those NEPs from the fixed values that we sample over
        self.fixed_NEP = copy.deepcopy(NEP_CONSTANTS_DICT)
        for name in sampled_NEP_param_names:
            if name in list(self.fixed_NEP.keys()):
                self.fixed_NEP.pop(name)
            
        # Construct a jitted lambda function for solving the TOV equations
        construct_family_lambda = lambda x: construct_family(x, ndat = self.ndat_TOV, min_nsat = self.min_nsat_TOV)
        self.construct_family_jit = jax.jit(construct_family_lambda)
                
        # TODO: add some tests to check if nb_CSE matches
        
        # TODO: remove me, this is for initial testing/exploration + might not work when jitted
        self.counter = 0
        
    @partial(jax.jit, static_argnums=(0,))
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
        M_TOV = jnp.max(masses_EOS)
        
        # Create a grid of masses for the likelihood calculation
        m_array = jnp.linspace(0, M_TOV, self.nb_masses)
        r_array = jnp.interp(m_array, masses_EOS, radii_EOS)
        
        # Evaluate for Maryland
        mr_grid = jnp.vstack([m_array, r_array])
        logy_maryland = maryland_posterior.logpdf(mr_grid)
        logL_maryland = logsumexp(logy_maryland) - jnp.log(len(logy_maryland))
        
        # Evaluate for Amsterdam
        logy_amsterdam = amsterdam_posterior.logpdf(mr_grid)
        logL_amsterdam = logsumexp(logy_amsterdam) - jnp.log(len(logy_amsterdam))
        
        L_maryland = jnp.exp(logL_maryland)
        L_amsterdam = jnp.exp(logL_amsterdam)
        L = 1/2 * (L_maryland + L_amsterdam)
        
        # NOTE: has to be removed if we want to use jax.jit or flowMC
        # np.savez(f"./data/{self.counter}.npz", masses_EOS = masses_EOS, radii_EOS = radii_EOS, logy_maryland = logy_maryland, logy_amsterdam = logy_amsterdam, L=L)
        # self.counter += 1
        
        return L