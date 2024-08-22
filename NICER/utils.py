import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
from jax.scipy.stats import gaussian_kde
import pandas as pd

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