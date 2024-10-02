import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
from jax.scipy.stats import gaussian_kde
import pandas as pd
import copy
from functools import partial

from jimgw.base import LikelihoodBase
from jimgw.transforms import NtoMTransform
from jimgw.prior import UniformPrior, CombinePrior
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD

from joseTOV.eos import MetaModel_with_CSE_EOS_model, MetaModel_EOS_model, construct_family
from joseTOV import utils

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
    "cs2_CSE_8": 0.9,
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

PSR_PATHS_DICT = {"J0030": {"maryland": "../../../data/J0030/J0030_RM_maryland.txt",
                            "amsterdam": "../../../data/J0030/ST_PST__M_R.txt"},
                  "J0740": {"maryland": "../../../data/J0740/J0740_NICERXMM_full_mr.txt",
                            "amsterdam": "../../../data/J0740/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat"}}
SUPPORTED_PSR_NAMES = list(PSR_PATHS_DICT.keys()) # we do not include the most recent PSR for now

data_samples_dict: dict[str, dict[str, pd.Series]] = {}
kde_dict: dict[str, dict[str, gaussian_kde]] = {}

for psr_name in PSR_PATHS_DICT.keys():

    # Get the paths
    maryland_path = PSR_PATHS_DICT[psr_name]["maryland"]
    amsterdam_path = PSR_PATHS_DICT[psr_name]["amsterdam"]

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

prex_posterior = gaussian_kde(np.loadtxt("../../../data/PREX/PREX_samples.txt", skiprows = 1).T)
crex_posterior = gaussian_kde(np.loadtxt("../../../data/CREX/CREX_samples.txt", skiprows = 1).T)

kde_dict["PREX"] = prex_posterior
kde_dict["CREX"] = crex_posterior

##################
### TRANSFORMS ###
##################

class MicroToMacroTransform(NtoMTransform):
    
    def __init__(self,
                 name_mapping: tuple[list[str], list[str]],
                 keep_names: list[str] = None,
                 # metamodel kwargs:
                 ndat_metamodel: int = 100,
                 # CSE kwargs
                 nmax_nsat: float = 25,
                 nb_CSE: int = 8,
                 # TOV kwargs
                 min_nsat_TOV: float = 1.0,
                 ndat_TOV: int = 100,
                 ndat_CSE: int = 100,
                 nb_masses: int = 100,
                 fixed_params: dict[str, float] = None
                ):
    
        # By default, keep all names
        if keep_names is None:
            keep_names = name_mapping[0]
        super().__init__(name_mapping, keep_names=keep_names)
    
        # Save as attributes
        self.ndat_metamodel = ndat_metamodel
        self.nmax_nsat = nmax_nsat
        self.nmax = nmax_nsat * 0.16
        self.nb_CSE = nb_CSE
        self.min_nsat_TOV = min_nsat_TOV
        self.ndat_TOV = ndat_TOV
        self.ndat_CSE = ndat_CSE
        self.nb_masses = nb_masses
        
        # Create the EOS object
        if nb_CSE > 0:
            eos = MetaModel_with_CSE_EOS_model(nmax_nsat=self.nmax_nsat,
                                               ndat_metamodel=self.ndat_metamodel,
                                               ndat_CSE=self.ndat_CSE,
                    )
            self.transform_func = self.transform_func_MM_CSE
        else:
            eos = MetaModel_EOS_model(nmax_nsat = self.nmax_nsat,
                                      ndat = self.ndat_metamodel)
        
            self.transform_func = self.transform_func_MM
        
        self.eos = eos
        
        # Remove those NEPs from the fixed values that we sample over
        if fixed_params is None:
            fixed_params = copy.deepcopy(NEP_CONSTANTS_DICT)
        
        self.fixed_params = fixed_params 
        for name in self.name_mapping[0]:
            if name in list(self.fixed_params.keys()):
                self.fixed_params.pop(name)
            
        # Construct a lambda function for solving the TOV equations, fix the given parameters
        self.construct_family_lambda = lambda x: construct_family(x, ndat = self.ndat_TOV, min_nsat = self.min_nsat_TOV)
        
    def transform_func_MM(self, params: dict[str, Float]) -> dict[str, Float]:
        
        params.update(self.fixed_params)
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP)
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        
        # Solve the TOV equations
        _, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
        
        return_dict = {"masses_EOS": masses_EOS, "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS,
                       "n": ns, "p": ps, "h": hs, "e": es, "dloge_dlogp": dloge_dlogps, "cs2": cs2}
        
        return return_dict


    def transform_func_MM_CSE(self, params: dict[str, Float]) -> dict[str, Float]:
        
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
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP, ngrids, cs2grids)
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        
        # Solve the TOV equations
        _, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
        
        # M_TOV = jnp.max(masses_EOS)
        # # Create a grid of masses for the likelihood calculation
        # m_array = jnp.linspace(0, M_TOV, self.nb_masses)
        # r_array = jnp.interp(m_array, masses_EOS, radii_EOS)
        # Lambdas_array = jnp.interp(m_array, masses_EOS, Lambdas_EOS)
        
        return_dict = {"masses_EOS": masses_EOS, "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS,
                       "n": ns, "p": ps, "h": hs, "e": es, "dloge_dlogp": dloge_dlogps, "cs2": cs2}
        
        return return_dict
    
def detector_frame_M_c_q_to_source_frame_m_1_m_2(params: dict) -> dict:
    
    M_c, q, d_L = params['M_c'], params['q'], params['d_L']
    H0 = params.get('H0', 67.4) # (km/s) / Mpc
    c = params.get('c', 299_792.4580) # km / s
    
    # Calculate source frame chirp mass
    z = d_L * H0 * 1e3 / c
    M_c_source = M_c / (1.0 + z)

    # Get source frame mass_1 and mass_2
    M_source = M_c_source * (1.0 + q) ** 1.2 / q**0.6
    m_1_source = M_source / (1.0 + q)
    m_2_source = M_source * q / (1.0 + q)

    return {'m_1': m_1_source, 'm_2': m_2_source}

class ChirpMassMassRatioToSourceComponentMasses(NtoMTransform):
        
    def __init__(
        self,
    ):
        name_mapping = (["M_c", "q", "d_L"], ["m_1", "m_2"])
        super().__init__(name_mapping=name_mapping, keep_names = "all")
        
        self.transform_func = detector_frame_M_c_q_to_source_frame_m_1_m_2
        
class ChirpMassMassRatioToLambdas(NtoMTransform):
    
    def __init__(
        self,
        name_mapping,
    ):
        super().__init__(name_mapping=name_mapping, keep_names = "all")
        
        self.mass_transform = ChirpMassMassRatioToSourceComponentMasses()
        
    def transform_func(self, params: dict[str, Float]) -> dict[str, Float]:
        
        masses_EOS = params["masses_EOS"]
        Lambdas_EOS = params["Lambdas_EOS"]
        
        # Get masses
        m_params = self.mass_transform.forward(params)
        m_1, m_2 = m_params["m_1"], m_params["m_2"]
        
        # Interpolate to get Lambdas
        lambda_1_interp = jnp.interp(m_1, masses_EOS, Lambdas_EOS, right = -1.0)
        lambda_2_interp = jnp.interp(m_2, masses_EOS, Lambdas_EOS, right = -1.0)
        
        return {"lambda_1": lambda_1_interp, "lambda_2": lambda_2_interp}
        
        
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
   
#############
### PRIOR ###
#############

my_nbreak = 2.0 * 0.16
NMAX_NSAT = 25
NMAX = NMAX_NSAT * 0.16
N = 100
NB_CSE = 8
width = (NMAX - my_nbreak) / (NB_CSE + 1)

### NEP priors
K_sat_prior = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat_prior = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
Z_sat_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])

E_sym_prior = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym_prior = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
Q_sym_prior = UniformPrior(-800.0, 800.0, parameter_names=["Q_sym"])
Z_sym_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sym"])

prior_list = [
    E_sym_prior,
    L_sym_prior, 
    K_sym_prior,
    Q_sym_prior,
    Z_sym_prior,

    K_sat_prior,
    Q_sat_prior,
    Z_sat_prior,
]

### CSE priors
prior_list.append(UniformPrior(1.0 * 0.16, 2.0 * 0.16, parameter_names=[f"nbreak"]))
for i in range(NB_CSE):
    left = my_nbreak + i * width
    right = my_nbreak + (i+1) * width
    prior_list.append(UniformPrior(left, right, parameter_names=[f"n_CSE_{i}"]))
    prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))

# Final point to end
prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))
prior = CombinePrior(prior_list)
sampled_param_names = prior.parameter_names
name_mapping = (sampled_param_names, ["masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])