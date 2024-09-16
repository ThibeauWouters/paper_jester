"""
Benchmarking the jose solver

TODO: implement it
"""

################
### PREAMBLE ###
################
import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"
import json
from typing import Callable

import time
import tqdm
import numpy as np
np.random.seed(43) # for reproducibility
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import corner

import jax
import jaxlib
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

print(jax.devices())

from jimgw.prior import UniformPrior, CombinePrior
from jimgw.jim import Jim
from joseTOV.eos import Interpolate_EOS_model, MetaModel_EOS_model, MetaModel_with_CSE_EOS_model, construct_family
from joseTOV import utils
from scipy.interpolate import interp1d
import paper_jose.utils as utils

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
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)

NEP_CONSTANTS_DICT = {
    "E_sat": -16,
    "K_sat": 230,
    "Q_sat": 0,
    "Z_sat": 0,
    
    "E_sym": 32,
    "L_sym": 50,
    "K_sym": 0,
    "Q_sym": 0,
    "Z_sym": 0
}

#################
### UTILITIES ###
#################

def SE(y_true: np.array, y_pred: np.array) -> np.array:
    """
    Squared error. 
    """
    return (y_true - y_pred)**2

def RSE(y_true: np.array, y_pred: np.array) -> np.array:
    """
    Relative squared error. 
    """
    return (y_true - y_pred)**2 / y_true**2

def RAE(y_true: np.array, y_pred: np.array) -> np.array:
    """
    Relative absolute error. 
    """
    return np.abs(y_true - y_pred) / y_true

def get_TOV_errors(m1: np.array,
                   r1: np.array,
                   l1: np.array,
                   m2: np.array,
                   r2: np.array,
                   l2: np.array,
                   error_fn: Callable = RAE,
                   use_TOV_limit: bool = True):
    """
    Index 1 is best taken to be the truth (e.g. Rahul's code), or more reliable model at least. So jose is index 2
    """
    
    if use_TOV_limit:
        negative_diffs = np.where(np.diff(m2) < 0.0)
        if len(negative_diffs) > 0:
            idx = negative_diffs[0][0]
            m2 = m2[:idx]
            r2 = r2[:idx]
            l2 = l2[:idx]
    
    assert len(m1) == len(r1), "Mismatch m1 and r1 shapes"
    assert len(m1) == len(l1), "Mismatch m1 and l1 shapes"
    
    assert len(m2) == len(r2), "Mismatch m2 and r2 shapes"
    assert len(m2) == len(l2), "Mismatch m2 and l2 shapes"
    
    # Pass on to unique, since we might have repeated masses due to MTOV limit
    m2 = np.unique(m2)
    r2 = np.unique(r2)
    l2 = np.unique(l2)
    
    # Get the predicted values of the second set at the masses of the first set
    r2_pred = interp1d(m2, r2, kind = "linear", fill_value = "extrapolate")(m1)
    l2_pred = interp1d(m2, l2, kind = "linear", fill_value = "extrapolate")(m1)
    
    # Compute the error
    errors_r = error_fn(r1, r2_pred)
    errors_l = error_fn(l1, l2_pred)
    
    return m1, errors_r, errors_l

def get_Delta_value(m: np.array, 
                    errors_r: np.array, 
                    errors_l: np.array,
                    m_value: float = 1.4):
    """
    Compute Delta_1.4 (or for another mass) as defined from given error arrays.
    """
    delta_r = interp1d(m, errors_r, kind = "linear", fill_value = "extrapolate")(m_value)
    delta_l = interp1d(m, errors_l, kind = "linear", fill_value = "extrapolate")(m_value)
    return delta_r, delta_l


###########################
### AUXILIARY FUNCTIONS ###
###########################

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

def load_micro_eos(micro_filename: str):
    """
    Load a micro EOS from one of Rahul's files.

    Args:
        micro_filename (str): Filename of the micro EOS. 

    Returns:
        tuple: Tuple of the n, p and e values.
    """
    micro_eos = np.genfromtxt(micro_filename)
    n, p, e = micro_eos[:, 0], micro_eos[:, 1], micro_eos[:, 2]
    return n, p, e

def load_macro_eos(macro_filename: str) -> tuple[list, list, list]:
    """
    Loads the macro EOS of given input file from Rahul's set
    NOTE: the order of mass radius and lambdas is a bit weird and TODO: should be changed perhaps to avoid confusion?

    Args:
        macro_filename (str): Filename.

    Returns:
        tuple[list, list, list]: Radius, mass and lambdas.
    """
    macro_eos = np.genfromtxt(macro_filename)
    r, m, l = macro_eos[:, 0], macro_eos[:, 1], macro_eos[:, 2]
    return r, m, l


class EOS_Sampler:
    
    def __init__(self,
                 N: int,
                 prior: CombinePrior,
                 transform: utils.MicroToMacroTransform,
                 make_plots: bool = False,
                 save_values: bool = False,
                 solve_nmma: bool = True,
                 outdir: str = "./outdir/",
                 clean_outdir: bool = False,
                 ):
        """
        Class for sampling the NEP values for the metamodel
        """
        
        self.prior = prior 
        self.transform = transform
        self.make_plots = make_plots
        self.save_values = save_values
        self.solve_nmma = solve_nmma
        self.outdir = outdir
        self.N = N
        self.key = jax.random.PRNGKey(42)
        self.counter = 0
        
        # If the outdir exists, then clean it:
        if clean_outdir:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            else:
                for filename in os.listdir(outdir):
                    if filename.endswith(".npz"):
                        os.remove(os.path.join(outdir, filename))
    
    
    def single_realization(self):
        
        key, subkey = jax.random.split(self.key)
        self.key = key
        
        # Sample a point from the prior
        params = self.prior.sample(subkey, 1)
        for key, value in params.items():
            if isinstance(value, jnp.ndarray):
                params[key] = value.at[0].get()
        
        # # Get EOS
        # if self.counter == 0:
        #     # Do the JIT compilation
        #     _ = jax.jit(self.transform.forward)(params)
            
        start = time.time()
        eos = self.transform.forward(params)
        end = time.time()
        runtime = end - start
        
        # Save it
        np.savez(os.path.join(self.outdir, f"{self.counter}.npz"), eos=eos, params=params, runtime=runtime)
        
        self.counter += 1
        
def report_timings(outdir: str = "./outdir/",
                   take_log: bool = True):
    
    # Load all the timings:
    timings = []
    timings_dict = {}
    for filename in os.listdir(outdir):
        if filename == "0.npz":
            continue
        if filename.endswith(".npz"):
            data = np.load(os.path.join(outdir, filename))
            timings.append(data["runtime"])
            
            # Append to the dict as well
            timings_dict[filename] = data["runtime"]
            
    timings = np.array(timings)
    
    # Make histogram
    plt.figure()
    if take_log:
        timings = np.log10(timings)
        plt.hist(timings, bins = 20, density = True)
        plt.xlabel(r"$\log_{10}(t)$")
    else:
        plt.hist(timings, bins = 20, density = True)
        plt.xlabel("Runtime [s]")
    plt.ylabel("Density")
    plt.title("Runtime histogram")
    plt.savefig("./figures/runtime_histogram.png", bbox_inches = "tight")
    
        
def main():
    
    NMAX_NSAT = 25
    NMAX = NMAX_NSAT * 0.16
    NB_CSE = 0
    my_nbreak = 2.0 * 0.16
    width = (NMAX - my_nbreak) / (NB_CSE + 1)
    
    ### NEP priors
    
    L_sym_prior = UniformPrior(20.0, 150.0, parameter_names=["L_sym"])
    K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
    K_sat_prior = UniformPrior(200.0, 300.0, parameter_names=["K_sat"])

    prior_list = [
        L_sym_prior, 
        K_sym_prior,
        K_sat_prior,
    ]
    
    ### CSE priors # TODO: implement
    
    prior = CombinePrior(prior_list)
    
    # Make the transform object
    name_mapping = (prior.parameter_names, ["masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])
    transform = utils.MicroToMacroTransform(name_mapping, 
                                            nmax_nsat=NMAX_NSAT,
                                            nb_CSE = NB_CSE,
                                            )
    N = 1_000

    sampler = EOS_Sampler(N, 
                          prior,
                          transform,
                          clean_outdir = False,)
    
    ## Do the sampling and TOV solving
    
    # for _ in tqdm.tqdm(range(N)):
    #     sampler.single_realization()
        
    report_timings()

if __name__ == "__main__":
    main()