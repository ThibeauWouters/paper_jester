"""
Test the robustness of universal relations with jose
"""

################
### PREAMBLE ###
################

import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
import shutil

import os
import tqdm
import copy
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

import paper_jose.utils as utils
import paper_jose.inference.utils_plotting as utils_plotting
plt.rcParams.update(utils_plotting.mpl_params)
import seaborn as sns

###########################
### UNIVERSAL RELATIONS ### 
###########################


def binary_love(lambda_symmetric: float, 
                mass_ratio: float,
                fit_coeffs: dict) -> float:
    """
    Computes lambda_antysymmetric from lambda_symmetric and mass_ratio. Note that this is only the fit, whereas typically the uncertainty would be marginalized over. See the CHZ paper: arXiv:1804.03221v2
    The code is copied from bilby/gw/conversion.py, changing np to jnp for JAX compatibility.
    
    Note: We take the fit coefficients as input, rather than hardcoding them, to allow for changing the fit coefficients later on.
    """
    lambda_symmetric_m1o5 = jnp.power(lambda_symmetric, -1. / 5.)
    lambda_symmetric_m2o5 = lambda_symmetric_m1o5 * lambda_symmetric_m1o5
    lambda_symmetric_m3o5 = lambda_symmetric_m2o5 * lambda_symmetric_m1o5

    q = mass_ratio
    q2 = jnp.square(mass_ratio)

    # Eqn.2 from CHZ, incorporating the dependence on mass ratio
    q_for_Fnofq = jnp.power(q, 10. / (3. - fit_coeffs["n_polytropic"]))
    Fnofq = (1. - q_for_Fnofq) / (1. + q_for_Fnofq)

    # Eqn 1 from CHZ, giving the lambda_antisymmetric_fitOnly (not yet accounting for the uncertainty in the fit)
    numerator = 1.0 + \
        (fit_coeffs["b11"] * q * lambda_symmetric_m1o5) + (fit_coeffs["b12"] * q2 * lambda_symmetric_m1o5) + \
        (fit_coeffs["b21"] * q * lambda_symmetric_m2o5) + (fit_coeffs["b22"] * q2 * lambda_symmetric_m2o5) + \
        (fit_coeffs["b31"] * q * lambda_symmetric_m3o5) + (fit_coeffs["b32"] * q2 * lambda_symmetric_m3o5)

    denominator = 1.0 + \
        (fit_coeffs["c11"] * q * lambda_symmetric_m1o5) + (fit_coeffs["c12"] * q2 * lambda_symmetric_m1o5) + \
        (fit_coeffs["c21"] * q * lambda_symmetric_m2o5) + (fit_coeffs["c22"] * q2 * lambda_symmetric_m2o5) + \
        (fit_coeffs["c31"] * q * lambda_symmetric_m3o5) + (fit_coeffs["c32"] * q2 * lambda_symmetric_m3o5)

    lambda_antisymmetric_fitOnly = Fnofq * lambda_symmetric * numerator / denominator

    return lambda_antisymmetric_fitOnly

# These are the coefficients reported in the CHZ paper: see Table I of CHZ
BINARY_LOVE_COEFFS = {
    "n_polytropic": 0.743,

    "b11": -27.7408,
    "b12": 8.42358,
    "b21": 122.686,
    "b22": -19.7551,
    "b31": -175.496,
    "b32": 133.708,
    
    "c11": -25.5593,
    "c12": 5.58527,
    "c21": 92.0337,
    "c22": 26.8586,
    "c31": -70.247,
    "c32": -56.3076
}

BINARY_LOVE_COEFFS_GODZIEBA = {
    "n_polytropic": 0.743,

    "b11": -18.32,
    "b12": 3.875,
    "b21": 28.06,
    "b22": -11.08,
    "b31": 43.56,
    "b32": 17.3,
    
    "c11": -18.37,
    "c12": 1.338,
    "c21": 15.99,
    "c22": 55.07,
    "c31": 98.56,
    "c32": -135.1
}

################
### SCORE_FN ### 
################


class UniversalRelationsScoreFn:
    """This is a class that stores stuff that might simplify the calculation of the error on the universal relations"""
    
    
    def __init__(self,
                 max_nb_eos: int = 100_000,
                 random_samples_outdir: str = "../doppelgangers/random_samples/",
                 nb_mass_samples: int = 1_000,
                 fixed_params: dict = {},
                 m_minval: float = 1.2,
                 m_maxval: float = 2.1,
                 m_length: int = 1_000):
        
        """
        Args
        ----
        max_nb_eos: int
            Maximum number of EOS to consider
        random_samples_outdir: str
            Directory where the EOS are stored
        m_minval: float
            Minimum value for the mass
        m_maxval: float
            Maximum value for the mass
        m_length: int
            Number of mass samples on which we evaluate the universal relations
        """
        
        # Set some attributes
        self.random_samples_outdir = random_samples_outdir
        self.max_nb_eos = max_nb_eos
        self.nb_mass_samples = nb_mass_samples
        self.m_minval = m_minval
        self.m_maxval = m_maxval
        self.m_length = m_length
        self.fixed_params = fixed_params
        
        # Create the mass array on which to interpolate the EOSs
        self.mass_array = jnp.linspace(self.m_minval, self.m_maxval, self.m_length)
        
        # Set the masses array to be used
        key = jax.random.PRNGKey(1)
        key, subkey = jax.random.split(key)
        first_batch = jax.random.uniform(subkey, shape=(self.nb_mass_samples,), minval = self.m_minval, maxval = self.m_maxval)
        key, subkey = jax.random.split(key)
        second_batch = jax.random.uniform(subkey, shape=(self.nb_mass_samples,), minval = self.m_minval, maxval = self.m_maxval)
        
        # Get the masses and ensure that m1 > m2
        self.m1 = jnp.maximum(first_batch, second_batch)
        self.m2 = jnp.minimum(first_batch, second_batch)
        self.q = self.m2 / self.m1
        
        print("Loading the EOS data in the UniversalRelationsScoreFn constructor")
        complete_lambda1_array = []
        complete_lambda2_array = []
        
        for i, file in enumerate(tqdm.tqdm(os.listdir(random_samples_outdir))):
            data = np.load(f"../doppelgangers/random_samples/{file}")
            _m, _l = data["masses_EOS"], data["Lambdas_EOS"]
            
            lambdas_array = jnp.interp(self.mass_array, _m, _l)
            
            if i > self.max_nb_eos:
                print("Max nb of EOS reached, quitting the loop")
                break
            
            lambda1_array = jnp.interp(self.m1, self.mass_array, lambdas_array)
            lambda2_array = jnp.interp(self.m2, self.mass_array, lambdas_array)
            
            # Stack it:
            complete_lambda1_array.append(lambda1_array)
            complete_lambda2_array.append(lambda2_array)
            
        # Convert to numpy array
        self.complete_lambda1_array = jnp.array(complete_lambda1_array)
        self.complete_lambda2_array = jnp.array(complete_lambda2_array)
        
        # Get the "true" Lambda_a and Lambda_s:
        self.lambda_symmetric = 0.5 * (self.complete_lambda1_array + self.complete_lambda2_array)
        self.lambda_asymmetric = 0.5 * (self.complete_lambda2_array - self.complete_lambda1_array)
        
    def score_fn(self, params: dict):
        """
        Params: in this case, this is the dict containing the parameters required by Binary love
        """
        
        # Call binary love
        params.update(self.fixed_params)
        binary_love_result = binary_love(self.lambda_symmetric, self.q, params)
        
        # Get the list of errors
        errors = (self.lambda_asymmetric - binary_love_result) / self.lambda_asymmetric
        
        # Convert to final error # TODO: choose the appropriate metric
        error = jnp.mean(abs(errors))
        
        return error
    
class UniversalRelationBreaker:
    
    def __init__(self, 
                 nb_mass_samples: int = 1_000,
                 m_minval: float = 1.2,
                 m_maxval: float = 2.1,
                 m_length: int = 1_000):
        
        # Create the mass array on which to interpolate the EOSs
        self.nb_mass_samples = nb_mass_samples
        self.m_minval = m_minval
        self.m_maxval = m_maxval
        self.m_length = m_length
        self.mass_array = jnp.linspace(self.m_minval, self.m_maxval, self.m_length)
        
        # Set the masses array to be used
        self.key = jax.random.PRNGKey(1)
        self.key, self.subkey = jax.random.split(self.key)
        
    def score_fn(self, 
                 params: dict,
                 transform: utils.MicroToMacroTransform,
                 binary_love_params: dict = BINARY_LOVE_COEFFS,
                 return_aux: bool = True):
        """
        Params are here the EOS params
        """
        
        # Solve TOV
        out = transform.forward(params)
        masses_EOS, Lambdas_EOS = out["masses_EOS"], out["Lambdas_EOS"]
        error = self.compute_error_from_NS(masses_EOS, Lambdas_EOS, binary_love_params, return_aux)
        
        if return_aux:
            return error, out
        else:
            return error
        
    def compute_error_from_NS(self, 
                              masses_EOS: Array, 
                              Lambdas_EOS: Array,
                              binary_love_params: dict = BINARY_LOVE_COEFFS):
        # Get the masses
        maxval = jnp.max(masses_EOS)
        first_batch = jax.random.uniform(self.subkey, shape=(self.nb_mass_samples,), minval = self.m_minval, maxval = maxval)
        self.key, self.subkey = jax.random.split(self.key)
        second_batch = jax.random.uniform(self.subkey, shape=(self.nb_mass_samples,), minval = self.m_minval, maxval = maxval)
        
        # Get the masses and ensure that m1 > m2
        self.m1 = jnp.maximum(first_batch, second_batch)
        self.m2 = jnp.minimum(first_batch, second_batch)
        self.q = self.m2 / self.m1
        
        # Interpolate the EOS
        lambda_1 = jnp.interp(self.m1, masses_EOS, Lambdas_EOS)
        lambda_2 = jnp.interp(self.m2, masses_EOS, Lambdas_EOS)
        
        # Get the symmetric and antisymmetric Lambdas
        lambda_symmetric  = 0.5 * (lambda_1 + lambda_2)
        lambda_asymmetric = 0.5 * (lambda_2 - lambda_1)
        
        # Call binary love
        binary_love_result = binary_love(lambda_symmetric, self.q, binary_love_params)
        
        # Evaluate the error
        errors = (lambda_asymmetric - binary_love_result) / lambda_asymmetric
        error = jnp.mean(abs(errors))
        
        return error
        
    # def score_fn(self, 
    #              params: dict,
    #              transform: utils.MicroToMacroTransform,
    #              binary_love_params: dict = BINARY_LOVE_COEFFS,
    #              return_aux: bool = True):
    #     """
    #     Params are here the EOS params
    #     """
        
    #     # Solve TOV
    #     out = transform.forward(params)
    #     masses_EOS, Lambdas_EOS = out["masses_EOS"], out["Lambdas_EOS"]
    #     maxval = jnp.max(masses_EOS)
        
    #     # Get the masses
    #     first_batch = jax.random.uniform(self.subkey, shape=(self.nb_mass_samples,), minval = self.m_minval, maxval = maxval)
    #     self.key, self.subkey = jax.random.split(self.key)
    #     second_batch = jax.random.uniform(self.subkey, shape=(self.nb_mass_samples,), minval = self.m_minval, maxval = maxval)
        
    #     # Get the masses and ensure that m1 > m2
    #     self.m1 = jnp.maximum(first_batch, second_batch)
    #     self.m2 = jnp.minimum(first_batch, second_batch)
    #     self.q = self.m2 / self.m1
        
    #     # Interpolate the EOS
    #     lambda_1 = jnp.interp(self.m1, masses_EOS, Lambdas_EOS)
    #     lambda_2 = jnp.interp(self.m2, masses_EOS, Lambdas_EOS)
        
    #     # Get the symmetric and antisymmetric Lambdas
    #     lambda_symmetric  = 0.5 * (lambda_1 + lambda_2)
    #     lambda_asymmetric = 0.5 * (lambda_2 - lambda_1)
        
    #     # Call binary love
    #     binary_love_result = binary_love(lambda_symmetric, self.q, binary_love_params)
        
    #     # Evaluate the error
    #     errors = (lambda_asymmetric - binary_love_result) / lambda_asymmetric
    #     error = jnp.mean(abs(errors))
        
    #     if return_aux:
    #         return error, out
    #     else:
    #         return error
        
        
################
### PLOTTING ###
################

def plot_binary_Love(binary_love_params: dict = BINARY_LOVE_COEFFS,
                     q_values: list[float] = [0.5, 0.75, 0.90, 0.99],
                     nb_samples: int = 50,
                     nb_eos: int = 100,
                     plot_binary_love: bool = True,
                     name: str = "default"
                     ):
    
    print("Making test plot for universal relation")
    key = jax.random.PRNGKey(1)
    plt.figure(figsize = (14, 10))
    colors = ["red", "green", "orange", "blue"]
    
    legend_fontsize = 21
    for i in tqdm.tqdm(range(nb_eos)):
        
        # Load from random_samples
        data = np.load(f"../doppelgangers/random_samples/{i}.npz")
        m, l = data["masses_EOS"], data["Lambdas_EOS"]
        
        # Sample the masses
        M_min, M_max = jnp.min(m), jnp.max(m)
        key, subkey = jax.random.split(key)
        m1_sampled = jax.random.uniform(subkey, shape=(nb_samples,), minval = M_min, maxval = M_max)
        
        for color, q in zip(colors, q_values):
            
            m2_sampled = q * m1_sampled
            
            # Get Lambdas:
            lambda1_sampled = jnp.interp(m1_sampled, m, l)
            lambda2_sampled = jnp.interp(m2_sampled, m, l)
            
            # Get lamda_symmetric and lambda_asymmetric
            lambda_symmetric_sampled  = 0.5 * (lambda2_sampled + lambda1_sampled)
            lambda_asymmetric_sampled = 0.5 * (lambda2_sampled - lambda1_sampled)
            
            # Plot it:
            if i == 0:
                plt.plot(lambda_symmetric_sampled, lambda_asymmetric_sampled, 'o', color = color, label = f"q = {q}", rasterized = True, zorder = 3, alpha = 0.5)
            else:
                plt.plot(lambda_symmetric_sampled, lambda_asymmetric_sampled, 'o', color = color, rasterized = True, zorder = 3, alpha = 0.5)
    
    # Plot the binary Love relation on top:
    if plot_binary_love:
        print("Plotting binary Love relation")
        lambda_symmetric_values = jnp.linspace(0, 5_000, 100)
        for i, q in enumerate(q_values):
            lambda_asymmetric_values = binary_love(lambda_symmetric_values, q, binary_love_params)
            if i == 0:
                plt.plot(lambda_symmetric_values, lambda_asymmetric_values, "--", linewidth = 4, color = "black", label = "Binary Love", zorder = 100)
            else:
                plt.plot(lambda_symmetric_values, lambda_asymmetric_values, "--", linewidth = 4, color = "black", zorder = 100)
            
            plt.fill_between(lambda_symmetric_values, 0.9 * lambda_asymmetric_values, 1.1 * lambda_asymmetric_values, color = "black", alpha = 0.25)
    
    print("Finalizing plot and saving it")
    plt.legend(fontsize = legend_fontsize)
    plt.xlabel(r"$\Lambda_{\rm s}$")
    plt.ylabel(r"$\Lambda_{\rm a}$")
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)
    plt.savefig(f"./figures/test_binary_love_{name}.png", bbox_inches = "tight")
    plt.savefig(f"./figures/test_binary_love_{name}.pdf", bbox_inches = "tight")

    plt.close()
    
def get_histograms(binary_love_params: dict = BINARY_LOVE_COEFFS,
                   q_values = [0.5, 0.75, 0.90, 0.99],
                   nb_samples = 100,
                   name: str = "default"):
    """
    Exploratory phase: plot histograms of fractional differences in Lambdas for different mass ratios
    """
    
    m1_sampled = jax.random.uniform(jax.random.PRNGKey(2), shape=(nb_samples,), minval = 1.2, maxval = 2.1)
    
    for q in tqdm.tqdm(q_values):
        all_errors = []
        for i, file in enumerate(tqdm.tqdm(os.listdir("../doppelgangers/random_samples/"))):
            data = np.load(f"../doppelgangers/random_samples/{file}")
            m, l = data["masses_EOS"], data["Lambdas_EOS"]
            
            # Sample the masses
            m2_sampled = q * m1_sampled
                
            # Get Lambdas:
            lambda1_sampled = jnp.interp(m1_sampled, m, l)
            lambda2_sampled = jnp.interp(m2_sampled, m, l)
            
            # Only keep those that are below certainn value - sufficient to check Lambda2 (largest), take 10 000  for CHZ paper
            mask = (lambda2_sampled < 10_000)
            lambda1_sampled = lambda1_sampled[mask]
            lambda2_sampled = lambda2_sampled[mask]
                
            # Get lamda_symmetric and lambda_asymmetric
            lambda_symmetric_sampled  = 0.5 * (lambda2_sampled + lambda1_sampled)
            lambda_asymmetric_sampled = 0.5 * (lambda2_sampled - lambda1_sampled)
            binary_love_values = binary_love(lambda_symmetric_sampled, q, binary_love_params)
            
            if not jnp.all(jnp.isfinite(lambda_asymmetric_sampled)):
                raise ValueError(f"File {i} has non-finite values in lambda_asymmetric_sampled")
            
            if not jnp.all(jnp.isfinite(binary_love_values)):
                raise ValueError(f"File {i} has non-finite values in binary_love_values")

            # Get the errors and add to the list
            errors = 100 * (lambda_asymmetric_sampled - binary_love_values) / lambda_asymmetric_sampled

            # Only add finite values:
            # Check if infinite errors:
            if not jnp.all(jnp.isfinite(errors)):
                nb_infinite = jnp.sum(jnp.isinf(errors))
                print(f"File {i} has nb_infinite = {nb_infinite}")
            
            errors = errors[jnp.isfinite(errors)]
            all_errors.extend(errors)

        # Make histogram
        plt.figure(figsize = (14, 10))
        plt.hist(abs(errors), bins = 20, color = "blue", linewidth = 4, label = f"q = {q}", density = True, histtype = "step")
        plt.xlabel(r"$\frac{\Lambda_{\rm a} - \Lambda_{\rm a}^{\rm fit}}{\Lambda_{\rm a}}$ (\%)", fontsize = 21)
        plt.ylabel("Density")
        plt.savefig(f"./figures/histogram_q_{q}_{name}.png", bbox_inches = "tight")
        plt.savefig(f"./figures/histogram_q_{q}_{name}.pdf", bbox_inches = "tight")
        plt.close()
        
def assess_binary_love_improvement_godzieba(binary_love_params_list: list[dict],
                                            names_list: list[str] = ["Default", "Recalibrated"],
                                            colors_list: list[str] = ["blue", "green"],
                                            nb_samples: int = 100,
                                            max_nb_eos: int = 1_000,
                                            eos_dir: str = "../doppelgangers/random_samples/",
                                            m_minval: float = 1.2,
                                            m_maxval: float = 2.1):
    """
    Make a plot similar to Fig 9 from arXiv:2012.12151v1
    
    Args
    ----
    nb_samples: int
        Number of samples to draw from the EOS
    max_nb_eos: int
        Maximum number of EOS to consider
    eos_dir: str
        Directory where the EOS are stored
    m_minval: float
        Minimum value for the mass
    m_maxval: float
        Maximum value for the mass
    """
    
    # Setup
    plt.subplots(nrows = 2, ncols = 2, figsize = (14, 10), sharey = True)
    all_files = os.listdir(eos_dir)
    
    # Iterate over the different binary love parameters sets
    for idx, (name, binary_love_params, color) in enumerate(zip(names_list, binary_love_params_list, colors_list)):
        key = jax.random.PRNGKey(1)
        key, subkey = jax.random.split(key)
        # Loop over all desired EOS files
        for i, file in enumerate(tqdm.tqdm(all_files)):
            if i >= max_nb_eos:
                print("Max number of iterations reached, exiting the loop now")
                break
            
            # Generate the masses for the plot
            first_batch = jax.random.uniform(subkey, shape=(nb_samples,), minval = m_minval, maxval = m_maxval)
            key, subkey = jax.random.split(key)
            second_batch = jax.random.uniform(subkey, shape=(nb_samples,), minval = m_minval, maxval = m_maxval)
            
            m1_sampled = jnp.maximum(first_batch, second_batch)
            m2_sampled = jnp.minimum(first_batch, second_batch)
            q = m2_sampled / m1_sampled
            
            # Load the EOS
            data = np.load(eos_dir + f"{i}.npz")
            m, l = data["masses_EOS"], data["Lambdas_EOS"]
            
            try: 
                # Get the Lambdas
                lambda1_sampled = jnp.interp(m1_sampled, m, l)
                lambda2_sampled = jnp.interp(m2_sampled, m, l)
                
                # Get the symmetric and antisymmetric Lambdas
                lambda_symmetric_sampled  = 0.5 * (lambda2_sampled + lambda1_sampled)
                lambda_asymmetric_sampled = 0.5 * (lambda2_sampled - lambda1_sampled)
                
                # Mask lambda_symmetric_sampled below 4000
                mask = (lambda_symmetric_sampled < 4000)
                lambda_symmetric_sampled = lambda_symmetric_sampled[mask]
                lambda_asymmetric_sampled = lambda_asymmetric_sampled[mask]
                q = q[mask]
            
                # Get binary Love
                binary_love_values = binary_love(lambda_symmetric_sampled, m2_sampled / m1_sampled, binary_love_params)
                
                # Get errors
                errors = (lambda_asymmetric_sampled - binary_love_values) / lambda_asymmetric_sampled
                
                # Limit error due to bad EOS for the plots
                mask = (abs(errors) < 1)
                errors = errors[mask]
                lambda_symmetric_sampled = lambda_symmetric_sampled[mask]
                q = q[mask]
                
            except Exception as e:
                print(f"Error with file {file}: {e}")
                continue
            
            # Plot
            if idx == 0:
                a, b = 1, 3
            else:
                a, b = 2, 4
            plt.subplot(int(f"22{a}"))
            plt.plot(lambda_symmetric_sampled, errors, 'o', color = color, alpha = 0.5, rasterized = True)
            plt.xlabel(r"$\Lambda_{\rm s}$")
            plt.ylabel(r"$(\Lambda_{\rm a} - \Lambda_{\rm a}^{\rm fit}) / \Lambda_{\rm a}$")
            plt.subplot(int(f"22{b}"))
            plt.xlabel(r"$q$")
            plt.ylabel(r"$(\Lambda_{\rm a} - \Lambda_{\rm a}^{\rm fit}) / \Lambda_{\rm a}$")
            plt.plot(q, errors, 'o', color = color, alpha = 0.5, rasterized = True)
        
    plt.savefig(f"./figures/improvement_godzieba_plot.png", bbox_inches = "tight")
    plt.savefig(f"./figures/improvement_godzieba_plot.pdf", bbox_inches = "tight")
    plt.close()
        
def make_godzieba_plot(binary_love_params: dict = BINARY_LOVE_COEFFS,
                       nb_samples: int = 100,
                       max_nb_eos: int = 1_000,
                       eos_dir: str = "../doppelgangers/random_samples/",
                       m_minval: float = 1.2,
                       m_maxval: float = 2.1,
                       name: str = "default"):
    """
    Make a plot similar to Fig 9 from arXiv:2012.12151v1
    
    Args
    ----
    nb_samples: int
        Number of samples to draw from the EOS
    max_nb_eos: int
        Maximum number of EOS to consider
    eos_dir: str
        Directory where the EOS are stored
    m_minval: float
        Minimum value for the mass
    m_maxval: float
        Maximum value for the mass
    """
    
    # Setup
    plt.subplots(nrows = 2, ncols = 1, figsize = (14, 10))
    key = jax.random.PRNGKey(1)
    key, subkey = jax.random.split(key)
    all_files = os.listdir(eos_dir)
    
    # Loop over all desired EOS files
    for i, file in enumerate(tqdm.tqdm(all_files)):
        if i >= max_nb_eos:
            print("Max number of iterations reached, exiting the loop now")
            break
        
        # Generate the masses for the plot
        first_batch = jax.random.uniform(subkey, shape=(nb_samples,), minval = m_minval, maxval = m_maxval)
        key, subkey = jax.random.split(key)
        second_batch = jax.random.uniform(subkey, shape=(nb_samples,), minval = m_minval, maxval = m_maxval)
        
        m1_sampled = jnp.maximum(first_batch, second_batch)
        m2_sampled = jnp.minimum(first_batch, second_batch)
        q = m2_sampled / m1_sampled
        
        # Load the EOS
        data = np.load(eos_dir + f"{i}.npz")
        m, l = data["masses_EOS"], data["Lambdas_EOS"]
        
        try: 
            # Get the Lambdas
            lambda1_sampled = jnp.interp(m1_sampled, m, l)
            lambda2_sampled = jnp.interp(m2_sampled, m, l)
            
            # Get the symmetric and antisymmetric Lambdas
            lambda_symmetric_sampled  = 0.5 * (lambda2_sampled + lambda1_sampled)
            lambda_asymmetric_sampled = 0.5 * (lambda2_sampled - lambda1_sampled)
            
            # Mask lambda_symmetric_sampled below 4000
            mask = (lambda_symmetric_sampled < 4000)
            lambda_symmetric_sampled = lambda_symmetric_sampled[mask]
            lambda_asymmetric_sampled = lambda_asymmetric_sampled[mask]
            q = q[mask]
        
            # Get binary Love
            binary_love_values = binary_love(lambda_symmetric_sampled, m2_sampled / m1_sampled, binary_love_params)
            
            # Get errors
            errors = lambda_asymmetric_sampled - binary_love_values
        except Exception as e:
            print(f"Error with file {file}: {e}")
            continue
        
        # Plot
        plt.subplot(211)
        plt.plot(lambda_symmetric_sampled, errors, 'o', color = "blue", alpha = 0.5, rasterized = True)
        plt.subplot(212)
        plt.plot(q, errors, 'o', color = "blue", alpha = 0.5, rasterized = True)
        
    # Make the final plot and save
    plt.subplot(211)
    plt.xlabel(r"$\Lambda_{\rm s}$")
    plt.ylabel(r"$\Lambda_{\rm a} - \Lambda_{\rm a}^{\rm fit}$")
    
    plt.subplot(212)
    plt.xlabel(r"$q$")
    plt.ylabel(r"$\Lambda_{\rm a} - \Lambda_{\rm a}^{\rm fit}$")
    plt.savefig(f"./figures/improvement_binary_Love_{name}.png", bbox_inches = "tight")
    plt.close()
    
def make_combined_godzieba_plot(binary_love_params: dict = BINARY_LOVE_COEFFS,
                                nb_samples: int = 100,
                                max_nb_eos: int = 1_000,
                                eos_dir_list: list[str] = ["../doppelgangers/random_samples/"],
                                colors_list: list[str] = ["blue"],
                                m_minval: float = 1.2,
                                m_maxval: float = 2.1):
    """
    Make a plot similar to Fig 9 from arXiv:2012.12151v1
    
    Args
    ----
    nb_samples: int
        Number of samples to draw from the EOS
    max_nb_eos: int
        Maximum number of EOS to consider
    eos_dir: str
        Directory where the EOS are stored
    m_minval: float
        Minimum value for the mass
    m_maxval: float
        Maximum value for the mass
    """
    
    print("Making combined Godzieba plot")
    
    # Setup
    plt.subplots(nrows = 2, ncols = 1, figsize = (14, 10))
    key = jax.random.PRNGKey(1)
    key, subkey = jax.random.split(key)
    all_files_list = [os.listdir(eos_dir) for eos_dir in eos_dir_list]
    
    # Loop over all desired EOS files
    alpha_list = [0.01, 0.2]
    for all_files, eos_dir, color, alpha in zip(all_files_list, eos_dir_list, colors_list, alpha_list):
        print(f"Checking the combined Godzieba plot for eos dir = {eos_dir}")
        for i, file in enumerate(tqdm.tqdm(all_files)):
            if i >= max_nb_eos:
                print("Max number of iterations reached, exiting the loop now")
                break
            
            # Generate the masses for the plot
            first_batch = jax.random.uniform(subkey, shape=(nb_samples,), minval = m_minval, maxval = m_maxval)
            key, subkey = jax.random.split(key)
            second_batch = jax.random.uniform(subkey, shape=(nb_samples,), minval = m_minval, maxval = m_maxval)
            
            m1_sampled = jnp.maximum(first_batch, second_batch)
            m2_sampled = jnp.minimum(first_batch, second_batch)
            q = m2_sampled / m1_sampled
            
            # Load the EOS, but there are two options for save formatting... Catch either
            maybe_file = eos_dir + f"{i}.npz"
            try:
                if os.path.exists(maybe_file):
                    file = maybe_file
                else:
                    file = os.path.join(eos_dir, file, "data", "0.npz")
                data = np.load(file)
                m, l = data["masses_EOS"], data["Lambdas_EOS"]
            except Exception as e:
                print(f"Could not load an EOS file: {e}")
            
            try: 
                # Get the Lambdas
                lambda1_sampled = jnp.interp(m1_sampled, m, l)
                lambda2_sampled = jnp.interp(m2_sampled, m, l)
                
                # Get the symmetric and antisymmetric Lambdas
                lambda_symmetric_sampled  = 0.5 * (lambda2_sampled + lambda1_sampled)
                lambda_asymmetric_sampled = 0.5 * (lambda2_sampled - lambda1_sampled)
                
                # Mask lambda_symmetric_sampled below 4000
                mask = (lambda_symmetric_sampled < 4000)
                lambda_symmetric_sampled = lambda_symmetric_sampled[mask]
                lambda_asymmetric_sampled = lambda_asymmetric_sampled[mask]
                q = q[mask]
            
                # Get binary Love
                binary_love_values = binary_love(lambda_symmetric_sampled, m2_sampled / m1_sampled, binary_love_params)
                
                # Get errors
                errors = lambda_asymmetric_sampled - binary_love_values
            except Exception as e:
                print(f"Error with file {file}: {e}")
                continue
            
            # Plot
            plt.subplot(211)
            plt.plot(lambda_symmetric_sampled, errors, 'o', color = color, alpha = alpha, rasterized = True)
            plt.subplot(212)
            plt.plot(q, errors, 'o', color = color, alpha = alpha, rasterized = True)
        
    # Make the final plot and save
    plt.subplot(211)
    plt.xlabel(r"$\Lambda_{\rm s}$")
    plt.ylabel(r"$\Lambda_{\rm a} - \Lambda_{\rm a}^{\rm fit}$")
    plt.ylim(-200, 200)
    
    plt.subplot(212)
    plt.xlabel(r"$q$")
    plt.ylabel(r"$\Lambda_{\rm a} - \Lambda_{\rm a}^{\rm fit}$")
    save_name = f"./figures/godzieba_plot_combined.png"
    print(f"Saving combined Godzieba plot to {save_name}")
    plt.ylim(-200, 200)
    
    print("Saving the combined Godzieba plot")
    plt.savefig(save_name, bbox_inches = "tight")
    plt.close()
    
############
### MAIN ### 
############

def run(score_fn_object: UniversalRelationsScoreFn,
        params: dict,
        nb_steps: int = 200,
        optimization_sign: float = -1,
        learning_rate: float = 1e-3):
    
    print("Starting parameters:")
    print(params)
    
    print("Computing by gradient ascent . . .")
    pbar = tqdm.tqdm(range(nb_steps))
    
    # Note: this does not return aux, contrary to doppelgangers run
    score_fn = jax.value_and_grad(score_fn_object.score_fn)
    
    for i in pbar:
        score, grad = score_fn(params)
        pbar.set_description(f"Iteration {i}: Score {score}")
        
        # Save: 
        np.savez(f"./outdir/{i}.npz", score = score, **params)
        
        # Do the updates
        params = {key: value + optimization_sign * learning_rate * grad[key] for key, value in params.items()}
        
    print("Computing DONE")
    
    return params
    
def assess_binary_love_accuracy(outdir_list: list[str] = ["../doppelgangers/random_samples/", "../doppelgangers/outdir/"],
                                colors_list: list[str] = ["blue", "red"],
                                binary_love_params: dict = BINARY_LOVE_COEFFS,
                                error_threshold: float = 1.0,
                                save_name: str = "./figures/accuracy_binary_love_error.png"):
    """
    Assessing how accurate a binary love relation is for a given batch of EOS in outdir and with the given binary Love parameters.
    
    Args
    ----
    outdir: str
        Directory where the EOS are stored
    binary_love_params: dict
        Parameters for the binary Love relation
    """
    
    # Do the optimization
    score_fn_object = UniversalRelationBreaker(nb_mass_samples = 1_000)
    errors_dict = {}
    
    # Iterate over the EOS and compute the error and append to the dict
    for outdir in outdir_list:
        print(f"Processing for: {outdir}")
        errors = []
        for _, file in enumerate(tqdm.tqdm(os.listdir(outdir))):
            try:
                if "random_samples" in outdir:
                    data = np.load(f"{outdir}/{file}")
                else:
                    file = os.path.join(outdir, file, "data", "0.npz")
                    data = np.load(file)
            except Exception as e:
                print(f"Could not load file {file}: {e}")
                continue
            
            masses_EOS, Lambdas_EOS = data["masses_EOS"], data["Lambdas_EOS"]
            error = score_fn_object.compute_error_from_NS(masses_EOS, Lambdas_EOS, binary_love_params)
            errors.append(error)
            
        print(f"Dropping all with error above {error_threshold}")
        errors = np.array(errors)
        errors = errors[errors < error_threshold]
        
        errors_dict[outdir] = errors
            
    # Make a histogram
    hist_kwargs = {"bins": 20, 
                   "linewidth": 4,
                   "density": True,
                   "histtype": "step"}
    
    plt.figure(figsize = (14, 10))
    for outdir, color in zip(outdir_list, colors_list):
        plt.hist(errors_dict[outdir], color = color, label = outdir, **hist_kwargs)
    
    plt.axvline(0.10, color = "black", linestyle = "--", linewidth = 4, label = "10 percent error")
    plt.xlabel("Error in binary Love")
    plt.ylabel("Density")
    plt.legend()
    print(f"Saving to {save_name}")
    plt.savefig(save_name, bbox_inches = "tight")
    plt.close()
        
    return errors
    
def compute_cdf(samples):
    # Sort the samples
    sorted_samples = np.sort(samples)
    
    # Compute the CDF values
    cdf_values = np.arange(1, len(samples) + 1) / len(samples)
    
    return sorted_samples, cdf_values
    
def assess_binary_love_improvement(params_list: list[dict],
                                   names_list: list[str] = ["Default", "Recalibrated"],
                                   colors_list: list[str] = ["blue", "green"],
                                   outdir: str = "../doppelgangers/random_samples/",
                                   error_threshold: float = 1.0,
                                   save_name: str = "./figures/improvement_binary_love_error.png",
                                   make_histogram: bool = True,
                                   make_cdf: bool = True):
    """
    Assessing how much more accurate a binary love relation has gotten
    
    Args
    ----
    outdir: str
        Directory where the EOS are stored
    binary_love_params: dict
        Parameters for the binary Love relation
    """
    
    # Do the optimization
    score_fn_object = UniversalRelationBreaker(nb_mass_samples = 1_000)
    errors_dict = {}
    
    # Iterate over the EOS and compute the error and append to the dict
    for params, name in zip(params_list, names_list):
        print(f"Processing for: {outdir}")
        errors = []
        for _, file in enumerate(tqdm.tqdm(os.listdir(outdir))):
            try:
                if "random_samples" in outdir:
                    data = np.load(f"{outdir}/{file}")
                else:
                    file = os.path.join(outdir, file, "data", "0.npz")
                    data = np.load(file)
            except Exception as e:
                print(f"Could not load file {file}: {e}")
                continue
            
            masses_EOS, Lambdas_EOS = data["masses_EOS"], data["Lambdas_EOS"]
            error = score_fn_object.compute_error_from_NS(masses_EOS, Lambdas_EOS, params)
            errors.append(error)
            
        print(f"Dropping all with error above {error_threshold}")
        errors = np.array(errors)
        errors = errors[errors < error_threshold]
        
        errors_dict[name] = errors
            
    # Make a histogram
    if make_histogram:
        plt.figure(figsize = (14, 10))
        hist_kwargs = {"bins": 20, 
                    "linewidth": 4,
                    "density": True,
                    "histtype": "step"}
        
        for name, color in zip(names_list, colors_list):
            plt.hist(errors_dict[name], color = color, label = name, **hist_kwargs)
        
        fs = 26
        plt.axvline(0.10, color = "black", linestyle = "--", linewidth = 4, label = "10 percent error")
        plt.xlabel("Binary Love score", fontsize = fs)
        plt.ylabel("Density", fontsize = fs)
        plt.legend(fontsize = fs)
        print(f"Saving to {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        save_name = save_name.replace(".png", ".pdf")
        print(f"Saving to {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        plt.close()
    
    if make_cdf:
        plt.figure(figsize = (14, 10))
        left = 999.0
        for name, color in zip(names_list, colors_list):
            samples = errors_dict[name]
            x_values, cdf_values = compute_cdf(samples)
            left = min(left, x_values[0])
            plt.plot(x_values, cdf_values, color = color, label = name, linewidth = 4)
            
            # Make some statements
            print(f"=== {name} ===")
            print(f"Mean error: {np.mean(samples)}")
            print(f"Median error: {np.median(samples)}")
            cdf_at_10 = np.interp(0.10, x_values, cdf_values)
            print(f"CDF at 0.10 percent: {cdf_at_10}")
            quantile_90 = np.percentile(samples, 90)
            print(f"Quantile 90 percent: {quantile_90}")
            
            plt.axvline(quantile_90, color = color, linestyle = "--", linewidth = 2)
            
        # plt.axvline(0.10, color = "black", linestyle = "--", linewidth = 4, label = "10 percent error")
        plt.axhline(y=0.90, xmin=-1, xmax=1, color = "black", linestyle = "-", linewidth = 2, label = r"90\%")
        plt.xlim(left = left, right = 0.4)
        plt.xlabel("Error in binary Love", fontsize = fs)
        plt.ylabel("CDF", fontsize = fs)
        plt.legend(fontsize = fs)
        print(f"Saving to {save_name.replace('.png', '_cdf.png')}")
        plt.savefig(save_name.replace('.png', '_cdf.png'), bbox_inches = "tight")
        plt.savefig(save_name.replace('.pdf', '_cdf.pdf'), bbox_inches = "tight")
        plt.close()
        
    return errors
    
    
def do_optimization(start_params: dict = BINARY_LOVE_COEFFS,
                    save_name_final_params: str = "./new_binary_love_params.npz",
                    make_plots: bool = False, 
                    keep_fixed: list[str] = []):
    
    # Call some plotting scripts -- exploring to check if the universal relations are working
    if make_plots:
        plot_binary_Love()
        get_histograms()
        make_godzieba_plot()
        
    start_params_copied = copy.deepcopy(start_params)
        
    # Remove the keep_fixed from the params to iterate on
    fixed_params = {key: start_params[key] for key in keep_fixed}
    for key in keep_fixed:
        start_params.pop(key)
        
    # Do the optimization
    score_fn_object = UniversalRelationsScoreFn(max_nb_eos = 100_000,
                                                nb_mass_samples = 1_000,
                                                fixed_params = fixed_params)
    
    # TODO: add keep_fixed
    final_params = run(score_fn_object = score_fn_object,
                       params = start_params,
                       nb_steps = 300,
                       optimization_sign = -1,
                       learning_rate = 1e-1)
    
    # Add the fixed parameters back again
    for key in keep_fixed:
        final_params[key] = start_params_copied[key]
    
    print("Final parameters:")
    print(final_params)
    
    # Save the final parameters
    np.savez(save_name_final_params, **final_params)
    
    # Make a plot for these new parameters
    if make_plots:
        name = "new"
        plot_binary_Love(binary_love_params = final_params, name = name)
        get_histograms(binary_love_params = final_params, name = name)
        make_godzieba_plot(binary_love_params = final_params, name = name)
    
    print("DONE")
    
    
def load_binary_love_params(file: str = "./new_binary_love_params.npz") -> dict:
    """
    Load the binary love parameters from the file
    """
    params = np.load(file)
    params = {key: value for key, value in params.items()}
    return params

def check_score_evolution(outdir: str = "./outdir/",
                          save_name: str = "./figures/score_evolution.png"):
    
    print("Checking the score evolution . . .")
    files = np.array(os.listdir(outdir))
    idx = np.argsort([int(file.split(".")[0]) for file in files])
    files = files[idx]
    
    scores = []
    for file in files:
        full_path = os.path.join(outdir, file)
        data = np.load(full_path)
        score = data["score"]
        scores.append(score)
        
    plt.figure(figsize = (14, 10))
    plt.plot(scores, "-o", linewidth = 4)
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    print("Checking the score evolution . . .")
    plt.savefig(save_name, bbox_inches = "tight")
    plt.close()
    
    
def main():
    
    # ### Make single Godzieba plot
    # make_godzieba_plot()
    
    ### Do optimization
    # do_optimization(keep_fixed = [])
    # check_score_evolution()
    
    # ### Make the combined Godzieba plot -- comparing the EOS (good vs bad, i.e., driven away from Binary Love)
    # eos_dir_list = ["../doppelgangers/random_samples/", "../doppelgangers/outdir/"]
    # colors = ["blue", "red"]
    # make_combined_godzieba_plot(eos_dir_list=eos_dir_list, colors_list = colors)
    
    ### Final assessment of improved binary Love relation -- how much did we improve?
    params = load_binary_love_params()
    # errors = assess_binary_love_accuracy(binary_love_params = params, save_name = "./figures/accuracy_binary_love_error_new.png")
    params_list = [BINARY_LOVE_COEFFS, params]
    assess_binary_love_improvement(params_list)
    # assess_binary_love_improvement_godzieba(params_list)
    
if __name__ == "__main__":
    main()