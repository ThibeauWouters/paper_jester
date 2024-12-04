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

import optax

import paper_jose.utils as utils
import paper_jose.inference.utils_plotting as utils_plotting
plt.rcParams.update(utils_plotting.mpl_params)
import seaborn as sns

VERBOSE = True

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
BINARY_LOVE_COEFFS_CHZ = {
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
    "b22": 11.08,
    "b31": 43.56,
    "b32": 17.3,
    
    "c11": -18.37,
    "c12": 1.338,
    "c21": 15.99,
    "c22": 55.07,
    "c31": 98.56,
    "c32": -135.1
}

BINARY_LOVE_COEFFS = BINARY_LOVE_COEFFS_CHZ

################
### SCORE_FN ### 
################


class UniversalRelationsScoreFn:
    """This is a class that stores stuff that might simplify the calculation of the error on the universal relations"""
    
    
    def __init__(self,
                 max_nb_eos: int = 100_000,
                 random_samples_outdir: str = "../benchmarks/random_samples/",
                 nb_mass_samples: int = 100,
                 fixed_params: dict = {},
                 m_minval: float = 1.0,
                 m_maxval: float = 2.1,
                 m_length: int = 1_000,
                 processed_data_filename: str = None,
                 learning_rate: float = 1e-3,
                 seed: int = 1):
        
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
        print(f"Fetching the EOS from {random_samples_outdir} to do the optimization")
        self.max_nb_eos = max_nb_eos
        self.nb_mass_samples = nb_mass_samples
        self.m_minval = m_minval
        self.m_maxval = m_maxval
        self.m_length = m_length
        self.fixed_params = fixed_params
        self.learning_rate = learning_rate
        
        # Set a random seed
        if seed is None:
            seed = np.random.randint(0, 1_000_000)
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        
        # Set the masses array to be used
        if processed_data_filename is None:
            print("UniversalRelationsScoreFn constructor will make its dataset now")
            key = jax.random.PRNGKey(1)
            
            complete_lambda1_array = []
            complete_lambda2_array = []
            
            complete_m1_array = []
            complete_m2_array = []
            complete_q_array = []
            
            negative_counter = 0
            for i, file in enumerate(tqdm.tqdm(os.listdir(random_samples_outdir))):
                if i == 0:
                    continue
                if i > self.max_nb_eos:
                    print("Max nb of EOS reached, quitting the loop")
                    break
                
                # Load ML curve and interpolate it
                full_filename = os.path.join(random_samples_outdir, file)
                data = np.load(full_filename)
                _m, _l = data["masses_EOS"], data["Lambdas_EOS"]
                mask = _l > 0
                _m = _m[mask]
                _l = _l[mask]
                
                mtov = jnp.max(_m)
                mass_array = jnp.linspace(self.m_minval, mtov, self.m_length)
                lambdas_array = jnp.interp(mass_array, _m, _l)
                
                # Mask away the negative lambdas and interpolate through them later on:
                nb_negative = jnp.sum(lambdas_array < 0)
                if nb_negative > 0:
                    print(f"File {i} has {nb_negative} negative lambdas, skipping")
                    negative_counter += 1
                    continue
                
                # Get masses for evaluation
                self.key, self.subkey = jax.random.split(key)
                first_batch = jax.random.uniform(self.subkey, shape=(self.nb_mass_samples,), minval = self.m_minval, maxval = mtov)
                self.key, self.subkey = jax.random.split(self.key)
                second_batch = jax.random.uniform(self.subkey, shape=(self.nb_mass_samples,), minval = self.m_minval, maxval = mtov)
                
                # Ensure that m1 > m2
                m1 = jnp.maximum(first_batch, second_batch)
                m2 = jnp.minimum(first_batch, second_batch)
                q = m2 / m1
                
                # Get the lambdas
                lambda1_array = jnp.interp(m1, mass_array, lambdas_array)
                lambda2_array = jnp.interp(m2, mass_array, lambdas_array)
                
                # Save the data
                complete_m1_array.append(m1)
                complete_m2_array.append(m2)
                complete_q_array.append(q)
                complete_lambda1_array.append(lambda1_array)
                complete_lambda2_array.append(lambda2_array)
                
            # Convert to numpy array
            self.m1 = jnp.array(complete_m1_array)
            self.m2 = jnp.array(complete_m2_array)
            self.q = jnp.array(complete_q_array)
            self.complete_lambda1_array = jnp.array(complete_lambda1_array)
            self.complete_lambda2_array = jnp.array(complete_lambda2_array)
            
            print(f"Total nb negative: {negative_counter}")
            
            # Check for NaNs
            if any([jnp.isnan(self.complete_lambda1_array).any(), jnp.isnan(self.complete_lambda2_array).any()]):
                raise ValueError("There are NaNs in the lambda arrays")
            
            if any([jnp.isnan(self.m1).any(), jnp.isnan(self.m2).any(), jnp.isnan(self.q).any()]):
                raise ValueError("There are NaNs in the mass arrays")
            
            # Check for negative Lambdas as well:
            if any([jnp.any(self.complete_lambda1_array < 0), jnp.any(self.complete_lambda2_array < 0)]):
                raise ValueError("There are negative Lambdas")
            
            # Check for infs:
            if any([jnp.any(jnp.isinf(self.complete_lambda1_array)), jnp.any(jnp.isinf(self.complete_lambda2_array))]):
                raise ValueError("There are infs in the lambda arrays")
            
            if any([jnp.any(jnp.isinf(self.m1)), jnp.any(jnp.isinf(self.m2)), jnp.any(jnp.isinf(self.q))]):
                raise ValueError("There are infs in the mass arrays")
            
            # Get the "true" Lambda_a and Lambda_s:
            self.lambda_symmetric = 0.5 * (self.complete_lambda1_array + self.complete_lambda2_array)
            self.lambda_asymmetric = 0.5 * (self.complete_lambda2_array - self.complete_lambda1_array)
            
            # Save the processed data to an array
            print("Saving the processed data to processed_data.npz")
            np.savez("processed_data.npz", lambda_symmetric = self.lambda_symmetric, lambda_asymmetric = self.lambda_asymmetric,
                    m1 = self.m1, m2 = self.m2, q = self.q)
        else:
            print(f"UniversalRelationsScoreFn constructor will load data from {processed_data_filename}")
            data = np.load(processed_data_filename)
            self.lambda_symmetric = data["lambda_symmetric"]
            self.lambda_asymmetric = data["lambda_asymmetric"]
            self.m1 = data["m1"]
            self.m2 = data["m2"]
            self.q = data["q"]
            
        # Apply a mask since a few EOS might be a bit off
        mask = (self.lambda_asymmetric < 5_000) * (self.lambda_symmetric < 5_000)
        self.m1 = self.m1[mask]
        self.m2 = self.m2[mask]
        self.q = self.q[mask]
        self.lambda_symmetric = self.lambda_symmetric[mask]
        self.lambda_asymmetric = self.lambda_asymmetric[mask]
            
        if VERBOSE:
            print("self.lambda_symmetric")
            print(self.lambda_symmetric)
            
            print("self.lambda_symmetric mean and std")
            print(np.mean(self.lambda_symmetric), np.std(self.lambda_symmetric))
            
            print("self.lambda_asymmetric")
            print(self.lambda_asymmetric)
            
            print(np.mean(self.lambda_asymmetric), np.std(self.lambda_asymmetric))
            
            # Make histogram for both of them
            ls = self.lambda_symmetric.flatten()
            plt.hist(ls, bins = 20, color = "blue", histtype = "step", linewidth = 4)
            plt.xlabel(r"$\Lambda_{\rm s}$")
            plt.ylabel("Density")
            plt.savefig("./figures/histogram_lambda_s.png", bbox_inches = "tight")
            plt.close()
            
            la = self.lambda_asymmetric.flatten()
            plt.hist(la, bins = 20, color = "blue", histtype = "step", linewidth = 4)
            plt.xlabel(r"$\Lambda_{\rm a}$")
            plt.ylabel("Density")
            plt.savefig("./figures/histogram_lambda_a.png", bbox_inches = "tight")
            plt.close()
            
            print("jnp.shape(self.complete_lambda1_array)")
            print(jnp.shape(self.complete_lambda1_array))
            
            print("jnp.shape(self.complete_lambda2_array)")
            print(jnp.shape(self.complete_lambda2_array))
            
            print("jnp.shape(self.m1)")
            print(jnp.shape(self.m1))
            
            print("jnp.shape(self.m2)")
            print(jnp.shape(self.m2))
            
            print("jnp.shape(self.q)")
            print(jnp.shape(self.q))
            
        
    def score_fn(self, params: dict, take_mean: bool = True):
        """
        Params: in this case, this is the dict containing the parameters required by Binary love
        """
        
        # Call binary love
        params.update(self.fixed_params)
        binary_love_result = binary_love(self.lambda_symmetric, self.q, params)
        
        errors = abs((self.lambda_asymmetric - binary_love_result) / self.lambda_asymmetric)
        if take_mean:
            error = jnp.mean(errors)
        else:
            error = errors
        
        return error
    
    def error_fn(self, 
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
                              binary_love_params: dict = BINARY_LOVE_COEFFS,
                              return_aux: bool = False,
                              take_mean: bool = True):
        # Get the masses
        maxval = jnp.max(masses_EOS)
        first_batch = jax.random.uniform(self.subkey, shape=(self.nb_mass_samples,), minval = self.m_minval, maxval = maxval)
        self.key, self.subkey = jax.random.split(self.key)
        second_batch = jax.random.uniform(self.subkey, shape=(self.nb_mass_samples,), minval = self.m_minval, maxval = maxval)
        
        # Get the masses and ensure that m1 > m2
        m1 = jnp.maximum(first_batch, second_batch)
        m2 = jnp.minimum(first_batch, second_batch)
        q = m2 / m1
        
        # Interpolate the EOS
        lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS)
        lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS)
        
        # Get the symmetric and antisymmetric Lambdas
        lambda_symmetric  = 0.5 * (lambda_1 + lambda_2)
        lambda_asymmetric = 0.5 * (lambda_2 - lambda_1)
        
        # Call binary love
        binary_love_result = binary_love(lambda_symmetric, q, binary_love_params)
        
        # Evaluate the error
        errors = (lambda_asymmetric - binary_love_result) / lambda_asymmetric
        if take_mean:
            error = jnp.mean(abs(errors))
        else:
            error = abs(errors)
        
        if return_aux:
            return error, q, lambda_symmetric
        else:
            return error
        
    def assess_binary_love_improvement(self,
                                       params_list: list[dict],
                                       names_list: list[str] = ["Default", "Recalibrated"],
                                       colors_list: list[str] = ["blue", "green"],
                                       error_threshold: float = jnp.inf,
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
        
        errors_dict = {}
        
        # Iterate over the given list of binary Love params and compute the error and append to the dict
        for params, name in zip(params_list, names_list):
            errors = self.score_fn(params, take_mean = False)
                
            print(f"Dropping all with error above {error_threshold}")
            errors = np.array(errors)
            errors = errors.flatten()
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
                print("\n")
                print(f"=== {name} ===")
                print(f"Mean error: {np.mean(samples)}")
                print(f"Median error: {np.median(samples)}")
                cdf_at_10 = np.interp(0.10, x_values, cdf_values)
                print(f"CDF at 0.10 percent: {cdf_at_10}")
                quantile_90 = np.percentile(samples, 90)
                print(f"Quantile 90 percent: {quantile_90}")
                
                plt.axvline(quantile_90, color = color, linestyle = "--", linewidth = 2)
                print("\n")
                
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
        if i == 0:
                continue
        
        # Load from random_samples
        data = np.load(f"../benchmarks/random_samples/{i}.npz")
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
        for i, file in enumerate(tqdm.tqdm(os.listdir("../benchmarks/random_samples/"))):
            if i == 0:
                continue
            
            data = np.load(f"../benchmarks/random_samples/{file}")
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
        
def make_godzieba_plot(binary_love_params: dict = BINARY_LOVE_COEFFS,
                       nb_mass_samples: int = 1_000,
                       max_nb_eos: int = 100_000,
                       eos_dir: str = "../benchmarks/random_samples/",
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
        
        if i == 0:
            continue
        
        # Generate the masses for the plot
        first_batch = jax.random.uniform(subkey, shape=(nb_mass_samples,), minval = m_minval, maxval = m_maxval)
        key, subkey = jax.random.split(key)
        second_batch = jax.random.uniform(subkey, shape=(nb_mass_samples,), minval = m_minval, maxval = m_maxval)
        
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
    
    
############
### MAIN ### 
############

def run(score_fn_object: UniversalRelationsScoreFn,
        params: dict,
        nb_steps: int = 200):
    
    print("Starting parameters:")
    print(params)
    
    print("Computing by gradient ascent . . .")
    pbar = tqdm.tqdm(range(nb_steps))
    
    # Combining gradient transforms using `optax.chain`.
    print(f"The learning rate is set to {score_fn_object.learning_rate}")
    gradient_transform = optax.adam(learning_rate=score_fn_object.learning_rate)
    opt_state = gradient_transform.init(params)
    
    # Note: this does not return aux, contrary to doppelgangers run
    score_fn = jax.value_and_grad(score_fn_object.score_fn)
    
    for i in pbar:
        score, grad = score_fn(params)
        pbar.set_description(f"Iteration {i}: Score {score}")
        
        # Save: 
        np.savez(f"./outdir/{i}.npz", score = score, **params)
        
        # Do the updates
        updates, opt_state = gradient_transform.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        
    print("Computing DONE")
    
    return params
    
    
def compute_cdf(samples):
    
    # If 2D array, change to 1D:
    if len(samples.shape) > 1:
        samples = samples.flatten()
    
    # Sort the samples
    sorted_samples = np.sort(samples)
    
    # Compute the CDF values
    cdf_values = np.arange(1, len(samples) + 1) / len(samples)
    
    return sorted_samples, cdf_values
    
    
    
def do_optimization(score_fn_object: UniversalRelationsScoreFn,
                    start_params: dict = BINARY_LOVE_COEFFS,
                    save_name_final_params: str = "./new_binary_love_params.npz",
                    keep_fixed: list[str] = [],
                    nb_steps: int = 1_000):
    
    start_params_copied = copy.deepcopy(start_params)
      
    # FIXME: this needs to be restored if desired  
    # # Remove the keep_fixed from the params to iterate on
    # fixed_params = {key: start_params[key] for key in keep_fixed}
    # for key in keep_fixed:
    #     start_params.pop(key)
    
    # TODO: add keep_fixed
    final_params = run(score_fn_object = score_fn_object,
                       params = start_params,
                       nb_steps = nb_steps)
    
    # Add the fixed parameters back again
    for key in keep_fixed:
        final_params[key] = start_params_copied[key]
    
    print("Final parameters:")
    print(final_params)
    
    # Save the final parameters
    np.savez(save_name_final_params, **final_params)
    
    print("DONE")
    
    
def load_binary_love_params(file: str = "./new_binary_love_params.npz") -> dict:
    """
    Load the binary love parameters from the file
    """
    params = np.load(file)
    params = {key: value for key, value in params.items()}
    return params

def check_score_evolution(outdir: str = "./outdir/",
                          save_name: str = "./figures/score_evolution.png",
                          plot_param_evolution: bool = False):
    
    print("Checking the score evolution . . .")
    files = np.array(os.listdir(outdir))
    idx = np.argsort([int(file.split(".")[0]) for file in files])
    files = files[idx]
    
    params_evolution: dict[str, list] = {}
    
    scores = []
    for file in files:
        full_path = os.path.join(outdir, file)
        data = np.load(full_path)
        score = data["score"]
        for key in data.keys():
            if key != "score":
                params_evolution[key] = params_evolution.get(key, [])
                params_evolution[key].append(float(data[key]))
        scores.append(score)
        
    plt.figure(figsize = (14, 10))
    plt.plot(scores, "-o", linewidth = 4)
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    print("Checking the score evolution . . .")
    plt.savefig(save_name, bbox_inches = "tight")
    plt.close()
    
    # For each parameter
    if plot_param_evolution:
        param_evolution_plotdir = "./figures/param_evolution/"
        if not os.path.exists(param_evolution_plotdir):
            print(f"Creating directory {param_evolution_plotdir}")
            os.makedirs(param_evolution_plotdir)
        
        for key, values in params_evolution.items():
            print(f"Making plot for {key}")
            plt.figure(figsize = (14, 10))
            plt.plot(values, "-o", linewidth = 4)
            plt.xlabel("Iteration")
            plt.ylabel(key)
            save_name = os.path.join(param_evolution_plotdir, f"{key}_evolution.png")
            plt.savefig(save_name, bbox_inches = "tight")
            plt.close()
            
    # Print the first and last score number:
    print(f"First score: {scores[0]}")
    print(f"Last score: {scores[-1]}")
    
def my_format(number: float,
              nb_round: int = 2) -> str:
    
    number = np.round(number, nb_round)
    text = str(number)
    return text
    
    
    
def print_table_coeffs(params: dict):
    
    keys = BINARY_LOVE_COEFFS.keys()
    for k in keys:
        if k == "n_polytropic":
            nb_round = 3
        else:
            nb_round = 2
            
        params[k] = my_format(params[k], nb_round=nb_round)
    
    text = "This work & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ \\\\"
    text = text.format(*(params[k] for k in keys))
    
    print("\n\n\n")
    print(text)
    print("\n\n\n")
    
def make_scatterplot_residuals(q: np.array, 
                               lambda_s: np.array, 
                               errors: np.array,
                               downsample_factor = 1_000):
    """Make a 2D scatterplot, with q and lambda_s the x and y axis, and the color the error"""
    
    # Flatten from 2D arrays to 1D
    q = q.flatten()
    lambda_s = lambda_s.flatten()
    errors = errors.flatten()
    
    # Downsample if so desired
    q = q[::downsample_factor]
    lambda_s = lambda_s[::downsample_factor]
    errors = errors[::downsample_factor]
    
    plt.figure(figsize = (14, 10))
    plt.scatter(q, lambda_s, c = errors, cmap = "coolwarm", s = 10, rasterized = True)
    plt.colorbar()
    plt.xlabel(r"$q$")
    plt.ylabel(r"$\Lambda_{\rm s}$")
    plt.yscale("log")
    plt.savefig("./figures/scatterplot_residuals.png", bbox_inches = "tight")
    plt.savefig("./figures/scatterplot_residuals.pdf", bbox_inches = "tight")
    plt.close()
    
def main():
    
    """Optimizing the universal relations fit itself"""
    
    random_samples_outdir = "../benchmarks/random_samples/"
    score_fn_object = UniversalRelationsScoreFn(random_samples_outdir=random_samples_outdir,
                                                max_nb_eos=100_000,
                                                nb_mass_samples = 100,
                                                learning_rate = 1e-1)
    
    # ### Make single Godzieba plot
    # make_godzieba_plot()
    
    ### Do optimization
    do_optimization(score_fn_object = score_fn_object,
                    start_params = BINARY_LOVE_COEFFS,
                    save_name_final_params = "./new_binary_love_params.npz",
                    nb_steps = 1_000)
    check_score_evolution(plot_param_evolution=False)
    
    ### Final assessment of improved binary Love relation -- how much did we improve?
    params = load_binary_love_params()
    
    ### Check how much we improved
    params_list = [BINARY_LOVE_COEFFS, params]
    score_fn_object.assess_binary_love_improvement(params_list)
    
    # ### For paper writing
    # print_table_coeffs(params)
   
    # ### Now, make a scatterplot of the residuals
    # make_scatterplot_residuals(q, lambda_s, errors)
    
if __name__ == "__main__":
    main()