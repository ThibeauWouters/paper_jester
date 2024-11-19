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

# We will import and build on top of DoppelgangerRun

from paper_jose.doppelgangers.doppelgangers import DoppelgangerRun


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


################
### SCORE_FN ### 
################


class UniversalRelationsScoreFn:
    """This is a class that stores stuff that might simplify the calculation of the error on the universal relations"""
    
    def __init__(self,
                 max_nb_eos: int = 100_000,
                 nb_mass_samples: int = 1_000,
                 random_samples_outdir: str = "../doppelgangers/random_samples/",
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
        
        # Create the mass array on which to interpolate the EOSs
        self.mass_array = jnp.linspace(self.m_minval, self.m_maxval, self.m_length)
        
        # Set the masses array to be used
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
        self.complete_lambda1_array = complete_lambda1_array
        self.complete_lambda2_array = complete_lambda2_array
        
        # Get the "true" Lambda_a and Lambda_s:
        self.lambda_symmetric = 0.5 * (self.complete_lambda1_array + self.complete_lambda2_array)
        self.lambda_asymmetric = 0.5 * (self.complete_lambda2_array - self.complete_lambda1_array)
        
    def score_fn(self, params: dict):
        """
        Params: in this case, this is the dict containing the parameters required by Binary love
        """
        
        # Call binary love
        binary_love_result = binary_love(self.lambda_symmetric, self.q, params)
        
        # Get the error
        errors = (self.lambda_asymmetric - binary_love_result) / self.lambda_asymmetric
        error = jnp.mean(errors)
        
        return error
        
        
################
### PLOTTING ###
################

def plot_binary_Love(q_values: list[float] = [0.5, 0.75, 0.90, 0.99],
                     nb_samples: int = 50,
                     nb_eos: int = 100,
                     plot_binary_love: bool = True):
    
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
            lambda_asymmetric_values = binary_love(lambda_symmetric_values, q, BINARY_LOVE_COEFFS)
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
    plt.savefig("./figures/test_binary_love.png", bbox_inches = "tight")
    plt.savefig("./figures/test_binary_love.pdf", bbox_inches = "tight")

    plt.close()
    
def get_histograms(q_values = [0.5, 0.75, 0.90, 0.99],
                   nb_samples = 100,):
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
            binary_love_values = binary_love(lambda_symmetric_sampled, q, BINARY_LOVE_COEFFS)
            
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
        plt.savefig(f"./figures/histogram_q_{q}.png", bbox_inches = "tight")
        plt.savefig(f"./figures/histogram_q_{q}.pdf", bbox_inches = "tight")
        plt.close()
        
def make_godzieba_plot(nb_samples: int = 1_000,
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
            
            # # Limit lambdas
            # mask = (lambda2_sampled < 10_000)
            # lambda1_sampled = lambda1_sampled[mask]
            # lambda2_sampled = lambda2_sampled[mask]
            # q = q[mask]
            
            # Get the symmetric and antisymmetric Lambdas
            lambda_symmetric_sampled  = 0.5 * (lambda2_sampled + lambda1_sampled)
            lambda_asymmetric_sampled = 0.5 * (lambda2_sampled - lambda1_sampled)
            
            # Mask lambda_symmetric_sampled below 4000
            mask = (lambda_symmetric_sampled < 4000)
            lambda_symmetric_sampled = lambda_symmetric_sampled[mask]
            lambda_asymmetric_sampled = lambda_asymmetric_sampled[mask]
            q = q[mask]
        
            # Get binary Love
            binary_love_values = binary_love(lambda_symmetric_sampled, m2_sampled / m1_sampled, BINARY_LOVE_COEFFS)
            
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
    # plt.ylim(-300, 300)
    
    plt.subplot(212)
    plt.xlabel(r"$q$")
    plt.ylabel(r"$\Lambda_{\rm a} - \Lambda_{\rm a}^{\rm fit}$")
    # plt.ylim(-300, 300)
    plt.savefig("./figures/godzieba_plot.png", bbox_inches = "tight")
    plt.close()
    
############
### MAIN ### 
############

def get_learning_rate(i, start_lr, total_epochs):
    # if i < total_epochs // 2:
    #     return start_lr
    # else:
    #     return start_lr / 10.0
    return start_lr

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
        
        # Do the updates
        learning_rate = get_learning_rate(i, learning_rate, nb_steps)
        params = {key: value + optimization_sign * learning_rate * grad[key] for key, value in params.items()}
        
    print("Computing DONE")
    print("Final parameters:")
    print(params)
    

def main(fixed_CSE: bool = False, # use a CSE, but have it fixed, vary only the metamodel
         metamodel_only = False, # only use the metamodel, no CSE used at all
         ):
    
    ### SETUP
    
    # Prior
    my_nbreak = 2.0 * 0.16
    if metamodel_only:
        NMAX_NSAT = 5
        NB_CSE = 0
    else:
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

    # Vary the CSE (i.e. include in the prior if used, and not set to fixed)
    if not metamodel_only and not fixed_CSE:
        # CSE priors
        prior_list.append(UniformPrior(1.0 * 0.16, 2.0 * 0.16, parameter_names=[f"nbreak"]))
        for i in range(NB_CSE):
            left = my_nbreak + i * width
            right = my_nbreak + (i+1) * width
            prior_list.append(UniformPrior(left, right, parameter_names=[f"n_CSE_{i}"]))
            prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))
    
    # Combine the prior
    prior = CombinePrior(prior_list)
    
    
    
    # # Get a doppelganger score
    # sampled_param_names = prior.parameter_names
    # name_mapping = (sampled_param_names, ["masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])
    # transform = utils.MicroToMacroTransform(name_mapping, nmax_nsat=NMAX_NSAT, nb_CSE=NB_CSE)
    
    # ### Optimizer run
    # np.random.seed(700)
    # for i in range(N_runs):
    #     seed = np.random.randint(0, 100_000)
    #     print(f" ====================== Run {i + 1} / {N_runs} with seed {seed} ======================")
    #     runner = DoppelgangerRun(prior, transform, "macro", seed, nb_steps = 200, learning_rate = learning_rate)
    #     params = runner.initialize_walkers()
        
    ### Call some plotting scripts
    # plot_binary_Love()
    # get_histograms()
    # make_godzieba_plot(nb_samples = 1_000)
    
    print("DONE")
    
if __name__ == "__main__":
    main()