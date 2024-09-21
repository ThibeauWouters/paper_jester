"""
Playground for testing the possibilities of EOS exploration with jose.
"""

################
### PREAMBLE ###
################

import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"
import shutil

import os
import tqdm
import time
import corner
import numpy as np
import pandas as pd
np.random.seed(42) # for reproducibility
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Union, Callable

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

# from jax.scipy.stats import gaussian_kde

def compute_gradient_descent(N: int,
                             prior: CombinePrior,
                             score_fn: Callable,
                             optimization_sign: float = -1, 
                             learning_rate: float = 1e-3, 
                             start_halfway: bool = True,
                             random_seed: int = 42,
                             clean_outdir: bool = False):
    """
    Compute the gradient ascent or descent (just call it descent here for simplicity) in order to find the doppelgangers in the EOS space.

    Args:
        N (int): Number of steps to perform
        prior (CombinePrior): The prior from which to sample.
        likelihood (utils.NICERLikelihood): TODO: unused, remove?
        score_fn (Callable): Score fn. 
        optimization_sign (float, optional): Either +1 or -1, deciding the sign put in front of the gradient for parameters update. Defaults to -1.
        learning_rate (float, optional): The learning rate to be used in the gradient descent. Defaults to 1e-3.
        start_halfway (bool, optional): Whether to use the midpoint of the prior as the starting point. Defaults to True.
        clean_outdir (bool, optional): Whether to clean the output directory before recomputing. Defaults to False.
    """
    
    # Get initial parameters
    if start_halfway:
        params = {}
        # All are uniform priors so this works for now, but be careful, might break later on
        for i, key in enumerate(prior.parameter_names):
            base_prior = prior.base_prior[i]
            lower, upper = base_prior.xmin, base_prior.xmax
            params[key] = 0.5 * (lower + upper)

    else:
        jax_key = jax.random.PRNGKey(random_seed)
        jax_key, jax_subkey = jax.random.split(jax_key)
        params = prior.sample(jax_subkey, 1)
        
        for key, value in params.items():
            if isinstance(value, jnp.ndarray):
                params[key] = value.at[0].get()
        
    print("Starting parameters:")
    print(params)
    
    # Define the score function in the desired jax format
    score_fn = jax.value_and_grad(score_fn, has_aux=True)
    # score_fn = jax.jit(score_fn)
    
    failed_counter = 0
    
    if clean_outdir:
        print("Cleaning the outdir ./computed_data/ . . .")
        shutil.rmtree("./computed_data/", ignore_errors=True)
        os.makedirs("./computed_data/")
    print("Computing by gradient ascent . . .")
    
    # Perform the gradient descent
    for i in tqdm.tqdm(range(N)):
        
        ((score, aux), grad) = score_fn(params)
        m, r, l = aux
        
        # print("grad")
        # print(grad)
        
        if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)):
            print(f"Iteration {i} has NaNs")
            
            failed_counter += 1
            print(f"Skipping")
            continue
        
        print(f"Iteration {i}: score = {score}")
        np.savez(f"./computed_data/{i}.npz", masses_EOS = m, radii_EOS = r, Lambdas_EOS = l, score = score, **params)
        
        params = {key: value + optimization_sign * learning_rate * grad[key] for key, value in params.items()}
        
    print("Computing DONE")
    print(f"Failed percentage: {np.round(100 * failed_counter/N, 2)}")
    return None

#################
### SCORE FNs ###
#################

def doppelganger_score(params: dict,
                       transform: utils.MicroToMacroTransform,
                       m_target: Array,
                       Lambdas_target: Array, 
                       r_target: Array,
                       m_min = 0.5, 
                       m_max = 2.1,
                       alpha: float = 1.0) -> float:
    
    # Solve the TOV equations
    out = transform.forward(params)
    m_model, r_model, Lambdas_model = out["masses_EOS"], out["radii_EOS"], out["Lambdas_EOS"]
    
    mtov_model = m_model[-1]
    mtov_target = m_target[-1]
    
    # Get a mass array and interpolate NaNs on top of it TODO: make argument
    masses = jnp.linspace(m_min, m_max, 100)
    my_Lambdas_model = jnp.interp(masses, m_model, Lambdas_model, left = 0, right = 0)
    my_Lambdas_target = jnp.interp(masses, m_target, Lambdas_target, left = 0, right = 0)
    
    my_r_model = jnp.interp(masses, m_model, r_model, left = 0, right = 0)
    my_r_target = jnp.interp(masses, m_target, r_target, left = 0, right = 0)
    
    # Define separate scores
    score_lambdas = jnp.mean(((my_Lambdas_target - my_Lambdas_model) / my_Lambdas_target)**2)
    score_r = - jnp.mean(((my_r_target - my_r_model) / my_r_target)**2)
    score_mtov = ((mtov_target - mtov_model) / mtov_target)**2
    
    score = score_lambdas + alpha * score_mtov
    
    return score, (m_model, r_model, Lambdas_model)

################
### PLOTTING ###
################

def plot_NS(N: int,
            scatter = False,
            m_target: Array = None,
            Lambdas_target: Array = None,
            r_target: Array = None,
            plot_mse: bool = False):
    
    # Read the EOS data
    all_masses_EOS = []
    all_radii_EOS = []
    all_Lambdas_EOS = []

    for i in range(N):
        try:
            data = np.load(f"./computed_data/{i}.npz")
            
            masses_EOS = data["masses_EOS"]
            radii_EOS = data["radii_EOS"]
            Lambdas_EOS = data["Lambdas_EOS"]
            
            all_masses_EOS.append(masses_EOS)
            all_radii_EOS.append(radii_EOS)
            all_Lambdas_EOS.append(Lambdas_EOS)
            
        except FileNotFoundError:
            print(f"File {i} not found")
            continue
        
    # N might have become smaller if we hit NaNs at some point
    N_max = len(all_masses_EOS)
    norm = mpl.colors.Normalize(vmin=0, vmax=N_max)
    # cmap = sns.color_palette("rocket_r", as_cmap=True)
    cmap = mpl.cm.viridis
        
    # Plot the target
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 6))
    plt.subplot(121)
    plt.plot(r_target, m_target, color = "red", zorder = 1e10)
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M \ [M_\odot]$")
    plt.subplot(122)
    plt.xlabel(r"$M \ [M_\odot]$")
    plt.ylabel(r"$\Lambda$")
    plt.plot(m_target, Lambdas_target, label=r"$\Lambda$", color = "red", zorder = 1e10)
    plt.yscale("log")
        
    for i in range(N_max):
        color = cmap(norm(i))
        
        # Mass-radius plot
        plt.subplot(121)
        plt.plot(all_radii_EOS[i], all_masses_EOS[i], color=color, linewidth = 2.0, zorder=i)
        if scatter:
            plt.scatter(all_radii_EOS[i], all_masses_EOS[i], color=color, s=5, zorder=i)
            
        # Mass-Lambdas plot
        plt.subplot(122)
        plt.plot(all_masses_EOS[i], all_Lambdas_EOS[i], color=color, linewidth = 2.0, zorder=i)
        if scatter:
            plt.scatter(all_masses_EOS[i], all_Lambdas_EOS[i], color=color, s=5, zorder=i)
        
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs[-1])
    cbar.set_label(r'Iteration number', fontsize = 22)
        
    plt.tight_layout()
    save_name = f"./figures/doppelganger_trajectory.png" 
    print(f"Saving to: {save_name}")
    plt.savefig(save_name, bbox_inches = "tight")
    
    # Also plot the progress of the errors
    if plot_mse:
        mse_errors = []
        for i in range(N_max):
            data = np.load(f"./computed_data/{i}.npz")
            mse_errors.append(data["score"])
            
        # Plot
        plt.figure(figsize=(6, 6))
        nb = [i+1 for i in range(len(mse_errors))]
        plt.plot(nb, mse_errors, color="black")
        plt.scatter(nb, mse_errors, color="black")
        plt.xlabel("Iteration number")
        plt.ylabel("MSE")
        
        plt.savefig("./figures/mse_errors.png", bbox_inches="tight")
        plt.close()


def main():
    
    ### PRIOR
    my_nbreak = 2.0 * 0.16
    NMAX_NSAT = 25
    NMAX = NMAX_NSAT * 0.16
    NB_CSE = 8
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

    # CSE priors
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
        
    ### Load the target
    target_filename = "./36022_macroscopic.dat"
    target_eos = np.genfromtxt(target_filename, skip_header=1, delimiter=" ").T
    r_target, m_target, Lambdas_target = target_eos[0], target_eos[1], target_eos[2]
        
    # Use it to get a doppelganger score
    
    transform = utils.MicroToMacroTransform(utils.name_mapping, 
                                        keep_names = ["E_sym", "L_sym"],
                                        nmax_nsat = utils.NMAX_NSAT,
                                        nb_CSE = utils.NB_CSE,
                                        )
    
    doppelganger_score_ = lambda params: doppelganger_score(params, transform, m_target, Lambdas_target, r_target)
        
    N = 100
    compute_gradient_descent(N, prior, doppelganger_score_, learning_rate = 0.001, start_halfway = False, random_seed = 43)
    
    # TODO: remove me?
    # # Plot the target
    # plt.subplots(1, 2, figsize=(12, 6))
    # plt.subplot(121)
    # plt.plot(r_target, m_target, color = "red", zorder = 1e10)
    # plt.xlabel(r"$R$ [km]")
    # plt.ylabel(r"$M \ [M_\odot]$")
    # plt.subplot(122)
    # plt.xlabel(r"$M \ [M_\odot]$")
    # plt.ylabel(r"$\Lambda$")
    # plt.plot(m_target, Lambdas_target, label=r"$\Lambda$", color = "red", zorder = 1e10)
    # plt.savefig("./figures/target_EOS.png", bbox_inches="tight")
    # plt.close()
    
    # Plot the doppelganger trajectory
    plot_NS(N, m_target = m_target, Lambdas_target = Lambdas_target, r_target = r_target)
    
if __name__ == "__main__":
    main()