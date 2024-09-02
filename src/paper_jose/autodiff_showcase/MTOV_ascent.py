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
from typing import Union

import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jimgw.prior import UniformPrior, CombinePrior

from jax.scipy.stats import gaussian_kde

import joseTOV.utils as jose_utils

import paper_jose.inference.utils_plotting as utils_plotting
import paper_jose.utils as utils
plt.rcParams.update(utils_plotting.mpl_params)

start = time.time()

def compute_MTOV_gradient_ascent(N, 
                                 prior: CombinePrior,
                                 likelihood: utils.NICERLikelihood,
                                 learning_rate = 1e-3,
                                 start_halfway: bool = True):
    
    # Get some initial parameters
    if start_halfway:
        params = {}
        # All are uniform priors so this works for now
        for i, key in enumerate(prior.parameter_names):
            base_prior = prior.base_prior[i]
            lower, upper = base_prior.xmin, base_prior.xmax
            params[key] = 0.5 * (lower + upper)

    else:
        jax_key = jax.random.PRNGKey(40)
        jax_key, jax_subkey = jax.random.split(jax_key)
        params = prior.sample(jax_subkey, 1)
        
        for key, value in params.items():
            if isinstance(value, jnp.ndarray):
                params[key] = value.at[0].get()
        
    print("Starting parameters:")
    print(params)
    

    def nep_to_MTOV(params):
        # Convert the NEP parameters to MTOV parameters
        
        macro_params = likelihood.transform.forward(params)
        m, r, l = macro_params["masses_EOS"], macro_params["radii_EOS"], macro_params["Lambdas_EOS"]
        mtov = jnp.max(m)
        return mtov, (m, r, l)
    
    nep_to_MTOV = jax.value_and_grad(nep_to_MTOV, has_aux=True)
    nep_to_MTOV = jax.jit(nep_to_MTOV)
    
    loss = nep_to_MTOV
    
    failed_counter = 0
    shutil.rmtree("./computed_data/", ignore_errors=True)
    os.makedirs("./computed_data/")
    print("Computing by gradient ascent . . .")
    
    for i in tqdm.tqdm(range(N)):
        
        ((loss_value, aux), grad) = loss(params)
        m, r, l = aux
        
        if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)):
            print(f"Iteration {i} has NaNs")
            
            failed_counter += 1
            print(f"Skipping")
            continue
        
        print(f"Iteration {i}: Loss = {loss_value}")
        # Save results
        np.savez(f"./computed_data/{i}.npz", masses_EOS = m, radii_EOS = r, L=0.0, **params)
        
        params = {key: value + learning_rate * grad[key] for key, value in params.items()}
        
    print("Computing DONE")
    print(f"Failed percentage: {np.round(100 * failed_counter/N, 2)}")
    return None

def compute_EOS_gradient_descent(N, 
                                 prior: CombinePrior,
                                 likelihood: utils.NICERLikelihood,
                                 learning_rate = 1e-3,
                                 start_halfway: bool = True,
                                 save_step: int = None):
    
    if save_step is None:
        save_step = N // 100
    
    # Get some initial parameters
    if start_halfway:
        params = {}
        # All are uniform priors so this works for now
        for i, key in enumerate(prior.parameter_names):
            base_prior = prior.base_prior[i]
            lower, upper = base_prior.xmin, base_prior.xmax
            params[key] = 0.5 * (lower + upper)

    else:
        jax_key = jax.random.PRNGKey(40)
        jax_key, jax_subkey = jax.random.split(jax_key)
        params = prior.sample(jax_subkey, 1)
        
        for key, value in params.items():
            if isinstance(value, jnp.ndarray):
                params[key] = value.at[0].get()
        
    print("Starting parameters:")
    print(params)
    
    # Load the true EOS
    df = pd.read_csv("./36022_microscopic.dat", header = None, names = ["n", "e", "p", "cs2"], skiprows = 1, delimiter = " ")
    n_true, p_true, e_true, cs2_true = df["n"].values, df["p"].values, df["e"].values, df["cs2"].values
    
    def eos_mse(params):
        transformed_params = likelihood.transform.forward(params)
        n, p, e, cs2 = transformed_params["n"], transformed_params["p"], transformed_params["e"], transformed_params["cs2"]
        
        n = n / jose_utils.fm_inv3_to_geometric
        p = p / jose_utils.MeV_fm_inv3_to_geometric
        e = e / jose_utils.MeV_fm_inv3_to_geometric
        
        # Interpolate the true EOS
        p_true_interp = jnp.interp(n, n_true, p_true)
        e_true_interp = jnp.interp(n, n_true, e_true)
        cs2_true_interp = jnp.interp(n, n_true, cs2_true)
        
        # mse_p = jnp.mean((p - p_true_interp)**2)
        # mse_e = jnp.mean((e - e_true_interp)**2)
        mse_cs2 = jnp.mean((cs2 - cs2_true_interp)**2)
        
        mse = mse_cs2
        
        return mse, (n, p, e, cs2)
    
    eos_mse = jax.value_and_grad(eos_mse, has_aux=True)
    eos_mse = jax.jit(eos_mse)
    
    loss = eos_mse
    
    failed_counter = 0
    shutil.rmtree("./computed_data/", ignore_errors=True)
    os.makedirs("./computed_data/")
    print("Computing by gradient ascent . . .")
    
    for i in tqdm.tqdm(range(N)):
        
        ((loss_value, aux), grad) = loss(params)
        n, p, e, cs2 = aux
        
        print(f"Iteration {i}: Loss = {loss_value}")
        # Save results
        if i % save_step == 0:
            np.savez(f"./computed_data/{i}.npz", n = n, p = p, e = e, cs2 = cs2, L=0.0, **params)
        
        params = {key: value - learning_rate * grad[key] for key, value in params.items()}
        
    np.savez(f"./computed_data/{i}.npz", n = n, p = p, e = e, cs2 = cs2, L=0.0, **params)
        
    print("Computing DONE")
    print(f"Failed percentage: {np.round(100 * failed_counter/N, 2)}")
    return None
    
        
def plot_NS(N: int,
            scatter = False):
    
    fig, ax = plt.subplots(figsize = (12, 6))

    # Read the EOS data
    all_masses_EOS = []
    all_radii_EOS = []

    for i in range(N):
        try:
            data = np.load(f"./computed_data/{i}.npz")
            
            masses_EOS = data["masses_EOS"]
            radii_EOS = data["radii_EOS"]
            
            all_masses_EOS.append(masses_EOS)
            all_radii_EOS.append(radii_EOS)
            
        except FileNotFoundError:
            print(f"File {i} not found")
            continue
        
    norm = mpl.colors.Normalize(vmin=0, vmax=N)
    # cmap = sns.color_palette("rocket_r", as_cmap=True)
    cmap = mpl.cm.viridis
        
    for i in range(N):
        # alpha = (i + 1) / N
        color = cmap(norm(i))
        plt.plot(all_radii_EOS[i], all_masses_EOS[i], color=color, linewidth = 2.0, zorder=1e10)
        if scatter:
            plt.scatter(all_radii_EOS[i], all_masses_EOS[i], color=color, s=5, zorder=1e10)
        
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'Iteration number', fontsize = 22)
        
    plt.xlim(5, 16)
    plt.ylim(0.75, 3.25)
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_{\odot}$]")

    plt.tight_layout()
    save_name = f"./figures/MTOV_ascent/MTOV_ascent.png" 
    print(f"Saving to: {save_name}")
    plt.savefig(save_name, bbox_inches = "tight")
    plt.close()

    print("DONE")
    end = time.time()
    print(f"Time taken: {end - start} s")
    
def plot_eos(N: int):
    
    # Plot the "true" EOS
    df = pd.read_csv("./36022_microscopic.dat", header = None, names = ["n", "e", "p", "cs2"], skiprows = 1, delimiter = " ")
    n_true, p_true, e_true, cs2_true = df["n"].values, df["p"].values, df["e"].values, df["cs2"].values
    
    plt.subplots(nrows = 2, ncols = 2, figsize = (12, 12))
    plt.subplot(221)
    plt.plot(n_true, p_true, color = "red")
    plt.xlabel("n")
    plt.ylabel("p")
    
    plt.subplot(222)
    plt.plot(n_true, e_true, color = "red")
    plt.xlabel("n")
    plt.ylabel("e")
    
    plt.subplot(223)
    plt.plot(n_true, cs2_true, color = "red")
    plt.xlabel("n")
    plt.ylabel("cs2")
    
    # Now plot the created EOS:
    kwargs = {"color": "black", "alpha": 0.1}
    for i in range(N):
        try:
            data = np.load(f"./computed_data/{i}.npz")
            
            n = data["n"]
            p = data["p"]
            e = data["e"]
            cs2 = data["cs2"]
            
            plt.subplot(221)
            plt.plot(n, p, **kwargs)
            
            plt.subplot(222)
            plt.plot(n, e, **kwargs)
            
            plt.subplot(223)
            plt.plot(n, cs2, **kwargs)
            
        except FileNotFoundError:
            print(f"File {i} not found")
            continue
    
    plt.savefig("./figures/EOS_descent.png", bbox_inches = "tight")
    plt.close()
    
    
def plot_trajectory(N: int,
                    plot_keys: list[str] = None):
    
    if plot_keys is None:
        plot_keys = list(utils.NEP_CONSTANTS_DICT.keys())
    
    plot_keys.remove("E_sat")

    # Read the EOS data
    all_masses_EOS = []
    all_radii_EOS = []
    all_params = []

    for i in range(N):
        try:
            data = np.load(f"./computed_data/{i}.npz")
            
            masses_EOS = data["masses_EOS"]
            radii_EOS = data["radii_EOS"]
            
            params = {key: data[key] for key in plot_keys}
            
            all_masses_EOS.append(masses_EOS)
            all_radii_EOS.append(radii_EOS)
            all_params.append(params)
            
        except FileNotFoundError:
            print(f"File {i} not found")
            continue
        
    # Get the TOV mass:
    all_masses_EOS = np.array(all_masses_EOS)
    all_MTOV = np.max(all_masses_EOS, axis=1)
    
    norm = mpl.colors.Normalize(vmin=0, vmax=N)
    cmap = mpl.cm.viridis
        
    colors = [cmap(norm(i)) for i in range(N)]
    for i, key in enumerate(plot_keys):
        fig = plt.subplots(figsize = (12, 12))
        param_data = np.array([params[key] for params in all_params])
        plt.plot(param_data, all_MTOV, color="red", linewidth = 2.0, zorder=1e10)
            
        # sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # cbar = fig.colorbar(sm, ax=ax)
        # cbar.set_label(r'Iteration number', fontsize = 22)
            
        plt.xlabel(f"{key}")
        plt.ylabel(r"$M_{\rm{TOV}}$ [$M_{\odot}$]")

        plt.tight_layout()
        save_name = f"./figures/MTOV_ascent/MTOV_ascent_param_{key}.png" 
        print(f"Saving to: {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        plt.close()

    print("DONE")
    end = time.time()
    print(f"Time taken: {end - start} s")
    
def main():
    
    N = 10
    
    prior = utils.prior
    transform = utils.MicroToMacroTransform(utils.name_mapping, 
                                            keep_names = ["E_sym", "L_sym"],
                                            nmax_nsat = utils.NMAX_NSAT,
                                            nb_CSE = utils.NB_CSE,
                                            )
    
    likelihood = utils.ZeroLikelihood(transform = transform)
    
    ### Choose which kind of gradient descent to perform
    
    ### MTOV
    # compute_MTOV_gradient_ascent(N, prior=prior, likelihood=likelihood, learning_rate = 0.001)
    # try:
    #     plot_NS(N)
    # except Exception as e:
    #     print(f"Error in plotting: {e}")
        
    ### EOS
    compute_EOS_gradient_descent(3_000,
                                 prior=prior,
                                 likelihood=likelihood,
                                 learning_rate = 0.1)
    try:
        plot_eos(N)
    except Exception as e:
        print(f"Error in plotting: {e}")
        
    try:
        plot_trajectory(N)
    except Exception as e:
        print(f"Error in plotting: {e}")
    
    print("DONE")
    
if __name__ == "__main__":
    main()