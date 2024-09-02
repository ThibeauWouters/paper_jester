"""
Playground for NICER inference: we sample some individual parameters, then solve the TOV equations and compute the NICER log likelihood. The results are saved and plotted to visually sanity-check the results. Meant for low number of samples and playing around.
"""

################
### PREAMBLE ###
################

import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
import shutil

import os
import tqdm
import time
import corner
import numpy as np
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

import utils
plt.rcParams.update(utils.mpl_params)

start = time.time()

def compute_gradient_ascent(N, 
                            prior: CombinePrior,
                            likelihood: utils.NICERLikelihood,
                            learning_rate = 1e-3,
                            start_halfway: bool = True,
                            loss_function: str = "MTOV"):
    
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
    
    
    if loss_function == "MTOV":
        def nep_to_MTOV(params):
            # Convert the NEP parameters to MTOV parameters
            
            macro_params = likelihood.transform.forward(params)
            m, r, l = macro_params["masses_EOS"], macro_params["radii_EOS"], macro_params["Lambdas_EOS"]
            mtov = jnp.max(m)
            return mtov, (m, r, l)
        
        nep_to_MTOV = jax.value_and_grad(nep_to_MTOV, has_aux=True)
        nep_to_MTOV = jax.jit(nep_to_MTOV)
        
        loss = nep_to_MTOV
        
    elif loss_function == "MTOV":
        def eos_mse(params):
            transformed_params = likelihood.transform.forward(params)
            n, p, e = likelihood.compute_eos(transformed_params)
            cs2 = jnp.gradient(p, e)
            
            return mtov, (m, r, l)
        
        nep_to_MTOV = jax.value_and_grad(nep_to_MTOV, has_aux=True)
        nep_to_MTOV = jax.jit(nep_to_MTOV)
        
        loss = nep_to_MTOV
        
    else:
        raise ValueError(f"Loss function {loss_function} not recognized")
    
    failed_counter = 0
    shutil.rmtree("./computed_data/", ignore_errors=True)
    os.makedirs("./computed_data/")
    print("Computing by gradient ascent . . .")
    
    for i in tqdm.tqdm(range(N)):
        
        ((mtov, aux), grad) = nep_to_MTOV(params)
        m, r, l = aux
        
        if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)):
            print(f"Iteration {i} has NaNs")
            
            failed_counter += 1
            print(f"Skipping")
            continue
        
        print(f"Iteration {i}: MTOV = {mtov}")
        # Save results
        np.savez(f"./computed_data/{i}.npz", masses_EOS = m, radii_EOS = r, L=0.0, **params)
        
        params = {key: value + learning_rate * grad[key] for key, value in params.items()}
        
    print("Computing DONE")
    print(f"Failed percentage: {np.round(100 * failed_counter/N, 2)}")
    return None
    
        
def plot_eos(N: int,
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
    
    N = 200
    
    NMAX_NSAT = 25
    NMAX = NMAX_NSAT * 0.16
    NB_CSE = 8
    my_nbreak = 2.0 * 0.16
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
        
        # n_CSE
        prior_list.append(UniformPrior(left, right, parameter_names=[f"n_CSE_{i}"]))
        
        # cs2_CSE
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))
    
    # Final point to end
    prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))
    
    prior = CombinePrior(prior_list)
    sampled_param_names = prior.parameter_names
    
    name_mapping = (sampled_param_names, ["masses_EOS", "radii_EOS", "Lambdas_EOS"])
    transform = utils.MicroToMacroTransform(name_mapping, 
                                            keep_names = ["E_sym", "L_sym"],
                                            nmax_nsat=NMAX_NSAT,
                                            nb_CSE = NB_CSE,
                                            )
    
    # Likelihood: choose which PSRs to perform inference on:
    psr_names = []
    likelihoods_list_NICER = [utils.NICERLikelihood(psr) for psr in psr_names]

    # REX_names = ["PREX"]
    REX_names = []
    likelihoods_list_REX = [utils.REXLikelihood(rex) for rex in REX_names]

    likelihoods_list = likelihoods_list_NICER + likelihoods_list_REX
    
    if len(likelihoods_list) == 0:
        # For testing stuff
        likelihood = utils.ZeroLikelihood(transform = transform)
    else:
        likelihood = utils.CombinedLikelihood(likelihoods_list,
                                              transform = transform)

    name_mapping = (sampled_param_names, ["masses_EOS", "radii_EOS", "Lambdas_EOS"])
    compute_gradient_ascent(N, prior=prior, likelihood=likelihood, learning_rate = 0.001)
    
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