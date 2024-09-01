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
import utils_plotting
plt.rcParams.update(utils_plotting.mpl_params)

start = time.time()

def compute_without_vmap(N,
                         prior: CombinePrior,
                         likelihood: utils.NICERLikelihood):
    jax_key = jax.random.PRNGKey(40)
    L_array = []
    failed_counter = 0
    shutil.rmtree("./computed_data/", ignore_errors=True)
    os.makedirs("./computed_data/")
    print("Computing likelihood without vmap . . .")
    for i in tqdm.tqdm(range(N)):
        
        jax_key, jax_subkey = jax.random.split(jax_key)
        
        params = prior.sample(jax_subkey, 1)
        for key, value in params.items():
            if isinstance(value, jnp.ndarray):
                params[key] = value.at[0].get()
        
        macro_params = likelihood.transform.forward(params)
        m, r, l = macro_params["masses_EOS"], macro_params["radii_EOS"], macro_params["Lambdas_EOS"]
        
        if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)):
            print(f"Iteration {i} has NaNs. \nParams")
            print(params)
            failed_counter += 1
            print(f"Skipping")
            continue
        
        L = likelihood.evaluate(macro_params, None)
        L_array.append(L)
        
    print("Computing likelihood without vmap DONE")
    print(f"Failed percentage: {np.round(100 * failed_counter/N, 2)}")
    return L_array


def compute_with_vmap(N,
                      prior,
                      likelihood):
    # FIXME: this is broken after adding transforms, need to update it
    jax_key = jax.random.PRNGKey(42)
    jax_key, jax_subkey = jax.random.split(jax_key)
    params = prior.sample(jax_subkey, N)

    print("Computing likelihood with vmap")
    likelihood_evaluate_vmap = jax.vmap(likelihood.evaluate, in_axes=(0, None))
    my_time_start = time.time()
    L_array = likelihood_evaluate_vmap(params, None)
    my_time_end = time.time()
    print(f"Time taken: {my_time_end - my_time_start} s")
    print("Computing likelihood with vmap DONE")
    
    return L_array

def plot_corner_data(PSR_NAME: str):
    
    print(f"Plotting the PSR data for PSR: {PSR_NAME}")
    # Fetch the samples
    amsterdam_samples = utils.data_samples_dict[PSR_NAME]["amsterdam"]
    maryland_samples = utils.data_samples_dict[PSR_NAME]["maryland"]
    
    # Plot
    corner_kwargs = utils.default_corner_kwargs.copy()
    hist_kwargs = {"density": True}
    corner_kwargs["labels"] = ["$R$ [km]", "$M$ [$M_{\odot}$]"]
    corner_kwargs["fill_contours"] = True
    corner_kwargs["alpha"] = 1.0
    corner_kwargs["zorder"] = 1e9
    corner_kwargs["no_fill_contours"] = True
    corner_kwargs["plot_contours"] = True

    corner_kwargs["color"] = utils_plotting.AMSTERDAM_COLOR
    hist_kwargs["color"] = utils_plotting.AMSTERDAM_COLOR

    corner_kwargs["hist_kwargs"] = hist_kwargs
    if PSR_NAME == "J0030":
        corner_kwargs["range"] = [(8, 17), (0.8, 2.25)]
    else:
        corner_kwargs["range"] = [(10, 19), (1.6, 2.75)]
    
    fig = corner.corner(np.array(amsterdam_samples[["R", "M"]]), weights=np.array(amsterdam_samples["weight"]), **corner_kwargs)

    corner_kwargs["color"] = utils_plotting.MARYLAND_COLOR
    hist_kwargs["color"] = utils_plotting.MARYLAND_COLOR
    corner_kwargs["hist_kwargs"] = hist_kwargs
    fig = corner.corner(np.array(maryland_samples[["R", "M"]]), fig=fig, weights=np.array(maryland_samples["weight"]), **corner_kwargs)

    fs = 24
    plt.text(0.65, 0.8, "Maryland", color=utils_plotting.MARYLAND_COLOR, fontsize=fs, transform=plt.gcf().transFigure)
    plt.text(0.65, 0.7, "Amsterdam", color=utils_plotting.AMSTERDAM_COLOR, fontsize=fs, transform=plt.gcf().transFigure)
    plt.savefig(f"./figures/corner_data_{PSR_NAME}.png")
    plt.close() 

def plot_eos_contours(N: int,
                      psr_name: Union[str, list[str]],
                      REX_name: Union[str, list[str]],
                      scatter = False):
    
    fig, ax = plt.subplots(figsize = (12, 6))

    if isinstance(psr_name, str):
        psr_name = [psr_name]
    suffix = "_".join(psr_name)    
    
    if isinstance(REX_name, str):
        REX_name = [REX_name]
    suffix += "_".join(REX_name)    
    
    for psr in psr_name:
    
        maryland_samples = utils.data_samples_dict[psr]["maryland"]
        amsterdam_samples = utils.data_samples_dict[psr]["amsterdam"]

        maryland_data_2d = jnp.array([maryland_samples["M"].values, maryland_samples["R"].values])
        amsterdam_data_2d = jnp.array([amsterdam_samples["M"].values, amsterdam_samples["R"].values])

        # First the data TODO: improve the plotting, the contours are ugly but matplotlib is annoying...
        for dataset, cmap in zip([maryland_data_2d, amsterdam_data_2d], [utils.MARYLAND_CMAP, utils.AMSTERDAM_CMAP]):

            data = dataset.T
            hist, xedges, yedges = np.histogram2d(data[:, 1], data[:, 0], bins=50)
            xcenters = (xedges[:-1] + xedges[1:]) / 2
            ycenters = (yedges[:-1] + yedges[1:]) / 2
            X, Y = np.meshgrid(xcenters, ycenters)
            plt.contour(X, Y, hist.T, levels=10, cmap = cmap)
            
    for rex in REX_name:
        cmap = utils_plotting.REX_CMAP_DICT[rex]

        posterior: gaussian_kde = utils.kde_dict[rex]
        samples = posterior.resample(jax.random.PRNGKey(0), (1000,))

        hist, xedges, yedges = np.histogram2d(samples[:, 1], samples[:, 0], bins=50)
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(xcenters, ycenters)
        plt.contour(X, Y, hist.T, levels=10, cmap = cmap)

    # Read the EOS data
    all_masses_EOS = []
    all_radii_EOS = []
    all_L = []

    for i in range(N):
        try:
            data = np.load(f"./computed_data/{i}.npz")
            
            masses_EOS = data["masses_EOS"]
            radii_EOS = data["radii_EOS"]
            L = data["L"]
            
            all_masses_EOS.append(masses_EOS)
            all_radii_EOS.append(radii_EOS)
            all_L.append(L)
        except FileNotFoundError:
            print(f"File {i} not found")
            continue
        
    # DEBUG
    print("Before normalizing this is the log L range:")
    print(np.min(all_L), np.max(all_L))
        
    # Then plot all the EOS data
    all_L = np.array(all_L)
    if np.max(all_L) == np.min(all_L):
        no_cmap = True
        all_L = np.ones_like(all_L)
    else:
        no_cmap = False
        all_L = (all_L - np.min(all_L))/(np.max(all_L) - np.min(all_L))

    norm = mpl.colors.Normalize(vmin=np.min(all_L), vmax=np.max(all_L))
    # cmap = mpl.cm.Greens
    # cmap = sns.color_palette("rocket_r", as_cmap=True)
    cmap = mpl.cm.Reds

    for i in range(len(all_L)):
        if no_cmap:
            color = "red"
        else:
            color = cmap(norm(all_L[i]))
        plt.plot(all_radii_EOS[i], all_masses_EOS[i], color=color, linewidth = 2.0, zorder=1e10) # alpha=all_L[i], 
        if scatter:
            plt.scatter(all_radii_EOS[i], all_masses_EOS[i], color=color, s=5, zorder=1e10)
        
    plt.xlim(5, 16)
    plt.ylim(0.75, 3.25)
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_{\odot}$]")

    if not no_cmap:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r'Normalized $\log \mathcal{L}_{\rm{NICER}}$', fontsize = 22)

    plt.tight_layout()
    save_name = f"./figures/contours_{suffix}.png" 
    print(f"Saving to: {save_name}")
    plt.savefig(save_name, bbox_inches = "tight")
    plt.close()

    print("DONE")
    end = time.time()
    print(f"Time taken: {end - start} s")
    
def main():
    # plot_corner_data("J0030")
    # plot_corner_data("J0740")
    
    NMAX_NSAT = 25
    NMAX = NMAX_NSAT * 0.16
    N = 10
    NB_CSE = 8
    my_nbreak = 2.0 * 0.16
    width = (NMAX - my_nbreak) / (NB_CSE + 1)
    
    ### NEP priors
    # L_sym_prior = UniformPrior(20.0, 150.0, parameter_names=["L_sym"])
    # K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
    # K_sat_prior = UniformPrior(200.0, 300.0, parameter_names=["K_sat"])
    
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
    compute_without_vmap(N, prior=prior, likelihood=likelihood)
    
    try:
        plot_eos_contours(N, 
                          psr_name = psr_names, 
                          REX_name = REX_names,
                          scatter = True)
    except Exception as e:
        print(f"Error in plotting: {e}")
    
    print("DONE")
    
if __name__ == "__main__":
    main()