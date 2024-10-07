import psutil
p = psutil.Process()
p.cpu_affinity([0])

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
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

from joseTOV.eos import MetaModel_with_CSE_EOS_model, MetaModel_EOS_model, MetaModel_with_NN_EOS_model, construct_family
from joseTOV import utils

from paper_jose.utils import MicroToMacroTransform

import numpy as np 
import matplotlib.pyplot as plt
params = {"axes.grid": True,
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

plt.rcParams.update(params)

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

    
def test_random_initialization():
    
    # Taken from emulators paper Ingo and Rahul
    NMAX_NSAT = 6.0
    NEP_CONSTANTS_DICT = {
        "E_sym": 33.0,
        "L_sym": 60.0,
        "K_sym": -200.0,
        "Q_sym": 0.0,
        "Z_sym": 0.0,
        
        "E_sat": -16.0,
        "K_sat": 230.0,
        "Q_sat": 0.0,
        "Z_sat": 0.0,
        
        "nbreak": 1.5 * 0.16,
    }
    
    # Define the transform
    name_mapping = (list(NEP_CONSTANTS_DICT.keys()) + ["nn_state"], ["masses_EOS", "radii_EOS", "Lambdas_EOS", "p_c_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])
    transform = MicroToMacroTransform(name_mapping=name_mapping,
                                      nmax_nsat = NMAX_NSAT,
                                      use_neuralnet=True)
    
    # Get the NN state
    key = jax.random.PRNGKey(1)
    state = transform.eos.initialize_nn_state(key)
    
    # Construct the EOS
    ns_og, ps_og, hs_og, es_og, dloge_dlogps_og, _, cs2_og = transform.eos.construct_eos(NEP_CONSTANTS_DICT, state.params)
    
    # Convert these to units that we use more often for visualization
    n = ns_og / utils.fm_inv3_to_geometric / 0.16
    p = ps_og / utils.MeV_fm_inv3_to_geometric
    e = es_og / utils.MeV_fm_inv3_to_geometric
    
    nmin = 0.1
    mask = (nmin < n) * (n < NMAX_NSAT)
    n = n[mask]
    p = p[mask]
    e = e[mask]
    cs2 = cs2_og[mask]
    
    # Make the plot
    plt.subplots(figsize = (14, 10), nrows = 1, ncols = 2)
    plt.subplot(221)
    c = "black"
    plt.plot(n, e, color = c)
    # plt.scatter(n_TOV, e_TOV, color = c)
    # plt.plot(n_target, e_target, color = "black", label = "Target")
    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$e$ [MeV fm$^{-3}$]")
    plt.axvline(1.5, color = "red", linestyle = "--")
    # plt.axvline(1.1 * 1.5, color = "red", linestyle = "--")
    plt.xlim(nmin, NMAX_NSAT)
    
    plt.subplot(222)
    plt.plot(n, p, color = c)
    # plt.scatter(n_TOV, p_TOV, color = c)
    # plt.plot(n_target, p_target, color = "black", label = "Target")
    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
    plt.axvline(1.5, color = "red", linestyle = "--")
    # plt.axvline(1.1 * 1.5, color = "red", linestyle = "--")
    plt.xlim(nmin, NMAX_NSAT)
    
    plt.subplot(223)
    plt.plot(n, cs2, color = c)
    # plt.scatter(n_TOV, cs2_TOV, color = c)
    # plt.plot(n_target, cs2_target, color = "black", label = "Target")
    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$c_s^2$")
    plt.axvline(1.5, color = "red", linestyle = "--")
    # plt.axvline(1.1 * 1.5, color = "red", linestyle = "--")
    plt.xlim(nmin, NMAX_NSAT)
    plt.ylim(0, 1)
    
    plt.subplot(224)
    
    plt.plot(e, p, color = c)
    
    # # Legend for n_TOV dots:
    # plt.plot(self.e_target[mask_target], self.p_target[mask_target], color = "black", label = "Target")
    # legend_elements = [Line2D([0], [0], marker='o', color='black', label=r'$n_{\rm{TOV}}$', markerfacecolor='black', markersize=10)]
    # plt.legend(handles=legend_elements, loc='best')
    # plt.scatter(e_TOV, p_TOV, color = c)
    # plt.xlim(e_min, e_max)
    
    plt.xlabel(r"$e$ [MeV fm$^{-3}$]")
    plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
    
    plt.tight_layout()
    plt.savefig("./figures/test.png", bbox_inches = "tight")
    plt.close()
    
    # Try solve the TOV equations:
    eos_tuple = (ns_og, ps_og, hs_og, es_og, dloge_dlogps_og)
    p_c_EOS, masses_EOS, radii_EOS, Lambdas_EOS = transform.construct_family_lambda(eos_tuple)
    
    print("masses")
    print(masses_EOS)
    
    print("radii")
    print(radii_EOS)
    
    print("Lambdas")
    print(Lambdas_EOS)
    
    # Make the plot
    plt.subplots(figsize = (14, 10), nrows = 1, ncols = 2)
    
    plt.subplot(121)
    plt.plot(radii_EOS, masses_EOS, color = "black")
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_\odot$]")
    
    plt.subplot(122)
    plt.plot(masses_EOS, Lambdas_EOS, color = "black")
    plt.xlabel(r"$M$ [$M_\odot$]")
    plt.ylabel(r"$\Lambda$")
    plt.yscale("log")
    
    plt.savefig("./figures/test_TOV.png", bbox_inches = "tight")
    plt.close()
    
def match_target_cs2(which: str):
    
    supported_which = ["hauke", "sine"]
    if which not in supported_which:
        raise ValueError(f"which must be one of {supported_which}")
    
    # Get the EOS
    if which == "hauke":
        # Load micro and macro targets
        micro_filename = "../doppelgangers/36022_microscopic.dat"
        macro_filename = "../doppelgangers/36022_macroscopic.dat"
        
       
    ### Setup of EOS and transform:
    NMAX_NSAT = 6.0
    
    # Define the transform
    name_mapping = (["nn_state"], ["masses_EOS", "radii_EOS", "Lambdas_EOS", "p_c_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])
    transform = MicroToMacroTransform(name_mapping=name_mapping,
                                      nmax_nsat = NMAX_NSAT,
                                      use_neuralnet=True)
    
    from paper_jose.doppelgangers.doppelgangers import DoppelgangerRun
    run = DoppelgangerRun(None, 
                          transform, 
                          "micro", 
                          42, 
                          micro_target_filename=micro_filename, 
                          macro_target_filename=macro_filename)
    
    # Get the NN state
    key = jax.random.PRNGKey(1)
    state = transform.eos.initialize_nn_state(key)
    
    params = {"nn_state": state.params}
    n, cs2 = run.run_micro(params)
    
    # Mask them
    min_nsat = 0.5
    max_nsat = 6.0
    
    mask = (min_nsat < n) * (n < max_nsat)
    mask_target = (min_nsat < run.n_target) * (run.n_target < max_nsat)
    
    # Make the plot
    plt.figure(figsize=(12, 6))
    plt.plot(n[mask], cs2[mask], label = "Result found", color = "red")
    plt.plot(run.n_target[mask_target], run.cs2_target[mask_target], label = "Target", color = "black")
    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$c_s^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/test_match.png", bbox_inches = "tight")
    plt.close()
    
def main():
    # test_random_initialization()
    match_target_cs2(which = "hauke")
    
if __name__ == "__main__":
    main()