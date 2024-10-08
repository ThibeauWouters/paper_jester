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
    plt.savefig("./figures/test.pdf", bbox_inches = "tight")
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
    
    plt.savefig("./figures/test_TOV.pdf", bbox_inches = "tight")
    plt.close()
    
def match_target_cs2(which: str, which_score: str):
    
    supported_which = ["hauke", "sine"]
    if which not in supported_which:
        raise ValueError(f"which must be one of {supported_which}")
    
    # Get the EOS
    if which == "hauke":
        # Load micro and macro targets
        micro_filename = "../doppelgangers/my_target_microscopic.dat"
        macro_filename = "../doppelgangers/my_target_macroscopic.dat"
        
    else:
        # TODO: get the macro for the sine cs2!
        # Load micro and macro targets
        micro_filename = "../doppelgangers/my_sine_wave.dat"
        macro_filename = "../doppelgangers/my_target_macroscopic.dat" # NOTE: WRONG FILE! Only use micro for now
       
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
                          which_score=which_score,
                          micro_target_filename=micro_filename,
                          macro_target_filename=macro_filename,
                          nb_steps = 500)
    
    # Get the NN state
    n, cs2 = run.run_nn()
    
    # Solve the TOV equations
    ns_og, ps_og, hs_og, es_og, dloge_dlogps_og, _, cs2_og = run.transform.eos.construct_eos(run.transform.fixed_params, run.transform.eos.state.params)
    eos_tuple = (ns_og, ps_og, hs_og, es_og, dloge_dlogps_og)
    _, m, r, l = transform.construct_family_lambda(eos_tuple)
    
    # Mask them
    min_nsat = 0.5
    max_nsat = 6.0
    
    mask = (min_nsat < n) * (n < max_nsat)
    mask_target = (min_nsat < run.n_target) * (run.n_target < max_nsat)
    
    # Make the plot
    plt.figure(figsize=(12, 6))
    plt.plot(n[mask], cs2[mask], label = "Result found", color = "red", zorder = 4)
    plt.plot(run.n_target[mask_target], run.cs2_target[mask_target], label = "Target", linestyle = "--", color = "black", zorder = 5)
    plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
    plt.ylabel(r"$c_s^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./figures/test_match.pdf", bbox_inches = "tight")
    plt.close()
    
    # Make the plot for TOV as well
    plt.subplots(figsize=(12, 6), nrows = 1, ncols = 2)
    mask = (m > 0.5) * (m < 3.0)
    
    plt.subplot(121)
    plt.plot(r[mask], m[mask], color = "black", zorder = 4)
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_\odot$]")
    
    plt.subplot(122)
    plt.plot(m[mask], l[mask], color = "black", zorder = 4)
    plt.xlabel(r"$M$ [$M_\odot$]")
    plt.ylabel(r"$\Lambda$")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("./figures/test_match_TOV.pdf", bbox_inches = "tight")
    plt.close()
    
def get_sine_EOS(break_density = 2.0):
    
    # Load Hauke's EOS
    data = np.loadtxt("../doppelgangers/my_target_microscopic.dat")
    n_target, cs2_target = data[:, 0] / 0.16, data[:, 3]
    
    # Start a sine wave after the break density
    n = np.linspace(0.1, 6, 1000)
    cs2_target = np.interp(n, n_target, cs2_target)
    e = np.zeros_like(n)
    p = np.zeros_like(n)
    
    cs2_at_break = np.interp(break_density, n, cs2_target)
    sine_wave = cs2_at_break + 0.25 * np.sin((n - break_density) * np.pi)
    
    cs2 = np.where(n < break_density, cs2_target, sine_wave)

    # Save as target as .dat file:
    data = np.column_stack((n * 0.16, e, p, cs2))
    np.savetxt("../doppelgangers/my_sine_wave.dat", data, delimiter=' ')
    
    # Make the plot
    plt.figure(figsize=(12, 6))
    plt.plot(n, cs2, label = "Sine wave", color = "red", zorder = 4)
    plt.scatter(n, cs2, color = "red", zorder = 4)
    plt.savefig("./figures/sine_wave.pdf", bbox_inches = "tight")
    plt.close()
    
def main():
    get_sine_EOS()
    # test_random_initialization()
    match_target_cs2(which = "sine", which_score = "micro")
    
if __name__ == "__main__":
    main()