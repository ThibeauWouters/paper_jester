import psutil
p = psutil.Process()
p.cpu_affinity([0])

import numpy as np
import jax
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

class MicroToMacroTransform(NtoMTransform):
    
    def __init__(self,
                 name_mapping: tuple[list[str], list[str]],
                 keep_names: list[str] = None,
                 # metamodel kwargs:
                 ndat_metamodel: int = 100,
                 # CSE kwargs
                 nmax_nsat: float = 25,
                 # TOV kwargs
                 min_nsat_TOV: float = 1.0,
                 ndat_TOV: int = 100,
                 ndat_CSE: int = 100,
                 nb_masses: int = 100,
                 fixed_params: dict[str, float] = None,
                ):
    
        # By default, keep all names
        if keep_names is None:
            keep_names = name_mapping[0]
        super().__init__(name_mapping, keep_names=keep_names)
    
        # Save as attributes
        self.ndat_metamodel = ndat_metamodel
        self.nmax_nsat = nmax_nsat
        self.nmax = nmax_nsat * 0.16
        self.min_nsat_TOV = min_nsat_TOV
        self.ndat_TOV = ndat_TOV
        self.ndat_CSE = ndat_CSE
        self.nb_masses = nb_masses
        
        eos = MetaModel_with_NN_EOS_model(nmax_nsat=self.nmax_nsat,
                                          ndat_metamodel=self.ndat_metamodel,
                                          ndat_CSE=self.ndat_CSE)
        
        self.eos = eos
        
        # Construct a lambda function for solving the TOV equations, fix the given parameters
        self.construct_family_lambda = lambda x: construct_family(x, ndat = self.ndat_TOV, min_nsat = self.min_nsat_TOV)
        
    def transform_func(self, params: dict[str, Float]) -> dict[str, Float]:
        
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        nn_state = params["nn_state"]
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, cs2 = self.eos.construct_eos(NEP, nn_state)
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        
        # Solve the TOV equations
        p_c_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
    
        return_dict = {"masses_EOS": masses_EOS, "radii_EOS": radii_EOS, "Lambdas_EOS": Lambdas_EOS, "p_c_EOS": p_c_EOS,
                    "n": ns, "p": ps, "h": hs, "e": es, "dloge_dlogp": dloge_dlogps, "cs2": cs2}

        return return_dict
    
def main():
    
    # Taken from emulators paper Ingo and Rahul
    NMAX_NSAT = 6.0
    NEP_CONSTANTS_DICT = {
        "E_sym": 33,
        "L_sym": 60,
        "K_sym": -200,
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
                                      nmax_nsat = NMAX_NSAT,)
    
    # Get the NN state
    key = jax.random.PRNGKey(1)
    state = transform.eos.initialize_nn_state(key)
    
    # Construct the EOS
    ns_og, ps_og, hs_og, es_og, dloge_dlogps_og, _, cs2_og = transform.eos.construct_eos(NEP_CONSTANTS_DICT, state)
    
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
    
    
if __name__ == "__main__":
    main()