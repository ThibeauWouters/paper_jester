"""
Using minimization as a test
"""

################
### PREAMBLE ###
################

import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
import shutil
import time

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

from paper_jose.doppelgangers.doppelgangers import DoppelgangerRun

def main(N_runs: int = 0,
         fixed_CSE: bool = True, # use a CSE, but have it fixed, vary only the metamodel
         metamodel_only = False, # only use the metamodel, no CSE used at all
         which_score: str = "macro" # score function to be used for optimization. Recommended: "macro"
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
        # K_sym_prior,
        # Q_sym_prior,
        # Z_sym_prior,

        # K_sat_prior,
        # Q_sat_prior,
        # Z_sat_prior,
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
    
    # Get a doppelganger score
    sampled_param_names = prior.parameter_names
    name_mapping = (sampled_param_names, ["masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])
    transform = utils.MicroToMacroTransform(name_mapping, nmax_nsat=NMAX_NSAT, nb_CSE=NB_CSE)
    
    # # Choose the learning rate
    # if fixed_CSE:
    #     learning_rate = 1e3
    # else:
    #     learning_rate = 1e-3
        
    # Initialize random doppelganger: this is to run postprocessing scripts below
    doppelganger = DoppelgangerRun(prior, 
                                   transform, 
                                   which_score, 
                                   -1, 
                                   nb_steps = 200,
                                   score_fn_has_aux=False)
    
    # ### Optimizer run
    # np.random.seed(345)
    # for i in range(N_runs):
    #     seed = np.random.randint(0, 100_000)
    #     print(f" ====================== Run {i + 1} / {N_runs} with seed {seed} ======================")
        
    #     doppelganger = DoppelgangerRun(prior, transform, which_score, seed, nb_steps = 200, learning_rate = learning_rate)
        
    #     # Do a run
    #     params = doppelganger.initialize_walkers()
    #     doppelganger.run(params)
        
    ### Try out JAX optimization
    from jax.scipy.optimize import minimize
    
    start_time = time.time()
    results = minimize(doppelganger.score_fn_macro_array, jnp.array([20.0, 90.0]), method = "BFGS")
    print("results")
    print(results)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} s = {(end_time - start_time) / 60} min")
        
    # doppelganger.export_target_EOS()
    # doppelganger.perturb_doppelganger(seed = 125, nb_perturbations=1)
        
    # doppelganger.finetune_doppelganger(seed = 750)
    
    # # Plot the MTOV correlations?
    # doppelganger.plot_pressure_mtov_correlations()
    
    # ### Meta plots of the final "real" doppelgangers
    
    # final_outdir = "./outdir/"
    # doppelganger.get_table(outdir=final_outdir, keep_real_doppelgangers = True, save_table = False)
    # doppelganger.plot_doppelgangers(final_outdir, keep_real_doppelgangers = True)
    
    # doppelganger.random_sample()
    
    print("DONE")
    
if __name__ == "__main__":
    main()