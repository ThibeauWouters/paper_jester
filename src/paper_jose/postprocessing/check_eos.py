"""
Check up on the final EOS
"""

# import psutil
# p = psutil.Process()
# p.cpu_affinity([0])
import os
import tqdm
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
print(jax.devices())

import jax.numpy as jnp
from jimgw.prior import UniformPrior, CombinePrior

import paper_jose.utils as utils

# Iterate over the final solutions

def check_eos(n_nep: int):
    
    # Prior stuff
    my_nbreak = 2.0 * 0.16
    NMAX_NSAT = 5
    NB_CSE = 0
    NMAX = NMAX_NSAT * 0.16
    width = (NMAX - my_nbreak) / (NB_CSE + 1)
    
    ### TODO: decide which parameters to keep fixed here
    if n_nep == 2:
        fixed_params_keys = ["K_sym", "K_sat", "Q_sym", "Q_sat", "Z_sym", "Z_sat"] # only up to first order
    elif n_nep == 4:
        fixed_params_keys = ["Q_sym", "Q_sat", "Z_sym", "Z_sat"] # only up to second order
    elif n_nep == 6:
        fixed_params_keys = ["Z_sym", "Z_sat"]
    elif n_nep == 8:
        fixed_params_keys = ["E_sym"] 
    else:
        raise ValueError(f"n_NEP = {n_nep} is not supported")
    
    # Here, we can (if so desired) fix the values of the CSE part of the EOS parametrization to be fixed to the target
    fixed_params = {k: v for k, v in utils.NEP_CONSTANTS_DICT.items() if k in fixed_params_keys}
    
    print(f"Doppelganger main has the following fixed params")
    for k, v in fixed_params.items():
        print(f"{k}: {v}")
    
    ### DEFINE TRANSFORM

    # NEP priors
    K_sat_prior = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
    Q_sat_prior = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
    Z_sat_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])

    E_sym_prior = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
    L_sym_prior = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
    K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
    Q_sym_prior = UniformPrior(-800.0, 800.0, parameter_names=["Q_sym"])
    Z_sym_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sym"])

    all_NEP_prior_list = [
        E_sym_prior,
        L_sym_prior, 
        K_sym_prior,
        Q_sym_prior,
        Z_sym_prior,

        K_sat_prior,
        Q_sat_prior,
        Z_sat_prior,
    ]
    
    prior_list = []
    for NEP_prior in all_NEP_prior_list:
        if NEP_prior.parameter_names[0] not in fixed_params_keys:
            prior_list.append(NEP_prior)
            
    print(f"Added the NEP priors to the prior list for the sampling, the prior list is now:")
    for p in prior_list:
        print(p)
    
    # Combine the prior
    prior = CombinePrior(prior_list)
    
    sampled_param_names = prior.parameter_names
    name_mapping = (sampled_param_names, ["logpc_EOS", "masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])
    transform = utils.MicroToMacroTransform(name_mapping,
                                            nmax_nsat=NMAX_NSAT,
                                            nb_CSE=NB_CSE,
                                            fixed_params = fixed_params)
    
    ### ITERATE OVER SOLUTIONS
    
    if n_nep == 8:
        directory = "../doppelgangers/campaign_results/E_sym_fixed_NEPs/"
    else:
        directory = f"../doppelgangers/campaign_results/{n_nep}_NEPs_new/"
        
    # Check if directory exists 
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist")
    
    # Iterate over the subdirs:
    subdir_list = os.listdir(directory)
    for subdir in tqdm.tqdm(subdir_list):
        data_dir = os.path.join(directory, subdir, "data")
        
        npz_files = os.listdir(data_dir)
        ids = [int(f.split(".")[0]) for f in npz_files]
        max_id = max(ids)
        
        final_file = os.path.join(data_dir, f"{max_id}.npz")
        data = np.load(final_file)
        
        NEPs = {"E_sym": data["E_sym"], "L_sym": data["L_sym"], "K_sym": data["K_sym"], "Q_sym": data["Q_sym"], "Z_sym": data["Z_sym"],
                "K_sat": data["K_sat"], "Q_sat": data["Q_sat"], "Z_sat": data["Z_sat"]}
        
        # Transform:
        out = transform.forward(NEPs)
    
def main():
    n_nep_list = [2, 4, 6, 8]
    n_nep_list = [8]
    for n_nep in n_nep_list:
        print(f" === Checking the EOS results for n_NEP = {n_nep_list} ===")
        check_eos(n_nep)
        
    print("DONE")
    
if __name__ == "__main__":
    main()