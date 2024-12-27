"""
Full-scale inference: we will use jim as flowMC wrapper
"""

################
### PREAMBLE ###
################
import os 
import time
import shutil
import numpy as np
np.random.seed(43) # for reproducibility
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import equinox as eqx
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.distributions import Normal, Transformed
from jimgw.prior import UniformPrior, CombinePrior
from jimgw.jim import Jim
import paper_jose.utils as utils
import utils_plotting
import sys

print(f"GPU found?")
print(jax.devices())

def main(outdir):
    
    NMAX_NSAT = 25
    NB_CSE = 8
    
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
    if NB_CSE > 0:
        nbreak_prior = UniformPrior(1.0 * 0.16, 2.0 * 0.16, parameter_names=[f"nbreak"])
        prior_list.append(nbreak_prior)
        for i in range(NB_CSE):
            # NOTE: the density parameters are sampled from U[0, 1], so we need to scale it, but it depends on break so will be done internally
            prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"n_CSE_{i}_u"]))
            prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))

        # Final point to end
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))

    # Construct the EOS prior and a transform here which can be used down below for creating the EOS plots after inference is completed
    eos_prior = CombinePrior(prior_list)
    eos_param_names = eos_prior.parameter_names
    all_output_keys = ["logpc_EOS", "masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"]
    name_mapping = (eos_param_names, all_output_keys)
    
    # This transform will be the same as my_transform, but with different output keys, namely, all EOS related quantities, for postprocessing
    my_transform_eos = utils.MicroToMacroTransform(name_mapping,
                                                   keep_names = ["E_sym", "L_sym", "nbreak"],
                                                   nmax_nsat = NMAX_NSAT,
                                                   nb_CSE = NB_CSE
                                                )
    
    ### POSTPROCESSING ###
    data = np.load(os.path.join(outdir, "results_production.npz"))
    params = {k: value for k, value in data.items() if k in eos_param_names}
    log_prob = data["log_prob"]

    # Generate the final EOS + TOV samples from the EOS parameter samples
    idx = np.random.choice(np.arange(len(log_prob)), size=10_000, replace=False)
    TOV_start = time.time()
    chosen_samples = {k: jnp.array(v[idx]) for k, v in params.items()}
    # NOTE: jax lax map helps us deal with batching, but a batch size multiple of 10 gives errors, therefore this weird number
    transformed_samples = jax.lax.map(jax.jit(my_transform_eos.forward), chosen_samples, batch_size = 4_999)
    TOV_end = time.time()
    print(f"Time taken for TOV map: {TOV_end - TOV_start} s")
    chosen_samples.update(transformed_samples)

    log_prob = log_prob[idx]
    np.savez(os.path.join(outdir, "eos_samples.npz"), log_prob=log_prob, **chosen_samples)
    
if __name__ == "__main__":
    outdir = sys.argv[1]
    main(outdir)