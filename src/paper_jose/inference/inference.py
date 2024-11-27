"""
Full-scale inference: we will use jim as flowMC wrapper
"""

################
### PREAMBLE ###
################
# import psutil
# p = psutil.Process()
# p.cpu_affinity([0])
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
import os 
import json

import time
import numpy as np
np.random.seed(43) # for reproducibility
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import corner

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

start = time.time()

##################
### LIKELIHOOD ###
##################

# Likelihood: choose which PSR(s) to perform inference on:
psr_names = ["J0030", "J0740"]
print(f"Loading PSR data from {psr_names}")
likelihoods_list_NICER = [utils.NICERLikelihood(psr) for psr in psr_names]

GW_id = "real"
print(f"Loading GW data from {GW_id}")
likelihoods_list_GW = [utils.GWlikelihood(GW_id)]

REX_names = ["PREX", "CREX"]
print(f"Loading PREX/CREX from {REX_names}")
likelihoods_list_REX = [utils.REXLikelihood(rex) for rex in REX_names]

my_transform = utils.MicroToMacroTransform(utils.name_mapping,
                                           keep_names = ["E_sym", "L_sym"],
                                           nmax_nsat = utils.NMAX_NSAT,
                                           nb_CSE = utils.NB_CSE
                                           )

likelihoods_list = likelihoods_list_NICER + likelihoods_list_REX + likelihoods_list_GW
likelihood = utils.CombinedLikelihood(likelihoods_list)

###########
### Jim ###
###########


# Sampler kwargs
mass_matrix = jnp.eye(utils.prior.n_dim)
local_sampler_arg = {"step_size": mass_matrix * 1e-2}
production_kwargs = {"n_loop_training": 5,
          "n_loop_production": 3,
          "n_chains": 500,
          "n_local_steps": 2,
          "n_global_steps": 25,
          "n_epochs": 10,
          "train_thinning": 1,
          "output_thinning": 1,
}

test_kwargs = {"n_loop_training": 2,
          "n_loop_production": 2,
          "n_chains": 10,
          "n_local_steps": 2,
          "n_global_steps": 5,
          "n_epochs": 5,
          "train_thinning": 1,
          "output_thinning": 1,
}
kwargs = test_kwargs

jim = Jim(likelihood,
          utils.prior,
          local_sampler_arg = local_sampler_arg,
          likelihood_transforms = [my_transform],
          **kwargs)

jim.sample(jax.random.PRNGKey(1))
jim.print_summary()

##############
### CORNER ###
##############

outdir = f"./outdir/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Training (just to count number of samples)
sampler_state = jim.sampler.get_sampler_state(training=True)
log_prob = sampler_state["log_prob"].flatten()
nb_samples_training = len(log_prob)


# Production (also for postprocessing plotting)
sampler_state = jim.sampler.get_sampler_state(training=False)
log_prob = sampler_state["log_prob"].flatten()
nb_samples_training = len(log_prob)

samples = jim.get_samples()
keys, samples = list(samples.keys()), np.array(list(samples.values()))
log_prob = np.array(sampler_state["log_prob"])
nb_samples_production = len(log_prob.flatten())
total_nb_samples = nb_samples_training + nb_samples_production

np.savez(os.path.join(outdir, "results_production.npz"), samples=samples, log_prob=log_prob, keys=keys)

print("Log prob range")
print(np.min(log_prob), np.max(log_prob))

utils_plotting.plot_corner(outdir, samples, keys)

### Plot the EOS

print("Plotting EOS")

samples = np.reshape(samples, (len(keys), -1))
named_values = dict(zip(keys, samples))
log_prob = log_prob.flatten()

# Highest likelihood EOS
max_idx = np.argmax(log_prob)
max_log_prob = log_prob[max_idx]
max_values = {k: v[max_idx] for k, v in named_values.items()}

transformed_max_log_prob = my_transform.forward(max_values)
np.savez(os.path.join(outdir, "max_log_prob.npz"), max_values=max_values, transformed_max_log_prob=transformed_max_log_prob)

# Sample a few EOS from the posterior samples:
N_samples = 200
idx = np.random.choice(len(log_prob), N_samples)
named_samples = {k: v[idx] for k, v in named_values.items()}
transformed_samples = jax.vmap(my_transform.forward)(named_samples)
np.savez(os.path.join(outdir, "eos_samples.npz"), named_samples=named_samples, transformed_samples=transformed_samples)

utils_plotting.plot_eos(outdir, transformed_max_log_prob, transformed_samples)

end = time.time()
runtime = end - start

print(f"Time taken: {runtime} s")
print(f"Number of samples generated: {total_nb_samples}")

# Save the runtime to a file as well
with open(outdir + "runtime.txt", "w") as f:
    f.write(f"{runtime}")

print("DONE")