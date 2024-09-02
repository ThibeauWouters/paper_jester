"""
Full-scale NICER inference: we will use jim as flowMC wrapper
"""

################
### PREAMBLE ###
################
import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
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
# jax.config.update("jax_platform_name", "cpu")

from jimgw.prior import UniformPrior, CombinePrior
from jimgw.jim import Jim

import utils
import utils_plotting

start = time.time()

##############
### PRIORS ###
##############

my_nbreak = 2.0 * 0.16
NMAX_NSAT = 25
NMAX = NMAX_NSAT * 0.16
N = 100
NB_CSE = 8
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
    prior_list.append(UniformPrior(left, right, parameter_names=[f"n_CSE_{i}"]))
    prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))

# Final point to end
prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))
prior = CombinePrior(prior_list)
sampled_param_names = prior.parameter_names


# Likelihood: choose which PSR(s) to perform inference on:
psr_names = ["J0030", "J0740"]
likelihoods_list_NICER = [utils.NICERLikelihood(psr) for psr in psr_names]

REX_names = ["PREX", "CREX"]
likelihoods_list_REX = [utils.REXLikelihood(rex) for rex in REX_names]

name_mapping = (sampled_param_names, ["masses_EOS", "radii_EOS", "Lambdas_EOS"])
my_transform = utils.MicroToMacroTransform(name_mapping, 
                                           keep_names = ["E_sym", "L_sym"],
                                           nmax_nsat=NMAX_NSAT,
                                           nb_CSE = NB_CSE
                                           )

likelihoods_list = likelihoods_list_NICER + likelihoods_list_REX
likelihood = utils.CombinedLikelihood(likelihoods_list)

###########
### Jim ###
###########


# Sampler kwargs
mass_matrix = jnp.eye(prior.n_dim)
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
kwargs = production_kwargs

jim = Jim(likelihood,
          prior,
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