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

from utils import default_corner_kwargs
import utils as NICER_utils

plt.rcParams.update(NICER_utils.mpl_params)

start = time.time()

##############
### PRIORS ###
##############

NBREAK_NSAT = 2.0
NBREAK = NBREAK_NSAT * 0.16

NMAX_NSAT = 25
NMAX = NMAX_NSAT * 0.16
N = 100
NB_CSE = 8
width = (NMAX - NBREAK) / (NB_CSE + 1)

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

for i in range(NB_CSE):
    left = NBREAK + i * width
    right = NBREAK + (i+1) * width
    prior_list.append(UniformPrior(left, right, parameter_names=[f"n_CSE_{i}"]))
    prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))

# Final point to end
prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))
prior = CombinePrior(prior_list)
sampled_param_names = prior.parameter_names

###########
### Jim ###
###########

name_mapping = (sampled_param_names, ["masses_EOS", "radii_EOS", "Lambdas_EOS"])
my_transform = NICER_utils.MicroToMacroTransform(name_mapping, 
                                                  NBREAK_NSAT,
                                                  nmax_nsat=NMAX_NSAT,
                                                  nb_CSE = NB_CSE,
                                                  ndat_TOV=100,
                                                  ndat_CSE=100
                                                  )

# Likelihood: choose which PSR(s) to perform inference on:
psr_names = ["J0740"]
likelihoods_list = [NICER_utils.NICERLikelihood(psr) for psr in psr_names]
likelihood = NICER_utils.CombinedLikelihood(likelihoods_list)

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
kwargs = test_kwargs

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

# Training (just to count number of samples)

sampler_state = jim.sampler.get_sampler_state(training=True)
log_prob = sampler_state["log_prob"].flatten()
nb_samples_training = len(log_prob)

sampler_state = jim.sampler.get_sampler_state(training=False)
log_prob = sampler_state["log_prob"]
nb_samples_production = len(log_prob.flatten())
total_nb_samples = nb_samples_training + nb_samples_production

chains = jim.get_samples(training = False)
names, samples = list(chains.keys()), np.array(list(chains.values()))

print("np.shape(samples)")
print(np.shape(samples))

outdir = f"./outdir/"
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
# Save the samples
np.savez(outdir + "results_production.npz", samples=samples, log_prob=log_prob)

samples = np.reshape(samples, (prior.n_dim, -1))

print("np.shape(samples)")
print(np.shape(samples))

print("np.shape(log_prob)")
print(np.shape(log_prob))

print("Log prob range")
print(np.min(log_prob), np.max(log_prob))

corner.corner(samples.T, labels = prior.parameter_names, **default_corner_kwargs)
plt.savefig(outdir + "corner.png")

end = time.time()
runtime = end - start

print(f"Time taken: {runtime} s")
print(f"Number of samples generated: {total_nb_samples}")

# Save the runtime to a file as well
with open(outdir + "runtime.txt", "w") as f:
    f.write(f"{runtime}")

print("DONE")