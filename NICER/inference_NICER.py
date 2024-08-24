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

import tqdm
import time
import numpy as np
np.random.seed(43) # for reproducibility
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import corner

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
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

L_sym_prior = UniformPrior(20.0, 150.0, parameter_names=["L_sym"])
K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
K_sat_prior = UniformPrior(200.0, 300.0, parameter_names=["K_sat"])

my_eps = 1e-3 # small nudge to avoid overlap in CSE gridpoints
NBREAK_NSAT = 2.0

n_CSE_0_prior = UniformPrior(NBREAK_NSAT * 0.16, 3.0 * 0.16 - my_eps, parameter_names=["n_CSE_0"])
n_CSE_1_prior = UniformPrior(3.0 * 0.16,  4.0 * 0.16 - my_eps, parameter_names=["n_CSE_1"])
n_CSE_2_prior = UniformPrior(4.0 * 0.16,  5.0 * 0.16 - my_eps, parameter_names=["n_CSE_2"])
n_CSE_3_prior = UniformPrior(5.0 * 0.16,  6.0 * 0.16 - my_eps, parameter_names=["n_CSE_3"])
n_CSE_4_prior = UniformPrior(6.0 * 0.16,  7.0 * 0.16 - my_eps, parameter_names=["n_CSE_4"])
n_CSE_5_prior = UniformPrior(7.0 * 0.16,  8.0 * 0.16 - my_eps, parameter_names=["n_CSE_5"])
n_CSE_6_prior = UniformPrior(8.0 * 0.16,  9.0 * 0.16 - my_eps, parameter_names=["n_CSE_6"])
n_CSE_7_prior = UniformPrior(9.0 * 0.16, 10.0 * 0.16, parameter_names=["n_CSE_7"])

# TODO: I am a bit scared about the bounds
my_eps = 1e-2
cs2_CSE_0_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_0"])
cs2_CSE_1_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_1"])
cs2_CSE_2_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_2"])
cs2_CSE_3_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_3"])
cs2_CSE_4_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_4"])
cs2_CSE_5_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_5"])
cs2_CSE_6_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_6"])
cs2_CSE_7_prior = UniformPrior(0.0 + my_eps, 1.0 - my_eps, parameter_names=["cs2_CSE_7"])


prior_list = [L_sym_prior, 
              K_sym_prior, 
              K_sat_prior,
            #   n_CSE_0_prior,
            #   n_CSE_1_prior,
            #   n_CSE_2_prior,
            #   n_CSE_3_prior,
            #   n_CSE_4_prior,
            #   n_CSE_5_prior,
            #   n_CSE_6_prior,
            #   n_CSE_7_prior,
            #   cs2_CSE_0_prior,
            #   cs2_CSE_1_prior,
            #   cs2_CSE_2_prior,
            #   cs2_CSE_3_prior,
            #   cs2_CSE_4_prior,
            #   cs2_CSE_5_prior,
            #   cs2_CSE_6_prior,
            #   cs2_CSE_7_prior
]

prior = CombinePrior(prior_list)
sampled_param_names = prior.parameter_names

###########
### Jim ###
###########

likelihood = NICER_utils.NICERLikelihood(sampled_param_names, NBREAK_NSAT)

mass_matrix = jnp.eye(prior.n_dim)
local_sampler_arg = {"step_size": mass_matrix * 1e-2}


test_kwargs = {"n_loop_training": 2,
               "n_loop_production": 2,
               "n_chains": 5,
               "n_local_steps": 5,
               "n_global_steps": 5,
               "n_epochs": 5
}

production_kwargs = {"n_loop_training": 5,
                     "n_loop_production": 3,
                     "n_chains": 500,
                     "n_local_steps": 3,
                     "n_global_steps": 25,
                     "n_epochs": 20,
                     "train_thinning": 1,
                     "output_thinning": 1,
}

jim = Jim(likelihood,
          prior,
          local_sampler_arg = local_sampler_arg,
          **production_kwargs)

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

outdir = f"./outdir_{NICER_utils.PSR_NAME}/"
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