"""
Full-scale NICER inference: we will use jim as flowMC wrapper
"""

################
### PREAMBLE ###
################

import os
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
jax.config.update("jax_platform_name", "cpu")

from jimgw.prior import UniformPrior, CombinePrior
from jimgw.jim import Jim

import utils as NICER_utils

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
                        levels=[0.68, 0.95],
                        plot_density=False,
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False)

start = time.time()

##############
### PRIORS ###
##############

L_sym_prior = UniformPrior(20.0, 150.0, parameter_names=["L_sym"])
K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
K_sat_prior = UniformPrior(200.0, 300.0, parameter_names=["K_sat"])

my_eps = 1e-3 # small nudge to avoid overlap in CSE gridpoints
NBREAK_NSAT = 2.0

# TODO: this is a first attempt so pretty cumbersome, improve in the future
n_CSE_0_prior = UniformPrior(NBREAK_NSAT + my_eps, 3.0 - my_eps, parameter_names=["n_CSE_0"])
n_CSE_1_prior = UniformPrior(3.0, 4.0 - my_eps, parameter_names=["n_CSE_1"])
n_CSE_2_prior = UniformPrior(4.0, 5.0 - my_eps, parameter_names=["n_CSE_2"])
n_CSE_3_prior = UniformPrior(5.0, 6.0 - my_eps, parameter_names=["n_CSE_3"])
n_CSE_4_prior = UniformPrior(6.0, 7.0 - my_eps, parameter_names=["n_CSE_4"])
n_CSE_5_prior = UniformPrior(7.0, 8.0 - my_eps, parameter_names=["n_CSE_5"])
n_CSE_6_prior = UniformPrior(8.0, 9.0 - my_eps, parameter_names=["n_CSE_6"])
n_CSE_7_prior = UniformPrior(9.0, 10.0 - my_eps, parameter_names=["n_CSE_7"])

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
        #       n_CSE_0_prior,
        #       n_CSE_1_prior,
        #       n_CSE_2_prior,
        #       n_CSE_3_prior,
        #       n_CSE_4_prior,
        #       n_CSE_5_prior,
        #       n_CSE_6_prior,
        #       n_CSE_7_prior,
        #       cs2_CSE_0_prior,
        #       cs2_CSE_1_prior,
        #       cs2_CSE_2_prior,
        #       cs2_CSE_3_prior,
        #       cs2_CSE_4_prior,
        #       cs2_CSE_5_prior,
        #       cs2_CSE_6_prior,
        #       cs2_CSE_7_prior
]

prior = CombinePrior(prior_list)
sampled_param_names = prior.parameter_names

###########
### Jim ###
###########

likelihood = NICER_utils.NICERLikelihood(sampled_param_names, NBREAK_NSAT)

mass_matrix = jnp.eye(prior.n_dim)
local_sampler_arg = {"step_size": mass_matrix * 1e-3}

jim = Jim(likelihood,
          prior,
          n_loop_training = 2,
          n_loop_production = 2,
          n_chains = 10,
          n_epochs = 20,
          jit = True,
          local_sampler_arg = local_sampler_arg)

jim.sample(jax.random.PRNGKey(0))

# ##############
# ### CORNER ###
# ##############

samples = jim.get_samples()

print("np.shape(samples)")
print(np.shape(samples))


print("DONE")
end = time.time()
print(f"Time taken: {end - start} s")