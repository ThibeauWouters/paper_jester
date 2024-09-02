"""
DEPRECATED: I will try to immediately go to the GW inference step-by-step. 
"""

### On CIT
import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
### 

import numpy as np
import matplotlib.pyplot as plt
import corner
import jax
import jax.numpy as jnp
from jaxtyping import Float, jaxtyped

from jimgw.prior import UniformPrior, CombinePrior
from jimgw.transforms import NtoMTransform
from jimgw.base import LikelihoodBase
from jimgw.jim import Jim

import jimgw.constants as constants

constant_H0 = 67.4

from flowMC import Sampler

import paper_jose.utils as utils

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

##################
### LIKELIHOOD ###
##################

class MyLikelihood(LikelihoodBase):
    
    true_M_c: Float
    true_q: Float
    true_d_L: Float
    
    true_lambda_1: Float
    true_lambda_2: Float
    
    def __init__(self,
                 true_M_c: Float,
                 true_q: Float,
                 true_d_L: Float,
                 true_lambda_1: Float,
                 true_lambda_2: Float):
        
        
        self.true_M_c = true_M_c
        self.true_q = true_q
        self.true_d_L = true_d_L
        
        true_params = {'M_c': true_M_c, 'q': true_q, 'd_L': true_d_L}
        true_source_masses_dict = utils.detector_frame_M_c_q_to_source_frame_m_1_m_2(true_params)
        
        true_m_1_source = true_source_masses_dict['m_1']
        true_m_2_source = true_source_masses_dict['m_2']
        
        self.true_m_1_source = true_m_1_source
        self.true_m_2_source = true_m_2_source
        
        self.true_lambda_1 = true_lambda_1
        self.true_lambda_2 = true_lambda_2
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """Test case: do the transformation and put a Gaussian at fiducial true mass values"""
        
        ### Put likelihood on source frame masses
        # m_1_source, m_2_source = params['m_1'], params['m_2']
        # m1_std = 0.1
        # m2_std = 0.1
        
        lambda_1, lambda_2 = params['lambda_1'], params['lambda_2']
        
        lambda_1_std = 50
        lambda_2_std = 50
        
        # val = -0.5 * (((m_1_source - self.true_m_1_source) / m1_std)**2 + ((m_2_source - self.true_m_2_source) / m2_std)**2)
        val = -0.5 * (((lambda_1 - self.true_lambda_1) / lambda_1_std)**2 + ((lambda_2 - self.true_lambda_2) / lambda_2_std)**2)
        
        return val
        
# Setup
true_M_c = 1.4
true_q = 0.9
true_d_L = 40
true_lambda_1 = 750.0
true_lambda_2 = 600.0

# Priors
eps = 0.5 # half of width of the chirp mass prior
mc_prior = UniformPrior(true_M_c - eps, true_M_c + eps, parameter_names=['M_c'])
q_prior = UniformPrior(0.125, 1.0, parameter_names=['q'])
d_L_prior = UniformPrior(20.0, 60.0, parameter_names=['d_L'])

E_sym_prior = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym_prior = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])

eos_prior_list = [E_sym_prior, L_sym_prior]

prior_list = [mc_prior, q_prior, d_L_prior] + eos_prior_list # + utils.prior_list 
combine_prior = CombinePrior(prior_list)

# Likelihood and transform
likelihood = MyLikelihood(true_M_c, 
                          true_q,
                          true_d_L,
                          true_lambda_1,
                          true_lambda_2)

name_mapping = ([p.parameter_names[0] for p in eos_prior_list], ["masses_EOS", "Lambdas_EOS"])
eos_transform = utils.MicroToMacroTransform(name_mapping,
                                           keep_names = "all",
                                           nmax_nsat = utils.NMAX_NSAT,
                                           nb_CSE = utils.NB_CSE
                                           )

name_mapping = (combine_prior.parameter_names, ["lambda_1", "lambda_2"])
transform = utils.ChirpMassMassRatioToLambdas(name_mapping,
                                              eos_transform)

### Jim

# Other stuff we have to give to Jim to make it work
step = 5e-3
local_sampler_arg = {"step_size": step * jnp.eye(combine_prior.n_dim)}

# Jim:
jim = Jim(likelihood, 
          combine_prior, 
          likelihood_transforms=[transform],
          n_chains = 10,
          parameter_names=['M_c', 'q', 'd_L'] + utils.sampled_param_names,
          n_loop_training=2,
          n_loop_production=2,
          n_local_steps = 2,
          n_global_steps = 5,
          n_epochs = 5,
          local_sampler_arg=local_sampler_arg)

jim.sample(jax.random.PRNGKey(3))
jim.print_summary()
    
# Go from Mc, q samples to m1, m2 samples
chains_named = jim.get_samples()

print("NB samples: ", np.shape(chains_named['M_c']))

### Prior space corner plot
chains = np.array([chains_named['M_c'], chains_named['q'], chains_named['d_L']]).T
chains = np.reshape(chains, (-1, 3))
corner.corner(chains, labels = ["M_c", "q", "d_L"], truths = np.array([true_M_c, true_q, true_d_L]), **default_corner_kwargs)

plt.savefig("./figures/test_lambdas_transform_with_distance_before.png", bbox_inches = 'tight')
plt.close()

# ### Transformed space:
# chains = jax.vmap(transform.forward)(chains_named)

# print("chains")
# print(chains)

# lambda_1, lambda_2 = chains_named['lambda_1'], chains_named['lambda_2']
# chains = np.vstack([lambda_1, lambda_2]).T

# corner.corner(chains, labels = ["lambda_1", "lambda_2"], truths = np.array([true_lambda_1, true_lambda_2]), **default_corner_kwargs)

# plt.savefig("./figures/test_lambdas_transform_with_distance_after.png", bbox_inches = 'tight')
# plt.close()