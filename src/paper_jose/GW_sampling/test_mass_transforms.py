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
    
    true_m_1_source: Float
    true_m_2_source: Float
    
    def __init__(self,
                 true_M_c: Float,
                 true_q: Float,
                 true_d_L: Float):
        
        
        self.true_M_c = true_M_c
        self.true_q = true_q
        self.true_d_L = true_d_L
        
        true_params = {'M_c': true_M_c, 'q': true_q, 'd_L': true_d_L}
        true_source_masses_dict = utils.detector_frame_M_c_q_to_source_frame_m_1_m_2(true_params)
        
        true_m_1_source = true_source_masses_dict['m_1']
        true_m_2_source = true_source_masses_dict['m_2']
        
        self.true_m_1_source = true_m_1_source
        self.true_m_2_source = true_m_2_source
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        """Test case: do the transformation and put a Gaussian at fiducial true mass values"""
        
        ### Put likelihood on source frame masses
        m_1_source, m_2_source = params['m_1'], params['m_2']
        m1_std = 0.1
        m2_std = 0.1
        return -0.5 * (((m_1_source - self.true_m_1_source) / m1_std)**2 + ((m_2_source - self.true_m_2_source) / m2_std)**2)
        
# Setup
true_M_c = 1.4
true_q = 0.9
true_d_L = 40

# Priors
eps = 0.5 # half of width of the chirp mass prior
mc_prior = UniformPrior(true_M_c - eps, true_M_c + eps, parameter_names=['M_c'])
q_prior = UniformPrior(0.125, 1.0, parameter_names=['q'])
d_L_prior = UniformPrior(20.0, 60.0, parameter_names=['d_L'])
combine_prior = CombinePrior([mc_prior, q_prior, d_L_prior])


# Likelihood and transform
likelihood = MyLikelihood(true_M_c, 
                          true_q,
                          true_d_L)

mass_transform = utils.ChirpMassMassRatioToSourceComponentMasses()

### Jim

# Other stuff we have to give to Jim to make it work
step = 5e-3
local_sampler_arg = {"step_size": step * jnp.eye(combine_prior.n_dim)}

# Jim:
jim = Jim(likelihood, 
          combine_prior, 
          likelihood_transforms=[mass_transform],
          n_chains = 100,
          parameter_names=['M_c', 'q', 'd_L'],
          n_loop_training=5,
          n_loop_production=3,
          local_sampler_arg=local_sampler_arg)

jim.sample(jax.random.PRNGKey(0))
jim.print_summary()
    
# Go from Mc, q samples to m1, m2 samples
chains_named = jim.get_samples()

### Prior space corner plot
chains = np.array([chains_named['M_c'], chains_named['q'], chains_named['d_L']]).T
chains = np.reshape(chains, (-1, 3))
corner.corner(chains, labels = ["M_c", "q", "d_L"], truths = np.array([true_M_c, true_q, true_d_L]), **default_corner_kwargs)

plt.savefig("./figures/test_mass_transform_with_distance_before.png", bbox_inches = 'tight')
plt.close()

### Transformed space:
m1m2_named = mass_transform.forward(chains_named)
m1, m2 = m1m2_named['m_1'], m1m2_named['m_2']
chains = np.array([m1, m2]).T
chains = np.reshape(chains, (-1, 2))

true_m1, true_m2 = likelihood.true_m_1_source, likelihood.true_m_2_source
corner.corner(chains, labels = ["m_1", "m_2"], truths = np.array([true_m1, true_m2]), **default_corner_kwargs)

plt.savefig("./figures/test_mass_transform_with_distance_after.png", bbox_inches = 'tight')
plt.close()