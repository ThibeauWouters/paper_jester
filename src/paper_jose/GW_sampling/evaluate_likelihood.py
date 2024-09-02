import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"

import jax.numpy as jnp
import jax

from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.single_event.waveform import RippleTaylorF2
from jimgw.prior import UniformPrior, CosinePrior, SinePrior, CombinePrior
from jimgw.single_event.transforms import MassRatioToSymmetricMassRatioTransform
from jimgw.base import LikelihoodBase

from flowMC.strategy.optimization import optimization_Adam

import paper_jose.utils as utils

import time
import pickle
import numpy as np
jax.config.update("jax_enable_x64", True)
import shutil
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import optax 
import sys
sys.path.append("../")
# import utils_plotting as utils
print(jax.devices())

################
### PREAMBLE ###
################

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
                        save=False)

params = {
    "axes.labelsize": 30,
    "axes.titlesize": 30,
    "text.usetex": True,
    "font.family": "serif",
}
plt.rcParams.update(params)

labels = [r'$M_c/M_\odot$', r'$q$', r'$\chi_1$', r'$\chi_2$', r'$\Lambda$', r'$\delta\Lambda$', r'$d_{\rm{L}}/{\rm Mpc}$',
               r'$t_c$', r'$\phi_c$', r'$\iota$', r'$\psi$', r'$\alpha$', r'$\delta$']
naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

data_path = "/home/thibeau.wouters/gw-datasets/GW170817/" # on CIT

start_runtime = time.time()

############
### BODY ###
############

### Data definitions

total_time_start = time.time()
gps = 1187008882.43
trigger_time = gps
fmin = 20.0
fmax = 2048.0
minimum_frequency = fmin
maximum_frequency = fmax
T = 128.0
duration = T
post_trigger_duration = 2.0
epoch = duration - post_trigger_duration
f_ref = fmin 
tukey_alpha = 2 / (T / 2)

### Getting detector data

# This is our preprocessed data obtained from the TXT files at the GWOSC website (the GWF gave me NaNs?)
H1.frequencies = np.genfromtxt(f'{data_path}H1_freq.txt')
H1_data_re, H1_data_im = np.genfromtxt(f'{data_path}H1_data_re.txt'), np.genfromtxt(f'{data_path}H1_data_im.txt')
H1.data = H1_data_re + 1j * H1_data_im

L1.frequencies = np.genfromtxt(f'{data_path}L1_freq.txt')
L1_data_re, L1_data_im = np.genfromtxt(f'{data_path}L1_data_re.txt'), np.genfromtxt(f'{data_path}L1_data_im.txt')
L1.data = L1_data_re + 1j * L1_data_im

V1.frequencies = np.genfromtxt(f'{data_path}V1_freq.txt')
V1_data_re, V1_data_im = np.genfromtxt(f'{data_path}V1_data_re.txt'), np.genfromtxt(f'{data_path}V1_data_im.txt')
V1.data = V1_data_re + 1j * V1_data_im

# Load the PSD

H1.psd = H1.load_psd(jnp.array(H1.frequencies), psd_file = data_path + "GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_H1_psd.txt")
L1.psd = L1.load_psd(jnp.array(L1.frequencies), psd_file = data_path + "GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_L1_psd.txt")
V1.psd = V1.load_psd(jnp.array(V1.frequencies), psd_file = data_path + "GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_V1_psd.txt")

# ### Define priors

# # Internal parameters
# Mc_prior = UniformPrior(1.18, 1.21, parameter_names=["M_c"])
# q_prior = UniformPrior(0.125, 1.0, parameter_names=["q"])
# s1z_prior = UniformPrior(-0.05, 0.05, parameter_names=["s1_z"])
# s2z_prior = UniformPrior(-0.05, 0.05, parameter_names=["s2_z"])
# lambda_1_prior = UniformPrior(0.0, 5000.0, parameter_names=["lambda_1"])
# lambda_2_prior = UniformPrior(0.0, 5000.0, parameter_names=["lambda_2"])
# dL_prior       = UniformPrior(1.0, 75.0, parameter_names=["d_L"])
# # dL_prior       = PowerLaw(1.0, 75.0, 2.0, parameter_names=["d_L"])
# t_c_prior      = UniformPrior(-0.1, 0.1, parameter_names=["t_c"])
# phase_c_prior  = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
# cos_iota_prior = CosinePrior(parameter_names=["iota"])
# psi_prior     = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
# ra_prior      = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
# sin_dec_prior = SinePrior(parameter_names=["dec"])

# prior_list = [
#         Mc_prior,
#         q_prior,
#         s1z_prior,
#         s2z_prior,
#         lambda_1_prior,
#         lambda_2_prior,
#         dL_prior,
#         t_c_prior,
#         phase_c_prior,
#         cos_iota_prior,
#         psi_prior,
#         ra_prior,
#         sin_dec_prior,
#     ]

# prior = CombinePrior(prior_list)

# # The following only works if every prior has xmin and xmax property, which is OK for Uniform and Powerlaw
# # bounds = jnp.array([[p.xmin, p.xmax] for p in prior.base_prior])
# bounds = []
# for p in prior.base_prior:
#     if isinstance(p, UniformPrior):
#         bounds.append([p.xmin, p.xmax])
#     else:
#         # This is sine or cosine
#         bounds.append([-1.0, 1.0])
# bounds = jnp.array(bounds)

bounds = None

### Create likelihood object

ref_params = {
    'M_c': 1.19793583,
    'eta': 0.24794374,
    's1_z': 0.00220637,
    's2_z': 0.0495,
    'lambda_1': 105.12916663,
    'lambda_2': 5.0,
    'd_L': 45.41592353,
    't_c': 0.00220588,
    'phase_c': 5.76822606,
    'iota': 2.46158044,
    'psi': 2.09118099,
    'ra': 5.03335133,
    'dec': 0.01679998
}

n_bins = 500
gw_likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], 
                                                    bounds=bounds, 
                                                    waveform=RippleTaylorF2(f_ref=f_ref), 
                                                    trigger_time=gps, 
                                                    duration=T, 
                                                    n_bins=n_bins, 
                                                    ref_params=ref_params)

# try:
#     print("Trying to unpickle the GW likelihood")
#     with open("gw_likelihood.pkl", "rb") as f:
#         gw_likelihood = pickle.load(f)

# except Exception as e:
#     print(f"Error while unpickling: {e}")
    
#     print("initializing from scratch")
    
#     gw_likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], 
#                                                     bounds=bounds, 
#                                                     waveform=RippleTaylorF2(f_ref=f_ref), 
#                                                     trigger_time=gps, 
#                                                     duration=T, 
#                                                     n_bins=n_bins, 
#                                                     ref_params=ref_params)

#     # Save with pickle: 
#     with open("gw_likelihood.pkl", "wb") as f:
#         pickle.dump(gw_likelihood, f)
    

### Likelihood

# class GWLikelihood(LikelihoodBase):
    
#     def __init(self, 
#                heterodyned_likelihood: HeterodynedTransientLikelihoodFD):
        
#         self.heterodyned_likelihood = heterodyned_likelihood
        
#     def evaluate():
        
    
### Load and preprocess the data

data_file = "./repro_GW170817_TaylorF2/old_outdir/results_production.npz"

data = np.load(data_file)
chains = data["chains"]
chains = np.reshape(chains, (chains.shape[0] * chains.shape[1], chains.shape[2]))
naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

named_chains = {n: chains[:, i] for i, n in enumerate(naming)}

# Drop lambdas
named_chains = {k: v for k, v in named_chains.items() if k not in ["lambda_1", "lambda_2"]}

# Convert cos_iota and sin_dec:
named_chains["iota"] = np.arccos(named_chains["cos_iota"])
named_chains["dec"] = np.arcsin(named_chains["sin_dec"])
named_chains.pop("cos_iota")
named_chains.pop("sin_dec")

# q to eta:
named_chains["eta"] = named_chains["q"] / (1 + named_chains["q"])**2
named_chains.pop("q")


### Transforms
eos_transform = utils.MicroToMacroTransform(utils.name_mapping,
                                           keep_names = "all",
                                           nmax_nsat = utils.NMAX_NSAT,
                                           nb_CSE = utils.NB_CSE
                                           )

name_mapping = (["M_c", "q", "d_L", "masses_EOS", "Lambdas_EOS"], ["lambda_1", "lambda_2"])
lambda_transform = utils.ChirpMassMassRatioToLambdas(name_mapping, eos_transform)

### Sample

N = 100
jax_key = jax.random.PRNGKey(0)
for i in range(N):
    
    jax_key, jax_subkey = jax.random.split(jax_key)
    eos_params = utils.prior.sample(jax_subkey, 1)
    
    print("eos_params")
    print(eos_params)
    
    NS_params = eos_transform.forward(eos_params)
    
    print("NS_params")
    print(NS_params)
    
    # GEt some random GW params
    idx = np.random.choice(np.arange(len(chains)))
    gw_params = {k: v[idx] for k, v in named_chains.items()}
    
    print("gw_params")
    print(gw_params)
    
    gw_params.update(NS_params)
    
    # Get the lambdas:
    params = lambda_transform.forward(gw_params)
    
    log_L = gw_likelihood.evaluate(params)
    
    break