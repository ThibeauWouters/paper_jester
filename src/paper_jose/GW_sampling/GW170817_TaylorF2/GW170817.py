import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
import copy
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.single_event.waveform import RippleTaylorF2
from jimgw.prior import Uniform, PowerLaw, Composite 
from jimgw.base import LikelihoodBase

from joseTOV.eos import MetaModel_with_CSE_EOS_model, construct_family
# from joseTOV import utils
# import paper_jose.utils as paper_jose_utils

import jax.numpy as jnp
import jax
from jaxtyping import Float
import time
import numpy as np
jax.config.update("jax_enable_x64", True)
import shutil
import numpy as np
import matplotlib.pyplot as plt
import optax 
# import sys
# sys.path.append("../")
import paper_jose.GW_sampling.utils_plotting as utils
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

# Taken from emulators paper Ingo and Rahul
NEP_CONSTANTS_DICT = {
    "E_sym": 32,
    "L_sym": 50,
    "K_sym": 0,
    "Q_sym": 0,
    "Z_sym": 0,
    
    "E_sat": -16,
    "K_sat": 230,
    "Q_sat": 0,
    "Z_sat": 0,
    
    "nbreak": 1.5,
    
    "n_CSE_0": 3 * 0.16,
    "n_CSE_1": 4 * 0.16,
    "n_CSE_2": 5 * 0.16,
    "n_CSE_3": 6 * 0.16,
    "n_CSE_4": 7 * 0.16,
    "n_CSE_5": 8 * 0.16,
    "n_CSE_6": 9 * 0.16,
    "n_CSE_7": 10 * 0.16,
    
    "cs2_CSE_0": 0.5,
    "cs2_CSE_1": 0.7,
    "cs2_CSE_2": 0.5,
    "cs2_CSE_3": 0.4,
    "cs2_CSE_4": 0.8,
    "cs2_CSE_5": 0.6,
    "cs2_CSE_6": 0.9,
    "cs2_CSE_7": 0.8,
    "cs2_CSE_8": 0.9,
}

def detector_frame_M_c_q_to_source_frame_m_1_m_2(params: dict) -> dict:
    
    M_c, q, d_L = params['M_c'], params['q'], params['d_L']
    H0 = params.get('H0', 67.4) # (km/s) / Mpc
    c = params.get('c', 299_792.4580) # km / s
    
    # Calculate source frame chirp mass
    z = d_L * H0 * 1e3 / c
    M_c_source = M_c / (1.0 + z)

    # Get source frame mass_1 and mass_2
    M_source = M_c_source * (1.0 + q) ** 1.2 / q**0.6
    m_1_source = M_source / (1.0 + q)
    m_2_source = M_source * q / (1.0 + q)

    return {'m_1': m_1_source, 'm_2': m_2_source}

##################
### LIKELIHOOD ###
##################

class EOSLikelihood(LikelihoodBase):
    """Wrapper around HeterodynedTransientLikelihoodFD that uses sampled EOS for the NS tidal deformabilities"""
    
    def __init__(self,
                 sampled_param_names: list[str],
                 gw_likelihood: HeterodynedTransientLikelihoodFD,
                 # metamodel kwargs:
                 ndat_metamodel: int = 100,
                 # CSE kwargs
                 nmax_nsat: float = 25,
                 nb_CSE: int = 8,
                 # TOV kwargs
                 min_nsat_TOV: float = 1.0,
                 ndat_TOV: int = 100,
                 ndat_CSE: int = 100,
                 nb_masses: int = 100,
                ):
            
        # Save as attributes
        self.ndat_metamodel = ndat_metamodel
        self.nmax_nsat = nmax_nsat
        self.nmax = nmax_nsat * 0.16
        self.nb_CSE = nb_CSE
        self.min_nsat_TOV = min_nsat_TOV
        self.ndat_TOV = ndat_TOV
        self.ndat_CSE = ndat_CSE
        self.nb_masses = nb_masses
        
        # Create the EOS object
        eos = MetaModel_with_CSE_EOS_model(nmax_nsat=self.nmax_nsat,
                                           ndat_metamodel=self.ndat_metamodel,
                                           ndat_CSE=self.ndat_CSE,
                )
        self.eos = eos
        
        # Save the GW likelihood 
        self.gw_likelihood = gw_likelihood
        
        # Remove those NEPs from the fixed values that we sample over
        self.fixed_params = copy.deepcopy(NEP_CONSTANTS_DICT)
        for name in sampled_param_names:
            if name in list(self.fixed_params.keys()):
                self.fixed_params.pop(name)
            
        # Construct a lambda function for solving the TOV equations, fix the given parameters
        self.construct_family_lambda = lambda x: construct_family(x, ndat = self.ndat_TOV, min_nsat = self.min_nsat_TOV)
        
    def get_eos_and_ns(self, params: dict[str, Float]) -> dict[str, Float]:
        
        params.update(self.fixed_params)
        
        # Separate the MM and CSE parameters
        NEP = {key: value for key, value in params.items() if "_sat" in key or "_sym" in key}
        NEP["nbreak"] = params["nbreak"]
        
        ngrids = jnp.array([params[f"n_CSE_{i}"] for i in range(self.nb_CSE)])
        cs2grids = jnp.array([params[f"cs2_CSE_{i}"] for i in range(self.nb_CSE)])
        
        # Append the final cs2 value, which is fixed at nmax 
        ngrids = jnp.append(ngrids, jnp.array([self.nmax]))
        cs2grids = jnp.append(cs2grids, jnp.array([params[f"cs2_CSE_{self.nb_CSE}"]]))
        
        # Create the EOS, ignore mu and cs2 (final 2 outputs)
        ns, ps, hs, es, dloge_dlogps, _, _ = self.eos.construct_eos(NEP, ngrids, cs2grids)
        eos_tuple = (ns, ps, hs, es, dloge_dlogps)
        
        # Solve the TOV equations, save only mass and Lambdas
        _, masses_EOS, _, Lambdas_EOS = self.construct_family_lambda(eos_tuple)
        
        return_dict = {"masses_EOS": masses_EOS, "Lambdas_EOS": Lambdas_EOS}
        
        return return_dict
    
    def transform_masses(self, params: dict[str, Float]) -> Float:
        return detector_frame_M_c_q_to_source_frame_m_1_m_2(params)
    
    def evaluate(self, params: dict[str, Float], data: dict) -> Float:
        
        eos_dict = self.get_eos_and_ns(params)
        source_masses = self.transform_masses(params)
        
        m, l = eos_dict["masses_EOS"], eos_dict["Lambdas_EOS"]
        m_1, m_2 = source_masses["m_1"], source_masses["m_2"]
        
        # Also transform to get eta
        params["eta"] = params["q"] / (1 + params["q"]) ** 2
        
        # Get lambdas
        params["lambda_1"] = jnp.interp(m_1, m, l, right = -1.0)
        params["lambda_2"] = jnp.interp(m_2, m, l, right = -1.0)
        
        return jax.lax.cond((params["lambda_1"] < 0.0) | (params["lambda_2"] < 0.0),
                            lambda _: -jnp.inf,
                            lambda x: self.gw_likelihood.evaluate(x, None),
                            params)
        

start_runtime = time.time()

############
### BODY ###
############

### Data definitions

total_time_start = time.time()
gps = 1187008882.43
trigger_time = gps
fmin = 23
fmax = 2048
minimum_frequency = fmin
maximum_frequency = fmax
T = 128
duration = T
post_trigger_duration = 2
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

H1.psd = H1.load_psd(H1.frequencies, psd_file = data_path + "GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_H1_psd.txt")
L1.psd = L1.load_psd(L1.frequencies, psd_file = data_path + "GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_L1_psd.txt")
V1.psd = V1.load_psd(V1.frequencies, psd_file = data_path + "GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle_V1_psd.txt")

### Define priors

# Internal parameters
Mc_prior = Uniform(1.18, 1.21, naming=["M_c"])
q_prior = Uniform(0.125, 1.0, naming=["q"])
s1z_prior = Uniform(-0.05, 0.05, naming=["s1_z"])
s2z_prior = Uniform(-0.05, 0.05, naming=["s2_z"])
# lambda_1_prior = Uniform(0.0, 5000.0, naming=["lambda_1"])
# lambda_2_prior = Uniform(0.0, 5000.0, naming=["lambda_2"])
dL_prior       = Uniform(1.0, 75.0, naming=["d_L"])
# dL_prior       = PowerLaw(1.0, 75.0, 2.0, naming=["d_L"])
t_c_prior      = Uniform(-0.1, 0.1, naming=["t_c"])
phase_c_prior  = Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
cos_iota_prior = Uniform(
    -1.0,
    1.0,
    naming=["cos_iota"],
    transforms={
        "cos_iota": (
            "iota",
            lambda params: jnp.arccos(
                jnp.arcsin(jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        )
    },
)
psi_prior     = Uniform(0.0, jnp.pi, naming=["psi"])
ra_prior      = Uniform(0.0, 2 * jnp.pi, naming=["ra"])
sin_dec_prior = Uniform(
    -1.0,
    1.0,
    naming=["sin_dec"],
    transforms={
        "sin_dec": (
            "dec",
            lambda params: jnp.arcsin(
                jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        )
    },
)

# Define the GW prior

gw_prior_list = [
        Mc_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        # lambda_1_prior,
        # lambda_2_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
    ]

gw_prior = Composite(gw_prior_list)

# The following only works if every prior has xmin and xmax property, which is OK for Uniform and Powerlaw
bounds = jnp.array([[p.xmin, p.xmax] for p in gw_prior.priors])

### Create likelihood object

ref_params = {
    'M_c': 1.19793583,
    'eta': 0.24794374,
    's1_z': 0.00220637,
    's2_z': 0.05,
    'lambda_1': 105.12916663,
    'lambda_2': 5.0,
    'd_L': 45.41592353,
    't_c': 0.00220588,
    'phase_c': 5.76822606,
    'iota': 2.46158044,
    'psi': 2.09118099,
    'ra': 3.41,
    'dec': -0.32
}

n_bins = 100

gw_likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], 
                                                 bounds=bounds, 
                                                 waveform=RippleTaylorF2(f_ref=f_ref), 
                                                 prior=gw_prior,
                                                 trigger_time=gps, 
                                                 duration=T, 
                                                 n_bins=n_bins, 
                                                 ref_params=ref_params)
print("Running with n_bins  = ", n_bins)

#################
### EOS PRIOR ###
#################

my_nbreak = 2.0 * 0.16
NMAX_NSAT = 25
NMAX = NMAX_NSAT * 0.16
N = 100
NB_CSE = 8
width = (NMAX - my_nbreak) / (NB_CSE + 1)

### NEP priors
# K_sat_prior = Uniform(150.0, 300.0, naming=["K_sat"])
# Q_sat_prior = Uniform(-500.0, 1100.0, naming=["Q_sat"])
# Z_sat_prior = Uniform(-2500.0, 1500.0, naming=["Z_sat"])

E_sym_prior = Uniform(28.0, 45.0, naming=["E_sym"])
L_sym_prior = Uniform(10.0, 200.0, naming=["L_sym"])
# K_sym_prior = Uniform(-300.0, 100.0, naming=["K_sym"])
# Q_sym_prior = Uniform(-800.0, 800.0, naming=["Q_sym"])
# Z_sym_prior = Uniform(-2500.0, 1500.0, naming=["Z_sym"])

eos_prior_list = [
    E_sym_prior,
    L_sym_prior, 
    # K_sym_prior,
    # Q_sym_prior,
    # Z_sym_prior,

    # K_sat_prior,
    # Q_sat_prior,
    # Z_sat_prior,
]

### CSE priors
# prior_list.append(Uniform(1.0 * 0.16, 2.0 * 0.16, naming=[f"nbreak"]))
# for i in range(NB_CSE):
#     left = my_nbreak + i * width
#     right = my_nbreak + (i+1) * width
#     prior_list.append(Uniform(left, right, naming=[f"n_CSE_{i}"]))
#     prior_list.append(Uniform(0.0, 1.0, naming=[f"cs2_CSE_{i}"]))
# eos_prior_list.append(Uniform(0.0, 1.0, naming=[f"cs2_CSE_{NB_CSE}"]))

sampled_param_names = [prior.naming[0] for prior in eos_prior_list]

print("sampled_param_names for EOS prior:")
print(sampled_param_names)

######################
### COMPLETE PRIOR ###
######################

complete_prior_list = gw_prior_list + eos_prior_list
prior = Composite(complete_prior_list)

##################
### LIKELIHOOD ###
##################

likelihood = EOSLikelihood(sampled_param_names, gw_likelihood)

# Local sampler args

eps = 5e-6
mass_matrix = jnp.eye(prior.n_dim)
# mass_matrix = mass_matrix.at[0,0].set(1e-5)
# mass_matrix = mass_matrix.at[1,1].set(1e-4)
# mass_matrix = mass_matrix.at[2,2].set(1e-3)
# mass_matrix = mass_matrix.at[3,3].set(1e-3)
# mass_matrix = mass_matrix.at[7,7].set(1e-5)
# mass_matrix = mass_matrix.at[11,11].set(1e-2)
# mass_matrix = mass_matrix.at[12,12].set(1e-2)
local_sampler_arg = {"step_size": mass_matrix * eps}

# Build the learning rate scheduler

n_loop_training = 400
n_epochs = 100
total_epochs = n_epochs * n_loop_training
start = int(total_epochs / 10)
start_lr = 1e-3
end_lr = 1e-5
power = 4.0
schedule_fn = optax.polynomial_schedule(
    start_lr, end_lr, power, total_epochs-start, transition_begin=start)

scheduler_str = f"polynomial_schedule({start_lr}, {end_lr}, {power}, {total_epochs-start}, {start})"

# Create jim object

outdir_name = "./outdir/"

jim = Jim(
    likelihood,
    prior,
    n_loop_training=10,
    n_loop_production=5,
    n_local_steps=3,
    n_global_steps=50,
    n_chains=1_000,
    n_epochs=25,
    learning_rate=0.001,
    max_samples=50000,
    momentum=0.9,
    batch_size=50000,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=1,
    output_thinning=1,    
    local_sampler_arg=local_sampler_arg,
    stopping_criterion_global_acc = 1.0,
    outdir_name=outdir_name
)

### Heavy computation begins
jim.sample(jax.random.PRNGKey(41))
### Heavy computation ends

# === Show results, save output ===

# Print a summary to screen:
jim.print_summary()
outdir = outdir_name

# Save and plot the results of the run
#  - training phase

name = outdir + f'results_training.npz'
print(f"Saving samples to {name}")
state = jim.Sampler.get_sampler_state(training=True)
chains, log_prob, local_accs, global_accs, loss_vals = state["chains"], state[
    "log_prob"], state["local_accs"], state["global_accs"], state["loss_vals"]
local_accs = jnp.mean(local_accs, axis=0)
global_accs = jnp.mean(global_accs, axis=0)
np.savez(name, log_prob=log_prob, local_accs=local_accs,
            global_accs=global_accs, loss_vals=loss_vals)

utils.plot_accs(local_accs, "Local accs (training)",
                "local_accs_training", outdir)
utils.plot_accs(global_accs, "Global accs (training)",
                "global_accs_training", outdir)
utils.plot_loss_vals(loss_vals, "Loss", "loss_vals", outdir)
utils.plot_log_prob(log_prob, "Log probability (training)",
                    "log_prob_training", outdir)

#  - production phase
name = outdir + f'results_production.npz'
state = jim.Sampler.get_sampler_state(training=False)
chains, log_prob, local_accs, global_accs = state["chains"], state[
    "log_prob"], state["local_accs"], state["global_accs"]
local_accs = jnp.mean(local_accs, axis=0)
global_accs = jnp.mean(global_accs, axis=0)
np.savez(name, chains=chains, log_prob=log_prob,
            local_accs=local_accs, global_accs=global_accs)

utils.plot_accs(local_accs, "Local accs (production)",
                "local_accs_production", outdir)
utils.plot_accs(global_accs, "Global accs (production)",
                "global_accs_production", outdir)
utils.plot_log_prob(log_prob, "Log probability (production)",
                    "log_prob_production", outdir)

# Finally, copy over this script to the outdir for reproducibility
shutil.copy2(__file__, outdir + "copy_script.py")

# print("Saving the jim hyperparameters")
# # Change scheduler from function to a string representation
# try:
#     jim.hyperparameters["learning_rate"] = scheduler_str
#     jim.Sampler.hyperparameters["learning_rate"] = scheduler_str
#     jim.save_hyperparameters(outdir=outdir)
# except Exception as e:
#     # Sometimes, something breaks, so avoid crashing the whole thing
#     print(f"Could not save hyperparameters in script: {e}")

# Plot the chains as corner plots
try: 
    utils.plot_chains(chains, "chains_production", outdir, labels = list(prior.naming), truths=None)
except Exception as e:
    print(f"Could not plot chains: {e}")
    print(f"Moving on however")


print("Finished successfully")

end_runtime = time.time()
runtime = end_runtime - start_runtime
print(f"Time taken: {runtime} seconds ({(runtime)/60} minutes)")

print(f"Saving runtime")
with open(outdir + 'runtime.txt', 'w') as file:
    file.write(str(runtime))
