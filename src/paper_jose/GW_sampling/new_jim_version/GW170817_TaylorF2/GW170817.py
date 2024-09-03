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

from flowMC.strategy.optimization import optimization_Adam


import time
import numpy as np
jax.config.update("jax_enable_x64", True)
import shutil
import numpy as np
import matplotlib.pyplot as plt
import optax 
import sys
sys.path.append("../")
import utils_plotting as utils
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

### Define priors

# Internal parameters
Mc_prior = UniformPrior(1.18, 1.21, parameter_names=["M_c"])
q_prior = UniformPrior(0.125, 1.0, parameter_names=["q"])
s1z_prior = UniformPrior(-0.05, 0.05, parameter_names=["s1_z"])
s2z_prior = UniformPrior(-0.05, 0.05, parameter_names=["s2_z"])
lambda_1_prior = UniformPrior(0.0, 5000.0, parameter_names=["lambda_1"])
lambda_2_prior = UniformPrior(0.0, 5000.0, parameter_names=["lambda_2"])
dL_prior       = UniformPrior(1.0, 75.0, parameter_names=["d_L"])
# dL_prior       = PowerLaw(1.0, 75.0, 2.0, parameter_names=["d_L"])
t_c_prior      = UniformPrior(-0.1, 0.1, parameter_names=["t_c"])
phase_c_prior  = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
cos_iota_prior = CosinePrior(parameter_names=["iota"])
psi_prior     = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior      = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
sin_dec_prior = SinePrior(parameter_names=["dec"])

prior_list = [
        Mc_prior,
        q_prior,
        s1z_prior,
        s2z_prior,
        lambda_1_prior,
        lambda_2_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        cos_iota_prior,
        psi_prior,
        ra_prior,
        sin_dec_prior,
    ]

prior = CombinePrior(prior_list)

### Transforms
likelihoods_transforms = [MassRatioToSymmetricMassRatioTransform]

# The following only works if every prior has xmin and xmax property, which is OK for Uniform and Powerlaw
# bounds = jnp.array([[p.xmin, p.xmax] for p in prior.base_prior])
bounds = []
for p in prior.base_prior:
    if isinstance(p, UniformPrior):
        bounds.append([p.xmin, p.xmax])
    else:
        # This is sine or cosine
        bounds.append([-1.0, 1.0])
bounds = jnp.array(bounds)

### Create likelihood object

# ref_params = {'M_c': 1.192426544510255, 's1_z': -0.008553472800168603, 's2_z': -0.0466910085532903, 'lambda_1': 3288.662589969991, 'lambda_2': 395.1047965180274, 'd_L': 7.7103376753023705, 't_c': 0.05134335146489427, 'phase_c': 3.7334150846779135, 'iota': -1.5556060449504978, 'psi': 2.934190277243148, 'ra': 0.16349361052294487, 'dec': 0.910094453267318, 'eta': 0.15587318073433384}

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
    'ra': 3.41, # 5.03335133,
    'dec': -0.33 # 0.01679998
}

n_bins = 500

likelihood = HeterodynedTransientLikelihoodFD([H1, L1, V1], 
                                              prior=prior, 
                                              bounds=bounds, 
                                              waveform=RippleTaylorF2(f_ref=f_ref), 
                                              trigger_time=gps, 
                                              duration=T, 
                                              n_bins=n_bins, 
                                              ref_params=ref_params,
                                              likelihood_transforms=likelihoods_transforms,
                                            #   fixing_parameters = {"ra": 3.41, "dec": -0.33}
                                            )
print("Running with n_bins  = ", n_bins)

# Local sampler args

eps = 1e-3
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

n_loop_training = 100
n_epochs = 100
total_epochs = n_epochs * n_loop_training
start = int(total_epochs / 10)
start_lr = 1e-4
end_lr = 1e-4
power = 4.0
schedule_fn = optax.polynomial_schedule(
    start_lr, end_lr, power, total_epochs-start, transition_begin=start)

scheduler_str = f"polynomial_schedule({start_lr}, {end_lr}, {power}, {total_epochs-start}, {start})"

# Create jim object

outdir_name = "./outdir/"
if not os.path.exists(outdir_name):
    os.makedirs(outdir_name)

# Adam_optimizer = optimization_Adam(n_steps=3000, learning_rate=0.01, noise_level=1)

jim = Jim(
    likelihood,
    prior,
    likelihood_transforms = likelihoods_transforms, 
    n_loop_training=n_loop_training,
    n_loop_production=5,
    n_local_steps=10,
    n_global_steps=1_000,
    n_chains=1_000,
    n_epochs=n_epochs,
    n_max_examples=30000,
    n_flow_samples=100000,
    momentum=0.9,
    batch_size=30000,
    learning_rate=schedule_fn,
    use_global=True,
    keep_quantile=0.0,
    train_thinning=1,
    output_thinning=30,    
    local_sampler_arg=local_sampler_arg,
    outdir_name=outdir_name,
    # strategies=[Adam_optimizer, "default"],
)

### Heavy computation begins
jim.sample(jax.random.PRNGKey(41))
### Heavy computation ends

# === Show results, save output ===

# Print a summary to screen:
jim.print_summary()
outdir = outdir_name

if not os.path.exists(outdir):
    os.makedirs(outdir)

# Save and plot the results of the run
#  - training phase

name = outdir + f'results_training.npz'
print(f"Saving samples to {name}")
state = jim.sampler.get_sampler_state(training=True)
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
state = jim.sampler.get_sampler_state(training=False)
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

# Plot the chains as corner plots
utils.plot_chains(chains, "chains_production", outdir, labels = prior.parameter_names, truths=None)

# Final steps

# Finally, copy over this script to the outdir for reproducibility
shutil.copy2(__file__, outdir + "copy_script.py")


print("Finished successfully")

end_runtime = time.time()
runtime = end_runtime - start_runtime
print(f"Time taken: {runtime} seconds ({(runtime)/60} minutes)")

print(f"Saving runtime")
with open(outdir + 'runtime.txt', 'w') as file:
    file.write(str(runtime))


print("DONE")