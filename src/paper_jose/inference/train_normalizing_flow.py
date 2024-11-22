"""Train a normalizing flow on a given GW dataset and save the model to be used for inference later on"""

### Imports
import psutil
import os 
p = psutil.Process()
p.cpu_affinity([0])
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"

import matplotlib.pyplot as plt
import corner
import numpy as np
import copy

### Stuff for nice plots
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
                        levels=[0.68, 0.9, 0.997],
                        plot_density=False, 
                        plot_datapoints=False, 
                        fill_contours=False,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        density=True,
                        save=False)

import jax
import jax.numpy as jnp  # JAX NumPy
# jax.config.update("jax_enable_x64", True)
import equinox as eqx # Neural networks
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.distributions import Normal

print("GPU found?")
print(jax.devices())


###################
### AUXILIARIES ###
###################

def make_cornerplot(chains_1: np.array, 
                    chains_2: np.array,
                    range: list[float],
                    name: str = "./figures/corner_plot_data_NF.png"):
    """
    Plot a cornerplot of the true data samples and the NF samples
    Note: the shape use is a bit inconsistent below, watch out.
    """

    # The training data:
    corner_kwargs = copy.deepcopy(default_corner_kwargs)
    hist_1d_kwargs = {"density": True, "color": "blue"}
    corner_kwargs["color"] = "blue"
    corner_kwargs["hist_kwargs"] = hist_1d_kwargs
    fig = corner.corner(chains_1.T, range=range, labels = [r"$M_c$", r"$q$", r"$\Lambda_1$", r"$\Lambda_2$"], **corner_kwargs)

    # The data from the normalizing flow
    corner_kwargs["color"] = "red"
    hist_1d_kwargs = {"density": True, "color": "red"}
    corner_kwargs["hist_kwargs"] = hist_1d_kwargs
    corner.corner(chains_2, range=range, fig = fig, **corner_kwargs)

    # Make a textbox just because that makes the plot cooler
    fs = 32
    plt.text(0.75, 0.75, "Training data", fontsize = fs, color = "blue", transform = plt.gcf().transFigure)
    plt.text(0.75, 0.65, "Normalizing flow", fontsize = fs, color = "red", transform = plt.gcf().transFigure)

    plt.savefig(name, bbox_inches = "tight")
    plt.close()
    
def get_source_masses(M_c, q, d_L, H0 = 67.4, c = 2.998 * 10**5):
    M_c_source = M_c/(1 + (H0 * d_L)/c)
    m_1 = M_c_source * ((1 + q) ** (1/5))/((q) ** (3/5))
    m_2 = M_c_source * ((q) ** (2/5)) * ((1+q) ** (1/5))
    return m_1, m_2

################    
### PREAMBLE ###
################
    
paths_dict = {"injection": f"./GW170817/GW170817_injection.npz",
              "real": f"/home/thibeau.wouters/TurboPE-BNS/real_events_no_taper/GW170817_NRTidalv2/backup/results_production.npz"
              }

WHICH = "real"
if WHICH not in paths_dict.keys():
    raise ValueError(f"WHICH must be one of {paths_dict.keys()}s")

print(f"Training the NF for the {WHICH} data run . . .")

############
### BODY ###
############

# Load the data
H0 = 67.4 # km/(s Mpc)
c = 2.998 * 10**5 # in km/s to make sure dimensions are correct in hubbles law

if WHICH == "injection":
    # Downsample the data from 350 000 to 35 000
    jump = 10
    path = paths_dict[WHICH]
    data = np.load(path)
    M_c, q, lambda_1, lambda_2, d_L = data["M_c"].flatten(), data["q"].flatten(), data["lambda_1"].flatten(), data["lambda_2"].flatten(), data["d_L"].flatten()
    M_c = M_c[::jump]
    q = q[::jump]
    d_L = d_L[::jump]
    lambda_1 = lambda_1[::jump]
    lambda_2 = lambda_2[::jump]
    
    # Compute the component masses
    m_1, m_2 = get_source_masses(M_c, q, d_L)

    
    print("Loaded data from injection")
    data = np.array([m_1, m_2, lambda_1, lambda_2])
else:
    jump = 10
    path = paths_dict[WHICH]
    data = np.load(path)
    chains = data["chains"]
    # naming = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'd_L', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']
    
    M_c = chains[:, :, 0].flatten()
    q = chains[:, :, 1].flatten()
    lambda_1 = chains[:, :,  4].flatten()
    lambda_2 = chains[:, :, 5].flatten()
    d_L = chains[:, :, 6].flatten()
    
    M_c = M_c[::jump]
    q = q[::jump]
    d_L = d_L[::jump]
    lambda_1 = lambda_1[::jump]
    lambda_2 = lambda_2[::jump]
    
    # Compute the component masses
    m_1, m_2 = get_source_masses(M_c, q, d_L)
    print("Loaded data from real event analysis")
    data = np.array([m_1, m_2, lambda_1, lambda_2])


print(f"Loaded data with shape {np.shape(data)}")
n_dim, n_samples = np.shape(data)
print(f"ndim = {n_dim}, nsamples = {n_samples}")
data_np = np.array(data)

num_epochs = 500
N_samples_plot = 10_000
flow_key, train_key, sample_key = jax.random.split(jax.random.key(0), 3)

x = data.T # shape must be (n_samples, n_dim)
x = np.array(x)
print("np.shape(x)")
print(np.shape(x))

# Get range from the data for plotting
my_range = np.array([[np.min(x.T[i]), np.max(x.T[i])] for i in range(n_dim)])
widen_array = np.array([[-0.2, 0.2], [-0.2, 0.2], [-100, 100], [-20, 20]])
my_range += widen_array
print(f"The range is {my_range}")

flow = block_neural_autoregressive_flow(
    key=flow_key,
    base_dist=Normal(jnp.zeros(x.shape[1])),
    nn_depth=5,
    nn_block_dim=8
)

flow, losses = fit_to_data(
    key=train_key,
    dist=flow,
    x=x,
    learning_rate=5e-4,
    max_epochs=num_epochs,
    max_patience=50
    )

plt.plot(losses["train"], label = "Train", color = "red")
plt.plot(losses["val"], label = "Val", color = "blue")
plt.yscale("log")
plt.legend()
plt.savefig(f"./figures/GW_NF_training_losses_{WHICH}.png")
plt.close()

# And sample the distribution
nf_samples = flow.sample(sample_key, (N_samples_plot, ))
nf_samples_np = np.array(nf_samples)

make_cornerplot(data_np, nf_samples_np, range=my_range, name=f"./figures/GW_NF_corner_{WHICH}.png")

# Save the model
save_path = f"./GW170817/NF_model_{WHICH}.eqx"
eqx.tree_serialise_leaves(save_path, flow)

loaded_model = eqx.tree_deserialise_leaves(save_path, like=flow)

# And sample the distribution
nf_samples_loaded = loaded_model.sample(sample_key, (N_samples_plot, ))
nf_samples_loaded_np = np.array(nf_samples_loaded)

log_prob = loaded_model.log_prob(nf_samples_loaded)

make_cornerplot(data_np, nf_samples_loaded_np, range=my_range, name=f"./figures/GW_NF_corner_{WHICH}_reloaded.png")