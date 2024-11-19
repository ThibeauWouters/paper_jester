"""
Plot the entire collection of random samples for inspection
"""
import numpy as np 
import os
import tqdm
import matplotlib.pyplot as plt

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

# # Improved corner kwargs
# default_corner_kwargs = dict(bins=40, 
#                         smooth=1., 
#                         show_titles=False,
#                         label_kwargs=dict(fontsize=16),
#                         title_kwargs=dict(fontsize=16), 
#                         color="blue",
#                         # quantiles=[],
#                         # levels=[0.9],
#                         plot_density=True, 
#                         plot_datapoints=False, 
#                         fill_contours=True,
#                         max_n_ticks=4, 
#                         min_n_ticks=3,
#                         truth_color = "red",
#                         save=False)

RANDOM_SAMPLES_DIR = "./random_samples/"
PLOT_KWARGS = {"color": "blue", 
               "linewidth": 2}
MAX_NB_EOS = 100

plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
all_files = os.listdir(RANDOM_SAMPLES_DIR)
for i, eos_file in tqdm.tqdm(enumerate(all_files)):
    full_path = os.path.join(RANDOM_SAMPLES_DIR, eos_file)
    data = np.load(full_path)
    
    m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
    
    plt.subplot(1, 2, 1)
    plt.plot(r, m, **PLOT_KWARGS)
    plt.subplot(1, 2, 2)
    plt.plot(m, l, **PLOT_KWARGS)
    
    if i > MAX_NB_EOS:
        print("Quitting, max number of EOS reached")
        break
    
# Finalize the plot
R_MIN, R_MAX = 9, 15
M_MIN, M_MAX = 0.75, 3
L_MIN, L_MAX = 1, 1e5
plt.subplot(1, 2, 1)
plt.xlabel(r"$R$ [km]")
plt.ylabel(r"$M$ [$M_{\odot}$]")
plt.xlim(R_MIN, R_MAX)
plt.ylim(M_MIN, M_MAX)

plt.subplot(1, 2, 2)
plt.xlabel(r"$M$ [$M_{\odot}$]")
plt.ylabel(r"$\Lambda$")
plt.yscale("log")
plt.xlim(M_MIN, M_MAX)
plt.ylim(L_MIN, L_MAX)

plt.savefig("./figures/MRL_random_samples.png")
plt.close()