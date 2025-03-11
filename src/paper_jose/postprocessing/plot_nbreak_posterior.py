import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import arviz

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

# Load the posterior samples
posterior_filename = "../inference/outdir_all_new_prior_no_chiEFT/results_production.npz"
posterior = np.load(posterior_filename)
nbreak = np.array(posterior["nbreak"]).flatten() / 0.16

eps = 0.00
# eps = 0.05
jump = 1
nbreak = nbreak[::jump]

# Mask chains getting stuck at boundaries
mask = (nbreak > 1.0 + eps) & (nbreak < 4.0 - eps)
nbreak = nbreak[mask]

print(f"The posterior is downsampled with jump factor {jump} and has {len(nbreak)} samples now")

# Get some uniform samples to represent the prior
prior = np.random.uniform(1.0, 4.0, size = len(nbreak))

plt.figure()
hist_kwargs = {"density": True, 
               "linewidth": 2,
               "histtype": "step",
               }

# Get the bins
_, edges = np.histogram(nbreak, bins = 50, density = True)

med = np.median(nbreak)
hdi_prob = 0.90
low, high = arviz.hdi(nbreak, hdi_prob = hdi_prob)
low = med - low
high = high - med

print(f"\nThe {hdi_prob:.2f} for nbreak is {med:.2f}-{low:.2f}+{high:.2f}")

# Plot them
plt.hist(nbreak, bins = edges[1:-1], label = "Posterior", color = "blue", zorder = 100, **hist_kwargs)
plt.hist(prior, bins = edges[1:-1], label = "Prior", color = "gray", zorder = 100, **hist_kwargs)
plt.xlabel(r"$n_{\rm break}$ [$n_{\rm sat}$]")
plt.ylabel("Density")

eps = 0.1
plt.xlim(1.0 + eps, 4.0 - eps)
plt.legend()
print(f"Saving histogram plot")
plt.savefig("./figures/nbreak_posterior.pdf", bbox_inches = "tight")
plt.close()


# # Plot a KDE:
# plt.figure()
# kde_posterior = gaussian_kde(nbreak)
# kde_prior = gaussian_kde(prior)

# x = np.linspace(min_value, max_value, 1000)
# plt.plot(x, kde_posterior(x), label = "Posterior", color = "blue")
# plt.plot(x, kde_prior(x), label = "Prior", color = "gray")
# plt.xlabel(r"$n_{\rm break}$ [$n_{\rm sat}$]")
# plt.ylabel("Density")
# plt.legend()
# print(f"Saving KDE plot")
# plt.savefig("./figures/nbreak_posterior_kde.pdf", bbox_inches = "tight")
# plt.close()