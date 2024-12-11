"""Quickly doing some postprocessing on the results of the inference."""

import numpy as np
import matplotlib.pyplot as plt
import corner
mpl_params = {"axes.grid": True,
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

plt.rcParams.update(mpl_params)

filename = "./outdir_prior/eos_samples.npz"
data = np.load(filename)
keys = list(data.keys())

print(f"The keys are: {keys}")

# samples = data["samples"]
# print(np.shape(samples))

m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]

nb_samples, nb_points = np.shape(m)
print(f"Number of samples: {nb_samples}")

# Plotting
samples_kwargs = {"color": "black", "alpha": 1}

fig, cax = plt.subplots(1, 2, figsize=(12, 6))

m_min, m_max = 1.0, 3.5
r_min, r_max = 8.0, 20.0
l_min, l_max = 0.0, 50_000.0

log_prob = data["log_prob"]
print("np.shape(log_prob)")
print(np.shape(log_prob))

# Get a colorbar for log prob, but normalized
norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
cmap = plt.get_cmap("viridis")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

for i in range(nb_samples):

    # Get color
    color = cmap(norm(log_prob[i]))
    samples_kwargs["color"] = color

    # Mass-radius plot
    plt.subplot(121)
    mask = (r[i] > r_min) * (r[i] < r_max) * (m[i] > m_min) * (m[i] < m_max) * (l[i] > l_min) * (l[i] < l_max)
    plt.plot(r[i][mask], m[i][mask], **samples_kwargs)
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_{\odot}$]")
    
    # Mass-Lambda plot
    plt.subplot(122)
    plt.plot(m[i][mask], l[i][mask], **samples_kwargs)
    plt.yscale("log")
    # plt.xlim(1.0, 3.5)
    # plt.ylim(0.0, 50_000)
    plt.xlabel(r"$M$ [$M_{\odot}$]")
    plt.ylabel(r"$\Lambda$")
    
# Save
sm.set_array([])
# plt.colorbar(sm, cax = cax, label=r"$\log P$")
plt.savefig("./figures/postprocessing.png", bbox_inches = "tight")
plt.savefig("./figures/postprocessing.pdf", bbox_inches = "tight")