import numpy as np 
import matplotlib.pyplot as plt
params = {"axes.grid": True,
        "text.usetex" : False,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        # "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

data = np.load("eos_samples.npz")
all_keys = list(data.keys())
keys = [k for k in all_keys if "log_prob" not in k]
samples = {k: data[k] for k in keys}

log_prob = data["log_prob"]

# Get maximum likelihood
max_log_prob_idx = np.argmax(log_prob)
max_log_prob_sample = {k: samples[k][max_log_prob_idx] for k in keys}

# Save it
np.savez("max_log_prob.npz", **max_log_prob_sample)

### Test loading it again
data = np.load("max_log_prob.npz")

m_jester = max_log_prob_sample["masses_EOS"]
r_jester = max_log_prob_sample["radii_EOS"]
l_jester = max_log_prob_sample["Lambdas_EOS"]

mask = m_jester > 1.0
m_jester = m_jester[mask]
r_jester = r_jester[mask]
l_jester = l_jester[mask]
    
hauke_filename = "../../doppelgangers/hauke_macroscopic.dat"
data = np.genfromtxt(hauke_filename, skip_header=1, delimiter=" ").T
r_hauke, m_hauke, l_hauke = data[0], data[1], data[2]

mask = m_hauke > 1.0
m_hauke = m_hauke[mask]
r_hauke = r_hauke[mask]
l_hauke = l_hauke[mask]

plt.subplots(1, 2, figsize=(10, 5))
plt.subplot(121)
plt.plot(r_jester, m_jester, color = "blue", label="Jester")
plt.plot(r_hauke, m_hauke, label="Koehn+", color = "red")
plt.xlabel(r"Radius [km]")
plt.ylabel(r"Mass [$M_\odot$]")

plt.subplot(122)
plt.plot(m_jester, l_jester, color = "blue", label="Jester")
plt.plot(m_hauke, l_hauke, label="Koehn+", color = "red")
plt.xlabel(r"Mass [$M_\odot$]")
plt.ylabel(r"Lambda")
plt.yscale("log")

plt.legend()
plt.savefig("max_log_prob.png", bbox_inches = "tight")
plt.savefig("max_log_prob.pdf", bbox_inches = "tight"))
plt.close()