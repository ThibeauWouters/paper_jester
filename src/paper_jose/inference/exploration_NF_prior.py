import numpy as np
import matplotlib.pyplot as plt
import corner

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

eos_samples_filename = "./outdir_prior/eos_samples.npz"
eos_samples = np.load(eos_samples_filename)

print("list(eos_samples.keys()")
print(list(eos_samples.keys()))

masses_EOS, radii_EOS, Lambdas_EOS = eos_samples["masses_EOS"], eos_samples["radii_EOS"], eos_samples["Lambdas_EOS"]

# Downsample data
jump = 1
masses_EOS = masses_EOS[::jump]
radii_EOS = radii_EOS[::jump]
Lambdas_EOS = Lambdas_EOS[::jump]

# Iterate over EOS and keep those that are fine
nb_samples = len(masses_EOS)
good_idx = np.ones(nb_samples, dtype=bool)

# TODO: This is a bit of a hack... 
for i in range(nb_samples):
    # First, sometimes the radius can be very large for low mass stars, which is unphysical
    bad_radii = (masses_EOS[i] > 1.0) * (radii_EOS[i] > 20.0)
    if any(bad_radii):
        good_idx[i] = False
        continue
    # Second, sometimes a negative Lambda was computed, remove that
    bad_Lambdas = (Lambdas_EOS[i] < 0.0)
    if any(bad_Lambdas):
        good_idx[i] = False
        continue
    # Finally, we want the TOV mass to be above 2.0 M_odot
    bad_MTOV = np.max(masses_EOS[i]) < 2.0
    if bad_MTOV:
        good_idx[i] = False
        continue
    
print("Number of good samples: ", np.sum(good_idx) / nb_samples)

masses_EOS = masses_EOS[good_idx]
radii_EOS = radii_EOS[good_idx]
Lambdas_EOS = Lambdas_EOS[good_idx]

print("np.shape(masses_EOS)")
print(np.shape(masses_EOS))
    
# Now sample some binaries from those EOS
N_samples = 40_000

m1_list = np.empty(N_samples)
m2_list = np.empty(N_samples)
Lambda1_list = np.empty(N_samples)
Lambda2_list = np.empty(N_samples)

mtov_list = []

for i in range(N_samples):
    # Randomly select an EOS
    idx = np.random.randint(0, len(masses_EOS))
    m, l = masses_EOS[idx], Lambdas_EOS[idx]
    mtov = np.max(m)
    
    mtov_list.append(mtov)
    
    # Use GW mass priors
    M_c_sample = np.random.uniform(1.18, 1.21)
    q_sample = np.random.uniform(0.5, 1.0)
    
    # Convert to m1, m2
    total_mass_sample = M_c_sample * (1 + q_sample) ** 1.2 / q_sample ** 0.6
    m1 = total_mass_sample / (1 + q_sample)
    m2 = m1 * q_sample

    # Use EOS to get Lambdas
    Lambda_1 = np.interp(m1, m, l)
    Lambda_2 = np.interp(m2, m, l)
    
    m1_list[i] = m1
    m2_list[i] = m2
    Lambda1_list[i] = Lambda_1
    Lambda2_list[i] = Lambda_2
    
print(f"m1 ranges from {np.min(m1_list)} to {np.max(m1_list)}")
print(f"m2 ranges from {np.min(m2_list)} to {np.max(m2_list)}")
    
range = [[np.min(m1_list), np.max(m1_list)], 
         [np.min(m2_list), np.max(m2_list)], 
         [0, 2000],
         [0, 6000]
         ]

print(f"Saving data")
np.savez("./NF/data/eos_prior_samples.npz", m1 = np.array(m1_list), m2 = np.array(m2_list), lambda_1 = np.array(Lambda1_list), lambda_2 = np.array(Lambda2_list))
print(f"Saving data DONE")

# Make a cornerplot of masses and Lambdas
data = np.array([m1_list, m2_list, Lambda1_list, Lambda2_list]).T
print("np.shape(data)")
print(np.shape(data))
figure = corner.corner(data, labels=[r"$m_1$ [$M_\odot$]", r"$m_2$ [$M_\odot$]", r"$\Lambda_1$", r"$\Lambda_2$"], range=range,**default_corner_kwargs)
plt.savefig("./figures/exploration_corner.png")
plt.close()