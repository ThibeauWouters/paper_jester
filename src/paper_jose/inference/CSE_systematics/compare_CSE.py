import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

NB_CSE_list = [8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
bins = 50
violin_data = {"nb_cse": [], "r14": []}

for nb_cse in NB_CSE_list:
    try:
        filename = f"./outdir_{nb_cse}/eos_samples.npz"
        data = np.load(filename)
    except Exception as e:
        print(f"Issue with nb_cse = {nb_cse}. Error:")
        print(e)
        continue

    # Go over the MR samples, and for each (that is not broken), get the radius at 1.4 Msun
    all_m, all_r = data["masses_EOS"], data["radii_EOS"]
    jump = 10
    all_m = all_m[::jump]
    all_r = all_r[::jump]
    
    r14_list = []
    bad_counter = 0
    for i in range(len(all_m)):
        _m, _r = all_m[i], all_r[i]
        bad_radii = (_m > 1.0) * (_r > 20.0)
        if any(bad_radii):
            bad_counter += 1
            continue
        
        bad_mtov = np.max(_m) < 1.4
        if bad_mtov:
            bad_counter += 1
            continue
        
        r14 = np.interp(1.4, _m, _r)
        # r14_list.append(r14)
        violin_data["nb_cse"].append(nb_cse)
        violin_data["r14"].append(r14)
        
    print(f"nb_cse = {nb_cse}, bad_counter = {bad_counter}")
    
      
# # Make the histogram
# plt.hist(r14_list, bins=bins, histtype="step", label=f"{nb_cse}", lw = 4, density=True)
    
df = pd.DataFrame(violin_data)
print("df")
print(df)

sns.violinplot(data=violin_data, x=df["nb_cse"], y=df["r14"])

plt.xlabel("Number of CSE grid points")
plt.ylabel(r"$R_{1.4}$ [km]")
plt.savefig("./figures/r14_histogram.png", bbox_inches = "tight")
plt.savefig("./figures/r14_histogram.pdf", bbox_inches = "tight")
plt.close()