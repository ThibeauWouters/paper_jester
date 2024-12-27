import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import arviz

import joseTOV.utils as jose_utils

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

# NOTE: I am removing 8 here just because 8 and 10 are almost the same
# NB_CSE_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
NB_CSE_list = [8, 10, 30, 40, 50, 60]
bins = 50
violin_data = {"nb_cse": [], 
               "r14": [],
               "MTOV": [],
               "p3nsat": []
               }

for nb_cse in NB_CSE_list:
    try:
        filename = f"./outdir_{nb_cse}/eos_samples.npz"
        data = np.load(filename)
    except Exception as e:
        print(f"Issue with nb_cse = {nb_cse}. Error:")
        print(e)
        continue

    # Go over the MR samples, and for each (that is not broken), get the radius at 1.4 Msun
    all_m, all_r, all_n, all_p = data["masses_EOS"], data["radii_EOS"], data["n"], data["p"]
    
    all_n = all_n / jose_utils.fm_inv3_to_geometric / 0.16
    all_p = all_p / jose_utils.MeV_fm_inv3_to_geometric
    
    jump = 100
    
    all_m = all_m[::jump]
    all_r = all_r[::jump]
    all_n = all_n[::jump]
    all_p = all_p[::jump]
    
    bad_counter = 0
    for i in range(len(all_m)):
        _m, _r = all_m[i], all_r[i]
        _n, _p = all_n[i], all_p[i]
        bad_radii = (_m > 1.0) * (_r > 20.0)
        if any(bad_radii):
            bad_counter += 1
            continue
        
        bad_mtov = np.max(_m) < 1.4
        if bad_mtov:
            bad_counter += 1
            continue
        
        # Compute some quantities
        r14 = np.interp(1.4, _m, _r)
        mtov = np.max(_m)
        p3nsat = np.interp(3, _n, _p)
        
        violin_data["nb_cse"].append(nb_cse)
        violin_data["r14"].append(r14)
        violin_data["MTOV"].append(mtov)
        violin_data["p3nsat"].append(p3nsat)
        
    print(f"nb_cse = {nb_cse}, bad_counter = {bad_counter}")
    
      
# # Make the histogram
# plt.hist(r14_list, bins=bins, histtype="step", label=f"{nb_cse}", lw = 4, density=True)
    
df = pd.DataFrame(violin_data)
print("df")
print(df)

# TODO: combining into one plot is a bit messy
all_keys = ["r14", "MTOV", "p3nsat"]
all_labels = [r"$R_{1.4}$ [km]", r"$M_{\rm TOV}$ [M$_\odot$]", r"$p_{3n_{\rm{sat}}}$ [MeV fm$^{-3}$]"]
figsize = (12, 6)

for i, (key, label) in enumerate(zip(all_keys, all_labels)):
    plt.figure(figsize=figsize)
    
    print(f"Plotting the data for {key}")

    ax = sns.violinplot(data=violin_data, 
                        x=df["nb_cse"], 
                        y=df[key],
                        inner = None, 
                        cut = 0, 
                        split = True,
                        fill = False, 
                        bw_adjust = 0.5,
                        linewidth = 2)

    plt.xlabel("Number of CSE grid points")
    plt.ylabel(label)
    
    if key == "r14":
        plt.ylim(bottom = 10)
    elif key == "MTOV":
        plt.ylim(bottom = 1.9)
    elif key == "p3nsat":
        plt.ylim(top = 300)

    # Get current x-axis ticks and labels
    current_ticks = ax.get_xticks()
    current_labels = ax.get_xticklabels()

    # Define the custom shifted label positions
    shift_x = 0.4
    shifted_labels = [label.get_text() for label in current_labels]
    shifted_positions = [t + shift_x for t in current_ticks]

    # Set shifted ticks and labels in the final plot
    ax.set_xticks(shifted_positions)
    final_nb_cse_values = df["nb_cse"].unique().tolist()
    ax.set_xticklabels(final_nb_cse_values)

    # Add a second set of ticks at the top

    median_list = df.groupby("nb_cse")[key].mean().values
    lower_list = df.groupby("nb_cse")[key].quantile(0.025).values
    upper_list = df.groupby("nb_cse")[key].quantile(0.975).values

    ax_top = ax.secondary_xaxis("top")
    tick_labels = [
        rf"${med:.2f}_{{-{med - low:.2f}}}^{{+{high - med:.2f}}}$"
        for med, low, high in zip(median_list, lower_list, upper_list)
    ]
    ax_top.set_xticks(shifted_positions)
    ax_top.set_xticklabels(tick_labels, rotation=45)

    plt.savefig(f"./figures/violinplots_{key}.png", bbox_inches = "tight")
    plt.savefig(f"./figures/violinplots_{key}.pdf", bbox_inches = "tight")
    plt.close()