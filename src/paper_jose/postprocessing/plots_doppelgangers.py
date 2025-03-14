import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os

import os
import copy
import arviz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as patches

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
print(jax.devices())

import jax.numpy as jnp
from jaxtyping import Array

import paper_jose.utils as utils
import joseTOV.utils as jose_utils
import paper_jose.inference.utils_plotting as utils_plotting
plt.rcParams.update(utils_plotting.mpl_params)
import seaborn as sns

legend_fontsize = 20
label_fontsize = 24
cbar_fontsize = 24
figsize = (8, 6)

params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.labelsize": 24,
        "figure.titlesize": 16
        }

plt.rcParams.update(params)

TARGET_KWARGS = {"color": "black",
                 "linestyle": "--",
                 "linewidth": 2,
                 "zorder": 1e10}

TARGET_SCATTER_KWARGS = {"color": "black",
                         "s": 10,
                         "marker": "x",
                         "zorder": 1e10}

def load_target(filename: str = "../doppelgangers/hauke_macroscopic.dat"):
    """Returns MRL of the target"""
    data = np.genfromtxt(filename, skip_header=1, delimiter=" ").T
    r_target, m_target, Lambdas_target = data[0], data[1], data[2]
    return m_target, r_target, Lambdas_target

def load_target_micro(filename: str = "../doppelgangers/my_target_microscopic.dat"):
    data = np.loadtxt(filename)
    n_target, e_target, p_target, cs2_target = data[:, 0] / 0.16, data[:, 1], data[:, 2], data[:, 3]
    
    return n_target, e_target, p_target, cs2_target

def get_n_TOV(n, p, p_c):
    """
    We find n_TOV by checking where we achieve the central pressure.

    Args:
        n (_type_): _description_
        p (_type_): _description_
        p_c (_type_): _description_
    """
    n_TOV = jnp.interp(p_c, p, n)
    return n_TOV

def plot_campaign_results(n_NEP: list[str],
                          target_filename = "../doppelgangers/my_target_macroscopic.dat",
                          plot_EOS: bool = True,
                          plot_NS: bool = True,
                          plot_EOS_errors: bool = True,
                          plot_NS_errors: bool = True,
                          plot_EOS_params: bool = True,
                          plot_correlations: bool = True):
    """
    Master function for making the plots about the second part of the paper
    """
    
    # From the number of varying NEP number, get the outdir
    outdir = f"../doppelgangers/campaign_results/{n_NEP}_NEPs/"
    print(f"Plotting the optimization campaigns results for outdir {outdir} . . .")
    
    # Load the target EOS and NS:
    m_target, r_target, l_target = load_target(target_filename)
    target_filename = target_filename.replace("macroscopic", "microscopic")
    n_target, e_target, p_target, cs2_target = load_target_micro(target_filename)
    
    # Initialize dictionary where we will save the results
    results = {"masses_EOS": [],
               "MTOV": [],
               "radii_EOS": [],
               "Lambdas_EOS": [],
               
               "pc_EOS": [],
               
               "R1.4": [],
               "Lambda1.4": [],
               
               "n": [],
               "p": [],
               "e": [],
               "cs2": [],
               
               "n_TOV": [],
               "p_TOV": [],
               
               "E_sym": [],
               "L_sym": [],
               "K_sym": [],
               "Q_sym": [],
               "Z_sym": [],
               
               "K_sat": [],
               "Q_sat": [],
               "Z_sat": []
               }
    
    NEP_keys = ["E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym", "K_sat", "Q_sat", "Z_sat"]
    
    # Gather the results: iterate over different optimization campaigns
    print(f"Looking at outdir {outdir}")
    subdirs = os.listdir(outdir)
    for subdir in subdirs:
        full_dir = os.path.join(outdir, subdir, "data")
        files = os.listdir(full_dir)
        
        # Get the final file
        all_files = [f for f in files if f.endswith(".npz") and "best" not in f]
        idx_list = [int(f.split(".")[0]) for f in all_files]
        last_idx = max(idx_list)

        # Get the results of the final step
        data = np.load(os.path.join(full_dir, f"{last_idx}.npz"))
        
        # Load the data
        masses_EOS = data["masses_EOS"]
        radii_EOS = data["radii_EOS"]
        Lambdas_EOS = data["Lambdas_EOS"]
        n = data["n"]
        p = data["p"]
        e = data["e"]
        cs2 = data["cs2"]
        
        n = n / jose_utils.fm_inv3_to_geometric / 0.16 # convert to nsat
        p = p / jose_utils.MeV_fm_inv3_to_geometric # convert to MeV/fm^3
        e = e / jose_utils.MeV_fm_inv3_to_geometric # convert to MeV/fm^3
        
        # Append -- this is sloppy but making it more succint is really annoying and not worth it (for now)
        results["masses_EOS"].append(masses_EOS)
        results["radii_EOS"].append(radii_EOS)
        results["Lambdas_EOS"].append(Lambdas_EOS)
        
        results["n"].append(n)
        results["p"].append(p)
        results["e"].append(e)
        results["cs2"].append(cs2)
        
        results["E_sym"].append(data["E_sym"])
        results["L_sym"].append(data["L_sym"])
        results["K_sym"].append(data["K_sym"])
        results["Q_sym"].append(data["Q_sym"])
        results["Z_sym"].append(data["Z_sym"])

        results["K_sat"].append(data["K_sat"])
        results["Q_sat"].append(data["Q_sat"])
        results["Z_sat"].append(data["Z_sat"])
        
        # Get n_TOV and p_TOV
        p_c_array = jnp.exp(data["logpc_EOS"]) / jose_utils.MeV_fm_inv3_to_geometric
        results["pc_EOS"].append(p_c_array)
        
        # Get it at TOV, so maximal, limit
        p_c = p_c_array[-1]
        n_TOV = float(get_n_TOV(n, p, p_c))
        p_TOV = float(np.interp(n_TOV, n, p))
        
        # Save it:
        results["n_TOV"].append(n_TOV)
        results["p_TOV"].append(p_TOV)
    
    # Convert all to numpy arrays
    for key in results.keys():
        results[key] = np.array(results[key])
        
    # Fetch the true params:
    TRUE_NEPS = utils.NEP_CONSTANTS_DICT
    
    # Report the density at 1.0 M_odot mass
    n_1M = []
    for i in range(len(results["masses_EOS"])):
        pc_1M = np.interp(1.0, results["masses_EOS"][i], results["pc_EOS"][i])
        n_1M_val = get_n_TOV(results["n"][i], results["p"][i], pc_1M) # not exactly n_TOV, but it works
        n_1M.append(n_1M_val)
        
    mean_, std_ = np.mean(n_1M), np.std(n_1M)
    
    print(f"Mean density at 1.0 M_odot: ({mean_:.2f} +/- {std_:.2f}) nsat")
    
    ### Also report the correlation coefficients pairwise between the NEP values:
    correlation_dict = {}
    n_vars = len(NEP_keys)
    for i, key1 in enumerate(NEP_keys):
        for j, key2 in enumerate(NEP_keys):
            if j > i:
                corr = np.corrcoef(results[key1], results[key2])[0, 1]
                # print(f"Correlation coefficient between {key1} and {key2}: {corr}")
                correlation_dict[f"{key1} {key2}"] = corr
                
    # Initialize an empty matrix
    correlation_matrix = np.full((n_vars, n_vars), np.nan)

    # Fill the lower-triangular part of the matrix with correlation values
    for key, corr_value in correlation_dict.items():
        var1, var2 = key.split(" ")[0], key.split(" ")[1]
        i, j = NEP_keys.index(var1), NEP_keys.index(var2)
        if i > j:
            correlation_matrix[i, j] = corr_value
        elif j > i:
            correlation_matrix[j, i] = corr_value

    # Check what the max range is for the colorbar
    abs_corr_matrix = abs(correlation_matrix.flatten())
    mask = np.isnan(abs_corr_matrix)
    abs_corr_matrix = abs_corr_matrix[~mask]
    max_range = np.max(abs_corr_matrix)

    # Plot using Seaborn
    if plot_correlations:
        print("Making the NEP correlations plot")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        NEP_labels = [r"$E_{\rm{sym}}$",
                      r"$L_{\rm{sym}}$",
                      r"$K_{\rm{sym}}$",
                      r"$Q_{\rm{sym}}$",
                      r"$Z_{\rm{sym}}$",
                      r"$K_{\rm{sat}}$",
                      r"$Q_{\rm{sat}}$",
                      r"$Z_{\rm{sat}}$"
                      ]
        
        NEP_labels_x, NEP_labels_y = copy.deepcopy(NEP_labels), copy.deepcopy(NEP_labels)
        NEP_labels_x[-1] = " "
        NEP_labels_y[0] = " "
        
        ax = sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={'label': 'Correlation coefficient', 
                      'shrink': 0.60,
                      'aspect': 10,
                      'pad': 0.0},
            xticklabels=NEP_labels_x,
            yticklabels=NEP_labels_y,
            mask=np.isnan(correlation_matrix),
            vmin=-max_range,
            vmax=max_range,
            square=True,
            cbar=False,
            annot_kws={"fontsize": 18}
        )
        
        # Ticks handling
        fs = 22
        ax.tick_params(axis='x', which='both', length=0, labelsize = fs)
        ax.tick_params(axis='y', which='both', length=0, labelsize = fs)

        plt.grid(False)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
        plt.margins(0,0)
        name = f"./figures/final_doppelgangers/{n_NEP}_NEPs_correlation_matrix.pdf"
        plt.savefig(name, bbox_inches = "tight", pad_inches=0.1)
        plt.close()
    
    ### Plots of EOS and NS
    alpha = 1.0
    if plot_EOS:
        print("Plotting the EOS")
        plt.subplots(nrows = 2, ncols = 2, figsize=(12, 10))
        
        # First, plot all the targets in each panel
        
        # e(n) 
        plt.subplot(2, 2, 1)
        plt.plot(n_target, e_target, **TARGET_KWARGS)
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"$e$ [MeV fm${}^{-3}$]")
        
        plt.subplot(2, 2, 2)
        plt.plot(n_target, p_target, **TARGET_KWARGS)
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"$p$ [MeV fm${}^{-3}$]")
        
        # cs2(n)
        plt.subplot(2, 2, 3)
        plt.plot(n_target, cs2_target, **TARGET_KWARGS)
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"$c_s^2$")
        
        plt.subplot(2, 2, 4)
        plt.plot(e_target, p_target, **TARGET_KWARGS)
        plt.xlabel(r"$e$ [MeV fm${}^{-3}$]")
        plt.ylabel(r"$p$ [MeV fm${}^{-3}$]")
        
        # Now loop over the recovered ones, cut off at nTOV, and plot them
        for i in range(len(results["n"])):
            n_TOV = results["n_TOV"][i]
            mask = results["n"][i] < n_TOV
            
            n = results["n"][i][mask]
            e = results["e"][i][mask]
            p = results["p"][i][mask]
            cs2 = results["cs2"][i][mask]
            
            plt.subplot(2, 2, 1)
            plt.plot(n, e, alpha=alpha)
        
            plt.subplot(2, 2, 2)
            plt.plot(n, p, alpha=alpha)
        
            plt.subplot(2, 2, 3)
            plt.plot(n, cs2, alpha=alpha)
        
            plt.subplot(2, 2, 4)
            plt.plot(e, p, alpha=alpha)
        
        plt.savefig(f"./figures/final_doppelgangers/{n_NEP}_NEPs_EOS.pdf", bbox_inches = "tight")
        plt.close()
        
    ### Plots of EOS and NS
    alpha = 0.1
    if plot_NS:
        print("Plotting the NS")
        plt.subplots(nrows = 1, ncols = 2, figsize=(12, 10))
        
        # First plot target in each panel
        
        # MR
        plt.subplot(1, 2, 1)
        plt.plot(r_target, m_target, **TARGET_KWARGS)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_\odot$]")
        
        # ML
        plt.subplot(1, 2, 2)
        plt.plot(m_target, l_target, **TARGET_KWARGS)
        plt.yscale("log")
        plt.xlabel(r"$M$ [$M_\odot$]")
        plt.ylabel(r"$\Lambda$")
        
        # Then the recovered ones
        for i in range(len(results["masses_EOS"])):
            plt.subplot(1, 2, 1)
            plt.plot(results["radii_EOS"][i], results["masses_EOS"][i])
            
            plt.subplot(1, 2, 2)
            plt.plot(results["masses_EOS"][i], results["Lambdas_EOS"][i])
        
        # Put a grey square below 1.0 M_odot:
        plt.subplot(1, 2, 1)
        plt.axhspan(0, 1.0, color="black", alpha=0.2)
        
        # Limit plot to terminate at lowest mass of target
        m_min = np.min(m_target)
        plt.ylim(bottom = m_min)
        
        plt.subplot(1, 2, 2)
        plt.axvspan(0, 1.0, color="black", alpha=0.2)
        m_min = np.min(m_target)
        plt.xlim(left = m_min)
        
        plt.savefig(f"./figures/final_doppelgangers/{n_NEP}_NEPs_NS.pdf", bbox_inches = "tight")
        plt.close()
        
    truth_kwargs = {"linewidth": 2,
                    "linestyle": "--",
                    "color": "black"}
        
    if plot_EOS_params:
        print(f"Plotting EOS params")
        
        plt.subplots(nrows = 2, ncols = 2, figsize=(14, 12))
        
        # Esym vs Lsym:
        plt.subplot(2, 2, 1)
        # plt.scatter(TRUE_NEPS["E_sym"], TRUE_NEPS["L_sym"], **TARGET_SCATTER_KWARGS)
        
        plt.axhline(TRUE_NEPS["L_sym"], **truth_kwargs)
        plt.axvline(TRUE_NEPS["E_sym"], **truth_kwargs)
        
        for i in range(len(results["E_sym"])):
            plt.scatter(results["E_sym"][i], results["L_sym"][i])
        plt.xlabel(r"$E_{\rm{sym}}$ [MeV]")
        plt.ylabel(r"$L_{\rm{sym}}$ [MeV]")
        
        # Ksym vs Ksat:
        plt.subplot(2, 2, 2)
        # plt.scatter(TRUE_NEPS["K_sym"], TRUE_NEPS["K_sat"], **TARGET_SCATTER_KWARGS)
        plt.axhline(TRUE_NEPS["K_sat"], **truth_kwargs)
        plt.axvline(TRUE_NEPS["K_sym"], **truth_kwargs)
        
        for i in range(len(results["E_sym"])):
            plt.scatter(results["K_sym"][i], results["K_sat"][i])
        plt.xlabel(r"$K_{\rm{sym}}$ [MeV]")
        plt.ylabel(r"$K_{\rm{sat}}$ [MeV]")
        
        # Qsym vs Qsat:
        plt.subplot(2, 2, 3)
        # plt.scatter(TRUE_NEPS["Q_sym"], TRUE_NEPS["Q_sat"], **TARGET_SCATTER_KWARGS)
        plt.axhline(TRUE_NEPS["Q_sat"], **truth_kwargs)
        plt.axvline(TRUE_NEPS["Q_sym"], **truth_kwargs)
        
        for i in range(len(results["E_sym"])):
            plt.scatter(results["Q_sym"][i], results["Q_sat"][i])
        plt.xlabel(r"$Q_{\rm{sym}}$ [MeV]")
        plt.ylabel(r"$Q_{\rm{sat}}$ [MeV]")
        
        # Zsym vs Zsat:
        plt.subplot(2, 2, 4)
        # plt.scatter(TRUE_NEPS["Z_sym"], TRUE_NEPS["Z_sat"], **TARGET_SCATTER_KWARGS)
        plt.axhline(TRUE_NEPS["Z_sat"], **truth_kwargs)
        plt.axvline(TRUE_NEPS["Z_sym"], **truth_kwargs)
        
        for i in range(len(results["E_sym"])):
            plt.scatter(results["Z_sym"][i], results["Z_sat"][i])
        plt.xlabel(r"$Z_{\rm{sym}}$ [MeV]")
        plt.ylabel(r"$Z_{\rm{sat}}$ [MeV]")
        
        plt.savefig(f"./figures/final_doppelgangers/{n_NEP}_NEPs_EOS_params.pdf", bbox_inches = "tight")
        plt.close()
    
    # TODO: implement this, if still desired?
    # if plot_EOS_errors:
    #     print("Plotting the EOS errors")
        
    # if plot_NS_errors:
    #     print("Plotting the NS errors")
    
    
def make_money_plot(target_filename: str):
    
    print("Making the money plot")
    
    all_numbers_NEP = [2, 4, 6, 8]
    all_NEP_names = ["E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym", "K_sat", "Q_sat", "Z_sat"]
    TRUE_LSYM = 70.0
    
    ### `all_results` will store for each NEP the values of the NEP parameters to make the violinplot
    all_results = {}
    for nb_NEP in all_numbers_NEP:
        campaign_outdir = f"../doppelgangers/campaign_results/{nb_NEP}_NEPs/"
        
        # Initialize a new dictionary for this number of varying NEPs, and initialize empty lists for each NEP
        all_results[nb_NEP] = {k: [] for k in all_NEP_names}
        
        # Iterate over all subdirs and fetch all of its NEP parameters
        for subdir in os.listdir(campaign_outdir):
            full_dir = os.path.join(campaign_outdir, subdir, "data")
            files = os.listdir(full_dir)
            try:
                # Get the final file
                all_files = [f for f in files if f.endswith(".npz") and "best" not in f]
                idx_list = [int(f.split(".")[0]) for f in all_files]
                last_idx = max(idx_list)
        
                # Get the results of the final step
                data = np.load(os.path.join(full_dir, f"{last_idx}.npz"))
                
                # Load the NEPs
                for NEP in all_NEP_names:
                    all_results[nb_NEP][NEP].append(float(data[NEP]))
            except Exception as e:
                print(f"Error in subdir {subdir}: {e}")
              
    ### For the second part of the plot, focus on the results of the 8 NEPs and get EOS and NS results
    outdir = f"../doppelgangers/campaign_results/8_NEPs/"
    
    # Load the target EOS and NS:
    m_target, r_target, l_target = load_target(target_filename)
    target_filename = target_filename.replace("macroscopic", "microscopic")
    n_target, e_target, p_target, cs2_target = load_target_micro(target_filename)
    
    # Initialize dictionary where we will save the results
    final_results = {"masses_EOS": [],
                     "MTOV": [],
                     "radii_EOS": [],
                     "Lambdas_EOS": [],
                     
                     "pc_EOS": [],
                     "n": [],
                     "p": [],
                     "e": [],
                     "cs2": [],
                     
                     "n_TOV": [],
                     "p_TOV": [],
               }
    
    # Gather the EOS and NS data: iterate over different optimization campaigns
    subdirs = os.listdir(outdir)
    for subdir in subdirs:
        full_dir = os.path.join(outdir, subdir, "data")
        files = os.listdir(full_dir)
        
        # Get the final file
        all_files = [f for f in files if f.endswith(".npz") and "best" not in f]
        idx_list = [int(f.split(".")[0]) for f in all_files]
        last_idx = max(idx_list)

        # Get the results of the final step
        data = np.load(os.path.join(full_dir, f"{last_idx}.npz"))
        
        # Load the data
        masses_EOS = data["masses_EOS"]
        radii_EOS = data["radii_EOS"]
        Lambdas_EOS = data["Lambdas_EOS"]
        n = data["n"]
        p = data["p"]
        e = data["e"]
        cs2 = data["cs2"]
        
        n = n / jose_utils.fm_inv3_to_geometric / 0.16 # convert to nsat
        p = p / jose_utils.MeV_fm_inv3_to_geometric # convert to MeV/fm^3
        e = e / jose_utils.MeV_fm_inv3_to_geometric # convert to MeV/fm^3
        
        # Append -- this is sloppy but making it more succint is really annoying and not worth it (for now)
        final_results["masses_EOS"].append(masses_EOS)
        final_results["radii_EOS"].append(radii_EOS)
        final_results["Lambdas_EOS"].append(Lambdas_EOS)
        
        final_results["n"].append(n)
        final_results["p"].append(p)
        final_results["e"].append(e)
        final_results["cs2"].append(cs2)
        
        # Get n_TOV and p_TOV
        p_c_array = jnp.exp(data["logpc_EOS"]) / jose_utils.MeV_fm_inv3_to_geometric
        final_results["pc_EOS"].append(p_c_array)
        
        # Get it at TOV, so maximal, limit
        p_c = p_c_array[-1]
        n_TOV = float(get_n_TOV(n, p, p_c))
        p_TOV = float(np.interp(n_TOV, n, p))
        
        # Save it:
        final_results["n_TOV"].append(n_TOV)
        final_results["p_TOV"].append(p_TOV)
    
    # Convert all to numpy arrays
    for key in final_results.keys():
        final_results[key] = np.array(final_results[key])
        
    # Fetch the true params:
    TRUE_NEPS = utils.NEP_CONSTANTS_DICT  

    ### START PLOTTING ###
    
    fig = plt.figure(figsize=(14, 8))  # Adjust the figure size as needed

    # Create a grid layout (2 rows, 3 columns) with gridspec, divide into desired plots
    gs = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3)

    ax_left = plt.subplot(gs[:, 0])

    ax1 = plt.subplot(gs[0, 1])  # Top-left
    ax2 = plt.subplot(gs[0, 2])  # Top-right
    ax3 = plt.subplot(gs[1, 1])  # Bottom-left
    ax4 = plt.subplot(gs[1, 2])  # Bottom-right

    data = [np.array(all_results[nb_NEP]["L_sym"]) for nb_NEP in all_numbers_NEP]
    data = [d[d < 200] for d in data]
    ax_left.violinplot(data, all_numbers_NEP, showmeans=False, showmedians=True)
    
    xlabels = [r"$E_{\rm{sym}},$"  + "\n" + r"$L_{\rm{sym}}$", 
               r"$+K_{\rm{sym}},$" + "\n" + r"$\phantom{+}K_{\rm{sat}}$", 
               r"$+Q_{\rm{sym}},$" + "\n" + r"$\phantom{+}Q_{\rm{sat}}$", 
               r"$+Z_{\rm{sym}},$" + "\n" + r"$\phantom{+}Z_{\rm{sat}}$", ]
    ax_left.set_xticks(all_numbers_NEP, labels=xlabels, rotation=0)
    
    # Plot the true Lsym line for comparison
    ax_left.axhline(y=TRUE_LSYM, color="black", linestyle="-", label="Target", alpha = 0.5)
    ax_left.legend(loc="upper left")
    ax_left.grid(False)
    # ax_left.set_xlabel("Varying nuclear empirical parameters")
    ax_left.set_ylabel(r"$L_{\rm{sym}}$ [MeV]")
    
    # Define the rectangle in the left panel (zoomed-in region)
    rect_delta_x = 0.5
    rect_x, rect_y = 8 - rect_delta_x, 30
    rect_width, rect_height = 2 * rect_delta_x, 170

    # Draw rectangle in left panel
    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, 
                            linewidth=2, edgecolor='red', facecolor='none')
    ax_left.add_patch(rect)
    
    # Draw connecting lines (correct coordinate mapping)
    fig.canvas.draw()  # Ensures correct layout calculation

    # Convert rectangle corners to figure coordinates
    corner1 = ax_left.transData.transform((rect_x + rect_width, rect_y))  # Bottom-right
    corner2 = ax_left.transData.transform((rect_x + rect_width, rect_y + rect_height))  # Top-right

    # Get figure coordinates of the right bounding box (use ax3 and ax2 as references)
    bbox_x1, bbox_y1 = ax3.get_position().x0, ax3.get_position().y0
    bbox_x2, bbox_y2 = ax2.get_position().x1, ax2.get_position().y1

    # Convert to figure coordinates
    corner1_fig = fig.transFigure.inverted().transform(corner1)
    corner2_fig = fig.transFigure.inverted().transform(corner2)

    # Draw connecting lines in figure space
    line1 = plt.Line2D([corner1_fig[0], bbox_x1], [corner1_fig[1], bbox_y1], transform=fig.transFigure, color="red", linestyle="--", linewidth=1.5)
    line2 = plt.Line2D([corner2_fig[0], bbox_x2], [corner2_fig[1], bbox_y2], transform=fig.transFigure, color="red", linestyle="--", linewidth=1.5)

    fig.lines.extend([line1, line2])
    
    # Finally make the plots in the other panels
    # TODO:
    
    plt.savefig("./figures/money_plots/money_plot.pdf", bbox_inches = "tight")
    plt.close()
    
    
def main():
    """Plots for single runs, which are shown at the start of the section"""
    
    # ### These are with the JESTER-generated target EOS
    target_filename="../doppelgangers/my_target_macroscopic.dat"

    ### These directories are from before January 2025, using the previous definition of the run problem
    # outdirs_list = ["../doppelgangers/campaign_results/Lambdas/04_12_2024_doppelgangers/",
    #                 "../doppelgangers/campaign_results/radii/04_12_2024_doppelgangers/"]
    
    # ### These are after receiving Ingo's comments.
    # N_NEP_LIST = [2]
    # N_NEP_LIST = [2, 4, 6, 8]
    # for n in N_NEP_LIST:
    #     plot_campaign_results(n, target_filename=target_filename)
    # plot_campaign_results("E_sym_fixed", target_filename=target_filename)
    
    # ### Make the final money plot
    make_money_plot(target_filename)
    print("DONE")

if __name__ == "__main__":
    main()