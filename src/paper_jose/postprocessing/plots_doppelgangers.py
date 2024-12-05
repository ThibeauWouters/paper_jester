import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter, LogLocator

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
        "axes.labelsize": 24,
        "figure.titlesize": 16
        }

plt.rcParams.update(params)

TARGET_KWARGS = {"color": "black",
                 "linewidth": 4,
                 "zorder": 1e10}

def load_target(filename: str = "../doppelgangers/hauke_macroscopic.dat"):
    """Returns MRL of the target"""
    data = np.genfromtxt(filename, skip_header=1, delimiter=" ").T
    r_target, m_target, Lambdas_target = data[0], data[1], data[2]
    return m_target, r_target, Lambdas_target

def plot_all_NS(main_dir: str = "../doppelgangers/real_doppelgangers/",
                m_min: float = 0.75,
                max_nb_steps: int = 100):
    
    all_dirs = os.listdir(main_dir)
    for dir in all_dirs:
        full_dir = os.path.join(main_dir, dir, "data")
        plot_NS(full_dir, max_nb_steps, save_name = f"./figures/doppelganger_dir_{dir}.png" ,m_min=m_min)
    

def plot_NS(dir: str,
            xticks_error_radii: list,
            yticks_error_Lambdas: list,
            target_filename: str = "../doppelgangers/hauke_macroscopic.dat",
            max_nb_steps: int = 999999,
            save_name: str = "./figures/doppelganger_trajectory.png",
            m_target: Array = None,
            r_target: Array = None,
            l_target: Array = None,
            m_min: float = 0.75,
            m_max: float = 2.3,
            nb_masses: int = 500,
            rasterized: bool = True):
    """
    Plot the doppelganger trajectory in the NS space.
    Args:
        m_min (float, optional): Minimum mass from which to compute errors and create the error plot. Defaults to 1.2.
    """
    
    if m_target is None:
        m_target, r_target, l_target = load_target(target_filename)
        
    if "hauke" in target_filename and "radius" in dir:
        mass_min, mass_max = 1.0, 2.5
        radius_min, radius_max = 11, 12.9
        lambda_min, lambda_max = 2, 5e3
    elif "hauke" in target_filename and "Lambdas" in dir:
        mass_min, mass_max = 1.0, 2.5
        radius_min, radius_max = 11, 13.4
        lambda_min, lambda_max = 2, 5e3
    else:
        mass_min, mass_max = 0.9, 3.0
        radius_min, radius_max = 11.75, 13.99
        lambda_min, lambda_max = 2, 5e3
        
    delta_r_min, delta_r_max = np.min(xticks_error_radii), np.max(xticks_error_radii)
    delta_l_min, delta_l_max = np.min(yticks_error_Lambdas), np.max(yticks_error_Lambdas)
        
    mask  = m_target > m_min
    m_target = m_target[mask]
    r_target = r_target[mask]
    l_target = l_target[mask]

    # Read the EOS data
    all_masses_EOS = []
    all_radii_EOS = []
    all_Lambdas_EOS = []
    
    # Get the files but first preprocess a bit
    _files = os.listdir(dir)
    _files = [f for f in _files if f.endswith(".npz") and "best" not in f]
    sort_idx = np.argsort([int(f.split(".")[0]) for f in _files])
    _files = [_files[i] for i in sort_idx]
    
    # Then pass for plotting
    files = [os.path.join(dir, f) for f in _files]
    
    for i, file in enumerate(files):
        if i>max_nb_steps:
            break
        data = np.load(file)
        
        masses_EOS = data["masses_EOS"]
        radii_EOS = data["radii_EOS"]
        Lambdas_EOS = data["Lambdas_EOS"]
        
        if not np.any(np.isnan(masses_EOS)) and not np.any(np.isnan(radii_EOS)) and not np.any(np.isnan(Lambdas_EOS)):
        
            all_masses_EOS.append(masses_EOS)
            all_radii_EOS.append(radii_EOS)
            all_Lambdas_EOS.append(Lambdas_EOS)
        
    # Get the errors:
    m_final, r_final, l_final = all_masses_EOS[-1], all_radii_EOS[-1], all_Lambdas_EOS[-1]
    mask = m_final > m_min
    m_final, r_final, l_final = m_final[mask], r_final[mask], l_final[mask]
    
    mass_array = np.linspace(m_min, m_max, nb_masses)
    
    r_target_interp = np.interp(mass_array, m_target, r_target)
    l_target_interp = np.interp(mass_array, m_target, l_target)
    
    r_final_interp = np.interp(mass_array, m_final, r_final)
    l_final_interp = np.interp(mass_array, m_final, l_final)
    
    errors_r = np.abs(r_target_interp - r_final_interp)
    errors_r = 1000 * errors_r
    errors_l = np.abs(l_target_interp - l_final_interp)
        
    # N might have become smaller than total predetermined number of runs if we hit NaNs at some point
    N_max = len(all_masses_EOS)
    norm = mpl.colors.Normalize(vmin=0, vmax=N_max)
    # cmap = sns.color_palette("rocket_r", as_cmap=True)
    # cmap = sns.color_palette("crest", as_cmap=True)
    cmap = sns.color_palette("flare", as_cmap=True)
        
    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1, 2], 'height_ratios': [1, 2]})
    
    ### Errors
    plt.subplot(2, 3, 5)
    plt.plot(errors_r, mass_array, color = "black", zorder=1e10)
    plt.xlabel(r"$|\Delta R|$ [m]")
    plt.xscale("log")
    plt.xlim(left = delta_r_min, right = delta_r_max)
    
    plt.subplot(2, 3, 3)
    plt.plot(mass_array, errors_l, color = "black", zorder=1e10)
    plt.ylabel(r"$|\Delta \Lambda|$")
    plt.yscale("log")
    plt.ylim(bottom = delta_l_min, top = delta_l_max)
    
    ### NS families
    plt.subplot(2, 3, 4)
    plt.plot(r_target, m_target, **TARGET_KWARGS)
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M \ [M_\odot]$")
    plt.xlim(left = radius_min, right = radius_max)
    plt.ylim(bottom = mass_min, top = mass_max)
    
    plt.subplot(2, 3, 6)
    plt.xlabel(r"$M \ [M_\odot]$")
    plt.ylabel(r"$\Lambda$")
    plt.plot(m_target, l_target, label = "Target", **TARGET_KWARGS)
    plt.yscale("log")
    plt.xlim(left = mass_min, right = mass_max)
    plt.ylim(bottom = lambda_min, top = lambda_max)
    
    for i in range(N_max):
        color = cmap(norm(i))
        
        m, r, l = all_masses_EOS[i], all_radii_EOS[i], all_Lambdas_EOS[i]
        mask = m > m_min
        m, r, l = m[mask], r[mask], l[mask]
        
        # Mass-radius plot
        plt.subplot(2, 3, 4)
        plt.plot(r, m, color=color, linewidth = 2.0, zorder=100 + i, rasterized=rasterized)
            
        # Mass-Lambdas plot
        plt.subplot(2, 3, 6)
        plt.plot(m, l, color=color, linewidth = 2.0, zorder=100 + i, rasterized=rasterized)
        
        # We put the legend in the mass-Lambdas plot
        plt.legend(fontsize = 24, numpoints = 10)
        
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Create a phantom axis for the colorbar relative to axs[1, 0]
    cbar_ax = inset_axes(axs[1, 0], 
                        width="100%",  # Colorbar width relative to the axis width
                        height="50%",   # Colorbar height relative to the axis height
                        loc='upper center',  # Place above the axis
                        bbox_to_anchor=(0.0, 1.2, 1, 0.1),  # Fine-tune position
                        bbox_transform=axs[1, 0].transAxes,  # Relative to axs[1, 0]
                        ) # borderpad=0
    
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r'Iteration', fontsize = 32)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    
    # Remove phantom plots
    axs[0, 0].remove()
    axs[0, 1].remove()
    
    # Share y-axis between the corresponding subplots
    axs[1, 1].sharey(axs[1, 0])
    axs[0, 2].sharex(axs[1, 2])

    # Disable ticks for some plots
    axs[1, 1].tick_params(left=False, labelleft=False)
    axs[0, 2].tick_params(bottom=False, labelbottom=False)
    
    # Mass-Lambdas ticks at the right
    axs[0, 2].yaxis.tick_right()
    axs[0, 2].yaxis.set_label_position("right")
    
    axs[1, 2].yaxis.tick_right()
    axs[1, 2].yaxis.set_label_position("right")
    
    # plt.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    print(f"Saving to: {save_name}")
    plt.savefig(save_name, bbox_inches = "tight")
    plt.savefig(save_name.replace(".png", ".pdf"), bbox_inches = "tight")
    plt.close()
    
def report_doppelganger(dir: str = "../doppelgangers/real_doppelgangers/7007/data/",
                        target_filename: str = "../doppelgangers/hauke_macroscopic.dat",
                        nb_masses: int = 500):
    
    # Load the final iteration of the doppelganger
    files = [f for f in os.listdir(dir) if f.endswith(".npz") and "best" not in f]
    idx = [int(f.split(".")[0]) for f in files]
    max_idx = max(idx)
    
    best_file = os.path.join(dir, f"{max_idx}.npz")
    data = np.load(best_file)
    
    m_doppelganger, r_doppelganger, l_doppelganger = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
    
    # Load the target
    m_target, r_target, l_target = load_target(target_filename)
    
    # Interpolate on the mass array used during optimization to report the final errors
    mass_array = np.linspace(1.2, 2.1, nb_masses)
    
    r_target_interp = np.interp(mass_array, m_target, r_target)
    l_target_interp = np.interp(mass_array, m_target, l_target)
    
    r_doppelganger_interp = np.interp(mass_array, m_doppelganger, r_doppelganger)
    l_doppelganger_interp = np.interp(mass_array, m_doppelganger, l_doppelganger)
    
    # Find the final max error
    max_error_radii = np.max(np.abs(r_target_interp - r_doppelganger_interp))
    max_error_Lambdas = np.max(np.abs(l_target_interp - l_doppelganger_interp))
    
    print("\n")
    print(f"Max error in radii: {max_error_radii} km")
    print(f"Max error in Lambdas: {max_error_Lambdas}")
    print("\n")
    
    # Difference at 1.4 solar masses:
    r14_target = np.interp(1.4, m_target, r_target)
    r14_doppelganger = np.interp(1.4, m_doppelganger, r_doppelganger)
    error_r14 = np.abs(r14_target - r14_doppelganger)
    
    l14_target = np.interp(1.4, m_target, l_target)
    l14_doppelganger = np.interp(1.4, m_doppelganger, l_doppelganger)
    error_l14 = np.abs(l14_target - l14_doppelganger)
    
    print("\n")
    print(f"Delta R1.4: {error_r14 * 1000} m")
    print(f"Delta L1.4: {error_l14}")
    print("\n")
    
def plot_campaign_results(outdirs_list: list[str],
                          target_filename = "../doppelgangers/my_target_macroscopic.dat",
                          add_units: bool = True):
    
    # Load the target EOS and NS:
    m_target, r_target, l_target = load_target(target_filename)
    
    if add_units:
        extra_string_MeV = r" [MeV]"
        extra_string_Modot = r" [M$_\odot$]"
        extra_string_km = r" [km]"
    else:
        extra_string_MeV = r""
        extra_string_Modot = r""
        extra_string_km = r""
        
    
    results = {"masses_EOS": [],
               "MTOV": [],
               "radii_EOS": [],
               "Lambdas_EOS": [],
               
               "R1.4": [],
               "Lambda1.4": [],
               
               "n": [],
               "p": [],
               
               "E_sym": [],
               "L_sym": [],
               "K_sym": [],
               "Q_sym": [],
               "Z_sym": [],
               
               "E_sat": [],
               "K_sat": [],
               "Q_sat": [],
               "Z_sat": []
               }
    
    prior_ranges = {"E_sym": [28.0, 45.0],
                    "L_sym": [10.0, 200.0],
                    "K_sym": [-300.0, 100.0],
                    "Q_sym": [-800.0, 800.0],
                    "Z_sym": [-2500.0, 1500.0],
                    "K_sat": [150.0, 300.0],
                    "Q_sat": [-500.0, 1100.0],
                    "Z_sat": [-2500.0, 1500.0],
                    }
    NEP_keys = list(prior_ranges.keys())
    
    # TODO: will augment this with MTOV, R1.4 and L1.4, but need to do some more preprocessing for that
    all_keys = NEP_keys + ["MTOV", "R1.4", "Lambda1.4"]
    
    # Gather the results: iterate over different optimization campaigns
    for outdir in outdirs_list:
        # Get separate results of each run
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
            MTOV = np.max(masses_EOS)
            radii_EOS = data["radii_EOS"]
            Lambdas_EOS = data["Lambdas_EOS"]
            n = data["n"]
            p = data["p"]
            
            n = n / jose_utils.fm_inv3_to_geometric / 0.16 # convert to nsat
            p = p / jose_utils.MeV_fm_inv3_to_geometric # convert to MeV/fm^3
            
            # Append -- this is sloppy but making it more succint is really annoying and not worth it (for now)
            results["masses_EOS"].append(masses_EOS)
            results["MTOV"].append(MTOV)
            results["radii_EOS"].append(radii_EOS)
            results["Lambdas_EOS"].append(Lambdas_EOS)
            results["n"].append(n)
            results["p"].append(p)
            
            results["E_sym"].append(data["E_sym"])
            results["L_sym"].append(data["L_sym"])
            results["K_sym"].append(data["K_sym"])
            results["Q_sym"].append(data["Q_sym"])
            results["Z_sym"].append(data["Z_sym"])

            results["K_sat"].append(data["K_sat"])
            results["Q_sat"].append(data["Q_sat"])
            results["Z_sat"].append(data["Z_sat"])
            
            r14 = np.interp(1.4, masses_EOS, radii_EOS)
            l14 = np.interp(1.4, masses_EOS, Lambdas_EOS)
            results["R1.4"].append(r14)
            results["Lambda1.4"].append(l14)
            
    # Convert all to numpy arrays
    for key in results.keys():
        results[key] = np.array(results[key])
        
    # Comput the "prior range" for MTOV, R1.4 and Lambda1.4
    prior_ranges["MTOV"] = [np.min(results["MTOV"]), np.max(results["MTOV"])]
    prior_ranges["R1.4"] = [np.min(results["R1.4"]), np.max(results["R1.4"])]
    prior_ranges["Lambda1.4"] = [np.min(results["Lambda1.4"]), np.max(results["Lambda1.4"])]
            
    # Fetch the true params:
    all_true_params = utils.NEP_CONSTANTS_DICT
    
    # Also compute MTOV, R1.4 and Lambda1.4 for the target:
    MTOV_target = np.max(m_target)
    R1_4_target = np.interp(1.4, m_target, r_target)
    Lambda1_4_target = np.interp(1.4, m_target, l_target)
    
    # Save in all_true_params
    all_true_params["MTOV"] = MTOV_target
    all_true_params["R1.4"] = R1_4_target
    all_true_params["Lambda1.4"] = Lambda1_4_target
    
    param_labels = [r"$E_{\rm{sym}}$" + extra_string_MeV,
                    r"$L_{\rm{sym}}$" + extra_string_MeV,
                    r"$K_{\rm{sym}}$" + extra_string_MeV,
                    r"$Q_{\rm{sym}}$" + extra_string_MeV,
                    r"$Z_{\rm{sym}}$" + extra_string_MeV,
                    r"$K_{\rm{sat}}$" + extra_string_MeV,
                    r"$Q_{\rm{sat}}$" + extra_string_MeV,
                    r"$Z_{\rm{sat}}$" + extra_string_MeV,
                    r"$M_{\rm{TOV}}$" + extra_string_Modot,
                    r"$R_{1.4}$" + extra_string_km,
                    r"$\Lambda_{1.4}$"
                    ]
    ### Finally, make the plots
    print(f"Making the plot")
    
    # Hyperparameters put here:
    xlabel_fontsize = 12
    ticks_fontsize = 12
    num_ticks = 5
    
    # Trying color combinations from https://www.wada-sanzo-colors.com/
    # combi #7
    
    # TRUE_COLOR = "#F37420"
    # DOPPELGANGER_COLOR = "#B4CDC2"
    
    # combi 30
    TRUE_COLOR = "#AB2439"
    DOPPELGANGER_COLOR = "#A2B0AD"
    
    
    fig, axes = plt.subplots(nrows = 1, ncols = len(param_labels), figsize=(len(param_labels) * 1.5, 6))
    for i, (key, label) in enumerate(zip(all_keys, param_labels)):
        # Select the subplot
        ax = axes[i]
        
        # Fetch the values:
        true_value = all_true_params[key]
        samples = results[key]
        if key in prior_ranges:
            range_min, range_max = prior_ranges[key]
            range_width = range_max - range_min
            eps = 0.05 * range_width
        
        # Jitter for better visualization
        jitter_eps = 0.0
        jitter_factor = 10
        jitter_x = np.random.uniform(-jitter_eps, jitter_eps, size=len(samples))
        x_positions = np.zeros_like(samples) + jitter_x
        
        ax.scatter(x_positions, samples, color=DOPPELGANGER_COLOR, alpha=0.75, s=50, label=label)
        ax.scatter(0, true_value, color=TRUE_COLOR, s=100, zorder=100)

        # Set the yticks:
        ticks = np.linspace(range_min, range_max, num_ticks)
        
        # Round ticks labels to integers just for visualization, unless for MTOV and R1.4 which are more precise:
        if key not in ["MTOV", "R1.4"]:
            ticks_labels = [f"{tick:.0f}" for tick in ticks]
        else:
            ticks_labels = [f"{tick:.2f}" for tick in ticks]
        ax.set_ylim(range_min, range_max)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Custom yticks
        ax.set_yticks(ticks)
        ax.set_yticklabels(
            ticks_labels,
            fontsize=ticks_fontsize,
            rotation=90,
        )
        ax.tick_params(axis='y', which='both', direction='in', pad=2)

        # Set the y-axis limits and label for clarity
        if jitter_eps > 0.0:
            ax.set_xlim(-jitter_factor*jitter_eps, jitter_factor*jitter_eps)
        ax.set_ylim(range_min - eps, range_max + eps)
        ax.set_xticks([])
        ax.set_xlabel(label, fontsize=xlabel_fontsize, rotation=0, labelpad=5, ha='center')
        plt.subplots_adjust(wspace=0.5)
        
    # Show the plot
    plt.savefig("./figures/final_doppelgangers/campaign_results.png")
    plt.close()


    
def main():
    
    """Plots for single runs, which are shown at the start of the section"""
    
    # ### These are the doppelganger runs where the target is Hauke's max L Set A EOS  
    # my_dir = "../doppelgangers/3133_radius/3133/data/"
    # xticks_error_radii = [0.1, 200]
    # yticks_error_Lambdas = [1, 1000]
    # plot_NS(my_dir, 
    #         xticks_error_radii,
    #         yticks_error_Lambdas,
    #         save_name="./figures/final_doppelgangers/doppelganger_trajectory_3133_radius.png")
    # report_doppelganger(my_dir)

    # my_dir = "../doppelgangers/3133_Lambdas/3133/data/"
    # xticks_error_radii = [30, 1300]
    # yticks_error_Lambdas = [0.0, 200]
    # plot_NS(my_dir, 
    #         xticks_error_radii,
    #         yticks_error_Lambdas,
    #         save_name="./figures/final_doppelgangers/doppelganger_trajectory_3133_Lambdas.png")
    # report_doppelganger(my_dir)

    # ### These are with the JESTER-generated target EOS
    # target_filename="../doppelgangers/my_target_macroscopic.dat"

    # my_dir = "../doppelgangers/campaign_results/Lambdas/04_12_2024_doppelgangers/1784/data"
    # xticks_error_radii = [1, 110]
    # yticks_error_Lambdas = [0.001, 200]
    # plot_NS(my_dir, 
    #         xticks_error_radii,
    #         yticks_error_Lambdas,
    #         target_filename=target_filename,
    #         save_name="./figures/final_doppelgangers/doppelganger_trajectory_Lambdas_04_12_seed_1784.png")
    # report_doppelganger(my_dir, target_filename=target_filename)
    
    
    """Plots for the larger campaign of runs"""
    
    outdirs_list = ["../doppelgangers/campaign_results/Lambdas/04_12_2024_doppelgangers/",
                    "../doppelgangers/campaign_results/radii/04_12_2024_doppelgangers/"]
    plot_campaign_results(outdirs_list)
    
    print("DONE")

if __name__ == "__main__":
    main()