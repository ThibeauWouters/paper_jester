import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
print(jax.devices())

import jax.numpy as jnp
from jaxtyping import Array

from paper_jose.universal_relations.universal_relations import UniversalRelationBreaker
import paper_jose.utils as utils
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
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "axes.labelsize": 24,
        # "legend.fontsize": 16,
        # "legend.title_fontsize": 16,
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
    

def plot_NS(dir: str = "../doppelgangers/real_doppelgangers/7007/data/",
            max_nb_steps: int = 100,
            save_name: str = "./figures/doppelganger_trajectory.png",
            m_target: Array = None,
            r_target: Array = None,
            l_target: Array = None,
            m_min: float = 0.75):
    """
    Plot the doppelganger trajectory in the NS space.
    Args:
        m_min (float, optional): Minimum mass from which to compute errors and create the error plot. Defaults to 1.2.
    """
    
    if m_target is None:
        m_target, r_target, l_target = load_target()
        
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
        
        if i == 0:
            print("list(data.keys())")
            print(list(data.keys()))
        
        masses_EOS = data["masses_EOS"]
        radii_EOS = data["radii_EOS"]
        Lambdas_EOS = data["Lambdas_EOS"]
        
        if not np.any(np.isnan(masses_EOS)) and not np.any(np.isnan(radii_EOS)) and not np.any(np.isnan(Lambdas_EOS)):
        
            all_masses_EOS.append(masses_EOS)
            all_radii_EOS.append(radii_EOS)
            all_Lambdas_EOS.append(Lambdas_EOS)
            
        
    # N might have become smaller if we hit NaNs at some point
    N_max = len(all_masses_EOS)
    norm = mpl.colors.Normalize(vmin=0, vmax=N_max)
    # cmap = sns.color_palette("rocket_r", as_cmap=True)
    cmap = mpl.cm.GnBu
        
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 6))
    
    # TODO: plot the target
    # Plot the target
    plt.subplot(121)
    plt.plot(r_target, m_target, **TARGET_KWARGS)
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M \ [M_\odot]$")
    plt.ylim(bottom = 1.0)
    
    plt.subplot(122)
    plt.xlabel(r"$M \ [M_\odot]$")
    plt.ylabel(r"$\Lambda$")
    plt.plot(m_target, l_target, label = "Target", **TARGET_KWARGS)
    plt.yscale("log")
    plt.xlim(left = 1.0)
    plt.ylim(bottom = 2, top = 5e3)
    
    for i in range(N_max):
        color = cmap(norm(i))
        
        m, r, l = all_masses_EOS[i], all_radii_EOS[i], all_Lambdas_EOS[i]
        mask = m > m_min
        m, r, l = m[mask], r[mask], l[mask]
        
        # Mass-radius plot
        plt.subplot(121)
        plt.plot(r, m, color=color, linewidth = 2.0, zorder=100 + i)
            
        # Mass-Lambdas plot
        plt.subplot(122)
        plt.plot(m, l, color=color, linewidth = 2.0, zorder=100 + i)
        
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs[-1])
    cbar.set_label(r'Iteration', fontsize = 32)
        
    plt.tight_layout()
    plt.legend(fontsize = 24, numpoints = 10)
    print(f"Saving to: {save_name}")
    plt.savefig(save_name, bbox_inches = "tight")
    plt.savefig(save_name.replace(".png", ".pdf"), bbox_inches = "tight")
    plt.close()
    
def report_doppelganger(dir: str = "../doppelgangers/real_doppelgangers/7007/data/",
                        nb_masses: int = 500,):
    
    # Load the final iteration of the doppelganger
    files = [f for f in os.listdir(dir) if f.endswith(".npz") and "best" not in f]
    idx = [int(f.split(".")[0]) for f in files]
    max_idx = max(idx)
    
    best_file = os.path.join(dir, f"{max_idx}.npz")
    data = np.load(best_file)
    
    m_doppelganger, r_doppelganger, l_doppelganger = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
    
    # Load the target
    m_target, r_target, l_target = load_target()
    
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
    
    
    
my_dir = "../doppelgangers/real_doppelgangers/209/data/" # this had a pretty nice trajectory just picking it for visualization purposes
plot_NS(my_dir, max_nb_steps=110)
# plot_all_NS()
report_doppelganger()