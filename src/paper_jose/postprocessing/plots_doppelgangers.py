import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
import shutil

import os
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns
from typing import Union, Callable
from collections import defaultdict

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
print(jax.devices())

import jax.numpy as jnp
from jimgw.prior import UniformPrior, CombinePrior
from jaxtyping import Array
import joseTOV.utils as jose_utils

from paper_jose.universal_relations.universal_relations import UniversalRelationBreaker
import paper_jose.utils as utils
import paper_jose.inference.utils_plotting as utils_plotting
plt.rcParams.update(utils_plotting.mpl_params)
import seaborn as sns


def plot_all_NS(main_dir: str = "../doppelgangers/real_doppelgangers/",
                m_min: float = 1.2,
                max_nb_steps: int = 100):
    
    all_dirs = os.listdir(main_dir)
    for dir in all_dirs:
        full_dir = os.path.join(main_dir, dir, "data")
        plot_NS(full_dir, m_min, max_nb_steps, save_name = f"./figures/doppelganger_dir_{dir}.png")
    

def plot_NS(dir: str = "../doppelgangers/real_doppelgangers/7945/data/",
            m_min: float = 1.2,
            max_nb_steps: int = 100,
            save_name: str = "./figures/doppelganger_trajectory.png"):
    """
    Plot the doppelganger trajectory in the NS space.
    Args:
        m_min (float, optional): Minimum mass from which to compute errors and create the error plot. Defaults to 1.2.
    """

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
            
        
    # N might have become smaller if we hit NaNs at some point
    N_max = len(all_masses_EOS)
    norm = mpl.colors.Normalize(vmin=0, vmax=N_max)
    # cmap = sns.color_palette("rocket_r", as_cmap=True)
    cmap = mpl.cm.viridis
        
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 6))
    
    # # TODO: plot the target
    # # Plot the target
    # plt.subplot(121)
    # plt.plot(self.r_target, self.m_target, color = "red", zorder = 1e10)
    # plt.xlabel(r"$R$ [km]")
    # plt.ylabel(r"$M \ [M_\odot]$")
    
    # plt.subplot(122)
    # plt.xlabel(r"$M \ [M_\odot]$")
    # plt.ylabel(r"$\Lambda$")
    # plt.plot(self.m_target, self.Lambdas_target, label=r"$\Lambda$", color = "red", zorder = 1e10)
    # plt.yscale("log")
    
    for i in range(N_max):
        color = cmap(norm(i))
        
        # Mass-radius plot
        plt.subplot(121)
        plt.plot(all_radii_EOS[i], all_masses_EOS[i], color=color, linewidth = 2.0, zorder=i)
            
        # Mass-Lambdas plot
        plt.subplot(122)
        plt.yscale('log')
        plt.plot(all_masses_EOS[i], all_Lambdas_EOS[i], color=color, linewidth = 2.0, zorder=i)
        
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs[-1])
    cbar.set_label(r'Iteration number', fontsize = 22)
        
    plt.tight_layout()
    print(f"Saving to: {save_name}")
    plt.savefig(save_name, bbox_inches = "tight")
    plt.close()
    
plot_all_NS()