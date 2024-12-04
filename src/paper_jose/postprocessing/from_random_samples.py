"""
Checking stuff about doppelgangers and random samples
"""

################
### PREAMBLE ###
################

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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

PATH = "/home/thibeau.wouters/projects/jax_tov_eos/paper_jose/src/paper_jose/doppelgangers/"
micro_target_filename = PATH + "hauke_microscopic.dat"
macro_target_filename = PATH + "hauke_macroscopic.dat"

data = np.genfromtxt(macro_target_filename, skip_header=1, delimiter=" ").T
r_target, m_target, Lambdas_target = data[0], data[1], data[2]

def make_differences_csv(random_samples_dir = "../benchmarks/random_samples/"):
    
    # Initialize where we will save the output
    filenames = []
    max_r_diffs = []
    max_l_diffs = []
    
    avg_r_diffs = []
    avg_l_diffs = []
    
    mass_array = np.linspace(1.2, 2.1, 500)

    for file in tqdm.tqdm(os.listdir(random_samples_dir)):
        full_file = os.path.join(random_samples_dir, file)
        data = np.load(full_file)
        
        # On the masses, evaluate and look for difference
        m = data["masses_EOS"]
        r = data["radii_EOS"]
        l = data["Lambdas_EOS"]
        
        r_interp = np.interp(mass_array, m, r)
        l_interp = np.interp(mass_array, m, l)
        
        r_target_interp = np.interp(mass_array, m_target, r_target)
        l_target_interp = np.interp(mass_array, m_target, Lambdas_target)
        
        r_diffs = np.abs(r_interp - r_target_interp)
        l_diffs = np.abs(l_interp - l_target_interp)
        
        max_r_diff = np.max(r_diffs)
        max_l_diff = np.max(l_diffs)
        
        avg_r_diff = np.mean(r_diffs)
        avg_l_diff = np.mean(l_diffs)
        
        filenames.append(file)
        max_r_diffs.append(max_r_diff)
        max_l_diffs.append(max_l_diff)
        
        avg_r_diffs.append(avg_r_diff)
        avg_l_diffs.append(avg_l_diff)

    output_dir = {"filenames": filenames,
                  "max_r_diffs": max_r_diffs,
                  "max_l_diffs": max_l_diffs}

    # Save it
    output_dir = pd.DataFrame(output_dir)
    output_dir.to_csv("random_samples_doppelgangers.csv", index=False)
    
    return

def analyze_random_samples():
        
        # Load the data
        data = pd.read_csv("random_samples_doppelgangers.csv")
        
        # Get the max differences
        max_r_diffs = data["max_r_diffs"]
        max_l_diffs = data["max_l_diffs"]
        
        # One sample seems off, ditch it
        mask = max_r_diffs < 20.0
        
        max_r_diffs = max_r_diffs[mask]
        max_l_diffs = max_l_diffs[mask]
        
        # Plot
        print("Plotting a scatterplot of the differences")
        plt.subplots(1, 1, figsize=(6, 6))
        plt.scatter(max_r_diffs, max_l_diffs, color="black", s=10)
        plt.xlabel("Max radius difference")
        plt.ylabel("Max Lambda difference")
        plt.yscale("log")
        plt.title("Differences between random samples and Hauke's data")
        plt.savefig("./figures/random_samples/scatter_differences_random_samples.png")
        plt.close()
        
        # Make histograms
        print("Plotting a histogram of the radius differences")
        plt.figure(figsize=(12, 6))
        plt.hist(max_r_diffs, bins=20, color="blue", linewidth = 4, histtype="step", density = True)
        plt.xlabel("Max radius difference")
        plt.ylabel("Density")
        plt.savefig("./figures/random_samples/histogram_radius_differences_random_samples.png")
        plt.close()
        
        plt.figure(figsize=(12, 6))
        print("Plotting a histogram of the lambdas differences")
        plt.hist(max_l_diffs, bins=20, color="blue", linewidth = 4, histtype="step", density = True)
        plt.xlabel("Max Lambdas difference")
        plt.ylabel("Density")
        plt.savefig("./figures/random_samples/histogram_Lambdas_differences_random_samples.png")
        plt.close()
        
def get_starting_points(csv_filename: str = "random_samples_doppelgangers.csv"):
    
    # Load the differences CSV:
    data = pd.read_csv(csv_filename)
    # Keep those that have max radius difference less than 250 m
    mask = data["max_l_diffs"] < 10.0
    data = data[mask]
    
    print("Selected data")
    print(data)
    
    # Export the dirs to a txt file
    with open("selected_dirs.txt", "w") as f:
        for dir in data["filenames"]:
            f.write(dir + "\n")

def main():
    
    # make_differences_csv()
    get_starting_points()
    # analyze_random_samples()
    
    return

if __name__ == "__main__":
    main()