import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import arviz
import json
import re

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

NB_CSE_list = [10, 20, 30, 40, 50] # 2, 5, # TODO: for 2 and 5, there is fixed CSE?

def fetch_runtime(nb_cse: int, verbose: bool = False):
    """
    Digs into the log file to fetch the walltime of the parameter estimation run.

    Args:
        nb_cse (int): Number of CSE points for this run to find the correct directory and therefore the file
    """
    
    # Open the file:
    filename = f"./outdir_{nb_cse}/log.out"
    with open(filename, "r") as f:
        log_text = f.read()
    
    # Extract Job Wall-clock time
    match = re.search(r"Job Wall-clock time:\s*(\d+):(\d+):(\d+)", log_text)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        runtime = hours * 3600 + minutes * 60 + seconds
        if verbose:
            print(f"Job Wall-clock is: {hours}:{minutes}:{seconds}")
            print(f"Job Wall-clock: {runtime:.2f} seconds")
    else:
        raise ValueError("Job Wall-clock time not found.")
    
    return float(runtime)
    
def postprocess_runs(output_label: str = "h100"):
    """
    Takes the run samples and saves ESS et cetera to relevant file.
    """
    keys_to_skip = ["E_sat", "log_prob", "Lambdas_EOS", "cs2", 'dloge_dlogp', 'e', 'h', 'logpc_EOS', 'masses_EOS', 'n', 'p', 'radii_EOS']
    
    # Store all results for different N_CSE in a dictionary
    all_results = {}
    for nb_cse in NB_CSE_list:
        # Store the results of this run here
        results = {}
        
        eos_samples = f"./outdir_{nb_cse}/results_production.npz"
        if not os.path.exists(eos_samples):
            print(f"File {eos_samples} does not exist.")
            return
        
        # Load the data
        run_results = np.load(eos_samples)
        all_keys = list(run_results.keys())
        keys = [k for k in all_keys if k not in keys_to_skip]
        
        n_dim = len(keys)
        print(f"Number of dimensions for N_CSE = {nb_cse}: {n_dim}")
        
        # Put the ESS of the samples into the results dictionary
        for key in keys:
            samples = np.array(run_results[key])
            ess = float(arviz.ess(samples))
            # print(f"Key {key}: ESS {ess}")
            results[key] = ess
            
        # Fetch the runtime:
        runtime = fetch_runtime(nb_cse)
        results["runtime"] = runtime
        
        # Save:
        all_results[nb_cse] = results
        
    # Dump:
    output_file = f"./data/{output_label}.json"
    print(f"Dumping the results to {output_file}")
    
    with open(output_file, "w") as f:
        json.dump(all_results, f)
    

def make_scaling_plot(plot_lines: bool = False):
    """
    Make a plot to show the scaling of jester as a function of number of parameters in the CSE. 
    """
    
    # TODO: extend this with all the different runs hardware stuff we have done.
    all_labels = ["h100"]
    colors_dict = {"h100": "green"}
    labels_dict = {"h100": "H100"}
    
    plt.figure(figsize = (8, 6))
    for label in all_labels:
        # Load the data
        with open(f"./data/{label}.json", "r") as f:
            data = json.load(f)
        
        y_values = []
        y_err = []
        
        # Loop over the different number of CSE points and fetch the runtime and ESS values
        for results in data.values():
            # Fetch the runtime and pop from the dictionary so that the remaining keys are all the sampled parameters
            runtime = results["runtime"]
            results.pop("runtime")
            
            all_ess_values = np.array(list(results.values()))
            all_runtime_per_ess = runtime / all_ess_values
            
            runtime_per_ess = np.mean(all_runtime_per_ess)
            err = np.std(all_runtime_per_ess)
            
            y_values.append(runtime_per_ess)
            y_err.append(err)
            
        # Make the plot
        plt.errorbar(NB_CSE_list, y_values, yerr = y_err, capsize = 5, fmt = "o", color = colors_dict[label], label = labels_dict[label])
        if plot_lines:
            plt.plot(NB_CSE_list, y_values, color = colors_dict[label])
           
    # Make the ticks equal to NB_CSE_list
    plt.xticks(NB_CSE_list)
           
    # Finalize the plot with labels etc
    plt.legend()
    plt.grid(False)
    plt.xlabel("Number of CSE grid points")
    plt.ylabel("Runtime/effective sample size [s]")
    
    # Add a secondary x-axis on top with number of parameters
    nb_parameters = [34, 54, 74, 94, 114]
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(NB_CSE_list)
    ax2.set_xticklabels(nb_parameters)
    ax2.set_xlabel("Total number of parameters")
    plt.grid(False)
    
    # Finally, save the figure
    plt.savefig("./figures/scaling_plot.pdf", bbox_inches = "tight")
    plt.close()

def main():
    ### This is used to get, after a run with certain hardware setup is done and finished, the postprocessed results
    # postprocess_runs()
    
    ### Here, we make the actual scaling plot. See the function for which labels etc are fetched.
    print("Making scaling plot")
    make_scaling_plot()
    print("Making scaling plot DONE")
    
if __name__ == "__main__":
    main()