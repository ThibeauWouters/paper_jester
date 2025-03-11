import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import arviz
import json
import re

import joseTOV.utils as jose_utils
from scipy.optimize import curve_fit
from scipy.stats import linregress

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

NB_CSE_list = np.array([10, 20, 30, 40, 50]) # NOTE: we did a run of 50, but for the scaling, let us focus on up to 40

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
    
def postprocess_runs(output_label: str = "H100"):
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
        all_results[str(nb_cse)] = results
        
    # Dump:
    print("all_results")
    print(all_results)
    
    output_file = f"./data/{output_label}.json"
    print(f"Dumping the results to {output_file}")
    
    with open(output_file, "w") as f:
        json.dump(all_results, f)
    

def make_scaling_plot(plot_lines: bool = True,
                      plot_linear_scaling: bool = False,
                      plot_linear_fit: bool = False):
    """
    Make a plot to show the scaling of jester as a function of number of parameters in the CSE. 
    """
    
    all_labels = ["A100", 
                  "H100"]
    # The colors are Combination 5 of https://www.wada-sanzo-colors.com/combinations/5 
    colors_dict = {"H100": "#437742",
                   "A100": "#064f6e"
                   }
    
    # legend_labels = {"H100": "NVIDIA H100 GPU",
    #                  "A100": "NVIDIA A100 GPU"
    # }
    
    legend_labels = {"H100": "H100",
                     "A100": "A100"
    }
    
    # Define a linear function for fitting
    def linear_func(x, a, b):
        return a * x + b
    
    # TODO: duplicate but otherwise does not work
    plt.figure(figsize = (6, 4))
    for i, label in enumerate(all_labels):
        nb_parameters = [34, 54, 74, 94, 114]
        NB_CSE_list = np.array([10, 20, 30, 40, 50]) # NOTE: we did a run of 50, but for the scaling plot, let us focus on up to 40 for clarity
        
        x_array = nb_parameters
        
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
        plot_label = legend_labels[label]
        c = colors_dict[label]
        
        # Limit to ditch the final CSE points runs
        limit_nb = 2
        x_array = x_array[:-limit_nb]
        y_values = y_values[:-limit_nb]
        y_err = y_err[:-limit_nb]
        
        plt.errorbar(x_array, y_values, yerr = y_err, capsize = 5, fmt = "o", color = c, label = plot_label)
        if plot_lines:
            plt.plot(x_array, y_values, color = colors_dict[label], alpha = 0.5)
            
        ratio_runtimes = y_values[-1] / y_values[0]
        
        print(f"ratio_runtimes for {label}: {ratio_runtimes:.2f}")
            
        if plot_linear_scaling:
            x_ = np.linspace(34, 94, 100)
            x0 = x_[0]
            
            # Show the scaling relation
            # y_scaling = y_values[0] * (x_ / x0) # linear scaling
            # y_scaling = y_values[0] * (x_ / x0) ** 2 # quadratic scaling
            y_scaling = y_values[0] * (x_ / x0) ** 1.5 # x sqrt(x) scaling
            
            # y_scaling = y_values[0] * (x_ / x0) * np.log((x_ / x0)) # log scaling
            
            if i == 0:
                plt.plot(x_, y_scaling, linestyle = "--", color = "gray", label = "Quadratic scaling")
            else:
                plt.plot(x_, y_scaling, linestyle = "--", color = "gray")
        plt.legend()
            
        # # Fit the data with a line
        # params, covariance = curve_fit(linear_func, x_array, y_values)
        # a_fit, b_fit = params
        # print(f"Label: {label}, a: {a_fit:.2f}, b: {b_fit:.2f}")
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x_array, y_values)
        
        print(f"Label {label} has slope {slope:.2f} and intercept {intercept:.2f}, and p-value {p_value:.6f}")
        
        # Generate fitted line
        if plot_linear_fit:
            eps = 1
            x_fit = np.linspace(min(x_array) - eps, max(x_array) + eps, 100)
            y_fit = linear_func(x_fit, slope, intercept)
            
            plt.plot(x_fit, y_fit, linestyle = "--", color = "gray")
            
    # Make the ticks equal to NB_CSE_list
    plt.xticks(x_array)
           
    # Finalize the plot with labels etc
    
    plt.grid(False)
    plt.xlabel("Number of EOS parameters")
    plt.ylabel("Runtime/effective sample [s]")
    
    # TODO: if we do not want to show number of grid points and only the parameters, this is redundant and can be deleted
    # # Extra axis to put the top x ticks for number of paramters
    # ax1 = plt.gca()
    # ax2 = ax1.twiny()
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_xticks(NB_CSE_list)
    # ax2.set_xticklabels(nb_parameters)
    # ax2.set_xlabel("Total number of parameters")
    
    plt.grid(False)
    left = x_array[0]
    right = x_array[-1]
    dx = 2
    plt.xlim(left - dx, right + dx)
    plt.ylim(0.45, 4.1)
    
    # Finally, save the figure
    plt.savefig("./figures/scaling_plot.pdf", bbox_inches = "tight")
    plt.close()

def main():
    # ### This is used to get, after a run with certain hardware setup is done and finished, the postprocessed results
    # postprocess_runs()
    
    ### Here, we make the actual scaling plot. See the function for which labels etc are fetched.
    print("Making scaling plot")
    make_scaling_plot()
    print("Making scaling plot DONE")
    
if __name__ == "__main__":
    main()