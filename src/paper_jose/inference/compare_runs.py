"""Compare different runs, e.g. their posterior to prior ratio or something"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
import sys
import corner
import tqdm
import argparse
import arviz
from scipy.spatial.distance import jensenshannon

np.random.seed(2)

from jimgw.prior import UniformPrior
import joseTOV.utils as jose_utils

def get_posterior_to_prior_ratio(outdir: str):
    
    # Load the posterior samples
    filename = os.path.join(outdir, "results_production.npz")
    data = np.load(filename)
    
    # Load the prior samples
    if "metamodel" in outdir:
        filename_prior = "./metamodel/outdir_prior/results_production.npz"
    else:
        filename_prior = "./outdir_prior/results_production.npz"
        
    data_prior = np.load(filename_prior)
        
    posterior_to_prior_widths = {}
    JSD_dict = {}
    
    # Debug; print the keys
    print(list(data.keys()))
    
    for key in data.keys():
        if key == "log_prob" or "mass" in key:
            continue
        else:
            # Load posterior width
            posterior_chains = np.array(data[key]).flatten()
            prior_chains = np.array(data_prior[key]).flatten()
            
            # Get the standard deviation
            posterior_width = np.std(posterior_chains)
            prior_width = np.std(prior_chains)
            
            posterior_to_prior_widths[key] = posterior_width / prior_width
            
            # Make histograms to then compute the Jensen Shanon Divergences
            hist_prior, _bins = np.histogram(prior_chains, bins=100, density=True)
            hist_posterior, _ = np.histogram(posterior_chains, bins=_bins, density=True)
            
            jsd = jensenshannon(hist_prior, hist_posterior)
            JSD_dict[key] = jsd
            
        # Save the final result in the outdir
        np.savez(os.path.join(outdir, "PPR.npz"), **posterior_to_prior_widths)
        np.savez(os.path.join(outdir, "JSD.npz"), **JSD_dict)
        
    # Print the results
    print(f"The PPR for {outdir}")
    for key, value in posterior_to_prior_widths.items():
        print(f"    {key}: {value}")
        
    print(f"The JSD for {outdir}")
    for key, value in JSD_dict.items():
        print(f"    {key}: {value}")

    # Check and print on Lsym as well:
    Lsym_posterior = np.array(data["L_sym"]).flatten()
    median = np.median(Lsym_posterior)
    low, high = arviz.hdi(Lsym_posterior, 0.95)
    low = median - low
    high = high - median
    
    print(f"\nFinal Lsym result: {median:.2f}-{low:.2f}+{high:.2f}")

def main():
    outdir = sys.argv[1]
    get_posterior_to_prior_ratio(outdir)

if __name__ == "__main__":
    main()