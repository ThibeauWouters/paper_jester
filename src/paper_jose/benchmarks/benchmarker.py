"""
Randomly sample EOSs and solve TOV for benchmarking purposes
"""

################
### PREAMBLE ###
################

import os
import shutil
import time
import sys
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
jax.config.update("jax_enable_x64", False)
print(jax.devices())

from jimgw.prior import UniformPrior, CombinePrior

import jax.numpy as jnp
from jimgw.prior import CombinePrior
import joseTOV.utils as jose_utils

import paper_jose.utils as utils
import paper_jose.inference.utils_plotting as utils_plotting
plt.rcParams.update(utils_plotting.mpl_params)
import seaborn as sns

from paper_jose.benchmarks.nmma_tov import EOS_with_CSE

#############
### PATHS ###
#############

KOEHN_SAMPLES_MICRO = "/home/twouters2/hauke_eos/micro/"
KOEHN_SAMPLES_MACRO = "/home/twouters2/hauke_eos/macro/"

###################
### BASIC SETUP ###
###################

USE_CSE = True

NMAX_NSAT = 25
NMAX = NMAX_NSAT * 0.16
NB_CSE = 8

### NEP priors
K_sat_prior = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
Q_sat_prior = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
Z_sat_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])

E_sym_prior = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
L_sym_prior = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
K_sym_prior = UniformPrior(-300.0, 100.0, parameter_names=["K_sym"])
Q_sym_prior = UniformPrior(-800.0, 800.0, parameter_names=["Q_sym"])
Z_sym_prior = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sym"])

prior_list = [
    E_sym_prior,
    L_sym_prior, 
    K_sym_prior,
    Q_sym_prior,
    Z_sym_prior,

    K_sat_prior,
    Q_sat_prior,
    Z_sat_prior,
]

### CSE priors
if USE_CSE:
    nbreak_prior = UniformPrior(1.0 * 0.16, 2.0 * 0.16, parameter_names=[f"nbreak"])
    prior_list.append(nbreak_prior)
    for i in range(NB_CSE):
        # NOTE: the density parameters are sampled from U[0, 1], so we need to scale it, but it depends on break so will be done internally
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"n_CSE_{i}_u"]))
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))

    # Final point to end
    prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))

# Construct the EOS prior and a transform here which can be used in postprocessing.py
eos_prior = CombinePrior(prior_list)
eos_param_names = eos_prior.parameter_names
all_output_keys = ["logpc_EOS", "masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"]
name_mapping = (eos_param_names, all_output_keys)
my_transform_eos = utils.MicroToMacroTransform(name_mapping,
                                               keep_names = ["E_sym", "L_sym"],
                                               nmax_nsat = NMAX_NSAT,
                                               nb_CSE = NB_CSE
                                               )

class Benchmarker:
    
    def __init__(self, 
                 transform: utils.MicroToMacroTransform,
                 batch_size: int = 5_000,
                 m_min: float = 1.0,
                 nb_masses: int = 200):
        
        self.transform = transform
        self.batch_size = batch_size
        self.m_min = m_min
        self.nb_masses = nb_masses
        
    def interpolate_eos(self, n: jnp.ndarray, p: jnp.ndarray, e: jnp.ndarray):
        return self.transform.eos.interpolate_eos(n, p, e)
        

    def load_koehn_samples(self, nb_samples: int = 1_000):
        """Load the micro and macro samples from the Koehn+ paper, to use for benchmarking purposes"""
        
        masses, radii, Lambdas = [], [], []
        ns_array, ps_array, es_array, hs_array, dloge_dlogps_array = [], [], [], [], []
        
        counter = 0
        for i in range(100_000):
            try: 
                # Load macro
                data = np.genfromtxt(os.path.join(KOEHN_SAMPLES_MACRO, f"{i}.dat"), skip_header=1, delimiter=" ").T
                r, m, l = data[0], data[1], data[2]
                
                if isinstance(m, float):
                    print(f"Skipping {i} since something went wrong. Here is the error message")
                    continue
                
                if len(m) < 5:
                    print(f"Skipping {i} since something went wrong. Here is the error message")
                    continue
                
                # Load micro
                data = np.loadtxt(os.path.join(KOEHN_SAMPLES_MICRO, f"{i}.dat"))
                n, e, p, cs2 = data[:, 0] / 0.16, data[:, 1], data[:, 2], data[:, 3]
                
                ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n, p, e)
                
                # Save all:
                _m = np.linspace(self.m_min, np.max(m), self.nb_masses)
                
                _r = np.interp(_m, m, r)
                _l = np.interp(_m, m, l)
                
                masses.append(_m)
                radii.append(_r)
                Lambdas.append(_l)
                
                ns_array.append(ns)
                ps_array.append(ps)
                hs_array.append(hs)
                es_array.append(es)
                dloge_dlogps_array.append(dloge_dlogps)
                
                # We had a succesful run, so add to counter
                counter += 1
                
            except Exception as e:
                print(f"Skipping {i} since something went wrong. Here is the error message")
                print(e)
                continue
            
            if counter == nb_samples:
                break
            
        # Convert all to jax numpy arrays before returning:
        ns_array = jnp.array(ns_array)
        ps_array = jnp.array(ps_array)
        hs_array = jnp.array(hs_array)
        es_array = jnp.array(es_array)
        dloge_dlogps_array = jnp.array(dloge_dlogps_array)
        masses = jnp.array(masses)
        radii = jnp.array(radii)
        Lambdas = jnp.array(Lambdas)
            
        # Convert it to a batch of EOS tuples
        eos_tuples = ns_array, ps_array, hs_array, es_array, dloge_dlogps_array
            
        return eos_tuples, masses, radii, Lambdas
    
    def benchmark(self, 
                  eos_tuples,
                  masses,
                  radii,
                  Lambdas):
        """Note: masses, radii and Lambdas are taken from the Koehn+ samples"""
        
        start = time.time()
        out = jax.lax.map(self.transform.construct_family_lambda, eos_tuples, batch_size = self.batch_size)
        end = time.time()
        
        print(f"Time taken: {end - start} s")
        
        return out
    
def main():
    
    benchmarker = Benchmarker(my_transform_eos)
    
    eos_tuples, masses, radii, Lambdas = benchmarker.load_koehn_samples()
    out = benchmarker.benchmark(eos_tuples, masses, radii, Lambdas)
    
    
if __name__ == "__main__":
    main()