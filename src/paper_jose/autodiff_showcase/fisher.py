"""Code for prospecting the correlations of EOS parameters using some kind of Fisher information matrix approach"""

import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
import shutil

import os
import tqdm
import time
import corner
import numpy as np
import pandas as pd
np.random.seed(42) # for reproducibility
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Union, Callable

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
print(jax.devices())

import jax.numpy as jnp
from jimgw.prior import UniformPrior, CombinePrior
from jaxtyping import Array
import joseTOV.utils as jose_utils

import paper_jose.utils as utils
import paper_jose.inference.utils_plotting as utils_plotting
plt.rcParams.update(utils_plotting.mpl_params)

# TODO: rename
class MyLikelihood:
    
    def __init__(self, 
                 transform: utils.MicroToMacroTransform,
                 R1_4_target: float):
        
        self.transform = transform
        self.R1_4_target = R1_4_target
        print(f"The target R1.4 is: {self.R1_4_target}")
        
    def evaluate(self, 
                 params: dict,
                 sigma_R: float = 0.1):
        
        # Get the R1.4 for this EOS
        macro = self.transform.forward(params)
        m, r = macro["masses_EOS"], macro["radii_EOS"]
        R1_4 = jnp.interp(1.4, m, r)
        
        # Gaussian likelihood:
        log_L = -0.5 * (R1_4 - self.R1_4_target) ** 2 / sigma_R ** 2
        return log_L
    
class Fisher:
    
    def __init__(self,
                 NB_CSE, 
                 filename: str = "my_hessian_values.npz"):
        
        self.filename = filename # to store the Fisher information matrix
    
        ### PRIOR
        my_nbreak = 2.0 * 0.16
        if NB_CSE > 0:
            NMAX_NSAT = 25
        else:
            NMAX_NSAT = 5
        
        NMAX = NMAX_NSAT * 0.16
        width = (NMAX - my_nbreak) / (NB_CSE + 1)

        # NEP priors
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

        # CSE priors
        if NB_CSE > 0:
            prior_list.append(UniformPrior(1.0 * 0.16, 2.0 * 0.16, parameter_names=[f"nbreak"]))
            for i in range(NB_CSE):
                left = my_nbreak + i * width
                right = my_nbreak + (i+1) * width
                prior_list.append(UniformPrior(left, right, parameter_names=[f"n_CSE_{i}"]))
                prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))
            prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))

        # Combine the priors
        self.prior = CombinePrior(prior_list)
        sampled_param_names = self.prior.parameter_names
        name_mapping = (sampled_param_names, ["masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])
        
        # Use it to get a doppelganger score
        self.transform = utils.MicroToMacroTransform(name_mapping, nmax_nsat = NMAX_NSAT, nb_CSE = NB_CSE)
        
        ### Get an R1.4 value as target, Use the center of the prior -- All are uniform priors so this works for now, but be careful
        self.params = {}
        for i, key in enumerate(self.prior.parameter_names):
            base_prior: UniformPrior = self.prior.base_prior[i]
            lower, upper = base_prior.xmin, base_prior.xmax
            self.params[key] = 0.5 * (lower + upper)
            
        out = self.transform.forward(self.params)
        r_target, m_target = out["radii_EOS"], out["masses_EOS"]
        
        R1_4_target = jnp.interp(1.4, m_target, r_target)
        self.likelihood = MyLikelihood(self.transform, R1_4_target)
        

    def compute_hessian_values(self,
                               use_outer_product: bool = False):
        
        if use_outer_product:
            print("WARNING: using outer products")
            # Take the gradient
            grad_fn = jax.grad(self.likelihood.evaluate)
            grad_values = grad_fn(self.params)
            grad_values = np.array(list(grad_values.values()))
            
            # my_hessian_values = np.einsum('ac,bd->abcd', grad_values, grad_values)
            my_hessian_values = np.outer(grad_values, grad_values)
            
        else:
            hessian = jax.hessian(self.likelihood.evaluate)
        
            print("hessian_values")
            hessian_values: dict = hessian(self.params)
            print(hessian_values)
        
            # Extract the Hessian as array
            my_hessian_values = []
            
            for _, row in hessian_values.items():
                for _, value in row.items():
                    my_hessian_values.append(float(value))
                
        names = self.prior.parameter_names
        
        # Dump it:
        np.savez(self.filename, hessian_values=my_hessian_values, names=names)
        
    
    def read_hessian_values(self,
                            take_log: bool = True,
                            verbose: bool = True):
        
        data = np.load(self.filename, allow_pickle = True)
        names = data["names"]
        n_dim = len(names)
        hessian_values = np.reshape(data["hessian_values"], (n_dim, n_dim))
        
        if verbose:
            print("Hessian values")
            print(hessian_values)
        
        plt.figure(figsize = (22, 22))
        if take_log:
            hessian_values = np.log(abs(hessian_values))
            cbar_label = 'log10(Absolute value of Hessian)'
        else:
            cbar_label = 'Hessian'
            
        plt.imshow(hessian_values, cmap='viridis', interpolation='none')
        cbar = plt.colorbar(shrink = 0.85)
        cbar.set_label(label=cbar_label, size = 24)
        plt.grid(False)
        plt.xticks(range(n_dim), data["names"], rotation = 90, fontsize = 24)
        plt.yticks(range(n_dim), data["names"], fontsize = 24)
        plt.savefig("./figures/hessian.png", bbox_inches='tight')
        plt.close()
    
# TODO: remove me
def compare_hessians():
    data = np.load("my_hessian_values.npz", allow_pickle = True)
    hessian_og = data["hessian_values"]
    N = int(np.sqrt(len(hessian_og)))
    hessian_og = hessian_og.reshape((N, N))
    
    data = np.load("my_hessian_values_outer.npz", allow_pickle = True)
    hessian_outer = data["hessian_values"]
    
    print("First element check")
    print(hessian_og[0, 0])
    print(hessian_outer[0, 0])
    
    print("Differences")
    print((hessian_og - hessian_outer) / hessian_og)
    
def main():
    
    ### Compute the Fisher
    fisher = Fisher(NB_CSE = 0, filename = "my_hessian_values.npz")
    
    fisher.compute_hessian_values()
    fisher.read_hessian_values()
    
    # compare_hessians() # TODO: implement this, check whether the Hessian and the outer products agree indeed
    
    print("DONE")
    
if __name__ == "__main__":
    main()