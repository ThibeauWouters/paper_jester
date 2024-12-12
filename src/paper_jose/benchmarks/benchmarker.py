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

import jax.numpy as jnp
from jimgw.prior import CombinePrior
import joseTOV.utils as jose_utils

import paper_jose.utils as utils
import paper_jose.inference.utils_plotting as utils_plotting
plt.rcParams.update(utils_plotting.mpl_params)
import seaborn as sns

from paper_jose.benchmarks.nmma_tov import EOS_with_CSE


class Benchmarker:
    
    """A class to randomly sample and solve for some EOSs"""
    
    def __init__(self,
                 prior: CombinePrior,
                 transform: utils.MicroToMacroTransform,
                 outdir: str = "./random_samples/",
                 random_seed: int = 0,
                 nb_samples: int = 2_000,
                 mtov_threshold: float = 2.1):
        
        self.prior = prior
        self.transform = transform
        self.jax_key = jax.random.PRNGKey(random_seed)
        self.outdir = outdir
        self.nb_samples = nb_samples
        self.mtov_threshold = mtov_threshold
        
    def initialize_walkers(self) -> dict:
        """
        Initialize the walker parameters in the EOS space given the random seed.

        Returns:
            dict: Dictionary of the starting parameters.
        """
        
        self.jax_key, jax_subkey = jax.random.split(self.jax_key)
        params = self.prior.sample(jax_subkey, 1)
            
        # This is needed, otherwise JAX will scream
        for key, value in params.items():
            if isinstance(value, jnp.ndarray):
                params[key] = value.at[0].get()
                        
        return params

    def plot_all_eos(self, 
                     color = "blue", 
                     use_mask = True,
                     verbose: bool = False):
        
        plt.subplots(nrows = 1, ncols = 2, figsize = (12, 8))
        files = os.listdir(self.outdir)
        mtov_list = np.zeros(len(files))
        r14_list = np.zeros(len(files))
        
        for i, file in enumerate(files):
            data = np.load(os.path.join(self.outdir, file))
            masses, radii, lambdas = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
            mask = radii < 20
            n_false = len(mask) - np.sum(mask)
            if verbose:
                print(f"EOS {i} has {n_false} radii > 20 km")
            if use_mask:
                masses, radii, lambdas = masses[mask], radii[mask], lambdas[mask]
                
            mtov_list[i] = np.max(masses)
            r14_list[i] = np.interp(1.4, masses, radii)
            
            plt.subplot(1, 2, 1)
            plt.plot(radii, masses, color = color, alpha = 0.1)
            plt.xlabel(r"Radius [km]")
            plt.ylabel(r"Mass [$M_\odot$]")
            plt.subplot(1, 2, 2)
            plt.plot(masses, lambdas, color = color, alpha = 0.1)
            plt.xlabel(r"Radius [km]")
            plt.ylabel(r"Lambda")
            plt.yscale("log")
            
        plt.savefig("./figures/all_eos.png")
        plt.close()
        
        # Make a histogram of MTOV
        plt.figure(figsize = (12, 6))
        plt.hist(mtov_list, bins = 20, color = color, density = True, histtype = "step", linewidth = 4)
        plt.xlabel(r"$M_{\rm TOV}$ [$M_\odot$]")
        plt.ylabel("Density")
        plt.savefig("./figures/mtov_hist.png")
        plt.close()
        
        # Make a histogram of MTOV
        plt.figure(figsize = (12, 6))
        plt.hist(r14_list, bins = 20, color = color, density = True, histtype = "step", linewidth = 4)
        plt.xlabel(r"$R_{1.4}$ [km]")
        plt.ylabel("Density")
        plt.savefig("./figures/r14_hist.png")
        plt.close()
        

    def random_sample(self):
        """
        Generate a sample from the prior, solve the TOV equations and return the results.        
        """
        
        print("Generating random samples for a batch of EOS . . . ")
        
        files = os.listdir(self.outdir)
        counter = len(files)
        
        pbar = tqdm.tqdm(range(self.nb_samples))
        for i in pbar:
            params = self.initialize_walkers()
            out = self.transform.forward(params)
            logpc, m, r, l = out["logpc_EOS"], out["masses_EOS"], out["radii_EOS"], out["Lambdas_EOS"]
            n, p, e, h, dloge_dlogp, cs2 = out["n"], out["p"], out["e"], out["h"], out["dloge_dlogp"], out["cs2"]
            
            if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)):
                print(f"Sample {i} has NaNs. Skipping this sample")
                continue
            
            # Only save if the TOV mass is high enough
            if jnp.max(m) > self.mtov_threshold:
                npz_filename = os.path.join(self.outdir, f"{counter}.npz")
                save_dict = {**params, **out}
                np.savez(npz_filename, **save_dict)
                counter += 1
                
            pbar.set_description(f"Sample: {i} MTOV: {np.max(m)} Counter: {counter}")
            
        return
    
    def make_crosscheck_plots(self, 
                              other_dir: str,
                              max_nb: int = 100,
                              colors: list[str] = ["blue", "red"],
                              other_label: str = "NMMA"):
        
        pbar = tqdm.tqdm(range(1, max_nb))
        for i in pbar:
            # Load jose result:
            jose_filename = os.path.join(self.outdir, f"{i}.npz")
            jose_data = np.load(jose_filename)
            m_jose, r_jose, l_jose = jose_data["masses_EOS"], jose_data["radii_EOS"], jose_data["Lambdas_EOS"]
            
            # Load other result:
            other_filename = os.path.join(other_dir, f"{i}.npz")
            other_data = np.load(other_filename, allow_pickle=True)
            m_other, r_other, l_other = other_data["masses_EOS"], other_data["radii_EOS"], other_data["Lambdas_EOS"]
            
            plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(r_jose, m_jose, color = colors[0], label = "JESTER")
            plt.plot(r_other, m_other, color = colors[1], linestyle = "--", label = other_label)
            plt.xlabel(r"Radius [km]")
            plt.ylabel(r"Mass [$M_\odot$]")
            
            plt.subplot(1, 2, 2)
            plt.plot(m_jose, l_jose, color = colors[0])
            plt.plot(m_other, l_other, color = colors[1], linestyle = "--", label = other_label)
            plt.xlabel(r"Mass [$M_\odot$]")
            plt.ylabel(r"Lambda")
            plt.yscale("log")
            
            plt.legend()
            plt.savefig(f"./figures/crosschecks/{i}.png")
            plt.close()
    
    
class CrosscheckSolver:
    
    def __init__(self, 
                 jose_dir: str = "./random_samples/",
                 outdir: str = "./nmma/"):
    
        self.jose_dir = jose_dir
        self.outdir = outdir
        
    def load_eos(self, idx: int) -> dict:
        """
        Load precomputed jose EOS and construct a dict from it that is used for TOV solver consumption (either jose or NMMA)
        
        Args:
            idx (int): The index of the EOS to load, taken from the random samples directory
        Returns:
            dict: The EOS dict that can be used for the TOV solver. Keys are n, p, e, h, dloge_dlogp, logpc_EOS
        """
        
        # Load data
        og_filename = os.path.join(self.jose_dir, f"{idx}.npz")
        data = np.load(og_filename)
        n, p, e, h, dloge_dlogp = data["n"], data["p"], data["e"], data["h"], data["dloge_dlogp"]
        logpc_EOS = data["logpc_EOS"]
        
        # Preprocess for NMMA consumption
        eos_dict = {"n": n / jose_utils.fm_inv3_to_geometric, 
                    "p": p / jose_utils.MeV_fm_inv3_to_geometric,
                    "e": e / jose_utils.MeV_fm_inv3_to_geometric,
                    "h": h,
                    "dloge_dlogp": dloge_dlogp,
                    "logpc_EOS": logpc_EOS}
        
        return eos_dict
    
    def solve_nmma(self, eos_dict: dict) -> tuple:
        """
        Solve the TOV equations using NMMA for a given EOS dict

        Args:
            eos_dict (dict): Dictionary containing the EOS parameters with keys n, p, e, h, dloge_dlogp, logpc_EOS

        Returns:
            tuple: Masses, radii and Lambdas of the EOS as well as timing for the solver part
        """
        n_max = max(eos_dict["n"])
        eos_nmma = EOS_with_CSE(eos_dict, n_lim = n_max)
        
        pc = jnp.exp(eos_dict["logpc_EOS"])
        pc = pc / jose_utils.MeV_fm_inv3_to_geometric
        
        start = time.time()
        masses_EOS, radii_EOS, Lambdas_EOS = eos_nmma.construct_family(pc)
        end = time.time()
        
        timing = end - start
        
        return masses_EOS, radii_EOS, Lambdas_EOS, timing
        
    def load_and_solve_nmma(self, idx: int) -> None:
        """
        Loads a precomputed jose EOS and solves the TOV equations using NMMA, see self.load_eos and self.solve_nmma for more details
        """
        eos_dict = self.load_eos(idx)
        masses_EOS, radii_EOS, Lambdas_EOS, timing= self.solve_nmma(eos_dict)
        
        np.savez(os.path.join(self.outdir, f"{idx}.npz"), timing = timing, masses_EOS = masses_EOS, radii_EOS = radii_EOS, Lambdas_EOS = Lambdas_EOS)
        
    def solve_jose(self, eos_dict: dict):
        raise NotImplementedError("Still have to implement this")
        # # Convert to tuple
        # ns, ps, hs, es, dloge_dlogps = eos_dict[]
        
    def load_and_solve_nmma(self, idx: int) -> None:
        """
        Loads a precomputed jose EOS and solves the TOV equations using NMMA, see self.load_eos and self.solve_nmma for more details
        """
        eos_dict = self.load_eos(idx)
        masses_EOS, radii_EOS, Lambdas_EOS, timing= self.solve_nmma(eos_dict)
        
        np.savez(os.path.join(self.outdir, f"{idx}.npz"), timing = timing, masses_EOS = masses_EOS, radii_EOS = radii_EOS, Lambdas_EOS = Lambdas_EOS)
        
        
    def load_and_solve_sequence(self,
                           max_nb: int = 100,
                           solver_name: str = "jose"):
        """
        Load and TOV-solve all of the desired EOS using either jose or NMMA.

        Args:
            max_nb (int, optional): Maximal number of EOS to solve for. Defaults to 100.
            solver_name (str, optional): Name of the solver, either jose or nmma (lowercase). Defaults to "jose".

        Raises:
            ValueError: In case the solver name is not supported
            NotImplementedError: Still have to implement jose
        """
        
        supported_solver_names = ["jose", "nmma"]
        solver_name = solver_name.lower()
        if solver_name not in supported_solver_names:
            raise ValueError(f"Solver name {solver_name} not supported. Choose from {supported_solver_names}")
        
        pbar = tqdm.tqdm(range(1, max_nb))
        for idx in pbar:
            if solver_name == "nmma":
                self.load_and_solve_nmma(idx)
            elif solver_name == "jose":
                self.load_and_solve_jose(idx)
                
            else:
                raise NotImplementedError("Only NMMA is implemented so far, perhaps can implement another cross check method later on as well")
            
def main():
    
    # Get the args passed from the command line
    args = sys.argv[1:]
    if len(args) != 4:
        raise ValueError("Usage: python benchmarker.py <platform_name> <jit> <vmap> <solver_name>")
    
    platform_name = args[0]
    use_jit = args[1]
    use_vmap = args[2]
    solver_name = args[3]
    
    print(f"Running benchmarker with:")
    print(f"    - platform_name: {platform_name}")
    print(f"    - use_jit: {use_jit}")
    print(f"    - use_vmap: {use_vmap}")
    print(f"    - solver_name: {solver_name}")
    
    jax.config.update('jax_platform_name', platform_name)
    
    ### Choose to create own prior or can also fetch the one from the utils
    prior = utils.prior
    transform = utils.MicroToMacroTransform(name_mapping=utils.name_mapping)
    benchmarker = Benchmarker(prior=prior, transform=transform, nb_samples=10_000)
    
    ### Create the random samples that we will use for benchmarking
    # benchmarker.random_sample()
    # benchmarker.plot_all_eos()
    
    ### Use another solver, e.g. NMMA
    max_nb = 10
    crosscheck = CrosscheckSolver()
    crosscheck.load_and_solve_all(max_nb = max_nb)
    
    ### Diagnosis/cross-check plots
    benchmarker.make_crosscheck_plots(other_dir = crosscheck.outdir,
                                      max_nb=max_nb)
    
if __name__ == "__main__":
    main()