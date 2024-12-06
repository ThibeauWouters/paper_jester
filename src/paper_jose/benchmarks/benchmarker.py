"""
Randomly sample EOSs and solve TOV for benchmarking purposes
"""

################
### PREAMBLE ###
################

import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
import shutil
import time

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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(jax.devices())

import jax.numpy as jnp
from jimgw.prior import UniformPrior, CombinePrior
from jaxtyping import Array
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
                 mtov_threshold: float = 1.5):
        
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
   
    def find_broken_eos(self, max_nb_eos: int = 10) -> list[int]:
        
        files = os.listdir("./random_samples/")
        idx = [f.split(".")[0] for f in files]
        # Sort them
        sort_idx = np.argsort(idx)
        idx = np.array(idx)[sort_idx]
        files = np.array(files)[sort_idx]
        
        broken_files = []
        
        for file in files:
            full_filename = os.path.join("./random_samples/", file)
            data = np.load(full_filename)
            masses, radii, lambdas = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
            if any(lambdas < 0):
                idx_number = int(file.split(".")[0])
                broken_files.append(idx_number)
                if len(broken_files) > max_nb_eos:
                    break
                
        return broken_files
            
    def debug(self, 
              idx_number: int, 
              figsize = (12, 8),
              save_name: str = "debug_bad"):
        
        # Load from random samples:
        file = os.path.join("./random_samples/", f"{idx_number}.npz")
        data = np.load(file)
        
        # Fetch the parameters
        param_keys = self.prior.parameter_names
        params = {key: data[key] for key in param_keys}
        
        # Solve TOV:
        out = self.transform.forward(params)
        
        # Get the NS properties and check them out
        m, r, l = out["masses_EOS"], out["radii_EOS"], out["Lambdas_EOS"]
        logpc_EOS = out["logpc_EOS"]
        
        print('m')
        print(m)
        
        print('r')
        print(r)
        
        print('l')
        print(l)
        
        print('logpc')
        print(logpc_EOS)
        
        # Show problematic indices
        problematic_lambdas = np.where(l < 0)
        print(f"Problematic lambdas: {problematic_lambdas}")
        print(f"Problematic logpc: {logpc_EOS[problematic_lambdas]}")
        
        n, p, e, h, dloge_dlogp, cs2 = out["n"], out["p"], out["e"], out["h"], out["dloge_dlogp"], out["cs2"]
        
        n = n / jose_utils.fm_inv3_to_geometric / 0.16
        p = p / jose_utils.MeV_fm_inv3_to_geometric
        e = e / jose_utils.MeV_fm_inv3_to_geometric
        
        # EOS
        plt.subplots(nrows = 2, ncols = 2, figsize = figsize)
        plt.subplot(221)
        plt.plot(n, p)
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"$p$ [MeV fm${}^{-3}$]")
        
        plt.subplot(222)
        plt.plot(n, e)
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"$e$ [MeV fm${}^{-3}$]")
        
        plt.subplot(223)
        plt.plot(n, cs2)
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"cs2")
        
        plt.subplot(224)
        plt.plot(n, h)
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"h")
        
        plt.savefig("./figures/debug_eos.png")
        plt.close()
        
        # TOV
        plt.subplots(nrows = 1, ncols = 2, figsize = figsize)
        plt.subplot(121)
        plt.plot(r, m)
        plt.xlabel(r"Radius [km]")
        plt.ylabel(r"Mass [$M_\odot$]")
        
        plt.subplot(122)
        plt.plot(m, l)
        plt.xlabel(r"Mass [$M_\odot$]")
        plt.ylabel(r"Lambda")
        plt.yscale("log")
        plt.savefig(f"./figures/{save_name}_tov.png")
        plt.close() 
    
class CrosscheckSolver:
    
    def __init__(self, 
                 jose_dir: str = "./random_samples/",
                 outdir: str = "./nmma/"):
    
        self.jose_dir = jose_dir
        self.outdir = outdir
        
    def load_and_solve_nmma(self, idx: int) -> None:
        
        """Load precomputed JESTER EOS and construct an NMMA EOS from it"""
        
        # Load data
        og_filename = os.path.join(self.jose_dir, f"{idx}.npz")
        data = np.load(og_filename)
        n, p, e = data["n"], data["p"], data["e"]
        logpc = data["logpc_EOS"]
        m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
        
        # Preprocess for NMMA consumption
        low_density_eos = {"n": n / jose_utils.fm_inv3_to_geometric, 
                           "p": p / jose_utils.MeV_fm_inv3_to_geometric,
                           "e": e / jose_utils.MeV_fm_inv3_to_geometric}
        
        n_max = max(low_density_eos["n"])
        eos_nmma = EOS_with_CSE(low_density_eos, n_lim = n_max)
        pc = jnp.exp(logpc)
        pc = pc / jose_utils.MeV_fm_inv3_to_geometric
        
        start = time.time()
        masses_EOS, radii_EOS, Lambdas_EOS = eos_nmma.construct_family(pc)
        end = time.time()
        
        np.savez(os.path.join(self.outdir, f"{idx}.npz"), timing = end-start, masses_EOS = masses_EOS, radii_EOS = radii_EOS, Lambdas_EOS = Lambdas_EOS)
        
    def load_and_solve_all(self,
                           max_nb: int = 100,
                           with_nmma: bool = True):
        
        pbar = tqdm.tqdm(range(1, max_nb))
        for idx in pbar:
            if with_nmma:
                self.load_and_solve_nmma(idx)
            else:
                raise NotImplementedError("Only NMMA is implemented so far, perhaps can implement another cross check method later on as well")
    
    
def main():
    
    ### Choose to create own prior or can also fetch the one from the utils
    prior = utils.prior
    transform = utils.MicroToMacroTransform(name_mapping = utils.name_mapping)
    benchmarker = Benchmarker(prior = prior, transform = transform, nb_samples=6_000)
    
    ### There is some weird spiky behavior that I need to locate the issue for and debug -- doing this here
    broken_files = benchmarker.find_broken_eos(max_nb_eos = 10)
    print(broken_files)
    benchmarker.debug(broken_files[9])
    # benchmarker.debug(1, save_name="debug_good")
    
    # ### Benchmark jose/jester:
    # benchmarker.random_sample()
    # benchmarker.plot_all_eos()
    
    # ### Use another solver, e.g. NMMA
    # max_nb = 10
    # crosscheck = CrosscheckSolver()
    # crosscheck.load_and_solve_all(max_nb = max_nb)
    
    # ### Diagnosis/cross-check plots
    # benchmarker.make_crosscheck_plots(other_dir = crosscheck.outdir,
    #                                   max_nb=max_nb)
    
if __name__ == "__main__":
    main()