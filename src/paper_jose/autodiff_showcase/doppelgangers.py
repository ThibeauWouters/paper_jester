"""
Playground for testing the possibilities of EOS exploration with jose.
"""

################
### PREAMBLE ###
################

import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"
import shutil

import os
import tqdm
import time
import corner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
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

import paper_jose.utils as utils
import paper_jose.inference.utils_plotting as utils_plotting
plt.rcParams.update(utils_plotting.mpl_params)

from paper_jose.autodiff_showcase.evolutionary_optimizer import EvolutionaryOptimizer
import seaborn as sns

##########################
### OPTIMIZATION CLASS ###
##########################
class OptimizationRun:
    
    def __init__(self,
                 score_fn: Callable,
                 prior: CombinePrior,
                 nb_walkers: int = 1,
                 nb_steps: int = 200,
                 optimization_sign: float = -1, 
                 learning_rate: float = 1e-3, 
                 start_halfway: bool = True,
                 random_seed: int = 42,
                 # Plotting
                 outdir_name: str = "computed_data",
                 plot_mse: bool = True,
                 plot_final_errors: bool = True,
                 plot_target: bool = True,
                 # Target
                 m_target: Array = None,
                 r_target: Array = None,
                 Lambdas_target: Array = None,
                 ):
        
        self.score_fn = score_fn
        self.prior = prior
        self.nb_steps = nb_steps
        self.nb_walkers = nb_walkers
        
        self.optimization_sign = optimization_sign
        self.learning_rate = learning_rate
        if self.nb_walkers > 1:
            # Option to start halfway only works if there is exactly one walker
            start_halfway = False
        self.start_halfway = start_halfway
        self.random_seed = random_seed
        self.outdir_name = outdir_name
        
        # Choose which type of run: single or vmap
        if self.nb_walkers == 1:
            self.run = self.run_single
        else:
            self.run = self.run_vmap
            
        self.m_target = m_target
        self.r_target = r_target
        self.Lambdas_target = Lambdas_target
            
        # Clean the outdir(s)
        if nb_walkers > 1:
            for i in range(self.nb_walkers):
                subdir = f"./{self.outdir_name}_{i}/"
                shutil.rmtree(subdir, ignore_errors=True)
                os.makedirs(subdir)
                os.makedirs(f"{subdir}figures/")
        else:
            # shutil.rmtree(self.outdir_name, ignore_errors=True)
            if not os.path.exists(self.outdir_name):
                print("Creating the outdir")
                os.makedirs(self.outdir_name)
                os.makedirs(f"{self.outdir_name}figures/")
            
        self.plot_mse = plot_mse
        self.plot_final_errors = plot_final_errors
        self.plot_target = plot_target
        
    def initialize_walkers(self):
        
        if self.start_halfway and self.nb_walkers == 1:
            params = {}
            # All are uniform priors so this works for now, but be careful, might break later on
            for i, key in enumerate(self.prior.parameter_names):
                base_prior: UniformPrior = self.prior.base_prior[i]
                lower, upper = base_prior.xmin, base_prior.xmax
                params[key] = 0.5 * (lower + upper)

        else:
            jax_key = jax.random.PRNGKey(self.random_seed)
            jax_key, jax_subkey = jax.random.split(jax_key)
            params = self.prior.sample(jax_subkey, self.nb_walkers)
            
        if self.nb_walkers == 1:
            # This is needed, otherwise JAX will scream
            for key, value in params.items():
                if isinstance(value, jnp.ndarray):
                    params[key] = value.at[0].get()
                        
        return params
        
    def run_single(self,
                   params: dict):
        """
        Compute the gradient ascent or descent (just call it descent here for simplicity) in order to find the doppelgangers in the EOS space.
        
        TODO: change to array, not to dict, if needed?
        """
        
        print("Starting parameters:")
        print(params)
        
        # Define the score function in the desired jax format
        self.score_fn = jax.value_and_grad(self.score_fn, has_aux=True)
        self.score_fn = jax.jit(self.score_fn)
        
        failed_counter = 0
        
        print("Computing by gradient ascent . . .")
        pbar = tqdm.tqdm(range(self.nb_steps))
        for i in pbar:
            ((score, aux), grad) = self.score_fn(params)
            m, r, l = aux
            
            if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)):
                print(f"Iteration {i} has NaNs. Exiting the computing loop now")
                break
            
            pbar.set_description(f"Iteration {i}: score = {score}")
            np.savez(f"{self.outdir_name}{i}.npz", masses_EOS = m, radii_EOS = r, Lambdas_EOS = l, score = score, **params)
            
            params = {key: value + self.optimization_sign * self.learning_rate * grad[key] for key, value in params.items()}
            
            max_error = compute_max_error(m, l, self.m_target, self.Lambdas_target)
            if max_error < 10.0:
                print(f"Early stopping at iteration {i} with max error: {max_error}")
                break
            
        print("Computing DONE")
        
        return None
    
    def run_vmap(self, params: dict):
        """
        Compute the gradient ascent or descent (just call it descent here for simplicity) in order to find the doppelgangers in the EOS space.
        """
            
        print("Starting parameters:")
        print(params)
        
        # Define the score function in the desired jax format
        self.score_fn = jax.value_and_grad(self.score_fn, has_aux=True)
        self.score_fn = jax.vmap(jax.jit(self.score_fn))
        
        failed_counter = 0
        
        print("Computing by gradient descent . . .")
        for i in tqdm.tqdm(range(self.nb_steps)):
            
            ((score, aux), grad) = self.score_fn(params)
            m, r, l = aux
            
            if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)):
                print(f"Iteration {i} has NaNs")
                
                failed_counter += 1
                print(f"Skipping")
                continue
            
            print(f"Iteration {i}: score = {score}")
            
            # Save it
            for j in range(self.nb_walkers):
                np.savez(f"./{self.outdir_name}_{j}/{i}.npz", masses_EOS = m[j], radii_EOS = r[j], Lambdas_EOS = l[j], score = score[j], **params)
            
            print("grads")
            print(grad)
            
            print("params")
            print(params)
            
            params = {key: value + self.optimization_sign * self.learning_rate * grad[key] for key, value in params.items()}
            
        print("Computing DONE")
        return None
    
    ################
    ### PLOTTING ###
    ################
    
    def plot_all_NS(self):
        for i in range(self.nb_walkers):
            subdir = f"./{self.outdir_name}_{i}/"
            self.plot_NS(subdir)
            
    def plot_NS(self, subdir: str, m_min = 1.2):
    
        # Read the EOS data
        all_masses_EOS = []
        all_radii_EOS = []
        all_Lambdas_EOS = []

        for i in range(self.nb_steps):
            try:
                data = np.load(f"{subdir}{i}.npz")
                
                masses_EOS = data["masses_EOS"]
                radii_EOS = data["radii_EOS"]
                Lambdas_EOS = data["Lambdas_EOS"]
                
                if not np.any(np.isnan(masses_EOS)) and not np.any(np.isnan(radii_EOS)) and not np.any(np.isnan(Lambdas_EOS)):
                
                    all_masses_EOS.append(masses_EOS)
                    all_radii_EOS.append(radii_EOS)
                    all_Lambdas_EOS.append(Lambdas_EOS)
                
            except FileNotFoundError:
                continue
            
        # N might have become smaller if we hit NaNs at some point
        N_max = len(all_masses_EOS)
        norm = mpl.colors.Normalize(vmin=0, vmax=N_max)
        # cmap = sns.color_palette("rocket_r", as_cmap=True)
        cmap = mpl.cm.viridis
            
        # Plot the target
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 6))
        plt.subplot(121)
        plt.plot(self.r_target, self.m_target, color = "red", zorder = 1e10)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M \ [M_\odot]$")
        plt.subplot(122)
        plt.xlabel(r"$M \ [M_\odot]$")
        plt.ylabel(r"$\Lambda$")
        plt.plot(self.m_target, self.Lambdas_target, label=r"$\Lambda$", color = "red", zorder = 1e10)
        plt.yscale("log")
        
        for i in range(N_max):
            color = cmap(norm(i))
            
            # Mass-radius plot
            plt.subplot(121)
            plt.plot(all_radii_EOS[i], all_masses_EOS[i], color=color, linewidth = 2.0, zorder=i)
                
            # Mass-Lambdas plot
            plt.subplot(122)
            plt.plot(all_masses_EOS[i], all_Lambdas_EOS[i], color=color, linewidth = 2.0, zorder=i)
            
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs[-1])
        cbar.set_label(r'Iteration number', fontsize = 22)
            
        plt.tight_layout()
        save_name = f"{subdir}figures/doppelganger_trajectory.png"
        print(f"Saving to: {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        plt.close()
        
        # Also plot the progress of the errors
        if self.plot_mse:
            plt.figure(figsize = (12, 6))
            mse_errors = []
            for i in range(N_max):
                data = np.load(f"{subdir}{i}.npz")
                mse_errors.append(data["score"])
                
            # Plot
            nb = [i+1 for i in range(len(mse_errors))]
            plt.plot(nb, mse_errors, color="black")
            plt.scatter(nb, mse_errors, color="black")
            plt.xlabel("Iteration number")
            plt.ylabel("MSE")
            plt.yscale("log")
            
            plt.savefig(f"{subdir}figures/mse_errors.png", bbox_inches = "tight")
            plt.close()
            
        if self.plot_final_errors:
            plt.figure(figsize = (12, 6))
            # Plot the errors of the final M, Lambda, R
            m_final = all_masses_EOS[-1]
            r_final = all_radii_EOS[-1]
            Lambda_final = all_Lambdas_EOS[-1]
            
            my_m_min = max(min(m_final), min(self.m_target))
            my_m_min = max(my_m_min, m_min)
            my_m_max = min(max(m_final), max(self.m_target))
            
            masses = jnp.linspace(my_m_min, my_m_max, 500)
            my_Lambdas_model = jnp.interp(masses, m_final, Lambda_final, left = 0, right = 0)
            my_Lambdas_target = jnp.interp(masses, self.m_target, self.Lambdas_target, left = 0, right = 0)
            
            # my_r_model = jnp.interp(masses, m_final, r_final, left = 0, right = 0)
            # my_r_target = jnp.interp(masses, m_target, r_target, left = 0, right = 0)
            
            errors = abs(my_Lambdas_model - my_Lambdas_target)
            max_error = max(errors)
            plt.plot(masses, errors, color = "black")
            plt.xlabel(r"$M \ [M_\odot]$")
            plt.ylabel(r"$\Delta \Lambda \ (L_\infty)$ ")
            plt.yscale("log")
            plt.title(f"Max error: {max_error}")
            save_name = f"{subdir}figures/final_errors.png"
            print(f"Saving to: {save_name}")
            plt.savefig(save_name, bbox_inches = "tight")
            
            plt.close()
            
    def plot_all_EOS(self):
        for i in range(self.nb_walkers):
            subdir = f"./{self.outdir_name}_{i}/"
            self.plot_EOS(subdir)
                
    def plot_EOS(self, subdir: str):
    
        parameter_names = self.prior.parameter_names
        eos_trajectory = {name: [] for name in parameter_names}
        
        for i in range(self.nb_steps):
            try:
                data = np.load(f"{subdir}{i}.npz")
                for name in parameter_names:
                    eos_trajectory[name].append(data[name])
            except FileNotFoundError:
                continue
                
        for name in parameter_names:
            values = eos_trajectory[name]
            plt.figure(figsize = (12, 6))
            plt.plot(values, color = "black")
            plt.xlabel("Iteration number")
            plt.title(name)
            save_name = f"{subdir}figures/trajectory_{name}.png"
            print(f"Saving to: {save_name}")
            plt.savefig(save_name, bbox_inches = "tight")
            plt.close()
            
            
class Result:
    
    def __init__(self, 
                 outdir: str,
                 optimizer: OptimizationRun,
                 m_target: Array,
                 r_target: Array,
                 Lambdas_target: Array,
                 n_target: Array,
                 p_target: Array,
                 e_target: Array,
                 cs2_target: Array):
    
        self.outdir = outdir
        self.optimizer = optimizer
        
        self.m_target = m_target
        self.r_target = r_target
        self.Lambdas_target = Lambdas_target
        
        self.n_target = n_target / 0.16
        self.p_target = p_target
        self.e_target = e_target
        self.cs2_target = cs2_target
        
    def show_table(self):
        
        subdirs = os.listdir(self.outdir)
        output = defaultdict(list)
        
        for subdir in subdirs:
            # Will save everything in a dict here
            output["subdir"].append(subdir)
            
            npz_files = [f for f in os.listdir(f"{self.outdir}/{subdir}") if f.endswith(".npz")]
            numbers = [int(file.split(".")[0]) for file in npz_files]
            final_numer = max(numbers)
            
            # Load the data
            data = np.load(f"{self.outdir}/{subdir}/{final_numer}.npz")
            keys: list[str] = data.keys()
            for key in keys:
                if key.endswith("_EOS"):
                    continue
                output[key].append(float(data[key]))
            
            # TODO: work in progress
            # Macro output: needs a bit more work
            m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
            
            max_error = compute_max_error(m, l, self.m_target, self.Lambdas_target)
            
            # Get Lambda 1.4 error:
            Lambda_1_4_model = jnp.interp(1.4, m, l, left = 0, right = 0)
            Lambda_1_4_target = jnp.interp(1.4, self.m_target, self.Lambdas_target, left = 0, right = 0)
            error_1_4 = abs(Lambda_1_4_model - Lambda_1_4_target)
            
            # Add to output:
            output["max_error"].append(max_error)
            output["error_1_4"].append(error_1_4)

        df = pd.DataFrame(output)
        # Sort based on score, lower to upper:
        df = df.sort_values("max_error")
        
        print("Postprocessing table:")
        print(df)
        
        self.df = df
        
        return output
    
    def plot_doppelgangers(self):
        
        ### First the NS
        
        doppelgangers_dict = {}
        for subdir in os.listdir(self.outdir):
            full_subdir = os.path.join(self.outdir, subdir)

            # Get the final
            npz_files = [f for f in os.listdir(full_subdir) if f.endswith(".npz")]
            if len(npz_files) == 0:
                
                raise ValueError("No npz files found in {}".format(full_subdir))

            ids = [int(f.split(".")[0]) for f in npz_files]
            final_id = max(ids)

            # Final npz
            final_npz = os.path.join(full_subdir, "{}.npz".format(final_id))

            # Load it
            data = np.load(final_npz)
            keys = list(data.keys())

            doppelgangers_dict[subdir] = {}
            for key in keys:
                doppelgangers_dict[subdir][key] = data[key]
                
        # Make the plot
        plt.subplots(figsize=(14, 8), nrows = 1, ncols = 2)

        plt.subplot(121)
        plt.plot(self.r_target, self.m_target, color="black", linewidth = 4, label = "Target")
        for key in doppelgangers_dict.keys():
            r, m = doppelgangers_dict[key]["radii_EOS"], doppelgangers_dict[key]["masses_EOS"]
            plt.plot(r, m)

        plt.xlim(10, 15)
        plt.ylim(0.5, 2.5)
        plt.xlabel(r"$r$ [km]")
        plt.ylabel(r"$M/M_{\odot}$")
        plt.grid(True)

        plt.subplot(122)
        plt.plot(self.m_target, self.Lambdas_target, color="black", linewidth = 4, label = "Target")
        for key in doppelgangers_dict.keys():
            m, l = doppelgangers_dict[key]["masses_EOS"], doppelgangers_dict[key]["Lambdas_EOS"]
            label = f"id = {key}"
            plt.plot(m, l, label=label)
        plt.xlabel(r"$M/M_{\odot}$")
        plt.ylabel(r"$\Lambda$")
        plt.yscale("log")
        plt.grid(True)
        plt.xlim(0.5, 2.5)
        plt.ylim(top = 1e5)
        plt.legend()
        plt.savefig("./figures/doppelgangers_NS.png", bbox_inches = "tight")
        plt.savefig("./figures/doppelgangers_NS.pdf", bbox_inches = "tight")
        plt.close()
        
        ### Second: the EOS
        
        param_names = self.optimizer.prior.parameter_names
        
        # Get the EOS
        for max_nsat, extra_id in zip([25.0, 2.0], ["MM_CSE", "MM"]):
            plt.subplots(figsize = (14, 10), nrows = 1, ncols = 2)
            for key in doppelgangers_dict.keys():
                
                label = f"id = {key}"
                params = {k: doppelgangers_dict[key][k] for k in param_names}
                
                out = self.optimizer.transform.forward(params)
                
                n = out["n"] / jose_utils.fm_inv3_to_geometric / 0.16
                e = out["e"] / jose_utils.MeV_fm_inv3_to_geometric
                p = out["p"] / jose_utils.MeV_fm_inv3_to_geometric
                cs2 = out["cs2"]
                
                # Limit everything to be up to the maximum saturation density
                mask = n < max_nsat
                n, e, p, cs2 = n[mask], e[mask], p[mask], cs2[mask]
                
                mask_target = self.n_target < max_nsat
                n_target, e_target, p_target, cs2_target = self.n_target[mask_target], self.e_target[mask_target], self.p_target[mask_target], self.cs2_target[mask_target]
                
                plt.subplot(221)
                plt.plot(n, e, label = label)
                plt.plot(n_target, e_target, color = "black", label = "Target")
                plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
                plt.ylabel(r"$e$ [MeV fm$^{-3}$]")
                
                plt.subplot(222)
                plt.plot(n, p, label = label)
                plt.plot(n_target, p_target, color = "black", label = "Target")
                plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
                plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
                
                plt.subplot(223)
                plt.plot(n, cs2, label = label)
                plt.plot(n_target, cs2_target, color = "black", label = "Target")
                plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
                plt.ylabel(r"$c_s^2$")
                plt.ylim(0, 1)
                
            plt.savefig(f"./figures/doppelgangers_EOS_{extra_id}.png", bbox_inches = "tight")
            plt.savefig(f"./figures/doppelgangers_EOS_{extra_id}.pdf", bbox_inches = "tight")
            plt.close()
            
        ### Now need to plot the EOS parameters:
        param_names_MM = [n for n in param_names if n.endswith("_sat") or n.endswith("_sym")]
        param_names_MM += ["subdir"]
        
        sns.pairplot(self.df[param_names_MM], hue = "subdir", plot_kws={"s": 100})
        plt.savefig("./figures/doppelgangers_EOS_params.png", bbox_inches = "tight")
        plt.savefig("./figures/doppelgangers_EOS_params.pdf", bbox_inches = "tight")
        plt.close()

#################
### SCORE FNs ###
#################

# Get maximum error in Lambda

def compute_max_error(mass_1,
                      Lambdas_1,
                      mass_2,
                      Lambdas_2):
    
    masses = jnp.linspace(1.2, 2.1, 500)
    my_Lambdas_model = jnp.interp(masses, mass_1, Lambdas_1, left = 0, right = 0)
    my_Lambdas_target = jnp.interp(masses, mass_2, Lambdas_2, left = 0, right = 0)
    errors = abs(my_Lambdas_model - my_Lambdas_target)
    return max(errors)

def mrse(x, y):
    return jnp.mean(((x - y) / y) ** 2)

def mrae(x, y):
    return jnp.mean(jnp.abs((x - y) / y))

def doppelganger_score(params: dict,
                       transform: utils.MicroToMacroTransform,
                       m_target: Array,
                       Lambdas_target: Array, 
                       r_target: Array,
                       m_min = 1.2,
                       m_max = 2.1,
                       N_masses: int = 100,
                       alpha: float = 1.0,
                       beta: float = 0.0,
                       gamma: float = 2.0,
                       delta: float = 0.0,
                       return_aux: bool = True,
                       error_fn: Callable = mrse) -> float:
    
    # Solve the TOV equations
    out = transform.forward(params)
    m_model, r_model, Lambdas_model = out["masses_EOS"], out["radii_EOS"], out["Lambdas_EOS"]
    
    mtov_model = m_model[-1]
    mtov_target = m_target[-1]
    
    # Get a mass array and interpolate NaNs on top of it TODO: make argument
    masses = jnp.linspace(m_min, m_max, N_masses)
    my_Lambdas_model = jnp.interp(masses, m_model, Lambdas_model, left = 0, right = 0)
    my_Lambdas_target = jnp.interp(masses, m_target, Lambdas_target, left = 0, right = 0)
    
    my_r_model = jnp.interp(masses, m_model, r_model, left = 0, right = 0)
    my_r_target = jnp.interp(masses, m_target, r_target, left = 0, right = 0)
    
    # Define separate scores
    score_lambdas = error_fn(my_Lambdas_model, my_Lambdas_target)
    score_r = error_fn(my_r_model, my_r_target)
    score_mtov = error_fn(mtov_model, mtov_target)
    
    # TODO: remove this?
    target_1_4 = jnp.interp(1.4, masses, my_Lambdas_target, left = 0, right = 0)
    model_1_4 = jnp.interp(1.4, masses, my_Lambdas_model, left = 0, right = 0)
    score_1_4 = jnp.max(jnp.abs((target_1_4 - model_1_4) / model_1_4))
    
    score = alpha * score_lambdas + beta * score_r + gamma * score_mtov + delta * score_1_4
    
    if return_aux:
        return score, (m_model, r_model, Lambdas_model)
    else:
        return score

       
######################
### BODY FUNCTIONS ### 
######################

def run_optimizer(metamodel_only: bool = False,
                  N_runs: int = 1):
    """
    Optimize a single EOS, mainly for testing the framework
    
    Args:
        method (str, optional): Either "single" or not "single". Defaults to "single".
    """
    
    ### PRIOR
    my_nbreak = 2.0 * 0.16
    if metamodel_only:
        NMAX_NSAT = 5
        NB_CSE = 0
    else:
        NMAX_NSAT = 25
        NB_CSE = 8
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

    if not metamodel_only:
        # CSE priors
        prior_list.append(UniformPrior(1.0 * 0.16, 2.0 * 0.16, parameter_names=[f"nbreak"]))
        for i in range(NB_CSE):
            left = my_nbreak + i * width
            right = my_nbreak + (i+1) * width
            prior_list.append(UniformPrior(left, right, parameter_names=[f"n_CSE_{i}"]))
            prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))
    
    # Combine the prior
    prior = CombinePrior(prior_list)
    
    # Use it to get a doppelganger score
    sampled_param_names = prior.parameter_names
    name_mapping = (sampled_param_names, ["masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])
    transform = utils.MicroToMacroTransform(name_mapping, 
                                            nmax_nsat=NMAX_NSAT,
                                            nb_CSE=NB_CSE,
                                            )
    
    # Get ready:
    target_filename = "./36022_macroscopic.dat"
    target_eos = np.genfromtxt(target_filename, skip_header=1, delimiter=" ").T
    r_target, m_target, Lambdas_target = target_eos[0], target_eos[1], target_eos[2]
    doppelganger_score_ = lambda params: doppelganger_score(params, transform, m_target, Lambdas_target, r_target)
    
    # Initialize the optimizer in case we have N_Runs = 0 so we can still use the prior etc
    optimizer = OptimizationRun(doppelganger_score_, 
                                prior,
                                nb_steps = 200,
                                learning_rate = 0.001,
                                nb_walkers = 1,
                                start_halfway=False,
                                random_seed=42,
                                outdir_name=f"./outdir_doppelganger/42/",
                                m_target = m_target,
                                r_target = r_target,
                                Lambdas_target = Lambdas_target)
    
    optimizer.transform = transform
    
    for i in range(N_runs):
        seed = np.random.randint(0, 1000)
        print(f" ====================== Run {i + 1} / {N_runs} with seed {seed} ======================")
        
        optimizer = OptimizationRun(doppelganger_score_, 
                                    prior,
                                    nb_steps = 200,
                                    learning_rate = 0.001,
                                    nb_walkers = 1,
                                    start_halfway=False,
                                    random_seed=seed,
                                    outdir_name=f"./outdir_doppelganger/{seed}/",
                                    m_target = m_target,
                                    r_target = r_target,
                                    Lambdas_target = Lambdas_target)
        
        params = optimizer.initialize_walkers()
        optimizer.run(params)
        
        optimizer.plot_NS(optimizer.outdir_name)
        optimizer.plot_EOS(optimizer.outdir_name)
        
    return optimizer
        
############
### MAIN ###
############

def main():
    ### Preprocessing steps etc
    target_filename = "./36022_macroscopic.dat"
    target_eos = np.genfromtxt(target_filename, skip_header=1, delimiter=" ").T
    r_target, m_target, Lambdas_target = target_eos[0], target_eos[1], target_eos[2]
    
    target_filename = "./36022_microscopic.dat"
    data = np.loadtxt(target_filename)
    n_target, e_target, p_target, cs2_target = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    
    
    # Print Lambda1.4 for the target
    target_1_4 = jnp.interp(1.4, m_target, Lambdas_target, left = 0, right = 0)
    print(f"Lambda1.4 target: {target_1_4}")
    
    ### Optimizer run
    np.random.seed(47)
    optimizer = run_optimizer(metamodel_only = False, N_runs = 0)
    
    ### Postprocessing with result:
    result = Result("./real_doppelgangers/", optimizer, m_target, r_target, Lambdas_target, n_target, p_target, e_target, cs2_target)
    result.show_table()
    result.plot_doppelgangers()
    
    print("DONE")
    
if __name__ == "__main__":
    main()