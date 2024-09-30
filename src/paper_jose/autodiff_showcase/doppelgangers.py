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

from paper_jose.autodiff_showcase.evolutionary_optimizer import EvolutionaryOptimizer
# from jax.scipy.stats import gaussian_kde

##########################
### OPTIMIZATION CLASS ###
##########################
class OptimizationRun:
    
    def __init__(self,
                 score_fn: Callable,
                 prior: CombinePrior,
                 nb_steps: int,
                 nb_walkers: int = 1,
                 optimization_sign: float = -1, 
                 learning_rate: float = 1e-3, 
                 start_halfway: bool = True,
                 random_seed: int = 42,
                 return_aux: bool = True,
                 outdir_name: str = "computed_data",
                 # Plotting
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
        if self.nb_walkers == 1:
            start_halfway = False
        self.start_halfway = start_halfway
        self.return_aux = return_aux
        self.random_seed = random_seed
        self.outdir_name = outdir_name
        
        # Choose which type of run: single or vmap
        if self.nb_walkers == 1:
            self.run = self.run_single
        else:
            self.run = self.run_vmap # FIXME: implement this
            
        self.m_target = m_target
        self.r_target = r_target
        self.Lambdas_target = Lambdas_target
            
        # Clean the outdir(s)
        for i in range(self.nb_walkers):
            subdir = f"./{outdir_name}_{i}/"
            shutil.rmtree(subdir, ignore_errors=True)
            os.makedirs(subdir)
            
        self.plot_mse = plot_mse
        self.plot_final_errors = plot_final_errors
        self.plot_target = plot_target
        
    def initialize_walkers(self):
        
        if self.start_halfway:
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
        score_fn = jax.value_and_grad(score_fn, has_aux=True)
        score_fn = jax.jit(score_fn)
        
        failed_counter = 0
        
        print("Computing by gradient ascent . . .")
        for i in tqdm.tqdm(range(self.nb_steps)):
            
            ((score, aux), grad) = score_fn(params)
            m, r, l = aux
            
            if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)):
                print(f"Iteration {i} has NaNs")
                
                failed_counter += 1
                print(f"Skipping")
                continue
            
            print(f"Iteration {i}: score = {score}")
            np.savez(f"./computed_data/{i}.npz", masses_EOS = m, radii_EOS = r, Lambdas_EOS = l, score = score, **params)
            
            params = {key: value + self.optimization_sign * self.learning_rate * grad[key] for key, value in params.items()}
            
        print("Computing DONE")
        print(f"Failed percentage: {np.round(100 * failed_counter / self.nb_steps, 2)}")
        return None

    def compute_gradient_descent_vmap(self,
                                      params: dict):
        """
        Compute the gradient ascent or descent (just call it descent here for simplicity) in order to find the doppelgangers in the EOS space.
        """
            
        print("Starting parameters:")
        print(params)
        
        # Define the score function in the desired jax format
        score_fn = jax.value_and_grad(score_fn, has_aux=True)
        score_fn = jax.jit(jax.vmap(score_fn))
        
        failed_counter = 0
        
        print("Computing by gradient descent . . .")
        for i in tqdm.tqdm(range(self.nb_steps)):
            
            ((score, aux), grad) = score_fn(params)
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
        print(f"Failed percentage: {np.round(100 * failed_counter / self.nb_steps, 2)}")
        return None
    
    def plot_all_NS(self):
        for i in range(self.nb_walkers):
            subdir = f"./{self.outdir_name}_{i}/"
            self.plot_NS(subdir)
            
    def plot_NS(self,
                subdir: str,
                m_min = 1.2, # TODO: change?
                ):
    
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
                print(f"File {i} not found")
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
        save_name = f"{subdir}doppelganger_trajectory.png"
        print(f"Saving to: {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        plt.close()
        
        # Also plot the progress of the errors
        if self.plot_mse:
            plt.figure(figsize = (12, 6))
            mse_errors = []
            for i in range(N_max):
                data = np.load(f"./computed_data/{i}.npz")
                mse_errors.append(data["score"])
                
            # Plot
            plt.figure(figsize=(6, 6))
            nb = [i+1 for i in range(len(mse_errors))]
            plt.plot(nb, mse_errors, color="black")
            plt.scatter(nb, mse_errors, color="black")
            plt.xlabel("Iteration number")
            plt.ylabel("MSE")
            
            plt.savefig(f"{subdir}mse_errors.png", bbox_inches = "tight")
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
            
            masses = jnp.linspace(my_m_min, my_m_max, 100)
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
            save_name = f"{subdir}final_errors.png"
            print(f"Saving to: {save_name}")
            plt.savefig(save_name, bbox_inches = "tight")
            
            plt.close()
       

#################
### SCORE FNs ###
#################

def doppelganger_score(params: dict,
                       transform: utils.MicroToMacroTransform,
                       m_target: Array,
                       Lambdas_target: Array, 
                       r_target: Array,
                       m_min = 0.5,
                       m_max = 2.1,
                       N_masses: int = 100,
                       alpha: float = 1.0,
                       beta: float = 0.0,
                       gamma: float = 2.0,
                       return_aux: bool = True) -> float:
    
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
    score_lambdas = jnp.mean(((my_Lambdas_target - my_Lambdas_model) / my_Lambdas_target)**2)
    score_r = jnp.mean(((my_r_target - my_r_model) / my_r_target)**2)
    score_mtov = ((mtov_target - mtov_model) / mtov_target)**2
    
    score = alpha * score_lambdas + beta * score_r + gamma * score_mtov
    
    if return_aux:
        return score, (m_model, r_model, Lambdas_model)
    else:
        return score

       
######################
### BODY FUNCTIONS ### 
######################

def run_optimizer(method: str,
                  metamodel_only: bool = False):
    """
    Optimize a single EOS, mainly for testing the framework
    
    Args:
        method (str, optional): Either "single" or not "single". Defaults to "single".
    """
    
    # Check if method is OK:
    supported_methods = ["single", "evosax", "vmap"]
    if method not in supported_methods:
        raise ValueError(f"Method {method} not supported. Choose one of {supported_methods}.")
    
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
    N = 100
    run = OptimizationRun(score_fn = doppelganger_score,)
    
    # if method == "single":
    #     print("Running for a single parameter set")
        
    #     doppelganger_score_ = lambda params: doppelganger_score(params, transform, m_target, Lambdas_target, r_target)
    #     N = 100
    #     compute_gradient_descent(N, prior, doppelganger_score_, learning_rate = 0.001, start_halfway = False, random_seed = 64, return_aux = True)
        
    #     # Plot the doppelganger trajectory
    #     plot_NS(N,
    #             m_target = m_target, 
    #             Lambdas_target = Lambdas_target, 
    #             r_target = r_target, 
    #             plot_final_errors = True)
        
    # elif method == "evosax":
    #     print("Running with evosax")
        
    #     doppelganger_score_ = lambda params: doppelganger_score(params, transform, m_target, Lambdas_target, r_target, return_aux = False)
        
    #     # Create bounds -- will only work with uniform priors
    #     bound = []
    #     for i in range(len(prior.base_prior)):
    #         bound.append([prior.base_prior[i].xmin, prior.base_prior[i].xmax])
            
    #     bound = np.array(bound)
        
    #     print("bound")
    #     print(bound)
            
    #     # TODO: Can derive everything from the prior, so simplify this!
    #     optimizer = EvolutionaryOptimizer(loss_func=doppelganger_score_, 
    #                                       prior = prior,
    #                                       n_dims=len(prior.parameter_names), 
    #                                       bound=bound,
    #                                       popsize=20, 
    #                                       n_loops=100, 
    #                                       seed=43, 
    #                                       verbose=True)
    
    #     start_time = time.time()
    #     print("Starting the optimizer")
    #     _ = optimizer.optimize()
    #     best_fit = optimizer.get_result()[0]
    #     print("Done optimizing!")
    #     end_time = time.time()
        
    #     print(f"Time elapsed: {end_time - start_time} s")
        
    # elif method == "vmap":
    #     print("Running with vmap")
        
    #     doppelganger_score_ = lambda params: doppelganger_score(params, transform, m_target, Lambdas_target, r_target)
    #     nb_steps = 10
    #     nb_walkers = 2
    #     compute_gradient_descent_vmap(nb_steps, 
    #                                   nb_walkers, 
    #                                   prior, 
    #                                   doppelganger_score_, 
    #                                   learning_rate = 0.001, 
    #                                   random_seed = 65)
        
    #     ### TODO: plot the trajectories for several of them
    #     # Plot the doppelganger trajectory
    #     plot_NS(nb_steps, 
    #             m_target = m_target, 
    #             Lambdas_target = Lambdas_target, 
    #             r_target = r_target, 
    #             plot_final_errors = True)


############
### MAIN ###
############

def main():
    run_optimizer(method = "vmap", metamodel_only=True)
    
if __name__ == "__main__":
    main()