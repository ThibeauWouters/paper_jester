"""
Find doppelgangers with Jose
"""

################
### PREAMBLE ###
################

# import psutil
# p = psutil.Process()
# p.cpu_affinity([0])
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
import optax

from paper_jose.universal_relations.universal_relations import UniversalRelationBreaker
import paper_jose.utils as utils
import paper_jose.inference.utils_plotting as utils_plotting
plt.rcParams.update(utils_plotting.mpl_params)
import seaborn as sns

##########################
### DOPPELGANGER CLASS ###
##########################

PATH = "./"

def mse(x: Array, y: Array) -> float:
    """Relative mean squared error between x and y."""
    return jnp.mean((x - y) ** 2)

def mrse(x: Array, y: Array) -> float:
    """Relative mean squared error between x and y."""
    return jnp.mean(((x - y) / y) ** 2)

def mae(x: Array, y: Array) -> float:
    """Relative mean squared error between x and y."""
    return jnp.mean(abs(x - y))

def mrae(x: Array, y: Array) -> float:
    """Relative mean absolute error between x and y."""
    return jnp.mean(jnp.abs((x - y) / y))

def maxae(x: Array, y: Array) -> float:
    """Relative mean absolute error between x and y."""
    return jnp.max(jnp.abs(x - y))

class DoppelgangerRun:
    
    def __init__(self,
                 prior: CombinePrior,
                 transform: utils.MicroToMacroTransform,
                 which_score: str = "macro",
                 random_seed: int = 42,
                 # Optimization hyperparameters
                 nb_steps: int = 200,
                 optimization_sign: float = -1, 
                 learning_rate: float = 1e-3,
                 # Plotting
                 outdir_name: str = "./outdir/",
                 plot_mse: bool = True,
                 plot_final_errors: bool = True,
                 plot_target: bool = True,
                 # Target
                 micro_target_filename: str = PATH + "hauke_microscopic.dat", # 36022
                 macro_target_filename: str = PATH + "hauke_macroscopic.dat",
                 load_params: bool = True,
                 fixed_params: dict = None,
                 # Other stuff
                 save_by_counter: bool = True,
                 ):
        
        # Set prior and transform
        self.prior = prior
        self.prior_lower = jnp.array([p.xmin for p in prior.base_prior])
        self.prior_upper = jnp.array([p.xmax for p in prior.base_prior])
        self.transform = transform
        self.which_score = which_score
        self.learning_rate = learning_rate
        
        # Load micro and macro targets
        data = np.loadtxt(micro_target_filename)
        n_target, e_target, p_target, cs2_target = data[:, 0] / 0.16, data[:, 1], data[:, 2], data[:, 3]
        
        # Interpolate the target: it is not so dense
        self.n_target = np.linspace(np.min(n_target), np.max(n_target), 1_000)
        self.e_target = np.interp(self.n_target, n_target, e_target)
        self.p_target = np.interp(self.n_target, n_target, p_target)
        self.cs2_target = np.interp(self.n_target, n_target, cs2_target)
        
        data = np.genfromtxt(macro_target_filename, skip_header=1, delimiter=" ").T
        self.r_target, self.m_target, self.Lambdas_target = data[0], data[1], data[2]
        
        self.mtov_target = jnp.max(self.m_target)
        print(f"The MTOV target is {self.mtov_target}")
        
        if self.which_score.lower() == "macro":
            self.score_fn = lambda params: self.doppelganger_score_macro(params)
            self.score_fn = jax.value_and_grad(self.score_fn, has_aux=True)
            self.score_fn = jax.jit(self.score_fn)
            
        elif self.which_score.lower() == "binary_love":
            print("Using the binary_love optimization function")
            breaker = UniversalRelationBreaker()
            self.score_fn = lambda params: breaker.score_fn(params, transform)
            print("NOTE: we are setting the optimization sign to 1.0 since we want to break it, i.e. maximize the error")
            optimization_sign = 1.0
            save_by_counter = True
        else:
            raise ValueError("Unknown score function")
            
        # Save the final things
        self.nb_steps = nb_steps
        self.optimization_sign = optimization_sign
        self.save_by_counter = save_by_counter
        
        # Outdir and plotting stuff
        self.outdir_name = outdir_name
        self.jax_key = jax.random.PRNGKey(random_seed)
        self.set_seed(random_seed)
        self.subdir_name = os.path.join(self.outdir_name, str(random_seed))
            
        
        if not os.path.exists(self.outdir_name):
            print("Creating the outdir")
            os.makedirs(self.outdir_name)
            
        self.plot_mse = plot_mse
        self.plot_final_errors = plot_final_errors
        self.plot_target = plot_target
        
        # Load the parameters of an existing doppelganger:
        self.fixed_params = fixed_params
        
        if self.fixed_params is None:
            print("No fixed_params given in Doppelganger run")
            if load_params:
                print("Will load params from earlier file")
                npz_filename = PATH + "real_doppelgangers/7945/data/199.npz"
                data = np.load(npz_filename)
                params_keys = list(utils.NEP_CONSTANTS_DICT.keys())
                params_keys.remove("E_sat")
                
                params = {key: float(data[key]) for key in params_keys}
                self.fixed_params = params
                
        # Update the fixed params dict in the transform if not included in prior (i.e. varied over)
        for key in self.fixed_params.keys():
            if key not in self.prior.parameter_names:
                self.transform.fixed_params[key] = self.fixed_params[key]
        
        # DEBUG
        print("Loaded the following fixed params:")
        print(self.transform.fixed_params)
            
            
        
    def set_seed(self, seed: int):
        self.random_seed = seed
        
        # Create outdirs for this seed
        self.subdir_name = os.path.join(self.outdir_name, str(seed))
        if os.path.exists(self.subdir_name):
            print("Subdir already exists")
            return
        else:
            os.makedirs(self.subdir_name)
            os.makedirs(f"{self.subdir_name}/figures/")
            os.makedirs(f"{self.subdir_name}/data/")
            print(f"Created subdir: {self.subdir_name}")
        
    def random_sample(self, 
                      N_samples: int = 2_000, 
                      outdir: str = PATH + "random_samples/"):
        """
        Generate a sample from the prior, solve the TOV equations and return the results.        
        """
        
        print("Generating random samples for a batch of EOS . . . ")
        
        counter = 0
        
        pbar = tqdm.tqdm(range(N_samples))
        for i in pbar:
            params = self.initialize_walkers()
            out = self.transform.forward(params)
            m, r, l = out["masses_EOS"], out["radii_EOS"], out["Lambdas_EOS"]
            n, p, e, cs2 = out["n"], out["p"], out["e"], out["cs2"]
            
            if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)):
                print(f"Sample {i} has NaNs. Skipping this sample")
                continue
            
            # Only save if the TOV mass is high enough
            if jnp.max(m) > 2.1:
                npz_filename = os.path.join(outdir, f"{counter}.npz")
                np.savez(npz_filename, masses_EOS = m, radii_EOS = r, Lambdas_EOS = l, n = n, p = p, e = e, cs2 = cs2, **params)
                counter += 1
                
            pbar.set_description(f"Sample: {i} MTOV: {np.max(m)} Counter: {counter}")
            
        return
    
    def normalize_forward(self, x, lower, higher):
        # value = (x - lower) / (higher - lower)
        # return value.astype('float')
        return x

    def normalize_backward(self, x, lower, higher):
        # value = x * (higher - lower) + lower
        # return value.astype('float')
        return x
    
    def normalize_forward_dict(self, params: dict) -> dict:
        return {key: self.normalize_forward(value, lower, higher) for key, value, lower, higher in zip(params.keys(), params.values(), self.prior_lower, self.prior_upper)}
    
    def normalize_backward_dict(self, params: dict) -> dict:
        return {key: self.normalize_backward(value, lower, higher) for key, value, lower, higher in zip(params.keys(), params.values(), self.prior_lower, self.prior_upper)}

    def doppelganger_score_macro(self, 
                                 params: dict,
                                 # For the masses for interpolation
                                 m_min = 1.2,
                                 m_max = 2.25,
                                 N_masses: int = 500,
                                 # Hyperparameters for score fn
                                 alpha: float = 1.0,
                                 beta: float = 0.0,
                                 gamma: float = 0.0,
                                 return_aux: bool = True,
                                 error_fn: Callable = mrae) -> float:
        
        """
        Doppelganger score function. 
        TODO: type hints

        Returns:
            _type_: _description_
        """
        
        # Solve the TOV equations
        unnormalized_params = self.normalize_backward_dict(params)
        out = self.transform.forward(unnormalized_params)
        m_model, r_model, Lambdas_model = out["masses_EOS"], out["radii_EOS"], out["Lambdas_EOS"]
        
        mtov_model = jnp.max(m_model)
        mtov_target = jnp.max(self.m_target)
        
        # Get a mass array and interpolate NaNs on top of it
        masses = jnp.linspace(m_min, m_max, N_masses)
        my_Lambdas_model = jnp.interp(masses, m_model, Lambdas_model, left = 0, right = 0)
        my_Lambdas_target = jnp.interp(masses, self.m_target, self.Lambdas_target, left = 0, right = 0)
        
        my_r_model = jnp.interp(masses, m_model, r_model, left = 0, right = 0)
        my_r_target = jnp.interp(masses, self.m_target, self.r_target, left = 0, right = 0)
        
        # Define separate scores
        score_lambdas = error_fn(my_Lambdas_model, my_Lambdas_target)
        score_r       = error_fn(my_r_model, my_r_target)
        score_mtov    = error_fn(mtov_model, mtov_target)
        
        score = alpha * score_lambdas + beta * score_r + gamma * score_mtov
        
        if return_aux:
            return score, out
        else:
            return score
        
    def run(self, params: dict) -> None:
        """
        Run the optimization loop for the doppelganger problem.

        Args:
            params (dict): Starting parameters.

        """
        
        print("Starting parameters:")
        print(params)
        
        print("Computing by gradient ascent . . .")
        pbar = tqdm.tqdm(range(self.nb_steps))
        
        params = self.normalize_forward_dict(params)
        
        
        # Combining gradient transforms using `optax.chain`.
        gradient_transform = optax.adam(learning_rate=self.learning_rate)
        opt_state = gradient_transform.init(params)
        
        try: 
            for i in pbar:
                ((score, aux), grad) = self.score_fn(params)
                
                m, r, l = aux["masses_EOS"], aux["radii_EOS"], aux["Lambdas_EOS"]
                logpc = aux["logpc_EOS"]
                n, p, e, cs2 = aux["n"], aux["p"], aux["e"], aux["cs2"]
                
                if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)):
                    print(f"Iteration {i} has NaNs. Exiting the computing loop now")
                    break
                
                if self.which_score == "macro":
                    max_error_Lambdas = compute_max_error(m, l, self.m_target, self.Lambdas_target,)
                    max_error_radii = compute_max_error(m, r, self.m_target, self.r_target)
                    pbar.set_description(f"Iteration {i}: score {score}, max_error_lambdas = {max_error_Lambdas}, max_error_radii = {max_error_radii}")
                
                elif self.which_score.lower() == "binary_love":
                    mtov = jnp.max(m)
                    pbar.set_description(f"Iteration {i}: Score: {score} MTOV: {mtov}")
                
                else:
                    pbar.set_description(f"Iteration {i}: score {score}")
                    
                ### Save
                if self.save_by_counter:
                    npz_filename = os.path.join(self.subdir_name, f"data/{i}.npz")
                else:
                    npz_filename = os.path.join(self.subdir_name, f"data/0.npz")
                
                
                if self.which_score == "macro":
                    np.savez(npz_filename, logpc_EOS = logpc, masses_EOS = m, radii_EOS = r, Lambdas_EOS = l, n = n, p = p, e = e, cs2 = cs2, score = score, max_error_Lambdas=max_error_Lambdas, max_error_radii=max_error_radii, **self.normalize_backward_dict(params))
                else:
                    np.savez(npz_filename, logpc_EOS = logpc, masses_EOS = m, radii_EOS = r, Lambdas_EOS = l, n = n, p = p, e = e, cs2 = cs2, score = score, **self.normalize_backward_dict(params))
                    
                # Check for early stoppings
                if self.which_score == "macro":
                    if max_error_Lambdas < 10.0: # and max_error_radii < 0.1:
                        print("Max error reached the threshold, exiting the loop")
                        break
            
                # Do the updates
                updates, opt_state = gradient_transform.update(grad, opt_state)
                params = optax.apply_updates(params, updates)
                
            print("Computing DONE")
            self.plot_NS()
        except Exception as e:
            print(f"Exception occurred: {e}")
        
            
    def run_nn(self, 
               max_nsat: float = 6.0,
               min_nsat: float = None) -> None:
        """
        Run the optimization loop for the doppelganger problem.

        Args:
            params (dict): Starting parameters.

        """
        
        import optax
        from flax.training.train_state import TrainState
        
        ### Optimizer and scheduler:
        
        # # Polynomial schedule
        # start = int(0.25 * self.nb_steps)
        # start_lr = 1e-3
        # end_lr = 1e-5
        # power = 4.0
        # scheduler = optax.polynomial_schedule(start_lr, end_lr, power, self.nb_steps - start, transition_begin=start)
        
        # # Exponential decay schedule
        scheduler = optax.exponential_decay(init_value=self.learning_rate,
                                            transition_steps=self.nb_steps,
                                            decay_rate=0.90)
        
        # Combining gradient transforms using `optax.chain`.
        # tx = optax.chain(optax.clip(1.0), optax.adam(learning_rate, 0.9))
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
            optax.scale_by_adam(),  # Use the updates from adam.
            optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            optax.scale(-1.0))
        
        # Get initial dummy input
        test_input = jnp.ones(self.transform.eos.ndat_CSE)
        params = self.transform.eos.nn.init(jax.random.PRNGKey(0), test_input)
        
        # Initialize the neural network
        state = TrainState.create(apply_fn = self.transform.eos.nn.apply, params = params, tx = tx)
        self.transform.eos.state = state
        
        # Below, only fetch params:
        params = params["params"]
        
        # Create the opt_state
        opt_state = tx.init(params)
        
        nbreak = self.transform.fixed_params["nbreak"]
        if min_nsat is None:
            min_nsat = nbreak / 0.16
            print(f"Min nsat is set to {min_nsat}")
        
        def score_fn_micro(params: dict):
            """Params should be a dict of metamodel parameters and an entry nn_state that maps to the state.params of the neural network"""
            out = self.transform.forward(params)
            n = out["n"] / jose_utils.fm_inv3_to_geometric / 0.16
            p = out["p"] / jose_utils.MeV_fm_inv3_to_geometric
            e = out["e"] / jose_utils.MeV_fm_inv3_to_geometric
            # cs2 = out["cs2"]
            # cs2_target = self.cs2_target[mask]
            # cs2_interp = jnp.interp(n_target, n, cs2)
            # mask = (self.n_target < max_nsat) * (self.n_target > min_nsat)
            # n_target = self.n_target[mask]
            
            max_e_MM = 600
            mask_target = self.e_target < max_e_MM
            e_target = self.e_target[mask_target]
            p_target = self.p_target[mask_target]
            
            p_of_e = jnp.interp(e_target, e, p)
            
            score_low = mrae(p_of_e, p_target)
            
            high_target_p = 200
            high_target_e = 1200
            
            p_at_target = jnp.interp(high_target_e, e, p)
            
            score_high = mrae(p_at_target, high_target_p)
            
            score = score_low + score_high
            
            return score, out
        
        score_fn_micro = jax.value_and_grad(score_fn_micro, has_aux = True)
        score_fn_micro = jax.jit(score_fn_micro)

        if self.which_score == "micro":
            print("Running neural network with micro target")
            score_fn = score_fn_micro
            
        elif self.which_score == "macro":
            print("Running neural network with macro target")
            score_fn = self.score_fn
        
        pbar = tqdm.tqdm(range(self.nb_steps))
        for i in pbar:
            ((score, aux), grad) = score_fn({"nn_state": params})
            
            # Fetch the grad for the NN separately
            grad_nn = grad["nn_state"]
           
            # TODO: decide what to put in the aux?
            m, r, l, n, cs2 = aux["masses_EOS"], aux["radii_EOS"], aux["Lambdas_EOS"], aux["n"], aux["cs2"]
            if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)) or np.any(np.isnan(n)) or np.any(np.isnan(cs2)):
                print(f"Iteration {i} has NaNs. Exiting the computing loop now")
                break
            
            npz_filename = os.path.join(self.subdir_name, f"data/{i}.npz")
            np.savez(npz_filename, masses_EOS = m, radii_EOS = r, Lambdas_EOS = l, n = n, cs2 = cs2, score = score)
                
            # Make the updates
            for i in range(self.nb_gradient_updates):
                # state = state.apply_gradients(grads=grad_nn)
                updates, opt_state = tx.update(grad_nn, opt_state)
                params = optax.apply_updates(params, updates)
                
            # Get the max error on Lambdas
            if self.which_score == "macro":
                max_error_lambdas = compute_max_error(m, l, self.m_target, self.Lambdas_target)
                max_error_radii = compute_max_error(m, r, self.m_target, self.r_target)
                
                pbar.set_description(f"Iteration {i}: score = {score} max error lambdas = {max_error_lambdas} max error radii = {max_error_radii}")
                if max_error_lambdas < 10.0 and max_error_radii < 0.1:
                    print("Max error reached the threshold, exiting the loop")
                    break
            else:
                pbar.set_description(f"Iteration {i}: score = {score}")
            
        print("Computing DONE")
        
        # Make the plot
        n = n / jose_utils.fm_inv3_to_geometric / 0.16
        e = aux["e"] / jose_utils.MeV_fm_inv3_to_geometric
        p = aux["p"] / jose_utils.MeV_fm_inv3_to_geometric
        
        mask = (n < max_nsat) * (n > min_nsat)
        mask_target = (self.n_target < max_nsat) * (self.n_target > min_nsat)
        plt.figure(figsize=(12, 6))
        plt.plot(n[mask], cs2[mask], label = "Result found", color = "red", zorder = 4)
        plt.plot(self.n_target[mask_target], self.cs2_target[mask_target], label = "Target", linestyle = "--", color = "black", zorder = 5)
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"$c_s^2$")
        plt.legend()
        plt.tight_layout()
        save_name = "../test/figures/test_match.pdf"
        print(f"Saving figure to: {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        plt.close()
        
        # TODO: this is code duplication, remove!
        plt.subplots(figsize = (14, 12), nrows = 2, ncols = 2)
        c = "red"
        mask = (n < max_nsat) * (n > min_nsat)
        mask_target = (self.n_target < max_nsat) * (self.n_target > min_nsat)
        plt.subplot(221)
        plt.plot(n[mask], e[mask], color = c)
        plt.plot(self.n_target[mask_target], self.e_target[mask_target], color = "black", label = "Target")
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"$e$ [MeV fm$^{-3}$]")
        plt.xlim(min_nsat, max_nsat)
        
        plt.subplot(222)
        plt.plot(n, p, color = c)
        plt.plot(self.n_target[mask_target], self.p_target[mask_target], color = "black", label = "Target")
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
        plt.xlim(min_nsat, max_nsat)
        
        plt.subplot(223)
        print("len(cs2[mask]")
        print(len(cs2[mask]))
        plt.plot(n[mask], cs2[mask], color = c)
        plt.scatter(n[mask], cs2[mask], color = c)
        plt.plot(self.n_target[mask_target], self.cs2_target[mask_target], color = "black", label = "Target")
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"$c_s^2$")
        plt.xlim(min_nsat, max_nsat)
        plt.ylim(0, 1)
        
        plt.subplot(224)
        e_min = 500
        e_max = 1500
        mask = (e_min < e) * (e < e_max)
        mask_target = (e_min < self.e_target) * (self.e_target < e_max)
        
        plt.plot(e[mask], p[mask], color = c)
        plt.plot(self.e_target[mask_target], self.p_target[mask_target], color = "black", label = "Target")
        plt.xlabel(r"$e$ [MeV fm$^{-3}$]")
        plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
        plt.tight_layout()
        save_name = "../test/figures/test_match_EOS.pdf"
        print(f"Saving figure to: {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        plt.close()
        
        # Make the plot
        plt.subplots(figsize = (14, 10), nrows = 1, ncols = 2)
        
        plt.subplot(121)
        plt.plot(r, m, color = "red", label = "NN")
        plt.plot(self.r_target, self.m_target, color = "black", label = "Target")
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_\odot$]")
        
        plt.subplot(122)
        plt.plot(m, l, color = "red", label = "NN")
        plt.plot(self.m_target, self.Lambdas_target, color = "black", label = "Target")
        plt.xlabel(r"$M$ [$M_\odot$]")
        plt.ylabel(r"$\Lambda$")
        plt.yscale("log")
        
        plt.legend()
        save_name = "../test/figures/test_match_TOV.pdf"
        print(f"Saving figure to {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        plt.close()
        
        
        
    def perturb_doppelganger(self, 
                             dir: str = "./real_doppelgangers/750/",
                             seed: int = 123,
                             nb_perturbations: int = 100,
                             nb_cse: int = 10,
                             nmax_nsat: int = 6):
        
        # TODO: change into initialization procedure? To start from a "local" region to find doppelgangers?
        
        # Load the final npz file from this dir:
        data_files = os.listdir(os.path.join(dir, "data/"))
        npz_files = [f for f in data_files if f.endswith(".npz") and "best" not in f]
        numbers = [int(f.split(".")[0]) for f in npz_files]
        final_number = max(numbers)
        
        data = np.load(os.path.join(dir, f"data/{final_number}.npz"))
        
        # Get the EOS parameters
        params = {key: data[key] for key in self.prior.parameter_names}
        nbreak = params["nbreak"]
        
        print("nbreak")
        print(nbreak / 0.16)
        
        # Define the CSE prior we will use
        prior_list = []
        
        # The first CSE point is rather tight:
        prior_list.append(UniformPrior((3.0 - 0.1) * 0.16, (3.0 - 0.1) * 0.16, parameter_names=["n_CSE_1"]))
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=["cs2_CSE_1"]))
        
        width = (nmax_nsat * 0.16 - nbreak) / (nb_cse + 1)
        for i in range(2, nb_cse):
            left = nbreak + i * width
            right = nbreak + (i + 1) * width
            prior_list.append(UniformPrior(left, right, parameter_names=[f"n_CSE_{i}"]))
            prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))
        
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{nb_cse}"]))
        prior = CombinePrior(prior_list)
        
        # Perturb a few times, save here::
        if not os.path.exists(f"perturbations/"):
            os.makedirs(f"perturbations/")
            
        key = jax.random.PRNGKey(seed)
        pbar = tqdm.tqdm(range(nb_perturbations))
        
        counter = 0
        for i in pbar:
            # Sample a new random key
            key, subkey = jax.random.split(key)
            new_params = prior.sample(subkey, 1)
            new_params = {k: float(v.at[0].get()) for k, v in new_params.items()}
            
            params.update(new_params)
            
            # Solve:
            out = self.transform.forward(params)
            m = out["masses_EOS"]
            r = out["radii_EOS"]
            l = out["Lambdas_EOS"]
            
            # Compute error
            max_error_Lambdas = compute_max_error(m, l, self.m_target, self.Lambdas_target)
            max_error_radii = compute_max_error(m, r, self.m_target, self.r_target)
            
            # Just define some random score for now
            score = max_error_Lambdas + max_error_radii
            if max_error_Lambdas < 10.0 and max_error_radii < 0.1:
                counter += 1
                np.savez(f"perturbations/{i}.npz", masses_EOS = m, radii_EOS = r, Lambdas_EOS = l, score = score, **params)
            
            pbar.set_description(f"Iteration {i}: max_error_lambdas = {max_error_Lambdas}, max_error_radii = {max_error_radii} doppelgangers found: {counter}")
        
        return
    
    def finetune_doppelganger(self, 
                              dir: str = "./real_doppelgangers/",
                              seed: int = 750):
        
        # Load the final npz file from this dir, this will be the starting point
        data_files = os.listdir(os.path.join(dir, str(seed), "data/"))
        npz_files = [f for f in data_files if f.endswith(".npz") and "best" not in f]
        numbers = [int(f.split(".")[0]) for f in npz_files]
        final_number = max(numbers)
        data = np.load(os.path.join(dir, str(seed), f"data/{final_number}.npz"))
        
        # Get the EOS parameters
        params = {key: data[key] for key in self.prior.parameter_names}
        
        # Perturb a few times, save here::
        if not os.path.exists(f"finetune/"):
            os.makedirs(f"finetune/")
            os.makedirs(f"finetune/{seed}")
            os.makedirs(f"finetune/{seed}/figures/")
            os.makedirs(f"finetune/{seed}/data/")
            
        pbar = tqdm.tqdm(range(self.nb_steps))
        for i in pbar:
            
            ((score, aux), grad) = self.score_fn_finetune(params)
            m, r, l = aux
            
            if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)):
                print(f"Iteration {i} has NaNs. Exiting the computing loop now")
                break
            
            npz_filename = os.path.join("./finetune/", str(seed), "data/{i}.npz")
            np.savez(npz_filename, masses_EOS = m, radii_EOS = r, Lambdas_EOS = l, score = score, **params)
            
            max_error_Lambdas = compute_max_error(m, l, self.m_target, self.Lambdas_target)
            max_error_radii = compute_max_error(m, r, self.m_target, self.r_target)
            
            # if max_error_Lambdas < 10.0:
            #     print("Max error reached the threshold, exiting the loop")
            #     break
            
            # Make the updates
            learning_rate = get_learning_rate(i, self.learning_rate, self.nb_steps)
            params = {key: value + self.optimization_sign * learning_rate * grad[key] for key, value in params.items()}
            
            pbar.set_description(f"Iteration {i}: max_error Lambdas = {max_error_Lambdas}, max_error radii = {max_error_radii}")
            
        print("Computing DONE")
        self.plot_NS(dir = f"./finetune/{seed}")
            
        return
    
    def plot_pressure_mtov_correlations(self, outdir: str = "./perturbations/"):
        
        print("Plotting pressure vs M_TOV correlations")
        
        from scipy.stats import pearsonr
        
        subdirs = os.listdir(outdir)
            
        pressures = []
        mtovs = []
        n_TOV_array = []
        param_names = self.prior.parameter_names
        
        if "perturbations" in outdir:
            print("Note: Looking at the perturbations")
            for file in os.listdir(outdir):
                data = np.load(os.path.join("./perturbations/", file))
                params = {key: data[key] for key in param_names}
                
                # Solve the EOS
                out = self.transform.forward(params)
                m = out["masses_EOS"]
                mtov = float(jnp.max(m))
                
                n = out["n"] / jose_utils.fm_inv3_to_geometric / 0.16
                p = out["p"] / jose_utils.MeV_fm_inv3_to_geometric
                
                # # Get the pressure at n_TOV
                # p_c_array = jnp.exp(out["p_c_EOS"]) / jose_utils.MeV_fm_inv3_to_geometric
                # p_c = p_c_array[-1]
                # n_TOV = get_n_TOV(n, p, p_c)
                # p_TOV = float(np.interp(n_TOV, n, p))
                # # p_TOV = float(np.interp(4.0, n, p))
                # pressures.append(p_TOV)
                # n_TOV_array.append(n_TOV)
                
                # Save:
                mtovs.append(mtov)
            
        else:
            for subdir in subdirs:
                # Get the datadir
                data_dir = os.path.join(outdir, subdir, "data")
                
                # Get the final
                npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz") and "best" not in f]
                max_number = max([int(f.split(".")[0]) for f in npz_files])
                data = np.load(os.path.join(data_dir, f"{max_number}.npz"))
                
                params = {key: data[key] for key in param_names}
                
                # Solve the EOS
                out = self.transform.forward(params)
                m = out["masses_EOS"]
                mtov = float(jnp.max(m))
                
                n = out["n"] / jose_utils.fm_inv3_to_geometric / 0.16
                p = out["p"] / jose_utils.MeV_fm_inv3_to_geometric
                
                # n_TOV = get_n_TOV(n, p, p_c)
                # # Get the pressure at n_TOV
                # p_c_array = jnp.exp(out["p_c_EOS"]) / jose_utils.MeV_fm_inv3_to_geometric
                # p_c = p_c_array[-1]
                # p_TOV = float(np.interp(n_TOV, n, p))
                # pressures.append(p_TOV)
                # n_TOV_array.append(n_TOV)
                
                # Save:
                mtovs.append(mtov)
            
        # Make a plot
        pressures = np.array(pressures)
        n_TOV_array = np.array(n_TOV_array)
        mtovs = np.array(mtovs)
        
        r, p_value = pearsonr(mtovs, n_TOV_array)
        print(f"Pearson correlation coefficient: {(r, p_value)}")
        
        plt.figure(figsize = (12, 6))
        plt.scatter(mtovs, pressures, color = "black", s = 16)
        # plt.scatter(mtovs, pressures, color = "black", s = 16)
        plt.xlabel(r"$M_{\rm TOV}$ [M$_\odot$]")
        plt.ylabel(r"$p_{\rm{TOV}}$ [MeV fm$^{-3}$]")
        # plt.ylabel(r"$n_{\rm{TOV}}$ [$n_{\rm{sat}}$]")
        # plt.ylabel(r"$p(4 n_{\rm{sat}})$ [MeV fm$^{-3}$]")
        plt.title("Pearson correlation coefficient: {:.2f}".format(r))
        plt.tight_layout()
        plt.savefig(f"./figures/pressure_mtov_correlations.png", bbox_inches = "tight")
        plt.savefig(f"./figures/pressure_mtov_correlations.pdf", bbox_inches = "tight")
        plt.close()
    
    
    def plot_NS(self, m_min: float = 1.2, dir: str = None):
        """
        Plot the doppelganger trajectory in the NS space.

        TODO: perhaps make m_min a class variable?
        
        Args:
            m_min (float, optional): Minimum mass from which to compute errors and create the error plot. Defaults to 1.2.
        """
    
        # Read the EOS data
        all_masses_EOS = []
        all_radii_EOS = []
        all_Lambdas_EOS = []
        
        if dir is None:
            dir = self.subdir_name

        for i in range(self.nb_steps):
            try:
                npz_file = os.path.join(self.subdir_name, f"data/{i}.npz")
                data = np.load(npz_file)
                
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
        mask = (self.m_target > m_min)
        m_target = self.m_target[mask]
        r_target = self.r_target[mask]
        Lambdas_target = self.Lambdas_target[mask]
        
        plt.plot(r_target, m_target, color = "red", zorder = 1e10)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M \ [M_\odot]$")
        
        plt.subplot(122)
        plt.xlabel(r"$M \ [M_\odot]$")
        plt.ylabel(r"$\Lambda$")
        plt.plot(m_target, Lambdas_target, label=r"$\Lambda$", color = "red", zorder = 1e10)
        plt.yscale("log")
        
        for i in range(N_max):
            color = cmap(norm(i))
            
            # Mass-radius plot
            plt.subplot(121)
            mask = (all_masses_EOS[i] > m_min)
            m = all_masses_EOS[i][mask]
            r = all_radii_EOS[i][mask]
            l = all_Lambdas_EOS[i][mask]
            
            plt.plot(r, m, color=color, linewidth = 2.0, zorder=i)
                
            # Mass-Lambdas plot
            plt.subplot(122)
            plt.plot(m, l, color=color, linewidth = 2.0, zorder=i)
            
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs[-1])
        cbar.set_label(r'Iteration number', fontsize = 22)
            
        plt.tight_layout()
        save_name = os.path.join(self.subdir_name, "figures/doppelganger_trajectory.png")
        print(f"Saving to: {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        plt.close()
        
        if self.plot_final_errors:
            plt.figure(figsize = (12, 6))
            # Plot the errors of the final M, Lambda, R
            m_final = all_masses_EOS[-1]
            Lambda_final = all_Lambdas_EOS[-1]
            
            max_error_print = compute_max_error(m_final, Lambda_final, self.m_target, self.Lambdas_target)
            
            my_m_min = max(min(m_final), min(self.m_target))
            my_m_min = max(my_m_min, m_min)
            my_m_max = min(max(m_final), max(self.m_target))
            
            masses = jnp.linspace(my_m_min, my_m_max, 500)
            my_Lambdas_model = jnp.interp(masses, m_final, Lambda_final, left = 0, right = 0)
            my_Lambdas_target = jnp.interp(masses, self.m_target, self.Lambdas_target, left = 0, right = 0)
            
            # my_r_model = jnp.interp(masses, m_final, r_final, left = 0, right = 0)
            # my_r_target = jnp.interp(masses, m_target, r_target, left = 0, right = 0)
            
            errors = abs(my_Lambdas_model - my_Lambdas_target)
            plt.plot(masses, errors, color = "black")
            plt.xlabel(r"$M \ [M_\odot]$")
            plt.ylabel(r"$\Delta \Lambda \ (L_\infty)$ ")
            plt.yscale("log")
            plt.title(f"Max error: {max_error_print}")
            save_name = os.path.join(self.subdir_name, "figures/final_errors.png")
            print(f"Saving to: {save_name}")
            plt.savefig(save_name, bbox_inches = "tight")
            
            print(f"FINAL RESULT: The max error in Lambdas was: {max_error_print}")
            
            plt.close()
            
    def plot_single_NS(self, m, r, l):
        """
        Plot the doppelganger trajectory in the NS space.

        TODO: perhaps make m_min a class variable?
        
        Args:
            m_min (float, optional): Minimum mass from which to compute errors and create the error plot. Defaults to 1.2.
        """
        
        # Plot the target
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 6))
        plt.subplot(121)
        plt.plot(self.r_target, self.m_target, color = "black", zorder = 1e10)
        plt.plot(r, m, color="red", linewidth = 2.0)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M \ [M_\odot]$")
        
        plt.subplot(122)
        plt.xlabel(r"$M \ [M_\odot]$")
        plt.ylabel(r"$\Lambda$")
        plt.plot(self.m_target, self.Lambdas_target, label=r"$\Lambda$", color = "black", zorder = 1e10)
        plt.plot(m, l, color="red", linewidth = 2.0)
        plt.yscale("log")
        plt.tight_layout()
        save_name = os.path.join(self.subdir_name, "figures/doppelganger_trajectory.png")
        print(f"Saving to: {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        plt.close()
        
    def plot_single_EOS(self, n, p, e):
        
        # Limit up to 3 nsat:
        mask = (n < 3.0) * (n > 0.5)
        n, p, e = n[mask], p[mask], e[mask]
        
        mask_target = (self.n_target < 3.0) * (self.n_target > 0.5)
        
        plt.plot(self.e_target[mask_target], self.p_target[mask_target], color = "black", zorder = 1e10, label = "Target")
        plt.plot(e, p, color = "red", linewidth = 2.0, zorder = 1e9, label = "EOS found")
        plt.xlabel(r"$e$ [MeV fm$^{-3}$]")
        plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
        
        save_name = os.path.join(self.subdir_name, "figures/doppelganger_trajectory_EOS.png")
        print(f"Saving to: {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        plt.close()
            
    def plot_EOS(self):
        
        # TODO: remove me if unused and not useful
    
        parameter_names = self.prior.parameter_names
        eos_trajectory = {name: [] for name in parameter_names}
        
        for i in range(self.nb_steps):
            try:
                npz_file = os.path.join(self.subdir_name, f"data/{i}.npz")
                data = np.load(npz_file)
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
            save_name = os.path.join(self.subdir_name, f"figures/trajectory_{name}.png")
            print(f"Saving to: {save_name}")
            plt.savefig(save_name, bbox_inches = "tight")
            plt.close()
            
    def get_table(self, outdir: str = None, keep_real_doppelgangers: bool = True, save_table: bool = False):
        """
        Postprocessing utility to show the table of the doppelganger runs.

        Args:
            outdir (str): Outdir with a collection, ideally, of real doppelgangers. 
        """
        
        if outdir is None:
            outdir = self.outdir_name
            
        subdirs = os.listdir(outdir)
        output = defaultdict(list)
        
        for subdir in subdirs:
            # Get the datadir
            data_dir = os.path.join(outdir, subdir, "data")
            
            # Get the final iteration number from the filenames
            npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
            
            # TODO: load best if it is there otherwise get the final number
            if "best.npz" in npz_files:
                npz_files.remove("best.npz")
            numbers = [int(file.split(".")[0]) for file in npz_files]
            try:
                final_number = max(numbers)
            except ValueError as e:
                print(f"There was a problem for subdir {subdir}: {e}")
                continue
            
            # Get the datadir
            npz_file = os.path.join(data_dir, f"{final_number}.npz")
            data = np.load(npz_file)
            keys: list[str] = data.keys()
            
            for key in keys:
                if key.endswith("_EOS") or key in ["n", "p", "e", "cs2", "max_error_Lambdas", "max_error_lambdas", "max_error_radii"]:
                    continue
                output[key].append(float(data[key]))
                
            
            # TODO: work in progress
            # Macro output: needs a bit more work
            m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
            
            max_error_Lambdas = compute_max_error(m, l, self.m_target, self.Lambdas_target)
            max_error_radii = compute_max_error(m, r, self.m_target, self.r_target)
            
            # Add to output:
            output["max_error_Lambdas"].append(float(max_error_Lambdas))
            output["max_error_radii"].append(float(max_error_radii))
            output["subdir"].append(subdir)
            
        df = pd.DataFrame(output)
        # Sort based on score, lower to upper:
        df = df.sort_values("max_error_Lambdas")
        
        if keep_real_doppelgangers:
            # Only limit to those with max error below 10:
            df = df[(df["max_error_Lambdas"] < 10.0) * (df["max_error_radii"] < 0.100)]
            
            print(f"Keeping only the real doppelgangers, there are: {len(df)} doppelgangers")
        
        print("Postprocessing table:")
        print(df)
        
        self.df = df
        
        nbreak = np.array(self.df["nbreak"].values)
        
        print(f"The nbreak values are {nbreak}")
        
        if save_table:
            filename = "doppelgangers_table.csv"
            df.to_csv(filename, index = False)
            print(f"Saved the doppelgangers table to: {filename}")
        
        # # New target:
        # target = df[df["subdir"] == "7945"]
        # target_dict = target.to_dict(orient = "series")
        # print(target_dict)
    
    def plot_doppelgangers(self, 
                           outdir: str,
                           # deciding which plots to make
                           plot_NS: bool = True,
                           plot_NS_no_lambdas: bool = True,
                           plot_NS_errors: bool = True,
                           plot_EOS: bool = True,
                           plot_EOS_params: bool = True,
                           show_legend: bool = False,
                           keep_real_doppelgangers: bool = True):
        """
        Plot everything related to the real doppelgangers that are found in the outdir.

        Args:
            outdir (str): Outdir of real doppelgangers.

        Raises:
            ValueError: In case there are no npz files for a specific run. 
        """
        param_names = self.prior.parameter_names
        
        doppelgangers_dict = {}
        
        if "perturbations" in outdir:
            print("Looking at perturbations")
            for file in os.listdir(outdir):
                # Load it
                data = np.load(os.path.join(outdir, file))
                keys = list(data.keys())
                
                # Check the max error of Lambdas:
                max_error_Lambdas = compute_max_error(data["masses_EOS"], data["Lambdas_EOS"], self.m_target, self.Lambdas_target)
                max_error_radii = compute_max_error(data["masses_EOS"], data["radii_EOS"], self.m_target, self.r_target)
                if keep_real_doppelgangers and (max_error_Lambdas > 10.0 or max_error_radii > 0.1):
                   continue
                else:
                     # Add it
                    doppelgangers_dict[file] = {}
                    for key in keys:
                        doppelgangers_dict[file][key] = data[key]
        
                # Add TOV masses:
                doppelgangers_dict[file]["M_TOV"]= np.max(data["masses_EOS"])
        
        else:
            
            for subdir in os.listdir(outdir):
                if "948" in subdir or "8436" in subdir:
                    continue
                    
                full_subdir = os.path.join(outdir, os.path.join(subdir, "data"))
                
                # Get the final
                npz_files = [f for f in os.listdir(full_subdir) if f.endswith(".npz")]
                
                # TODO: load best if it is there otherwise get the final number
                if "best.npz" in npz_files:
                    npz_files.remove("best.npz")
                
                if len(npz_files) == 0:
                    print(f"No npz files found in {full_subdir}. Skipping")
                    continue

                ids = [int(f.split(".")[0]) for f in npz_files]
                final_id = max(ids)

                # Final npz
                final_npz = os.path.join(full_subdir, "{}.npz".format(final_id))

                # Load it
                data = np.load(final_npz)
                keys = list(data.keys())
                
                # Check the max error of Lambdas:
                max_error_Lambdas = compute_max_error(data["masses_EOS"], data["Lambdas_EOS"], self.m_target, self.Lambdas_target)
                max_error_radii = compute_max_error(data["masses_EOS"], data["radii_EOS"], self.m_target, self.r_target)
                if keep_real_doppelgangers and (max_error_Lambdas > 10.0 or max_error_radii > 0.1):
                    continue
                else:
                    # Add it
                    doppelgangers_dict[subdir] = {}
                    for key in keys:
                        doppelgangers_dict[subdir][key] = data[key]
        
                # Add TOV masses:
                doppelgangers_dict[subdir]["M_TOV"]= np.max(data["masses_EOS"])
            
        # Get colors based on MTOV mass:
        all_mtov = [doppelgangers_dict[key]["M_TOV"] for key in doppelgangers_dict.keys()]
        sorted_mtov_idx = np.argsort(all_mtov)
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(doppelgangers_dict))]
        colors = [colors[i] for i in sorted_mtov_idx]
        
        ### First the NS
        if plot_NS:
            
            if plot_NS_no_lambdas:
                plt.figure(figsize=(10, 8))
                legend_x_position = -0.3
            else:
                plt.subplots(figsize=(14, 8), nrows = 1, ncols = 2)
                plt.subplot(121)
                legend_x_position = -0.4

            # Radius
            plt.plot(self.r_target, self.m_target, color="black", linewidth = 4, label = "Target", zorder = 1e10)
            deviation_radius = 0.1
            alpha_radius = 0.75
            plt.plot(self.r_target - deviation_radius, self.m_target, color="black", linestyle = "--", linewidth = 4, alpha = alpha_radius, zorder = 1e10, label = r"$\pm {}$ km".format(deviation_radius))
            plt.plot(self.r_target + deviation_radius, self.m_target, color="black", linestyle = "--", linewidth = 4, alpha = alpha_radius, zorder = 1e10)
            for i, key in enumerate(doppelgangers_dict.keys()):
                label = f"id = {key}"
                
                r, m = doppelgangers_dict[key]["radii_EOS"], doppelgangers_dict[key]["masses_EOS"]
                if show_legend:
                    plt.plot(r, m, zorder = 1e9, label=label, color = colors[i])
                else:
                    plt.plot(r, m, zorder = 1e9, color = colors[i])

            r_min = 11
            r_max = 12.75
            plt.xlim(r_min, r_max)
            plt.ylim(1.0, 2.5)
            plt.xlabel(r"$R$ [km]")
            
            # Also show the M_TOV constraint from Hauke's paper -- first for set A
            r_ = np.linspace(r_min, r_max, 100)
            plt.fill_between(r_, 2.26 - 0.22, 2.26 + 0.45, color = "grey", alpha = 0.25, label = r"$M_{\rm{TOV}}$ (Set A)", zorder = 1)
            # plt.plot(r_, [2.26 for _ in r], color = "grey", zorder = 1)
            
            plt.fill_between(r_, 2.31 - 0.2, 2.31 + 0.08, color = "grey", alpha = 0.5, label = r"$M_{\rm{TOV}}$ (Set C)", zorder = 1)
            # plt.plot(r_, [2.31 for _ in r], color = "grey", zorder = 1)
            
            plt.ylabel(r"$M/M_{\odot}$")
            plt.grid(True)
            
            plt.legend(bbox_to_anchor=(legend_x_position, 0.5), loc='center')

            if not plot_NS_no_lambdas:
                plt.subplot(122)
                plt.plot(self.m_target, self.Lambdas_target, color="black", linewidth = 4, label = "Target", zorder = 1e10)
                for i, key in enumerate(doppelgangers_dict.keys()):
                
                    m, l = doppelgangers_dict[key]["masses_EOS"], doppelgangers_dict[key]["Lambdas_EOS"]
                    plt.plot(m, l, zorder = 1e9, color = colors[i])
                plt.xlabel(r"$M/M_{\odot}$")
                plt.ylabel(r"$\Lambda$")
                plt.yscale("log")
                plt.grid(True)
                plt.xlim(0.5, 2.5)
                plt.ylim(top = 1e5)
                
                plt.savefig("./figures/doppelgangers_NS.png", bbox_inches = "tight")
                plt.savefig("./figures/doppelgangers_NS.pdf", bbox_inches = "tight")
            else:
                plt.savefig("./figures/doppelgangers_NS_no_lambdas.png", bbox_inches = "tight")
                plt.savefig("./figures/doppelgangers_NS_no_lambdas.pdf", bbox_inches = "tight")
                
            plt.close()
        
        if plot_NS_errors:
            # Errors lambdas
            print("Plotting the errors on Lambdas")
            plt.figure(figsize=(14, 8))
            masses = jnp.linspace(1.2, 2.1, 500)
            lambdas_target = jnp.interp(masses, self.m_target, self.Lambdas_target, left = 0, right = 0)
            for i, key in enumerate(doppelgangers_dict.keys()):
                m, l = doppelgangers_dict[key]["masses_EOS"], doppelgangers_dict[key]["Lambdas_EOS"]
                lambdas_model = jnp.interp(masses, m, l, left = 0, right = 0)
                plt.plot(masses, abs(lambdas_model - lambdas_target), label = f"id = {key}", color = colors[i])
            
            if show_legend:    
                plt.legend()
            plt.ylim(bottom = 1e-2)
            plt.xlabel(r"$M/M_{\odot}$")
            plt.ylabel(r"abs($\Delta \Lambda$)")
            # plt.yscale("log")
            plt.savefig("./figures/doppelgangers_NS_errors_L.png", bbox_inches = "tight")
            plt.savefig("./figures/doppelgangers_NS_errors_L.pdf", bbox_inches = "tight")
            plt.close()
            
            # Errors lambdas
            print("Plotting the errors on radii")
            plt.figure(figsize=(14, 8))
            radii_target = jnp.interp(masses, self.m_target, self.r_target, left = 0, right = 0)
            for i, key in enumerate(doppelgangers_dict.keys()):
                m, r = doppelgangers_dict[key]["masses_EOS"], doppelgangers_dict[key]["radii_EOS"]
                radii_model = jnp.interp(masses, m, r, left = 0, right = 0)
                plt.plot(masses, 1000 * abs(radii_model - radii_target), label = f"id = {key}", color = colors[i])
                
            if show_legend:
                plt.legend()
            plt.ylim(bottom = 1e-4)
            plt.xlabel(r"$M/M_{\odot}$")
            plt.ylabel(r"abs($\Delta R$ [m])")
            # plt.yscale("log")
            plt.savefig("./figures/doppelgangers_NS_errors_R.png", bbox_inches = "tight")
            plt.savefig("./figures/doppelgangers_NS_errors_R.pdf", bbox_inches = "tight")
            plt.close()
        
        ### Second: the EOS
        
        if plot_EOS:
            print("Plotting EOS curves")
            plt.subplots(figsize = (14, 10), nrows = 1, ncols = 2)
            for i, key in enumerate(doppelgangers_dict.keys()):
                
                if i > 50:
                    break
                
                label = f"id = {key}"
                params = {k: doppelgangers_dict[key][k] for k in param_names}
                
                out = self.transform.forward(params)
                
                n = out["n"] / jose_utils.fm_inv3_to_geometric / 0.16
                e = out["e"] / jose_utils.MeV_fm_inv3_to_geometric
                p = out["p"] / jose_utils.MeV_fm_inv3_to_geometric
                cs2 = out["cs2"]
                
                n = out["n"] / jose_utils.fm_inv3_to_geometric / 0.16
                e = out["e"] / jose_utils.MeV_fm_inv3_to_geometric
                p = out["p"] / jose_utils.MeV_fm_inv3_to_geometric
                cs2 = out["cs2"]
                
                # # p_c is saved in log space, so take exp and make sure we do unit conversion properly
                # p_c_array = jnp.exp(out["p_c_EOS"]) / jose_utils.MeV_fm_inv3_to_geometric
                # # Get the p_c either at TOV mass - or - right at 2 M_odot:
                # p_c = p_c_array[-1]
                # # p_c = jnp.interp(2.0, out["masses_EOS"], p_c_array)
                
                # n_TOV = get_n_TOV(n, p, p_c)
                
                # Limit everything to be up to the maximum saturation density
                nmin = 0.5 
                nmax = 1.2
                mask = (n > nmin) * (n < nmax)
                n, e, p, cs2 = n[mask], e[mask], p[mask], cs2[mask]
                
                mask_target = (self.n_target > nmin) * (self.n_target < nmax)
                n_target, e_target, p_target, cs2_target = self.n_target[mask_target], self.e_target[mask_target], self.p_target[mask_target], self.cs2_target[mask_target]
                
                # # Find the index at which n reaches n_TOV
                # p_TOV = jnp.interp(n_TOV, n, p)
                # e_TOV = jnp.interp(n_TOV, n, e)
                # cs2_TOV = jnp.interp(n_TOV, n, cs2)
                
                c = colors[i]
                
                plt.subplot(221)
                plt.plot(n, e, label = label, color = c)
                # plt.scatter(n_TOV, e_TOV, color = c)
                plt.plot(n_target, e_target, color = "black", label = "Target")
                plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
                plt.ylabel(r"$e$ [MeV fm$^{-3}$]")
                plt.xlim(nmin, nmax)
                
                plt.subplot(222)
                plt.plot(n, p, label = label, color = c)
                # plt.scatter(n_TOV, p_TOV, color = c)
                plt.plot(n_target, p_target, color = "black", label = "Target")
                plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
                plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
                plt.xlim(nmin, nmax)
                
                plt.subplot(223)
                plt.plot(n, cs2, label = label, color = c)
                # plt.scatter(n_TOV, cs2_TOV, color = c)
                plt.plot(n_target, cs2_target, color = "black", label = "Target")
                plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
                plt.ylabel(r"$c_s^2$")
                plt.xlim(nmin, nmax)
                plt.ylim(0, 1)
                
                plt.subplot(224)
                e_min = 200
                e_max = 1500
                mask = (e_min < e) * (e < e_max)
                mask_target = (e_min < self.e_target) * (self.e_target < e_max)
                
                plt.plot(e[mask], p[mask], label = label, color = c)
                plt.plot(self.e_target[mask_target], self.p_target[mask_target], color = "black", label = "Target")
                
                # Legend for n_TOV dots:
                legend_elements = [Line2D([0], [0], marker='o', color='black', label=r'$n_{\rm{TOV}}$', markerfacecolor='black', markersize=10)]
                plt.legend(handles=legend_elements, loc='best')
                
                # plt.scatter(e_TOV, p_TOV, color = c)
                plt.xlabel(r"$e$ [MeV fm$^{-3}$]")
                plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
                plt.xlim(e_min, e_max)
                
            plt.savefig(f"./figures/doppelgangers_EOS.png", bbox_inches = "tight")
            plt.savefig(f"./figures/doppelgangers_EOS.pdf", bbox_inches = "tight")
            plt.close()
            
        ### Now need to plot the EOS parameters:
        if plot_EOS_params and hasattr(self, "df"):
            print("Plotting the EOS parameters")
            param_names_MM = [n for n in param_names if n.endswith("_sat") or n.endswith("_sym")]
            param_names_MM += ["subdir"]
            
            sns.pairplot(self.df[param_names_MM], hue = "subdir", plot_kws={"s": 100})
            plt.savefig("./figures/doppelgangers_EOS_params.png", bbox_inches = "tight")
            plt.savefig("./figures/doppelgangers_EOS_params.pdf", bbox_inches = "tight")
            plt.close()

    def export_target_EOS(self, dir: str = "./real_doppelgangers/7945/data/"):
        """Get my own target file"""
        
        npz_files = [f for f in os.listdir(dir) if f.endswith(".npz")]
        if "best.npz" in npz_files:
            npz_files.remove("best.npz")
            
        # Get final number
        numbers = [int(file.split(".")[0]) for file in npz_files]
        final_number = max(numbers)
        npz_file = os.path.join(dir, f"{final_number}.npz")
        data = np.load(npz_file)
        keys: list[str] = data.keys()
        for key in keys:
            if key.endswith("_EOS"):
                continue
            
        # Get the params
        params = {k: data[k] for k in self.prior.parameter_names}
            
        # Get the EOS
        out = self.transform.forward(params)
        
        n, p, e, cs2 = out["n"], out["p"], out["e"], out["cs2"]
        n = n / jose_utils.fm_inv3_to_geometric
        p = p / jose_utils.MeV_fm_inv3_to_geometric
        e = e / jose_utils.MeV_fm_inv3_to_geometric
        
        # Save it as .dat file:
        data = np.column_stack((n, e, p, cs2))
        np.savetxt('my_target_microscopic.dat', data, delimiter=' ')
        
        m, r, l = out["masses_EOS"], out["radii_EOS"], out["Lambdas_EOS"]
        
        # Save it as .dat file:
        data = np.column_stack((r, m, l))
        np.savetxt('my_target_macroscopic.dat', data, delimiter=' ')

        print("Saved new target!")        

#################
### UTILITIES ###
#################

def get_learning_rate(i, start_lr, total_epochs):
    if i < total_epochs // 2:
        return start_lr
    else:
        return start_lr / 2.0
    
def get_learning_rate_micro(i, start_lr, total_epochs, final_lr_multiplier = 0.01):
    """Linearly go from start_lr to 0.1 * start_lr"""
    if i < total_epochs // 2:
        return start_lr
    else:
        return start_lr - (i - total_epochs // 2) * (start_lr - final_lr_multiplier * start_lr) / (total_epochs // 2)

def get_n_TOV(n, p, p_c):
    """
    We find n_TOV by checking where we achieve the central pressure.

    Args:
        n (_type_): _description_
        p (_type_): _description_
        p_c (_type_): _description_
    """
    n_TOV = jnp.interp(p_c, p, n)
    return n_TOV

def compute_max_error(mass_1: Array, 
                      x_1: Array, 
                      mass_2: Array, 
                      x_2: Array, 
                      m_min: float = 1.2, 
                      m_max: float = 2.25,
                      nb_masses: int = 500) -> float:
    """
    Compute the maximal deviation between Lambdas or radii ("x") for two given NS families. Note that we interpolate on a given grid
    The default maximal mass is chosen from Hauke's EOS.

    Args:
        mass_1 (Array): Mass array of the first family.
        Lambdas_1 (Array): Lambdas array of the first family.
        mass_2 (Array): Mass array of the second family.
        Lambdas_2 (Array): Lambdas array of the second family.

    Returns:
        float: Maximal deviation found for the Lambdas.
    """
    masses = jnp.linspace(m_min, m_max, nb_masses)
    my_x_1 = jnp.interp(masses, mass_1, x_1, left = 0, right = 0)
    my_x_2 = jnp.interp(masses, mass_2, x_2, left = 0, right = 0)
    errors = abs(my_x_1 - my_x_2)
    return max(errors)

    

############
### MAIN ### 
############

def get_sine_EOS(break_density = 2.0):
    
    # TODO: can do this experiment to see if we can get a sine wave from the MM+CSE model, to check "coverage"
    
    # Load Hauke's EOS
    data = np.loadtxt("./my_target_microscopic.dat")
    n_target, cs2_target = data[:, 0] / 0.16, data[:, 3]
    
    # Start a sine wave after the break density
    n = np.linspace(0.1, 6, 1000)
    cs2_target = np.interp(n, n_target, cs2_target)
    e = np.zeros_like(n)
    p = np.zeros_like(n)
    
    cs2_at_break = np.interp(break_density, n, cs2_target)
    sine_wave = cs2_at_break + 0.25 * np.sin((n - break_density) * np.pi)
    
    cs2 = np.where(n < break_density, cs2_target, sine_wave)

    # Save as target as .dat file:
    data = np.column_stack((n * 0.16, e, p, cs2))
    np.savetxt("./sine/my_sine_wave_microscopic.dat", data, delimiter=' ')
    
    # Dummy values for the mass, radius and Lambdas
    m = np.zeros_like(n)
    r = np.zeros_like(n)
    l = np.zeros_like(n)
    
    data = np.column_stack((m, r, l))
    np.savetxt("./sine/my_sine_wave.dat", data, delimiter=' ')
    
    # # Make the plot
    # plt.figure(figsize=(12, 6))
    # plt.plot(n, cs2, label = "Sine wave", color = "red", zorder = 4)
    # plt.scatter(n, cs2, color = "red", zorder = 4)
    # plt.savefig("./figures/sine_wave.pdf", bbox_inches = "tight")
    # plt.close()
    
def load_starting_params(parameter_names: list[str],
                         filename: str = "../postprocessing/starting_points.txt",
                         random_samples_dir: str = "../benchmarks/random_samples"):
    """Load good starting points (already low score) as found in the batch of random samples"""
    
    filenames = np.loadtxt(filename, dtype = str)
    starting_params = load_starting_params_from_samples(parameter_names, filenames, random_samples_dir=random_samples_dir)
            
    return filenames, starting_params

def load_starting_params_from_samples(parameter_names: list[str],
                                      filenames: list[str],
                                      random_samples_dir: str = "../benchmarks/random_samples"):
    starting_params = []
    for i, file in enumerate(filenames):
        params_dict = {}
        data = np.load(os.path.join(random_samples_dir, file))
        for name in parameter_names:
            params_dict[name] = float(data[name])
        starting_params.append(params_dict)
    return starting_params


def initialize_walkers(prior: CombinePrior,
                       seed: int = 42) -> dict:
    """
    Initialize the walker parameters in the EOS space given the random seed.

    Returns:
        dict: Dictionary of the starting parameters.
    """
    
    
    key = jax.random.split(jax.random.PRNGKey(seed))
    params = prior.sample(key, 1)
        
    # This is needed, otherwise JAX will scream
    for key, value in params.items():
        if isinstance(value, jnp.ndarray):
            params[key] = value.at[0].get()
                    
    return params
    

def main(N_runs: int = 0,
         from_starting_points: bool = True, # whether to start from the given starting points from benchmark random samples
         fixed_CSE: bool = False, # use a CSE, but have it fixed, vary only the metamodel
         metamodel_only = False, # only use the metamodel, no CSE used at all
         which_score: str = "macro" # score function to be used for optimization. Recommended: "macro"
         ):
    
    ### SETUP
    
    # Prior
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

    # Vary the CSE (i.e. include in the prior if used, and not set to fixed)
    if not metamodel_only and not fixed_CSE:
        # CSE priors
        prior_list.append(UniformPrior(1.0 * 0.16, 2.0 * 0.16, parameter_names=[f"nbreak"]))
        for i in range(NB_CSE):
            left = 2.0 * 0.16
            right = NMAX
            prior_list.append(UniformPrior(left, right, parameter_names=[f"n_CSE_{i}"]))
            prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{NB_CSE}"]))
    
    # Combine the prior
    prior = CombinePrior(prior_list)
    
    # prior = utils.prior
    sampled_param_names = prior.parameter_names
    name_mapping = (sampled_param_names, ["logpc_EOS", "masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])
    transform = utils.MicroToMacroTransform(name_mapping, 
                                            nmax_nsat=utils.NMAX_NSAT, 
                                            nb_CSE=utils.NB_CSE,
                                            fixed_params = {})
    
    # Choose the learning rate
    if fixed_CSE:
        learning_rate = 1e3
    else:
        learning_rate = 1e-3
    
    # Initialize random doppelganger: this is to run postprocessing scripts below
    doppelganger = DoppelgangerRun(prior, transform, which_score, -1, nb_steps = 300)
    
    ### Optimizer run
    if from_starting_points:
        
        ### Option 1: give your directories
        # filenames = ["3679.npz", "3843.npz"]
        filenames = ["3133.npz"]
        starting_params = load_starting_params_from_samples(prior.parameter_names, filenames)
        
        ### Option 2: load preselected directories
        # filenames, starting_params = load_starting_params(prior.parameter_names)
        
        print(starting_params)
        N_runs = len(starting_params)
        print(f"N_runs is now set to {N_runs}")
        fixed_params_keys = []
    else:
        fixed_params_keys = []
        
    s = 26096
    seed = s
    np.random.seed(s)
    
    for i in range(N_runs):
        if from_starting_points:
            idx = filenames[i].split(".")[0]
            print(f" ====================== Run {i + 1} / {N_runs} with dir idx {idx} ======================")
            seed = int(idx)
        else:
            print(f" ====================== Run {i + 1} / {N_runs} with seed {seed} ======================")
        # Initialize walker either randomly or from the given starting points
        if from_starting_points:
            params = starting_params[i]
        else:
            # Generate the seed for the next run
            params = initialize_walkers(prior, seed)
            
        # Get the desired fixed params
        print(params.keys())
        fixed_params = {key: params[key] for key in list(params.keys()) if key in fixed_params_keys}
        
        print("fixed_params")
        print(fixed_params)
        
        # Define the doppelganger object
        doppelganger = DoppelgangerRun(prior, 
                                       transform, 
                                       which_score, 
                                       seed, 
                                       nb_steps = 200, 
                                       learning_rate = learning_rate,
                                       fixed_params=fixed_params,
                                       load_params = False)
        
        # Do the run
        doppelganger.run(params)
        
        # Generate new seed for next run
        seed = np.random.randint(0, 100_000)
        
    # doppelganger.export_target_EOS()
    # doppelganger.perturb_doppelganger(seed = 125, nb_perturbations=1)
        
    # doppelganger.finetune_doppelganger(seed = 750)
    
    # # Plot the MTOV correlations?
    # doppelganger.plot_pressure_mtov_correlations()
    
    # ### Meta plots of the final "real" doppelgangers
    
    # final_outdir = "./outdir/"
    # doppelganger.get_table(outdir=final_outdir, keep_real_doppelgangers = True, save_table = False)
    # doppelganger.plot_doppelgangers(final_outdir, keep_real_doppelgangers = True)
    
    # doppelganger.random_sample(outdir="./test_new_prior/")
    
    print("DONE")
    
if __name__ == "__main__":
    main()