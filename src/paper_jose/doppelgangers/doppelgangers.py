"""
Find doppelgangers with Jose
"""

################
### PREAMBLE ###
################

import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
import shutil

import os
import tqdm
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
import seaborn as sns

##########################
### DOPPELGANGER CLASS ###
##########################

class DoppelgangerRun:
    
    def __init__(self,
                 prior: CombinePrior,
                 transform: utils.MicroToMacroTransform,
                 random_seed: int,
                 # Optimization hyperparameters
                 nb_steps: int = 200,
                 optimization_sign: float = -1, 
                 learning_rate: float = 1e-3, 
                 # Plotting
                 outdir_name: str = "./outdir/",
                 plot_mse: bool = True,
                 plot_final_errors: bool = True,
                 plot_target: bool = True,
                 clean_outdir: bool = False,
                 # Target
                 micro_target_filename: str = "./36022_microscopic.dat",
                 macro_target_filename: str = "./36022_macroscopic.dat",
                 ):
        
        # Set prior and transform
        self.prior = prior
        self.transform = transform
        
        # Load micro and macro targets
        data = np.loadtxt(micro_target_filename)
        self.n_target, self.e_target, self.p_target, self.cs2_target = data[:, 0] / 0.16, data[:, 1], data[:, 2], data[:, 3]
        
        data = np.genfromtxt(macro_target_filename, skip_header=1, delimiter=" ").T
        self.r_target, self.m_target, self.Lambdas_target = data[0], data[1], data[2]
        
        # Define the doppelganger score function
        self.score_fn = lambda params: doppelganger_score(params, transform, self.m_target, self.Lambdas_target, self.r_target)
        
        # Save the final things
        self.nb_steps = nb_steps
        self.optimization_sign = optimization_sign
        self.learning_rate = learning_rate
        
        # Outdir and plotting stuff
        self.outdir_name = outdir_name
        self.set_seed(random_seed)
        self.subdir_name = os.path.join(self.outdir_name, str(random_seed))
            
        if clean_outdir:
            shutil.rmtree(self.outdir_name, ignore_errors=True)
        
        if not os.path.exists(self.outdir_name):
            print("Creating the outdir")
            os.makedirs(self.outdir_name)
            
        self.plot_mse = plot_mse
        self.plot_final_errors = plot_final_errors
        self.plot_target = plot_target
        
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
        
    def initialize_walkers(self) -> dict:
        """
        Initialize the walker parameters in the EOS space given the random seed.

        Returns:
            dict: Dictionary of the starting parameters.
        """
        
        jax_key = jax.random.PRNGKey(self.random_seed)
        jax_key, jax_subkey = jax.random.split(jax_key)
        params = self.prior.sample(jax_subkey, 1)
            
        # This is needed, otherwise JAX will scream
        for key, value in params.items():
            if isinstance(value, jnp.ndarray):
                params[key] = value.at[0].get()
                        
        return params
        
    def run(self, params: dict) -> None:
        """
        Run the optimization loop for the doppelganger problem.

        Args:
            params (dict): Starting parameters.

        """
        
        print("Starting parameters:")
        print(params)
        
        # Define the score function in the desired jax format
        self.score_fn = jax.value_and_grad(self.score_fn, has_aux=True)
        self.score_fn = jax.jit(self.score_fn)
        
        print("Computing by gradient ascent . . .")
        pbar = tqdm.tqdm(range(self.nb_steps))
        for i in pbar:
            ((score, aux), grad) = self.score_fn(params)
            m, r, l = aux
            
            if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)):
                print(f"Iteration {i} has NaNs. Exiting the computing loop now")
                break
            
            pbar.set_description(f"Iteration {i}: score = {score}")
            npz_filename = os.path.join(self.subdir_name, f"data/{i}.npz")
            np.savez(npz_filename, masses_EOS = m, radii_EOS = r, Lambdas_EOS = l, score = score, **params)
            
            params = {key: value + self.optimization_sign * self.learning_rate * grad[key] for key, value in params.items()}
            
            max_error = compute_max_error(m, l, self.m_target, self.Lambdas_target)
            # if max_error < 10.0:
            #     print(f"Early stopping at iteration {i} with max error: {max_error}")
            #     break
            
        print("Computing DONE")
    
    def plot_NS(self, m_min: float = 1.2):
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
        save_name = os.path.join(self.subdir_name, "figures/doppelganger_trajectory.png")
        print(f"Saving to: {save_name}")
        plt.savefig(save_name, bbox_inches = "tight")
        plt.close()
        
        if self.plot_final_errors:
            plt.figure(figsize = (12, 6))
            # Plot the errors of the final M, Lambda, R
            m_final = all_masses_EOS[-1]
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
            save_name = os.path.join(self.subdir_name, "figures/final_errors.png")
            print(f"Saving to: {save_name}")
            plt.savefig(save_name, bbox_inches = "tight")
            
            print(f"FINAL RESULT: The max error was: {max_error}")
            
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
            
    def show_table(self, show_real_doppelgangers: bool = False):
        """
        Postprocessing utility to show the table of the doppelganger runs.

        Args:
            outdir (str): Outdir with a collection, ideally, of real doppelgangers. 
        """
        
        subdirs = os.listdir(self.outdir_name)
        output = defaultdict(list)
        
        for subdir in subdirs:
            # Get the datadir
            data_dir = os.path.join(self.outdir_name, subdir, "data")
            
            # Get the final iteration number from the filenames
            npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
            numbers = [int(file.split(".")[0]) for file in npz_files]
            try:
                final_number = max(numbers)
            except ValueError as e:
                print(f"There was a problem for subdir {subdir}: {e}")
                continue
            
            # Get the datadir
            output["subdir"].append(subdir)
            npz_file = os.path.join(data_dir, f"{final_number}.npz")
            data = np.load(npz_file)
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
        
        if show_real_doppelgangers:
            # Only limit to those with max error below 10:
            df = df[df["max_error"] < 10.0]
        
        print("Postprocessing table:")
        print(df)
        
        self.df = df
    
    def plot_doppelgangers(self, outdir: str):
        """
        Plot everything related to the real doppelgangers that are found in the outdir.

        Args:
            outdir (str): Outdir of real doppelgangers.

        Raises:
            ValueError: In case there are no npz files for a specific run. 
        """
        
        ### First the NS
        
        doppelgangers_dict = {}
        for subdir in os.listdir(outdir):
            full_subdir = os.path.join(outdir, os.path.join(subdir, "data"))

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
        print("Plotting NS families")
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
        
        # Errors lambdas
        print("Plotting the errors on Lambdas")
        plt.figure(figsize=(14, 8))
        masses = jnp.linspace(1.2, 2.1, 500)
        lambdas_target = jnp.interp(masses, self.m_target, self.Lambdas_target, left = 0, right = 0)
        for key in doppelgangers_dict.keys():
            m, l = doppelgangers_dict[key]["masses_EOS"], doppelgangers_dict[key]["Lambdas_EOS"]
            lambdas_model = jnp.interp(masses, m, l, left = 0, right = 0)
            plt.plot(masses, abs(lambdas_model - lambdas_target), label = f"id = {key}")
            
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
        for key in doppelgangers_dict.keys():
            m, r = doppelgangers_dict[key]["masses_EOS"], doppelgangers_dict[key]["radii_EOS"]
            radii_model = jnp.interp(masses, m, r, left = 0, right = 0)
            plt.plot(masses, abs(radii_model - radii_target), label = f"id = {key}")
            
        plt.legend()
        plt.ylim(bottom = 1e-4)
        plt.xlabel(r"$M/M_{\odot}$")
        plt.ylabel(r"abs($\Delta R$ [km])")
        # plt.yscale("log")
        plt.savefig("./figures/doppelgangers_NS_errors_R.png", bbox_inches = "tight")
        plt.savefig("./figures/doppelgangers_NS_errors_R.pdf", bbox_inches = "tight")
        plt.close()
        
        ### Second: the EOS
        param_names = self.prior.parameter_names
        
        print("Plotting EOS curves")
        for max_nsat, extra_id in zip([25.0, 2.0], ["MM_CSE", "MM"]):
            plt.subplots(figsize = (14, 10), nrows = 1, ncols = 2)
            for key in doppelgangers_dict.keys():
                
                label = f"id = {key}"
                params = {k: doppelgangers_dict[key][k] for k in param_names}
                
                out = self.transform.forward(params)
                
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
        print("Plotting the EOS parameters")
        param_names_MM = [n for n in param_names if n.endswith("_sat") or n.endswith("_sym")]
        param_names_MM += ["subdir"]
        
        sns.pairplot(self.df[param_names_MM], hue = "subdir", plot_kws={"s": 100})
        plt.savefig("./figures/doppelgangers_EOS_params.png", bbox_inches = "tight")
        plt.savefig("./figures/doppelgangers_EOS_params.pdf", bbox_inches = "tight")
        plt.close()

#################
### SCORE FNs ###
#################

def compute_max_error(mass_1: Array, Lambdas_1: Array, mass_2: Array, Lambdas_2: Array, m_min: float = 1.2, m_max: float = 2.1) -> float:
    """
    Compute the maximal deviation between Lambdas for two given NS families. Note that we interpolate on a given grid

    Args:
        mass_1 (Array): Mass array of the first family.
        Lambdas_1 (Array): Lambdas array of the first family.
        mass_2 (Array): Mass array of the second family.
        Lambdas_2 (Array): Lambdas array of the second family.

    Returns:
        float: Maximal deviation found for the Lambdas.
    """
    masses = jnp.linspace(m_min, m_max, 500)
    my_Lambdas_model = jnp.interp(masses, mass_1, Lambdas_1, left = 0, right = 0)
    my_Lambdas_target = jnp.interp(masses, mass_2, Lambdas_2, left = 0, right = 0)
    errors = abs(my_Lambdas_model - my_Lambdas_target)
    return max(errors)

def mrse(x: Array, y: Array) -> float:
    """Relative mean squared error between x and y."""
    return jnp.mean(((x - y) / y) ** 2)

def mrae(x: Array, y: Array) -> float:
    """Relative mean absolute error between x and y."""
    return jnp.mean(jnp.abs((x - y) / y))

def doppelganger_score(params: dict,
                       transform: utils.MicroToMacroTransform,
                       m_target: Array,
                       Lambdas_target: Array, 
                       r_target: Array,
                       # For the masses for interpolation
                       m_min = 1.2,
                       m_max = 2.1,
                       N_masses: int = 100,
                       # Hyperparameters for score fn
                       alpha: float = 1.0,
                       beta: float = 0.0,
                       gamma: float = 2.0,
                       return_aux: bool = True,
                       error_fn: Callable = mrse) -> float:
    
    """
    Doppelganger score function. 
    TODO: type hints

    Returns:
        _type_: _description_
    """
    
    # Solve the TOV equations
    out = transform.forward(params)
    m_model, r_model, Lambdas_model = out["masses_EOS"], out["radii_EOS"], out["Lambdas_EOS"]
    
    mtov_model = m_model[-1]
    mtov_target = m_target[-1]
    
    # Get a mass array and interpolate NaNs on top of it
    masses = jnp.linspace(m_min, m_max, N_masses)
    my_Lambdas_model = jnp.interp(masses, m_model, Lambdas_model, left = 0, right = 0)
    my_Lambdas_target = jnp.interp(masses, m_target, Lambdas_target, left = 0, right = 0)
    
    my_r_model = jnp.interp(masses, m_model, r_model, left = 0, right = 0)
    my_r_target = jnp.interp(masses, m_target, r_target, left = 0, right = 0)
    
    # Define separate scores
    score_lambdas = error_fn(my_Lambdas_model, my_Lambdas_target)
    score_r       = error_fn(my_r_model, my_r_target)
    score_mtov    = error_fn(mtov_model, mtov_target)
    
    score = alpha * score_lambdas + beta * score_r + gamma * score_mtov
    
    if return_aux:
        return score, (m_model, r_model, Lambdas_model)
    else:
        return score
       
############
### MAIN ### 
############

def main(metamodel_only = False, N_runs: int = 1):
    
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
    
    # Get a doppelganger score
    sampled_param_names = prior.parameter_names
    name_mapping = (sampled_param_names, ["masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])
    transform = utils.MicroToMacroTransform(name_mapping, nmax_nsat=NMAX_NSAT, nb_CSE=NB_CSE)
    
    ### Optimizer run
    np.random.seed(63)
    for i in range(N_runs):
        seed = np.random.randint(0, 10_000)
        print(f" ====================== Run {i + 1} / {N_runs} with seed {seed} ======================")
        
        doppelganger = DoppelgangerRun(prior, transform, seed)
        
        params = doppelganger.initialize_walkers()
        # doppelganger.run(params)
        # doppelganger.plot_NS()
    
    # ### Postprocessing with result:
    doppelganger.show_table(show_real_doppelgangers = True) # do a meta-analysis of the runs
    # final_outdir = "./real_doppelgangers/"
    ### doppelganger.plot_doppelgangers(final_outdir)
    
    print("DONE")
    
if __name__ == "__main__":
    main()