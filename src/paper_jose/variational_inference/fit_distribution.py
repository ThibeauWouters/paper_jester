"""
Can we fit simple distributions in a variational inference like manner and learn something from that?
"""

################
### PREAMBLE ###
################

import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"
import corner
import shutil

import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import jax
jax.config.update("jax_enable_x64", False)
# jax.config.update('jax_platform_name', 'cpu')
print(jax.devices())

import jax.numpy as jnp
from jimgw.prior import UniformPrior, CombinePrior
from jaxtyping import Array

import optax

from paper_jose.universal_relations.universal_relations import UniversalRelationBreaker
import paper_jose.utils as utils
import paper_jose.inference.utils_plotting as utils_plotting
plt.rcParams.update(utils_plotting.mpl_params)
import seaborn as sns

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

TARGET_EOS_DIR = "../doppelgangers/real_doppelgangers/7945/data/"

class GaussianPrior:
    
    """Multivariate Gaussian distribution of which covariance matrix entries are parameters"""
    
    def __init__(self, 
                 mean: Array,
                 param_names: list[str]):
        
        self.mean = mean
        self.param_names = param_names
        self.n_dim = len(mean)
        
    def get_covariance(self,
                       covariance_parameters: dict):
        
        cov = jnp.zeros((self.n_dim, self.n_dim))
        
        # Fill with the appropriate entries:
        for i, param_name in enumerate(self.param_names):
            cov = cov.at[i, i].set(covariance_parameters[f"sigma_{param_name}_{param_name}"])
            for j in range(i+1, self.n_dim):
                other_param_name = self.param_names[j]
                cov = cov.at[i, j].set(covariance_parameters[f"sigma_{param_name}_{other_param_name}"])
                cov = cov.at[j, i].set(cov[i, j])
        
        return cov
        
    def sample(self, 
               covariance_parameters: dict, 
               key: jax.random.PRNGKey,
               nb_samples: int = 100) -> Array:
        
        z = jax.random.normal(key, shape=(nb_samples, len(self.mean)))
        cov = self.get_covariance(covariance_parameters)
        L = jnp.linalg.cholesky(cov)
        x = self.mean + z @ L.T
        
        return x
    
    def add_name(self, x: Array):
        x_named = {name: x[i] for i, name in enumerate(self.param_names)}
        return x_named
    
    
class DistributionFitter:
    
    def __init__(self, 
                 prior: CombinePrior = None,
                 transform: utils.MicroToMacroTransform = None,
                 target_eos_dir: str = TARGET_EOS_DIR,
                 nb_samples: int = 20,
                 nb_samples_plot: int = 1_000,
                 nb_gradient_steps: int = 100,
                 learning_rate: float = 3e2,
                 clean_outdir: bool = True):
        
        # Set some attributes
        self.target_eos_dir = target_eos_dir
        self.nb_samples = nb_samples
        self.nb_samples_plot = nb_samples_plot
        self.nb_gradient_steps = nb_gradient_steps
        self.learning_rate = learning_rate
        self.transform = transform
        self.clean_outdir = clean_outdir
        
        # Load the target EOS
        self.target_eos = load_final_eos_from_dir(self.target_eos_dir)
        self.naming = prior.parameter_names
        print("self.naming")
        print(self.naming)
        self.true_values = {name: self.target_eos[name] for name in self.naming}
        self.out_true = transform.forward(self.true_values)
        self.m_true, self.r_true, self.l_true = self.out_true["masses_EOS"], self.out_true["radii_EOS"], self.out_true["Lambdas_EOS"]
        self.true_r14 = get_R14(self.m_true, self.r_true)
        
        # Get the standard deviations
        widths = []
        for prior in prior.base_prior:
            widths.append(prior.xmax - prior.xmin)
        scale_factor = 0.25
        widths = scale_factor * np.array(widths)
        self.widths = widths
        print(f"Standard deviations of the Gaussian (rescaled widths) are: {widths}")
        
        mean = np.array([self.target_eos[param_name] for param_name in self.naming])
        print(f"Mean of the Gaussian is: {mean}")
        
        # Now define a multivariate Gaussian
        self.gaussian = GaussianPrior(mean, self.naming)
        
        # Get the score_fn
        lambda_score_fn = lambda params: score_fn(params, self.transform, self.gaussian, nb_samples=nb_samples, return_aux=True)
        lambda_score_fn = jax.value_and_grad(lambda_score_fn, has_aux=True)
        lambda_score_fn = jax.jit(lambda_score_fn)
        
        self.lambda_score_fn = lambda_score_fn
        
    def initialize_covariance(self):
        
        # Define the initial covariance matrix with zero off-diagonal elements
        covariance_parameters = {}
        nb_params = len(self.naming)
        for i, param_name in enumerate(self.naming):
            covariance_parameters[f"sigma_{param_name}_{param_name}"] = self.widths[i]
        for i, param_name in enumerate(self.naming):
            for j in range(i+1, nb_params):
                other_param_name = self.naming[j]
                covariance_parameters[f"sigma_{param_name}_{other_param_name}"] = 0.0

        return covariance_parameters
        
    def run(self, covariance_parameters: dict) -> None:
        
        # Get the optax right:
        gradient_transform = optax.adam(learning_rate=self.learning_rate)
        opt_state = gradient_transform.init(covariance_parameters)
        
        # Clean the outdir
        outdir = "./outdir/"
        if os.path.exists(outdir) and self.clean_outdir:
            shutil.rmtree(outdir)
            print(f"Cleaned up the directory: {outdir}")
            
            os.makedirs(outdir)
            print(f"Recreated the directory: {outdir}")
        else:
            print(f"Directory does not exist: {outdir}")
        
        print(f"Starting with covariance matrix parameters:")
        print(covariance_parameters)
        
        # Now do gradient based optimization
        pbar = tqdm.tqdm(range(self.nb_gradient_steps))
        for i in pbar:
            
            # Compute the gradient
            ((score, aux), grad) = self.lambda_score_fn(covariance_parameters)
            pbar.set_description(f"Score: {score}")
            
            m, r, l = aux["masses_EOS"], aux["radii_EOS"], aux["Lambdas_EOS"]
            n, p, e, cs2 = aux["n"], aux["p"], aux["e"], aux["cs2"]
            
            if np.any(np.isnan(m)) or np.any(np.isnan(r)) or np.any(np.isnan(l)):
                print(f"Iteration {i} has NaNs. Exiting the computing loop now")
                break
            
            # TODO: decide whether we want to save all stuff or only params
            # np.savez(f"./outdir/{i}.npz", masses_EOS = m, radii_EOS = r, Lambdas_EOS = l, n = n, p = p, e = e, cs2 = cs2, score = score, **covariance_parameters)
            np.savez(f"./outdir/{i}.npz", score = score, **covariance_parameters)

            # Do the updates
            updates, opt_state = gradient_transform.update(grad, opt_state)
            covariance_parameters = optax.apply_updates(covariance_parameters, updates)
            
        print("Done computing. Final covariance parameters")
        print(covariance_parameters)
        
    def generate_plots(self, 
                       covariance_parameters: dict, 
                       save_name: str = ""):
        """Generates the plots for a single distribution's EOS and NS properties"""
        
        print(f"Making diagnosis plots for {save_name}")
        samples = self.gaussian.sample(covariance_parameters, jax.random.PRNGKey(0), nb_samples=self.nb_samples_plot)
        samples = np.array(samples)
        
        corner.corner(samples, truths = np.array(self.gaussian.mean), labels=self.naming, **default_corner_kwargs)
        plt.savefig(f"./figures/distribution_EOS_{save_name}.png", bbox_inches="tight")
        plt.close()
        
        # Try out the TOV solver
        print(f"Calling TOV solver now")
        samples_named = self.gaussian.add_name(samples.T)
        out = jax.vmap(self.transform.forward)(samples_named)
        
        plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
        m, r, l = out["masses_EOS"], out["radii_EOS"], out["Lambdas_EOS"]
        for i in range(self.nb_samples_plot):
            plt.subplot(121)
            plt.plot(r[i], m[i], color="blue", alpha=0.25, zorder = 10)
            plt.plot(self.r_true[i], self.m_true[i], color="red", zorder = 11)
            plt.xlabel(r"Radius [km]")
            plt.ylabel(r"Mass [M$_\odot$]")
            
            
            plt.subplot(122)
            plt.plot(m[i], l[i], color="blue", alpha=0.25)
            plt.plot(self.m_true[i], self.l_true[i],  color="red", zorder = 11)
            plt.xlabel(r"Mass [M$_\odot$]")
            plt.ylabel(r"$\Lambda$")
            plt.yscale("log")
            
        plt.savefig(f"./figures/distribution_NS_{save_name}.png", bbox_inches="tight")
        plt.close()
        
        # Make a histogram of the R1.4 values
        radii_14 = jax.vmap(get_R14)(m, r)
        plt.hist(radii_14, bins=20, histtype="step", color="blue", density = True)
        plt.axvline(self.true_r14, color="red", linestyle="-", label="True R1.4")
        plt.xlabel(r"R$_{1.4}$ [km]")
        plt.ylabel(r"Density")
        plt.savefig(f"./figures/histogram_R14_{save_name}.png", bbox_inches="tight")
        plt.close()
    
def load_final_eos_from_dir(eos_dir: str) -> dict:
    """
    In the specified eos_dir, find and load the file with highest counter, which was the final file found by the doppelganger optimizer.
    Return this info, which contains all we need to know about the target EOS.
    """
    files = os.listdir(eos_dir)
    numbers = [f.split('.')[0] for f in files]
    
    max_nb = max([int(nb) for nb in numbers])
    filename = os.path.join(eos_dir, f"{max_nb}.npz")
    data = np.load(filename)
    return data
    
def get_R14(mass: Array, radii: Array):
    return jnp.interp(1.4, mass, radii)

def score_fn(covariance_parameters: dict, 
             transform: utils.MicroToMacroTransform,
             gaussian: GaussianPrior,
             nb_samples: int = 500,
             return_aux: bool = True):
    
    # Try out sample:
    # TODO: do we want the key to be fixed? Or random?
    samples = gaussian.sample(covariance_parameters, jax.random.PRNGKey(0), nb_samples=nb_samples)
    params = gaussian.add_name(samples.T)
    out = jax.vmap(transform.forward)(params)
    
    # Compute the score: get the R1.4 values for all eos
    masses, radii = out["masses_EOS"], out["radii_EOS"]
    radii_14 = jax.vmap(get_R14)(masses, radii)
    # score = jnp.max(radii_14) - jnp.min(radii_14)
    score = jnp.std(radii_14)
    
    if return_aux:
        return score, out
    else:
        return score
    
    
def main(param_names_to_run = ["E_sym", "L_sym"]):
    ### Define the prior
    my_nbreak = 2.0 * 0.16
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

    all_prior_list: list[UniformPrior] = [
        E_sym_prior,
        L_sym_prior, 
        K_sym_prior,
        Q_sym_prior,
        Z_sym_prior,

        K_sat_prior,
        Q_sat_prior,
        Z_sat_prior,
    ]
    all_param_names = [p.parameter_names[0] for p in all_prior_list]
    
    prior_list = []
    for key in all_param_names:
        if key in param_names_to_run:
            prior_list.append(all_prior_list[all_param_names.index(key)])

    # Combine the prior
    prior = CombinePrior(prior_list)
    
    # Define the transform for EOS code and TOV solver
    sampled_param_names = prior.parameter_names
    print("sampled_param_names")
    print(sampled_param_names)
    name_mapping = (sampled_param_names, ["masses_EOS", "radii_EOS", "Lambdas_EOS", "n", "p", "h", "e", "dloge_dlogp", "cs2"])
    transform = utils.MicroToMacroTransform(name_mapping, nmax_nsat=NMAX_NSAT, nb_CSE=NB_CSE)
    
    # Generate the plots
    fitter = DistributionFitter(prior = prior, 
                                transform = transform, 
                                learning_rate = 1e-1,
                                nb_samples = 100)
    
    covariance_parameters = fitter.initialize_covariance()
    # fitter.generate_plots(covariance_parameters, save_name="initial")
    
    # Do the run
    fitter.run(covariance_parameters)
    
    # Load the final obtained values
    eos = load_final_eos_from_dir("./outdir/")
    final_covariance_parameters = {}
    
    for key in eos.keys():
        if "sigma" in key:
            final_covariance_parameters[key] = eos[key]
    fitter.generate_plots(final_covariance_parameters, save_name="final")
    
if __name__ == "__main__":
    main()