################
### PREAMBLE ###
################

import os
import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", False)
print(jax.devices())

import jax.numpy as jnp
from jimgw.prior import CombinePrior
import joseTOV.utils as jose_utils

import paper_jose.utils as utils
import paper_jose.inference.utils_plotting as utils_plotting
plt.rcParams.update(utils_plotting.mpl_params)

from paper_jose.debug.rahul import tov_solve

# Disable TeX rendering
plt.rcParams.update({"text.usetex": False})

class Debugger:
    """A class that can be used to debug the EOS and TOV code."""
    
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
              save_name: str = "debug_bad",
              make_sample: bool = False,
              seed: int = 0,
              solve_rahul: bool = True):
        
        # Load from random samples:
        if make_sample:
            
            accept = False
            jaxkey = jax.random.PRNGKey(seed)
            while not accept:
            
                # Fetch the parameters
                jaxkey, jaxsubkey = jax.random.split(jaxkey)
                params = self.prior.sample(jaxsubkey, 1)
                for key, value in params.items():
                    if isinstance(value, jnp.ndarray):
                        params[key] = value.at[0].get()
                print(f"params: {params}")
                
                # Solve TOV:
                out = self.transform.forward(params)
                
                # Get the NS properties and check them out
                m, r, l = out["masses_EOS"], out["radii_EOS"], out["Lambdas_EOS"]
                mtov = np.max(m)
                accept = True
                # if mtov < 1.6:
            
        else:
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
        
        n, p, e, h, dloge_dlogp, cs2 = out["n"], out["p"], out["e"], out["h"], out["dloge_dlogp"], out["cs2"]
        
        n = n / jose_utils.fm_inv3_to_geometric / 0.16
        p = p / jose_utils.MeV_fm_inv3_to_geometric
        e = e / jose_utils.MeV_fm_inv3_to_geometric
        
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
        
        if solve_rahul:
            # Solve the TOV equations
            out_rahul = tov_solve(n, p, e, cs2)
            r_rahul, m_rahul, l_rahul, p_c_rahul = out_rahul
        
        for nmax, suffix in zip([2, 25], ["zoom", "normal"]):
            # EOS
            mask = n < nmax
            plt.subplots(nrows = 2, ncols = 2, figsize = figsize)
            plt.subplot(221)
            plt.plot(n[mask], p[mask])
            plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
            plt.ylabel(r"$p$ [MeV fm${}^{-3}$]")
            
            plt.subplot(222)
            plt.plot(n[mask], e[mask])
            plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
            plt.ylabel(r"$e$ [MeV fm${}^{-3}$]")
            
            plt.subplot(223)
            plt.plot(n[mask], cs2[mask])
            plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
            plt.ylabel(r"cs2")
            
            plt.subplot(224)
            plt.plot(n[mask], h[mask])
            plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
            plt.ylabel(r"h")
            
            plt.savefig(f"./figures/debug_eos_{suffix}.png")
            plt.close()
        
        # TOV
        plt.subplots(nrows = 1, ncols = 2, figsize = figsize)
        plt.subplot(121)
        plt.plot(r, m, label = "jester")
        if solve_rahul:
            plt.plot(r_rahul, m_rahul, linestyle = "--", color = "red", label = "Rahul")
        plt.xlabel(r"Radius [km]")
        plt.ylabel(r"Mass [$M_\odot$]")
        
        plt.subplot(122)
        plt.plot(m, l, label = "jester")
        if solve_rahul:
            plt.plot(m_rahul, l_rahul, linestyle = "--", color = "red", label = "Rahul")
        plt.xlabel(r"Mass [$M_\odot$]")
        plt.ylabel(r"Lambda")
        plt.yscale("log")
        plt.savefig(f"./figures/{save_name}_tov.png")
        plt.close() 
        
        logpc_EOS = out["logpc_EOS"]
        pc = jnp.exp(logpc_EOS)
        pc = pc / jose_utils.MeV_fm_inv3_to_geometric
        
        plt.plot(pc, m)
        plt.xlabel(r"pc_EOS MeV fm-3")
        plt.ylabel(r"Mass [$M_\odot$]")
        plt.legend()
        plt.savefig(f"./figures/{save_name}_pc_vs_m.png")
        plt.close() 
    
    
def main():
    ### Choose to create own prior or can also fetch the one from the utils
    prior = utils.prior
    transform = utils.MicroToMacroTransform(name_mapping=utils.name_mapping)
    debugger = Debugger(prior=prior, transform=transform, nb_samples=10_000)
    
    # bad seeds: 5, 8
    debugger.debug(idx_number = 1, make_sample = True, seed = 8)
    exit()

if __name__ == "__main__":
    main()