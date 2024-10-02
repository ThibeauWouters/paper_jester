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
import copy
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

class MinMaxScalerJax(object):
    """
    MinMaxScaler like sklearn does it, but for JAX arrays since sklearn might not be JAX-compatible?
    
    Note: assumes that input has dynamical range: it will not catch errors due to constant input (leading to zero division)
    """
    
    def __init__(self,
                 min_val: dict = None,
                 max_val: dict = None):
        
        self.min_val = min_val
        self.max_val = max_val
    
    def transform(self, x: dict) -> dict:
        out = {k: (x[k] - self.min_val[k]) / (self.max_val[k] - self.min_val[k]) for k in x.keys()}
        return out
    
    def inverse_transform(self, x: dict) -> dict:
        out = {k: x[k] * (self.max_val[k] - self.min_val[k]) + self.min_val[k] for k in x.keys()}
        return out
    
    def fit_transform(self, x: dict) -> dict:
        # self.fit(x)
        return self.transform(x)
class MyLikelihood:
    
    def __init__(self, 
                 transform: utils.MicroToMacroTransform,
                 R1_4_target: float,
                 scaler: MinMaxScalerJax = None,
                 sigma_R: float = 1.0):
        
        self.transform = transform
        self.R1_4_target = R1_4_target
        self.sigma_R = sigma_R
        print(f"The target R1.4 is: {self.R1_4_target}")
        
        self.scaler = scaler
        
    def get_R_1_4(self, params: dict):
        # Get the R1.4 for this EOS
        params = self.scaler.inverse_transform(params)
        macro = self.transform.forward(params)
        m, r = macro["masses_EOS"], macro["radii_EOS"]
        R1_4 = jnp.interp(1.4, m, r)
        return R1_4
        
    def evaluate(self, params: dict):
        R1_4 = self.get_R_1_4(params)
        log_L = -0.5 * (R1_4 - self.R1_4_target) ** 2 / self.sigma_R ** 2
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
            E_sym_prior, # 0
            L_sym_prior, # 1
            K_sym_prior, # 2
            Q_sym_prior, # 3
            Z_sym_prior, # 4

            K_sat_prior, # 2
            Q_sat_prior, # 3
            Z_sat_prior, # 4
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
        
        # Now make the scaler
        min_val = {k: float(prior.xmin) for k, prior in zip(sampled_param_names, self.prior.base_prior)}
        max_val = {k: float(prior.xmax) for k, prior in zip(sampled_param_names, self.prior.base_prior)}
        
        scaler = MinMaxScalerJax(min_val = min_val, max_val = max_val)
        
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
        self.likelihood = MyLikelihood(self.transform, R1_4_target, scaler=scaler)
        

    def compute_hessian_values(self, use_outer_product: bool = False):
        
        
        # # TODO: remove me
        _grad_fn = jax.grad(self.likelihood.get_R_1_4)
        params_ = self.likelihood.scaler.transform(self.params)
        _grad_values = _grad_fn(params_)
        
        print("Sanity checking:")
        print("_grad_values")
        print(_grad_values)
        
        if use_outer_product:
            print("WARNING: using the outer product with the radius")
            # Take the gradient
            params = self.likelihood.scaler.transform(self.params)
            
            grad_fn = jax.grad(self.likelihood.get_R_1_4)
            grad_values = grad_fn(params)
            grad_values = np.array(list(grad_values.values()))
            
            # my_hessian_values = np.einsum('ac,bd->abcd', grad_values, grad_values)
            my_hessian_values = np.outer(grad_values, grad_values)
            my_hessian_values = - my_hessian_values / self.likelihood.sigma_R ** 2
            
        else:
            grad_fn = jax.grad(self.likelihood.get_R_1_4)
            params = self.likelihood.scaler.transform(self.params)
            
            hessian = jax.hessian(self.likelihood.evaluate)
            hessian_values: dict = hessian(params)
        
            # Extract the Hessian as array
            my_hessian_values = []
            
            for _, row in hessian_values.items():
                for _, value in row.items():
                    my_hessian_values.append(float(value))
                
        names = self.prior.parameter_names
        n_dim = len(names)
        
        # Dump it:
        my_hessian_values = np.reshape(my_hessian_values, (n_dim, n_dim))
        np.savez(self.filename, hessian_values=my_hessian_values, names=names)
        
        return my_hessian_values
        
    
    def read_hessian_values(self,
                            verbose: bool = False,
                            take_log: bool = True,
                            plot_matrix: bool = False):
        
        data = np.load(self.filename, allow_pickle = True)
        names = data["names"]
        n_dim = len(names)
        # hessian_values = np.reshape(data["hessian_values"], (n_dim, n_dim))
        hessian_values = data["hessian_values"]
        
        if verbose:
            print("Hessian values")
            print(hessian_values)
            
        if plot_matrix:
            plot_matrix(hessian_values, take_log = take_log, names = data["names"])
            
        return hessian_values
        
    def check_gaussianity(self, 
                          param_name: str = "E_sym",
                          param_min: float = 28.0,
                          param_max: float = 45.0,
                          N_params: int = 100,
                          plot: bool = False):
        
        param_values = np.linspace(param_min, param_max, N_params)
        log_likelihood_list = []
        
        # Start at true values
        params = copy.deepcopy(self.params)
        
        truth = params[param_name]
        
        for value in param_values:
            params[param_name] = value
            log_likelihood_list.append(self.likelihood.evaluate(params))
            
        if plot:
            plt.plot(param_values, log_likelihood_list)
            plt.axvline(truth, color = "red", label = "True value")
            plt.xlabel(param_name)
            plt.ylabel("Log likelihood")
            plt.savefig(f"./figures/check_gaussianity/{param_name}.png")
            plt.close()
            
        return param_values, log_likelihood_list
    
    def check_gaussianity_2d(self, 
                             param_name_1: str = "E_sym",
                             param_min_1: float = 28.0,
                             param_max_1: float = 45.0,
                             param_name_2: str = "K_sym",
                             param_min_2: float = 28.0,
                             param_max_2: float = 45.0,
                             N_params: int = 100,
                             plot: bool = False):
        
        param_values_1 = np.linspace(param_min_1, param_max_1, N_params)
        param_values_2 = np.linspace(param_min_2, param_max_2, N_params)
        log_likelihood_list = []
        
        # Start at true values
        params = copy.deepcopy(self.params)
        truths = np.array([params[param_name_1], params[param_name_2]])
        
        for value_1 in param_values_1:
            for value_2 in param_values_2:
                params[param_name_1] = value_1
                params[param_name_2] = value_2
                log_likelihood_list.append(self.likelihood.evaluate(params))
            
        if plot:
            plt.plot(param_values, log_likelihood_list)
            plt.axvline(truth, color = "red", label = "True value")
            plt.xlabel(param_name)
            plt.ylabel("Log likelihood")
            plt.savefig(f"./figures/check_gaussianity/{param_name}.png")
            plt.close()
            
        return param_values, log_likelihood_list
    
    def invert_hessian_gwfast(self,
                              hessian: np.array,
                              verbose: bool = True):
        
        # from gwfast.fisherTools import CovMatr
        import mpmath
        
        FisherM = hessian
        FisherMatrixOr = copy.deepcopy(FisherM)
    
        reweighted=False
        CovMatr = np.zeros(FisherM.shape)
        cho_failed = 0
        typeuse='float64'
        # FisherM = FisherMatrix.astype(typeuse)
        
        # Check for NaN values
        ff = mpmath.matrix(FisherM.astype(typeuse))
        try:
            # Conditioning of the original Fisher
            
            # Checks this by computing the eigenvalues of the FIM
            E, _ = mpmath.eigh(ff)
            E = np.array(E.tolist(), dtype=typeuse)
            if np.any(E < 0) and verbose:
                print('Matrix is not positive definite!')
                print("E")
                print(E)

            cond = np.max(np.abs(E))/np.min(np.abs(E))
            if verbose:
                print('Condition of original matrix: %s' %cond)
            
            try:
                # Normalize by the diagonal
                ws =  mpmath.diag([ 1/mpmath.sqrt(ff[i, i]) for i in range(FisherM.shape[-2]) ])
                FisherM_ = ws*ff*ws
                # Conditioning of the new Fisher
                EE, _ = mpmath.eigh(FisherM_)
                E = np.array(EE.tolist(), dtype=typeuse)
                cond = np.max(np.abs(E))/np.min(np.abs(E))
                if verbose:
                    print('Condition of the new matrix: %s' %cond)
                reweighted=True
            except ZeroDivisionError:
                print('The Fisher matrix has a zero element on the diagonal. The normalization procedure will not be applied. Consider using a prior.')
                FisherM_ = ff
            
            # Gi
            invMethodIn='inv'
            alt_method = 'svd'
            invMethod = invMethodIn
            if np.any(E<0):
                if verbose:
                    print('Matrix is not positive definite at position!')
                if invMethodIn=='cho':
                    cho_failed+=1
                    invMethod=alt_method
                    if verbose:
                        print('Cholesky decomposition not usable. Using method %s' %invMethod)
            elif invMethod=='cho':
                try:
                    # In rare cases, the choleski decomposition still fails even if the eigenvalues are positive...
                    # likely for very small eigenvalues
                    c = (mpmath.cholesky(FisherM_))**-1
                    print("c is computed!")
                except Exception as e:
                    print(e)
                    invMethod=alt_method
                    print('Cholesky decomposition not usable. Eigenvalues seem ok but cholesky decomposition failed. Using method %s' %invMethod)
                    #print('Eigenvalues: %s' %str(E))
                    cho_failed+=1

            if invMethod=='inv':
                    cc = FisherM_**-1
            elif invMethod=='cho':
                    #c = cF**-1
                    cc = c.T*c
            elif invMethod=='svd':
                    U, Sm, V = mpmath.svd_r(FisherM_)
                    S = np.array(Sm.tolist(), dtype=typeuse)
                    
                    # TODO: check if needed?
                    # if ((truncate) and (np.abs(cond)>condNumbMax)):
                    #     if verbose:
                    #         print('Truncating singular values below %s' %svals_thresh)
                        
                    #     maxev = np.max(np.abs(S))
                    #     Sinv = mpmath.matrix(np.array([1/s if np.abs(s)/maxev>svals_thresh else 1/(maxev*svals_thresh) for s in S ]).astype(typeuse))
                    #     St = mpmath.matrix(np.array([s if np.abs(s)/maxev>svals_thresh else maxev*svals_thresh for s in S ]).astype(typeuse))
                        
                    #     # Also copute truncated Fisher to quantify inversion error consistently
                    #     truncFisher = U*mpmath.diag([s for s in St])*V
                    #     truncFisher = (truncFisher+truncFisher.T)/2
                    #     FisherMatrixOr[:, :, k] = np.array(truncFisher.tolist(), dtype=typeuse)
                        
                    #     if verbose:
                    #         truncated = np.abs(S)/maxev<svals_thresh #np.array([1 if np.abs(s)/maxev>svals_thresh else 0 for s in S ]
                    #         print('%s singular values truncated' %(truncated.sum()))
                    # else:
                    Sinv = mpmath.matrix(np.array([1/s for s in S ]).astype(typeuse))
                    St = S
                    
                    cc=V.T*mpmath.diag([s for s in Sinv])*U.T
                    
            # elif invMethod=='svd_reg':
                
            #         U, Sm, V = mpmath.svd_r(FisherM_)
                    
            #         S = np.squeeze(np.array(Sm.tolist(), dtype=typeuse))
            #         Um = np.array(U.tolist(), dtype=typeuse)
            #         Vm = np.array(V.tolist(), dtype=typeuse)

            #         kVal = sum(S > svals_thresh)
                                                
            #         Sinv = mpmath.matrix(np.array([1/s  for s in S ]).astype(typeuse))
            #         cc = mpmath.matrix(Um[:, 0:kVal] @ np.diag(1. / S[0:kVal]) @ Vm[0:kVal, :])
                                            
                    
            # elif invMethod=='lu':
            #         P, L, U = mpmath.lu(FisherM_)
            #         ll = P*L
            #         llinv = ll**-1
            #         uinv=U**-1
            #         cc = uinv*llinv
                    
                    
            #### END OF INVERSION METHODS
            
            # Enforce symmetry.
            cc = (cc+cc.T)/2
            
            if reweighted:
                # Undo the reweighting
                CovMatr_ = ws*cc*ws
            else:
                CovMatr_ = cc

            CovMatr =  np.array(CovMatr_.tolist(), dtype=typeuse)
        
        except Exception as e:
            # Eigenvalue decomposition failed
            print(e)
            raise ValueError("Eigenvalue decomposition failed")
    
        return CovMatr
    
def invert_hessian(hessian):
    return np.linalg.inv(hessian)

def plot_matrix(matrix: np.array,
                    names: list[str],
                    take_log: bool = True,
                    save_name: str = "hessian.png"):
        
        n_dim = np.shape(matrix)[0]
        
        plt.figure(figsize = (22, 22))
        if take_log:
            matrix = np.log(abs(matrix))
            cbar_label = 'log10(Absolute value)'
        else:
            cbar_label = 'Matrix'
            
        plt.imshow(matrix, cmap='viridis', interpolation='none')
        cbar = plt.colorbar(shrink = 0.85)
        cbar.set_label(label=cbar_label, size = 24)
        plt.grid(False)
        plt.xticks(range(n_dim), names, rotation = 90, fontsize = 24)
        plt.yticks(range(n_dim), names, fontsize = 24)
        plt.savefig(f"./figures/{save_name}", bbox_inches='tight')
        plt.close()
        
# TODO: remove me
def compare_hessians():
    data = np.load("my_hessian_values.npz", allow_pickle = True)
    hessian_og = data["hessian_values"]
    # N = int(np.sqrt(len(hessian_og)))
    # hessian_og = hessian_og.reshape((N, N))
    
    data = np.load("my_hessian_values_radius.npz", allow_pickle = True)
    hessian_outer = data["hessian_values"]
    
    print("First element check")
    print(hessian_og[0, 0])
    print(hessian_outer[0, 0])
    
    print("ratio")
    print(hessian_og / hessian_outer)
    
def check_all_gaussianity(fisher: Fisher):
    
    for prior in fisher.prior.base_prior:
        param_name = prior.parameter_names[0]
        print(f"Checking {param_name} . . .")
        fisher.check_gaussianity(param_name = param_name,
                                 param_min=prior.xmin,
                                 param_max=prior.xmax,
                                 plot = True)
    
def main():
    
    ### Compute the Fisher
    fisher = Fisher(NB_CSE = 0, filename = "my_hessian_values.npz")
    
    hessian = fisher.compute_hessian_values(use_outer_product = False)
    # hessian = fisher.read_hessian_values()
    
    # compare_hessians() # check if the radius thing is done correctly and gives identical results
    
    print("hessian")
    print(hessian)
    
    ### Inversion of Fisher matrix
    inv = invert_hessian(hessian)
    print(inv)
    
    identity = np.dot(hessian, inv)
    
    print("Identity")
    print(identity)
    
    ### Gaussianity
    # check_all_gaussianity(fisher)
    
    # corrcoef = np.corrcoef(inv)
    # print("corrcoef")
    # print(corrcoef)
    
    plot_matrix(hessian, names = fisher.prior.parameter_names, save_name = "hessian.png", take_log = False)
    plot_matrix(inv, names = fisher.prior.parameter_names, save_name = "inv.png", take_log = False)
    # plot_matrix(hessian[:-2, :-2], names = fisher.prior.parameter_names[:-2], save_name = "hessian.png", take_log = False)
    # plot_matrix(inv[:-2, :-2], names = fisher.prior.parameter_names[:-2], save_name = "inv.png", take_log = False)
    plot_matrix(identity, names = fisher.prior.parameter_names, save_name = "identity.png", take_log = False)
    
    print("DONE")
    
if __name__ == "__main__":
    main()