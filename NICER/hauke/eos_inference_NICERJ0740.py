#!/home/koehn/anaconda3/envs/nmma/bin/python3
import sys
import os, os.path
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import pandas as pd
import scipy.integrate as integrate
import corner
import tqdm

eos_path_macro = "../eos/samples/MRL"

#oldpath: maryland_path = "./data/J0740/J0740_RM_maryland.txt"
maryland_path = "./data/J0740/J0740_NICERXMM_full_mr.txt"
#oldpath: amsterdam_path = "./data/J0740/A_NICER_VIEW_OF_PSR_J0740+6620/STU/NICERxXMM/FI_H/run10/NICER_x_XMM_J0740__XPSI_STU_NSX_FIH__weight_mass_radius.txt"
amsterdam_path = "./data/J0740/new_mr_samples_amsterdam/J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat"

#load the radius-mass posterior samples from the data
maryland_samples = pd.read_csv(maryland_path, sep=" ", names=["R","M", "weight"] , skiprows = 6)
#if pd.isna(maryland_samples["weight"]).any()#
#	print("Warning: weights not properly specified, assuming constant weights instead.")
#	maryland_samples["weight"] = np.ones(len(maryland_samples["weight"]))
amsterdam_samples = pd.read_csv(amsterdam_path, sep=" ", names=["M", "R"])

#get a smooth interpolator for the posterior
#maryland_chosen_samples = np.random.choice(np.arange(0, len(maryland_samples["R"]), 1), size = neff, p = maryland_samples['weight']/np.sum(maryland_samples['weight']), replace = False)
maryland_posterior= stats.gaussian_kde([maryland_samples["M"], maryland_samples["R"]], weights = maryland_samples["weight"])

#np.random.seed(42)
#neff = min(eff_post_sample_size,len(amsterdam_samples["R"]))
#amsterdam_chosen_samples = np.random.choice(np.arange(0, len(amsterdam_samples["R"]), 1), size = neff, p = amsterdam_samples['weight']/np.sum(amsterdam_samples['weight']), replace = False)
amsterdam_posterior= stats.gaussian_kde([amsterdam_samples["M"], amsterdam_samples["R"]])



#initialize likelihood array for each of the EOS and sample lists for R1.4
files = next(os.walk(eos_path_macro))[2]
NEOS=len(files)



#split the EOS workload differently
splits = np.split(np.arange(0, NEOS), size)
work = splits[rank].copy()
save_maryland=[]
save_amsterdam=[]

print("This is processor ", rank, "and I am doing EOS from ", work[0]+1, " to ", work[-1]+1, ".")
#loop over the EOS and calculate the likelihood for each
comm.Barrier()
for j in work:
    R, M = np.loadtxt(eos_path_macro+'/{}.dat'.format(j+1), usecols = [0,1]).T #loads the R-M curve
    R = np.array([R]).flatten(); M=np.array([M]).flatten()
    masses = np.arange(0, M.max(),0.02)
    radius_masses = np.interp(masses, M, R)
    logy_maryland = maryland_posterior.logpdf(np.vstack([masses, radius_masses]))
    save_maryland.append(scipy.special.logsumexp(logy_maryland)-np.log(len(logy_maryland)))
    #logy_amsterdam = amsterdam_posterior.logpdf(np.vstack([masses, radius_masses]))
    #save_amsterdam.append(scipy.special.logsumexp(logy_amsterdam)-np.log(len(logy_amsterdam)))
comm.Barrier()

save_maryland = comm.gather(save_maryland,root=0)
save_amsterdam = comm.gather(save_amsterdam,root=0)

if rank==0:
   logL_maryland = np.array([l for sublist in save_maryland for l in sublist]) #flatten the save list
   logL_maryland -= scipy.special.logsumexp(logL_maryland)
   #logL_amsterdam = np.array([l for sublist in save_amsterdam for l in sublist])
   #logL_amsterdam -= scipy.special.logsumexp(logL_amsterdam)

   L_maryland = np.exp(logL_maryland)
   #L_amsterdam = np.exp(logL_amsterdam)
   #L = 1/2*(L_maryland+L_amsterdam)
   np.savetxt("eos_likelihood_J0740_maryland", L_maryland)#np.array([L, L_maryland, L_amsterdam]).T)

else:
	None
