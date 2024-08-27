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

eos_path_params = "../eos/samples/params"
data_posterior = stats.gaussian_kde(np.loadtxt("./data/PREX_samples.txt", skiprows = 1).T)


#initialize likelihood array for each of the EOS and sample lists for R1.4
files = next(os.walk(eos_path_params))[2]
NEOS=len(files)


#split the EOS workload differently
splits = np.split(np.arange(0, NEOS), size)
work = splits[rank].copy()
save = []

print("This is processor ", rank, "and I am doing EOS from", work[0]+1, " to", work[-1]+1, ".")

#loop over the EOS and calculate the likelihood for each
comm.Barrier()
for j in work:
    params = np.loadtxt(eos_path_params+'/{}.dat'.format(j+1))
    EFT_breakdown_density, K_sat, Q_sat, Z_sat, E_sym, L_sym, K_sym, Q_sym, Z_sym = params

    logl = data_posterior.logpdf((E_sym, L_sym))

    save.append(logl)

comm.Barrier()

save = comm.gather(save,root=0)

if rank==0:
   logL = np.array([l for sublist in save for l in sublist]) #flatten the save list
   logL -= scipy.special.logsumexp(logL)

   L = np.exp(logL)
   np.savetxt("eos_likelihood_PREX", L)

else:
	None
