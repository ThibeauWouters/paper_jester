"""
Extra script to analyze some results of inference runs etc.
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

import os
import tqdm
import time
import numpy as np
np.random.seed(42) # for reproducibility
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jimgw.prior import UniformPrior, CombinePrior

import utils as NICER_utils
plt.rcParams.update(NICER_utils.mpl_params)

### Load and plot some data

# computed_data = "./computed_data/"
# all_L_sym = []
# all_L = []

# for file in os.listdir(computed_data):
#     full_path = os.path.join(computed_data, file)
#     data = np.load(full_path)
    
#     all_L_sym.append(data["L_sym"])
#     all_L.append(data["L"])

# plt.plot(all_L_sym, all_L, "o")
# plt.xlabel(r"$L_{\rm{sym}}$")
# plt.ylabel(r"$\log \mathcal{L}_{\rm{NICER}}$")
# plt.savefig("./figures/L_sym_vs_L.png", bbox_inches = 'tight')
# plt.close()

outdir = "./outdir_J0030/"

