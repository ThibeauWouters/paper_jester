"""
Benchmarking the jose solver

TODO: implement it
"""

################
### PREAMBLE ###
################
import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
import json

import time
import numpy as np
np.random.seed(43) # for reproducibility
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import corner

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")

from jimgw.prior import UniformPrior, CombinePrior
from jimgw.jim import Jim

import paper_jose.utils as utils
import utils_plotting

start = time.time()

my_transform = utils.MicroToMacroTransform(utils.name_mapping,
                                           keep_names = ["E_sym", "L_sym"],
                                           nmax_nsat = utils.NMAX_NSAT,
                                           nb_CSE = utils.NB_CSE
                                           )