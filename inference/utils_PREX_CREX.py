# TODO: merge!

# import psutil
# p = psutil.Process()
# p.cpu_affinity([0])
# import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"

# import numpy as np
# import jax
# jax.config.update('jax_platform_name', 'cpu')
# import jax.numpy as jnp
# from jax.scipy.special import logsumexp
# from jaxtyping import Array, Float
# from jax.scipy.stats import gaussian_kde
# import pandas as pd
# import copy
# from functools import partial

# from joseTOV.eos import MetaModel_with_CSE_EOS_model, construct_family
# from joseTOV import utils
# from jimgw.base import LikelihoodBase
# from jimgw.transforms import NtoMTransform
# from jax.scipy.stats import gaussian_kde

# import matplotlib.pyplot as plt
# import corner

# params = {"axes.grid": True,
#         "text.usetex" : True,
#         "font.family" : "serif",
#         "ytick.color" : "black",
#         "xtick.color" : "black",
#         "axes.labelcolor" : "black",
#         "axes.edgecolor" : "black",
#         "font.serif" : ["Computer Modern Serif"],
#         "xtick.labelsize": 16,
#         "ytick.labelsize": 16,
#         "axes.labelsize": 16,
#         "legend.fontsize": 16,
#         "legend.title_fontsize": 16,
#         "figure.titlesize": 16}

# plt.rcParams.update(params)

# # Improved corner kwargs
# default_corner_kwargs = dict(bins=40, 
#                         smooth=1., 
#                         show_titles=False,
#                         label_kwargs=dict(fontsize=16),
#                         title_kwargs=dict(fontsize=16), 
#                         color="blue",
#                         # quantiles=[],
#                         # levels=[0.9],
#                         plot_density=True, 
#                         plot_datapoints=False, 
#                         fill_contours=True,
#                         max_n_ticks=4, 
#                         min_n_ticks=3,
#                         truth_color = "red",
#                         save=False)

# ############
# ### DATA ###
# ############

# # # Create samples for both of them:

# # N_samples = 1_000
# # key = jax.random.PRNGKey(0)
# # key, subkey = jax.random.split(key)
# # prex_samples = prex_posterior.resample(subkey, (N_samples,))
# # key, subkey = jax.random.split(key)
# # crex_samples = crex_posterior.resample(subkey, (N_samples,))

# # # Create a corner plot

# # fig = corner.corner(np.array(prex_samples.T), labels=[r"$E_{\rm sym}$", r"$L_{\rm sym}$"], **default_corner_kwargs)
# # default_corner_kwargs["color"] = "red"
# # corner.corner(np.array(crex_samples.T), fig=fig, **default_corner_kwargs)
# # fs = 24
# # plt.text(0.75, 0.85, "PREX", color="blue", fontsize=fs, transform=plt.gcf().transFigure)
# # plt.text(0.75, 0.75, "CREX", color="red", fontsize=fs, transform=plt.gcf().transFigure)
# # plt.savefig("./figures/data_corner_plot.png")
# # plt.close()

# # print("DONE")