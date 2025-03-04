## data

Contains postprocessed data: for a particular label, the runs were done with that setup (e.g. `h100` refers to the runs done with a single NVIDIA H100 GPU et cetera). Here, we have a dictionary saved as JSON with the keys being the number of CSE grid points used for the EOS, and the values being themselves dictionaries with for each sampled parameter the corresponding ESS. An additional key `runtime` stores the runtime as given in the log files. See `scaling_plot.py` for further information regarding how this is done.

- H100 scaling results/runs: [permalink](https://github.com/ThibeauWouters/paper_jose/tree/c653fc0f029b88acc0c3d64b8e9e4bdf2209ac4e/src/paper_jose/inference/CSE_systematics). 