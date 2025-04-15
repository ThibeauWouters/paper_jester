# paper_jester
Code for the paper introducing [Jester](https://github.com/nuclear-multimessenger-astronomy/jester): Jax-based EoS and Tov solvER. 

The repository is split into two main, active branches:
- `new_main`: Code and results for the gradient-based optimization scheme part of the paper. The master function used to generate the results can be found at [this line](https://github.com/ThibeauWouters/paper_jose/blob/5516f5bc4947ffadce8b793894d157a32b443a02/src/paper_jose/doppelgangers/doppelgangers.py#L1613)
- `inference`: Code and results for the Bayesian inference runs performed in the paper. The master function used to generate the results can be found at [this link](https://github.com/ThibeauWouters/paper_jose/blob/389c3d43ecaa4385fb312145508c9f7226b1ebd6/src/paper_jose/inference/inference.py#L166). 
