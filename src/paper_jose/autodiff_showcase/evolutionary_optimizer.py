
from jimgw.prior import CombinePrior
from evosax import CMA_ES
from typing import Callable

import time
import tqdm
import jax
import jax.numpy as jnp
import numpy as np

class EvolutionaryOptimizer:
    
            # def y(x):
            # return -self.evaluate_original(prior.transform(prior.add_name(x)), {})

    def __init__(self,
                 loss_func: Callable,
                 prior: CombinePrior, 
                 n_dims: int,
                 bound, # TODO: which type?
                 popsize: int = 10,
                 n_loops: int = 100,
                 seed: int = 9527,
                 verbose: bool = True,
                 **kwargs
                 ):
        
        print("np.shape(bound)")
        print(np.shape(bound))
        
        print("n_dims")
        print(n_dims)
        
        self.loss_func = jax.jit(jax.vmap(loss_func))
        self.prior = prior
        self.n_dims = n_dims 
        self.bound = bound
        self.n_loops = n_loops
        self.seed = seed
        self.verbose = verbose
        
        elite_ratio = kwargs.get("elite_ratio", 0.5)
        self.strategy = CMA_ES(num_dims=n_dims, popsize=popsize, elite_ratio=elite_ratio)
        # TODO: es params is broken?
        self.es_params = self.strategy.default_params.replace(clip_min=0, clip_max=1)
        self.history = []
        self.state = None

    def optimize(self, keep_history_step = 0):
        """
        Optimize the objective function.
        """
        
        rng = jax.random.PRNGKey(self.seed)
        key, subkey = jax.random.split(rng)
        progress_bar = tqdm.tqdm(range(self.n_loops), "Generation: ") if self.verbose else range(self.n_loops)
        self.state = self.strategy.initialize(key, self.es_params)
        
        if keep_history_step > 0:
            self.history = []
            for i in progress_bar:
                key, self.state, theta = self.optimize_step(subkey, self.state, self.loss_func, self.bound)
                if i%keep_history_step == 0: self.history.append(theta)
                if self.verbose: progress_bar.set_description(f"Generation: {i}, Fitness: {self.state.best_fitness:.4f}")
            self.history = jnp.array(self.history)
        else:
            for i in progress_bar:
                key, self.state, _ = self.optimize_step(subkey, self.state, self.loss_func, self.bound)
                if self.verbose: progress_bar.set_description(f"Generation: {i}, Fitness: {self.state.best_fitness:.4f}")
                
    def optimize_step(self, key: jax.random.PRNGKey, state, objective: callable, bound):
        key, subkey = jax.random.split(key)
        x, state = self.strategy.ask(subkey, state, self.es_params)
        theta = x * (bound[:, 1] - bound[:, 0]) + bound[:, 0]
        print("theta")
        print(np.shape(theta))
        
        theta_named = self.prior.add_name(theta.T)
        
        print("theta_named")
        print(theta_named)
        
        fitness = objective(theta_named)
        state = self.strategy.tell(x, fitness.astype(jnp.float32), state, self.es_params)
        return key, state, theta

    def get_result(self):
        """
        Get the best member and the best fitness.

        Returns
        -------
        best_member : (ndims,) ndarray
            The best member.
        best_fitness : float
            The best fitness.
        """

        best_member = self.state.best_member* (self.bound[:, 1] - self.bound[:, 0]) + self.bound[:, 0]
        best_fitness = self.state.best_fitness
        return best_member, best_fitness