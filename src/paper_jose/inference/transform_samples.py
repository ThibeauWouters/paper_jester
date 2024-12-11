import numpy as np
import argparse
import jax
import jax.numpy as jnp
import os
import time
from paper_jose.inference.inference import my_transform_eos, eos_param_names
import paper_jose.inference.utils_plotting

def parse_arguments():
    parser = argparse.ArgumentParser(description="Full-scale inference script with customizable options.")
    parser.add_argument("--outdir", 
                        type=str, 
                        default="./outdir/", 
                        help="Directory where the samples are stored (default: './outdir/')")
    parser.add_argument("--N-samples", 
                        type=int, 
                        default=1_000, 
                        help="Number of samples for which to solve the TOV equations")
    parser.add_argument("--use-batching", 
                        type=bool, 
                        default=False,
                        help="Whether to do some batching for the TOV solver to make it more memory efficient")
    return parser.parse_args()

def main(args):
    
    # Load samples
    filename = os.path.join(args.outdir, "results_production.npz")
    data = np.load(filename)
    log_prob = data["log_prob"]
    
    max_N_samples = len(log_prob)
    if args.N_samples > max_N_samples:
        print(f"Requested {args.N_samples} samples, but only {max_N_samples} are available. Adjusting that.")
        args.N_samples = max_N_samples
    
    samples_named = {}
    for param in eos_param_names:
        samples_named[param] = data[param]
    
    print(f"Transforming the samples")
    idx = np.random.choice(np.arange(len(log_prob)), size=args.N_samples, replace=False)
    TOV_start = time.time()
    chosen_samples = {k: jnp.array(v[idx]) for k, v in samples_named.items()}
    # transformed_samples = jax.vmap(my_transform_eos.forward)(chosen_samples)
    # NOTE: jax lax map helps us deal with batching, but a batch size multiple of 10 gives errors, therefore this weird number
    transformed_samples = jax.lax.map(my_transform_eos.forward, chosen_samples, batch_size = 4_999)
    TOV_end = time.time()
    print(f"Time taken for TOV map: {TOV_end - TOV_start} s")
    chosen_samples.update(transformed_samples)

    log_prob = log_prob[idx]
    np.savez(os.path.join(args.outdir, "eos_samples.npz"), log_prob=log_prob, **chosen_samples)
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)