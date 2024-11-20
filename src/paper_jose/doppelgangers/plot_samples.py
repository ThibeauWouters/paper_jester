"""
Plot the entire collection of random samples for inspection
"""
import numpy as np 
import os
import copy
import tqdm
import matplotlib.pyplot as plt
import joseTOV.utils as jose_utils

params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

RANDOM_SAMPLES_DIR = "./random_samples/"
OUTDIR = "./outdir/"
PLOT_KWARGS = {"color": "blue", 
               "linewidth": 2,
               "alpha": 0.2}

MAX_NB_EOS = 1_000
SCORE_THRESHOLD_LOW = 0.50 # if lower than this, skip
SCORE_THRESHOLD_HIGH = 0.99 # if higher than this, skip

def check_valid_ns(mass, radius, lambdas):
    mask = mass > 1.0
    mass = mass[mask]
    radius = radius[mask]
    lambdas = lambdas[mask]
    
    if any(radius > 14.5):
        return False
    else:
        return True

def plot_macro():
    print("Making the macro plot")
    
    plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
    plot_kwargs = copy.deepcopy(PLOT_KWARGS)

    ### Plot the batch of random samples
    plot_kwargs["alpha"] = 0.1
    all_files = os.listdir(RANDOM_SAMPLES_DIR)
    for i, eos_file in tqdm.tqdm(enumerate(all_files)):
        full_path = os.path.join(RANDOM_SAMPLES_DIR, eos_file)
        data = np.load(full_path)
        
        m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
        if not check_valid_ns(m, r, l):
            print(f"There might be something wrong with {eos_file}")
            continue
        
        plt.subplot(1, 2, 1)
        plt.plot(r, m, **plot_kwargs)
        plt.subplot(1, 2, 2)
        plt.plot(m, l, **plot_kwargs)
        
        if i > MAX_NB_EOS:
            print("Quitting, max number of EOS reached")
            break

    ### Plot the "broken" EOS
    all_files = os.listdir(OUTDIR)
    plot_kwargs["color"] = "red"
    plot_kwargs["alpha"] = 0.75
    
    for i, eos_file in tqdm.tqdm(enumerate(all_files)):
        
        try:
            full_path = os.path.join(OUTDIR, eos_file, "data", "0.npz")
            data = np.load(full_path)
            
            # Check if score is "extreme" enough
            score = data["score"]
            if (score > SCORE_THRESHOLD_HIGH) and (score < SCORE_THRESHOLD_LOW):
                print(f"Skipping {eos_file} due to threshold constraints")
                continue
            
            m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
            if not check_valid_ns(m, r, l):
                print(f"There might be something wrong with {eos_file}")
                continue
            
        except Exception as e:
            print(f"Error: {e}")
        
        plt.subplot(1, 2, 1)
        plt.plot(r, m, **plot_kwargs)
        plt.subplot(1, 2, 2)
        plt.plot(m, l, **plot_kwargs)
        
        if i > MAX_NB_EOS:
            print("Quitting, max number of EOS reached")
            break
        
    # Finalize the plot
    R_MIN, R_MAX = 7, 15
    M_MIN, M_MAX = 0.75, 3
    L_MIN, L_MAX = 1, 1e5
    plt.subplot(1, 2, 1)
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M$ [$M_{\odot}$]")
    plt.xlim(R_MIN, R_MAX)
    plt.ylim(M_MIN, M_MAX)

    plt.subplot(1, 2, 2)
    plt.xlabel(r"$M$ [$M_{\odot}$]")
    plt.ylabel(r"$\Lambda$")
    plt.yscale("log")
    plt.xlim(M_MIN, M_MAX)
    plt.ylim(L_MIN, L_MAX)

    plt.savefig("./figures/MRL_random_samples.png")
    plt.close()
    
def plot_micro():
    print("Making the micro plot")
    
    plt.subplots(figsize = (24, 24), nrows = 2, ncols = 2)
    dirs = [RANDOM_SAMPLES_DIR, OUTDIR]
    colors = ["blue", "red"]
    plot_kwargs = copy.deepcopy(PLOT_KWARGS)
    
    for dir, col in zip(dirs, colors):
        print(f"Plotting for {dir} . . .")
        plot_kwargs["color"] = col
        all_files = os.listdir(dir)
        
        for i, eos_file in tqdm.tqdm(enumerate(all_files)):
            if dir == RANDOM_SAMPLES_DIR:
                try:
                    full_path = os.path.join(RANDOM_SAMPLES_DIR, eos_file)
                    data = np.load(full_path)
                    
                    n, p, e, cs2 = data["n"], data["p"], data["e"], data["cs2"]
                
                    m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
                    if not check_valid_ns(m, r, l):
                        print(f"There might be something wrong with {eos_file}")
                        continue
            
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            else:
                try:
                    full_path = os.path.join(OUTDIR, eos_file, "data", "0.npz")
                    data = np.load(full_path)
                    
                    # Check if score is "extreme" enough
                    score = data["score"]
                    if (score > SCORE_THRESHOLD_HIGH) and (score < SCORE_THRESHOLD_LOW):
                        print(f"Skipping {eos_file} due to threshold constraints")
                        continue
                    
                    n, p, e, cs2 = data["n"], data["p"], data["e"], data["cs2"]
                    
                    m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
                    if not check_valid_ns(m, r, l):
                        print(f"There might be something wrong with {eos_file}")
                        continue
                    
                except Exception as e:
                    print(f"Error: {e}")
                    continue

            if i > MAX_NB_EOS:
                print("Quitting, max number of EOS reached")
                break
        
            # label = f"id = {key}"
            # params = {k: doppelgangers_dict[key][k] for k in param_names}
            
            n = n / jose_utils.fm_inv3_to_geometric / 0.16
            e = e / jose_utils.MeV_fm_inv3_to_geometric
            p = p / jose_utils.MeV_fm_inv3_to_geometric
            cs2 = cs2
            
            plt.subplot(221)
            plt.plot(n, e, color = col)
            plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
            plt.ylabel(r"$e$ [MeV fm$^{-3}$]")
            # plt.xlim(nmin, nmax)
            
            plt.subplot(222)
            plt.plot(n, p, color = col)
            plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
            plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
            # plt.xlim(nmin, nmax)
            
            plt.subplot(223)
            plt.plot(n, cs2, color = col)
            plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
            plt.ylabel(r"$c_s^2$")
            # plt.xlim(nmin, nmax)
            plt.ylim(0, 1)
            
            plt.subplot(224)
            # TODO: if we want it
            # e_min = 200
            # e_max = 1500
            # mask = (e_min < e) * (e < e_max)
            # mask_target = (e_min < self.e_target) * (self.e_target < e_max)
            mask = [True for _ in e]
            plt.plot(e[mask], p[mask], color = col)
            
            plt.xlabel(r"$e$ [MeV fm$^{-3}$]")
            plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
            # plt.xlim(e_min, e_max)
            
    plt.savefig(f"./figures/breaking_EOS.png", bbox_inches = "tight")
    plt.savefig(f"./figures/breaking_EOS.pdf", bbox_inches = "tight")
    plt.close()
        
def main():
    plot_macro()
    plot_micro()
    
if __name__ == "__main__":
    main()