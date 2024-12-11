"""Quickly doing some postprocessing on the results of the inference."""

import numpy as np
import matplotlib.pyplot as plt
import os
import corner
import tqdm
import argparse

import joseTOV.utils as jose_utils

mpl_params = {"axes.grid": True,
        "text.usetex" : False,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        # "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}
plt.rcParams.update(mpl_params)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Full-scale inference script with customizable options.")
    parser.add_argument("--outdir", 
                        type=str, 
                        default="./outdir/", 
                        help="Directory to save output files (default: './outdir/')")
    parser.add_argument("--max-samples", 
                        type=int,
                        default=2_000, 
                        help="Max number of samples to plot")
    # TODO: check if this is ok
    # parser.add_argument("--plot-prior", 
    #                     type=bool, 
    #                     default=True, 
    #                     help="Whether to also plot the samples from the prior, assuming they are located in ./outdir_prior/ (default: True)")
    return parser.parse_args()


def main(args):
    
    # Load the EOS/TOV data samples
    filename = os.path.join(args.outdir, "eos_samples.npz")
    data = np.load(filename)
    m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
    n, p, e, cs2 = data["n"], data["p"], data["e"], data["cs2"]
    
    n = n / jose_utils.fm_inv3_to_geometric / 0.16
    p = p / jose_utils.MeV_fm_inv3_to_geometric
    e = e / jose_utils.MeV_fm_inv3_to_geometric

    nb_samples = np.shape(m)[0]
    print(f"Number of samples: {nb_samples}")

    # Plotting
    samples_kwargs = {"color": "black",
                      "alpha": 1.0,
                      "rasterized": True}

    plt.subplots(1, 2, figsize=(12, 6))

    m_min, m_max = 1.0, 3.75
    r_min, r_max = 5.5, 18.0
    l_min, l_max = 1.0, 50_000.0

    log_prob = data["log_prob"]
    print("np.shape(log_prob)")
    print(np.shape(log_prob))

    # Get a colorbar for log prob, but normalized
    norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
    cmap = plt.get_cmap("viridis")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    print("Creating EOS plot . . .")
    mtov_list = []
    r14_list = []
    for i in tqdm.tqdm(range(nb_samples)):

        # Get color
        color = cmap(norm(log_prob[i]))
        samples_kwargs["color"] = color
        
        # Mass-radius plot
        plt.subplot(121)
        mask = (r[i] > 0.75 * r_min) * (r[i] < 1.25 * r_max) * (m[i] > 0.75 * m_min) * (m[i] < 1.25 * m_max) * (l[i] > 0.75 * l_min) * (l[i] < 1.25 * l_max)
        # mask = [True for _ in range(len(r[i]))]
        plt.plot(r[i][mask], m[i][mask], **samples_kwargs)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_{\odot}$]")
        plt.xlim(r_min, r_max)
        plt.ylim(m_min, m_max)
        
        # Mass-Lambda plot
        plt.subplot(122)
        plt.plot(m[i][mask], l[i][mask], **samples_kwargs)
        plt.yscale("log")
        plt.xlim(m_min, m_max)
        plt.ylim(l_min, l_max)
        plt.xlabel(r"$M$ [$M_{\odot}$]")
        plt.ylabel(r"$\Lambda$")
        
        if i > args.max_samples:
            break
        
        # Add MTOV and R1.4
        mtov = np.max(m[i])
        mtov_list.append(mtov)
        if mtov < 1.4:
            continue
        r14 = np.interp(1.4, m[i], r[i])
        if r14 > 20.0:
            continue
        r14_list.append(r14)
        
    # Save
    sm.set_array([])
    # plt.colorbar(sm, cax = cax, label=r"$\log P$")
    plt.savefig(os.path.join(args.outdir, "postprocessing.png"), bbox_inches = "tight")
    plt.savefig(os.path.join(args.outdir, "postprocessing.pdf"), bbox_inches = "tight")
    plt.close()
    print("Creating EOS plot . . . DONE")

    ### Build a histogram of the TOV masses and R1.4 and Lambda1.4 values
    print("Creating histograms . . .")
    bins = 25

    plt.subplots(1, 2, figsize=(18, 6))
    plt.subplot(121)
    plt.hist(mtov_list, bins=bins, color="blue", histtype="step", lw=2, density = True)
    plt.xlabel(r"$M_{\rm TOV}$ [$M_{\odot}$]")
    plt.ylabel("Density")

    plt.subplot(122)
    plt.hist(r14_list, bins=bins, color="blue", histtype="step", lw=2, density = True)
    plt.xlabel(r"$R_{1.4}$ [km]")
    plt.ylabel("Density")

    plt.savefig(os.path.join(args.outdir, "postprocessing_histograms.png"), bbox_inches = "tight")
    plt.savefig(os.path.join(args.outdir, "postprocessing_histograms.pdf"), bbox_inches = "tight")
    plt.close()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)