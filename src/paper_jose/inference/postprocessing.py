"""Quickly doing some postprocessing on the results of the inference."""

import numpy as np
import matplotlib.pyplot as plt
import os
import corner
import tqdm
import argparse

import joseTOV.utils as jose_utils

mpl_params = {"axes.grid": True,
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
plt.rcParams.update(mpl_params)

HAUKE_MICRO_EOS = "/home/twouters2/hauke_eos/micro/"
HAUKE_MACRO_EOS = "/home/twouters2/hauke_eos/macro/"
HAUKE_WEIGHTS = "/home/twouters2/hauke_eos/weights/"
HAUKE_RESULTS_FILE = "/home/twouters2/hauke_eos/histogram_data.npz"

# These indices correspond to the bad samples, i.e. very very soft (broken) EOS 
BAD_INDICES_HAUKE = np.array([
    697, 2768, 6748, 7013, 7818, 10463, 16304, 16721, 18136, 19907, 24908, 
    24944, 25641, 25662, 31398, 33688, 33760, 34836, 40810, 41568, 42493, 
    45180, 49071, 49847, 50850, 50895, 56866, 57065, 57216, 58638, 59395, 
    59701, 63154, 64116, 65709, 66545, 68317, 68778, 70636, 71686, 73827, 
    74637, 75090, 75459, 76844, 77111, 79144, 80969, 81691, 83593, 89061, 
    90706, 91702, 94944, 97836, 98124, 98652, 99150
])
BAD_INDICES_HAUKE -= 1
# TODO: might accidentally have missed the final one? Not sure, but shapes do not match...
BAD_INDICES_HAUKE = np.append(BAD_INDICES_HAUKE, [99_999])

def gather_hauke_results():
    """We gather the histograms for the most important quantities taken from https://multi-messenger.physik.uni-potsdam.de/eos_constraints/ and downloaded locally to the cluster"""
    
    mtov_list = []
    r14_list = []
    ntov_list = []
    p3nsat_list = []
    
    for i in tqdm.tqdm(range(1, 100_000)):
        # Load micro
        try: 
            data = np.loadtxt(os.path.join(HAUKE_MICRO_EOS, f"{i}.dat"))
            n, e, p, cs2 = data[:, 0] / 0.16, data[:, 1], data[:, 2], data[:, 3]
            
            # Load macro
            data = np.genfromtxt(os.path.join(HAUKE_MACRO_EOS, f"{i}.dat"), skip_header=1, delimiter=" ").T
            r, m, l = data[0], data[1], data[2]
            
        except Exception as e:
            print(f"Skipping {i} since something went wrong. Here is the error message")
            print(e)
            continue
        
        if isinstance(m, np.float64):
            print(f"Skipping {i} since something is off")
            print(r)
            print(m)
            continue
        
        # Get useful quantities
        mtov = np.max(m)
        mtov_list.append(mtov)
        
        r14 = np.interp(1.4, m, r)
        r14_list.append(r14)
        p3nsat = np.interp(3, n, p)
        p3nsat_list.append(p3nsat)
        
        # FIXME: n_TOV is not possible with the current information?
        
    # Save the result
    np.savez(HAUKE_RESULTS_FILE, mtov_list=mtov_list, r14_list=r14_list, ntov_list=ntov_list, p3nsat_list=p3nsat_list)
    print("Results saved to", HAUKE_RESULTS_FILE)
        
def fetch_hauke_weights(weights_name: str):
    allowed = ["ALL", "GW170817", "J0740", "PREX", "CREX", "J0030", "PREX_CREX_NICERS", "prior"]
    if weights_name not in allowed:
        raise ValueError(f"weights_name must be one of {allowed}")
    filename = os.path.join(HAUKE_WEIGHTS, f"{weights_name}.txt")
    print(f"Loading weights from {filename}")
    
    weights = np.genfromtxt(filename)
    
    # These were the file numbers which started from 1 so the indices are off by one
    print("Weights at these bad indices")
    print(weights[BAD_INDICES_HAUKE])
    weights = np.delete(weights, BAD_INDICES_HAUKE)
    
    return weights

def make_plots(outdir: str,
               plot_NS: bool = True,
               plot_EOS: bool = True,
               plot_histograms: bool = True,
               max_samples: int = 2_000,
               hauke_string: str = ""):
    
    filename = os.path.join(outdir, "eos_samples.npz")
    
    data = np.load(filename)
    m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
    logpc_EOS = data["logpc_EOS"]
    n, p, e, cs2 = data["n"], data["p"], data["e"], data["cs2"]
    
    n = n / jose_utils.fm_inv3_to_geometric / 0.16
    p = p / jose_utils.MeV_fm_inv3_to_geometric
    e = e / jose_utils.MeV_fm_inv3_to_geometric
    
    pc_EOS = np.exp(logpc_EOS) / jose_utils.MeV_fm_inv3_to_geometric

    nb_samples = np.shape(m)[0]
    print(f"Number of samples: {nb_samples}")

    # Plotting
    samples_kwargs = {"color": "black",
                      "alpha": 1.0,
                      "rasterized": True}

    plt.subplots(1, 2, figsize=(12, 6))

    m_min, m_max = 0.3, 3.75
    r_min, r_max = 5.5, 18.0
    l_min, l_max = 1.0, 50_000.0

    log_prob = data["log_prob"]
    log_prob = log_prob[:max_samples + 1]
    log_prob = np.exp(log_prob) # so actually no longer log prob but prob... whatever
    print("np.shape(log_prob)")
    print(np.shape(log_prob))

    # Get a colorbar for log prob, but normalized
    norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
    cmap = plt.get_cmap("YlGn")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    if plot_NS:
        print("Creating NS plot . . .")
        bad_counter = 0
        for i in tqdm.tqdm(range(nb_samples)):

            if i >= max_samples:
                break

            # Get color
            normalized_value = norm(log_prob[i])
            color = cmap(normalized_value)
            if "prior" in outdir:
                samples_kwargs["color"] = "gray"
                samples_kwargs["zorder"] = 1e10
                samples_kwargs["alpha"] = 0.1
            else:
                samples_kwargs["color"] = color
                samples_kwargs["zorder"] = 1e10 + normalized_value
            
            if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
                bad_counter += 1
                continue
        
            if any(l[i] < 0):
                bad_counter += 1
                continue
            
            # Mass-radius plot
            plt.subplot(121)
            plt.plot(r[i], m[i], **samples_kwargs)
            plt.xlabel(r"$R$ [km]")
            plt.ylabel(r"$M$ [$M_{\odot}$]")
            plt.xlim(r_min, r_max)
            plt.ylim(m_min, m_max)
            
            # Mass-Lambda plot
            plt.subplot(122)
            plt.plot(m[i], l[i], **samples_kwargs)
            plt.yscale("log")
            plt.xlim(m_min, m_max)
            plt.ylim(l_min, l_max)
            plt.xlabel(r"$M$ [$M_{\odot}$]")
            plt.ylabel(r"$\Lambda$")
            
        print(f"Bad counter: {bad_counter}")
        # Beautify the plots a bit
        plt.subplot(121)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_{\odot}$]")
        plt.xlim(r_min, r_max)
        plt.ylim(m_min, m_max)
        
        # Mass-Lambda plot
        plt.subplot(122)
        plt.yscale("log")
        plt.xlim(m_min, m_max)
        plt.ylim(l_min, l_max)
        plt.xlabel(r"$M$ [$M_{\odot}$]")
        plt.ylabel(r"$\Lambda$")
            
        # Save
        sm.set_array([])
        # plt.colorbar(sm, cax = cax, label=r"$\log P$")
        plt.savefig(os.path.join(outdir, "postprocessing_NS.png"), bbox_inches = "tight")
        plt.savefig(os.path.join(outdir, "postprocessing_NS.pdf"), bbox_inches = "tight")
        plt.close()
        print("Creating NS plot . . . DONE")
    
    if plot_EOS:
        raise NotImplementedError("Not implemented yet.")
    
    if plot_histograms:
        # Get the Hauke results if desired
        if len(hauke_string) > 0:
            print(f"Reading Hauke data")
            hauke_histogram_data = np.load(HAUKE_RESULTS_FILE)
            weights = fetch_hauke_weights(hauke_string)
            
            for key in hauke_histogram_data.keys():
                print(f"Shape of {key}: {np.shape(hauke_histogram_data[key])}")
            
            print("np.shape(weights)")
            print(np.shape(weights))
            
            print(f"Reading Hauke data DONE")
    
        ### Build a histogram of the TOV masses and R1.4 and Lambda1.4 values
        print("Creating histograms . . .")
        
        mtov_list = []
        r14_list = []
        ntov_list = []
        p3nsat_list = []
        
        negative_counter = 0
        for i in tqdm.tqdm(range(nb_samples)):
            _m, _r, _l = m[i], r[i], l[i]
            _pc = pc_EOS[i]
            _n, _p, _e, _cs2 = n[i], p[i], e[i], cs2[i]
            
            if any(_l < 0):
                negative_counter += 1
                continue
            
            mtov = np.max(_m)
            r14 = np.interp(1.4, _m, _r)
            
            p3nsat = np.interp(3, _n, _p)
            
            pc_TOV = np.interp(mtov, _m, _pc)
            n_TOV = np.interp(pc_TOV, _p, _n)
            
            # Append all
            mtov_list.append(mtov)
            if mtov > 1.4:
                r14_list.append(r14)
            ntov_list.append(n_TOV)
            p3nsat_list.append(p3nsat)
            
        print(f"Negative counter: {negative_counter}")
        
        bins = 50
        hist_kwargs = dict(histtype="step", lw=2, density = True, bins=bins)
        plt.subplots(2, 2, figsize=(18, 12))
        plt.subplot(221)
        plt.hist(mtov_list, color="blue", label = "Jester", **hist_kwargs)
        if len(hauke_string) > 0:
            plt.hist(hauke_histogram_data["mtov_list"], color="red", weights=weights, label = "Hauke", **hist_kwargs)
        plt.xlabel(r"$M_{\rm TOV}$ [$M_{\odot}$]")
        plt.ylabel("Density")

        plt.subplot(222)
        plt.hist(r14_list, color="blue", label = "Jester", **hist_kwargs)
        if len(hauke_string) > 0:
            has_r14 = np.where(hauke_histogram_data["mtov_list"] > 1.4, True, False)
            r14_weights = weights[has_r14]
            r14_list = hauke_histogram_data["r14_list"][has_r14]
            plt.hist(r14_list, color="red", weights=r14_weights, label = "Jester", **hist_kwargs)
        plt.xlabel(r"$R_{1.4}$ [km]")
        plt.ylabel("Density")
        
        plt.subplot(223)
        plt.hist(ntov_list, color="blue", label = "Jester", **hist_kwargs)
        # FIXME: cannot comput n_TOV with the current information from Hauke?
        # if len(hauke_string) > 0:
        #     plt.hist(hauke_histogram_data["r14_list"], bins=bins, color="red", histtype="step", lw=2, density = True, weights=weights)
        plt.xlabel(r"$n_{\rm{TOV}}$ [$n_{\rm{sat}}$]")
        plt.ylabel("Density")
        
        plt.subplot(224)
        plt.hist(p3nsat_list, color="blue", label = r"\texttt{Jester}", **hist_kwargs)
        if len(hauke_string) > 0:
            plt.hist(hauke_histogram_data["p3nsat_list"], color="red", weights=weights, label = "Koehn+", **hist_kwargs)
        plt.xlabel(r"$p_{3n_{\rm{sat}}}$ [MeV fm$^{-3}$]")
        plt.ylabel("Density")
        plt.legend(fontsize = 24)

        plt.savefig(os.path.join(outdir, "postprocessing_histograms.png"), bbox_inches = "tight")
        plt.savefig(os.path.join(outdir, "postprocessing_histograms.pdf"), bbox_inches = "tight")
        plt.close()

def main():
    
    print(f"Running main")
    # ### Gather Hauke results -- get the raw data for the histograms
    # gather_hauke_results()
    
    ### Single postprocessing
    suffix_list = ["prior"] # "J0740", "GW170817", "J0030",
    for suffix in suffix_list:
        outdir = f"./outdir_{suffix}/"
        print(f"Making plots for {outdir}")
        make_plots(outdir,
                   plot_NS=True,
                   plot_EOS=False,
                   plot_histograms=True,
                   hauke_string = suffix)

if __name__ == "__main__":
    main()
    
    