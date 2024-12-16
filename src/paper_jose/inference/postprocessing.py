"""Quickly doing some postprocessing on the results of the inference."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import corner
import tqdm
import argparse
import arviz

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

def load_gw170817_injection_eos():
    """Returns MRL of the target"""
    filename = "/home/twouters2/psds/hauke_eos.npz"
    data = np.load(filename)
    r_target, m_target, Lambdas_target = data["radii_EOS"], data["masses_EOS"], data["Lambdas_EOS"]
    print("np.max(m_target)")   
    print(np.max(m_target))
    return m_target, r_target, Lambdas_target

def gather_hauke_results():
    """We gather the histograms for the most important quantities taken from https://multi-messenger.physik.uni-potsdam.de/eos_constraints/ and downloaded locally to the cluster"""
    
    mtov_list = []
    r14_list = []
    l14_list = []
    ntov_list = []
    p3nsat_list = []
    
    for i in tqdm.tqdm(range(1, 100_000)):
        try: 
            # Load micro
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
        r14 = np.interp(1.4, m, r)
        l14 = np.interp(1.4, m, l)
        
        # Append them
        mtov_list.append(mtov)
        r14_list.append(r14)
        l14_list.append(l14)
        p3nsat = np.interp(3, n, p)
        p3nsat_list.append(p3nsat)
        
        # FIXME: n_TOV is not possible with the current information?
        
    # Save the result
    np.savez(HAUKE_RESULTS_FILE, mtov_list=mtov_list, r14_list=r14_list, l14_list=l14_list, ntov_list=ntov_list, p3nsat_list=p3nsat_list)
    print("Results saved to", HAUKE_RESULTS_FILE)
        
def fetch_hauke_weights(weights_name: str):
    allowed = ["all", "GW170817", "J0740", "PREX", "CREX", "J0030", "PREX_CREX_NICERS", "radio", "prior"]
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
               max_samples: int = 3_000,
               hauke_string: str = "",
               reweigh_prior: bool = True):
    
    filename = os.path.join(outdir, "eos_samples.npz")
    
    if "GW170817_injection" in outdir:
        print(f"Loading the EOS and NS used for the GW170817 injection")
        m_target, r_target, l_target = load_gw170817_injection_eos()
    
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

    # Sample requested number of indices randomly:
    log_prob = data["log_prob"]
    log_prob = np.exp(log_prob) # so actually no longer log prob but prob... whatever
    
    max_log_prob_idx = np.argmax(log_prob)
    indices = np.random.choice(nb_samples, max_samples, replace=False, p=log_prob/np.sum(log_prob))
    indices = np.append(indices, max_log_prob_idx)

    # Get a colorbar for log prob, but normalized
    norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
    cmap = sns.color_palette("crest", as_cmap=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    if plot_NS:
        print("Creating NS plot . . .")
        bad_counter = 0
        for i in tqdm.tqdm(indices):

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
            
            if any((m[i] > 1.0) * (r[i] > 20.0)):
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
        if "GW170817_injection" in outdir:
            print(f"Plotting the EOS and NS used for the GW170817 injection")
            plt.plot(r_target, m_target, color="red", linestyle = "--", lw=2, label="Injection", zorder = 1e100)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_{\odot}$]")
        plt.xlim(r_min, r_max)
        plt.ylim(m_min, m_max)
        
        # Mass-Lambda plot
        plt.subplot(122)
        if "GW170817_injection" in outdir:
            print(f"Plotting the EOS and NS used for the GW170817 injection")
            plt.plot(m_target, l_target, color="red", linestyle = "--", lw=2, label="Injection", zorder = 1e100)
        plt.xlim(m_min, m_max)
        plt.ylim(l_min, l_max)
        plt.xlabel(r"$M$ [$M_{\odot}$]")
        plt.ylabel(r"$\Lambda$")
        plt.yscale("log")
            
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
        l14_list = []
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
            l14 = np.interp(1.4, _m, _l)
            
            p3nsat = np.interp(3, _n, _p)
            
            pc_TOV = np.interp(mtov, _m, _pc)
            n_TOV = np.interp(pc_TOV, _p, _n)
            
            # Append all
            mtov_list.append(mtov)
            if mtov > 1.4:
                r14_list.append(r14)
                l14_list.append(l14)
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
        r14_list = np.array(r14_list)
        mask = r14_list < 20.0
        plt.hist(r14_list[mask], color="blue", label = "Jester", **hist_kwargs)
        if len(hauke_string) > 0:
            has_r14 = np.where(hauke_histogram_data["mtov_list"] > 1.4, True, False)
            
            r14_weights = weights[has_r14]
            r14_list_hauke = hauke_histogram_data["r14_list"][has_r14]
            
            # Also ditch all R14 above 20 km, since that signals something went wrong?
            keep_idx = r14_list_hauke < 20.0
            r14_list_hauke = r14_list_hauke[keep_idx]
            r14_weights = r14_weights[keep_idx]
            
            plt.hist(r14_list_hauke, color="red", weights=r14_weights, label = "Koehn+", **hist_kwargs)
        plt.xlabel(r"$R_{1.4}$ [km]")
        plt.ylabel("Density")
        
        plt.subplot(223)
        l14_list = np.array(l14_list)
        mask = r14_list < 20.0
        plt.hist(l14_list[mask], color="blue", label = "Jester", **hist_kwargs)
        if len(hauke_string) > 0:
            has_l14 = np.where(hauke_histogram_data["mtov_list"] > 1.4, True, False)
            
            l14_weights = weights[has_l14]
            l14_list_hauke = hauke_histogram_data["l14_list"][has_l14]
            
            # Also ditch all Lambda1.4 that are too large or negative
            keep_idx = (l14_list_hauke < 2_000) * (l14_list_hauke > 0.0)
            l14_list_hauke = l14_list_hauke[keep_idx]
            l14_weights = l14_weights[keep_idx]
            
            plt.hist(l14_list_hauke, color="red", weights=l14_weights, label = "Koehn+", **hist_kwargs)
        # FIXME: cannot comput n_TOV with the current information from Hauke?
        # plt.hist(ntov_list, color="blue", label = "Jester", **hist_kwargs)
        # if len(hauke_string) > 0:
        #     plt.hist(hauke_histogram_data["r14_list"], bins=bins, color="red", histtype="step", lw=2, density = True, weights=weights)
        # plt.xlabel(r"$n_{\rm{TOV}}$ [$n_{\rm{sat}}$]")
        
        plt.xlabel(r"$\Lambda_{1.4}$")
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
        
    # If this is the run where we combine all constraints, then also make the master plot
    if "all" in outdir:
        print(f"This is the all-constraints run. Therefore, we also make the master plot!")
        NB_POINTS = 100
        nmin_grid = 0.5 
        nmax_grid = 8.0
        
        # Load the data again, this is just because I am too lazy to check if we overwrite variables or not and I want it in this same function to automate the workflow... this code becomes worse and worse every day :) :) :)
        filename = os.path.join(outdir, "eos_samples.npz")
        
        data = np.load(filename)
        log_prob = data["log_prob"]
        
        m_min = 1.0
        m, r, l = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
        logpc_EOS = data["logpc_EOS"]
        pc_EOS = np.exp(logpc_EOS) / jose_utils.MeV_fm_inv3_to_geometric
        
        n, p, e, cs2 = data["n"], data["p"], data["e"], data["cs2"]
        n = n / jose_utils.fm_inv3_to_geometric / 0.16
        p = p / jose_utils.MeV_fm_inv3_to_geometric
        e = e / jose_utils.MeV_fm_inv3_to_geometric
        
        # TODO: find an efficient way to get n_TOV?
        # last_pc = pc_EOS[:, -1]
        # n_TOV = np.interp(last_pc, p, n)
        
        # Get the maximum log prob index
        max_log_prob_idx = np.argmax(log_prob)
        
        # Get Koehn data
        hauke_data = np.genfromtxt("../doppelgangers/hauke_macroscopic.dat", skip_header=1, delimiter=" ").T
        r_hauke, m_hauke, Lambdas_hauke = hauke_data[0], hauke_data[1], hauke_data[2]
        
        # First comparison plot of max log prob:
        plt.subplots(1, 2, figsize=(12, 6))
        plt.subplot(121)
        _r, _m, _l = r[max_log_prob_idx], m[max_log_prob_idx], l[max_log_prob_idx]
        mask_jester = _m > 0.5
        mask_hauke = m_hauke > 0.5
        
        plt.plot(_r[mask_jester], _m[mask_jester], color="blue", label="Jester", lw=2)
        plt.plot(r_hauke[mask_hauke], m_hauke[mask_hauke], color="red", label="Koehn+", lw=2)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_{\odot}$]")
        plt.ylim(bottom = 0.75)
        
        plt.subplot(122)
        plt.plot(_m[mask_jester], _l[mask_jester], color="blue", label="Jester", lw=2)
        plt.plot(m_hauke[mask_hauke], Lambdas_hauke[mask_hauke], color="red", label="Koehn+", lw=2)
        plt.yscale("log")
        plt.xlabel(r"$M$ [$M_{\odot}$]")
        plt.ylabel(r"$\Lambda$")
        plt.xlim(left = 0.75)
        plt.legend()
        
        plt.savefig(os.path.join(outdir, "master_max_log_prob_comparison.png"), bbox_inches = "tight")
        plt.savefig(os.path.join(outdir, "master_max_log_prob_comparison.pdf"), bbox_inches = "tight")
        plt.close()
        
        # Now, for the combined posteriors plots for EOS and NS, taking inspiration from Fig 26 of Koehn+
        
        # TODO: do subplots, but as test case, let us check out cs2
        n_grid = np.linspace(nmin_grid, nmax_grid, NB_POINTS)
        m_grid = np.linspace(0.75, 3.0, NB_POINTS)
        
        # Interpolate all EOS cs2 on this n_grid
        cs2_interp_array = np.array([np.interp(n_grid, n[i], cs2[i]) for i in range(nb_samples)]).T
        r_interp_array = np.array([np.interp(m_grid, m[i], r[i], left = -1, right = -1) for i in range(nb_samples)]).T
        
        plt.subplots(nrows = 1, ncols = 2, figsize=(12, 6))
        arrays = [r_interp_array, cs2_interp_array]
        for plot_idx in range(2):
            plt.subplot(1, 2, plot_idx + 1)
            interp_array = arrays[plot_idx]
            median_values = []
            low_values = []
            high_values = []
            
            for i in range(NB_POINTS):
                # Determine median
                values_here = interp_array[i]
                mask = values_here > 0
                values_here = values_here[mask]
                median = np.median(values_here)
                median_values.append(median)
                
                # Use arviz to compute the 90% CI
                low, high = arviz.hdi(values_here, hdi_prob = 0.95)
                low_values.append(low)
                high_values.append(high)
        
            # Now, make the final plot
            if plot_idx == 0:
                m_max, r_max = m[max_log_prob_idx], r[max_log_prob_idx]
                mask = m_max > 0.75
                plt.plot(r_max[mask], m_max[mask], color="blue")
                plt.fill_betweenx(m_grid, low_values, high_values, color="blue", alpha=0.25)
            else:
                # cs2_max = cs2_interp_array.T[max_log_prob_idx]
                plt.plot(n_grid, median_values, color="blue")
                plt.fill_between(n_grid, low_values, high_values, color="blue", alpha=0.25)
        
        # Add the labels here manually
        plt.subplot(121)
        plt.xlabel(r"$R$ [km]")
        plt.ylabel(r"$M$ [$M_\odot$]")
        plt.ylim(bottom = 0.75, top = 2.5)
        
        # Add the labels here manually
        plt.subplot(122)
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"$c_s^2$")
        plt.axhline(0.33, linestyle = "--", color="black")
        plt.savefig(os.path.join(outdir, "master_plot.png"), bbox_inches = "tight")
        plt.savefig(os.path.join(outdir, "master_plot.pdf"), bbox_inches = "tight")
        plt.close()
        

def main():
    
    if len(sys.argv) < 2:
        print("No outdir_suffix provided, therefore, we assume you want to gather the results for the Koehn+ paper")
        gather_hauke_results()
        exit()
        
    outdir = sys.argv[1]
    suffix = outdir.split("outdir_")[1]
    if suffix.endswith("/"):
        suffix = suffix[:-1]
    
    # Check if we have a run with the given suffix
    allowed_suffixes = []
    for entry in os.listdir("."):
        if os.path.isdir(entry) and entry.startswith("outdir_"):
            # Extract suffix after "outdir_"
            allowed_suffixes.append(entry.split("outdir_")[1])

    if suffix not in allowed_suffixes:
        print(f"Error: There is no run with the suffix '{suffix}'. Allowed suffixes are: {allowed_suffixes}")
        sys.exit(1)
        
    print("suffix")
    print(suffix)
        
    ### Single postprocessing
    hauke_string = suffix.split("_")[0]
    outdir = f"./outdir_{suffix}/"
    print(f"Making plots for {outdir}")
    make_plots(outdir,
                plot_NS=False,
                plot_EOS=False, # TODO: implement this!
                plot_histograms=False,
                hauke_string=hauke_string)

if __name__ == "__main__":
    main()
    
    