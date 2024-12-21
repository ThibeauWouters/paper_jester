"""Quickly doing some postprocessing on the results of the inference."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
import sys
import corner
import tqdm
import argparse
import arviz

np.random.seed(2)

import joseTOV.utils as jose_utils

# from paper_jose.inference.inference import prior_list # FIXME: no longer possible to import this

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

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)

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
    allowed = ["all", "GW170817", "J0740", "PREX", "CREX", "J0030", "PREX_CREX_NICERS", "radio", "chiEFT", "prior"]
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

def lambda_1_lambda_2_to_lambda_tilde(eta, lambda_1, lambda_2):
    lambda_plus = lambda_1 + lambda_2
    lambda_minus = lambda_1 - lambda_2
    lambda_tilde = 8 / 13 * (
        (1 + 7 * eta - 31 * eta**2) * lambda_plus +
        (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta**2) * lambda_minus)

    delta_lambda_tilde = 1 / 2 * (
        (1 - 4 * eta) ** 0.5 * (1 - 13272 / 1319 * eta + 8944 / 1319 * eta**2) *
        lambda_plus + (1 - 15910 / 1319 * eta + 32850 / 1319 * eta ** 2 +
                    3380 / 1319 * eta ** 3) * lambda_minus)
    
    return lambda_tilde, delta_lambda_tilde

def make_plots(outdir: str,
               plot_R_and_p: bool = True,
               plot_EOS: bool = False,
               plot_histograms: bool = True,
               max_samples: int = 3_000,
               hauke_string: str = ""):
    
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

    plt.subplots(1, 2, figsize=(11, 6))

    m_min, m_max = 0.3, 3.75
    r_min, r_max = 5.5, 18.0
    l_min, l_max = 1.0, 50_000.0
    
    if "all" in outdir:
        m_min, m_max = 0.3, 3.0
        r_min, r_max = 9.0, 18.0
        l_min, l_max = 1.0, 50_000.0

    # Sample requested number of indices randomly:
    log_prob = data["log_prob"]
    log_prob = np.exp(log_prob) # so actually no longer log prob but prob... whatever
    
    max_log_prob_idx = np.argmax(log_prob)
    indices = np.random.choice(nb_samples, max_samples, replace=False) # p=log_prob/np.sum(log_prob)
    indices = np.append(indices, max_log_prob_idx)

    # Get a colorbar for log prob, but normalized
    norm = plt.Normalize(vmin=np.min(log_prob), vmax=np.max(log_prob))
    cmap = sns.color_palette("crest", as_cmap=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    if plot_R_and_p:
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
            
            # Pressure as a function of density
            plt.subplot(122)
            last_pc = pc_EOS[i, -1]
            n_TOV = np.interp(last_pc, p[i], n[i])
            mask = (n[i] > 0.5) * (n[i] < n_TOV)
            plt.plot(n[i][mask], p[i][mask], **samples_kwargs)
            
            # # Mass-Lambda plot
            # plt.subplot(122)
            # plt.plot(m[i], l[i], **samples_kwargs)
            # plt.yscale("log")
            # plt.xlim(m_min, m_max)
            # plt.ylim(l_min, l_max)
            # plt.xlabel(r"$M$ [$M_{\odot}$]")
            # plt.ylabel(r"$\Lambda$")
            
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
        
        plt.subplot(122)
        # plt.xlim(m_min, m_max)
        # plt.ylim(l_min, l_max)
        plt.xlabel(r"$n$ [$n_{\rm{sat}}$]")
        plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
        
        # plt.subplot(122)
        # if "GW170817_injection" in outdir:
        #     print(f"Plotting the EOS and NS used for the GW170817 injection")
        #     plt.plot(m_target, l_target, color="red", linestyle = "--", lw=2, label="Injection", zorder = 1e100)
        # plt.xlim(m_min, m_max)
        # plt.ylim(l_min, l_max)
        # plt.xlabel(r"$M$ [$M_{\odot}$]")
        # plt.ylabel(r"$\Lambda$")
        # plt.yscale("log")
            
        # Save
        sm.set_array([])
        # Add the colorbar
        fig = plt.gcf()
        # cbar = plt.colorbar(sm, ax=fig.axes)
        # Add a single colorbar at the top spanning both subplots
        cbar_ax = fig.add_axes([0.15, 0.94, 0.7, 0.03])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Normalized posterior probability", fontsize = 16)
        cbar.set_ticks([])
        cbar.ax.xaxis.labelpad = 5
        cbar.ax.tick_params(labelsize=0, length=0)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.get_offset_text().set_visible(False)
        cbar.set_label(r"Normalized posterior probability")

        plt.savefig(os.path.join(outdir, "postprocessing_NS.png"), bbox_inches = "tight", dpi=300)
        plt.savefig(os.path.join(outdir, "postprocessing_NS.pdf"), bbox_inches = "tight", dpi=300)
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
        
        mass_at_2nat_list = []
        
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
            
            p_at_2nsat = np.interp(2.0, _n, _p)
            mass_at_2nat = np.interp(p_at_2nsat, _pc, _m)
            mass_at_2nat_list.append(mass_at_2nat)
            
        print(f"Negative counter: {negative_counter}")
        
        mass_at_2nat_list = np.array(mass_at_2nat_list)
        median = np.median(mass_at_2nat_list)
        low, high = arviz.hdi(mass_at_2nat_list, hdi_prob = 0.95)
        low = median - low
        high = high - median
        print(f"Mass at 2 nsat: {median:.2f}-{low:.2f}+{high:.2f}")
        
        bins = 50
        hist_kwargs = dict(histtype="step", lw=2, density = True, bins=bins)
        plt.subplots(2, 2, figsize=(18, 12))
        plt.subplot(221)
        plt.hist(mtov_list, color="blue", label = "Jester", **hist_kwargs)
        if len(hauke_string) > 0:
            plt.hist(hauke_histogram_data["mtov_list"], color="red", weights=weights, label = "Hauke", **hist_kwargs)
        plt.xlabel(r"$M_{\rm TOV}$ [$M_{\odot}$]")
        plt.ylabel("Density")
        
        mtov_list = np.array(mtov_list)
        median = np.median(mtov_list)
        low, high = arviz.hdi(mtov_list, hdi_prob = 0.95)
        low = median - low
        high = high - median
        print(f"TOV mass: {median:.2f}-{low:.2f}+{high:.2f}")
        plt.title(r"$M_{\rm TOV}$: " + f"{median:.2f} - {low:.2f} + {high:.2f}")

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
        
        r14_list = np.array(r14_list[mask])
        median = np.median(r14_list)
        low, high = arviz.hdi(r14_list, hdi_prob = 0.95)
        low = median - low
        high = high - median
        print(f"R1.4: {median:.2f}-{low:.2f}+{high:.2f}")
        plt.title(r"$R_{1.4}$ [km]: " + f"{median:.2f}-{low:.2f}+{high:.2f}")
        
        plt.subplot(223)
        # l14_list = np.array(l14_list)
        # mask = r14_list < 20.0
        # plt.hist(l14_list[mask], color="blue", label = "Jester", **hist_kwargs)
        # if len(hauke_string) > 0:
        #     has_l14 = np.where(hauke_histogram_data["mtov_list"] > 1.4, True, False)
            
        #     l14_weights = weights[has_l14]
        #     l14_list_hauke = hauke_histogram_data["l14_list"][has_l14]
            
        #     # Also ditch all Lambda1.4 that are too large or negative
        #     keep_idx = (l14_list_hauke < 2_000) * (l14_list_hauke > 0.0)
        #     l14_list_hauke = l14_list_hauke[keep_idx]
        #     l14_weights = l14_weights[keep_idx]
            
        #     plt.hist(l14_list_hauke, color="red", weights=l14_weights, label = "Koehn+", **hist_kwargs)
        # plt.xlabel(r"$\Lambda_{1.4}$")
        # if len(hauke_string) > 0:
        #     plt.hist(hauke_histogram_data["r14_list"], bins=bins, color="red", histtype="step", lw=2, density = True, weights=weights)
        
        plt.hist(ntov_list, color="blue", label = "Jester", **hist_kwargs)
        plt.xlabel(r"$n_{\rm{TOV}}$ [$n_{\rm{sat}}$]")
        plt.ylabel("Density")
        
        ntov_list = np.array(ntov_list)
        median = np.median(ntov_list)
        low, high = arviz.hdi(ntov_list, hdi_prob = 0.95)
        low = median - low
        high = high - median
        print(f"n_TOV: {median:.4f} - {low:.4f} + {high:.4f}")
        plt.title(r"$n_{\rm{TOV}}$ [$n_{\rm{sat}}$]: " + f"{median:.4f} - {low:.4f} + {high:.4f}")
        
        plt.subplot(224)
        plt.hist(p3nsat_list, color="blue", label = r"\texttt{Jester}", **hist_kwargs)
        if len(hauke_string) > 0:
            plt.hist(hauke_histogram_data["p3nsat_list"], color="red", weights=weights, label = "Koehn+", **hist_kwargs)
        plt.xlabel(r"$p_{3n_{\rm{sat}}}$ [MeV fm$^{-3}$]")
        plt.ylabel("Density")
        plt.legend(fontsize = 24)
        
        p3nsat_list = np.array(p3nsat_list)
        median = np.median(p3nsat_list)
        low, high = arviz.hdi(p3nsat_list, hdi_prob = 0.95)
        low = median - low
        high = high - median
        print(f"p3nsat: {median:.2f}-{low:.2f}+{high:.2f}")
        plt.title(r"$p_{3n_{\rm{sat}}}$ [MeV fm$^{-3}$]: " + f"{median:.4f} - {low:.4f} + {high:.4f}")

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
        
        ### Check how many cs2 curves are above or below 0.33
        counter_cs2_above_033 = 0
        for i in range(nb_samples):
            mask = n[i] < 4.0
            if np.any(cs2[i][mask] > 0.33):
                counter_cs2_above_033 += 1
        
        print(f"Percentage of EOS samples that are above 0.33: {(counter_cs2_above_033 / nb_samples) * 100:.2f}%")
   
def compare_lambda_posteriors(max_samples = 100_000):
    
    # Hauke_data
    hauke_data = np.load("NF/data/GW170817_marginalized_samples.npz")
    data = np.load("outdir_GW170817/eos_samples.npz")
    m_eos_prior, r_eos_prior, l_eos_prior = data["masses_EOS"], data["radii_EOS"], data["Lambdas_EOS"]
    data_agnostic = np.load("outdir_GW170817_agnostic/eos_samples.npz")
    m_agnostic_prior, r_agnostic_prior, l_agnostic_prior = data_agnostic["masses_EOS"], data_agnostic["radii_EOS"], data_agnostic["Lambdas_EOS"]
    
    mass_1_GW_170817 = data["mass_1_GW170817"]
    mass_2_GW_170817 = data["mass_2_GW170817"]
    
    M_c_list = (mass_1_GW_170817 * mass_2_GW_170817)**(3/5) / (mass_1_GW_170817 + mass_2_GW_170817)**(1/5)
    q_list = mass_2_GW_170817 / mass_1_GW_170817
    
    my_M_c_list = []
    my_q_list = []
    my_lambda_tilde_list = []
    my_delta_lambda_list = []
    
    KEYS = ["EOS prior", "Agnostic prior"]
    samples_dict = {}
    
    for key, m, r, l in zip(KEYS, [m_eos_prior, m_agnostic_prior], [r_eos_prior, r_agnostic_prior], [l_eos_prior, l_agnostic_prior]):
        for i in tqdm.tqdm(range(max_samples)):
        
            if any(np.isnan(m[i])) or any(np.isnan(r[i])) or any(np.isnan(l[i])):
                continue
        
            if any(l[i] < 0):
                continue
            
            if any((m[i] > 1.0) * (r[i] > 20.0)):
                continue
            
            # Get the Lambdas
            masses_EOS, Lambdas_EOS = m[i], l[i]
            lambda_1 = np.interp(mass_1_GW_170817[i], masses_EOS, Lambdas_EOS)
            lambda_2 = np.interp(mass_2_GW_170817[i], masses_EOS, Lambdas_EOS)
            q = q_list[i]
            eta = q / (1 + q)**2
            
            lambda_tilde, delta_lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(eta, lambda_1, lambda_2)
            
            # Append all desired:
            my_M_c_list.append(M_c_list[i])
            my_q_list.append(q_list[i])
            my_lambda_tilde_list.append(lambda_tilde)
            my_delta_lambda_list.append(delta_lambda_tilde)
            
        # Save:
        samples_dict[key] = np.array([my_M_c_list, my_q_list, my_lambda_tilde_list, my_delta_lambda_list]).T
    
    # # Gather all the samples
    # low, high = arviz.hdi(M_c_list, hdi_prob = 0.95)
    # low = np.median(M_c_list) - low
    # high = high - np.median(M_c_list)
    # print(f"Source frame chirp mass measurement Jim:  {np.median(M_c_list):.4f} - {low:.4f} + {high:.4f}")
    
    # Now do the same for Hauke:
    m_1, m_2, lambda_1, lambda_2 = hauke_data["m_1"], hauke_data["m_2"], hauke_data["lambda_1"], hauke_data["lambda_2"]
    M_c = (m_1 * m_2)**(3/5) / (m_1 + m_2)**(1/5)
    low, high = arviz.hdi(M_c, hdi_prob = 0.95)
    low = np.median(M_c) - low
    high = high - np.median(M_c)
    print(f"Source frame chirp mass measurement Hauke: {np.median(M_c):.4f} - {low:.4f} + {high:.4f}")
    q = m_2 / m_1
    eta = q / (1 + q)**2
    lambda_tilde, delta_lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(eta, lambda_1, lambda_2)
    
    mask = lambda_tilde < 3_000
    M_c = M_c[mask]
    q = q[mask]
    lambda_tilde = lambda_tilde[mask]
    delta_lambda_tilde = delta_lambda_tilde[mask]
    
    hauke_samples = np.array([M_c, q, lambda_tilde, delta_lambda_tilde]).T
    
    # Make the corner plot
    labels = [r"$M_c$ [$M_{\odot}$]", r"$q$", r"$\tilde{\Lambda}$", r"$\delta \tilde{\Lambda}$"]
    my_range = [[1.18, 1.20],
                [0.60, 1.0],
                [0.0, 2000.0],
                [-500, 500]]
    
    corner_kwargs = copy.deepcopy(default_corner_kwargs)
    
    print(f"Making lambdas cornerplot")
    # EOS prior
    hist_kwargs = {"density": True, "color": "green"}
    corner_kwargs["hist_kwargs"] = hist_kwargs
    corner_kwargs["color"] = "green"
    
    fig = corner.corner(samples_dict[KEYS[0]], labels = labels, **corner_kwargs)
    
    # Agnostic prior
    hist_kwargs = {"density": True, "color": "blue"}
    corner_kwargs["hist_kwargs"] = hist_kwargs
    corner_kwargs["color"] = "blue"
    
    corner.corner(samples_dict[KEYS[0]], labels = labels, fig=fig, **corner_kwargs)
    
    # Koehn+
    hist_kwargs = {"density": True, "color": "red"}
    corner_kwargs["hist_kwargs"] = hist_kwargs
    corner_kwargs["color"] = "red"
    corner.corner(hauke_samples, fig=fig, labels = labels, range=my_range, **corner_kwargs)
    
    fs = 36
    plt.text(0.75, 0.90, "EOS prior", color="green", transform=plt.gcf().transFigure, fontsize=fs)
    plt.text(0.75, 0.80, "Agnostic prior", color="blue", transform=plt.gcf().transFigure, fontsize=fs)
    plt.text(0.75, 0.70, "Koehn+", color="red", transform=plt.gcf().transFigure, fontsize=fs)
    
    plt.savefig(os.path.join("GW170817_lambdas_cornerplot.png"), bbox_inches = "tight")
    plt.close()
    print(f"Making lambdas cornerplot DONE")
    
    # Also make a basic histogram of lambda tilde:
    print(f"Making lambdas histogram plot")
    plt.figure(figsize=(12, 6))
    bins = 50
    hist_kwargs = dict(histtype="step", lw=2, density = True, bins=bins)
    plt.hist(samples_dict[KEYS[0]][:, 2], color="green", label = "EOS prior", **hist_kwargs)
    plt.hist(samples_dict[KEYS[1]][:, 2], color="blue", label = "Agnostic prior", **hist_kwargs)
    plt.hist(hauke_samples[:, 2], color="red", label = "Koehn+", **hist_kwargs)
    plt.xlabel(r"$\tilde{\Lambda}$")
    plt.ylabel("Density")
    plt.xlim(0, 3000)
    plt.legend()
    plt.savefig("GW170817_lambdas_histogram.png", bbox_inches = "tight")
    plt.close()
    print(f"Making lambdas histogram plot DONE")
    
def compare_priors_radio_nuclear():
    
    # Load hauke histogram data
    hist_data = 0
    
def report_NEPs(suffix):
    
    filename = f"outdir_{suffix}/eos_samples.npz"
    data = np.load(filename)
    NEP_keys = ["E_sym", "L_sym", "K_sym", "Q_sym", "Z_sym", "K_sat", "Q_sat", "Z_sat"]

    NEPs = {}
    
    for key in NEP_keys:
        NEPs[key] = data[key]
    
    ### Correlation coefficients
    corrcoefs = {}
    print(f"NEP Correlation coefficients:")
    for i, key in enumerate(NEP_keys):
        for j, key2 in enumerate(NEP_keys):
            if j > i:
                corrcoef = np.corrcoef(NEPs[key], NEPs[key2])[0, 1]
                corrcoefs[f"{key}_{key2}"] = corrcoef
                
                print(f"    {key} - {key2}: {corrcoef}")
            
    # Report some stuff
    print(f"Maximal corrcoef: {max(corrcoefs.values())}")
    print(f"Minimal corrcoef: {min(corrcoefs.values())}")
    
    ### Posterior to prior ratios
    print(f"NEP Posterior to prior ratios:")
    ppr_dict = {}
    for prior in prior_list:
        # All priors are uniform in 1D
        name = prior.parameter_names[0]
        if "sat" not in name and "sym" not in name:
            continue
        else:
            prior_width = prior.xmax - prior.xmin
            prior_sigma = np.sqrt(1/12) * prior_width
            
            posterior_sigma = np.std(NEPs[name])
            ppr_dict[name] = posterior_sigma / prior_sigma
            
    print(f"Posterior to prior ratios")
    for key in ppr_dict.keys():
        print(f"    {key}: {ppr_dict[key]}")
        
    print(f"Maximal posterior to prior ratio: {max(ppr_dict.values())}")
    print(f"Minimal posterior to prior ratio: {min(ppr_dict.values())}")
        
    for key in NEP_keys:
        values = np.array(NEPs[key])
        median = np.median(values)
        low, high = arviz.hdi(values, hdi_prob = 0.95)
        low = median - low
        high = high - median
        
        print(f"{key}: {median:.2f}-{low:.2f}+{high:.2f}")


def main():
    
    # TODO: make this simpler
    # if len(sys.argv) < 2:
    #     print("No outdir_suffix provided, therefore, we assume you want to gather the results for the Koehn+ paper")
    #     gather_hauke_results()
    #     exit()
        
    if len(sys.argv) < 2:
        # compare_lambda_posteriors()
        compare_priors_radio_nuclear()
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
                plot_R_and_p=True,
                plot_EOS=False, # TODO: deprecate this?
                plot_histograms=True,
                hauke_string=hauke_string)
    
    if "all" in suffix:
        # Additionally, check the NEPs
        report_NEPs(suffix)
    
if __name__ == "__main__":
    main()
    
    