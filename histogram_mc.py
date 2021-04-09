"""
Lucas J. Koerner, koerner.lucas@stthomas.edu
April 14, 2020
Updates: Aug 2020

Monte carlo simulation of dToF sensor to determine
    noise impacts of various instrument parameters

"""
import os
import itertools
import pdb
import shutil
import pickle as pkl
from datetime import datetime  # Current date time in local system
import numpy as np
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad, trapz

d = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

np.random.seed(12)

PLT = False
SAVE_FIGS = True
figure_dir = 'figures/'
figure_dir_tocopy = 'figures_copy/'
bck_print = False
PRINT_FITS = False
INTEGRATE_MASK = False
PLT_HIST = True

if PLT:
    # setup plotting
    plt.ion()
    marker = itertools.cycle(('.', '+', 's', '*'))

_c = 3.0e8

# dictionary of the simulation configuration
cfg = {}

# for continuous time photon generation
cfg['cps'] = 100000  # this is the background counts per second
cfg['frame_time'] = 33e-3

# need to keep simulating past the estimated stop time
# and then chop at the stop time
# make 20% longer to get to stop time
cfg['num_events'] = int(cfg['cps'] * cfg['frame_time'] * 1.2)
cfg['tdc_std'] = 100e-12
cfg['tdc_std_arr'] = np.array([1]) * cfg['tdc_std']
cfg['sig_tdelay'] = 5e-9

# required for TDC quantization
cfg['tdc_width'] = 150e-12
cfg['tdc_bins'] = 176
cfg['tdc_fs'] = cfg['tdc_width'] * cfg['tdc_bins']

# update frame-time to be an integer number of laser pulses
cfg['frame_time'] = int(cfg['frame_time'] / cfg['tdc_fs']) * cfg['tdc_fs']

cfg['span_sigma'] = 3  # 3 sigma in terms of IRF left and right for fits
cfg['delay'] = 12e-9

cfg['xtol'] = None
cfg['ftol'] = None
cfg['pileup'] = False

# save the configuration for each simulation iteration (since some things
# change)
cfg_list = []


def mc_background(cfg, repeats):
    """
    Create a list of arrays of background event times.
    """

    bck_res = []  # list of numpy arrays

    for _ in range(repeats):
        inter_event_times = np.random.exponential(
            1 / cfg['cps'], cfg['num_events'])
        event_times = np.cumsum(inter_event_times)

        # how to do this and still keep information for photon pileup?
        truncated_times = event_times[event_times < cfg['frame_time']]

        # other information that may be useful (i.e. for photon pile-up):
        # modulo_times = truncated_times % cfg['tdc_fs']
        # divisor_times = (truncated_times // cfg['tdc_fs']).astype(int)
        bck_res.append(truncated_times)

    return bck_res


def mc_signal(cfg, ph_sig, delay, repeats, PILEUP=False):
    """
    Poisson sample of ph_sig (average photons in a frame time? )
    Need to do this modulo the tdc_fs to keep information for photon pile-up
    """

    sig_res = []

    pulses = int(cfg['frame_time'] / cfg['tdc_fs'])
    ph_per_pulse = ph_sig / (cfg['frame_time'] / cfg['tdc_fs'])

    for _ in range(repeats):
        # if photon pileup
        if PILEUP:  # works but is terribly slow
            signal = np.array([])
            for p_num in range(pulses):
                s = np.random.normal(loc=delay + p_num * cfg['tdc_fs'], scale=cfg['tdc_std'],
                                     size=np.random.poisson(ph_per_pulse))
                signal = np.append(signal, s)

            # [n_sig, bins2] = np.histogram(signal, bins=bins) # defines the right-most bin edge
            #tdc_result = n + n_sig
        else:
            # signal = np.random.normal(loc = delay, scale = cfg['tdc_std'], size = np.random.poisson(ph_sig))
            signal = np.random.default_rng().normal(loc=delay, scale=cfg['tdc_std'],
                                                    size=np.random.poisson(ph_sig))
        sig_res.append(signal)

    return sig_res  # returns a list of np arrays


def combine_sig_bck(cfg, bck, sig):
    """

    """
    combined = []
    for n in range(np.shape(sig)[0]):
        # background modulo
        modulo_bck = bck[n] % cfg['tdc_fs']
        total = np.append(modulo_bck, sig[n])

        combined.append(total)

    return combined


def quantize(cfg, tdc_times, PLT=True):
    """
    Given a continuous time array of photon arrivals
    quantize into a histogram of TDC outputs
    """

    bins_seq = np.linspace(
        0,
        (cfg['tdc_bins'] - 1) * cfg['tdc_width'],
        cfg['tdc_bins'])
    [n, bin_edges] = np.histogram(tdc_times, bins=bins_seq)

    if PLT:
        # for plotting
        width = 0.9 * (bin_edges[1] - bin_edges[0])  # bin width
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # array of bin centers
        plt.bar(centers, n, align='center', width=width)
        plt.show()

    return n, bin_edges


def find_peak(tdc_result, bins, fit_methods=['cm', 'gauss', 'gauss_wt']):
    """
    Locate the peak using various fitting fit_methods
        mode (required), center of mass, Gaussiam least square,
        Gauss weighted least squares,
        and the method of Thompson (CoM weighted by Gaussian)
    """

    center = (bins[:-1] + bins[1:]) / 2  # array of bin centers

    idx = np.argmax(tdc_result)
    mode = center[idx]

    low_idx = int(np.max([0,
                          idx + np.floor(-cfg['span_sigma'] * cfg['tdc_std'] / cfg['tdc_width'])]))
    high_idx = int(np.min([len(center) - 2,
                           idx + np.ceil(cfg['span_sigma'] * cfg['tdc_std'] / cfg['tdc_width'])]))
    high_idx = high_idx + 1  # numpy slicing is not inclusive of last element

    # even if we hit an edge of the TDC we must span at least 3 bins otherwise
    # the fit fails
    if high_idx == (len(center) - 1):
        low_idx = np.min([low_idx, high_idx - 3])
    if low_idx == 0:
        high_idx = np.max([high_idx, low_idx + 3])

    # center of mass around the mode
    if 'cm' in fit_methods:
        cm = np.sum(center[low_idx: high_idx] * tdc_result[low_idx: high_idx]
                    ) / np.sum(tdc_result[low_idx: high_idx])
    else:
        cm = None

    def gauss(x, a, mu, sigma):
        return a * 1 / ((2 * np.pi)**0.5) * \
            np.exp(-1 / 2 * ((x - mu) / sigma)**2)

    if 'gauss' in fit_methods:
        try:
            if (cfg['xtol'] is not None) and (cfg['ftol'] is not None):
                fit = optimization.curve_fit(gauss, center[low_idx: high_idx],
                                             tdc_result[low_idx: high_idx],
                                             [np.max(tdc_result) * cfg['tdc_std'] / cfg['tdc_width'] / np.sqrt(np.pi),
                                                 mode,
                                              cfg['tdc_std']],
                                             ftol=cfg['ftol'], xtol=cfg['xtol'])
            else:
                fit = optimization.curve_fit(gauss, center[low_idx: high_idx],
                                             tdc_result[low_idx: high_idx],
                                             [np.max(tdc_result) * cfg['tdc_std'] / cfg['tdc_width'] / np.sqrt(np.pi),
                                              mode,
                                              cfg['tdc_std']])
            gfit = fit[0][1]
        except RuntimeError:
            gfit = np.nan
    else:
        gfit = None

    if 'gauss_wt' in fit_methods:
        try:
            if (cfg['xtol'] is not None) and (cfg['ftol'] is not None):
                fit = optimization.curve_fit(gauss, center[low_idx: high_idx],
                                             tdc_result[low_idx: high_idx],
                                             [np.max(tdc_result) * cfg['tdc_std'] / cfg['tdc_width'] / np.sqrt(np.pi),
                                                 mode,
                                              cfg['tdc_std']],
                                             sigma=np.sqrt(
                                                 tdc_result[low_idx: high_idx]),
                                             ftol=cfg['ftol'], xtol=cfg['xtol'])
            else:
                fit = optimization.curve_fit(gauss, center[low_idx: high_idx],
                                             tdc_result[low_idx: high_idx],
                                             [np.max(tdc_result) * cfg['tdc_std'] / cfg['tdc_width'] / np.sqrt(np.pi),
                                                 mode,
                                              cfg['tdc_std']],
                                             sigma=np.sqrt(tdc_result[low_idx: high_idx]))
            gfit_wt = fit[0][1]

        except RuntimeError:
            gfit_wt = np.nan
    else:
        gfit_wt = None

    if 'gauss_subtract' in fit_methods:
        bck_gnd = np.delete(tdc_result, np.arange(low_idx, high_idx + 1))
        tdc_subtract = np.mean(bck_gnd)
        try:
            if (cfg['xtol'] is not None) and (cfg['ftol'] is not None):
                fit = optimization.curve_fit(gauss, center[low_idx: high_idx],
                                             tdc_result[low_idx: high_idx] -
                                             tdc_subtract,
                                             [np.max(tdc_result) * cfg['tdc_std'] / cfg['tdc_width'] / np.sqrt(np.pi),
                                                 mode,
                                              cfg['tdc_std']],
                                             ftol=cfg['ftol'], xtol=cfg['xtol'])
            else:
                fit = optimization.curve_fit(gauss, center[low_idx: high_idx],
                                             tdc_result[low_idx: high_idx] -
                                             tdc_subtract,
                                             [np.max(tdc_result) * cfg['tdc_std'] / cfg['tdc_width'] / np.sqrt(np.pi),
                                              mode,
                                              cfg['tdc_std']])
            gfit_subtract = fit[0][1]
        except RuntimeError:
            gfit_subtract = np.nan
    else:
        gfit_subtract = None
        tdc_subtract = None
    if PLT_HIST:
        fz = 14
        width = (0.9 * (bins[1] - bins[0])) * 1e9  # bin width
        xmin = (center[idx] - 3e-9) * 1e9
        xmax = (center[idx] + 3e-9) * 1e9
        fig, ax = plt.subplots()
        ax.bar(center * 1e9, tdc_result, align='center', width=width)

        gauss_func = gauss(center, fit[0][0], fit[0][1], fit[0][2])
        ax.plot(center * 1e9, gauss_func, 'm', linewidth=2)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([0, np.max(tdc_result) + 3])
        ax.set_ylabel('Q')
        ax.set_xlabel('t [ns]')
        ax.text(9.3, b_per_bin + 10,
                '$b = {:.1f}$'.format(b_per_bin),
                color='k', fontsize=fz)
        ax.arrow(9.5, 0, 0, b_per_bin,
                 width=0.005, head_width=0.1, head_length=5,
                 color='k', length_includes_head=True)
        ax.axhline(b_per_bin, linestyle='--', color='k')

        mu = fit[0][1] * 1e9
        std_dev = fit[0][2] * 1e9
        amp = fit[0][0]

        print(mu - std_dev)
        print(std_dev)
        print('TDC bin width ={}'.format(cfg['tdc_width']))

        height_1sigma_width = 0.88249
        max_amp = np.max(gauss_func)

        ax.text(mu - 1.5, height_1sigma_width * max_amp, s='$\sigma$ = 100 ps',
                fontsize=fz)
        ax.text(
            12.5,
            height_1sigma_width *
            max_amp,
            'N={}'.format(N),
            fontsize=fz)
        ax.text(13.0, b_per_bin + 25,
                'a = {} ps'.format(int(cfg['tdc_width'] * 1e12)),
                fontsize=fz, color='g')
        ax.axvline(bins[90] * 1e9, 0, 0.3, color='g')
        ax.axvline(bins[91] * 1e9, 0, 0.3, color='g')

        if SAVE_FIGS:
            figname = 'histogram_example'
            for e in ['.png', '.eps']:
                fig.savefig(os.path.join(figure_dir,
                                         figname + e))
                if figure_dir_tocopy is not None:
                    shutil.copy2(
                        os.path.join(
                            figure_dir,
                            figname + e),
                        figure_dir_tocopy)
        # pdb.set_trace()

    if not INTEGRATE_MASK:  # don't integrate mask

        if 'gauss_mask' in fit_methods:
            # implement method of Thompson, 2002
            # calculate mask with starting x0, calculate x0, iterate
            x0 = mode
            iterations = 0
            x0_diff = 99
            while ((x0_diff > 1e-15) and (iterations < 200)):
                mask = np.exp(-1 / 2 * ((center - x0) / cfg['tdc_std'])**2)
                x0_new = np.sum(center * mask * tdc_result) / \
                    np.sum(mask * tdc_result)
                iterations = iterations + 1
                x0_diff = np.abs(x0 - x0_new)
                # print(x0_diff)
                x0 = x0_new
            mask_fit = x0
            #print('Mask fit = {}; iterations = {}'.format(mask_fit, iterations))
        else:
            mask_fit = None

    if INTEGRATE_MASK:  # integrate
        if 'gauss_mask' in fit_methods:
            # implement method of Thompson, 2002
            # iterate until convergence or a maximum number of iterations
            # calculate mask with starting x0, calculate x0, iterate
            def gauss_func(x, x0, tdc_std):
                return np.exp(-1 / 2 * ((x - x0) / tdc_std)**2)
            x0 = mode
            iterations = 0
            max_iterations = 200  # as used in Thompson, pg 2776
            x0_diff = 99

            while ((x0_diff > 1e-13) and (iterations < max_iterations)):
                # signal photon sweep, very few went beyond 30 iterations, most at or below 10
                # plateaued at around 4 ps

                # integrate mask using quad
                # mask_int = np.array([])
                # dx = bins[1] - bins[0]
                # for b in bins[:-1]:  # all but the last bin edge
                #     mask_tmp = quad(gauss_func, b, b+dx, args=(x0, cfg['tdc_std']))
                #     mask_int = np.append(mask_int, mask_tmp[0])
                # mask = mask_int/dx

                # quad is slow, use trapz instead
                # mask_trapz = np.array([])
                # dx = bins[1] - bins[0]
                # trap_steps = 20
                # for b in bins[:-1]:  # all but the last bin edge
                #     y_tmp = map(gauss_func, np.linspace(b, b+dx, trap_steps), [x0]*trap_steps, [cfg['tdc_std']]*trap_steps )
                #     mask_trapz = np.append(mask_trapz, trapz(list(y_tmp)))
                # mask = mask_trapz/(trap_steps - 1)

                # trapz is slow lookup best mask (this has precision limits at
                # high photon counts, high precision)

                idx = (np.abs(mask_lookup_vals - x0)).argmin()
                mask = mask_lookup[idx]

                x0_new = np.sum(center * mask * tdc_result) / \
                    np.sum(mask * tdc_result)
                iterations = iterations + 1
                x0_diff = np.abs(x0 - x0_new)
                x0 = x0_new

            mask_fit = x0
        else:
            mask_fit = None

    if PRINT_FITS:
        print('Mode = {} \nCoM = {} \nGauss = {} \nGauss Wt = {} \nMask fit = {} '.format(mode,
                                                                                          cm,
                                                                                          gfit,
                                                                                          gfit_wt,
                                                                                          mask_fit))

    return mode, cm, gfit, gfit_wt, mask_fit, iterations, gfit_subtract, tdc_subtract


N_arr = np.logspace(1.5, 5.5, 100)

# how many times to simulate (each simulation covers 'frame_time' seconds)
cfg['n_repeats'] = 50
if PLT_HIST:
    cfg['n_repeats'] = 1

print('Background photons per tdc bin = {}'.format(
    cfg['cps'] * cfg['frame_time'] / cfg['tdc_bins']))

fits = {}  # dictionary to store arrays of results
for k in ['mode', 'cm', 'gfit', 'gfit_wt', 'gfit_subtract',
          'tdc_background',
          'mask_fit', 'mask_fit_iters', 'N', 'delay',
          'max_val', 'tdc_width', 'tdc_bins', 'tdc_std',
          'cps']:
    fits[k] = np.array([])

n_num = 0
last_std_lookup = 0
time_array = []

# N_arr = [50, 300, 5000] # total number of photons
#N_arr = np.logspace(1.7, 4, 50)
N_arr = [300]  # for background sweep

# this is the background counts per second (about 18.75 counts per bin)
cfg['cps'] = 100000
cfg['tdc_bins_arr'] = np.linspace(12, 23, 12)
cfg['tdc_bins_arr'] = np.append(cfg['tdc_bins_arr'], np.linspace(24, 46, 12))
cfg['tdc_bins_arr'] = np.append(
    cfg['tdc_bins_arr'], np.arange(
        48, 256, step=8))
cfg['tdc_bins_arr'] = np.append(
    cfg['tdc_bins_arr'], np.arange(
        256, 512, step=32))
cfg['tdc_bins_arr'] = np.append(
    cfg['tdc_bins_arr'], np.arange(
        512, 2112, step=64))
cfg['tdc_bins_arr'] = [176]

cfg['tdc_std_arr'] = [100e-12]

# for back in np.linspace(10000, 50000000, 200):
# for back in np.logspace(3, 9, 400):
for back in [0.1]:
    # for back in [cfg['cps']]:
    cfg['cps'] = back * 1600000.0
    # make 20% longer to get to stop time
    cfg['num_events'] = int(cfg['cps'] * cfg['frame_time'] * 1.2)

    for N in N_arr:
        d_temp = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
        print('Loop! Value of N = {} at time = {}'.format(N, d_temp))

        bck_res = mc_background(cfg, cfg['n_repeats'])

        for tdc_std_m in cfg['tdc_std_arr']:
            for b in cfg['tdc_bins_arr']:  # vary TDC bins
                # for b in [16, 2048]: # vary TDC bins
                if PLT_HIST:
                    b_per_bin = cfg['cps'] * cfg['frame_time'] / b
                    print('Number of background per bin: {}'.format(b_per_bin))
                # FWHM / 2.355 for standard deviation
                cfg['tdc_std'] = tdc_std_m
                cfg['tdc_bins'] = int(b)
                cfg['tdc_width'] = cfg['tdc_fs'] / cfg['tdc_bins']

                for n in range(cfg['n_repeats']):
                    delay = cfg['delay'] + np.random.random() * \
                        cfg['tdc_width']
                    sig_res = mc_signal(cfg, N, delay,
                                        cfg['n_repeats'], cfg['pileup'])
                    c = combine_sig_bck(cfg, bck_res, sig_res)
                    tdc_data, bins = quantize(cfg, c[n], PLT=PLT)
                    print(
                        'total number of photons = {}'.format(
                            np.sum(tdc_data)))

                    if (n == 0) and (last_std_lookup !=
                                     tdc_std_m) and INTEGRATE_MASK:
                        last_std_lookup = tdc_std_m
                        # build a look up table for Gaussian mask

                        def gauss_func(x, x0, tdc_std):
                            return np.exp(-1 / 2 * ((x - x0) / tdc_std)**2)

                        mask_lookup_vals = np.linspace(cfg['delay'] - 3 * cfg['tdc_width'],
                                                       cfg['delay'] + 3 * cfg['tdc_width'], 100000)  # was steps of 0.15 ps; now 0.009 ps
                        mask_lookup = []
                        dx = bins[1] - bins[0]

                        for x0 in mask_lookup_vals:
                            mask_trapz = np.array([])
                            trap_steps = 20
                            for b in bins[:-1]:  # all but the last bin edge
                                y_tmp = map(gauss_func, np.linspace(
                                    b, b + dx, trap_steps), [x0] * trap_steps, [cfg['tdc_std']] * trap_steps)
                                mask_trapz = np.append(
                                    mask_trapz, trapz(list(y_tmp)))
                            mask = mask_trapz / (trap_steps - 1)
                            mask_lookup.append(mask)

                    mode, cm, gfit, gfit_wt, mask_fit, mask_fit_iters, gfit_subtract, tdc_background = find_peak(tdc_data,
                                                                                                                 bins, ['cm', 'gauss', 'gauss_mask', 'gauss_subtract'])

                    fits['mode'] = np.append(fits['mode'], mode)
                    fits['cm'] = np.append(fits['cm'], cm)
                    fits['gfit'] = np.append(fits['gfit'], gfit)
                    fits['gfit_wt'] = np.append(fits['gfit_wt'], gfit_wt)
                    fits['mask_fit'] = np.append(fits['mask_fit'], mask_fit)
                    fits['mask_fit_iters'] = np.append(
                        fits['mask_fit_iters'], mask_fit_iters)

                    fits['N'] = np.append(fits['N'], N)
                    fits['delay'] = np.append(fits['delay'], delay)
                    fits['max_val'] = np.append(fits['max_val'],
                                                np.max(tdc_data))
                    fits['tdc_width'] = np.append(
                        fits['tdc_width'], cfg['tdc_width'])
                    fits['tdc_bins'] = np.append(
                        fits['tdc_bins'], cfg['tdc_bins'])
                    fits['tdc_std'] = np.append(
                        fits['tdc_std'], cfg['tdc_std'])
                    fits['cps'] = np.append(fits['cps'], cfg['cps'])
                    fits['gfit_subtract'] = np.append(
                        fits['gfit_subtract'], gfit_subtract)
                    fits['tdc_background'] = np.append(
                        fits['tdc_background'], tdc_background)

            # pickle the data frame (partial) and configuration
            df = pd.DataFrame.from_dict(fits)
            d_temp = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
            time_array.append(d_temp)

            # running list of the configurations used for each iteration (each
            # n_num)
            cfg_list.append(cfg)

            print('Pickle loop {} at {}'.format(n_num, d))
            # Python 3: open(..., 'wb')
            with open('sims/sim_res_temp_{}_{}.pkl'.format(d, n_num), 'wb') as f:
                pkl.dump([df, cfg], f)
            n_num = n_num + 1

# pickle the data frame and configuration
df = pd.DataFrame.from_dict(fits)
d_temp = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
time_array.append(d_temp)
print('Pickle loop {} at {}'.format(n_num, d))
with open('sims/sim_res_temp_{}_{}.pkl'.format(d, n_num), 'wb') as f:  # Python 3: open(..., 'wb')
    pkl.dump([df, cfg_list], f)

# quick plot to check results
if PLT:
    for method in ['cm', 'gfit', 'mask_fit']:
        plt.figure()
        for b in df.tdc_bins.unique():
            summary = np.array([])

            dfb = df.loc[(df['tdc_bins'] == b)]
            N_unq = dfb.N.unique()
            for v in N_unq:
                cm = dfb.loc[dfb['N'] == v][method]
                dly = dfb.loc[dfb['N'] == v]['delay']
                noise = np.std(cm - dly)
                summary = np.append(summary, noise)

            plt.loglog(N_unq, summary, marker='*',
                       label='Bins = {}'.format(b))
        plt.legend()
        plt.title(method)

# ----------------------------------------------------
# helper functions used to test methods


def swp_ftol_xtol(bins, tdc_result, ftol_arr, xtol_arr):
    """
    almost any tolerance to 1e-2 is equivalent.
    """

    res = {}
    res['gfit'] = np.array([])
    res['gfit_err'] = np.array([])
    res['ftol'] = np.array([])
    res['xtol'] = np.array([])
    res['fit'] = []

    for ftol in ftol_arr:
        for xtol in xtol_arr:

            center = (bins[:-1] + bins[1:]) / 2  # array of bin centers
            # locate signal using a few methods
            idx = np.argmax(tdc_result)
            mode = center[idx]
            low_idx = int(np.max([0,
                                  idx + np.floor(-cfg['span_sigma'] * cfg['tdc_std'] / cfg['tdc_width'])]))
            high_idx = int(np.min([len(center) - 1,
                                   idx + np.ceil(cfg['span_sigma'] * cfg['tdc_std'] / cfg['tdc_width'])]))

            # numpy slicing is not inclusive of last element (this makes it
            # symmetric)
            high_idx = high_idx + 1

            print(low_idx)
            print(high_idx)

            def gauss(x, a, mu, sigma):
                return a * 1 / ((2 * np.pi)**0.5) * \
                    np.exp(-1 / 2 * ((x - mu) / sigma)**2)

            try:
                fit = optimization.curve_fit(gauss, center[low_idx: high_idx],
                                             tdc_result[low_idx: high_idx],
                                             [np.max(tdc_result) * cfg['tdc_std'] / cfg['tdc_width'] / np.sqrt(np.pi),
                                                 mode,
                                              cfg['tdc_std']],
                                             ftol=ftol, xtol=xtol)
                print(fit)
                gfit = fit[0][1]
            except RuntimeError:
                gfit = np.nan
                fit = None

            res['ftol'] = np.append(res['ftol'], ftol)
            res['xtol'] = np.append(res['xtol'], xtol)
            res['gfit'] = np.append(res['gfit'], gfit)
            res['gfit_err'] = np.append(
                res['gfit_err'], np.sqrt(
                    np.diag(
                        fit[1]))[1])
            res['fit'].append(fit)

    return res


def summarize(tf):
    """
    summarize the results of a dataframe
    """
    n_unq = np.unique(tf['N'])

    for v in n_unq:
        print('Photon count = {}'.format(v))
        for k in tf.keys():
            idx = np.argwhere(tf['N'] == v)
            if k not in ['N', 'delay']:
                try:
                    print('{} Method:  Mean = {}; Std = {}'.format(k,
                                                                   np.mean(
                                                                       tf[k][idx] - tf['delay'][idx]),
                                                                   np.std(tf[k][idx] - tf['delay'][idx])))
                except BaseException:
                    pass


def get_vector(tf, k, calc, remove_baseline=True):

    n_unq = np.unique(tf['N'])
    summary = np.array([])
    for v in n_unq:
        idx = np.argwhere(tf['N'] == v)
        if remove_baseline:
            summary = np.append(summary, calc(tf[k][idx] - tf['delay'][idx]))
        else:
            summary = np.append(summary, calc(tf[k][idx]))
    return n_unq, summary


def test_mask_int():
    """
    test the Thompson Mask method (with integration of the mask)
    """

    bins = np.linspace(0, 300e-12, 31)
    center = (bins[:-1] + bins[1:]) / 2  # array of bin centers
    x0 = 145e-12
    cfg = {'tdc_std': 30e-12}

    def gauss_func(x, x0, tdc_std):
        return np.exp(-1 / 2 * ((x - x0) / tdc_std)**2)

    mask = np.exp(-1 / 2 * ((center - x0) / cfg['tdc_std'])**2)

    mask_int = np.array([])
    dx = bins[1] - bins[0]
    for b in bins[:-1]:  # all but the last bin edge
        mask_tmp = quad(gauss_func, b, b + dx, args=(x0, cfg['tdc_std']))
        mask_int = np.append(mask_int, mask_tmp[0])
    mask_int = mask_int / dx

    # quad is slow
    mask_trapz = np.array([])
    dx = bins[1] - bins[0]
    trap_steps = 20
    for b in bins[:-1]:  # all but the last bin edge
        y_tmp = map(gauss_func, np.linspace(b, b + dx, trap_steps),
                    [x0] * trap_steps, [cfg['tdc_std']] * trap_steps)
        mask_trapz = np.append(mask_trapz, trapz(list(y_tmp)))
    mask_trapz = mask_trapz / (trap_steps - 1)

    plt.plot(mask)
    plt.plot(mask_int)
    plt.plot(mask_trapz)
    plt.show()
    # Conclusion: the integration method produces a result nearly identical to evaluation
    # at the center of each bin
