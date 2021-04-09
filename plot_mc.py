'''
Lucas J. Koerner, koer2434@stthomas.edu
April 14, 2020

Plot results of Monte Carlo simulation of dToF
to determine noise impacts of various instrument parameters
'''
import os
import itertools
import shutil
import pickle as pkl
from datetime import datetime  # Current date time in local system
import matplotlib.pyplot as plt
import numpy as np
from analytical_plots import vary_bin_size, vs_signal, vs_background

d = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

np.random.seed(123)

# setup plotting
marker = itertools.cycle(('+', '.', 'x', '*'))
line_style = itertools.cycle(('-', '--'))
plt.rcParams['grid.color'] = 'd3d3d3'
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.75
plt.rcParams['grid.alpha'] = 0.75

# flags to determine if plots are made and figures saved
PLT = False
SAVE_FIGS = True
figure_dir = 'figures/'
figure_dir_tocopy = 'figures_copy/'

print('-' * 50)

leg_str = {'mode': 'mode',
           'cm': 'CoM',
           'gfit': 'Gauss LSQ',
           'gfit_wt': 'Wt. Gauss',
           'mask_fit': 'mask CoM'}


def load_file(filename):
    """loads pickled results of Monte Carlo sims
    Args:
        filename: the file name

    Returns:
        pandas data frame of the results
        dictionary of the simulation configuration
    """

    with open('sims/{}'.format(filename), 'rb') as f:  # Python 3: open(..., 'wb')
        p_l = pkl.load(f)
    df = p_l[0]
    cfg = p_l[1]
    return df, cfg

# --------
#   Each fit method, vs. photon number and for each bin size at constant tdc RMS
# --------


filename = 'sim_res_temp_08_31_2020__08_11_05_50.pkl'  # (1e-15 x0_diff)
df, cfg = load_file(filename)

xscale = [40, 1e4]
yscale = [0.9, 50]

legend_label = {
    'cm': 'CoM',
    'gfit_subtract': 'Gauss fit',
    'mask_fit': 'mask CoM'}
marker = itertools.cycle(('.', '+', 'x', '*'))  # reset marker

tdc_std_unq = np.sort(df.tdc_std.unique())
for tdc_std_val in tdc_std_unq:
    fig, ax = plt.subplots()
#    for method in ['mode', 'cm', 'gfit', 'mask_fit', 'gfit_subtract']:
    for method in ['gfit_subtract', 'mask_fit', 'cm']:
        for b in np.sort(df.tdc_bins.unique()):
            summary = np.array([])

            dfb = df.loc[(df['tdc_bins'] == b) & (
                df['tdc_std'] == tdc_std_val)]
            N_unq = dfb.N.unique()
            for v in N_unq:
                t_meas = dfb.loc[dfb['N'] == v][method]
                dly = dfb.loc[dfb['N'] == v]['delay']
                noise = np.std(t_meas - dly)
                summary = np.append(summary, noise)

            ax.loglog(N_unq, summary * 1e12, marker=next(marker), linestyle='None',
                      label='MC: {}'.format(legend_label[method]))
    vs_signal(ax)
    ax.set_xlim(xscale)
    ax.set_ylim(yscale)
    ax.set_ylabel(r'$\delta$ [ps]')
    ax.set_xlabel('N [Signal photons]')
    ax.legend(prop={'size': 11}, framealpha=1)
    plt.grid(True)

    if SAVE_FIGS:
        figname = 'vs_photons_forbinsz_tdcrms{}'.format(
            int(tdc_std_val * 1e12))
        for e in ['.png', '.eps']:
            fig.savefig(os.path.join(figure_dir,
                                     figname + e))
            if figure_dir_tocopy is not None:
                shutil.copy2(
                    os.path.join(
                        figure_dir,
                        figname + e),
                    figure_dir_tocopy)

# --------
#   sweep background
# --------
marker = itertools.cycle(('.', '+', 'x', '*'))  # reset marker
filename = 'sim_res_temp_08_31_2020__08_21_12_347.pkl'
df, cfg = load_file(filename)
cfg = [cfg]
xscale = [1, 1e4]
yscale = [2, 40]

tdc_std_unq = np.sort(df.tdc_std.unique())

for tdc_std_val in tdc_std_unq:
    # for method in ['mode', 'cm', 'gfit', 'mask_fit', 'gfit_subtract']:
    fig, ax = plt.subplots()
    for method in ['gfit_subtract']:
        for b in np.sort(df.tdc_bins.unique()):
            summary = np.array([])

            dfb = df.loc[(df['tdc_bins'] == b) & (
                df['tdc_std'] == tdc_std_val)]
            N_unq = dfb.cps.unique()
            for v in N_unq:
                t_meas = dfb.loc[dfb['cps'] == v][method]
                dly = dfb.loc[dfb['cps'] == v]['delay']
                noise = np.std(t_meas - dly)
                summary = np.append(summary, noise)

            ax.loglog(N_unq * cfg[-1]['frame_time'] / cfg[-1]['tdc_bins'],
                      summary * 1e12, marker=next(marker), fillstyle='none', linestyle='None',
                      label='MC: {}'.format(legend_label[method]))

    # overlay analytical data
    vs_background(ax)

    # decorate the plot
    ax.set_xlim(xscale)
    ax.set_ylim(yscale)
    ax.set_ylabel(r'$\delta$ [ps]')
    ax.set_xlabel('b [Background photons / bin]')
    ax.legend(prop={'size': 11}, framealpha=1)
    plt.grid(True)

    if SAVE_FIGS:
        figname = 'vs_background_tdcrms{}'.format(int(tdc_std_val * 1e12))
        for e in ['.png', '.eps']:
            fig.savefig(os.path.join(figure_dir,
                                     figname + e))
            if figure_dir_tocopy is not None:
                shutil.copy2(
                    os.path.join(
                        figure_dir,
                        figname + e),
                    figure_dir_tocopy)

# --------
#   Each fit method, vs. background number and for each tdc RMS at constant bin size
# --------

filename = 'sim_res_temp_08_28_2020__21_55_58_100.pkl'
df, cfg = load_file(filename)

xscale = [10, 350000]
yscale = [0.1, 10000]

tdc_std_unq = np.sort(df.tdc_std.unique())
for tdc_std_val in tdc_std_unq:
    for method in ['mode', 'cm', 'gfit', 'mask_fit']:
        fig, ax = plt.subplots()
        for b in np.sort(df.tdc_bins.unique()):
            summary = np.array([])

            dfb = df.loc[(df['tdc_bins'] == b) & (
                df['tdc_std'] == tdc_std_val)]
            cps_unq = dfb.cps.unique()
            for v in cps_unq:
                t_meas = dfb.loc[dfb['cps'] == v][method]
                dly = dfb.loc[dfb['cps'] == v]['delay']
                noise = np.std(t_meas - dly)
                summary = np.append(summary, noise)

            ax.loglog(cps_unq * cfg[-1]['frame_time'] / cfg[-1]['tdc_bins'], 
                summary * 1e12, marker='*',
                label='Width = {:.0f} ps'.format(np.unique(dfb['tdc_width'])[0] * 1e12))

        # decorate the plot
        ax.set_xlim(xscale)
        ax.set_ylim(yscale)
        ax.set_ylabel(r'$\delta$ [ps]')
        ax.set_xlabel('N [Signal photons]')
        ax.legend(prop={'size': 11}, framealpha=1)
        plt.grid(True)
        plt.title(method)

        if SAVE_FIGS:
            for e in ['.png', '.eps']:
                fig.savefig(os.path.join(figure_dir,
                                         'vs_background'.format(method, int(tdc_std_val * 1e12)) + e))

# --------
#   Each fit method, vs. photon number and for each tdc RMS at constant bin size
# --------

tdc_width_unq = np.sort(df.tdc_width.unique())
for tdc_width_val in tdc_width_unq:
    for method in ['mode', 'cm', 'gfit', 'mask_fit']:
        fig, ax = plt.subplots()
        for tdc_std in np.sort(df.tdc_std.unique()):
            summary = np.array([])

            dfb = df.loc[(df['tdc_width'] == tdc_width_val)
                         & (df['tdc_std'] == tdc_std)]
            N_unq = dfb.N.unique()
            bin_width = dfb.tdc_width.unique()
            for v in N_unq:
                cm = dfb.loc[dfb['N'] == v][method]
                dly = dfb.loc[dfb['N'] == v]['delay']
                noise = np.std(cm - dly)
                summary = np.append(summary, noise)
            ax.loglog(N_unq, summary * 1e12, marker='*',
                      label=r'$\sigma = {0:3.1f} \; ps$'.format(tdc_std * 1e12))
        ax.set_xlim(xscale)
        ax.set_ylim(yscale)
        ax.set_ylabel(r'$\delta$ [ps]')
        ax.set_xlabel('N [Signal photons]')
        ax.legend(prop={'size': 11}, framealpha=1)
        plt.grid(True)
        # plt.title(method + ' bin-width = {:.1f} [ps]'.format(bin_width[0]*1e12))

        # fig.tight_layout()
        if SAVE_FIGS:
            for e in ['.png', '.eps']:
                fig.savefig(os.path.join(figure_dir,
                                         'vs_photons_forbinsz_mode_{}_tdcbin_{}'.format(method, int(tdc_width_val * 1e12)) + e))
        # plt.close('all')


# --------
#   Each fit method, vs. IRF at constant photon number and bin size
# --------

# constant photon number sweep IRF
filename = 'sim_res_08_14_2020__16_54_02.pkl'
df, cfg = load_file(filename)

tdc_std_unq = np.sort(df.tdc_std.unique())
N_unq = df.N.unique()
tdc_bin_unq = np.sort(df.tdc_width.unique())

for N in N_unq:
    for method in ['mode', 'cm', 'gfit', 'mask_fit']:
        fig, ax = plt.subplots()
        for b in np.sort(df.tdc_bins.unique()):
            summary = np.array([])
            dfb = df.loc[(df['tdc_bins'] == b) & (df['N'] == N)]
            tdc_std_unq = dfb.tdc_std.unique()
            for v in tdc_std_unq:
                t_meas = dfb.loc[dfb['tdc_std'] == v][method]
                dly = dfb.loc[dfb['tdc_std'] == v]['delay']
                noise = np.std(t_meas - dly)
                summary = np.append(summary, noise)

            ax.loglog(tdc_std_unq * 1e12, summary * 1e12, marker='*',
                      label='Width = {:.0f} ps'.format(np.unique(dfb['tdc_width'])[0] * 1e12))

        # decorate the plot
        # ax.set_xlim(xscale)
        # ax.set_ylim(yscale)
        ax.set_ylabel(r'$\delta$ [ps]')
        ax.set_xlabel('TDC IRF [ps]')
        ax.legend(prop={'size': 11}, framealpha=1)
        plt.grid(True)
        # plt.title(method + ' tdc-std = {:.1f} ps'.format(tdc_std_unq[2]*1e12))

        if SAVE_FIGS:
            for e in ['.png', '.eps']:
                fig.savefig(os.path.join(figure_dir,
                                         'vs_tdcirf_forbinsz_mode_{}_Nphotons_{}'.format(method, int(N)) + e))
        # plt.close('all')


method = 'gfit'
tdc_std_sort = np.sort(df.tdc_std.unique())
dfb = df.loc[(df['tdc_bins'] == 64) & (df['tdc_std'] == tdc_std_sort[1])]

fig, axs = plt.subplots(1, 2)
summary = np.array([])
bins_seq = np.linspace(-2e-8, 2e-8, 100)

N_unq = dfb.N.unique()[0:15]
for v in N_unq:
    cm = dfb.loc[dfb['N'] == v][method]
    dly = dfb.loc[dfb['N'] == v]['delay']

    timing_values = dfb.loc[dfb['N'] == v]['mode'] - dly
    print('Mode: N = {:.1f}; max_val = {:.3e}, min_val = {:.3e}, avg = {:.3e}, std = {:.3e}'.format(v,
                                                                                                    np.max(timing_values), np.min(
                                                                                                        timing_values), np.mean(timing_values),
                                                                                                    np.std(timing_values)))

    timing_values = cm - dly
    print('      N = {:.1f}; max_val = {:.3e}, min_val = {:.3e}, avg = {:.3e}, std = {:.3e}'.format(v,
                                                                                                    np.max(timing_values), np.min(
                                                                                                        timing_values), np.mean(timing_values),
                                                                                                    np.std(timing_values)))
    print('-' * 50)
    noise = np.std(cm - dly)
    [hist, bin_edges] = np.histogram(timing_values, bins=bins_seq)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axs[0].plot(bin_centers, hist, marker='*', label='N={:.3f}'.format(v))
    axs[1].plot(v, noise, marker='*')

axs[0].legend()
# plt.title(method)


# --------
#   Each fit method, vs. bin size at constant photon count (vary IRF in legend)
# --------

filename = 'sim_res_temp_09_01_2020__23_14_19_15.pkl'
df, cfg = load_file(filename)

xscale = [0, 8.5]
yscale = [5, 300]
tdc_bins_unq = np.sort(df.tdc_bins.unique())
cps_unq = np.sort(df.cps.unique())[1:]
irf_unq = np.sort(df.tdc_std.unique())

irf_unq = [100e-12]

for irf in irf_unq:
    for tdc_num in tdc_bins_unq:
        print('Num of bins: {}, bck per bin = {}'.format(tdc_num,
                                                         cfg[-1]['cps'] * cfg[-1]['frame_time'] / tdc_num))
    N_unq = df.N.unique()
    for N in N_unq:
        # for method in ['mode', 'cm', 'gfit', 'mask_fit', 'gfit_subtract']:
        for method in ['cm', 'gfit_subtract']:

            fig, ax = plt.subplots()
            plot_colors = []

            for b in cps_unq:
                summary = np.array([])
                dfb = df.loc[(df['tdc_std'] == irf) & (
                    df['N'] == N) & (df['cps'] == b)]
                tdc_width_unq = dfb.tdc_width.unique()
                for v in tdc_width_unq:
                    t_meas = dfb.loc[dfb['tdc_width'] == v][method]
                    dly = dfb.loc[dfb['tdc_width'] == v]['delay']
                    noise = np.std(t_meas - dly)
                    summary = np.append(summary, noise)

                p = ax.semilogy(tdc_width_unq / irf, summary * 1e12, 
                    marker=next(marker), linestyle='None',
                    label='N/b = {:.2f}'.format(N / (b * cfg[-1]['frame_time'] / 176)))
                plot_colors.append(p[0].get_color())

            # add in analytical plots
            vary_bin_size(ax, itertools.cycle(plot_colors))

            ax.set_xlim(xscale)
            ax.set_ylim(yscale)
            ax.set_ylabel(r'$\delta$ [ps]')
            ax.set_xlabel(r'$a/\sigma$')
            ax.legend(prop={'size': 11}, framealpha=1)
            plt.grid(True)
            # plt.title(method + ' N = {}, IRF = {:.1f} [ps] '.format(N, irf*1e12))

            if SAVE_FIGS:
                figname = 'vs_tdcbins_forstd_method_{}_irf_{}'.format(
                    method, int(irf * 1e12))
                for e in ['.png', '.eps']:
                    fig.savefig(os.path.join(figure_dir,
                                             figname + e))
                    if figure_dir_tocopy is not None:
                        shutil.copy2(
                            os.path.join(
                                figure_dir,
                                figname + e),
                            figure_dir_tocopy)


method = 'gfit'
tdc_std_sort = np.sort(df.tdc_std.unique())
dfb = df.loc[(df['tdc_bins'] == 64) & (df['tdc_std'] == tdc_std_sort[1])]

fig, axs = plt.subplots(1, 2)
summary = np.array([])
bins_seq = np.linspace(-2e-8, 2e-8, 100)

N_unq = dfb.N.unique()[0:15]
for v in N_unq:
    cm = dfb.loc[dfb['N'] == v][method]
    dly = dfb.loc[dfb['N'] == v]['delay']

    timing_values = dfb.loc[dfb['N'] == v]['mode'] - dly
    print('Mode: N = {:.1f}; max_val = {:.3e}, min_val = {:.3e}, avg = {:.3e}, std = {:.3e}'.format(v,
                                                                                                    np.max(timing_values), np.min(
                                                                                                        timing_values), np.mean(timing_values),
                                                                                                    np.std(timing_values)))

    timing_values = cm - dly
    print('      N = {:.1f}; max_val = {:.3e}, min_val = {:.3e}, avg = {:.3e}, std = {:.3e}'.format(v,
                                                                                                    np.max(timing_values), np.min(
                                                                                                        timing_values), np.mean(timing_values),
                                                                                                    np.std(timing_values)))
    print('-' * 50)
    noise = np.std(cm - dly)
    [hist, bin_edges] = np.histogram(timing_values, bins=bins_seq)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axs[0].plot(bin_centers, hist, marker='*', label='N={:.3f}'.format(v))
    axs[1].plot(v, noise, marker='*')

axs[0].legend()
