"""
Lucas J. Koerner, koerner.lucas@stthomas.edu
Aug, 2020

Create plots of analytical functions of ToF noise
Uses winick_sweep and thompson_arr from analytical_calcs

"""
import os
import shutil
import itertools
from datetime import datetime  # Current date time in local system
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from analytical_calcs import winick_sweep, thompson_arr


d = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
SAVE_FIGS = True
figure_dir = 'figures/'
figure_dir_tocopy = 'figures_copy/'

line_styles = itertools.cycle(('-', '-.', '--', ':'))


def vary_bin_size(ax=None, plot_colors=None):
    """
    Vary SNR, vary bin size using just MLE
    scale background by bin size

    Args:
        ax: an axes to plot onto, if None a figure / axes is created
        plot_colors: specify the color of the plot

    Returns:
        None (but saves figures)
    """

    sigma = 100e-12
    N = 300
    b = 10  # at what bin size?
    bin_width = np.linspace(0.05 * sigma, 10 * sigma, 40)
    if ax is None:
        fig, ax = plt.subplots()

    for b in [N / 200, N / 20, N / 2, N * 2]:

        prec_thomp_vpix = thompson_arr(
            sigma, bin_width, N, b / sigma * bin_width)
        b = b / 100e-12
        rms_error_arr = winick_sweep(bin_width, N, b, sigma,
                                     noise_scaled=True)

        if plot_colors is None:
            ax.semilogy(bin_width / sigma, rms_error_arr * 1e12, linestyle='-',
                        label='N/b = {:.1f}'.format(N / (b * 100e-12)))
        else:
            ax.semilogy(bin_width / sigma, rms_error_arr * 1e12,
                        linestyle='-', color=next(plot_colors))
            # label = 'N/b = {:.1f}'.format(N/(b*100e-12)))

    ax.semilogy(bin_width / sigma, bin_width / np.sqrt(12) * 1e12,
                linestyle='--', label='qtz. limit')

    if ax is None:
        ax.set_ylabel(r'$\delta$ [ps]')
        ax.set_xlabel(r'(TDC res.)/$\sigma$')
        ax.set_ylim([4, 600])
        ax.legend(prop={'size': 11}, framealpha=1)
        plt.grid(True)
        fig.tight_layout()

        if SAVE_FIGS:
            figname = 'vs_pix_size_SNRvary_scalednoise'
            for e in ['.png', '.eps']:
                fig.savefig(os.path.join(figure_dir,
                                         figname + e))
                if figure_dir_tocopy is not None:
                    shutil.copy2(os.path.join(
                        figure_dir, figname + e), figure_dir_tocopy)


def vary_irf(ax=None, plot_colors=None):
    """
    Vary IRF at a few bin sizes

    Args:
        ax: an axes to plot onto, if None a figure / axes is created
        plot_colors: specify the color of the plot

    Returns:
        None (but saves figures)
    """

    sigma = 100e-12
    a = 150e-12
    N = 300
    bin_arr = [100e-12, 200e-12, 400e-12, 800e-12]
    b = 50
    sigma_arr = np.linspace(0.5 * sigma, 8 * sigma, 200)
    if ax is None:
        fig, ax = plt.subplots()
        decorate = True
    else:
        decorate = False

    for a in bin_arr:

        prec_thomp_vpix = thompson_arr(
            sigma_arr, a, N * (sigma_arr / np.min(sigma_arr)), b / np.min(bin_arr) * a)

        # sweep two variables with Winick expression (MLE)
        rms_error_arr = np.array([])
        for nw, sw in zip(N * (sigma_arr / np.min(sigma_arr)), sigma_arr):
            rms_error = winick_sweep(a, nw, b / np.min(bin_arr) * a, [sw],
                                     noise_scaled=False)
            rms_error_arr = np.append(rms_error_arr, rms_error)

        if plot_colors is None:
            ax.semilogy(sigma_arr * 1e12, prec_thomp_vpix * 1e12, linestyle=next(line_styles),
                        label='TDC res. = {} [ps]'.format(int(a * 1e12)))
            ax.semilogy(sigma_arr * 1e12, rms_error_arr * 1e12, linestyle=next(line_styles),
                        label='CRB. TDC res. = {} [ps]'.format(int(a * 1e12)))
        else:
            ax.semilogy(sigma_arr * 1e12, prec_thomp_vpix * 1e12, linestyle=next(line_styles),
                        color=next(plot_colors),
                        label='TDC res. = {} [ps]'.format(int(a * 1e12)))

    if decorate:
        ax.set_ylabel(r'$\delta$ [ps]')
        ax.set_xlabel(r'$\sigma$ [ps]')
        ax.set_ylim([4, 300])
        ax.set_xlim([50, 800])
        ax.legend(prop={'size': 11}, framealpha=1)
        plt.grid(True)
        fig.tight_layout()

        if SAVE_FIGS:
            figname = 'vs_IRF_fewbinsizes'
            for e in ['.png', '.eps']:
                fig.savefig(os.path.join(figure_dir,
                                         figname + e))
                if figure_dir_tocopy is not None:
                    shutil.copy2(os.path.join(
                        figure_dir, figname + e), figure_dir_tocopy)


def vary_irf_laser_casestudy(ax=None, plot_colors=None):
    """
    Vary IRF at a single bin size

    Args:
        ax: an axes to plot onto, if None a figure / axes is created
        plot_colors: specify the color of the plot

    Returns:
        None (but saves figures)
    """

    sigma = 100e-12
    N = 300
    bin_arr = [150e-12]
    b = 50
    sigma_arr = np.linspace(1 * sigma, 8 * sigma, 200)
    if ax is None:
        fig, ax = plt.subplots()
        decorate = True
    else:
        decorate = False

    for a in bin_arr:
        # sweep two variables with Winick
        prec_thomp_vpix = thompson_arr(
            sigma_arr, a, N * (sigma_arr / np.min(sigma_arr))**2, b / np.min(bin_arr) * a)
        rms_error_arr = np.array([])
        for nw, sw in zip(N * (sigma_arr / np.min(sigma_arr)), sigma_arr):
            rms_error = winick_sweep(a, nw, b / np.min(bin_arr) * a, [sw],
                                     noise_scaled=False)
            rms_error_arr = np.append(rms_error_arr, rms_error)
        if plot_colors is None:
            ax.semilogy(sigma_arr * 1e12, rms_error_arr * 1e12, linestyle=next(line_styles),
                        label='N $\propto \sigma$'.format(int(a * 1e12)))
        else:
            ax.semilogy(sigma_arr * 1e12, rms_error_arr * 1e12, linestyle=next(line_styles),
                        color=next(plot_colors),
                        label='TDC res. = {} [ps]'.format(int(a * 1e12)))

    for a in bin_arr:
        # sweep two variables with Winick
        rms_error_arr = winick_sweep(a, N, b / np.min(bin_arr) * a, sigma_arr,
                                     noise_scaled=False)
        if plot_colors is None:
            ax.semilogy(sigma_arr * 1e12, rms_error_arr * 1e12, linestyle=next(line_styles),
                        label='N = {}'.format(int(N)))
        else:
            ax.semilogy(sigma_arr * 1e12, rms_error_arr * 1e12, linestyle=next(line_styles),
                        color=next(plot_colors),
                        label='N = {}'.format(int(N)))

    if decorate:
        ax.set_ylabel(r'$\delta$ [ps]')
        ax.set_xlabel(r'$\sigma$ [ps]')
        ax.set_ylim([4, 300])
        ax.set_xlim([50, 800])
        ax.legend(prop={'size': 11}, framealpha=1)
        plt.grid(True)
        fig.tight_layout()

        if SAVE_FIGS:
            figname = 'vs_IRF_fewbinsizes_scaleN'
            for e in ['.png', '.eps']:
                fig.savefig(os.path.join(figure_dir,
                                         figname + e))
                if figure_dir_tocopy is not None:
                    shutil.copy2(os.path.join(
                        figure_dir, figname + e), figure_dir_tocopy)


def vs_signal(ax=None):
    """
    Vary the number of detected photons (N)

    Args:
        ax: an axes to plot onto, if None a figure / axes is created

    Returns:
        None (but saves figures)
    """

    sigma = 100e-12
    N = np.logspace(1.7, 4, 30)
    bin_width = 1.5 * sigma
    b = 18.75  # background
    prec_thomp = thompson_arr(sigma, bin_width, N, b)

    fund_limit = sigma / np.sqrt(N)

    rms_error_arr = winick_sweep(bin_width, N, b, sigma,
                                 noise_scaled=False)

    if ax is None:
        fig, ax = plt.subplots()

    ax.loglog(N, np.asarray(prec_thomp) * 1e12, marker='None', linestyle='-',
              label='Thompson')
    ax.loglog(N, rms_error_arr * 1e12, marker='None', linestyle='-.',
              label='CRB')
    ax.loglog(N, fund_limit * 1e12, marker='None', linestyle='--',
              label='Fund. Limit')

    if ax is None:
        ax.set_ylabel(r'$\delta$ [ps]')
        ax.set_xlabel('Signal Photons')
        ax.set_ylim([0.8, 100])
        ax.set_xlim([40, 1e4])
        ax.legend(prop={'size': 11}, framealpha=1)
        plt.grid(True)
        fig.tight_layout()

        if SAVE_FIGS:
            figname = 'Fund_Thompson_CRLB_vs_signalcount'
            for e in ['.png', '.eps']:
                fig.savefig(os.path.join(figure_dir,
                                         figname + e))
                if figure_dir_tocopy is not None:
                    shutil.copy2(os.path.join(
                        figure_dir, figname + e), figure_dir_tocopy)


def vs_background(ax=None):
    """
    Vary the number of background photons (b)

    Args:
        ax: an axes to plot onto, if None a figure / axes is created

    Returns:
        None (but saves figures)
    """

    sigma = 100e-12
    N = 1000
    bin_width = 1.5 * sigma
    b = np.logspace(3, 9, 40) * 33e-3 / 176
    prec_thomp = thompson_arr(sigma, bin_width, N, b)
    rms_error_arr = winick_sweep(bin_width, N, b, sigma,
                                 noise_scaled=False)

    if ax is None:
        fig, ax = plt.subplots()

    ax.loglog(b, np.asarray(prec_thomp) * 1e12, linestyle='-',
              label='Thompson')
    ax.loglog(b, rms_error_arr * 1e12, linestyle='-.',
              label='CRB')

    if ax is None:
        fig.show()
        ax.set_ylabel(r'$\delta$ [ps]')
        ax.set_xlabel('Background Photons')
        ax.set_ylim([0.8, 1000])
        ax.set_xlim([40, 1e6])
        ax.legend(prop={'size': 11}, framealpha=1)
        plt.grid(True)
        fig.tight_layout()

        if SAVE_FIGS:
            figname = 'Thompson_CRLB_vsbackground'
            for e in ['.png', '.eps']:
                fig.savefig(os.path.join(figure_dir,
                                         figname + e))
                if figure_dir_tocopy is not None:
                    shutil.copy2(os.path.join(
                        figure_dir, figname + e), figure_dir_tocopy)


def vs_background_multiplesignal(ax=None):
    """
    Vary the number background (b) at a few levels of detected photons (N)

    Args:
        ax: an axes to plot onto, if None a figure / axes is created
            if an axes is provided it is not decorated

    Returns:
        None (but saves figures)
    """

    sigma = 100e-12
    N = 1000
    bin_width = 1.5 * sigma
    b = np.logspace(0, 6, 100)

    if ax is None:
        fig, ax = plt.subplots()
        decorate = True
    else:
        decorate = False

    line_styles = itertools.cycle(('-', '-.', '--', ':'))

    for N in [100, 1000, 10000]:
        rms_error_arr = winick_sweep(bin_width, N, b, sigma,
                                     noise_scaled=False)
        t = ax.loglog(b, rms_error_arr * 1e12, linestyle=next(line_styles),
                      label='N = {}'.format(N))

        b_sqrt2 = ((sigma**2 + bin_width**2 / 12) * bin_width * N) / \
            (4 * np.pi**(0.5) * sigma**3)
        print(b_sqrt2)
        ax.loglog(b_sqrt2, np.sqrt((sigma**2 + bin_width**2 / 12) / N) *
                  np.sqrt(2) * 1e12, marker='*', color=t[0].get_color())

    if decorate:  # decorate the plot
        fig.show()
        ax.set_ylabel(r'$\delta$ [ps]')
        ax.set_xlabel('b [background photons / bin]')
        ax.set_ylim([0.8, 100])
        ax.set_xlim([1, 1e4])
        ax.legend(prop={'size': 11}, framealpha=1)
        plt.grid(True)
        fig.tight_layout()

        if SAVE_FIGS:
            figname = 'CRB_vsbackground_multipleN'
            for e in ['.png', '.eps']:
                fig.savefig(os.path.join(figure_dir,
                                         figname + e))
                if figure_dir_tocopy is not None:
                    shutil.copy2(os.path.join(
                        figure_dir, figname + e), figure_dir_tocopy)


def vary_exposure(ax=None):
    """
    Vary the exposure time at a few SNR levels (b is a fraction of N)

    Args:
        ax: an axes to plot onto, if None a figure / axes is created
            if an axes is provided it is not decorated

    Returns:
        None (but saves figures)
    """

    sigma = 100e-12
    N = np.logspace(0, 6, 60)
    bin_width = 1.5 * sigma

    if ax is None:
        fig, ax = plt.subplots()

    for b_perc_N in [0.0001, 0.01, 0.1, 1, 100]:  # background

        prec_thomp = thompson_arr(sigma, bin_width, N, N * b_perc_N)
        rms_error_arr = winick_sweep(bin_width, N, b_perc_N, sigma,
                                     noise_scaled=True)

        ax.loglog(N, np.asarray(prec_thomp) * 1e12, marker='None', linestyle='-',
                  label='Thompson. SNR = {}'.format(1 / b_perc_N))
        ax.loglog(N, rms_error_arr * 1e12, marker='None', linestyle='-.',
                  label='CRB. SNR = {}'.format(1 / b_perc_N))

    if ax is None:
        ax.set_ylabel(r'$\delta$ [ps]')
        ax.set_xlabel('Signal Photons')
        ax.set_ylim([0.8, 100])
        ax.set_xlim([40, 1e4])
        ax.legend(prop={'size': 11}, framealpha=1)
        plt.grid(True)
        fig.tight_layout()

        if SAVE_FIGS:
            figname = 'Fund_Thompson_CRLB_vs_exposure_time'
            for e in ['.png', '.eps']:
                fig.savefig(os.path.join(figure_dir,
                                         figname + e))
                if figure_dir_tocopy is not None:
                    shutil.copy2(os.path.join(
                        figure_dir, figname + e), figure_dir_tocopy)


def vs_signal_regimes(ax=None):
    """
    Vary the number of signal photons and look at trend of noise for
    regimes of SNR

    Args:
        ax: an axes to plot onto, if None a figure / axes is created
            if an axes is provided it is not decorated

    Returns:
        the figure axes
    """

    sigma = 100e-12
    N = np.logspace(0, 4, 1000)
    bin_width = 2 * sigma
    b = 30  # background
    N_knee = (48 * np.pi**0.5 * sigma**3 /
              (bin_width * (12 * sigma**2 + bin_width**2))) * b
    print('Expecting a change in dependence at N = {}'.format(N_knee))
    prec_thomp = thompson_arr(sigma, bin_width, N, b)
    rms_error_arr = winick_sweep(bin_width, N, b, sigma,
                                 noise_scaled=False)

    prec_thomp_star = thompson_arr(sigma, bin_width, np.asarray([N_knee]), b)
    rms_error_arr_star = winick_sweep(bin_width, np.asarray([N_knee]), b, sigma,
                                      noise_scaled=False)

    if ax is None:
        fig, ax = plt.subplots()
        decorate = True
    else:
        decorate = False

    t = ax.loglog(N, np.asarray(prec_thomp) * 1e12, marker='None', linestyle='-',
                  label='Thompson')
    t = ax.loglog(N_knee, np.asarray(prec_thomp_star) * 1e12,
                  marker='*', linestyle='None', color=t[0].get_color())
    t = ax.loglog(N, rms_error_arr * 1e12, marker='None', linestyle='-.',
                  label='CRB')

    ax.axvline(N_knee, 0, 1, color='k', linestyle='--')
    ax.text(1.5, 20, 'Background \nlimited')
    ax.text(1400, 20, 'Signal \nlimited')

    idx = N < 10
    coeff = np.polyfit(
        np.log10(
            N[idx]), np.log10(
            rms_error_arr[idx] * 1e12), 1)
    print(coeff)
    ax.text(2.6, 600, 'm = {:.2f}'.format(coeff[0]))

    idx = N > 1000
    coeff = np.polyfit(
        np.log10(
            N[idx]), np.log10(
            rms_error_arr[idx] * 1e12), 1)
    print(coeff)
    ax.text(2e3, 3, 'm = {:.2f}'.format(coeff[0]))

    if decorate:
        ax.set_ylabel(r'$\delta$ [ps]')
        ax.set_xlabel('N [Signal photons]')
        ax.set_ylim([1, 1000])
        ax.set_xlim([1, 1e4])
        ax.legend(prop={'size': 11}, framealpha=1)
        plt.grid(True)
        fig.tight_layout()

        if SAVE_FIGS:
            figname = 'SNRregimes_vs_signalcount'
            for e in ['.png', '.eps']:
                fig.savefig(os.path.join(figure_dir,
                                         figname + e))
                if figure_dir_tocopy is not None:
                    shutil.copy2(os.path.join(
                        figure_dir, figname + e), figure_dir_tocopy)
    return ax


def thompson_crb_heatmap(DUAL=True, ax=None, plot_colors=None):
    """
    generate a heat map of percent deviation between Thompson estimation and CRB
        x is a/sigma
        y is N/b
        make axes logarithmic

    Args:
        DUAL: if TRUE make two heatmaps
        ax:   axes to plot onto
        plot_colors:

    Returns:
        the axes, the array of error by Thompson, array of errors by CRB, and the colorbar
    """

    bin_width = 150e-12
    b = 50

    if DUAL:
        sigma_arr = np.flip(np.logspace(
            np.log10(1 / 4), np.log10(4), 7) * bin_width)
        N_arr = np.flip(np.logspace(np.log10(0.5), 2, 10) * b)
    else:
        sigma_arr = np.flip(np.logspace(
            np.log10(1 / 4), np.log10(4), 10) * bin_width)
        N_arr = np.flip(np.logspace(np.log10(0.5), 2, 10) * b)

    if ax is None:
        fig, ax = plt.subplots()
        decorate = True
    else:
        decorate = False

    for N in N_arr:
        # sweep two variables with Winick
        prec_thomp_vpix = thompson_arr(sigma_arr, bin_width, N, b)
        try:
            thompson_err_arr = np.vstack((thompson_err_arr, prec_thomp_vpix))
        except BaseException:
            thompson_err_arr = prec_thomp_vpix

        crb_err = winick_sweep(bin_width, N, b, sigma_arr,
                               noise_scaled=False)
        try:
            crb_err_arr = np.vstack((crb_err_arr, crb_err))
        except BaseException:
            crb_err_arr = crb_err

    if DUAL:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.93, 6.27))
    else:
        fig, (ax2) = plt.subplots(1, 1, figsize=(6.27, 6.27))

    if DUAL:
        # image of the log of the error in picoseconds.
        # Need to scale the colorbar
        crb_err_ps = crb_err_arr * 1e12
        im = ax.imshow(crb_err_ps, cmap='gray',
                       norm=colors.LogNorm(vmin=crb_err_ps.min(), vmax=crb_err_ps.max()))

        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(r"CRB $\delta$ RMS ps", fontdict={'fontsize': 10})
        plt.show()

        for (j, i), label in np.ndenumerate(crb_err_ps):
            if (i % 2 == 0 and j % 2 == 0):
                label = int(np.round(label))
                ax.text(i, j, label, ha='center', va='center', color='green')
            # ax2.text(i,j,label,ha='center',va='center')

        # We want to show all ticks...
        ax.set_xticks(np.arange(0, len(sigma_arr), 2))
        ax.set_yticks(np.arange(0, len(N_arr), 2))
        # ... and label them with the respective list entries
        sigma_arr_str = ['{:.1f}'.format(bin_width / i)
                         for i in sigma_arr[::2]]
        ax.set_xticklabels(sigma_arr_str)

        N_arr_str = ['{:.1f}'.format(i / b) for i in N_arr[::2]]
        ax.set_yticklabels(N_arr_str)

        ax.set_xlabel(r'a/$\sigma$')
        ax.set_ylabel('N/b')

    # percent difference
    perc_diff = (crb_err_arr - thompson_err_arr) / crb_err_arr * 100
    im2 = ax2.imshow(perc_diff, cmap='gray')
    cbar = ax2.figure.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    plt.show()

    # ax2.set_title('Thompson, CRB % diff.', fontdict = {'fontsize' : 10})
    plt.show()

    for (j, i), label in np.ndenumerate(perc_diff):
        if (i % 2 == 0 and j % 2 == 0):
            label = int(np.round(label))
            if label < 20:
                ax2.text(i, j, label, ha='center', va='center', color='green')
            else:
                ax2.text(i, j, label, ha='center', va='center', color='black')

    # We want to show all ticks...
    ax2.set_xticks(np.arange(0, len(sigma_arr), 2))
    ax2.set_yticks(np.arange(0, len(N_arr), 2))
    # ... and label them with the respective list entries
    sigma_arr_str = ['{:.1f}'.format(bin_width / i) for i in sigma_arr[::2]]
    ax2.set_xticklabels(sigma_arr_str)

    N_arr_str = ['{:.1f}'.format(i / b) for i in N_arr[::2]]
    ax2.set_yticklabels(N_arr_str)

    ax2.set_xlabel(r'a/$\sigma$')
    ax2.set_ylabel('N/b')

    if 0:
        ax.set_ylabel(r'$\delta$ [ps]')
        ax.set_xlabel(r'$\sigma$ [ps]')
        ax.set_ylim([4, 300])
        ax.set_xlim([50, 800])
        ax.legend(prop={'size': 11}, framealpha=1)
        plt.grid(True)
    fig.tight_layout()

    SAVE_FIGS = True
    if SAVE_FIGS:
        if DUAL:
            figname = 'heatmap_diff'
        else:
            figname = 'heatmap_diff_only'
        for e in ['.png', '.eps']:
            fig.savefig(os.path.join(figure_dir,
                                     figname + e))
            if figure_dir_tocopy is not None:
                shutil.copy2(os.path.join(
                    figure_dir, figname + e), figure_dir_tocopy)

    return ax, thompson_err_arr, crb_err_arr, cbar


def optimal_a(N=300, b=10, ax=None):
    """
    search for the optimal bin size

    Args:
        N: number of signal photons
        b: number of background photons
        ax:   axes to plot onto

    Returns:
        the axes
    """

    sigma = 100e-12
    bin_normalize = sigma
    bin_width = np.linspace(20e-12, 2000e-12, 500)
    print('SNR at a = sigma: {:.2f}'.format(N / (b / bin_normalize * sigma)))

    prec_thomp = thompson_arr(
        sigma,
        bin_width,
        N,
        b /
        bin_normalize *
        bin_width)
    crb_err = winick_sweep(bin_width, N, b / bin_normalize, sigma,
                           noise_scaled=True)

    if ax is None:
        fig, ax = plt.subplots()

    t = ax.loglog(bin_width, np.asarray(prec_thomp) * 1e12, marker='None', linestyle='-',
                  label='Thompson')
    t = ax.loglog(bin_width, np.asarray(crb_err) * 1e12, marker='None', linestyle='-',
                  label='CRB')

    for increase in [0.1, 0.2, 0.41, 0.5]:
        idx = np.argmin(np.abs(prec_thomp - (1 + increase) * prec_thomp[0]))
        print('{} \% increase a/sigma {:.2f}'.format(np.round(increase *
                                                              100), bin_width[idx] / sigma))

    print('CRB' + '---' * 40)

    for increase in [0.1, 0.2, 0.41, 0.5]:
        idx = np.argmin(np.abs(crb_err - (1 + increase) * crb_err[0]))
        print('{} \% increase a/sigma {:.2f}'.format(np.round(increase *
                                                              100), bin_width[idx] / sigma))

    ax.legend()
    ax.set_ylim([1, 50])
    return ax


if __name__ == "__main__":

    print('Replicate Winick 1986 Fig. 1 (CRB)')
    lms = 1  # average number (not a rate)
    sigma = 1

    fig, ax = plt.subplots()
    for lmn in [0.1, 1, 10]:  # vary the background

        pix_size_arr = np.linspace(0.05 * sigma, 10 * sigma, 100)
        rms_error_arr = winick_sweep(pix_size_arr, lms, lmn, sigma,
                                     noise_scaled=False)

        ax.semilogy(pix_size_arr / sigma, rms_error_arr / sigma**0.5,
                    linestyle='None', marker='*',
                    label='s/b = {:.1f}'.format(lms / lmn))
        ax.set_ylim([1, 1e3])
        ax.set_xlim([0, 10])
        ax.set_xlabel(r'Bin size/$\sigma$')
        ax.set_ylabel('Normalized RMS error')
        ax.legend(framealpha=1)

    print('Winick 1986 method with real values (CRB)')
    lms = 100  # average number of signal photons (not a rate)
    sigma = 100e-12

    fig, ax = plt.subplots()
    for lmn in [0.001 * lms, 0.1 *
                lms, 1 * lms, 10 * lms]:  # vary background photons
        pix_size_arr = np.linspace(0.05 * sigma, 10 * sigma, 100)
        rms_error_arr = winick_sweep(pix_size_arr, lms, lmn, sigma,
                                     noise_scaled=False)  # noise is constant per bin (independent of bin size)

        ax.semilogy(pix_size_arr / sigma, rms_error_arr * 1e12,
                    linestyle='None', marker='*',
                    label='s/b = {:.1f}'.format(lms / lmn))
        ax.set_ylim([5, 5000])
        ax.set_xlim([0, 10])
        ax.set_xlabel(r'Bin size/$\sigma$')
        ax.set_ylabel(r'$\delta$ [ps]')
        ax.legend(framealpha=1)

    # -----------------------------------------
    # Regimes of SNR
    # -----------------------------------------
    vs_signal_regimes()

    # -----------------------------------------
    # Thompson percent difference
    # -----------------------------------------
    thompson_crb_heatmap(DUAL=False)
