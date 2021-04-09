"""
Lucas J. Koerner, koer2434@stthomas.edu
Aug, 2020

Goals:
1) Analytically estimate precision limits 
2) create contained functions that can be used in other modules 
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.integrate import quad

np.random.seed(123)

# Thompson 
def prec_thompson(sigma, bin_width, N, b):
    '''
    # Thompson, 2002, eqn. 14

    inputs: 
        sigma: instrument response function (Gaussian sigma)
        bin_width: width of the TDC bin (in seconds)
        N: total number of photons 
        b: background photons per bin (note that in Thompson b is the background noise)
    

    '''
    return np.sqrt((sigma**2 + bin_width**2/12)/N + \
        4*np.sqrt(np.pi)*sigma**3*b/(bin_width*N**2))

def thompson_arr(sigma, bin_width, N, b):
    ''' 
    inputs: either scalar or array
                all arrays must be the same size 
                expected use is a single parameter that is an array
    '''
    # get lengths
    ls = len(sigma) if hasattr(sigma, "__len__") else 1
    lbw = len(bin_width) if hasattr(bin_width, "__len__") else 1
    ln = len(N) if hasattr(N, "__len__") else 1
    lbck = len(b) if hasattr(b, "__len__") else 1

    lvals = np.unique(np.sort([ls, lbw, ln, lbck]))

    if (ls == lbw == ln == lbck == 1):
        return np.asarray(list(map(prec_thompson, [sigma], [bin_width], [N], [b])))
    elif (lvals[0] == 1) and (len(lvals) == 2):
        arr_len = lvals[1]
        s_in = [sigma]*arr_len if ls == 1 else sigma
        bw_in = [bin_width]*arr_len if lbw == 1 else bin_width
        n_in = [N]*arr_len if ln == 1 else N
        bck_in = [b]*arr_len if lbck == 1 else b
        
        return np.asarray(list(map(prec_thompson, s_in, bw_in, n_in, bck_in)))

    else: 
        print('Thompson arr: all array parameters must be the same length') 
        return None

    return np.asarray(list(map(prec_thompson, [sigma]*len(N), [bin_width]*len(N), N, [b]*len(N))))

'''
Winick 1986: 
 note that the methodology in Winick assigns a background rate per pixel 
   so in effect the background goes up as the pixels shrink... 
   not true in our work (background is constant per picosecond not per bin)

Wolfram: differentiate (integrate 1/(sqrt(2*pi)*sigma)*e^(-(x-eps)^2/(2*sigma^2)) dx from x1 to x2) wrt eps
Wolfram: (integrate 1/(sqrt(2*pi)*sigma)*e^(-(x-eps)^2/(2*sigma^2)) dx from x1 to x2) 
'''

def winick_onepix(eps_x, xi, dx, lms, lmn, sigma):
    # must sum over pixels and then average over eps_x
    gi_p = 1/(np.sqrt(2*np.pi)*sigma)*(np.exp(-(xi+dx/2-eps_x)**2/(2*sigma**2)) - np.exp(-(xi-dx/2-eps_x)**2/(2*sigma**2)))
    gi = 1/2*(erf((xi+dx/2-eps_x)/(np.sqrt(2)*sigma)) - erf((xi-dx/2-eps_x)/(np.sqrt(2)*sigma)))
    return (lms*gi_p)**2/(lms*gi + lmn)

def sum_pixels(eps_x, dx, lms, lmn, sigma):
    sum_res = 0

    # -2*dx, +2*dx ensures a minimum of 5 pixels (the extent doesn't significantly impact results)
    pix_arr = np.arange(int(-sigma*5/dx)*dx - 2*dx, (int(sigma*5/dx)+1)*dx + 2*dx, dx )

    for p in pix_arr:
        sum_res += winick_onepix(eps_x, p, dx, lms, lmn, sigma)
    return sum_res 

def winick_sweep(dx, lms, lmn, sigma,
    noise_scaled = True, integrate_steps = 100):
    '''
    Calculate RMS noise per Winick 1986 equation 36
    inputs:
        dx: bin size
        lms: average signal 
        lmn: average background 
        sigma: instrument response function 
        noise_scaled: if True the noise is scaled by the bin size. 
                so input should be in units of background/second
        integrate_steps: number of steps when averaging over bin
    returns:
        array of rms errors 
    '''
    # figure out which is an array and sweep over it 
    swp_cnt = 0
    if isinstance(dx, (np.ndarray, list)):
        swp_cnt += 1
        swp_var = 'dx'
        swp_arr = dx
    if isinstance(lms, (np.ndarray, list)):
        swp_cnt += 1
        swp_var = 'lms'
        swp_arr = lms
    if isinstance(lmn, (np.ndarray, list)):
        swp_cnt += 1
        swp_var = 'lmn'
        swp_arr = lmn
    if isinstance(sigma, (np.ndarray, list)):
        swp_cnt += 1
        swp_var = 'sigma'
        swp_arr = sigma

    if (swp_cnt > 1):
        print('Error in winick_sweep: only one input parameter may be an array')
        return 
    elif (swp_cnt == 0):
        print('Error in winick_sweep: one input parameter must be an array')
        return 

    rms_error_arr = np.array([])

    for swp in swp_arr: # 
        if noise_scaled and ((swp_var == 'dx') or (swp_var == 'lms')): 
            lmn_scaled = lmn*swp
        else:
            lmn_scaled = lmn

        # average over eps_x
        sum_arr = np.array([])

        if swp_var == 'dx':
            dx_tmp = swp
        else:
            dx_tmp = dx
        step = dx_tmp/integrate_steps
        for eps_x in np.arange(-dx_tmp/2, dx_tmp/2, step):

            if swp_var == 'dx':
                sum_arr = np.append(sum_arr, 
                    1/sum_pixels(eps_x, swp, lms, lmn_scaled, sigma))
            if swp_var == 'lms':
                sum_arr = np.append(sum_arr, 
                    1/sum_pixels(eps_x, dx, swp, lmn_scaled, sigma))
            if swp_var == 'lmn':
                sum_arr = np.append(sum_arr, 
                    1/sum_pixels(eps_x, dx, lms, swp, sigma))
            if swp_var == 'sigma':
                sum_arr = np.append(sum_arr, 
                    1/sum_pixels(eps_x, dx, lms, lmn_scaled, swp))
            
        rms_error = (np.mean(sum_arr))**(0.5) 
        rms_error_arr = np.append(rms_error_arr, rms_error)

    return rms_error_arr


def snr_knee(sigma, bin_width):
    '''
    From eq. 10 of the paper (first TIM submission)
    using the Thompson equation calculate the SNR where background equals IRF and quantization
    '''

    return (48*np.sqrt(np.pi)*sigma**3)/(12*sigma**2*bin_width + bin_width**3)


def N_for_precision(precision, sigma, bin_width, b):
    '''
    From eq. 10 of the paper (first TIM submission)
    using the Thompson equation calculate the SNR where background equals IRF and quantization
    '''

    numer = (sigma**2 + a**2/12) + np.sqrt( (sigma**2 + a**2/12)**2 + 4*precision**2*(4*np.sqrt(np.pi)*sigma**3*b)/bin_width )
    denom = 2*sigma**2

    return numer/denom


def exp_search(sigma, bin_width, N, b, target_prec):
    '''
    Search the winick expression for an exposure time to hit a target precision given
    detector parameters and N,b 

    inputs:
        sigma: seconds 
        bin_width: in seconds
        N: total signal photons 
        b: background photons per bin
        target_prec: in mm

    returns: exposure time factor change, estimated precision (in mm)
    '''
    DEBUG_PRINTS = False 
    crb_err = winick_sweep(bin_width, [N], b, sigma, 
            noise_scaled = False)[0]  # returns in seconds 

    print('Original SNR: {:.2f}; with CRB predicted RMS error of: {:.2f} [mm]'.format(N/b, 
        crb_err*3.0e8/2*1000))

    # two step search 

    # coarse 
    exp_time = np.logspace(-3, 3, 50) # factor of 1.33 
    crb_search = np.array([])

    for t in exp_time:
        crb_err = winick_sweep(bin_width, N*t, b*t, [sigma], 
            noise_scaled = False)[0]
        crb_search = np.append(crb_search, crb_err*3.0e8/2*1000)
    idx = np.argmin(np.abs(crb_search - target_prec))
    t_coarse = exp_time[idx]

    if DEBUG_PRINTS:
        print('Coarse t = {}'.format(t_coarse))

    # fine 
    exp_time = np.logspace(np.log10(exp_time[0]/exp_time[1]), 
        np.log10(exp_time[1]/exp_time[0]), 100) 
    crb_search = np.array([])
    for t in exp_time:
        crb_err = winick_sweep(bin_width, N*t*t_coarse, b*t*t_coarse, [sigma], 
            noise_scaled = False)[0]
        crb_search = np.append(crb_search, crb_err*3.0e8/2*1000)
    idx = np.argmin(np.abs(crb_search - target_prec))
    
    t_fine = exp_time[idx]
    t_optimal = t_fine * t_coarse

    if DEBUG_PRINTS:
        print('Optimal t = {}'.format(t_optimal))

    print('Increase exposure time by x{:.2f}; for precision of {:.2f} [mm]'.format(t_optimal, 
        crb_search[idx]))

    return t_optimal, crb_search[idx]
