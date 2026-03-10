#!/usr/bin/env python3
"""
Fit UV power-law slopes (f_lambda ∝ λ^β) using forward-modelling + emcee.

Notes:
 - Input catalog fluxes/errors are expected in f_nu [erg s^-1 cm^-2 Hz^-1].
 - Convert catalog f_nu -> f_lambda at each filter pivot (observed-frame).
 - Select filters with >=50% of rest-frame transmission in 1500-3000 Å.
 - Mark detections as S/N >= 3. Detections are used directly.
 - If <2 detections, include remaining candidate filters as 2σ upper limits in the likelihood.
 - Fit lnA and β with emcee using a censored likelihood (Gaussian for detections,
   one-sided Gaussian-CDF for upper limits).
 - Excludes rest-frame pivot < 1216 Å where possible.

Created: Tuesday 10th March 2026.
"""

import numpy as np
from pathlib import Path
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy import special
import emcee

# ----------------- Configuration -----------------
#! Directories
cat_dir = Path.cwd().parent / "data" / "catalogues"
filter_dir = Path.cwd().parent / "data" / "filters"
catalogue_file = cat_dir / "COSMOS_z6_real_errors.fits"

#! Filter dict: keys are names, values are the transmission curve filenames.
filter_dict = {'HSC-G_DR3': 'g_HSC.txt',
               'HSC-R_DR3': 'r_HSC.txt',
               'HSC-I_DR3': 'i_HSC.txt',
               'HSC-NB0816_DR3': 'nb816_HSC.txt',
               'HSC-Z_DR3': 'z_HSC.txt',
               'HSC-NB0921_DR3': 'nb921_HSC.txt',
               'HSC-Y_DR3': 'y_HSC.txt',
               'Y': 'VISTA_Y.txt',
               'J': 'VISTA_J.txt',
               'H': 'VISTA_H.txt',
               'Ks': 'VISTA_Ks.txt',
               'f115w': 'f115w_angstroms.txt',
               'f150w': 'f150w_angstroms.txt',
               'f277w': 'f277w_angstroms.txt',
               'f444w': 'f444w_angstroms.txt',
               'VIS': 'Euclid_VIS.txt',
               'Ye': 'Euclid_Y.txt',
               'Je': 'Euclid_J.txt',
               'He': 'Euclid_H.txt',
               'ch1cds': 'irac_ch1.txt',
               'ch2cds': 'irac_ch2.txt',
               }

#! MCMC params
nwalkers = 50
nburn = 800
nprod = 1200

#! Rest-frame pivot wlen for normalisation
lambda0 = 2000.0

#! Minimum sigma for use of a filter in fitting
sn_threshold = 3.0

c_A_s = 2.99792458e18  # speed of light in Å s^-1

#! Output dirs
out_dir = Path.cwd() / "uv_slope_results_sn3_with_limits"
out_dir.mkdir(exist_ok=True)


#? Functions for loading filters, computing model fluxes, and likelihoods.

def load_filter(path):
    """Load a filter transmission curve."""
    data = np.loadtxt(path)
    wave = data[:, 0].astype(float)
    trans = data[:, 1].astype(float)
    trans = np.clip(trans, 0.0, None)
    return wave, trans

def filter_pivot(wave, trans):
    """Find the pivot wlen of a filter given its transmission curve."""
    den = np.trapz(trans, wave)
    if den <= 0:
        return np.median(wave)
    return np.trapz(wave * trans, wave) / den

def model_band_flux_lnA_beta(lnA, beta, wave_obs, trans_obs, z, lambda0=lambda0):
    """
    Predict observed band-averaged f_lambda.
    lnA: ln(A) where A = f_lambda_rest(lambda0) (we are fitting for A and beta)
    wave_obs: observed-frame sampling (Å)
    trans_obs: transmission
    """
    A = np.exp(lnA)
    wave_rest = wave_obs / (1.0 + z)
    f_lambda_rest = A * (wave_rest / lambda0) ** beta
    f_lambda_obs = f_lambda_rest / (1.0 + z)
    num = np.trapz(trans_obs * f_lambda_obs, wave_obs)
    den = np.trapz(trans_obs, wave_obs)
    if den <= 0:
        return 0.0
    return num / den

#? Likelihood functions

def log_prior(lnA, beta):
    if not np.isfinite(lnA) or not np.isfinite(beta):
        return -np.inf
    if beta < -10.0 or beta > 5.0:
        return -np.inf
    if lnA < -300.0 or lnA > 200.0:
        return -np.inf
    return 0.0

def log_likelihood_censored(lnA, beta, waves_list, trans_list, z,
                            obs_fluxes, obs_errs, is_upper):
    """
    For each band:
      - if is_upper==False: Gaussian likelihood using obs_fluxes, obs_errs
      - if is_upper==True: one-sided Gaussian CDF P(F_obs < limit | model)
        obs_fluxes contains the limit (e.g. 2*sigma) for upper limits
    """
    lnlike = 0.0
    for w, t, f_obs, ferr, upper in zip(waves_list, trans_list, z * 0 + 0, obs_fluxes, is_upper):
        pass

def log_likelihood(lnA, beta, waves_list, trans_list, z, obs_fluxes, obs_errs, is_upper):
    lnlike = 0.0
    for w, t, f_obs, ferr, upper in zip(waves_list, trans_list, obs_fluxes, obs_errs, is_upper):
        if ferr <= 0 or not np.isfinite(f_obs):
            return -np.inf
        model_f = model_band_flux_lnA_beta(lnA, beta, w, t, z, lambda0=lambda0)
        if upper:
            arg = (f_obs - model_f) / (np.sqrt(2) * ferr)
            cdf = 0.5 * (1.0 + special.erf(arg))
            if cdf <= 0:
                return -np.inf
            lnlike += np.log(cdf)
        else:
            chi = (f_obs - model_f) / ferr
            lnlike += -0.5 * (chi * chi + np.log(2 * np.pi * (ferr ** 2)))
    return lnlike

def log_posterior(theta, waves_list, trans_list, z, obs_fluxes, obs_errs, is_upper):
    lnA, beta = theta
    lp = log_prior(lnA, beta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(lnA, beta, waves_list, trans_list, z, obs_fluxes, obs_errs, is_upper)
    return lp + ll

#? Functions to process each object in catalogue.

def fit_object(obj, filters_all, filter_dict_local):
    z = obj['Zphot']

    #* collect candidate filters: >=50% of rest transmission area inside 1500-3000 Å
    candidate_names = []
    pivots_obs = {}
    for fname in filters_all:
        if fname not in filter_dict_local:
            continue
        fpath = filter_dir / filter_dict_local[fname]
        if not fpath.exists():
            continue
        w, t = load_filter(fpath)
        rest_w = w / (1.0 + z)
        total_area = np.trapz(t, w)
        if total_area <= 0:
            continue
        mask = (rest_w >= 1500.0) & (rest_w <= 3000.0)
        if np.sum(mask) < 3:
            continue
        in_area = np.trapz(t[mask], w[mask])
        if in_area / total_area >= 0.5:
            candidate_names.append(fname)
            pivots_obs[fname] = filter_pivot(w, t)

    if len(candidate_names) == 0:
        return {'ok': False, 'reason': 'no_filters_in_window'}

    # convert catalog f_nu -> f_lambda
    flux_nu = []
    err_nu = []
    for fname in candidate_names:
        flux_nu.append(obj[f'flux_{fname}'])
        err_nu.append(obj[f'err_real_{fname}'])
    flux_nu = np.array(flux_nu)
    err_nu = np.array(err_nu)

    pivots = np.array([pivots_obs[f] for f in candidate_names])  # Å
    flux_lambda = flux_nu * c_A_s / (pivots ** 2)
    err_lambda = err_nu * c_A_s / (pivots ** 2)

    # classify detections and upper limits
    sn = np.zeros_like(flux_lambda)
    valid = np.isfinite(flux_lambda) & np.isfinite(err_lambda) & (err_lambda > 0)
    sn[valid] = flux_lambda[valid] / err_lambda[valid]
    is_det = sn >= sn_threshold
    is_ul = ~is_det  # remaining candidate bands are treated as upper limits when needed

    n_det = np.sum(is_det)
    # if >=2 detections: fit using detections only (ignore upper limits)
    # if <2 detections: include upper limits (2*sigma) in the likelihood
    use_upper_limits = n_det < 2

    # prepare filter sampling arrays for those used in likelihood 
    use_names = []
    waves_list = []
    trans_list = []
    obs_fluxes = []
    obs_errs = []
    is_upper = []

    for fname, det_flag, flux_lam, err_lam in zip(candidate_names, is_det, flux_lambda, err_lambda):
        if det_flag:
            # detection: include
            fpath = filter_dir / filter_dict_local[fname]
            w, t = load_filter(fpath)
            mask = t > 1e-8
            if np.sum(mask) < 3:
                mask = slice(None)
            waves_list.append(w[mask])
            trans_list.append(t[mask])
            obs_fluxes.append(flux_lam)
            obs_errs.append(err_lam)
            is_upper.append(False)
            use_names.append(fname)
        else:
            # non-detection: include only if we decided to use upper limits
            if use_upper_limits:
                fpath = filter_dir / filter_dict_local[fname]
                w, t = load_filter(fpath)
                mask = t > 1e-8
                if np.sum(mask) < 3:
                    mask = slice(None)
                waves_list.append(w[mask])
                trans_list.append(t[mask])
                # use 2*sigma as the upper-limit value (in f_lambda)
                limit = 2.0 * err_lam
                obs_fluxes.append(limit)
                obs_errs.append(err_lam)
                is_upper.append(True)
                use_names.append(fname)
            else:
                # skip this upper limit if not needed
                pass

    if len(waves_list) == 0:
        return {'ok': False, 'reason': 'no_valid_bands_after_sn_and_selection', 'candidates': candidate_names}

    # exclude bands with rest pivot < 1216 Å where possible, keepingat least two bands overall
    piv_rest = np.array([filter_pivot(w, t) for w, t in zip(waves_list, trans_list)]) / (1.0 + z)
    good_mask = piv_rest >= 1216.0
    if np.sum(good_mask) >= 2:
        waves_list = [w for w, g in zip(waves_list, good_mask) if g]
        trans_list = [t for t, g in zip(trans_list, good_mask) if g]
        obs_fluxes = np.array([f for f, g in zip(obs_fluxes, good_mask) if g])
        obs_errs = np.array([e for e, g in zip(obs_errs, good_mask) if g])
        is_upper = np.array([u for u, g in zip(is_upper, good_mask) if g])
        used_names = [n for n, g in zip(use_names, good_mask) if g]
    else:
        # keep all (not enough bands to drop)
        obs_fluxes = np.array(obs_fluxes)
        obs_errs = np.array(obs_errs)
        is_upper = np.array(is_upper)
        used_names = use_names

    if len(waves_list) < 1:
        return {'ok': False, 'reason': 'no_bands_after_igm_cut'}

    # initial guess
    if n_det >= 2:
        # use detected bands for initial guess
        det_indices = [i for i, u in enumerate(is_upper) if not u]
        pivots_rest_for_guess = np.array([filter_pivot(waves_list[i], trans_list[i]) for i in det_indices]) / (1 + z)
        fluxes_for_guess = np.array([obs_fluxes[i] for i in det_indices])
        errs_for_guess = np.array([obs_errs[i] for i in det_indices])
    else:
        pivots_rest_for_guess = np.array([filter_pivot(w, t) for w, t in zip(waves_list, trans_list)]) / (1 + z)
        fluxes_for_guess = np.maximum(obs_fluxes, 1e-40)  # limits used as conservative fluxes
        errs_for_guess = obs_errs

    # quick log-linear fit for initial beta
    x = np.log10(pivots_rest_for_guess)
    y = np.log10(np.maximum(fluxes_for_guess, 1e-40))
    yerr_approx = errs_for_guess / (fluxes_for_guess * np.log(10) + 1e-40)
    wght = 1.0 / (yerr_approx**2 + 1e-12)
    S = np.sum(wght); Sx = np.sum(wght * x); Sy = np.sum(wght * y)
    Sxx = np.sum(wght * x * x); Sxy = np.sum(wght * x * y)
    denom = (S * Sxx - Sx * Sx)
    beta0 = -2.0 if denom == 0 else (S * Sxy - Sx * Sy) / denom
    lambda_med = np.median(pivots_rest_for_guess)
    A0 = np.median(fluxes_for_guess * (1 + z)) * (lambda_med / lambda0) ** (-beta0)
    lnA0 = np.log(A0) if (A0 > 0 and np.isfinite(A0)) else -50.0

    # emcee setup
    ndim = 2
    p0 = np.vstack([lnA0 + 1e-3 * np.random.randn(nwalkers),
                    beta0 + 1e-2 * np.random.randn(nwalkers)]).T

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                    args=(waves_list, trans_list, z, obs_fluxes, obs_errs, is_upper))

    # run MCMC
    pos, prob, state = sampler.run_mcmc(p0, nburn, progress=False)
    sampler.reset()
    sampler.run_mcmc(pos, nprod, progress=False)

    samples = sampler.get_chain(flat=True)
    lnA_med, beta_med = np.median(samples, axis=0)
    lnA_lo, beta_lo = np.percentile(samples, 16, axis=0)
    lnA_hi, beta_hi = np.percentile(samples, 84, axis=0)

    A_med = np.exp(lnA_med)

    result = {
        'ok': True,
        'used_filters': used_names,
        'A_median': A_med,
        'A_err_minus': A_med - np.exp(lnA_lo),
        'A_err_plus': np.exp(lnA_hi) - A_med,
        'beta_median': beta_med,
        'beta_lo': beta_lo,
        'beta_hi': beta_hi,
        'samples': samples,
        'sampler': sampler,
        'pivots_rest': np.array([filter_pivot(w, t) for w, t in zip(waves_list, trans_list)]) / (1 + z),
        'obs_fluxes': obs_fluxes,
        'obs_errs': obs_errs,
        'is_upper': is_upper,
        'used_names': used_names,
        'n_detections': int(n_det),
        'n_candidates': len(candidate_names),
    }
    return result


##################! RUNNING THE CODE, LOOPING THROUGH CATALOGUE OBJECTS ##################
def main():
    t = Table.read(catalogue_file)
    t.sort('Muv')
    # testing - run on subset!!
    # t = t[0:10]

    flux_cols = [col for col in t.colnames if col.startswith('flux_')]
    filters_all = [col.split('flux_')[1] for col in flux_cols]

    for i, obj in enumerate(t):
        z = obj['Zphot']
        ID = obj['ID']
        print(f"Object {i}, ID {ID}: z={z:.4f}")
        try:
            res = fit_object(obj, filters_all, filter_dict)
        except Exception as e:
            print(f"  Fit error: {e}")
            continue

        if not res['ok']:
            print(f"  Skipped: {res.get('reason')} (candidates: {res.get('n_candidates','N/A')})")
            continue

        beta = res['beta_median']
        beta_lo = res['beta_lo']
        beta_hi = res['beta_hi']
        print(f"  beta = {beta:.3f} (+{beta_hi - beta:.3f}/-{beta - beta_lo:.3f}), detections={res['n_detections']}, used={res['used_filters']}")

        # diagnostic plots
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(121)
        piv = res['pivots_rest']
        fx = res['obs_fluxes']
        ferr = res['obs_errs']
        is_upper = res['is_upper'].astype(bool)
        if np.any(~is_upper):
            ax.errorbar(piv[~is_upper], fx[~is_upper], yerr=ferr[~is_upper], fmt='o', label='detections')
        if np.any(is_upper):
            ax.errorbar(piv[is_upper], fx[is_upper], yerr=ferr[is_upper], fmt='v', label='2σ upper limits')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Rest λ [Å]'); ax.set_ylabel('f_λ [erg s^-1 cm^-2 Å^-1]')
        ax.set_title(f'Obj {i} z={z:.3f}  β={beta:.3f}')

        # overplot median model predictions through used filters
        lnA_med = np.log(res['A_median'])
        model_points = []
        piv_obs_plot = []
        for fname in res['used_names']:
            w, t = load_filter(filter_dir / filter_dict[fname])
            model_points.append(model_band_flux_lnA_beta(lnA_med, beta, w, t, z, lambda0=lambda0))
            piv_obs_plot.append(filter_pivot(w, t) / (1 + z))
        model_points = np.array(model_points)
        piv_obs_plot = np.array(piv_obs_plot)
        ax.scatter(piv_obs_plot, model_points, marker='s', facecolors='none', edgecolors='C2', label='model (median)')
        ax.legend()

        ax2 = fig.add_subplot(122)
        ax2.hist(res['samples'][:, 1], bins=40, histtype='stepfilled', alpha=0.7)
        ax2.axvline(beta, color='k', linestyle='--')
        ax2.set_xlabel('β'); ax2.set_title('Posterior β')

        fig.tight_layout()
        outname = out_dir / f"obj_{i}_ID_{ID}_z{z:.3f}_beta{beta:.3f}.pdf"
        fig.savefig(outname)
        plt.close(fig)

        # write results
        with open(out_dir / f"obj_{i}_ID_{ID}_results.txt", "w") as fh:
            fh.write(f"obj {i} z {z:.5f}\n")
            fh.write(f"n_candidates {res['n_candidates']} n_detections {res['n_detections']}\n")
            fh.write(f"used_filters: {res['used_filters']}\n")
            fh.write(f"A = {res['A_median']:.5e} +{res['A_err_plus']:.5e} -{res['A_err_minus']:.5e}\n")
            fh.write(f"beta = {beta:.5f} +{beta_hi - beta:.5f} -{beta - beta_lo:.5f}\n")

    print("Done.")

if __name__ == '__main__':
    main()
