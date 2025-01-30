import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from scripts.zscores.zscore_to_tm import zscore_to_tm
from scripts.vectors_matrices.named_array import *
from controls import *


def historical_tms_to_zscores(historical_tms, rho_init, banks, portfolios,
                              bank_portfolios, recompute_historical_zscores=False,
                              verbose=False):
    # file for h5 saving and loading
    h5_file = f'data/output/{first_hash}/zscores.h5'

    # if computation is set to True
    if recompute_historical_zscores:

        # Set up the empty output arrays
        historical_zscores = NamedArray(names=['bank', 'portfolio_type', 'historical_tm_quarter'],
                                        data=np.full((n_banks, n_portfolio_types, n_history_quarters_tm), np.nan))
        zscore_variance = NamedArray(names=['bank', 'portfolio_type'],
                                     data=np.full((n_banks, n_portfolio_types), np.nan))
        rhos = NamedArray(names=['bank', 'portfolio_type'],
                          data=np.full((n_banks, n_portfolio_types), np.nan))
        upper_bounds = NamedArray(names=['bank', 'portfolio_type', 'stage3', 'stage3'],
                                  data=np.full((n_banks, n_portfolio_types, 3, 3), np.nan))
        lower_bounds = NamedArray(names=['bank', 'portfolio_type', 'stage3', 'stage3'],
                                  data=np.full((n_banks, n_portfolio_types, 3, 3), np.nan))

        # loop over banks
        for ib, b in enumerate(banks):
            # loop over portfolios
            for ip, p in enumerate(portfolios):
                # if the tuple (ib, ip) is not in the array of all valid bank portfolios, skip
                if not (ib, ip) in bank_portfolios:
                    continue

                # To look at one bank-portfolio only
                # if not (ib, ip) == (7, 0):
                #     continue

                if verbose:
                    print(f'Bank #{ib} ({b}), portfolio #{ip} ({p})')
                # take slice of the historical_tms for that bank and portfolio
                tm_slice = historical_tms.data[ib, ip, :, :, :]

                # if the slice is all NaNs, skip
                if np.all(np.isnan(tm_slice)):
                    if verbose:
                        print('   > Slice is all NaNs, skipping')
                    continue

                # compute the z-scores for that slice
                bp_hist_zscores, bp_zscore_var, bp_rho, bp_bounds \
                    = bank_portfolio_tms_to_zscore(tm_slice, verbose=True)
                bp_upper_bounds = np.hstack((10 ** 8 * np.ones((3, 1)), bp_bounds))
                bp_lower_bounds = np.hstack((bp_bounds, -10 ** 8 * np.ones((3, 1))))

                # put the results in the output arrays
                historical_zscores.data[ib, ip, :] = bp_hist_zscores
                zscore_variance.data[ib, ip] = bp_zscore_var
                rhos.data[ib, ip] = bp_rho
                upper_bounds.data[ib, ip, :, :] = bp_upper_bounds
                lower_bounds.data[ib, ip, :, :] = bp_lower_bounds

        # save the results to h5
        save_named_array(h5_file, historical_zscores, 'historical_zscores')
        save_named_array(h5_file, zscore_variance, 'zscore_variance')
        save_named_array(h5_file, rhos, 'rhos')
        save_named_array(h5_file, upper_bounds, 'upper_bounds')
        save_named_array(h5_file, lower_bounds, 'lower_bounds')

    # if computation is set to False, just read from h5 file
    else:
        historical_zscores = load_named_array(h5_file, 'historical_zscores')
        zscore_variance = load_named_array(h5_file, 'zscore_variance')
        rhos = load_named_array(h5_file, 'rhos')
        upper_bounds = load_named_array(h5_file, 'upper_bounds')
        lower_bounds = load_named_array(h5_file, 'lower_bounds')

    # return results, whether computed or read from h5
    return historical_zscores, zscore_variance, rhos, upper_bounds, lower_bounds


def bank_portfolio_tms_to_zscore(tm_slice, verbose=True):
    distance_to_bounds = 4e-6
    variance_tolerance = 0.05
    avg_tm = np.nanmean(tm_slice, axis=0)
    # print('avg_tm', avg_tm)
    # print('tm_slice', tm_slice)

    bounds = norm.ppf(1 - np.cumsum(avg_tm[:, 0:2], axis=1))
    upper_bounds = np.hstack((10 ** 8 * np.ones((3, 1)), bounds))
    lower_bounds = np.hstack((bounds, -10 ** 8 * np.ones((3, 1))))
    tm_slice_non_nan_mask = ~np.all(np.isnan(tm_slice), axis=(1, 2))
    tm_slice_short = tm_slice[tm_slice_non_nan_mask, :, :]

    def error(params):
        # print(params)
        rho = params[0]
        zscores = params[1:]
        # Vectorized calculation of tms_fitted
        tms_fitted = np.array([zscore_to_tm(NamedArray(data=np.array([z_t]), names=['']), upper_bounds, lower_bounds, np.array([rho])).data for z_t in zscores])
        # Vectorized error calculation
        # errors = np.nansum((tms_fitted - tm_slice_short) ** 2, axis=(1, 2))
        errors = np.nanmean((tms_fitted - tm_slice_short) ** 2, axis=(1, 2))
        full_error_term = np.nansum(errors) + (np.nanvar(zscores) - 1) ** 2
        return full_error_term

    default_rates = tm_slice_short[:, 0, 2]
    default_rates_norm = -(default_rates - np.nanmean(default_rates)) / np.nanstd(default_rates)
    # replace all nans in default_rates_norm by 0
    default_rates_norm[np.isnan(default_rates_norm)] = 0

    initial_params = np.insert(default_rates_norm, 0, 0.1)
    nlc = NonlinearConstraint(lambda x: np.nanvar(x[1:]), 1-variance_tolerance, 1+variance_tolerance)
    bounds_params = [(0 + distance_to_bounds, 1 - distance_to_bounds)] + [(-10, 10)] * len(default_rates_norm)

    # measure time of minimization
    start = time.time()
    result = minimize(error, initial_params, method='SLSQP', bounds=bounds_params, constraints=nlc)
    # result = minimize(error, initial_params, method='SLSQP', bounds=bounds_params)
    end = time.time()
    print('Time elapsed:', end - start)

    optimal_rho = result.x[0]
    zscore_vector = result.x[1:]
    zscore_variance = np.nanvar(zscore_vector)

    zscore_vector_full = np.full(tm_slice.shape[0], np.nan)
    zscore_vector_full[tm_slice_non_nan_mask] = zscore_vector

    # if the number of non-nan z-scores is less than 3, replace all with nan
    if np.sum(~np.isnan(zscore_vector_full)) < 3:
        zscore_vector_full = np.full(tm_slice.shape[0], np.nan)
        zscore_variance = np.nan
        optimal_rho = np.nan

    print('Optimal rho:', optimal_rho)
    print('Zscore variance:', zscore_variance, '\n')
    print('Zscore vector:', zscore_vector_full, '\n')

    return zscore_vector_full, zscore_variance, optimal_rho, bounds