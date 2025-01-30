import numpy as np
from concurrent.futures import ProcessPoolExecutor

from scripts.zscores.zscore_to_tm import zscore_to_tm
from scripts.vectors_matrices.named_array import *


def compute_for_bank_portfolio(ib, b, ip, p, historical_tms, bank_portfolios, verbose):
    # Check if the tuple (ib, ip) is a valid bank portfolio
    if not (ib, ip) in bank_portfolios:
        return None

    if verbose:
        print(f'Bank #{ib} ({b}), portfolio #{ip} ({p})')

    # Get the slice of the historical_tms for that bank and portfolio
    tm_slice = historical_tms.data[ib, ip, :, :, :]
    if np.all(np.isnan(tm_slice)):
        if verbose:
            print('   > Slice is all NaNs, skipping')
        return None

    # Compute the z-scores for that slice
    bp_hist_zscores, bp_zscore_var, bp_rho, bp_bounds = bank_portfolio_tms_to_zscore(tm_slice, verbose=True)
    bp_upper_bounds = np.hstack((10 ** 8 * np.ones((3, 1)), bp_bounds))
    bp_lower_bounds = np.hstack((bp_bounds, -10 ** 8 * np.ones((3, 1))))

    # Structure to hold results
    return (ib, ip, bp_hist_zscores, bp_zscore_var, bp_rho, bp_upper_bounds, bp_lower_bounds)


def historical_tms_to_zscores(historical_tms, rho_init, banks, portfolios, bank_porfolios,
                              recompute_historical_zscores=False, verbose=False):

    # file for h5 saving and loading
    h5_file = 'data/derived/zscores.h5'

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

    if recompute_historical_zscores:
        # Prepare the structure to hold all results
        results = []

        # Setup parallel processing using ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            # Prepare a list of tasks
            tasks = [(ib, b, ip, p, historical_tms, verbose) for ib, b in enumerate(banks) for ip, p in enumerate(portfolios)]
            results = list(executor.map(lambda args: compute_for_bank_portfolio(*args), tasks))

        # After collecting all results, integrate them back into your data structure
        for result in results:
            if result is not None:
                ib, ip, bp_hist_zscores, bp_zscore_var, bp_rho, bp_upper_bounds, bp_lower_bounds = result
                historical_zscores.data[ib, ip, :] = bp_hist_zscores
                zscore_variance.data[ib, ip] = bp_zscore_var
                rhos.data[ib, ip] = bp_rho
                upper_bounds.data[ib, ip, :, :] = bp_upper_bounds
                lower_bounds.data[ib, ip, :, :] = bp_lower_bounds

        # Save results to h5
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

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(tm_slice[:, 0, 2])
    # plt.title(f'S1 Default rates for bank {39} and portfolio {2}')
    # plt.show()

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
        errors = np.nansum((tms_fitted - tm_slice_short) ** 2, axis=(1, 2))
        full_error_term = np.nansum(errors) + (np.nanvar(zscores) - 1) ** 2
        return full_error_term

    default_rates = tm_slice_short[:, 0, 2]
    default_rates_norm = -(default_rates - np.nanmean(default_rates)) / np.nanstd(default_rates)
    # replace all nans in default_rates_norm by 0
    default_rates_norm[np.isnan(default_rates_norm)] = 0

    # print(tm_slice_short)
    # print(default_rates_norm)
    # tms_fitted_tmp = np.array(
    #     [zscore_to_tm(NamedArray(data=np.array([z_t]), names=['']), upper_bounds, lower_bounds, np.array([0.1])).data
    #      for z_t in default_rates_norm])
    # print(tms_fitted_tmp)

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