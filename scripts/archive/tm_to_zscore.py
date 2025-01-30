import numpy as np
from scipy.optimize import fmin
from scipy.stats import norm
from scipy.optimize import minimize
import h5py

from scripts.zscores.zscore_to_tm import zscore_to_tm
from scripts.vectors_matrices.valid_tm import is_valid_tm
from scripts.vectors_matrices.named_array import *


def historical_tms_to_zscore(tms, initial_rho=0.1, verbose=False):
    """
    Function computing z-scores from historical matrices using gradient descent to find optimal rho.
    :param tms: array of historical transition matrices
    :param initial_rho: initial guess for the rho value
    :return: z-scores, z-score variance, rho
    """

    # check that the input is a 3D array of shape (n_matrices, 3, 3)
    if tms.ndim != 3:
        raise ValueError('Input not 3D array.')

    if tms.shape[1] != 3 or tms.shape[2] != 3:
        raise ValueError('Input not 3x3 matrices.')

    # check if those matrices are proper transition matrices
    if not np.all(is_valid_tm(tms)):
        raise ValueError('Input not transition matrix.')

    # check if all entries in tms are NaNs, if so don't run the optimization, return NaNs
    if np.isnan(tms).all():
        zscore_vector = np.full((len(tms),), np.nan)
        zscore_variance = np.nan
        optimal_rho = np.nan
        upper_bounds = np.full((3, 3), np.nan)
        lower_bounds = np.full((3, 3), np.nan)
        print(f'    avg_mat fully NaN')
        return zscore_vector, zscore_variance, optimal_rho, upper_bounds, lower_bounds

    # compute average historical transition matrix, mean should ignore Nans
    avg_tm = np.nanmean(tms, axis=0)

    # check if avg_tm is a proper transition matrix
    if not np.all(is_valid_tm(avg_tm)):
        raise ValueError('Average TM not transition matrix.')

    # Define upper and lower bounds
    upper_bounds = np.fliplr(norm.ppf(np.cumsum(np.fliplr(avg_tm), axis=1)[:, 0:2]))
    upper_bounds = np.hstack((10 ** 8 * np.ones((3, 1)), upper_bounds))
    lower_bounds = np.hstack((upper_bounds[:, 1:3], -10 ** 8 * np.ones((3, 1))))

    # Function to compute the variance of z-scores for a given rho
    def zscore_variance_distance_to_one(rho):
        # compute the vector of z-scores for that given rho
        zscore_tmp = np.array([single_tm_to_zscore(tm.reshape(3, 3), upper_bounds, lower_bounds, rho) for tm in tms])
        # compute the variance
        zs_variance = np.nanvar(zscore_tmp)
        if verbose:
            print(f'    rho = {rho}, zvar = {zs_variance}')
        # output the distance of the variance to 1, which is the target value
        objective = (zs_variance - 1)**2
        return objective

    # Use gradient descent to find the optimal rho
    satisfactory_zscore_variance_delta2 = 2e-4

    def callback(intermediate_result):
        # print('callback', intermediate_result.fun)
        if intermediate_result.fun < satisfactory_zscore_variance_delta2:
            raise StopIteration

    optimization_options = {
        'maxfun': 40,  # Increase the number of iterations
        'eps': 1e-5,  # Set a smaller tolerance for step size
        'ftol': 1e-3,  # Set a smaller tolerance for function value change
        'finite_diff_rel_step': 1e-3,  # Set a smaller step size for numerical gradient computation
    }

    optimization_upper_bound = 0.3
    optimal_rho_result = minimize(zscore_variance_distance_to_one, np.array(initial_rho),
                                  bounds=[(0.001, optimization_upper_bound)], method='L-BFGS-B', jac='3-point',
                                  options=optimization_options, callback=callback)
    nfev = optimal_rho_result.nfev
    # if the optimization fails, try again with difKferent range and intitial value
    if not optimal_rho_result.fun < satisfactory_zscore_variance_delta2:
        optimization_upper_bound = 0.6
        initial_rho = 0.2
        if verbose:
            print('optimization_upper_bound', optimization_upper_bound)
        optimal_rho_result = minimize(zscore_variance_distance_to_one, np.array(initial_rho),
                                      bounds=[(0.001, optimization_upper_bound)], method='L-BFGS-B', jac='2-point',
                                      options=optimization_options)
        nfev += optimal_rho_result.nfev
        # if that fails too, try with a larger bounds and initial value
        if not (optimal_rho_result.success & (optimal_rho_result.fun < 4e-4)):
            optimization_upper_bound = 0.9
            initial_rho = optimal_rho_result.x[0] * 1.1
            if verbose:
                print('optimization_upper_bound', optimization_upper_bound)
            optimal_rho_result = minimize(zscore_variance_distance_to_one, np.array(initial_rho),
                                          bounds=[(0.001, optimization_upper_bound)], method='L-BFGS-B', jac='3-point',
                                          options=optimization_options)
            nfev += optimal_rho_result.nfev

    # save the optimal rho that was found
    optimal_rho = optimal_rho_result.x[0]

    # Compute z-scores with the optimal rho
    zscore_vector = np.array([single_tm_to_zscore(tm.reshape(3, 3), upper_bounds,
                                                  lower_bounds, optimal_rho) for tm in tms])
    zscore_variance = np.nanvar(zscore_vector)

    # print output and return the results
    print(f'    looped {nfev} times, rho = {optimal_rho}, zvar = {zscore_variance}')
    # print(f'{zscore_vector}')
    return zscore_vector, zscore_variance, optimal_rho, upper_bounds, lower_bounds


def single_tm_to_zscore(tm, upper_bounds, lower_bounds, rho):
    """
    Given a transition matrix (TM) and the upper/lower bounds from the average TM as well
    as a correlation parameter rho, the function computes and returns the z-score.
    :param tm: transition matrix
    :param upper_bounds: upper bounds of the average transition matrix
    :param lower_bounds: lower bounds of the average transition matrix
    :param rho: correlation parameter
    :return: z-score
    """

    # if the TM is all NaNs, return NaN
    if np.isnan(tm).all():
        return np.nan

    def devnest(z):
        # transform rho, ub, lb into 2d/4d vector so that they can be passed to zscore_to_tm
        rho2d = np.array([rho])
        # upper_bounds4d = upper_bounds[np.newaxis, np.newaxis, :, :]
        # lower_bounds4d = lower_bounds[np.newaxis, np.newaxis, :, :]

        tm_pred = zscore_to_tm(NamedArray(data=z, names=['']), upper_bounds, lower_bounds, rho2d)
        error_term = np.nansum((tm - tm_pred.data) ** 2)
        return error_term

    z_out = fmin(devnest, 1, disp=False)
    return z_out[0]


def all_historical_tms_to_zscores(historical_tms, rho_init, banks, portfolios,
                                  bank_porfolios, recompute_historical_zscores=False, verbose=False):
    """
    Takes in all historical transition matrices and computes the z-score time series for
    each bank and portfolio
    """

    # file for h5 saving and loading
    h5_file = 'data/derived/zscores.h5'

    # if computation is set to True
    if recompute_historical_zscores:

        # Set up the empty output arrays
        historical_zscores = NamedArray(names=['bank', 'portfolio_type', 'historical_year'],
                                        data=np.full((n_banks, n_portfolio_types, n_history_years), np.nan))
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
                if not (ib, ip) in bank_porfolios:
                    continue
                if (ib != 1) or (ip != 3):
                    continue
                if verbose:
                    print(f'Bank #{ib} ({b}), portfolio #{ip} ({p})')
                # take slice of the historical_tms for that bank and portfolio
                tm_slice = historical_tms.data[ib, ip, :, :, :]

                # compute the z-scores for that slice
                bp_hist_zscores, bp_zscore_var, bp_rho, bp_upper_bounds, bp_lower_bounds = (
                    historical_tms_to_zscore(tm_slice, rho_init, verbose=False))

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
