import numpy as np

from scripts.vectors_matrices.named_array import *


def compute_stage_1_ecl(cl_scenarios, cl_rate_scenarios, current_portfolios, tm_scenarios_4x4,
                        lgd, interest_rates, verbose=False):

    # Get the stage 1 exposure
    if current_portfolios.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage5']:
        raise ValueError(f'Names of current_portfolios are not as expected')

    s1_exposure = NamedArray(names=current_portfolios.names[:4].copy(),
                             data=current_portfolios.data[:, :, :, :, 0].copy())
    validate_named_array(s1_exposure)
    if verbose:
        non_nan_exposure = np.any(~np.isnan(s1_exposure.data), axis=(0, 3))
        n_non_missing = np.sum(non_nan_exposure)
        print(f'[ecl s1] nb bank-ptf with current exposure: {n_non_missing}')

    # Get the LGD vector for quarter 0 (the first simlated quarter) and stage 1
    if lgd.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage5', 'future_quarter']:
        raise ValueError(f'Names of lgd are not as expected')
    next_quarter_lgd = NamedArray(names=lgd.names[:4].copy(),
                               data=lgd.data[:, :, :, :, 0, 0].copy())
    validate_named_array(next_quarter_lgd)
    if verbose:
        non_nan_lgd = np.any(~np.isnan(next_quarter_lgd.data), axis=(0, 3))
        n_non_missing = np.sum(non_nan_lgd)
        indices = np.argwhere((non_nan_exposure) & (~non_nan_lgd))
        print(f'[ecl s1] nb bank-ptf with next quarter LGD: {n_non_missing} (missing {indices.shape[0]})')

    # Get the stage 1 PD
    if tm_scenarios_4x4.names != ['scenario', 'bank', 'portfolio_type', 'future_quarter', 'stage4m', 'stage4m']:
        raise ValueError(f'Names of tm_scenarios_4x4 are not as expected')
    s1_pd = NamedArray(names=tm_scenarios_4x4.names[:3].copy() + ['collateral_type'],
                       data=tm_scenarios_4x4.data[:, :, :, 0, 0, 2].copy()[..., np.newaxis])
    validate_named_array(s1_pd)

    if verbose:
        non_nan_pd = np.any(~np.isnan(s1_pd.data), axis=(0, 3))
        n_non_missing = np.sum(non_nan_pd)
        indices = np.argwhere((non_nan_exposure) & (~non_nan_pd))
        print(f'[ecl s1] nb bank-ptf with PD: {n_non_missing} (missing {indices.shape[0]})')

    # Discounting half a quarter
    discount_factor = NamedArray(names=interest_rates.names.copy(),
                                 data=(1 + interest_rates.data.copy()) ** 0.5)
    discount_factor.data = discount_factor.data[np.newaxis, :, :]
    discount_factor.names = ['scenario'] + discount_factor.names
    validate_named_array(discount_factor)

    if verbose:
        n_non_missing = np.sum(np.any(~np.isnan(discount_factor.data), axis=0))
        print(f'[ecl s1] nb bank-ptf with discount factor: {n_non_missing}')

    # Compute stage 1 ECL by summing over collateral types
    # check if names of s1_exposure and next_quarter_lgd are the same
    if s1_exposure.names != next_quarter_lgd.names:
        raise ValueError(f'Names of s1_exposure and next_quarter_lgd are not the same: '
                         f'{s1_exposure.names} vs {next_quarter_lgd.names}')
    # check if names of s1_exposure and s1_pd are the same
    if s1_exposure.names != s1_pd.names:
        raise ValueError(f'Names of s1_exposure and s1_pd are not the same: '
                         f'{s1_exposure.names} vs {s1_pd.names}')
    s1_ecl_data = np.nansum(s1_exposure.data * next_quarter_lgd.data * s1_pd.data, axis=3) / discount_factor.data

    # Define array with nas where there is no s1_exposure (np.nansum workaround)
    s1_exposure_not_nan = NamedArray(names=discount_factor.names.copy(),
                                     data=np.where(np.any(~np.isnan(s1_exposure.data * next_quarter_lgd.data * s1_pd.data),
                                                          axis=(0, 3)), 1, np.nan)[np.newaxis, ...])
    validate_named_array(s1_exposure_not_nan)

    # Define s1_ecl as a NamedArray
    s1_ecl = NamedArray(names=discount_factor.names.copy(),
                        data=s1_ecl_data * s1_exposure_not_nan.data)
    validate_named_array(s1_ecl)

    # Compute ECL rates by dividing by the full S1 exposure
    full_s1_exposure = NamedArray(names=s1_exposure.names[:3].copy(),
                                  data=np.nansum(s1_exposure.data, axis=3))  # summing over collateral types
    validate_named_array(full_s1_exposure)

    s1_ecl_rel = NamedArray(names=s1_ecl.names.copy(),
                            data=s1_ecl.data / full_s1_exposure.data)

    if verbose:
        n_non_missing = np.sum(np.any(~np.isnan(s1_ecl.data), axis=0))
        print(f'[ecl s1] nb bank-ptf with ECL: {n_non_missing}')
        n_non_missing = np.sum(np.any(~np.isnan(s1_ecl.data/full_s1_exposure.data), axis=0))
        print(f'[ecl s1] nb bank-ptf with relative ECL: {n_non_missing}')

    # Save the stage 1 ECL in the output array
    if cl_scenarios.names != ['scenario', 'bank', 'portfolio_type', 'stage3']:
        raise ValueError(f'Names of cl_scenarios are not as expected')

    cl_scenarios.data[:, :, :, 0] = s1_ecl.data
    cl_rate_scenarios.data[:, :, :, 0] = s1_ecl_rel.data

    return cl_scenarios, cl_rate_scenarios
