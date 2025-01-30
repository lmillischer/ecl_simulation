import numpy as np
import h5py

from scripts.ecl.ecl_stage1and2 import compute_stage_1and2_ecl
from scripts.ecl.ecl_stage3 import compute_stage_3_ecl

from scripts.vectors_matrices.named_array import *
from controls import *


def calculate_ecl(current_portfolios, historical_portfolios, tm_scenarios_4x4, lgd, interest_rates, do_ecl, verbose=False):

    out_path = f'data/output/{first_hash}/'

    # If ecl scenarios to be recomputed
    if do_ecl:
        # Dimensions of the output vector: (bank, portfolio, stage 1/2/3)
        #  NB. not using the stages vector as it contains 3b1, 3b2, 3b3 (default vintage buckets)
        ecl_scenarios_abs = NamedArray(names=['scenario', 'bank', 'portfolio_type', 'stage3'],
                                  data=np.full((n_scenarios, n_banks, n_portfolio_types, 3), np.nan))
        validate_named_array(ecl_scenarios_abs)

        ecl_scenarios_rel = NamedArray(names=['scenario', 'bank', 'portfolio_type', 'stage3'],
                                       data=np.full((n_scenarios, n_banks, n_portfolio_types, 3), np.nan))
        validate_named_array(ecl_scenarios_rel)

        # Reshape current portfolios, so it has the right shape and can be read by the functions below
        current_portfolios_ecl = NamedArray(names=current_portfolios.names.copy(),
                                            data=current_portfolios.data.copy())
        current_portfolios_ecl.data = current_portfolios_ecl.data[np.newaxis, ...]
        current_portfolios_ecl.names = ['scenario'] + current_portfolios_ecl.names
        validate_named_array(current_portfolios_ecl)

        # If do_cycle_neutral, we scale the stage 1, 2, 3 exposures so that they match the long-run split up
        if do_cycle_neutral:
            # compute, date by date, the historical weights of each of the 3 stages
            hist_weights = historical_portfolios.data / np.nansum(historical_portfolios.data, axis=2, keepdims=True)
            hist_weights = np.nan_to_num(hist_weights, nan=0.)  # replace 0 by nan
            # average over time
            hist_weights = np.nanmean(hist_weights, axis=3)[np.newaxis, :, :, np.newaxis, :]
            # nomalize the weights so that they sum to 1
            #    when all three stages are nan and replaced by 0 this drags down the average
            hist_weights = hist_weights / np.nansum(hist_weights, axis=4, keepdims=True)
            # compute current relative sizes of stage 3b1, 3b2, 3b3
            stage3_weights = np.nan_to_num(current_portfolios_ecl.data[:, :, :, :, 2:], nan=0)
            stage3_weights = stage3_weights / np.nansum(current_portfolios_ecl.data[:, :, :, :, 2:], axis=4, keepdims=True)
            stage3_weights = np.nan_to_num(stage3_weights, nan=1/3)
            # now multiply the relative sizes of stage 3b1, 3b2, 3b3 by the historical weight of stage 3
            stage3_weights = stage3_weights * hist_weights[:, :, :, :, 2:]
            # drop the 3rd dimension from hist_weights and append stage3_weights
            # repeat hist_weights along axis 3 (collateral_type) to make it compatible with stage3_weights
            hist_weights = np.repeat(hist_weights[:, :, :, :, :2], 4, axis=3)
            hist_weights = np.concatenate([hist_weights[:, :, :, :, :2], stage3_weights], axis=4)
            # test that the sum of the weights is 1 when it is not nan
            sum_weights = np.nansum(hist_weights, axis=4)
            if not np.all(np.isclose(sum_weights, 1) | np.isclose(sum_weights, 0)):
                raise ValueError(f'Weights do not sum to 1: {sum_weights}')
            # compute current reweighted portfolios as the product of current portfolios summed over the stage
            #   dimension and the historical weights
            current_reweighted = np.nansum(current_portfolios_ecl.data, axis=4, keepdims=True) * hist_weights
            # replace the current portfolios by the reweighted ones
            current_portfolios_ecl.data = current_reweighted

        # Compute stage 1 ECL
        ecl_scenarios_abs, ecl_scenarios_rel = (
            compute_stage_1and2_ecl(ecl_scenarios_abs, ecl_scenarios_rel, current_portfolios_ecl, tm_scenarios_4x4,
                                    lgd, interest_rates, stage1or2=1, lifetime_losses=False, verbose=verbose))

        # Compute stage 2 ECL
        ecl_scenarios_abs, ecl_scenarios_rel = (
            compute_stage_1and2_ecl(ecl_scenarios_abs, ecl_scenarios_rel, current_portfolios_ecl, tm_scenarios_4x4,
                                    lgd, interest_rates, stage1or2=2, lifetime_losses=True, verbose=verbose))

        # Compute stage 3 ECL
        ecl_scenarios_abs, ecl_scenarios_rel = (
            compute_stage_3_ecl(ecl_scenarios_abs, ecl_scenarios_rel, current_portfolios_ecl,
                                tm_scenarios_4x4, lgd, interest_rates))

        # Save NamedArray
        save_named_array(f'{out_path}/cl_scenarios.h5', ecl_scenarios_abs, 'cl_scenarios')
        save_named_array(f'{out_path}/cl_rate_scenarios.h5', ecl_scenarios_rel, 'cl_rate_scenarios')

    # If ecl scenarios to be read from file
    else:
        ecl_scenarios_abs = load_named_array(f'{out_path}/cl_scenarios.h5', 'cl_scenarios')
        ecl_scenarios_rel = load_named_array(f'{out_path}/cl_rate_scenarios.h5', 'cl_rate_scenarios')

    # Check before returning
    validate_named_array(ecl_scenarios_abs)
    validate_named_array(ecl_scenarios_rel)

    # Check if ecl always positive and below 1+max(costs)
    if np.any(ecl_scenarios_rel.data < 0):
        raise ValueError(f'ecl_scenarios_rel has negative values')
    elif verbose:
        print(f'ecl_scenarios_rel has no negative values')
    if np.any(ecl_scenarios_rel.data > 1 + max(costs.values())):
        # diplsay the bank and portfolio with the highest ecl
        max_value = np.nanmax(ecl_scenarios_rel.data)
        indices = np.where(ecl_scenarios_rel.data == max_value)
        indices = list(zip(*indices))
        # print(f'{indices} has the highest ecl: {max_value}')
        print(f'Highest ecl: {max_value}')
        raise ValueError(f'ecl_scenarios_rel has values above 1+costs')
    elif verbose:
        print(f'ecl_scenarios_rel has no values above 1+costs')

    return ecl_scenarios_abs, ecl_scenarios_rel
