import numpy as np

from scripts.vectors_matrices.named_array import *


def amortization_to_repayment(amort_prof_named, n_scenario_quarters, do_tm_scenarios=False):
    """
    This function transforms the amortization profiles (e.g. 100% 80% 40% 0%) into repayment profiles,
    i.e. what fraction will be repaid in quarter N (20% 50% 100% 0% in the example above)
    :return:
    """

    out_path = f'data/output/{first_hash}/'

    # If repayment profiles to be recomputed
    if do_tm_scenarios:

        amort_prof = amort_prof_named.data

        # Trim the amortization profiles to the number of future quarters
        amort_prof = amort_prof[:, :, :, :n_scenario_quarters + 1]

        # Along dimension 4, compute the difference between each element and the next one
        # and divide by the previous element unless that one is 0 then replace by 1

        # Array of previous elements where 0s are replaced by 1s
        prev_elem = amort_prof[:, :, :, :-1].copy()
        prev_elem[prev_elem == 0] = 1

        # Array of differences between each element and the next one
        diff_elem = np.diff(amort_prof, axis=3)
        diff_elem[diff_elem == np.nan] = 1

        repayment_prof = - diff_elem / prev_elem

        # Invert dimensions 2 and 3
        repayment_prof = np.swapaxes(repayment_prof, 2, 3)

        # To avoid having nans in the repayment profile, replace them by 0% as nothing is amortizing any more
        repayment_prof[np.isnan(repayment_prof)] = 0

        # Define NamedArray
        repayment_prof = NamedArray(names=['bank', 'portfolio_type', 'future_quarter', 'stage3'],
                                    data=repayment_prof)
        validate_named_array(repayment_prof)

        # Save to h5 file
        save_named_array(f'{out_path}/repay_scenarios.h5', repayment_prof, 'repayment_prof')

    # If repayment profiles to be read from file
    else:
        repayment_prof = load_named_array(f'{out_path}/repay_scenarios.h5', 'repayment_prof')

    # output
    return repayment_prof
