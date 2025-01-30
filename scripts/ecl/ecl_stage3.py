import numpy as np

from scripts.vectors_matrices.named_array import *


def compute_stage_3_ecl(cl_scenarios, cl_rate_scenarios, current_portfolios, tm_scenarios_4x4,
                        lgd, interest_rates, verbose=False):

    # tm_scenarios_4x4 will be manipulated, so just take a copy
    if tm_scenarios_4x4.names != ['scenario', 'bank', 'portfolio_type', 'future_quarter', 'stage4m', 'stage4m']:
        raise ValueError(f'Names of tm_scenarios_4x4 are not as expected')

    tm_scenarios_4x4_s3 = NamedArray(names=tm_scenarios_4x4.names.copy(),
                                     data=tm_scenarios_4x4.data.copy())

    # Get the stage 3 exposure
    if current_portfolios.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage5']:
        raise ValueError(f'Names of current_portfolios are not as expected')

    s3_exposure = NamedArray(names=['stage3b' if n == 'stage5' else n for n in current_portfolios.names],
                             data=current_portfolios.data[:, :, :, :, 2:5])
    validate_named_array(s3_exposure)

    # Get the first quarter LGD and stage 3 buckets (idx=2:5)
    if lgd.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage5', 'future_quarter']:
        raise ValueError(f'Names of lgd are not as expected')

    next_quarter_lgd = NamedArray(names=['stage3b' if n == 'stage5' else n for n in lgd.names[:5]],
                               data=lgd.data[:, :, :, :, 2:5, 0])
    validate_named_array(next_quarter_lgd)

    # Get the stage 3 cure rate, based on quarter 0 TMs
    if tm_scenarios_4x4.names != ['scenario', 'bank', 'portfolio_type', 'future_quarter', 'stage4m', 'stage4m']:
        raise ValueError(f'Names of tm_scenarios_4x4 are not as expected')
    s3_cure_namevec = [n for n in tm_scenarios_4x4_s3.names if n != 'future_quarter'] + ['collateral_type', 'stage3b']
    s3_cure_data = tm_scenarios_4x4_s3.data.copy()[:, :, :, 0, :, :][..., np.newaxis, np.newaxis]
    s3_cure = NamedArray(names=s3_cure_namevec,
                         data=s3_cure_data)
    validate_named_array(s3_cure)
    # We simply repeat the cure rate vector 3 times along the bucket dimension
    s3_cure.data = np.repeat(s3_cure.data, 3, axis=-1)
    validate_named_array(s3_cure)

    if s3_cure.names != ['scenario', 'bank', 'portfolio_type', 'stage4m', 'stage4m', 'collateral_type', 'stage3b']:
        raise ValueError(f'Names of s3_cure are not as expected')

    # For bucket 1, we take the transition matrix to the power of 3 for a 3-quarter horizon cure rate
    bucket1_tms = s3_cure.data[..., 0]
    tm_power_of_3 = np.einsum('ijklmn,ijkmon->ijklon', bucket1_tms, bucket1_tms)
    tm_power_of_3 = np.einsum('ijklmn,ijkmon->ijklon', tm_power_of_3, bucket1_tms)
    s3_cure.data[..., 0] = tm_power_of_3
    # For bucket 2, no need to multiply the transition matrices, we will take the 1-quarter cure rates, so no change
    # For bucket 3, we will set all cure rates to 0 so no need to make matrix multiplications
    s3_cure.data[..., 2] = 0

    # Next step: get 3-1 and 3-2 cure rates
    #  first stage index=2 as we are looking for transitions FROM stage 3 (idx=2)
    #  second stage index=0:2 as we are looking for transitions TO stages 1 and 2
    #  summing over the two selected destination sages, i.e. axis 3 (stage 1 axis having been collapsed)
    s3_cure.data = np.nansum(s3_cure.data[:, :, :, 2, 0:2, :, :], axis=3)
    s3_cure.names = [n for n in s3_cure.names if n != 'stage4m']
    validate_named_array(s3_cure)

    # Compute stage 3 ECL by summing over collateral types and buckets
    #  NB: No discounting for stage 3
    # check if names of s3_exposure and next_quarter_lgd are the same
    if s3_exposure.names != next_quarter_lgd.names:
        raise ValueError(f'Names of s3_exposure ({s3_exposure.names}) and next_quarter_lgd ({next_quarter_lgd.names}) '
                         f'do not match.')
    # check if names of s3_exposure and s3_cure are the same
    if s3_exposure.names != s3_cure.names:
        raise ValueError(f'Dimensions of s3_exposure ({s3_exposure.names}) and s3_cure ({s3_cure.names}) '
                         f'do not match.')
    # do the calculation
    s3_ecl_data = np.nansum(s3_exposure.data * next_quarter_lgd.data * (1 - s3_cure.data), axis=3)  # sum: collateral types
    s3_ecl_data = np.nansum(s3_ecl_data, axis=3)  # summ over buckets
    s3_ecl = NamedArray(names=list(s3_exposure.names[:-2]),
                        data=s3_ecl_data)
    validate_named_array(s3_ecl)

    # Compute ECL rates by dividing by the full S3 exposure
    full_s3_exp_data = np.nansum(s3_exposure.data, axis=3)  # summing over collateral types
    full_s3_exp_data = np.nansum(full_s3_exp_data, axis=3)  # summing over buckets
    full_s3_exposure = NamedArray(names=list(s3_exposure.names[:-2]),
                                  data=full_s3_exp_data)  # summing over collateral types and buckets
    validate_named_array(full_s3_exposure)

    # Save the stage 3 ECL in the output array
    cl_scenarios.data[:, :, :, 2] = s3_ecl.data
    cl_rate_scenarios.data[:, :, :, 2] = s3_ecl.data / full_s3_exposure.data

    return cl_scenarios, cl_rate_scenarios
