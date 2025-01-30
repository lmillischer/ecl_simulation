import numpy as np

from scripts.vectors_matrices.named_array import *


def compute_stage_2_ecl(cl_scenarios, cl_rate_scenarios, current_portfolios, tm_scenarios_4x4,
                        lgd, interest_rates, verbose=False):

    # tm_scenarios_4x4_s2 will be manipulated, so just take a copy
    if tm_scenarios_4x4.names != ['scenario', 'bank', 'portfolio_type', 'future_quarter', 'stage4m', 'stage4m']:
        raise ValueError(f'Names of tm_scenarios_4x4 are not as expected')

    tm_scenarios_4x4_s2 = NamedArray(names=tm_scenarios_4x4.names.copy(),
                                     data=tm_scenarios_4x4.data.copy())

    # to avoid double counting set the recovery rates from stage 3 to stages 1 and 2 to 0
    tm_scenarios_4x4_s2.data[..., 2, :] = [0, 0, 1, 0]

    # Get the stage 2 exposure
    if current_portfolios.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage5']:
        raise ValueError(f'Names of current_portfolios are not as expected')

    s2_exposure = NamedArray(names=current_portfolios.names[:4].copy(),
                             data=current_portfolios.data[:, :, :, :, 1])
    validate_named_array(s2_exposure)

    # Get the COL flows to stage 3 from exposures initially in stage 2
    #  step 0: duplicat the s2_exposure along the first axis to have it shape (n_scearios, ....)
    s2_exposure.data = np.repeat(s2_exposure.data, n_scenarios, axis=0)
    validate_named_array(s2_exposure)
    #  step 1: add a dimension for the stage (size 4) and put 0s for all except stage 2
    s2_exposure.data = s2_exposure.data[..., np.newaxis]
    s2_exposure.data = np.concatenate((np.zeros(s2_exposure.data.shape),
                                       s2_exposure.data,
                                       np.zeros(s2_exposure.data.shape),
                                       np.zeros(s2_exposure.data.shape)), axis=4)
    s2_exposure.names.append('stage4m')
    validate_named_array(s2_exposure)
    #  step 2: add a dimension for future quarters and put 0s for all except quarter 0
    s2_exposure.data = s2_exposure.data[..., np.newaxis]
    future_quarters_shape = list(s2_exposure.data.shape)
    future_quarters_shape[-1] = n_scenario_quarters
    futures_quarters_with_zeros = np.zeros(tuple(future_quarters_shape))
    s2_exposure.data = np.concatenate((s2_exposure.data, futures_quarters_with_zeros), axis=5)
    s2_exposure.names.append('future_snapshot')
    validate_named_array(s2_exposure)

    if s2_exposure.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage4m', 'future_snapshot']:
        raise ValueError(f'Names of s2_exposure are not as expected')

    # Check that all the exposure in stage 1, 3 and M in quarter 0 are 0
    for stage in [0, 2, 3]:
        if np.any(s2_exposure.data[:, :, :, :, stage, 0] != 0):
            raise ValueError(f'There are exposures in stage {stage} in quarter 0')


    # step 3: perform matrix multiplication of the exposure by the transition matrix
    #    add a dimension for the collateral stat_type (size 4)
    tm_scenarios_4x4_s2.data = tm_scenarios_4x4_s2.data[:, :, :, np.newaxis, ...]
    tm_scenarios_4x4_s2.names = tm_scenarios_4x4_s2.names[:3] + ['collateral_type'] + tm_scenarios_4x4_s2.names[3:]
    validate_named_array(tm_scenarios_4x4_s2)
    #    move future_quarter dimension to the end
    tm_scenarios_4x4_s2.data = np.moveaxis(tm_scenarios_4x4_s2.data, 4, -1)
    tm_scenarios_4x4_s2.names = tm_scenarios_4x4_s2.names[:4] + tm_scenarios_4x4_s2.names[5:] + ['future_quarter']
    validate_named_array(tm_scenarios_4x4_s2)

    # define the flow to stage 3 array (caution: future_quarter instead of future_snapshot)
    fts3_names = s2_exposure.names.copy()[:-1] + ['future_quarter']
    flow_to_stage_3 = NamedArray(names=fts3_names,
                                 data=np.zeros(s2_exposure.data.shape[:-1] + (n_scenario_quarters,)))
    # we are only interested in the first to stages, so keep only [:2] of the 5th dimension
    flow_to_stage_3.data = flow_to_stage_3.data[..., :2, :]
    flow_to_stage_3.names[4] = 'stage2'

    validate_named_array(flow_to_stage_3)

    #  loop over future quarters, for each quarter fill the s2_exposure (last dimenson) with the result of the
    #  multiplication of 4x4 transition matrix by the 4x1 exposure vector
    for quarter in range(n_scenario_quarters):
        # make matrix multiplication to get the stock in the next quarter, save that in s2_exposure
        s2_exposure.data[..., quarter+1] = np.einsum('ijklm,ijklmn->ijkln', s2_exposure.data[..., quarter],
                                                  tm_scenarios_4x4_s2.data[..., quarter])
        #   NB: last dimension of s2_exposure is future_snapshot, last dimension of tm_scenarios_4x4_s2 is future_quarter
        #       therefore getting from s2_exp[y] to s2_exp[y+1] one considers tm_scen[y] for multiplication

        # for each quarter multiply the stock of stage 1 and stage 2 by the TM elements T13 and T23 and in flow_to_stage_3
        flow_to_stage_3.data[..., quarter] = s2_exposure.data[..., :2, quarter] * tm_scenarios_4x4_s2.data[..., :2, 2, quarter]

    # Define discount factor
    # add new dimension for the future quarter, repeat the interest rates along that dimension
    interest_rates.data = interest_rates.data[:, :, np.newaxis]
    interest_rates.data = np.repeat(interest_rates.data, n_scenario_quarters, axis=-1)
    interest_rates.names.append('future_quarter')

    discount_factor = NamedArray(names=interest_rates.names.copy(),
                                 data=(1 + interest_rates.data))
    # loop over future quarters and pu the discount factor to the power of the quarter
    for quarter in range(n_scenario_quarters):
        discount_factor.data[..., quarter] = discount_factor.data[..., quarter] ** (quarter + 0.5)
    discount_factor.data = discount_factor.data[np.newaxis, :, :]
    discount_factor.names = ['scenario'] + discount_factor.names
    validate_named_array(discount_factor)

    # -------------------------------------------------------------------
    # Perform the ECL calculation
    # -------------------------------------------------------------------

    # Step 1: multiply the flow to stage 3 by the LGD (selecting stage 1 and 2 LGDs)
    if lgd.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage5', 'future_quarter']:
        raise ValueError(f'Names of lgd are not as expected')
    if flow_to_stage_3.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage2', 'future_quarter']:
        raise ValueError(f'Names of flow_to_stage_3 are not as expected')
    s2_ecl = NamedArray(names=flow_to_stage_3.names.copy(),
                        data=flow_to_stage_3.data * lgd.data[..., :2, :])
    validate_named_array(s2_ecl)

    # Step 2: sum over stages (1 and 2) and collateral types
    s2_ecl.data = np.nansum(s2_ecl.data, axis=4)
    s2_ecl.data = np.nansum(s2_ecl.data, axis=3)
    s2_ecl.names = s2_ecl.names[:3] + s2_ecl.names[5:]
    validate_named_array(s2_ecl)

    # Step 3: multiply by the discount factor, then sum over future quarters
    if s2_ecl.names != ['scenario', 'bank', 'portfolio_type', 'future_quarter']:
        raise ValueError(f'Names of s2_ecl are not as expected')
    s2_ecl.data = np.nansum(s2_ecl.data / discount_factor.data, axis=3)
    s2_ecl.names = s2_ecl.names[:3]
    validate_named_array(s2_ecl)

    # Compute ECL rates by dividing by the full S1 exposure (sum over coll_type, select stage 2 and quarter 0)
    if s2_exposure.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage4m', 'future_snapshot']:
        raise ValueError(f'Names of s2_exposure are not as expected')
    full_s2_exp_data = np.nansum(s2_exposure.data[..., 1, 0], axis=3)
    full_s2_exposure = NamedArray(names=s2_exposure.names[:3].copy(),
                                  data=full_s2_exp_data)
    validate_named_array(full_s2_exposure)

    # define relative ECL
    s2_ecl_rel = NamedArray(names=s2_ecl.names.copy(),
                            data=s2_ecl.data / full_s2_exposure.data)

    if s2_ecl.names != ['scenario', 'bank', 'portfolio_type']:
        raise ValueError(f'Names of s2_ecl are not as expected')

    # Save the stage 2 ECL in the output array
    cl_scenarios.data[:, :, :, 1] = s2_ecl.data
    cl_rate_scenarios.data[:, :, :, 1] = s2_ecl_rel.data

    return cl_scenarios, cl_rate_scenarios
