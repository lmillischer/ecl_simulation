import numpy as np

from scripts.vectors_matrices.named_array import *


def compute_stage_1and2_ecl(ecl_scenarios_abs, ecl_scenarios_rel, current_portfolios, tm_scenarios_4x4,
                            lgd, interest_rates, stage1or2, lifetime_losses, verbose=False):

    # stage1or2 can only be 1 or 2
    if stage1or2 not in [1, 2]:
        raise ValueError(f'stage1or2 must be 1 or 2, not {stage1or2}')

    # lifetime_losses can only be True or False and for stage 2 must be True
    if lifetime_losses not in [True, False]:
        raise ValueError(f'lifetime_losses must be True or False, not {lifetime_losses}')
    if stage1or2 == 2:
        lifetime_losses = True

    # tm_scenarios_4x4_s1or2 will be manipulated, so just take a copy
    if tm_scenarios_4x4.names != ['scenario', 'bank', 'portfolio_type', 'future_quarter', 'stage4m', 'stage4m']:
        raise ValueError(f'Names of tm_scenarios_4x4 are not as expected')

    tm_scenarios_4x4_s1or2 = NamedArray(names=tm_scenarios_4x4.names.copy(),
                                        data=tm_scenarios_4x4.data.copy())
    # check if tm_scenarios_4x4_s1or2 contains nans
    if np.all(np.isnan(tm_scenarios_4x4_s1or2.data)):
        raise ValueError(f'tm_scenarios_4x4_s1or2 all nans')

    # to avoid double counting set the recovery rates from stage 3 to stages 1 and 2 to 0
    #    otherwise loans could default (to stage 3), generate a loss, then cure and default again
    tm_scenarios_4x4_s1or2.data[..., 2, :] = [0, 0, 1, 0]

    # Get the stage 1 or 2 exposure
    if current_portfolios.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage5']:
        raise ValueError(f'Names of current_portfolios are not as expected')

    s1or2_exposure = NamedArray(names=current_portfolios.names[:4].copy(),
                                data=current_portfolios.data[:, :, :, :, stage1or2-1].copy())
    validate_named_array(s1or2_exposure)

    # Get the LCU flows to stage 3 from exposures initially in stage 1/2
    #  step 0: duplicat the s1or2_exposure along the first axis to have it shape (n_scearios, ....)
    s1or2_exposure.data = np.repeat(s1or2_exposure.data, n_scenarios, axis=0)
    validate_named_array(s1or2_exposure)
    #  step 1: add a dimension for the stage (size 4) and put 0s for all except stage 1 or 2
    s1or2_exposure.data = s1or2_exposure.data[..., np.newaxis]
    if stage1or2 == 1:
        s1or2_exposure.data = np.concatenate((s1or2_exposure.data,
                                              np.zeros(s1or2_exposure.data.shape),
                                              np.zeros(s1or2_exposure.data.shape),
                                              np.zeros(s1or2_exposure.data.shape)), axis=4)
    elif stage1or2 == 2:
        s1or2_exposure.data = np.concatenate((np.zeros(s1or2_exposure.data.shape),
                                           s1or2_exposure.data,
                                           np.zeros(s1or2_exposure.data.shape),
                                           np.zeros(s1or2_exposure.data.shape)), axis=4)

    s1or2_exposure.names.append('stage4m')
    validate_named_array(s1or2_exposure)
    #  step 2: add a dimension for future quarters and put 0s for all except quarter 0
    s1or2_exposure.data = s1or2_exposure.data[..., np.newaxis]
    future_quarters_shape = list(s1or2_exposure.data.shape)
    future_quarters_shape[-1] = n_scenario_quarters
    futures_quarters_with_zeros = np.zeros(tuple(future_quarters_shape))
    s1or2_exposure.data = np.concatenate((s1or2_exposure.data, futures_quarters_with_zeros), axis=5)
    s1or2_exposure.names.append('future_snapshot')
    validate_named_array(s1or2_exposure)

    if s1or2_exposure.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage4m', 'future_snapshot']:
        raise ValueError(f'Names of s1or2_exposure are not as expected')

    # Check that all the exposure in stages other than s1/s2 quarter 0 are 0 (either 2-3-M or 1-3-M)
    stages_that_must_be_zero = [1, 2, 3] if stage1or2 == 1 else [0, 2, 3]
    for stage in stages_that_must_be_zero:
        if np.any(s1or2_exposure.data[:, :, :, :, stage, 0] != 0):
            raise ValueError(f'There are exposures in stage {stage} in quarter 0')

    # step 3: perform matrix multiplication of the exposure by the transition matrix
    #    add a dimension for the collateral stat_type (size 4)
    tm_scenarios_4x4_s1or2.data = tm_scenarios_4x4_s1or2.data[:, :, :, np.newaxis, ...]
    tm_scenarios_4x4_s1or2.names = tm_scenarios_4x4_s1or2.names[:3] + ['collateral_type'] + tm_scenarios_4x4_s1or2.names[3:]
    validate_named_array(tm_scenarios_4x4_s1or2)
    #    move future_quarter dimension to the end
    tm_scenarios_4x4_s1or2.data = np.moveaxis(tm_scenarios_4x4_s1or2.data, 4, -1)
    tm_scenarios_4x4_s1or2.names = tm_scenarios_4x4_s1or2.names[:4] + tm_scenarios_4x4_s1or2.names[5:] + ['future_quarter']
    validate_named_array(tm_scenarios_4x4_s1or2)

    # define the flow to stage 3 array (caution: future_quarter instead of future_snapshot)
    fts3_names = s1or2_exposure.names.copy()[:-1] + ['future_quarter']
    flow_to_stage_3 = NamedArray(names=fts3_names,
                                 data=np.zeros(s1or2_exposure.data.shape[:-1] + (n_scenario_quarters,)))
    # we are only interested in the first two stages (they are the only one that can default)
    #    so keep only [:2] of the 5th dimension
    flow_to_stage_3.data = flow_to_stage_3.data[..., :2, :]
    flow_to_stage_3.names[4] = 'stage2'
    validate_named_array(flow_to_stage_3)

    # loop over future quarters, for each quarter fill the s1or2_exposure (last dimension) with the result of the
    #    multiplication of 4x4 transition matrix by the 4x1 exposure vector
    quarters_to_loop_over = n_scenario_quarters if lifetime_losses else 4  # if not lifelong losses then 1Y = 4Q losses
    for quarter in range(quarters_to_loop_over):
        # make matrix multiplication to get the stock in the next quarter, save that in s1or2_exposure
        s1or2_exposure.data[..., quarter+1] = np.einsum('ijklm,ijklmn->ijkln', s1or2_exposure.data[..., quarter],
                                                  tm_scenarios_4x4_s1or2.data[..., quarter])
        # NB: last dimension of s1or2_exposure is future_snapshot, last dimension of tm_scenarios_4x4_s1or2 is
        # future_quarter therefore getting from s2_exp[y] to s2_exp[y+1] one considers tm_scen[y] for multiplication

        # for each quarter multiply the stock of stage 1 and stage 2 by the TM elements T13 and T23 and in flow_to_stage_3
        flow_to_stage_3.data[..., quarter] \
            = s1or2_exposure.data[..., :2, quarter] * tm_scenarios_4x4_s1or2.data[..., :2, 2, quarter]

        # check if flow_to_stage_3 contains nans
        if np.all(np.isnan(flow_to_stage_3.data[..., quarter])):
            if np.all(np.isnan(s1or2_exposure.data[..., :2, quarter])):
                raise ValueError(f'flow_to_stage_3 all nans because of exposure')
            elif np.all(np.isnan(tm_scenarios_4x4_s1or2.data[..., :2, 2, quarter])):
                raise ValueError(f'flow_to_stage_3 all nans because of TM')
            else:
                raise ValueError(f'flow_to_stage_3 all nans')

    # Define discount factor (for that make a copy of interest rates, not to change the original)
    ir_use = NamedArray(names=interest_rates.names.copy(),
                        data=interest_rates.data.copy())
    # add new dimension for the future quarter, repeat the interest rates along that dimension
    ir_use.data = ir_use.data[:, :, np.newaxis]
    ir_use.data = np.repeat(ir_use.data, n_scenario_quarters, axis=-1)
    ir_use.names.append('future_quarter')

    discount_factor = NamedArray(names=ir_use.names.copy(),
                                 data=(1 + ir_use.data))
    # loop over future quarters and pu the discount factor to the power of the quarter
    for quarter in range(n_scenario_quarters):
        discount_factor.data[..., quarter] = discount_factor.data[..., quarter] ** (quarter + 0.5)
    discount_factor.data = discount_factor.data[np.newaxis, :, :]
    discount_factor.names = ['scenario'] + discount_factor.names
    validate_named_array(discount_factor)

    # check if discount_factor contains nans
    if np.all(np.isnan(discount_factor.data)):
        raise ValueError(f'discount_factor all nans')
    elif np.all(discount_factor.data == 0):
        raise ValueError(f'discount_factor all zeros')

    # -------------------------------------------------------------------
    # Perform the ECL calculation
    # -------------------------------------------------------------------

    # Step 1: multiply the flow to stage 3 by the LGD (selecting stage 1 and 2 LGDs)
    if lgd.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage5', 'future_quarter']:
        raise ValueError(f'Names of lgd are not as expected')
    if flow_to_stage_3.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage2', 'future_quarter']:
        raise ValueError(f'Names of flow_to_stage_3 are not as expected')
    s1or2_ecl = NamedArray(names=flow_to_stage_3.names.copy(),
                        data=flow_to_stage_3.data * lgd.data[..., :2, :])
    validate_named_array(s1or2_ecl)

    # Step 2: sum over stages (1 and 2) and collateral types
    s1or2_ecl.data = np.nansum(s1or2_ecl.data, axis=4)
    s1or2_ecl.data = np.nansum(s1or2_ecl.data, axis=3)
    s1or2_ecl.names = s1or2_ecl.names[:3] + s1or2_ecl.names[5:]
    validate_named_array(s1or2_ecl)

    # Step 3: multiply by the discount factor, then sum over future quarters
    if s1or2_ecl.names != ['scenario', 'bank', 'portfolio_type', 'future_quarter']:
        raise ValueError(f'Names of s1or2_ecl are not as expected')
    s1or2_ecl.data = np.nansum(s1or2_ecl.data / discount_factor.data, axis=3)
    s1or2_ecl.names = s1or2_ecl.names[:3]
    validate_named_array(s1or2_ecl)
    if np.all(np.isnan(s1or2_ecl.data)):
        raise ValueError(f's1or2_ecl all nans')
    # check if s1or2_ecl contains inf
    if np.any(np.isinf(s1or2_ecl.data)):
        raise ValueError(f's1or2_ecl contains inf')

    # Compute ECL rates by dividing by the full S1 exposure (sum over coll_type, select stage 2 and quarter 0)
    if s1or2_exposure.names != ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage4m', 'future_snapshot']:
        raise ValueError(f'Names of s1or2_exposure are not as expected')
    full_s1or2_exp_data = np.nansum(s1or2_exposure.data[..., stage1or2-1, 0], axis=3)
    full_s1or2_exposure = NamedArray(names=s1or2_exposure.names[:3].copy(),
                                     data=full_s1or2_exp_data)
    validate_named_array(full_s1or2_exposure)
    # check if full_s1or2_exposure contains 0
    if np.all(full_s1or2_exposure.data == 0):
        raise ValueError(f'full_s1or2_exposure all zeros')

    # define relative ECL
    s1o2_ecl_rel = NamedArray(names=s1or2_ecl.names.copy(),
                              data=s1or2_ecl.data / full_s1or2_exposure.data)
    # check if s1o2_ecl_rel contains inf
    if np.any(np.isinf(s1o2_ecl_rel.data)):
        raise ValueError(f's1o2_ecl_rel contains inf')

    if s1or2_ecl.names != ['scenario', 'bank', 'portfolio_type']:
        raise ValueError(f'Names of s1or2_ecl are not as expected')

    # Save the stage 2 ECL in the output array
    ecl_scenarios_abs.data[:, :, :, stage1or2-1] = s1or2_ecl.data
    ecl_scenarios_rel.data[:, :, :, stage1or2-1] = s1o2_ecl_rel.data

    return ecl_scenarios_abs, ecl_scenarios_rel
