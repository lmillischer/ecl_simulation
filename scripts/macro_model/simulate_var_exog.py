import numpy as np

from scripts.macro_model.check_var_stability import var_is_stable
from scripts.vectors_matrices.named_array import *
import time

from controls import *


# Simulate macro models
def simulate_var(init_cond, use_exog, exog_scenarios, init_exog,
                 coefs, coef_vcov, res_vcov, non_neg_vars,
                 var_order, n_scenario_quarters, n_scenarios,
                 do_coef_uncertainty=True, do_resid_uncertainty=True,
                 verbose=False):

    # Determine the number of macro variables from intial conditions
    n_endog_vars = init_cond.shape[1]
    n_exog_vars = int((coefs.shape[0] - 1)/var_order - coefs.shape[1])

    # Check if we have valid exogeneous scenarios and initial conditions
    if use_exog:
        if np.any(np.isnan(exog_scenarios)):
            raise ValueError('Exogeneous scenarios needed')
        if exog_scenarios.shape != (n_scenarios, n_scenario_quarters, n_exog_vars):
            raise ValueError('Exogeneous scenarios have wrong shape.')
        init_exog = np.array(init_exog)
        if np.any(np.isnan(init_exog)):
            raise ValueError('Initial exogeneous conditions needed')
        # if init_exog.shape != (var_order_exog, n_exog_vars):
        #     raise ValueError(f'Initial exogeneous conditions have wrong shape: '
        #                      f'{init_exog.shape} instead of {(var_order, n_exog_vars)}')

        # now add the initial conditions on top of the exogeneous scenarios
        # repeat init_exog n_scenarios times along a newly created first axis
        init_exog = np.repeat(init_exog[np.newaxis, :, :], n_scenarios, axis=0)
        exog_scenarios = np.concatenate((init_exog, exog_scenarios), axis=1)

    # Create a nan array to store the simulation results
    #  it should be of dimensions n_scenarios, n_scenario_quarters + var_order, n_endog_vars
    macro_scenario_array \
        = np.full((n_scenarios, n_scenario_quarters + var_order, n_endog_vars),
                  fill_value=np.NaN)

    # Loop through simulation runs
    for r in range(n_scenarios):
        if verbose and r % 100 == 0:
            print(f'    Simulating scenario {r}')

        # If do_coef_uncertainty is True, draw a random set of coefficients
        if do_coef_uncertainty:
            # Draw a random set of coefficients and use them if the VAR is stable
            stable_drawn_coefficients = False
            while not stable_drawn_coefficients:
                coefs_drawn = (np.random.multivariate_normal(np.array(coefs).flatten(), np.array(coef_vcov))
                               .reshape(coefs.shape))
                # now we test the stability of the drawn coefficients
                if not use_exog:
                    # if no exogeneous variables, no coefficients need to be dropped for the test
                    stable_drawn_coefficients = var_is_stable(coefs_drawn)[0]
                else:
                    # if exogeneous variables, we need to drop the exogeneous coefficients
                    exog_idx = list(range(1, len(coefs)))
                    for nv in coefs.columns:
                        for icn, cn in enumerate(coefs.index):
                            if nv in cn:
                                exog_idx.remove(icn)
                    # drop lines in drawn coefficients corresponding to exog_idx
                    coefs_drawn_endog = np.delete(coefs_drawn, exog_idx, axis=0)
                    stable_drawn_coefficients = var_is_stable(coefs_drawn_endog)[0]
            coefs_to_use = coefs_drawn
        else:
            coefs_to_use = coefs

        # Initialize the first var_order rows of the output matrices with the initial conditions
        macro_scenario_array[r, 0:var_order, ] = init_cond

        # Loop through periods for simulation
        for h in range(var_order - 1, var_order + n_scenario_quarters - 1):
            # Create an array to store lagged values
            x = np.array([])
            for p in range(1, var_order + 1):
                x = np.append(x, macro_scenario_array[r, h + 1 - p, :])
                x = np.append(x, exog_scenarios[r, h + 1 - p, :]) if use_exog else x

            # Calculate the value for the next period using the VAR model equation
            next_quarter = np.dot(np.append(1, x), coefs_to_use)

            # Noise term sampled from a multivariate normal distribution
            if do_resid_uncertainty:
                noise = np.random.multivariate_normal(np.zeros(n_endog_vars), res_vcov)
            else:
                noise = np.zeros(n_endog_vars)

            # Calculate the value for the current period
            macro_scenario_array[r, h + 1, :] = next_quarter + noise

    # Trim the output array to remove historical data which we needed only to project the VAR forward
    macro_scenario_array = macro_scenario_array[:, var_order:, :]

    # Ensure variables flagged as "non-negative" are not negative
    for idx in non_neg_vars:
        macro_scenario_array[:, :, idx] = np.maximum(macro_scenario_array[:, :, idx], 0)

    # Return the simulated data
    return macro_scenario_array


# Main function called from main.py returning scenarios
def simulate_scenarios_exog(macro_data, macro_names, macro_names_print, future_macro_path,
                            coefs, coef_vcov, residual_vcov,
                            coefs_exog, coef_vcov_exog, residual_vcov_exog, exog_vars,
                            var_order_endog, var_order_exog, n_scenario_quarters, n_scen,
                            do_cycle_neutral,
                            do_coef_uncertainty, do_resid_uncertainty,
                            verbose=True, do_macro_var=True):

    out_path = f'data/output/{first_hash}/'

    # If scenarios to be simulted from scratch
    if do_macro_var:

        if verbose:
            print(f'    Simulating {n_scen} macro scenarios ...')
        start_time = time.time()

        # transform first most recent var_order rows of macro_data into initial conditions array
        if not do_cycle_neutral:
            init_cond_x = macro_data.sort_index().tail(var_order_exog)
            init_cond_exog = np.array(init_cond_x[exog_vars])
            init_cond_n = macro_data.sort_index().tail(var_order_endog)
            init_cond_endo = np.array(init_cond_n.drop(columns=exog_vars))
        else:
            # if we compute cycle neutral losses, we use as initial variables the average macro variables
            print('macro_data', macro_data.shape)
            init_cond_x = np.mean(np.array(macro_data[exog_vars]), axis=0)[np.newaxis, :]
            init_cond_exog = np.repeat(init_cond_x, var_order_exog, axis=0)
            init_cond_n = np.mean(np.array(macro_data.drop(columns=exog_vars)), axis=0)[np.newaxis, :]
            init_cond_endo = np.repeat(init_cond_n, var_order_endog, axis=0)

        # Now follow these steps:
        #  1 - simulate exogeneous variables
        #  2 - recenter exogeneous variables
        #  3 - simulate endogeneous variables with exog as input
        #  4 - recenter endogeneous variables

        # STEP 1 - Simulate exogeneous scenarios

        # non-negative variable:
        nn_idx_exog = [exog_vars.index('FFR')]

        scenarios_exog_not_recentered \
            = simulate_var(init_cond_exog, use_exog=False, exog_scenarios=np.nan, init_exog=None,
                           coefs=coefs_exog, coef_vcov=coef_vcov_exog,
                           res_vcov=residual_vcov_exog, non_neg_vars=nn_idx_exog,
                           var_order=var_order_exog, n_scenario_quarters=n_scenario_quarters,
                           n_scenarios=n_scen, do_coef_uncertainty=do_coef_uncertainty,
                           do_resid_uncertainty=do_resid_uncertainty,
                           verbose=False)

        # STEP 2 - Recenter exogeneous scenarios
        future_macro_path_exog = future_macro_path[exog_vars]
        if not do_cycle_neutral:
            scenarios_exog_recentered = recenter_scenarios(scenarios_exog_not_recentered, future_macro_path_exog, exog_vars)
        else:
            # if we simulate cycle neutral macro paths, we do not recenter
            scenarios_exog_recentered = scenarios_exog_not_recentered.copy()

        # STEP 3 - Simulate endogeneous scenarios with exogeneous as input
        nn_idx_endog = [macro_names.index('UR'), macro_names.index('STR')]  # non-negative variables

        scenarios_endog_not_recentered \
            = simulate_var(init_cond_endo, use_exog=True, exog_scenarios=scenarios_exog_recentered,
                           init_exog=init_cond_exog,
                           coefs=coefs, coef_vcov=coef_vcov,
                           res_vcov=residual_vcov, non_neg_vars=nn_idx_endog,
                           var_order=var_order_endog, n_scenario_quarters=n_scenario_quarters,
                           n_scenarios=n_scen, do_coef_uncertainty=do_coef_uncertainty,
                           do_resid_uncertainty=do_resid_uncertainty,
                           verbose=verbose)

        time_elapsed = time.time() - start_time
        if verbose:
            print(f'    Simulated scenarios in {time_elapsed:.0f} seconds')

        # STEP 4 - Recenter the scenarios
        future_macro_path_endog = future_macro_path.drop(columns=exog_vars)
        if not do_cycle_neutral:
            scenarios_endog_recentered = recenter_scenarios(scenarios_endog_not_recentered, future_macro_path_endog,
                                                  future_macro_path_endog.columns)
        else:
            # as above if we simulate cycle neutral scenarios, we do not recenter
            scenarios_endog_recentered = scenarios_endog_not_recentered.copy()

        if verbose:
            print('    Recentering done')

        # add the delta variable (e.g. of unemplyoment)
        variables_to_be_deltaed = ['UR']
        scenarios_endog_recentered, macro_names, macro_names_print \
            = add_delta_variable(scenarios_endog_recentered, variables_to_be_deltaed,
                                 macro_names, macro_names_print)
        scenarios_endog_not_recentered, _, _ \
            = add_delta_variable(scenarios_endog_not_recentered, variables_to_be_deltaed,
                                 macro_names.copy(), macro_names_print.copy())
        macro_data_numpy = np.array(macro_data)[np.newaxis, :, :]
        avg_macro_data_numpy = np.nanmean(np.array(macro_data), axis=0)[np.newaxis, np.newaxis, :]
        # we averaged over past quarters, so we repeat it n_quarters times
        avg_macro_data_numpy = np.repeat(avg_macro_data_numpy, macro_data_numpy.shape[1], axis=1)
        macro_data_enhanced, _, _ = add_delta_variable(macro_data_numpy, variables_to_be_deltaed,
                                                       macro_names.copy(), macro_names_print.copy())
        average_data_enhanced, _, _ = add_delta_variable(avg_macro_data_numpy, variables_to_be_deltaed,
                                                         macro_names.copy(), macro_names_print.copy())

        if verbose:
            print('    Added delta variable')

        # Concatenate engogenous and exogenous variables
        scenarios_full_recentered = np.concatenate((scenarios_endog_recentered, scenarios_exog_recentered), axis=2)
        scenarios_full_not_recentered = np.concatenate((scenarios_endog_not_recentered, scenarios_exog_not_recentered), axis=2)

        # add the annual change variables
        variables_to_be_annualized = ['RGDPQ', 'INFQ', 'HPQ', 'FXQ', 'WAGQ', 'OILQ']
        scenarios_full_recentered, _, _ \
            = add_annual_variable(scenarios_full_recentered, variables_to_be_annualized,
                                  macro_names.copy(), macro_names_print.copy())
        scenarios_full_not_recentered, _, _ = add_annual_variable(scenarios_full_not_recentered, variables_to_be_annualized,
                                                                  macro_names.copy(), macro_names_print.copy())
        average_data_enhanced, _, _ = add_annual_variable(average_data_enhanced, variables_to_be_annualized,
                                                          macro_names.copy(), macro_names_print.copy())
        macro_data_enhanced, macro_names, macro_names_print = add_annual_variable(macro_data_enhanced, variables_to_be_annualized,
                                                                                  macro_names, macro_names_print)
        average_data_enhanced = average_data_enhanced[0, :, :]
        macro_data_enhanced = macro_data_enhanced[0, :, :]

        # Define NamedArray
        scenarios_full_recentered = NamedArray(names=['scenario', 'future_quarter', 'macro_var'],
                                          data=scenarios_full_recentered)
        scenarios_full_not_recentered = NamedArray(names=['scenario', 'future_quarter', 'macro_var'],
                                                  data=scenarios_full_not_recentered)
        macro_data_enhanced = NamedArray(names=['historical_macro_quarter', 'macro_var'], data=macro_data_enhanced)
        average_data_enhanced = NamedArray(names=['historical_macro_quarter', 'macro_var'], data=average_data_enhanced)

        # Save the macro scenarios and macro_names in an h5 file
        save_named_array(f'{out_path}/macro_scenarios.h5', scenarios_full_recentered, 'macro_scenarios')
        save_named_array(f'{out_path}/macro_scenarios.h5', scenarios_full_not_recentered,
                         'macro_scenarios_before_recentering')
        save_named_array(f'{out_path}/macro_scenarios.h5', macro_data_enhanced, 'macro_data_enhanced')
        save_named_array(f'{out_path}/macro_scenarios.h5', average_data_enhanced, 'average_data_enhanced')
        with h5py.File(f'{out_path}/macro_names.h5', 'w') as h5file:
            h5file.create_dataset('macro_names', data=macro_names)
            h5file.create_dataset('macro_names_print', data=macro_names_print)

    # If scenarios to be read from file
    else:
        scenarios_full_recentered = load_named_array(f'{out_path}/macro_scenarios.h5', 'macro_scenarios')
        scenarios_full_not_recentered \
            = load_named_array(f'{out_path}/macro_scenarios.h5', 'macro_scenarios_before_recentering')
        macro_data_enhanced = load_named_array(f'{out_path}/macro_scenarios.h5', 'macro_data_enhanced')
        average_data_enhanced = load_named_array(f'{out_path}/macro_scenarios.h5', 'average_data_enhanced')
        with h5py.File(f'{out_path}/macro_names.h5', 'r') as h5file:
            macro_names = h5file['macro_names'][()]
            macro_names = [byte.decode('utf-8') for byte in macro_names]
            macro_names_print = h5file['macro_names_print'][()]
            macro_names_print = [byte.decode('utf-8') for byte in macro_names_print]

    # Check before returning
    validate_named_array(scenarios_full_not_recentered)
    validate_named_array(scenarios_full_recentered)
    validate_named_array(macro_data_enhanced)
    validate_named_array(average_data_enhanced)

    return (scenarios_full_recentered, scenarios_full_not_recentered, macro_data_enhanced, average_data_enhanced,
            macro_names, macro_names_print)


# Function to recenter scenarios around a given path
def recenter_scenarios(scenarios, manual_future_macro_path, macro_names, q_hori=n_scenario_quarters):
    # Check if columns of hand_input match macro_names
    if list(manual_future_macro_path.columns) != list(macro_names):
        print('macro_names                     ', macro_names)
        print('manual_future_macro_path.columns', list(manual_future_macro_path.columns))
        raise ValueError('The columns of manual_future_macro_path do not match macro_names')

    # compute the mean of all variables over all scenarios
    mean_scenarios = np.mean(scenarios, axis=0)

    # if the length of future_macro_path is smaller than n_horizon, raise an error
    if len(manual_future_macro_path) < q_hori:
        raise ValueError(f'The length of future_macro_path ({len(manual_future_macro_path)}) lower than n_horizon ({q_hori})')

    # remove the last rows of future_macro_path if its length is greater than n_horizon
    if len(manual_future_macro_path) > q_hori:
        manual_future_macro_path = manual_future_macro_path.head(q_hori)
    elif len(manual_future_macro_path) < q_hori:
        raise ValueError('The length of future_macro_path lower than n_horizon')

    # We will sustract the mean and add the recentered scenarios
    future_macro_array = np.array(manual_future_macro_path.values)

    # For that, whenever no value is given for the recentered scenario, we wil use that
    #   of the mean so no recentering is done, then only a subset is recentered

    # replace nan values in future_macro_array by equivalent value in mean_scenarios
    future_macro_array[np.isnan(future_macro_array)] = mean_scenarios[np.isnan(future_macro_array)]

    # add fake scenario dimension to future_macro_array and mean to be able to broadcast
    future_macro_array = future_macro_array[np.newaxis, :, :]
    mean_scenarios = mean_scenarios[np.newaxis, :, :]

    # recenter the scenarios (demeaning + adding the future path)
    recentered_scenarios = scenarios - mean_scenarios + future_macro_array

    return recentered_scenarios


def add_delta_variable(scenarios_before_delta, variables_to_be_delta, macro_names, macro_names_print):
    """Adds a variable to the scenarios_y array that is the delta of the variable_to_be_delta variable"""

    # check types
    if type(macro_names) != list:
        raise ValueError('macro_names is not a list')
    if type(macro_names_print) != list:
        raise ValueError('macro_names_print is not a list')

    # check dimension of scenarios_before_delta
    if scenarios_before_delta.ndim != 3:
        raise ValueError('scenarios_before_delta is not of dimension 3')

    # get the index of the variables to be delta'ed, i.e. diff taken (macro_names a numpy index)
    for var2d in variables_to_be_delta:
        idx_to_be_delta = macro_names.index(var2d)

        delta = np.diff(scenarios_before_delta[:, :, idx_to_be_delta], axis=1)
        delta = np.hstack((np.full((delta.shape[0], 1), np.nan), delta))

        scenarios_before_delta = np.insert(scenarios_before_delta, idx_to_be_delta + 1, delta, axis=2)

        # insert the name of the delta variable in macro_names and macro_names_print
        macro_names.insert(idx_to_be_delta + 1, f'D{var2d}')
        # find the index of the variable in macro_names_print
        print_name_to_be_delta = macro_names_print[macro_names.index(var2d)]
        macro_names_print.insert(idx_to_be_delta + 1, f'Delta {print_name_to_be_delta}')

    return scenarios_before_delta, macro_names, macro_names_print


def add_annual_variable(scnearios_before_annual, variables_to_be_annualized, macro_names, macro_names_print):
    """Adds a variable to the scenarios_y array that is the annual change of the variable_to_be_annualized variable"""

    macro_names_loc = macro_names.copy()

    # check types
    if type(macro_names_loc) != list:
        raise ValueError('macro_names is not a list')
    if type(macro_names_print) != list:
        raise ValueError('macro_names_print is not a list')

    # check if macro_names is the same length as scenarios_before_annual.shape[2]
    if len(macro_names_loc) != scnearios_before_annual.shape[2]:
        raise ValueError('macro_names is not the same length as scenarios_before_annual.shape[2]')

    # check dimension of scnearios_before_annual: scenarios, future_quarters, variables
    if scnearios_before_annual.ndim != 3:
        raise ValueError('scnearios_before_annual is not of dimension 3')

    scenarios_after_annual = scnearios_before_annual.copy()

    for var2a in variables_to_be_annualized:
        # get the index of the variables to be annualized (macro_names a numpy index)
        idx_to_be_ann = macro_names_loc.index(var2a)

        # compute level of the variable to be annualized
        level = np.cumprod(1 + scenarios_after_annual[:, :, idx_to_be_ann] / 100, axis=1)
        level_last_year = np.roll(level, 4, axis=1)
        # we don't know the past values, so we can't compute the annual change for the first 4 quarters
        # we start by computing the trend by taking the ratio of the first and last observation of levels
        trend = (level[:, -1] / level[:, 0]) ** (1 / (level.shape[1] - 1))
        # we then append four observations at the beginning of level_last year
        for i in range(4, -1, -1):
            level_last_year[:, i] = level_last_year[:, i + 1] / trend
        annual_change = 100 * (level / level_last_year - 1)

        scenarios_after_annual = np.insert(scenarios_after_annual, idx_to_be_ann + 1, annual_change, axis=2)

        # insert the name of the delta variable in macro_names and macro_names_print
        macro_names_loc.insert(idx_to_be_ann + 1, f'{var2a.replace("Q", "A")}')
        # find the index of the variable in macro_names_print
        print_name_to_be_delta = macro_names_print[macro_names_loc.index(var2a)]
        macro_names_print.insert(idx_to_be_ann + 1, f'{print_name_to_be_delta} annual change')

    return scenarios_after_annual, macro_names_loc, macro_names_print


def q_to_y(scenarios_q, history):
    print('Transforming quarterly scenarios to yearly scenarios ...')

    # Check scenarios_q is of dimension 3
    if scenarios_q.ndim != 3:
        raise ValueError('scenarios_q is not of dimension 3')

    # Check that the number of columns in scenarios_q + history is a multiple of 4
    if (scenarios_q.shape[1] + history.shape[0]) % 4 != 0:
        raise ValueError('The number of columns in scenarios_q is not a multiple of 4')

    # Check the number of columns in history is equal to the number of columns in scenarios_q
    if history.shape[1] != scenarios_q.shape[2]:
        raise ValueError('The number of columns in history is not equal to the number of columns in scenarios_q')

    # append history on top of scenario
    history = np.tile(history, (scenarios_q.shape[0], 1, 1))
    scenarios_q = np.concatenate((history, scenarios_q), axis=1)

    # function taking means of chunks of 4 rows (for stock variables)
    def yearly_means_3d(array):
        # Check if the second dimension is divisible by 4 quarters
        if array.shape[1] % 4 != 0:
            raise ValueError("The size of the second dimension should be divisible by 4.")

        # Calculate the new shape
        new_shape = (array.shape[0], -1, 4, array.shape[2])
        reshaped_array = array.reshape(new_shape)

        # Compute the means along the new axis
        return np.mean(reshaped_array, axis=2)

    # function calculating the year on year growth rate of the annual sum (for flow variables)
    def yearly_level_growth_3d(array):
        # Convert to logarithmic differences
        log_diffs = array / 100

        # Calculate the cumulative sum of the logarithmic differences along axis 1
        cumulative_log_diffs = np.cumsum(log_diffs, axis=1)

        # Start from ln(100) and add the cumulative logarithmic differences
        loglevels = np.log(100) + cumulative_log_diffs

        # Apply exp to get the final transformed values
        q_levels = np.exp(loglevels)

        # Check if the second dimension is divisible by 4
        if q_levels.shape[1] % 4 != 0:
            raise ValueError("The size of the second dimension should be divisible by 4.")

        # Reshape the array for summing in chunks of 4 along the second axis
        new_shape = (q_levels.shape[0], -1, 4, q_levels.shape[2])
        reshaped_q_levels = q_levels.reshape(new_shape)

        # Compute the sum of each chunk
        y_levels = np.sum(reshaped_q_levels, axis=2)

        # Calculate the growth rate
        y_growth = y_levels[:, 1:, :] / y_levels[:, :-1, :] - 1

        return y_growth * 100

    # convert quarterly scenarios to yearly scenarios using proper methodology for each variable
    rgdp_infq = yearly_level_growth_3d(scenarios_q[:, :, :2])
    urx_sov3_ts = yearly_means_3d(scenarios_q[:, :, 2:5])[:, 1:, :]
    rppq_fxq_wq = yearly_level_growth_3d(scenarios_q[:, :, 5:8])

    # yearly scneario the horizontal combination of rgdp_infq, urx_sov3_ts, rppq_fxq_wq
    scenarios_y = np.concatenate((rgdp_infq, urx_sov3_ts, rppq_fxq_wq), axis=2)

    # check if scenarios_y is of the right shape
    n_scenarios = scenarios_q.shape[0]
    n_future_years = (scenarios_q.shape[1] - 1) // 4
    n_macro_vars = scenarios_q.shape[2]
    if scenarios_y.shape != (n_scenarios, n_future_years, n_macro_vars):
        raise ValueError('scenarios_y is not of the right shape')

    return scenarios_y