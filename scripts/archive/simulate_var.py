
from scripts.macro_model.check_var_stability import var_is_stable
from scripts.vectors_matrices.named_array import *


# Simulate macro models
def simulate_var(init_cond, coefs, coef_vcov, res_vcov, var_order, n_scenario_quarters, n_scenarios,
                 do_coef_uncertainty=True, verbose=False):
    # Determine the number of macro variables from intial conditions
    n_macro_vars = init_cond.shape[1]

    # Create a nan array to store the simulation results
    #  it should be of dimensions n_scenarios, n_scenario_quarters + var_order, n_macro_vars
    macro_scenario_array = np.full((n_scenarios, n_scenario_quarters + var_order, n_macro_vars),
                                   fill_value=np.NaN)

    # Loop through simulation runs
    for r in range(n_scenarios):
        if verbose:
            print(f'    Simulating scenario {r}')

        # If do_coef_uncertainty is True, draw a random set of coefficients
        if do_coef_uncertainty:
            # Draw a random set of coefficients and use them if the VAR is stable
            stable_drawn_coefficients = False
            while not stable_drawn_coefficients:
                coefs_drawn = (np.random.multivariate_normal(np.array(coefs).flatten(), coef_vcov)
                               .reshape(coefs.shape))
                stable_drawn_coefficients = var_is_stable(coefs_drawn)[0]
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

            # Calculate the value for the next period using the VAR model equation
            next_quarter = np.dot(np.append(1, x), coefs_to_use)

            # Noise term sampled from a multivariate normal distribution
            noise = np.random.multivariate_normal(np.zeros(n_macro_vars), res_vcov)

            # Calculate the value for the current period
            macro_scenario_array[r, h + 1, :] = next_quarter + noise

    # Trim the output array to remove historical data which we needed only to project the VAR forward
    macro_scenario_array = macro_scenario_array[:, var_order:, :]

    # Ensure unemployment (column 2) and sovereign yield (column 3) are not negative
    macro_scenario_array[:, :, 2:4] = np.maximum(macro_scenario_array[:, :, 2:4], 0)

    # Return the simulated data
    return macro_scenario_array


# Main function called from main.py returning scenarios
def simulate_scenarios(macro_data, macro_names, macro_names_print, future_macro_path, coefs, coef_vcov, residual_vcov,
                       var_order, n_scenario_quarters, n_scen, verbose=True, do_macro_var=True):

    if verbose:
        print(f'Simulating {n_scen} macro scenarios ...')

    # If scenarios to be simulted from scratch
    if do_macro_var:

        # Drop all the rows from macro_data that correspond to a incomplete quarter (i.e. not 4 quarters)
        #    indeed, we want to simulate a full number of quarters forward,
        #    this can be changed if we move to a quarterly data model

        # transform first most recent var_order rows of macro_data into initial conditions array
        init_cond = np.array(macro_data.sort_index().tail(var_order))

        # simulate scenarios
        scen_no_recentering_q = simulate_var(init_cond, coefs, coef_vcov, residual_vcov,
                                             var_order, n_scenario_years, n_scen, verbose=True)
        if verbose:
            print('    Simulated scenarios before recentering')

        # convert scenarios from quarterly to yearly
        history_for_q2y = np.array(macro_data.sort_index().tail(4))   # I need a full year of history
        scenarios_before_recentering = q_to_y(scen_no_recentering_q, history_for_q2y)
        if verbose:
            print('    Q to Y conversion done')

        # recenter the scenarios
        recentered_scenarios = recenter_scenarios(scenarios_before_recentering, macro_data,
                                                  future_macro_path, macro_names)
        if verbose:
            print('    Recentering done')

        # add the delta variable (e.g. of unemplyoment)
        recentered_scenarios, macro_names, macro_names_print \
                = add_delta_variable(recentered_scenarios, 'URX',
                                     macro_names, macro_names_print)
        scenarios_before_recentering, _, _ \
                = add_delta_variable(scenarios_before_recentering, 'URX',
                                     macro_names.copy(), macro_names_print.copy())
        if verbose:
            print('    Added delta variable')

        # also convert the historical data from quarterly to yearly
        macro_data_np = np.array(macro_data)[np.newaxis, :, :]
        fake_history = np.empty((0, 8))
        # create empty np array of dimension 2
        historical_macro_data_y = q_to_y(macro_data_np, fake_history)
        historical_macro_data_y, *_ \
            = add_delta_variable(historical_macro_data_y, 'URX',
                                 macro_names.copy(), macro_names_print.copy())
        # drop the first fake scenario dimension
        historical_macro_data_y = historical_macro_data_y[0, :, :]

        historical_macro_data_y = NamedArray(names=['historical_macro_year', 'macro_var'],
                                             data=historical_macro_data_y)
        validate_named_array(historical_macro_data_y)

        # Define NamedArray
        recentered_scenarios = NamedArray(names=['scenario', 'future_year', 'macro_var'],
                                          data=recentered_scenarios)
        scenarios_before_recentering = NamedArray(names=['scenario', 'future_year', 'macro_var'],
                                                  data=scenarios_before_recentering)

        # Save the macro scenarios and macro_names in an h5 file
        save_named_array('data/derived/macro_scenarios.h5', recentered_scenarios, 'macro_scenarios')
        save_named_array('data/derived/macro_scenarios.h5', scenarios_before_recentering, 'macro_scenarios_before_recentering')
        save_named_array('data/derived/macro_scenarios.h5', historical_macro_data_y, 'historical_macro_data_y')
        with h5py.File('data/derived/macro_names.h5', 'w') as h5file:
            h5file.create_dataset('macro_names', data=macro_names)
            h5file.create_dataset('macro_names_print', data=macro_names_print)

    # If scenarios to be read from file
    else:
        recentered_scenarios = load_named_array('data/derived/macro_scenarios.h5', 'macro_scenarios')
        scenarios_before_recentering \
            = load_named_array('data/derived/macro_scenarios.h5', 'macro_scenarios_before_recentering')
        historical_macro_data_y = load_named_array('data/derived/macro_scenarios.h5', 'historical_macro_data_y')
        with h5py.File('data/derived/macro_names.h5', 'r') as h5file:
            macro_names = h5file['macro_names'][()]
            macro_names = [byte.decode('utf-8') for byte in macro_names]
            macro_names_print = h5file['macro_names_print'][()]
            macro_names_print = [byte.decode('utf-8') for byte in macro_names_print]

    # Check before returning
    validate_named_array(scenarios_before_recentering)
    validate_named_array(recentered_scenarios)
    validate_named_array(historical_macro_data_y)

    return recentered_scenarios, scenarios_before_recentering, macro_names, macro_names_print, historical_macro_data_y


# Function to recenter scenarios around a given path
def recenter_scenarios(scenarios, macro_data, manual_future_macro_path, macro_names, y_hori=10):

    # Check if columns of hand_input match macro_names
    if list(manual_future_macro_path.columns) != list(macro_names):
        print('macro_names                     ', macro_names)
        print('manual_future_macro_path.columns', list(manual_future_macro_path.columns))
        raise ValueError('The columns of manual_future_macro_path do not match macro_names')

    # compute the mean of all variables over all scenarios
    mean_scenarios = np.mean(scenarios, axis=0)

    # if the length of future_macro_path is smaller than n_horizon, raise an error
    if len(manual_future_macro_path) < y_hori:
        raise ValueError('The length of future_macro_path lower than n_horizon')

    # remove the last rows of future_macro_path if its length is greater than n_horizon
    if len(manual_future_macro_path) > y_hori:
        manual_future_macro_path = manual_future_macro_path.head(y_hori)

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


def add_delta_variable(scenarios_y, variable_to_be_delta, macro_names, macro_names_print):
    """Adds a variable to the scenarios_y array that is the delta of the variable_to_be_delta variable"""

    # check types
    if type(macro_names) != list:
        raise ValueError('macro_names is not a list')
    if type(macro_names_print) != list:
        raise ValueError('macro_names_print is not a list')

    # get the index of the variables to be delta'ed, i.e. diff taken (macro_names a numpy index)
    idx_to_be_delta = macro_names.index(variable_to_be_delta)

    delta = np.diff(scenarios_y[:, :, idx_to_be_delta], axis=1)
    delta = np.hstack((np.full((delta.shape[0], 1), np.nan), delta))

    scenarios_y = np.insert(scenarios_y, idx_to_be_delta + 1, delta, axis=2)

    # insert the name of the delta variable in macro_names and macro_names_print
    macro_names.insert(idx_to_be_delta + 1, f'D{variable_to_be_delta}')
    # find the index of the variable in macro_names_print
    print_name_to_be_delta = macro_names_print[macro_names.index(variable_to_be_delta)]
    macro_names_print.insert(idx_to_be_delta + 1, f'Delta {print_name_to_be_delta}')

    return scenarios_y, macro_names, macro_names_print


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