import pandas as pd

from data.file_paths import *

from scripts.vectors_matrices.named_array import *


def get_bma_parameters(portfolio_types, n_coef):

    # prepare the empty outputs
    coefficients = np.full((n_portfolio_types, n_coef), np.nan)
    residual_se = np.full((n_banks, n_portfolio_types), np.nan)
    vcov = np.full((n_portfolio_types, n_coef, n_coef), np.nan)

    # loop over portfolios and read in the data
    for ip, p in enumerate(portfolio_types):
        sheet_name = f'BMA P{ip}'

        # save the coefficients
        portfolio_coefs = pd.read_excel(input_path, sheet_name=sheet_name, nrows=1, skiprows=1)
        portfolio_coefs.drop(columns='Unnamed: 0', inplace=True)
        coefficients[ip, :] = np.nan_to_num(portfolio_coefs.to_numpy())
        # save the residual standard errors
        portfolio_se = pd.read_excel(input_path, sheet_name=sheet_name, nrows=1, skiprows=3)
        residual_se[:, ip] = portfolio_se.iloc[0].to_numpy()[1:n_banks+1]

        # save the covariance matrix
        portfolio_cov = pd.read_excel(input_path, sheet_name=sheet_name, skiprows=7,
                                      header=None)
        portfolio_cov.drop(columns=0, inplace=True)
        vcov[ip, :, :] = portfolio_cov.to_numpy()

    return coefficients, residual_se, vcov


def macro_to_zscores(portfolio_types, macro_scenarios, do_macro_to_zscores,
                     historical_zscores, historical_or_average_macro_enhanced,
                     do_cycle_neutral,
                     do_coef_uncertainty, do_resid_uncertainty, verbose=True):

    out_path = f'data/output/{first_hash}/'

    # if computation set to True
    if do_macro_to_zscores:

        # BMA model has 2 coefs per macro variable (contemp. and lagged) and an intercept per bank
        n_coefs = n_banks + n_ar_bma + 2 * n_macro_variables

        # read in BMA parameters
        coefs, residual_stderr, coef_vcov = get_bma_parameters(portfolio_types, n_coefs)

        # set up empty zscore scenarios
        n_scenarios = macro_scenarios.data.shape[0]
        coef_for_scenarios = np.full((n_scenarios, n_banks, n_portfolio_types, n_coefs), np.nan)

        # prepare the array of coefficient vectors, drawn from the multivariate normal
        for ip, p in enumerate(portfolio_types):
            coef_array = np.random.multivariate_normal(coefs[ip], coef_vcov[ip], size=(n_scenarios, n_banks))
            coef_for_scenarios[:, :, ip, :] = coef_array

        # get the contemporaneous and lagged coefficients
        bank_fe = coef_for_scenarios[:, :, :, :n_banks]
        bank_fe = np.diagonal(bank_fe, axis1=1, axis2=3)  # for every bank we only need the corresponding bank FE
        bank_fe = np.moveaxis(bank_fe, 1, 2)

        # now get the other coefficients
        idx_ar_end = n_banks + n_ar_bma
        ar_coef = coef_for_scenarios[:, :, :, n_banks:idx_ar_end]
        idx_contemp_end = idx_ar_end + n_macro_variables
        contem_coefs = coef_for_scenarios[:, :, :, idx_ar_end:idx_contemp_end]
        idx_lagged_end = idx_contemp_end + n_macro_variables
        lagged_coefs = coef_for_scenarios[:, :, :, idx_contemp_end:idx_lagged_end]

        # create empty zscore scenarios
        zscore_scenario_data = np.full((n_scenarios, n_banks, n_portfolio_types, n_scenario_quarters+n_ar_bma), np.nan)
        # then intialise the first n_ar_bma lines of zscore_data with historical zscores
        # repeat the historical zscores along a new first dimension n_scenario times
        tmp_hzscores = np.repeat(historical_zscores.data[np.newaxis, ...], n_scenarios, axis=0)
        if not do_cycle_neutral:
            zscore_scenario_data[:, :, :, :n_ar_bma] = tmp_hzscores[:, :, :, -n_ar_bma:]
        else:
            # if we compute cycle neutral losses, we set initial z-scores to 0
            zscore_scenario_data[:, :, :, :n_ar_bma] = 0

        # prepare initial macro conditions by adding one line to the historical macro data
        macro_scen_for_use = macro_scenarios.data
        init_macro = historical_or_average_macro_enhanced.data[-1, :]
        init_macro = np.repeat(init_macro[np.newaxis, :], n_scenarios, axis=0)[:, np.newaxis, :]
        macro_scen_for_use = np.concatenate((init_macro, macro_scen_for_use), axis=1)
        macro_scen_for_use = macro_scen_for_use[:, np.newaxis, np.newaxis, :, :]
        # fill nas in macro_scen_for_use with 0
        macro_scen_for_use[np.isnan(macro_scen_for_use)] = 0
        # todo: later compute DUR in a proper way with macro history

        # now loop over future quarters
        for h in range(1, n_scenario_quarters+1):
            # compute dot product of the lagged coefficients with lagged macro vars
            dotprod_lagged = (macro_scen_for_use[:, :, :, h-1, :] * lagged_coefs).sum(axis=3)

            # compute dot product of the contemporaneous coefficients with contemporaneous macro vars
            dotprod_contemp = (macro_scen_for_use[:, :, :, h, :] * contem_coefs).sum(axis=3)

            # compute dot product of the autoregressive coefficients with the zscores
            lagged_zscores = zscore_scenario_data[:, :, :, h-n_ar_bma:h]
            # if lagged_zscores are missing at the start, use the average historical z-scores
            if (h == 1) and (np.any(np.isnan(lagged_zscores))):
                lagged_zscores[np.isnan(lagged_zscores)] = np.nanmean(tmp_hzscores[:, :, :, :], axis=3)[..., np.newaxis][np.isnan(lagged_zscores)]
                # if still missing, set the zscore to average of all other banks in that period
                # start by creating an array of the right dimension that contains average over banks of the latest observed z-scores
                avg_zs = np.nanmean(tmp_hzscores[:, :, :, -n_ar_bma:], axis=1)[:, np.newaxis, ...]
                # repeat n_banks time along axis 1
                avg_zs = np.repeat(avg_zs, n_banks, axis=1)
                lagged_zscores[np.isnan(lagged_zscores)] = avg_zs[np.isnan(lagged_zscores)]
            lagged_zscores = lagged_zscores[:, :, :, ::-1]  # reverse the order of the zscores
            dotprod_ar = (lagged_zscores * ar_coef).sum(axis=3)

            # draw from bank-specific residuals
            if do_resid_uncertainty:
                epsilon = np.random.normal(loc=0, scale=residual_stderr[np.newaxis, :, :], size=(n_scenarios, n_banks, n_portfolio_types))
            else:
                epsilon = 0

            # sum the terms and save in zscore_scenario_data
            zscore_scenario_data[:, :, :, h] = dotprod_lagged + dotprod_contemp + dotprod_ar + bank_fe + epsilon
            # print('zscore_scenario_data', zscore_scenario_data[0, xb, xp, :])

        # Now that the loop is done, drop the first n_ar_bma lines
        zscore_scenario_data = zscore_scenario_data[:, :, :, n_ar_bma:]

        # Define NamedArray
        zscore_scenarios = NamedArray(names=['scenario', 'bank', 'portfolio_type', 'future_quarter'],
                                      data=zscore_scenario_data)
        validate_named_array(zscore_scenarios)

        # Save zscore scenarios to h5 file
        save_named_array(f'{out_path}/zscore_scenario_data.h5', zscore_scenarios, 'zscore_scenarios')

    # if computation set to False, just read from h5 file
    else:
        zscore_scenarios = load_named_array(f'{out_path}/zscore_scenario_data.h5', 'zscore_scenarios')

    # Return results
    return zscore_scenarios
