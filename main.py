# Laurent Millischer, laurent@millischer.eu, 2023

# -------------------------------------------------------------------
# Setting up the infrastructure
# -------------------------------------------------------------------

# Import libraries
import os
import shutil
import warnings

# Import custom modules
from scripts.read_data.get_input_data import get_input_data
from scripts.macro_model.calibrate_var import calibrate_var, calibrate_var_exog
from scripts.macro_model.simulate_var_exog import simulate_scenarios_exog
from scripts.zscores.tm_to_zscore import historical_tms_to_zscores
from scripts.zscores.macro_to_zscores import macro_to_zscores
from scripts.zscores.zscore_scenarios_to_tms import zscore_scenarios_to_3x3tms
from scripts.vectors_matrices.add_repayments_to_tm import three_to_four
from scripts.vectors_matrices.amortization_to_repayment import amortization_to_repayment
from scripts.lgd.get_lgd_scenarios import get_lgd_scenarios
from scripts.lgd.value_to_sale import value_to_sale, find_binomial_parameters, find_negative_binomial_parameters
from scripts.ecl.ecl_calculation import calculate_ecl
from scripts.data_handling.memory_usage import *

from scripts.plotting.plot_ecl import plot_ecl
from scripts.plotting.plot_exposure import plot_exposure
from scripts.plotting.plot_zscores import *
from scripts.plotting.plot_lgd import *
from scripts.plotting.plot_tms import *
from scripts.plotting.plot_macro import plot_macro
from scripts.tests.tests_dashboard import run_tests

# in the file below, you can decide what modules to run
from controls import *

# Set working directory to the directory of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Ignore warnings
warnings.filterwarnings('ignore', category=UserWarning, module='intel')

# -------------------------------------------------------------------
# Prepare output folder
# -------------------------------------------------------------------

folder_path = f'data/output/{first_hash}'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
description_file_path = f'{folder_path}/_description.txt'
if not os.path.exists(description_file_path):
    with open(description_file_path, 'w') as file:
        file.write(run_full_desc)
if not os.path.exists(f'{folder_path}/controls.py'):
    shutil.copy('controls.py', f'{folder_path}')

# -------------------------------------------------------------------
# Read in the input data
# -------------------------------------------------------------------

print('Reading input data...')
(banks, portfolio_types, stages, collateral_types, bank_portfolios, bank_portfolios_stages, historical_quarters, current_portfolios,
 historical_portfolios, amortization_profiles, historical_tms, historical_macro_data, macro_names, macro_names_print,
 future_macro_path, interest_rates) = get_input_data(read_from_excel=read_input_data_from_excel, verbose=True)
if verbose:
    print_memory_usage()
    get_largest_objects()

# -------------------------------------------------------------------
# [BOX T] Compute historical z-scores
# -------------------------------------------------------------------

print('Computing historical z-scores...')
historical_zscores, zscore_variance, rhos, upper_bounds, lower_bounds \
    = historical_tms_to_zscores(historical_tms, 0.1, banks, portfolio_types,
                                bank_portfolios, recompute_historical_zscores,
                                verbose=True)
if verbose:
    print_memory_usage()
    get_largest_objects()

# -------------------------------------------------------------------
# [BOXES V] The macro VAR model
# -------------------------------------------------------------------

print('Calibrating the VAR model...')
# Calibrate the VAR model
# model, coefs, coef_vcov, residuals, res_vcov = calibrate_var(historical_macro_data, var_order)
exog_vars = ['OILQ', 'FFR']
endog_output, exog_output = calibrate_var_exog(historical_macro_data, var_order_endog, var_order_exog, exog_vars, do_calibrate_var)
model, coefs, coef_vcov, residuals, res_vcov = endog_output
model_exog, coefs_exog, coef_vcov_exog, residuals_exog, res_vcov_exog = exog_output

print('Simulating macro scenarios...')
# Simulate scenarios, an array of dimensions n_scen, n_horizon, n_macro_vars
(macro_scenarios, scenarios_before_recentering, historical_macro_data_enhanced, average_macro_data_enhanced,
 macro_names, macro_names_print) \
    = simulate_scenarios_exog(historical_macro_data, macro_names, macro_names_print,
                              future_macro_path,
                              coefs, coef_vcov, res_vcov,
                              coefs_exog, coef_vcov_exog, res_vcov_exog, exog_vars,
                              var_order_endog, var_order_exog, n_scenario_quarters, n_scenarios,
                              do_cycle_neutral=do_cycle_neutral,
                              do_coef_uncertainty=True, do_resid_uncertainty=True,
                              verbose=True, do_macro_var=do_simulate_var)
if verbose:
    print_memory_usage()
    get_largest_objects()

# -------------------------------------------------------------------
# [BOX Z] Converting macro scenarios to z-scores
# -------------------------------------------------------------------

print('Converting macro scenarios to z-scores...')

historical_or_average_macro_enhanced = historical_macro_data_enhanced if not do_cycle_neutral \
    else average_macro_data_enhanced
zscore_scenarios = macro_to_zscores(portfolio_types, macro_scenarios, do_macro_to_zscores,
                                    historical_zscores, historical_or_average_macro_enhanced,
                                    do_cycle_neutral=do_cycle_neutral,
                                    do_coef_uncertainty=True, do_resid_uncertainty=True, verbose=True)
if verbose:
    print_memory_usage()
    get_largest_objects()

# -------------------------------------------------------------------
# [BOX T and BOX A] Convert z-scores to 4x4 TM sceanrios
# -------------------------------------------------------------------

print('Converting z-scores to transition matrices...')
tm_scenarios_3x3 = zscore_scenarios_to_3x3tms(zscore_scenarios, rhos, upper_bounds, lower_bounds, do_tm_scenarios)
print('    - 3x3 TMs done')
repayment_profiles = amortization_to_repayment(amortization_profiles, n_scenario_quarters, do_tm_scenarios)
print('    - Repayment profiles done')
tm_scenarios_4x4 = three_to_four(tm_scenarios_3x3, repayment_profiles, do_tm_scenarios)
if verbose:
    print_memory_usage()
    get_largest_objects()

# -------------------------------------------------------------------
# [BOXES L] LGD model and simulation
# -------------------------------------------------------------------

print('Running the LGD model...')

# if do_v2s:
    # r, p = find_negative_binomial_parameters(target_mean=3.92*4, target_std=4.72*4)
    # n, p = find_binomial_parameters(target_mean=3.92, target_std=4.72)
    # mean and std of the number of years to sale are 3.92 and 4.72
    # these parameters are not reachable with a binomial distribution
    # taking closest values: n=100, p=0.157


v2s_mu, v2s_sigma2 = value_to_sale(n=n_v2s, p=p_v2s, num_draws=10000, macro_scenarios=macro_scenarios,
                                   macro_names=macro_names, do_v2s=do_v2s, interest_rates=interest_rates,
                                   banks=banks, portfolio_types=portfolio_types,
                                   bank_portfolios=bank_portfolios)
if verbose:
    print_memory_usage()
    get_largest_objects()

sr_sigma2 = sr_sigma ** 2
lgd, cr_mu, cr_sigma2 \
    = get_lgd_scenarios(banks, portfolio_types, collateral_types, stages, v2s_mu, v2s_sigma2, sr_mu, sr_sigma2,
                        costs, macro_scenarios, macro_names, tm_scenarios_3x3, do_lgd, historical_tms,
                        amortization_profiles, verbose=True)
if verbose:
    print_memory_usage()
    get_largest_objects()

# -------------------------------------------------------------------
# [BOX E] ECL
# -------------------------------------------------------------------

print('Calculating ECL...')
cl_scenarios, cl_rate_scenarios = calculate_ecl(current_portfolios, historical_portfolios, tm_scenarios_4x4, lgd, interest_rates, do_ecl,
                                                verbose=True)


# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------

if do_plotting:

    print('Plotting...')

    do_exposure_plot = False
    do_macro_plot = True
    do_zscore_plot = False
    do_tm_plot = True
    do_lgd_plot = True
    do_ecl_plot = True

    # for (ib, b) in enumerate(banks):
    #     print(f'Bank {ib}: {b}')

    only_bank_portfolio = [(0, 0), (0, 1), (0, 2), (0, 3)]
    # only_bank_portfolio = [(7, 0), (17, 3), (18, 0), (27, 0), (29, 2), (38, 0), (39, 3)]
    # only_bank_portfolio = [(39, 3), (37, 3), (30, 3)]
    # only_bank_portfolio = [(39, 3)]
    # only_bank_portfolio = [(29,2)]

    # Historical exposures
    if do_exposure_plot:
        plot_exposure(historical_portfolios, current_portfolios, banks, portfolio_types, bank_portfolios,
                      collateral_types, only_bank_portfolio, historical_quarters)

    # Macro past and macro-scenarios
    if do_macro_plot:
        plot_macro(historical_macro_data_enhanced, scenarios_before_recentering, macro_names, macro_names_print,
                   historical_quarters, prefix='_no_recentering')
        plot_macro(historical_macro_data_enhanced, macro_scenarios, macro_names, macro_names_print, historical_quarters)

    # Zscores
    if do_zscore_plot:
        plot_zscore_scenarios(zscore_scenarios, banks, portfolio_types, bank_portfolios,
                              historical_zscores, only_bank_portfolio, historical_quarters, verbose=True)

    # Transition matrices
    if do_tm_plot:
        plot_tm_scenarios(tm_scenarios_3x3, banks, portfolio_types, bank_portfolios, historical_tms,
                          only_bank_portfolio, historical_quarters, historical_portfolios, verbose=True)

    # LGD distributions
    if do_lgd_plot:
        plot_lgd_scenarios(lgd, banks, portfolio_types, bank_portfolios, collateral_types,
                           only_bank_portfolio, historical_quarters, verbose=True)
        plot_cr_scenarios(cr_mu, cr_sigma2, banks, portfolio_types, bank_portfolios, collateral_types,
                          v2s_mu, v2s_sigma2, sr_mu, sr_sigma2, only_bank_portfolio, verbose=True)


    # ECL distributions
    if do_ecl_plot:
        for stage in [1, 2, 3]:
            for stype in ['rel']:  # ['rel', 'abs']:
                plot_ecl(banks, portfolio_types, cl_scenarios if stype == 'abs' else cl_rate_scenarios,
                         bank_portfolios, only_bank_portfolio, stage=stage, stat_type=stype, verbose=False)
