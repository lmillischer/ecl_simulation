import datetime as dt
import hashlib
import base64

# -------------------------------------------------------------------
# Dashboard
# -------------------------------------------------------------------

# a way to run all steps in one go
do_all_steps = False

read_input_data_from_excel = do_all_steps or True
recompute_historical_zscores = True
do_calibrate_var = do_all_steps or True
do_simulate_var = do_all_steps or True
do_macro_to_zscores = do_all_steps or True
do_tm_scenarios = do_all_steps or True
do_v2s = True
do_lgd = do_all_steps or True
do_ecl = do_all_steps or True

do_cycle_neutral = False

do_plotting = False
do_testing = False

verbose = False

# -------------------------------------------------------------------
# Input parameters
# -------------------------------------------------------------------

model_snapshot_date = dt.date(2023, 9, 30)

# Dimensions
n_banks = 3
n_portfolio_types = 4
n_collateral_types = 4
n_scenario_quarters = 70
n_history_quarters_tm = 62
n_history_quarters_macro = 63

# Macro
var_order_endog = 1     # number of lags in the endogeneous VAR
var_order_exog = 2      # number of lags in the exogeneous VAR
n_scenarios = 3000      # number of simulated scenarios
n_macro_variables = 17  # 8 endog + 2 exog + 7 derived variable
n_ar_bma = 1            # number of autoregressive terms in the BMA model

# LGD paramters

# Sales ratio distribution
sr_mu = 0
sr_sigma = 0.05

# Time to sale parameters
n_v2s, p_v2s = 100, 0.157  # 100*0.157 = 15.7 quarters = 3.9 years

# Costs
costs = {'None': 0.05, 'Financial': 0.05, 'Other': 0.05, 'Real Estate': 0.10}

# Colors for plotting different collateral types
colors = ['C0', 'C3', 'C2', 'C1', 'C4']

# Intervals for plotting
plot_intervals = [13, 25, 38, 50]

# -------------------------------------------------------------------------------------------------------------
# Remember run name and id

run_description = '6 June - switching BMA model uncertainty off'

run_full_desc = (f'{run_description} - n_scen: {n_scenarios} - snap_date: {model_snapshot_date} '
                 f'- n_scenario_quarters: {n_scenario_quarters}{" - cycle-neutral" if do_cycle_neutral else ""}')


def hash_string_to_length_8(input_string):
    # Create a SHA-256 hash of the input string
    sha256_hash = hashlib.sha256(input_string.encode()).digest()

    # Encode the hash using base64
    base64_hash = base64.urlsafe_b64encode(sha256_hash).decode()

    # Truncate or pad the encoded hash to ensure it is 8 characters long
    return base64_hash[:8]

first_hash = hash_string_to_length_8(run_full_desc)
print(f'Run hash: {first_hash}')
