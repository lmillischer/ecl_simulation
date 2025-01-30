# Laurent Millischer, laurent@millischer.eu, 2023

import pandas as pd

from scripts.read_data.get_historical_tms import get_historical_transition_matrices
from scripts.read_data.get_amortization_profiles import get_amortization_profile
from scripts.read_data.get_historical_macro_data import get_historical_macro_data
from scripts.read_data.get_future_macro_path import get_future_macro_path
from scripts.read_data.get_portfolios import get_current_portfolios, get_historical_portfolios
from scripts.read_data.get_interest_rates import get_interest_rates
from scripts.vectors_matrices.named_array import *

from controls import *


def get_input_data(read_from_excel=False, verbose=False):
    if read_from_excel:
        if verbose:
            print('Reading input data from Excel files')

        # read current portfolios
        (banks, portfolio_types, bank_portfolios, bank_portfolio_stages, stages, collateral_types,
         current_portfolios) = get_current_portfolios()
        historical_portfolios = get_historical_portfolios(banks, portfolio_types)
        if verbose:
            print(f'   - current and historical porftolios done')

        # read historical transition matrices
        historical_tms, historical_tm_periods = (
            get_historical_transition_matrices(all_banks=banks, portfolio_types=portfolio_types,
                                               bank_portfolios=bank_portfolios, verbose=verbose))
        if verbose:
            print(f'   - historical TMs done: {historical_tms.data.shape}')

        # read historical macro data
        historical_macro_data, macro_names_print = get_historical_macro_data(historical_tm_periods)
        if verbose:
            print(f'   - historical macro data done up until {max(historical_macro_data.index)}')

        # read amortization profiles
        amortization_profiles = get_amortization_profile(banks, portfolio_types, bank_portfolios)

        # read portfolio interest rates
        interest_rates = get_interest_rates(banks, portfolio_types, bank_portfolios)

        # read future macro path (hand input into macro scenarios)
        future_macro_path = get_future_macro_path()

        # these are the names of macro variables we will use throughout the model
        macro_names = list(historical_macro_data.columns)

        # then save the NamedArrays in h5 files
        tmp_folder_path = f'data/output/{first_hash}/'
        h5_out_name = f'{tmp_folder_path}/input_named_data.h5'
        save_named_array(h5_out_name, historical_tms, 'historical_tms')
        save_named_array(h5_out_name, amortization_profiles, 'amortization_profiles')
        save_named_array(h5_out_name, current_portfolios, 'current_portfolios')
        save_named_array(h5_out_name, interest_rates, 'interest_rates')
        save_named_array(h5_out_name, historical_portfolios, 'historical_portfolios')

        # lists to h5 files
        with h5py.File(f'{tmp_folder_path}/input_data.h5', 'w') as h5file:
            h5file.create_dataset('banks', data=banks)
            h5file.create_dataset('stages', data=stages)
            h5file.create_dataset('collateral_types', data=collateral_types)
            h5file.create_dataset('historical_tm_periods', data=historical_tm_periods)
            h5file.create_dataset('portfolio_types', data=portfolio_types)
            h5file.create_dataset('bank_portfolios', data=bank_portfolios)
            h5file.create_dataset('bank_portfolio_stages', data=bank_portfolio_stages)
            h5file.create_dataset('macro_names', data=macro_names)
            h5file.create_dataset('macro_names_print', data=macro_names_print)

        # pandas dataframes to h5 files
        historical_macro_data.to_hdf(f'{tmp_folder_path}/input_data.h5', key='historical_macro_data', mode='a')
        future_macro_path.to_hdf(f'{tmp_folder_path}/input_data.h5', key='future_macro_path', mode='a')

    else:
        # instead of painfully looping through Excel files, just read the h5 files
        with h5py.File(f'data/output/{first_hash}/input_data.h5', 'r') as h5file:
            banks = h5file['banks'][()]
            stages = h5file['stages'][()]
            stages = [byte.decode('utf-8') for byte in stages]  # decode strings
            collateral_types = h5file['collateral_types'][()]
            collateral_types = [byte.decode('utf-8') for byte in collateral_types]  # decode strings
            portfolio_types = h5file['portfolio_types'][()]
            portfolio_types = [byte.decode('utf-8') for byte in portfolio_types]  # decode strings
            macro_names = h5file['macro_names'][()]
            macro_names = [byte.decode('utf-8') for byte in macro_names]  # decode strings
            macro_names_print = h5file['macro_names_print'][()]
            macro_names_print = [byte.decode('utf-8') for byte in macro_names_print]  # decode strings
            historical_tm_periods = h5file['historical_tm_periods'][()]
            historical_tm_periods = [byte.decode('utf-8') for byte in historical_tm_periods]  # decode strings

            # get bank_portfolios and transform from np.array back into a list
            bank_portfolios = h5file['bank_portfolios'][()]
            bank_portfolios = list(map(tuple, bank_portfolios))
            bank_portfolio_stages = h5file['bank_portfolio_stages'][()]
            bank_portfolio_stages = list(map(tuple, bank_portfolio_stages))

        # read NamedArrays
        tmp_folder_path = f'data/output/{first_hash}/'
        current_portfolios = load_named_array(f'{tmp_folder_path}/input_named_data.h5', 'current_portfolios')
        amortization_profiles = load_named_array(f'{tmp_folder_path}/input_named_data.h5', 'amortization_profiles')
        interest_rates = load_named_array(f'{tmp_folder_path}/input_named_data.h5', 'interest_rates')
        historical_tms = load_named_array(f'{tmp_folder_path}/input_named_data.h5', 'historical_tms')
        historical_portfolios = load_named_array(f'{tmp_folder_path}/input_named_data.h5', 'historical_portfolios')

        # read pandas dataframes
        historical_macro_data = pd.read_hdf(f'{tmp_folder_path}/input_data.h5', key='historical_macro_data')
        future_macro_path = pd.read_hdf(f'{tmp_folder_path}/input_data.h5', key='future_macro_path')

    if verbose:
        # print fraction of missing values in current portfolios for each stage
        n_non_missing = np.sum(np.any(~np.isnan(current_portfolios.data[:, :, :, :]), axis=(2, 3)))
        print(f'    Nb bank-ptf with current exposure value: {n_non_missing}')

    # Return data
    return (banks, portfolio_types, stages, collateral_types, bank_portfolios, bank_portfolio_stages, historical_tm_periods,
            current_portfolios, historical_portfolios, amortization_profiles, historical_tms, historical_macro_data,
            macro_names, macro_names_print, future_macro_path, interest_rates)
