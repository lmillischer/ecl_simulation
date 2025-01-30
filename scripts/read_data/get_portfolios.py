# Laurent Millischer, laurent@millischer.eu, 2023

import pandas as pd
from data.file_paths import input_path

from scripts.vectors_matrices.named_array import *


def get_historical_portfolios(banks, portfolio_types):
    # Read Excel file
    df = pd.read_excel(input_path, sheet_name='Historical exposure')

    # Rename columns, define year
    df.rename(columns={'ID_IMF': 'bank',
                       'Portfolio': 'portfolio',
                       'Stages': 'stage',
                       'Exposure_stocks': 'exposure'}, inplace=True)
    df['quarter'] = pd.PeriodIndex(df['Period'], freq='Q')
    df.drop(columns=['Period'], inplace=True)

    # define empty array to store historical portfolios
    historical_portfolios = NamedArray(names=['bank', 'portfolio_type', 'stage3', 'historical_snapshot'],
                                       data=np.full(shape=(len(banks), len(portfolio_types),
                                                           3, len(df['quarter'].unique())), fill_value=np.nan))
    validate_named_array(historical_portfolios)

    # Loop through banks, portfolios, and stages to fill the array
    full_list_of_snapshots = df.quarter.unique()
    for ib, b in enumerate(banks):
        for ip, p in enumerate(portfolio_types):
            for ix, x in enumerate([1, 2, 3]):
                tmp_df = df[(df['bank'] == b) & (df['portfolio'] == p) & (df['stage'] == x)]
                # fill tmp_df with missing quarters from full_list_of_snapshots
                tmp_df = pd.merge(pd.DataFrame({'quarter': full_list_of_snapshots}), tmp_df, on='quarter', how='left')
                for iy, y in enumerate(tmp_df['quarter']):
                    historical_portfolios.data[ib, ip, ix, iy] = tmp_df[tmp_df['quarter'] == y]['exposure']

    return historical_portfolios


def get_current_portfolios():
    """
    Function to read portfolios from an Excel file and store them in a numpy array
    :return: portfolio array of dimensions (bank, portfolio, stage, year)
    """

    # Read Excel file
    df = pd.read_excel(input_path, sheet_name='Current exposure')

    # Rename columns, define year
    df.rename(columns={'ID_IMF': 'bank',
                       'Portfolio': 'portfolio',
                       'CollateralType': 'collateral_type',
                       'Exposure_stocks': 'exposure'}, inplace=True)

    # define new variables
    df['Bucket'] = df['Bucket'].str.replace('Bucket ', '-')
    df['Bucket'].fillna('', inplace=True)
    df['collateral_type'].fillna('None', inplace=True)
    df['stage'] = df['Stages'].astype(str) + df['Bucket']

    # drop variables that are not needed
    df.drop(columns=['Period', 'Stages', 'Bucket'], inplace=True)

    # define main lists to be output
    banks = list(df['bank'].unique())
    if len(banks) != n_banks:
        raise ValueError(f'Number of banks is {len(banks)}, but {n_banks} in the control file')
    portfolio_types = list(df['portfolio'].unique())
    if len(portfolio_types) != n_portfolio_types:
        raise ValueError(f'Number of portfolio types is {len(portfolio_types)}, but {n_portfolio_types} in the control file')
    collateral_types = list(df['collateral_type'].unique())
    stages = list(df['stage'].unique())

    # Create empty array to store portfolios
    current_portfolios = np.full(shape=(len(banks), len(portfolio_types), len(collateral_types), len(stages)),
                                 fill_value=np.nan)

    # create empty list of (bank, portfolio) tuples
    bank_portfolios = []
    bank_portfolio_stages = []

    # Loop through banks, portfolios, and stages to fill the array
    for ib, b in enumerate(banks):
        for ip, p in enumerate(portfolio_types):
            # add the tuple (bank, portfolio) to the list if there is an observation
            if len(df[(df['bank'] == b) & (df['portfolio'] == p)]) > 0:
                bank_portfolios.append((ib, ip))
            for ic, c in enumerate(collateral_types):
                for ix, x in enumerate(stages):
                    tmp_df = df[(df['bank'] == b) &
                                (df['portfolio'] == p) &
                                (df['collateral_type'] == c) &
                                (df['stage'] == x)]
                    if len(tmp_df) > 0:
                        exposure = tmp_df.iloc[0]['exposure']
                        current_portfolios[ib, ip, ic, ix] = exposure

    # Which bank-portfolio-stages have exposure currently?
    tmp = np.nansum(current_portfolios, axis=2)  # sum over collateral type
    current_exp = np.empty((tmp.shape[0], tmp.shape[1], 3))
    current_exp[:, :, 0] = tmp[:, :, 0]
    current_exp[:, :, 1] = tmp[:, :, 1]
    current_exp[:, :, 2] = np.sum(tmp[:, :, 2:5], axis=2)  # sum over stages 3b1, 3b2, 3b3

    for ib, b in enumerate(banks):
        for ip, p in enumerate(portfolio_types):
            for s in [1, 2, 3]:
                exposure = current_exp[ib, ip, s-1]
                if np.isnan(exposure) or exposure == 0:
                    continue
                bank_portfolio_stages.append((ib, ip, s-1))

    # Define NamedArray
    current_portfolios = NamedArray(names=['bank', 'portfolio_type', 'collateral_type', 'stage5'],
                                    data=current_portfolios)
    validate_named_array(current_portfolios)

    return banks, portfolio_types, bank_portfolios, bank_portfolio_stages, stages, collateral_types, current_portfolios