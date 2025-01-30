# Laurent Millischer, laurent@millischer.eu, 2023

import pandas as pd
import numpy as np
from data.file_paths import input_path

from scripts.vectors_matrices.named_array import *


def get_amortization_profile(all_banks, portfolio_types, bank_portfolios):
    """
    Function to read amortization profiles from an Excel file and store them in a numpy array
    the list of banks and porfolios from the transition matrix is used as input
    :param all_banks: list of banks to loop over
    :param portfolio_types: list of portfolios to loop over
    :return: Amortization array of dimensions (bank, portfolio, stage, future_quarter)
    """

    # Read Excel file
    df = pd.read_excel(input_path, sheet_name='Amortization profiles')

    # Rename columns
    df.rename(columns={'ID_FMI': 'bank',
                       'Portfolio': 'portfolio',
                       'Stages': 'stage'}, inplace=True)

    # Drop columns where 'stage' is 3 because stage 3 does not amortize, it's defaulted
    df = df[df['stage'] != 3]

    # In the df there are column called q0, q1, ..., qN, find the highest N
    # and use it to define the number of future quarters
    amortization_horizon = max([int(x[1:]) for x in df.columns if x.startswith('q')])

    # check if column q0 is filled only with 1
    if not all(df['q0'] == 1):
        raise ValueError("Column q0 should be filled with 1s")

    # Create empty array to store amortization profiles
    amortization_profiles = np.full(shape=(len(all_banks), len(portfolio_types), 3,
                                           amortization_horizon + 1), fill_value=np.nan)

    # later we will be keeping only columns that start with 'q'
    columns_to_keep = [c for c in df.columns if c.startswith('q')]

    # Loop through banks, portfolios, and stages to fill the array
    for ib, b in enumerate(all_banks):
        for ip, p in enumerate(portfolio_types):

            if (ib, ip) not in bank_portfolios:
                continue

            for s in [1, 2]:
                # print(f"Processing bank {b}, portfolio {p}, stage {s}")
                tmp_df = df[(df['bank'] == b) &
                            (df['portfolio'] == p) &
                            (df['stage'] == s)]
                if len(tmp_df) == 0:
                    continue
                tmp_profile = tmp_df[columns_to_keep].iloc[0].values
                if len(tmp_profile) > 0:
                    amortization_profiles[ib, ip, s - 1, :] = tmp_profile

    # Set all the stage 3 amortization to 1, i.e. there is no amortization
    amortization_profiles[:, :, 2, :] = 1.

    # Define NamedArray
    amortization_profiles = NamedArray(names=['bank', 'portfolio_type', 'stage3', 'future_quarter_amort'],
                                       data=amortization_profiles)
    validate_named_array(amortization_profiles)

    return amortization_profiles
