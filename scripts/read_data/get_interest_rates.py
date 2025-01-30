# Laurent Millischer, laurent@millischer.eu, 2023

import pandas as pd
import numpy as np

from data.file_paths import input_path
from scripts.vectors_matrices.named_array import NamedArray


def get_interest_rates(banks, portfolio_types, bank_portfolios):
    # read interest rates from Excel file
    df = pd.read_excel(input_path, sheet_name='Interest rates')

    # rename columns
    df = df.rename(columns={'Bank_ID': 'bank',
                            'Portfolio': 'portfolio_type',
                            'Quarterly_weighted_interest_rate': 'interest_rate'})

    # create empty array
    interest_rates = np.full((len(banks), len(portfolio_types)), np.nan)

    # make sure that all banks and portfolio types are included
    for ib, b in enumerate(banks):
        for ip, p in enumerate(portfolio_types):
            # do not read interest rates for portfolios that have no exposure
            if (ib, ip) not in bank_portfolios:
                continue

            # if there is a value in the Excel file, save it in the interest_rates array
            if len(df[(df.bank == b) & (df.portfolio_type == p)]):
                interest_rates[ib, ip] = df[(df.bank == b) & (df.portfolio_type == p)].iloc[0]['interest_rate']

    # not doing this: they are already quarterly
    # # now convert annual into quarterly rates
    interest_rates = interest_rates / 100
    # interest_rates = interest_rates ** (1 / 4) - 1

    # define NamedArray
    interest_rates = NamedArray(names=['bank', 'portfolio_type'],
                                data=interest_rates)

    return interest_rates
