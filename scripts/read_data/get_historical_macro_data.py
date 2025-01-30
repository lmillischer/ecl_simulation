# Laurent Millischer, laurent@millischer.eu, 2023

import pandas as pd
from data.file_paths import input_path
from controls import *


# This function reads in the historical macro data and returns a dataframe
def get_historical_macro_data(historical_tm_periods):

    # Read the historical macro data from an Excel file, no column names
    df = pd.read_excel(io=input_path, sheet_name=f'Historical macro data', usecols='B:L',
                       skiprows=2, nrows=64) # todo: make this dynamic

    # Set the first column as index
    df.rename(columns={'Unnamed: 1': 'period'}, inplace=True)
    # Convert the period to a quarterly period object
    df['period'] = pd.PeriodIndex(df['period'], freq='Q')
    # Drop starting periods that are after the model_snapshot_date
    df = df[df['period'] <= pd.Period(model_snapshot_date, 'Q')]
    df.set_index('period', inplace=True)

    # Read the names of the macro variables from the second row of the excel file
    names = pd.read_excel(io=input_path, sheet_name=f'Historical macro data', usecols='C:L',
                          skiprows=1, nrows=1, header=None)
    names = names.values[0]

    if type(names) != list:
        names = names.tolist()

    return df, names