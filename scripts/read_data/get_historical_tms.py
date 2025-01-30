# Laurent Millischer, laurent@millischer.eu, 2023
import numpy as np
import pandas as pd
from scripts.vectors_matrices.valid_tm import is_valid_2d_tm
from data.file_paths import input_path

from scripts.vectors_matrices.named_array import *


def tm_from_raw_data(df, verbose=False):
    """
    Function to create a transition matrix from a dataframe
    :param df: from and to columns are the row and column indices, t_prob is the value
    :return: 3x3 transition matrix
    """

    # Check if dataframe has the right columns and length 9
    if not (len(df) == 9 and 'from' in df.columns and 'to' in df.columns and 't_prob' in df.columns):
        if verbose:
            print(df)
        print(df)
        print(df.dtypes)
        raise ValueError("Dataframe does not have the right columns or length.")

    # Initialize a 3x3 matrix with zeros
    mat = np.full(shape=(3, 3), fill_value=0.0)

    # Loop through each row of the dataframe
    for i in range(len(df)):
        # Extract 'from', 'to', and 't_prob' values from the current row
        from_idx = int(df.iloc[i]['from']) - 1
        to_idx = int(df.iloc[i]['to']) - 1
        t_prob = df.iloc[i]['t_prob']

        # Assign the transition probability value to the appropriate cell in the matrix
        mat[from_idx, to_idx] = t_prob

    # If a full row is 0, replace it with a full row of NaNs
    for i in range(3):
        if np.all(mat[i, :] == 0):
            mat[i, :] = np.nan

    # Check if valid transition matrix
    if not is_valid_2d_tm(mat):
        raise ValueError("Matrix is not a valid transition matrix.")

    # Return the resulting matrix
    return mat


def get_historical_transition_matrices(all_banks, portfolio_types, bank_portfolios,
                                       verbose=False):
    """
    Function to read historical transition historical_tms from an Excel file and store them in a numpy array
    :return: transition matrix array (banks x years x portfolios x 3 x 3)
    """

    # Read the historical transition matrix data from an Excel file, missing as NaNs
    raw_tms = pd.read_excel(input_path, sheet_name='Historical TMs')

    # Define a few variables
    raw_tms.rename(columns={'ID_IMF': 'bank',
                            'Portfolio': 'portfolio'}, inplace=True)
    # Move by one quarter because the transition matrix is for the next quarter
    raw_tms['period'] = (raw_tms['Period'] + dt.timedelta(days=90)).dt.to_period('Q')
    # Drop starting periods that are after the model_snapshot_date
    raw_tms = raw_tms[raw_tms['period'] <= pd.Period(model_snapshot_date, 'Q')]

    # drop variables that are not needed
    raw_tms.drop(columns=['Period', 'Name_IMF'], inplace=True)

    # drop rows in which either from or to are outside [1, 2, 3]
    raw_tms = raw_tms[(raw_tms['from'] >= 1) & (raw_tms['from'] <= 3)]
    raw_tms = raw_tms[(raw_tms['to'] >= 1) & (raw_tms['to'] <= 3)]

    # Define sets of all banks, years, and portfolios
    historical_tm_periods = raw_tms['period'].unique().astype(str)

    # Create a numpy array to store historical_tms
    historical_tms = np.full(shape=(len(all_banks), len(portfolio_types), len(historical_tm_periods), 3, 3), fill_value=np.nan)

    # Check if all banks and portfolios with exposure have historical transition matrices
    tick_list = bank_portfolios.copy()

    # Loop through all banks, years, and portfolios
    for ib, b in enumerate(all_banks):
        for ip, p in enumerate(portfolio_types):
            # if bank b has no exposure in portfolio p in the latest year, skip
            if (ib, ip) not in bank_portfolios:
                continue
            for iy, y in enumerate(historical_tm_periods):
                # if verbose:
                #     print(f"Processing bank {b}, period {y}, portfolio {p}")
                # Extract the data for the current bank, period (year or quarter), and portfolio
                tmp_df = raw_tms[
                    (raw_tms['bank'] == b) &
                    (raw_tms['period'] == y) &
                    (raw_tms['portfolio'] == p)]

                # Extract the matrix for the current bank, period, and portfolio
                if len(tmp_df):
                    tmp_tm = tm_from_raw_data(tmp_df)
                else:
                    continue

                # Check if matrix is 3x3
                if tmp_tm.shape != (3, 3):
                    raise ValueError("Matrix is not 3x3.")

                # If a line is [1 0 0] or [0 1 0] or [0 0 1], replace it with a line of epsilon defaults
                #   this is because only an infinitely high z-score can lead to such lines

                epsilon = 0.0001
                for i in [0, 1, 2]:
                    if np.all(tmp_tm[i, :] == np.array([1, 0, 0])):
                        if i == 0:
                            tmp_tm[i, :] = np.array([1-10*epsilon, 9*epsilon, epsilon])
                        else:
                            tmp_tm[i, :] = np.array([0.9, 0.1 - epsilon, epsilon])
                    if np.all(tmp_tm[i, :] == np.array([0, 1, 0])):
                            tmp_tm[i, :] = np.array([5*epsilon, 1-10*epsilon, 5*epsilon])
                    if np.all(tmp_tm[i, :] == np.array([0, 0, 1])):
                        if i == 2:
                            tmp_tm[i, :] = np.array([epsilon, 9*epsilon, 1-10*epsilon])
                        else:
                            tmp_tm[i, :] = np.array([epsilon, 0.1 - epsilon, 0.9])



                # Store the matrix in the numpy array
                historical_tms[ib, ip, iy, :, :] = tmp_tm

            # check if all transition matrices for bank b and portfolio p are NaNs
            if not np.all(np.isnan(historical_tms[ib, ip, :, :, :])):
                tick_list.remove((ib, ip))

    if verbose:
        print('These banks and portfolios have no historical transition matrices:', tick_list)

    # Define NamedArray
    historical_tms = NamedArray(names=['bank', 'portfolio_type', 'historical_tm_quarter', 'stage3', 'stage3'],
                                data=historical_tms)
    validate_named_array(historical_tms)

    # Return historical_tms
    return historical_tms, historical_tm_periods
