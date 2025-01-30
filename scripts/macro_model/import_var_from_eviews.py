# tmp functions to rearrange the imports from the eviews-generated excel file

import pandas as pd

def rearrange_dataframe(df, columns=True, rows=True, exclude_row1=False):

    if exclude_row1:
        # save first row then drop it
        row1 = df.iloc[0, :]
        df = df.iloc[1:, :]

    N = df.shape[0] // 2
    new_order = [i for j in zip(range(1, N + 1), range(N + 1, 2 * N + 1)) for i in j]

    # Adjust for zero-based index
    new_order = [i - 1 for i in new_order]

    # Rearrange rows and columns
    if columns:
        df = df.iloc[:, new_order]
    if rows:
        df = df.iloc[new_order, :]

    if exclude_row1:
        df = pd.concat([row1.to_frame().T, df])

    return df


def rearrange_endog(df):
    order = []
    for i in range(11):  # loop over 8 endogenous variables
        for j in range(8):  # loop over 11 explanatory variables
            order.append(i+j*11)
    df = df.iloc[order, :]
    df = df.iloc[:, order]
    return df