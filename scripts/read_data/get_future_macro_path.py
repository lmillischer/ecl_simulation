# Laurent Millischer, lmillischer@jvi.org

import pandas as pd

from data.file_paths import input_path


# Function to get the path of the future macro data
def get_future_macro_path():

    # Read the future macro path from a CSV file
    future_macro_path = pd.read_excel(input_path, sheet_name='Macro recentering', index_col='q_ahead')

    # Return the resulting dataframe
    return future_macro_path

