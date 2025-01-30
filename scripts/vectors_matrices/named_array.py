import numpy as np
import h5py

from controls import *

valid_names_and_dimensions = {'scenario': n_scenarios,
                              'bank': n_banks,
                              'portfolio_type': n_portfolio_types,
                              'future_quarter': n_scenario_quarters,
                              'future_quarter_amort': 121,  # 30 years of amortization + current snapshot
                              'future_snapshot': n_scenario_quarters + 1,  # because quarter 0 in the array
                              'historical_tm_quarter': n_history_quarters_tm,
                              'historical_macro_quarter': n_history_quarters_macro,
                              'historical_snapshot': 60,
                              'stage2': 2,  # stage 1, 2
                              'stage3': 3,  # stage 1, 2, 3
                              'stage3b': 3,  # stage 3b1, 3b2, 3b3
                              'stage4': 4,  # stage 1, 2, 3b1, 3b2
                              'stage4m': 4,  # stage 1, 2, 3, matured
                              'stage5': 5,  # stage 1, 2, 3b1, 3b2, 3b3
                              'collateral_type': n_collateral_types,
                              'macro_var': n_macro_variables}

class NamedArray:
    """A numpy array with named dimensions to keep track"""
    def __init__(self, names: list, data: np.ndarray):
        if not isinstance(names, list):
            raise TypeError("names must be a list")
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array")

        self.names = names
        self.data = data


def save_named_array(h5file, named_array, group_name):
    with h5py.File(h5file, 'a') as file:
        if group_name in file:
            del file[group_name]
        group = file.create_group(group_name)
        group.create_dataset('data', data=named_array.data)
        group.create_dataset('names', data=np.array(named_array.names, dtype='S'))  # Saving names as a string array


def load_named_array(h5file, group_name):
    with h5py.File(h5file, 'r') as file:
        group = file[group_name]
        data = group['data'][:]
        names = list(group['names'][:].astype(str))  # Converting back to Python strings
    return NamedArray(names, data)


def validate_named_array(named_array):
    # check that length of names is the same as the number of dimensions of data
    if len(named_array.names) != len(named_array.data.shape):
        raise ValueError(f'Length of names ({len(named_array.names)}) '
                         f'does not match number of dimensions of data ({len(named_array.data.shape)}).')

    # check that the types are correct
    if not isinstance(named_array.names, list):
        raise TypeError("names must be a list")
    if not isinstance(named_array.data, np.ndarray):
        raise TypeError("data must be a numpy array")

    for in_, name in enumerate(named_array.names):
        # check that all items of names in the list of authorized dimensions
        if name not in valid_names_and_dimensions:
            raise ValueError(f'Name "{name}" not in list of authorized dimensions.')
        # check that the length of the data dimension matches
        if ((named_array.data.shape[in_] != valid_names_and_dimensions[name]) and
                (named_array.data.shape[in_] != 1)):  # dimension can be 1 for broadcasting
            raise ValueError(f'Dimension #{in_+1} of data (size={named_array.data.shape[in_]}) '
                             f'does not match length of {name} (size={valid_names_and_dimensions[name]}).')
