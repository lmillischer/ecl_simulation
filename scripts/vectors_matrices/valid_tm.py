import numpy as np

from scripts.vectors_matrices.named_array import *


def is_valid_2d_tm(tm):
    """
    Function checking whether a 2D array is a valid transition matrix
    :param tm: array to be checked
    :return: True or False
    """

    # Check shape, should be either 3x3 or 4x4
    if not (tm.shape == (3, 3)) or (tm.shape == (4, 4)):
        print(f'Shape of the matrix not 3x3 or 4x4 but rather {tm.shape}.')
        return False

    # Check if there are no NaN values in the matrix that are not part of a full row of NaNs
    # A full row is valid because it would just mean there was no loan in that stage at t-1
    if np.isnan(tm).sum() > 0:
        # Check if all values in a line are NaNs
        nan_mask = np.isnan(tm)
        full_nan_rows = np.all(nan_mask, axis=1)

        # Check if any NaN is not in a full NaN row
        for i in range(3):
            if np.any(nan_mask[i]) and not full_nan_rows[i]:
                print('There are NaNs in the matrix')
                return False

    # Check if all values are between 0 and 1 unless they are NaNs
    if np.any(tm > 1) or np.any(tm < 0):
        print('Some matrix elements are outside [0, 1]')
        return False

    # Check if the sum of each of the rows is very close to 1 or NaN
    if not np.allclose(np.sum(tm, axis=1), 1, atol=1e-8):
        # Check if the row that does not sum to 1 is a full NaN row
        if not np.any(np.isnan(tm[np.sum(tm, axis=1) != 1])):
            print('Some rows do not sum to 1.')
            return False

    # If survived all checks, return true
    return True


def is_valid_tm_old(tm_array):
    """
    Function checking whether an in put is made of valid transition matrices. If the input is 2D
    it just runs the tests, if the array is 3D, it loops through the first dimension and checks
    if the slices are all valid transition matrices.
    :param tm_array: 2D or 3D array
    :return: True/False or a 1D array of True/False
    """
    if tm_array.ndim == 2:
        # If the array is 2D, apply the function directly
        return is_valid_2d_tm(tm_array)
    elif tm_array.ndim == 3:
        # If the array is 3D, apply the function to each 2D slice
        result = np.array([is_valid_2d_tm(tm_array[i]) for i in range(tm_array.shape[0])])
        return result
    else:
        raise ValueError("Input must be a 2D or 3D array")


def is_valid_tm(tm):

    # Check if tm is a NamedArray
    if not isinstance(tm, NamedArray):
        raise TypeError("tm must be a NamedArray")

    # Get the two dimensions of tm the names of which are stage3 or stage4 otherwise raise error
    if tm.names.count('stage3') == 2:
        # if so get index of both occurences
        stage_dim = [i for i, x in enumerate(tm.names) if x == 'stage3']
    elif tm.names.count('stage4m') == 2:
        # if so get index of both occurences
        stage_dim = [i for i, x in enumerate(tm.names) if x == 'stage4m']
    else:
        raise ValueError("tm must have a stage3 or stage4 dimension")

    # Check if all values are between 0 and 1 unless they are NaNs
    if np.any(tm.data > 1) or np.any(tm.data < 0):
        # get one example of a value outside [0, 1]
        idx = np.where((tm.data > 1) | (tm.data < 0))
        idx = list(zip(*idx))
        one_idx = idx[0]
        print(one_idx)
        wrong_tm = tm.data[one_idx[0], one_idx[1], one_idx[2], one_idx[3], :, :]
        print(wrong_tm)
        print('Some matrix elements are outside [0, 1]')
        return False

    # Check if the sum of each of the rows is very close to 1 or NaN
    sum_rows = np.sum(tm.data.copy(), axis=stage_dim[1])

    close_to_one_or_nans = np.logical_or(np.isclose(sum_rows, 1, atol=1e-6), np.isnan(sum_rows))

    if not np.all(close_to_one_or_nans):
        # get an example of a matrix with a row not summing to 1
        idx = np.where(~close_to_one_or_nans)
        idx = list(zip(*idx))
        one_idx = idx[0]
        print(one_idx)
        wrong_tm = tm.data[one_idx[0], one_idx[1], one_idx[2], one_idx[3], :, :].copy()
        print(wrong_tm)
        print(np.nansum(wrong_tm, axis=1))
        print('Not all rows sum to 1 or NaN.')
        return False

    # Check if there are no NaN values in the matrix that are not part of a full row of NaNs

    # start by getting the rows that are full of NaNs
    nan_sums = np.isnan(sum_rows)

    # a numpy array of the actual rows (length 3 or 4) of which the sum is nan
    list_of_nan_sum_rows = tm.data[nan_sums]

    # that array should be full of NaNs
    if not np.all(np.isnan(list_of_nan_sum_rows)):
        print('nan_sums', nan_sums.shape)
        print('test', list_of_nan_sum_rows.shape)
        idx = np.where(~np.isnan(list_of_nan_sum_rows))
        idx = list(zip(*idx))
        print(idx[0])
        # list of indices where nan_sum is True
        idx_nansum = np.where(nan_sums)
        idx_nansum = list(zip(*idx_nansum))
        pick_nansum = 2
        print('nanindex', idx_nansum[2])
        print('example of a nansum tm')
        nansumtm = tm.data[idx_nansum[pick_nansum][0], idx_nansum[pick_nansum][1], idx_nansum[pick_nansum][2], idx_nansum[pick_nansum][3], :, :]
        print(nansumtm)
        print('Some nans are not in full-nan rows')
        return False

    # If all tests passed, return True
    return True
