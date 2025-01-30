import numpy as np

from scripts.vectors_matrices.named_array import *
from scripts.vectors_matrices.valid_tm import is_valid_tm


def three_to_four(tm_named, repay_named, do_tm_scenarios=False):

    out_path = f'data/output/{first_hash}/'

    # If 4x4 matrices to be recomputed
    if do_tm_scenarios:

        # Extract the np.array from the NamedArrays
        tm = tm_named.data
        repay = repay_named.data

        # Check if the last two dimensions of tm are (3, 3)
        if tm.shape[-2:] != (3, 3):
            raise ValueError('Input not 3x3 matrices.')

        # Check if dimensions of repay match the second, third, forth and fifth dimensions of tm
        if tm.shape[1:5] != repay.shape:
            print('tm.shape[1:5]:', tm.shape[1:5])
            print('repay.shape:', repay.shape)
            raise ValueError('Input not matching dimensions.')

        # Set up a nan array with the same dimensions as tm but with the last two dimensions being 4
        four_by_four = np.full(tm.shape[:-2] + (4, 4), np.nan)

        # Set the fourth row of the 4x4 matrix to [0, 0, 0, 1] (repaid state is absorbing)
        four_by_four[..., 3, :] = np.array([0, 0, 0, 1])

        # Set the third column of the 4x4 matrix to the default vector from the 3x3 matrix
        four_by_four[..., :3, 2] = tm[..., :, 2]

        # Set the top 3 items in the fourth column to the repayment vector
        four_by_four[..., :3, 3] = repay[..., :] * (1 - tm[..., :, 2])

        # Compute the sum along lines of the last two dimensions of four_by_four

        # Set the first two columns of 4x4 matrix those of 3x3 matrix (such that sum of 4x4 rows is 1)
        scale = 1 - four_by_four[..., :3, -2:].sum(axis=-1, keepdims=True)

        norm = tm[..., :, :2].sum(axis=-1, keepdims=True)
        norm[norm == 0] = 1

        four_by_four[..., :3, :2] = tm[..., :, :2] / norm * scale

        # Define NamedArray
        four_by_four = NamedArray(names=['scenario', 'bank', 'portfolio_type', 'future_quarter', 'stage4m', 'stage4m'],
                                  data=four_by_four)
        validate_named_array(four_by_four)

        # Save to h5 file
        save_named_array(f'{out_path}/tm_scenarios_4x4.h5', four_by_four, 'four_by_four')

    # If 4x4 matrices to be read from file
    else:
        four_by_four = load_named_array(f'{out_path}/tm_scenarios_4x4.h5', 'four_by_four')

    # check if the output is a valid transition matrix
    if not is_valid_tm(four_by_four):
        print(four_by_four.data)
        raise ValueError('Output not transition matrix.')

    # output
    return four_by_four
