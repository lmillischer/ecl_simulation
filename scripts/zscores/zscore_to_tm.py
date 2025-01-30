import numpy as np
from scipy.stats import norm

from scripts.vectors_matrices.valid_tm import is_valid_tm
from scripts.vectors_matrices.named_array import NamedArray, validate_named_array


def zscore_to_tm(zscore_in, upper_bounds, lower_bounds, rho):
    """
    Using the upper/lower bounds of the average historical transition matrix and the correlation
    parameter rho, the function transforms the (systematic) zscore into a transition matrix.
    :param zscore: z-score
    :param upper_bounds: matrix with upper bounds of the average TM
    :param lower_bounds: matrix with lower bounds of the average TM
    :param rho: correlation parameter
    :return: 3x3 z-score conditional transition matrix
    """

    zscore = NamedArray(names=zscore_in.names.copy(),
                        data=zscore_in.data.copy())

    if not type(zscore) == NamedArray:
        raise ValueError('zscore_named must be a NamedArray.')

    # reshape all inputs so that they have matching shapes
    if zscore.data.shape == (1, ):
        zscore.data = zscore.data[..., np.newaxis]
        rho = NamedArray(data=rho, names=[''])
        upper_bounds = NamedArray(data=upper_bounds, names=['stage3', 'stage3'])
        lower_bounds = NamedArray(data=lower_bounds, names=['stage3', 'stage3'])
        dim_names = ['stage3', 'stage3']
    elif zscore.names == ['scenario', 'bank', 'portfolio_type', 'future_quarter']:
        zscore.data = zscore.data[..., np.newaxis, np.newaxis]
        rho.data = rho.data[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]
        upper_bounds.data = upper_bounds.data[np.newaxis, :, :, np.newaxis, :, :]
        lower_bounds.data = lower_bounds.data[np.newaxis, :, :, np.newaxis, :, :]
        dim_names = ['scenario', 'bank', 'portfolio_type', 'future_quarter', 'stage3', 'stage3']

    # perform the calculation
    tm = (norm.cdf((upper_bounds.data - np.sqrt(rho.data) * zscore.data) / np.sqrt(1 - rho.data)) -
          norm.cdf((lower_bounds.data - np.sqrt(rho.data) * zscore.data) / np.sqrt(1 - rho.data)))

    # Prepare NamedArray for outputting
    tm = NamedArray(names=dim_names,
                    data=tm)
    validate_named_array(tm)

    # Check if valid transition matrices
    if not is_valid_tm(tm):
        raise ValueError('Output not transition matrix.')

    return tm
