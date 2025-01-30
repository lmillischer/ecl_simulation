import numpy as np


def var_is_stable(var_coef_matrix):
    """
    Compute roots of comp coef matrix and tell whether the model is stable or not.

    Parameters:
    var_coef_matrix : 2D array-like

    Returns:
    out : int
        1 if the process is stable, 0 if the process is explosive.
    roots : array
        Sorted array of roots.
    """
    # print(var_coef_matrix)
    # The coefficient matrix coming out of statsmodels is never square
    # we drop the first line of the matrix, corresponding to the intercept
    var_coef_matrix = np.array(var_coef_matrix)  # make sure it is a numpy array
    var_coef_matrix = var_coef_matrix[1:, :]

    n_lines, n_columns = var_coef_matrix.shape
    # If the matrix is square, we can compute the eigenvalues directly
    if n_lines == n_columns:
        roots = np.abs(np.linalg.eigvals(var_coef_matrix.T))
    # If the matrix is not square, we need to build the companion matrix
    else:
        tmp_add = np.eye(n_lines)[:, :n_columns - n_lines]
        companion_matrix = np.hstack([var_coef_matrix, tmp_add])

        # roots are the absolute values of the eigenvalues
        roots = np.abs(np.linalg.eigvals(companion_matrix))

    maxroots = np.max(roots)

    is_stable = True if maxroots < 1 else False  # True - process stable, False - process explosive
    # print(f'   Is the process stable? {is_stable}')
    return is_stable, roots
