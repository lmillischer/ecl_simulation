import numpy as np


def project_portfolios(portfolios, transition_matrices):
    """
    Process the data by multiplying portfolio vectors with transition matrices.

    :param portfolios: Array of shape (n_banks, n_portfolios, 4) with portfolios (stage 1, 2, 3, repaid)
    :param transition_matrices: Array of shape (n_scenarios, n_banks, n_portfolios, n_quarters, 4, 4) with transition matrices
    :return: Numpy array of shape (n_scenarios, n_banks, n_portfolios, n_quarters, 4)
    """
    n_scenarios, n_banks, n_portfolios, n_quarters, _, _ = transition_matrices.shape
    output = np.full((n_scenarios, n_banks, n_portfolios, n_quarters, 4), fill_value=np.nan)

    # -----------------------------------------------------------------------------
    # Option 1: using einsum. Elegant, quick, but needs to be tested
    # -----------------------------------------------------------------------------

    # Expand dimensions of portfolios to align with transition_matrices for broadcasting
    expanded_portfolios = portfolios[np.newaxis, :, :, np.newaxis, :]
    output[:, :, :, 0, :] = np.einsum('...ij,...j->...i', transition_matrices[:, :, :, 0, :, :], expanded_portfolios[:, :, :, 0, :])
    for quarter in range(1, n_quarters):
        output[:, :, :, quarter, :] = np.einsum('...ij,...j->...i', transition_matrices[:, :, :, quarter, :, :], output[:, :, :, quarter - 1, :])

    # -----------------------------------------------------------------------------
    # Option 2: with loops, slower, less elegant, easier to understand
    # -----------------------------------------------------------------------------

    # for scenario in range(n_scenarios):
    #     for bank in range(n_banks):
    #         for portfolio in range(n_portfolios):
    #             vector = portfolios[bank, portfolio]
    #             for quarter in range(n_quarters):
    #                 matrix = transition_matrices[scenario, bank, portfolio, quarter]
    #                 vector = np.dot(matrix, vector)
    #                 output[scenario, bank, portfolio, quarter] = vector

    # before returning, drop the fourth item of the last dimension
    # (it is the repaid portfolio, which is not needed anymore)
    output = output[:, :, :, :, 0:3]

    return output
