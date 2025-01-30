import pandas as pd


def get_zscore_model_parameters():
    """
    Function to read the z-score model parameters from an Excel file
    :return: coefficients, coefficient variance-covariance matrix, and residual variance
    """

    # Get the coefficients from
    zscore_model_coefs = pd.read_excel()
    zscore_model_coef_vcov = pd.read_excel()
    zscore_model_residual_variance = pd.read_excel()


    return zscore_model_coefs, zscore_model_coef_vcov, zscore_model_residual_variance