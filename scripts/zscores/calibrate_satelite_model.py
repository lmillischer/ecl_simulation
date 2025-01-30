import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def calibrate_satelite_model(macro_vars, z_scores):
    """
    Calibrates a linear model to explain z-scores with macro variables.

    :param macro_vars: 2D array of historical macro variables (shape: years x variables)
    :param z_scores: 1D array of z-scores (length: years)
    :return: tuple (model, r_squared, mse) where model is the calibrated linear model,
             r_squared is the coefficient of determination, and mse is the mean squared error.
    """

    # Check that length of z_scores is the same as the length of macro_vars
    if z_scores.shape[0] != macro_vars.shape[0]:
        raise ValueError('Length of z_scores and macro_vars do not match.')

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(macro_vars, z_scores)

    # Make predictions
    predictions = model.predict(macro_vars)

    # Calculate goodness of fit
    r_squared = r2_score(z_scores, predictions)
    mse = mean_squared_error(z_scores, predictions)

    return model, r_squared, mse
