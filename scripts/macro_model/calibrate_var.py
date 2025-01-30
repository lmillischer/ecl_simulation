# Laurent Millischer, lmillischer@jvi.org

from statsmodels.tsa.api import VAR, VARMAX
import numpy as np
import pandas as pd
import warnings

from scripts.macro_model.import_var_from_eviews import rearrange_dataframe, rearrange_endog

warnings.filterwarnings("ignore", category=FutureWarning)


# Function to calibrate the VAR model based on historical data
def calibrate_var(macro_data, n_lags):
    # VAR calibration
    macro_data.index = pd.Period(macro_data.index, freq='Q')
    model = VAR(macro_data)
    results = model.fit(n_lags)

    # Save coefficients, residuals, and variance-covariance matrix
    coefs = results.params
    coef_vcov = results.cov_params()
    residuals = results.resid
    res_vcov = results.sigma_u

    return model, coefs, coef_vcov, residuals, res_vcov


def calibrate_var_exog(macro_data, var_order_endog, var_order_exog, exog_vars, do_calibrate_var, verbose=False):

    if do_calibrate_var:
        # prepare macro data
        macro_data.index = pd.PeriodIndex(macro_data.index, freq='Q')
        coefs = macro_data.copy()
        for col in exog_vars:
            coefs[f'{col}'] = coefs[col].shift(-1)  # in order to use contemporaneous exogenous variables shift by -1
        # for the VAR calibration to run, fill the NAs forward. this does not matter as we will
        #    discard the equations for the exogeneous variables anyway
        coefs.fillna(method='ffill', inplace=True)

        # define endogenous and exogenous variables
        df_exog = macro_data[exog_vars]

        # VARMAX calibration
        model = VAR(coefs)
        results = model.fit(var_order_endog)

        # Save coefficients, residuals, and variance-covariance matrix of the endogenous variables
        coefs = results.params
        coefs = coefs.drop(columns=exog_vars)  # we are interested only in the endogenous variables
        coef_vcov = results.cov_params()
        # loop through all levels of the index to drop the exogenous variables
        for idx in coef_vcov.index:
            if idx[1] in exog_vars:
                coef_vcov.drop(idx, inplace=True)
        for col in coef_vcov.columns:
            if col[1] in exog_vars:
                coef_vcov.drop(col, axis=1, inplace=True)
        residuals = results.resid
        residuals.drop(columns=exog_vars, inplace=True)
        res_vcov = np.cov(residuals, rowvar=False)

        if verbose:
            print(results.summary())

        # Now calibrate a VAR model only for the exogenous variables
        model_x = VAR(df_exog)
        results_x = model_x.fit(var_order_exog)

        coefs_x = results_x.params
        coef_vcov_x = results_x.cov_params()
        residuals_x = results_x.resid
        res_vcov_x = results_x.sigma_u
        if verbose:
            print(results_x.summary())

        # print('endo var coefs\n', coefs)

    # else read in the calibrated parameters from an excel file
    else:
        incl_or_excl = 'Incl'  # 'Incl' or 'Excl'
        sheet_endog = f'Domestic VARX - {incl_or_excl} Pandemic'
        sheet_exog = f'Foreign VAR - {incl_or_excl} Pandemic'

        # we don't need model and residuals
        model = None
        residuals = None
        model_x = None
        residuals_x = None

        # coef matrix endog
        coefs = pd.read_excel('data/input/Domestic VARX and Foreign VAR.xlsx',
                                  sheet_name=sheet_endog,
                                  usecols='B:J', skiprows=7, nrows=45, engine='openpyxl')
        coefs = coefs.dropna(subset=['Unnamed: 1'])
        coefs = pd.concat([coefs[coefs['Unnamed: 1'] == 'C'], coefs[coefs['Unnamed: 1'] != 'C']],
                              ignore_index=True)  # move line where 'Unnamed: 1' is 'C' to the top
        coefs.set_index('Unnamed: 1', inplace=True)  # set 'Unnamed: 1' as index
        coefs = coefs.apply(pd.to_numeric, errors='coerce')  # convert to numeric

        # coef matrix exog
        coefs_x = pd.read_excel('data/input/Domestic VARX and Foreign VAR.xlsx',
                              sheet_name=sheet_exog,
                              usecols='B:D', skiprows=7, nrows=21, engine='openpyxl')
        coefs_x = coefs_x.dropna(subset=['Unnamed: 1'])
        coefs_x = pd.concat([coefs_x[coefs_x['Unnamed: 1'] == 'C'], coefs_x[coefs_x['Unnamed: 1'] != 'C']],
                          ignore_index=True)  # move line where 'Unnamed: 1' is 'C' to the top
        coefs_x.set_index('Unnamed: 1', inplace=True)  # set 'Unnamed: 1' as index
        coefs_x = coefs_x.apply(pd.to_numeric, errors='coerce')  # convert to numeric
        coefs_x = rearrange_dataframe(coefs_x, columns=False, rows=True, exclude_row1=True)

        # coef_vcov matrix
        coef_vcov = pd.read_excel('data/input/Domestic VARX and Foreign VAR.xlsx',
                                  sheet_name=sheet_endog,
                                  usecols='N:CW', skiprows=3, nrows=89, engine='openpyxl')
        coef_vcov = rearrange_endog(coef_vcov)
        # coef_vcov matrix exog
        coef_vcov_x = pd.read_excel('data/input/Domestic VARX and Foreign VAR.xlsx',
                                  sheet_name=sheet_exog,
                                  usecols='H:Q', skiprows=3, nrows=10, engine='openpyxl')
        coef_vcov_x = rearrange_dataframe(coef_vcov_x)

        # residuals variance covariance matrix
        res_vcov = pd.read_excel(io='data/input/Domestic VARX and Foreign VAR.xlsx', sheet_name=sheet_endog,
                                 usecols='C:J', skiprows=56, nrows=1, header=None, engine='openpyxl').values[0]
        res_vcov = np.diag(res_vcov**2)
        # residuals variance covariance matrix exog
        res_vcov_x = pd.read_excel('data/input/Domestic VARX and Foreign VAR.xlsx',
                                 sheet_name=sheet_exog,
                                 usecols='C:D', skiprows=32, nrows=1, header=None, engine='openpyxl').values[0]
        res_vcov_x = np.diag(res_vcov_x**2)

    return ((model, coefs, coef_vcov, residuals, res_vcov),
            (model_x, coefs_x, coef_vcov_x, residuals_x, res_vcov_x))
