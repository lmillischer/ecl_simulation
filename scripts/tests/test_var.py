import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

from scripts.macro_model.calibrate_var import calibrate_var
from scripts.macro_model.check_var_stability import var_is_stable


def test_var(nlags=1):

    # read in test excel from B4 to cell F94
    df = pd.read_excel('data/tests/Model Testing.xlsx', sheet_name=f'VAR({nlags}) Estimation', usecols='B:F', skiprows=3, nrows=90)

    # set first column as index
    df.rename(columns={'Unnamed: 1': 'date'}, inplace=True)
    df.set_index('date', inplace=True)

    model, coefs, coef_vcov, residuals, res_vcov = calibrate_var(df, nlags)

    # Define the new order for levels
    new_order_index = ['RGDPQ', 'INFQ', 'URX', 'RPPQ', 'L1.RGDPQ', 'L1.INFQ', 'L1.URX', 'L1.RPPQ', 'const']
    new_order_columns = ['RGDPQ', 'INFQ', 'URX', 'RPPQ', 'L1.RGDPQ', 'L1.INFQ', 'L1.URX', 'L1.RPPQ', 'const']

    # Reorder the index and columns according to the new order
    coef_vcov = coef_vcov.reorder_levels([1, 0], axis=0).sort_index(axis=0)
    coef_vcov = coef_vcov.reorder_levels([1, 0], axis=1).sort_index(axis=1)

    # Now reorder according to the specific order provided by the user
    coef_vcov = coef_vcov.reindex(new_order_index, level=1, axis=0)
    coef_vcov = coef_vcov.reindex(new_order_index, level=0, axis=0)
    coef_vcov = coef_vcov.reindex(new_order_columns, level=1, axis=1)
    coef_vcov = coef_vcov.reindex(new_order_columns, level=0, axis=1)

    is_stable, roots = var_is_stable(coefs)

    def get_var_name(variable, local_vars):
        return [name for name, value in local_vars.items() if value is variable]

    for outdf in [coefs, res_vcov, residuals, coef_vcov]:
        for col in outdf.select_dtypes(include=['float', 'int']):
            outdf[col] = outdf[col].apply(lambda x: f'{x:.5f}'.replace(',', '.'))
            outdf.to_excel(f'data/tests/tmp_testvar_{get_var_name(outdf, locals())[0]}{nlags}.xlsx', index=True, header=True)
