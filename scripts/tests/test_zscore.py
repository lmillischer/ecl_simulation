import pandas as pd

from scripts.archive.tm_to_zscore import historical_tms_to_zscore


def test_zscore():

    # read in test excel from C6 to K35
    df = pd.read_excel('../data/tests/Model Testing.xlsx', sheet_name=f'Z-Score Estimation', usecols='C:K', skiprows=5, nrows=30)

    # reshape the dataframe to a numpy array of shape (len(df), 3, 3)
    tms = df.to_numpy().reshape(len(df), 3, 3)

    zscore_vector, zscore_variance, out_rho, upper_bounds, lower_bounds = historical_tms_to_zscore(tms, 0.1, verbose=True)

    def get_var_name(variable, local_vars):
        return [name for name, value in local_vars.items() if value is variable]

    for outdf in [zscore_vector, zscore_variance, out_rho]:
        print('\n', get_var_name(outdf, locals())[0])
        print(outdf)
        if outdf is zscore_vector:
            outdf = pd.DataFrame(outdf, columns=['Column1'])
            outdf.to_excel(f'tmp_testzscore_zscore.xlsx', index=True, header=True)


test_zscore()
