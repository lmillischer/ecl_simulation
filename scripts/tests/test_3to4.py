import pandas as pd

from scripts.vectors_matrices.add_repayments_to_tm import three_to_four

def test_3to4():

    # read in test excel from C6 to K35
    df = pd.read_excel('../data/tests/Model Testing.xlsx', sheet_name=f'TM3 to TM4', usecols='C:G', skiprows=6, nrows=3)

    # tm is the first three columns of the df
    tm = df.to_numpy()[:, 0:3]

    # repay is the last column of the df
    repay = df.to_numpy()[:, 4].reshape(3, 1)

    # test the three_to_four function
    four_by_four = three_to_four(tm, repay)
    print(four_by_four)



test_3to4()

