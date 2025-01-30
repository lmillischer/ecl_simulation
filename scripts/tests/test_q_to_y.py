import pandas as pd
import numpy as np

from scripts.archive.simulate_var import q_to_y


def test_q_to_y(do_2d=False):

    # read in columns A:I of the quarterly data
    dfq = pd.read_excel('data/tests/macro_scenario_example_for_QtoYconversion.xlsx', sheet_name='Sheet1',
                        usecols='A:I', nrows=40, skiprows=1)

    dfy = pd.read_excel('data/tests/macro_scenario_example_for_QtoYconversion.xlsx', sheet_name='Sheet1',
                        usecols='AH:AP', nrows=9, skiprows=3)

    for df in [dfq, dfy]:
        # rename first column to date independently of its name
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        # set the index to the first column
        df.set_index('date', inplace=True)


    # prepare in puts for q_to_y
    # history is the a numpy array of the first seven rows of dfq
    history = dfq.head(7).to_numpy()
    print('history shape: ', history.shape)
    # scenarios_q is all the remaining rows of dfq
    scenarios_q = dfq.tail(34).to_numpy()

    if not do_2d:

        # transform scenarios_q into a 3d numpy array
        scenarios_q = np.expand_dims(scenarios_q, axis=0)

        test_y = q_to_y(scenarios_q, history)
        print(np.array(dfy))
        print(test_y)

        diff = np.array(dfy) - test_y

        print(diff)

        test_passed = np.allclose(np.array(dfy), test_y)

        print(test_passed)

    # test the whole thing in two dimensions, i.e. with multiple scenarios
    else:
        # transform scenarios_q into a 3d numpy array
        scenarios_q = np.expand_dims(scenarios_q, axis=0)

        # repeat the scenarios 100 times
        scenarios_q = np.repeat(scenarios_q, 100, axis=0)
        print('Quarterly scenarios shape:', scenarios_q.shape)

        test_y = q_to_y(scenarios_q, history)
        print('Yearly scenarios shape:', test_y.shape)

        test_passed = np.allclose(np.array(dfy), test_y)

        print(test_passed)
