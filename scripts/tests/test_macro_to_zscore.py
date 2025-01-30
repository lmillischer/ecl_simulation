import numpy as np

from scripts.zscores.macro_to_zscores import macro_to_zscores


def test_macro_to_zscores():
    # Test data
    all_banks = [f'Bank{i}' for i in range(43)]
    portfolio_types = ['Type1', 'Type2']
    macro_names = [f'Macro{i}' for i in range(9)]
    scenario_count = 5
    horizon_count = 3
    macro_scenarios = np.random.rand(scenario_count, horizon_count, len(macro_names))

    # Call the function
    zscore_result = macro_to_zscores(portfolio_types, macro_scenarios)

    # Print the results
    print('zscore_result.shape:', zscore_result.shape)

    return True