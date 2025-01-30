
from scripts.tests.test_macro_to_zscore import test_macro_to_zscores


def run_tests():

    test_macro_to_zscores()


# check that even with very high LGDs and PDs, ECLs don't go above 100% of the exposure