
import h5py
import numpy as np

from scripts.vectors_matrices.named_array import load_named_array, NamedArray
from scripts.zscores.zscore_to_tm import zscore_to_tm

chosen_run_hash = 'XW8M2m2x'

ecl_scenarios_rel = load_named_array('../../data/output/XW8M2m2x/cl_rate_scenarios.h5', 'cl_rate_scenarios')
current_portfolios = load_named_array('../../data/output/XW8M2m2x/input_named_data.h5', 'current_portfolios')
historical_tms = load_named_array('../../data/output/XW8M2m2x/input_named_data.h5', 'historical_tms')
upper_bounds = load_named_array('../../data/output/XW8M2m2x/zscores.h5', 'upper_bounds')
lower_bounds = load_named_array('../../data/output/XW8M2m2x/zscores.h5', 'lower_bounds')
rhos = load_named_array('../../data/output/XW8M2m2x/zscores.h5', 'rhos')
lgd = load_named_array('../../data/output/XW8M2m2x/lgd.h5', 'lgd')
tm_scenarios_3x3 = load_named_array('../../data/output/XW8M2m2x/tm_scenarios_3x3.h5', 'tm_scenarios_3x3')

with h5py.File('../../data/output/XW8M2m2x/input_data.h5', 'r') as h5file:
    banks = h5file['banks'][:]
    portfolio_types = h5file['portfolio_types'][:]
    portfolio_types = [byte.decode('utf-8') for byte in portfolio_types]  # decode strings
    bank_portfolio_stages = h5file['bank_portfolio_stages'][:]
    bank_portfolio_stages = list(map(tuple, bank_portfolio_stages))

# for ib, b in enumerate(banks):
#     print(f'Bank #{ib} is {b}')


def which_ptfs_missing():

    # loop over banks and portfolio types
    for s in [1, 2, 3]:
        n_ptf_with_exposure = 0
        n_ptf_missing = 0
        print(f'\nStage {s}')

        for ib, b in enumerate(banks):
            for ip, p in enumerate(portfolio_types):

                if (ib, ip, s-1) not in bank_portfolio_stages:
                    continue

                n_ptf_with_exposure += 1

                if np.all(np.isnan(ecl_scenarios_rel.data[:, ib, ip, s-1])):
                    print(f'Bank #{ib} and portfolio type {p} are missing in stage {s}')
                    n_ptf_missing += 1

        print(f'Number of portfolio types with exposure: {n_ptf_with_exposure}')
        print(f'Number of portfolio types missing: {n_ptf_missing}')

    # loop over banks and portfolio types
    print('\nHistorical TMs')
    for ib, b in enumerate(banks):
        for ip, p in enumerate(portfolio_types):
            if (ib, ip) not in bank_portfolio_stages:
                continue

            # check if historical TMs are all missing
            if np.all(np.isnan(upper_bounds.data[ib, ip, :, :])):
                print(f'Bank #{ib} and portfolio type {p} are missing in historical TMs')
    print(historical_tms.names)


which_ptfs_missing()


def study_specific_bank_portfolio(bank, portfolio_type, stage):

    # print the TM history for that bank and portfolio
    print(historical_tms.names)
    htm = historical_tms.data[bank, portfolio_type, :, :, :]
    haverage_tm = np.mean(htm, axis=0)
    print('Historical average TM')
    # print(haverage_tm)
    print(np.nanmean(htm, axis=0))

    print('Historical TMs')
    print(historical_tms.data[bank, portfolio_type, :, :, :])

    # print the upper bounds (= average historical TM)
    # print('\nUpper bounds:')
    ub = upper_bounds.data[bank, portfolio_type, :, :]
    lb = lower_bounds.data[bank, portfolio_type, :, :]
    rho = rhos.data[bank, portfolio_type]
    # print(ub)
    # print(lb)

    # print LGD
    print('\nLGD:')
    # ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage5', 'future_quarter']
    lgd_bp = lgd.data[:, bank, portfolio_type, :, stage-1, :]
    lgd_bp = np.nanmean(lgd_bp, axis=(0, 2))
    print(lgd_bp)
    # for (7, 0) lgd is somehow missing

    # print ECL
    # print('ecl_scenarios_rel', ecl_scenarios_rel.names)
    print('\nECL:')
    print(np.mean(ecl_scenarios_rel.data[:, bank, portfolio_type, stage-1]))
    print(ecl_scenarios_rel.data[:, bank, portfolio_type, stage-1])

    # print current exposure
    print('\nCurrent exposure:')
    print(current_portfolios.names)
    print(current_portfolios.data[bank, portfolio_type, :, stage-1])

    # print default rates
    print('\nDefault rates:')
    print(tm_scenarios_3x3.names)
    # dr_array = tm_scenarios_3x3.data[:, bank, portfolio_type, :, stage-1, 2]

    def print_tm_by_hand(z_in):

        named_zin = NamedArray(names=[], data=np.array([z_in]))

        tm_by_hand = zscore_to_tm(named_zin, ub, lb, np.array(rho))
        print(tm_by_hand.data)

    # print(f'rho = {rho}')
    # for z in [-2, 0, 2]:
    #     print(f'\n TM for z={z}')
    #     print_tm_by_hand(z)

    # plot a histogram
    # import matplotlib.pyplot as plt
    # plt.hist(dr_array.flatten(), bins=100)
    # plt.show()

    # print('median', 100*np.median(, axis=0))
    # print('mean', 100*np.mean(tm_scenarios_3x3.data[:, bank, portfolio_type, :, stage-1, 2], axis=0))
    # print('mean', 100*np.nanmean(tm_scenarios_3x3.data[:, bank, portfolio_type, :, stage-1, 2], axis=0))
    # print('min', 100*np.min(tm_scenarios_3x3.data[:, bank, portfolio_type, :, stage - 1, 2], axis=0))
    # print('max', 100*np.max(tm_scenarios_3x3.data[:, bank, portfolio_type, :, stage - 1, 2], axis=0))


# study_specific_bank_portfolio(39, 3, 1)
