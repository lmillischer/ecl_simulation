
import numpy as np
import h5py
import pandas as pd

from scripts.lgd.project_lgd_into_the_future import project_lgd
from scripts.vectors_matrices.named_array import *
from controls import *
from data.file_paths import input_path


def get_basic_lgds(portfolio_types, collateral_types, verbose=False):

    # create empty array to fill
    basic_lgds = NamedArray(names=['portfolio_type', 'collateral_type', 'stage5'],
                                data=np.full(shape=(n_portfolio_types, n_collateral_types, 5),
                                            fill_value=np.nan))
    validate_named_array(basic_lgds)

    # read excel
    df = pd.read_excel(input_path, sheet_name='Basic LGDs', na_values=['NA', 'NULL', ''], keep_default_na=False)

    # loop over portfolio stat_type
    for ip, p in enumerate(portfolio_types):
        for ic, c in enumerate(collateral_types):
            for b in range(3):
                lgd_value = df[(df.portfolio.str.contains(p)) &
                               (df.coll_type == c) &
                               (df.bucket == b+1)]['lgd'].iloc[0]
                if b == 0:  # bucket 1 LGD valid for stage 1, 2 and 3b1
                    basic_lgds.data[ip, ic, 0:3] = lgd_value
                else:
                    basic_lgds.data[ip, ic, b + 2] = lgd_value  # '+2' because stage 3b1 is index 2
                if verbose:
                    print(f'ptf {p} coll {c} bucket {b+1}, LGD = {lgd_value}')
    #

    # add three empty dimensions: scenarios, banks, quarters
    basic_lgds.data = basic_lgds.data[np.newaxis, np.newaxis, :, :, :, np.newaxis].astype(np.float32)
    basic_lgds.names = ['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage5', 'future_quarter']
    validate_named_array(basic_lgds)

    return basic_lgds


def get_collateralization_ratios(banks, portfolio_types, collateral_types, verbose=False):

    # read excel
    df = pd.read_excel(input_path, sheet_name='Collateralization ratios', skiprows=5,
                       decimal=',')

    # rename columns
    df.rename(columns={'Bank': 'bank',
                       'Collateral_Type': 'collateral_type',
                       'Portfolio': 'portfolio',
                       'Stages': 'stage',
                       'Bucket': 'bucket'
                       }, inplace=True)

    # at this stage, keep only limit = 200
    df = df[df['Limit'] == 200]  # this is the uppler limit of the lognormal fit

    # drop size and limit column
    df.drop(columns=['size_group', 'Limit'], inplace=True)

    # create empty array to fill
    cr_mu = np.full(shape=(n_banks, n_portfolio_types, n_collateral_types, 5),
                    fill_value=np.nan).astype(np.float32)
    cr_sigma2 = np.full(shape=(n_banks, n_portfolio_types, n_collateral_types, 5),
                        fill_value=np.nan).astype(np.float32)

    # loop over banks
    for ib, b in enumerate(banks):
        # loop over portfolio stat_type
        for ip, p in enumerate(portfolio_types):
            # loop over collateral stat_type
            for ic, c in enumerate(collateral_types):
                # loop over stage
                for s in range(3):
                    if verbose:
                        print(f'bank {b} ptf {p} coll {c} stage {s}')
                    # loop over bucket only for stage 3
                    if s == 2:
                        for bucket in range(3):
                            # get the mu and sigma
                            subdf = df[(df.bank == b) &
                                       (df.portfolio == p) &
                                       (df.collateral_type == c) &
                                       (df.stage == s+1) &
                                       (df.bucket == f'Bucket {bucket+1}')]
                            mu = subdf['lognormal_mu'].iloc[0] if len(subdf) > 0 else np.nan
                            sigma = subdf['lognormal_sigma'].iloc[0] if len(subdf) > 0 else np.nan
                            # save the mu and sigma
                            stage_idx = 2 + bucket  # 2 is the index of stage 3
                            cr_mu[ib, ip, ic, stage_idx] = mu
                            cr_sigma2[ib, ip, ic, stage_idx] = sigma ** 2
                    # for stage 1 and 2
                    else:
                        # get the mu and sigma
                        subdf = df[(df.bank == b) &
                                   (df.portfolio == p) &
                                   (df.collateral_type == c) &
                                   (df.stage == s+1)]
                        mu = subdf['lognormal_mu'].iloc[0] if len(subdf) > 0 else np.nan
                        sigma = subdf['lognormal_sigma'].iloc[0] if len(subdf) > 0 else np.nan
                        # save the mu and sigma
                        cr_mu[ib, ip, ic, s] = mu
                        cr_sigma2[ib, ip, ic, s] = sigma ** 2

    # Define NamedArray
    cr_mu = NamedArray(names=['bank', 'portfolio_type', 'collateral_type', 'stage5'],
                       data=cr_mu.astype(np.float32))
    cr_sigma2 = NamedArray(names=['bank', 'portfolio_type', 'collateral_type', 'stage5'],
                           data=cr_sigma2.astype(np.float32))
    validate_named_array(cr_mu)
    validate_named_array(cr_sigma2)

    # Output
    return cr_mu, cr_sigma2


def get_lgd_scenarios(banks, portfolio_types, collateral_types, stages, v2s_mu, v2s_sigma2, sr_mu, sr_sigma2,
                      costs, macro_scenarios, macro_names, tm_scenarios_3x3, do_lgd, historical_tms,
                      amortization_profiles, verbose=False):

    out_path = f'data/output/{first_hash}/'

    # If lgd scenarios to be recomputed
    if do_lgd:

        # Get the basic lgds from Basic LGDs
        basic_lgds = get_basic_lgds(portfolio_types, collateral_types)

        # Get the lognormal fit collateralization ratios for all banks, portfolio types and collateral types
        cr_mu, cr_sigma2 = get_collateralization_ratios(banks, portfolio_types, collateral_types, verbose=False)
        if verbose:
            print('   - Fixed LGDs and collateralization ratios read')

        # With all the information, create the lgd scenarios
            lgd = project_lgd(collateral_types, n_scenarios, stages, n_scenario_quarters, amortization_profiles,
                          basic_lgds, cr_mu, cr_sigma2, v2s_mu, v2s_sigma2, sr_mu, sr_sigma2,
                          macro_scenarios, macro_names, tm_scenarios_3x3, historical_tms, verbose=verbose)
        if verbose:
            print('   - LGD scenarios created')

        # Add the costs
        costs_by_collateral = NamedArray(names=['collateral_type'],
                                         data=np.array([costs[c] for c in collateral_types]))
        for ict, ct in enumerate(collateral_types):
            lgd.data[:, :, :, ict, :, :] += costs_by_collateral.data[ict]

        # If all lgd are nan, raise error
        if np.isnan(lgd.data).all():
            raise ValueError(f'All lgd values are nan.')

        # save lgd in an h5 file
        save_named_array(f'{out_path}/lgd.h5', lgd, 'lgd')
        save_named_array(f'{out_path}/cr_mu.h5', cr_mu, 'cr_mu')
        save_named_array(f'{out_path}/cr_sigma2.h5', cr_sigma2, 'cr_sigma2')

    # If lgd scenarios to be read from file
    else:
        lgd = load_named_array(f'{out_path}/lgd.h5', 'lgd')
        cr_mu = load_named_array(f'{out_path}/cr_mu.h5', 'cr_mu')
        cr_sigma2 = load_named_array(f'{out_path}/cr_sigma2.h5', 'cr_sigma2')

    return lgd, cr_mu, cr_sigma2
