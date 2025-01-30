from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import os

from controls import *
from scripts.vectors_matrices.named_array import *


def plot_lgd_scenarios(lgd, banks, portfolio_types, bank_portfolios, collateral_types,
                       only_bank_portfolio, historical_periods, verbose=False):

    lgd_scenarios = NamedArray(names=lgd.names.copy(),
                               data=100*lgd.data.copy())

    # loop over banks
    for ib, b in enumerate(banks):

        # loop over portfolios
        for ip, p in enumerate(portfolio_types):
            if not (ib, ip) in bank_portfolios:
                continue

            if only_bank_portfolio != []:
                if not (ib, ip) in only_bank_portfolio:
                    continue

            # loop over stage 1 and 2 and buckets 1, 2, 3 of stage 3
            for s in [1, 2, 3, 4, 5]:

                if s < 3:
                    sprint = s
                elif s == 3:
                    sprint = '3 bucket 1'
                elif s == 4:
                    sprint = '3 bucket 2'
                elif s == 5:
                    sprint = '3 bucket 3'

                if verbose:
                    print(f'Plotting LGDs for bank {b} (#{ib}), portfolio {p}, stage {sprint}')

                # get zscore scenarios
                slice_3d = lgd_scenarios.data[:, ib, ip, :, s-1, :].copy()

                if np.all(np.isnan(slice_3d)):
                    continue

                fig, ax = plt.subplots()

                for ic, c in enumerate(collateral_types):

                    slice2d = slice_3d[:, ic, :]

                    if np.all(np.isnan(slice2d)):
                        continue

                    # Preparing DataFrame for seaborn
                    df = pd.DataFrame(slice2d.T).reset_index().rename(columns={'index': 'quarter'})
                    # reshape long
                    df = pd.melt(df, id_vars=['quarter']).drop(columns=['variable'])
                    # print(df)

                    # Plot the generated scenarios
                    label_added = False
                    for interval in plot_intervals:
                        if not label_added:
                            sns.lineplot(df, x='quarter', y='value', estimator='median', errorbar=('pi', interval),
                                         color=colors[ic], label=f'Collateral: {c}')
                            label_added = True
                        else:
                            sns.lineplot(df, x='quarter', y='value', estimator='median', errorbar=('pi', interval),
                                         color=colors[ic])

                plt.ylabel(f'LGD (%)')
                plt.title(f'Stage {sprint} LGD: Bank {b} - {p} Lending Portfolio')
                plt.xlabel('')

                # set xticks to be every ny years
                quarter_index = pd.period_range(start=historical_periods[-1], freq='Q', periods=slice_3d.shape[2]).shift(1)
                ny = 2
                plt.xticks(np.arange(0, len(quarter_index), ny * 4),
                           quarter_index[np.arange(0, len(quarter_index), ny * 4)], rotation=45)

                # add a dotted dark red line for the last observation
                plt.axvline(x=0, color='maroon', linestyle='--')
                plt.legend()

                plt.tight_layout()

                # If folder does not exist, create it
                if not os.path.exists(f'plots/{first_hash}/lgd/scenarios'):
                    os.makedirs(f'plots/{first_hash}/lgd/scenarios')
                if not os.path.exists(f'plots/{first_hash}/banks/b{b}_{p}'):
                    os.makedirs(f'plots/{first_hash}/banks/b{b}_{p}')

                # save
                plt.savefig(f'plots/{first_hash}/lgd/scenarios/lgd_bank{b}_{p}_s{sprint}.png', dpi=300)
                plt.savefig(f'plots/{first_hash}/banks/b{b}_{p}/3. lgd_bank{b}_{p}_s{sprint}.png', dpi=300)
                plt.close()


def plot_cr_scenarios(cr_mu, cr_sigma2, banks, portfolio_types, bank_portfolios, collateral_types,
                      v2s_mu, v2s_sigma2, sr_mu, sr_sigma2, only_bank_portfolio,
                      verbose=False):

    # loop over banks
    for ib, b in enumerate(banks):

        # loop over portfolios
        for ip, p in enumerate(portfolio_types):
            if not (ib, ip) in bank_portfolios:
                continue

            if only_bank_portfolio != []:
                if not (ib, ip) in only_bank_portfolio:
                    continue

            # loop over stage 1 and 2 and buckets 1, 2, 3 of stage 3
            for s in [1, 2, 3, 4, 5]:

                if s < 3:
                    sprint = s
                elif s == 3:
                    sprint = '3 bucket 1'
                elif s == 4:
                    sprint = '3 bucket 2'
                elif s == 5:
                    sprint = '3 bucket 3'

                if verbose:
                    print(f'Plotting CRs for bank {b} (#{ib}), portfolio {p}, stage {sprint}')

                # count the number of non-missing values
                n_non_missing = np.sum(~np.isnan(cr_mu.data[ib, ip, :, s-1]))
                if n_non_missing == 0:
                    continue

                for ic, c in enumerate(collateral_types):
                    # plot a lognormal distribution
                    mu = cr_mu.data[ib, ip, ic, s-1]
                    sigma2 = cr_sigma2.data[ib, ip, ic, s-1]

                    if np.isnan(mu):
                        continue

                    x = np.linspace(0, 2.5, 1000)
                    plt.plot(x, stats.lognorm.pdf(x, s=sigma2, scale=np.exp(mu)),
                             label=f'{c} Collateral',
                             # label=f'{c}  (mu={mu:.2f}, sigma={sigma2:.2f})',
                             color=colors[ic])

                    if c == 'Real Estate':
                        v2s_mu_tmp = v2s_mu.data[ib, ip]
                        v2s_sigma2_tmp = v2s_sigma2.data[ib, ip]
                        plt.plot(x, stats.lognorm.pdf(x, s=sigma2 + v2s_sigma2_tmp + sr_sigma2,
                                                      scale=np.exp(mu + v2s_mu_tmp + sr_mu)), '--',
                                 label=f'{c} (with time-to-sale)',
                                 color=colors[ic])


                # add vertical line at 1
                plt.axvline(x=1, color='black', linestyle='--')

                plt.xlabel(f'Collateralization Ratio')
                plt.ylabel(f'Density')
                plt.yticks([])
                plt.title(f'Stage {sprint} Collateralization: Bank {b} - {p} Lending Portfolio')
                plt.legend()

                # If folder does not exist, create it
                if not os.path.exists(f'plots/{first_hash}/lgd/collat_ratio'):
                    os.makedirs(f'plots/{first_hash}/lgd/collat_ratio')
                if not os.path.exists(f'plots/{first_hash}/banks/b{b}_{p}'):
                    os.makedirs(f'plots/{first_hash}/banks/b{b}_{p}')

                # save
                plt.savefig(f'plots/{first_hash}/lgd/collat_ratio/cr_bank{b}_{p}_s{sprint}.png', dpi=300)
                plt.savefig(f'plots/{first_hash}/banks/b{b}_{p}/5. cr_bank{b}_{p}_s{sprint}.png', dpi=300)
                plt.close()
