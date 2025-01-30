from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

from controls import *
from scripts.vectors_matrices.named_array import *


def plot_tm_scenarios(tm_scenarios_3x3_in, banks, portfolio_types, bank_portfolios, historical_tms,
                      only_bank_portfolio, historical_periods, historical_exposure, verbose=False):

    tmp_3x3 = tm_scenarios_3x3_in.data.copy()
    tm_scenarios_3x3 = NamedArray(names=tm_scenarios_3x3_in.names.copy(),
                                  data=100*tmp_3x3)

    # loop over banks
    for ib, b in enumerate(banks):

        # loop over portfolios
        for ip, p in enumerate(portfolio_types):
            if not (ib, ip) in bank_portfolios:
                continue

            if only_bank_portfolio != []:
                if not (ib, ip) in only_bank_portfolio:
                    continue

            # loop over stage 1 and 2 and 3
            for s in [1, 2, 3]:
                if verbose:
                    print(f'Plotting default rates for bank {b} (#{ib}), portfolio {p}, stage {s}')

                # get zscore scenarios
                slice_2d = tm_scenarios_3x3.data[:, ib, ip, :, s-1, 2].copy()

                if np.all(np.isnan(slice_2d)):
                    if verbose:
                        print('   all nan, not plotting')
                    continue

                # get historical pds
                histo_tm = 100*historical_tms.data[ib, ip, :, s-1, 2].copy()
                max_hist = histo_tm.shape[0]  # remember max index of historical data
                histo_tm = np.repeat(histo_tm[np.newaxis, ...], slice_2d.shape[0], axis=0)

                # concatenate historical data on top of scenarios along axis 1
                full_scen_hist = np.concatenate([histo_tm, slice_2d], axis=1)

                # Preparing DataFrame for seaborn
                df = pd.DataFrame(full_scen_hist.T).reset_index().rename(columns={'index': 'quarter'})
                # reshape long
                df = pd.melt(df, id_vars=['quarter']).drop(columns=['variable'])

                # Plot the generated scenarios
                for interval in plot_intervals:
                    sns.lineplot(df, x='quarter', y='value', estimator='median', errorbar=('pi', interval), color='C0')
                plt.ylabel(f'Quarterly Default Rate (%)')
                plt.title(f'Stage {s} Default Rate: Bank {b} - {p} Lending Portfolio')
                if s == 3:
                    plt.title(f'Stage {s} Non-Cure Rate: Bank {b} - {p} Lending Portfolio')
                plt.xlabel('')

                # set xticks to be every ny years
                quarter_index = pd.period_range(start=historical_periods[0], freq='Q', periods=full_scen_hist.shape[1])
                ny = 2
                plt.xticks(np.arange(0, len(quarter_index), ny * 4),
                           quarter_index[np.arange(0, len(quarter_index), ny * 4)], rotation=45)

                # add a dotted dark red line for the latest historical data
                plt.axvline(x=max_hist-1, color='maroon', linestyle='--')

                plt.tight_layout()

                # if folder does not exist, create it
                if not os.path.exists(f'plots/{first_hash}/tms'):
                    os.makedirs(f'plots/{first_hash}/tms')
                if not os.path.exists(f'plots/{first_hash}/banks/b{b}_{p}'):
                    os.makedirs(f'plots/{first_hash}/banks/b{b}_{p}')

                # save figure
                plt.savefig(f'plots/{first_hash}/tms/defaultrate_bank{b}_{p}_s{s}.png', dpi=300)
                plt.savefig(f'plots/{first_hash}/banks/b{b}_{p}/2. defaultrate_bank{b}_{p}_s{s}.png', dpi=300)
                plt.close()
