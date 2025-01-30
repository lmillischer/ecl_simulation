import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from controls import *


def plot_exposure(historical_portfolios, current_portfolios, banks, portfolio_types, bank_portfolios,
                  collateral_types, only_bank_portfolio, historical_periods):

    current_portfolio_stage12 = current_portfolios.data[:, :, :, 0:2]
    current_portfolio_stage3 = np.nansum(current_portfolios.data[:, :, :, 2:5], axis=3, keepdims=True)
    current_portfolio_3stages = np.concatenate((current_portfolio_stage12, current_portfolio_stage3), axis=3)
    current_portfolio_all_collateral = np.nansum(current_portfolio_3stages, axis=2)

    # loop over banks and portfolios, plot historical exposure
    for ib, b in enumerate(banks):
        for ip, p in enumerate(portfolio_types):

            # ----------------------------------------
            # Plot historical exposure
            # ----------------------------------------

            # if the tuple (ib, ip) is not in the array of all valid bank portfolios, skip
            if not (ib, ip) in bank_portfolios:
                continue

            if only_bank_portfolio != []:
                if not (ib, ip) in only_bank_portfolio:
                    continue

            print(f'Plotting exposure for bank {b}, portfolio {p} ({ip})')

            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            plt.plot(historical_portfolios.data[ib, ip, 0, :], label=f'Stage 1')
            plt.plot(historical_portfolios.data[ib, ip, 1, :], label=f'Stage 2')
            plt.plot(historical_portfolios.data[ib, ip, 2, :], label=f'Stage 3')

            # x labels quarterly
            quarter_index = pd.period_range(start=historical_periods[0], freq='Q',
                                            periods=historical_portfolios.data.shape[3])
            ny = 2
            plt.xticks(np.arange(0, len(quarter_index), ny * 4),
                       quarter_index[np.arange(0, len(quarter_index), ny * 4)],
                       rotation=45)

            # add 3 dots at x=12 and y = current_portfolios_all_collateral[ib, ip, 0-1-2]
            last_year = historical_portfolios.data.shape[3] - 1
            plt.scatter([last_year, last_year, last_year], current_portfolio_all_collateral[ib, ip, :], color='black', marker='o', s=10,
                        label='Current exposure')

            plt.title(f'Total exposure bank {b}, portfolio {p} ({ip})')
            plt.legend()

            # If folder does not exist, create it
            if not os.path.exists(f'plots/{first_hash}/exposure'):
                os.makedirs(f'plots/{first_hash}/exposure')
            if not os.path.exists(f'plots/{first_hash}/banks/b{b}_{p}'):
                os.makedirs(f'plots/{first_hash}/banks/b{b}_{p}')

            # save
            plt.savefig(f'plots/{first_hash}/exposure/exposure_bank{b}_{p}.png')
            plt.savefig(f'plots/{first_hash}/banks/b{b}_{p}/1. hist_exposure_bank{b}_{p}.png')
            plt.close()

            # -------------------------------------------------------------------------
            # Plot composition of current exposure by collateral type
            # -------------------------------------------------------------------------

            for s in range(1, 4):

                prev_exp = 0

                tot_exposure = np.nansum(current_portfolio_3stages[ib, ip, :, s - 1])
                if tot_exposure == 0:
                    continue

                for ic, c in enumerate(collateral_types):

                    exp_for_plotting = 100*np.nansum(current_portfolio_3stages[ib, ip, ic, s-1])/tot_exposure
                    plt.bar(0, np.array(exp_for_plotting), bottom=prev_exp, label=c, color=colors[ic])
                    prev_exp += exp_for_plotting

                # Customizing plot
                plt.ylabel('Share of Total Exposure (%)')
                plt.title(f'Collateral Types: Bank {b} - {p} Lending Portfolio Stage {s}')
                plt.gca().set_xticks([])
                plt.legend()
                plt.tight_layout()

                # Save plot
                plt.savefig(f'plots/{first_hash}/exposure/exposure_bank{b}_{p}_s{s}_comp.png', dpi=300)
                plt.savefig(f'plots/{first_hash}/banks/b{b}_{p}/1. exposure_bank{b}_{p}_s{s}_comp.png', dpi=300)
                plt.close()

