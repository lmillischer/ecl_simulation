import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

from controls import *


def plot_historical_zscores(portfolio_types, macro_data, historical_zscores, banks,
                            bank_portfolios, macro_names):


    # from macro_data, take only the years that are in historical_years
    gdp = macro_data.data[:-1, macro_names.index('RGDP')]
    unemp = macro_data.data[:-1, macro_names.index('URX')]

    # take only first 6 banks
    historical_zscores = historical_zscores.data[:, :, :]

    x1 = np.array(range(2009, 2022))

    # loop over portfolios
    for ip, p in enumerate(portfolio_types):

        fig, ax = plt.subplots()
        default_fig_width, default_fig_height = fig.get_size_inches()
        aspect_ratio = default_fig_width / default_fig_height

        # Specify your desired height in cm and convert to inches
        desired_height_cm = 15  # for example
        desired_height_in = desired_height_cm / 2.54

        # Calculate the corresponding width to maintain the aspect ratio
        calculated_width_in = desired_height_in * aspect_ratio

        fig, ax1 = plt.subplots(figsize=(calculated_width_in, desired_height_in), dpi=300)
        ax1.set_ylabel('Z-Score')

        # loop over banks
        for ib, b in enumerate(banks):
            y = historical_zscores[ib, ip, :]

            ax1.plot(x1, y, '-', color='lightblue')

        # change ytitle color
        ax1.yaxis.label.set_color('cadetblue')
        ax1.tick_params(axis='y', colors='cadetblue')

        # plot gdp with using a yaxis on the right
        ax2 = ax1.twinx()

        if p in ['Commercial', 'Microcredit']:
            ax2.plot(x1, gdp, '-', color='black', label='Real GDP growth')
            ax2.set_ylabel('Real GDP growth [%]')
        else:
            ax2.plot(x1, unemp, '-', color='black', label='Unemployment')
            ax2.set_ylabel('Unemployment [%]')

        plt.title(f'Historical Z-Score: {p} Lending Porfolios')
        plt.xlabel('')
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig(f"plots/zscores/zscore_{p}.png")
        plt.close()

        do_single_banks = False

        if do_single_banks:
            # loop over all banks and plot zscores for that bank and portfolio
            for ib, b in enumerate(banks):
                if not (ib, ip) in bank_portfolios:
                    continue

                fig, ax1 = plt.subplots(figsize=(calculated_width_in, desired_height_in), dpi=300)
                ax1.set_ylabel('z-score')

                y = historical_zscores[ib, ip, :]

                ax1.plot(x1, y, '-', color='lightblue')

                # plot gdp with using a yaxis on the right
                ax2 = ax1.twinx()
                if p in ['Commercial']:
                    ax2.plot(x1, gdp, '-', color='black', label='Real GDP growth')
                    ax2.set_ylabel('Real GDP growth [%]')
                else:
                    ax2.plot(x1, unemp, '-', color='black', label='Unemployment')
                    ax2.set_ylabel('Unemployment [%]')

                # add legend
                ax1.legend()

                plt.title(f'Historical z-score for {ib} in {p}')
                plt.xlabel('')
                plt.tight_layout()

                # Save the plot as a PNG file
                plt.savefig(f"../../plots/zscores/all_ptfs/zscore_{b}_{p}.png")


def plot_zscore_scenarios(zscore_scenarios, banks, portfolio_types, bank_portfolios,
                          historical_zscores, only_bank_portfolio, historical_periods, verbose=False):

    for ib, b in enumerate(banks):
        for ip, p in enumerate(portfolio_types):
            if not (ib, ip) in bank_portfolios:
                continue

            if only_bank_portfolio != []:
                if not (ib, ip) in only_bank_portfolio:
                    continue

            if verbose:
                print('   Plotting zscore scenarios for bank', b, 'portfolio', p)

            # get zscore scenarios
            slice_2d = zscore_scenarios.data[:, ib, ip, :]

            # get historical zscores
            histo_z = historical_zscores.data[ib, ip, :].copy()
            max_hist = histo_z.shape[0]  # remember max index of historical data
            histo_z = np.repeat(histo_z[np.newaxis, ...], slice_2d.shape[0], axis=0)

            # concatenate historical data on top of scenarios along axis 1
            full_scen_hist = np.concatenate([histo_z, slice_2d], axis=1)

            # Preparing DataFrame for seaborn
            df = pd.DataFrame(full_scen_hist.T).reset_index().rename(columns={'index': 'quarter'})
            # reshape long
            df = pd.melt(df, id_vars=['quarter']).drop(columns=['variable'])

            # If df all nan, continue
            if df['value'].isnull().all():
                if verbose:
                    print('      > all nan, not plotting')
                continue

            # Plot the generated scenarios before recentering
            for interval in plot_intervals:
                sns.lineplot(df, x='quarter', y='value', estimator='median', errorbar=('pi', interval), color='C0')
            plt.ylabel(f'Z-Score')
            plt.title(f'Z-Score: Bank {b} - {p} Lending Portfolio')
            plt.xlabel('')

            # set xticks to be every ny years
            quarter_index = pd.period_range(start=historical_periods[0], freq='Q', periods=full_scen_hist.shape[1])
            ny = 2
            plt.xticks(np.arange(0, len(quarter_index), ny*4), quarter_index[np.arange(0, len(quarter_index), ny*4)], rotation=45)

            # add a dotted dark red line for the year 2022
            plt.axvline(x=max_hist-1, color='maroon', linestyle='--')

            plt.tight_layout()

            # If folder does not exist, create it
            if not os.path.exists(f'plots/{first_hash}/zscores/zscore_densities'):
                os.makedirs(f'plots/{first_hash}/zscores/zscore_densities')
            if not os.path.exists(f'plots/{first_hash}/banks/b{b}_{p}'):
                os.makedirs(f'plots/{first_hash}/banks/b{b}_{p}')

            # Save the plot as a PNG file
            plt.savefig(f'plots/{first_hash}/zscores/zscore_densities/zscores_bank{b}_{p}.png', dpi=300)
            plt.savefig(f'plots/{first_hash}/banks/b{b}_{p}/4. zscores_bank{b}_{p}.png', dpi=300)
            plt.close()

