import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from controls import *


def plot_macro(historical_macro_data, macro_scenarios, macro_names, macro_names_print, historical_periods, prefix=''):

    #  historical data for scenarios, repeating the historcal data N times
    tmp_hist_macro = np.repeat(historical_macro_data.data[np.newaxis, ...], macro_scenarios.data.shape[0], axis=0)

    # concatenate historical data on top of scenarios along axis 1
    full_macro_scenarios = np.concatenate([tmp_hist_macro, macro_scenarios.data], axis=1)

    print_macro_names = ['Quarterly Real GDP Growth (QoQ, %)', 'Quarterly Real GDP Growth (YoY, %)',
                         'Consumer Price Inflation (QoQ, %)', 'Consumer Price Inflation (YoY, %)',
                         'Unemployment Rate (%)', 'Change in Unemployment Rate (QoQ, %)',
                         'Three-Year Sovereign Bond Yield (%)',
                         'Term Spread (p.p.)',
                         'House Price Growth (QoQ, %)', 'House Price Growth (YoY, %)',
                         'USD-COP Exchange Rate Growth (QoQ, %)', 'USD-COP Exchange Rate Growth (YoY, %)',
                         'Nominal Wage Growth (QoQ, %)', 'Nominal Wage Growth (YoQ, %)',
                         'Oil Price Growth (QoQ, %)', 'Oil Price Growth (YoY, %)',
                         'Fed Funds Rate (%)']

    # check length of print_macro_names
    if len(print_macro_names) != len(macro_names):
        print(macro_names)
        raise ValueError(f'Length of print_macro_names ({len(print_macro_names)}) does not match length of macro_names ({len(macro_names)})')

    # Plot the generated scenarios before recentering
    for i, var in enumerate(macro_names):
        print(f'Plotting {var} ({i+1}/{len(macro_names)})')
        if var == 'DURX':
            continue

        slice_2d = full_macro_scenarios[:, :, i]

        # Preparing DataFrame for seaborn
        df = pd.DataFrame(slice_2d.T).reset_index().rename(columns={'index': 'quarter'})
        # df['year'] = df['year'] + 2009
        # reshape long
        df = pd.melt(df, id_vars=['quarter']).drop(columns=['variable'])
        # print(df)

        # Plot the generated scenarios before recentering
        for interval in plot_intervals:
            sns.lineplot(df, x='quarter', y='value', estimator='median', errorbar=('pi', interval), color='C0')
        plt.ylabel(f'{print_macro_names[i]}')
        plt.title(f'{print_macro_names[i]}')
        plt.xlabel('')

        # set xticks to be every ny years
        quarter_index = pd.period_range(start=historical_periods[0], freq='Q', periods=full_macro_scenarios.shape[1])
        ny = 2
        plt.xticks(np.arange(0, len(quarter_index), ny * 4), quarter_index[np.arange(0, len(quarter_index), ny * 4)],
                   rotation=45)

        # add a dotted dark red line for the year 2022
        max_hist = len(historical_periods)
        plt.axvline(x=max_hist, color='maroon', linestyle='--')

        plt.tight_layout()

        # If folder does not exist, create it
        if not os.path.exists(f'plots/{first_hash}/macro'):
            os.makedirs(f'plots/{first_hash}/macro')

        # save
        plt.savefig(f'plots/{first_hash}/macro/{var}{prefix}.png', dpi=300)
        plt.close()

