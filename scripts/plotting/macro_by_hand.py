import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from controls import *
from scripts.vectors_matrices.named_array import *


def plot_macro_overlay(first_hash, second_hash, label_1, label_2):

    first_folder = f'../../data/output/{first_hash}/'
    second_folder = f'../../data/output/{second_hash}/'

    historical_macro_enhanced = load_named_array(f'{first_folder}/macro_scenarios.h5', 'macro_data_enhanced')
    macro_scenarios = load_named_array(f'{first_folder}/macro_scenarios.h5', 'macro_scenarios')
    macro_scenarios_2 = load_named_array(f'{second_folder}/macro_scenarios.h5', 'macro_scenarios')
    with h5py.File(f'{first_folder}/macro_names.h5', 'r') as h5file:
        macro_names = h5file['macro_names'][()]
        macro_names = [byte.decode('utf-8') for byte in macro_names]
    with h5py.File(f'{first_folder}/input_data.h5', 'r') as h5file:
        historical_periods = h5file['historical_tm_periods'][()]
        historical_periods = [byte.decode('utf-8') for byte in historical_periods]  # decode strings


    # historical data for scenarios, repeating the historcal data N times
    tmp_hist_macro = np.repeat(historical_macro_enhanced.data[np.newaxis, ...], macro_scenarios.data.shape[0], axis=0)

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
        slice_2d_2 = macro_scenarios_2.data[:, :, i]

        # Preparing DataFrame for seaborn
        df = pd.DataFrame(slice_2d.T).reset_index().rename(columns={'index': 'quarter'})
        df_2 = pd.DataFrame(slice_2d_2.T).reset_index().rename(columns={'index': 'quarter'})
        # reshape long
        df = pd.melt(df, id_vars=['quarter']).drop(columns=['variable'])
        df_2 = pd.melt(df_2, id_vars=['quarter']).drop(columns=['variable'])
        # shift quarters by the length of historical data
        df_2['quarter'] = df_2['quarter'] + len(historical_periods)

        # Plot the generated scenarios before recentering
        for interval in plot_intervals:
            sns.lineplot(df, x='quarter', y='value', estimator='median', errorbar=('pi', interval), color='C0',
                         label=f'{label_1 if interval == plot_intervals[0] else ""}')
            sns.lineplot(df_2, x='quarter', y='value', estimator='median', errorbar=('pi', interval), color='C1',
                         label=f'{label_2 if interval == plot_intervals[0] else ""}')
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
        plt.savefig(f'../../plots/{first_hash}/macro_overlay/{second_hash}/{var}.png', dpi=300)
        plt.close()


# comparing recentered with baseline
plot_macro_overlay('7CQsoo1e', 'uCJG4UnY', label_1='Historical and VAR', label_2='Recentered')
# compare cycle-neutral with baseline
# plot_macro_overlay('7CQsoo1e', 'sRZWj7e8', label_1='Historical and VAR', label_2='Cycle-neutral')


