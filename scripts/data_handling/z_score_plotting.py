import pandas as pd
import matplotlib.pyplot as plt
from scripts.vectors_matrices.named_array import *


h5_file = '../../data/derived/zscores.h5'
historical_zscores = load_named_array(h5_file, 'historical_zscores')
historical_macro_data = pd.read_hdf('../../data/input/input_data.h5', key='historical_macro_data').reset_index()
print(historical_macro_data)


with h5py.File('../../data/derived/input_data.h5', 'r') as h5file:
    historical_tm_periods = h5file['historical_tm_periods'][()]
    historical_tm_periods = [byte.decode('utf-8') for byte in historical_tm_periods]  # decode strings
    bank_portfolios = h5file['bank_portfolios'][()]
    portfolio_types = h5file['portfolio_types'][()]
    portfolio_types = [byte.decode('utf-8') for byte in portfolio_types]  # decode strings

for ip, p in enumerate(portfolio_types):
    # overlay all the z-scores for a given portfolio type
    zscores = historical_zscores.data[:, ip, :]

    # plot them
    fig, ax = plt.subplots()
    ax.plot(zscores.T, color='lavender', alpha=0.5)
    # compute median and plot
    median = np.nanmedian(zscores, axis=0)
    ax.plot(median, color='midnightblue', linewidth=1, label='Median z-score')
    # compute p25 and p75 and plot
    p25 = np.nanpercentile(zscores, 25, axis=0)
    ax.plot(p25, color='midnightblue', linewidth=1, linestyle='dashed', label='Interquartile range')
    p75 = np.nanpercentile(zscores, 75, axis=0)
    ax.plot(p75, color='midnightblue', linewidth=1, linestyle='dashed')

    plt.legend()

    ax.set_title(f'Z-scores for Portfolio Type {p}')
    # set x-ticks to historical_tm_periods
    # keep only every 4th item in historical_tm_periods replace the rest with ''
    xtickslab = ['' if i % 4 != 0 else historical_tm_periods[i] for i in range(len(historical_tm_periods))]
    ax.set_xticks(range(len(historical_tm_periods)))
    ax.set_xticklabels(xtickslab, rotation=45)
    ax.set_ylabel('Z-score')
    plt.tight_layout()
    plt.savefig(f"../../plots/zscores/historical/zscore_{p}.png")
    plt.close()

# print indices of z-scores that are lower than -7
print('Indices of z-scores that are lower than -7:')
print(np.argwhere(historical_zscores.data < -7))
print(historical_zscores.names)