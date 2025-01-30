import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm
import os

from controls import first_hash


def plot_ecl(banks, portfolio_types, ecl_scenarios, bank_portfolios, only_bank_portfolio,
             stage, stat_type='abs', verbose=False):

    # Check if stat_type is 'abs' or 'rel'
    if stat_type not in ['abs', 'rel']:
        raise ValueError(f'Unknown type: {stat_type}')

    # Delete all plots in folder
    folder = f'plots/{first_hash}/ecl/stage{stage}/{stat_type}/'
    # if the folder exists
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    # For every bank and portfolio, plot a histogram of stage 1 ECL
    for ib, b in enumerate(banks):
        for ip, p in enumerate(portfolio_types):

            if (ib, ip) not in bank_portfolios:
                continue

            if only_bank_portfolio != []:
                if not (ib, ip) in only_bank_portfolio:
                    continue

            if verbose:
                print(f'Plotting ECL bank #{ib} ({b}) - {p}')

            ecl = ecl_scenarios.data[:, ib, ip, stage-1].copy()
            ecl = ecl * 100 if stat_type == 'rel' else ecl

            # if ecl all nan, continue
            if np.all(np.isnan(ecl)):
                if verbose:
                    print('  all nan, not plotting')
                continue

            # in ECL, replace 0 with a very small number
            ecl[ecl == 0] = 1e-10

            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

            # Plot the histogram

            # set the max value of the historgram to a high percentile of ecl
            max_ecl = 5 * np.nanmean(ecl)
            high_quant = np.nanquantile(ecl, 0.95)
            if high_quant < max_ecl:
                max_ecl = high_quant
            if (stage == 3) & (stat_type == 'rel'):
                max_ecl = 110
            plt.hist(ecl, bins=100, density=True, label='CL scenarios', range=(0, max_ecl))

            # Add a horizontal red line at the ECL
            label_string = f'ECL = {ecl.mean():.1f}%' if stat_type == 'rel' \
                else f'ECL = {ecl.mean()/1e9:.0f} bn COP'
            plt.axvline(ecl.mean(), color='red', linestyle='dashed', linewidth=1, label=label_string)

            # fit the histogram with a lognormal and plot the fit
            # shape, loc, scale = lognorm.fit(ecl, floc=0)
            # xmin, xmax = min(ecl), max_ecl
            # x = np.linspace(xmin, xmax, 1000)
            # pdf = lognorm.pdf(x, shape, loc, scale)
            # plt.plot(x, pdf, 'k', linewidth=2)

            plt.gca().set_yticks([])
            plt.title(f'Stage {stage} ECL: Bank #{b} - {p} Lending Portfolio')
            plt.xlabel(f'Scenario loss {"[COP]" if stat_type == "abs" else "[%]"}')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()

            # if folder does not exist, create it
            if not os.path.exists(f'plots/{first_hash}/ecl/stage{stage}/{stat_type}/'):
                os.makedirs(f'plots/{first_hash}/ecl/stage{stage}/{stat_type}/')
            if not os.path.exists(f'plots/{first_hash}/banks/b{b}_{p}/'):
                os.makedirs(f'plots/{first_hash}/banks/b{b}_{p}/')

            # save the figure
            plt.savefig(f'plots/{first_hash}/ecl/stage{stage}/{stat_type}/ecl_s{stage}_bank{b}_{p}.png', dpi=300)
            plt.savefig(f'plots/{first_hash}/banks/b{b}_{p}/0. ecl_{stat_type}_s{stage}_bank{b}_{p}.png', dpi=300)
            plt.close()

    # count the number of plots in the plot folder
    n_plots = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
    print(f'[stage {stage}/{stat_type}] Produced {n_plots} plots')
