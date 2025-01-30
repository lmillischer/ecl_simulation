import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, lognorm, nbinom
import seaborn as sns
import h5py

from scripts.vectors_matrices.named_array import *
from controls import *


def find_binomial_parameters(target_mean, target_std):
    # Calculate p using the mean and standard deviation formulas
    p = 1 - (target_std ** 2 / target_mean)
    if not (0 < p < 1):
        raise ValueError("The provided target values are not feasible for a binomial distribution.")

    # Calculate n using the mean and p
    n = target_mean / p

    # Ensure n is an integer since it's the number of trials in a binomial distribution
    n = round(n)

    return n, p


def find_negative_binomial_parameters(target_mean, target_std):
    target_variance = target_std ** 2
    # Calculate p using the mean and variance formulas
    p = target_mean / target_variance
    if not (0 < p < 1):
        raise ValueError("The provided target values are not feasible for a negative binomial distribution.")

    # Calculate r using the mean and p
    r = target_mean * p / (1 - p)

    return r, p


def value_to_sale(n, p, num_draws, macro_scenarios, macro_names, do_v2s, interest_rates,
                  banks, portfolio_types, bank_portfolios, verbose=False):
    """
    Simulates draws from a negative binomial distribution and plots the initial and final distributions.
    Adjusts the final distribution to ensure all values are positive before fitting a lognormal distribution.

    Args:
    n (int): Number of trials in the binomial distribution.
    p (float): Probability of success in each trial in the binomial distribution.
    num_draws (int): Number of draws to simulate.
    """

    h5file_v2s = f'data/output/{first_hash}/value_to_sale.h5'

    # If parameters to be computed
    if do_v2s:

        # Take the houseprices only
        rpp_idx = macro_names.index('HPQ')
        hp_scenarios = macro_scenarios.data[:, :, rpp_idx] / 100 + 1
        # convert ratios to levels
        hp_scenarios = np.cumprod(hp_scenarios, axis=1)

        # Draw from the negative binomial distribution
        # sampled_quartes_to_sale = nbinom.rvs(r, p, size=num_draws)
        sampled_quartes_to_sale = binom.rvs(n, p, size=num_draws)
        if verbose:
            print('q2s mean', np.mean(sampled_quartes_to_sale))
            print('q2s std', np.std(sampled_quartes_to_sale))


        # Define named arrays
        v2s_mu = NamedArray(names=['bank', 'portfolio_type'],
                            data=np.full(shape=(n_banks, n_portfolio_types), fill_value=np.nan))
        v2s_sigma2 = NamedArray(names=['bank', 'portfolio_type'],
                                data=np.full(shape=(n_banks, n_portfolio_types), fill_value=np.nan))

        # Loop over banks
        for ib, b in enumerate(banks):

            # Loop over portfolio types
            for ip, p in enumerate(portfolio_types):

                if (ib, ip) not in bank_portfolios:
                    continue

                print(f' v2s: bank {b} ptf {p}')

                irate = interest_rates.data[ib, ip]
                if np.isnan(irate):
                    if verbose:
                        print('     - interest rate missing')
                    irate = np.nanmean(interest_rates.data[:, ip])
                elif verbose:
                    print(f'    - interest rate: {irate}')
                discount_rate = 1 + irate

                # Draw from houseprice scenarios
                final_sample = np.array([])

                for q2sale in sampled_quartes_to_sale:
                    dr = discount_rate ** (q2sale + 0.5)
                    sampled_element = np.random.choice(hp_scenarios[:, q2sale]/dr, size=10)
                    final_sample = np.append(final_sample, sampled_element)
                if verbose:
                    print(f'    - mean v2s = {np.mean(final_sample):.2f}')

                # Fit a lognormal distribution to the adjusted combined histogram

                # select subsample where value is below 2
                final_sample_low = final_sample[final_sample < 1.5]
                sigma, loc, mu2 = lognorm.fit(final_sample_low, floc=0)
                mu = np.log(mu2)
                v2s_mu.data[ib, ip] = mu
                v2s_sigma2.data[ib, ip] = sigma ** 2

                # Plotting
                plt.figure(figsize=(12, 6))

                # Plot the original binomial distribution
                plt.subplot(1, 2, 1)
                sns.histplot(sampled_quartes_to_sale, kde=False, color='blue')
                plt.title('Original Binomial Distribution')
                plt.xlabel('Years to sale')
                plt.ylabel('Frequency')

                # Plot the final distribution with the fitted lognormal curve
                plt.subplot(1, 2, 2)
                plt.hist(final_sample, bins=200, density=True, color='green')
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                lognorm_shape = lognorm.pdf(x, sigma, loc, mu2)
                plt.plot(x, lognorm_shape, 'k', linewidth=2)
                plt.title('Adjusted Final Distribution with Lognormal Fit')
                plt.xlabel('House price ratio')
                plt.ylabel('Density')
                # add text box with fitted mu and sigma
                textstr = '\n'.join((
                    r'$\mu=%.2f$' % (mu, ),
                    r'$\sigma=%.2f$' % (sigma, )))
                plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                            verticalalignment='top')
                plt.tight_layout()

                # save plot
                plt.savefig(f'plots/lgd/value_to_sale/b{b}_{p}_v2s.png')
                plt.close()

        # Plot the house price distirbution in quarters
        for qt in range(20):
            plt.figure(figsize=(12, 6))
            sns.histplot(hp_scenarios[:, qt], kde=False, color='green')
            plt.title(f'House price distribution in quarter {qt}')
            plt.xlabel('House price')
            plt.ylabel('Frequency')
            plt.xlim(0, 2)
            plt.savefig(f'plots/lgd/house_price_future/house_price_quarter{qt}.png')
            plt.tight_layout()
            plt.close()

        # Save mu and sigma to h5 file
        save_named_array(h5file_v2s, v2s_mu, 'v2s_mu')
        save_named_array(h5file_v2s, v2s_sigma2, 'v2s_sigma')

    # if parameters to be loaded
    else:
        v2s_mu = load_named_array(h5file_v2s, 'v2s_mu')
        v2s_sigma2 = load_named_array(h5file_v2s, 'v2s_sigma')

    return v2s_mu, v2s_sigma2
