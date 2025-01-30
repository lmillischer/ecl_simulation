import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def draw_lognorm(mu, sigma, name, min_plot=0., max_plot=10.):
    """
    Plot a histrogram drawn from a lognormal distribution and the density overlaid for
    pure illustrative purposes only.
    """

    # Drawing 1000 samples from the log-normal distribution
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=1000)

    # Generate values for x
    x = np.linspace(min_plot, max_plot, 1000)  # Adjust the range as needed
    # Calculate lognormal density
    lognormal = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    # Plotting the histogram and the lognormal density distribution
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=30, density=True, alpha=0.6, label='Histogram of samples')
    plt.plot(x, lognormal, 'r', linewidth=2, label=f'Lognormal distribution (μ={mu}, σ={sigma})')
    # plt.title('Lognormal Distribution with Histogram Overlay')
    plt.xlabel(f'{name}')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    # Save the updated plot as a PNG file
    updated_file_path = f'../../plots/lgd_distributions/lognormal_hist_{name}.png'
    plt.savefig(updated_file_path)


# draw_lognorm(0, 0.6, 'collateralization_ratio', min_plot=0, max_plot=6)
draw_lognorm(-.1, 0.1, 'actual_sales_ratio', min_plot=0.5, max_plot=1.5)
