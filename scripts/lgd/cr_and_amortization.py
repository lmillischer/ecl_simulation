import numpy as np
import matplotlib.pyplot as plt

# Function to generate initial loans
def generate_initial_loans(n, mature=True):
    exposures = np.random.lognormal(mean=0, sigma=1, size=n)
    # over_coll = np.random.lognormal(mean=0, sigma=1, size=n)
    # collateral = exposures + over_coll
    coll_ratio = np.random.lognormal(mean=0.9, sigma=0.1, size=n)
    collateral = exposures * coll_ratio
    times = np.random.uniform(0 if mature else 9.99, 10, size=n)
    return np.array(list(zip(exposures, collateral, times)))

# Function to update loans for a year
def update_loans(loans):
    # Update exposures
    loans[:, 0] = loans[:, 0] * (loans[:, 2] / (loans[:, 2] + 1))
    # Decrease time to amortization by 1
    loans[:, 2] -= 1
    # Remove loans that have reached maturity
    loans = loans[loans[:, 2] > 0]
    return loans

# Function to plot distributions
def plot_distributions(loans, year, mature):

    print(year)
    exposures = loans[:, 0]
    collaterals = loans[:, 1]
    collateralization_ratios = collaterals / exposures

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.hist(exposures, density=False, bins=500, alpha=0.7, range=(0, 15))
    plt.title(f'Exposure Distribution in Year {year}')
    plt.xlabel('Exposure')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    # plot histogram as denstiy

    plt.hist(collateralization_ratios, density=True, bins=500, alpha=0.7, range=(0, 35))
    # fit a lognormal distribution to the data and plot it
    mu, sigma = np.log(collateralization_ratios).mean(), np.log(collateralization_ratios).std()
    x = np.linspace(0, 35, 1000)
    plt.plot(x, np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2)) / (x * sigma * np.sqrt(2 * np.pi)),
             'k--', lw=1, label='Lognormal fit')
    # add text with mu and sigma
    plt.text(0.95, 0.95, f'$\mu$ = {mu:.2f}\n$\sigma$ = {sigma:.2f}', transform=plt.gca().transAxes,
             horizontalalignment='right', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.title(f'Collateralization Ratio Distribution in Year {year}')
    plt.xlabel('Collateralization Ratio')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'../../plots/cr_and_amort/year_{year}{"_mature" if mature else ""}.png')
    plt.close()


# Initialize loans
mature = True
loans = generate_initial_loans(1000000, mature)

# Plot initial distributions and update loans for 10 years
plot_distributions(loans, 0, mature)
for year in range(1, 10):
    loans = update_loans(loans)
    plot_distributions(loans, year, mature)
