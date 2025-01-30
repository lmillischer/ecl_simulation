import numpy as np
import matplotlib.pyplot as plt


def plot_bank_composition_portfolios(portfolios, banks, portfolio_names, normalize=False):
    """
    Take as input the array of shape (n_banks, n_years, n_porfolios, n_stages, 1) sum over the portfolios and years plot the stacked bar chart over time for each stage
    """

    # Replace nans with zeros in portfolio
    portfolios = np.nan_to_num(portfolios)

    # keep only most recent year
    portfolios = portfolios[:, -1, :, :]

    # Sum over stages
    portfolios = portfolios.sum(axis=2)

    # Define the colors
    colors = ['blue', 'orange', 'green', 'red']

    # Define the labels
    labels = portfolio_names

    # Define the x-axis
    x = np.arange(portfolios.shape[0]) + 1

    # If normalize=True, normalize the array such that each row equals to 1
    if normalize:
        portfolios = portfolios / portfolios.sum(axis=1)[:, None]

    # Plot the stacked bar chart
    plt.bar(x, portfolios[:, 0], color=colors[0], label=labels[0])
    plt.bar(x, portfolios[:, 1], bottom=portfolios[:, 0], color=colors[1], label=labels[1])
    plt.bar(x, portfolios[:, 2], bottom=portfolios[:, 0] + portfolios[:, 1], color=colors[2], label=labels[2])
    plt.bar(x, portfolios[:, 3], bottom=portfolios[:, 0] + portfolios[:, 1] + portfolios[:, 2], color=colors[3], label=labels[3])

    # Add some labels
    plt.xticks(x, labels=banks)
    plt.xticks(rotation=45)
    plt.xlabel('Bank')
    plt.gca().tick_params(axis='x', labelsize=8)
    plt.ylabel('Portfolio [COR]')
    plt.title('Composition of portfolios by stage')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/descriptive/bank_composition_portfolios{"_normalize" if normalize else ""}.png', dpi=300)
    plt.close()


def plot_bank_composition_stages(portfolios, banks, normalize=False):
    """
    Take as input the array of shape (n_banks, n_years, n_porfolios, n_stages, 1) sum over the portfolios and years plot the stacked bar chart over time for each stage
    """

    # Replace nans with zeros in portfolio
    portfolios = np.nan_to_num(portfolios)

    # keep only most recent year
    portfolios = portfolios[:, -1, :, :]

    # Sum over portfolios
    portfolios = portfolios.sum(axis=1)

    # Define the colors
    colors = ['blue', 'orange', 'green']

    # Define the labels
    labels = ['Stage 1', 'Stage 2', 'Stage 3']

    # Define the x-axis
    x = np.arange(portfolios.shape[0]) + 1

    # If normalize=True, normalize the array such that each row equals to 1
    if normalize:
        portfolios = portfolios / portfolios.sum(axis=1)[:, None]

    # Plot the stacked bar chart
    plt.bar(x, portfolios[:, 0], color=colors[0], label=labels[0])
    plt.bar(x, portfolios[:, 1], bottom=portfolios[:, 0], color=colors[1], label=labels[1])
    plt.bar(x, portfolios[:, 2], bottom=portfolios[:, 0] + portfolios[:, 1], color=colors[2], label=labels[2])

    # Add some labels
    plt.xticks(x, labels=banks)
    plt.xticks(rotation=45)
    plt.xlabel('Bank')
    plt.gca().tick_params(axis='x', labelsize=8)
    plt.ylabel('Portfolio [COR]')
    plt.title('Composition of portfolios by stage')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/descriptive/bank_composition_stages{"_normalize" if normalize else ""}.png', dpi=300)
    plt.close()


def plot_historical_composition_stages(portfolios, historical_years, normalize=False):
    """
    Take as input the array of shape (n_banks, n_years, n_porfolios, n_stages, 1) sum over the portfolios and banks plot the stacked bar chart over time for each stage
    :param portfolios:
    :return:
    """

    # Replace nans wiht zeros in portfolio
    portfolios = np.nan_to_num(portfolios)

    # Sum over banks and portfolios
    portfolios = portfolios.sum(axis=0).sum(axis=1)

    # Define the colors
    colors = ['blue', 'orange', 'green']

    # Define the labels
    labels = ['Stage 1', 'Stage 2', 'Stage 3']

    # Define the x-axis
    x = np.arange(len(portfolios))

    # If normalize=True, normalize the array such that each row equals to 1
    if normalize:
        portfolios = portfolios / portfolios.sum(axis=1)[:, None]

    # Plot the stacked bar chart
    plt.bar(x, portfolios[:, 0], color=colors[0], label=labels[0])
    plt.bar(x, portfolios[:, 1], bottom=portfolios[:, 0], color=colors[1], label=labels[1])
    plt.bar(x, portfolios[:, 2], bottom=portfolios[:, 0] + portfolios[:, 1], color=colors[2], label=labels[2])

    # Add some labels

    historical_years = np.append(historical_years, 2022)
    plt.xticks(x, labels=historical_years+[2022])
    plt.xticks(rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Portfolio [COR]')
    plt.title('Historical composition of portfolios by stage')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/descriptive/historical_composition_stages{"_normalize" if normalize else ""}.png', dpi=300)
    plt.close()


def plot_historical_composition_portfolios(portfolios, all_porfolios, historical_years, normalize=False):
    """
    Take as input the array of shape (n_banks, n_years, n_porfolios, n_stages, 1) sum over the stages and banks plot the stacked bar chart over time for each stage
    :param portfolios:
    :return:
    """

    # Replace nans wiht zeros in portfolio
    portfolios = np.nan_to_num(portfolios)

    # Sum over banks and portfolios
    portfolios = portfolios.sum(axis=0).sum(axis=2)

    # Define the colors
    colors = ['blue', 'orange', 'green', 'red']
    labels = all_porfolios


    # Define the x-axis
    x = np.arange(len(portfolios)) + 1

    # If normalize=True, normalize the array such that each row equals to 1
    if normalize:
        portfolios = portfolios / portfolios.sum(axis=1)[:, None]

    # Plot the stacked bar chart
    plt.bar(x, portfolios[:, 0], color=colors[0], label=labels[0])
    plt.bar(x, portfolios[:, 1], bottom=portfolios[:, 0], color=colors[1], label=labels[1])
    plt.bar(x, portfolios[:, 2], bottom=portfolios[:, 0] + portfolios[:, 1], color=colors[2], label=labels[2])
    plt.bar(x, portfolios[:, 3], bottom=portfolios[:, 0] + portfolios[:, 1] + portfolios[:, 2], color=colors[3], label=labels[3])

    # Add some labels
    historical_years = np.append(historical_years, 2022)
    plt.xticks(x, labels=historical_years)
    plt.xticks(rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Portfolio [COR]')
    plt.title('Historical composition of portfolios by stage')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/descriptive/historical_composition_portfolios{"_normalize" if normalize else ""}.png', dpi=300)
    plt.close()
