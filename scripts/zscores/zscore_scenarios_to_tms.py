import numpy as np

from scripts.zscores.zscore_to_tm import zscore_to_tm
from scripts.vectors_matrices.named_array import save_named_array, load_named_array
from controls import *

def zscore_scenarios_to_3x3tms(zscore_scenarios, rhos, upper_bounds, lower_bounds, do_tm_scenarios=False):

    out_path = f'data/output/{first_hash}/'
    # If scenarios to be recomputed
    if do_tm_scenarios:

        # if upper bounds are missing for a bank/portfolio, replace them with average historical TMs of all banks of that portfolio type
        for p in range(n_portfolio_types):
            for b in range(n_banks):
                if np.all(np.isnan(upper_bounds.data[b, p, :, :])):
                    upper_bounds.data[b, p, :, :] = np.nanmean(upper_bounds.data[:, p, :, :], axis=0)
                if np.all(np.isnan(lower_bounds.data[b, p, :, :])):
                    lower_bounds.data[b, p, :, :] = np.nanmean(lower_bounds.data[:, p, :, :], axis=0)
                if np.all(np.isnan(rhos.data[b, p])):
                    rhos.data[b, p] = np.nanmean(rhos.data[:, p], axis=0)

        tm_scenarios_3x3 = zscore_to_tm(zscore_scenarios, upper_bounds, lower_bounds, rhos)

        # save in h5 file
        save_named_array(f'{out_path}/tm_scenarios_3x3.h5', tm_scenarios_3x3, 'tm_scenarios_3x3')

    # If scenarios to be read from file
    else:
        tm_scenarios_3x3 = load_named_array(f'{out_path}/tm_scenarios_3x3.h5', 'tm_scenarios_3x3')

    return tm_scenarios_3x3
