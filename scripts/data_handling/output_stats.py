
from main import *
import pandas as pd
import numpy as np

from scripts.vectors_matrices.named_array import *

print('historical_tms', historical_tms.names)

print('historical_portfolios', historical_portfolios.names)

# get T12 and T23
default_rates = NamedArray(names=['bank', 'portfolio_type', 'historical_tm_quarter', 'stage2'],
                           data=historical_tms.data[:, :, :, :2, 2].copy())
validate_named_array(default_rates)
print('default_rates', default_rates.names)

# historical stage 1 and 2 exposure
hist_exp = NamedArray(names=['bank', 'portfolio_type', 'historical_tm_quarter', 'stage2'],
                      data=historical_portfolios.data[:, :, :2, :].copy())
hist_exp.data = np.swapaxes(hist_exp.data, 2, 3)
validate_named_array(hist_exp)
print('hist_exp', hist_exp.names)

# multiply default rates by exposure, then sum over stages and banks
dr_times_exp = np.nansum((default_rates.data * hist_exp.data), axis=(3, 0))
print('dr_times_exp', dr_times_exp.shape)

# divide by total exposure

total_exp = np.nansum(hist_exp.data, axis=(3, 0))
print('total_exp', total_exp.shape)
dr_weighted = dr_times_exp / total_exp
print('dr_weighted', dr_weighted.shape)

# export to excel as a table
default_rates = pd.DataFrame(dr_weighted.T, index=range(2009, 2022), columns=portfolio_types)
default_rates.to_excel('data/derived/defaults_through_time.xlsx')

print(default_rates)
