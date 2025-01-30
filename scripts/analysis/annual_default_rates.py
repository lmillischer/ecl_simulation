import numpy as np
import pandas as pd

from scripts.vectors_matrices.named_array import *

run_hash = 'XW8M2m2x'

historical_portfolios = load_named_array(f'../../data/output/{run_hash}/input_named_data.h5', 'historical_portfolios')
historical_tms = load_named_array(f'../../data/output/{run_hash}/input_named_data.h5', 'historical_tms')

print('Historical portfolios')
print(historical_portfolios.names)
print(historical_portfolios.data.shape)

print('Historical TMs')
print(historical_tms.names)
print(historical_tms.data.shape)

# keep only default rates from 1>3 and 2>3
historical_drs = historical_tms.data[:, :, :, 0:2, 2]
# periods go from Q2 2008 to Q3 2023 (62 periods)
# to fit with the historical exposure data, we drop the first four periods
historical_drs = historical_drs[:, :, 4:, :]
print('\nDRs')
print(historical_drs.data.shape)

# get the exposure in 1 and 2 historically
historical_s12_exposure = historical_portfolios.data[:, :, 0:2, :]
# switch last two dimensions
historical_s12_exposure = np.moveaxis(historical_s12_exposure, 2, 3)
# snapshots go from beggining Q2 2009 to beginning of Q1 2024 (60 snapshots)
# so we drop the last two snapshots
historical_s12_exposure = historical_s12_exposure[:, :, :-2, :]
print('Exposure')
print(historical_s12_exposure.shape)

# get the flow to default from 1>3 and 2>3
flow_to_default = historical_s12_exposure * historical_drs
print('Flow to default all')
print(flow_to_default.shape)

# sum flows over banks and stages
flow_to_default = np.nansum(flow_to_default, axis=(0, 3))
print('Flow to default collapsed')
print(flow_to_default.shape)

# sum by chunks of 4 quarters (only 3 quarters in year 2009 and 2023)
group_sizes = [3] + [4] * 13 + [3]  # 3 + 13*4 + 3 = 58
indices = np.cumsum([0] + group_sizes)
flow_to_default_yearly = np.zeros((4, len(group_sizes)))
for i in range(len(group_sizes)):
    flow_to_default_yearly[:, i] = np.nansum(flow_to_default[:, indices[i]:indices[i + 1]], axis=1)
# scale the first and last year by 4/3 as there are only 3 quarters
flow_to_default_yearly[:, 0] *= 4 / 3
flow_to_default_yearly[:, -1] *= 4 / 3
# swap dimensions
flow_to_default_yearly = np.swapaxes(flow_to_default_yearly, 0, 1)
print('Flow to default by year')
print(flow_to_default_yearly.shape)
print(flow_to_default_yearly)

# sum historical exposure over banks and stages
historical_s12_exposure = np.nansum(historical_s12_exposure, axis=(0, 3))
# print('Exposure collapsed')
# print(historical_s12_exposure.shape)
yearly_hist_exposure = historical_s12_exposure
exposure_yearly = np.zeros((4, len(group_sizes)))
for i in range(len(group_sizes)):
    exposure_yearly[:, i] = np.nanmean(historical_s12_exposure[:, indices[i]:indices[i + 1]], axis=1)
# scale the first and last year by 4/3 as there are only 3 quarters
# swap dimensions
exposure_yearly = np.swapaxes(exposure_yearly, 0, 1)
print('Exposure by year')
print(exposure_yearly.shape)
print(exposure_yearly)

# default rates by year
drs_yearly = flow_to_default_yearly/exposure_yearly
print('DRs by year')
print(drs_yearly.shape)

# export drs_yearly to Excel
df = pd.DataFrame(drs_yearly)
df.columns = ['Commercial', 'Consumer', 'Microcredit', 'Mortgage']
df.index = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
df.to_excel('../../data/analysis/default_rates_yearly.xlsx', index=True)



