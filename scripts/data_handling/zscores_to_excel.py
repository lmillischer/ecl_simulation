import pandas as pd

from scripts.vectors_matrices.named_array import *

h5_file = '../../data/derived/zscores.h5'
# h5_file = '../../data/derived/zscores - 12 May mean statt sum.h5'
# h5_file = '../../data/derived/zscores - 13 May mean statt sum.h5'
historical_zscores = load_named_array(h5_file, 'historical_zscores')
rhos = load_named_array(h5_file, 'rhos')
with h5py.File('../../data/derived/input_data.h5', 'r') as h5file:
    historical_tm_periods = h5file['historical_tm_periods'][()]
    bank_portfolios = h5file['bank_portfolios'][()]

# Create PeriodIndex from the string array
historical_tm_periods = historical_tm_periods.astype(str)
period_index = pd.PeriodIndex(historical_tm_periods, freq='Q')

for bp in bank_portfolios:
    # check if historical_zscores is all NaNs
    if np.all(np.isnan(historical_zscores.data[bp[0], bp[1], :])):
        print(f'Bank {bp[0]}, Portfolio {bp[1]} has all NaNs')

# count bank_portfolios
n_bptf = len(bank_portfolios)
# how many historical z-score series are non-all-nan?
n_nonnan_zscore = 0
for b in range(historical_zscores.data.shape[0]):
    for p in range(historical_zscores.data.shape[1]):
        if not np.all(np.isnan(historical_zscores.data[b, p, :])):
            n_nonnan_zscore += 1

print(f'Number of bank-portfolio-timeframe with non-all-nan z-scores: {n_nonnan_zscore} and we have {n_bptf} bank-portfolios')


# Create a Pandas Excel writer using XlsxWriter as the engine
with pd.ExcelWriter('../../data/derived/historical_zscores_for_bma_quarterly.xlsx', engine='xlsxwriter') as writer:
    for i in range(historical_zscores.data.shape[1]):
        # Slice the array to get a (43, 13) matrix
        data_slice = historical_zscores.data[:, i, :]

        # Convert the sliced array to a DataFrame
        df = pd.DataFrame(data_slice.T)
        df.index = historical_tm_periods

        # Write the DataFrame to a specific sheet
        df.to_excel(writer, sheet_name=f'Porfolio {i}')

# output rhos, reset index call it bank
rhodf = pd.DataFrame(rhos.data).reset_index()
rhodf.columns = ['bank', 'Commercial', 'Consumer', 'Microcredit', 'Mortgage']
rhodf.to_excel('../../data/derived/optimal_rhos.xlsx', index=False)