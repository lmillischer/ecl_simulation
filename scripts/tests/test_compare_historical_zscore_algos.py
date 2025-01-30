import os

from scripts.vectors_matrices.named_array import *
import matplotlib.pyplot as plt

print(os.getcwd())

# read in the data from the h5 files
historical_tms = load_named_array('../data/derived/input_named_data.h5', 'historical_tms')
print('historical_tms', historical_tms.names)


zscore_old = load_named_array('../data/derived/test/zscores_tts.h5', 'historical_zscores')
zscore_new = load_named_array('../data/derived/test/zscores_tts2.h5', 'historical_zscores')

rho_old = load_named_array('../data/derived/test/zscores_tts.h5', 'rhos')
rho_new = load_named_array('../data/derived/test/zscores_tts2.h5', 'rhos')

# compare the two
# print(zscore_old.data.shape)
# print(zscore_new.data.shape)
# print(rho_old.data.shape)
# print(rho_new.data.shape)

# calculate mean and std of absolute difference, ignoring nans
abs_diff_z = np.abs(zscore_old.data - zscore_new.data)
abs_diff_rho = np.abs(rho_old.data - rho_new.data)

counter_z = 0
counter_rho = 0
for i in range(43):
    for j in range(4):
        mean_z = np.nanmean(abs_diff_z[i, j, :])
        if mean_z > 0.1:
            counter_z += 1

            print(i, j, '  mean_z:  ', mean_z)
            # print(np.nanstd(abs_diff_z[i, j, :]))

print(f'Number of banks with mean z-score difference > 0.03: {counter_z}')

sel_i = 38
sel_j = 3

# print(abs_diff_z[sel_i, sel_j, :])
print('\nnew')
print(zscore_new.data[sel_i, sel_j, :])
print('var', np.nanstd(zscore_new.data[sel_i, sel_j, :]))
print('rho', rho_new.data[sel_i, sel_j])

print('\nold')
print(zscore_old.data[sel_i, sel_j, :])
print('var', np.nanstd(zscore_old.data[sel_i, sel_j, :]))
print('rho', rho_old.data[sel_i, sel_j])

default_rates = historical_tms.data[sel_i, sel_j, :, 0, 2]
default_rates2 = historical_tms.data[sel_i, sel_j, :, 1, 2]
default_rates3 = historical_tms.data[sel_i, sel_j, :, 2, 2]

# Plot zscore new and zscore old vs range(13)
if True:
    plt.plot(range(2009, 2022), zscore_new.data[sel_i, sel_j, :], label='new')
    plt.plot(range(2009, 2022), zscore_old.data[sel_i, sel_j, :], label='old')
    # plot default rates on right y axis
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(range(2009, 2022), -default_rates, color='green', label='default rates (S1)', lw=3)
    # ax2.plot(range(2009, 2022), -default_rates2, color='green', label='default rates (S2)', lw=3)
    # ax2.plot(range(2009, 2022), -default_rates3, color='green', label='default rates (S3)', lw=3)
    ax.legend()
    plt.show()
