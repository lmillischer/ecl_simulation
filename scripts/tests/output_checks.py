from scripts.vectors_matrices.named_array import *

def check_historical_zscores():

    # read z-scores from h5 file
    zs_file = '../../data/derived/zscores.h5'
    historical_zscores = load_named_array(zs_file, 'historical_zscores')
    tm_file = '../../data/derived/input_named_data.h5'
    historical_tms = load_named_array(tm_file, 'historical_tms')
    with h5py.File('../../data/derived/input_data.h5', 'r') as h5file:
        bank_portfolios = h5file['bank_portfolios'][:]
        portfolios = h5file['portfolio_types'][:]
        banks = h5file['banks'][:]
        print(portfolios)

    # STEP 1: check why 3 portfolios don't have z-score data
    print('\nSTEP 1 ')
    print(f'There are {len(bank_portfolios)} bank portfolios')
    for ib, ip in bank_portfolios:
        if np.all(np.isnan(historical_zscores.data[ib, ip, :])):
            print(f'Bank #{ib} ({banks[ib]}), portfolio #{ip} has no z-score data')
            if np.all(np.isnan(historical_tms.data[ib, ip, :, :, :])):
                print('   > No transition matrix data either')
            # else:
            #     print(historical_tms.data[ib, ip, :, :, :])

    # STEP 2: do all holes in the z-scores correspond to missing historical TMs?
    print('\nSTEP 2 ')
    # loop over banks, portfolios and historical quarters
    for ib, ip in bank_portfolios:
        for iy in range(historical_zscores.data.shape[2]):
            if np.all(np.isnan(historical_zscores.data[ib, ip, iy])):
                if np.all(np.isnan(historical_tms.data[ib, ip, iy])):
                    pass
                    # print(f'Bank #{ib} ({banks[ib]}), portfolio #{ip} has no z-score data for quarter {iy}')
                else:
                    print(f'Bank #{ib} ({banks[ib]}), portfolio #{ip} has no z-score data for quarter {iy}, but has transition matrix data')
                    print(historical_tms.data[ib, ip, iy])

    # STEP 3: count how often a z-score value is repeated
    print('\nSTEP 3 ')
    zscore_counts = {}
    for ib, ip in bank_portfolios:
        ztmp = historical_zscores.data[ib, ip, :]
        # how often is the most frequent value in ztmp repeated (ztmp are floats)?
        ztmp = ztmp[~np.isnan(ztmp)]
        ztmp = np.round(ztmp, 4)
        ztmp = ztmp.astype(str)
        ztmp = ztmp[ztmp != 'nan']
        ztmp = ztmp.tolist()
