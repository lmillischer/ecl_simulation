import numpy as np
import gc
from scipy.stats import lognorm, norm

from scripts.lgd.frye_jacobs import project_basic_lgds
from scripts.vectors_matrices.named_array import *
from scripts.data_handling.memory_usage import print_memory_usage


def cr_to_lgd(cr_mu, cr_sigma2):
    if cr_mu.data.shape != cr_sigma2.data.shape:
        raise ValueError('cr_mu and cr_sigma2 must have the same dimensions')

    # Access np arrays
    sigma_array = np.sqrt(cr_sigma2.data)
    mu_array = cr_mu.data

    # Integral of f(x)
    cdf_at_1 = lognorm.cdf(1, sigma_array, scale=np.exp(mu_array))

    # Integral of x * f(x)
    # For lognormal, E[X] = exp(mu + sigma^2 / 2), so for truncated at 1, we adjust
    mean = np.exp(mu_array + sigma_array ** 2 / 2)
    upper_cdf = norm.cdf((np.log(1) - mu_array) / sigma_array - sigma_array)
    # truncation_cdf = norm.cdf((np.log(1) - mu_array) / sigma_array)
    truncated_expectation = mean * upper_cdf

    # Integral of (1 - x) * f(x)
    integral = cdf_at_1 - truncated_expectation

    lgd = NamedArray(names=cr_mu.names,
                     data=integral)
    validate_named_array(lgd)

    return lgd


def project_lgd(collateral_types, n_scenarios, stages, n_scenario_quarters, amortization_profiles,
                basic_lgds, cr_mu, cr_sigma2, v2s_mu, v2s_sigma2, sr_mu, sr_sigma2,
                macro_scenarios, macro_names, tm_scenarios_3x3, historical_tms, verbose=False):

    # project collateralization ratios into the future
    cr_lgd = project_cr(collateral_types, n_scenarios, n_scenario_quarters,
                        cr_mu, cr_sigma2, v2s_mu, v2s_sigma2, sr_mu, sr_sigma2,
                        macro_scenarios, macro_names, amortization_profiles, verbose=verbose)
    if verbose:
        print('   - Collateralization ratios projected')

    # project basic lgds into the future using frye-jacobs
    basic_lgd_scenarios = project_basic_lgds(basic_lgds, tm_scenarios_3x3, historical_tms)
    if verbose:
        print('   - Basic LGDs projected')

    # define the empty lgd array
    lgd_scenarios = NamedArray(names=['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage5', 'future_quarter'],
                               data=np.full(shape=(n_scenarios, n_banks, n_portfolio_types,
                                                   n_collateral_types, 5, n_scenario_quarters),
                                            fill_value=np.nan))

    # then fill the lgd_scenarios array in 3 steps

    # Step 1: set stage 3 bucket 3 LGD to 100%
    lgd_scenarios.data[:, :, :, :, stages.index('3-3'), :] = 1

    # Step 2: for collateral stat_type None, set LGD to Basic LGD (including Frye-Jacobs, except for stage 3 bucket 3)
    stages_not_3b3 = [i for i, s in enumerate(stages) if s != '3-3']

    lgd_scenarios.data[:, :, :, collateral_types.index('None'), stages_not_3b3, :] \
        = basic_lgd_scenarios.data[:, :, :, collateral_types.index('None'), :, :]

    # Step 3: for all other collateral types, set LGD to Basic x collateralization ratio
    # get collateral_type indices all but "None"
    coll_type_not_none = [i for i, c in enumerate(collateral_types) if c != 'None']

    lgd_scenarios.data[:, :, :, coll_type_not_none, :4, :] \
        = (basic_lgd_scenarios.data[:, :, :, coll_type_not_none, :, :]
           * cr_lgd.data[:, :, :, coll_type_not_none, :4, :])

    return lgd_scenarios


def project_cr(collateral_types, n_scenarios, n_scenario_quarters,
               cr_mu_named, cr_sigma2_named, v2s_mu, v2s_sigma2, sr_mu, sr_sigma2,
               macro_scenarios_named, macro_names, amortization_profiles, verbose=False):
    """Project the collateralization ratios into the future, taking into account the house price scenarios"""

    # extract the np.array from the NamedArrays
    cr_mu = cr_mu_named.data
    cr_sigma2 = cr_sigma2_named.data
    macro_scenarios = macro_scenarios_named.data

    # Starting with the collateralization ratios (cr_mu and cr_sigma2) with shape:
    #  n_banks, n_portfolio_types, n_collateral_types, n_stages
    #  we add fake dimensions for the scenarios and the quarters

    # add a dimension as the first dimension, repeating cr_mu n_scenario times
    cr_mu = np.repeat(cr_mu[np.newaxis, ...], n_scenarios, axis=0).astype(np.float32)
    cr_sigma2 = np.repeat(cr_sigma2[np.newaxis, ...], n_scenarios, axis=0).astype(np.float32)

    # add a dimension as the last dimension, repeating cr_mu n_sceario_quarters times
    cr_mu = np.repeat(cr_mu[..., np.newaxis], n_scenario_quarters, axis=-1)
    cr_sigma2 = np.repeat(cr_sigma2[..., np.newaxis], n_scenario_quarters, axis=-1)

    # Keep only the real estate price of the macro scenario (find index in macro_names first)
    # also convert % growth rates to ratios
    rpp_idx = macro_names.index('HPQ')
    hp_scenarios = macro_scenarios[:, :, rpp_idx]/100 + 1
    # convert ratios to levels
    hp_scenarios = np.cumprod(hp_scenarios, axis=1).astype(np.float32)
    if verbose:
        print('   - [cr] House price scenarios processed')

    # STEP 1: Shift the cr_mu by log of the house price growth
    # for that adjust the dimensions of hp_scenarios
    hp_scenarios = hp_scenarios[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    # repeat along the 4th dimension (collateral types)
    hp_scenarios = np.repeat(hp_scenarios, cr_mu.shape[3], axis=3)
    # find the index of the RE collateral stat_type
    re_idx = collateral_types.index('Real Estate')
    # set hp_scenario to 1 for all non-RE collateral types, i.e. for all collateral types except re_idx
    hp_scenarios[:, :, :, np.arange(len(collateral_types)) != re_idx, :, :] = 1
    # finally shift the cr_mu by log of the house price growth
    cr_mu += np.log(hp_scenarios).astype(np.float32)
    del hp_scenarios
    gc.collect()
    if verbose:
        print('   - [cr] CR_mu shifted by house price growth')

    # STEP 2: Shift the cr_mu and cr_sigma2 by the sales ratio and v2s (=value to sale) parameters,
    #   but only in the Real Estate collateral stat_type
    # Start by adjusting v2s_mu to the shape
    # add a first dimension (scenarios) and repeat n_scenario times

    v2s_mu.data = np.repeat(v2s_mu.data[np.newaxis, ...], n_scenarios, axis=0).astype(np.float32)
    v2s_mu.names = ['scenario'] + v2s_mu.names
    validate_named_array(v2s_mu)
    v2s_sigma2.data = np.repeat(v2s_sigma2.data[np.newaxis, ...], n_scenarios, axis=0)
    v2s_sigma2.names = ['scenario'] + v2s_sigma2.names
    validate_named_array(v2s_sigma2)
    # add a last dimension (stage5) and repeat 5 times
    v2s_mu.data = np.repeat(v2s_mu.data[..., np.newaxis], 5, axis=-1)
    v2s_mu.names = v2s_mu.names + ['stage5']
    validate_named_array(v2s_mu)
    v2s_sigma2.data = np.repeat(v2s_sigma2.data[..., np.newaxis], 5, axis=-1)
    v2s_sigma2.names = v2s_sigma2.names + ['stage5']
    validate_named_array(v2s_sigma2)
    # add a last dimension (future_quarter) and repeat n_scenario_quarters times
    v2s_mu.data = np.repeat(v2s_mu.data[..., np.newaxis], n_scenario_quarters, axis=-1)
    v2s_mu.names = v2s_mu.names + ['future_quarter']
    validate_named_array(v2s_mu)
    v2s_sigma2.data = np.repeat(v2s_sigma2.data[..., np.newaxis], n_scenario_quarters, axis=-1)
    v2s_sigma2.names = v2s_sigma2.names + ['future_quarter']
    validate_named_array(v2s_sigma2)

    cr_mu[:, :, :, re_idx, :, :] += sr_mu + v2s_mu.data
    cr_sigma2[:, :, :, re_idx, :, :] += sr_sigma2 + v2s_sigma2.data
    if verbose:
        print('   - [cr] CR_mu and CR_sigma2 shifted')

    # STEP 3: Shift cr_mu for all portfolio and collateral types (not None, obviously) by half the amortization
    amort_mu = NamedArray(names=amortization_profiles.names.copy(),
                          data=amortization_profiles.data.copy())
    # keep only future_quarters # of quarters of amortization
    amort_mu.data = amort_mu.data[:, :, :, :valid_names_and_dimensions['future_quarter']]
    amort_mu.names[-1] = 'future_quarter'
    # take 1-amorization because that is the cumulated repayment until that quarter
    amort_mu.data = 1 - amort_mu.data
    # add a scenario dimension
    amort_mu.data = np.repeat(amort_mu.data[np.newaxis, ...], n_scenarios, axis=0)
    amort_mu.names = ['scenario'] + amort_mu.names
    validate_named_array(amort_mu)
    # add a collateral_type dimension in position 3
    amort_mu.data = np.repeat(amort_mu.data[..., np.newaxis], len(collateral_types), axis=-1)
    amort_mu.data = np.moveaxis(amort_mu.data, -1, 3)
    amort_mu.data[:, :, :, :, collateral_types.index('None'), :] = 0  # set repayment to 0 when there is no collateral
    amort_mu.names = amort_mu.names[:3] + ['collateral_type'] + amort_mu.names[3:]
    validate_named_array(amort_mu)
    # finally, in dimension with index 4, take the elements at index 2 and repeat it twice
    extracted_slice = amort_mu.data[..., 2, :]  # Extract the (..., 2, :) slice
    repeated_slice = np.repeat(extracted_slice[..., np.newaxis, :], 2, axis=-2)  # Repeat the slice twice
    amort_mu.data = np.concatenate((amort_mu.data, repeated_slice), axis=-2)  # Concatenate along the fifth axis
    amort_mu.names[4] = 'stage5'
    validate_named_array(amort_mu)
    # now we increase mu by half the amortization
    cr_mu -= np.log(1 - amort_mu.data/2)

    # decrease memory use
    if verbose:
        print_memory_usage()
    del amort_mu
    cr_mu = cr_mu.astype(np.float32)
    cr_sigma2 = cr_sigma2.astype(np.float32)
    if verbose:
        print_memory_usage()

    # define the named array for cr_mu and cr_sigma2
    cr_mu = NamedArray(names=['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage5', 'future_quarter'],
                       data=cr_mu)
    validate_named_array(cr_mu)
    cr_sigma2 = NamedArray(names=['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage5', 'future_quarter'],
                           data=cr_sigma2)
    validate_named_array(cr_sigma2)

    # now convert mu and sigma into lde
    gc.collect()
    cr_lgd = cr_to_lgd(cr_mu, cr_sigma2)
    validate_named_array(cr_lgd)

    # delete unused variables
    del cr_mu, cr_sigma2

    return cr_lgd
