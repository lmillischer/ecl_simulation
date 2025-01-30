from scipy.stats import norm
import numpy as np

from scripts.vectors_matrices.named_array import *


def project_basic_lgds(basic_lgds, tm_scenarios_3x3, historical_tms, rho=0.15):
    """Project the basic LGDs into the future using Frye-Jacobs"""

    # --------------------------------------------------------------------------------------------------------
    # First we compute the k parameter for every bank-portfolio
    # --------------------------------------------------------------------------------------------------------

    # for that we need the current PD for each bank-portfolio-stage
    longrun_pds = np.nanmean(historical_tms.data[:, :, :, :, 2], axis=2)

    # to avoid nans, we replace longrun_pds with 0.01% where they are 0 or missing
    epsilon_pd = 0.0001
    longrun_pds[longrun_pds == 0] = epsilon_pd
    longrun_pds[np.isnan(longrun_pds)] = epsilon_pd

    if np.all(np.isnan(longrun_pds)):
        raise ValueError('  [Frye-Jacobs] All PDs are NaN')
    longrun_pds = longrun_pds[np.newaxis, :, :, np.newaxis, :]  # add new empty dimensions for scenarios and collateral types
    # extend the last dimension of longrun_pds from 3 to 4 by copying the last slice twice (same longrun_pds for stage 3b1 and 3b2)
    longrun_pds = np.concatenate((longrun_pds, longrun_pds[:, :, :, :, 2, np.newaxis]), axis=4)

    longrun_pds = NamedArray(names=['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage4'],
                     data=longrun_pds)
    # and we need the current lgds (which depend only on portfolio and stage)
    lgds = NamedArray(names=basic_lgds.names[:5].copy(),
                      data=basic_lgds.data[:, :, :, :, :-1, 0].copy())  # we drop stage 3b3 and select quarter 0 (current quarter)
    lgds.names = ['stage4' if item == 'stage5' else item for item in lgds.names]  # dim now stage4 (ie. 1, 2, 3b1, 3b2)
    # compute the k parameter
    validate_named_array(longrun_pds)
    validate_named_array(lgds)
    if longrun_pds.names != lgds.names:
        print('longrun_pds names:', longrun_pds.names)
        print('longrun_pds dims', longrun_pds.data.shape)
        print('lgds names:', lgds.names)
        print('lgds dims', lgds.data.shape)
        raise ValueError('longrun_pds and lgds must have the same dimensions')

    # --------------------------------------------------------------------------------------------------------
    # Get future PIT PDs
    # --------------------------------------------------------------------------------------------------------

    # first we get the future PDs
    future_pds = tm_scenarios_3x3.data[:, :, :, :, :, 2]  # select column 2 of the TM (PD) for all scenarios and quarters
    # extend the last dimension of longrun_pds from 3 to 4 by copying the last slice twice
    #   (same longrun_pds, i.e. non-cure rates for stage 3b1 and 3b2)
    future_pds = np.concatenate((future_pds, future_pds[:, :, :, :, 2, np.newaxis]), axis=4)
    # add an empty dimension at the end for collateral types (before stages)
    future_pds = future_pds[:, :, :, :, np.newaxis, :]
    # move the future quarters dimension to the end
    future_pds = np.moveaxis(future_pds, 3, 5)
    future_pds = NamedArray(names=['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage4', 'future_quarter'],
                            data=future_pds)
    validate_named_array(future_pds)

    # --------------------------------------------------------------------------------------------------------
    # For each bank-portfolio-stage we look for the long term LGD that ensures LGD_pit = Basic LGD
    # --------------------------------------------------------------------------------------------------------

    longrun_lgd = longrun_lgd_frye_jacobs(lgd_pit=lgds.data[...],
                                          pd_pit=future_pds.data[..., 0],
                                          pd_longrun=longrun_pds.data[...],
                                          rho=rho)
    longrun_lgd = NamedArray(names=['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage4'],
                             data=longrun_lgd)
    validate_named_array(longrun_lgd)

    # --------------------------------------------------------------------------------------------------------
    # Compute k for Frye-Jacobs application
    # --------------------------------------------------------------------------------------------------------

    k = (norm.ppf(longrun_pds.data) - norm.ppf(longrun_pds.data * longrun_lgd.data)) / np.sqrt(1 - rho)

    # final dimensionality: scenarios x banks x portfolio_types x collateral_types x stages (onyl 4) x quarters
    k = NamedArray(names=['scenario', 'bank', 'portfolio_type', 'collateral_type', 'stage4', 'future_quarter'],
                   data=k)

    # --------------------------------------------------------------------------------------------------------
    # Then we project the LGDs into the future using future PDs and the k parameter
    # --------------------------------------------------------------------------------------------------------

    # then project LGDs forward using the Frye-Jacobs formula
    basic_lgd_scenarios = NamedArray(names=future_pds.names,
                                  data=norm.cdf(norm.ppf(future_pds.data) - k.data[..., np.newaxis]) / future_pds.data)
    basic_lgd_scenarios.data = np.minimum(basic_lgd_scenarios.data, 1)  # cap LGDs at 100%
    validate_named_array(basic_lgd_scenarios)

    # return the scenarios
    return basic_lgd_scenarios


def longrun_lgd_frye_jacobs(lgd_pit, pd_pit, pd_longrun, rho):
    "Find the long run LGD that ensures LGD_pit = Basic LGD using Frye-Jacobs formula"

    # print('')
    # # print('lgds', lgd_pit.names)
    # print('lgd pit', lgd_pit.data.shape)
    # # print('longrun_pds', pd_longrun.names)
    # print('pd longrun', pd_longrun.shape)
    # # print('future_pds', pd_pit.names)
    # print('pd pit', pd_pit.data.shape)

    inner_term = np.sqrt(1 - rho) * (norm.ppf(lgd_pit * pd_pit) - norm.ppf(pd_pit))
    lgd_longrun = norm.cdf(inner_term + norm.ppf(pd_longrun))/pd_longrun

    return lgd_longrun
