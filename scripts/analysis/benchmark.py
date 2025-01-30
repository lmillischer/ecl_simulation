
import pandas as pd
import datetime as dt
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

from data.file_paths import input_path
from scripts.vectors_matrices.named_array import load_named_array


def prepare_provision_benchmark(run_code, ccy_code='', ccy_quantile=0.98):

    """
    This function prepares a file where for each bank-ptf-stage we obtain current & model cyclical & countercyclical provisions
    """

    # -------------------------------------------------------------------------------
    # Get the current cyclical provisions by stage
    current_cyc_prov = pd.read_excel('../../data/derived/current_cyclical_provisions.xlsx')
    current_cyc_prov.rename(columns={'ID_IMF': 'bank',
                                     'Portfolio': 'portfolio',
                                     'Exposure_stocks': 'outstanding',
                                     'Stages': 'stage',
                                     'DeterministicM_&_Procyclical_without_Voluntary_Prov': 'current_cyc_prov'
                                     }, inplace=True)
    current_cyc_prov = current_cyc_prov[['bank', 'portfolio', 'stage', 'outstanding', 'current_cyc_prov']]
    # now collapse and sum by bank, portfolio and stage
    current_cyc_prov = current_cyc_prov.groupby(['bank', 'portfolio', 'stage']).sum().reset_index()
    # per bank and portfolio compute a column that indicates the split of current_cyc_prov between stages
    current_cyc_prov['current_cyc_share'] = current_cyc_prov['current_cyc_prov'] / current_cyc_prov.groupby(['bank', 'portfolio'])['current_cyc_prov'].transform('sum')

    # -------------------------------------------------------------------------------
    # Get the current countercyclical provisions (by stage will be done later)

    current_ccy_prov = pd.read_excel('../../'+input_path, sheet_name='Historical provisions')
    current_ccy_prov = current_ccy_prov[current_ccy_prov['Period'].dt.date == dt.date(2023, 12, 31)]

    # rename columns
    current_ccy_prov.rename(columns={'ID_IMF': 'bank',
                                   'Portfolio': 'portfolio'
                                   }, inplace=True)
    current_ccy_prov['current_ccy_prov_banklevel'] = current_ccy_prov['General_prov'] + current_ccy_prov['Countercyclical_Prov']
    current_ccy_prov = current_ccy_prov[['bank', 'portfolio', 'current_ccy_prov_banklevel']]

    # -------------------------------------------------------------------------------
    # Get the model cyclical provisions (by stage)

    ecl_scenarios_abs = load_named_array(f'../../data/output/{run_code}/cl_scenarios.h5', 'cl_scenarios')

    with h5py.File('../../data/derived/input_data.h5', 'r') as h5file:
        banks = h5file['banks'][:]
        portfolio_types = h5file['portfolio_types'][:]
        portfolio_types = [byte.decode('utf-8') for byte in portfolio_types]  # decode strings

    # Compute the average ECL
    model_cyc_prov = np.mean(ecl_scenarios_abs.data, axis=0)
    # Convert the 3-dimensional np.array to a 2-dimensional pd.DataFrame
    dim1, dim2, dim3 = model_cyc_prov.shape
    index_1, index_2, index_3 = np.meshgrid(np.arange(dim1), np.arange(dim2), np.arange(dim3), indexing='ij')
    index_1 = index_1.flatten()
    index_2 = index_2.flatten()
    index_3 = index_3.flatten()
    values = model_cyc_prov.flatten()
    model_cyc_prov = pd.DataFrame({
        'ib': index_1,
        'ip': index_2,
        'stage': index_3+1,
        'model_cyc_prov': values
    })
    # transform the column ib into banks using the banks array
    model_cyc_prov['bank'] = banks[model_cyc_prov['ib']]
    # transform the column ip into portfolios using the portfolio_types array

    def get_string_from_ptf_list(index):
        return portfolio_types[index]
    model_cyc_prov['portfolio'] = model_cyc_prov['ip'].apply(get_string_from_ptf_list)
    model_cyc_prov = model_cyc_prov[['bank', 'portfolio', 'stage', 'model_cyc_prov']]

    # -------------------------------------------------------------------------------
    # Get the model countercyclical provisions

    if ccy_code != '':
        ecl_scenarios_ccy = load_named_array(f'../../data/output/{ccy_code}/cl_scenarios - baseline cycle-neutral.h5', 'cl_scenarios')

        # Compute the quantile of losses along axis 0 (scenarios) then sum over stages
        model_ccy_prov = np.quantile(ecl_scenarios_ccy.data, ccy_quantile, axis=0)
        # Convert the 3-dimensional np.array to a 2-dimensional pd.DataFrame
        dim1, dim2, dim3 = model_ccy_prov.shape
        index_1, index_2, index_3 = np.meshgrid(np.arange(dim1), np.arange(dim2), np.arange(dim3), indexing='ij')
        index_1 = index_1.flatten()
        index_2 = index_2.flatten()
        index_3 = index_3.flatten()
        values = model_ccy_prov.flatten()
        model_ccy_prov = pd.DataFrame({
            'ib': index_1,
            'ip': index_2,
            'stage': index_3 + 1,
            'model_ccy_prov': values
        })
        # transform the column ib into banks using the banks array
        model_ccy_prov['bank'] = banks[model_ccy_prov['ib']]
        # transform the column ip into portfolios using the portfolio_types array
        model_ccy_prov['portfolio'] = model_ccy_prov['ip'].apply(get_string_from_ptf_list)
        model_ccy_prov = model_ccy_prov[['bank', 'portfolio', 'stage', 'model_ccy_prov']]

    # -------------------------------------------------------------------------------
    # Get capital

    capital = pd.read_excel('../../data/input/CAPITALRATES_20240515.xlsx')
    capital = capital[['ID_FMI', 'cet1_%']]
    capital.rename(columns={'ID_FMI': 'bank',
                            'cet1_%': 'cet1r'}, inplace=True)
    capital.dropna(subset='cet1r', inplace=True)

    # -------------------------------------------------------------------------------
    # Get the RWA

    rwa = pd.read_excel('../../' + input_path, sheet_name='RWA')
    rwa.rename(columns={'ID_FMI': 'bank',
                        'RWA': 'rwa_banklevel'}, inplace=True)
    rwa = rwa[['bank', 'rwa_banklevel']]

    # -------------------------------------------------------------------------------
    # Merge everything

    # merge current cyc and ccy provisions
    provision_df = pd.merge(current_cyc_prov, current_ccy_prov, on=['bank', 'portfolio'], how='outer', indicator=True)
    print(provision_df.groupby('_merge', observed=True).count())
    provision_df.drop(columns='_merge', inplace=True)
    provision_df['current_ccy_prov'] = provision_df['current_ccy_prov_banklevel'] * provision_df['current_cyc_share']
    provision_df.drop(columns=['current_ccy_prov_banklevel', 'current_cyc_share'], inplace=True)

    # merge with model cyclical
    provision_df = pd.merge(provision_df, model_cyc_prov, on=['bank', 'portfolio', 'stage'], how='outer', indicator=True)
    print(provision_df.groupby('_merge', observed=True).count())
    provision_df.drop(columns='_merge', inplace=True)

    # merge with model countercyclical
    if ccy_code != '':
        provision_df = pd.merge(provision_df, model_ccy_prov, on=['bank', 'portfolio', 'stage'], how='outer', indicator=True)
        print(provision_df.groupby('_merge', observed=True).count())
        provision_df.drop(columns='_merge', inplace=True)
    else:
        provision_df['model_ccy_prov'] = np.nan

    # merge with rwa
    provision_df = pd.merge(provision_df, rwa, on='bank', indicator=True)
    print(provision_df.groupby('_merge', observed=True).count())
    provision_df.drop(columns='_merge', inplace=True)

    # merge with capital
    provision_df = pd.merge(provision_df, capital, on='bank', indicator=True)
    print(provision_df.groupby('_merge', observed=True).count())
    provision_df.drop(columns='_merge', inplace=True)

    # save to excel
    provision_df.fillna(0, inplace=True)
    provision_df.to_excel(f'../../data/output/{run_code}/provision_benchmark.xlsx', index=False)

    # collapse by bank and portfolio, some variables are summed, some taken the max
    provision_df = provision_df.groupby(['bank', 'portfolio']).agg({
        'outstanding': 'sum',
        'current_cyc_prov': 'sum',
        'current_ccy_prov': 'sum',
        'model_cyc_prov': 'sum',
        'model_ccy_prov': 'sum',
        'rwa_banklevel': 'max',
        'cet1r': 'max'
    }).reset_index()

    # save to excel
    provision_df.to_excel(f'../../data/output/{run_code}/provision_benchmark_ptf_level.xlsx', index=False)


def draw_sensitivity(baseline_code='', baseline_name='',
                     alternative_code='', alternative_name='',
                     do_by_stages=False,
                     do_countercyclical=False):
    "Can also be used to draw just portfolio-level ECL distributions when no alternative is provided."

    if alternative_code == '' and do_countercyclical:
        raise ValueError('Cannot do countercyclical without a cycle neutral distribution')

    # infrastructure
    with h5py.File(f'../../data/output/{baseline_code}/input_data.h5', 'r') as h5file:
        banks = h5file['banks'][:]
        portfolio_types = h5file['portfolio_types'][:]
        portfolio_types = [byte.decode('utf-8') for byte in portfolio_types]  # decode strings

    # get ECL scnearios
    ecl_scenarios_abs_baseline = load_named_array(f'../../data/output/{baseline_code}/cl_scenarios.h5', 'cl_scenarios')
    if alternative_code != '':
        ecl_scenarios_abs_alternative = load_named_array(f'../../data/output/{alternative_code}/cl_scenarios.h5', 'cl_scenarios')

    # aggregate
    if not do_by_stages:  # if we don't do by stages, we sum over stages and banks
        ecl_scenarios_abs_baseline = np.nansum(ecl_scenarios_abs_baseline.data, axis=(1, 3))
        if alternative_code != '':
            ecl_scenarios_abs_alternative = np.nansum(ecl_scenarios_abs_alternative.data, axis=(1, 3))
    else:   # if we do by stages, we only sum over banks
        ecl_scenarios_abs_baseline = np.nansum(ecl_scenarios_abs_baseline.data, axis=1)
        if alternative_code != '':
            ecl_scenarios_abs_alternative = np.nansum(ecl_scenarios_abs_alternative.data, axis=1)

    # current provisions
    if not do_by_stages:
        provision_df = pd.read_excel(f'../../data/output/{baseline_code}/provision_benchmark_ptf_level.xlsx')
        provision_df = provision_df.groupby('portfolio').sum().reset_index()  # sum over all banks
    else:
        provision_df = pd.read_excel(f'../../data/output/{baseline_code}/provision_benchmark.xlsx')
        provision_df = provision_df.groupby(['portfolio', 'stage']).sum().reset_index()  # sum over all banks
        provision_df = provision_df[provision_df['stage'] != 0]

    # drop useless columns and stage=0 (missing values)
    provision_df.drop(columns=['bank', 'rwa_banklevel', 'cet1r'], inplace=True)

    # -------------------------------------------------------------------
    # Now do the plotting

    if do_by_stages:
        for ip, p in enumerate(portfolio_types):
            for stat_type in ['rel']:  # ['abs', 'rel']:
                for stage in range(1, 4):
                    ecl_baseline = ecl_scenarios_abs_baseline[:, ip, stage-1]
                    if alternative_code != '':
                        ecl_alternative = ecl_scenarios_abs_alternative[:, ip, stage-1]
                    else:
                        ecl_alternative = None

                    # compute current provisions
                    current_provisions = provision_df.loc[(provision_df.portfolio == p) &
                                                          (provision_df.stage == stage), 'current_cyc_prov'].values[0]
                    current_ccyp = provision_df.loc[(provision_df.portfolio == p) &
                                                    (provision_df.stage == stage), 'current_ccy_prov'].values[0]
                    if stat_type == 'rel':
                        total_outstanding = provision_df.loc[(provision_df.portfolio == p) &
                                                             (provision_df.stage == stage), 'outstanding'].values[0]
                        current_provisions = current_provisions / total_outstanding * 100
                        current_ccyp = current_ccyp / total_outstanding * 100
                        ecl_baseline = ecl_baseline / total_outstanding * 100
                        if alternative_code != '':
                            ecl_alternative = ecl_alternative / total_outstanding * 100

                    draw_plot(ecl_baseline, ecl_alternative, alternative_code, baseline_code,
                              current_provisions, stat_type,
                              do_countercyclical, current_ccyp, p,
                              stage, do_by_stages=True)
    else:
        for ip, p in enumerate(portfolio_types):
            for stat_type in ['rel']:  # ['abs', 'rel']:
                ecl_baseline = ecl_scenarios_abs_baseline[:, ip]
                if alternative_code != '':
                    ecl_alternative = ecl_scenarios_abs_alternative[:, ip]
                else:
                    ecl_alternative = None

                # compute current provisions
                current_provisions = provision_df.loc[provision_df.portfolio == p, 'current_cyc_prov'].values[0]
                current_ccyp = provision_df.loc[provision_df.portfolio == p, 'current_ccy_prov'].values[0]
                if stat_type == 'rel':
                    total_outstanding = provision_df.loc[provision_df.portfolio == p, 'outstanding'].values[0]
                    current_provisions = current_provisions / total_outstanding * 100
                    current_ccyp = current_ccyp / total_outstanding * 100
                    ecl_baseline = ecl_baseline / total_outstanding * 100
                    if alternative_code != '':
                        ecl_alternative = ecl_alternative / total_outstanding * 100

                draw_plot(ecl_baseline, ecl_alternative, alternative_code, baseline_code,
                          current_provisions, stat_type,
                          do_countercyclical, current_ccyp, p)


def draw_plot(ecl_baseline, ecl_alternative, alternative_code, baseline_code,
              current_provisions, stat_type,
              do_countercyclical, current_ccyp, p,
              stage=None, do_by_stages=False):

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Plot the histogram

    # set the max value of the historgram to a high percentile of ecl_baseline
    max_mult = 4
    max_quant = 0.99
    max_ecl = max_mult * np.nanmean(ecl_baseline)
    if alternative_code != '':
        if np.nanmean(ecl_alternative) * max_mult > max_ecl:
            max_ecl = np.nanmean(ecl_alternative) * max_mult
    high_quant = np.nanquantile(ecl_baseline, max_quant)
    if alternative_code != '':
        high_quant = max(np.nanquantile(ecl_baseline, max_quant),
                         np.nanquantile(ecl_alternative, max_quant))
    if high_quant < max_ecl:
        max_ecl = high_quant
    plt.hist(ecl_baseline, bins=100, density=True, label='ECL baseline', range=(0, max_ecl))
    if alternative_code != '':
        plt.hist(ecl_alternative, bins=100, density=True, label=f'ECL {alternative_code}', range=(0, max_ecl), alpha=0.4)

    # Add a horizontal line at the ECL
    model_cyclical = ecl_baseline.mean()
    label_string = f'Model provisions = {model_cyclical:.1f}%' if stat_type == 'rel' \
        else f'Model provisions = {model_cyclical / 1e9:,.0f} bn COP'
    plt.axvline(model_cyclical, color='black', linestyle='dashed', linewidth=1, label=label_string)
    if (alternative_code != '') and not do_countercyclical:
        plt.axvline(ecl_alternative.mean(), color='orange', linestyle='dashed', linewidth=1,
                    label=f'Model provisions ({alternative_code}) = {ecl_alternative.mean():.1f}%')
    if do_countercyclical:  # draw a line at the implied countercyclical provisions
        model_total_prov = model_cyclical+current_ccyp
        label_model_ccyp = f'Model total provisions = {model_total_prov:.1f}%' if stat_type == 'rel' \
            else f'Model total provisions = {(model_total_prov) / 1e9:,.0f} bn COP'
        plt.axvline(model_total_prov, color='orange', linestyle='solid', linewidth=1, label=label_model_ccyp)

    # Add vertical line at the current provision
    label_current = f'Current cyclical provisions = {current_provisions:.1f}%' if stat_type == 'rel' \
        else f'Current cyclical provisions = {current_provisions / 1e9:,.0f} bn COP'
    plt.axvline(current_provisions, color='green', linestyle='dashed', linewidth=2, label=label_current)
    if do_countercyclical:
        label_current_ccyp = f'Current total provisions = {current_ccyp+current_provisions:.1f}%' if stat_type == 'rel' \
            else f'Current total provisions = {(current_ccyp+current_provisions) / 1e9:,.0f} bn COP'
        plt.axvline(current_provisions+current_ccyp, color='green', linestyle='solid', linewidth=2, label=label_current_ccyp)

    # Add legend
    legend = plt.legend()
    plt.draw()

    # Compute the implied quantile of the cycle neutral distribution
    if do_countercyclical:
        print(f'\nPortfolio {p}')
        sorted_ecl_cycle_neutral = np.sort(ecl_alternative)
        print(f'Maximum ECL in cycle neutral distribution: {sorted_ecl_cycle_neutral[-1]:.1f}')
        quantile = 100 * sorted_ecl_cycle_neutral.searchsorted(model_total_prov) / len(sorted_ecl_cycle_neutral)
        print(f'Quantile of cycle neutral distribution: {quantile:.1f}')

        # Write text on plot
        plt.text(0.02, 0.5, f'CCyP-implied quantile of\ncycle-neutral distribution:\n{quantile:.1f}%',
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform=ax.transAxes)

    # Finish plot and save
    plt.gca().set_yticks([])
    plt.title(f'System-wide Provisions - {p} Lending Portfolio{f" (Stage {stage})" if do_by_stages else ""}')
    plt.xlabel(f'Scenario loss {"[COP]" if stat_type == "abs" else "[%]"}')
    plt.ylabel('Density')
    # legend top left
    # plt.legend(loc='upper left')

    plt.tight_layout()

    # if folders do not exist, create them
    if not os.path.exists(f'../../plots/{baseline_code}/summary/'):
        os.makedirs(f'../../plots/{baseline_code}/summary/')
    if not os.path.exists(f'../../plots/{baseline_code}/sensitivity/'):
        os.makedirs(f'../../plots/{baseline_code}/sensitivity/')
    if not os.path.exists(f'../../plots/{baseline_code}/countercyclical/'):
        os.makedirs(f'../../plots/{baseline_code}/countercyclical/')

    # save the figure
    if alternative_code == '':  # if just one run
        plt.savefig(f'../../plots/{baseline_code}/summary/system_wide_ecl_{p}{f"_stage{stage}" if do_by_stages else ""}.png', dpi=300)
    elif not do_countercyclical:  # if two runs but no countercyclical, we are running sensitivity
        plt.savefig(f'../../plots/{baseline_code}/sensitivity/system_wide_ecl_{p} - {alternative_code}.png', dpi=300)
    else:  # else we are running the countercyclical plots
        plt.savefig(f'../../plots/{baseline_code}/countercyclical/ecl_{p} - {alternative_code}.png', dpi=300)
    plt.close()


def bank_by_bank_impact(run_code):
    df = pd.read_excel(f'../../data/output/{run_code}/provision_benchmark_ptf_level.xlsx')
    df['del_provisions'] = df['model_cyc_prov'] - df['current_cyc_prov']
    df['cet1'] = df['rwa_banklevel'] * df['cet1r'] / 100

    # compute impact by bank over all portfolios
    df_sum = df.groupby('bank').agg({
        'outstanding': 'sum',
        'current_cyc_prov': 'sum',
        'current_ccy_prov': 'sum',
        'model_cyc_prov': 'sum',
        'model_ccy_prov': 'sum',
        'del_provisions': 'sum',
        'cet1': 'max',
        'rwa_banklevel': 'max',
        'cet1r': 'max'
    }).reset_index()
    df_sum['portfolio'] = 'Total'

    # add df_sum to the bottom of df (append does not work)
    df = pd.concat([df, df_sum], ignore_index=True)

    # compute the impact on CET1
    df['cet1_after'] = df['cet1'] - df['del_provisions']

    # compute the impact on RWAs
    # start by computing the RW density only for portfolio = 'Total'
    df_total = df.loc[df['portfolio'] == 'Total'].copy()
    df_total['rwa_density'] = df_total['rwa_banklevel'] / df_total['outstanding']
    # then merge it back to the main df
    df = pd.merge(df, df_total[['bank', 'rwa_density']], on='bank', how='left')
    df['rwa_after'] = df['rwa_banklevel'] - df['rwa_density'] * df['del_provisions']

    # compute the impact on CET1 ratio
    df['cet1r_after'] = 100 * df['cet1_after'] / df['rwa_after']
    df['del_cet1r'] = df['cet1r_after'] - df['cet1r']

    # replace all inf with nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # save to excel
    df.to_excel(f'../../data/output/{run_code}/bank_by_bank_impact.xlsx', index=False)

    # -------------------------------------------------------------------
    # Now do the plotting

    # plot a boxplot for portfolio = 'Total' of the del_cet1r
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    df_box = df[['bank', 'portfolio', 'del_cet1r']]
    # reshape wide
    print(df_box)
    df_box = df_box.pivot(columns='portfolio', values='del_cet1r', index='bank')
    print(df_box)
    data_for_boxplots = [df_box[col].dropna() for col in df_box.columns.tolist()]
    plt.boxplot(data_for_boxplots, positions=[1, 2, 3, 4, 5], widths=0.6, vert=True)
    plt.title('Impact of the Change in Cyclical Provisions on bank CET1 Ratios')
    plt.xticks([1, 2, 3, 4, 5], ['Commercial', 'Consumer', 'Microcredit', 'Mortgage', 'Total'])

    # annotate
    plt.ylabel('Impact bank CET1 ratios [%]')

    # save and close
    plt.tight_layout()
    plt.savefig(f'../../plots/{run_code}/summary/impact_on_cet1_ratio.png', dpi=300)
    plt.close()



# prepare_provision_benchmark(ccy_quantile=0.98)
# prepare_provision_benchmark('7CQsoo1e')
# bank_by_bank_impact('7CQsoo1e')
# draw_sensitivity('7CQsoo1e', '', do_by_stages=True, do_countercyclical=False)
