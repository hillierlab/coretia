import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wellmap


def load_lum(path: Path):
    """
       Reads a 96 well place stored in xlsx or csv. Returns a [row, col, Lum] long pandas table from a 96 plate

       Parameters
       ----------
       path - points to a xlsx file, both file have same name only extension differs.

       Returns
       -------
       dataframe with columns [row, col, Lum] where row spans A-G, col 1-12, Lum is NaN or float

       """

    if 'csv' in path.suffix:
        df = pd.read_csv(path, sep=";").rename(columns={'Lum': 'row'})
    elif 'xls' in path.suffix:
        df = pd.read_excel(path, usecols='A:M', nrows=8)
        df = df.rename({'Unnamed: 0': 'row'}, axis=1)

    df1 = df.melt(id_vars=['row'], var_name='col', value_name='Lum')
    #df2 = df1.dropna()  # empty row/cols should be removed to match layout
    return df1


def import_plate(toml_path: Path, path_guess='{0.stem}'):
    try:
        layout, data = wellmap.load(toml_path, data_loader=load_lum, path_guess=path_guess+'.xlsx')
    except wellmap.LayoutError:
        layout, data = wellmap.load(toml_path, data_loader=load_lum, path_guess=path_guess+'.csv')
    except Exception as e1:
        print(f"Neither {toml_path.stem}.xlsx nor .csv is found or it cannot be properly parsed.")
        raise

    # If firefly reporter is used in any of the plates, fill-in other plates with 'nanoluc'
    if 'reporter' in layout:
        layout['reporter'] = layout['reporter'].astype(str)
        # Replace the 'nan' string (which represents the former NaN values) with 'nanoluc'
        layout['reporter'] = layout['reporter'].replace('nan', 'nanoluc')

    meta = parse_multiplate_meta(toml_path)
    data['well'] = data['row'] + data['col'].astype(str)
    df1 = pd.merge(layout, data, on=['well', 'path'])
    meta, plates = merge_meta(df1, meta)

    # Check if both categories are present in df
    category_pair = extract_category_pair(meta, plates)

    # extract sample IDs, drop categories (e.g. 'DMEM|FBS')
    if category_pair is not None:
        pattern = '|'.join(category_pair)
        df1['condition'] = df1['plate'].str.extract(rf"({pattern})", expand=False)
        # Group by 'plate' and fill NaN values in 'sample' with the corresponding sample string for each plate
        cat_in_df = df1['condition'].isin(category_pair)
        if category_pair is not None and not all(cat_in_df):
            raise RuntimeError(f"One of {category_pair} defined in TOML is not present in df: {cat_in_df}")
    else:
        df1['condition'] = df1['plate']
    df1['sample'] = df1.groupby('plate')['sample'].transform(lambda x: x.ffill().bfill())

    # Check for NaNs in control columns
    if df1.loc[df1['negative_control'] == 1, 'Lum'].isna().any():
        raise ValueError(f"Negative control contains NaN values in TOML file: {toml_path}")
    if df1.loc[df1['no_antibody_control'] == 1, 'Lum'].isna().any():
        raise ValueError(f"No-antibody control contains NaN values in TOML file: {toml_path}")

    # Define the pattern for columns to drop (starting with 'well', 'row', or 'col')
    pattern = r'^(well|row|col)'

    # Find columns matching the pattern
    cols_to_drop = df1.filter(regex=pattern).columns

    # Drop the unwanted columns
    df_filtered = df1.drop(columns=cols_to_drop)

    extras = extras_from_meta(meta, plates)
    # Check if all plates have groupby or aggregator has a groupby
    all_d_c = [d_c_from_meta(meta[pl1].extras, toml_path) for pl1 in plates]
    if not all([a1 == all_d_c[0] for a1 in all_d_c]):
        raise RuntimeError(
            f"Multiplate TOML has inconsistent groupby defined, this should not occur as aggregator TOML overrides individual ones.")
    d_c = all_d_c[0]
    return df_filtered, extras, plates, d_c, category_pair


def extract_category_pair(meta, plates):
    """
    Extract category pairs for plotting based on metadata or plate names.
    """
    if 'concat' in meta and 'categories' in meta['concat'].extras['plot']:
        category_pair = meta['concat'].extras['plot']['categories']
        if len(category_pair) != 2:
            raise NotImplementedError(
                f"Grouping curves along two categories is supported. Found: {category_pair}"
            )
        return category_pair

    if len(plates) == 2:
        common, category_pair = extract_common_and_remainders(plates)
        if len(category_pair) == 2:
            print(f"Processing two curves with sample_id {common} and conditions {category_pair}.")
            return category_pair
        print("TOML plate names do not define a pair of conditions.")
    return None


def merge_meta(df, meta):
    """
    Clone parameters into each plate from aggregator TOML.
    """
    plates = df['plate'].unique()  # array

    # If multiplate: clone settings from aggregator to plate extras dicts
    extra_keys_clone = ['basic_stats',
                        'plot', 'n_bootstrap']  # normalize can be defined both in aggregator TOML or in each plate, leave that in extras['concat']
    if 'concat' in meta:
        for pk1 in plates:
            if 'normalize' in meta[pk1].extras:
                warnings.warn(
                    f"Multiplate TOML file uses normalize.groupby from the aggregator TOML file, not in individual plate TOML files. {meta}")
            meta[pk1].extras.update({ek1: ev1 for ek1, ev1 in meta['concat'].extras.items() if ek1 in extra_keys_clone})
    return meta, plates


def parse_multiplate_meta(toml_path: Path):
    """
    Parses extra parameters from each well into a dict.

    Parameters
    ----------
    toml_path: relative or absolute path to aggregator TOML that lists plate TOML files under [concat]

    Returns
    -------
    dict or dicts
    """
    import tomllib
    with open(toml_path, 'rb') as agg:
        plates = tomllib.load(agg)
    if 'meta' not in plates or 'concat' not in plates['meta']:
        # not multiplate, return dict with plate name as key
        cfg, dummy, dummy, meta1 = wellmap.config_from_toml(toml_path)
        if 'expt' not in cfg or 'plate' not in cfg['expt']:
            raise wellmap.LayoutError(f"TOML {toml_path} must have an ['expt'] plate = 'name' section defined.")
        return {cfg['expt']['plate']: meta1}
    pathp = Path(toml_path).parent  # individual plate TOML files are assumed to be specified relative to same directory
    cfg, dummy, dummy, meta1 = wellmap.config_from_toml(toml_path)
    meta = {'concat': meta1}  # store extras specified in the aggregator file
    if 'merge' in cfg:
        raise DeprecationWarning(f"merge keyword is not used in wellmap formats, this code is never running")
    for pk1, pv1 in plates['meta']['concat'].items():
        if isinstance(pv1, str):
            meta[pk1] = wellmap.config_from_toml(pathp/pv1)[3]
        elif isinstance(pv1, list):
            # This is a workaround, when merging from [concat] and plate = [path1, path2]
            meta[pk1] = wellmap.config_from_toml(pathp/pv1[0])[3]
    return meta


def import_multiplate(toml_path: Path):
    """
    Loads an aggregator TOML file.

    Returns:
        x1: Array of dilutions, n x d where n is the number of curves, d is the number of dilutions
        y1: Array of luminance values n x d x r where r is the number of technical repeats
        data: dataframe after subtracting negative control and normalized with no-antibody control
        df: raw imported dataframe
        meta:
        plates: list of strings with the names of each sub-TOML file
    """
    if not isinstance(toml_path, (str, Path)):
        TypeError("toml_path must be one aggregator TOML path")
    toml_path = Path(toml_path)

    df, extras, plates, d_c, category_pair = import_plate(toml_path)

    x, data = apply_neg_pos_control(df, d_c, plates)
    y = list(data.values())
    for k1, v1 in data.items():
        if len(v1) != len(y[0]):
            print(f"{k1} has {len(v1)} dilution points, not all curves have same number of dilution points.")
    x1 = list(x.values())
    y1 = y

    if d_c not in ['dilution', 'conc_ng']:
        raise NotImplementedError(
            "TOML file has to have a basic_stats.groupby list containing 'dilution' or 'conc_ng'.")

    return x1, y1, (x, data), df, extras, plates, d_c


def normalize_luminescence(df):
    # normalize the luminescence values for each sample and fbs

    # Extract no-antibody and negative control luminescence values for the current sample and fbs
    negative_control_mean = df[df['negative_control'] == 1]['Lum'].mean()
    no_antibody_control_mean = df[df['no_antibody_control'] == 1]['Lum'].mean()

    # Normalize the luminescence values
    df['Lum_normalized'] = (df['Lum'] - negative_control_mean) / (no_antibody_control_mean - negative_control_mean)
    if any(df['Lum_normalized'].isna()):
        print(f"Missing values in plate map {df['plate'].dropna().unique()}: {df['Lum_normalized']}.")

    if not all(df['Lum_normalized'].dropna() > -0.04):
        raise RuntimeError(f"Possible error in plate map {df['plate'].dropna().unique()}, negative normalized luminescence values detected: {df['Lum_normalized']}.")
    return df


def normalize(df, d_c):
    """
    Normalize to no-antibody control, removing negative control offset.
    Parameters
    ----------
    df:
    well well0 row_x col_x  ... negative_control row_y col_y       Lum
    0     A1   A01     A     1  ...              NaN     A     1    8905.0
    1     A2   A02     A     2  ...              NaN     A     2   11692.0
    d_c: 'conc_ng' or 'dilution'

    Returns
    -------
                        Lum  fbs        sample  conc_ng
    plate
    ADK9_1 DMEM 0   0.193792   2%  ADK9 in DMEM    0.313
                1   0.254486   2%  ADK9 in DMEM    0.313

    """

    # Filter out the no-antibody control (dilution value of 1.0)
    gblist = ['Lum', 'no_antibody_control', 'negative_control', 'sample', 'condition', 'plate', d_c]
    mlist = ['Lum', 'sample', 'condition', d_c]
    all_single_dilution = None

    df_normalized = df.groupby(['plate'])[gblist].apply(normalize_luminescence)
    df_normalized = df_normalized.drop(columns=['Lum'])
    df_normalized = df_normalized.rename(columns={'Lum_normalized': 'Lum'})
    df_model = df_normalized[mlist].copy()

    df1 = df_model.dropna(subset=mlist)
    return df1, all_single_dilution


def curve_is_full_sigmoid(y_curve, thr_low=0.15, thr_high=0.85, debug = False):
    y = np.array(y_curve)
    if y.ndim != 2:
        raise RuntimeError(f"y_curve must have shape N_dilutions x N_technical_repeats, got {y.shape}.")
    y = y.mean(axis=1)  # Ensure it's a NumPy array

    # Check conditions: second value > thr_low and second-to-last value < thr_high
    if np.all(y[:2] < thr_low) and np.all(y[-2:] > thr_high):
        is_sigmoid = True
    else:
        is_sigmoid = False
    if debug:
        print(f"Curve: {y}, sigmoid: {is_sigmoid}")
    return is_sigmoid

def basic_stats(df: pd.DataFrame, **kwargs):
    """
    Compute mean, sem, CV along one column.
    Parameters
    ----------
    df: dataframe returned by wellmap.load
    stats_groupby: group repeated measurements along this variable, e.g. capsid

    Returns
    -------
    dataframe where measurements repeated for the same value of 'column' are returned as mean, sem, CV
    """

    params = kwargs.get('basic_stats')
    if params is None or 'groupby' not in params:
        raise ValueError("TOML file has to define basic_stats.groupby = 'capsid' or another parameter.")
    else:
        groupby = params['groupby']
    avg_f = params.get('avg_f', 'mean')  # 'median' or 'mean' to be used to calculate average of technical replicates
    std_f = params.get('std_f', 'std')  # 'median_abs_deviation' or 'std'
    if std_f == 'median_abs_deviation':
        import scipy.stats
        std_fo = scipy.stats.median_abs_deviation
    else:
        std_fo = std_f  # e.g. 'mean' or other string handled by .agg()

    # If there is only one no-antibody control on the plate and not one no-antibody control for every condition (groupby[0], e.g. 'capsid')
    # then copy the no-antibody control to all groups
    pcont = df.loc[df['no_antibody_control']==1]  # rows containing technical replicates of no-antibody control
    try:
        gkeys = df[groupby[0]].dropna().unique()  # rows not having no-antibody control assigned yet
    except:
        raise ValueError(f"groupby {groupby[0]} not found in df columns: {df.columns}")
    all_no_antibody_control_na = all(pcont[groupby[0]].isna())
    if all_no_antibody_control_na:  # one no-antibody control on the plate for all conditions on plate, clone values for each condition
        df.loc[df['no_antibody_control']==1, groupby[0]] = gkeys[0]  # use the existing rows as no-antibody control for first condition
        for gkey in gkeys[1:]:  # iterate through conditions e.g. capsid types; start from second condition
            temp_pcont = pcont.copy()
            temp_pcont[groupby[0]] = gkey
            df = pd.concat([df, temp_pcont])

    # all no-antibody controls have a condition assigned, check if all conditions are covered
    try:
        assert all(gkeys == df.loc[df['no_antibody_control']==1, groupby[0]].unique())
    except:
        raise ValueError(f"Not all conditions: {groupby[0]}:{gkeys} have a no-antibody control. ")

    # Important to only retrieve Lum (numeric value), otherwise agg below will complain about non-numeric columns
    data = {'negative_control': df.groupby('negative_control')[['Lum']],
            'treated': df.groupby(groupby)[['Lum']]  # NC is implicitly dropped as it contains NaNs  group_keys=['no_antibody_control','plot_order']
            }

    # compute mean of NC then subtract from all values
    data_a = {k1: data[k1].agg([avg_f, std_fo, 'sem']) for k1 in data.keys()}  # remove Lum from column multiindex
    if len(data_a['negative_control']) != 1:
        raise RuntimeError(f"Negative control should be one number, it is {data_a['negative control']}")

    data_a['treated']['mean-NC'] = data_a['treated'][('Lum',avg_f)]-data_a['negative_control'][('Lum',avg_f)].tolist()
    for k1 in data.keys():
        data_a[k1][('Lum','CV')] = data_a[k1][('Lum',std_f)]/data_a[k1][('Lum',avg_f)]  # coefficient of variation
    out_cols = [('Lum', avg_f), ('Lum', std_f), ('Lum','sem'), ('Lum','CV')]
    for ekey in ['no_antibody_control', 'plot_order']:  # if extra tags defined in plate TOML, forward those from plate dataframe along with calculated stats
        if ekey in data_a['treated'].columns:
            out_cols.append((ekey,avg_f))

    outdf = data_a['treated'][out_cols]
    return outdf # negative control subtracted


def apply_neg_pos_control(df, d_c, plates, include_no_antibody_control=False, threshold=0.15):
    """
    From each well, subtracts median of negative control and normalizes to median of no-antibody control.

    Parameters
    ----------
    df
    meta
    plates: list of str containing plate names
    threshold: float, if normalized luminance at lowest dilution (e.g. 1/4) is higher than this value, a Hill fit will not be good enough.
                if normalized luminance at highest dilution (e.g. 1/256) is lower than this value, a Hill fit will not be good enough.

    Returns
    -------
    x: dict containing dilution values [4,8,16, etc].
        Keys are [plate name + ' ' + groupby value],
        e.g. x['DMEM Human 2'] if in anett_aav9.toml:
        basic_stats.groupby = ['sample', 'dilution']
        [meta.concat]
        'DMEM' = '20230222_human1.toml'
        and sample='Human 2' was set in '20230222_human1.toml'
    data: same keys as x and each key holds the normalized, negative control corrected luminance values.

    """

    data, x = {}, {}
    if 'sample' in df.columns:
        gbk = 'sample'
    else:
        raise NotImplementedError("If plates using dilution or conc_ng series but do not define sample as parameter, use groupby[0].")
    for pl1 in plates:
        df1 = df[df['plate'] == pl1]
        pcval = 1 if d_c == 'dilution' else 0

        neg_cont = df1[df1['negative_control'] == 1]['Lum'].median()
        lum_col = df1['Lum']
        lum_col -= neg_cont
        pc_median = df1[df1['no_antibody_control'] == 1]['Lum'].median()
        if pd.isna(pc_median):
            raise RuntimeError(f"{df1['plate'].unique()} no-antibody control median returned NaN.")
        lum_col /= pc_median
        df1.loc[:,'Lum'] = lum_col.astype(float)

        for gk, gv in df1.groupby(gbk):
            y_reps = []
            x1 = []

            # Generate x for no-antibody control as lowest dilution/4
            series = np.sort(gv[d_c].unique())
            if series[0] == pcval:
                Warning(
                    f"no-antibody control has no {d_c} defined, yet lowest dilution point is {series[0]}.")
                lowest_dil_conc = series[1]
            else:
                lowest_dil_conc = series[0]
            pc_x = lowest_dil_conc / 4

            # no-antibody control has no [dilution/conc_ng] or sample defined, hence will not pop up during groupby:
            if include_no_antibody_control and (
                    any(pd.isna(df1[df1['no_antibody_control'] == 1][d_c]))
                    or pc_x not in [gk1 for gk1, gd1 in gv.groupby(d_c)]):
                # add no-antibody control with lowest dilution or conc_ng/4 (2 log2 units)
                x1.append(pc_x)
                y_reps.append(np.array(df1[df1['no_antibody_control'] == 1]['Lum']))

            # collect into dilution, lum vector with repeated values for same dilution
            for gk1, gd1 in gv.groupby(d_c):
                if (gk1 == pcval):  # if needed, pc is included above
                    # print(f"{pl1}: not repeating no-antibody control.")
                    continue  # discard no-antibody control here if dilution=1 or conc_ng=0 is set
                else:
                    y_reps.append(np.array(gd1['Lum']))
                    x1.append(gk1)
                cval = np.array(gd1['Lum']).mean()
                top_i = -1 if d_c == 'dilution' else 0  # e.g. 256 for 'dilution' and 0.02 for conc_ng
                bottom_i = 0 if d_c == 'dilution' else -1  # e.g. 4 for 'dilution' and 0.15 for conc_ng
                if gk1 == series[top_i] and cval < 1-threshold:
                      print(f"{pl1}: Lowest cc (highest dilution) {series[0]} is too low, curve[{series[0]}]={cval} starts below {1 - threshold}")
                elif gk1 == series[bottom_i] and cval > threshold:
                    print(
                        f"{pl1}: Highest cc (lowest dilution) {series[-1]} is too high, curve[{series[-1]}]={cval} ends higher than {threshold}")
            data[pl1 + ' ' + str(gk)] = np.array(y_reps)
            x[pl1 + ' ' + str(gk)] = np.array(x1)
    return x, data


def extract_common_and_remainders(strings):
    """

    Parameters
    ----------
    strings: e.g. ['20240725 AAV9', '20240731 AAV9']

    Returns
    -------
    'AAV9', ['20240725', '20240731']
    """
    # Split each string into tokens
    split_strings = [s.split() for s in strings]

    # Convert to sets for easy comparison
    sets = [set(parts) for parts in split_strings]

    # Find the common tokens using set intersection
    common_tokens = set.intersection(*sets)

    # Determine the common part (one of the common tokens)
    common_part = ' '.join(common_tokens) if common_tokens else None

    # Extract the remainders by removing common tokens from each string
    remainders = []
    for parts in split_strings:
        differing_parts = [part for part in parts if part not in common_tokens]
        remainders.append(' '.join(differing_parts))

    return common_part, remainders

def extract_nd50_reference(reference_toml):
    """
    Exports a reference pickle file to be loaded by multi-plate analysis that use the same plates.

    Parameters
    ----------
    reference_toml: str

    Returns
    -------

    """
    from coretia import process
    nd50_mlh = process.process(reference_toml)
    return nd50_mlh


def d_c_from_meta(extras, toml_path, default_groupby=True):
    params = extras.get('basic_stats')
    if not params or 'groupby' not in params:
        if default_groupby:
            # For TOMLs with no analysis/plotting instructions, try to infer from file name
            return 'conc_ng' if 'adk' in str(toml_path) else 'dilution'
        else:
            raise ValueError("TOML file must define basic_stats.groupby = 'capsid' or another parameter.")
    groupby = params['groupby']

    return extras['basic_stats']['groupby'][1] if 'basic_stats' in extras else 'dilution' if 'adk' not in str(toml_path) else 'conc_ng'

def extras_from_meta(meta, plates):
    if 'concat' in meta:
        extras = meta['concat'].extras
    elif len(plates) == 1:
        extras = meta[plates[0]].extras
    else:
        raise NotImplementedError(f"Don't know how to handle {len(plates)} plates without 'concat' in meta")
    return extras