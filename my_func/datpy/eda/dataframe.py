import numpy as np
import pandas as pd
from seaborn import histplot
import matplotlib.pyplot as plt


def display(df, max_rows=300, max_columns=100):
    from IPython.display import display
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_columns):
        display(df)  # need display to show the dataframe when using with in jupyter
        # some pandas stuff

# ANALYST DATA


def nan_and_unique(df_raw: pd.DataFrame):
    """
    Return each columns in table: count the not-nan, the unique value, the not-nan/total rate, the unique/not_nan rate
    """
    total_row = pd.Series(
        df_raw.shape[0], index=df_raw.columns, name="TOTAL_ROW")
    count_nan = pd.Series(df_raw.isna().sum(), name="COUNT_NAN")
    count_not_nan = (total_row - count_nan).rename("COUNT_NOT_NAN")
    count_unique = df_raw.nunique().rename("COUNT_UNIQUE")
    count_duplicate = (count_not_nan - count_unique).rename("COUNT_DUPLICATED")
    rate_not_nan = (
        count_not_nan / (df_raw.shape[0])).rename("NOT_NAN_ON_TOTAL_RATE")
    rate_unique_on_not_nan = pd.Series(count_unique.values/(count_not_nan+1e-50).values,
                                       index=count_not_nan.index, name="UNIQUE_ON_NOT_NAN_RATE")
    rate_duplicated_on_not_nan = (
        1-rate_unique_on_not_nan).rename('DUPLICATED_ON_NOT_NAN_RATE')
    res = pd.concat([total_row, count_nan, count_not_nan, count_unique,
                    rate_not_nan, rate_unique_on_not_nan, rate_duplicated_on_not_nan], axis=1)
    return res.convert_dtypes()


def check_unique_multivalues(df_raw: pd.DataFrame, key: str, key_compare_list: list = [], key_exception: list = []):
    """
    Return all rows have key is duplicated but other values are difference.
    """
    key_compare_list = [i for i in df_raw.columns if i !=
                        key] if key_compare_list == [] else key_compare_list
    table = df_raw[[key]+key_compare_list].drop_duplicates()
    if table[key].is_unique:
        print("Khong co duplicate key and difference other values")
    else:
        res = table[(table.duplicated(key, keep=False)) & (
            table[key].isin(key_exception) == False) & (table[key].isna() == False)]
        no_key_duplicated = res[key].nunique()
        print("Cap {} co {} value bi duplicated".format(
            "-".join(res.columns), no_key_duplicated))
        return res.sort_values([key] + key_compare_list)


def save_xlsx(list_dfs, xls_path):
    with pd.ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer, 'sheet%s' % n)
        writer.save()
        writer.close()


def print_df_to_pdf(df, pdf):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values,
                         colLabels=df.columns, loc='center')
    pdf.savefig(fig, bbox_inches='tight')


def cut_off_percentile_dis(df, cols_cutoff: list, lower: float = 0.25, upper: float = 0.75, IQR=True):
    lower_band = df[cols_cutoff].quantile(lower)
    upper_band = df[cols_cutoff].quantile(upper)
    IQR = (upper_band - lower_band) if IQR else 0
    df_new = df[~((df[cols_cutoff] < (lower_band - 1.5 * IQR)) |
                  (df[cols_cutoff] > (upper_band + 1.5 * IQR))).any(axis=1)]
    print("""Remove {remove_count}/{total_count} rows ({remove_rate}%), Remain {remain_count}/{total_count} rows ({remain_rate}%)
    """.format(remove_count=len(df)-len(df_new), remove_rate=round((len(df)-len(df_new))/len(df), 4)*100,
               remain_count=len(df_new), remain_rate=round(len(df_new)/len(df), 4)*100, total_count=len(df)))
    return df_new


def plot_histogram_density(sr: pd.Series, name_t: str = "", bin_range: tuple = None):
    var = sr.name
    if (bin_range is None) == False:
        sr = sr.loc[(sr >= sr.quantile(bin_range[0])) &
                    (sr <= sr.quantile(bin_range[1]))]
    histplot(sr, kde=True)
    plt.title("_".join([name_t, var]))
#     plt.savefig("_".join([name_t,var])+".png")
    plt.show()
#     return fig
#     pp.save


def quantile_statistic(sr: pd.Series, name_t: str = "", bin_range: tuple = None):
    global sr2
    if (bin_range is None) == False:
        sr2 = sr.loc[(sr >= sr.quantile(bin_range[0])) & (
            sr <= sr.quantile(bin_range[1]))].copy()
    else:
        sr2 = sr.copy()
    var = sr2.name
    func = ["quantile({})".format(i) for i in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
                                               0.9, 0.95, 0.99]] + ["max()", "min()", "mean()", "median()", "mode().loc[0]"]
    return pd.DataFrame([[eval("sr2.{}".format(i)) for i in func]], columns=func, index=["_".join([name_t, var])])

# tinh day du - do on dinh cua phan phoi qua thoi gian


def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(
                expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack(
                [np.percentile(expected_array, b) for b in breakpoints])

        expected_percents = np.histogram(expected_array, breakpoints)[
            0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[
            0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum([sub_psi(expected_percents[i], actual_percents[i])
                           for i in range(0, len(expected_percents))])

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:, i], actual[:, i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i, :], actual[i, :], buckets)

    return(psi_values)
