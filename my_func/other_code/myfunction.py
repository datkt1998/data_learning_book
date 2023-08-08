
# from sys import path
# path.append(r"C:\Users\datatech2\Desktop\Dat")
# path.append(r"E:\Sharing Folder")
# from Function_data_analyst import *

# from vnaddress import VNAddressStandardizer

from sys import path,modules
import numpy as np
import pandas as pd
from unidecode import unidecode
from seaborn import displot,histplot
import cx_Oracle
import json
import requests
import matplotlib.pyplot as plt

#connection
def OutputTypeHandler(cursor, name, defaultType, size, precision, scale):
    if defaultType == cx_Oracle.CLOB:
        return cursor.var(cx_Oracle.LONG_STRING, arraysize = cursor.arraysize)
    elif defaultType == cx_Oracle.BLOB:
        return cursor.var(cx_Oracle.LONG_BINARY, arraysize = cursor.arraysize)

def read_data_from_db( db_con , table_name: str = None , col_name = "*" ,
    offset_rows : int = 0, n_records : int = -1, chunksize : int = None  , sql_query : str=None, thread_num : int = 0):

    error_exception = None
    n_records = "all" if n_records==-1 else n_records
    
    if type(db_con)==cx_Oracle.Connection:
        db_con.outputtypehandler = OutputTypeHandler
        offset_clause= " offset {} rows ".format(offset_rows)
        num_records_clause = "" if n_records == "all" else " fetch next {} rows only".format(n_records)
        combine_clause = offset_clause + num_records_clause

    else: # sqlite3
        offset_clause= "" if offset_rows==0 else " offset {} ".format(offset_rows)
        num_records_clause = "limit -1" if n_records == "all" else " limit {} ".format(n_records)
        combine_clause =  num_records_clause + offset_clause

    if sql_query is None:
        cols=col_name if type(col_name)==str else ", ".join(col_name)
        sql_query="""
        select {} from {} {}
        """.format(cols,table_name,combine_clause)

    if "DROP TABLE" in sql_query:
        pd.read_sql_query(sql=sql_query, con=db_con)
        return None 

    res=pd.read_sql_query(sql=sql_query, con=db_con,chunksize=chunksize)
    print("Bảng {} offset {} dòng, {} records".format(table_name,offset_rows,n_records) + ("" if chunksize is None else ", chunksize {}".format(chunksize)))
    return res


# API
# encode/decode encrypt
def code_crypt(data, t_crypt="ec",max_data=50000):
    headers = {"Content-Type": "application/json"}
    type_crypt = { "dc": "http://192.168.45.45:8779/dc",
                  "ec": "http://192.168.45.45:8779/en",
                 "dcs": "http://192.168.45.45:8779/dcs",
                 "ecs": "http://192.168.45.45:8779/ens"}
    if t_crypt not in ["ec","dc"]:
        print("nhap sai t_crypt !")
    elif type(data)==str :
        return requests.post(type_crypt[t_crypt], data=data).text
    elif type(data)==list : 
        t_crypt = t_crypt+"s"
        sublist_data=[data[x:x+max_data] for x in range(0, len(data), max_data)]
        res=[]
        for sublist in sublist_data:
            data_request =  {"lisd": sublist}
            request_res = requests.post(type_crypt[t_crypt], json=data_request, headers=headers).json()
            res += request_res
        return res
    else:
        print("nhap sai dang du lieu data !")

# from vnaddress import VNAddressStandardizer

def split_address_api(
    data, key_return=["province_short", "district_short", "ward_short"], max_data=200,if_error= None #VNAddressStandardizer
):
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    res = None

    def return_if_error(x,if_error):
        return if_error(x, comma_handle = False).execute() if if_error is not None else None
    
    def split_1_address(data : str):
            url = "http://192.168.45.45:8030/single_address"
            try:
                r = requests.post(url, data=json.dumps({"address": data}), headers=headers)
                res = [json.loads(r.text)["address"][i] for i in key_return]
                assert res != [""]*len(key_return)
            except:
                try:
                    data = return_if_error(data,if_error)
                    r = requests.post(url, data=json.dumps({"address": data}), headers=headers)
                    res = [json.loads(r.text)["address"][i] for i in key_return]
                except:
                    res=[""]*len(key_return)
            return res
        
    def split_list_address(data : list):
        url = "http://192.168.45.45:8030/address_list"
        sublist_data = [data[x : x + max_data] for x in range(0, len(data), max_data)]
        res = []
        total_amount = len(data)
        amount=0
        for sublist in sublist_data:
            amount += max_data
            try:
                response = requests.post(url, data=json.dumps({"address_list": sublist}), headers=headers).json()
                response_key_return=[[a["address"][i]  for i in key_return] if (a["address"] !="") else ([""]*len(key_return)) for a in response]
            except:
                response_key_return=[split_1_address(add1) for add1 in sublist]
            res += response_key_return
            print("",end="\r")
            print("Done {}/{}".format(amount,total_amount),end="")
        return res
    
    try:
        if (data != data) or (data==""):
            return None
        if type(data) == str:
            res = split_1_address(data)
        elif type(data) == list:
            res = split_list_address(data)
        else:
            print("Sai type of data !")
        return res
    except:
        raise

# Phan tich Address
def address_compare_score(A, B, A_split_yet=False, B_split_yet=False):
    unidecode_f = np.vectorize(lambda x: unidecode(x))
    A_split = unidecode_f(np.array(split_address_api(A) if A_split_yet == False else A))
    B_split = unidecode_f(np.array(split_address_api(B) if B_split_yet == False else B))
    ar = A_split == B_split
    axis = None if len(ar.shape) == 1 else 1
    score = np.cumprod(ar, axis=axis).sum(axis=1)
    return score

# nhận diện và standardize address
# from vnaddress import VNAddressStandardizer
# address = VNAddressStandardizer(raw_address = "Q1", comma_handle = True)
# address.execute()

# AddressCorrection
# from sys import path,modules
# path.append(r"E:\Sharing Folder\my-modules\VN-address-correction")
# from address_correction import AddressCorrection
# address_correction = AddressCorrection()
# address_correction.address_correction('o cho dua dong da ha noi')





# STANDARDIZE

def ID_standardize(x):
    try:
        x = str(x).split(".")[0].replace(",","").replace(" ","")
        res =  "{:012.0f}".format(float(x)) if (type(x)==str and x.isnumeric()) else x
        res = res.replace("000","",1) if len(x)<=9 and res.startswith("000") else res
    except:
        res = x
    return res

def SI_standardize(x):
    try:
        x = str(x).split(".")[0].replace(",","").replace(" ","")
        res =  "{:010.0f}".format(float(x)) if (type(x)==str and x.isnumeric()) else x
    except:
        res = x
    return res

def Phone_standardize(x):
    DAUSO = {'162':'32','163':'33','164':'34','165':'35','166':'36','167':'37','168':'38',
             '169':'39','120':'70','121':'79','122':'77','126':'76','128':'78','123':'83',
             '124':'84','125':'85','127':'81','129':'82','186':'56','188':'58','199':'59'}
    
    str_x = str(x).split(".")[0].replace(",","").replace(" ","")
    if str_x.isnumeric() and len(str_x)>= 9:
        if len(str_x)>=10 and str_x.startswith("0"):
            str_x=str_x.replace("0","",1)
        elif len(str_x)>=11 and str_x.startswith("84"):
            str_x=str_x.replace("84","",1)
        else:
            str_x = str_x
            
    if len(str_x) == 10:
        for key in DAUSO.keys():
            if str_x.startswith(key):
                str_x = str_x.replace(key,DAUSO[key],1)
                break
    if len(str_x)==9:
        return "0"+str_x
    else:
        return np.nan

# CLEAN DATA

def str_clean(x):
    if type(x)==str:
        list_nan_encrypt =[
        "NqH2T0OqGhYeJKBEFFzNmg==","WT0GPRKcQxiTXjDpOncVhw==","8zeFSJ6JoUFhmf9cA1NRDg==",
        "JTejTlFoNwzfCBJMcR0HiQ==","V+0CZRxDddEVQw2rxwmiwA==","lVKQ/5YMX3x+hOdbtW0F1w==",
        "Yh9tPt/GcByge8grvB8R/w==",
    ]
        x=x.strip()
        if (x=="") or (x.lower()=="null") or (x.lower()=="nan") or (x in list_nan_encrypt):
            return np.nan
        else:
            return x
    elif x != x: # None or nan
        return np.nan
    else:
        return x

def DFCleaner(df_raw: pd.DataFrame):
    df_raw=df_raw.applymap(str_clean)
    return df_raw

def convert_dt(df_raw: pd.DataFrame, cols_list: list, inplace: bool=False):
    if inplace==False:
        df_raw=df_raw.copy()
    df_raw[cols_list] = df_raw[cols_list].applymap(lambda x: pd.to_datetime(x ,format="%Y%m%d"))
    if inplace==False:
        return df_raw

def reformat_name(df_raw: pd.DataFrame, col_name: str , inplace:bool=False):
    if inplace==False:
        df_raw=df_raw.copy()
    df_raw[col_name]=df_raw[col_name].map(lambda x:unidecode(x).replace(" ","").lower() if type(x)==str else np.nan)
    if inplace==False:
        return df_raw



# ANALYST DATA
def nan_and_unique(df_raw: pd.DataFrame):
    """
    Return each columns in table: count the not-nan, the unique value, the not-nan/total rate, the unique/not_nan rate
    """
    total_row=pd.Series(df_raw.shape[0],index=df_raw.columns, name="TOTAL_ROW")
    count_nan=pd.Series(df_raw.isna().sum(), name="COUNT_NAN")
    count_not_nan = (total_row - count_nan).rename("COUNT_NOT_NAN")
    count_unique = df_raw.nunique().rename("COUNT_UNIQUE")
    count_duplicate= (count_not_nan - count_unique).rename("COUNT_DUPLICATED")
    rate_not_nan = (count_not_nan/ (df_raw.shape[0])).rename("NOT_NAN_ON_TOTAL_RATE")
    rate_unique_on_not_nan = pd.Series(count_unique.values/(count_not_nan+1e-50).values,
                                            index=count_not_nan.index,name="UNIQUE_ON_NOT_NAN_RATE")
    rate_duplicated_on_not_nan = (1-rate_unique_on_not_nan).rename('DUPLICATED_ON_NOT_NAN_RATE')
    res = pd.concat([total_row,count_nan,count_not_nan,count_unique,rate_not_nan,rate_unique_on_not_nan,rate_duplicated_on_not_nan],axis=1)
    return res.convert_dtypes()

def check_unique_multivalues(df_raw: pd.DataFrame, key: str, key_compare_list: list=[] ):
    """
    Return all rows have key is duplicated but other values are difference.
    """
    key_compare_list = [i for i in df_raw.columns if  i != key] if key_compare_list == [] else key_compare_list
    table=df_raw[[key]+key_compare_list].drop_duplicates()
    if table[key].is_unique:
        print("Khong co duplicate key and difference other values")
    else:
        res = table[table.duplicated(key,keep=False)]
        no_key_duplicated = res[key].nunique()
        print("Cap {} co {} value bi duplicated".format("-".join(res.columns), no_key_duplicated))
        return res.sort_values([key]+ key_compare_list)
    
def save_xlsx(list_dfs, xls_path):
    with pd.ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer,'sheet%s' % n)
        writer.save()
        writer.close()


def print_df_to_pdf(df,pdf):
    fig, ax =plt.subplots(figsize=(12,4))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='center')
    pdf.savefig(fig, bbox_inches='tight')


def cut_off_percentile_dis(df,cols_cutoff:list,lower:float=0.25, upper:float=0.75,IQR=True):
    lower_band=df[cols_cutoff].quantile(lower)
    upper_band=df[cols_cutoff].quantile(upper)
    IQR = (upper_band - lower_band) if IQR else 0
    df_new= df[~((df[cols_cutoff] < (lower_band - 1.5 * IQR)) |(df[cols_cutoff] > (upper_band + 1.5 * IQR))).any(axis=1)]
    print("""Remove {remove_count}/{total_count} rows ({remove_rate}%), Remain {remain_count}/{total_count} rows ({remain_rate}%)
    """.format(remove_count= len(df)-len(df_new),remove_rate= round((len(df)-len(df_new))/len(df),4)*100 ,
    remain_count= len(df_new), remain_rate= round(len(df_new)/len(df),4)*100, total_count= len(df)))
    return df_new


def plot_histogram_density(sr: pd.Series, name_t: str="", bin_range: tuple=None):
    var=sr.name
    if (bin_range is None)==False:
        sr=sr.loc[(sr >= sr.quantile(bin_range[0])) & (sr <= sr.quantile(bin_range[1]))]
    histplot(sr,kde=True)
    plt.title("_".join([name_t,var]))
#     plt.savefig("_".join([name_t,var])+".png")
    plt.show()
#     return fig
#     pp.save


def quantile_statistic(sr: pd.Series, name_t: str="", bin_range: tuple=None):
    global sr2
    if (bin_range is None)==False:
        sr2=sr.loc[(sr >= sr.quantile(bin_range[0])) & (sr <= sr.quantile(bin_range[1]))].copy()
    else:
        sr2=sr.copy()
    var=sr2.name
    func=["quantile({})".format(i) for i in [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]] +["max()","min()","mean()","median()","mode().loc[0]"]
    return pd.DataFrame([[eval("sr2.{}".format(i)) for i in func]],columns=func,index=["_".join([name_t,var])])
    
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

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

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

        psi_value = np.sum([sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))])

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)