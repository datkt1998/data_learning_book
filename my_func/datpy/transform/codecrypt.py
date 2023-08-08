import requests
import tqdm
import numpy as np

def cryptlist(data:list,type_crypt :str,headers, max_data=50000,use_tqdm = False):
    sublist_data=[data[x:x+max_data] for x in range(0, len(data), max_data)]
    res=[]
    loop_data = tqdm(sublist_data) if use_tqdm else sublist_data
    for sublist in loop_data:
        data_request =  {"lisd": sublist}
        request_res = requests.post(type_crypt, json=data_request, headers=headers).json()
        res += request_res
    return res

def encrypt(data, max_data=50000,use_tqdm = False):
    headers = {"Content-Type": "application/json"}
    type_crypt = {"ec": "http://192.168.45.45:8779/en",
                 "ecs": "http://192.168.45.45:8779/ens"}

    if type(data)==str :
        return requests.post(type_crypt['ec'], data=data).text
    elif type(data)==list :
        data = [i if i == i else '' for i in data] # replace nan
        res = cryptlist(data,type_crypt['ecs'],headers, max_data=max_data,use_tqdm = use_tqdm)
        res = list(np.where(np.array(data) != "",res, None))
        return res
    else:
        return None

def decrypt(data, max_data=50000,use_tqdm = False):
    headers = {"Content-Type": "application/json"}
    type_crypt = {"dc": "http://192.168.45.45:8779/dc",
                 "dcs": "http://192.168.45.45:8779/dcs"}

    if type(data)==str :
        return requests.post(type_crypt['dc'], data=data).text
    elif type(data)==list :
        data = [i if i == i else 'NqH2T0OqGhYeJKBEFFzNmg==' for i in data] # replace nan
        res = cryptlist(data,type_crypt['dcs'],headers, max_data=max_data,use_tqdm = use_tqdm)
        return res
    else:
        return None

def encrypt_df(df, *cols):
    for col in cols :
        if type(col)==str :
            df[col] = encrypt(df[col].tolist())
        elif type(col) == list:
            for subcol in col:
                df[subcol] = encrypt(df[subcol].tolist())

def decrypt_df(df, *cols):
    for col in cols :
        if type(col)==str :
            df[col] = decrypt(df[col].tolist())
        elif type(col) == list:
            for subcol in col:
                df[subcol] = decrypt(df[subcol].tolist())