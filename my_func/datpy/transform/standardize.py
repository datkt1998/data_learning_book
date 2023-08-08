#%%
import json
import requests
import numpy as np
from unidecode import unidecode
from string import punctuation
import re 
import tqdm
from difflib import get_close_matches
import pandas as pd
import os
from datpy.helptool.web_action import download_mapping_telco


class address:
    """function với thông tin địa chỉ

    Returns:
        _type_: _description_
    """

    list_province = ['An Giang', 'Bà Rịa - Vũng Tàu', 'Bạc Liêu', 'Bắc Giang', 
    'Bắc Kạn', 'Bắc Ninh', 'Bến Tre', 'Bình Dương', 'Bình Định', 'Bình Phước', 
    'Bình Thuận', 'Cà Mau', 'Cao Bằng', 'Cần Thơ', 'Đà Nẵng', 'Đắk Lắk', 
    'Đắk Nông', 'Điện Biên', 'Đồng Nai', 'Đồng Tháp', 'Gia Lai', 'Hà Giang', 
    'Hà Nam', 'Hà Nội', 'Hà Tĩnh', 'Hải Dương', 'Hải Phòng', 'Hậu Giang', 
    'Hòa Bình', 'Hưng Yên', 'Khánh Hòa', 'Kiên Giang', 'Kon Tum', 'Lai Châu', 
    'Lạng Sơn', 'Lào Cai', 'Lâm Đồng', 'Long An', 'Nam Định', 'Nghệ An', 
    'Ninh Bình', 'Ninh Thuận', 'Phú Thọ', 'Phú Yên', 'Quảng Bình', 'Quảng Nam', 
    'Quảng Ngãi', 'Quảng Ninh', 'Quảng Trị', 'Sóc Trăng', 'Sơn La', 'Tây Ninh', 
    'Thái Bình', 'Thái Nguyên', 'Thanh Hóa', 'Thành phố Hồ Chí Minh', 
    'Thừa Thiên Huế', 'Tiền Giang', 'Trà Vinh', 'Tuyên Quang', 'Vĩnh Long', 
    'Vĩnh Phúc', 'Yên Bái']
    unidecode_province = {unidecode(i).lower():i for i in list_province}
    unidecode_province.update({unidecode(i).lower().replace(' ',""):i for i in list_province})
    unidecode_province.update({'brvt':'Bà Rịa - Vũng Tàu',
        "hn":'Hà Nội','tphn':'Hà Nội','ho chi minh':'Thành phố Hồ Chí Minh',
        'hcm': 'Thành phố Hồ Chí Minh', 'ba ria vung tau': 'Bà Rịa - Vũng Tàu',
        'ba ria': 'Bà Rịa - Vũng Tàu', 'vung tau': 'Bà Rịa - Vũng Tàu','hue':'Thừa Thiên Huế',
        'daklak':'Đắk Lắk', 'daknong':'Đắk Nông', 'tphcm': 'Thành phố Hồ Chí Minh',
        'bac can': 'Bắc Kạn', 'ha tay':'Hà Nội','kontum': 'Kon Tum',
    })
    listkey = list(unidecode_province.keys())

    def __init__(self,):
        pass

    def splitAdress(
        data, 
        key_return=[ "ward_short", "district_short","province_short"], 
        max_data=500,
        if_error= None #VNAddressStandardizer
    ):
        """Phân tách địa chỉ thành các level

        Args:
            data (_type_): dữ liệu địa chỉ hoặc list địa chỉ
            key_return (list, optional): các level muốn get. Defaults to [ "ward_short", "district_short","province_short"].
            max_data (int, optional): số lượng địa chỉ gửi lên tối đa cho 1 request API. Defaults to 500.
            if_error (_type_, optional): kết quả trả Nếu lỗi . Defaults to None#VNAddressStandardizer.

        Returns:
            _type_: _description_
        """
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        res = None

        def return_if_error(x,if_error):
            return if_error(x, comma_handle = False).execute() if if_error is not None else None
        
        def split_1_address(data : str):
                url = "http://192.168.45.45:8030/single_address"
                try:
                    r = requests.post(url, data=json.dumps({"address": data}), headers=headers)
                    res = [json.loads(r.text)["address"][i].title() for i in key_return]
                    assert res != [""]*len(key_return)
                except:
                    try:
                        data = return_if_error(data,if_error)
                        r = requests.post(url, data=json.dumps({"address": data}), headers=headers)
                        res = [json.loads(r.text)["address"][i].title() for i in key_return]
                    except:
                        res=[""]*len(key_return)
                return res
            
        def split_list_address(data : list):
            url = "http://192.168.45.45:8030/address_list"
            sublist_data = [data[x : x + max_data] for x in range(0, len(data), max_data)]
            res = []
            # total_amount = len(data)
            amount=0
            for sublist in sublist_data:
                amount += max_data
                try:
                    response = requests.post(url, data=json.dumps({"address_list": sublist}), headers=headers).json()
                    response_key_return=[[a["address"][i].title()  for i in key_return] if (a["address"] !="") else ([""]*len(key_return)) for a in response]
                except:
                    response_key_return=[split_1_address(add1) for add1 in sublist]
                res += response_key_return
                print("",end="\r")
                # print("Done {}/{}".format(min(amount,total_amount),total_amount),end="")
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
    
    def standardize(data):
        """Chuẩn hóa địa chỉ

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        res = address.splitAdress(data)
        if type(data) == list:
            return list(map(lambda x: ", ".join(x),res))
        if type(data) == str:
            return ", ".join(res)

    def get_province(address_data:str):
        """Xác định tỉnh thành từ địa chỉ

        Args:
            address_data (str): _description_

        Returns:
            _type_: _description_
        """
        if (address_data != address_data) or (address_data is None):
            return np.nan

        def rfind_nth(string, substring=" ", n=1, max_num_char = 3):
            end = string.rfind(substring)
            while end >= 0 and n > 1:
                end = string.rfind(substring,0, end)
                n -= 1
            return " ".join(string[end+1:].split(" ")[:max_num_char])

        add_replacepunc = address_data.translate(str.maketrans(punctuation, ' '*len(punctuation))) #map punctuation to space
        add_clean = unidecode(add_replacepunc).replace("  "," ").strip().lower()
        for count_word_use in range(7):
            check = rfind_nth(add_clean,n = count_word_use)
            for minscore in [1, 0.99, 0.95, 0.9, 0.85]:
                res = get_close_matches(check,address.listkey, n = 1, cutoff=minscore)
                if len(res)>0:
                    return address.unidecode_province[res[0]]  
        # return address.splitAdress(data = address_data, key_return = ["province_short"])[0]
        return np.nan


    # Phan tich Address
    def address_compare_score(A, B):
        """So sánh 2 địa chỉ A và B

        Args:
            A (_type_): _description_
            B (_type_): _description_

        Returns:
            _type_: _description_
        """
        A_split = np.array([unidecode(x) for x in address.splitAdress(A)])
        B_split = np.array([unidecode(x) for x in address.splitAdress(B)])
        score = np.cumprod(A_split == B_split).sum()
        return score


class text:
    """Chuẩn hóa dữ liệu text
    """
    def __init__():
        pass
    
    def encrypt(value):
        url = "http://192.168.45.45:8779/en"
        res = requests.post(url, data=value).text
        return res

    def compare_name(name1,name2):
        if type(name1) != str or type(name2) != str:
            return None
        if name1.isnumeric() or name2.isnumeric():
            return None
        else:
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            url = 'http://192.168.45.45:8030/compare_names'
            pair = json.dumps({"name1" : name1, "name2" : name2})
            x = requests.post(url, data = pair,headers=headers)
            res = json.loads(x.text)['match']
            return res

    def remove_punctuation(str_data):
        """
        Xóa bỏ ký tự đặc biệt và đuôi float .0
        """
        str_data = re.sub(' +', ' ', str(str_data)).strip()
        for e in ['.0','.00',',0',',00']:
            if str_data.endswith(e):
                str_data = str_data[:-len(e)]
                break
        str_data = str_data.translate(str.maketrans('', '', punctuation)).strip()
        return str_data

    def clean(value, space = True, end_dotzero = False, punc = False, nan = True, use_unidecode = False):
        """Convert nan/none value ở dạng string thành NaN

        Returns:
            _type_: _description_
        """
        def space_clean(value):
            return re.sub(' +', ' ', value).strip()

        def end_dotzero_clean(value):
            for e in ['.0','.00',',0',',00']:
                if value.endswith(e):
                    value = value[:-len(e)]
                    break
            return value

        def punctuation_clean(value):
            return value.translate(str.maketrans('', '', punctuation))

        def nan_clean(value):
            value = str(value)
            nan_list = ['', '#n/a', '#n/a n/a', '#na',
                        '#ref!', '(null)', ',', '-1.#ind',
                        '-1.#qnan', '-nan', '.', '1.#ind',
                        '1.#qnan', '<na>', 'n/a', 'na',
                        'nan', 'none', 'null',]
            condi = (value.lower() in nan_list)
            if condi :
                return np.nan
            else:
                return value

        if (value is np.nan) or (value is None) or (nan_clean(value) is np.nan): # None or nan
            return np.nan
        if type(value) == str:
            if space: value = space_clean(value)
            if end_dotzero: value = end_dotzero_clean(value)
            if nan: value = nan_clean(value)
            if punc: value = punctuation_clean(value)
            if use_unidecode: value = unidecode(value)
        return value





class IDcard(text):

    """ 
    Đánh giá một chuỗi có phải là căn cước công dân hợp lệ hay không
    Các giá trị trả về:
    Type        Ý nghĩa                             Mô tả cách phân loại
    1           CMND                                9 ký tự số hợp lệ
    2           CMND 9 số có tỉnh                   9 ký tự số + 3 ký tự chữ ở cuối
    3           CCCD/CMND 12 số                     12 ký tự số hợp lệ
    4           Hộ chiếu VN                         Ký tự chữ ở đầu + 7 ký tự số ở cuối
    5           Hộ chiếu nước ngoài                 Có 8/9 ký tự, bao gồm cả ký tự chữ và số
    6           Nghi ngờ CMND mất 0                 8 ký tự số, khi thêm 0 được CMND hợp lệ
    7           Nghi ngờ CCCD/CMND 12 số mất 0      11 ký tự số, khi thêm 0 được CCCD/CMND 12 số hợp lệ
    8           Không hợp lệ                        Các trường hợp còn lại
    """

    CCCDrule = {
    "001": ["Hà Nội"],
    "002": ["Hà Giang"],
    "004": ["Cao Bằng"],
    "006": ["Bắc Kạn"],
    "008": ["Tuyên Quang"],
    "010": ["Lào Cai"],
    "011": ["Điện Biên"],
    "012": ["Lai Châu"],
    "014": ["Sơn La"],
    "015": ["Yên Bái"],
    "017": ["Hòa Bình"],
    "019": ["Thái Nguyên"],
    "020": ["Lạng Sơn"],
    "022": ["Quảng Ninh"],
    "024": ["Bắc Giang"],
    "025": ["Phú Thọ"],
    "026": ["Vĩnh Phúc"],
    "027": ["Bắc Ninh"],
    "030": ["Hải Dương"],
    "031": ["Hải Phòng"],
    "033": ["Hưng Yên"],
    "034": ["Thái Bình"],
    "035": ["Hà Nam"],
    "036": ["Nam Định"],
    "037": ["Ninh Bình"],
    "038": ["Thanh Hóa"],
    "040": ["Nghệ An"],
    "042": ["Hà Tĩnh"],
    "044": ["Quảng Bình"],
    "045": ["Quảng Trị"],
    "046": ["Thừa Thiên Huế"],
    "048": ["Đà Nẵng"],
    "049": ["Quảng Nam"],
    "051": ["Quảng Ngãi"],
    "052": ["Bình Định"],
    "054": ["Phú Yên"],
    "056": ["Khánh Hòa"],
    "058": ["Ninh Thuận"],
    "060": ["Bình Thuận"],
    "062": ["Kon Tum"],
    "064": ["Gia Lai"],
    "066": ["Đắk Lắk"],
    "067": ["Đắk Nông"],
    "068": ["Lâm Đồng"],
    "070": ["Bình Phước"],
    "072": ["Tây Ninh"],
    "074": ["Bình Dương"],
    "075": ["Đồng Nai"],
    "077": ["Bà Rịa - Vũng Tàu"],
    "079": ["Hồ Chí Minh"],
    "080": ["Long An"],
    "082": ["Tiền Giang"],
    "083": ["Bến Tre"],
    "084": ["Trà Vinh"],
    "086": ["Vĩnh Long"],
    "087": ["Đồng Tháp"],
    "089": ["An Giang"],
    "091": ["Kiên Giang"],
    "092": ["Cần Thơ"],
    "093": ["Hậu Giang"],
    "094": ["Sóc Trăng"],
    "095": ["Bạc Liêu"],
    "096": ["Cà Mau"],
        }
    # Chứng minh nhân dân 9 số
    CMNDrule = {
    "01": ["Hà Nội"],
    "02": ["Hồ Chí Minh"],
    "03": ["Hải Phòng"],
    "04": ["Điện Biên", "Lai Châu"],
    "05": ["Sơn La"],
    "06": ["Lào Cai", "Yên Bái"],
    "07": ["Hà Giang", "Tuyên Quang"],
    "08": ["Lạng Sơn", "Cao Bằng"],
    "090": ["Thái Nguyên"],
    "091": ["Thái Nguyên"],
    "092": ["Thái Nguyên"],
    "095": ["Bắc Kạn"],
    "10": ["Quảng Ninh"],
    "11": ["Hà Tây", "Hòa Bình", "Hà Nội"],
    "12": ["Bắc Giang", "Bắc Ninh"],
    "13": ["Phú Thọ", "Vĩnh Phúc"],
    "14": ["Hải Dương", "Hưng Yên"],
    "15": ["Thái Bình"],
    "16": ["Nam Định", "Hà Nam", "Ninh Bình"],
    "17": ["Thanh Hóa"],
    "18": ["Nghệ An", "Hà Tĩnh"],
    "19": ["Quảng Bình", "Quảng Trị", "Thừa Thiên - Huế"],
    "20": ["Quảng Nam", "Đà Nẵng"],
    "21": ["Quảng Ngãi", "Bình Định"],
    "22": ["Khánh Hòa", "Phú Yên"],
    "230": ["Gia Lai"],
    "231": ["Gia Lai"],
    "23": ["Kon Tum"],
    "24": ["Đắk Lắk"],
    "245": ["Đắk Nông"],
    "25": ["Lâm Đồng"],
    "26": ["Ninh Thuận", "Bình Thuận"],
    "27": ["Đồng Nai", "Bà Rịa - Vũng Tàu"],
    "280": ["Bình Dương"],
    "281": ["Bình Dương"],
    "285": ["Bình Phước"],
    "29": ["Tây Ninh"],
    "30": ["Long An"],
    "31": ["Tiền Giang"],
    "32": ["Bến Tre"],
    "33": ["Vĩnh Long", "Trà Vinh"],
    "34": ["Đồng Tháp"],
    "35": ["An Giang"],
    "36": ["Cần Thơ", "Hậu Giang", "Sóc Trăng"],
    "37": ["Kiên Giang"],
    "38": ["Cà Mau", "Bạc Liêu"],
        }
    typeError = {
    '1': "CMND", 
    '2': "CMND 9 số có tỉnh", 
    '3': "CCCD/CMND 12 số", 
    "4": "Hộ chiếu VN", 
    "5": "Hộ chiếu nước ngoài",
    "6": "Nghi ngờ CMND mất 0", 
    "7": "Nghi ngờ CCCD/CMND 12 số mất 0",
    "8": "Không hợp lệ"
        }
    typeStandard = {
    "1": "PASSPORT_VN", 
    "2": "PASSPORT_NN",
    "3": "CMND_9", 
    "4": "CCCD_12",
    "5": "unknown"
        }

    def __init__(self,value=None):
        self.value = str(value)

    def cleankey(self,error = 'ignore') :
        """Làm sạch input

        Args:
            error (str, optional): _description_. Defaults to 'ignore'.

        Returns:
            _type_: _description_
        """

        try:
            cleaned = text.clean(self.value, True, True, True, True, False).replace(" ","")
            if cleaned is None or cleaned != cleaned:
                return np.nan
            runtime = 0
            while (len([s for s in cleaned if s.isdigit()]) > 12) and (runtime <30):
                runtime+=1
                if cleaned[0] == '0':
                    cleaned = cleaned[1:]
                else:
                    continue

            if runtime == 30:
                raise

            res = cleaned
        except:
            if error == 'ignore':
                res = self.value
            if error == 'nan':
                res = np.nan
        return res


    def typeIDcode(self,):
        """Trả ra mã code của ID theo nhóm
        """
        def isAllNumber(text):
            text = str(text)
            for char in text:
                if not char.isnumeric():
                    return False
            return True 

        def isAllAlpha(text):
            text = str(text)
            for char in text:
                if not char.isalpha():
                    return False 
            return True 

        def is9IDCard(text):
            if text[0:2] in IDcard.CMNDrule.keys() or text[0:3] in IDcard.CMNDrule.keys():
                return True
            else:
                return False

        def is12IDCard(text):
            if text[0:3] not in IDcard.CCCDrule.keys():
                # Sai ma tinh 
                return False 
            elif int(text[3]) not in range(10): # ?
                # print("Sai gioi tinh")
                return False
            elif int(text[3]) > 1 and int(text[4:6]) > 25:
                # sinh năm >= 2000 và năm sinh >= 2025
                # print("Sai nam sinh")
                return False
            return True 
        
        def get_codetypeID(text):
            if isAllAlpha(text):
                # Chỉ chứa ký tự chữ 
                return "8"
            if len(text) == 12:
                if isAllNumber(text):
                    if is12IDCard(text):
                    # "CCCD/CMND 12 số: 12 ký tự số hợp lệ",
                        return '3'
                    else:
                        return '8'
                elif isAllNumber(text[0:9]) and is9IDCard(text[0:9]) and isAllAlpha(text[9:12]):
                    # CMND 9 số có tỉnh: 9 ký tự số + 3 ký tự chữ ở cuối
                    return '2'
                else:
                    return '8'

            elif len(text) == 11 and isAllNumber(text):
                if is12IDCard("0"+text) :
                # Nghi ngờ CCCD/CMND 12 số mất 0: 11 ký tự số, khi thêm 0 được CCCD/CMND 12 số hợp lệ
                    return '7'
                else:
                    return '8'
            elif len(text) == 10 and isAllNumber(text):
                if is12IDCard("00"+text) :
                # Nghi ngờ CCCD/CMND 12 số mất 0: 11 ký tự số, khi thêm 0 được CCCD/CMND 12 số hợp lệ
                    return '7'
                else:
                    return '8'

            elif len(text) == 8:
                if re.match("[a-zA-Z][0-9]{7}", text):
                    # Hộ chiếu VN: Ký tự chữ ở đầu + 7 ký tự số ở cuối
                    return '4'
                elif isAllNumber(text): 
                    if is9IDCard("0" + text) :
                    # Nghi ngờ CMND mất 0: 8 ký tự số, khi thêm 0 được CMND hợp lệ
                        return '6'
                    else:
                        # 8 ký tự số, vẫn sai khi thêm 0
                        return "8"#"6a"
                elif re.match("[a-zA-Z0-9]{8}", text):
                    return "5"#"5a"
                else:
                    return '8'

            elif len(text) == 9:
                if isAllNumber(text):
                    if is9IDCard(text):
                    # CMND: 9 ký tự số hợp lệ
                        return '1'
                    else:
                        # 9 ký tự số, sai mã tỉnh 
                        return "8"#"1a" 
                elif re.match("[a-zA-Z0-9]{9}", text):
                    return "5"#"5b"
                else:
                    return '8'

            else:
                return '8'

        return get_codetypeID(self.cleankey())
    
    def typeIDraw(self):
        """Trả ra type of ID raw

        Returns:
            _type_: _description_
        """
        return IDcard.typeError[self.typeIDcode()]

    def typeIDstandard(self,error = 'ignore', value = None):
        """Trả ra type of ID sau khi chuẩn hóa

        Args:
            error (str, optional): _description_. Defaults to 'ignore'.

        Returns:
            _type_: _description_
        """

        if value is not None:
            if (value != value):
                return value
            elif type(value) == str:
                checkvalue = unidecode(value.lower()).replace(" ","")
                if len([i for i in ['pp','passport', 'hochieu'] if i in checkvalue ])>0:
                    return 'PASSPORT'
                elif len([i for i in ['cccd','cmnd12', 'cc','chip', 'cancuoccongdan'] if i in checkvalue])>0:
                    return IDcard.typeStandard['4']
                elif len([i for i in ['cmnd','cmt','chungminhthu','chungminhnhandan'] if i in value.lower()])>0:
                    return IDcard.typeStandard['3']
            elif error == 'ignore':
                return value
        else:
            cleaned = self.cleankey(error=error)
            if cleaned != cleaned:
                return np.nan
            typeID = self.typeIDcode()
            if ((typeID =='3') and (len(cleaned) == 12)) or (typeID =='7'):
                return IDcard.typeStandard['4']
            elif (typeID in ['1','2','6']) or ((typeID =='3') and (len(cleaned) == 9)):
                return IDcard.typeStandard['3']
            elif ( typeID =='4' ):
                return IDcard.typeStandard['1']
            elif ( typeID =='5' ):
                return IDcard.typeStandard['2']
            else:
                return IDcard.typeStandard['5']

    def standardize(self,error = 'ignore', encrypt = False):
        """Hàm chuẩn hóa ID

        Args:
            error (str, optional): _description_. Defaults to 'ignore'.
            encrypt (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        cleaned = self.cleankey(error=error)
        cleaned = cleaned.upper() if type(cleaned) == str else cleaned
        typeID = self.typeIDcode()
        if (typeID in ['1','3','4','5']):
            res = cleaned
        elif (typeID == '2'):
            res = cleaned[:-3]
        elif (typeID == '6'):
            res = "{:09.0f}".format(int(cleaned))
        elif (typeID == '7'):
            res = "{:012.0f}".format(int(cleaned))
        else:
            res = cleaned

        if encrypt:
            res = text.encrypt(res)

        return res



class Phone(text):
    """Lấy thông tin từ 1 số điện thoại

    Args:
        text (_type_): _description_
    """

    def __init__(self,value = None, dialcode = "84", error = 'ignore', encrypt = False):
        self.value = str(value)
        self.dialcode = dialcode
        self.error = error
        self.is_encrypt = encrypt
        self.chuyendausodidong = [
            ['162','32'],['163','33'],['164','34'],['165','35'],['166','36'],['167','37'],['168','38'],['169','39'],
            ['120','70'],['121','79'],['122','77'],['126','76'],['128','78'],['123','83'],['124','84'],['125','85'],
            ['127','81'],['129','82'],['182','52'],['186','56'],['188','58'],['199','59']
            ]
        self.chuyendausocodinh = [
            ['76', '296'], ['64', '254'], ['281', '209'], ['240', '204'], ['781', '291'], ['241', '222'], ['75', '275'],
            ['56', '256'], ['650', '274'], ['651', '271'], ['62', '252'], ['780', '290'], ['710', '292'], ['26', '206'],
            ['511', '236'], ['500', '262'], ['501', '261'], ['230', '215'], ['61', '251'], ['67', '277'], ['59', '269'],
            ['219', '219'], ['351', '226'], ['4', '24'], ['39', '239'], ['320', '220'], ['31', '225'], ['711', '293'],
            ['8', '28'], ['218', '218'], ['321', '221'], ['8', '258'], ['77', '297'], ['60', '260'], ['231', '213'],
            ['63', '263'], ['25', '205'], ['20', '214'], ['72', '272'], ['350', '228'], ['38', '238'], ['30', '229'],
            ['68', '259'], ['210', '210'], ['57', '257'], ['52', '232'], ['510', '235'], ['55', '255'], ['33', '203'],
            ['53', '233'], ['79', '299'], ['22', '212'], ['66', '276'], ['36', '227'], ['280', '208'], ['37', '237'],
            ['54', '234'], ['73', '273'], ['74', '294'], ['27', '207'], ['70', '270'], ['211', '211'], ['29', '216'],
        ]
        self.dausodidong = [i[1] for i in self.chuyendausodidong] + \
                            ['86','96','97','98','88','91','94','89','90','93','92','56','58','99','87']
        self.dausocodinh = [i[1] for i in self.chuyendausocodinh]
        # self.typephone = 'unknown'
        self.cleaned = self.standardize()

    def cleankey(self):
        """Hàm làm sạch input

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        self.typephone = 'unknown'
        dialcode = self.dialcode
        def isAllNumber(text):
            for char in text:
                if not char.isnumeric():
                    return False
            return True 
        cleaned =  text.clean(self.value, True, True, True, True, False).replace(" ","")
        assert isAllNumber(cleaned)

        runtime = 0

        while (len(cleaned) >= 9) and (runtime<30) and (self.typephone == 'unknown'):

            cleaned = cleaned[5:] if bool(re.match('8400[0-9]{1,1}',cleaned[:5])) and (len(cleaned)>=14) else cleaned
            cleaned = cleaned[1:] if cleaned[0] == '0' else cleaned
            if ((cleaned[:len(dialcode)] == dialcode) and len(cleaned)>=(9+len(dialcode))):
                cleaned = cleaned[len(dialcode):]

            if (len(cleaned) == 9) and (self.typephone == 'unknown'):
                for i in self.dausodidong:
                    if cleaned.startswith(i):
                        self.typephone = 'so_di_dong'
                        break

            if (len(cleaned) in [9,10]) and (self.typephone == 'unknown'):
                for i in self.dausocodinh:
                    if cleaned.startswith(i) and (len(cleaned[len(i):]) in [7,8]): # ví du 0220 3736 596 (HD), 024 392 63087 (HN)
                        self.typephone = 'so_co_dinh'
                        break
                    

            if (len(cleaned) == 10) and (self.typephone == 'unknown'):
                for pair in self.chuyendausodidong:
                    if cleaned.startswith(pair[0]):
                        cleaned = cleaned.replace(pair[0],pair[1],1)
                        self.typephone = 'so_di_dong'
                        break

            if (len(cleaned) in [9,10]) and (self.typephone == 'unknown'):
                for pair in self.chuyendausocodinh:
                    if cleaned.startswith(pair[0]) and (len(cleaned[len(pair[0]):]) in [7,8]): # ví du 0220 3736 596 (HD), 024 392 63087 (HN)
                        cleaned = cleaned.replace(pair[0],pair[1],1)
                        self.typephone = 'so_co_dinh'
                        break

            runtime+=1

        if runtime == 30:
            raise ValueError
        else:
            return cleaned

    def standardize(self):
        """Hàm chuẩn hóa SĐT

        Returns:
            _type_: _description_
        """
        try:
            res = self.dialcode + self.cleankey()
        except:
            if self.error == 'ignore':
                res = self.value
            if self.error == 'nan':
                res =  np.nan
        if self.is_encrypt:
            res = text.encrypt(res)
        return res

    def pay_type(paytype):
        """Hàm chuẩn hóa thông tin paytype

        Args:
            paytype (_type_): _description_

        Returns:
            _type_: _description_
        """
        if paytype != paytype:
            return np.nan
        paytype = text.clean(str(paytype), True, True, True, True, True).lower()
        if len([i for i in ['pre','truoc','tt'] if i in paytype])>0:
            return 'prepaid'
        elif len([i for i in ['post','sau','ts'] if i in paytype])>0:
            return 'postpaid'
        else :
            return np.nan

    def map_telco(list_of_phonenumbers, show = False):
        """Hàm get thông tin nhà mạng từ 1 list SĐT từ web brandname

        Args:
            list_of_phonenumbers (_type_): _description_
        """
        def merge_mapping_telco( filedir ):
            mapped = pd.DataFrame()
            with pd.ExcelFile(filedir) as excelFile:
                for telco in excelFile.sheet_names:
                    if telco.lower() not in ['sai số','trùng số']:
                        sheetfile = excelFile.parse(telco)
                        sheetfile['TELCO'] = telco
                        mapped = pd.concat([mapped,sheetfile],ignore_index=True)
            return mapped

        download_location = os.getcwd()
        data = pd.DataFrame(list_of_phonenumbers, columns = ['RAW']).drop_duplicates()
        data['MSISDN'] = data['RAW'].map(lambda x: Phone(x).standardize())
        data[['MSISDN']].drop_duplicates().to_excel(os.path.join(download_location,'map_telco.xlsx'),index=False)
        download_mapping_telco( download_location = download_location , show = show)
        downloadedfile = os.path.join(download_location,
        [i for i in os.listdir(download_location) if i.startswith('DataTelco_')][0])
        mapped = merge_mapping_telco(downloadedfile)
        mapped['MSISDN'] = mapped['MSISDN'].map(lambda x: Phone(x).standardize())
        data_res = data.merge(mapped,how='left')
        os.remove(os.path.join(download_location,'map_telco.xlsx'))
        os.remove(downloadedfile)
        return data_res

class Person:
    """Chuẩn hóa thông tin của người
    """
    def __init__(self):
        pass

    def gender_standardize(gender):
        """Chuẩn hóa giới tính"""
        if gender != gender:
            return np.nan
        gender = text.clean(str(gender), True, True, True, True, True).lower()
        if gender in ['m', 'male','nam']:
            return 'M'
        elif gender in ['f', 'female','nu']:
            return 'F'
        else:
            return np.nan
    
    def birthyear_standardize(birthyear):
        """Chuẩn hóa năm sinh

        Args:
            birthyear (_type_): _description_

        Returns:
            _type_: _description_
        """
        res = np.nan
        try:
            str_birthyear = text.clean(str(birthyear), True, True, True, True, False).lower()
            if re.match("^[0-9]{4,4}$",str_birthyear):
                if (int(str_birthyear) > 1930) and (int(str_birthyear) < 2030):
                    return str_birthyear
            return res
        except:
            return res


#%%
# Phone("8484919891000").standardize()
#%%

class SI(text):
    """Chuẩn hóa mã số BHXH

    Args:
        text (_type_): _description_
    """
    def __init__(self,value):
        self.value = str(value)

    def cleankey(self):

        def isAllNumber(text):
            for char in text:
                if not char.isnumeric():
                    return False
            return True

        def returnNumberSI(text):
            text = "{:010.0f}".format(int(text))
            if len(text) > 10:
                raise ValueError
            return text 


        cleaned = text.clean(self.value, True, True, True, True, False).replace(" ","")

        if isAllNumber(cleaned):
            cleaned = returnNumberSI(cleaned)
            return cleaned
        
        if re.match("0*[0-9]{7,10}[a-zA-Z]{0,3}$", cleaned):
            alpha_len = len([i for i in cleaned if i.isalpha()])
            number_comp = returnNumberSI(cleaned[:-alpha_len])
            cleaned = number_comp + cleaned[-alpha_len:]
            return cleaned
        else:
            raise

    def standardize(self,error = 'ignore', encrypt = False):
        try:
            res =  self.cleankey().upper()
        except:
            if error == 'ignore':
                res = self.value
            if error == 'nan':
                res =  np.nan

        if encrypt:
            res = text.encrypt(res)
        return res       



class TaxID(text):
    def __init__(self,value):
        self.value = np.nan if ((type(value) == dict) or (value != value) or (value is None)) else str(value)
        self.type = np.nan
        self.cleaned = value if (value != value) else self.taxID_clean() 

    def taxID_clean(self, value_input=None):
        value = self.value if value_input is None else value_input
        if re.match("0*[0-9]{8,10}(-[0-9]{1,3})?$", value):
            res = re.findall("[0-9]{8,10}-[0-9]{1,3}$", value)
            if len(res) == 0:
                res = re.findall("[0-9]{8,10}$", value)
                if len(res) == 0:
                    raise ValueError
                else:
                    self.type = 'dv_chu_quan'
                    return "{:010.0f}".format(int(res[0]))
            else:
                self.type = 'dv_chi_nhanh'
                return "{:010.0f}-{:03.0f}".format(int(res[0].split("-")[0]), int(res[0].split("-")[1]))
        else:
            if value_input is None:
                value_clean = text.clean(self.value, True, True, True, True, False).replace(" ","").replace('.','').replace(',','')
                return self.taxID_clean(value_clean)
            else :
                return self.clean()