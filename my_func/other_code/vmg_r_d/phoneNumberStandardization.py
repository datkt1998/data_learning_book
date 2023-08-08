from re import X
from string import punctuation
import numpy as np
import pymongo

from helpers.helper import use_crypt_star_list

myclient = pymongo.MongoClient("mongodb://192.168.18.50:27017/")
mydb = myclient["phone_check"]
mycol = mydb["phone_check"]

def remove_punctuation(str_data):
    str_data = str(str_data).strip()
    
    for e in ['.0','.00',',0',',00']:
        if str_data.endswith(e):
            str_data = str_data[:-len(e)]
            break
    str_data = str_data.translate(str.maketrans('', '', punctuation))
    return str_data

def Phone_standardize(filter_list, field, des):   
    DAUSO = {'162':'32','163':'33','164':'34','165':'35','166':'36',
             '167':'37','168':'38','169':'39','120':'70','121':'79',
             '122':'77','126':'76','128':'78','123':'83','124':'84',
             '125':'85','127':'81','129':'82','182':'52','186':'56',
             '188':'58','199':'59',
             "76": "296", "64": "254", "281": "209",
             '240':'204','781':'291', "241": "222", "75": "275", "56": "256",
             '650':'274','651':'271', "62": "252", "780": "290", "710": "292",
             '26':'206','511':'236', "500": "262", "501": "261", "230": "215",
             '61':'251','67':'277', "59": "269", "219": "219", "351": "226",
             '4':'24','39':'239', "320": "220", "31": "225", "711": "293",
             '8':'28','218':'218', "321": "221", "8": "258", "77": "297",
             '60':'260','231':'213', "63": "263", "25": "205", "20": "214",
             '72':'272','350':'228', "38": "238", "30": "229", "68": "259",
             '210':'210','57':'257', "52": "232", "510": "235", "55": "255",
             '33':'203','53':'233', "79": "299", "22": "212", "66": "276",
             '36':'227','280':'208', "37": "237", "54": "234", "73": "273",
             '74':'294','27':'207', "70": "270", "211": "211", "29": "216"
             }
    DAUSODTB = {
            "296", "254", "209", '204','291',"222","275", "256",
             '274', '271', "252", "290", "292",
             '206','236', "262", "261", "215",
             '251','277', "269", "219", "226",
             '239', "220","225","293",
             '218', "221", "258", "297",
             '260','213',"263","205","214",
             '272','228',"238","229","259",
             '210','257',"232","235","255",
             '203','233',"299","212","276",
             '227','208',"237","234","273",
             '294','207',"270","211","216",
    }

    finalList = []
    
    encrypt_phone_list = [row[field] for row in filter_list]

    if des: encrypt_phone_list = use_crypt_star_list(encrypt_phone_list, t_crypt="dcs")

    for index, x in enumerate(encrypt_phone_list):
        str_x = remove_punctuation(x).replace(" ","")
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
            # return "84"+str_x
            finalList.append("84"+str_x)
        else:
            if len(str_x) == 10:
                    i = 0
                    for dauso in DAUSODTB:
                        if str_x.startswith(dauso):
                            i = 1
                            # file = open("mobidtb.txt", "a")
                            # file.write("84"+str_x + "   " + filter_list[index]["ID_NO"] + "\n")
                            # file.close
                            finalList.append("84"+str_x)
                            break
                    if i==0:
                        # file = open("mobi.txt", "a")
                        # file.write(str_x + "   " + filter_list[index]["ID_NO"] + "\n")
                        # file.close
                        finalList.append(None)
            else:
                #if str_x is not None and str_x != "None":
                    # file = open("mobi.txt", "a")
                    # file.write(str_x + "   " + filter_list[index]["ID_NO"] + "\n")
                    # file.close
                finalList.append(None)
            
    finalListEnc = use_crypt_star_list(finalList, t_crypt="ens")

    for index, x in enumerate(finalListEnc):
        filter_list[index][field] = x
    return(filter_list)