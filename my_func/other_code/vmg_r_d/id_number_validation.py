import re 


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
    if text[0:2] in CMND_province_code_and_name.keys() or text[0:3] in CMND_province_code_and_name.keys():
        return True
    else:
        return False

def is12IDCard(text):
    if text[0:3] not in province_code_and_name_rule.keys():
        # Sai ma tinh 
        return False 
    elif int(text[3]) not in range(10): # ?
        # print("Sai gioi tinh")
        return False 
    elif int(text[3]) > 1 and int(text[4:6]) > 21:
        # print("Sai nam sinh")
        return False
    return True 

# CMND 12 so 
province_code_and_name_rule = {
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
CMND_province_code_and_name = {
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

def validateIdCardNumber(text):
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
    text = str(text)
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
    elif len(text) == 8:
        if re.match("[a-zA-Z][0-9]{7}", text):
            # Hộ chiếu VN: Ký tự chữ ở đầu + 7 ký tự số ở cuối
            return '4'
        elif isAllNumber(text): 
            if is9IDCard("0" + text) or is9IDCard(text + "0"):
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
    elif len(text) == 11 and isAllNumber(text):
        if is12IDCard("0"+text) or is12IDCard(text + "0"):
        # Nghi ngờ CCCD/CMND 12 số mất 0: 11 ký tự số, khi thêm 0 được CCCD/CMND 12 số hợp lệ
            return '7'
        else:
            return '8'
    else:
        return '8'

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

