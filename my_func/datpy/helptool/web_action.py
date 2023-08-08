from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pyotp
from time import sleep
import os
#%% FUNCTION

def check_display(driver, xpath):
    return driver.find_element(by="xpath", value=xpath).is_displayed()

def driver_action(driver, link_or_xpath, type, time_wait=15, list_key=[]):
    if type == "get_link":
        driver.get(link_or_xpath)
        sleep(0.3)
    if type == "click":
        stt = 1
        while stt < 3:
            try:
                ele = WebDriverWait(driver, time_wait).until(
                    EC.visibility_of_element_located((By.XPATH, link_or_xpath)))
                ele.click()
                stt = 3
            except:
                stt += 1
            sleep(0.3)
    if type == "sendkey":
        ele = WebDriverWait(driver, time_wait).until(
            EC.visibility_of_element_located((By.XPATH, link_or_xpath)))
        for key in list_key:
            ele.send_keys(key)
        sleep(0.3)
    if type == 'getText':
        return driver.find_elements_by_xpath(link_or_xpath)[0].get_attribute("value")


def load_chrome_driver(download_location, showbrowser = True):
    # check and download driver
    s = Service(ChromeDriverManager().install())
    # load driver
    chromeOptions = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_location}
    chromeOptions.add_experimental_option("prefs", prefs)
    if showbrowser == False:
        chromeOptions.add_argument('headless')
    driver = webdriver.Chrome(service=s, options=chromeOptions)
    sleep(4)
    return driver

def sign_in(driver, Config):
    driver_action(driver, Config['portal'], type="get_link")  # go to portal
    driver_action(driver, Config['user_box'], type="sendkey", list_key=[Keys.CONTROL + 'a', Keys.COMMAND + 'a' , Keys.BACKSPACE])
    driver_action(driver, Config['user_box'], type="sendkey", list_key=[Config["user"]])
    driver_action(driver, Config['password_box'],type="sendkey", list_key=[Keys.CONTROL + 'a', Keys.COMMAND + 'a' , Keys.BACKSPACE])
    driver_action(driver, Config['password_box'], type="sendkey", list_key=[Config["password"]])
    driver_action(driver, Config['signin_button'], type="click")
    sleep(5)

def download_data(driver, day_start, day_end, Config):
    if check_display(driver, Config['report_tab']) == False: driver_action(driver, Config['sidebar'], type="click")  # side bar
    if check_display(driver, Config['typedata_xpath']) == False: driver_action(driver, Config['report_tab'], type="click")  # report tab
    driver_action(driver, Config['typedata_xpath'], type="click")  # request tab
    driver_action(driver, Config['day_start'],type="sendkey", list_key=[Keys.CONTROL + 'a', Keys.COMMAND + 'a' , Keys.BACKSPACE, day_start])  # day_start
    driver_action(driver, Config['day_end'],type="sendkey", list_key=[Keys.CONTROL + 'a', Keys.COMMAND + 'a' , Keys.BACKSPACE, day_end])  # day_end
    driver_action(driver, Config['download_button'], type="click")

def download_mapping_telco( download_location , show = False):

    def mapping_telco_crawl(driver,download_location, filemap = None):
        file_map_telco = download_location + r"/map_telco.xlsx" if filemap is None else filemap
        site = r"https://portal3.brandsms.vn/pages/v3/login.html"
        driver_action(driver, site, type="get_link")
        sleep(0.1)
        userBox_fullXpath = '/html/body/div/div[2]/form/div/span/div[1]/input'
        driver_action(driver, userBox_fullXpath, type="sendkey", list_key=['datatech'])
        sleep(0.1)
        passBox_fullXpath = '/html/body/div/div[2]/form/div/span/div[2]/input'
        driver_action(driver, passBox_fullXpath, type="sendkey", list_key=['123@dataTech'])
        sleep(0.1)
        sign_in_button = '/html/body/div/div[2]/form/div/div/div/button'
        driver_action(driver, sign_in_button, type="click")
        sleep(0.1)
        totp = pyotp.TOTP('UEFTG3BTKRG3X4NP')
        otpBox_fullXpath = '/html/body/div/div[2]/form/div/div[1]/div/input'
        driver_action(driver, otpBox_fullXpath, type="sendkey", list_key=[totp.now()])
        sleep(0.1)
        sign_in_button = '/html/body/div/div[2]/form/div/div[2]/div/button'
        driver_action(driver, sign_in_button, type="click",time_wait=10)
        sleep(3)

        if check_display(driver, "/html/body/div/aside[1]/section/ul/li[1]/ul/li[3]/a") == False:
            driver_action(driver, "/html/body/div/aside[1]/section/ul/li[1]/a/span[1]", type="click")  # side bar

        if check_display(driver, "/html/body/div/aside[1]/section/ul/li[1]/ul/li[3]/ul/li[1]/a/span") == False:
            driver_action(driver, "/html/body/div/aside[1]/section/ul/li[1]/ul/li[3]/a", type="click")

        driver_action(driver, "/html/body/div/aside[1]/section/ul/li[1]/ul/li[3]/ul/li[1]/a/span", type="click")

        choose_file_xpath = r'/html/body/div/div[1]/section/section/div/div/div/div[1]/div[1]/div/div/input'
        map_telco_dir = file_map_telco
        driver_action(driver, choose_file_xpath, type="sendkey", list_key=[map_telco_dir])

        get_telco_xpath = '/html/body/div/div[1]/section/section/section/div[2]/ul/li[1]/button'
        driver_action(driver, get_telco_xpath, type="click")
        sleep(5)
        
        download_xpath  = "/html/body/div/div[1]/section/section/section/div[2]/ul/li[2]/button"
        driver_action(driver, download_xpath, type="click")
        sleep(5)

    print('Trình duyệt đang chạy ngầm...')
    restart_time = 0
    restart_allowed = True
    while (restart_time <= 2) and restart_allowed:
        driver = load_chrome_driver(download_location, show)
        try:
            mapping_telco_crawl(driver,download_location)
            
            if len([i for i in os.listdir(download_location) 
                if 'DataTelco' in i]) == 0:
                raise
            driver.quit()
            restart_allowed = False
            # print('Đã tải xong file mapping_telco')
        except Exception as e:
            restart_time = restart_time + 1
            # print(str(e))
            driver.quit()
            if restart_time <= 2:
                print('Khởi động lại trình duyệt do lỗi !')
