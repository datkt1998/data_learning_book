import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

import os
import logging


class LinkInfo:

    def __init__(self, url):
        self.url = url
        self.domains = ['vietnambiz.vn']

    def verify_link(self):
        try:
            domain = urlparse(self.url).netloc
            if domain in self.domains:
                return True
            else:
                raise ValueError('the domain is not in the pre-defined list')
        except Exception as e:
            raise e

    def get_link(self):
        self.verify_link()
        r = requests.get(self.url)
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.find_all("title")[0].text.strip()
        content = soup.find_all('meta', {'id': 'metaDes', 'name': 'description'})[
            0]['content'].strip()
        return title, content


class Mylog:
    """
    Setup trình ghi log
    # logging.debug("This is a debug message")
    # logging.info("This is an informational message")
    # logging.warning("Careful! Something does not look right")
    # logging.error("You have encountered an error")
    # logging.critical("You are in trouble")
    """

    def __init__(self, log_folder=".", log_evel=logging.INFO, logger_name=None):
        self.folder = log_folder
        self.level = log_evel
        self.name = logger_name
        self.filedir = self.gen_logfile()

    def gen_logfile(self, ):
        """Tạo log file

        Returns:
            _type_: _description_
        """
        filename = 'log_prediction.log'
        return os.path.join(self.folder, filename)

    def get_logger(self, filedir=None):
        """Tạo logger

        Args:
            filedir (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        filedir = self.filedir if filedir is None else filedir
        # log_format = "%(asctime)s|%(name)s|%(module)s(%(lineno)d)|%(levelname)s|>%(message)s"
        log_format = "%(asctime)s|> %(message)s"
        logging.basicConfig(filename=filedir, level=self.level, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
        return logging.getLogger(self.name)

    def __call__(self):
        return self.get_logger()
