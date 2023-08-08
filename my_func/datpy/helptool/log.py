import os
import functools
import logging
from datetime import datetime


class Mylog:
    """
    Setup trình ghi log
    # logging.debug("This is a debug message")
    # logging.info("This is an informational message")
    # logging.warning("Careful! Something does not look right")
    # logging.error("You have encountered an error")
    # logging.critical("You are in trouble")
    """

    def __init__(self, loggerName=None , Logfolder = None):
        self.logFormatter = logging.Formatter(
            "%(asctime)s|%(name)s|%(module)s(%(lineno)d)|%(levelname)s|>%(message)s",
            '%Y-%m-%d %H:%M:%S')
        self.name = loggerName
        self.filename = 'log_{}.log'.format(datetime.now().strftime("%Y_%m_%d_%H_%M"))
        self.filedir = os.path.join(Logfolder, self.filename) if Logfolder is not None else self.filename


    def get_logger(self, toFile = True, toConsole = False):
        """Tạo logger

        Args:
            filedir (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # filedir =None # self.filedir if filedir is None else filedir
        
        # logging.basicConfig(filename=filedir, level=self.level,
        #                     format=log_format, datefmt='%Y-%m-%d %H:%M:%S',
        #                     )
        # return logging.getLogger(self.name)
        Logger = logging.getLogger(self.name)
        if toFile:
            fileHandler = logging.FileHandler(self.filedir)
            fileHandler.setFormatter(self.logFormatter)
            fileHandler.setLevel(logging.INFO)
            Logger.addHandler(fileHandler)
        if toConsole or (not toFile):
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(self.logFormatter)
            consoleHandler.setLevel(logging.INFO)
            Logger.addHandler(consoleHandler)
        return Logger


def logs(func=None, logger: Mylog = None, Success_message=None, Failure_message=None):
    """
    log apply to function
    """

    if not isinstance(logger, logging.RootLogger):
        logger = logger()
        assert isinstance(logger, logging.RootLogger)

    def decorator_log(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            # appendInfo = appendInfo if type(appendInfo) == list else [appendInfo]
            signature = ", ".join(args_repr + kwargs_repr)
            try:
                result = func(*args, **kwargs)
                logger.info((f"Success in \'{func.__qualname__}\' with args {signature}")
                            if Success_message is None else Success_message)
                return result
            except Exception as e:
                logger.error((f"Exception raised in \'{func.__qualname__}\' with args {signature} --> "+str(
                    e)) if Failure_message is None else Failure_message)
                # raise e
        return wrapper
    if func is None:
        return decorator_log
    else:
        return decorator_log(func)
