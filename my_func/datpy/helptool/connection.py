# %%
import io
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
import pickle
import cx_Oracle
import pandas as pd
import ftplib
import os
from datetime import datetime
from pathlib import Path
from pyunpack import Archive
from sqlalchemy import create_engine, types
import sqlite3
from tqdm import tqdm as tqdm_
from munch import DefaultMunch
import shutil
import pymongo


def runtime(func):
    """Decorator show time to run function

    Args:
        func (_type_): Function muốn tính time
    """
    def func_wrapper(*args, **kwargs):
        start = datetime.now()
        res = func(*args, **kwargs)
        stop = datetime.now()
        print("--> Finish in {}s".format(str(stop - start).split(".")[0]))
        return res
    return func_wrapper


def showlog(mylogger=None, level='error'):
    """Lựa chọn muốn print ra trên console hay ghi vào file log
    Returns:
        _type_: _description_
    """
    if mylogger is None:
        return print
    else:
        level = level.lower()
        if level == 'error':
            return mylogger.error
        elif level == 'warning':
            return mylogger.warning
        elif level == 'info':
            return mylogger.info
        elif level == 'critical':
            return mylogger.critical


class FtpServer:
    """Setup kết nối tới FTP server
    """

    def __init__(self, **kwargs,):
        """
        Nếu truyền dict thì FtpServer(**dict)
        """
        kwargs = DefaultMunch.fromDict(kwargs)
        self.hostname = kwargs['hostname']
        self.username = kwargs['username']
        self.password = kwargs['password']
        self.logger = kwargs['logger']
        self.connect()

    def connect(self):
        """Kết nối tới FTP server

        Returns:
            _type_: _description_
        """
        self.ftp = ftplib.FTP(self.hostname, self.username, self.password)
        self.ftp.encoding = "utf-8"
        showlog(mylogger=self.logger, level='info')(
            f'Success connecting to server {self.hostname}')
        return self.ftp

    def listfile(self, folder=None):
        """list files tại currentdir hoặc specific dir

        Args:
            folder (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        server = self.connect()
        if folder is not None:
            server.cwd(folder)
        return server.nlst()

    def listdir(self, folder=None, countfile=0):
        """list pathfile và subpathfile tại currentdir hoặc specific dir

        Args:
            folder (_type_, optional): _description_. Defaults to None.
            countfile (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        server = self.connect()
        listdir = []
        if folder is not None:
            server.cwd(folder)
        currentdir = server.pwd()
        listfile = server.nlst()
        listfile = [i for i in listfile if 'ignore' not in i.lower()]
        for f in listfile:
            path = currentdir + "/" + f
            try:
                server.cwd(path)
                append_file = self.listdir(folder=path, countfile=countfile)
                listdir += append_file
                countfile += len(append_file)
            except:
                listdir.append(path)
                countfile += 1
        print(f"Count files in ftp's folder \'{folder}\': ", countfile)
        showlog(mylogger=self.logger, level='info')(
            f"Count files in ftp's folder '{folder}': {countfile}")
        return listdir

    def toFtpFile(self, filedirFTP, folderSaveLocal):
        return FtpFile(self.configs_database, filedirFTP, folderSaveLocal)


class FtpFile(FtpServer):
    """Action cho 1 file nhất định trên ftp server

    Args:
        FtpServer (_type_): _description_
    """

    def __init__(self, hostname, username, password, logger, filedirFTP, folderSaveLocal):
        super().__init__(hostname, username, password, logger)
        self.filedirFTP = filedirFTP
        self.folderLocal = folderSaveLocal
        self.filename = os.path.basename(self.filedirFTP)
        self.filedirLOCAL = os.path.join(self.folderLocal, self.filename)

    def getsize(self):
        """Get thông tin size của file

        Returns:
            _type_: _description_
        """
        a = self.connect()
        sizefile = a.size(self.filedirFTP)
        if sizefile > 1024**3:
            return "{:.2f} GB".format(sizefile/1024**3)
        elif sizefile > 1024**2:
            return "{:.2f} MB".format(sizefile/1024**2)
        elif sizefile > 1024:
            return "{:.2f} KB".format(sizefile/1024)
        else:
            return "{} bytes".format(sizefile)

    def checkFolder(self):
        """Remove all file tin folder save trên local, tránh trường hợp lưu nhiều gây đầy bộ nhớ
        """
        Path(self.folderLocal).mkdir(parents=True, exist_ok=True)
        listfiles = os.listdir(self.folderLocal)
        for filename in listfiles:
            os.remove(os.path.join(self.folderLocal, filename))
            showlog(mylogger=self.logger, level='info')(
                f'Removed {filename} in \'{self.folderLocal}\'')

    # @runtime
    def unpackFile(self, delZipFile=True):
        """Giải nén file nếu dạng file ở dạng nén

        Args:
            delZipFile (bool, optional): _description_. Defaults to True.
        """
        pathFile = self.filedirLOCAL
        try:
            Archive(pathFile,).extractall(os.path.dirname(pathFile))
            showlog(mylogger=self.logger, level='info')(
                f'Unpacked file {self.filename}')
            os.remove(pathFile)
            showlog(mylogger=self.logger, level='info')(
                f'Removed file {pathFile}')
        except:
            showlog(mylogger=self.logger, level='error')(
                f'Failed to unpack file {self.filename}')

    def getlistdir(self, unpack=False):
        if unpack and (os.path.splitext(self.filedirLOCAL)[1] in ['.zip', '.rar', '.gz', '.bz2', '.7z']):
            self.unpackFile()
        self.unpack_filelist = [os.path.join(
            self.folderLocal, i) for i in os.listdir(self.folderLocal)]
        showlog(mylogger=self.logger, level='info')(
            f'Get all files {self.filename}')

    # @runtime
    def downloadFile(self):
        """Download file từ ftp về local
        """
        FTPdir = os.path.dirname(self.filedirFTP)
        sizefile = self.getsize()
        ftp = self.connect()
        ftp.cwd(FTPdir)
        assert (self.filename in ftp.nlst())
        with open(self.filedirLOCAL, 'wb') as fobj:
            ftp.retrbinary('RETR ' + self.filename, fobj.write)
            showlog(mylogger=self.logger, level='info')(
                f'Downloaded {self.filedirFTP} ({sizefile})')

    def process(self, run_download=True):
        """Process chạy chính

        Args:
            run_download (bool, optional): _description_. Defaults to True.
        """
        if run_download:
            self.checkFolder()
            self.downloadFile()
        self.getlistdir()


class LocalFile:
    """Action cho 1 file nhất định trên local server
    """

    def __init__(self, folderRawLocal, folderSaveLocal, **kwargs):
        self.logger = kwargs['logger']
        self.folderRawLocal = folderRawLocal
        self.folderSaveLocal = folderSaveLocal
        self.listfilesdir = [os.path.join(
            self.folderRawLocal, i) for i in os.listdir(self.folderRawLocal)]

    def getsize(self, filedir):
        """Get thông tin size của file

        Returns:
            _type_: _description_
        """

        sizefile = os.path.getsize(filedir)
        if sizefile > 1024**3:
            return "{:.2f} GB".format(sizefile/1024**3)
        elif sizefile > 1024**2:
            return "{:.2f} MB".format(sizefile/1024**2)
        elif sizefile > 1024:
            return "{:.2f} KB".format(sizefile/1024)
        else:
            return "{} bytes".format(sizefile)

    def process(self, pathFile):

        def checkFolder(folder):
            """Remove all file tin folder save trên local, tránh trường hợp lưu nhiều gây đầy bộ nhớ
            """
            Path(folder).mkdir(parents=True, exist_ok=True)
            listfiles = os.listdir(folder)
            for filename in listfiles:
                os.remove(os.path.join(folder, filename))
                showlog(mylogger=self.logger, level='info')(
                    f'Removed {filename} in \'{folder}\'')

        def unpackFile(pathFile, delZipFile=True):
            try:
                Archive(pathFile,).extractall(os.path.dirname(pathFile))
                showlog(mylogger=self.logger, level='info')(
                    f'Unpacked file {pathFile}')
                os.remove(pathFile)
                showlog(mylogger=self.logger, level='info')(
                    f'Removed file {pathFile}')
            except:
                showlog(mylogger=self.logger, level='error')(
                    f'Failed to unpack file {pathFile}')

        checkFolder(self.folderSaveLocal)
        shutil.copyfile(pathFile, self.folderSaveLocal)
        newpath = os.path.join(self.folderSaveLocal,
                               os.path.basename(pathFile))
        if os.path.splitext(newpath) in ['.zip', '.rar', '.gz', '.bz2', '.7z']:
            self.unpackFile(newpath)
        self.unpack_filelist = [os.path.join(
            self.folderSaveLocal, i) for i in os.listdir(self.folderSaveLocal)]
        showlog(mylogger=self.logger, level='info')(
            f'Get all files {pathFile}')


class Database:
    """Setup kết nối tới database oracle/sqlite và các thao tác: đọc, ghi, tạo, xóa, phân quyền
    """

    def __init__(self, **kwargs):
        self.kwargs = DefaultMunch.fromDict(kwargs)
        self.hostname = self.kwargs['hostname']
        self.username = self.kwargs['username']
        self.password = self.kwargs['password']
        self.port = self.kwargs['port']
        self.service_name = self.kwargs['service_name']
        self.path = self.kwargs['path']
        self.uri = self.kwargs['uri']
        self.database_name = self.kwargs['database_name']
        self.collection_name = self.kwargs['collection_name']
        self.type = self.kwargs['type']
        self.logger = self.kwargs['logger']
        self.connect()

    def connect(self, printStatus=True):
        self.conn = None
        try:
            if (self.type == 'oracle') or (self.service_name is not None):  # oracle
                try:
                    self.conn = cx_Oracle.connect(
                        user=self.username, password=self.password, dsn=f"{self.hostname}:{self.port}/{self.service_name}")
                    self.engine = create_engine(
                        f'oracle+cx_oracle://{self.username}:{self.password}@{self.hostname}:{self.port}/?service_name={self.service_name}')
                    self.type = 'oracle'
                    if printStatus:
                        showlog(mylogger=self.logger, level='info')(
                            f"Success to connect to ORACLE {self.hostname}:{self.port}/{self.service_name}")
                except:
                    self.conn = None

            elif self.type == 'sqlite3' or (self.path is not None):
                self.conn = sqlite3.connect(self.path)
                self.engine = sqlite3.connect(self.path)
                self.type = 'sqlite3'
                filename = os.path.basename(self.path)
                if printStatus:
                    showlog(mylogger=self.logger, level='info')(
                        f"Success to connect to Sqlite3 {filename}")

            elif self.conn is None:  # mongodb
                if self.uri is None:
                    hostname = self.hostname if self.hostname is not None else 'localhost'
                    port = self.port if self.port is not None else "27017"
                    if self.username is not None:
                        self.uri = f'mongodb://{self.username}:{self.password}@{hostname}:{port}/admin?authSource=admin&authMechanism=SCRAM-SHA-1'
                    else:
                        self.uri = f'mongodb://{hostname}:{port}/'
                self.client = pymongo.MongoClient(self.uri)

                if self.database_name is not None:
                    self.database = self.client[self.database_name]
                    if self.collection_name is not None:
                        self.collection = self.database[self.collection_name]
                self.type = 'mongodb'
                if printStatus:
                    showlog(mylogger=self.logger, level='info')(
                        f"Success to connect to MongoDB {hostname}:{port}")

        except Exception as e:
            showlog(mylogger=self.logger, level='error')(
                'Fail to connect to database !')
            raise e


class DatabaseSQL(Database):
    def __init__(self, **kwargs):
        super(DatabaseSQL, self).__init__(**kwargs)

    # @runtime
    def drop(self, tablename, schema=None):
        # Drop table if exists
        cursor = self.conn.cursor()
        tablename = "{}.{}".format(
            schema, tablename) if schema is not None else tablename
        showlog(mylogger=self.logger, level='warning')(
            f'Droping {tablename.upper()} table if exists.')
        cursor.execute(
            f"BEGIN EXECUTE IMMEDIATE 'DROP TABLE {tablename.upper()}'; EXCEPTION WHEN OTHERS THEN NULL; END;")
        showlog(mylogger=self.logger, level='warning')(
            f'Droped {tablename.upper()} table if exists.')

    def create(self, tablename: str, typeCol: dict, schema=None):

        def check_exists_table(tablename, schema, conn):
            sql = f"""
            select count(*) from user_tables 
            where table_name = '{tablename}'
            and tablespace_name= '{schema}'
            """
            cnt = pd.read_sql_query(sql, conn).iloc[0, 0]
            if cnt == 0:
                showlog(mylogger=self.logger, level='warning')(
                    f'There are no {tablename.upper()} table')
                return False
            else:
                # hp.cfg['log'].info(f'There are no {tablename.upper()} table')
                return True

        if check_exists_table(tablename, schema, self.engine) == False:
            try:
                tablename = ("{}.{}".format(schema, tablename) if (schema is not None) else tablename).upper()
                cursor = self.conn.cursor()
                schemaCol = ", ".join(
                    ["{} {}".format(i, typeCol[i]) for i in typeCol.keys()])
                cursor.execute(f"CREATE TABLE {tablename} ({schemaCol})")
                showlog(mylogger=self.logger, level='info')(
                    f'Created {tablename.upper()} table in {schema.upper()}')
                return True
            except:
                showlog(mylogger=self.logger, level='error')(
                    f'Fail to created {tablename.upper()} table in {schema.upper()}')

    def describe(self, tablename):
        if self.configs_database.type == 'oracle':
            return pd.read_sql_query(f"Select COLUMN_NAME, DATA_TYPE, DATA_LENGTH from ALL_TAB_COLUMNS where TABLE_NAME = \'{tablename}\' ", self.conn)
        else:
            raise "Not set describe for non-Oracle connection"

    def getdtype(dataSchema):
        """Convert dtype từ dict trên yaml sang sqlalchemy, tạo tham số khi đẩy dữ liệu lên database

        Args:
            dataSchema (_type_): _description_
        """
        def convert_tool(x: str):
            if x.lower() == 'date':
                return types.DATE()
            elif x.lower().startswith('varchar2'):
                lenght_varchar2 = int(x[x.index("(")+1:x.index(")")])
                return types.VARCHAR(lenght_varchar2)
                # return types.CLOB()
            elif 'float' in x.lower():
                return types.FLOAT()
            elif 'integer' in x.lower():
                return types.INTEGER()
        return {i: convert_tool(dataSchema[i]) for i in dataSchema.keys()}

    # @logs(logger = hp.cfg['log'])
    def upload(self, data, dataSchema, tablename: str, schema=None, chunksize=5000, if_exists='append', filename=None, logIndex=True):
        try:
            dty = Database.getdtype(
                dataSchema) if dataSchema is not None else None
            data.to_sql(tablename.lower(), schema=schema, con=self.engine, if_exists=if_exists,
                        chunksize=chunksize, index=False, dtype=dty,)

        except Exception as e:
            withIndex = f' from {data.index[0]} to {data.index[-1]}' if logIndex else ""
            showlog(mylogger=self.logger, level='error')(
                f"Fail to upload data {filename}{withIndex} with error: {e}")

    def access(self, toUser, tablename, access='select', schema=None):
        """
        grant select/insert/update/delete on <schema>.<table_name> to <username>;
        """
        cursor = self.conn.cursor()
        tablename = "{}.{}".format(
            schema, tablename) if schema is not None else tablename
        cursor.execute(f"""grant {access} on {tablename} to {toUser};""")
        self.conn.commit()
        cursor.close()
        print(f'Set {toUser} to {access} in {tablename} !')

    def createIndex(self, indexname, tablename, cols, schema=None):
        """
        CREATE INDEX <indexname> ON <schema.tablename> (cols);
        """
        cursor = self.conn.cursor()
        cols_list = cols if type(cols) != list else ", ".join(cols)
        tablename = "{}.{}".format(
            schema, tablename) if schema is not None else tablename
        cursor.execute(
            f"""CREATE INDEX {indexname} ON {tablename} ({cols_list});""")
        self.conn.commit()
        cursor.close()
        # conn.close()
        print(f'Set {indexname} as index to {cols_list} in {tablename} !')

    # @runtime
    def read(self, table_name_sql: str = None, col_name="*",
             offset_rows: int = 0, n_records: int = -1, chunksize: int = None, position=0):
        self.connect(False)
        if (table_name_sql is None):
            if type(self.conn) == cx_Oracle.Connection:
                return pd.read_sql_query("SELECT OWNER,TABLE_NAME,TABLESPACE_NAME  FROM all_tables", self.engine)
            else:
                return pd.read_sql_query("SELECT *  FROM sqlite_master", self.engine)

        if type(self.conn) == cx_Oracle.Connection:
            offset_clause = " offset {} rows ".format(offset_rows)
            num_records_clause = "" if n_records == - \
                1 else " fetch next {} rows only".format(n_records)
            combine_clause = offset_clause + num_records_clause
        else:  # sqlite3
            offset_clause = "" if offset_rows == 0 else " offset {} ".format(
                offset_rows)
            num_records_clause = "limit -1" if n_records == - \
                1 else " limit {} ".format(n_records)
            combine_clause = num_records_clause + offset_clause

        if 'select ' not in table_name_sql.lower():
            cols = col_name if type(col_name) == str else ", ".join(col_name)
            sql = """
            select {} from {} {}
            """.format(cols, table_name_sql, combine_clause)
        else:
            sql = table_name_sql + " " + combine_clause
        tablename = sql.split(' ')[sql.lower().split(' ').index('from')+1]
        res = pd.read_sql_query(sql=sql, con=self.engine, chunksize=chunksize)
        if chunksize is not None:
            res = tqdm_(res, desc=tablename, position=position)
        # print("Bảng {} offset {} dòng, {} records".format(table_name,offset_rows,n_records) + ("" if chunksize is None else ", chunksize {}".format(chunksize)))
        return res


class Mongo:
    pass


class GDrive:

    def __init__(self, client_secret_json=None):
        self.client_secret_json = client_secret_json
        self.service = self.Create_Service()

    def Create_Service(self, api_name='drive', api_version='v3', scopes=['https://www.googleapis.com/auth/drive']):
        # print(self.client_secret_json, api_name, api_version, scopes, sep='-')
        CLIENT_SECRET_FILE = self.client_secret_json
        API_SERVICE_NAME = api_name
        API_VERSION = api_version
        SCOPES = scopes
        # print(SCOPES)

        cred = None

        pickle_file = f'token_{API_SERVICE_NAME}_{API_VERSION}.pickle'
        # print(pickle_file)

        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as token:
                cred = pickle.load(token)
        elif not cred or not cred.valid:
            if cred and cred.expired and cred.refresh_token:
                cred.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CLIENT_SECRET_FILE, SCOPES)
                cred = flow.run_local_server()

            with open(pickle_file, 'wb') as token:
                pickle.dump(cred, token)

        try:
            service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
            # print(API_SERVICE_NAME, 'service created successfully')
            return service
        except Exception as e:
            print('Unable to connect.')
            print(e)
            return None

    def get_files(self, folderID, scanSubfolder=True, folderdir=""):
        query = f"parents = '{folderID}'"
        resource = self.service.files().list(q=query).execute().get('files')
        if not scanSubfolder:
            return resource
        else:
            for i in resource:
                i['folderdir'] = folderdir
                if i['mimeType'].endswith('folder'):
                    new_fol_dir = f"{i['folderdir']}\\{i['name']}" if folderdir != '' else i['name']
                    resource += self.get_files(
                        i['id'], scanSubfolder=scanSubfolder, folderdir=new_fol_dir)
            return resource

    def downloadFile(self, fileID, file_name, save_folder):
        request = self.service.files().get_media(fileId=fileID)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fd=fh, request=request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            # print(file_name,': {:2.0f}%'.format(status.progress() * 100), end = "\r")
        fh.seek(0)
        with open(os.path.join(save_folder, file_name), 'wb') as f:
            f.write(fh.read())
            f.close()

    def downloadFolder(self, folderID, folder_name, save_folder, position=0):
        listfile = self.get_files(folderID, scanSubfolder=False)
        folderpath = os.path.join(save_folder, folder_name)
        Path(folderpath).mkdir(parents=True, exist_ok=True)
        for file in tqdm_(listfile, desc=folder_name, position=position):
            if file['mimeType'].endswith('folder'):
                self.downloadFolder(
                    file['id'], file['name'], folderpath, position=position+1)
            else:
                self.downloadFile(file['id'], file['name'], folderpath)
