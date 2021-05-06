from .base import AbstractDataset

import pandas as pd

from sqlalchemy import create_engine
import mysql.connector as connection
import pymysql

db_connection_str = 'mysql+pymysql://root:deipss@127.0.0.1/lotus'

from datetime import date


class DbDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'db'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['movies.dat',
                'ratings.dat',
                'users.dat']

    def load_ratings_df(self):
        df = self.read_ratings()
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

    # Reading ratings file:
    def read_ratings(self):
        db_connection = create_engine(db_connection_str)
        cnx = db_connection.connect()
        df = pd.read_sql('SELECT uid as uid, mid as sid,score as rating,addtime as timestamp FROM rating',
                         con=cnx)
        cnx.close()
        print("read_ratings() 评分长度=%s" % len(df))
        return df



