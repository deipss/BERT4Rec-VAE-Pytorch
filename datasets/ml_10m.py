from .base import AbstractDataset

import pandas as pd

from datetime import date


class ML10MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-10m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['movies.dat',
                'ratings.dat']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df


