from .base import AbstractDataset

import pandas as pd

from datetime import date


class FashionDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'fashion'

    @classmethod
    def url(cls):
        return 'test'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['AMAZON_FASHION.csv',
                'meta_AMAZON_FASHION.json']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('AMAZON_FASHION.csv')
        df = pd.read_csv(file_path, sep=',', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df


