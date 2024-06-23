from .base import AbstractDataset

import pandas as pd


class AppDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'app'

    @classmethod
    def url(cls):
        return 'test'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['Appliances.csv',
                'meta_Appliances.json']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('Appliances.csv')
        df = pd.read_csv(file_path, sep=',', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df


