from .base import AbstractDataset

import pandas as pd

from datetime import date


class BehaviorDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'behavior'

    @classmethod
    def url(cls):
        return 'test'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['UserBehavior.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('UserBehavior.csv')
        df = pd.read_csv(file_path, sep=',', header=None)
        df.columns = ['uid', 'sid', 'smeta','behavior', 'timestamp']
        return df


