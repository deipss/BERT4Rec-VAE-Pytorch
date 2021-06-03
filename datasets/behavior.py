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
        df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['uid', 'sid', 'smeta','behavior', 'timestamp']
        return df


    def generate_meta_map(self, smap):
        df = self.load_ratings_df()
        del df['uid']
        del df['behavior']
        del df['timestamp']
        meta_set = set()
        meta_set.update(list(df['smeta'].values))
        set_len = len(meta_set)
        umap = {u: i + 1 for i, u in enumerate(meta_set)}
        # 位运算
        df['smeta'] = df['smeta'].map(lambda x: [umap[x]])
        df['smeta'] = df['smeta'].map(lambda x: x + [0] * (set_len - len(x)))
        set_smap = smap.keys()
        df['sid'] = df['sid'].map(lambda x: smap[x] if x in set_smap else x)

        map = df.set_index('sid').T.to_dict('int')
        return map['smeta'],len(meta_set)