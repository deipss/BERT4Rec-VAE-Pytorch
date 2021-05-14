from .base import AbstractDataset

import pandas as pd

from datetime import date


class ML20MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-20m'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['genome-scores.csv',
                'genome-tags.csv',
                'links.csv',
                'movies.csv',
                'ratings.csv',
                'README.txt',
                'tags.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.csv')
        df = pd.read_csv(file_path)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

    def generate_meta_map(self, smap):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('movies.csv')
        df = pd.read_csv(file_path)
        df.columns = ['sid', 'sname', 'smeta']
        df['smeta'] = df['smeta'].map(lambda x: x.split("|"))
        meta_set = set()
        for i in df['smeta'].values:
            meta_set.update(i)
        del df['sname']
        set_len = len(meta_set)
        umap = {u: i + 1 for i, u in enumerate(meta_set)}
        # 位运算
        df['smeta'] = df['smeta'].map(lambda x: [umap[v] for v in x])
        df['smeta'] = df['smeta'].map(lambda x: x + [0] * (set_len - len(x)))
        set_smap = smap.keys()
        df['sid'] = df['sid'].map(lambda x: smap[x] if x in set_smap else x)

        map = df.set_index('sid').T.to_dict('int')
        return map['smeta'], len(meta_set)
