from .base import AbstractDataset

import pandas as pd

from datetime import date


class ML1MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ml-1m'

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
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df

    def generate_meta_map(self, smap):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('movies.dat')
        df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['sid', 'sname', 'smeta']
        df['smeta'] = df['smeta'].map(lambda x: x.split("|")[0].strip())
        del df['sname']

        meta_set = set(df['smeta'].values)
        umap = {u: i for i, u in enumerate(meta_set)}

        df['smeta'] = df['smeta'].map(lambda x: umap[x])
        set_smap = smap.keys()
        df['sid'] = df['sid'].map(lambda x: smap[x] if x in set_smap else x)

        map = df.set_index('sid').T.to_dict('int')
        return map['smeta']
