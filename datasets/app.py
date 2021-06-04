from .base import AbstractDataset

import pandas as pd
import json

from datetime import date


class AppDataset(AbstractDataset):

    def parse(self, path):
        g = open(path, mode='r')
        for l in g:
            yield json.loads(l)

    def getDF(self, path):
        df = {}
        for d in self.parse(path):
            df[d['asin']] = d.get('category', [])
        return df

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

    def generate_meta_map(self, smap):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('meta_Appliances.json')
        _map = self.getDF(file_path)
        meta_set = set()
        for v in _map.values():
            meta_set.update(v)
        df = self.load_ratings_df()
        del df['uid']
        del df['rating']
        del df['timestamp']
        df['smeta'] = df['sid'].map(lambda x: _map.get(x, []))
        set_len = len(meta_set)
        umap = {u: i + 1 for i, u in enumerate(meta_set)}
        # 位运算
        df['smeta'] = df['smeta'].map(lambda x: [umap[v] for v in x])
        df['smeta'] = df['smeta'].map(lambda x: x + [0] * (set_len - len(x)))
        set_smap = smap.keys()
        df['sid'] = df['sid'].map(lambda x: smap[x] if x in set_smap else x)

        map = df.set_index('sid').T.to_dict('int')
        return map['smeta'], len(meta_set)
