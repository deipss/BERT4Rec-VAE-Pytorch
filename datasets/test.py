import pandas as pd

def run():
    file_path = ('/home/deipss/BERT4Rec-VAE-Pytorch-master/Data/ml-1m/movies.dat')
    df = pd.read_csv(file_path, sep='::', header=None)
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
    df['smeta'] = df['smeta'].map(lambda x: x + [0] *( set_len - len(x)))
    smap={}
    set_smap = smap.keys()
    df['sid'] = df['sid'].map(lambda x: smap[x] if x in set_smap else x)

    map = df.set_index('sid').T.to_dict('int')
    a = map['smeta']
    pass


import pandas as pd
import json

import pandas as pd
import gzip


def parse(path):
    g = open(path, mode='r')
    for l in g:
        yield json.loads(l)


def getDF(path):
    df = {}
    for d in parse(path):
        if not 'category' in d.keys():
            print(d)
        print(d.get('category', []))
        df[d['asin']] = d.get('category', [])
    return df


if __name__ == '__main__':

    folder_path2 = '/home/deipss/BERT4Rec-VAE-Pytorch-master/Data/card/meta_Gift_Cards.json'
    df = getDF(folder_path2)
    categories_set = set()
    for v in df.values():
        categories_set.update(v)
    print(categories_set)
    print(len(categories_set))
