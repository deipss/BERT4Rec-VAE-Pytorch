import pandas as pd

if __name__ == '__main__':
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