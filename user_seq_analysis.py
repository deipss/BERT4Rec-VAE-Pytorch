import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width',1000)



def generate_meta_map():
    file_path = './Data/ml-1m/movies.dat'
    df = pd.read_csv(file_path, sep='::', header=None)
    df.columns = ['sid', 'sname', 'smeta']
    return df


def load_ratings_df():
    file_path = './Data/ml-1m/ratings.dat'
    df = pd.read_csv(file_path, sep='::', header=None)
    df.columns = ['uid', 'sid', 'rating', 'timestamp']
    return df


if __name__ == '__main__':
    mdf = generate_meta_map()
    del mdf['sname']
    map = mdf.set_index('sid').T.to_dict('int')['smeta']
    df = load_ratings_df()
    df = df[df['uid'].isin([4958, 1, 22, 31, 44, 55, 6, 77, 8, 9])]
    u = df.sort_values('timestamp', ascending=False).groupby('uid').head(10)
    u['meta'] = u['sid'].apply(lambda x: map[x])
    # u['meta'] = u['meta'].map(lambda x: mdf.loc[mdf['sid'] == x]['smeta'])
    # print(u)
    u.to_csv('user_seq.csv', header = 0)  # 不保存列名


