import pandas as pd

if __name__ == '__main__':
    file_path = '/home/deipss/BERT4Rec-VAE-Pytorch-master/Data/card/Gift_Cards.csv'
    df = pd.read_csv(file_path, sep=',', header=None)
    df.columns = ['uid', 'sid', 'rating', 'timestamp']
    cnt = df['sid'].value_counts()
    top50 = cnt.head(50)
    top50 = top50.index.tolist
    print(type(top50))
    print(top50)

