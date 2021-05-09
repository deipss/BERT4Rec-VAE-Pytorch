import torch
import pandas as pd
from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
E = {}
mid2idx = {}
midx2id = {}


def get_E():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader, dataloader = dataloader_factory(args)
    print('template = %s\t model_code = %s\n' % (args.template, args.model_code))
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    # trainer.train()
    best_model = torch.load(os.path.join('/home/deipss/BERT4Rec-VAE-Pytorch-master/experiments/test_bert_128_ml-1m_2021-05-07_0/models/best_acc_model.pth')).get('model_state_dict')
    model.load_state_dict(best_model)
    model.eval()
    global E
    global mid2idx
    global midx2id
    mid2idx = dataloader.smap
    midx2id = {i: s for i, s in mid2idx.items()}
    E = model.bert.embedding.token.weight.data


# 数据植入cuda
def to_device(d):
    return Variable(d).to(device)


def evaluate(emb, top_k, movie_lists):
    rst = []
    for m in movie_lists:
        c = to_device(torch.LongTensor([m]))
        cos_result = F.cosine_similarity(emb[c], emb).sort(descending=True)
        cos_dis_idx = cos_result[1]  # 下标
        cos_dis = cos_result[0]  # 分数
        rank_list = cos_dis_idx.data.cpu().numpy()[:top_k]  # 前K个最近距离的项目
        rank_list = rank_list.tolist()  # numpy array 转成 python list
        rst.append(rank_list)
    return rst


def search_neighbor_item():
    # 读取pandas
    df = generate_meta_map()
    # m_list = [12, 17, 71, 177, 36, 23, 76, 44, 98, 337]
    # # 加载模型
    get_E()
    # rst = evaluate(E,11,m_list)
    rst = [[12, 59, 77, 623, 2359, 2415, 28, 3261, 1043, 608, 1552], [17, 1495, 1030, 87, 1440, 345, 821, 2325, 77, 2415, 450], [71, 1091, 3327, 1994, 2865, 1177, 866, 1163, 1059, 1579, 1863], [177, 966, 73, 3440, 656, 1287, 1335, 807, 2628, 1671, 520], [36, 1289, 2040, 3352, 3336, 424, 1592, 3028, 2894, 1615, 1013], [23, 2063, 14, 1962, 1793, 2404, 1645, 769, 375, 745, 2706], [76, 2060, 65, 928, 3052, 519, 3069, 1578, 179, 1524, 2630], [44, 1241, 1099, 2505, 1576, 2975, 2904, 2099, 1020, 2884, 2378], [98, 2704, 2913, 3557, 216, 604, 1619, 464, 85, 1864, 3173], [337, 3310, 59, 2829, 2410, 852, 2154, 3372, 1028, 472, 787]]

    print(rst)
    for i in rst:
        target = midx2id[i[0]]
        print('idx=%d=  mid=%d' % (i[0],midx2id[i[0]]))
        print(get_info_by_sid(df,target))
        for k in i[1:]:
            item_mid = midx2id[k]
            print(get_info_by_sid(df, item_mid))


def generate_meta_map():
    file_path = '/home/deipss/BERT4Rec-VAE-Pytorch-master/Data/ml-1m/movies.dat'
    df = pd.read_csv(file_path, sep='::', header=None)
    df.columns = ['sid', 'sname', 'smeta']
    return df


def get_info_by_sid(df, sid):
    return df.loc[df['sid'] == sid].values.tolist()


if __name__ == '__main__':
    args.mode = 'train'
    i = 128
    args.bert_hidden_units = i
    args.dae_latent_dim = i
    args.vae_latent_dim = i
    args.dim = i
    search_neighbor_item()
