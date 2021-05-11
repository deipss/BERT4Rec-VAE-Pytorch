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
    best_model = torch.load(os.path.join(
        '/home/deipss/BERT4Rec-VAE-Pytorch-master/experiments/test_bert_128_ml-1m_2021-05-11_0_meta/models/best_acc_model.pth')).get(
        'model_state_dict')
    model.load_state_dict(best_model)
    model.eval()
    global E
    global mid2idx
    global midx2id
    mid2idx = dataloader.smap
    midx2id = {i: s for s, i in mid2idx.items()}
    E = model.bert.embedding.token.weight.data


# 数据植入cuda
def to_device(d):
    return Variable(d).to(device)


def evaluate(emb, top_k, movie_lists):
    rst = []
    cos_v = []
    for m in movie_lists:
        c = to_device(torch.LongTensor([m]))
        cos_result = F.cosine_similarity(emb[c], emb).sort(descending=True)
        cos_dis_idx = cos_result[1]  # 下标
        cos_dis = cos_result[0]  # 分数
        rank_list = cos_dis_idx.data.cpu().numpy()[:top_k]  # 前K个最近距离的项目
        rank_list = rank_list.tolist()  # numpy array 转成 python list
        rank_list_cos = cos_dis.data.cpu().numpy()[:top_k]  # 前K个最近距离的项目
        rank_list_cos = rank_list_cos.tolist()  # numpy array 转成 python list
        rst.append(rank_list)
        cos_v.append(rank_list_cos)
    return rst, cos_v


def search_neighbor_item():
    # 读取pandas
    df = generate_meta_map()
    m_list = [ 17, 71, 44, 98,501]
    # # 加载模型
    get_E()
    m_list = [mid2idx[i] for i in m_list]
    rst, cos_v = evaluate(E, 11, m_list)

    print(rst)
    for i, v in zip(rst, cos_v):
        print('emb_id=%d\tmovie_id=%d' % (i[0], midx2id[i[0]]))
        print(get_info_by_sid(df, midx2id[i[0]]))
        for k, c in zip(i[1:], v[1:]):
            if k not in midx2id.keys():
                continue
            item_mid = midx2id[k]
            print(get_info_by_sid(df, item_mid))


def generate_meta_map():
    file_path = '/home/deipss/BERT4Rec-VAE-Pytorch-master/Data/ml-1m/movies.dat'
    df = pd.read_csv(file_path, sep='::', header=None)
    df.columns = ['sid', 'sname', 'smeta']
    return df


def get_info_by_sid(df, sid):
    info = df.loc[df['sid'] == sid].values.tolist()[0]
    return '' + str(info[0]) + '\t' + str(info[1]) + '\t' + str(info[2])


if __name__ == '__main__':
    # df = generate_meta_map()
    # get_info_by_sid(df, 12)
    args.mode = 'train'
    i = 128
    args.bert_hidden_units = i
    args.dae_latent_dim = i
    args.vae_latent_dim = i
    args.dim = i
    search_neighbor_item()
