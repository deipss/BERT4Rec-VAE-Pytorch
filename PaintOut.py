import matplotlib.pyplot as plt
import numpy as np
import os
import json

lw = 1.7
markersize = 5.7


def data_to_line(data):
    r_s = '\t'
    n_s = '\t'
    Recall = 'Recall@'
    NDCG = 'NDCG@'
    for i in [1, 5, 10, 15, 20, 25]:
        r_s += str(round(data[Recall + str(i)], 4)) + '\t'
        n_s += str(round(data[NDCG + str(i)], 4)) + '\t'
    return r_s + n_s


def data_load():
    dir = './experiments'
    for root, dirs, files in os.walk(dir):
        infos = root.split('_')
        for f in files:
            if 'test_metrics.json' == f and '1m' in root:
                if 'cnn' == infos[2]:
                    infos[1] = infos[1] + infos[2]
                    infos[2] = infos[3]
                    infos[3] = infos[4]
                data_file = open(root + '\\\\' + f)
                data = json.load(data_file)
                print('%12s\t%3s\t%10s\t%s'
                      % (infos[1], infos[2], infos[3], data_to_line(data)))



def data_load_bert_cnn():
    dir = './experiments'
    for root, dirs, files in os.walk(dir):
        for sub_dir in dirs:
            path = root + '/' + sub_dir
            if 1 ==1  and '25_' in sub_dir:
                try:
                    data, data_config = {}, {}
                    data_file = open(path + '/logs/' + 'test_metrics.json')
                    data = json.load(data_file)
                    data_file = open(path + '/' + 'config.json')
                    data_config = json.load(data_file)

                    print('%12s\t%3s\t%10s\t%d\t%d\t%s'
                          % ('Meta_'+data_config['model_code'], data_config['dim'], data_config['dataset_code'],
                             data_config['bert_num_blocks'], data_config['bert_num_heads'], data_to_line(data)))
                except BaseException:
                    pass
                    # print()

def paint_dim_1_4():
    x = ['32', '64', '128', '256']
    fig = plt.figure(figsize=(40, 10), dpi=80)
    # 创建子图1
    sub1 = fig.add_subplot(1, 2, 1)
    sub1.plot(x, [0.7037, 0.7442, 0.7558, 0.7671], '^-' ,color='coral', label=r'ConvBERT4Rec@20M', lw=lw, markersize=markersize)
    sub1.plot(x, [0.6818, 0.7180, 0.7499, 0.7588], 's-' ,color='green', label=r'BERT4Rec@20M', lw=lw, markersize=markersize)
    sub1.plot(x, [0.7029, 0.7476, 0.7684, 0.7594], '^--',color='coral', label=r'ConvBERT4Rec@1M', lw=lw, markersize=markersize)
    sub1.plot(x, [0.6740, 0.7171, 0.7489, 0.7513], 's--',color='green', label=r'BERT4Rec@1M', lw=lw, markersize=markersize)
    # sub1.title.set_text('MovieLens 20M')
    sub1.set_xlabel('dim', fontsize=font_size)
    sub1.set_ylabel('Recall@10', fontsize=font_size)
    sub1.tick_params(labelsize=font_size)
    sub1.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub3 = fig.add_subplot(1, 2, 2)
    sub3.plot(x, [0.4523, 0.4926, 0.5073, 0.5216], '^-', color='coral', label=r'ConvBERT4Rec@20M', lw=lw, markersize=markersize)
    sub3.plot(x, [0.4291, 0.4633, 0.5005, 0.5142], 's-', color='green', label=r'BERT4Rec@20M', lw=lw, markersize=markersize)
    sub3.plot(x, [0.4784, 0.5286, 0.5457, 0.5393], '^--',color='coral',  label=r'ConvBERT4Rec@1M', lw=lw, markersize=markersize)
    sub3.plot(x, [0.4193, 0.4866, 0.5367, 0.5402], 's--',color='green',  label=r'BERT4Rec@1M', lw=lw, markersize=markersize)
    # sub3.title.set_text('MovieLens 20M')
    sub3.set_xlabel('dim', fontsize=font_size)
    sub3.set_ylabel('NDCG@10', fontsize=font_size)
    sub3.tick_params(labelsize=font_size)
    sub3.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)


    # 创建子图1
    # sub4 = fig.add_subplot(2, 2, 3)
    # sub4.plot(x, [0.7029, 0.7476, 0.7684, 0.7594], '^-', label=r'ConvBERT4Rec', lw=lw, markersize=markersize)
    # sub4.plot(x, [0.6740, 0.7171, 0.7489, 0.7513], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    # sub4.plot(x, [0.3676, 0.3756, 0.3772, 0.3619], '.-', label=r'ADE', lw=lw, markersize=markersize)
    # sub4.plot(x, [0.3800, 0.3656, 0.3682, 0.3604], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    # sub4.title.set_text('MovieLens 1M')
    # sub4.set_xlabel('dim')
    # sub4.set_ylabel('Recall@10')
    # sub4.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)
    #
    # # 创建子图2
    # sub2 = fig.add_subplot(2, 2, 4)
    # sub2.plot(x, [0.4784, 0.5286, 0.5457, 0.5393], '^-', label=r'ConvBERT4Rec', lw=lw, markersize=markersize)
    # sub2.plot(x, [0.4193, 0.4866, 0.5367, 0.5402], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    # sub2.plot(x, [0.3902, 0.3971, 0.3887, 0.3776], '.-', label=r'ADE', lw=lw, markersize=markersize)
    # sub2.plot(x, [0.3959, 0.3773, 0.3855, 0.3771], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    # sub2.title.set_text('MovieLens 1M')
    # sub2.set_xlabel('dim')
    # sub2.set_ylabel('NDCG@10')
    # sub2.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    plt.legend(loc=0, bbox_to_anchor=(1.05, 1),fontsize=font_size)
    # plt.tight_layout()
    plt.show()


def paint_top_1_4():
    x = ['5', '10', '15', '20', '25']
    fig = plt.figure(figsize=(40, 10), dpi=80)


    # 创建子图1
    sub1 = fig.add_subplot(1, 2, 1)
    sub1.plot(x, [0.6058, 0.7442, 0.8113, 0.8532, 0.8812], '^-', color='coral', label=r'ConvBERT4Rec@20M', lw=lw, markersize=markersize)
    sub1.plot(x, [0.5740, 0.7180, 0.7909, 0.8367, 0.8676], 's-', color='green', label=r'BERT4Rec@20M', lw=lw, markersize=markersize)
    sub1.plot(x, [0.6336, 0.7476, 0.8108, 0.8519, 0.8848], '^--',color='coral',  label=r'ConvBERT4Rec@1M', lw=lw, markersize=markersize)
    sub1.plot(x, [0.5883, 0.7171, 0.7823, 0.8289, 0.8647], 's--',color='green',  label=r'BERT4Rec@1M', lw=lw, markersize=markersize)
    # sub1.title.set_text('MovieLens 20M')
    sub1.set_xlabel('K',fontsize=font_size)
    sub1.set_ylabel('Recall@K',fontsize=font_size)
    sub1.tick_params(labelsize=font_size)
    sub1.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub3 = fig.add_subplot(1, 2, 2)
    sub3.plot(x, [0.4477, 0.4926, 0.5104, 0.5203, 0.5265], '^-', color='coral', label=r'ConvBERT4Rec@20M', lw=lw, markersize=markersize)
    sub3.plot(x, [0.4166, 0.4633, 0.4827, 0.4935, 0.5002], 's-', color='green', label=r'BERT4Rec@20M', lw=lw, markersize=markersize)
    sub3.plot(x, [0.4917, 0.5286, 0.5454, 0.5551, 0.5623], '^--',color='coral',  label=r'ConvBERT4Rec@1M', lw=lw, markersize=markersize)
    sub3.plot(x, [0.4449, 0.4866, 0.5039, 0.5149, 0.5228], 's--',color='green',  label=r'BERT4Rec@1M', lw=lw, markersize=markersize)
    # sub3.title.set_text('MovieLens 20M')
    sub3.set_xlabel('K',fontsize=font_size)
    sub3.set_ylabel('NDCG@K',fontsize=font_size)
    sub3.tick_params(labelsize=font_size)
    sub3.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)
    #
    # # 创建子图1
    # sub4 = fig.add_subplot(2, 2, 3)
    # sub4.plot(x, [0.6336, 0.7476, 0.8108, 0.8519, 0.8848], '^-', label=r'ConvBERT4Rec', lw=lw, markersize=markersize)
    # sub4.plot(x, [0.5883, 0.7171, 0.7823, 0.8289, 0.8647], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    # sub4.plot(x, [0.4034, 0.3756, 0.3744, 0.3774, 0.3908], '.-', label=r'ADE', lw=lw, markersize=markersize)
    # sub4.plot(x, [0.3861, 0.3656, 0.3704, 0.3845, 0.4015], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    # sub4.title.set_text('MovieLens 1M')
    # sub4.set_xlabel('K')
    # sub4.set_ylabel('Recall@K')
    # sub4.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)
    #
    # # 创建子图2
    # sub2 = fig.add_subplot(2, 2, 4)
    # sub2.plot(x, [0.4917, 0.5286, 0.5454, 0.5551, 0.5623], '^-', label=r'ConvBERT4Rec', lw=lw, markersize=markersize)
    # sub2.plot(x, [0.4449, 0.4866, 0.5039, 0.5149, 0.5228], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    # sub2.plot(x, [0.4246, 0.3971, 0.3875, 0.3830, 0.3853], '.-', label=r'ADE', lw=lw, markersize=markersize)
    # sub2.plot(x, [0.3978, 0.3773, 0.3728, 0.3749, 0.3802], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    # sub2.title.set_text('MovieLens 1M')
    # sub2.set_xlabel('K')
    # sub2.set_ylabel('NDCG@K')
    # sub2.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    plt.legend(loc=0, bbox_to_anchor=(1.05, 1),fontsize=font_size)
    # plt.tight_layout()
    plt.show()

font_size=17

if __name__ == '__main__':
    data_load_bert_cnn()
    # data_load()
