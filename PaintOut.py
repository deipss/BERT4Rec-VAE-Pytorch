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
            if '20m' in sub_dir and '_bert_' in sub_dir:
                data, data_config = {}, {}
                data_file = open(path + '/logs/' + 'test_metrics.json')
                data = json.load(data_file)
                data_file = open(path + '/' + 'config.json')
                data_config = json.load(data_file)
                print('%12s\t%3s\t%10s\t%d\t%d\t%s'
                      % (data_config['model_code'], data_config['dim'], data_config['dataset_code'],
                         data_config['kernel_size'], data_config['stride'], data_to_line(data)))


def paint_dim_1_4():
    x = ['32', '64', '128', '256']
    fig = plt.figure(figsize=(40, 10), dpi=80)
    # 创建子图1
    sub1 = fig.add_subplot(2, 2, 1)
    sub1.plot(x, [0.7037, 0.7442, 0.7558, 0.7671], '^-', label=r'BERT_CNN4Rec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.6818, 0.7180, 0.7499, 0.7588], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.3153, 0.3159, 0.3160, 0.3079], '.-', label=r'ADE', lw=lw, markersize=markersize)
    sub1.plot(x, [0.2977, 0.2952, 0.2836, 0.2763], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    sub1.title.set_text('MovieLens 20M')
    sub1.set_xlabel('dim')
    sub1.set_ylabel('HR@10')
    sub1.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub3 = fig.add_subplot(2, 2, 2)
    sub3.plot(x, [0.4523, 0.4926, 0.5073, 0.5216], '^-', label=r'BERT_CNN4Rec', lw=lw, markersize=markersize)
    sub3.plot(x, [0.4291, 0.4633, 0.5005, 0.5142], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub3.plot(x, [0.2981, 0.2980, 0.2990, 0.2910], '.-', label=r'ADE', lw=lw, markersize=markersize)
    sub3.plot(x, [0.2829, 0.2809, 0.2685, 0.2626], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    sub3.title.set_text('MovieLens 20M')
    sub3.set_xlabel('dim')
    sub3.set_ylabel('NDCG@10')
    sub3.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图1
    sub4 = fig.add_subplot(2, 2, 3)
    sub4.plot(x, [0.7029, 0.7476, 0.7684, 0.7594], '^-', label=r'BERT_CNN4Rec', lw=lw, markersize=markersize)
    sub4.plot(x, [0.6740, 0.7171, 0.7489, 0.7513], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub4.plot(x, [0.3676, 0.3756, 0.3772, 0.3619], '.-', label=r'ADE', lw=lw, markersize=markersize)
    sub4.plot(x, [0.3800, 0.3656, 0.3682, 0.3604], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    sub4.title.set_text('MovieLens 1M')
    sub4.set_xlabel('dim')
    sub4.set_ylabel('HR@10')
    sub4.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub2 = fig.add_subplot(2, 2, 4)
    sub2.plot(x, [0.4784, 0.5286, 0.5457, 0.5393], '^-', label=r'BERT_CNN4Rec', lw=lw, markersize=markersize)
    sub2.plot(x, [0.4193, 0.4866, 0.5367, 0.5402], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub2.plot(x, [0.3902, 0.3971, 0.3887, 0.3776], '.-', label=r'ADE', lw=lw, markersize=markersize)
    sub2.plot(x, [0.3959, 0.3773, 0.3855, 0.3771], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    sub2.title.set_text('MovieLens 1M')
    sub2.set_xlabel('dim')
    sub2.set_ylabel('NDCG@10')
    sub2.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    plt.legend(loc=0, bbox_to_anchor=(1.05, 1))
    # plt.tight_layout()
    plt.show()


def paint_top_1_4():
    x = ['5', '10', '15', '20', '25']
    fig = plt.figure(figsize=(40, 10), dpi=80)
    # 创建子图1
    sub1 = fig.add_subplot(2, 2, 1)
    sub1.plot(x, [0.6058, 0.7442, 0.8113, 0.8532, 0.8812], '^-', label=r'BERT_CNN4Rec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.5740, 0.7180, 0.7909, 0.8367, 0.8676], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.2909, 0.3159, 0.3487, 0.3805, 0.4087], '.-', label=r'ADE', lw=lw, markersize=markersize)
    sub1.plot(x, [0.2761, 0.2952, 0.3244, 0.3542, 0.3823], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    sub1.title.set_text('MovieLens 20M')
    sub1.set_xlabel('K')
    sub1.set_ylabel('HR@K')
    sub1.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub3 = fig.add_subplot(2, 2, 2)
    sub3.plot(x, [0.4477, 0.4926, 0.5104, 0.5203, 0.5265], '^-', label=r'BERT_CNN4Rec', lw=lw, markersize=markersize)
    sub3.plot(x, [0.4166, 0.4633, 0.4827, 0.4935, 0.5002], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub3.plot(x, [0.2955, 0.2980, 0.3067, 0.3164, 0.3258], '.-', label=r'ADE', lw=lw, markersize=markersize)
    sub3.plot(x, [0.2814, 0.2809, 0.2881, 0.2969, 0.3060], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    sub3.title.set_text('MovieLens 20M')
    sub3.set_xlabel('K')
    sub3.set_ylabel('NDCG@K')
    sub3.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图1
    sub4 = fig.add_subplot(2, 2, 3)
    sub4.plot(x, [0.6336, 0.7476, 0.8108, 0.8519, 0.8848], '^-', label=r'BERT_CNN4Rec', lw=lw, markersize=markersize)
    sub4.plot(x, [0.5883, 0.7171, 0.7823, 0.8289, 0.8647], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub4.plot(x, [0.4034, 0.3756, 0.3744, 0.3774, 0.3908], '.-', label=r'ADE', lw=lw, markersize=markersize)
    sub4.plot(x, [0.3861, 0.3656, 0.3704, 0.3845, 0.4015], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    sub4.title.set_text('MovieLens 1M')
    sub4.set_xlabel('K')
    sub4.set_ylabel('HR@K')
    sub4.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub2 = fig.add_subplot(2, 2, 4)
    sub2.plot(x, [0.4917, 0.5286, 0.5454, 0.5551, 0.5623], '^-', label=r'BERT_CNN4Rec', lw=lw, markersize=markersize)
    sub2.plot(x, [0.4449, 0.4866, 0.5039, 0.5149, 0.5228], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub2.plot(x, [0.4246, 0.3971, 0.3875, 0.3830, 0.3853], '.-', label=r'ADE', lw=lw, markersize=markersize)
    sub2.plot(x, [0.3978, 0.3773, 0.3728, 0.3749, 0.3802], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    sub2.title.set_text('MovieLens 1M')
    sub2.set_xlabel('K')
    sub2.set_ylabel('NDCG@K')
    sub2.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    plt.legend(loc=0, bbox_to_anchor=(1.05, 1))
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    paint_dim_1_4()
    # data_load()
