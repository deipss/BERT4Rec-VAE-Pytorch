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
            if 1 ==1  and '05-16' in sub_dir:
                try:
                    data, data_config = {}, {}
                    data_file = open(path + '/logs/' + 'test_metrics.json')
                    data = json.load(data_file)
                    data_file = open(path + '/' + 'config.json')
                    data_config = json.load(data_file)
                    if data_config['blocks_1m_test'] == False:
                        continue
                    print('%12s\t%3s\t%10s\t%d\t%d\t%s'
                          % ('Meta'+data_config['model_code'], data_config['dim'], data_config['dataset_code'],
                             data_config['kernel_size'], data_config['stride'], data_to_line(data)))
                except BaseException:
                    print()

def paint_dim_1_4():
    x = ['32', '64', '128', '256']
    fig = plt.figure(figsize=(40, 10), dpi=80)

    # 创建子图1
    sub1 = fig.add_subplot(2, 2, 1)
    sub1.plot(x, [0.7144 ,0.7515 ,0.7729 ,0.7647 ], '^-', color='black', label=r'CM-BERT4Rec', lw=lw,              markersize=markersize)
    sub1.plot(x, [0.6981 ,0.7310 ,0.7496 ,0.7555 ], 's-', color='blue', label=r'MetaBERT4Rec', lw=lw,              markersize=markersize)
    sub1.plot(x, [0.7029 ,0.7476 ,0.7684 ,0.7594 ], 'o-', color='coral', label=r'ConvBERT4Rec', lw=lw,              markersize=markersize)
    sub1.plot(x, [0.6740 ,0.7171 ,0.7489 ,0.7513 ], '1--', color='red', label=r'BERT4Rec', lw=lw,              markersize=markersize)
    sub1.title.set_text('MovieLens 1M')
    sub1.set_xlabel('dim', fontsize=font_size)
    sub1.set_ylabel('Recall@10', fontsize=font_size)
    sub1.tick_params(labelsize=font_size)
    sub1.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub3 = fig.add_subplot(2, 2, 2)
    sub3.plot(x, [0.4963 ,0.5305 ,0.5495 ,0.5423 ], '^-', color='black', label=r'CM-BERT4Rec', lw=lw,              markersize=markersize)
    sub3.plot(x, [0.4511 ,0.5066 ,0.5376 ,0.5438 ], 's-', color='blue', label=r'MetaBERT4Rec', lw=lw,              markersize=markersize)
    sub3.plot(x, [0.4784 ,0.5286 ,0.5457 ,0.5393 ], 'o-', color='coral', label=r'ConvBERT4Rec', lw=lw,              markersize=markersize)
    sub3.plot(x, [0.4193 ,0.4866 ,0.5367 ,0.5402 ], '1--', color='red', label=r'BERT4Rec', lw=lw,              markersize=markersize)
    sub3.title.set_text('MovieLens 1M')
    sub3.set_xlabel('dim', fontsize=font_size)
    sub3.set_ylabel('NDCG@10', fontsize=font_size)
    sub3.tick_params(labelsize=font_size)
    sub3.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图1
    sub2 = fig.add_subplot(2, 2, 3)
    sub2.plot(x, [0.7234 ,0.7424 ,0.7558 ,0.7608 ], '^-', color='black', label=r'CM-BERT4Rec', lw=lw,              markersize=markersize)
    sub2.plot(x, [0.7073 ,0.7326 ,0.7495 ,0.7613 ], 's-', color='blue', label=r'MetaBERT4Rec', lw=lw,              markersize=markersize)
    sub2.plot(x, [0.7037 ,0.7442 ,0.7558 ,0.7671 ], 'o-', color='coral', label=r'ConvBERT4Rec', lw=lw,              markersize=markersize)
    sub2.plot(x, [0.6818 ,0.7180 ,0.7499 ,0.7588 ], '1--', color='red', label=r'BERT4Rec', lw=lw,              markersize=markersize)
    sub2.title.set_text('MovieLens 20M')
    sub2.set_xlabel('dim', fontsize=font_size)
    sub2.set_ylabel('Recall@10', fontsize=font_size)
    sub2.tick_params(labelsize=font_size)
    sub2.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub4 = fig.add_subplot(2, 2, 4)
    sub4.plot(x, [0.4709 ,0.4905 ,0.5066 ,0.5152 ], '^-', color='black', label=r'CM-BERT4Rec', lw=lw,              markersize=markersize)
    sub4.plot(x, [0.4512 ,0.4798 ,0.4992 ,0.5148 ], 's-', color='blue', label=r'MetaBERT4Rec', lw=lw,              markersize=markersize)
    sub4.plot(x, [0.4523 ,0.4926 ,0.5073 ,0.5216 ], 'o-', color='coral', label=r'ConvBERT4Rec', lw=lw,              markersize=markersize)
    sub4.plot(x, [0.4291 ,0.4633 ,0.5005 ,0.5142 ], '1--', color='red', label=r'BERT4Rec', lw=lw,              markersize=markersize)
    sub4.title.set_text('MovieLens 20M')
    sub4.set_xlabel('dim', fontsize=font_size)
    sub4.set_ylabel('NDCG@10', fontsize=font_size)
    sub4.tick_params(labelsize=font_size)
    sub4.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    plt.legend(loc=0, bbox_to_anchor=(1.05, 1), fontsize=font_size)
    # plt.tight_layout()
    plt.show()


def paint_top_1_4():
    x = ['5', '10', '15', '20', '25']
    fig = plt.figure(figsize=(40, 10), dpi=80)


    # 创建子图1
    sub1 = fig.add_subplot(2, 2, 1)
    sub1.plot(x, [0.6607 ,0.7729 ,0.8293 ,0.8665 ,0.8940 ], '^-', color='black', label=r'CM-BERT4Rec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.6462 ,0.7496 ,0.8081 ,0.8466 ,0.8768 ], 's-', color='blue', label=r'MetaBERT4Rec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.6565 ,0.7684 ,0.8293 ,0.8658 ,0.8954 ], 'o-',color='coral',  label=r'ConvBERT4Rec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.6414 ,0.7489 ,0.8116 ,0.8548 ,0.8802], '1--',color='red',  label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub1.title.set_text('MovieLens 1M')
    sub1.set_xlabel('K',fontsize=font_size)
    sub1.set_ylabel('Recall@K',fontsize=font_size)
    sub1.tick_params(labelsize=font_size)
    sub1.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub3 = fig.add_subplot(2, 2, 2)
    sub3.plot(x, [0.5132 ,0.5495 ,0.5645 ,0.5732 ,0.5793 ], '^-', color='black', label=r'CM-BERT4Rec', lw=lw, markersize=markersize)
    sub3.plot(x, [0.5040 ,0.5376 ,0.5530 ,0.5621 ,0.5687 ], 's-', color='blue', label=r'MetaBERT4Rec', lw=lw, markersize=markersize)
    sub3.plot(x, [0.5094 ,0.5457 ,0.5618 ,0.5704 ,0.5769 ], 'o-',color='coral',  label=r'ConvBERT4Rec', lw=lw, markersize=markersize)
    sub3.plot(x, [0.5019 ,0.5367 ,0.5534 ,0.5636 ,0.5692 ], '1--',color='red',  label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub3.title.set_text('MovieLens 1M')
    sub3.set_xlabel('K',fontsize=font_size)
    sub3.set_ylabel('NDCG@K',fontsize=font_size)
    sub3.tick_params(labelsize=font_size)
    sub3.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图1
    sub2 = fig.add_subplot(2, 2, 3)
    sub2.plot(x, [0.6203 ,0.7558 ,0.8213 ,0.8610 ,0.8876 ], '^-', color='black', label=r'CM-BERT4Rec', lw=lw,              markersize=markersize)
    sub2.plot(x, [0.6141 ,0.7495 ,0.8148 ,0.8538 ,0.8820 ], 's-', color='blue', label=r'MetaBERT4Rec', lw=lw,              markersize=markersize)
    sub2.plot(x, [0.6221 ,0.7558 ,0.8214 ,0.8606 ,0.8868 ], 'o-', color='coral', label=r'ConvBERT4Rec', lw=lw,              markersize=markersize)
    sub2.plot(x, [0.6153 ,0.7499 ,0.8161 ,0.8557 ,0.8828 ], '1--', color='red', label=r'BERT4Rec', lw=lw,              markersize=markersize)
    sub2.title.set_text('MovieLens 20M')
    sub2.set_xlabel('K', fontsize=font_size)
    sub2.set_ylabel('Recall@K', fontsize=font_size)
    sub2.tick_params(labelsize=font_size)
    sub2.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub4 = fig.add_subplot(2, 2, 4)
    sub4.plot(x, [0.4626 ,0.5066 ,0.5240 ,0.5333 ,0.5392 ], '^-', color='black', label=r'CM-BERT4Rec', lw=lw,              markersize=markersize)
    sub4.plot(x, [0.4552 ,0.4992 ,0.5165 ,0.5258 ,0.5319 ], 's-', color='blue', label=r'MetaBERT4Rec', lw=lw,              markersize=markersize)
    sub4.plot(x, [0.4639 ,0.5073 ,0.5247 ,0.5340 ,0.5397 ], 'o-', color='coral', label=r'ConvBERT4Rec', lw=lw,              markersize=markersize)
    sub4.plot(x, [0.4568 ,0.5005 ,0.5181 ,0.5275 ,0.5334 ], '1--', color='red', label=r'BERT4Rec', lw=lw,              markersize=markersize)
    sub4.title.set_text('MovieLens 20M')
    sub4.set_xlabel('K', fontsize=font_size)
    sub4.set_ylabel('NDCG@K', fontsize=font_size)
    sub4.tick_params(labelsize=font_size)
    sub4.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    plt.legend(loc=0, bbox_to_anchor=(1.05, 1),fontsize=font_size)
    # plt.tight_layout()
    plt.show()

font_size=17

if __name__ == '__main__':
    # paint_dim_1_4()
    #
    data_load_bert_cnn()
