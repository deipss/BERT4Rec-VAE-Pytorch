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


def paint_dim_1_4():
    x = ['32', '64', '128', '256']
    fig = plt.figure(figsize=(40, 10), dpi=80)
    # 创建子图1
    sub1 = fig.add_subplot(1, 4, 1)
    sub1.plot(x, [0.7037, 0.7442, 0.7558, 0.7671], '^-', label=r'BERT_CNN4Rec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.6818, 0.7180, 0.7499, 0.7588], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.3153, 0.3159, 0.3160, 0.3079], '.-', label=r'ADE', lw=lw, markersize=markersize)
    sub1.plot(x, [0.2977, 0.2952, 0.2836, 0.2763], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    sub1.title.set_text('MovieLens 20M')
    sub1.set_xlabel('dim')
    sub1.set_ylabel('HR@10')
    sub1.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub3 = fig.add_subplot(1, 4, 2)
    sub3.plot(x, [0.4523, 0.4926, 0.5073, 0.5216], '^-', label=r'BERT_CNN4Rec', lw=lw, markersize=markersize)
    sub3.plot(x, [0.4291, 0.4633, 0.5005, 0.5142], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub3.plot(x, [0.2981, 0.2980, 0.2990, 0.2910], '.-', label=r'ADE', lw=lw, markersize=markersize)
    sub3.plot(x, [0.2829, 0.2809, 0.2685, 0.2626], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    sub3.title.set_text('MovieLens 20M')
    sub3.set_xlabel('dim')
    sub3.set_ylabel('NDCG@10')
    sub3.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图1
    sub4 = fig.add_subplot(1, 4, 3)
    sub4.plot(x, [0.7029 , 0.7476 , 0.7684 , 0.7594 ], '^-', label=r'BERT_CNN4Rec', lw=lw, markersize=markersize)
    sub4.plot(x, [0.6740 , 0.7171 , 0.7489 , 0.7513 ], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub4.plot(x, [0.3676 , 0.3756 , 0.3772 , 0.3619 ], '.-', label=r'ADE', lw=lw, markersize=markersize)
    sub4.plot(x, [0.3800 , 0.3656 , 0.3682 , 0.3604 ], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    sub4.title.set_text('MovieLens 1M')
    sub4.set_xlabel('dim')
    sub4.set_ylabel('HR@10')
    sub4.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub2 = fig.add_subplot(1, 4, 4)
    sub2.plot(x, [0.4784 , 0.5286 , 0.5457 , 0.5393 ], '^-', label=r'BERT_CNN4Rec', lw=lw, markersize=markersize)
    sub2.plot(x, [0.4193 , 0.4866 , 0.5367 , 0.5402 ], 's-', label=r'BERT4Rec', lw=lw, markersize=markersize)
    sub2.plot(x, [0.3902 , 0.3971 , 0.3887 , 0.3776 ], '.-', label=r'ADE', lw=lw, markersize=markersize)
    sub2.plot(x, [0.3959 , 0.3773 , 0.3855 , 0.3771 ], 'x-', label=r'VDE', lw=lw, markersize=markersize)
    sub2.title.set_text('MovieLens 1M')
    sub2.set_xlabel('dim')
    sub2.set_ylabel('NDCG@10')
    sub2.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    plt.legend(loc=0, bbox_to_anchor=(1.05, 1))
    # plt.tight_layout()
    plt.show()


def paint_topK_1_4():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fig = plt.figure(figsize=(40, 10), dpi=80)
    lw = 1.7
    markersize = 6.7
    # 创建子图1
    sub1 = fig.add_subplot(1, 4, 1)
    sub1.plot(x, [0.0348, 0.0662, 0.0967, 0.1298, 0.1582, 0.1849, 0.2045, 0.2331, 0.251, 0.2742], '^-',
              label=r'pi2v@$\alpha$=0.2', lw=lw, markersize=markersize)
    sub1.plot(x, [0.0428, 0.0849, 0.1238, 0.154, 0.1849, 0.2179, 0.2411, 0.2721, 0.2882, 0.3103], 's-',
              label=r'pi2v@$\alpha$=0.4', lw=lw, markersize=markersize)
    sub1.plot(x, [0.0419, 0.0823, 0.119, 0.1497, 0.1806, 0.2114, 0.2423, 0.2595, 0.2856, 0.3154], '.-',
              label=r'pi2v@$\alpha$=0.6', lw=lw, markersize=markersize)
    sub1.plot(x, [0.033, 0.064, 0.0976, 0.1243, 0.1512, 0.1771, 0.2017, 0.2239, 0.2444, 0.2701], 'x-',
              label=r'pi2v@$\alpha$=0.8', lw=lw, markersize=markersize)
    sub1.plot(x, [0.0375, 0.083, 0.1135, 0.1406, 0.1807, 0.2017, 0.231, 0.2527, 0.2692, 0.2914], '1--',
              label=r'Item2vec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.0428, 0.0811, 0.1222, 0.1552, 0.1885, 0.2232, 0.2489, 0.2708, 0.2838, 0.3075], '<--',
              label=r'User2vec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.0287, 0.0605, 0.0879, 0.1169, 0.1414, 0.1712, 0.1967, 0.2326, 0.2494, 0.2793], 'o--',
              label=r'Prod2vec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.0135, 0.0274, 0.0432, 0.0616, 0.0814, 0.1035, 0.1273, 0.1541, 0.1822, 0.2127], '+--', label=r'SVD',
              lw=lw, markersize=markersize)
    sub1.title.set_text('MovieLens 100K')
    sub1.set_xlabel('top K')
    sub1.set_ylabel('HR@10')
    sub1.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub2 = fig.add_subplot(1, 4, 2)
    sub2.plot(x, [0.0348, 0.0542, 0.0694, 0.0846, 0.0957, 0.1054, 0.1106, 0.122, 0.1259, 0.1325], '^-',
              label=r'pi2v@$\alpha$=0.2', lw=lw, markersize=markersize)
    sub2.plot(x, [0.0428, 0.0695, 0.0896, 0.1021, 0.1136, 0.1262, 0.133, 0.1446, 0.1468, 0.1545], 's-',
              label=r'pi2v@$\alpha$=0.4', lw=lw, markersize=markersize)
    sub2.plot(x, [0.0419, 0.068, 0.0861, 0.098, 0.1104, 0.1216, 0.1328, 0.135, 0.1444, 0.1552], '.-',
              label=r'pi2v@$\alpha$=0.6', lw=lw, markersize=markersize)
    sub2.plot(x, [0.033, 0.0525, 0.0696, 0.0804, 0.0901, 0.1, 0.1078, 0.1159, 0.1194, 0.1296], 'x-',
              label=r'pi2v@$\alpha$=0.8', lw=lw, markersize=markersize)
    sub2.plot(x, [0.0375, 0.0676, 0.0816, 0.0925, 0.1107, 0.1171, 0.1283, 0.1343, 0.139, 0.1455], '1--',
              label=r'Item2vec', lw=lw, markersize=markersize)
    sub2.plot(x, [0.0428, 0.0662, 0.0878, 0.1016, 0.1161, 0.1296, 0.1369, 0.1433, 0.1495, 0.1562], '<--',
              label=r'User2vec', lw=lw, markersize=markersize)
    sub2.plot(x, [0.0287, 0.0492, 0.0621, 0.0755, 0.0831, 0.0951, 0.103, 0.1165, 0.1198, 0.129], 'o--',
              label=r'Prod2vec', lw=lw, markersize=markersize)
    sub2.plot(x, [0.0135, 0.0223, 0.0302, 0.0381, 0.0458, 0.0536, 0.0616, 0.0700, 0.0785, 0.0873], '+--', label=r'SVD',
              lw=lw, markersize=markersize)
    sub2.title.set_text('MovieLens 100K')
    sub2.set_xlabel('top K')
    sub2.set_ylabel('NDCG@10')
    sub2.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图3
    sub3 = fig.add_subplot(1, 4, 3)
    sub3.plot(x, np.random.rand(10), '^--', label=r'pi2v@$\alpha$=0.2', lw=lw, markersize=markersize)
    sub3.plot(x, np.random.rand(10), 's--', label=r'pi2v@$\alpha$=0.4', lw=lw, markersize=markersize)
    sub3.plot(x, np.random.rand(10), '.--', label=r'pi2v@$\alpha$=0.6', lw=lw, markersize=markersize)
    sub3.plot(x, np.random.rand(10), 'x--', label=r'pi2v@$\alpha$=0.8', lw=lw, markersize=markersize)
    sub3.plot(x, np.random.rand(10), '1--', label=r'item2vec', lw=lw, markersize=markersize)
    sub3.title.set_text('MovieLens 1M')
    sub3.set_xlabel('top K')
    sub3.set_ylabel('HR@10')
    sub3.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图4
    sub4 = fig.add_subplot(1, 4, 4)
    sub4.plot(x, np.random.rand(10), '^--', label=r'pi2v@$\alpha$=0.2', lw=lw, markersize=markersize)
    sub4.plot(x, np.random.rand(10), 's--', label=r'pi2v@$\alpha$=0.4', lw=lw, markersize=markersize)
    sub4.plot(x, np.random.rand(10), '.--', label=r'pi2v@$\alpha$=0.6', lw=lw, markersize=markersize)
    sub4.plot(x, np.random.rand(10), 'x--', label=r'pi2v@$\alpha$=0.8', lw=lw, markersize=markersize)
    sub4.plot(x, np.random.rand(10), '1--', label=r'item2vec', lw=lw, markersize=markersize)
    sub4.title.set_text('MovieLens 1M')
    sub4.set_xlabel('top K')
    sub4.set_ylabel('HR@10')
    sub4.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    plt.legend(loc=0, bbox_to_anchor=(1.05, 1))
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    paint_dim_1_4()
    # data_load()
