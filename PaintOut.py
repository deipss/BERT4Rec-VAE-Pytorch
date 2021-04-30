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
    sub1.plot(x, [0.2212, 0.2473, 0.2535, 0.2772], '^-', label=r'pi2v@$\alpha$=0.2', lw=lw, markersize=markersize)
    sub1.plot(x, [0.2647, 0.2885, 0.297, 0.314], 's-', label=r'pi2v@$\alpha$=0.4', lw=lw, markersize=markersize)
    sub1.plot(x, [0.2545, 0.278, 0.3023, 0.3139], '.-', label=r'pi2v@$\alpha$=0.6', lw=lw, markersize=markersize)
    sub1.plot(x, [0.2213, 0.2388, 0.2549, 0.2693], 'x-', label=r'pi2v@$\alpha$=0.8', lw=lw, markersize=markersize)
    sub1.plot(x, [0.2551, 0.2656, 0.2875, 0.2912], '1--', label=r'Item2vec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.245, 0.2743, 0.2913, 0.307], '<--', label=r'User2vec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.1966, 0.2215, 0.2438, 0.2721], 'o--', label=r'Prod2vec', lw=lw, markersize=markersize)
    sub1.plot(x, [0.2127, 0.2127, 0.2127, 0.2127], '+--', label=r'SVD', lw=lw, markersize=markersize)
    sub1.title.set_text('MovieLens 100K')
    sub1.set_xlabel('dim')
    sub1.set_ylabel('HR@10')
    sub1.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图2
    sub2 = fig.add_subplot(1, 4, 2)
    sub2.plot(x, [0.1047, 0.1164, 0.1212, 0.1347], '^-', label=r'pi2v@$\alpha$=0.2', lw=lw, markersize=markersize)
    sub2.plot(x, [0.1253, 0.1407, 0.1457, 0.1558], 's-', label=r'pi2v@$\alpha$=0.4', lw=lw, markersize=markersize)
    sub2.plot(x, [0.1173, 0.1313, 0.1454, 0.1529], '.-', label=r'pi2v@$\alpha$=0.6', lw=lw, markersize=markersize)
    sub2.plot(x, [0.1023, 0.1119, 0.1204, 0.1289], 'x-', label=r'pi2v@$\alpha$=0.8', lw=lw, markersize=markersize)
    sub2.plot(x, [0.1225, 0.1294, 0.1409, 0.1447], '1--', label=r'Item2vec', lw=lw, markersize=markersize)
    sub2.plot(x, [0.1155, 0.1319, 0.1409, 0.1545], '<--', label=r'User2vec', lw=lw, markersize=markersize)
    sub2.plot(x, [0.0887, 0.1012, 0.1115, 0.1256], 'o--', label=r'Prod2vec', lw=lw, markersize=markersize)
    sub2.plot(x, [0.0873, 0.0873, 0.0873, 0.0873], '+--', label=r'SVD', lw=lw, markersize=markersize)
    sub2.title.set_text('MovieLens 100K')
    sub2.set_xlabel('dim')
    sub2.set_ylabel('NDCG@10')
    sub2.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图3
    sub3 = fig.add_subplot(1, 4, 3)
    sub3.plot(x, np.random.rand(4), '^--', label=r'pi2v@$\alpha$=0.2', lw=lw, markersize=markersize)
    sub3.plot(x, np.random.rand(4), 's--', label=r'pi2v@$\alpha$=0.4', lw=lw, markersize=markersize)
    sub3.plot(x, np.random.rand(4), '.--', label=r'pi2v@$\alpha$=0.6', lw=lw, markersize=markersize)
    sub3.plot(x, np.random.rand(4), 'x--', label=r'pi2v@$\alpha$=0.8', lw=lw, markersize=markersize)
    sub3.plot(x, np.random.rand(4), '1--', label=r'item2vec', lw=lw, markersize=markersize)
    sub3.title.set_text('MovieLens 1M')
    sub3.set_xlabel('dim')
    sub3.set_ylabel('HR@10')
    sub3.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

    # 创建子图4
    sub4 = fig.add_subplot(1, 4, 4)
    sub4.plot(x, np.random.rand(4), '^--', label=r'pi2v@$\alpha$=0.2', lw=lw, markersize=markersize)
    sub4.plot(x, np.random.rand(4), 's--', label=r'pi2v@$\alpha$=0.4', lw=lw, markersize=markersize)
    sub4.plot(x, np.random.rand(4), '.--', label=r'pi2v@$\alpha$=0.6', lw=lw, markersize=markersize)
    sub4.plot(x, np.random.rand(4), 'x--', label=r'pi2v@$\alpha$=0.8', lw=lw, markersize=markersize)
    sub4.plot(x, np.random.rand(4), '1--', label=r'item2vec', lw=lw, markersize=markersize)
    sub4.title.set_text('MovieLens 1M')
    sub4.set_xlabel('dim')
    sub4.set_ylabel('NDCG@10')
    sub4.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.2)

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
    # paint_topK_1_4()
    data_load()
