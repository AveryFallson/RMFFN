import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import torch
import open3d as o3d


def pca(X, d):
    # Centralization
    means = np.mean(X, 0)
    X = X - means
    # Covariance Matrix
    covM = np.dot(X.T, X)
    eigval, eigvec = np.linalg.eig(covM)
    indexes = np.argsort(eigval)[-d:]
    W = eigvec[:, indexes]
    return np.dot(X, W)

def randomcolor():
    colorArr = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 15)]
    return "#" + color



def tsne(fts):
    iris = load_iris()  # 使用sklearn自带的测试文件
    iris['data'] = fts
    new_w = TSNE(n_components=2, learning_rate=10, init='pca', random_state=12).fit_transform(iris.data)
    return new_w


if __name__ == '__main__':

    # --------------data-input-------------------

    a = np.load("tsne_train_ke.npy", allow_pickle=True).item()

    fts_q = a['fts']    # (object_num, dim)
    las_q = a['las']    # (object_num)  label需要从小到大排序
    mode = "tsne"       # pca, load, tsne
    dim = "2d"          # 2d or 3d

    # -------------------------------------------
    spl = [0]
    spl_count = 0
    spl_count_15 = 0
    spl_15 = [0]
    categories = np.max(las_q) + 1
    # g_list = [0, 2, 4, 7, 8, 12, 14, 17, 26, 33, 1, 5, 21, 23]
    # g_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    g_list = range(40)
    # color_slt = ['#FCCF42', '#D99F38', '#F09D4A', '#D96E38', '#FA5939', '#FC3632', '#D92B86', '#E63BF0', '#9A2BD9',
    #              '#6E2AFA', '#324AFC', '#2B73D9', '#3DBDF0', '#2BD9D6', '#2AFAB3']
    color_slt = ['#726684', '#0948D0', '#7CF21C', '#75E7AC', '#5B2F61', '#F7B610', '#170468', '#C5044D', '#9D7DB3',
                 '#225E7C', '#D5056F', '#ECFB2C', '#578873', '#2C1D17', '#E61411', '#CDCB9C', '#63A85D', '#A5E191',
                 '#E2B383', '#E73B6E', '#67A197', '#6084EC', '#D30FFD', '#BABA71', '#07CE41', '#2D835B', '#1A43B4',
                 '#DD7F7F', '#9757AD', '#C011CB', '#0C2558', '#B5DD1E', '#AB65F7', '#D758CF', '#02A448', '#898E7A',
                 '#D779D5', '#76BC80', '#250F3F', '#E42CDA']

    for i in range(categories):
        spl_count = spl_count + list(las_q).count(i)
        spl.append(spl_count)

        if g_list.count(i) > 0:
            spl_count_15 += list(las_q).count(i)
            spl_15.append(spl_count_15)

            if i == 0:
                fts_q_15 = fts_q[spl[i]:spl[i + 1],:]
                las_q_15 = las_q[spl[i]:spl[i + 1]]
            else:
                fts_q_15 = np.concatenate((fts_q_15, fts_q[spl[i]:spl[i + 1],:]), 0)
                las_q_15 = np.concatenate((las_q_15, las_q[spl[i]:spl[i + 1]]), 0)

    if mode == "pca":
        new_w = pca(fts_q, int(dim[0]))
    else:
        if mode == "tsne":
            print(fts_q_15.shape)
            print(las_q_15.shape)
            iris = load_iris()  # 使用sklearn自带的测试文件
            iris['data'] = fts_q_15
            iris['target'] = las_q_15
            new_w = TSNE(n_components=int(dim[0]), learning_rate=10, init='pca', random_state=12).fit_transform(iris.data)
            print(new_w.shape)
            exit()
            np.save('tsne.npy', new_w)
        else:
            if mode == "load":
                new_w = np.load('tsne.npy')
            else:
                print("pca, load or tsne")

    # new_w = np.load('tsne.npy')
    # print(new_w.shape)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(new_w)
    # o3d.io.write_point_cloud('tsne.ply', pcd)
    # exit()


    if dim == "3d":  # 3d图像
        ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
        ax.set_title('t-SNE')  # 设置本图名称
        color_list = []

        # for i in range(categories):
        # unsup: 3 16 22 27
        # norm: 1 5 10 11 21 23 24
        # small: 6 9 13 15 18 19
        # [0, 2, 4, 7, 8, 12, 14, 17, 26, 33]
        for i in range(len(spl_15)-1):
            x = new_w[spl_15[i]:spl_15[i + 1], 0]
            y = new_w[spl_15[i]:spl_15[i + 1], 1]
            z = new_w[spl_15[i]:spl_15[i + 1], 2]
            ax.scatter(x, y, z, c=color_slt[i], marker='.', linewidths=0.0)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色,或随机上色
            # ax.scatter(x, y, z, c=sec_color)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色,或随机上色
        ax.set_xlabel('X')  # 设置x坐标轴
        ax.set_ylabel('Y')  # 设置y坐标轴
        ax.set_zlabel('Z')  # 设置z坐标轴
        plt.show()

    else:  # 2d图像
        plt.figure(figsize=(12, 12))
        print(new_w.shape)
        print(spl_15)
        # color_slt = []
        for i in range(len(spl_15)-1):
            x = new_w[spl_15[i]:spl_15[i + 1], 0]
            y = new_w[spl_15[i]:spl_15[i + 1], 1]
            plt.scatter(x, y, c=color_slt[i], marker='.', linewidths=0.0)
            # current_color = randomcolor()
            # color_slt.append(current_color)
            # plt.scatter(x, y, c=current_color, marker='.', linewidths=0.5)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色,或随机上色
        print(color_slt)
        plt.colorbar()
        plt.show()
