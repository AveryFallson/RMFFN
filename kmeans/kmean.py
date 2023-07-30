import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import sklearn.metrics as metrics
import time
import scipy.spatial
import os
import shutil
from pathlib import Path
from utils import *
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris

# save_path = Path('/home/dh/kgm/MN40-clust')
save_path = Path('/home/dh/kgm/mn40_ke/TTrain')
# load_path = Path('/home/dh/kgm/OS-MN40')

# def create_file(name, label):
#     if os.path.exists(load_path/'query'/str(name)):
#         shutil.copytree(load_path/'query'/str(name), save_path/str(label)/str(name))
#     if os.path.exists(load_path/'target'/str(name)):
#         shutil.copytree(load_path/'target'/str(name), save_path/str(label)/str(name))
#     return 0

def tsne(fts):
    iris = load_iris()  # 使用sklearn自带的测试文件
    iris['data'] = fts
    new_w = TSNE(n_components=2, learning_rate=10, init='pca', random_state=12).fit_transform(iris.data)
    return new_w

def create_file(name_list, path_list, label):
    name = name_list[0]
    path = path_list[0]
    if os.path.exists(path):
        shutil.copytree(path, save_path/str(label)/str(name))
    return 0

def exft_kms(start, end, logger_kms, ni_clusters=25, train_data=False):
    # logger_kms = get_logger('kmeans', 'kmeans')
    for i in range(ni_clusters):
        if os.path.exists('/home/dh/kgm/mn40_ke/TTrain/'+str(i)):
            shutil.rmtree('/home/dh/kgm/mn40_ke/TTrain/'+str(i))
        os.mkdir('/home/dh/kgm/mn40_ke/TTrain/'+str(i))
    q_fts = np.load('kmeans/q_fts.npy')
    t_fts = np.load('kmeans/t_fts.npy')
    q_name = list(np.load('kmeans/q_name.npy'))
    t_name = list(np.load('kmeans/t_name.npy'))
    q_path = list(np.load('kmeans/q_path.npy'))
    t_path = list(np.load('kmeans/t_path.npy'))
    q_las = list(np.load('kmeans/q_las.npy'))
    t_las = list(np.load('kmeans/t_las.npy'))

    if train_data:
        tr_fts = np.load('kmeans/tr_fts.npy')
        tr_name = list(np.load('kmeans/tr_name.npy'))
        tr_path = list(np.load('kmeans/tr_path.npy'))
        tr_las = list(np.load('kmeans/tr_las.npy'))
        print(tr_fts.shape,'-------------tr_fts.shape-------------')

        all_las = q_las + t_las + tr_las
        all_name = q_name + t_name + tr_name
        all_path = q_path + t_path + tr_path
        all_fts = np.concatenate((q_fts, t_fts, tr_fts), axis=0)
        # all_fts = all_fts[:, 2048:3368]
    else:
        all_las = q_las + t_las
        all_name = q_name + t_name
        all_path = q_path + t_path
        all_fts = np.concatenate((q_fts, t_fts), axis=0)

    all_st = time.time()
    clf = KMeans(n_clusters=ni_clusters, random_state=9)
    all_pred = clf.fit_predict(all_fts)  # pred_result
    all_sec = time.time() - all_st
    print(f"Kmeans time cost: {all_sec // 60 // 60} hours {all_sec // 60 % 60} minutes!")

    all_st = time.time()
    np.save("kmeans/tsne.npy", {'fts': tsne(all_fts), 'true_las': all_las, 'kms_las': all_pred})
    all_sec = time.time() - all_st
    print(f"tsne time cost: {all_sec // 60 // 60} hours {all_sec // 60 % 60} minutes!")


    for i in range(ni_clusters):
        list_pred = list(all_pred)
        print(list_pred.count(i), "/", i)
    center = clf.cluster_centers_                   # center_points
    print("center_points_size:  ", center.shape)
    print("pred_size:           ", all_pred.shape)
    print("score:               ", metrics.calinski_harabasz_score(all_fts, all_pred))
    dist_mat = scipy.spatial.distance.cdist(center, all_fts, 'cosine')
    print("dist_metric:         ", dist_mat.shape)

    logger_kms.info('                   ')
    logger_kms.info('---save model list:')
    for cls in range(ni_clusters):
        index = np.argsort(dist_mat[cls])
        name_list = []
        for i in range(start, end):
            if all_pred[index[i]] == cls:
                create_file(all_name[index[i]], all_path[index[i]], cls)
                name_list.append(str(all_name[index[i]][0]))
        logger_kms.info(name_list)

def main():
    print("hello")

if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")