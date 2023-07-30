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


def extract_feat(args, load_name, input_class=8):
    model = UniModel(args=args, n_class=input_class)
    model.eval()

    model.to(args.device)

    path_load = os.path.join('/home/dh/kgm/OS-MN40-hm/experiment/checkpoints', load_name)
    pretrained = torch.load(path_load)
    model.load_state_dict(pretrained)

    queryDataset = GetDataTrain(dataType='query')
    targetDataset = GetDataTrain(dataType='target')

    queryLoader = torch.utils.data.DataLoader(queryDataset, batch_size=16, shuffle=False,
                                                 num_workers=args.j, pin_memory=True, drop_last=False)
    targetLoader = torch.utils.data.DataLoader(targetDataset, batch_size=16, shuffle=False,
                                                 num_workers=args.j, pin_memory=True, drop_last=False)


    ftss_q = []
    lass_q = []
    ftss_t = []
    lass_t = []
    all_lbls, all_preds = [], []

    for idx, input_data in enumerate(tqdm(queryLoader)):
        target = input_data['target_mv'].reshape(-1)
        # target = target.to(args.device)
        data_mv = input_data['data_mv'].to(args.device)
        data_pc = input_data['data_pc'].to(args.device)
        data_mesh1 = input_data['data_mesh1'].to(args.device)
        data_mesh2 = input_data['data_mesh2'].to(args.device)
        data_vox = input_data['data_vox'].to(args.device)
        label_rand1 = label_rand.repeat(data_mv.shape[0], 1, 1)
        if idx == 0:
            name_q = input_data['name_all']
            path_q = input_data['path_all']
            np.save("kmeans/q_name.npy", name_q)
            np.save("kmeans/q_path.npy", path_q)
            del name_q, path_q

        with torch.no_grad():
            out, fts = model(data_mv, data_pc, data_mesh1, data_mesh2, data_vox, True, label_rand1)
            conf, preds = torch.max(out, 1)

        ftss_q.append(fts.cpu().data)
        lass_q.append(target.cpu().data)
    ftss_q = torch.cat(ftss_q, dim=0).numpy()
    lass_q = torch.cat(lass_q, dim=0).numpy()
    np.save("kmeans/q_fts.npy", ftss_q)
    np.save("kmeans/q_las.npy", lass_q)

    for idx, input_data in enumerate(tqdm(targetLoader)):
        target = input_data['target_mv'].reshape(-1)
        # target = target.to(args.device)
        data_mv = input_data['data_mv'].to(args.device)
        data_pc = input_data['data_pc'].to(args.device)
        data_mesh1 = input_data['data_mesh1'].to(args.device)
        data_mesh2 = input_data['data_mesh2'].to(args.device)
        data_vox = input_data['data_vox'].to(args.device)
        label_rand1 = label_rand.repeat(data_mv.shape[0], 1, 1)
        if idx == 0:
            name_t = input_data['name_all']
            path_t = input_data['path_all']
            np.save("kmeans/t_name.npy", name_t)
            np.save("kmeans/t_path.npy", path_t)
            del name_t, path_t

        with torch.no_grad():
            out, fts = model(data_mv, data_pc, data_mesh1, data_mesh2, data_vox, True, label_rand1)
            out = F.softmax(out / 1, dim=-1)
            conf, preds = torch.max(out, 1)

        ftss_t.append(fts.cpu().data)
        lass_t.append(target.cpu().data)
        all_preds.append(preds.cpu().data)
        all_lbls.append(target.cpu().data)
    ftss_t = torch.cat(ftss_t, dim=0).numpy()
    lass_t = torch.cat(lass_t, dim=0).numpy()
    np.save("kmeans/t_fts.npy", ftss_t)
    np.save("kmeans/t_las.npy", lass_t)

    map_modal = []
    for idx, start_end in enumerate([(0, 512), (512, 1024), (1024, 1536), (1536, 2048), (2048, 3648), (0, 3648)]):
        dist_mat_modal = scipy.spatial.distance.cdist(ftss_q[:, start_end[0]:start_end[1]], ftss_t[:, start_end[0]:start_end[1]], 'cosine')
        map_modal.append(map_score(dist_mat_modal, lass_q, lass_t))
    res = {
        "map_img": map_modal[0],
        "map_pt": map_modal[1],
        "map_mesh": map_modal[2],
        "map_vox": map_modal[3],
        "map_fuse": map_modal[4],
        "map_all": map_modal[5]
    }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)

    return 0



def main():
    print("hello")

if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")