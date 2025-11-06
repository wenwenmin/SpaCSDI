import os
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
import scanpy as sc
import networkx as nx
def Kmeans_cluster(adata, num_cluster, used_obsm='latent', key_added_pred="kmeans_cluster", random_seed=2024):
    np.random.seed(random_seed)
    cluster_model = KMeans(n_clusters=num_cluster, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
    cluster_labels = cluster_model.fit_predict(adata.obsm[used_obsm])
    adata.obs[key_added_pred] = cluster_labels
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('int')
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('category')
    return adata

def create_dictionary_gnn(adata, use_rep, use_label, batch_name, k = 50,  verbose = 1, mask_rate=0.5):

    cell_names = adata.obs_names

    batch_list = adata.obs[batch_name]
    cells = []
    remains_cells = []
    for i in batch_list.unique():
        cells.append(cell_names[batch_list == i])
        remains_cells.append(cell_names[batch_list != i])

    mnns = dict()
    u_unique_set = None
    n_unique_set = None
    for idx, b_name in enumerate(batch_list.unique()):
        key_name = str(b_name) + "_" + "rest"
        mnns[key_name] = {}

        new = list(cells[idx])
        ref = list(remains_cells[idx])

        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]
        gt1 = adata[new].obs[use_label]
        gt2 = adata[ref].obs[use_label]
        names1 = new
        names2 = ref
        match, u_unique_set, n_unique_set = GNN(ds1, ds2, gt1, gt2, names1, names2, u_unique_set, n_unique_set, knn=k, mask_rate=mask_rate)
        if (verbose > 0):
            print('Processing datasets {0} have {1} nodes or edges'.format(b_name, len(match)))

        if len(match) > 0:
            G = nx.Graph()
            G.add_edges_from(match)
            node_names = np.array(G.nodes)
            anchors = list(node_names)
            adj = nx.adjacency_matrix(G)
            tmp = np.split(adj.indices, adj.indptr[1:-1])

            for i in range(0, len(anchors)):
                key = anchors[i]
                i = tmp[i]
                names = list(node_names[i])
                mnns[key_name][key]= names
    return(mnns)


def GNN(target_slice_ds, rest_slice_ds, gt1, gt2, names1, names2, u_unique_set=None, n_unique_set=None, knn = 20, approx = False, mask_rate=0.5):
    if u_unique_set is None:
        u_unique_set = set()
    if n_unique_set is None:
        n_unique_set = set()

    similarity = torch.matmul(torch.tensor(target_slice_ds), torch.transpose(torch.tensor(rest_slice_ds), 1, 0))
    _, I_knn = similarity.topk(k=knn, dim=1, largest=True, sorted=False)

    mask = torch.rand(I_knn.shape) < mask_rate
    I_knn[mask] = -1
    match_lst = []
    for i in range(I_knn.shape[0]):
        gt = gt1.iloc[i]  # 正确的 Pandas 访问方式
        gt_tmp = set(gt2[gt2 == gt].index)
        for j in I_knn[i]:
            if j == -1:
                continue
            if names2[j] not in gt_tmp:
                continue
            item = (names1[i], names2[j])
            ex_item = (names2[j], names1[i])

            if ex_item in u_unique_set:
                n_unique_set.add(ex_item)
                continue
            if item not in u_unique_set:
                u_unique_set.add(item)
                match_lst.append(item)
    return match_lst, u_unique_set, n_unique_set