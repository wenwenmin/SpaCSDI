import anndata as ad
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import os
import torch
import numpy as np
import scanpy as sc
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans


import os

import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt

import model.simulation_st
import anndata

from scipy.sparse import issparse,csr_matrix
from sklearn.preprocessing import normalize
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

import model.data_downsample

def find_sc_top_marker_genes(scdata,  celltype_key, top_gene_per_type=200,method='t-test'):

    sc_ad = scdata.copy()

    sc.tl.rank_genes_groups(sc_ad, celltype_key, method=method, use_raw=False, top_genes_per_group=top_gene_per_type)

    # Initialize lists and dictionary to store results
    combined_genes = []
    gene_dict = {}  # To store genes with p-values and log fold changes

    for cell_type in sc_ad.obs[celltype_key].unique():
        # Get the marker genes for the current cell type
        marker_genes = sc.get.rank_genes_groups_df(sc_ad, group=cell_type).sort_values('pvals_adj')

        for i, row in marker_genes.iterrows():
            gene_name = row['names']
            # Ensure gene information is stored correctly in the dictionary
            if gene_name not in gene_dict:
                gene_dict[gene_name] = []
            gene_dict[gene_name].append({
                'cell_type': cell_type,
                'p_value': row['pvals_adj'],
                'log_fold_change': row['logfoldchanges']
            })

        # Get the top genes for the current cell type
        top_genes = marker_genes['names'].values[:top_gene_per_type]
        combined_genes.extend(top_genes)

    # Remove duplicates and preserve the order in the original list
    combined_genes = list(dict.fromkeys(combined_genes))

    return combined_genes, gene_dict

def dist_adj_real_one(stdata, spatial_x, spatial_y, space_dist_neighbors=20, link_method='hard', space_dist_threshold=None):
    # Perform k-NN search
    knn = NearestNeighbors(n_neighbors=space_dist_neighbors, metric='minkowski', p=2)  # 明确 p=2
    knn.fit(stdata.obs[[spatial_x, spatial_y]])
    dist, ind = knn.kneighbors()

    # Initialize sparse adjacency matrix (lil_matrix for efficient row updates)
    A_space = lil_matrix((stdata.shape[0], stdata.shape[0]), dtype=float)

    # Create adjacency matrix based on link method
    for i in range(ind.shape[0]):
        for j in range(ind.shape[1]):
            if space_dist_threshold is None or dist[i, j] < space_dist_threshold:
                if link_method == 'hard':
                    A_space[i, ind[i, j]] = 1
                    A_space[ind[i, j], i] = 1
                else:  # 'soft' method
                    if dist[i, j] > 1e-6:  # 避免除零错误
                        A_space[i, ind[i, j]] = 1 / dist[i, j]
                        A_space[ind[i, j], i] = 1 / dist[i, j]

    # Convert to csr_matrix for efficient calculations
    A_space = csr_matrix(A_space)

    # Store adjacency matrix in stdata.uns (not obs)
    stdata.obsm['dist_adj'] = A_space

    return stdata


def dist_adj_pseodo(stdata, space_dist_neighbors=20):
    true_labels = stdata.obsm['label']
    # 确保 true_labels 是数值型矩阵
    if not isinstance(true_labels, np.ndarray):
        true_labels = np.array(true_labels)

    # 计算余弦相似度
    cos_sim = cosine_similarity(true_labels)

    # 取前 (k+1) 个最近邻，然后去掉自己
    k_index = torch.topk(torch.tensor(cos_sim), k=space_dist_neighbors + 1, dim=1)[1]
    k_index = k_index[:, 1:]  # 去掉自身索引

    # 创建邻接矩阵（稀疏矩阵）
    A_space = lil_matrix((stdata.shape[0], stdata.shape[0]), dtype=np.float32)

    # 构建邻接矩阵
    for i in range(k_index.shape[0]):
        for j in k_index[i].tolist():  # 转换为整数索引
            if j < stdata.shape[0]:  # 确保索引合法
                A_space[i, j] = 1
                A_space[j, i] = 1

    # 转换为 CSR 格式
    A_space = csr_matrix(A_space)

    # 存入 stdata
    stdata.obsm['dist_adj'] = A_space # 如果 stdata.uns 不支持稀疏矩阵，转换为 NumPy 数组

    return stdata


def exp_adj(data, exp_dist_neighbors=15):
    # 降维
    if data.shape[1] > 2500:
        dim = 200
    else:
        dim = 100
    sc.tl.pca(data, n_comps=dim, svd_solver='arpack', random_state=None)
    input_data = data.obsm['X_pca']  # PCA降维后的数据

    # 计算余弦相似度
    cos_sim = cosine_similarity(input_data)

    k_index = torch.topk(torch.tensor(cos_sim), k=exp_dist_neighbors + 1, dim=1)[1]
    k_index = k_index[:, 1:]  # 去掉自身索引

    # 创建邻接矩阵（稀疏矩阵）
    A_space = lil_matrix((data.shape[0],data.shape[0]), dtype=np.float32)

    # 构建邻接矩阵
    for i in range(k_index.shape[0]):
        for j in k_index[i].tolist():  # 转换为整数索引
            if j < data.shape[0]:  # 确保索引合法
                A_space[i, j] = 1
                A_space[j, i] = 1

    # 转换为 CSR 格式
    A_space = csr_matrix(A_space)

    # 存入 stdata
    # data.uns['exp_adj'] = A_space.toarray()  # 如果 stdata.uns 不支持稀疏矩阵，转换为 NumPy 数组
    data.obsm['exp_adj'] = A_space

    return data

def _mask_by_genes(x, mask_rate=0.2):
        use_x = x.clone()
        if isinstance(use_x, np.ndarray):
            mask = np.random.choice([True, False], size=use_x.shape, p=[mask_rate, 1 - mask_rate], replace=False)
        elif isinstance(use_x, torch.Tensor):
            mask = torch.rand(use_x.shape) < mask_rate
        else:
            raise TypeError("type error!")
        mask = mask.to(use_x.device)
        use_x[mask] = 0
        return use_x, mask


def SaveLossPlot(SavePath, metric_logger, loss_type, output_prex):
    # Ensure the save path exists
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)

    # Create a figure for plotting
    plt.figure(figsize=(15, 15))  # Optional: adjust figure size

    for i in range(len(loss_type)):
        # Check if the loss data is available
        if loss_type[i] not in metric_logger:
            print(f"Warning: {loss_type[i]} not found in metric_logger!")
            continue  # Skip this loss type if not found

        # Get the loss data
        metric_data = metric_logger[loss_type[i]]

        # Ensure metric_data is a tensor on CPU before plotting
        if isinstance(metric_data, torch.Tensor):
            # Move tensor to CPU if it is on GPU
            metric_data = metric_data.cpu().numpy()

        # Check if metric_data is actually a 1D array or compatible for plotting
        # Plot the data in the corresponding subplot
        plt.subplot(3, 3, i + 1)  # Adjust to your preferred grid layout
        plt.plot(metric_data)
        plt.title(loss_type[i], x=0.5, y=0.95)  # Centered title with a slight offset

    # Save the plot to the specified path
    imgName = os.path.join(SavePath, output_prex + '.png')
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.savefig(imgName)
    plt.close()  # Close the plot to free up memory



def Kmeans_cluster(adata, num_cluster, used='latent', key_added_pred="G3STNET", random_seed=2024):
    np.random.seed(random_seed)
    cluster_model = KMeans(n_clusters=num_cluster, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
    cluster_labels = cluster_model.fit_predict(adata.obsm[used])
    adata.obs[key_added_pred] = cluster_labels
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('int')
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('category')
    return adata





def normalized_data(ad,target_sum=None):
    ad_norm = sc.pp.normalize_total(ad,inplace=False,target_sum=1e4)
    ad_norm  = sc.pp.log1p(ad_norm['X'])
    # ad_norm  = sc.pp.scale(ad_norm)
    # ad_norm = normalize(ad_norm,axis=1)
    ad.layers['norm'] = ad_norm
    return ad
def filter_model_genes(
    sc_ad,
    st_ad,
    celltype_key,
    layer='norm',
    deg_method=None,
    log2fc_min=0.5,
    pval_cutoff=0.01,
    n_top_markers=500,
    n_top_hvg=None,
    pct_diff=None,
    pct_min=0.1,
):
    # Remove duplicate genes from st_ad
    if len(set(st_ad.var_names)) != len(st_ad.var_names):
        print("Removing duplicate genes from st_ad")
        # Create a boolean mask where True indicates the first occurrence of each gene
        mask = ~st_ad.var_names.duplicated()
        st_ad = st_ad[:, mask].copy()

    # Compute overlapping genes
    # overlaped_genes = np.intersect1d(sc_ad.var_names, st_ad.var_names)
    #
    # sc_ad = sc_ad[:,overlaped_genes].copy()
    # st_ad = st_ad[:,overlaped_genes].copy()
    # 在 filter_model_genes 函数中

    # if n_top_hvg is None:
    st_genes = st_ad.var_names
    # else:
    #     sc.pp.highly_variable_genes(st_ad, n_top_genes=n_top_hvg, flavor='seurat_v3')
    #     st_genes = st_ad.var_names[st_ad.var['highly_variable'] == True]

    sc_ad = sc_ad[:, st_genes].copy()
    sc_genes = find_sc_markers(sc_ad, celltype_key, layer, deg_method, log2fc_min, pval_cutoff, n_top_markers, pct_diff, pct_min)
    used_genes = np.intersect1d(sc_genes,st_genes)
    sc_ad = sc_ad[:,used_genes].copy()
    st_ad = st_ad[:,used_genes].copy()
    sc.pp.filter_cells(sc_ad, min_genes=1)
    sc.pp.filter_cells(st_ad, min_genes=1)

    print(f'### This Sample Used gene numbers is: {len(used_genes)}')
    return sc_ad, st_ad


def find_sc_markers(sc_ad, celltype_key, layer='norm', deg_method=None, log2fc_min=0.5, pval_cutoff=0.01,
                    n_top_markers=500, pct_diff=None, pct_min=0.1):
    print('### Finding marker genes...')
    # filter celltype contain only one sample.
    filtered_celltypes = list(
        sc_ad.obs[celltype_key].value_counts()[(sc_ad.obs[celltype_key].value_counts() == 1).values].index)
    if len(filtered_celltypes) > 0:
        sc_ad = sc_ad[sc_ad.obs[~(sc_ad.obs[celltype_key].isin(filtered_celltypes))].index, :].copy()
    sc.tl.rank_genes_groups(sc_ad, groupby=celltype_key, pts=True, layer=layer, use_raw=False, method=deg_method)
    marker_genes_dfs = []
    for c in np.unique(sc_ad.obs[celltype_key]):
        tmp_marker_gene_df = sc.get.rank_genes_groups_df(sc_ad, group=c, pval_cutoff=pval_cutoff, log2fc_min=log2fc_min)
        if (tmp_marker_gene_df.empty is not True):
            tmp_marker_gene_df.index = tmp_marker_gene_df.names
            tmp_marker_gene_df.loc[:, celltype_key] = c
            if pct_diff is not None:
                pct_diff_genes = sc_ad.var_names[np.where((sc_ad.uns['rank_genes_groups']['pts'][c] -
                                                           sc_ad.uns['rank_genes_groups']['pts_rest'][c]) > pct_diff)]
                tmp_marker_gene_df = tmp_marker_gene_df.loc[np.intersect1d(pct_diff_genes, tmp_marker_gene_df.index), :]
            if pct_min is not None:
                # pct_min_genes = sc_ad.var_names[np.where((sc_ad.uns['rank_genes_groups']['pts'][c]) > pct_min)]
                tmp_marker_gene_df = tmp_marker_gene_df[tmp_marker_gene_df['pct_nz_group'] > pct_min]
            if n_top_markers is not None:
                tmp_marker_gene_df = tmp_marker_gene_df.sort_values('logfoldchanges', ascending=False)
                tmp_marker_gene_df = tmp_marker_gene_df.iloc[:n_top_markers, :]
            marker_genes_dfs.append(tmp_marker_gene_df)
    marker_gene_df = pd.concat(marker_genes_dfs, axis=0)
    print(marker_gene_df[celltype_key].value_counts())
    all_marker_genes = np.unique(marker_gene_df.names)
    return all_marker_genes

def generate_sm_stdata(sc_ad,num_sample,celltype_key,n_threads,cell_counts,clusters_mean,cells_mean,cells_min,cells_max,cell_sample_counts,cluster_sample_counts,ncell_sample_list,cluster_sample_list):
    sm_data,sm_labels = simulation_st.generate_simulation_data(sc_ad,num_sample=num_sample,celltype_key=celltype_key,downsample_fraction=None,data_augmentation=False,n_cpus=n_threads,
                                                               cell_counts=cell_counts,clusters_mean=clusters_mean,cells_mean=cells_mean,cells_min=cells_min,
                                                               cells_max=cells_max,cell_sample_counts=cell_sample_counts,cluster_sample_counts=cluster_sample_counts,
                                                               ncell_sample_list=ncell_sample_list,cluster_sample_list=cluster_sample_list)
    sm_data_mtx = csr_matrix(sm_data)
    sm_ad = anndata.AnnData(sm_data_mtx)
    sm_ad.var.index = sc_ad.var_names
    sm_labels = (sm_labels.T/sm_labels.sum(axis=1)).T
    sm_ad.obsm['label'] = pd.DataFrame(sm_labels,columns=np.array(sc_ad.obs[celltype_key].value_counts().index.values),index=sm_ad.obs_names)
    return sm_ad

def downsample_sm_spot_counts(sm_ad,st_ad,n_threads):
    fitdistrplus = importr('fitdistrplus')
    lib_sizes = robjects.FloatVector(np.array(st_ad.X.sum(1)).reshape(-1))
    res = fitdistrplus.fitdist(lib_sizes,'lnorm')
    loc = res[0][0]
    scale = res[0][1]
    sm_mtx_count = sm_ad.X.toarray()
    sample_cell_counts = np.random.lognormal(loc,scale,sm_ad.shape[0])
    sm_mtx_count_lb = data_downsample.downsample_matrix_by_cell(sm_mtx_count,sample_cell_counts.astype(np.int64), n_cpus=n_threads, numba_end=False)
    sm_ad.X = csr_matrix(sm_mtx_count_lb)


def check_data_type(ad):
    if issparse(ad.X):
        ad.X = ad.X.toarray()
    if ad.X.dtype != np.float32:
        ad.X =ad.X.astype(np.float32)
    return ad

def SaveLossPlot(SavePath, metric_logger, loss_type, output_prex):
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    for i in range(len(loss_type)):
        plt.subplot(3, 3, i+1)
        plt.plot(metric_logger[loss_type[i]])
        plt.title(loss_type[i], x = 0.5, y = 0.5)
    imgName = os.path.join(SavePath, output_prex +'.png')
    plt.savefig(imgName)
    plt.close()