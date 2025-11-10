import scipy
import torch
import gudhi
import itertools
import hnswlib
import torch.linalg
import random
import copy
import scipy.sparse
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
import torch.optim as optim

from scipy import sparse
from scipy import stats
from typing import Optional
from annoy import AnnoyIndex
from scipy.spatial import distance
from collections import Counter
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from torch_geometric.nn import GCNConv

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from typing import List, Optional, Union, Any
from torch_geometric.nn.conv import MessagePassing


from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics import adjusted_rand_score as ari_score


def get_mean_reconstruct_sc(adata, label_list, adata_label, class_rep = None):
    index_type_l = []
    for i in range(len(label_list)):
        indexes = [index for index in range(len(adata_label)) if adata_label[index] == label_list[i]]
        index_type_l.append([indexes[j] for j in range(len(indexes))])
    if class_rep == 'reconstruct':
        sc_recon = adata.obsm['reconstruct'].copy()
    elif class_rep == 'embedding':
        sc_recon = adata.obsm['embedding'].copy()
    else:
        if scipy.sparse.issparse(adata.X):
            sc_recon = adata.X.todense() # spot*genes
        else:
            sc_recon = adata.X.copy()
    sc_recon_m_l = []
    for j in range(len(label_list)):
        sc_recon_j = sc_recon[index_type_l[j]]
        sc_recon_m_l.append(np.mean(sc_recon_j, axis = 0))
    sc_recon_m = np.array(sc_recon_m_l)
    #sc_recon_mean = pd.DataFrame(sc_recon_m.T,columns=label_list)
    sc_recon_m1 = sc_recon_m.reshape(len(label_list),sc_recon_j.shape[1])
    return sc_recon_m1


def label_to_int(adata, label_name):
    adata_label_o = np.array(adata.obs[label_name].copy())
    label_list = list(set(adata.obs[label_name].tolist()))
    adata_label = adata_label_o.copy()
    for i in range(len(label_list)):
        need_index = np.where(adata.obs[label_name]==label_list[i])[0]
        if len(need_index):
            adata_label[need_index] = i
    adata.obs['pre_label'] = adata_label
    return adata, label_list,adata_label_o


def identify_anchors(adata_l, section_ids, st_flag, iter_comb, use_rep_anchor, label_name, mnn_neigh, agg_neigh = 10):
    if use_rep_anchor == 'Agg':
        n=len(adata_l)
        for i in range(n):
            adata_i =  adata_l[i].copy()                    
            adata_l[i].obsm['Agg'] = generate_aggegate_data(adata_i, agg_neigh = agg_neigh) 
           

    adata_l_st = []
    for st_i in st_flag:
        adata_l_st.append(adata_l[st_i])

    adata_concat_st = ad.concat(adata_l_st, label="slice_name", keys=section_ids) ## note: here will generate repeat slice names
    adata_concat_st.obs["batch_name"] = adata_concat_st.obs["slice_name"].astype('category')      

    mnns, mnns_adj = compute_mnn(adata_concat_st, adata_l_st, use_rep_anchor, iter_comb = iter_comb, batch_name='batch_name', k=mnn_neigh, metric='euclidean', method = 'approx')
    
    # for each data pair, check the accuracy
    if label_name in adata_concat_st.obs.keys():
        print(check_mnn_accuracy(mnns_adj, adata_concat_st, iter_comb, label_name, batch_name='batch_name'))
     
    anchors_l = covert_mnns(adata_concat_st, mnns_adj, iter_comb, label_name, batch_name='batch_name')    
    return anchors_l


def select_confident_cells(adata, label_name, focus_cell_type, q_cutoff = 0.75):
    if focus_cell_type is not None:
        adata_sc = adata.copy()
        sc.pp.scale(adata_sc)
        # calculate leiden clustering
        sc.pp.pca(adata_sc)
        sc.pp.neighbors(adata_sc)
        label_type = adata_sc.obs[label_name].tolist()
    
        snn_sc = np.asmatrix(adata_sc.obsp['connectivities'].todense())
        min_cluster = 20
        index_type_l = []
        for i in range(len(focus_cell_type)):
            indexes = [index for index in range(len(label_type)) if label_type[index] == focus_cell_type[i]]
            if len(indexes) > min_cluster:
                snn_i = snn_sc[np.ix_(indexes, indexes)]
                col_si = snn_i.mean(1)
                thresh_i = np.quantile(np.array(col_si)[:,0], q_cutoff)
                index_i = np.where(col_si > thresh_i)[0]
                if len(index_i) > min_cluster:
                    index_type_l.extend([indexes[index_i[j]] for j in range(len(index_i))])
                else:
                    index_type_l.extend([indexes[j] for j in range(len(indexes))])
            else:
                index_type_l.extend([indexes[j] for j in range(len(indexes))])
        union_cell=list(set(index_type_l))
        adata = adata[union_cell,:]
    return adata



def graph_alpha(spatial_locs, n_neighbors=10):
    """
    Construct a geometry-aware spatial proximity graph of the spatial spots of cells by using alpha complex.
    :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
    :type adata: class:`anndata.annData`
    :param n_neighbors: the number of nearest neighbors for building spatial neighbor graph based on Alpha Complex
    :type n_neighbors: int, optional, default: 10
    :return: a spatial neighbor graph
    :rtype: class:`scipy.sparse.csr_matrix`
    """
    A_knn = kneighbors_graph(spatial_locs, n_neighbors=n_neighbors, mode='distance')
    estimated_graph_cut = A_knn.sum() / float(A_knn.count_nonzero())
    spatial_locs_list = spatial_locs.tolist()
    n_node = len(spatial_locs_list)
    alpha_complex = gudhi.AlphaComplex(points=spatial_locs_list)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=estimated_graph_cut ** 2)
    skeleton = simplex_tree.get_skeleton(1)
    initial_graph = nx.Graph()
    initial_graph.add_nodes_from([i for i in range(n_node)])
    for s in skeleton:
        if len(s[0]) == 2:
            initial_graph.add_edge(s[0][0], s[0][1])

    extended_graph = nx.Graph()
    extended_graph.add_nodes_from(initial_graph)
    extended_graph.add_edges_from(initial_graph.edges)

    # add self edges
    for i in range(n_node):
        try:
            extended_graph.add_edge(i, i)
        except:
            pass

   # return nx.to_numpy_matrix(extended_graph)
    return nx.to_numpy_array(extended_graph)

def buid_adj_for_sc(adata_sc):
    cell_type = adata_sc.obs['ref'].tolist()
    cell_type_uniq = list(set(cell_type))
    snn_sc = np.zeros([len(cell_type), len(cell_type)])
    for i in range(len(cell_type_uniq)):
        id_i = [j for j in range(len(cell_type)) 
                if cell_type[j] == cell_type_uniq[i]]
        adata_sc_i = adata_sc[id_i].copy()
        sc.pp.scale(adata_sc_i)
        sc.tl.pca(adata_sc_i)
        sc.pp.neighbors(adata_sc_i)
        snn_i = adata_sc_i.obsp['connectivities']
        snn_sc[np.ix_(id_i,id_i)] = to_dense_array(snn_i)
    adata_sc.obsp['adj_f'] = sparse.csr_matrix(snn_sc)
    return adata_sc

# aggregated original X with graph obtained using embedding
def generate_aggegate_data(adata, agg_neigh):
    sc.pp.scale(adata)
    # calculate leiden clustering
    sc.pp.pca(adata)
    adata.obsm['X_pca_old'] = adata.obsm['X_pca'].copy()
    adata.obsm['X_pca'] = adata.obsm['embedding']
    sc.pp.neighbors(adata)
    snn_source = adata.obsp['connectivities'].todense()
    if scipy.sparse.issparse(adata.X): 
        X = adata.X.todense() 
    else:
        X = adata.X
    X_mg = X.copy()
    for i in range(X.shape[0]):
        # detect non-zero of snn1_g
        index_i = snn_source[i,:].argsort().A[0]
        index_i = index_i[(X.shape[0]-agg_neigh):X.shape[0]]
        X_mg[i,:] = X[index_i,:].mean(0)
    return X_mg



def nn_approx(ds1, ds2, names1, names2, knn=50):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M = 16)
    p.set_ef(10)
    p.add_items(ds2)
    ind,  distances = p.knn_query(ds1, k=knn)
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match


def nns(ds1, ds2, names1, names2, knn=50, metric_p=2):
    # Find nearest neighbors of first dataset.
    knn = min(knn, min(ds1.shape[0], ds2.shape[0]))
    nns_ = NearestNeighbors(n_neighbors=knn).fit(ds2)
    ind = nns_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def nn_annoy(ds1, ds2, names1, names2, knn = 20, metric='euclidean', n_trees = 50, save_on_disk = True):
    """ Assumes that Y is zero-indexed. """
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    if(save_on_disk):
        a.on_disk_build('annoy.index')
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def mnn(ds1, ds2, names1, names2, knn = 20, metric='euclidean', save_on_disk = True, method = 'annoy'):
    if method == 'annoy':
        match1 = nn_annoy(ds1, ds2, names1, names2, knn=knn, metric=metric)#, save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2 = nn_annoy(ds2, ds1, names2, names1, knn=knn, metric=metric)#, save_on_disk = save_on_disk)
        mutual = match1 & set([ (b, a) for a, b in match2 ])
    if method == 'approx':
        # Find nearest neighbors in first direction.
        # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)#, save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)#, save_on_disk = save_on_disk)
        mutual = match1 & set([ (b, a) for a, b in match2 ])
    else:
        match1 = nns(ds1, ds2, names1, names2, knn=knn)
        match2 = nns(ds2, ds1, names2, names1, knn=knn)
        # Compute mutual nearest neighbors.
        mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual

def compute_mnn(adata, adata_l, use_rep, iter_comb, batch_name, k = 20, metric='euclidean', save_on_disk = True, method = [], verbose = 1):
       
    cell_names = adata.obs_names
    batch_list = adata.obs[batch_name]
    cells = []
    for i in batch_list.unique():        
        if use_rep in adata[batch_list == i].obsm.keys():
            cells.append(cell_names[batch_list == i])
        else:
            cells.append(None)
      
    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(cells)), 2))

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    mnns = dict()
    mnns_adj = dict()
    
    for comb in iter_comb:
        i = comb[0]
        j = comb[1]
        key_name1 = batch_name_df.loc[i].values[0] + "+" + batch_name_df.loc[j].values[0]
        mnns[key_name1] = {} # for multiple-slice setting, the key_names1 can avoid the mnns replaced by previous slice-pair
        if(verbose > 0):
            print('Processing datasets {}'.format((i, j)))
        
        
        if adata_l is not None:
            new = adata_l[j].obs_names
            ref = adata_l[i].obs_names
            if use_rep not in adata_l[j].obsm.keys():
                use_rep = 'embedding'
                
            ds1 = adata_l[j].obsm[use_rep]
            ds2 = adata_l[i].obsm[use_rep]
        else:
            new = list(cells[j])
            ref = list(cells[i])
            if use_rep not in adata.obsm.keys():
                use_rep = 'embedding'
            ds1 = adata[new].obsm[use_rep] # note adata has no representation
            ds2 = adata[ref].obsm[use_rep]
            
        names1 = new
        names2 = ref
        
        ds1 = np.asarray(ds1)
        ds2 = np.asarray(ds2)
        # if k>1，one point in ds1 may have multiple MNN points in ds2.
        match = mnn(ds1, ds2, names1, names2, knn=k, metric=metric, save_on_disk = save_on_disk, method = method)

        G = nx.Graph()
        G.add_edges_from(match)
        node_names = np.array(G.nodes)
        anchors = list(node_names)
        adj = nx.adjacency_matrix(G)
        #adj = nx.to_numpy_matrix(G)
        tmp = np.split(adj.indices, adj.indptr[1:-1])

        # return the similarity id between each pair
        cells1 = ref.tolist()
        cells2 = new.tolist()
        cells12 = cells1
        for p in cells2:
            cells12.append(p)   
            
        cross_adj = np.zeros((len(cells12),len(cells12)))    
        for index_an in range(0, len(anchors)):
            key = anchors[index_an]
            k_new = tmp[index_an]
            names = list(node_names[k_new])
            mnns[key_name1][key]= names
            
            need_index1 = cells12.index(key)
            need_index2 = []
            for t in range(0, len(names)):
                need_index2.append(cells12.index(names[t])) 
                cross_adj[need_index1,need_index2] = 1
            mnns_adj[key_name1] = cross_adj
    return mnns, mnns_adj

# detect each pair of mnn
def get_mnn_index(mnn_adj):
    mnn_id = []
    s_id = []
    t_id = []
    for i in range(mnn_adj.shape[0]):
        c_i = mnn_adj[i,:].nonzero()[0]
        # build pairs
        c_i = c_i[c_i > i]
        if len(c_i) > 0:
            for j in range(len(c_i)):
                mnn_id.append([i,c_i[j]])
                s_id.append(i)
                t_id.append(c_i[j])
    return s_id, t_id

######## check the mnn_adj
def check_mnn_accuracy(mnns_adj, adata_concat,iter_comb,label_name, batch_name):
    if label_name in adata_concat.obs.keys():
        
        cell_names = adata_concat.obs_names
        batch_list = adata_concat.obs[batch_name]
        
        cells = []
        for i in batch_list.unique():
            cells.append(cell_names[batch_list == i])

        batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
        if iter_comb is None:
            iter_comb = list(itertools.combinations(range(len(cells)), 2))
            
        r_record = np.zeros([1,len(iter_comb)])    
        for t in range(0,len(iter_comb)):
            comb = iter_comb[t]
            i = comb[0]
            j = comb[1]
            key_name1 = batch_name_df.loc[i].values[0] + "+" + batch_name_df.loc[j].values[0]
            mnn_adj = mnns_adj[key_name1]    
            ref = list(cells[i])
            new = list(cells[j])
            adata_list = [adata_concat[ref], adata_concat[new]]
            adata = ad.concat(adata_list)
            label = adata.obs[label_name]
            con_same = np.zeros((mnn_adj.shape[0],2))
            for p in range(mnn_adj.shape[0]):
                c_p = mnn_adj[p,:].nonzero()[0]
                label_p = label[c_p]
                if len(c_p) > 0:
                    con_same[p,0] = sum(label_p == label[p])/len(c_p)
                    con_same[p,1] = 1
            r_record[0,t] = sum(con_same[con_same[:,1].nonzero(),0][0])/len(con_same[:,1].nonzero()[0])
        r = np.mean(r_record)
    else:
        r = 0
        print('There is no label information!')
    return r



#from annoy import AnnoyIndex
def acquire_pairs(X, Y, k, metric):
    f = X.shape[1] # the node in the corresponding data are randomly select across the similarity matrix, X is from embeddings 
    t1 = AnnoyIndex(f, metric)
    t2 = AnnoyIndex(f, metric)
    for i in range(len(X)):
        t1.add_item(i, X[i])
    for i in range(len(Y)):
        t2.add_item(i, Y[i])
    t1.build(10)
    t2.build(10)

    mnn_mat = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t2.get_nns_by_vector(item, k) for item in X])
    for i in range(len(sorted_mat)):
        mnn_mat[i,sorted_mat[i]] = True
    _ = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t1.get_nns_by_vector(item, k) for item in Y])
    for i in range(len(sorted_mat)):
        _[sorted_mat[i],i] = True
    mnn_mat = np.logical_and(_, mnn_mat)
    pairs = [(x, y) for x, y in zip(*np.where(mnn_mat>0))]
    return pairs

def create_pairs_dict(pairs):
    pairs_dict = {}
    for x,y in pairs:
        if x not in pairs_dict.keys():
            pairs_dict[x] = [y]
        else:
            pairs_dict[x].append(y)
    return pairs_dict


def find_indices_element(list_to_check, item_to_find):
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]

def find_indices_list(list_to_check, list_to_find):
    indexes = []
    for i in list_to_find:
        id_i = [idx for idx, value in enumerate(list_to_check) if value == i]
        if len(id_i) > 0:
            indexes.extend(id_i)
    return indexes



def most_frequent(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def fusion_anchors(s_id_s1, t_id_s1, s_id_s2, t_id_s2, anchor_type):
    st_id_1 = [str(s_id_s1[i])+'_'+str(t_id_s1[i]) for i in range(len(s_id_s1))]
    st_id_2 = [str(s_id_s2[i])+'_'+str(t_id_s2[i]) for i in range(len(s_id_s2))]
    if anchor_type  == 'union':
        st_id_new = list(set(st_id_1).union(set(st_id_2)))
    elif anchor_type  == 'intersection':
        st_id_new = list(set(st_id_1).intersection(set(st_id_2)))
    elif anchor_type  == 'st':
        st_id_new =list(set(st_id_1))
    elif anchor_type  == 'sc':
        st_id_new =list(set(st_id_2))
    s_id = []
    t_id = []
    for k in range(len(st_id_new)):
        s_id.append(int(st_id_new[k].split('_')[0]))
        t_id.append(int(st_id_new[k].split('_')[1]))                
    return s_id, t_id


# check the similarity between them
def check_id(s_id,t_id, adata_concat, label_name = None):
    if label_name in adata_concat.obs.keys():
        con_same = np.zeros((len(s_id),2))
        labels_1 = []
        labels_2 = []
        label = adata_concat.obs[label_name]
        for i in range(len(s_id)):
            label_i = label[s_id[i]]
            label_j = label[t_id[i]]
            if label_i == label_j:
                con_same[i,1] = 1
        
            labels_1.append(label_i)
            labels_2.append(label_j)
        if len(s_id)>0:
            r = len(con_same[:,1].nonzero()[0])/len(s_id)
        else:
            r = 0
    else:
        r = 0
    return r

def covert_mnns(adata_concat, mnns_adj,iter_comb, label_name, batch_name):
    # remove the mnn pairs with smaller marker similarity
    # check each pairs to filter
    cell_names = adata_concat.obs_names
    batch_list = adata_concat.obs[batch_name]
        
    cells = []
    for i in batch_list.unique():
        cells.append(cell_names[batch_list == i])

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(cells)), 2))
     
    anchors_l = dict()
    for t in range(0,len(iter_comb)):
        comb = iter_comb[t]
        i = comb[0]
        j = comb[1]
        key_name1 = batch_name_df.loc[i].values[0] + "+" + batch_name_df.loc[j].values[0]
        mnn_adj = mnns_adj[key_name1]            
        ref = list(cells[i])
        new = list(cells[j])    
        adata_source = adata_concat[ref]
        adata_target = adata_concat[new]
        adata_concat_t = ad.concat([adata_source, adata_target])
        
        # get index
        s_id_t, t_id_t = get_mnn_index(mnn_adj)
        
        if label_name in adata_concat_t.obs.columns:
            r_3 = check_id(s_id_t,t_id_t, adata_concat_t, label_name)
            print('The ratio of filtered mnn pairs:', r_3)
            
        anchors_l[key_name1] = [s_id_t, t_id_t]
    
    return anchors_l


def filter_mnns(anchors_l,section_ids,adata_l_st,adata_sc,dev_method, smooth = False,cor_cutoff = 0.8):
    if dev_method == 'pfgw' or dev_method == 'otmap':
        if smooth:
            adj_sc = compute_adj_smooth(adata_sc,adata_l_st[0].uns['deconvolution']['sc type'].index.tolist())            
            adj_sc_norm = normalizeAdjacency(adj_sc)
        anchors_l_new = dict()
        keys_l = list(anchors_l.keys())
        for t in range(len(keys_l)):
            s_id = anchors_l[keys_l[t]][0]
            t_id = anchors_l[keys_l[t]][1]
             
            l_i = [p for p in range(len(section_ids)) if section_ids[p] == keys_l[t].split('+')[0]][0]
            l_j = [p for p in range(len(section_ids)) if section_ids[p] == keys_l[t].split('+')[1]][0]
            ds1 = adata_l_st[l_i].uns['deconvolution']['ot'].to_numpy().T       
            ds2 = adata_l_st[l_j].uns['deconvolution']['ot'].to_numpy().T
            if smooth:
                ds1_s = ds1@adj_sc_norm
                ds2_s = ds2@adj_sc_norm
            else:
                ds1_s = ds1.copy()
                ds2_s = ds2.copy()
            # compute the smilarity of each pairs
            t_id_o = [t_id[p]- ds1.shape[0] for p in range(len(t_id))]  
            d_sc = np.zeros([len(s_id),1])
            for i in range(len(s_id)):
                d_sc[i,0] = stats.spearmanr(ds1_s[s_id[i],:], ds2_s[t_id_o[i],:]).statistic
            # remove the smaller ones
            #z_d_sc = stats.zscore(d_sc[:,0])
            f_id = np.where(d_sc > cor_cutoff)[0]
            s_id_f = [s_id[f_id[q]] for q in range(len(f_id))]
            t_id_f = [t_id[f_id[q]] for q in range(len(f_id))]
            anchors_l_new[keys_l[t]] = [s_id_f, t_id_f]
            
    if dev_method == 'mlp':
        anchors_l_new = dict()
        keys_l = list(anchors_l.keys())
        for t in range(len(keys_l)):
            s_id = anchors_l[keys_l[t]][0]
            t_id = anchors_l[keys_l[t]][1]
             
            l_i = [p for p in range(len(section_ids)) if section_ids[p] == keys_l[t].split('+')[0]][0]
            l_j = [p for p in range(len(section_ids)) if section_ids[p] == keys_l[t].split('+')[1]][0]
            ds1 = adata_l_st[l_i].uns['deconvolution'].to_numpy().T     
            ds2 = adata_l_st[l_j].uns['deconvolution'].to_numpy().T
            ds1_s = ds1.copy()
            ds2_s = ds2.copy()
            # compute the smilarity of each pairs
            t_id_o = [t_id[p]- ds1.shape[0] for p in range(len(t_id))]  
            d_sc = np.zeros([len(s_id),1])
            for i in range(len(s_id)):
                d_sc[i,0] = stats.spearmanr(ds1_s[s_id[i],:], ds2_s[t_id_o[i],:]).statistic
            # remove the smaller ones
            #z_d_sc = stats.zscore(d_sc[:,0])
            f_id = np.where(d_sc > cor_cutoff)[0]
            s_id_f = [s_id[f_id[q]] for q in range(len(f_id))]
            t_id_f = [t_id[f_id[q]] for q in range(len(f_id))]
            anchors_l_new[keys_l[t]] = [s_id_f, t_id_f]
        
            
    return anchors_l_new

# filter the mnn pairs from representation that some one has zero values
def filter_zeroshot_mnns(adata_concat, adata_l, mnns_adj, use_rep, iter_comb, min_cut = 0,batch_name='batch_name'):
    cell_names = adata_concat.obs_names
    batch_list = adata_concat.obs[batch_name]
        
    cells = []
    for i in batch_list.unique():
        cells.append(cell_names[batch_list == i])

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(cells)), 2))
     
    anchors_l = dict()
    for t in range(0,len(iter_comb)):
        comb = iter_comb[t]
        i = comb[0]
        j = comb[1]
        key_name1 = batch_name_df.loc[i].values[0] + "+" + batch_name_df.loc[j].values[0]
        mnn_adj = mnns_adj[key_name1]            
        
        adata_source = adata_l[i]
        adata_target = adata_l[j]
        
        if use_rep == 'deconvolution':
            ds1 = adata_source.uns[use_rep]['ot'].to_numpy().T
            ds2 = adata_target.uns[use_rep]['ot'].to_numpy().T
        else:
            ds1 = adata_source.obsm[use_rep]
            ds2 = adata_target.obsm[use_rep]
        
        ds = np.concatenate((ds1,ds2), axis = 0)
        
        # get index
        s_id_t, t_id_t = get_mnn_index(mnn_adj)
        # filter zeroshot pairs
        # check the sum of these corresponding ot
        rs = np.sum(ds[s_id_t,],axis = 1)
        rt = np.sum(ds[t_id_t,],axis = 1)
        id_s = [k for k in range(len(s_id_t)) if rs[k] > min_cut]
        id_t = [k for k in range(len(t_id_t)) if rt[k] > min_cut]
        id_left = list(set(id_s).intersection(id_t))
        s_id_left = [s_id_t[id_left[k]] for k in range(len(id_left))]
        t_id_left = [t_id_t[id_left[k]] for k in range(len(id_left))]
        anchors_l[key_name1] = [s_id_left, t_id_left]
    return anchors_l
    

def compute_bandwidth(X):
    from scipy.spatial.distance import pdist, squareform
    pairwise_dists = squareform(pdist(X))
    median_dist = np.median(pairwise_dists)
    bandwidth = median_dist / np.sqrt(2)
    return bandwidth

def mmd_rbf(source, target, kernel, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params: 
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul: 
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    sum(kernel_val): 多个核矩阵之和
    '''
    
    #n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    
    
    xx, yy, zz = torch.mm(source, source.t()), torch.mm(target, target.t()), torch.mm(source, target.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = compute_bandwidth(np.vstack((source.detach().numpy(), target.detach().numpy())))
    
    
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_range = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    #XX, YY, XY = (torch.zeros(xx.shape).to(device),
    #              torch.zeros(xx.shape).to(device),
    #              torch.zeros(xx.shape).to(device))
    XX, YY, XY = (torch.zeros(xx.shape),
                  torch.zeros(xx.shape),
                  torch.zeros(xx.shape))

    if kernel == "multiscale":
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
        
    if kernel == "rbf":
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
    
    loss=torch.mean(XX)+torch.mean(YY)-2*torch.mean(XY)
    return loss

def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list


def compute_distance(adata, metric = 'knn',n_neighbors = 5):
    if metric == 'knn':
        D = compute_distance_from_adj(adata, n_neighbors)
    else:
        D = distance.cdist(adata.obsm['embedding'],adata.obsm['embedding'],metric = metric)# euclidean
    return D


def compute_adj(adata, n_neighbors):
    """
    Calculates adj

    param: adata - AnnData object
    
    return: adj
    
    """
    #sc.pp.scale(adata)
    #sc.tl.pca(adata)
    if 'embedding' in adata.obsm.keys(): 
        if isinstance(adata.obsm['embedding'],np.ndarray):
            adata.obsm['X_pca'] = adata.obsm['embedding']
        else:
            adata.obsm['X_pca'] = adata.obsm['embedding'].to_numpy()
        
    else:
        adata_n = adata.copy()
        sc.pp.scale(adata_n)
        sc.tl.pca(adata_n)
        adata.obsm['X_pca'] = adata_n.obsm['X_pca']
    sc.pp.neighbors(adata, n_neighbors)
    adj = adata.obsp['connectivities'].toarray()
    
    return adj


def compute_adj_smooth(adata,label, p_adj_cutoff = 0.05, lf_cutoff = 0.1):
    """
    Calculates adj

    param: adata - AnnData object
    
    return: adj
    
    """
    sc.tl.rank_genes_groups(adata, "Ground Truth", method="wilcoxon")
    
    similarity_markers = np.zeros([len(label), len(label)])
    
    for i in range(len(label)):
        p_adj_i = adata.uns['rank_genes_groups']['pvals_adj'][label[i]]
        index_p_adi = set(np.where(p_adj_i < p_adj_cutoff)[0].tolist())
        lf_i = adata.uns['rank_genes_groups']['logfoldchanges'][label[i]]
        if pd.isna(lf_i).any():
            index_i = index_p_adi
        else:
            index_lf_i = set(np.where(np.abs(lf_i) > lf_cutoff)[0].tolist())
            index_i = index_p_adi.intersection(index_lf_i)
        if len(index_i) > 0:
            gene_names_i = adata.uns['rank_genes_groups']['names'][label[i]]
            genes_i = gene_names_i[list(index_i)]    
        for j in range(len(label)):
            p_adj_j = adata.uns['rank_genes_groups']['pvals_adj'][label[j]]
            index_p_adj = set(np.where(p_adj_j < p_adj_cutoff)[0].tolist())
            lf_j = adata.uns['rank_genes_groups']['logfoldchanges'][label[j]]
            if pd.isna(lf_j).any():
                index_j = index_p_adj
            else:
                index_lf_j = set(np.where(np.abs(lf_j) > lf_cutoff)[0].tolist())
                index_j = index_p_adj.intersection(index_lf_j)
            if len(index_j) > 0:
                gene_names_j = adata.uns['rank_genes_groups']['names'][label[j]]
                genes_j = gene_names_j[list(index_j)]
                d1 = len(set(genes_i.tolist()).intersection(set(genes_j.tolist())))
                d2 = len(set(genes_i.tolist()).union(set(genes_j.tolist())))
                
                if d2 == 0:
                    similarity_markers[i,j] = 0 # because there is no significant markers, the reason might be bad cluster or there are some similar clusters
                else:
                    similarity_markers[i,j] = d1/d2
        
    adj_s = normalizeAdjacency(similarity_markers)
    return adj_s

def compute_distance_from_adj(adata, n_neighbors):
    """
    Calculates cost distance for spatial data or single cell data based on embedding.

    param: adata - AnnData object
    
    return: D_s - cost distance
    
    """
    adj = compute_adj(adata, n_neighbors)
    D_s = np.sqrt(2*(1-adj))
    return D_s

# normalized using similarity matrix
def normalizeAdjacency(W):
    """
    NormalizeAdjacency: Computes the degree-normalized adjacency matrix

    Input:

        W (np.array): adjacency matrix

    Output:

        A (np.array): degree-normalized adjacency matrix
    """
    # Check that the matrix is square
    assert W.shape[0] == W.shape[1]
    # Compute the degree vector
    d = np.sum(W, axis = 1)
    # Invert the square root of the degree
    d = 1/np.sqrt(d)
    # And build the square root inverse degree matrix
    D = np.diag(d)
    # Return the Normalized Adjacency
    return D @ W @ D 


## Covert a sparse matrix into a dense matrix
to_dense_array = lambda X: np.array(X.todense()) if isinstance(X,sparse.csr.spmatrix) else X

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep] 

def generalized_kl_divergence(X, Y):
    """
    Returns pairwise generalized KL divergence (over all pairs of samples) of two matrices X and Y.

    param: X - np array with dim (n_samples by n_features)
    param: Y - np array with dim (m_samples by n_features)

    return: D - np array with dim (n_samples by m_samples). Pairwise generalized KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.matrix([np.dot(X[i], log_X[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X, log_Y.T)
    sum_X = np.sum(X, axis=1)
    sum_Y = np.sum(Y, axis=1)
    D = (D.T - sum_X).T + sum_Y.T
    return np.asarray(D)

def kl_divergence(X, Y):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.
    
    param: X - np array with dim (n_samples by n_features)
    param: Y - np array with dim (m_samples by n_features)
    
    return: D - np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    X = np.array(X)
    Y = np.array(Y)
    X = X/np.sum(X,axis=1, keepdims=True)
    Y = Y/np.sum(Y,axis=1, keepdims=True)
    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.matrix([np.dot(X[i],log_X[i].T) for i in range(X.shape[0])])
    D = X_log_X.T - np.dot(X,log_Y.T)
    return np.asarray(D)

