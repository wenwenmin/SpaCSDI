import math
import torch
import itertools
import random
import scipy.sparse
import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
import torch.nn as nn
from scipy import sparse
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from typing import List, Optional, Union, Any

from SMILE.analysis import mclust_init
from SMILE.utils import get_mean_reconstruct_sc, label_to_int, identify_anchors, mmd_rbf


def get_mlp(adata_l,tag_l,section_ids, hidden_features, out_features, theta, lr, label_name, class_rep, gamma):
    # train a classifier for sc
    ref_sc_id = []
    n = len(adata_l)
    for i in range(n):
        if tag_l[i] == 'single cell': 
            ref_sc_id.append(i)

    adata_sc = adata_l[ref_sc_id[0]].copy()
    # return the corresponding label information
    
    adata_sc, label_list, adata_sc_label = label_to_int(adata_sc, label_name)
    
    label_sc = torch.tensor(adata_sc.obs['pre_label'])
        
    label_predictor_sc = LabelPredictor_sc(hidden_features, out_features, class_features = len(set(list(label_sc.numpy()))))
    optimizer_C_sc = optim.Adam(label_predictor_sc.parameters(), lr=lr)  
    class_criterion = nn.CrossEntropyLoss() 
    print('Training classifier...')
    emb = adata_sc.obsm['embedding'].copy()
    hidden_emb = torch.tensor(emb)
    
    # average across cell types
    sc_mean = get_mean_reconstruct_sc(adata_sc, label_list, adata_sc_label,class_rep)
    sc_recon_mean = torch.FloatTensor(sc_mean)
    
    # Normalize features for consistence between ST and scRNA-seq
    #sc_recon_mean = F.normalize(sc_recon_mean, p=2, eps=1e-12, dim=1)# cell type*genes
    # add constraint on within slice and between slices
    for epoch in range(200):
        label_predictor_sc.train()
        class_logits_sc = label_predictor_sc(hidden_emb)
        loss_sc = class_criterion(class_logits_sc, label_sc.long())
        loss_reconstruct_l = []
        loss_class_st_between = []
        loss_class_st_within = []
        class_prob_sc_l = []
        for i in range(n):
            if tag_l[i] != 'single cell':
                class_logits_sc_i = label_predictor_sc(torch.tensor(adata_l[i].obsm['embedding'].copy()))
                #class_prob_sc_i = class_logits_sc_i.softmax(-1) # spot*cell type
                class_prob_sc_i = F.softmax(class_logits_sc_i,dim=1)
                # print(class_prob_sc_i.shape)
                class_prob_sc_l.append(class_prob_sc_i)
                class_dists_i = torch.cdist(class_prob_sc_i,class_prob_sc_i, p=2)
                class_dists_i = torch.div(class_dists_i, torch.max(class_dists_i))
                z_dists_i = torch.cdist(torch.tensor(adata_l[i].obsm['embedding']), torch.tensor(adata_l[i].obsm['embedding']), p=2)
                z_dists_i = torch.div(z_dists_i, torch.max(z_dists_i))
                            
                n_items = class_prob_sc_i.size(dim=0) * class_prob_sc_i.size(dim=0)
                penalty_dev_i = torch.div(torch.sum(torch.mul(1.0 - z_dists_i, class_dists_i)), n_items)
                loss_class_st_within.append(penalty_dev_i)
                # reconstruction
                if class_rep == 'reconstruct':
                    st_recon_i = torch.FloatTensor(adata_l[i].obsm['reconstruct']) # spot*genes
                elif class_rep == 'embedding':
                    st_recon_i = torch.FloatTensor(adata_l[i].obsm['embedding']) # spot*out_features
                else:
                    if scipy.sparse.issparse(adata_l[i].X):
                        st_recon_i = torch.FloatTensor(adata_l[i].X.todense()) # spot*genes

                    else:
                        st_recon_i = torch.FloatTensor(adata_l[i].X) # spot*genes
                #st_recon_i = F.normalize(st_recon_i, p=2, eps=1e-12, dim=1)
                #sc_i = torch.matmul(class_prob_sc_i, torch.transpose(sc_recon_mean,0,1))# spot*genes
            
                sc_i = torch.matmul(class_prob_sc_i, sc_recon_mean)# spot*genes
                # use constrastive learning
                emb_graph = torch.from_numpy(adata_l[i].obsp['adj_t'].astype("float32"))
                #loss_recon_i1 = Noise_Cross_Entropy(sc_i, st_recon_i,emb_graph)
                
                #sc_i = F.normalize(sc_i, p=2, eps=1e-12, dim=1)
                loss_recon_i = F.mse_loss(sc_i, st_recon_i, reduction='mean')
                #loss_recon_i = loss_recon_i1 + loss_recon_i2
                #loss_recon_i = mmd_rbf(sc_i, st_recon_i, kernel = 'rbf')
                loss_reconstruct_l.append(loss_recon_i)

        loss_reg_st_within = torch.sum(torch.stack(loss_class_st_within))/n
        loss_rec = torch.sum(torch.stack(loss_reconstruct_l))/n
        
        loss_sc_all = gamma*loss_rec+loss_sc + theta*loss_reg_st_within
        optimizer_C_sc.zero_grad()
        loss_sc_all.backward()
        optimizer_C_sc.step()
          
        if epoch % 10 == 0:
            print('epoch {:>3d}: overall loss: {:6.4f},sc classifier loss: {:6.4f},representation loss: {:6.4f},within spatial regularization loss: {:6.4f}'.format(epoch, loss_sc_all,loss_sc,loss_rec,loss_reg_st_within))

    with torch.no_grad():
                                                                                                                
        acc_sc = 0
        # check the classifier accuracy                       
        preds_i = label_predictor_sc(hidden_emb)
        pred_cls_i = preds_i.data.max(1)[1]
        acc_sc += pred_cls_i.eq(label_sc.data).cpu().sum()
        print("single cell data classification: Avg Accuracy = {:2%}".format(acc_sc/len(label_sc))) 
        for i in range(n):
            if tag_l[i] != 'single cell':
                hidden_emb_i = torch.tensor(adata_l[i].obsm['embedding'].copy()) 
                preds_i = label_predictor_sc(hidden_emb_i)
                #class_prob_sc_i = preds_i.softmax(-1)
                class_prob_sc_i = F.softmax(preds_i,dim=1)
                mlp_m = pd.DataFrame(class_prob_sc_i.cpu().detach().numpy().T, index = label_list, columns = adata_l[i].obs.index) # classifier order      
                adata_l[i].obsm['deconvolution'] = class_prob_sc_i.cpu().detach().numpy()
                adata_l[i].uns['deconvolution'] = mlp_m
    return adata_l


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
    
    
class Discriminator_sep(nn.Module):
    def __init__(self, n_h):
        super(Discriminator_sep, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits    
    
    
class LabelPredictor_sc(nn.Module):
    def __init__(self, hidden_features, out_features, class_features):
        super(LabelPredictor_sc, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(out_features, hidden_features),
            nn.ReLU(),

            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),

            nn.Linear(hidden_features, class_features)
        )

    def forward(self, h):
        c = self.layer(h)
        return c
    
    
    
class LabelPredictor_st(nn.Module):
    def __init__(self, hidden_features, out_features, class_features):
        super(LabelPredictor_st, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(out_features, hidden_features),
            nn.ReLU(),

            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),

            nn.Linear(hidden_features, class_features),
        )

    def forward(self, h):
        c = self.layer(h)
        return c
    
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 

def permutation(feature):
    # fix_seed(FLAGS.random_seed) 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 


def get_weighted_graph(adj_f, adj_t, adj_i, att):
    # where there are more than two adjs
    if adj_f is not None:
        n = adj_f.shape[0]        
    if adj_t is not None:
        n = adj_t.shape[0]
    if adj_i is not None:
        n = adj_i.shape[0]
    if adj_f is not None and adj_t is not None and adj_i is None: # feature+spatial
        att1 = att[:,0,:]
        att2 = att[:,1,:]
        adj1 = torch.mm(torch.diag(att1.view(n)), adj_f)
        adj2 = torch.mm(torch.diag(att2.view(n)), adj_t)
        adj12 = adj1+adj2
        adj = (adj12+torch.transpose(adj12, dim0=0, dim1=1))/2
    if adj_f is not None and adj_t is None and adj_i is not None: # feature+image
        att1 = att[:,0,:]
        att2 = att[:,1,:]
        adj1 = torch.mm(torch.diag(att1.view(n)), adj_f)
        adj2 = torch.mm(torch.diag(att2.view(n)), adj_i)
        adj12 = adj1+adj2
        adj = (adj12+torch.transpose(adj12, dim0=0, dim1=1))/2
    
    if adj_f is None and adj_t is not None and adj_i is not None: # spatial+image
        att1 = att[:,0,:]
        att2 = att[:,1,:]
        adj1 = torch.mm(torch.diag(att1.view(n)), adj_t)
        adj2 = torch.mm(torch.diag(att2.view(n)), adj_i)
        adj12 = adj1+adj2
        adj = (adj12+torch.transpose(adj12, dim0=0, dim1=1))/2
        
    if adj_f is not None and adj_t is not None and adj_i is not None: # feature+spatial+image
        att1 = att[:,0,:]
        att2 = att[:,1,:]
        att3 = att[:,2,:]
        adj1 = torch.mm(torch.diag(att1.view(n)), adj_f)
        adj2 = torch.mm(torch.diag(att2.view(n)), adj_t)
        adj3 = torch.mm(torch.diag(att3.view(n)), adj_i)
        adj123 = adj1+adj2+adj3
        adj = (adj123+torch.transpose(adj123, dim0=0, dim1=1))/3   
    return adj
    
def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list

def get_weight(adj):
    # create sparse matrix
    row_col = []
    edge_arr = []
    rows, cols = adj.nonzero()
    edge_nums = adj.getnnz() 
    for i in range(edge_nums):
        row_col.append([rows[i], cols[i]])
        edge_arr.append(adj.data[i])
    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_weight = torch.tensor(np.array(edge_arr), dtype=torch.float)
    return edge_index, edge_weight

class GraphsEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GraphsEncoder, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, out_features)
        self.norm = torch.nn.BatchNorm1d(hidden_features)
        
        self.conv3 = GCNConv(out_features, hidden_features)
        self.conv4 = GCNConv(hidden_features, in_features)
                
        self.attentionf = Attention(self.out_features)
        self.disc = Discriminator_sep(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        

    def forward(self, feat, feat_a, adj_f, adj_t, adj_i):
        hiden_emb, att, h1_f, h1_t, h1_i, h2, ret = None, None, None, None, None, None, None
        # print("feat:",feat, feat_a, adj_f, adj_t, adj_i)
        # GCN on feature space to obtain Zf
        if adj_f is not None:
            # extract edge list and edge weight
            edge_index_f, edge_weight_f = get_weight(sparse.coo_matrix(adj_f.cpu().detach().numpy()))
            h1_f = self.conv1(feat, edge_index_f, edge_weight=edge_weight_f)
            h1_f = self.norm(h1_f)
            h1_f = F.relu(h1_f)
            h1_f = self.conv2(h1_f, edge_index_f, edge_weight=edge_weight_f)
        
        # GCN on spatial to obtain Zt
        if adj_t is not None:
            edge_index_t, edge_weight_t = get_weight(sparse.coo_matrix(adj_t.cpu().detach().numpy()))
            h1_t = self.conv1(feat, edge_index_t, edge_weight=edge_weight_t)
            h1_t = self.norm(h1_t)
            h1_t = F.relu(h1_t)
            h1_t = self.conv2(h1_t, edge_index_t, edge_weight=edge_weight_t)
        
            # GCN on corruption spatial to obtain Za     
            h1_a = self.conv1(feat_a, edge_index_t,edge_weight=edge_weight_t)
            h1_a = self.norm(h1_a)
            h1_a = F.relu(h1_a)
            h1_a = self.conv2(h1_a, edge_index_t, edge_weight=edge_weight_t)
        
            # build local summary vectors
            g = self.read(h1_t, adj_t) 
            g = self.sigm(g)  
            ret = self.disc(g, h1_t, h1_a)  
        else:
            ret = None
        
        
        # GCN on image space to obtain Zi
        if adj_i is not None:
            edge_index_i, edge_weight_i = get_weight(sparse.coo_matrix(adj_i.cpu().detach().numpy()))
            h1_i = self.conv1(feat, edge_index_i, edge_weight=edge_weight_i)
            h1_i = self.norm(h1_i)
            h1_i = F.relu(h1_i)
            h1_i = self.conv2(h1_i, edge_index_i, edge_weight=edge_weight_i)
                    
        
        if adj_f is not None and adj_t is None and adj_i is None: # only feature
            hiden_emb = h1_f
            att = None
            h1_t = None
            h1_i = None
            h2 = self.conv3(hiden_emb, edge_index_f, edge_weight=edge_weight_f)
            h2 = self.norm(h2)
            h2 = F.relu(h2)
            h2 = self.conv4(h2, edge_index_f, edge_weight=edge_weight_f)
            h2 = F.relu(h2)
    
        if adj_f is None and adj_t is not None and adj_i is None: # only spatial
            hiden_emb = h1_t
            att = None
            h1_f = None
            h1_i = None  
            h2 = self.conv3(hiden_emb, edge_index_t, edge_weight=edge_weight_t)
            h2 = self.norm(h2)
            h2 = F.relu(h2)
            h2 = self.conv4(h2, edge_index_t, edge_weight=edge_weight_t)
            h2 = F.relu(h2)
            
        if adj_f is None and adj_t is None and adj_i is not None: # only image
            hiden_emb = h1_i
            att = None
            h1_f = None
            h1_t = None 
            h2 = self.conv3(hiden_emb, edge_index_i, edge_weight=edge_weight_i) 
            h2 = self.norm(h2)
            h2 = F.relu(h2)
            h2 = self.conv4(h2, edge_index_i, edge_weight=edge_weight_i)
            h2 = F.relu(h2)
            
        # attention used to combine Zt or Zf or optimal zi
        if adj_f is not None and adj_t is not None and adj_i is None: # feature+spatial
            hiden_emb = torch.stack([h1_f, h1_t], dim=1)
            hiden_emb, att = self.attentionf(hiden_emb)
             # get weighted graph 
            adj_w = get_weighted_graph(adj_f, adj_t, adj_i, att)
            # extract edge list and edge weight
            edge_index, edge_weight = get_weight(sparse.coo_matrix(adj_w.cpu().detach().numpy()))
            # decoder with GCN on weighted graph
            h2 = self.conv3(hiden_emb, edge_index, edge_weight=edge_weight)
            h2 = self.norm(h2)
            h2 = F.relu(h2)
            h2 = self.conv4(h2, edge_index, edge_weight=edge_weight)
            h2 = F.relu(h2)
            h1_i = None
            
        if adj_f is not None and adj_t is None and adj_i is not None: # feature+image
            hiden_emb = torch.stack([h1_f, h1_i], dim=1)
            hiden_emb, att = self.attentionf(hiden_emb)
            # get weighted graph 
            adj_w = get_weighted_graph(adj_f, adj_t, adj_i, att)
            # extract edge list and edge weight
            edge_index, edge_weight = get_weight(sparse.coo_matrix(adj_w.cpu().detach().numpy()))
            # decoder with GCN on weighted graph
            h2 = self.conv3(hiden_emb, edge_index, edge_weight=edge_weight)
            h2 = self.norm(h2)
            h2 = F.relu(h2)
            h2 = self.conv4(h2, edge_index, edge_weight=edge_weight)
            h2 = F.relu(h2)
            h1_t = None
            
            
        if adj_f is None and adj_t is not None and adj_i is not None: # spatial+image
            hiden_emb = torch.stack([h1_t, h1_i], dim=1)
            hiden_emb, att = self.attentionf(hiden_emb) 
            # get weighted graph 
            adj_w = get_weighted_graph(adj_f, adj_t, adj_i, att)
            # extract edge list and edge weight
            edge_index, edge_weight = get_weight(sparse.coo_matrix(adj_w.cpu().detach().numpy()))
            # decoder with GCN on weighted graph
            h2 = self.conv3(hiden_emb, edge_index, edge_weight=edge_weight)
            h2 = self.norm(h2)
            h2 = F.relu(h2)
            h2 = self.conv4(h2, edge_index, edge_weight=edge_weight)
            h2 = F.relu(h2)
            h1_f = None
            
        if adj_f is not None and adj_t is not None and adj_i is not None: # feature+spatial+image
            hiden_emb = torch.stack([h1_f,h1_t, h1_i], dim=1)
            hiden_emb, att = self.attentionf(hiden_emb)
            # get weighted graph 
            adj_w = get_weighted_graph(adj_f, adj_t, adj_i, att)
            # extract edge list and edge weight
            edge_index, edge_weight = get_weight(sparse.coo_matrix(adj_w.cpu().detach().numpy()))
            # decoder with GCN on weighted graph
            h2 = self.conv3(hiden_emb, edge_index, edge_weight=edge_weight)
            h2 = self.norm(h2)
            h2 = F.relu(h2)
            h2 = self.conv4(h2, edge_index, edge_weight=edge_weight)
            h2 = F.relu(h2)
           
        return hiden_emb, att, h1_f, h1_t, h1_i, h2, ret
        
def SMILE(adata_l, tag_l, section_ids, multiscale, n_clusters_l, in_features, feature_method, hidden_features, out_features, iter_comb,
          anchors_all, use_rep_anchor, alpha, beta, lamb, theta, gamma, edge_weights, add_topology, add_feature,
          add_image, add_sc, spatial_regularization_strength,
            lr=1e-3, subepochs = 200, epochs = 400, class_rep = None,
          label_name='Ground Truth', anchor_size = 5000, agg_neigh = 10, mnn_neigh=30,
          random_seed=1, gpu=0, regularization_acceleration=True, edge_subset_sz=1000000):
    '''
      Args:
        adata_l: the adata list 
        tag_l: the data type list
        section_ids: the name of each slice
        multiscale: logisctical True of False
        in_features: the input feature dimension used in the model
        hidden_features: the hidden dimension used in the model (defalut: 512)
        out_features: the dimesion of latent embeddings (default: 50)
        label_source: the label information of source data (can combine with adata_l)
        spatial_coords_l: the list of spatial coordinates (can combine with adata_l)
        use_rep: the method used for computing mnn pairs
        label_name: the column in adata, which is used to check the accuracy (can combine with adata_l)
        
    '''
    
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    n = len(adata_l)
    feat_l = []
    feat_a_l = []
    spatial_coords_l = []
    adj_f_l = []
    edge_index_f_l = []
    adj_t_l = []
    edge_index_t_l = []
    adj_i_l = []
    edge_index_i_l = []
    for i in range(n):
        if scipy.sparse.issparse(adata_l[i].X): 
            feat_l.append(torch.from_numpy(adata_l[i].X.todense()).to(torch.float))
        else:
            feat_l.append(torch.from_numpy(adata_l[i].X).to(torch.float))
        feat_a_l.append(permutation(feat_l[i]))
        # spatial coords
        if add_topology and 'spatial' in adata_l[i].obsm.keys(): 
            spatial_coords_l.append(torch.tensor(adata_l[i].obsm['spatial']).float()) 
        else:
            spatial_coords_l.append(None)
        # adj_f  
        if add_feature and 'adj_f' in adata_l[i].obsp.keys(): 
            adj_f_l.append(torch.from_numpy(np.multiply(adata_l[i].obsp['adj_f'].todense().astype("float32"),edge_weights[1])))
            edge_index_f_l.append(sparse_mx_to_torch_edge_list(sparse.csr_matrix(adata_l[i].obsp['adj_f'])))

        else:
            if tag_l[i] == 'single cell' and add_feature == False:
                adj_f_l.append(torch.from_numpy(np.multiply(adata_l[i].obsp['adj_f'].todense().astype("float32"),edge_weights[1])))
                edge_index_f_l.append(sparse_mx_to_torch_edge_list(sparse.csr_matrix(adata_l[i].obsp['adj_f'])))    
            else:
                adj_f_l.append(None) 
                edge_index_f_l.append(None)
            
        # adj_t
        if add_topology and 'adj_t' in adata_l[i].obsp.keys(): 
            adj_t_l.append(torch.from_numpy(np.multiply(adata_l[i].obsp['adj_t'].astype("float32"),edge_weights[0])))
            edge_index_t_l.append(sparse_mx_to_torch_edge_list(sparse.csr_matrix(adata_l[i].obsp['adj_t'])))

        else:
            adj_t_l.append(None) 
            edge_index_t_l.append(None)        
            
                    
        # adj_i
        if add_image and 'adj_i' in adata_l[i].obsp.keys():
            if scipy.sparse.issparse(adata_l[i].obsp['adj_i']): 
                adj_i_l.append(torch.from_numpy(np.multiply(adata_l[i].obsp['adj_i'].todense().astype("float32"),edge_weights[2])))
            else:
                adj_i_l.append(torch.from_numpy(np.multiply(adata_l[i].obsp['adj_i'].astype("float32"),edge_weights[2])))
            edge_index_i_l.append(sparse_mx_to_torch_edge_list(sparse.csr_matrix(adata_l[i].obsp['adj_i'])))

        else:
            adj_i_l.append(None)
            edge_index_i_l.append(None)
    
    ref_sc_id = []
    ref_st_id = []
    for i in range(n):
        if tag_l[i] == 'single cell':
            ref_sc_id.append(i)# usually there is one ref
        if tag_l[i] == 'ST_ref':
            ref_st_id.append(i) # refernece for spatial data

    if len(ref_sc_id) > 0:
        st_flag = [j for j in range(n) if j!= ref_sc_id[0]]
    else:
        st_flag = [j for j in range(n)]
            
    if iter_comb is None:        
        if len(ref_sc_id) > 0:
            iter_comb_a = list(itertools.combinations(range(n), 2))
            iter_comb = []
            for comb in iter_comb_a:
                if comb[0] != ref_sc_id[0] and comb[1] != ref_sc_id[0] :
                    iter_comb.append(comb)
        else:
            iter_comb = list(itertools.combinations(range(n), 2))
                                                                 
    if feature_method is None or feature_method == 'GCNConv':
        feature_extractor = GraphsEncoder(in_features = in_features, hidden_features = hidden_features, out_features = out_features)
    # if feature_method == 'LGCN_mlp':
    #     feature_extractor = GraphsEncoder_LGCN_mlp(in_features = in_features, hidden_features = hidden_features, out_features = out_features)
    # if feature_method == 'SGCN':
    #     feature_extractor = GraphsEncoder_SGCN(in_features = in_features, hidden_features = hidden_features, out_features = out_features)
    # if feature_method == 'sparse':
    #     feature_extractor = GraphsEncoder_sparse(in_features = in_features, hidden_features = hidden_features, out_features = out_features)
    
    optimizer_F = optim.Adam(feature_extractor.parameters(), lr=lr)

    # if the label is available, train classifier for each data
    label_predictor_st_l, optimizer_C_st_l = [], []
    for i in range(n):
        if tag_l[i] == 'ST_ref':
            label_i = torch.tensor(adata_l[i].obs['ref'].to_numpy().astype(int))
            label_predictor_st_i = LabelPredictor_st(hidden_features, out_features, class_features = len(set(list(label_i.numpy()))))
            label_predictor_st_l.append(label_predictor_st_i)
            optimizer_C_st_l.append(optim.Adam(label_predictor_st_i.parameters(), lr=lr))
        elif tag_l[i] != 'single cell' and multiscale == True and len(n_clusters_l) > 1:
            label_predictor_st_i = LabelPredictor_st(hidden_features, out_features, class_features = n_clusters_l[i])
            label_predictor_st_l.append(label_predictor_st_i)
            optimizer_C_st_l.append(optim.Adam(label_predictor_st_i.parameters(), lr=lr))    
        else:
            label_predictor_st_l.append(None)
            optimizer_C_st_l.append(None)
            
    if len(ref_sc_id) > 0 and multiscale and len(n_clusters_l) == 1:
        label_predictor_st = LabelPredictor_st(hidden_features, out_features, class_features = n_clusters_l[0])
        optimizer_C_st = optim.Adam(label_predictor_st.parameters(), lr=lr)
    else:
        label_predictor_st = None
        optimizer_C_st = None
                
    running_F_loss, running_D_loss, running_C_st_loss = 0.0, 0.0, 0.0    
    
    
    print('Pretraining to extract embeddings of spots...')
    for epoch in range(subepochs):    
        # Step 1 : train feature extractor step
        feature_extractor.train()
        optimizer_F.zero_grad()      
        hidden_emb_l, h1_f_l, h1_t_l, h1_i_l, h2_l, ret_l = [], [] ,[], [], [], []
        loss_l = []
        loss_c_l = []
        loss_c_train_l = []
        for i in range(n):
            if tag_l[i] != 'single cell':            
                if feature_method == 'LGCN_mlp':
                    hidden_emb, _, h1_f, h1_t, h1_i, h2, ret = feature_extractor(feat_l[i], feat_a_l[i], edge_index_f_l[i], adj_f_l[i], edge_index_t_l[i], adj_t_l[i], edge_index_i_l[i], adj_i_l[i])
        
                if feature_method is None or feature_method == 'GCNConv' or feature_method == 'SGCN' or feature_method == 'sparse':
                    hidden_emb, _, h1_f, h1_t, h1_i, h2, ret = feature_extractor(feat_l[i], feat_a_l[i], adj_f_l[i], adj_t_l[i], adj_i_l[i])                                                        
                                                                   
                hidden_emb_l.append(hidden_emb)
                h1_f_l.append(h1_f)
                h1_t_l.append(h1_t)
                h1_i_l.append(h1_i)
                h2_l.append(h2)
                ret_l.append(ret)
                if h1_t is not None:
                    ret_CSL = torch.from_numpy(np.concatenate([np.ones([feat_l[i].shape[0], 1]), np.zeros([feat_l[i].shape[0], 1])], axis=1))
                    loss_CSL = nn.BCEWithLogitsLoss()               
                    loss_sl = loss_CSL(ret, ret_CSL)
                else:
                    loss_sl = 0
            
                loss_feat = F.mse_loss(feat_l[i], h2)
                loss_feat_i = loss_feat
                loss_i1 = loss_feat_i + alpha*loss_sl
            
                if spatial_coords_l[i] is not None:
                    if regularization_acceleration:
                        nodes = feat_l[i].shape[0]
                        cell_random_subset_1, cell_random_subset_2 = torch.randint(0, nodes , (edge_subset_sz,)), torch.randint(0, nodes, (edge_subset_sz,))
                        z1, z2 = torch.index_select(h1_t_l[i], 0, cell_random_subset_1), torch.index_select(h1_t_l[i], 0, cell_random_subset_2)
                        c1, c2 = torch.index_select(spatial_coords_l[i], 0, cell_random_subset_1), torch.index_select(spatial_coords_l[i], 0, cell_random_subset_2)
                        pdist = torch.nn.PairwiseDistance(p=2)

                        z_dists = pdist(z1, z2)
                        z_dists = z_dists / torch.max(z_dists)
                        sp_dists = pdist(c1, c2)
                        sp_dists = sp_dists / torch.max(sp_dists)
                        n_items = z_dists.size(dim=0)
                        penalty_i = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items)
                    else:
                        z_dists = torch.cdist(h1_t_l[i],h1_t_l[i], p=2)
                        z_dists = torch.div(z_dists, torch.max(z_dists))
                        sp_dists = torch.cdist(spatial_coords_l[i], spatial_coords_l[i], p=2)
                        sp_dists = torch.div(sp_dists, torch.max(sp_dists))
                        n_items = h1_t_l[i].size(dim=0) * h1_t_l[i].size(dim=0)
                        penalty_i = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items)                    
                else:
                    penalty_i = 0
                
                loss_i = loss_i1 + spatial_regularization_strength * penalty_i
            
                
                if tag_l[0] == 'ST_ref' and label_predictor_st_l[i] is not None:
                    label_predictor_i = label_predictor_st_l[i]
                    optimizer_C_i = optimizer_C_st_l[i]                            
                    class_criterion = nn.CrossEntropyLoss()                
                    label_i = torch.tensor(adata_l[i].obs['ref'].to_numpy().astype(int))
                    class_logits_i = label_predictor_i(hidden_emb.detach())
                    loss_ci_train = class_criterion(class_logits_i, label_i.long())
                    loss_ci_train.backward() ####
                    optimizer_C_i.step()
                    optimizer_C_i.zero_grad() 
                    loss_ci_train = loss_ci_train.item()
                
                    # then transfer to F
                    class_logits_i = label_predictor_i(hidden_emb)
                    loss_ci = class_criterion(class_logits_i, label_i.long())
                else:
                    loss_ci_train = 0
                    loss_ci = 0

            
                loss_c_l.append(loss_ci)
                loss_c_train_l.append(loss_ci_train)
            
                loss_fi = loss_i + lamb*loss_ci            
                loss_l.append(loss_fi)
            
        # unpack loss       
        loss_F = torch.sum(torch.stack(loss_l))
        loss_F.backward()
        optimizer_F.step()
        running_F_loss = loss_F.item()
        running_C_st_loss = math.fsum(loss_c_train_l)   
        
        if epoch % 10 == 0:
            print('epoch {:>3d}: train spatial C loss: {:6.4f}, train F loss: {:6.4f},'.format(epoch, running_C_st_loss, running_F_loss))
           
    with torch.no_grad():        
        for i in range(n):
            if feature_method == 'LGCN_mlp':
                hidden_emb, att, h1_f, h1_t, h1_i, h2, ret = feature_extractor(feat_l[i], feat_a_l[i], edge_index_f_l[i], adj_f_l[i], edge_index_t_l[i], adj_t_l[i], edge_index_i_l[i], adj_i_l[i])
        
            if feature_method is None or feature_method == 'GCNConv' or feature_method == 'SGCN' or feature_method == 'sparse':
                hidden_emb, att, h1_f, h1_t, h1_i, h2, ret = feature_extractor(feat_l[i], feat_a_l[i], adj_f_l[i], adj_t_l[i], adj_i_l[i])                                                        
                                                       
            adata_l[i].obsm['embedding'] = hidden_emb.cpu().detach().numpy()
            
            
            acc_st_i = 0
            if tag_l[i] == 'ST_ref':
                # check the classifier accuracy                       
                label_predictor_i = label_predictor_st_l[i]
                preds_i = label_predictor_i(hidden_emb)
                label_i = torch.tensor(adata_l[i].obs['ref'].to_numpy().astype(int))
                #adata_l[i].obsm['pre'] = label_i
            
                pred_cls_i = preds_i.data.max(1)[1]
                acc_st_i += pred_cls_i.eq(label_i.data).cpu().sum()
                print("spatial data classification: Avg Accuracy = {:2%}".format(acc_st_i/len(label_i))) 
                
            adata_l[i].obsm['embedding'] = hidden_emb.cpu().detach().numpy()
            if att is not None:
                adata_l[i].obsm['attention'] = att.cpu().detach().numpy()
            if h1_f is not None:
                adata_l[i].obsm['hidden_feature'] = h1_f.cpu().detach().numpy()
                
            if h1_t is not None:
                adata_l[i].obsm['hidden_spatial'] = h1_t.cpu().detach().numpy()
                
            if h1_i is not None:
                adata_l[i].obsm['hidden_image'] = h1_i.cpu().detach().numpy()
                
            adata_l[i].obsm['reconstruct'] = h2.cpu().detach().numpy()       

    # represent spot
    print('Training classifier...')
    if len(ref_sc_id) > 0 and multiscale:
        # identify clustering on the embedding of each slice
        adata_l = get_mlp(adata_l,tag_l,section_ids, hidden_features, out_features, theta, lr,label_name, class_rep,gamma)
        

        if len(n_clusters_l) > 1:
            y_pred_l = []
            for i in range(n):
                if tag_l[i] != 'single cell':
                    devfeatures = adata_l[i].obsm['deconvolution']
                    y_pred = mclust_init(devfeatures, n_clusters_l[i])# used as cluster constraint for each slice
                    y_pred_new = [y_pred[s]-1 for s in range(len(y_pred))]
                    y_pred_l.append(y_pred_new)
                    adata_l[i].uns['deconvolution_pre'] = adata_l[i].uns['deconvolution'].copy()
        else:
            devfeatures = []
            for i in range(n):
                if tag_l[i] != 'single cell':
                    devfeatures.append(torch.tensor(adata_l[i].obsm['deconvolution']).float())
                    adata_l[i].uns['deconvolution_pre'] = adata_l[i].uns['deconvolution'].copy()
            devfeatures_all = torch.cat(devfeatures).numpy()
            y_pred = mclust_init(devfeatures_all, n_clusters_l[0])
            y_pred_new = [y_pred[s]-1 for s in range(len(y_pred))]


    print('Identifying anchors...')
    anchors_l = identify_anchors(adata_l, section_ids, st_flag, iter_comb, use_rep_anchor, label_name, mnn_neigh)
    keys_l = list(anchors_l.keys())
    
    print('Aligning by anchors...')
    for epoch in range(subepochs, epochs): 
        feature_extractor.train() #
        optimizer_F.zero_grad()
        hidden_emb_l, h1_f_l, h1_t_l, h1_i_l, h2_l, ret_l = [], [] ,[], [], [], []
        loss_l = []
        loss_c_train_l = []
        loss_c_l = []
        for i in range(n):
            if tag_l[i] != 'single cell': 
                if feature_method == 'LGCN_mlp':
                    hidden_emb, _, h1_f, h1_t, h1_i, h2, ret = feature_extractor(feat_l[i], feat_a_l[i], edge_index_f_l[i], adj_f_l[i], edge_index_t_l[i], adj_t_l[i], edge_index_i_l[i], adj_i_l[i])
        
                if feature_method is None or feature_method == 'GCNConv' or feature_method == 'SGCN' or feature_method == 'sparse':
                    hidden_emb, _, h1_f, h1_t, h1_i, h2, ret = feature_extractor(feat_l[i], feat_a_l[i], adj_f_l[i], adj_t_l[i], adj_i_l[i])                                                        
            
                adata_l[i].obsm['embedding'] = hidden_emb.cpu().detach().numpy()
                adata_l[i].obsm['reconstruct'] = h2.cpu().detach().numpy()
                hidden_emb_l.append(hidden_emb)
                h1_f_l.append(h1_f)
                h1_t_l.append(h1_t)
                h1_i_l.append(h1_i)
                h2_l.append(h2)
                ret_l.append(ret)
                if h1_t is not None:
                    ret_CSL = torch.from_numpy(np.concatenate([np.ones([feat_l[i].shape[0], 1]), np.zeros([feat_l[i].shape[0], 1])], axis=1))
                    loss_CSL = nn.BCEWithLogitsLoss()               
                    loss_sl = loss_CSL(ret, ret_CSL)
                else:
                    loss_sl = 0
                loss_feat = F.mse_loss(feat_l[i], h2)
                loss_feat_i = loss_feat
                loss_i1 = loss_feat_i + alpha*loss_sl          
            
                if spatial_coords_l[i] is not None:
                    if regularization_acceleration:
                        nodes = feat_l[i].shape[0]
                        cell_random_subset_1, cell_random_subset_2 = torch.randint(0, nodes , (edge_subset_sz,)), torch.randint(0, nodes, (edge_subset_sz,))
                        z1, z2 = torch.index_select(h1_t_l[i], 0, cell_random_subset_1), torch.index_select(h1_t_l[i], 0, cell_random_subset_2)
                        c1, c2 = torch.index_select(spatial_coords_l[i], 0, cell_random_subset_1), torch.index_select(spatial_coords_l[i], 0, cell_random_subset_2)
                        pdist = torch.nn.PairwiseDistance(p=2)

                        z_dists = pdist(z1, z2)
                        z_dists = z_dists / torch.max(z_dists)
                        sp_dists = pdist(c1, c2)
                        sp_dists = sp_dists / torch.max(sp_dists)
                        n_items = z_dists.size(dim=0)
                        penalty_i = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items)
                    else:
                        z_dists = torch.cdist(h1_t_l[i],h1_t_l[i], p=2)
                        z_dists = torch.div(z_dists, torch.max(z_dists))
                        sp_dists = torch.cdist(spatial_coords_l[i], spatial_coords_l[i], p=2)
                        sp_dists = torch.div(sp_dists, torch.max(sp_dists))
                        n_items = h1_t_l[i].size(dim=0) * h1_t_l[i].size(dim=0)
                        penalty_i = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items)                    
                else:
                    penalty_i = 0
        
                loss_i = loss_i1 + spatial_regularization_strength * penalty_i

                
                if label_predictor_st_l[i] is not None:
                    label_predictor_i = label_predictor_st_l[i]
                    optimizer_C_i = optimizer_C_st_l[i]                            
                    class_criterion = nn.CrossEntropyLoss()
                    if len(n_clusters_l) > 1:
                        label_i = torch.tensor(np.array(y_pred_l[i]).astype(int))
                    else:
                        label_i = torch.tensor(adata_l[i].obs['ref'].to_numpy().astype(int))
                    class_logits_i = label_predictor_i(hidden_emb.detach())
                    loss_ci_train = class_criterion(class_logits_i, label_i.long())
                    loss_ci_train.backward() ####
                    optimizer_C_i.step()
                    optimizer_C_i.zero_grad() 
                    loss_ci_train = loss_ci_train.item()
                    
                    # then transfer to F
                    class_logits_i = label_predictor_i(hidden_emb)
                    loss_ci = class_criterion(class_logits_i, label_i.long())

                    label_predictor_st_l[i] = label_predictor_i
                else:
                    loss_ci_train = 0
                    loss_ci = 0

                
                loss_c_l.append(loss_ci)
                loss_c_train_l.append(loss_ci_train)
                loss_fi = loss_i + lamb*loss_ci
                loss_l.append(loss_fi)
                
        running_C_st_loss = math.fsum(loss_c_train_l)
        # unpack loss       
        loss_F = torch.sum(torch.stack(loss_l))
        
        # classifier based on deconvolution
        if label_predictor_st is not None:
            emb_all = torch.cat(hidden_emb_l)
            class_criterion = nn.CrossEntropyLoss()                
            label_st = torch.tensor(np.array(y_pred_new).astype(int))
            class_logits_st = label_predictor_st(emb_all.detach())
            loss_c_st_train = class_criterion(class_logits_st, label_st.long())
            loss_c_st_train.backward() ####
            optimizer_C_st.step()
            optimizer_C_st.zero_grad() 
            running_C_st_loss = loss_c_st_train.item()
            # then transfer to F
            class_logits_st = label_predictor_st(emb_all)
            loss_c_st = class_criterion(class_logits_st, label_st.long())
            loss_F = loss_F + lamb*loss_c_st
            
        # do alignment， MMD or Discrimintor+ anchors mixture for each pair
        loss_align_D = []
        for t in range(0, len(iter_comb)):
            comb = iter_comb[t]
            i = comb[0]
            j = comb[1] 
            # take out the corresponding embeddings of the pair
            emb_source_t = adata_l[i].obsm['embedding']
            s_id_anchor = anchors_l[keys_l[t]][0]
            t_id_anchor = anchors_l[keys_l[t]][1]
            
            if anchor_size is None or anchor_size >= len(s_id_anchor):
                anchor_len = len(s_id_anchor)
                if anchor_len > 10:
                    # take out the anchors
                    s_id = s_id_anchor
                    t_id = t_id_anchor
                    t_id_o = [t_id[p]- emb_source_t.shape[0] for p in range(len(t_id))]                        
                    loss_mmd_t = mmd_rbf(hidden_emb_l[i][s_id,:], hidden_emb_l[j][t_id_o,:], kernel = 'rbf')                       
            else:
                if anchors_all == True:
                    # divided to different groups
                    group_num = int(len(s_id_anchor)/anchor_size)
                    if group_num > 1:
                        loss_mmd_t_g = []
                        for group in range(group_num-1):
                            # compute distance for each anchor pair. high distance has high priority
                            anchor_index = list(range(group*anchor_size,(group+1)*anchor_size))
                            # take out the anchors
                            s_id = [s_id_anchor[anchor_index[p]] for p in range(anchor_size)]
                            t_id = [t_id_anchor[anchor_index[p]] for p in range(anchor_size)]
                            t_id_o = [t_id[p]- emb_source_t.shape[0] for p in range(len(t_id))]                    
                            # MMD on feature embeddings              
                            loss_mmd_t_gi = mmd_rbf(hidden_emb_l[i][s_id,:], hidden_emb_l[j][t_id_o,:], kernel = 'rbf')
                            loss_mmd_t_g.append(loss_mmd_t_gi)
                        loss_mmd_t = torch.sum(torch.stack(loss_mmd_t_g))/group_num
                    elif len(s_id_anchor) > 0:
                        # take out the anchors
                        anchor_len = len(s_id_anchor)
                        anchor_index = torch.randint(0, len(s_id_anchor), (anchor_len,))
                        s_id = [s_id_anchor[anchor_index[p]] for p in range(anchor_len)]
                        t_id = [t_id_anchor[anchor_index[p]] for p in range(anchor_len)]
                        t_id_o = [t_id[p]- emb_source_t.shape[0] for p in range(len(t_id))]    
                        loss_mmd_t = mmd_rbf(hidden_emb_l[i][s_id,:], hidden_emb_l[j][t_id_o,:], kernel = 'rbf')
                        
                        
                      
                if anchors_all == False:
                    anchor_len = min([anchor_size,len(s_id_anchor)])
                    anchor_index = torch.randint(0, len(s_id_anchor), (anchor_len,))
                    # take out the anchors
                    s_id = [s_id_anchor[anchor_index[p]] for p in range(anchor_len)]
                    t_id = [t_id_anchor[anchor_index[p]] for p in range(anchor_len)]
                    t_id_o = [t_id[p]- emb_source_t.shape[0] for p in range(len(t_id))]    
                    loss_mmd_t = mmd_rbf(hidden_emb_l[i][s_id,:], hidden_emb_l[j][t_id_o,:], kernel = 'rbf')
                                       
            loss_align_D.append(loss_mmd_t)
                              
        loss_F_all = loss_F + beta * torch.sum(torch.stack(loss_align_D))
  
        loss_F_all.backward(retain_graph = True)
        optimizer_F.step()
        running_F_loss = loss_F.item()
        running_D_loss = torch.sum(torch.stack(loss_align_D)).detach()
        
            
        if epoch % 10 == 0:
            print('epoch {:>3d}: total loss:{:6.4f}, train F loss: {:6.4f}, train C loss: {:6.4f}, train D loss: {:6.4f}'.format(epoch, loss_F_all, running_F_loss, running_C_st_loss, running_D_loss))
    flag = 0
    
    with torch.no_grad():
        hidden_emb_l = []
        for i in range(n):
            feat_a_i = permutation(feat_l[i])
            
            if feature_method == 'LGCN_mlp':
                hidden_emb, att, h1_f, h1_t, h1_i, h2, _ = feature_extractor(feat_l[i], feat_a_i, edge_index_f_l[i], adj_f_l[i], edge_index_t_l[i], adj_t_l[i], edge_index_i_l[i], adj_i_l[i])
        
            if feature_method is None or feature_method == 'GCNConv' or feature_method == 'SGCN' or feature_method == 'sparse':
                hidden_emb, att, h1_f, h1_t, h1_i, h2, _ = feature_extractor(feat_l[i], feat_a_i, adj_f_l[i], adj_t_l[i], adj_i_l[i])

            hidden_emb_l.append(hidden_emb)
            adata_l[i].obsm['embedding'] = hidden_emb.cpu().detach().numpy()
            if att is not None:
                adata_l[i].obsm['attention'] = att.cpu().detach().numpy()
            if h1_f is not None:
                adata_l[i].obsm['hidden_feature'] = h1_f.cpu().detach().numpy()
                
            if h1_t is not None:
                adata_l[i].obsm['hidden_spatial'] = h1_t.cpu().detach().numpy()
                
            if h1_i is not None:
                adata_l[i].obsm['hidden_image'] = h1_i.cpu().detach().numpy()
                
            adata_l[i].obsm['reconstruct'] = h2.cpu().detach().numpy()

            
            
            if i == 0 and tag_l[0] == 'ST_ref':
                # check the classifier accuracy                       
                label_predictor_i = label_predictor_st_l[i]
                preds_i = label_predictor_i(torch.tensor(adata_l[i].obsm['embedding'].copy()))
                label_i = torch.tensor(adata_l[i].obs['ref'].to_numpy().astype(int))
                #adata_l[i].obsm['pre'] = label_i
            
                pred_cls_i = preds_i.data.max(1)[1]
                acc_st_i += pred_cls_i.eq(label_i.data).cpu().sum()
                print("spatial data classification: Avg Accuracy = {:2%}".format(acc_st_i/len(label_i)))
                flag = 1

            # predict other ST data, using the reference predictor
            if flag == 1:
                for s in range(n):
                    if tag_l[s] != 'single cell':
                        pre_st_s = label_predictor_i(torch.tensor(adata_l[s].obsm['embedding'].copy()))                           
                        adata_l[s].obs['st_pre'] = pre_st_s.data.max(1)[1].numpy()

            if multiscale and tag_l[0] != 'ST_ref' and tag_l[i] != 'single cell' and len(n_clusters_l) > 1:
                label_predictor_i = label_predictor_st_l[i]
                preds_i = label_predictor_i(torch.tensor(adata_l[i].obsm['embedding'].copy()))
                pred_cls_i = preds_i.data.max(1)[1]
                adata_l[i].obs['pd_cluster'] = pred_cls_i
                #if 'ref' in adata_l[i].obs.keys():
                #    label_i = torch.tensor(adata_l[i].obs['ref'].to_numpy().astype(int))
                #    acc_st_i += pred_cls_i.eq(label_i.data).cpu().sum()
                #    print("spatial data classification: Avg Accuracy = {:2%}".format(acc_st_i/len(label_i)))
        if len(ref_sc_id) > 0 and multiscale and len(n_clusters_l) == 1 and tag_l[0] != 'ST_ref':
            emb_all = torch.cat(hidden_emb_l[0:(n-1)])
            preds_st = label_predictor_st(emb_all)
            pred_cls_st = preds_st.data.max(1)[1]
            adata_concat_st = ad.concat(adata_l[0:(n-1)], label="slice_name", keys=section_ids)
            adata_concat_st.obs['pd_cluster'] = pred_cls_st
            # divided to each slice
            for i in range(n-1):
                adata_i = adata_concat_st[adata_concat_st.obs['slice_name'] == section_ids[i]]
                adata_l[i].obs['pd_cluster'] = adata_i.obs['pd_cluster']
                #if 'ref' in adata_l[i].obs.keys():
                #    label_i = torch.tensor(adata_i.obs['ref'].to_numpy().astype(int))
                #    label_pre = torch.tensor(adata_i.obs['pd_cluster'].to_numpy().astype(int))
                #    acc_st_i = label_pre.eq(label_i.data).cpu().sum()
                #    print("spatial data classification: Avg Accuracy = {:2%}".format(acc_st_i/len(label_i)))
                
    print('Updating classifier...')
    
    if len(ref_sc_id) > 0 and multiscale:
        #print('Identifying anchors...')
        #anchors_l = identify_anchors(adata_l, section_ids, st_flag, iter_comb, use_rep_anchor, label_name, mnn_neigh)

        adata_l = get_mlp(adata_l,tag_l,section_ids,hidden_features, out_features, theta,lr,label_name, class_rep,gamma)   
                    
    return adata_l


