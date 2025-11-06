from collections import defaultdict

from model.Moduel import *
from model.utils import _mask_by_genes
from model.utils import *
from model.find_anchor import *
from scipy.sparse import csr_matrix

import scipy.sparse as sp
def extract_subgraph(adj_csr: csr_matrix, nodes_idx: np.ndarray):
    """
    从大邻接矩阵 adj_csr 中提取以 nodes_idx 为节点的子邻接矩阵
    返回PyG格式的 edge_index 和 edge_weight
    """
    # nodes_idx 要是排序过的，方便处理；如果没排序，这里先排序
    nodes_idx = np.sort(nodes_idx)

    # 抽取子矩阵（行列都选nodes_idx）
    sub_adj = adj_csr[nodes_idx, :][:, nodes_idx]  # shape: (len(nodes_idx), len(nodes_idx))

    coo = sub_adj.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    edge_weight = torch.tensor(coo.data, dtype=torch.float32)

    return edge_index, edge_weight
"""
每次循环：
1.更新判别器
2.更新编码、解码器
3.更新编码、分类器
"""
class SpaCSDInet(nn.Module):
    def __init__(self,
                 seed, device,pattern,exp_bia,dis_bia,w_exp,w_dis,
                 adatas, alldata, alldata_scvi,
                 hidden_num_exp, output_num_exp,
                 hidden_num_dis, output_num_dis,
                 hidden_num_dec,
                 latent_num_batch, batch_num,
                 hidden_pre, cell_type_num,
                 epochs, num_cluster, outdir):
        super(SpaCSDInet, self).__init__()
        self.batch_size = 2048
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)
        self.device = device
        self.adatas = adatas
        self.pattern=pattern
        self.alldata = alldata
        self.alldata_scvi = alldata_scvi
        self.hidden_num_exp = hidden_num_exp
        self.output_num_exp = output_num_exp
        self.hidden_num_dis = hidden_num_dis
        self.output_num_dis = output_num_dis
        self.hidden_num_dec = hidden_num_dec
        self.latent_num_batch = latent_num_batch
        self.batch_num = batch_num
        self.hidden_pre = hidden_pre
        self.cell_type_num = cell_type_num
        self.epochs = epochs
        self.num_cluster = num_cluster
        self.outdir = outdir
        self.exp_bia=exp_bia
        self.dis_bia=dis_bia
        self.w_dis=w_dis
        self.w_exp=w_exp
        if self.pattern=='combined':
           self.interlay_num = output_num_exp + output_num_dis
        elif self.pattern=='over' :
            if  self.output_num_exp==self.output_num_dis:
                self.interlay_num = output_num_exp
            else:
                print("维度不同无法相加！！！")
        else:
            print("没有该模式！！！")

        self.anchor_pair = None

    def start(self):
        self.section_ids = np.array(self.alldata.obs['batch_name_new'].unique())
        if sp.issparse(self.alldata.X):
            # 稀疏矩阵先转 dense 再转 torch
            self.X_true = torch.FloatTensor(self.alldata.X.toarray().copy()).to(self.device)
        else:
            # 已经是 ndarray
            self.X_true = torch.FloatTensor(self.alldata.X.copy()).to(self.device)
        # self.X_true = torch.FloatTensor(self.alldata.X.toarray().copy()).to(self.device)
        self.X_scvi = torch.FloatTensor(self.alldata_scvi.X.copy()).to(self.device)
        self.input_dim = self.X_scvi.shape[-1]
        self.expgraph = ExpGraphConv(self.input_dim, self.hidden_num_exp, self.output_num_exp)
        self.disgraph = DisGraphConv(self.input_dim, self.hidden_num_dis, self.output_num_dis)
        self.discriminator = Discriminator_batch(self.interlay_num, self.latent_num_batch, self.batch_num)
        self.decoder = Decoder(self.interlay_num, self.hidden_num_dec, self.X_true.shape[-1])
        self.predictor = Predictor(self.interlay_num, self.hidden_pre, self.cell_type_num)

        self.dist_adj = self.alldata.obsm['dist_adj']
        self.exp_adj = self.alldata.obsm['exp_adj']
        batch_name_new = self.alldata.obs['batch_name_new']
        if batch_name_new.dtype == 'object' or batch_name_new.dtype.name == 'category':
            batch_name_new = batch_name_new.astype('category').cat.codes  # 转换为整数编码
        self.batch_true = torch.tensor(batch_name_new.values, dtype=torch.long).to(self.device)

        self.stage1_optimizer = torch.optim.Adam(
            params=list(self.discriminator.parameters()),
            lr=0.002,
            weight_decay=0.002,
        )

        self.stage2_optimizer = torch.optim.Adam(
            params=list(self.expgraph.parameters()) +
                   list(self.disgraph.parameters()) +
                   list(self.decoder.parameters()),
            lr=0.001,
            weight_decay=0.0002,
        )

        self.stage3_optimizer = torch.optim.Adam(
            params=list(self.predictor.parameters())+
                   list(self.expgraph.parameters()) +
                   list(self.disgraph.parameters()),
            lr=0.002,
            weight_decay=0.002,
        )
        self.expgraph.to(self.device)
        self.disgraph.to(self.device)
        self.discriminator.to(self.device)
        self.decoder.to(self.device)
        self.predictor.to(self.device)

    def train_stage1(self):
        self.disgraph.eval()
        self.expgraph.eval()
        self.discriminator.train()
        h_dis = self.disgraph(self.X_scvi, self.dist_adj)
        h_exp = self.expgraph(self.X_scvi, self.exp_adj)
        if self.pattern=='combined':
           h_combined = torch.cat((self.dis_bia*h_dis, self.exp_bia*h_exp), dim=1)
        elif self.pattern=='over' :
            h_combined = self.dis_bia*h_dis+self.exp_bia*h_exp
        dis_loss = self.discriminator(h_combined, self.batch_true)
        all_loss = dis_loss
        self.stage1_optimizer.zero_grad()
        all_loss.backward()
        self.stage1_optimizer.step()
        all_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        return all_loss



    def train_stage2(self,batch_size=20000):
        self.disgraph.train()
        self.expgraph.train()
        self.decoder.train()
        self.discriminator.eval()
        h_dis = self.disgraph(self.X_scvi, self.dist_adj)
        h_exp = self.expgraph(self.X_scvi, self.exp_adj)
        if self.pattern=='combined':
           h_combined = torch.cat((self.dis_bia*h_dis, self.exp_bia*h_exp), dim=1)
        elif self.pattern=='over' :
            h_combined = self.dis_bia*h_dis+self.exp_bia*h_exp
        dis_loss = self.discriminator(h_combined, self.batch_true)
        # dis_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        # 每个切片的spots数量
        self.slice_lens = [adata.shape[0] for adata in self.adatas]
        self.slice_offsets = np.concatenate([[0], np.cumsum(self.slice_lens)])
        with torch.amp.autocast(device_type=self.device):
            X_recon_loss_all = []
            for i in range(len(self.adatas)):
                adata_i = self.adatas[i]
                if (adata_i.obs['Istrue'] == 1).all():
                    start = int(self.slice_offsets[i])
                    end = int(self.slice_offsets[i + 1])
                    # 1) 取该切片的 h 子特征
                    h_sub = h_combined[start:end, :]

                    # 2) dist 子邻接（已是CSR）
                    dist_csr = adata_i.obsm['dist_adj']
                    if not sp.isspmatrix_csr(dist_csr):
                        dist_csr = dist_csr.tocsr(copy=False)

                    exp_csr = self.exp_adj[start:end, start:end]
                    if not sp.isspmatrix_csr(exp_csr):
                        exp_csr = exp_csr.tocsr(copy=False)

                    A_csr = (self.w_dis * dist_csr) + (self.w_exp * exp_csr)
                    A_csr.eliminate_zeros()  # 清理零值条目
                    if sp.issparse(self.adatas[i].X):
                        _, mask = _mask_by_genes(torch.FloatTensor(self.adatas[i].X.toarray().copy()), mask_rate=0.1)
                        X_recon_loss_sub = self.decoder(h_sub, A_csr,
                                                        torch.FloatTensor(self.adatas[i].X.toarray().copy()).to(
                                                            self.device), mask=mask)
                    else:
                        _, mask = _mask_by_genes(torch.FloatTensor(self.adatas[i].X.copy()), mask_rate=0.1)
                        X_recon_loss_sub = self.decoder(h_sub, A_csr,
                                                        torch.FloatTensor(self.adatas[i].X.copy()).to(
                                                            self.device), mask=mask)

                    X_recon_loss_all.append(X_recon_loss_sub)
        if self.anchor_pair is not None:
            anchor, positive, negative = self.anchor_pair
            triple_loss = self.triplet_loss(h_combined, anchor, positive, negative)

        else:
            triple_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        X_recon_loss = torch.stack(X_recon_loss_all).sum()
        all_loss = 0.7*X_recon_loss - 0.2*dis_loss + triple_loss
        # all_loss =  X_recon_loss -  dis_loss + triple_loss
        self.stage2_optimizer.zero_grad()
        all_loss.backward()
        self.stage2_optimizer.step()
        return X_recon_loss, dis_loss, triple_loss

    def train_stage3(self,batch_size=20000):
        self.disgraph.train()
        self.expgraph.train()
        self.predictor.train()
        h_dis = self.disgraph(self.X_scvi, self.dist_adj)
        h_exp = self.expgraph(self.X_scvi, self.exp_adj)
        if self.pattern=='combined':
           h_combined_all = torch.cat((self.dis_bia*h_dis, self.exp_bia*h_exp), dim=1)
        elif self.pattern=='over' :
            h_combined_all = self.dis_bia*h_dis+self.exp_bia*h_exp

        for i in range(len(self.adatas)):
            if (self.adatas[i].obs['Istrue'] == 0).all():
                length = self.adatas[i].shape[0]
                self.len_sm=length
                h_combined = h_combined_all[-length:, :]
                pre_label, pre_loss = self.predictor(h_combined, self.adatas[i].obsm['label'])
                self.cell_pro = self.adatas[i].obsm['label']
                loss = pre_loss
                self.stage3_optimizer.zero_grad()
                loss.backward()
                self.stage3_optimizer.step()

                return pre_loss

    def train(self, t_step=50, early_stop_epochs=100, convergence=0.0004):
        self.start()
        metric_logger = defaultdict(list)
        early_stop_count = 0
        best_loss = np.inf
        """dataset1,2:t_step=30,dataset3:50"""
        for epoch in range(self.epochs):
            if (epoch+1) % t_step == 0:
                # 寻找锚点
                self.anchor_pair = self.get_anchor(Isreal=True)
            for _ in range(1):
                loss_stage1 = self.train_stage1()
            for _ in range(1):
                rec_loss, dis_loss, triple_loss = self.train_stage2()
            for _ in range(1):
                pre_loss = self.train_stage3()

            if epoch >= 0:
                metric_logger['pre_loss'].append(pre_loss.cpu().detach().numpy())

                metric_logger['rec_loss'].append(rec_loss.cpu().detach().numpy())

                metric_logger['dis_loss'].append(dis_loss.cpu().detach().numpy())

                metric_logger['loss_stage1'].append(loss_stage1.cpu().detach().numpy())

                metric_logger['triple_loss'].append(triple_loss.cpu().detach().numpy())


            if (epoch) % 20 == 0:
                print(
                    '============= Epoch {:02d}/{:02d} in stage ============='.format(epoch + 1, self.epochs))
                print(
                    "loss_stage1=%f, dis_loss=%f, rec_loss=%f,triple_loss=%f,pre_loss=%f" % (
                        loss_stage1, dis_loss, rec_loss, triple_loss, pre_loss ))
            # total_loss = 0.7*rec_loss  - 0.2*dis_loss + triple_loss + pre_loss + loss_stage1
            total_loss = rec_loss - dis_loss + triple_loss + pre_loss + loss_stage1
            current_loss = total_loss.cpu().detach().numpy()
            if best_loss - current_loss > convergence:
                if best_loss > current_loss:
                    best_loss = current_loss
                early_stop_count = 0
            else:
                early_stop_count += 1
            if early_stop_count > early_stop_epochs:
                print('Stop trainning because of loss convergence')
                break


        SaveLossPlot(self.outdir, metric_logger,
                     loss_type=['loss_stage1', 'dis_loss', 'rec_loss', 'triple_loss', 'pre_loss',
                                ], output_prex='Loss_metric_plot_stage')


    def predicted_ST(self, save_dir=None,batch_size=20000):
        h_dis_st = self.disgraph.inference(self.X_scvi, self.dist_adj)
        h_exp_st = self.expgraph.inference(self.X_scvi, self.exp_adj)
        if self.pattern == 'combined':
            h_combined = torch.cat((self.dis_bia * h_dis_st, self.exp_bia * h_exp_st), dim=1)
        elif self.pattern == 'over':
            h_combined = self.dis_bia * h_dis_st + self.exp_bia * h_exp_st

        trim_len = h_combined.shape[0] - self.len_sm

        pre_labels = self.predictor.predicted(h_combined)
        pre_labels_df = pd.DataFrame(
            data=pre_labels.detach().cpu().numpy()[:trim_len],
            columns=self.cell_pro.columns,
            index=self.alldata.obs_names[:trim_len]
        )
        self.alldata.uns['deconv_result'] = pre_labels_df

        self.alldata.obsm['latent_no_batch'] = h_combined.detach().cpu().numpy()

        start = 0
        for i in range(len(self.adatas)):
            adata_i = self.adatas[i]
            if (adata_i.obs['Istrue'] == 1).all():
                start = int(self.slice_offsets[i])
                end = int(self.slice_offsets[i + 1])
                # 1) 取该切片的 h 子特征
                h_sub = h_combined[start:end, :]

                # 2) dist 子邻接（已是CSR）
                dist_csr = adata_i.obsm['dist_adj']
                if not sp.isspmatrix_csr(dist_csr):
                    dist_csr = dist_csr.tocsr(copy=False)

                exp_csr = self.exp_adj[start:end, start:end]
                if not sp.isspmatrix_csr(exp_csr):
                    exp_csr = exp_csr.tocsr(copy=False)

                A_csr = (self.w_dis * dist_csr) + (self.w_exp * exp_csr)
                A_csr.eliminate_zeros()  # 清理零值条目
                X_recon_sub = self.decoder.inference(h_sub, A_csr)

                recon_df = pd.DataFrame(
                    data=X_recon_sub.detach().cpu().numpy(),
                    columns=self.adatas[i].var_names,
                    index=self.adatas[i].obs_names
                )

                key = f'x_recon_{i + 1}'
                if save_dir is not None:
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f'{key}.csv')
                        recon_df.to_csv(save_path)
                        # print(f"[Info] Saved {key} to {save_path}")
                else:
                        print(f"[Warning] Could not save {key}; no save_dir provided.")

        return self.alldata

    def triplet_loss(self, emb, anchor, positive, negative, margin=1.0):
        anchor_arr = emb[anchor]
        positive_arr = emb[positive]
        negative_arr = emb[negative]
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        return tri_output



    def clustering(self, adata, num_cluster, used_obsm, key_added_pred, random_seed=2024):
        adata = Kmeans_cluster(adata, num_cluster=num_cluster, used_obsm=used_obsm, key_added_pred=key_added_pred,
                               random_seed=random_seed)
        return adata

    def get_anchor(self, Isreal, verbose=0, random_seed=2024,batch_size=20000):
        self.disgraph.eval()
        if Isreal:

            h_dis = self.disgraph(self.X_scvi, self.alldata.obsm['dist_adj'])
            h_exp = self.expgraph(self.X_scvi, self.alldata.obsm['exp_adj'])
            if self.pattern == 'combined':
                latent_emb_dis = torch.cat((self.dis_bia * h_dis, self.exp_bia * h_exp), dim=1)
            elif self.pattern == 'over':
                latent_emb_dis = self.dis_bia * h_dis + self.exp_bia * h_exp
            # latent_emb_dis = torch.cat((h_dis, h_exp), dim=1)
            self.alldata.obsm['latent'] = latent_emb_dis.data.cpu().numpy()
            key_pred = 'domain'
            self.adata = self.clustering(self.alldata, num_cluster=self.num_cluster, used_obsm='latent',
                                         key_added_pred=key_pred, random_seed=random_seed)

            gnn_dict = create_dictionary_gnn(self.adata, use_rep='latent', use_label=key_pred,
                                             batch_name='batch_name_new',
                                             k=60, verbose=verbose)
            anchor_ind = []
            positive_ind = []
            negative_ind = []
            for batch_pair in gnn_dict.keys():
                batchname_list = self.adata.obs['batch_name_new'][gnn_dict[batch_pair].keys()]

                cellname_by_batch_dict = dict()
                for batch_id in range(len(self.section_ids)):
                    cellname_by_batch_dict[self.section_ids[batch_id]] = self.adata.obs_names[
                        self.adata.obs['batch_name_new'] == self.section_ids[batch_id]].values

                anchor_list = []
                positive_list = []
                negative_list = []
                for anchor in gnn_dict[batch_pair].keys():
                    anchor_list.append(anchor)
                    positive_spot = gnn_dict[batch_pair][anchor][0]
                    positive_list.append(positive_spot)
                    section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                    negative_list.append(
                        cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

                batch_as_dict = dict(zip(list(self.adata.obs_names), range(0, self.adata.shape[0])))
                anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))
            anchor_pair = (anchor_ind, positive_ind, negative_ind)
            return anchor_pair




"""
每次循环：
1.更新判别器
2.更新编码、解码器
3.更新编码、分类器
4.针对Domain识别
"""
class SpaCSDInet_domain(nn.Module):
    def __init__(self,
                 seed, device,pattern,exp_bia,dis_bia,w_exp,w_dis,
                 adatas, alldata, alldata_scvi,
                 hidden_num_exp, output_num_exp,
                 hidden_num_dis, output_num_dis,
                 hidden_num_dec,
                 latent_num_batch, batch_num,
                 hidden_pre, domain_num,
                 epochs, num_cluster, outdir):
        super(SpaCSDInet_domain, self).__init__()
        self.batch_size = 2048
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)
        self.device = device
        self.adatas = adatas
        self.pattern=pattern
        self.alldata = alldata
        self.alldata_scvi = alldata_scvi
        self.hidden_num_exp = hidden_num_exp
        self.output_num_exp = output_num_exp
        self.hidden_num_dis = hidden_num_dis
        self.output_num_dis = output_num_dis
        self.hidden_num_dec = hidden_num_dec
        self.latent_num_batch = latent_num_batch
        self.batch_num = batch_num
        self.hidden_pre = hidden_pre
        self.domain_num = domain_num
        self.epochs = epochs
        self.num_cluster = num_cluster
        self.outdir = outdir
        self.exp_bia=exp_bia
        self.dis_bia=dis_bia
        self.w_dis=w_dis
        self.w_exp=w_exp
        if self.pattern=='combined':
           self.interlay_num = output_num_exp + output_num_dis
        elif self.pattern=='over' :
            if  self.output_num_exp==self.output_num_dis:
                self.interlay_num = output_num_exp
            else:
                print("维度不同无法相加！！！")
        else:
            print("没有该模式！！！")

        self.anchor_pair = None

    def start(self):
        self.section_ids = np.array(self.alldata.obs['batch_name_new'].unique())
        if sp.issparse(self.alldata.X):
            # 稀疏矩阵先转 dense 再转 torch
            self.X_true = torch.FloatTensor(self.alldata.X.toarray().copy()).to(self.device)
        else:
            # 已经是 ndarray
            self.X_true = torch.FloatTensor(self.alldata.X.copy()).to(self.device)
        self.X_scvi = torch.FloatTensor(self.alldata_scvi.X.copy()).to(self.device)
        self.input_dim = self.X_scvi.shape[-1]
        self.expgraph = ExpGraphConv(self.input_dim, self.hidden_num_exp, self.output_num_exp)
        self.disgraph = DisGraphConv(self.input_dim, self.hidden_num_dis, self.output_num_dis)
        self.discriminator = Discriminator_batch(self.interlay_num, self.latent_num_batch, self.batch_num)
        self.decoder = Decoder(self.interlay_num, self.hidden_num_dec, self.X_true.shape[-1])
        self.predictor = Predictor(self.interlay_num, self.hidden_pre, self.domain_num)

        self.dist_adj = self.alldata.obsm['dist_adj']
        self.exp_adj = self.alldata.obsm['exp_adj']
        batch_name_new = self.alldata.obs['batch_name_new']
        if batch_name_new.dtype == 'object' or batch_name_new.dtype.name == 'category':
            batch_name_new = batch_name_new.astype('category').cat.codes  # 转换为整数编码
        self.batch_true = torch.tensor(batch_name_new.values, dtype=torch.long).to(self.device)

        self.stage1_optimizer = torch.optim.Adam(
            params=list(self.discriminator.parameters()),
            lr=0.002,
            weight_decay=0.002,
        )

        self.stage2_optimizer = torch.optim.Adam(
            params=list(self.expgraph.parameters()) +
                   list(self.disgraph.parameters()) +
                   list(self.decoder.parameters()),
            lr=0.001,
            weight_decay=0.0002,
        )

        self.stage3_optimizer = torch.optim.Adam(
            params=list(self.predictor.parameters())+
                   list(self.expgraph.parameters()) +
                   list(self.disgraph.parameters()),
            lr=0.002,
            weight_decay=0.002,
        )
        self.expgraph.to(self.device)
        self.disgraph.to(self.device)
        self.discriminator.to(self.device)
        self.decoder.to(self.device)
        self.predictor.to(self.device)

    def train_stage1(self):
        self.disgraph.eval()
        self.expgraph.eval()
        self.discriminator.train()
        h_dis = self.disgraph(self.X_scvi, self.dist_adj)
        h_exp = self.expgraph(self.X_scvi, self.exp_adj)
        if self.pattern=='combined':
           h_combined = torch.cat((self.dis_bia*h_dis, self.exp_bia*h_exp), dim=1)
        elif self.pattern=='over' :
            h_combined = self.dis_bia*h_dis+self.exp_bia*h_exp
        dis_loss = self.discriminator(h_combined, self.batch_true)
        all_loss = dis_loss
        self.stage1_optimizer.zero_grad()
        all_loss.backward()
        self.stage1_optimizer.step()
        all_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        return all_loss



    def train_stage2(self,batch_size=20000):
        self.disgraph.train()
        self.expgraph.train()
        self.decoder.train()
        self.discriminator.eval()
        h_dis = self.disgraph(self.X_scvi, self.dist_adj)
        h_exp = self.expgraph(self.X_scvi, self.exp_adj)
        if self.pattern=='combined':
           h_combined = torch.cat((self.dis_bia*h_dis, self.exp_bia*h_exp), dim=1)
        elif self.pattern=='over' :
            h_combined = self.dis_bia*h_dis+self.exp_bia*h_exp
        dis_loss = self.discriminator(h_combined, self.batch_true)
        # 每个切片的spots数量
        self.slice_lens = [adata.shape[0] for adata in self.adatas]
        self.slice_offsets = np.concatenate([[0], np.cumsum(self.slice_lens)])
        with torch.amp.autocast(device_type=self.device):
            X_recon_loss_all = []
            for i in range(len(self.adatas)):
                adata_i = self.adatas[i]
                if (adata_i.obs['Istrue'] == 1).all():
                    start = int(self.slice_offsets[i])
                    end = int(self.slice_offsets[i + 1])
                    # 1) 取该切片的 h 子特征
                    h_sub = h_combined[start:end, :]

                    # 2) dist 子邻接（已是CSR）
                    dist_csr = adata_i.obsm['dist_adj']
                    if not sp.isspmatrix_csr(dist_csr):
                        dist_csr = dist_csr.tocsr(copy=False)

                    exp_csr = self.exp_adj[start:end, start:end]
                    if not sp.isspmatrix_csr(exp_csr):
                        exp_csr = exp_csr.tocsr(copy=False)

                    A_csr = (self.w_dis * dist_csr) + (self.w_exp * exp_csr)
                    A_csr.eliminate_zeros()  # 清理零值条目
                    if sp.issparse(self.adatas[i].X):
                        _, mask = _mask_by_genes(torch.FloatTensor(self.adatas[i].X.toarray().copy()), mask_rate=0.1)
                        X_recon_loss_sub = self.decoder(h_sub, A_csr,
                                                        torch.FloatTensor(self.adatas[i].X.toarray().copy()).to(
                                                            self.device), mask=mask)
                    else:
                        _, mask = _mask_by_genes(torch.FloatTensor(self.adatas[i].X.copy()), mask_rate=0.1)
                        X_recon_loss_sub = self.decoder(h_sub, A_csr,
                                                        torch.FloatTensor(self.adatas[i].X.copy()).to(
                                                            self.device), mask=mask)


                    X_recon_loss_all.append(X_recon_loss_sub)
        if self.anchor_pair is not None:
            anchor, positive, negative = self.anchor_pair
            triple_loss = self.triplet_loss(h_combined, anchor, positive, negative)
        #
        else:
            triple_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        X_recon_loss = torch.stack(X_recon_loss_all).sum()
        all_loss = 0.7*X_recon_loss - 0.2*dis_loss + triple_loss
        # all_loss =  X_recon_loss -  dis_loss + triple_loss
        self.stage2_optimizer.zero_grad()
        all_loss.backward()
        self.stage2_optimizer.step()
        return X_recon_loss, dis_loss, triple_loss


    def train_stage3(self,batch_size=20000):
        self.disgraph.train()
        self.expgraph.train()
        self.predictor.train()
        h_dis = self.disgraph(self.X_scvi, self.dist_adj)
        h_exp = self.expgraph(self.X_scvi, self.exp_adj)
        if self.pattern=='combined':
           h_combined_all = torch.cat((self.dis_bia*h_dis, self.exp_bia*h_exp), dim=1)
        elif self.pattern=='over' :
            h_combined_all = self.dis_bia*h_dis+self.exp_bia*h_exp
        start = 0
        pre_loss_all = 0
        for i in range(len(self.adatas)):
            length = self.adatas[i].shape[0]
            if 'IsLable' in self.adatas[i].obs:
                if (self.adatas[i].obs['IsLable'] == 1).all():
                    self.len_sm = length
                    self.len_start_sm = start
                    h_combined = h_combined_all[start:start + length, :]
                    pre_label, pre_loss = self.predictor(h_combined, self.adatas[i].obsm['label'])
                    self.cell_pro_column = self.adatas[i].uns['label_categories']
                    pre_loss_all = pre_loss_all + pre_loss
                    start += length
            else:
                start = start + length
        self.stage3_optimizer.zero_grad()
        pre_loss_all.backward()
        self.stage3_optimizer.step()
        return pre_loss_all




    def train(self, t_step=50, early_stop_epochs=100, convergence=0.0004):
        self.start()
        metric_logger = defaultdict(list)
        early_stop_count = 0
        best_loss = np.inf
        """dataset1,2:t_step=30,dataset3:50"""
        for epoch in range(self.epochs):
            if (epoch+1) % t_step == 0:
                # 寻找锚点
                self.anchor_pair = self.get_anchor(Isreal=True)
            for _ in range(1):
                loss_stage1 = self.train_stage1()
            for _ in range(1):
                rec_loss, dis_loss, triple_loss = self.train_stage2()
            for _ in range(1):
                pre_loss = self.train_stage3()

            if epoch >= 0:
                metric_logger['pre_loss'].append(pre_loss.cpu().detach().numpy())

                metric_logger['rec_loss'].append(rec_loss.cpu().detach().numpy())

                metric_logger['dis_loss'].append(dis_loss.cpu().detach().numpy())

                metric_logger['loss_stage1'].append(loss_stage1.cpu().detach().numpy())

                metric_logger['triple_loss'].append(triple_loss.cpu().detach().numpy())


            if (epoch) % 20 == 0:
                print(
                    '============= Epoch {:02d}/{:02d} in stage ============='.format(epoch + 1, self.epochs))
                print(
                    "loss_stage1=%f, dis_loss=%f, rec_loss=%f,triple_loss=%f,pre_loss=%f" % (
                        loss_stage1, dis_loss, rec_loss, triple_loss, pre_loss ))
            # total_loss = 0.7*rec_loss  - 0.2*dis_loss + triple_loss + pre_loss + loss_stage1
            total_loss = rec_loss - dis_loss + triple_loss + pre_loss + loss_stage1
            current_loss = total_loss.cpu().detach().numpy()
            if best_loss - current_loss > convergence:
                if best_loss > current_loss:
                    best_loss = current_loss
                early_stop_count = 0
            else:
                early_stop_count += 1
            if early_stop_count > early_stop_epochs:
                print('Stop trainning because of loss convergence')
                break


        SaveLossPlot(self.outdir, metric_logger,
                     loss_type=['loss_stage1', 'dis_loss', 'rec_loss', 'triple_loss', 'pre_loss',
                                ], output_prex='Loss_metric_plot_stage')


    def predicted_ST(self, save_dir=None,batch_size=20000):
        h_dis_st = self.disgraph.inference(self.X_scvi, self.dist_adj)
        h_exp_st = self.expgraph.inference(self.X_scvi, self.exp_adj)
        if self.pattern == 'combined':
            h_combined = torch.cat((self.dis_bia * h_dis_st, self.exp_bia * h_exp_st), dim=1)
        elif self.pattern == 'over':
            h_combined = self.dis_bia * h_dis_st + self.exp_bia * h_exp_st

        trim_len = h_combined.shape[0] - self.len_sm

        pre_labels = self.predictor.predicted(h_combined)
        pre_labels_df = pd.DataFrame(
            data=pre_labels.detach().cpu().numpy()[:trim_len],
            columns=self.cell_pro_column,
            index=self.alldata.obs_names[:trim_len]
        )
        self.alldata.uns['deconv_result'] = pre_labels_df

        self.alldata.obsm['latent_no_batch'] = h_combined.detach().cpu().numpy()

        start = 0
        for i in range(len(self.adatas)):
            adata_i = self.adatas[i]
            if (adata_i.obs['Istrue'] == 1).all():
                start = int(self.slice_offsets[i])
                end = int(self.slice_offsets[i + 1])
                # 1) 取该切片的 h 子特征
                h_sub = h_combined[start:end, :]

                # 2) dist 子邻接（已是CSR）
                dist_csr = adata_i.obsm['dist_adj']
                if not sp.isspmatrix_csr(dist_csr):
                    dist_csr = dist_csr.tocsr(copy=False)

                exp_csr = self.exp_adj[start:end, start:end]
                if not sp.isspmatrix_csr(exp_csr):
                    exp_csr = exp_csr.tocsr(copy=False)

                A_csr = (self.w_dis * dist_csr) + (self.w_exp * exp_csr)
                A_csr.eliminate_zeros()  # 清理零值条目
                X_recon_sub = self.decoder.inference(h_sub, A_csr)

                recon_df = pd.DataFrame(
                    data=X_recon_sub.detach().cpu().numpy(),
                    columns=self.adatas[i].var_names,
                    index=self.adatas[i].obs_names
                )

                key = f'x_recon_{i + 1}'
                if save_dir is not None:
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f'{key}.csv')
                        recon_df.to_csv(save_path)
                        # print(f"[Info] Saved {key} to {save_path}")
                else:
                        print(f"[Warning] Could not save {key}; no save_dir provided.")

        return self.alldata

    def triplet_loss(self, emb, anchor, positive, negative, margin=1.0):
        anchor_arr = emb[anchor]
        positive_arr = emb[positive]
        negative_arr = emb[negative]
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        return tri_output



    def clustering(self, adata, num_cluster, used_obsm, key_added_pred, random_seed=2024):
        adata = Kmeans_cluster(adata, num_cluster=num_cluster, used_obsm=used_obsm, key_added_pred=key_added_pred,
                               random_seed=random_seed)
        return adata

    def get_anchor(self, Isreal, verbose=0, random_seed=2024,batch_size=20000):
        self.disgraph.eval()
        if Isreal:

            h_dis = self.disgraph(self.X_scvi, self.alldata.obsm['dist_adj'])
            h_exp = self.expgraph(self.X_scvi, self.alldata.obsm['exp_adj'])
            if self.pattern == 'combined':
                latent_emb_dis = torch.cat((self.dis_bia * h_dis, self.exp_bia * h_exp), dim=1)
            elif self.pattern == 'over':
                latent_emb_dis = self.dis_bia * h_dis + self.exp_bia * h_exp
            # latent_emb_dis = torch.cat((h_dis, h_exp), dim=1)
            self.alldata.obsm['latent'] = latent_emb_dis.data.cpu().numpy()
            key_pred = 'domain'
            self.adata = self.clustering(self.alldata, num_cluster=self.num_cluster, used_obsm='latent',
                                         key_added_pred=key_pred, random_seed=random_seed)

            gnn_dict = create_dictionary_gnn(self.adata, use_rep='latent', use_label=key_pred,
                                             batch_name='batch_name_new',
                                             k=60, verbose=verbose)
            anchor_ind = []
            positive_ind = []
            negative_ind = []
            for batch_pair in gnn_dict.keys():
                batchname_list = self.adata.obs['batch_name_new'][gnn_dict[batch_pair].keys()]

                cellname_by_batch_dict = dict()
                for batch_id in range(len(self.section_ids)):
                    cellname_by_batch_dict[self.section_ids[batch_id]] = self.adata.obs_names[
                        self.adata.obs['batch_name_new'] == self.section_ids[batch_id]].values

                anchor_list = []
                positive_list = []
                negative_list = []
                for anchor in gnn_dict[batch_pair].keys():
                    anchor_list.append(anchor)
                    positive_spot = gnn_dict[batch_pair][anchor][0]
                    positive_list.append(positive_spot)
                    section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                    negative_list.append(
                        cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

                batch_as_dict = dict(zip(list(self.adata.obs_names), range(0, self.adata.shape[0])))
                anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))
            anchor_pair = (anchor_ind, positive_ind, negative_ind)
            return anchor_pair




