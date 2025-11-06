
import scanpy as sc
from utils import *
import os
import numba
import anndata as ad
import os
import scvi

def data_prepare_and_merge(scdata, stdatas, outfilr,array_x,array_y):
    scdata = scdata.copy()
    scdata.var_names_make_unique()
    mask = ~scdata.var_names.duplicated()
    scdata = scdata[:, mask].copy()
    sc.pp.normalize_total(scdata, target_sum=1e4)
    sc.pp.log1p(scdata)
    sc.pp.filter_cells(scdata, min_genes=1)
    # 获取共同基因
    common_genes = scdata.var_names
    # 对每个其他数据集进行处理
    for idx, data in enumerate(stdatas):

        # 重新索引数据，确保数据只包含选定的基因
        stdatas[idx] = data.copy()
        stdatas[idx].var_names_make_unique()

        # 4️⃣ 预处理数据
        sc.pp.normalize_total(stdatas[idx], target_sum=1e4)
        sc.pp.log1p(stdatas[idx])
        mask = ~stdatas[idx].var_names.duplicated()
        stdatas[idx] = stdatas[idx][:, mask].copy()


        common_genes = common_genes.intersection(stdatas[idx].var_names)

    # 确保 scdata 按照共同基因进行过滤
    scdata = scdata[:, common_genes].copy()
    # 保存 scRNA 过滤后的数据
    scdata.write_h5ad(os.path.join(outfilr, 'scRNA_commmon_genes.h5ad'))

    # 创建一个空列表，用于存储带有 batch 标记的 ST 数据
    st_datasets_with_batch = []

    # 处理每个 ST 数据集，添加 batch_name 并进行过滤
    for i, st_data in enumerate(stdatas):
        st_data.var_names_make_unique()
        # 过滤基因列表中的数据
        st_data_filtered = st_data[:, common_genes].copy()
        # 设置 obs['batch_name'] 列，标记为第 i 个批次
        st_data_filtered.obs['batch_name_new'] = i+1
        st_data_filtered.obs['Istrue']=1
        st_data_filtered.obs_names = st_data_filtered.obs_names.astype(str) + f'_{i+1}'
        sc.pp.filter_cells(st_data_filtered, min_genes=1)
        # st_data_filtered=dist_adj_real_one(st_data_filtered,array_x,array_y)
        # 将处理好的数据集加入列表
        st_datasets_with_batch.append(st_data_filtered)
        # 保存过滤后的 ST 数据
        st_data_filtered.write_h5ad(os.path.join(outfilr, f'Spatial_ST{i + 1}_commmon_genes.h5ad'))

    stdatas1 = ad.concat(st_datasets_with_batch, axis=0)
    stdatas1.write_h5ad(os.path.join(outfilr, 'Merger_ST.h5ad'))

    return scdata,stdatas1


# def get_scvi_latent(
#         st_ad,
#         sm_ad,
#         n_layers,
#         n_latent,
#         gene_likelihood='zinb',
#         dispersion='gene-batch',
#         max_epochs=1000,
#         early_stopping=True,
#         batch_size=4096,
# ):
#
#
#     st_ad.obs["batch"] = 'real'
#     sm_ad.obs["batch"] = 'simulated'
#
#     adata = sc.concat([st_ad,sm_ad])
#     adata.layers["counts"] = adata.X.copy()
#
#     scvi.model.SCVI.setup_anndata(
#         adata,
#         layer="counts",
#         batch_key="batch"
#     )
#
#     vae = scvi.model.SCVI(adata, n_layers=n_layers, n_latent=n_latent, gene_likelihood=gene_likelihood,
#                           dispersion=dispersion)
#     vae.train(max_epochs=max_epochs, early_stopping=early_stopping, batch_size=batch_size)
#     adata.obsm["X_scVI"] = vae.get_latent_representation()
#     # print("现在的adata.shape", adata.shape)
#     st_scvi_ad = anndata.AnnData(adata[adata.obs['batch'] != 'simulated'].obsm["X_scVI"])
#     sm_scvi_ad = anndata.AnnData(adata[adata.obs['batch'] == 'simulated'].obsm["X_scVI"])
#
#     st_scvi_ad.obs = st_ad.obs
#     st_scvi_ad.obsm = st_ad.obsm
#
#     sm_scvi_ad.obs = sm_ad.obs
#     sm_scvi_ad.obsm = sm_ad.obsm
#
#     sm_scvi_ad = check_data_type(sm_scvi_ad)
#     st_scvi_ad = check_data_type(st_scvi_ad)
#
#     sm_data = sm_scvi_ad.X
#     sm_labels = sm_scvi_ad.obsm['label'].values
#     st_data = st_scvi_ad.X
#     return sm_scvi_ad, st_scvi_ad
def get_scvi_latent(
        st_ad,
        sm_ad,
        n_layers,
        n_latent,
        gene_likelihood='zinb',
        dispersion='gene-batch',
        max_epochs=1000,
        early_stopping=True,
        batch_size=4096,
):
    st_ad.obs["batch"] = 'real'
    sm_ad.obs["batch"] = 'simulated'

    adata = sc.concat([st_ad, sm_ad])
    adata.layers["counts"] = adata.X.copy()

    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key="batch"
    )

    vae = scvi.model.SCVI(
        adata,
        n_layers=n_layers,
        n_latent=n_latent,
        gene_likelihood=gene_likelihood,
        dispersion=dispersion
    )
    vae.train(max_epochs=max_epochs, early_stopping=early_stopping, batch_size=batch_size)
    adata.obsm["X_scVI"] = vae.get_latent_representation()

    st_scvi_ad = anndata.AnnData(adata[adata.obs['batch'] != 'simulated'].obsm["X_scVI"])
    sm_scvi_ad = anndata.AnnData(adata[adata.obs['batch'] == 'simulated'].obsm["X_scVI"])

    st_scvi_ad.obs = st_ad.obs.copy()
    st_scvi_ad.obsm = st_ad.obsm.copy()

    sm_scvi_ad.obs = sm_ad.obs.copy()
    sm_scvi_ad.obsm = sm_ad.obsm.copy()

    sm_scvi_ad = check_data_type(sm_scvi_ad)
    st_scvi_ad = check_data_type(st_scvi_ad)

    return sm_scvi_ad, st_scvi_ad

def data_prepare(
    sc_ad,
    st_ad,
    celltype_key,
    h5ad_file_path,
    data_file_path,
    n_layers,
    n_latent,
    deg_method:str='wilcoxon',
    n_top_markers:int=600,
    n_top_hvg:int=None,
    log2fc_min=0.5,
    pval_cutoff=0.01,
    pct_diff=None,
    pct_min=0.1,
    sm_size:int=20000,
    cell_counts=None,
    clusters_mean=None,
    #cells_mean=5,
    #cells_min=2,
    #cells_max=10,
    cells_mean=10,
    cells_min=1,
    cells_max=20,
    cell_sample_counts=None,
    cluster_sample_counts=None,
    ncell_sample_list=None,
    cluster_sample_list=None,
    n_threads=4,
        #seed=42,82
    seed=42,

):
    print('Setting global seed:', seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    simulation_st.numba_set_seed(seed)
    numba.set_num_threads(n_threads)

    sc_ad = normalized_data(sc_ad, target_sum=1e4)
    st_ad = normalized_data(st_ad, target_sum=1e4)
    print("原始st的形状：",st_ad.shape)
    sc_ad, st_ad = filter_model_genes(
        sc_ad,
        st_ad,
        celltype_key=celltype_key,
        deg_method=deg_method,
        n_top_markers=n_top_markers,
        n_top_hvg=n_top_hvg,
        log2fc_min=log2fc_min,
        pval_cutoff=pval_cutoff,
        pct_diff=pct_diff,
        pct_min=pct_min
    )
    print("筛选后st的形状：", st_ad.shape)
    """产生模拟数据并进行下采样"""
    sm_ad =generate_sm_stdata(sc_ad,num_sample=sm_size,celltype_key=celltype_key,n_threads=n_threads,cell_counts=cell_counts,clusters_mean=clusters_mean,cells_mean=cells_mean,
                             cells_min=cells_min,cells_max=cells_max,cell_sample_counts=cell_sample_counts,cluster_sample_counts=cluster_sample_counts,
                             ncell_sample_list=ncell_sample_list,cluster_sample_list=cluster_sample_list)

    downsample_sm_spot_counts(sm_ad,st_ad,n_threads=n_threads)
    os.makedirs(h5ad_file_path, exist_ok=True)
    sc_adcopy = sc_ad
    if 'cluster_p_balance' in sc_adcopy.uns:
        del sc_adcopy.uns['cluster_p_balance']
        del sc_adcopy.uns['cluster_p_sqrt']
        del sc_adcopy.uns['cluster_p_unbalance']
    """将scdata和stdata进行预处理后存储，以便其他Baselines使用"""
    sc_adcopy.write_h5ad(data_file_path + '\Scdata_filter.h5ad')
    sm_ad.write_h5ad(data_file_path + '\Sm_STdata_filter.h5ad')
    st_ad.write_h5ad(data_file_path + '\Real_STdata_filter.h5ad')
    """对scdata和stdata利用scvi工具进行降维"""
    sm_scvi_ad, st_scvi_ad = get_scvi_latent(st_ad,sm_ad,n_layers,n_latent)
    sm_scvi_ad.write_h5ad(h5ad_file_path + '\sm_scvi_ad.h5ad')
    st_scvi_ad.write_h5ad(h5ad_file_path + '\st_scvi_ad.h5ad')
