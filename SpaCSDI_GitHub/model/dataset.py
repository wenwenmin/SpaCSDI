from data_prepare import *
import anndata as ad
import os
scfile_path = "D:\pythonplaces\deconv-mulST\Dataset_origin\CRC\CRC_10X_sparse.h5ad"
scdata = ad.read_h5ad(scfile_path)
"""E:\多切片数据集\直肠癌\st1.h5ad"""
base_path = r"E:\多切片数据集\直肠癌"
crc_names = [f"st{i}.h5ad" for i in range(1, 8)]  # CRC1–CRC14
file_names = crc_names
file_paths = [os.path.join(base_path, fname) for fname in file_names]

# 2. 加载所有数据
stdatas = [ad.read_h5ad(fpath) for fpath in file_paths]

"""---------------1.数据准备----------------------"""
outfilr='D:\pythonplaces\deconv-mulST\Result\Dataset4'
if not os.path.exists(outfilr):
    os.makedirs(outfilr)
scdata,stdatas1=data_prepare_and_merge(scdata, stdatas, outfilr,array_x='pxl_row_in_fullres',array_y='pxl_col_in_fullres')

outfile = 'D:\pythonplaces\deconv-mulST\Result\Dataset4'
datafile = 'D:\pythonplaces\deconv-mulST\Result\Dataset4'
if not os.path.exists(outfile):
    os.makedirs(outfile)
cell_key='Cell_subtype'
data_prepare(sc_ad=scdata, st_ad=stdatas1, celltype_key=cell_key,
                     h5ad_file_path=outfile, data_file_path=datafile,
                     n_layers=2, n_latent=2048,sm_size=12000,n_top_markers=500)