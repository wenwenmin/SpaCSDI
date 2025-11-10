import pandas as pd
import scanpy as sc
import anndata
import numpy as np
import sys
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import SPACEL
from SPACEL import Splane

adata = sc.read('D:\pythonplaces\deconv-mulST\Baseline\Spoint\Result\Dataset4\st.h5ad')
# adata.obsm['spatial'] = adata.obs[['new_x', 'new_y']].values
section_ids = [1,2,3,4,5,6,7]
adata_list = []

for idx, sid in enumerate(section_ids):
    sub_adata = adata[adata.obs['batch_name_new'] == sid]
    adata_list.append(sub_adata)

import importlib
import base_model

importlib.reload(base_model)

splane_model = Splane.init_model(adata_list, n_clusters=4,use_gpu=False,n_neighbors=8,k=1,gnn_dropout=0.5)
splane_model.train(d_l=0.5)
splane_model.identify_spatial_domain()

sc.concat(adata_list).write(f'D:\pythonplaces\SPACEL-main\SPACEL\Splane\Result\dataset4/dataset4_st1.h5ad')