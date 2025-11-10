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

adata = sc.read('D:\pythonplaces\deconv-mulST\Baseline\Spoint\Result\Dataset1\st.h5ad')

section_ids = [151673, 151674, 151675, 151676]
adata_list = []

for idx, sid in enumerate(section_ids):
    sub_adata = adata[adata.obs['batch_name'] == sid]
    adata_list.append(sub_adata)

import importlib
import base_model

importlib.reload(base_model)

splane_model = Splane.init_model(adata_list, n_clusters=7,use_gpu=False,n_neighbors=8,k=1,gnn_dropout=0.5)
splane_model.train(d_l=0.5)
splane_model.identify_spatial_domain()

sc.concat(adata_list).write(f'D:\pythonplaces\SPACEL-main\SPACEL\Splane\Result\DLPFC/DLPFC_st.h5ad')