"""读取数据"""
import copy
import multiprocessing as mp
from Baseline.Spoint.model import init_model
import anndata as ad
import torch
import os
import Baseline.Spoint.data_utils
import tracemalloc
import time
def main(cell_key):
        start_time = time.time()
        tracemalloc.start()
        # sc_file = 'D:\pythonplaces\deconv-mulST\Result\Dataset3\paient_H\scRNA_commmon_genes.h5ad'
        # st_file = 'D:\pythonplaces\deconv-mulST\Result\Dataset3\paient_H\Merger_ST.h5ad'
        sc_file = 'D:\pythonplaces\deconv-mulST\Result\Dataset2/all-weeks\scRNA_commmon_genes.h5ad'
        st_file = 'D:\pythonplaces\deconv-mulST\Result\Dataset2/all-weeks\Merger_ST.h5ad'
        st_ad1 = ad.read_h5ad(st_file)
        sc_data1 = ad.read_h5ad(sc_file)
        st_ad = copy.deepcopy(st_ad1)
        sc_data = copy.deepcopy(sc_data1)
        output_path = 'D:\pythonplaces\deconv-mulST\Baseline\Spoint\Result\Dataset5'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        model = init_model(sc_ad=sc_data,
                                   st_ad=st_ad,
                                   celltype_key=cell_key,
                                   n_top_markers=400,
                                   n_top_hvg=2500)
        # model.train
        model.model_train(sm_lr=0.01,
                    st_lr=0.01)

        pre = model.deconv_spatial()

        pre.to_csv(output_path + "/proportion1.csv")

        st_ad = model.st_ad
        st_ad.write(output_path + '/st.h5ad')

        celltype = st_ad.obs.loc[:, pre.columns].max(0)
        st_ad.uns['celltypes']=celltype
        st_ad.write(output_path + '/st2.h5ad')

        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time  # 计算总时间
        print(f"spoint Total time taken: {total_time:.2f} seconds")  # 打印总时间
        current, peak = tracemalloc.get_traced_memory()
        print(f"[Peak Memory] Current: {current / 1024 / 1024:.2f} MB; Peak: {peak / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    main('celltype')
