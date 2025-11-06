import  anndata as ad

from model.SpaCSDI_net import *
from  model.find_anchor import *
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)
import tracemalloc
import time
import gc
def main():
        """反卷积"""
        print("-----dataset:deconvlution-----------")
        start_time = time.time()
        tracemalloc.start()
        files = [
            'D:\pythonplaces\deconv-mulST\Result\Dataset1\All-silce\st1.h5ad',
            'D:\pythonplaces\deconv-mulST\Result\Dataset1\All-silce\st2.h5ad',
            'D:\pythonplaces\deconv-mulST\Result\Dataset1\All-silce\st3.h5ad',
            'D:\pythonplaces\deconv-mulST\Result\Dataset1\All-silce\st4.h5ad',
            'D:\pythonplaces\deconv-mulST\Result\Dataset1\All-silce\Sm_STdata_filter.h5ad'
        ]
        datas = [ad.read_h5ad(f) for f in files]

        alldata = ad.read_h5ad('D:\pythonplaces\deconv-mulST\Result\Dataset1\All-silce/all_data.h5ad')
        alldata_scvi = ad.read_h5ad('D:\pythonplaces\deconv-mulST\Result\Dataset1\All-silce/alldatas_scvi_ad.h5ad')

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        outdir = r'D:\pythonplaces\SpaCSDI\Result\dataset1\over'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        modle = SpaCSDInet(
            seed=40, device=device, pattern='over', exp_bia=1, dis_bia=1, w_exp=1, w_dis=1,
            adatas=datas, alldata=alldata, alldata_scvi=alldata_scvi,
            hidden_num_exp=1024, output_num_exp=256,
            hidden_num_dis=1024, output_num_dis=256,
            hidden_num_dec=512,
            latent_num_batch=512, batch_num=5,
            hidden_pre=256, cell_type_num=33,
            epochs=600, num_cluster=7, outdir=outdir
        )
        modle.train()
        result = modle.predicted_ST(outdir)
        result_path = os.path.join(outdir, "pre_labels.h5ad")
        result.write_h5ad(result_path)
        model_path = os.path.join(outdir, "model.pth")
        torch.save(modle.state_dict(), model_path)
        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time  # 计算总时间
        print(f"stereoscope Total time taken: {total_time:.2f} seconds")  # 打印总时间
        current, peak = tracemalloc.get_traced_memory()
        print(f"[Peak Memory] Current: {current / 1024 / 1024:.2f} MB; Peak: {peak / 1024 / 1024:.2f} MB")
        del modle, datas, alldata, alldata_scvi, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("-----dataset:domain-----------")
        start_time = time.time()
        tracemalloc.start()
        files = [
            'D:\pythonplaces\deconv-mulST\Result\Dataset1\All-silce\st1.h5ad',
            'D:\pythonplaces\deconv-mulST\Result\Dataset1\All-silce\st2_lable.h5ad',
            'D:\pythonplaces\deconv-mulST\Result\Dataset1\All-silce\st3.h5ad',
            'D:\pythonplaces\deconv-mulST\Result\Dataset1\All-silce\st4.h5ad',
        ]
        datas = [ad.read_h5ad(f) for f in files]

        alldata = ad.read_h5ad('D:\pythonplaces\deconv-mulST\Result\Dataset1\All-silce\Domain/all_data.h5ad')
        alldata_scvi = ad.read_h5ad('D:\pythonplaces\deconv-mulST\Result\Dataset1\All-silce\st_scvi_ad.h5ad')

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(device)
        outdir = r'D:\pythonplaces\SpaCSDI\Result\dataset1\domain_over'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        modle = SpaCSDInet_domain(
            seed=40, device=device, pattern='over', exp_bia=1, dis_bia=1, w_exp=1, w_dis=1,
            adatas=datas, alldata=alldata, alldata_scvi=alldata_scvi,
            hidden_num_exp=1024, output_num_exp=256,
            hidden_num_dis=1024, output_num_dis=256,
            hidden_num_dec=512,
            latent_num_batch=512, batch_num=5,
            hidden_pre=256, domain_num=7,
            epochs=600, num_cluster=7, outdir=outdir
        )

        modle.train()
        result = modle.predicted_ST()
        result_path = os.path.join(outdir, "pre_domain.h5ad")
        result.write_h5ad(result_path)
        model_path = os.path.join(outdir, "model_domain.pth")
        torch.save(modle.state_dict(), model_path)
        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time  # 计算总时间
        print(f"stereoscope Total time taken: {total_time:.2f} seconds")  # 打印总时间
        current, peak = tracemalloc.get_traced_memory()
        print(f"[Peak Memory] Current: {current / 1024 / 1024:.2f} MB; Peak: {peak / 1024 / 1024:.2f} MB")
        del modle, datas, alldata, alldata_scvi, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()




main()


