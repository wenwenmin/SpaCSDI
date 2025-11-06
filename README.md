# SpaCSDI

We introduce SpaCSDI (Cross-slice Deconvolution and Integration), a unified end-to-end framework that simultaneously performs cell-type deconvolution, cross-slice alignment, spatial domain identification, and expression denoising within a shared latent representation space. SpaCSDI employs a dual-graph encoder that couples local spatial topology with global transcriptional similarity and integrates self-supervised, contrastive, and adversarial learning to enhance cross-slice consistency while preserving biological interpretability, whereas supervised signals further guide accurate inference of cell-type composition and spatial domains.
![(Variational)](fig1.png)


## System environment
To run `SpaCSDI`, you need to install [PyTorch](https://pytorch.org) with GPU support first. The environment supporting SpaCSDI and baseline models is specified in the `requirements.txt` file.

## Datasets
The publicly available  datasets were used in this study. You can download them from 10.5281/zenodo.17539983



## Run SpaCSDI models
After configuring the environment, download the datasets from the data repository and place them in the datasets/ directory. Then, use the provided notebooks (data_prepare.ipynb) and scripts (dataset.py) to generate the simulated datasets and intermediate files. Finally, update the file paths in main_code.py as needed, and run the script to start the experiment.

## Citing

SpaCSDI: Cross-slice Deconvolution and Integration in Multi-sample Spatial Transcriptomics

## Contact
If you have any questions, please contact huanglin212@aliyun.com and minwenwen@ynu.edu.cn
