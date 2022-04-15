# Quick navigation

- [repositories](awesome_paper_list_and_repos.md)
- [Datasets](dataset.md)
- [papers](#papers)
  - [Non-DL based approach](non_dl_papers.md)
  - [DL based approach](#DL-based-approach)
    - [2014-2016](2014-2016_papers.md)
    - [2017](2017_papers.md)
    - [2018](2018_papers.md)
    - [2019](2019_papers.md)
    - [2020](2020_papers.md)
	- [2021](2021_papers.md)
    - [2022](#2022)
- [Super Resolution workshop papers](workshops.md)
- [Super Resolution survey](sr_survey.md)

# Awesome-Super-Resolution（in progress）

Collect some super-resolution related papers, data and repositories.

## papers

### DL based approach

Note this table is referenced from [here](https://github.com/LoSealL/VideoSuperResolution/blob/master/README.md#network-list-and-reference-updating)

### 2022
More years papers, plase check Quick navigation

| Title                  | Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|Detail-Preserving Transformer for Light Field Image Super-Resolution | DPT | [AAAI2022](https://arxiv.org/pdf/2201.00346.pdf) | [code](https://github.com/BITszwang/DPT) |Light Field, Detail-Preserving Transformer |
|What Hinders Perceptual Quality of PSNR-oriented Methods? | DECLoss | [arxiv](https://arxiv.org/pdf/2201.01034.pdf) | - |Detail Enhanced Contrastive Loss |
|SCSNet: An Efficient Paradigm for Learning Simultaneously Image Colorization and Super-Resolution | SCSNet | [arxiv](https://arxiv.org/pdf/2201.04364.pdf) | - |Image Colorization and Super-Resolution |
|Coarse-to-Fine Embedded PatchMatch and Multi-Scale Dynamic Aggregation for Reference-based Super-Resolution | AMSA | [arxiv](https://arxiv.org/pdf/2201.04358.pdf) | - |RefSR |
|Efficient Non-Local Contrastive Attention for Image Super-Resolution | ENLCA | [arxiv](https://arxiv.org/pdf/2201.03794.pdf) | - |SISR |
|Revisiting L1 Loss in Super-Resolution: A Probabilistic View and Beyond | - | [arxiv](https://arxiv.org/pdf/2201.10084.pdf) | - |SISR, posterior Gaussian distribution, replace L1 loss |
|Scale-arbitrary Invertible Image Downscaling | AIDN | [arxiv](https://arxiv.org/pdf/2201.12576.pdf) | - |***Image Rescaling***, Conditional Resampling Module |
|VRT: A Video Restoration Transformer | VRT | [arxiv](https://arxiv.org/pdf/2201.12288.pdf) | [code](https://github.com/JingyunLiang/VRT) |***VideoSR***, Video Restoration Transformer, temporal mutual self attention and parallel warping |
|Fast Online Video Super-Resolution with Deformable Attention Pyramid | DAP | [arxiv](https://arxiv.org/pdf/2202.01731.pdf) | - |***VideoSR***, fast, deformable attention pyramid |
|Revisiting RCAN: Improved Training for Image Super-Resolution | RCAN-it | [arxiv](https://arxiv.org/pdf/2201.11279.pdf) | [code](https://github.com/zudi-lin/rcan-it) |SISR, train tricks |
|Towards Bidirectional Arbitrary Image Rescaling: Joint Optimization and Cycle Idempotence  | BAIRNet | [arxiv](https://arxiv.org/pdf/2203.00911.pdf) | - |Image Rescaling,  be robust in cycle idempotence test|
|Disentangling Light Fields for Super-Resolution and Disparity Estimation  | DistgSSR | [TPAMI 2022](https://arxiv.org/pdf/2202.10603.pdf) | [code](https://github.com/YingqianWang/DistgSSR) |Light Field|
|Fast Neural Architecture Search for Lightweight Dense Prediction Networks  | LDP-Sup | [arxiv](https://arxiv.org/pdf/2203.01994.pdf) | - |SISR, NAS, Lightweights|
|Learning the Degradation Distribution for Blind Image Super-Resolution  | PDM-SR | [CVPR2022](https://arxiv.org/pdf/2203.04962.pdf) | [code](https://github.com/greatlog/UnpairedSR) |blind SR, probabilistic degradation model, unpaired sr|
|Reference-based Video Super-Resolution Using Multi-Camera Video Triplets | RefVSR | [CVPR2022](https://arxiv.org/abs/2203.14537) | [code](https://github.com/codeslake/RefVSR) |Reference-based VSR, Multi-Camera VSR, [RealMCVSR Dataset](https://junyonglee.me/datasets/RealMCVSR), Bidirectional recurrent framework|
|Deep Constrained Least Squares for Blind Image Super-Resolution | DCLS | [CVPR2022](https://arxiv.org/pdf/2202.07508.pdf) | [code](https://github.com/megvii-research/DCLS-SR) | Blind SR, a dynamic deep linear kernel, Deep Constrained Least Squares |
|Blind Image Super Resolution with Semantic-Aware Quantized Texture Prior |  QuanTexSR | [arxiv](https://arxiv.org/pdf/2202.13142.pdf) | [code](https://github.com/chaofengc/QuanTexSR) | Blind SR, Quantized Texture Prior, Semantic-Guided QTP Pretraining |
|Unfolded Deep Kernel Estimation for Blind Image Super-resolution | UDKE | [arxiv](https://arxiv.org/pdf/2203.05568.pdf) | [code](https://github.com/natezhenghy/UDKE) | Blind SR, unfolded deep kernel estimation |
|Efficient Long-Range Attention Network for Image Super-resolution | ELAN | [arxiv](https://arxiv.org/pdf/2203.05568.pdf) | [code](https://github.com/xindongzhang/ELAN) | SISR SOTA, efficient long-range attention block, group-wise multi-scale self-attention, better results against the transformer-based SR  |
|STDAN: Deformable Attention Network for Space-Time Video Super-Resolution | STDAN | [arxiv](https://arxiv.org/pdf/2203.06841.pdf) | - | ***VideoSR***, long-short term feature interpolation, spatial-temporal deformable feature aggregation  |
|Rich CNN-Transformer Feature Aggregation Networks for Super-Resolution | ACT | [arxiv](https://arxiv.org/pdf/2203.07682.pdf) | - | SISR SOTA, ViT+CNN  |
|Hybrid Pixel-Unshuffled Network for Lightweight Image Super-Resolution | HPUN | [arxiv](https://arxiv.org/pdf/2203.08921.pdf) | - | Lightweight SISR SOTA, Down-sample, Pixel-unshuffle |
|A Text Attention Network for Spatial Deformation Robust Scene Text Image Super-resolution | TATT | [arxiv](https://arxiv.org/pdf/2203.09388.pdf) | [code](https://github.com/mjq11302010044/TATT) | Scene Text SR, CNN and Transformer, text structure consistency loss |
|ARM: Any-Time Super-Resolution Method | ARM | [arxiv](https://arxiv.org/pdf/2203.10812.pdf) | [code](https://github.com/chenbong/ARM-Net) | SISR, Edge-to-PSNR lookup,tradeoff between computation overhead and performance |
|RSTT: Real-time Spatial Temporal Transformer for Space-Time Video Super-Resolution| RSTT | [CVPR2022](https://arxiv.org/pdf/2203.14186.pdf) | [code](https://github.com/llmpass/RSTT) | ***VideoSR***, spatialtemporal transformer  |
|Efficient and Degradation-Adaptive Network for Real-World Image Super-Resolution| DASR | [arxiv](https://arxiv.org/pdf/2203.14216.pdf) | [code](https://github.com/csjliang/DASR) | RealSR, degradation-adaptive  |
|Look Back and Forth: Video Super-Resolution with Explicit Temporal Difference Modeling| ETDM | [CVPR2022](https://arxiv.org/pdf/2204.07114.pdf) | - | ***VideoSR***, A novel back-and-forth refinement strategy, A new framework to explore the temporal diff LR and HR |
