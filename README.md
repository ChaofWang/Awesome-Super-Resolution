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
|Lightweight Monocular Depth with a Novel Neural Architecture Search Method  | LDP-Sup | [arxiv](https://arxiv.org/pdf/2203.01994.pdf) | - |SISR, NAS, Lightweights|
|Learning the Degradation Distribution for Blind Image Super-Resolution  | PDM-SR | [CVPR2022](https://arxiv.org/pdf/2203.04962.pdf) | [code](https://github.com/greatlog/UnpairedSR) |blind SR, probabilistic degradation model, unpaired sr|
|Reference-based Video Super-Resolution Using Multi-Camera Video Triplets | RefVSR | [CVPR2022](https://arxiv.org/abs/2203.14537) | [code](https://github.com/codeslake/RefVSR) |Reference-based VSR, Multi-Camera VSR, [RealMCVSR Dataset](https://junyonglee.me/datasets/RealMCVSR), Bbdirectional recurrent framework|
