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
    - [2021](#2021)
- [Super Resolution workshop papers](workshops.md)
- [Super Resolution survey](sr_survey.md)

# Awesome-Super-Resolution（in progress）

Collect some super-resolution related papers, data and repositories.

## papers

### DL based approach

Note this table is referenced from [here](https://github.com/LoSealL/VideoSuperResolution/blob/master/README.md#network-list-and-reference-updating)

### 2021
More years papers, plase check Quick navigation

| Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| TrilevelNAS            | [arxiv](https://arxiv.org/pdf/2101.06658.pdf)            | -              | Trilevel Architecture Search Space      |
| SplitSR                | [arxiv](https://arxiv.org/pdf/2101.07996.pdf)            | -              | lightweight,on Mobile Devices      |
| BurstSR                | [arxiv](https://arxiv.org/pdf/2101.10997.pdf)            | -              | multi-frame sr, new BurstSR dataset      |
| USTVSRNet              | [arxiv](https://arxiv.org/pdf/2102.13011.pdf)            | -              | ***VSR***, Unconstrained video super-resolution,general-ized  pixelshuffle  layer.     |
| ClassSR                | [cvpr21](https://arxiv.org/pdf/2103.04039.pdf)           | [code](https://github.com/Xiangtaokong/ClassSR)         | classification，lightweight | 
| FADN                   | [arxiv](https://arxiv.org/pdf/2103.08357.pdf)            | -              |  Efficient SR，DCT，Mask Predictor，dynamic resblocks，frequency mask loss|
| SESR                   | [arxiv](https://arxiv.org/pdf/2103.09404.pdf)            | -              |  Super-Efficient SR, overparameterization|
| Adapted VSR            | [arxiv](https://arxiv.org/pdf/2103.10081.pdf)            | -              |  ***VSR***, Self-Supervised Adaptationn|
| Generic Perceptual Loss| [cvpr21](https://arxiv.org/pdf/2103.10571.pdf)           | -              |  Random VGG w/o pretrianed, work at semantic segmentation, depthestimation  and  instance  segmentation|
| U3D-RDN                | [AAAI21](https://arxiv.org/pdf/2103.11744.pdf)           | -              |  ***VSR***, Dual Subnet, Multi-stage Communicated Up-sampling|
| UltraSR                | [arxiv](https://arxiv.org/pdf/2103.12716.pdf)            | -              |  Implicit Image Function Based, Arbitrary-Scale|
| BSRNet/BSRGAN          | [arxiv](https://arxiv.org/pdf/2103.14006.pdf)            | -              |  randomly shuffled blur, downsampling and noise degradations for degradation model |
| D2C-SR                 | [arxiv](https://arxiv.org/pdf/2103.14373.pdf)            | -              |  RealSR， divergence stage with a triple loss ，convergence stage|
| MDF loss               | [arxiv](https://arxiv.org/pdf/2103.14616.pdf)            | [code](https://github.com/gfxdisp/mdf)              |  Multi-Scale Discriminative Feature loss|
| TLSR                   | [arxiv](https://arxiv.org/pdf/2103.15290.pdf)            | [code](https://github.com/YuanfeiHuang/TLSR)        | Blind SR， Transitive Learning   |
| OVSR                   | [arxiv](https://arxiv.org/pdf/2103.15683.pdf)            | -              | ***VSR***， new framework， precursor net and successor net  |
| Beby-GAN               | [arxiv](https://arxiv.org/pdf/2103.15295.pdf)            | [code](https://github.com/Jia-Research-Lab/Simple-SR)  |relaxing the immutable one-to-one constraint, Best-Buddy Loss  |
| FKP                    | [cvpr21](https://arxiv.org/pdf/2103.15977.pdf)           | [code](https://github.com/JingyunLiang/FKP)            |Blind SR, flow-based kernel prio|
| SRVC                   | [arxiv](https://arxiv.org/pdf/2104.02322.pdf)            | -            |Video Compression， Adaptive Conv|
| CMDSR                  | [arxiv](https://arxiv.org/pdf/2104.03926.pdf)            | -            |Blind SR, Conditional Meta-Network|
| SR3                    | [arxiv](https://arxiv.org/pdf/2104.07636.pdf)            | -            |Repeated Refinement, better than  SOTA GAN|
| BAM                    | [arxiv](https://arxiv.org/pdf/2104.07566.pdf)            | [code](https://github.com/dandingbudanding/BAM_A_lightweight_but_efficient_Balanced_attention_mechanism_for_super_resolution)            |balanced Attention Mechanism |
| KASR                   | [arxiv](https://arxiv.org/pdf/2104.09008.pdf)            | -              |realsr, Kernel Agnostic  |
| DeCoNAS                | [arxiv](https://arxiv.org/pdf/2104.09048.pdf)            | -              |Densely Constructed Search Space  |
| A2N                    | [arxiv](https://arxiv.org/pdf/2104.09497.pdf)            | [code](https://github.com/haoyuc/A2N)    |Attention in Attention， lightweight  |
| TMNet                  | [arxiv](https://arxiv.org/pdf/2104.10642.pdf)            | [code](https://github.com/CS-GangXu/TMNet)    | ***VSR***， interpolate frames,  Temporal Modulation Block  |
| TSAN                   | [arxiv](https://arxiv.org/pdf/2104.10488.pdf)            | [code](https://github.com/Jee-King/TSAN)    | SISR， multi-context attentiveblock  |
| BasicVSR++             | [arxiv](https://arxiv.org/pdf/2104.13371.pdf)            | [code](https://github.com/open-mmlab/mmediting)    | ***VSR***， 3 champions and 1 runner-up in NTIRE 2021   |

#### New NTIRE 2021 started! go [here](https://data.vision.ee.ethz.ch/cvl/ntire21/) 



