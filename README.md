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

| Title                  | Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|Trilevel Neural Architecture Search for Efficient Single Image Super-Resolution        | TrilevelNAS            | [arxiv](https://arxiv.org/pdf/2101.06658.pdf)            | -              | Trilevel Architecture Search Space      |
|SplitSR: An End-to-End Approach to Super-Resolution on Mobile Devices                  | SplitSR                | [arxiv](https://arxiv.org/pdf/2101.07996.pdf)            | -              | lightweight,on Mobile Devices      |
|Deep Burst Super-Resolution                                                            | BurstSR                | [arxiv](https://arxiv.org/pdf/2101.10997.pdf)            | -              | multi-frame sr, new BurstSR dataset      |
|Learning for Unconstrained Space-Time Video Super-Resolution                           | USTVSRNet              | [arxiv](https://arxiv.org/pdf/2102.13011.pdf)            | -              | ***VSR***, Unconstrained video super-resolution,general-ized  pixelshuffle  layer.     |
|ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic| ClassSR            | [cvpr21](https://arxiv.org/pdf/2103.04039.pdf)           | [code](https://github.com/Xiangtaokong/ClassSR)         | classification，lightweight | 
|Learning Frequency-aware Dynamic Network for Efficient Super-Resolution                | FADN                   | [arxiv](https://arxiv.org/pdf/2103.08357.pdf)            | -              |  Efficient SR，DCT，Mask Predictor，dynamic resblocks，frequency mask loss|
|Collapsible Linear Blocks for Super-Efficient Super Resolution                         | SESR                   | [arxiv](https://arxiv.org/pdf/2103.09404.pdf)            | -              |  Super-Efficient SR, overparameterization|
|Self-Supervised Adaptation for Video Super-Resolution                                  | Adapted VSR            | [arxiv](https://arxiv.org/pdf/2103.10081.pdf)            | -              |  ***VSR***, Self-Supervised Adaptationn|
|Generic Perceptual Loss for Modeling Structured Output Dependencies                    | Generic Perceptual Loss| [cvpr21](https://arxiv.org/pdf/2103.10571.pdf)           | -              |  Random VGG w/o pretrianed, work at semantic segmentation, depthestimation  and  instance  segmentation|
|Large Motion Video Super-Resolution with Dual Subnet and Multi-Stage Communicated Upsampling| U3D-RDN           | [AAAI21](https://arxiv.org/pdf/2103.11744.pdf)           | -              |  ***VSR***, Dual Subnet, Multi-stage Communicated Up-sampling|
|UltraSR: Spatial Encoding is a Missing Key for Implicit Image Function-based Arbitrary-Scale Super-Resolution| UltraSR         | [arxiv](https://arxiv.org/pdf/2103.12716.pdf)            | -              |  Implicit Image Function Based, Arbitrary-Scale|
|Designing a Practical Degradation Model for Deep Blind Image Super-Resolution          | BSRNet/BSRGAN          | [arxiv](https://arxiv.org/pdf/2103.14006.pdf)            | -              |  randomly shuffled blur, downsampling and noise degradations for degradation model |
|D2C-SR: A Divergence to Convergence Approach for Image Super-Resolution                | D2C-SR                 | [arxiv](https://arxiv.org/pdf/2103.14373.pdf)            | -              |  RealSR， divergence stage with a triple loss ，convergence stage|
|Training a Better Loss Function for Image Restoration                                  | MDF loss               | [arxiv](https://arxiv.org/pdf/2103.14616.pdf)            | [code](https://github.com/gfxdisp/mdf)              |  Multi-Scale Discriminative Feature loss|
|Transitive Learning: Exploring the Transitivity of Degradations for Blind Super-Resolution| TLSR                | [arxiv](https://arxiv.org/pdf/2103.15290.pdf)            | [code](https://github.com/YuanfeiHuang/TLSR)        | Blind SR， Transitive Learning   |
|Omniscient Video Super-Resolution                                                      | OVSR                   | [arxiv](https://arxiv.org/pdf/2103.15683.pdf)            | -              | ***VSR***， new framework， precursor net and successor net  |
|Best-Buddy GANs for Highly Detailed Image Super-Resolution                             | Beby-GAN               | [arxiv](https://arxiv.org/pdf/2103.15295.pdf)            | [code](https://github.com/Jia-Research-Lab/Simple-SR)  |relaxing the immutable one-to-one constraint, Best-Buddy Loss  |
|Flow-based Kernel Prior with Application to Blind Super-Resolution                     | FKP                    | [cvpr21](https://arxiv.org/pdf/2103.15977.pdf)           | [code](https://github.com/JingyunLiang/FKP)            |Blind SR, flow-based kernel prio|
|Efficient Video Compression via Content-Adaptive Super-Resolution                      | SRVC                   | [arxiv](https://arxiv.org/pdf/2104.02322.pdf)            | -            |Video Compression， Adaptive Conv|
|Conditional Meta-Network for Blind Super-Resolution with Multiple Degradations         | CMDSR                  | [arxiv](https://arxiv.org/pdf/2104.03926.pdf)            | -            |Blind SR, Conditional Meta-Network|
|Image Super-Resolution via Iterative Refinement                                        | SR3                    | [arxiv](https://arxiv.org/pdf/2104.07636.pdf)            | -            |Repeated Refinement, better than  SOTA GAN|
|BAM: A Lightweight and Efficient Balanced Attention Mechanism for Single Image Super Resolution| BAM            | [arxiv](https://arxiv.org/pdf/2104.07566.pdf)            | [code](https://github.com/dandingbudanding/BAM_A_lightweight_but_efficient_Balanced_attention_mechanism_for_super_resolution)            |balanced Attention Mechanism |
|Kernel Agnostic Real-world Image Super-resolution                                      | KASR                   | [arxiv](https://arxiv.org/pdf/2104.09008.pdf)            | -              |realsr, Kernel Agnostic |
|Neural Architecture Search for Image Super-Resolution Using Densely Constructed Search Space: DeCoNAS| DeCoNAS  | [arxiv](https://arxiv.org/pdf/2104.09048.pdf)            | -              |Densely Constructed Search Space  |
|Attention in Attention Network for Image Super-Resolution                              | A2N                    | [arxiv](https://arxiv.org/pdf/2104.09497.pdf)            | [code](https://github.com/haoyuc/A2N)    |Attention in Attention， lightweight  |
|Temporal Modulation Network for Controllable Space-Time Video Super-Resolution         | TMNet                  | [arxiv](https://arxiv.org/pdf/2104.10642.pdf)            | [code](https://github.com/CS-GangXu/TMNet)    | ***VSR***， interpolate frames,  Temporal Modulation Block  |
|A Two-Stage Attentive Network for Single Image Super-Resolution                        | TSAN                   | [arxiv](https://arxiv.org/pdf/2104.10488.pdf)            | [code](https://github.com/Jee-King/TSAN)    | SISR， multi-context attentiveblock  |
|BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment   | BasicVSR++             | [arxiv](https://arxiv.org/pdf/2104.13371.pdf)            | [code](https://github.com/open-mmlab/mmediting)    | ***VSR***， 3 champions and 1 runner-up in NTIRE 2021   |
|SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models              | SRDiff                 | [arxiv](https://arxiv.org/pdf/2104.14951.pdf)            |    | Diffusion Probabilistic Models   |
|SRWarp: Generalized Image Super-Resolution under Arbitrary Transformation              | SRWarp                 | [cvpr21](https://arxiv.org/pdf/2104.10325.pdf)            |[code](https://github.com/sanghyun-son/srwarp)    |  arbitrary transformation   |
|Cross-MPI: Cross-scale Stereo for Image Super-Resolutionusing Multiplane Images        | Cross-MPI              | [cvpr21](https://arxiv.org/pdf/2011.14631.pdf)            |-   |  RefSR, plane-aware attention, coarse-to-fine guided upsampling    |
|Lightweight Image Super-Resolution with Hierarchical and DifferentiableNeural Architecture Search   | DLSR      | [arxiv](https://arxiv.org/pdf/2105.03939.pdf)            |[code](https://github.com/DawnHH/DLSR-PyTorch)    |  Lightweight   |
|HINet: Half Instance Normalization Network for Image Restoration   | HINet      | [arxiv](https://arxiv.org/pdf/2105.06086.pdf)            |-  |  Half Instance Normalization Block, 1st place on the NTIRE 2021 Image Deblurring Challenge - Track2. JPEG Artifacts   |
|FDAN: Flow-guided Deformable Alignment Network for Video Super-Resolution   | FDAN      | [arxiv](https://arxiv.org/pdf/2105.05640.pdf)            |-  |  ***VSR***, Flowguided Deformable Module  |

#### New NTIRE 2021 started! go [here](https://data.vision.ee.ethz.ch/cvl/ntire21/) 



