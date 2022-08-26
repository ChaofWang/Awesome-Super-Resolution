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
|C3-STISR: Scene Text Image Super-resolution with Triple Clues| C3-STISR | [IJCAI2022](https://arxiv.org/pdf/2204.14044.pdf) | [code](https://github.com/zhaominyiz/C3-STISR) | Scene text SR |
|Lightweight Bimodal Network for Single-Image Super-Resolution via Symmetric CNN and Recursive Transformer| LBNet | [IJCAI2022](https://arxiv.org/pdf/2204.13286.pdf) | [code](https://github.com/IVIPLab/LBNet) | Lightweight SISR,  Symmetric CNN, Recursive Transformer |
|Attentive Fine-Grained Structured Sparsity for Image Restoration| LBNet | [CVPR2022](https://arxiv.org/pdf/2204.12266.pdf) | [code](https://github.com/JungHunOh/SLS_CVPR2022) | Layer-wise N:M structured Sparsity pruning|
|A New Dataset and Transformer for Stereoscopic Video Super-Resolution| Trans-SVSR | [CVPR2022](https://arxiv.org/pdf/2204.10039.pdf) | [code](https://github.com/H-deep/Trans-SVSR/) | Stereo video super-resolution|
|Accelerating the Training of Video Super-Resolution| - | [arxiv](https://arxiv.org/pdf/2205.05069.pdf) | [code](https://github.com/TencentARC/Efficient-VSR-Training) |  ***VideoSR***, 6.2× speedup in wall-clock training time|
|Metric Learning based Interactive Modulation for Real-World Super-Resolution| MM-RealSR | [arxiv](https://arxiv.org/pdf/2205.05065.pdf) | [code](https://github.com/TencentARC/MM-RealSR) | Metric Learning based Interactive Modulation |
|Activating More Pixels in Image Super-Resolution Transformer | HAT |[arxiv](https://arxiv.org/pdf/2205.04437.pdf) | [code](https://github.com/chxy95/HAT) | SISR,SOTA,  Hybrid Attention Transformer,  more than 1dB |
|SPQE: Structure-and-Perception-Based Quality Evaluation for Image Super-Resolution | - |[arxiv](https://arxiv.org/pdf/2205.03584.pdf) |-  | SR-IQA |
|Spatial-Temporal Space Hand-in-Hand:Spatial-Temporal Video Super-Resolution via Cycle-Projected Mutual Learning | CycMu-Net  |[arxiv](https://arxiv.org/pdf/2205.05264.pdf) |[code](https://github.com/hhhhhumengshun/CycMuNet)  | ***ST-VideoSR***,Cycle-Projected Mutual Learning |
|RepSR: Training Efficient VGG-style Super-Resolution Networks with Structural Re-Parameterization and Batch Normalization |RepSR  |[arxiv](https://arxiv.org/pdf/2205.05671.pdf) |[code](https://github.com/TencentARC/RepSR)  | Efficient SISR, lightweight, VGG-like, Structural Re-Parameterization and Batch Normalization|
|Blueprint Separable Residual Network for Efficient Image Super-Resolution |BSRN  |[arxiv](https://arxiv.org/pdf/2205.05996.pdf) |[code](https://github.com/xiaom233/BSRN )  | Efficient SISR, lightweight, blueprint separable convolution|
|Evaluating the Generalization Ability of Super-Resolution Networks | SRGA  |[arxiv](https://arxiv.org/pdf/2205.07019.pdf) |-  | Generalization Assessment Index, Patch-based Image Evaluation Set|
|Residual Local Feature Network for Efficient Super-Resolution | RLFN  |[arxiv](https://arxiv.org/pdf/2205.07514.pdf) |[code](https://github.com/fyan111/RLFN)  | Efficient SISR, lightweight, Residual Local Feature Network|
|Textural-Structural Joint Learning for No-Reference Super-Resolution Image Quality Assessment | NRIQA  |[arxiv](https://arxiv.org/pdf/2205.13847.pdf) |[code](https://github.com/yuqing-liu-dut/NRIQA_SR)  | No-Reference Super-Resolution Image Quality Assessment|
|ShuffleMixer: An Efficient ConvNet for Image Super-Resolution | ShuffleMixer  |[arxiv](https://arxiv.org/pdf/2205.15175.pdf) |[code](https://github.com/sunny2109/MobileSR-NTIRE2022)  | Efficient SISR, lightweight, point wises MLP |
|Real-Time Super-Resolution for Real-World Images on Mobile Devices | -  |[arxiv](https://arxiv.org/pdf/2206.01777.pdf) |-  | Efficient SISR, lightweight, 50fps |
|Real-World Image Super-Resolution by Exclusionary Dual-Learning | RWSR-EDL   |[arxiv](https://arxiv.org/pdf/2206.02609.pdf) |[code](https://github.com/House-Leo/RWSR-EDL)  |Dual Learning |
|Learning Trajectory-Aware Transformer for Video Super-Resolution |TTVSR   |[CVPR2022 oral](https://arxiv.org/pdf/2204.04216.pdf) |[code](https://github.com/researchmm/TTVSR)  |***VideoSR***, Trajectory-awareTransformer|
|LAR-SR: A Local Autoregressive Model for Image Super-Resolution |LAR-SR   |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_LAR-SR_A_Local_Autoregressive_Model_for_Image_Super-Resolution_CVPR_2022_paper.pdf) |-  |GAN-based model，Autoregressive |
|Memory-Augmented Non-Local Attention for Video Super-Resolution | MANA |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Memory-Augmented_Non-Local_Attention_for_Video_Super-Resolution_CVPR_2022_paper.pdf) |[code](https://github.com/jiy173/MANA)  |***VideoSR***, Memory-Augmented Attention|
|Learning Graph Regularisation for Guided Super-Resolution | - |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/de_Lutio_Learning_Graph_Regularisation_for_Guided_Super-Resolution_CVPR_2022_paper.pdf) |[code](https://github.com/prs-eth/graph-super-resolution)  |graph-super-resolution|
|videoINR: Learning Video Implicit Neural Representation for Continuous Space-Time Super-Resolution | VideoINR |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_VideoINR_Learning_Video_Implicit_Neural_Representation_for_Continuous_Space-Time_Super-Resolution_CVPR_2022_paper.pdf) |[code](https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution)  |***ST-VideoSR***,Video Implicit Neural Representation |
|Stable Long-Term Recurrent Video Super-Resolution | MRVSR |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Chiche_Stable_Long-Term_Recurrent_Video_Super-Resolution_CVPR_2022_paper.pdf) |[code](https://github.com/bjmch/MRVSR)  |***VideoSR***,  Lipschitz stability  |
|Blind Image Super-resolution with Elaborate Degradation Modeling on Noise and Kernel | BSRDM |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Yue_Blind_Image_Super-Resolution_With_Elaborate_Degradation_Modeling_on_Noise_and_CVPR_2022_paper.pdf) |[code](https://github.com/zsyOAOA/BSRDM)  | Monte Carlo EM|
|Reflash Dropout in Image Super-Resolution | - |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Kong_Reflash_Dropout_in_Image_Super-Resolution_CVPR_2022_paper.pdf) |-  |  Dropout in sr|
|Reference-based Video Super-Resolution Using Multi-Camera Video Triplets | RefVSR |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Reference-Based_Video_Super-Resolution_Using_Multi-Camera_Video_Triplets_CVPR_2022_paper.pdf) |[code](https://github.com/codeslake/RefVSR)  | RefVSR|
|SphereSR: 360◦ Image Super-Resolution with Arbitrary Projection via Continuous Spherical Image Representation | SphereSR |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Yoon_SphereSR_360deg_Image_Super-Resolution_With_Arbitrary_Projection_via_Continuous_Spherical_CVPR_2022_paper.pdf) |-  | SphereSR|
|Investigating Tradeoffs in Real-World Video Super-Resolution | RealBasicVSR |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Chan_Investigating_Tradeoffs_in_Real-World_Video_Super-Resolution_CVPR_2022_paper.pdf) |[code](https://github.com/ckkelvinchan/RealBasicVSR)  | RealBasicVSR|
|Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites | HDR-DSP |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Nguyen_Self-Supervised_Super-Resolution_for_Multi-Exposure_Push-Frame_Satellites_CVPR_2022_paper.pdf) |[code](https://centreborelli.github.io/HDR-DSP-SR)  | Self-Supervised|
|Texture-based Error Analysis for Image Super-Resolution | - |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Magid_Texture-Based_Error_Analysis_for_Image_Super-Resolution_CVPR_2022_paper.pdf) |-  | Texture Embedding|
|A Text Attention Network for Spatial Deformation Robust Scene Text Image Super-resolution | TATT |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_A_Text_Attention_Network_for_Spatial_Deformation_Robust_Scene_Text_CVPR_2022_paper.pdf) |[code](https://github.com/mjq11302010044/TATT)  | Scene Text SR|
|MNSRNet: Multimodal Transformer Network for 3D Surface Super-Resolution | MNSRNet |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_MNSRNet_Multimodal_Transformer_Network_for_3D_Surface_Super-Resolution_CVPR_2022_paper.pdf) |-  | 3D surface super-resolution|
|Task Decoupled Framework for Reference-based Super-Resolution | - |[CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Task_Decoupled_Framework_for_Reference-Based_Super-Resolution_CVPR_2022_paper.pdf) |-  | -|
|Joint Super-Resolution and Inverse Tone-Mapping:A Feature Decomposition Aggregation Network and A New Benchmark | FDAN |[TIP](https://arxiv.org/pdf/2207.03367.pdf) |-  | joint SR-ITM|
|Cross-receptive Focused Inference Network for Lightweight Image Super-Resolution | CFIN |[arxiv](https://arxiv.org/pdf/2207.02796.pdf) |-  | SISR, Lightweight|
|Degradation-Guided Meta-Restoration Network for Blind Super-Resolution | DMSR |[arxiv](https://arxiv.org/pdf/2207.00943.pdf) |-  | blind SR|
|Residual Sparsity Connection Learning for Efficient Video Super-Resolution| RSCL |[arxiv](https://arxiv.org/pdf/2206.07687.pdf) |-  | ***VideoSR***,Residual Sparsity Connection|
|AnimeSR: Learning Real-World Super-Resolution Models for Animation Videos| AnimeSR |[arxiv](https://arxiv.org/pdf/2206.07038.pdf) |-  | Animation Video SR|
|Learning a Degradation-Adaptive Network for Light Field Image Super-Resolution| LF-DAnet |[arxiv](https://arxiv.org/pdf/2206.06214.pdf) |[code](https://github.com/YingqianWang/LF-DAnet)  | Light Field SR|
|CADyQ: Content-Aware Dynamic Quantization for Image Super-Resolution| CADyQ |[ECCV2022](https://arxiv.org/pdf/2207.10345.pdf) |[code](https://github.com/Cheeun/CADyQ)  | |
|Towards Interpretable Video Super-Resolution via Alternating Optimization| DAVSR |[ECCV2022](https://arxiv.org/pdf/2207.10345.pdf) |[code](https://github.com/caojiezhang/DAVSR)  | ***STVSR***, Motion Blur, Motion Aliasing |
|Reference-based Image Super-Resolution with Deformable Attention Transformer | DATSR |[ECCV2022](https://arxiv.org/pdf/2207.11938.pdf) |[code](https://github.com/caojiezhang/DATSR)  | RefSR, Correspondence Matching, Texture Transfer, Deformable Attention Transformer |
|Learning Series-Parallel Lookup Tables for Efficient Image Super-Resolution | SPLUT |[ECCV2022](https://arxiv.org/pdf/2207.12987.pdf) |[code](https://github.com/zhjy2016/SPLUT)  |  SISR，look-up table, series-parallel network |
|Learning Spatiotemporal Frequency-Transformer for Compressed Video Super-Resolution | FTVSR |[ECCV2022](https://arxiv.org/pdf/2208.03012.pdf) |[code](https://github.com/researchmm/FTVSR)  |  ***VSR***，Transformer, Frequency Learning, Compression |
|Image Super-Resolution with Deep Dictionary | SRDD |[ECCV2022](https://arxiv.org/pdf/2207.09228.pdf) |[code](https://github.com/shuntama/srdd)  |  SISR,Deep Dictionary, Sparse Representation |
|Learning Mutual Modulation for Self-Supervised Cross-Modal Super-Resolution | MMSR |[ECCV2022](https://arxiv.org/pdf/2207.09156.pdf) |[code](https://github.com/palmdong/MMSR)  |  Mutual Modulation, Self-Supervised Super-Resolution, Cross-Modal, Multi-Modal |
|Compiler-Aware Neural Architecture Search for On-Mobile Real-time Super-Resolution | - |[arxiv](https://arxiv.org/pdf/2207.12577.pdf) |[code](https://github.com/wuyushuwys/compiler-aware-nas-sr)  | SISR, Mobile Real-time，nas |
|Enhancing Image Rescaling using Dual Latent Variables in Invertible Neural Network| DLV-IRN |[ACMM 2022](https://arxiv.org/pdf/2207.11844.pdf) |- | Image Rescaling |
|Perception-Distortion Balanced ADMM Optimization for Single-Image Super-Resolution | PDASR |[arxiv](https://arxiv.org/pdf/2208.03324.pdf) |[code](https://github.com/Yuehan717/PDASR)  |  Perception-Distortion Trade-Off, Constrained Optimization |
|Adaptive Local Implicit Image Function for Arbitrary-scale Super-resolution | A-LIIF |[ICIP2022](https://arxiv.org/pdf/2208.04318.pdf) |[code](https://github.com/LeeHW-THU/A-LIIF)  |   implicit image function |
|Rethinking Alignment in Video Super-Resolution Transformers | - |[arxiv](https://arxiv.org/pdf/2207.08494.pdf) |[code](https://github.com/XPixelGroup/RethinkVSRAlignment)  |  ***VSR***, feature alignment |
|SwinFIR: Revisiting the SwinIR with Fast Fourier Convolution and Improved Training for Image Super-Resolution | SwinFIR |[arxiv](https://arxiv.org/pdf/2208.11247.pdf) |- |  SISR, Fast Fourier Convolution，SOTA |
