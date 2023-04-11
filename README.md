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
    - [2022](2022_papers.md)
    - [2023](#2023)
- [Super Resolution workshop papers](workshops.md)
- [Super Resolution survey](sr_survey.md)

# Awesome-Super-Resolution（in progress）

Collect some super-resolution related papers, data and repositories.

## papers

### DL based approach

Note this table is referenced from [here](https://github.com/LoSealL/VideoSuperResolution/blob/master/README.md#network-list-and-reference-updating)

### 2023
More years papers, plase check Quick navigation

| Title                  | Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|Efficient Image Super-Resolution with Feature Interaction Weighted Hybrid Network | FIWHN | [AAAI2022](https://arxiv.org/pdf/2212.14181.pdf) | - |- |
|DIVA: Deep Unfolded Network from Quantum Interactive Patches for Image Restoration | DIVA | [arxiv](https://arxiv.org/pdf/2301.00247.pdf) | - |- |
|DLGSANet: Lightweight Dynamic Local and Global Self-Attention Networks for Image Super-Resolution | DLGSANet | [arxiv](https://arxiv.org/pdf/2301.02031.pdf) | [code](https://neonleexiang.github.io/DLGSANet) |- |
|The Best of Both Worlds: A Framework for Combining Degradation Prediction with High Performance Super-Resolution Networks | - | [MDPI Sensors](https://doi.org/10.3390/s23010419) | [code](https://github.com/um-dsrg/RUMpy) |- |
|Image Super-Resolution using Efficient Striped Window Transformer | ESWT | [arxiv](https://arxiv.org/pdf/2301.09869.pdf) | [code](https://github.com/Fried-Rice-Lab/FriedRiceLab) |SISR|
|A statistically constrained internal method for single image super-resolution | - | [arxiv](https://arxiv.org/pdf/2302.01648.pdf) |  |-|
|OSRT: Omnidirectional Image Super-Resolution with Distortion-aware Transformer | OSRT | [arxiv](https://arxiv.org/pdf/2302.03453.pdf) | - |ODISR|
|Denoising Diffusion Probabilistic Models for Robust Image Super-Resolution in the Wild | SR3+ | [arxiv](https://arxiv.org/pdf/2302.07864.pdf) | - |blind SR, DDPMs|
|Kernelized Back-Projection Networks for Blind Super Resolution | KCBPN/KBPN | [arxiv](https://arxiv.org/pdf/2302.08478.pdf) | [code](https://github.com/Yuki-11/KBPN) |blind SR|
|Improving Scene Text Image Super-Resolution via Dual Prior Modulation Network | DPMN | [arxiv](https://arxiv.org/pdf/2302.10414.pdf) | [code](https://github.com/jdfxzzy/DPMN) |Scene text SR|
|CDPMSR: CONDITIONAL DIFFUSION PROBABILISTIC MODELS FOR SINGLE IMAGE SUPER-RESOLUTION | CDPMSR | [arxiv](https://arxiv.org/pdf/2302.12831.pdf) | - |DDPMs|
|Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution | SAFMN | [arxiv](https://arxiv.org/pdf/2302.13800.pdf) | [code](https://github.com/sunny2109/SAFMN) |SISR, lightweight|
|Efficient and Explicit Modelling of Image Hierarchies for Image Restoration | GRL | [CVPR23](https://arxiv.org/pdf/2303.00748.pdf) | [code](https://github.com/ofsoundof/GRL-Image-Restoration) |SISR|
|Zero Shot Image Restoration Using Denoising Diffusion Null-Space Model  | DDNM | [ICLR23](https://arxiv.org/pdf/2212.00490.pdf) | [code](https://github.com/wyhuai/DDNM/tree/main) |DDNM|
|Unlimited-Size Diffusion Restoration | - | [arxiv](https://arxiv.org/pdf/2303.00354.pdf) | [code](https://github.com/wyhuai/DDNM/tree/main) |DDNM|
|OPE-SR: Orthogonal Position Encoding for Designing a Parameter-free Upsampling Module in Arbitrary-scale Image Super-Resolution | OPE | [CVPR23](https://arxiv.org/pdf/2303.01091.pdf) | - |Arbitrary-scale SR|
|Super-Resolution Neural Operator | SRNO | [CVPR23](https://arxiv.org/pdf/2303.02584.pdf) | [code](https://github.com/2y7c3/Super-Resolution-Neural-Operator) |continuous SR|
|Self-Asymmetric Invertible Network for Compression-Aware Image Rescaling | SAIN | [AAAI23](https://arxiv.org/pdf/2303.02353.pdf) | - |Image Rescaling|
|QuickSRNet: Plain Single-Image Super-Resolution Architecture for Faster Inference on Mobile Platforms | QSRNet | [arxiv](https://arxiv.org/pdf/2303.04336.pdf) | - |SISR, lightweight|
|Burstormer: Burst Image Restoration and Enhancement Transformer | Burstormer | [CVPR23](https://arxiv.org/pdf/2304.01194.pdf) | [code](https://github.com/akshaydudhane16/Burstormer) ||
|Better “CMOS” Produces Clearer Images:Learning Space-Variant Blur Estimation for Blind Image Super-Resolution | CMOS | [CVPR23](https://arxiv.org/pdf/2304.03542.pdf) | [code](https://github.com/ByChelsea/CMOS) |Blind SR|
|Waving Goodbye to Low-Res: A Diffusion-Wavelet Approach for Image Super-Resolution | DiWa | [arxiv](https://arxiv.org/abs/2304.01994) | - |Diffusion-Wavelet|
|Generative Diffusion Prior for Unified Image Restoration and Enhancement | GDP | [CVPR23](https://arxiv.org/pdf/2304.01247.pdf) | [code](https://generativediffusionprior.github.io) |unified image recovery|
|Tunable Convolutions with Parametric Multi-Loss Optimization | - | [CVPR23](https://arxiv.org/pdf/2304.00898.pdf) |   | |
|SOSR: Source-Free Image Super-Resolution with Wavelet Augmentation Transformer| SOSR | [arxiv](https://arxiv.org/pdf/2303.17783.pdf) |    |
|Cascaded Local Implicit Transformer for Arbitrary-Scale Super-Resolution | CLIT | [arxiv](https://arxiv.org/pdf/2303.16513.pdf) | [code](https://github.com/jaroslaw1007/CLIT) | |
|Implicit Diffusion Models for Continuous Super-Resolution | IDM | [arxiv](https://arxiv.org/pdf/2303.16491.pdf) | [code](https://github.com/Ree1s/IDM) | |
|Unlocking Masked Autoencoders as Loss Function for Image and Video Restoration| - | [arxiv](https://arxiv.org/pdf/2303.16411.pdf) |   | |
|Learning Generative Structure Prior for Blind Text Image Super-resolution | MARCONet | [CVPR23](https://arxiv.org/pdf/2303.14726.pdf) | [code](https://github.com/csxmli2016/MARCONet) |Blind Text SR |
|Toward DNN of LUTs: Learning Efficient Image Restoration with Multiple Look-Up Tables | MuLUT | [arxiv](https://arxiv.org/pdf/2303.14506.pdf) | [code](https://github.com/ddlee-cn/MuLUT) | |
|Incorporating Transformer Designs into Convolutions for Lightweight Image Super-Resolution | TCSR | [arxiv](https://arxiv.org/pdf/2303.14324.pdf) | [code](https://github.com/Aitical/TCSR) |LightweightSR |
|Learning Spatial-Temporal Implicit Neural Representations for Event-Guided Video Super-Resolution | egvsr | [CVPR23](https://arxiv.org/pdf/2303.13767.pdf) | [code](https://vlis2022.github.io/cvpr23/egvsr) |VSR |
|Human Guided Ground-truth Generation for Realistic Image Super-resolution | HGGT | [CVPR23](https://arxiv.org/pdf/2303.13069.pdf) | [code](https://github.com/ChrisDud0257/HGGT) |RealSR |
|EBSR: Enhanced Binary Neural Network for Image Super-Resolution | EBSR | [arxiv](https://arxiv.org/pdf/2303.12270.pdf) | |Binary NN |
|A High-Frequency Focused Network for Lightweight Single Image Super-Resolution | HFFN | [arxiv](https://arxiv.org/pdf/2303.11701.pdf) |- |LightweightSR |
|Learning Data-Driven Vector-Quantized Degradation Model for Animation Video Super-Resolution | VQD-SR | [arxiv](https://arxiv.org/pdf/2303.09826.pdf) |- |Animation VSR |
|SRFormer: Permuted Self-Attention for Single Image Super-Resolution | SRFormer | [arxiv](https://arxiv.org/pdf/2303.09735.pdf) | - | |
|A High-Performance Accelerator for Super-Resolution Processing on Embedded GPU | - | [arxiv](https://arxiv.org/pdf/2303.08999.pdf) |- | |
|DeblurSR: Event-Based Motion Deblurring Under the Spiking Representation | DeblurSR | [arxiv](https://arxiv.org/pdf/2303.08977.pdf) | [code](https://github.com/chensong1995/DeblurSR) | |
|ResDiff: Combining CNN and Diffusion Model for Image Super-Resolution | ResDiff | [arxiv](https://arxiv.org/pdf/2303.08714.pdf) | - |  |
|Towards High-Quality and Efficient Video Super-Resolution via Spatial-Temporal Data Overfitting | STDO | [CVPR23](https://arxiv.org/pdf/2303.08331.pdf) | [code](https://github.com/coulsonlee/STDO-CVPR2023) |VSR |
|Synthesizing Realistic Image Restoration Training Pairs: A Diffusion Approach | - | [arxiv](https://arxiv.org/pdf/2303.06994.pdf) | - | |
|Recursive Generalization Transformer for Image Super-Resolution | RGT | [arxiv](https://arxiv.org/pdf/2303.06373.pdf) | - |  |
|Local Implicit Normalizing Flow for Arbitrary-Scale Image Super-Resolution | LINF | [CVPR23](https://arxiv.org/pdf/2303.05156.pdf) | - |  |
|LMR: A Large-Scale Multi-Reference Dataset for Reference-based Super-Resolution | MRefSR | [arxiv](https://arxiv.org/pdf/2303.04970.pdf) | [code](https://github.com/wdmwhh/MRefSR) |  |
