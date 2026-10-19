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
    - [2023](2023_papers.md)
	- [2024](2024_papers.md)
    - [2025](2025_papers.md)
- [Super Resolution workshop papers](workshops.md)
- [Super Resolution survey](sr_survey.md)

# Awesome-Super-Resolution（in progress）

Collect some super-resolution related papers, data and repositories.

## papers

### DL based approach

Note this table is referenced from [here](https://github.com/LoSealL/VideoSuperResolution/blob/master/README.md#network-list-and-reference-updating)

### 2026
More years papers, plase check Quick navigation

| Title                  | Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|Binarized High-Efficiency RAW Video Restoration and Beyond |- | [TPAMI](https://arxiv.org/abs/2608.16756) |  | binary neural network, RAW, video, efficient, lightweight, restoration |
|ENAF: A Multi-Exit Network with an Adaptive Patch Fusion for Large Image Super Resolution |ENAF | [WACV 2025](https://arxiv.org/abs/2608.15349) | [code](https://github.com/nmduonggg/ENAF) | CNN, dynamic network, multi-exit, patch fusion, efficient, super-resolution |
|Flow Straight to Reality: Perceptually Consistent Flow Matching for Efficient Image Restoration |PCFlow | [ECCV 2026](https://arxiv.org/abs/2608.10544) | [code](https://github.com/aiimaginglab/PCFlow) | flow matching, perceptual, distortion-perception, efficient, image, restoration |
|Preserve More Details: Mitigating Content Drift in Real-World Image Super-Resolution |- | [ACM MM 2026](https://arxiv.org/abs/2608.09373) |  | diffusion, stable diffusion, real-world, fidelity, image, super-resolution |
|Beyond Uniform Restoration: Empowering All-in-One Restoration with Pixel-Level Multimodal Guidance |- | [ACM MM 2026](https://arxiv.org/abs/2608.09482) |  | multimodal, prompt, pixel-level guidance, all-in-one, image, restoration |
|Bend the Basics: Degradation-Aware Deformable Tokenization for All-in-One Image Restoration |- | [ICML 2026](https://arxiv.org/abs/2608.06832) |  | transformer, deformable, tokenization, degradation-aware, all-in-one, restoration |
|Recurrent Sinusoidal INRs for Efficient High-Fidelity Representation |- | [ECCV 2026](https://arxiv.org/abs/2607.21485) |  | INR, sinusoidal, recurrent, spectral, efficient, representation |
|Decoupling Cross-Modality Manifold Discrepancy: Leveraging Visible Diffusion Priors for Infrared Super-Resolution |Shift-IISR | [ACM MM 2026](https://arxiv.org/abs/2607.21174) | [code](https://github.com/Assassink8/Shift-IISR) | diffusion, cross-modality, infrared, prior, image, super-resolution |
|The Second LoViF 2026 Challenge on Real-World All-in-One Image Restoration: Methods and Results |LoViF | [ECCV 2026 Workshop](https://arxiv.org/abs/2607.21118) |  | challenge, benchmark, real-world, all-in-one, image, restoration |
|QuReC: All-in-One Image Restoration with Query-Specific Guidance and Local-Global Response Calibration |QuReC | [ACM MM 2026](https://arxiv.org/abs/2607.15097) | [code](https://github.com/zhoushen1/QuReC) | transformer, query, attention, calibration, all-in-one, restoration |
|NanoVSR: Towards Real-Time Video Super-Resolution on Edge Devices |NanoVSR | [ECCV 2026](https://arxiv.org/abs/2607.10495) | [code](https://github.com/filippawlicki/nanovsr) | CNN, flow-free, real-time, edge device, video, super-resolution |
|Perceiving Better Moments: Cover Frame Reselection and Enhancement for Live Photos with the Live2K Dataset |Live2K | [ECCV 2026](https://arxiv.org/abs/2607.04151) |  | dataset, benchmark, live photos, frame selection, video, enhancement |
|AVSR-Diff: Scale-Agnostic Diffusion Priors for Temporally Consistent Arbitrary-Scale Video Super-Resolution |AVSR-Diff | [ECCV 2026](https://arxiv.org/abs/2607.00987) | [code](https://kaist-viclab.github.io/AVSR-Diff/) | diffusion, INR, arbitrary-scale, temporal consistency, video, super-resolution |
|SkelEM: Training-Signal Decoupling of Skeleton and Diffusion for Self-supervised Axial Super-Resolution in Volume Microscopy |SkelEM | [ECCV 2026](https://arxiv.org/abs/2606.30012) |  | diffusion, self-supervised, microscopy, anisotropic, axial, super-resolution |
|DTI: Dynamic Trajectory Initialization for Generative Face Video Super-Resolution |DTI | [ECCV 2026](https://arxiv.org/abs/2606.29198) |  | diffusion, generative prior, face, trajectory, video, super-resolution |
|FreqOrtho-SR: Frequency-Guided Orthogonal Expert Learning for Real-World Image Super-Resolution |FreqOrtho-SR | [ECCV 2026](https://arxiv.org/abs/2606.28745) | [code](https://github.com/sonhm3029/FreqOrtho-SR) | diffusion, frequency, mixture-of-experts, real-world, image, super-resolution |
|Dual-Prior Guided Null-Space Learning with Mixture-of-Splines for Arbitrary Medical Slice Super-Resolution |DP-NSL | [ECCV 2026](https://arxiv.org/abs/2606.26716) | [code](https://github.com/DeepMed-Lab-ECNU/Medical-Image-Reconstruction) | null-space, spline, arbitrary-scale, medical, slice, super-resolution |
|TaskTok: Delving into Task Tokens for Task-driven Image Restoration |TaskTok | [ECCV 2026](https://arxiv.org/abs/2606.26615) | [code](https://github.com/jimmy9704/TaskTok) | token, generative prior, task-driven, efficient, image, restoration |
|LogicIR: Logic Gate Networks for Image Restoration |LogicIR | [ECCV 2026](https://arxiv.org/abs/2606.26609) | [code](https://github.com/jimmy9704/LogicIR) | logic gate network, lightweight, efficient, inference, image, restoration |
|Fidelity- and Perception-Aware Local Implicit Attention for Arbitrary-Scale Image Super-Resolution |FPLIA | [ECCV 2026](https://arxiv.org/abs/2606.21910) | [code](https://github.com/XUSean0118/FPLIA) | INR, attention, arbitrary-scale, fidelity, perceptual, super-resolution |
|Linear Recurrent Unit with Semantic Modulation for Image Super-Resolution |LSM | [CVPR 2026](https://arxiv.org/abs/2606.19901) | [code](https://github.com/MingyuChoi-run/LSM) | linear recurrent unit, state space, semantic modulation, efficient, image, super-resolution |
|EvTexture++: Event-Driven Texture Enhancement for Video Super-Resolution |EvTexture++ | [TPAMI](https://arxiv.org/abs/2606.13580) | [code](https://github.com/DachunKai/EvTexture) | event camera, recurrent, texture, video, super-resolution |
|Resolving Endpoint Underfitting in Diffusion Bridges via Noise Alignment |NADB | [CVPR 2026](https://arxiv.org/abs/2605.28962) | [code](https://github.com/gyr02/NADB) | diffusion bridge, noise alignment, image, translation, restoration |
|Expandable, Compressible, Mineable: Open-World Thermal Image Restoration |ECMRNet | [ICML 2026](https://arxiv.org/abs/2605.16967) | [code](https://github.com/Kust-lp/ECMRNet) | continual learning, open-world, thermal, infrared, all-in-one, restoration |
|Image Restoration via Diffusion Models with Dynamic Resolution |SubDAPS | [ICML 2026](https://arxiv.org/abs/2605.14267) | [code](https://github.com/StarNextDay/SubDAPS.git) | diffusion, posterior sampling, dynamic resolution, efficient, image, restoration |
|TOC-SR: Task-Optimal Compact diffusion for Image Super Resolution |TOC-SR | [ECCV 2026 Workshop](https://arxiv.org/abs/2605.02767) |  | diffusion, distillation, one-step, bayesian optimization, efficient, super-resolution |
|Breaking Degradation Coupling: A Structural Entropy Guided Decoupled Framework and Benchmark for Infrared Enhancement |SEGD | [CVPR 2026](https://arxiv.org/abs/2604.22886) | [code](https://github.com/Kust-lp/SEGD) | structural entropy, benchmark, thermal infrared, all-in-one, image, enhancement |
| Trust but Verify: Adaptive Conditioning for Reference-Based Diffusion Super-Resolution via Implicit Reference Correlation Modeling |Ada-RefSR | [ICLR2026](https://arxiv.org/abs/2602.01864) | [code](https://github.com/vivoCameraResearch/AdaRefSR) | diffusion, one-step, reference-based, attention, image, super-resolution |
|LiveMoments: Reselected Key Photo Restoration in Live Photos via Reference-guided Diffusion |LiveMoments | [ICLR 2026](https://arxiv.org/abs/2604.12286) | [code](https://github.com/OpenVeraTeam/LiveMoments) | diffusion, flow-matching, reference-based, image, restoration, 4K |
|Energy-oriented Diffusion Bridge for Image Restoration with Foundational Diffusion Models |E-Bridge | [arxiv](https://arxiv.org/abs/2604.10983) | [code](https://jinnh.github.io/E-Bridge/) | diffusion, bridge, image, restoration, one-step, FLUX |
|SAT: Selective Aggregation Transformer for Image Super-Resolution |SAT | [CVPR 2026](https://arxiv.org/abs/2604.07994) | [code](https://github.com/PhuTran1005/SAT) | transformer, attention, image, efficient, reconstruction, super-resolution |
|DRIFT: Deep Restoration, ISP Fusion, and Tone-mapping |DRIFT | [CVPR 2026](https://arxiv.org/abs/2604.03402) |  | GAN, NAFNet, image, multi-frame, denoising, super-resolution |
|VOSR: A Vision-Only Generative Model for Image Super-Resolution |VOSR | [CVPR 2026](https://arxiv.org/abs/2604.03225) | [code](https://github.com/cswry/VOSR) | diffusion, DiT, DINOv2, distillation, one-step, super-resolution |
|The Eleventh NTIRE 2026 Efficient Super-Resolution Challenge Report |NTIRE | [CVPR 2026](https://arxiv.org/abs/2604.03198) |  | CNN, distillation, lightweight, benchmark, super-resolution |
|Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration |IQPIR | [CVPR 2026](https://arxiv.org/abs/2603.29773) |[code](https://github.com/fengyang1399-pixel/IQPIR)  | GAN, transformer, image, real-world, codebook, perceptual |
|Restore, Assess, Repeat: A Unified Framework for Iterative Image Restoration |RAR | [CVPR 2026](https://arxiv.org/abs/2603.26385) | [code](https://restore-assess-repeat.github.io) | flow, diffusion, image, all-in-one, blind, restoration |
|EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions |EgoXtreme | [CVPR 2026](https://arxiv.org/abs/2603.25135) | [code](https://taegyoun88.github.io/EgoXtreme/) | dataset, benchmark, real-world, 6dof, egocentric, motion-blur |
|DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution |DUO-VSR | [CVPR 2026](https://arxiv.org/abs/2603.22271) | [code](https://cszy98.github.io/DUO-VSR/) | diffusion, DiT, GAN, one-step, video, super-resolution |
|OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging |OmniFM | [CVPR 2026](https://arxiv.org/abs/2603.21660) |  | transformer, attention, medical, reconstruction, federated learning, frequency |
|Face2Scene: Using Facial Degradation as an Oracle for Diffusion-Based Scene Restoration |Face2Scene | [CVPR 2026](https://arxiv.org/abs/2603.16570) |[code](https://amirhossein-kz.github.io/face2scene/)  | diffusion, SD-Turbo, LoRA, one-step, blind, restoration |
|Empowering Semantic-Sensitive Underwater Image Enhancement with VLM |- | [AAAI 2026](https://arxiv.org/abs/2603.12773) |  | underwater, cross-attention, perceptual, LLaVA, image, enhancement |
|UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution |UCAN | [CVPR 2026](https://arxiv.org/abs/2603.11680) | [code](https://github.com/hokiyoshi/UCAN) | attention, CNN, lightweight, large-kernel, linear-attention, super-resolution |
|Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis |UniCAC | [CVPR 2026](https://arxiv.org/abs/2603.12083) | [code](https://github.com/XiaolongQian/UniCAC) | benchmark, CNN, transformer, blind, aberration correction, image |
|Compressed-Domain-Aware Online Video Super-Resolution |CDA-VSR | [CVPR 2026](https://arxiv.org/abs/2603.07694) | [code](https://github.com/sspBIT/CDA-VSR) | CNN, video, compressed-domain, motion vectors, online, super-resolution |
|Disentangled Textual Priors for Diffusion-based Image Super-Resolution |DTPSR | [CVPR 2026](https://arxiv.org/abs/2603.07430) |[code](https://github.com/JL6666JL/DTPSR)  | diffusion, cross-attention, frequency-aware, real-world, blind, super-resolution |
|Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset |Real-IISR | [CVPR 2026](https://arxiv.org/abs/2603.04745) | [code](https://github.com/JZD151/Real-IISR) | autoregressive, codebook, infrared, real-world, super-resolution |
|CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video |CubeComposer | [CVPR 2026](https://arxiv.org/abs/2603.04291) | [code](https://lg-li.github.io/project/cubecomposer) | diffusion, autoregressive, video, attention, 4K, 360 |
|All-in-One Image Restoration via Causal-Deconfounding Wavelet-Disentangled Prompt Network |CWP-Net | [TIP](https://arxiv.org/abs/2603.03839) |  | CNN, prompt, wavelet, blind, all-in-one, image |
|FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution |FiDeSR | [CVPR 2026](https://arxiv.org/abs/2603.02692) | [code](https://github.com/Ar0Kim/FiDeSR) | diffusion, distillation, one-step, fidelity, image, super-resolution |
|Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration |DATPRL-IR | [ICLR 2026](https://arxiv.org/abs/2603.01725) | [code](https://github.com/GuangluDong0728/DATPRL-IR) | transformer, prompt, distillation, image, all-in-one, restoration |
|Downstream Task Inspired Underwater Image Enhancement: A Perception-Aware Study from Dataset Construction to Network Design |DTIUIE | [TIP](https://arxiv.org/abs/2603.01767) | [code](https://github.com/oucailab/DTIUIE) | CNN, attention, underwater, perceptual, image |
|Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery |RepTRFD | [CVPR 2026](https://arxiv.org/abs/2603.01034) | [code](https://github.com/YangyangXu2002/RepTRFD) | INR, tensor, reconstruction, inpainting, point cloud |
|Neural Discrimination-Prompted Transformers for Efficient UHD Image Restoration and Enhancement |UHDPromer | [IJCV](https://arxiv.org/abs/2603.00853) | [code](https://github.com/supersupercong/uhdpromer) | transformer, attention, image, lightweight, restoration, 4K |
|ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration |ShiftLUT | [CVPR 2026](https://arxiv.org/abs/2603.00906) | [code](https://github.com/Sailor-t/ShiftLUT) | LUT, CNN, efficient, lightweight, image, restoration |
|AlignVAR: Towards Globally Consistent Visual Autoregression for Image Super-Resolution |AlignVAR | [CVPR 2026](https://arxiv.org/abs/2603.00589) |  | autoregressive, attention, image, efficient, perceptual, super-resolution |
|SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting |SR3R | [CVPR 2026](https://arxiv.org/abs/2602.24020) |[code](https://xiangfeng66.github.io/SR3R/)  | transformer, 3D, gaussian splatting, feed-forward, super-resolution |
|DACESR: Degradation-Aware Conditional Embedding for Real-World Image Super-Resolution |DACESR | [TIP](https://arxiv.org/abs/2602.23890) | [code](https://github.com/nathan66666/DACESR.git) | Mamba, contrastive learning, real-world, image, super-resolution |
|Physics-Consistent Diffusion for Efficient Fluid Super-Resolution via Multiscale Residual Correction |ReMD | [CVPR 2026](https://arxiv.org/abs/2603.00149) | [code](https://github.com/lizhihao2022/ReMD) | diffusion, fluid, wavelet, multigrid, super-resolution |
|GSNR: Graph Smooth Null-Space Representation for Inverse Problems |GSNR | [CVPR 2026](https://arxiv.org/abs/2602.20328) |  | graph laplacian, plug-and-play, diffusion, image, reconstruction, inverse problems |
|Deep LoRA-Unfolding Networks for Image Restoration |LoRA | [TIP](https://arxiv.org/abs/2602.18697) |  | deep unfolding, LoRA, reconstruction, compressive sensing, image, efficient |
|Vascular anatomy-aware self-supervised pre-training for X-ray angiogram analysis |XA-SSL | [AAAI 2026](https://arxiv.org/abs/2602.11536) | [code](https://github.com/Dxhuang-CASIA/XA-SSL) | transformer, masked image modeling, medical, pre-training, reconstruction |
|Eliminating VAE for Fast and High-Resolution Generative Detail Restoration |GenDR-Pix | [ICLR 2026](https://arxiv.org/abs/2602.10630) | | diffusion, distillation, one-step, real-world, 4K, super-resolution |
|NanoSD: Edge Efficient Foundation Model for Real Time Image Restoration |NanoSD | [CVPR 2026](https://arxiv.org/abs/2601.09823) |  | diffusion, stable diffusion, distillation, edge device, real-time, super-resolution |


