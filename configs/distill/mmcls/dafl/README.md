# DAFL

> [Data-Free Learning of Student Networks](https://arxiv.org/abs/1904.01186)

<!-- [ALGORITHM] -->

## Abstract

Learning portable neural networks is very essential for computer vision for the purpose that pre-trained heavy deep models can be well applied on edge devices such as mobile phones and micro sensors. Most existing deep neural network compression and speed-up methods are very effective for training compact deep models, when we can directly access the training dataset. However, training data for the given deep network are often unavailable due to some practice problems (e.g. privacy, legal issue, and transmission), and the architecture of the given network are also unknown except some interfaces. To this end, we propose a novel framework for training efficient deep neural networks by exploiting generative adversarial networks (GANs). To be specific, the pre-trained teacher networks are regarded as a fixed discriminator and the generator is utilized for deviating training samples which can obtain the maximum response on the discriminator. Then, an efficient network with smaller model size and computational complexity is trained using the generated data and the teacher network, simultaneously. Efficient student networks learned using the pro- posed Data-Free Learning (DAFL) method achieve 92.22% and 74.47% accuracies using ResNet-18 without any training data on the CIFAR-10 and CIFAR-100 datasets, respectively. Meanwhile, our student network obtains an 80.56% accuracy on the CelebA benchmark.

![pipeline](/docs/en/imgs/model_zoo/dafl/pipeline.png)

## Results and models

### Classification

| Location | Dataset  |                                                   Teacher                                                    |                                                   Student                                                    |  Acc  | Acc(T) | Acc(S) |                          Config                           | Download                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| :------: | :------: | :----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: | :---: | :----: | :----: | :-------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  logits  | ImageNet | [resnet34](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet34_8xb32_in1k.py) | [resnet18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py) | 71.54 | 73.62  | 69.90  | [config](./wsld_cls_head_resnet34_resnet18_8xb32_in1k.py) | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) \|[model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v0.1/distill/wsld/wsld_cls_head_resnet34_resnet18_8xb32_in1k/wsld_cls_head_resnet34_resnet18_8xb32_in1k_acc-71.54_20211222-91f28cf6.pth?versionId=CAEQHxiBgMC6memK7xciIGMzMDFlYTA4YzhlYTRiMTNiZWU0YTVhY2I5NjVkMjY2) \| [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v0.1/distill/wsld/wsld_cls_head_resnet34_resnet18_8xb32_in1k/wsld_cls_head_resnet34_resnet18_8xb32_in1k_20211221_181516.log.json?versionId=CAEQHxiBgIDLmemK7xciIGNkM2FiN2Y4N2E5YjRhNDE4NDVlNmExNDczZDIxN2E5) |

## Citation

```latex
@inproceedings{DBLP:conf/iccv/ChenW0YLSXX019,
  author    = {Hanting Chen, Yunhe Wang, Chang Xu, Zhaohui Yang, Chuanjian Liu, Boxin Shi,
               Chunjing Xu, Chao Xu and Qi Tian},
  title     = {Data-Free Learning of Student Networks},
  booktitle = {2019 {IEEE/CVF} International Conference on Computer Vision, {ICCV}
               2019, Seoul, Korea (South), October 27 - November 2, 2019},
  pages     = {3513--3521},
  publisher = {{IEEE}},
  year      = {2019},
  url       = {https://doi.org/10.1109/ICCV.2019.00361},
  doi       = {10.1109/ICCV.2019.00361},
  timestamp = {Mon, 17 May 2021 08:18:18 +0200},
  biburl    = {https://dblp.org/rec/conf/iccv/ChenW0YLSXX019.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
