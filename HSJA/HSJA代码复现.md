# HSJA代码复现

原文链接：http://arxiv.org/abs/1904.02144

本文采用的数据集为CIFAR10，网络模型为ResNet。

## HSJA对抗样本与原始样本对比

### untargeted HSJA对抗样本

![untargeted-l2-0](./Results/untargeted-l2-0.jpg)

![untargeted-l2-1](./Results/untargeted-l2-1.jpg)

### targeted HSJA对抗样本

![targeted-l2-0](./Results/targeted-l2-0.jpg)

![targeted-l2-1](./Results/targeted-l2-1.jpg)

上面4张图中，左图是原始图片，右图是对抗图片。我们不难看出，上面4组图片左右图人肉眼难以分辨。这与原文展示的结果类似。

下图为原文的对抗样本与原始样本对比。

![Visualized trajectories of HopSkipJumpAttack for optimizing `2 distance on randomly selected images in CIFAR-10 and ImageNet](./Results/Visualized%20trajectories%20of%20HopSkipJumpAttack%20for%20optimizing%20%602%20distance%20on%20randomly%20selected%20images%20in%20CIFAR-10%20and%20ImageNet.png)