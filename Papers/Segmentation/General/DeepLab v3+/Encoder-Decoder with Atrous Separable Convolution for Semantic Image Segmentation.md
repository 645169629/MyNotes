## Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

> ECCV 2018 "DeepLab v3+"

### Abstract

​	Spatial pyramid pooling module 或 encode-decoder结构被用于语义分割网络结构中。第一种网络可以编码多尺度的上下文信息，通过在多个尺度（感受野）对输入特征进行过滤或池化操作。第二种网络通过逐渐恢复空间信息来捕获更好的边缘。本文提出结合两种方法的优势。我们提出的模型，DeepLabv3+，拓展了DeepLabv3，增加了一个decoder模块来refine分割结果，特别是目标的边界。我们进一步研究了Xception模块，并在Atrous 空间金字塔池化和decoder模块应用了逐深度可分离卷积，得到了一个更快更强的编解码网络。在PASCAL VOC 2012和Cityscapes数据集上达到了89%和82.1%。

### Introduction

​	本文考虑两种网络结构：1）使用空间金字塔池化模块，通过对不同分辨率的特征池化，捕获丰富的上下文信息。2）使用编解码结构，可以获得sharp的目标边界。

​	为了获得多个尺度的上下文信息，DeepLabv3应用了一些平行的，不同rate的膨胀卷积（称为Atrous Spatial Pyramid Pooling）,而PSPNet以不同的网格尺度，执行了池化操作。尽管丰富的语义信息被编码到了最后的特征图上，目标边界细节的信息却因为卷积池化丢失了。可以通过应用atrous 卷积获得更密集的特征图来减轻这一现象。但是由于网络的设计和显存的限制，不可以提取比输入分辨率小8倍甚至4倍的特征图。如ResNet-101，当提取1/16的特征图时，最后三个残差块（9层）都必须dilated。如果特征图是1/8，则会影响26个残差块（78层）。因此，提取密集特征是计算开销很大。另一方面，encoder-decoder模型计算很快（没有特征dilated），并且在decoder path逐渐恢复目标sharp边缘。为了结合两种方法的优势，我们提出将多尺度上下文信息包含到encoder-decoder网络的encoder模块。

​	如图1所示。DeepLabv3编码了丰富的语义信息。（根据atrous 卷积，可以控制encoder特征的密度）。并且，decoder模块可以恢复边界细节。

​	收到近期逐深度可分隔卷积（depthwise separable convolution）的启发，我们也研究了该操作（应用Xception模型，并应用atrous separable convolution 到ASPP和decoder模块），获得了更快的速度，更好的准确率。

贡献：

1）提出新的encoder-decoder结构，利用了DeepLabv3作为encoder模块和简单的decoder模块。

2）在我们的结构中，我们可以任意控制encoder提取特征的分辨率（通过atrous convolution），其他encoder-decoder模型是不行的。

3）应用了Xception以及在ASPP和decoder上应用了depthwise separable convolution，获得了更快更强的网络。

4）我们达到了新state-of-art。

5）开源代码。

### Related Work

​	利用上下文信息：1）多尺度输入（图像金字塔）；2）概率图模型（如DenseCRF）。

**Spatial pyramid pooling: ** PSPNet以不同网格尺度执行空间金字塔池化，DeepLab采用多个平行atrous convolution（不同rate，称为Atrous Spatial Pyramid Pooling，ASPP）。这些模型利用多尺度信息获得了不错的结果。

**Encoder-decoder:** 1）包括enccoder模块，逐步减少特征图，捕获更高的语义信息；2）decodder模块，逐渐恢复空间信息。

**Depthwise separable convolution: ** Depthwise separable convolution 或 group convolution可以减少计算开销以及参数数量，同时保持性能。特别的，我们应用了Xception 模型，得到了更快更强的网络。

### Methods

#### 3.1 Encoder-Decoder with Atrous Convolution

**Atrous convolution: ** 