## DeepLab：Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

>

### Abstract

三个贡献：1. 空洞卷积使我们可以明确控制特征响应的分辨率，同时在不增加参数数量的同时扩大感受野。2. 提出了atrous spatial pyramid polling（ASPP）以在多尺度分割目标。ASPP使用多个采样率和感受野下的filter来探索卷积特征层，因此可以在多尺度上捕获目标和图像上下文。3. 将DCNNs与概率图模型（Fully Connected CRFs）相结合提高目标边界定位。

### 1. Introduction

	DCNN应用于语义分割的挑战：1）减少的特征分辨率；2）目标存在多尺度；3）由于DCNN的不变性，定位准确度会降低。

解决第一个挑战：我们移除了DCNNs后面几层降采样和max pooling层，并将后面巻积层的filter替换为上采样filter，生成以更高采样率计算的特征图。Filter上采样相当于在filter中插入空洞。接着进行双线性插值，将特征响应返回原图大小。

解决第二个挑战：在卷积前对特征层，以多个比例进行重采样。我们通过使用多个并行的空洞卷积来实现，称为“atrous spatial pyramid pooling”。

解决第三个挑战：使用Fully-Connected CRF。

![f1](images\f1.png)

### 3. Methods

#### 3.1 Atrous Convolution for Dense Feature Extraction and Field-of-View Evlargement

$$
y[i] = \sum^K_{k=1}x[i+r\cdot k]w[k]
$$

$\ r\ $表示采用输入信号的比率。

![f2](images\f2.png)

![f3](images\f3.png)

#### 3.2 Multiscale Image Representations Using Atrous Spatial Pyramid Pooling

DCNNs可以隐式的学习多尺度的表示，只要数据集中包含不同尺度的目标。显式的考虑尺度可以提升DCNN能力。

本文使用两种方式：

1. 使用平行的DCNN分支，从多个rescaled版本中提取score map。接着对多个平行分支的特征图使用双线性插值返回原图分辨率，并融合他们（每个位置取最大响应）。

2. 使用多个平行的atrous卷积层（不同的采样率），最后进行融合。

    ![f4](images\f4.png)

#### 3.3 Structured Prediction with Fully-Connected Conditional Random Fields for Accurate Boundary Recovery

