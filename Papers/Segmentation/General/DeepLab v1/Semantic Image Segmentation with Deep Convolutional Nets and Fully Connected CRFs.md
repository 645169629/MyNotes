## Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs

>

### Abstract

	深度卷积神经网络最后一层的响应定位不足，不能进行准确的分割。这是由于DCNNs的高度不变性的特点。为了克服深度网络定位不准的特点，本文将最后一层响应与全连接条件随机场相结合。DeepLab在VOC2012测试集上达到IOU71.6%。

### 1. Introduction

	虽然深度网络的不变性被high-level视觉任务所需要，但它却会阻碍low-level任务，如语义分割（我们需要准确的定位，而不是空间细节的抽象信息）。
	
	有两点阻碍了DCNNs应用于image labeling任务：`降采样`，`空间不变性`。解决第一个问题：atrous算法。解决第二个问题：全连接CRF。

DeepLab的三点优势：1. 速度快；2. 准确度高；3. 简洁。

### 3. Convolutional Neural Networks for Dense Image Labeling

#### 3.1 Efficient Dense Sliding Window Feature Extraction with the hole Algorithm

![f1](images\f1.png)

#### 3.2 Controlling the Receptive Field Size and Accelerating Dense Computation with Convolutional Nets

	降采样第一个FC 层到 4x4（或3x3）。

### 4. Detailed Boundary Recovery:Fully-Connected Conditional Random Fields and Multi-Scale Prediction

#### 4.1 Deep Convolutional Networks and the Localization Challenge

![f2](images\f2.png)

#### 4.2 Fully-Connected Conditional Random Fields for Accurate Localization

能量函数:
$$
E(x) = \sum_i\theta(x_i)+\sum_{ij}\theta_{ij}(x_i,x_j)
$$
其中$\ x\ $为每个像素赋的值。$\ \theta_i(x_i)=-log\ P(x_i)\ $，其中$\ P(x_i)\ $为DCNN计算的像素$\ i\ $的赋值概率。$\theta_{ij}(x_i,x_j)=\mu(x_i,x_j)\sum^K_{m=1}w_m\cdot k^m(f_i,f_j)$其中$\ \mu(x_i,x_j)=1\ $如果$\ x_i\ne x_j\ $，其余为0。每个$\ k^m\ $都是高斯核，基于像素$\ i\ $和像素$\ j\ $提取的特征（$\ f\ $），由$\ w_m\ $加权。特殊的，核为：
$$
w_1\ exp(-\frac{\parallel p_i-p_j\parallel^2}{2\sigma^2_{\alpha}}-\frac{\parallel I_i-I_j\parallel^2}{2\sigma^2_{\alpha}})+w_2\ exp(-\frac{\parallel p_i-p_j\parallel^2}{2\sigma^2_{\gamma}})
$$
其中第一个核取决于像素位置和像素颜色强度，第二个核只取决于像素位置。$\sigma_\alpha$，$\sigma_\beta$，$\sigma_\gamma$控制高斯核的尺度。

#### 4.3 MultiScale Prediction

![f3](images\f3.png)