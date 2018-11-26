## RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation

> CVPR 2017

### Abstract

​	我们提出RefineNet，一个通用多路精修（refinement）网络，可以明确的利用降采样步骤中可用的信息，使用远程的残差连接来进行高分辨率预测。这样，捕获高层语义特征的深层可以直接使用前面巻积层细粒度的特征来精修。RefineNet的各个组件采用残差连接，可以有效的进行端到端训练。接着，我们提出链式残差池化，可以有效的捕获丰富的背景上下文。在PASCAL VOC2012数据集上IoU达到了83.4%。

![f1](images\f1.png)

### Introduction

​	卷积、池化通常把图像缩小32倍，因此丢失了图像结构细节。

​	一种解决方法是学习反卷积filter，作为上采样操作以产生高分辨率特征图。但是反卷积操作不能恢复降采样丢失的低层视觉特征，因此，他们不能输出准确的高分辨率预测。低层视觉信息对于边界或细节的预测至关重要。DeepLab运用了膨胀卷积，不降低图像分辨率的情况下增大感受野。但是该策略有两个限制：1）需要在很多膨胀（高分辨率）特征图（通常是高维特征）上做卷积，计算开销大；高维、高分辨率特征图需要大量显存。这限制了高分辨率预测的计算，通常输出大小被限制为原图的1/8。2）膨胀卷积是对特征的粗糙降采样，可能会丢失重要的细节。

​	另一类方法利用了中间层特征来产生高分辨率预测，如FCN、Hypercolumns。期望中间层的特征可以描述物体部件的中层表示，同时保留了空间信息。这些信息是对低层空间视觉信息（边、角、圈）和高层语义信息的补充。

​	我们认为所有层的特征对于语义分割都是有帮助的。高层特征帮助识别类别，低层特征产生细节的边界。如何利用中层特征依然是个问题。我们提出一个网络结构来利用中层特征。

主要贡献：

  1. 提出RefineNet，以循环的方式，使用细粒度低层特征精修低分辨率语义特征。
  2. RefineNet中所有组件使用了恒等映射的残差连接，梯度可以直接短程和远程传播，可以有效的端到端训练。
  3. 提出链式残差池化，捕获背景上下文信息。
  4. RefineNet在7个公共数据集上都到了state-of-the-art。

### Related Work

​	解决低分辨率预测的几种方法：1）DeepLab-CRF直接输出中-分辨率得分图，使用CRF修正边界。2）CRF-RNN将该方法拓展为端到端的。3）反卷积方法。

​	利用中间层特征的方法：1）FCN产生多个分辨率的预测，并平均起来；2）Hypercolumn融合了中间层特征。3）SegNet、UNet利用了跳跃连接。

### Proposed Method

​	图2（c）展示了一种可能的安排。

![f2](images\f2.png)

#### Multi-Path Refinement

​	在我们标准的多路结构中，我们将预训练的ResNet分为四个block，使用4个级联的RefineNet 单元结构，每个单元都直接连接ResNet block的输出以及前一个RefineNet block。注意到，这种设计不是唯一的。如，一个RefineNet block可以接受多个ResNet block的输入。我们分析 2级联的版本，单block方法以及一个2尺度7路结构。

​	每个ResNet输出都经过一个巻积层来调整维度。尽管所有RefineNet的结构相同，他们的参数不绑定。RefineNet-4只有一个输入，作为额外的卷积集来调整ResNet的weight到语义分割任务上。RefineNet-3有2路输入。RefineNet-3的目的是使用ResNet block-3的高分辨率特征来修正RefineNet-4的低分辨率特征。相似的，RefineNet-2和RefineNet-1通过融合高层信息和高分辨率-低层信息来重复这种逐阶段的修正。最终，高分辨率特征图送入softmax层，产生得分图，得分图使用双线性插值上采样到原图大小。

​	很重要的一点是我们在ResNet 和RefineNet的block之间引入了远程残差连接。在前传时，这些残差连接可以传送低层特征。在训练时，残差连接可以使梯度直接传播到前面的巻积层，帮助端到端训练。

#### RefineNet

​	RefineNet block如图3（a）所示。

![f3](images\f3.png)

**Residual convolution unit** 每个RefineNet block前面包含一些调整卷积，来fine-tune预训练ResNet。每个输入路径都经过两个残差卷积单元（RCU）。

**Multi-resolution fusion** 所有路径的输入接着通过multi-resolution fusion block融合为一个高分辨率特征图。该block首先是两个卷积用于调整输入，接着上采样到输入最大的分辨率。最后，所有特征图求和融合。

**Chained residual pooling** 输出特征图通过链式残差池化block，如图3（d）所示。该block是为了捕获背景上下文。使用多个窗口尺寸来池化特征并使用可学习的weight来将其融合。特别的，该模块是多个池化block，每个block包含一个最大池化层和一个巻积层。所有池化block输出特征图与输入特征图融合到一起。在一个池化block中，每个池化操作后面都有卷积操作，作为加和融合的权重层。

**Output convolutions** RefineNet block最后一步是另一个RCU。每个block间有三个RCU。在最后softmax预测之前也有两个RCU。目的是对多路融合特征图进行非线性操作。

#### Identity Mappings in RefineNet

​	Short-range残差连接是一个RCU或残差池化组件中的捷径连接，long-range残差连接是RefineNet模块和ResNet block之间的连接。