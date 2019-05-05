## Objects as Points

### Abstract

​	Detection识别一副图像中的boxes。几乎所有的目标检测器都是几乎穷举可能的目标位置并分类。这是浪费的，低效的并且需要额外的后处理。本文提出一个不一样的方法。我们将目标建模为一个点——bounding box的中心点。我们的检测器使用关键点检测来寻找中心点，然后回归其他目标属性，如size，3D location，orientation以及pose等。我们基于center point的方法，CenterNet，简单快速，并且比基于bounding box的检测器更准确。CenterNet在MS COCO数据集上达到了最好的速度-准确度权衡，28.1%AP/142FPS，37.4%AP/52FPS，45.1%AP/1.4FPS。

![f1](images\f1.png)

### Introduction

​	现有的目标检测器通过一个轴平行的bounding box 来表示每个目标，然后将检测任务看作是对大量可能的bounding box的分类任务。对于每个bounding box，分类器决定图像内容是目标还是背景。一阶段检测器在整幅图像上滑动可能bounding box，称为anchors，然后直接分类，而不需要指定box中的内容。两阶段检测器计算每个可能box的特征，然后对特征分类。后处理，也就是非极大值抑制，去除冗余的检测。后处理很难differentiate 以及训练，因此目前大部分的检测器不是端到端训练的。尽管如此，该想法已经达到了很好的商用价值。基于滑窗的目标检测器很浪费，因为需要穷举所有可能的位置。

​	本文提出一个更简单有效的方法。将目标用一个中心点来表示，其他属性直接从图像特征中回归。目标检测则成为了一个关键点检测问题。我们将图像送入一个fcn，生成一个heatmap。heatmap中的峰值 表示目标中心位置。每个峰值出的图像特征预测目标bounding box的height和weight。

​	我们的方法很通用，并可以很容易拓展到其他任务。我们提供了3D目标检测和多人姿态检测的实验，通过预测每个中心点额外的输出。对于3D bounding box估计，我们回归目标绝对depth，3D bounding box dimensions以及object orientation。对于姿态估计，我们将2D 关节位置看作是对于center的offset，直接从中心点位置回归它们。CenterNet运行速度很快。Resnet-18和上采样，我们的网络达到142FPS以及28.1%AP。使用关键点检测网络DLA-34，我们的网络达到37.4%AP/52FPS。使用Hourglass-104关键点检测网络，以及多尺度测试，我们的网络达到了45.1%AP/1.4FPS。

### Related Work

![f3](images\f3.png)

**Object detection by region classification.** RCNN、Fast-RCNN.

**Object detection with implicit anchors.** Faster RCNN在检测网络中生成proposal。采样固定形状的anchor，分类每个anchor为前景或背景。>0.7的overlap称为前景，<0.3的overlap称为背景。生成的proposal之后被在分类一次。一些对一阶段检测器的提升包括anchor形状先验，不同的特征分辨率以及采样的loss re-weighting。

​	中心点可以看作是一个形状任意的anchor，但也有一些不同。第一，CenterNet赋值anchor仅基于位置，而不是用box overlap。没有手工设置阈值来区分前背景。第二，每个目标只有一个anchor，因此不需要NMS。我们只是简单的提取keypoint heatmap的局部峰值。第三，CenterNet使用大的输出分辨率（stride of 4）（相比于传统的检测器，16），这消除了多种anchor的需要。

**Object detection by keypoint estimation. ** CornetNet检测bounding box的两个角作为关键点，ExtremeNet检测top-，left-，bottom-，right-most，center points。但是他们在关键点检测之后需要一个组合分组，slow了算法。

**Monocular 3D object detection. **

![f2](images\f2.png)

### Preliminary

​	设$\ I\in R^{W\times H\times 3}\ $为输入图像。我们的目标是产生关键点heatmap $\ \hat{Y}\in [0,1]^{\frac{W}{R}\times\frac{H}{R}\times C}\ $，其中$\ R\ $是输出stride，$\ C\ $是关键点type的数量。姿态检测中$\ C=17\ $，目标检测中$\ C=80\ $（类别数）。$\ R=4\ $。$\ \hat{Y}_{x,y,c}=1\ $表示检测的关键点，$\ \hat{Y}_{x,y,c}=0\ $表示背景。我们使用了不同的FCN encoder-decoder网络来预测$\ \hat{Y}\ $：hourglass network，up-convolutional ResNet，DLA。

​	对于每个类别$\ c\ $的关键点$\ p\in R^2\ $，我们计算其低分辨率等效$\ \widetilde{p}=\lfloor\frac{p}{R}\rfloor\ $。然后我们使用一个高斯核$\ Y_{xyc}=\mathrm{exp}(-\frac{(x-\widetilde{p}_x)^2+(y-\widetilde{p}_y)^2}{2\sigma^2_p})\ $将所有ground truth关键点分布到一个heatmap $\ Y\in[0,1]^{\frac{W}{R}\times\frac{H}{R}\times C}\ $，其中$\ \sigma_p\ $是目标大小自适应标准差。如果同一类的两个高斯重叠，我们取element-wise maximum。训练目标penalty-reduced逐像素逻辑回归加上focal loss：
$$
L_k = \frac{1}{N}\sum_{xyc}
\left\{
\begin{array}{**lr**}
(1-\hat{Y}_{xyc})^\alpha\log(\hat{Y}_{xyc}),& \ if\ Y_{xyc}=1\\
(1-Y_{xyc})^\beta(\hat{Y}_{xyc})^\alpha &\ \ otherwise\\
\log(1-\hat{Y}_{xyc})
\end{array}
\right.
$$
其中$\ \alpha\ $和$\ \beta\ $是focal loss的超参，$\ N\ $是图像中关键点数量。使用N归一化以将所有正focal loss instance归一化到1。$\ \alpha=2，\beta=4\ $。

​	为了恢复由于output stride的离散化误差，对于每个中心点我们额外预测一个局部offset$\ \hat{O}\in\mathcal{R}^{\frac{W}{R}\times\frac{H}{R}\times 2}\ $。所有的类别共享offset预测。offset使用L1 loss来训练。
$$
L_{off}=\frac{1}{N}\sum_p|\hat{O}_{\widetilde{p}}-(\frac{p}{R}-\widetilde{p})|
$$
监督信息只在关键点位置$\ \widetilde{p}\ $，其他位置忽略。

### Objects as Points

​	设$\ (x_1^{(k)},y_1^{(k)},x_2^{(k)},y_2^{(k)})\ $为目标$\ k\ $的bounding box，类别为$\ c_k\ $。其中心点为$\ p_k = (\frac{x_1^{(k)}+x_2^{(k)}}{2},\frac{y_1^{(k)}+y_2^{(k)}}{2})\ $。我们使用关键点检测器$\ \hat{Y}\ $来预测所有的中心点。此外，我们为每个目标回归目标大小$\ s_k = (x_2^{(k)}-x_1^{(k)},y_2^{(k)}-y^{(k)}_1)\ $。为了限制计算开销，我们对于所有类别使用单个尺寸预测$\ \hat{S}\in \mathcal{R}^{\frac{W}{R}\times\frac{H}{R}\times 2}\ $。我们使用在中心点使用L1 loss：
$$
L_{size} = \frac{1}{N}\sum^N_{k=1}|\hat{S}_{p_k}-s_k|
$$
我们不归一化尺度，直接使用原始像素坐标。我们使用一个常量$\ \lambda_{size}\ $来scale loss。整体的训练目标为：
$$
L_{det} = L_k+\lambda_{size}L_{size}+\lambda_{off}L_{off}
$$
$\ \lambda_{size}=0.1,\lambda_{off}=1\ $。我们使用一个网络来预测关键点$\ \hat{Y}\ $，偏移$\ \hat{O}\ $以及大小$\ \hat{S}\ $。网络在每个位置总共预测$C+4$个输出。所有输出共享fcn backbone。对于每个模态，特征通过一个3x3卷积，ReLU以及1x1卷积。图4展示了网络的输出。

#### From points to bounding boxes

​	在inference时，我们首先为每个类别单独提取heatmap的峰值。我们检测所有比其8邻域大的相应，然后取前100个峰值。设$\ \hat{P}_c\ $为检测的$\ n\ $个中心点集 $\ \hat{P}=\{(\hat{x}_i,\hat{y}_i)\}^n_{i=1}\ $。每个关键点位置是一个整数坐标$\ (x_i,y_i)\ $。我们使用关键点值$\ \hat{Y}_{x_iy_ic}\ $作为其检测置信度，在该位置产生bounding box：
$$
（\hat{x}_i+\delta \hat{x}_i - \hat{w}_i/2, \hat{y}_i+\delta\hat{y}_i-\hat{h}_i/2,\\
\hat{x}_i+\delta\hat{x}_i+\hat{w}_i/2,\hat{y}_i+\delta\hat{y}_i+\hat{h}_i/2)
$$
其中$\ (\delta\hat{x}_i,\delta\hat{y}_i) = \hat{O}_{\hat{x_i},\hat{y}_i}\ $是offset预测，$\ (\hat{w}_i,\hat{h}_i) = \hat{S}_{\hat{x}_i,\hat{y}_i}\ $是size预测。

#### 3D detection

#### Human pose estimation

![f4](images\f4.png)

### Implementation details

我们实验了4种结构：ResNet-18，ResNet-101，DLA-34和Hourglass-104。我们在ResNets和DLA-34中使用可变形卷积。

**Hourglass ** Hourglass 网络降采样输入4x，使用两个连续的hourglass模块。每个hourglass模块都是5层的down-和up-卷积网络以及跳跃连接。该网络比较大，但产生最好的关键点检测性能。

**ResNet**  Xiao等使用三个上卷积网络来增强标准resnet，使得可以输出高分辨率（stride 4）。我们首先将三个上采样层的通道数改为256，128，64，然后在每个上采样之前加一个3x3可变形卷积层。

**DLA**  Deep Layer Aggregation是一个有层级跳跃连接的图像分类网络。我们利用全卷积上采样版本的DLA用于密集预测。我们使用可变形卷积来增强跳跃连接。特别的，在每个上采样层，我们替换原来的卷积为可变形卷积。我们在每个输出head上加了一个3x3卷积层，最后一个1x1卷积来产生输出。

**Training** 我们在512x512上训练，产生输出分辨率为128x128。我们使用随机翻转，随机缩放，裁剪以及色彩抖动来进行数据增强，使用Adam来优化。我们不使用增强来训练3D估计分支。对于resnet和DLA-34，训练batch-size为128（8 GPUs），学习率5e-4，140eopch。在90和120epoch，learning rate 减小10x。对于Hourglass-104，我们follow ExtremeNet，使用batch-size29（5GPUs，主GPU batch-size 4），lr 2.5e-4，50epoch，40epoch减小10x。对于detection，我们从ExtremeNet来fine-tune Hourglass-104。

**Inference** 我们使用三种测试增强：no，翻转，翻转和多尺度。

### Experiments

​	我们在MS COCO数据集上评估目标检测性能，（train2017）包含118k训练数据，（val2017）包含5k验证数据以及（test-dev）包含20k保留测试数据。

![t1](images\t1.png)

![t2](images\t2.png)

![t3](images\t3.png)

### Conclusion

我们提出一种新的目标表示：点。我们的CenterNet建立在关键点估计网络上，寻找目标中心，并回归他们的大小。该算法简单，快速，准确，并且端到端differentiable，不使用NMS后处理。改想法是通用的并且应用广泛。CneterNet可以估计目标额外的属性，如pose，3D orientation，depth和extent。

