## Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks

### Abstract

​	在无约束条件下的人脸检测（detection）和对齐（alignment）具有挑战性，因为姿势多变，照明变化以及遮挡。最近的研究显示深度学习在这两个任务上可以达到很好的效果。本文中，我们提出一个深度级联多任务框架，利用detection和alignment之间的固有关联来提升性能。特别的，我们的框架利用了级联结构，有三个阶段（从粗到细的方式）来预测脸和landmark位置。此外，我们提出了一种新的在线苦难样本挖掘策略，进一步提高了性能。我们的方法在FDDB，WIDER FACE，AFLW三个数据集上超过了state-of-the-art，并且达到实时。

### Introduction

​	【2】等提出的人脸检测器利用了Haar-Like特征和AdaBoost来训练一个级联分类器，达到了很好的性能。但是，一些研究认为这类检测器在巨大变形时性能会下降。此外，【5，6，7】提出DPM来检测人类。但是，这种方法计算开销大，需要大量标注。【11】训练了一个CNN用于面部属性识别来在面部区域获得高响应，从而产生人脸候选框。但是，该方法结构复杂，开销大。【19】等使用了级联CNN来人脸检测，但是需要额外开销进行bb矫正，并且忽略了landmarks定位以及bb回归的固有关系。

​	Face alignment研究主要分为两类，基于回归的方法以及模板过滤方法。【22】提出使用脸部属性识别作为辅助任务，以提高性能。

​	但是，大部分研究忽略了detection和alignment两个任务的固有关系。尽管有一些研究尝试联合解决他们，但还是有限制。如【18】联合进行alignment和detection，使用随机森林和像素值差异特征。但是，手工特征限制了性能。【20】使用多任务CNN来提高多视角人类检测，但是初始检测窗是使用弱检测器。

​	另一方面，困难样本挖掘很重要。但是，传统方法是离线方式，增加了手动操作。

​	本文提出一个级联CNN多任务学习框架。包含三个阶段，第一阶段，使用浅CNN产生候选窗。第二阶段，使用一个更复杂CNN来过滤掉非面部窗口。第三阶段，使用更强的CNN来精修结果。

### Approach

![f1](images\f1.png)

​	如图1，给定一幅图，我们首先resize到不同尺度，构建图像金字塔作为级联网络的输入：

**Stage 1:**  利用一个全卷积网络，Proposal Network(P-Net)，获取候选面部窗以及其bb回归向量。接着使用bb回归向量来矫正候选窗。接着使用NMS来合并高度重叠的候选框。

**Stage 2:**  所有候选窗送入另一个CNN，RefineNetwork(R-Net)，进一步过滤掉错误候选，然后使用bb回归和NMS。

**Stage 3:**  与第二阶段累死，最终输出五个landmark位置。

![t1](images\t1.png)

![f2](images\f2.png)

​	训练阶段结果如表1.网络结构如图2所示。我们利用了三个任务来训练CNN检测器：分类，bb回归和面部landmark定位。

1）分类：对于每个样本，使用交叉熵损失：
$$
L^{det}_{i} = -(y^{det}_ilog(p_i)+(1-y^{det}_i)(1-log(p_i)))
$$
其中$\ p_i\ $表示网络输出的概率。$\ y^{det}_i\in\{0,1\}\ $表示ground truth。

2）bb回归：对于每个候选窗，计算与最近ground truth之间的offset。对于每个样本使用欧几里得损失：
$$
L^{box}_i = \parallel \hat{y}^{box}_i-y^{box}_i\parallel^2_2
$$
其中$\ \hat{y}^{box}_i\ $表示网络获取的结果，$\ y^{box}_i\ $为ground-truth。包括四个值：left top height width。

3）面部landmark定位：与bb回归类似：
$$
L^{landmark}_i = \parallel \hat{y}^{landmark}_i - y^{landmark}_i\parallel^2_2
$$
有五个面部landmark，包括左眼，有眼，鼻子，左嘴角，右嘴角。

4）Multi-source training：因为每个CNN使用了不同的任务，因此在训练时有不同的训练图像类型。因此，有些损失函数是不使用的，如背景区域只计算$\ L^{det}_i\ $，其他损失为0。整体学习目标为：
$$
min\sum^N_{i=1}\sum_{j\in det,box,landmark}\alpha_j\beta_i^jL^j_i
$$
其中，N为训练样本数，$\ \alpha_j\ $表示任务重要性。（P-Net和R-Net中$\ \alpha_{det} = 1,\alpha_{box}=0.5,\alpha_{landmark}=0.5\ $，O-Net中$\alpha_{det} = 1,\alpha_{box}=0.5,\alpha_{landmark}=1\ $） .$\ \beta^j_i\in\{0,1\}\ $是样本类型指示器。

5）在线困难样本挖掘：在每个mini-batch，我们对损失排序，并选择前70%作为困难样本。接着在回传时只计算这些样本的梯度。即，我们忽略简单样本，他们对于增强检测器没什么用。

### Experiments

​	FDDB包括在包括了2845张图像的5171张脸部标注。WIDER FACE数据集包括32203张图像中393707个标注bb，其中50%用于测试（根据图像难度分为了三个子集），40%用于训练，剩下的用于验证。AFLW包括24386张面部的landmark标注，我们使用与【22】相同的测试子集。

*A. Training Data*

​	因为我们同时执行detection和alignment，因此我们使用四种不同类型的标注数据：1）Negatives：与任何ground-truth IoU小于0.3的区域；2）Positives：与某个ground-truth IoU高于0.65的；3）Part faces：与某个ground-truth IoU在0.4和0.65之间的；4）Landmark faces：标记了5个区域的脸。Negatives和Positives用于训练分类，Positives和Part faces用于bb回归。整个训练集比例是3:1:1:2。

1）P-Net：我们随机裁剪一些WIDER FACE中的一些块来手机正样本，负样本，部分样本。接着从CelebA中裁剪脸作为landmark faces

2）R-Net：使用第一阶段来检测WIDER FACE，以获得正样本、负样本和部分样本，landmark faces是从CelebA中检测得到。

3）O-Net：与R-Net类似。

*B. The effectiveness of online hard sample mining*

![f3](images\f3.png)

​	图3(a)展示了两个不同P-Net的结果。可以看出在线困难样本挖掘可以提升1.5%的性能。

*C. The effectiveness of joint detection and alignment*

​	图3(b)显示联合任务提高了分类和bb回归多任务的性能。

*D. Evaluation on face detection*

![f4](images\f4.png)

​	图4(a)-(d)显示了我们的方法比state-of-the-art方法好很多。

*E. Evaluation on face alignment*

![f5](images\f5.png)

​	图5展示了我们的方法比state-of-the-art方法要好。并且在嘴角定位没那么好，可能是因为表情变化小。

*F. Runtime efficiency*

![t2](images\t2.png)

​	如表2。