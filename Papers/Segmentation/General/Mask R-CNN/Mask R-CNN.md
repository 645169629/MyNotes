## Mask R-CNN

> ICCV 2017

### Abstract

​	本文提出一个简单，灵活，通用的实例分割框架。我们的方法检测图像中的目标，同时为每个实例生成分割mask。本方法称为Mask R-CNN，在Faster R-CNN基础上增加了一个分支用于预测目标mask（与bb recognition分支平行）。Mask R-CNN很容易训练并且只增加了小部分开销，速度为5fps。而且，Mask R-CNN很容易拓展到其他任务，如姿势估计。

![f1](images\f1.png)

### Introduction

​	实例分割因为需要检测所有目标，同时分割每个实例而具有挑战性。需要结合目标检测和语义分割。

​	我们的方法在Faster R-CNN上增加了一个分支，预测每个RoI的分割mask，与现有的分类和bb regression并行。如图1所示。mask分支是一个小FCN，应用于每个RoI。

​	Faster R-CNN不是输入和输出之间像素的对齐。这在RoIPool（关键操作）如何关注各个实例，执行粗的空间量化用于特征提取可以看出。为了解决这种不对齐，我们提出一个简单，不需要量化的层，称为RoIAlign，可以保留准确的空间位置。尽管看起来像是镜像改变，RoIAlign有很大影响：1）提高了mask准确度10%到50%。2）我们发现解耦mask以及类别预测很重要：我们每个类预测一个二维mask，依靠网络的RoI分类分支类预测类别。FCNs通常执行逐像素多类别分类，同时进行分割和分类，这对于实例分割是不好的。

​	将keypoint作为one-hot 二分mask，只需要很小的改动Mask R-CNN就可以应用于检测实例姿势。

![f2](images\f2.png)

### Related Work

**R-CNN: **基于区域的CNN（R-CNN）方法致力于管理许多候选框，并单独在每个RoI上运行CNN。R-CNN被拓展，可以使用RoIPool关注特征图上的RoI，使得速度更快也更准确。Faster R-CNN通过RPN学习注意力机制进一步提升。

**Instance Segmentation: ** 由于R-CNN很有效，很多实例分割方法都是基于segment proposal。DeepMask以及之后的研究学习分割候选，接着由Fast R-CNN分类。在这些方法中，分割在识别之前，速度慢，不准确。同样的，【7】提出一个多阶段级联方法，从bb proposal中预测segment proposal，接着进行分类。但是我们的方法是基于平行预测mask和类别。

​	最近，【21】结合了segment proposal系统与目标检测系统，用于全卷积实例分割（FCIS）。共同的思想是以全卷积预测位置敏感的输出通道。这些通道同时解决目标类别，框以及mask，使得系统很快。但是FCIS在重叠实例上有误差，会产生假的边界（如图5所示）。

### Mask R-CNN

​	Mask R-CNN在概念上很简单：Faster R-CNN对于每个候选目标有两个输出，一个类别标签，一个bb offset；因此我们增加了第三个分支用于输出目标mask。

**Faster R-CNN: ** Faster R-CNN包含了两个阶段。第一个阶段称为Region Proposal Network（RPN），生成候选目标框。第二个阶段，使用RoIPool对每个候选框提取特征，并且进行分类和bb回归。两个阶段使用的特征可以共享。

**Mask R-CNN: ** Mask R-CNN也是两阶段，第一阶段同样为RPN。第二阶段，与预测类别和bb offset，Mask R-CNN也输出每个RoI的二分类mask。这与大部分最近的系统是相反的，它们是的分类依赖于mask预测。

​	训练时，我们定义每个采样的RoI的多任务损失为：$\ L = L_{cls}+L_{box}+L_{mask}\ $。分类损失$\ L_{cls}\ $和bb损失$\ L_{box}\ $与【9】中定义的一致。mask分支对于每个RoI有一个$\ Km^2\ $维输出，编码了$\ K\ $个二分类mask，大小为$\ m\times m\ $。我们采用了逐像素sigmoid，并且定义$\ L_{mask}\ $为平均二分类交叉熵损失。对于一个RoI，其对应ground-truth为类别k，$\ L_{mask}\ $只在第k个mask上定义。

​	$\ L_{mask}\ $的定义使得网络可以每个类别的mask，而不需要在类别之间进行比较；我们使用专用的分类分支来预测列表，已选择出输出mask。这解耦了mask和类别预测。这与FCNs语义分割不同。

**Mask Representation: ** mask编码了输入目标的空间布局。因此，与类别标签和box offset不同的是，不需要通过全连接层将输出压缩到很小的输出向量，可以直接通过卷积提供的像素到像素的相应来提取空间结构。

​	特别的，我们使用FCN为每个RoI预测一个$\ m\times m\ $的mask。这使得mask分支的每一层都保持$\ m\times m\ $的目标空间布局，而不需要将其压缩成向量表示而减少空间维度。

​	这种像素到像素的方法需要我们的RoI特征很好的对齐，以保证逐像素空间响应。

**RoIAlign: ** RoIPool用于从每个RoI提取一个小特征（如7x7）。RoIPool首先将floating数值RoI量化到特征图的离散粒度，接着细分为空间bins，最后聚合（max pooling）每个bin覆盖的特征值。量化是，在一个连续坐标$\ x\ $上计算$\ [x/16]\ $，其中16是特征图stride，[·]是取整；同样的，在分成bins时（如7x7）也执行了量化。这些量化会引入RoI与提取特征的不对齐，虽然不会影响分类，但是会影响像素级别的mask。

​	我们提出RoIAlign层来解决该问题，移除了RoIPool严格的量化，适当的对于输入和提取特征。我们提出的改变很简单：我们避免RoI或bins边界的量化（即，使用x/16而不是[x/16]）。使用双线性插值来计算每个RoI bin四个采样位置的输入特征，并集成结果（max 或 average）。

**Network Architecture: ** 对于backbone，我们使用了ResNet-50、ResNet-101以及ResNeXt-50、ResNeXt-101，还有ResNet-FPN。

对于网络head，如图3。ResNet-C4 backbone的head包括了ResNet的第五阶段（res5）。对于FPN，backbone已经包括了res5，因此，可以使用更少filter的head。

![f3](images\f3.png)

#### Implementation Details

​	使用Fast/Faster R-CNN的超参数。

**Training: **与Fast R-CNN一样，与gt IoU大于0.5的为pos，否则为neg。$\ L_{mask}\ $只在pos RoI上定义。

​	图像resize到800。每个mini-batch，每个GPU包含2张图，每张图包含N个采样RoI，pos:neg = 1:3。C4backbone N = 64，FPN N=512。8个GPU，160K迭代，lr=0.02，在120K次迭代减小十倍。weight decay 0.0001，momentum 0.9。

​	RPN anchors 包括5个尺度，3个比例。RPN单独训练，不与Mask R-CNN共享特征。RPN和Mask R-CNN backbone一样，可共享。

**Inference: ** 测试时，proposal为  C4:300 ，FPN：1000。我们在这些proposal上跑box 预测分支，接着NMS，mask分支接着被应用于最高得分的100个检测框上。mask branch对每个RoI预测K个mask，但我们只是用第k个，k由分类分支确定。$\ m\times m\ $的mask接着被resize到RoI的大小，阈值为0.5分类。

### Experiments

​	我们使用COCO数据集和评估标准AP（超过IoU的平均值），$AP_{50},AP_{75},AP_{S},AP_{M},AP_{L}$。图2和图4为Mask R-CNN输出。图5比较了Mask R-CNN和FCIS+++。

**Ablation Experiments**	

​	如表2所示。

![t1](images\t1.png)

![t2](images\t2.png)

![t3](images\t3.png)

![t4](images\t4.png)

![t5](images\t5.png)

![t6](images\t6.png)

![f4](images\f4.png)

![f5](images\f5.png)

![f6](images\f6.png)