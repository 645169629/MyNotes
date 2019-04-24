## Bottom-up Object Detection by Grouping Extreme and Center Points

### Abstract

​	sota方法穷举目标位置并分类。本文中，我们显示 bottom-up方法表现很好。我们使用标准关键点检测网络来检测边界点（top-most，left-most，bottom-most，right-most）以及一个center 点。我们group五个关键点（如果他们几何对齐）到一个bounding box中。Object detection则成了基于外观的关键点估计问题，不需要区域分类或隐式的特征学习。我们提出的方法在COCO test-dev上达到了43.2%。此外，我们的extreme points直接形成一个粗八角形mask，在COCO Mask AP上达到了18.9%，比原始的bounding box的Mask AP好很多。Extreme point 指导的分割进一步提升到了34.6的Mask AP。

![f1](images\f1.png)

### Introduction

​	Top-down方法主导 了目标检测很多年。流行的检测器将目标检测问题转换为矩形区域分类问题，要么是明确的裁剪区域或区域特征（两阶段方法）或隐式的设置固定大小的anchors来代理区域（一阶段方法）。但是top-down检测不是没有限制。矩形bounding box不是自然的目标表示。许多物体不是轴平行的box，将它们放入一个box中会引入大量的背景像素。此外，top-down方法穷举大量的box位置而没有真正理解视觉语法的组成，这样计算开销很大。最后，box是不好的物体表示，他们传达了很少的物体细节信息。

​	本文中，我们提出一个bottom-up检测框架，ExtremeNet。我们使用sota 关键点估计框架来寻找extreme点，通过预测每个物体类别的四个多峰heatmap。此外，我们每个类别使用一个heatmap来预测物体center。我们使用几何的方法将extreme points分组成物体。只有四个点的几何中心在中心heatmap中以高于预定义阈值的值被预测时，才group 四个extreme 点。我们迭代所有$\ O(n^2)\ $个extreme point 预测组合，并选择可用的。因为预测点$\ n\ $很小，对于COCO $\ n\le40\ $，所以暴力算法已经足够。

​	我们不是第一个使用关键点来目标检测的。CornerNet预测bounding box的两个相对的角，使用关联嵌入性特征来group corner点。我们的方法与其有两个方面不同：关键点定义和分组。corner常常在物体之外，而没有很强的外观特征。而Extreme points，在物体上，具有一致的局部外观特征。第二个不同是CornerNet是几何分组。我们的检测框架是完全基于外观，没有隐式特征学习的。

​	我们的方法比所有的一阶段方法都要好，与两阶段方法达到同等水平。

![f2](images\f2.png)

### Related Work

**Two-stage object detectors ** 

**One-stage object detector **

**Deformable Part Model **

**Grouping in bottom-up human pose estimation ** 决定哪些关键点属于同一个人是bottom-up多人姿态估计的关键。有多种方案：Newell等提出为每个关键点学习一个联合特征，使用集成loss 来训练。Cao等学习一个affinity field，集成了连接的关键点之间的边界。Papandreous等学习人类骨架树上父节点的位移，每个关键点一个2维特征。Nie等学习关于物体中心的偏移特征。

​	不同于上述方法，我们的center grouping是完全基于外观的方法，利用了extreme points和他们center的几何结构。

**Implicit keypoint detection ** 

![f3](images\f3.png)

### Preliminaries

**Extreme and center points ** 设$\ (x^{(tl)},y^{(tl)},x^{(br)},y^{(br)})\ $表示bounding box。点击左上角和右上角形成框不准确。Papadopoulos等提出通过点击extreme points来标注bounding box。$\ (x^{(t)},y^{(t)}),(x^{(l)},y^{(l)}),(x^{(b)},y^{(b)}),(x^{(r)},y^{(r)})\ $，以及center point$\ (\frac{x^{(l)}+x^{(r)}}{2},\frac{y^{(t)}+y^{(b)}}{2})\ $。

**Keypoint detection ** 关键点估计通常使用全卷积encoder-decoder网络来预测多通道heatmap用于多个Keypoint类型。网络在高斯图上使用L2 loss或逐像素logistic regression loss。SOTA关键点估计网络，104-layer HourglassNetwork，以全卷积方式训练。他们每个输出通道回归一个heatmap $\ \hat{Y}\in(0,1)^{H\times W}\ $。训练由多峰高斯heatmap$\ Y\in(0,1)^{(H\times W)}\ $指导，其中每个关键点定义了高斯核的均值。标准差要么是固定的要么设置成物体大小的比例向。高斯heatmap在L2 loss中作为回归目标或在logistic regression中作为weight map来减少positive 位置的惩罚。

**CornerNet**  CornerNet预测两组heatmaps用于box的两个角。为了平衡正负位置，它们使用修改的focal loss。

![g1](images\g1.png)

其中$\ \alpha=2\ $和$\ \beta=4\ $为超参。$\ N\ $是图像中物体的数量。

​	为了像素的准确性，CornerNet额外的为每个corner回归关键点的offset$\ \Delta^{(a)}\ $。该回归恢复了部分在下采样中丢失的信息。offsetmap使用smooth L1 loss来训练：
$$
L_{off} = \frac{1}{N}\sum^{N}_{k=1}SL_1(\Delta^{(a)},\vec{x}/s-\lfloor\vec{x}/s\rfloor)
$$
其中s是下采样factor，$\ \vec{x}\ $是keypoint坐标。

​	CornerNet接着使用相关嵌入groups对角到检测结果。

### ExtremeNet for Object detection

​	ExtremeNet使用HourglassNetwork，每个类别检测五个关键点。我们follow CornerNet的训练设置，loss和offset预测。offset预测类别无关，但与extreme-point有关。我们网络的输出为$\ 5\times C\ $heatmaps和$\ 4\times 2\ $offset maps，其中$\ C\ $是类别数。

#### Center Grouping

​	我们grouping 算法的输入是5个heatmaps per class：一个center heatmap$\ \hat{Y}^{(c)}\in(0,1)^{H\times W}\ $，四个extreme heatmaps$\ \hat{Y}^{(t)},\hat{Y}^{(l)},\hat{Y}^{(b)},\hat{Y}^{(r)}\in(0,1)^{H\times W}\ $。给定一个heatmap，我们通过检测所有的峰值来提取对应的关键点。peak是高于$\ \tau_p\ $的像素位置。

​	给定从heatmaps中提取的四个extreme points $\ t,b,r,l\ $，我们计算它们的集合中心$\ c = (\frac{l_x+t_x}{2},\frac{t_y+b_y}{2})\ $。如果该中心在center map$\ \hat{Y}^{(c)}\ $中具有高响应，我们认为extreme points为可用的检测：$\ \hat{Y}^{(c)}_{c_x,c_y}\ge\tau_c\ $。我们接着迭代所有四元组$\ t,b,r,l\ $。我们每个类别单独的提取检测结果。算法1总结了这一过程。我们设$\ \tau_p=0.1,\tau_c=0.1\ $。

![a1](images\a1.png)

#### Ghost box suppression

​	center grouping可能给三个相同大小的等距共线物体产生高得分虚警检测。中心物体有两个选择，选择正确的小框，或预测更大的包含extreme points在其邻域的大框。我们称这些虚警检测为“ghost”。

​	我们提出一种简单的后处理步骤来移除“ghost”。通过定义，ghost框包含许多更小的检测。我们使用一种soft NMS。如果一个框中包含的所有可的得分超过3倍其得分，则将其得分除以2。

#### Edge aggregation

​	Extreme 点有时不是唯一确定的。如果垂直或水平的边形成了extreme点，该边上任一点都可能被当作为extreme point。因此，我们的网络沿着边产生弱响应，而不是单一强响应。这种弱相应有两个问题：弱响应可能会低于peak选择阈值而错过extreme 点；尽管检测到了关键点，其得分也会降低。

​	我们使用edge aggregation来解决该问题。对于每个extreme point，作为局部最大值提取，我们聚合其得分要么以垂直方向，（用于left和right），要么以水平方向（用于top和bottom）。我们聚合所有单调递减的得分，然后沿着聚合方向在局部最小值处停止。特别的，设$\ m\ $为一个extreme 点，$\ N^{(m)}_i=\hat{Y}_{m_x+i,m_y}\ $为在该点的垂直或水平线段。设$\ i_0<0,0<i_1\ $为两个最接近的局部最小$\ N^{(m)}_{i_{0}-1}>N^{(m)}_{i_0}\ $和$\ N^{(m)}_{i_1}<N^{(m)}_{i_1+1}\ $。边聚合更新关键点得分为$\ \widetilde{Y}_m=\hat{Y}_m+\lambda_{aggr}\sum^{i_1}_{i=i_0}N^{(m)}_i\ $，其中$\ \lambda_{aggr}\ $为聚合权重，在我们的实验中设置为0.1。

#### Extreme Instance Segmentaion

​	Extreme 点包含关于物体更多的信息。我们提出一个简单的方法来近似物体mask，通过使用八边形，其变的中心点为extreme 点。具体一点，对于一个extreme 点，我们在它对应的边的两个方向上把它延伸到整个边长度的1/4。当遇到corner时阶段分割。接着连接四个分割的端点形成八边形。

​	我们使用Deep Exctreme Cut，将提供的extreme 点转换为实例分割mask。

### Experiments

![t1](images\t1.png)

![t2](images\t2.png)

![t3](images\t3.png)

