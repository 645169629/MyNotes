## Fast R-CNN

> 2015

### Abstract

​	Fast R-CNN基于之前的研究，高效的分类proposal。比R-CNN训练快9倍，测试快213倍，mAP更高。

### Introduction

​	目标检测两大挑战：1）要处理大量的proposal；2）候选框只有大致定位，需要refine。本文提出单阶段训练算法，联合学习分类proposal和refine空间位置。

#### 1.1 R-CNN and SPPnet

R-CNN三大缺点：

​	1）**Training is a multi-stage pipeline. ** R-CNN先使用log loss fine-tune。接着使用Conv特征训练SVMs。这些SVM作为目标检测		器，替代fine-tune学到的softmax分类。第三个阶段学习bb回归。

​	2）**Training is expensive in space and time. ** 对于SVM和bb回归训练，特征从每个proposal中提取并写到磁盘上。

​	3）**Object detection is slow. ** 测试时，每个测试图像，从每个proposal中提取特征。

​	R-CNN太慢是因为每个proposal都经过计算，没有共享计算。空间金字塔池化网络（SPPnets）共享了计算。SPPnet对整幅图像计算特征图，然后从共享特征图上提取特征向量。对proposal中的部分特征图，使用max-pooling，提取固定大小特征（如6x6）。pool多个输出大小，接着concat。

​	SPPnet的训练也是多阶段pipeline，包括提取特征，fine-tune网络，训练SVM，训练bb回归器，特征也被写到硬盘。但是与R-CNN不同的是，fine-tuning算法不能在空间金字塔池化之前更新巻积层，折现值了网络的accuracy。

#### 1.2 Controbutions

提出新的训练算法改进R-CNN和SPPnet。Fast R-CNN几个优点：

​	1）更高的mAP。

​	2）单阶段训练，使用多任务loss。

​	3）训练可以更新所有网络层。

​	4）不需要硬盘存储。

### Fast R-CNN architecture and training

![f1](images\f1.png)

图1展示了Fast R-CNN的结构。Fast R-CNN接收整幅图像和proposal作为输入。接着先对整幅图做conv和max pooling，生成特征图。接着对于每个proposal，RoI pooling层提取固定长度特征向量。每个特征向量送到一系列fc层，接着分为两个分支：一个产生K+1个类别的softmax概率，一个对K个类别，产生4个值。每4个值编码了精修的K个类别之一的bb位置。

#### 2.1 The RoI pooling layer

​	RoI pooling使用max pooling将感兴趣区域中的特征转换为固定大小$\ H\times W\ $,其中$\ H,W\ $是层超参，与RoI无关。每个RoI为一个特征图上的矩形窗，$(r,c,h,w)$，表示左上角和宽高。将$\ h\times w\ $的RoI窗转换成$\ H\times W\ $的网格，每个子窗口大小约为$\ h/H\times w/W\ $,接着使用max-pooling，获取对应格子的相应。

#### 2.2 Initializing from pre-trained networks

使用预训练模型初始化Fast R-CNN，经过三个变化：

​	1）最后一个max pooling转换为RoI pooling层，设置H和W来兼容网络第一个全连接层（如，VGG16，H=W=7）。

​	2）网络最后的fc层和softmax层替换为两个分支（分类&回归）。

​	3）修改网络，接受两个输入：图像list和RoI list。

#### 2.3 Fine-tuning for detection

​	SPP 层每个训练样本（即RoI）来自于不同的图像，回传十分低效。这种低效是因为每个RoI可能有很大的感受野，通常超过输入图像。因为前传必须处理全部感受野，训练输入很大（通常是整幅图）。

​	我们提出一个更有效的训练方法，利用了特征共享。在训练Fast R-CNN时，mini-batch分层采样，首先采样N个图像，接着每个图像采样R/N个RoI。同一图像的RoI共享计算。使N很小，减少了mini-batch的计算。例如，N=2，R=128，这种训练方式比在128张图上采样一个RoI快64倍。

​	除了分层采样，Fast R-CNN训练时同时优化softmax分类器和bb回归器。

**Multi-task loss. ** Fast R-CNN有两个分支输出层。第一个分支输出每个RoI的离散概率$\ p=(p_0,...,p_K)\ $（K+1个类）。K+1个fc层输出经过softmax得到p。第二个分支输出bb回归偏移$\  t^k=(t^k_x,t^k_y,t^k_w,t^k_h)\ $。其中$\ t^k\ $表示尺度不变平移和log空间h/w转换。

每个RoI都标记了gt类别$\ u\ $以及gt bb回归目标$\ v\ $。multi-task loss L定义为：
$$
L(p,u,t^u,v) = L_{cls}(p,u)+\lambda [u\ge 1]L_{loc}(t^u,v)
$$
其中$\ L_{cls}(p,u) = -\log p_u\ $。对于bb回归，target $\ v=(v_x,v_y,v_w,v_h)\ $，预测$\ t^u=(t^u_x,t^u_y,t^u_w,t^u_h)\ $：
$$
L_{loc}(t^u,v)=\sum_{i\in \{x,y,w,h\}}smooth_{L_1}(t^u_i-v_i)
$$
其中：
$$
smooth_{L_1}(x)=
\begin{cases}
0.5x^2&  \text{if|x|<1}\\
|x|-0.5&  \text{otherwise}
\end{cases}
$$
归一化回归目标$\ v\ $到0均值，统一方差。

**Mini-batch sampling. ** 在fine-tuning时，每个SGD mini-batch包含N=2幅图像。mini-batsh-size=128，每张图取64个RoI。我们取25%与gt IoU大于等于0.5的proposal。这些RoI标记为目标类别，其他RoI采样proposal与gt IoU在[0.1,0.5)的。这些RoI标位背景。训练时，图像水平翻转，概率为0.5，没有其他数据增广。

**Back-propagation through RoI pooling layers. ** 设$\ x_i\in\mathbb{R}\ ​$为第i个激活输入到RoI pooling层，$\ y_{rj}\ ​$为第r个RoI的第j个输出。RoI pooling计算$\ y_{rj}=x_{i*(r,j)}\ ​$，其中$\ i^*(r,j)=argmax_{i'\in R(r,j)}x_{i'}\ ​$。$\ R(r,j)\ ​$为输入在子窗口中的索引集。

​	RoI pooling层回传计算损失对每个$\ x_i\ $的偏导
$$
\frac{\partial L}{\partial x_i}=\sum_r\sum_j[i=i*(r,j)]\frac{\partial L}{\partial y_{rj}}
$$
对于每个mini-batch RoI r和每个pooling输出单元$\ y_{rj}\ $，$\ \partial L/\partial y_{rj}\ $叠加起来。回传中$\ \partial L/\partial y_{rj}\ $已经在上一层计算好了。

**SGD hyper-parameters. ** 用于softmax分类和bb回归的fc层使用零均值、标注出0.01/0.001的高斯分布初始化。Biases初始化为0。每层学习率：weight 1，biases 2，全局学习率0.001.对于VOC07或VOC12 trainval 30K mini-batch迭代，接着学习率下降到0.0001用于剩下10K迭代。momentum 0.9，decay 0.0005。

#### 2.4 Scale invariance

### Fast R-CNN detection

​	网络输入list 图像，以及R个proposal 的list。测试时，R约为2000。

#### 3.1 Truncated SVD for faster detection

​	检测大部分时间在计算fc层，可以使用truncated SVD来加速。

权值矩阵W（$\ u\times v\ $ ）可以使用SVD近似为:
$$
W\approx U\sum_tV^T
$$
其中，U是$\ u\times t\ $的矩阵，包含了W前t个左奇异值向量，$\sum_t$是$\ t\times t\ $的对角矩阵，包含W的前t个奇异值，$\ V\ $是$\ v\times t\ $的矩阵，包含W前t个右奇异值矩阵。Truncated SVD将参数从$\ uv\ $减少到了$\ t(u+v)\ $。为了压缩网络，一个fc层（W）替换为两个fc层。第一个使用权重矩阵$\ \sum_t V^T\ $（没有bias），第二个使用U（使用原来W相关的bias）。