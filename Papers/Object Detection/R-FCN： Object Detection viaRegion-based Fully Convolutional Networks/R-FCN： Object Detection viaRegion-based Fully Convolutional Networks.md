### R-FCN:Object Detection via Region-based Fully Convolutional Networks

#### Abstract

我们提出一种基于区域，全卷积网络用于准确有效的物体检测。与之前基于区域，应用一个开销大的每区域子网络上百次的检测器（如，Fast/Faster R-CNN）相比，我们基于区域的检测器是全卷积的，几乎所有计算在整幅图像是共享的。为了达到该目的，我们提出位置敏感的得分图来解决图像分类中平移不变型与物体检测中平移不变性的两难境地。因此我们的方法可以自然的采用全卷积图像分类器主干，如最新的残差网络（ResNets），用于目标检测。我们使用101层的ResNet，在PASCAL VOC数据集上获得了竞争力的结果（如，2007集上83.6%的mAP）。同时，我们的结果达到了测试速度170ms每幅图，比Faster R-CNN对应部分快2.5-20倍。

#### 1. Introduction

普遍用于目标检测的深度网络家族可以根据感兴趣区域（RoI）池化层分成两个子网络：(i)共享的，全卷积子网络，独立于RoIs；(ii)一个RoI-wise子网络，不共享计算。这种分解是之前由先驱分类结构得到的，如AlexNet和VGG Nets（设计包含两个子网络），一个卷积子网络以空间池化层结尾，紧接着几个全连接层。因此图像分类网络最后的空间池化层自然的转变为目标检测网络中的RoI池化层。

但是最近的state-of-the-art图像分类网络如残差网络（ResNets）和GoogLeNets被设计为全卷积的。通过类比，使用全部卷积层来构造目标检测结构中共享的，卷积子网络，使Roi-wise子网络没有隐层看起来很自然。但是，根据本工作经验的调查，这种做法最后会得到相当差的检测准确度，不匹配网络优秀的分类准确度。为了补救这一问题，在ResNet文中Faster R-CNN检测器中的RoI pooling层被有意的插入到两个巻积层集合之间——这使得RoI-wise子网络更深，提高了准确度，并由于不共享的每个RoI计算，以更低速度为代价。

我们认为前面提到的有意设计是由增加图像分类的平移不变性 vs 目标检测相关的平移变化性的两难困境造成的。一方面，图像级分类任务支持平移不变性——图像中一个目标的移位不会被区别出来。因此，尽可能平移不变的深度（全）卷积结构是更好的选择，这一点可由ImageNet 分类前几名的结果证明。另一方面，目标检测任务需要在一定程度上平移变换的定位表示。举个例子，候选框中某物体的偏移应该产生对于描述候选框重叠目标程度的有用响应。我们假设图像分类网络中更深的巻积层对于偏移越不敏感。为了解决这种困境，ResNet文中的检测pipeline将RoI池化层插入到巻积层之间——这一区域特定操作打破了偏移不变性，之后RoI巻积层在评估不同区域时不再是偏移不变的。但是，这种设计牺牲了训练和测试效率，由于其引入了大量的region-wise层（见表1）。

![t1](E:\研究生\论文\R-FCN： Object Detection viaRegion-based Fully Convolutional Networks\images\t1.png)

本文中，我们设计一个框架称为基于区域的全卷积网络（R-FCN）用于目标检测。我们的网络包含共享的全卷积结构如FCN[15]中一样。为了将偏移变化性加入到FCN中，我们使用一组专门的巻积层构建了一系列位置敏感得分图作为FCN输出。每个得分图编码了相对空间位置的位置信息（如，“目标的左侧”）。在FCN之上，我们加了一个位置敏感的RoI池化层以从得分图中获取信息，后面没有卷积/全连接层。整个结构通过端到端学习。所有可学习层都是巻积的并在整幅图上共享，还编码了目标检测需要的空间信息。图1展示了关键思想，表1比较了基于区域检测器的方法。

![f1](E:\研究生\论文\R-FCN： Object Detection viaRegion-based Fully Convolutional Networks\images\f1.png)

使用101层残差网络（ResNet-101）作为主干，我们的R-FCN产生了竞争力的结果，在PASCAL VOC 2007上为83.6%mAP，在2012上为82.0%mAP。同时我们的结果达到了测试速度170ms每幅图，比Faster R-CNN+ResNet-101对应部分快2.5到20倍。这些实验证明了我们的方法可以解决偏移不变/变换的困境，并且全卷积图像层分类器如ResNet可以高效的转变为全卷积目标检测器。

#### 2. Our approach

**Overview** 跟随R-CNN，我们采用了流行的两阶段目标检测策略，包含：(i)区域提案，(ii)区域分类。尽管不依赖于区域提案的方法确实存在，基于区域的系统依然在许多benchmarks上准确度靠前。我们使用区域提案网络（RPN，全卷积结构）来提取候选区域。根据[18]，我们在RPN和R-FCN之间共享特征。图2展示了系统的概况。

![f2](E:\研究生\论文\R-FCN： Object Detection viaRegion-based Fully Convolutional Networks\images\f2.png)

给定提案区域（RoIs），R-FCN结构设计用于将RoIs分类为目标类别或背景。在R-FCN中，所有可学习的权重层都是卷积的并且在整幅图上计算。最后的巻积层对于每个类别产生一组$\ k^2\ $位置敏感得分图，因此输出层有$\ k^2(C+1)\ $个通道（C个目标类别，+1代表背景）。$\ k^2\ $得分图对应于一个描述相对位置的$\ k\times k\ $的空间网格。举个例子，$\ k\times k=3\times 3\ $，9得分图编码了某目标类别的{左上，中上，右上，...，右下}。

**R-FCN**以位置敏感的RoI池化层结尾。该层汇聚了上一个巻积层的输出并为每个RoI生成得分。与[8,6]不同的是，我们位置敏感RoI层执行选择性池化（selective pooling），每个$\ k\times k\ $的bin聚集了一组$\ k\times k\ $得分图中一个得分图的响应。通过端到端训练，该RoI层引导最后的巻积层学习特定的位置敏感得分图。图1展示了这一思想。图3和图4可视化了一个例子。细节介绍如下。

![f3](E:\研究生\论文\R-FCN： Object Detection viaRegion-based Fully Convolutional Networks\images\f3.png)

![f4](E:\研究生\论文\R-FCN： Object Detection viaRegion-based Fully Convolutional Networks\images\f4.png)

**Backbone architecture** 本文的R-FCN是基于ResNet-101的，尽管其他网络同样适用。ResNet-101有100个巻积层，紧接着一个全局平均池化以及1000类全连接层。我们移除平均池化层和fc层只使用巻积层来计算特征图。ResNet-101中最后的卷积块是2048维的，我们附加了一个随机初始化的1024维$\ 1\times 1\ $卷积层以减少维度（为了准确，这在表1中的深度上加了1）。接着我们使用$\ k^2(C+1)\ $个通道的巻积层来生成得分图。

**Position-sensitive score maps & Position-sensitive RoI pooling** 为了明确的将位置信息编码入每个RoI，我们使用一个规则网格将每个RoI矩形分成$\ k\times k\ $个bins。对于一个$\ w\times  h\ $的RoI矩形，一个bin的尺寸大约为$\ \frac{w}{k}\times\frac{h}{k}\ $。在我们的方法中，最后的巻积层用来产生每个类别的$\ k^2\ $得分图。在第$\ (i,j)\ $个bin中，我们定义池化第$\ (i,j)\ $个得分图的位置敏感RoI池化操作为
$$
r_c(i,j|\Theta)=\sum_{(x,y)\in\mathbb{bin}(i,j)}z_{i,j,c}(x+x_0,y+y_0|\Theta)/n\ \ \ \ \ \ \ \ \ \ \ \ (1)
$$
其中$\ r_c(i,j)\ $为对于第$\ c\ $个类别，第$\ (i,j)\ $个bin的池化响应，$\ z_{i,j,c}\ $为$\ k^2(C+1)\ $得分图中的一个得分图，$\ (x_0,y_0)\ $表示RoI的左上角，$\ n\ $为bin中的像素数，$\ \Theta\ $表示网络所有的可学习参数。第$\ (i,j)\ $个bin跨度为$\ \lfloor i\frac{w}{k}\rfloor\leq x<\lceil(i+1)\frac{w}{k}\rceil\ $以及$\ \lfloor j\frac{h}{k}\le y<\lceil(j+1)\frac{h}{k}\rceil\ $。公式1的操作如图1所示，其中一种颜色表示一对$(i,j)$。公式1执行平均池化，但最大值池化也可以使用。

$k^2$位置敏感得分接着对RoI投票。本文中我们通过对得分取平均来投票，为每个RoI产生一个$(C+1)$维向量：$\ r_c(\Theta=\sum_{i,j}r_c(i,j|\Theta))\ $。接着我们计算所有类别的softmax响应：$\ s_c(\Theta)=e^{r_c(\Theta)}/\sum^C_{c'=0}e^{r_{c'}(\Theta)}\ $。它们在训练时用来评估交叉熵损失，在inference时对RoIs排名。

我们进一步以相似的方式处理了边界框回归。除了上面$\ k^2(C+1)\ $维巻积层，我们附加了一个姊妹4$k^2\ $维巻积层用于边界框回归。位置敏感RoI池化在一组4$k^2\ $图上执行，每个RoI产生一个4$k^2\ $维向量。接着通过平均投票被聚合为一个4维向量。这个4维向量参数化边界框维$\ t=(t_x,t_y,t_w,t_h)\ $。注意我们执行的是未知类别的边界框回归，特定类别的对应部分（即，具有一个4$k^2C$维输出层）也是可以的。

位置敏感得分图部分是受到[3]中设计FCNs用于实例级语义分割的启发。我们进一步引入位置敏感RoI池化层以学习得分图用于目标检测。在RoI层之后没有可学习层，使得region-wise计算几乎无开销并加速了训练和inference。

**Training** 使用先前计算的区域提案，很容易端到端训练R-FCN结构。根据[6]，我们定义在每个RoI上的损失函数为交叉熵损失和边界框回归损失：$\ L(s,t_x,y,w,h)=L_{cls}(S_{c^*})+\lambda[c^*>0]L_{reg}(t,t^*)\ $。这里$\ c^*\ $为RoI的ground-truth标注($\ c^*=0\ $表示背景)。$\ L_{cls}(S_{c^*})=-log(s_{c*})\ $分类交叉熵损失，$\ L_{reg}\ $维边界框回归损失，$\ t^*\ $表示ground truth 框。$\ [c^*>0]\ $为指示器，内容为真则为1否则为0。我们设置平衡权值$\ \lambda=1\ $。我们定义正样本为与ground-truth框有至少0.5IoU重叠的RoI，其他为负样本。

我们的方法在训练时很容易应用在线困难样本挖掘（OHEM）。我们微不足道的前RoI计算使得样本挖掘几乎没有开销。假设每幅图$\ N\ $个提案，在前传时，我们评估全部$\ N\ $个提案的损失。接着我们对所有RoIs根据损失排序并选出$\ B\ $个具有最高损失的RoIs。反向传播的执行根据所选的样本。由于我们的前RoI计算微不足道，前传时间几乎不受$\ N\ $影响而OHEM Fast R-CNN训练时间可能为其两倍。表3中我们提供了全面时间统计。

我们使用权值衰减0.0005，动量为0.9。默认我们使用单尺度训练：图像被resize到短的一边为600像素。每个GPU存有1幅图并选择$\ B\ $=128个RoIs用于反向传播。我们使用8个GPU训练模型（所以有效的mini-batch为8x）。我们使用20k mini-batches学习率0.001，10k mini-batches学习率0.0001 fine-tune R-FCN。为了使R-FCN和RPN共享特征(图2)，我们应用了[18]中的4步交替训练$^3$，交替的训练RPN和R-FCN。

**Inference** 如图2所示，RPN和R-FCN之间共享的特征图在尺度为600的图像上计算。接着RPN部分提案RoIs，在其上R-FCN部分估计category-wise得分并回归边界框。在inference时，我们评估300个RoIs以公平比较。结果通过非最大值抑制（NMS，阈值为0.3IoU）进行后处理。

**A trous and stride** 我们的全卷积结构享受网络修改（用于语义分割的FCNs广泛使用）的好处。特别的，我们将ResNet-101中有效的步长从32像素减少到16像素，提高了得分图分辨率。所有之前的层以及conv4阶段都保持不变；在第一个conv5块stride=2的操作修改为stride=1，conv5阶段所有卷积过滤器通过“hole 算法”修改以补偿减少的步长。为了公平比较，RPN在conv4阶段（与R-FCN共享）计算，以此RPN可以不受$\grave{a}\ \ trous$影响。下表展示了R-FCN的消融结果（$k\times k=7\times 7$，无困难样本挖掘）。$\ \grave{a}\ \ trous\ $提高了mAP2.6个百分点。

![tt1](E:\研究生\论文\R-FCN： Object Detection viaRegion-based Fully Convolutional Networks\images\tt1.png)

**Visualization**  图3和图4中我们可视化了R-FCN学到的位置敏感得分图，$\ k\times k=3\times3\ $。这些专门的图期望在特定的目标相对位置有较强激活。例如，"top-center-sensitive"得分图展示了目标大致中上位置的高得分。如果一个候选框准确的与真实目标重叠（图3），RoI中大部分$\ k^2\ $bins都被强激活，而它们的投票导致了高得分。相反的，如果一个候选框没有正确的重叠真实目标（图4），RoI中一些$\ k^2\ $bins没有被激活，因此得分很低。

#### 3. Reltaed Work

R-CNN展示了以深度网络使用区域提案的有效性。R-CNN在裁剪和warped的区域上评估卷积网络，并且计算在区域之间不共享（表1）。SPPnet，Fast R-CNN以及Faster R-CNN是"半卷积的"，其中一个卷积子网络执行整幅图的共享计算而另一个子网络评估单个区域。

![t2](E:\研究生\论文\R-FCN： Object Detection viaRegion-based Fully Convolutional Networks\images\t2.png)

还有一些目标检测器可以看作是“全卷积”模型。OverFeat通过在共享卷积特征图上滑动多尺度窗口来检测目标；相似的，在Fast R-CNN和[12]中，研究了滑窗代替区域提案。在这些方法中，可以把单尺度滑窗看作单个巻积层。Faster R-CNN中的RPN组件为全卷积检测器，预测多个关于参考框（anchors）的边界框。原始的RPN为未知类别的，但其类别特定的对于能够部分也适用。

另一类目标检测器根据全连接层在整幅图上生成整体目标检测结果。

#### 4. Experiments

##### 4.1 Experiments on PASCAL VOC

我们在PASCAL VOC 上进行了实验。我们在VOC2007和VOC2012（“07+12”）trainval数据集上训练了模型，并在VOC 2007测试集上进行了评估。目标检测准确度有平均精度度量(mAP)。

**Comparisons with Other Fully Convolutional Strategies**

尽管全卷积检测器时可用的，但其达到好的准确度是不容易的。我们研究了如下全卷积策略（或几乎全卷积策略，只有一个分类fc层），使用ResNet-101：

**Naive Faster R-CNN** 如introduction中讨论的，可以使用ResNet-101中全部巻积层来计算共享特征图，并在最后的巻积层之后（conv5之后）应用RoI池化层。一个21类全连接层对每个RoI进行评估（这种变体几乎是全卷积的）。$\grave{a}\ \ trous$被使用以进行公平比较。

**Class-specific RPN** 该RPN根据[18]训练，除了2类卷积分类器层被替换为21类卷积分类器层。为了公平比较，对于该类别特定RPN我们使用ResNet-101的conv5层，以及$\ \grave{a}\ trous\ $。

**R-FCN without position-sensitivity** 通过设置$\ k=1\ $我们移除了R-FCN的位置敏感性。这等价于在每个RoI中进行全局池化。

*Analysis* 表2展示了结果。我们注意到标准(非naive)Faster R-CNN在ResNet文中达到了76.4%mAP，使用ResNet-101（见表3），在conv4和conv5之间插入了RoI池化层。作为对比，naive Faster R-CNN(在conv5之后应用RoI池化)大幅降低了mAP，为68.9%（表2）。根据经验这一比较证明了通过对Faster R-CNN系统的层之间插入RoI池化层获得的空间信息的重要性。

class-specific RPN mAP为67.6%，比标准Faster R-CNN低大约9个百分点。事实上class-specific RPN与Fast R-CNN的特殊形式（使用密集滑窗作为提案）相似。

另一方面，我们的R-FCN系统具有更好的准确度（表2）。76.6%mAP与标准Faster R-CNN相当。这些结果表明我们的位置敏感策略可以编码有用的空间信息用于定位目标，而不用在RoI池化后使用任何可学习层。

位置敏感性的重要性进一步通过设置$\ k=1\ $被证明，其中R-FCN不能收敛。在这种情况下，不能在RoI中明确的捕获空间信息。而且，如果RoI 池化输出分辨率为 1$\times$1 ，naive Faster R-CNN可以收敛，但mAP会进一步降低到61.7%。

![t3](E:\研究生\论文\R-FCN： Object Detection viaRegion-based Fully Convolutional Networks\images\t3.png)

![t4](E:\研究生\论文\R-FCN： Object Detection viaRegion-based Fully Convolutional Networks\images\t4.png)

![t5](E:\研究生\论文\R-FCN： Object Detection viaRegion-based Fully Convolutional Networks\images\t5.png)

##### Comparisons with Faster R-CNN Using ResNet-101

接着我们与标准"Faster R-CNN+ResNet-101"（最强的竞争对手，在PASCAL VOC，MS COCO以及ImageNet benchmark上表现靠前）做比较。我们使用$\ k\times k=7\times7\ $。表3展示了比较。Faster R-CNN对每个区域评估一个10层子网络以达到高准确度，但R-FCN每区域靠小微不足道。在测试时300个RoI，Faster R-CNN每幅图0.42s，比R-FCN的0.17s慢2.5倍。R-FCN也比Faster R-CNN训练要快。而且，困难样本挖掘不增加R-FCN训练开销（表3）。可以训练R-FCN，从2000个RoI中挖掘，在这种情况下Faster R-CNN要慢6倍（2.9s vs 0.46s）。但是实验表明从大量候选集中挖掘并不会有益。所以我们使用300个RoI用于训练和inference。

表4展示了更多比较。根据[8]中的多尺度训练，我们在每个训练迭代中resize图像，尺度从{400,500,600,700,800}像素中随机选择。我们在测试时使用600像素单尺度，因此不会增加测试时间开销。mAP为80.5%。另外，我们在MS COCO *trainval*集上训练了模型，不能够在PASCAL VOC集上fine-tune。R-FCN达到了83.6%mAP，与"Faster R-CNN +++"系统（也使用ResNet-101）接近。注意到我们的结果测试速度为0.17s每幅图，比Faster R-CNN+++的3.36s（其进一步加入了可迭代边界框回归，上下文以及多尺度测试）快20倍。这些比较也在PASCAL VOC 2012测试集上进行（表5）。

**On the Impact of Depth**

下表展示了R-FCN使用不同深度ResNet的结果。当深度从50升到101时，我们的检测准确度提升，但在深度为152时饱和。

![tt3](E:\研究生\论文\R-FCN： Object Detection viaRegion-based Fully Convolutional Networks\images\tt3.png)

**On the Impact of Region Proposals**

R-FCN可以简单的应用其他区域提案方法，如Selective Search和Edge Boxes。下表展示了使用不同提案的结果。R-FCN展示了竞争力的结果，展示了我们方法的泛化力。

![tt4](E:\研究生\论文\R-FCN： Object Detection viaRegion-based Fully Convolutional Networks\images\tt4.png)

##### Experiments on MS COCO

接着我们在80类别的MS COCO数据集上进行评估。我们的实验涉及80k 训练集，40k验证集以及20k测试集。前90k迭代学习率设为0.001，后30k迭代学习率为0.0001。有效mini-batch为8。我们将[18]中交替训练从4步拓展到5步（即，增加一个RPN训练步骤），这些许的提升了准确度。我们也发现2步骤训练足够达到相当的准确度但特征不是共享的。

结果如表6。我们的单尺度训练R-FCN baseline 验证结果为48.9%/27.6%。这与Faster R-CNN baseline(48.4%/27.2%)相当，但我们测试快2.5倍。值得注意的是我们的方法在小尺寸目标上表现好。我们多尺度训练的R-FCN验证集结果为49.1%/27.8%，测试集结果为51.5%/29.2%。考虑到COCO中尺度的变化范围，我们进一步评估了多尺度测试版本，使用测试尺度为{200,400,600,800,1000}，mAP为53.2%/31.5%。这一结果与MS COCO2015竞赛中的第一名（Faster R-CNN +++，ResNet-101，55.7%/34.9%）结果相近。然而，我们的方法更简单并没有加入上下文或可迭代边界框回归等，并在训练和测试都更快。

![t6](E:\研究生\论文\R-FCN： Object Detection viaRegion-based Fully Convolutional Networks\images\t6.png)

#### 5. Conclusion and Future Work

我们提出了基于区域的全卷积网络，一个简单准确高效的目标检测框架。我们的系统自然的采用了state-of-the-art图像分类主干，如ResNet，设计为全卷积的。我们的方法准确度与Faster R-CNN相当，但在训练和Inference都更快。

我们有意使文中的R-FCN系统简单。还有一些FCN的拓展可用于语义分割，以及基于区域的拓展用于目标检测。