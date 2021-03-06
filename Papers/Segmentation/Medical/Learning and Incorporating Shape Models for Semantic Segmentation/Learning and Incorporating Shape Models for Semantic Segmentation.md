## Learning and Incorporating Shape Models for Semantic Segmentation

> MICCAI 2017

### Abstract

​	语义分割已经广泛的使用全卷积网络FCN来解决，并获得不错的结果，一直是最近分割挑战的先驱。但是，FCN方法不一定包括了局部几何信息，如平滑度和形状，而传统图像分析技术在解决分割和跟踪问题时受其影响较多。本文中，我们解决了将形状先验包含进FCN分割框架的问题。我们展示了在解决对比度和伪影损失的情况下，该形状先验的鲁棒性。我们的实验展示了对U-Net约5%的提升，在超声肾脏分割问题中。

### 1. Introduction

​	由于医学volumes的模态和解剖结构，对其分割具有挑战性。传统方法如活动轮廓（active contours）使用线性/非线性的形状模型，解决了分割问题的不定性。最近，FCN被成功的应用于2D/3D医学图像分割，光流，恢复（restoration）等。虽然FCN很成功的将上下文引入了学习，其还存在一些缺点最近的研究工作尝试去解决。首先，局部几何信息如平滑度和拓扑结构并没有可靠明确的被捕获。其次，这需要足够有代表性的训练数据来从本质上建模前景，背景，形状以及前面这些实体的上下文关系。在训练数据有限时，FCN的一些失败的模式也很难去解释或提高。

​	受到传统方法的启发，我们提出使用先验形状信息来增强FCN框架。在FCN中明确的建模形状有两点好处：（1） 我们发现对训练数据中外观差异的泛化更好了；（2）数据增强策略对于FCN的鲁棒表现是必不可少的。特别是对于医学数据，很难想象真实的外观变化可以使FCN解决低分辨率或伪影的问题。解耦形状模型，可以很容易构建形状类别的数据增强策略，以捕获不变性，从而提升预测性能。我们在2D超声B-mode图像上的肾脏解剖分割问题中证明了我们方法的有效性。

​	总的来说，本文的贡献如下：

- （1）学习一个非线性形状模型并对对任意mask映射到形状流形空间。我们也讨论了两种新的数据增强策略来实现形状卷积自动编码。

- （2）通过一个新的损失函数（对预测分割mask与学得形状模型之间的差异惩罚）明确的将形状模型并入FCN。

- （3）证明了提出的方法的优越性，在dice重叠上提升了约5%，增加的网络复杂度小于1%。

### 2. Related Work

​	在训练数据有限时，FCN的失败模式很难解释或提高。在最近的一项研究中，展示了平行网络通过权值共享或新的损失函数进行绑定，明确的，联合的对外观和形状进行建模，提高了FCN的鲁棒性。

​	此外，结合图像的几何特性（如，形状或平滑度）在解决分割，光流等图像范围的预测问题时很关键。在[1]中，作者通过在组织分割的多标注问题中加入平滑度和拓扑先验，解决了局部几何问题。对于3D形状分割，[11]中通过表面映射层（通过基于表面的条件随机场处理），结合了多个FCN（标注置信度）的输出，以获得一致的标注。另一类研究考虑使用深度网络来学习形状先验，该网络随后在变化的框架下以经典的方式使用。在[3,5]中，使用深度Boltzmann machines来学习形状先验，但以变化的形式来用于图像分割和图像修复任务。在[14]中，提出了一个分割网络，其中使用了一个预训练的analysis网络，以获得图像特征，该特征通过FCN来获得全局分割mask。这些全局mask接着使用analysis网络低层的权值来精修。

​	在我们的工作中，我们级联两个CNN达到了受形状先验影响的分割。我们网络的主要不同如下：（1）在FCN内部以优雅的方式来包含形状规则化，而不是标注置信度或不完整形状的后处理步骤。所提方法的动机是FCN的输出可能不位于真实形状的流形上，因此需要被映射到正确的位置。该映射通过自动编码器（AE）来实现，其在训练中隐含的提供了形状先验。在测试时，分割结果直接从FCN的输出获得。（2）一种可以加到其他几何或拓扑先验的通用形式。（3）使用简单网络实现形状正规化，该网络使用两个数据增强策略训练。

### 3. Our Approach

​	FCN是CNN的拓展，用于像素级预测，本质上具有层级的反卷积层作用于CNN特征图，已给出图像输出。每个反卷积层都与相应的卷积层连接，以保持上采样时的精细细节。FCN对于将空间上下文引入预测很有效，优势是对于前传操作很快速。在标准FCN形式中，如U-Net，给定训练样本，图像和分割mask对，$I_k$，$S_k$，$k=1,2,...,N$，该框架学习由参数$\ w\ $定义的预测器$\hat{S}_w[·]$，最小化训练损失，如$\ RMSE:=\frac{1}{N}\sum^N_{k=1}|S_k-\hat{S}_w[I_k]|^2\ $

​	本文中，我们修改上述损失以加入形状先验。虽然有很多线性/非线性的分割形状先验表示，我们使用*convolutional autoencoders*（CAE）作为形状表示，使其能简单的集成到现有的FCN实现中。

​		由有效形状（通过ground truth 训练mask定义，$\ S_k,k=1,2,...N\ $）组成的有效形状潜在空间标记为$\ \mathcal{M}\ $。假设我们可以学得一个p维形状映射（*encoder*）*E*以及一个（*decoder*）*R*。为了可以插入到分割框架中，映射*E*应该可以接受任意形状*S*并将其映射到*M*上一个有效的表示。因此，与解码器*R*的组合，即（$R\ \circ\ E[S]$）为将S映射到$\ \mathcal{M}\ $上有效的形状。可以将$\ R\ \circ\ E\ $看作是在分割损失函数中扮演*convolutional de-noising autoencoder*（CDAE）的角色。记$\ \hat{S_k}=\hat{S}_w[I_k]\ $ ，我们修改损失为：
$$
L[w] = \frac{1}{N}\sum^N_{k=1}|\hat{S}_k-(R\ \circ\ E)[\hat{S}_k]|^2\ +\ \lambda_1|E[S_k]-E[\hat{S_k}]|^2\ +\ \lambda_2|S_k-\hat{S}_k|^2
$$
第一项通过最小化映射误差驱使预测形状$\ \hat{S}_k\ $贴近于形状空间$\ \mathcal{M}\ $。第二项驱动编码的ground truth mask表示与预测mask之间的距离。最后一项尝试在学得的形状空间$\ \mathcal{M}\ $中保持ground truth形状的变化。在FCN原始实现中，如U-Net，由于损失函数是基于欧几里得距离，网络参数必须预测从输入图像到高维形状的复杂变换，因此需要足够的代表性训练数据来从本质上建模外观，形状以及其上下文相互关系。在我们提出的方法中，网络复杂度的很大一部分由自动编码器承担，由于预测形状$\ \hat{S}_k\ $与ground truth$\ S_k\ $之间的距离是基于编码表示的（见图1）。

![f1](images\f1.png)

### 4. Architectures

​	在本节中，我们解释神经网络模型，以实现公式1。我们构建了两个FCN的级联，一个用于分割，一个用于形状规则化。如图2所示。分割网络操作输入图像，形状规则化网络约束预测形状在训练形状定义的$\ \mathcal{M}\ $中。

![f2](images\f2.png)

#### 4.1 Segmentation Network

 	我们的分割网络为原始U-Net结构（如图3a），该结构是医学图像分割中最成功和流行的CNN结构。U-Net具有分解-综合块，以及对应层的跳跃级连接。

![f3](images\f3.png)

#### 4.2 Shape Regularization Network

​	本网络的目的是操作不完整的，不足/超过的分割形状mask，并将其转换到训练形状的manifold中。我们提出使用卷积自动编码器来实现形状规则化，如图3b所示。形状规则化网络包含了形状编码和解码块，使用卷积核非线性映射的组合将不完整的形状映射到潜在的表示。我们假设编码器可以提供一个简明的，紧凑的，不会受到输入形状误差影响的潜在空间表示，解码器块从中可以准确的重建完整的形状。在编码器和解码器之间没有跳跃的连接。

​	级联网络不同部分的信息流如图2所示。shape completion网络的编码器和重建层的输出——$\ E[\hat{S}_k]\ $以及$\ (R\ \circ\ E)[\hat{S}_k]\ $影响公式1中的前两项，而分割网络的输出控制第三项。形状规则化网络单独在噪声增强的形状（4.3节）上预训练，接着被加入级联结构。它通过公式1更新分割网络，产生形状规则化U-Net（SR-UNet）。

#### 4.3 Implementation Details

​	我们的分割网络包含卷积层和上/降采样层，总共10层，平均分布在U-Net的两侧。可训练参数约$\ 14\times 10^6\ $，我们使用ReLUs以及batch normalization分别作为激活单元和规则化。直观的，我们希望shape completion 网络比较简单，因此，我们构建了卷积自动编码器，可训练参数为$\ 12\times10^3\ $，使网络复杂度增加少于1%。公式1中典型的$\ \lambda\ $约为0.5，这些值的变化并不会产生显著的改变。我们接着描述shape completion 网络的预训练。

#### 4.4 Data Augmentation for Shape Regularization Network

![f4](images\f4.png)

对于形状规则化网络，其训练时必须使用不准确的形状作为输入，ground truth masks作为输出。我们执行了两种数据分割策略以产生这些不完整的形状：

- （a）**Random corruption of shapes** 我们使用随机高平均强度的corruption核，在形状随机种子位置上滑过并侵蚀。我们重复这一步多次以产生多个实例，如图4a所示。

- （b）**Intermediate U-Net predications** 我们采样U-Net在收敛前不同epoch的预测结果，并将其作为shape completion网络的输入，如图4b所示。思想是使得CAE学得将U-Net的失败模式补充完整。

### 5. Kidney Segmentation from U/S B-Mode Images

​	使用自动化检测，分割方法来加速临床工作流提供了很多好处，如操作独立性，提高的临床结果等。自动的2D超声纵向扫描肾脏分割具有挑战性，由于许多原因——肾脏形状，尺寸和方向的变化性，获取扫描平面的差异，内部区域（肾窦）的变化性以及邻近结构的影响，如隔膜 ，肝脏以及脂肪层。任何存在的病理或异常会严重改变观察到的纹理，接着可能会进一步加剧超声问题，如伪影，斑点，对散射体敏感等。此外，自动算法期望可以在不同扫描协议，不同探头图像，不同获取或重建设置下工作。

**Data** 本实验的目的是证明我们的方法对于UNet的鲁棒性和泛化性。数据集包含总共231张B-mode图像，从两个不同扫描，不同的获取设置中得到。图像包含了各种挑战-病理，伪影，不完整的肾脏获取，畸形并包含了从成人到小孩的。我们使用100幅图像训练，剩余图像测试。结果展示了我们方法的优势。

### 6. Results

​	我们使用Dice系数来比较我们的结果以及专家标注的ground truth。我们称我们形状规则化的FCN为SR-UNet_1以及SR-UNet_2，分别对应两种不同的数据增强策略。表1中，可以看出SR-UNet_1,2提高了Dice重叠4-5%，在挑战问题上有较大提升。不出意外的，使用噪声U-Net预测构建的的shape completion网络结果更好，因为它明确的作用于失败模式，但有趣的是，合成数据增强效果同样好。图5展示了SR-UNet可以将复杂结构完整化，即使有严重的病理存在。如第一行，伪影移除了肾脏右侧几乎所有信息。然而，级联网络可以达到接近ground truth。类似的在第3行，囊肿的存在影响了U-Net而SR-UNet可以达到更准确的结果。在第2，4行，肾脏大部分受到畸形影响，使得U-Net分割变差，而我们的方法在第二行产生了接近完美的分割结果，在第4行提高了很多。我们新的新装规则化方法是通用的，可以被包含进任何语义分割网络。我们选择与医学图像分割中较为流行的U-Net进行比较。

![f5](images\f5.png)

![t1](images\t1.png)

​	虽然U-Net在一些医学图像分析问题中特别有效，但我们的结果表明至少在有限数据场景下U-Net对于形状表现不好，特别是纹理和局部信息由于病理无法获知的情况下。相关不希望的特征导致产生不连续的分割。虽然其他技术如一些精心设计的后处理可以解决这些问题，但我们的方法提供了一个自然，鲁棒的方法，将渴望的形状特征集成入深度网络的训练过程中。

### 4. Discussion

​	将形状先验集成入神经网络的训练损失中，可以极大提高预测结果，如我们的分割实验所证明的。尽管一些情况十分具有挑战，我们认为我们的贡献是FCN应用于临床设置的重要一步，从而产生有意义，可解释的输出。此外，在我们的公式中，将形状先验拓展至3D分割很简单。虽然我们使用卷积自动编码器来获得形状先验，可选的还有Boltzmann machines，linear shape dictionaries等。而且，形状只是解剖结构中一个几何属性，更有意义的先验（如纹理，尺寸等）都可以被集成入训练目标，从而更鲁棒，更稳定。