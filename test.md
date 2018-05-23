### Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation
@(Paper Notes)[Text Detection and Recognition]

场景文本检测的难点：
**外因**：成像环境复杂，易受噪声、模糊、光照不均匀等因素的影响。这在一般的目标检测中也存在。
**内因**：（1）文本区域在图像中可能是任意方向，因此文本框（boundingbox）需要是可旋转的矩形或者多边形；（2）文本行的长宽比变化较大；（3）文本可能以字符、单词或者文本行的形式出现，在定位边界的时候可能会发生混淆。

基于深度学习的场景文本检测方法可以分为两大类，第一类将场景文本视为一般的目标检测问题，通过回归文本框来定位场景文本，但是会受到任意方向和文本区域长宽比变化较大的影响。因此需要考虑文本序列的特性，在通用目标检测框架上进行改进。第二类是采用语义分割的方法，对文本区域直接进行分割，但是大多需要复杂的后处理过程。

基于目标检测的方法：Faster R-CNN -> CTPN(水平方向)、RRPN(任意方向)，SSD -> TextBoxes、SegLink，DenseBox -> EAST。

论文中将检测和分割这两大类方法进行结合，提出了通过定位文本边界框的角点，并在相对位置分割文本区域来检测场景文本的方法。在推理阶段，候选边框通过对角点的采样和分组得到，并进一步通过分割图进行打分，然后使用非极大值抑制（NMS）滤除多余边框。与之前的方法相比，能够自然地处理不同方向的长文本，并且不需要复杂的后处理过程。

#### Architecture
检测流程如下：
- 首先检测文本的角点（左上角、右上角、右下角和左下角），而不是直接检测文本框，可以适应任意方向任意长宽比的文本
- 预测位置敏感分割图（类似于R-FCN），可以提供有效的文本位置信息
- 对角点进行采样和分组，生成候选文本框，并通过分割信息消除不合理的边框
![Alt text | center](./1526994721459.png)

网络结构如下图所示，主要由三部分组成，主干网络，角点检测模块，位置敏感分割模块。
![Alt text](./1526995378464.png)

#### Feature Extraction
主干网络采用DSSD进行特征提取，包括VGG16卷积层、额外的卷积层（VGG的fc6、fc7全连接层替换为卷积层conv6、conv7，以及另外4个卷积层conv8、conv9、conv10、conv11，上图少画了一个卷积层）、反卷积层（解码部分）。反卷积和跨层连接用于融合高层特征和低层特征，得到具有丰富语义信息和位置信息的特征金字塔，实现多尺度的目标检测。用于预测文本角点的特征图有$F_3、F_4、F_7、F_8、F_9、F_{10}、F_{11}$，相邻特征图之间的空间尺寸为2倍的关系。与DSSD不同的时，每个特征图的输出维度均为256，这样预测模块可以在不同特征图之间共享。

#### Corner Detection
一个文本框可以由参数$R=(x,y,w,h,\theta)$表示，而文本框的角点可以用一个正方形表示，正方形的中心为文本框（groundtruth）顶点坐标，正方形边长为文本框的短边长度，因此角点可以表示为$C=(x_c,y_c,ss,ss)$，实际上用三个参数即可表示。角点检测模块输出两个分支，一个表示角点的得分值，另一个表示角点的坐标偏移值。$w',h'$表示特征图的宽高，k表示default box的个数，q表示角点类型（左上、左下、右上、右下），一共4个，因此可以看做是一个4类的目标检测问题。
![Alt text | center](./1526996575594.png)
进行角点检测时在不同特征图考虑的尺寸，这个尺寸是针对角点（正方形）而言，因此只有一种长宽比1。
![Alt text | center](./1526997341666.png)

#### Position-Sensitive Segmentation
位置敏感分割图将文本框划分为$g*g$个bins，g取值为2，因此文本框一共划分为4个子块，分别为左上部分、右上部分、右下部分、左下部分。对所有的文本框都进行这样的划分，将每个bin看作一类，最后的分割图包含$g*g$个通道，相当于一个4类的语义分割问题。也就是说将所有像素分类到对应的bin中，如果不属于任意一个bin，则为背景像素。下图为一个文本框转换为分割图标签的结果。在进行位置敏感分割时，只用到了$F_3、F_4、F_7、F_8、F_9$一共5个特征图，并且后面添加了两个连续的Conv1x1-BN-ReLU-Deconv2x2 blocks模块，因此输出分割图和输入图像尺寸一致。
![Alt text | center](./1526997597568.png)

#### Training
损失函数为角点检测和位置敏感分割的多任务损失，公式如下：
![Alt text | center](./1526998410322.png)
$L_{conf}$为角点得分损失，采用交叉熵损失函数；$L_{loc}$为角点的定位偏移损失，采用Smooth L1损失函数；$L_{deg}$为位置敏感分割损失，采用Dice损失函数。$N_c$为正样本box的数量，$N_s$为分割图中的像素数量，$\lambda_1,\lambda_2$为权衡因子，取值分别为1、10。
![Alt text | center](./1526998661291.png)
![Alt text | center](./1526998746968.png)
$y_l=(\Delta x, \Delta y, \Delta ss, \Delta ss)$为坐标偏移的标签值，$y_l=(\Delta \hat x, \Delta \hat y, \Delta \hat {ss}, \Delta \hat {ss})$为预测值。default box坐标为$B = (x_b, y_b, ss_b, ss_b) $，真实角点坐标为$C = (x_c, y_c, ss_c, ss_c)$
![Alt text | center](./1526999067173.png)
位置敏感分割采用Dice Loss
![Alt text | center](./1527036089719.png)
$y_s$为位置敏感分割的标定值（label map），$p_s$为分割预测值，$y_s p_s$为逐像素相乘然后求和，$y_s + p_s$表示像素总和。后面一项$\dfrac{2y_s p_s}{y_s + p_s}$实际上为$F_1$ score，取值范围为[0,1]，用于衡量两者（$y_s$和$p_s$）的相似性。Dice Loss可参考[here](https://dev.to/andys0975/what-is-dice-loss-for-image-segmentation-3p85)

#### Inference
##### Sample and Grouping
在测试阶段，保留所有得分大于0.5的角点（horizontal square），然后通过非极大值抑制滤除多余的角点。最后对角点进行采样和分组生成候选框。角点有文本框顶点位置和短边长度信息，因此一个旋转的矩形框可以由相邻的两个角点和与两个角点组成的边相垂直的边来确定，在角点类型和位置都已知的情况下，短边方向也是确定的。可以组合的角点需要满足的先验条件：
- 角点的相对位置关系必须合理，例如左上角点的x坐标应该比右上角点小；而且组合的两个角点必须是相邻的角点，即（左上，右上）、（左上，左下）、（右上，右下）、（左下、右下）中的一种。
- 构成矩形框的短边长度（角点边长）应该大于一个阈值，例如5
- 两个角点的边长之比（长边/短边）应该小于1.5，即$max(ss_1,ss_2)/min(ss_1,ss_2)\leq1.5$

##### Scoring
对角点采样和分组生成的候选框并不一定是真实的文本框，因此还需根据位置分割敏感图对候选框进行打分筛选。论文提出了一种旋转的位置敏感 ROI 平均池化层（Rotated Position-Sensitive ROI Average pooling layer），将分割图上的候选框按偏转角度方向划分为$g*g$个网格（bins），然后在每个网格内统计前景像素（非零值）的得分总和P以及数量C，然后取平均得到每个网格的分数，最后对所有网格的分数取平均，作为该候选框的最终得分值，如果大于0.6则认为是文本框。处理流程和算法pipeline如下所示：
![Alt text | center](./1527038357007.png)
![Alt text | center](./1527038371837.png)
注意：在求每个网格的得分值时，只对前景像素取平均，而不是对整个网格区域的所有像素取平均，这样可以避免背景点的影响，使得分更精确。

#### Conclusion
论文提出了一种融合目标检测和分割思想的场景文本检测方法，可以实现端到端的训练，检测文本区域的角点而不是直接检测文本框可以处理任意方向任意长宽比的文本，避免了基于Anchor或者default box的方法中长宽比和尺度不好设置的情况（论文中用到的box是针对角点的，不用考虑长宽比）。而且检测的每个角点是相互独立的，多个特征层检测到的角点可以放到统一的集合里面（进行NMS处理），然后根据要求进行采样分组，生成候选框，而不是在每个特征层单独检测文本框后再进行多层融合。结合位置敏感分割图，可以有效地处理各种粒度的文本，例如字符、单词、文本行等，使检测结果更加精确。

#### reference
- [Paper: Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation](https://arxiv.org/pdf/1802.08948.pdf)
- https://www.cnblogs.com/lillylin/p/8495124.html
- https://www.jiqizhixin.com/articles/03011
- https://zhuanlan.zhihu.com/p/34208702
- https://dev.to/andys0975/what-is-dice-loss-for-image-segmentation-3p85