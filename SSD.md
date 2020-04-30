# SSD: Single Shot MultiBox Detector

## 1. 简介

​      目标检测主流的算法主要分为两个类型：

* **two-stage方法**：如R-CNN系算法，其主要思路是先通过Selective Search或者RPN网络产生一系列稀疏的候选框，然后对这些候选框进行分类与回归，two-stage方法的优势是准确度高；
* **one-stage方法**：如YOLO系列和SSD，其主要思路是均匀地在图片的不同位置进行密集抽样，抽样时可以采用不同尺度和长宽比，然后利用CNN网络提取特征后直接进行分类与回归，整个过程只需要一步，所以其优势是速度快；但是均匀的密集采样的一个重要缺点是正样本与负样本(背景)极其不均衡，导致模型准确度稍低。

​      本文介绍SSD(Single Shot MultiBox Detector)算法，MultiBox表明SSD是多框预测, SSD出现下YOLOv1算法之后，是结合YOLOv1和Faster RCNN提出来的。SSD算法在准确度和速度上都比YOLOv1要好很多。Faster R-CNN系列先通过CNN网络得到候选框，然后再进行分类与回归，而YOLO和SSD可以一步到位完成检测。与YOLOv1不同的是，SSD采用CNN来直接进行检测，而不是像YOLOv1那样在全连接层之后做检测。其实采用卷积直接做检测只是SSD相比Yolo的其中一个不同点，另外还有两个重要的改变，一是SSD提取了不同尺度的特征图来做检测，大尺度特征图可以用来检测小物体，而小尺度特征图用来检测大物体；二是SSD采用了不同尺度和长宽比的先验框（Prior boxes, Default boxes 即Anchor）。YOLOv1算法缺点是难以检测小目标，而且定位不准，但是这几点重要改进使得SSD在一定程度上克服这些缺点。

### 1.1 SSD设计思想

* SSD网络的设计思想来源于YOLOv1的框架结果和Faster R-CNN的anchor boxes:

<center class="half">    
    <img src="../pic/fasterrcnn.jpg" width="40%" title="Faster R-CNN"/>    
    <img src="../pic/yolov1.jpg" width="50%" title="YOLOv1"/>     
    <br><text>图1</text><br/>
</center>

主要有以下三点:

**（1）采用多尺度特征图用于检测**

​          多尺度即采用大小不同的特征图，CNN网络一般前面的特征图比较大，后面会逐渐采用stride=2的卷积或者pool来降低特征图大小，比较大的特征图来用来检测相对较小的目标，而小的特征图负责检测大目标，如图1所示，8x8的特征图可以划分更多的单元，但是其每个单元的先验框尺度比较小, 所以用来检测小目标；4X4的特征图用来检测大目标。

<center class="half">    
    <img src="../pic/feature_map.jpg" width="50%" title="feature_map"/>      
    <br><text>图2</text><br/>
</center>

**（2）采用卷积进行检测**

​        与YOLOv1最后采用全连接层不同，SSD直接采用卷积对不同的特征图来进行提取检测结果。对于形状为  $m \times n \times p$的特征图，只需要采用$3 \times 3 \times p$这样比较小的卷积核得到检测值。

**（3）设置先验框**

​        SSD借鉴了Faster R-CNN中anchor的理念，每个cell设置尺度或者长宽比不同的先验框，预测的边界框以这些先验框为基准，在一定程度上减少训练难度。一般情况下，每个cell会设置多个先验框，其尺度和长宽比存在差异，如图5所示，可以看到每个单元使用了4个不同的先验框，图片中猫和狗分别采用最适合它们形状的先验框来进行训练。

<center class="half">    
    <img src="../pic/feature_map2.jpg" width="70%" title="feature_map"/>      
    <br><text>图2</text><br/>
</center>

### 1.2 SSD预测分析

SSD预测的Bbox分为两个部分：

* 第一部分是各个类别的置信度，其中SSD将背景也当做一个类别，如果数据集中待检测目标共有$c$个类别，SSD其实需要预测$c+1$个置信度值，其中第一个置信度指的是属于背景的置信度。

* 第二部分是Bbox的位置($cx,xy,w,h$) , 即Bbox的中心坐标以及宽高。但是预测值其实只是Bbox相对于先验框anchor的偏移量。Anchor的位置用$d=\left(d^{c x}, d^{c y}, d^{w}, d^{h}\right)$ 表示，对应的Bbox坐标用$b=\left(b^{c x}, b^{c y}, b^{w}, b^{h}\right)$ 表示，那么Bbox坐标的预测值$l$计算公式如下所示:
  $$
  \begin{array}{c}
  l^{c x}=\left(b^{c x}-d^{c x}\right) / d^{w}, l^{c y}=\left(b^{c y}-d^{c y}\right) / d^{h} \\
  l^{w}=\log \left(b^{w} / d^{w}\right), l^{h}=\log \left(b^{h} / d^{h}\right)
  \end{array}
  $$

​      上面这个过程为Bbox的编码(encode)，预测时，需要反向计算进行解码(decode)，从预测值$l$ 中得到Bbox真实坐标$b$ ：
$$
\begin{aligned}
b^{c x}=& d^{w} l^{c x}+d^{c x}, b^{c y}=d^{y} l^{c y}+d^{c y} \\
b^{w} &=d^{w} \exp \left(l^{w}\right), b^{h}=d^{h} \exp \left(l^{h}\right)
\end{aligned}
$$

### 1.3 SSD成果

​		SSD在PASCAL VOC、COCO、ILSVRC数据集上均获得了当时的SOFA结果，在VOC2007的测试结果为，`59FPS`和`74.3%`的mAP, SSD无论是速度还是精度均高于当时最好的模型-Faster R-CNN.SSD的改进设计，使得在输入分辨率较低时能够保证检测的精度，同时网络是end-to-end的设计，训练简单，在检测速度和精度之间取得较好的trade-off(平衡)。

## 2. 网络结构

SSD网络结构示意图如下图3所示：

<center class="half">    
    <img src="../pic/ssd.jpg" width="85%" title="SSD"/>      
    <br><text>图3</text><br/>
</center>

​       SSD使用VGG16作为主干网络，VGG16首先在ILSVRC CLS-LOC数据集进行预训练，然后在VGG16的基础上新增了6个卷积层来得到更多的特征图用于提取特征。SSD的网络结构如图3所示；上面是SSD模型, 下面是YOLOv1模型，可以明显看到SSD利用了多尺度的特征图做检测; 模型的输入图片大小是$300 \times 300$。

​		VGG16网络中的fc6 和 fc7全连接层转换为卷积层,  改变pool5池化层的卷积核大小( filter=2 × 2 stride=2 改变为filter=3 × 3 stride=1), 去除fc8全连接层和所有的Dropout层，在fc6全连接层上使用空洞卷积(dilated convolution)弥补损失的感受野(在不增加参数与模型复杂度的条件下指数级扩大卷积的视野)。

* 空洞卷积(dilated convolution)

  空洞卷积避免了池化带来的问题(内部数据结构丢失、空间层级化信息丢失、小物体信息无法重建)， 相比原来的正常卷积操作，空洞卷积多了一个超参数-扩张率(dilation rate), 指的是kernel的间隔数量; 

<center class="half">    
    <img src="../pic/stand_convolution.webp" width="40%" title="Stand Convolution"/>     
    <img src="../pic/dilated_convolution.webp" width="40%" title="Dilated Convolution"/>     
    <br><text>图4&ensp;标准卷积(3X3 s=1) &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; 图5&ensp;空洞卷积(3X3 dilation rate=2)</text><br/>
</center>

​         如图3中所示 从VGG网络后面新增的卷积层Conv7，Conv8_2，Conv9_2，Conv10_2，Conv11_2中提取特征图，然后用于检测，加上Conv4_3层，共提取了6个特征图，其大小分别是$(38,38),(19,19),(10,10),(5,5),(3,3),(1,1)$ , 其中不同特征图的cell设置的先验框数目不同，但是同一个特征图上每个cell设置的先验框数目是相同的。

​       先验框的设置，包括尺度和长宽比两个方面。对于先验框的尺度，遵守一个线性递增规则：随着特征图大小降低，先验框尺度线性增加. 计算公式如下：
$$
s_{k}=s_{\min }+\frac{s_{\max }-s_{\min }}{m-1}(k-1), k \in[1, m]
$$
​      其中$m$是特征图的个数，这里是5 ，因为第一层(Conv4_3 layer)是单独设置的，$s_{k}$表示先验框大小相对于图片的比例，论文中$s_{k}的取值范围是$$s_{min}=0.2$和$s_{max}=0.9$。对于第一个特征图，其先验框的尺度比例一般设置为$s_{min}/2=0.1$，那么尺度为$300 \times 0.1=30$。对于后面的特征图，先验框尺度按照上面公式线性增加，根据上面的公式可以得到各个特征图的尺度为60,111,162,213,264。对于长宽比，一般选取$a_{r} \in\left\{1,2,3, \frac{1}{2}, \frac{1}{3}\right\}$，对于特定的长宽比，按如下公式计算先验框的宽度与高度（下面的$s_{k}$均指的是先验框实际尺度，而不是尺度比例）
$$
w_{k}^{a}=s_{k} \sqrt{a_{r}}, h_{k}^{a}=s_{k} / \sqrt{a_{r}}
$$
​        默认情况下，每个特征图会有一个$a_{r}=1$且尺度为$s_{k}$的先验框，除此之外，还会设置一个尺度为$s^{'}_{k}= \sqrt{s_{k}s_{k+1}}$且$a_{r}=1$的先验框，这样每个特征图都设置了两个长宽比为1但大小不同的正方形先验框。注意最后一个特征图需要参考一个虚拟$s_{m+1}=300 \times 105 / 100=315$来计算$\boldsymbol{s}_{m}^{\prime}$。因此，每个特征图一共有6个先验框$\left\{1,2,3, \frac{1}{2}, \frac{1}{3}, 1^{\prime}\right\}$，但是在代码实现中，Conv4_3，Conv10_2和Conv11_2层仅使用4个先验框，它们不使用长宽比为$3, \frac{1}{3}$的先验框。每个单元先验框的中心点分布在各个单元的中心，即$\left(\frac{i+0.5}{\left|f_{k}\right|}, \frac{j+0.5}{\left|f_{k}\right|}\right), i, j \in\left[0,\left|f_{k}\right|\right)$，其中$|f_{k}|$为特征图的大小。

​        提取特征图以后，对特征图进行卷积得到预测结果，预测值包括类别置信度和边界框位置，各采用一次$3 \times 3$卷积来完成。令$n_{k}$为该特征图所采用的先验框数目，那么类别置信度需要的卷积核数量为$n_{k} \times c$。边界框位置需要的卷积核数量为$n_{k} \times 4$。由于每个先验框都会预测一个边界框，所以SSD 300预测的边界框数量为：
$$
38 \times 38 \times 4+19 \times 19 \times 6+10 \times 10 \times 6+5 \times 5 \times 6+3 \times 3 \times 4+1 \times 1 \times 4=8732
$$
数量很大，因此SSD本质上是密集采样。

## 3. 训练过程

### 3.1 先验框匹配

​        在训练过程中，首先要确定训练图片中的Ground Truth与哪个先验框来进行匹配，与之匹配的先验框所对应的边界框将负责预测它。先验框与Ground Truth的匹配原则主要有两点。

* 首先，对于图片中每个Ground Truth 找到与其IoU最大的先验框，该先验框与其匹配，这样，可以保证每个Ground Truth一定有一个先验框与之匹配。与Ground Truth匹配的先验框为正样本， 若一个先验框没有与任何Ground Truth进行匹配，该先验框只能与背景匹配则称为负样本。一个图片中Ground Truth是非常少的，而先验框却很多，
* 如果仅按上述原则匹配，会有很多先验框会是负样本，正负样本极其不平衡，所以需要第二个原则：对于剩余的未匹配先验框，若某个Ground Truth的 IoU大于某个阈值(论文中为0.5)，那么该先验框也与这个Ground Truth进行匹配。这意味着某个Ground Truth可能与多个先验框匹配，这是可以的。但是反过来却不可以，因为一个先验框只能匹配一个Ground Truth，如果多个Ground Truth与某个先验框IoU大于阈值，那么先验框只与IOU最大的那个Ground Truth进行匹配。该一定在第一个原则之后进行。如果某个Ground Truth所对应最大IoU小于阈值，并且所匹配的先验框却与另外一个Ground Truth的IoU大于阈值，那么该先验框应该匹配前者，首先要确保某个Ground Truth一定有一个先验框与之匹配。由于先验框很多，某个Ground Truth的最大 IoU肯定大于阈值，所以只需按照第二个原则就可以。

​       尽管一个Ground Truth可以与多个先验框匹配，但是Ground Truth相对先验框还是太少了，所以负样本相对正样本会很多。为了保证正负样本尽量平衡，SSD算法采用了Hard Negative Mining，对负样本进行抽样，按照置信度误差进行降序排列，选取误差的较大的top-k作为训练的负样本，且证正负样本比例为1:3。

### 3.2 损失函数

​        损失函数包括两部分：位置误差(Locatization Loss)与置信度误差(Confidence Loss)：
$$
L(x, c, l, g)=\frac{1}{N}\left(L_{c o n f}(x, c)+\alpha L_{l o c}(x, l, g)\right)
$$
​        其中$N$是先验框正样本的数量。这里$x_{i j}^{p} \in\{1,0\}$是一个指示函数，当$x_{i j}^{p}=1$) 时表示第$i$个先验框与第$j$个Ground Truth匹配，并且Ground Truth的类别为$p$。$c$为类别置信度预测值。$l$为先验框的所对应边界框的位置预测值，而$g$是Ground Truth的位置参数。

* 位置误差-Smooth L1 loss，定义如下：

$$
\begin{array}{c}
L_{l o c}(x, l, g)=\sum_{i \in P o s}^{N} \sum_{m \in\{c x, c y, w, h\}} x_{i j}^{k} \operatorname{smooth}_{\mathrm{L} 1}\left(l_{i}^{m}-\hat{g}_{j}^{m}\right) \\
\hat{g}_{j}^{c x}=\left(g_{j}^{c x}-d_{i}^{c x}\right) / d_{i}^{w} \quad \hat{g}_{j}^{c y}=\left(g_{j}^{c y}-d_{i}^{c y}\right) / d_{i}^{h} \\
\hat{g}_{j}^{w}=\log \left(\frac{g_{j}^{w}}{d_{i}^{w}}\right) \quad \hat{g}_{j}^{h}=\log \left(\frac{g_{j}^{h}}{d_{i}^{h}}\right)
\end{array}
$$

$$
\operatorname{smooth}_{L_{1}}(x)=\left\{\begin{array}{ll}
0.5 x^{2} & \text { if }|x|<1 \\
|x|-0.5 & \text { otherwise }
\end{array}\right.
$$

$x^{p}_{ij}$是指示函数，正样本为1负样本为0，所以只有正样本产生$L_{loc}$误差。要先对Ground Truth的$g$进行编码得到$\hat g$ ，因为预测值$l$也是编码值，若设置variance_encoded_in_target=True，编码时要加上variance：
$$
\begin{aligned}
\hat{g}_{j}^{c x} &=\left(g_{j}^{c x}-d_{i}^{c x}\right) / d_{i}^{w} / \text {variance}[0], \hat{g}_{j}^{c y}=\left(g_{j}^{c y}-d_{i}^{c y}\right) / d_{i}^{h} / \text {variance}[1] \\
\hat{g}_{j}^{w} &=\log \left(g_{j}^{w} / d_{i}^{w}\right) / \text {variance}[2], \hat{g}_{j}^{h}=\log \left(g_{j}^{h} / d_{i}^{h}\right) / \text {variance}[3]
\end{aligned}
$$

* 置信度误差-Confidence Loss: 

$$
L_{c o n f}(x, c)=-\sum_{i \in P o s}^{N} x_{i j}^{p} \log \left(\hat{c}_{i}^{p}\right)-\sum_{i \in N e g} \log \left(\hat{c}_{i}^{0}\right) \quad \text { where } \quad \hat{c}_{i}^{p}=\frac{\exp \left(c_{i}^{p}\right)}{\sum_{p} \exp \left(c_{i}^{p}\right)}
$$

其权重系数$\alpha$是通过交叉验证的方法设置为1.

### 3.3 数据扩增

​		采用数据扩增(Data Augmentation)可以提升SSD的性能，主要采用的技术有水平翻转(horizontal flip)、随机裁剪加颜色扭曲(random crop & color distortion)、随机采集块域(Randomly sample a patch )、获取小目标训练样本）