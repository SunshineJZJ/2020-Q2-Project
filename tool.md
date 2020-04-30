SSD: Single Shot MultiBox Detector [[paper\]](https://arxiv.org/abs/1512.02325)[[code\]](https://github.com/weiliu89/caffe/tree/ssd)[[slide\]](http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf)[[video\]](https://drive.google.com/file/d/0BzKzrI_SkD1_R09NcjM1eElLcWc/view)[[author\]](http://www.cs.unc.edu/~wliu/)

使用Tensorflow框架加快训练速度时需要两个改变：

* 通过启用TF_ENABLE_WINOGRAD_NONFUSED;
* 改变提供给channel first而不是channel last的维度(data_format ='channels_first');对卷积操作启用WINOGRAD。



### SSD研究背景

​         在目标检测领域，one-stage方法速度较快但精度低，two-stage方法精度高但速度低，精度和速度之间的平衡问题一直未解决，SSD网络的涉及思想来源于YOLO的框架结果和Faster R-CNN的anchor boxes。

<center class="half">    
    <img src="../pic/fasterrcnn.jpg" width="40%" title="Faster R-CNN"/>    
    <img src="../pic/yolov1.jpg" width="50%" title="YOLOv1"/>   
</center>


### SSD成果

​		SSD在PASCAL VOC、COCO、ILSVRC数据集上均获得了当时的SOFA结果，在VOC2007的测试结果为，59FPS和74.3%的mAP, SSD无论是速度还是精度均高于当时最好的模型-Faster R-CNN.SSD的改进设计，使得在输入分辨率较低时能够保证检测的精度，同时网络是end-to-end的设计，训练简单，在检测速度和精度之间取得较好的trade-off(平衡)。

### SSD算法主要特点

- 特征提取主干网络：VGG16，fc6 和 fc7层转换为卷积层,  改变pool5层的卷积核大小，从 filter=2 × 2 stride=2 改变为filter=3 × 3 stride=1, 去除全连接层fc8，在fc6上使用空洞卷积(dilated convolution)弥补损失的感受野；并且增加了一些分辨率递减的卷积层；
- SSD摈弃了proposal的生成阶段，使用anchor机制，这里的anchor就是位置和大小固定的box，可以理解成事先设置好的固定的proposal;
- SSD使用不同深度的卷积层预测不同大小的目标，对于小目标使用分辨率较大的较低层，即在低层特征图上设置较小的anchor，高层的特征图上设置较大anchor；
- 预测层：使用3x3的卷积对每个anchor的类别和位置直接进行回归;
- SSD使用的Data Augmentation对效果影响很大;





