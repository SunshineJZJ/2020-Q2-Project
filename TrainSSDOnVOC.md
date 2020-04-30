# Train SSD On VOC  By TF1.6 教程

## 0. 环境准备

**创建CPU或GPU虚拟环境,  训练推荐GPU环境, 预测时CPU、GPU均可**

创建Tensorflow GPU虚拟环境

```python
# conda命令创建虚拟环境
conda env create -f conda-gpu.yml
# 激活虚拟环境
conda activate yolov3-tf2-gpu
# 退出虚拟环境
conda deactivate
```

主要依赖库及版本：

```python
python==3.5.6
pip
matplotlib
caffe-gpu
opencv-python==3.3.1
tensorflow-gpu==1.6.0
scipy==1.1.0
lxml
tqdm
```

使用命令conda  install caffe-gpu仅是安装caffe的Python接口.

**以下步骤均在虚拟环境下进行,请先激活虚拟环境**


## 1. 准备数据集

`Pascal VOC Dataset`官网, 点击[链接](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)或用下面命令将数据集下载到./dataset/PASCAL_VOC/目录：

```bash
# 2007年的训练数据
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
# 2007年的测试数据
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# 2012年的训练数据
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# 解压
tar -xvf VOCtest_06-Nov-2007.tar -C VOC2007TEST
tar -xvf VOCtrainval_06-Nov-2007.tar  -C VOC2007
tar -xvf VOCtrainval_11-May-2012.tar  -C VOC2012
```

目录结构如下所示：

```python
./dataset/PASCAL_VOC/
 	   |->VOC2007/
 	   |    |->Annotations/
 	   |    |->ImageSets/
 	   |    |->...
 	   |->VOC2012/
 	   |    |->Annotations/
 	   |    |->ImageSets/
 	   |    |->...
 	   |->VOC2007TEST/
 	   |    |->Annotations/
 	   |    |->...
```

### 1.1. 目标检测数据集PASCAL VOC简介

<center class="half">    
    <img src="../pic/voc_dataset.jpg" width="40%"/>     
</center>

​		PASCAL VOC挑战赛 （The [PASCAL](http://pascallin2.ecs.soton.ac.uk/) Visual Object Classes ）是一个世界级的计算机视觉挑战赛, [PASCAL](http://www.pascal-network.org/)全称：Pattern Analysis, Statical Modeling and Computational Learning，是一个由欧盟资助的网络组织。

​		很多优秀的计算机视觉模型比如分类，定位，检测，分割，动作识别等模型都是基于PASCAL VOC挑战赛及其数据集上推出的，尤其是一些目标检测模型（比如大名鼎鼎的R CNN系列、YOLO、SSD等）。

​		PASCAL VOC从2005年开始举办挑战赛，每年的内容都有所不同，从最开始的分类，到后面逐渐增加检测，分割，人体布局，动作识别（Object Classification 、Object Detection、Object Segmentation、Human Layout、Action Classification）等内容，数据集的容量以及种类也在不断的增加和改善。

数据集下载解压后如下图所示：

<center class="half">    
    <img src="../pic/voc.jpg" width="40%"/>     
</center>

* JPEGImages文件夹存储了17125张图片，目前已知：有11540张用于检测任务

  其中：训练集train:5717, 验证集val:5823)

* Annotations文件夹存放的是xml文件,存储每张图片中标注信息,每张图片都有一个同名的xml文件

  <center class="half">    
      <img src="../pic/voc_dataset_annotation.jpg" width="40%"/>     
      <img src="../pic/voc_dataset_annotaion_1.jpg" width="50%"/>
  </center>

* ImageSets中：

  <center class="half">    
      <img src="../pic/voc_image.jpg" width="60%"/>     
  </center>

  * Action存储人的动作，Layout存储人的部位，Segmentation存储分割，Main存储检测索引;
  
  * Main中，每个类都有对应的classname_train.txt、classname_val.txt和classname_trainval.txt三个索引文件，分别对应训练集，验证集和训练验证集（即训练集+验证集）。训练集：5717、验证集：5823 和trainval：11540。
  
    <center class="half">    
        <img src="../pic/voc_main.jpg" width="50%"/>     
        <img src="../pic/voc_main_1.jpg" width="40%"/>
    </center>
  
    每个类别的数量及图片都是一样的，区别在name_train.txt name_val.txt中对name类别进行了划分，对应的类别用1表示，否则用-1表示
  
  * VOC2012中没有了测试集，而是采用在线评测的方式；

* SegmentationClass：语义分割图片：不同类别之间用不同颜色区分，不分割背景.

  <center class="half">    
      <img src="../pic/segClass.png" width="40%"/>     
  </center>
  
* SegmentationObject：实例分割: 不用类别之间及相同类别之间均用不同颜色区分，不分割背景.

  <center class="half">    
      <img src="../pic/segObject.png" width="40%"/>     
  </center>

## 2. 生成TFRecord格式数据集

程序位置 ./dataset/convert_tfrecords.py

将训练集/验证集存储为TFRecord文件，结果保存在 ./dataset/tfrecords/目录下

```bash
python dataset/convert_tfrecords.py \
    --dataset_directory=./dataset/PASCAL_VOC \
    --output_directory=./dataset/tfrecords
```

## 3. 训练

先下载VGG16预训练权重[**pre-trained VGG-16 model (reduced-fc)**](https://drive.google.com/drive/folders/184srhbt8_uvLKeWW_Yo8Mc5wTyc0lJT7)并解压到./model目录

开启训练：

```python
python train_ssd.py
```

### 3.1 训练参数说明

* num_readers: 并行读取数据的数量；

* num_preprocessing_threads：创建batch数据的线程数；
* num_cpu_threads: 训练时CPU线程数;
* gpu_memory_fraction: GPU显存使用率;
* data_dir: LMDB数据集路径;
* num_classes: 数据集类别数+1;
* model_dir: 模型训练结果保存路径;
* log_every_n_steps：终端显示日志频率;
* save_summary_steps：评价指标(loss,accuracy)保存频率;
* save_checkpoints_secs: checkpoints保存频率(单位：秒);
* train_image_size: 输入图片大小;
* train_epochs：训练的轮数;
* max_number_of_steps：训练的最大步数;
* batch_size：batch size大小;
* data_format: 数据格式(通道在前 or 通道在后);
* negative_ratio：loss处正负样本比例(1:3);
* match_threshold：loss match阈值;
* neg_threshold: loss负样本阈值;
* tf_random_seed：随机数种子;
* weight_decay：权重衰减系数;
* momentum：动量系数;
* learning_rate：学习率大小;
* end_learning_rate：学习率衰减最小值;
* decay_boundaries: 学习率衰减边界(step数);
* lr_decay_factors：学习率衰减系数;
* checkpoint_path: fine-tune时，已训练模型的checkpoint保存路径;
* checkpoint_model_scope: 模型里面的作用域名
* model_scope：模型域名;
* checkpoint_exclude_scopes: 加载模型时需要排除的变量;
* ignore_missing_vars: 加载checkpoints时忽略丢失的变量;
* multi_gpu：是否使用GPU多卡训练;



### 3.2 计算mAP值

```python
python eval_ssd.py
python voc_eval.py
```

<center class="half">    
    <img src="../pic/mAP.jpg" width="40%"/>     
</center>

论文中测试结果：

```latex
mAP: 74.3%
```

### 3.3 训练结果可视化

使用tensorboard工具可视化log文件

```python
tensorboard --logdir=./logs
```

<center class="half">    
    <img src="../pic/loss.jpg" width="40%"/>     
</center>

训练时batch_size设置的很小，所以图中loss会出现震荡。

## 4. 检测

```python
python val_ssd.py \
--image_path  ./demo/test.jpg \
--image_out_path  ./demo/test_result.jpg
```

输出结果：

<center class="half">    
    <img src="../pic/test_result.jpg" width="40%"/>     
</center>

