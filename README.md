# Yolov5
DeepVAC-compliant Yolov5 implementation   
- 20210202: add Yolov5S & Yolov5L      
- TODO

# 简介
本项目实现了符合DeepVAC规范的Yolov5。

**项目依赖**

- deepvac >= 0.2.5
- pytorch >= 1.6.0
- torchvision >= 0.7.0

# 如何运行本项目

## 1. 阅读[DeepVAC规范](https://github.com/DeepVAC/deepvac)
可以粗略阅读，建立起第一印象。

## 2. 准备运行环境
使用Deepvac规范指定[Docker镜像](https://github.com/DeepVAC/deepvac#2-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)。

## 3. 准备数据集
```bash
bash data/script/get_coco.sh
```
获取coco2017数据集。     

## 4. 修改配置文件

修改config.py文件。主要修改内容：
- 指定预训练模型路径。     
```python
config.model_path = <pretrained-model-path>
```

- 指定训练集、验证集目录和对应的标注txt文件, custom dataset必须为标准COCO格式数据集。     
```python
config.train.img_folder = <your-custom-train-img-folder>
config.train.annotation = <your-custom-train-annotation-path>

config.val.img_folder = <your-custom-val-img-folder>
config.val.annotation = <your-custom-val-annotation-path>
```

- 指定测试集目录，可以通过plot参数来进行可视化。           
```python
config.test.img_folder = <test-images-folder>
config.test.plot = True
```

- 修改分类数
```
config.class_num = 80
```

## 5. 训练

### 5.1 单卡训练
执行命令：
```bash
python3 train.py
```

### 5.2 分布式训练

在config.py中修改如下配置：
```python
#dist_url，单机多卡无需改动，多机训练一定要修改
config.dist_url = "tcp://localhost:27030"

#rank的数量，一定要修改
config.world_size = 2
```
然后执行命令：

```bash
python train.py --rank 0 --gpu 0
python train.py --rank 1 --gpu 1
```

## 6. 测试

指定要测试模型的路径，在config.py指定待测模型路径：

```python
config.model_path = <trained-model-path>
```

然后运行测试脚本：

```bash
python3 test.py
```

## 7， 更多功能
如果要在本项目中开启如下功能：
- 预训练模型加载
- checkpoint加载
- 使用tensorboard
- 启用TorchScript
- 转换ONNX
- 转换NCNN
- 转换CoreML
- 开启量化
- 开启自动混合精度训练

请参考[DeepVAC](https://github.com/DeepVAC/deepvac)。
