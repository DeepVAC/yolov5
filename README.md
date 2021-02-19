# Yolov5
DeepVAC-compliant Yolov5 implementation   

# 简介
本项目实现了符合DeepVAC规范的Yolov5   

**项目依赖**

- deepvac >= 0.2.6
- pytorch >= 1.8.0
- torchvision >= 0.7.0

# 如何运行本项目

## 1. 阅读[DeepVAC规范](https://github.com/DeepVAC/deepvac)
可以粗略阅读，建立起第一印象   

## 2. 准备运行环境
使用Deepvac规范指定[Docker镜像](https://github.com/DeepVAC/deepvac#2-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)   

## 3. 准备数据集
- 获取coco2017数据集      
浏览器操作：     
[coco2017labels.zip](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip)     
[train2017.zip](http://images.cocodataset.org/zips/train2017.zip)     
[val2017.zip](http://images.cocodataset.org/zips/val2017.zip)     
[test2017.zip](http://images.cocodataset.org/zips/test2017.zip)       

命令行操作：   
```bash
curl -L https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip -o coco2017labels.zip
mkdir -p data/coco
unzip -q coco2017labels -d data/coco   
rm coco2017labels.zip
```

- 解压coco2017数据集

- 数据集配置
在config.py文件中作如下配置：     

```python
config.train.img_folder = <train2017-extract-folder/train2017/>      
config.train.annotation = <coco2017labels-extract-folder/instances_train2017.json>       
config.val.img_folder = <val2017-extract-folder/val2017/>           
config.val.annotation = <coco2017labels-extract-folder/instances_val2017.json>        
config.test.input_dir = <test2017-extract-folder/test2017/>           
config.test.plot = True  # 可视化             
```

- 如果是自己的数据集，那么必须要符合标准coco标注格式

## 4. 训练相关配置

- 指定预训练模型路径(config.model_path)       
[yolov5s & yolov5l](https://pan.baidu.com/share/init?surl=oA4uZUlWUtEq2dOMlBZ8hg) 提取码: g4tu
- 指定训练分类数量(config.class_num)    
- 是否采用混合精度训练(config.amp)     
- 是否采用ema策略(config.ema)      
- 是否采用梯度积攒到一定数量在进行反向更新梯度策略(config.nominal_batch_factor)     
- dataloader相关配置(config.train, config.val)     

```python
config.model_path = <pretrained-model-path>
config.class_num = 80
config.amp = False
config.ema = True
config.nominal_batch_factor = 4  # 在样本数量积攒至64后再进行反向更新梯度
config.train.shuffle = True
config.train.batch_size = 16
config.train.num_workers = 8
config.train.pin_memory = True
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

- 测试相关配置

```python
config.class_num = <class_num>
config.test.input_dir = <test-data-path>
config.test.idx_to_cls = <class-index-to-class-name-maps>
config.test.plot = <True or False>  # optional
config.test.plot_dir = <path-to-save-images>  # optional
```

- 加载模型(*.pth)

```python
config.model_path = <trained-model-path>
```


- 运行测试脚本：

```bash
python3 test.py
```

## 7. 使用torchscript模型
如果训练过程中未开启config.script_model_path开关，可以在测试过程中转化torchscript模型     
- 转换torchscript模型(*.pt)     

```python
config.ema = False
config.script_model_path = "output/script.pt"
```
  按照步骤6完成测试，torchscript模型将保存至config.script_model_path指定文件位置      

- 加载torchscript模型

```python
config.jit_model_path = <torchscript-model-path>
```

## 8. 使用静态量化模型
如果训练过程中未开启config.static_quantize_dir开关，可以在测试过程中转化静态量化模型     
- 转换静态模型(*.sq)     

```python
config.ema = False
config.static_quantize_dir = "output/script.sq"
```
  按照步骤6完成测试，静态量化模型将保存至config.static_quantize_dir指定文件位置      

- 加载静态量化模型

```python
config.jit_model_path = <static-quantize-model-path>
```
当前yolov5-1支持静态量化模型导出，但在test过程中会出现upsample错误，我们推测是pytorch bug导致了这个问题，目前这个bug已经加入TODO    


## 9. 更多功能
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

请参考[DeepVAC](https://github.com/DeepVAC/deepvac)

## 10. TODO
- 20210201 项目增加了对Yolov5S和Yolov5L的支持    
- 20210219 修复了torchscript模型C++推理代码在cuda上CUDNN_STATUS_INTER_ERROR问题(在modules/model.py中重写Fcous模块）     
- 修复在test过程中，静态量化模型报错问题    
- 增加对Yolov5M的支持    
- 增加对Yolov5x的支持    
- 时刻同步 https://github.com/ultralytics/yolov5
