# Mask2Former baseline for Component, Crack

## 简介
使用Mask2Former模型在Electronic Component, Crack数据集上进行语义实例分割任务，Mask2Former模型是CVPR2022所提出的SOTA模型，适用于通用的分割任务（实例分割、语义分割、全景分割）

## 使用说明
在```./myconfig``` 文件夹内修改对应任务的yaml配置文件，运行```train.py```进行训练测试
示例：
```
python train.py --num-gpus 1  --config-file ./myconfig/elec/instance maskformer2_R50_bs16_50ep.yaml 
```
## 结果展示

### Electronic Component
```
|   AP   |  AP50  |  AP75  |   APm  |   APl  |
|:------:|:------:|:------:|:------:|:------:|
| 68.260 | 87.728 | 75.141 | 69.150 | 94.440 |
```

### Crack
```
IOU: 85.02
```