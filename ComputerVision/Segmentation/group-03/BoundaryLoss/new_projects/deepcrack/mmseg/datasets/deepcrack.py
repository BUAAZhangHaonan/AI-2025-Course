# Copyright (c) OpenMMLab. All rights reserved.
"""
DeepCrack Dataset for Crack Detection

This dataset is designed for crack detection tasks using the DeepCrack dataset.
The dataset contains images and corresponding binary segmentation masks for crack detection.

Dataset structure:
├── data/
│   ├── DeepCrack/
│   │   ├── train_img/          # Training images (JPG format)
│   │   ├── train_lab/          # Training labels (PNG format, binary mask)
│   │   ├── test_img/           # Test images (JPG format)
│   │   └── test_lab/           # Test labels (PNG format, binary mask)

Reference:
@article{liu2019deepcrack,
  title={DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation},
  author={Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xie, Renping and Li, Li},
  journal={Neurocomputing},
  volume={338},
  pages={139--153},
  year={2019},
  doi={10.1016/j.neucom.2019.01.036}
}
"""

import os
import os.path as osp
from typing import List, Optional

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class DeepCrackDataset(BaseSegDataset):
    """DeepCrack dataset for crack detection.
    
    This dataset is used for binary crack segmentation tasks.
    The dataset contains images with corresponding binary masks where:
    - 0: background (non-crack pixels)
    - 1: crack pixels
    
    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as
            specify classes to load. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img_path=None, seg_map_path=None).
        img_suffix (str): Suffix of images. Default: '.jpg'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Defaults to None which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=True``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default to False.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
    """
    
    # 数据集元信息定义
    METAINFO = dict(
        # 类别定义：背景和裂缝
        classes=('background', 'crack'),
        # 调色板定义：背景为黑色(0,0,0)，裂缝为白色(255,255,255)
        palette=[[0, 0, 0], [255, 255, 255]],
        # 类别名称映射
        class_names=['background', 'crack']
    )

    def __init__(self,
                 ann_file: str = '',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img_path='', seg_map_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[List[int]] = None,
                 serialize_data: bool = True,
                 pipeline: List = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        """
        初始化DeepCrack数据集
        
        Args:
            ann_file: 标注文件路径，对于DeepCrack数据集通常为空字符串
            img_suffix: 图像文件后缀，默认为'.jpg'
            seg_map_suffix: 分割掩码文件后缀，默认为'.png'
            metainfo: 数据集元信息
            data_root: 数据根目录
            data_prefix: 数据路径前缀，包含img_path和seg_map_path
            filter_cfg: 数据过滤配置
            indices: 数据索引
            serialize_data: 是否序列化数据
            pipeline: 数据处理管道
            test_mode: 是否为测试模式
            lazy_init: 是否延迟初始化
            max_refetch: 最大重试次数
            ignore_index: 忽略的标签索引
            reduce_zero_label: 是否将标签0标记为忽略
            backend_args: 后端参数
        """
        # 调用父类初始化方法
        super().__init__(
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            backend_args=backend_args
        )

    def load_data_list(self) -> List[dict]:
        """
        加载数据列表
        
        对于DeepCrack数据集，我们需要从img_dir和ann_dir中自动匹配图像和标签文件
        文件名（不包括扩展名）必须完全匹配
        
        Returns:
            List[dict]: 数据信息列表，每个元素包含img_path和seg_map_path
        """
        # 获取图像和标签目录路径
        # 如果data_prefix中的路径已经包含data_root，则直接使用
        if self.data_prefix['img_path'].startswith(self.data_root):
            img_dir = self.data_prefix['img_path']
            ann_dir = self.data_prefix['seg_map_path']
        else:
            img_dir = osp.join(self.data_root, self.data_prefix['img_path'])
            ann_dir = osp.join(self.data_root, self.data_prefix['seg_map_path'])
        
        print(f"Debug: self.data_root = {self.data_root}")
        print(f"Debug: self.data_prefix = {self.data_prefix}")
        print(f"Debug: img_dir = {img_dir}")
        print(f"Debug: ann_dir = {ann_dir}")
        
        
        # 检查目录是否存在
        if not osp.exists(img_dir):
            raise FileNotFoundError(f'Image directory {img_dir} does not exist')
        if not osp.exists(ann_dir):
            raise FileNotFoundError(f'Annotation directory {ann_dir} does not exist')
        
        # 获取所有图像文件
        img_files = []
        for file in os.listdir(img_dir):
            if file.endswith(self.img_suffix):
                img_files.append(file)
        
        data_list = []
        
        # 为每个图像文件创建数据项
        for img_file in img_files:
            # 获取文件名（不包括扩展名）
            img_name = osp.splitext(img_file)[0]
            # 对应的标签文件名
            ann_file = img_name + self.seg_map_suffix
            
            # 检查标签文件是否存在
            ann_path = osp.join(ann_dir, ann_file)
            if not os.path.exists(ann_path):
                print(f'Warning: Annotation file {ann_path} does not exist, skipping {img_file}')
                continue
            
            # 创建数据项
            data_info = dict(
                img_path=osp.join(img_dir, img_file),
                seg_map_path=ann_path,
                seg_fields=['gt_seg_map'],  # 指定分割字段，应该是gt_seg_map
                reduce_zero_label=self.reduce_zero_label  # 添加reduce_zero_label字段
            )
            data_list.append(data_info)
        
        print(f'Loaded {len(data_list)} samples from DeepCrack dataset')
        return data_list
