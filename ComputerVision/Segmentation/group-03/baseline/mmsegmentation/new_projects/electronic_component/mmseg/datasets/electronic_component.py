# Copyright (c) OpenMMLab. All rights reserved.
"""
Electronic Component Dataset for Segmentation

This dataset is designed for electronic component segmentation tasks.
The dataset contains RGB images and corresponding segmentation masks.

Dataset structure:
├── data/
│   ├── electronic_component/
│   │   ├── images/
│   │   │   ├── train/          # Training RGB images (PNG)
│   │   │   ├── val/            # Validation RGB images (PNG)
│   │   │   └── test/           # Test RGB images (PNG)
│   │   ├── mask/
│   │   │   ├── train/          # Training segmentation masks (PNG)
│   │   │   ├── val/            # Validation segmentation masks (PNG)
│   │   │   └── test/           # Test segmentation masks (PNG)
│   │   └── annotations/        # COCO format annotations (optional)
"""

import os
import os.path as osp
from typing import List, Optional

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class ElectronicComponentDataset(BaseSegDataset):
    """Electronic Component dataset for segmentation.
    
    This dataset is used for segmenting electronic components from images.
    The task is to distinguish between:
    - 0: background
    - 1: electronic component (all instances merged)
    
    Note: The original mask contains instance IDs (0-255), but we convert
    them to semantic segmentation (background vs component).
    
    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset.
        data_root (str, optional): The root directory for data.
        data_prefix (dict, optional): Prefix for data paths.
        img_suffix (str): Suffix of images. Default: '.png'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default to False.
    """
    
    # 数据集元信息定义
    METAINFO = dict(
        # 类别定义：背景和电子元件
        classes=('background', 'component'),
        # 调色板定义：背景为黑色(0,0,0)，电子元件为红色(255,0,0)
        palette=[[0, 0, 0], [255, 0, 0]],
        # 类别名称映射
        class_names=['background', 'component']
    )

    def __init__(self,
                 ann_file: str = '',
                 img_suffix='.png',
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
        初始化Electronic Component数据集
        
        Args:
            ann_file: 标注文件路径，对于此数据集通常为空字符串
            img_suffix: 图像文件后缀，默认为'.png'
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
        
        对于Electronic Component数据集，我们需要从img_dir和ann_dir中
        自动匹配图像和标签文件。文件名必须完全匹配。
        
        Returns:
            List[dict]: 数据信息列表，每个元素包含img_path和seg_map_path
        """
        # 获取图像和标签目录路径
        if self.data_prefix['img_path'].startswith(self.data_root):
            img_dir = self.data_prefix['img_path']
            ann_dir = self.data_prefix['seg_map_path']
        else:
            img_dir = osp.join(self.data_root, self.data_prefix['img_path'])
            ann_dir = osp.join(self.data_root, self.data_prefix['seg_map_path'])
        
        print(f"[ElectronicComponent] Loading data from:")
        print(f"  - Images: {img_dir}")
        print(f"  - Masks:  {ann_dir}")
        
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
        for img_file in sorted(img_files):
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
                seg_fields=['gt_seg_map'],  # 指定分割字段
                reduce_zero_label=self.reduce_zero_label
            )
            data_list.append(data_info)
        
        print(f'[ElectronicComponent] Loaded {len(data_list)} samples')
        return data_list








