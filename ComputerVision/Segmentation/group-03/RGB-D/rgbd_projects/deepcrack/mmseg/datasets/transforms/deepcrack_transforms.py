"""
DeepCrack数据集的自定义transforms
"""

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ConvertDeepCrackLabels(BaseTransform):
    """
    将DeepCrack标签从255转换为1
    
    DeepCrack数据集的标签格式：
    - 0: 背景
    - 255: 裂缝
    
    需要转换为：
    - 0: 背景
    - 1: 裂缝
    """
    
    def transform(self, results: dict) -> dict:
        """
        转换标签值
        
        Args:
            results: 包含gt_seg_map的字典
            
        Returns:
            转换后的results
        """
        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            # 将255转换为1
            gt_seg_map[gt_seg_map == 255] = 1
            results['gt_seg_map'] = gt_seg_map
        
        return results
    
    def __repr__(self) -> str:
        return self.__class__.__name__

