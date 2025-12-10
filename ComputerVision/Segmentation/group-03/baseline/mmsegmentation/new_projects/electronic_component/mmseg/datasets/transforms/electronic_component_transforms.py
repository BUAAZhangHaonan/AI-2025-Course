"""
Electronic Component数据集的自定义transforms
"""

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ConvertInstanceToSemantic(BaseTransform):
    """
    将实例分割标签转换为语义分割标签
    
    Electronic Component数据集的mask包含实例ID（0-255）：
    - 0: 背景
    - 1-254: 不同的电子元件实例
    - 255: 边界/ignore区域
    
    转换规则：
    - 0 → 0 (背景)
    - 1-254 → 1 (电子元件，合并所有实例)
    - 255 → 255 (ignore_index，保持不变)
    
    Example:
        >>> transform = ConvertInstanceToSemantic()
        >>> results = dict(gt_seg_map=np.array([0, 1, 2, 100, 255]))
        >>> results = transform(results)
        >>> print(results['gt_seg_map'])  # [0, 1, 1, 1, 255]
    """
    
    def __init__(self):
        """初始化转换器"""
        super().__init__()
    
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
            
            # 🔥 处理RGB格式的mask：如果是3通道，取第一个通道
            # Electronic Component的mask是RGB格式的PNG，但三个通道值相同
            if len(gt_seg_map.shape) == 3 and gt_seg_map.shape[2] == 3:
                # 取第一个通道（三个通道值相同）
                gt_seg_map = gt_seg_map[:, :, 0]
            
            # 创建新的标签图
            # 0 保持为 0 (背景)
            # 1-254 映射为 1 (电子元件)
            # 255 保持为 255 (ignore)
            new_seg_map = np.zeros_like(gt_seg_map)
            new_seg_map[gt_seg_map == 0] = 0  # 背景
            new_seg_map[(gt_seg_map > 0) & (gt_seg_map < 255)] = 1  # 电子元件
            new_seg_map[gt_seg_map == 255] = 255  # ignore
            
            results['gt_seg_map'] = new_seg_map
        
        return results
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

