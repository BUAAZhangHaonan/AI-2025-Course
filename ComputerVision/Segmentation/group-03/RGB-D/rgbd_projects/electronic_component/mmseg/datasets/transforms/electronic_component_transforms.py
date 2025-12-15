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


@TRANSFORMS.register_module()
class LoadDepthFromFile(BaseTransform):
    """
    从.npy文件加载深度图并归一化

    Args:
        to_float32 (bool): 是否转换为float32. Default: True
        normalize (bool): 是否归一化深度值到[0, 1]. Default: True
        depth_scale (float): 深度缩放因子，用于将深度值缩放到合理范围. Default: 1.0

    Example:
        >>> transform = LoadDepthFromFile(normalize=True)
        >>> results = dict(depth_path='path/to/depth.npy')
        >>> results = transform(results)
        >>> print(results['depth'].shape)  # (H, W, 1)
    """

    def __init__(self, to_float32=True, normalize=True, depth_scale=1.0):
        super().__init__()
        self.to_float32 = to_float32
        self.normalize = normalize
        self.depth_scale = depth_scale

    def transform(self, results: dict) -> dict:
        """
        加载并处理深度图

        Args:
            results: 包含depth_path的字典

        Returns:
            添加了depth字段的results
        """
        if 'depth_path' not in results:
            raise KeyError('depth_path not found in results')

        # 加载深度图 (.npy文件)
        depth = np.load(results['depth_path'])

        # 确保深度图是2D的
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]  # 取第一个通道

        # 应用深度缩放
        depth = depth * self.depth_scale

        # 归一化到[0, 1]
        if self.normalize:
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = np.zeros_like(depth)

        # 转换为float32
        if self.to_float32:
            depth = depth.astype(np.float32)

        # 添加通道维度 (H, W) -> (H, W, 1)
        depth = depth[:, :, np.newaxis]

        results['depth'] = depth
        results['depth_path'] = results['depth_path']

        # 将depth添加到seg_fields，这样RandomFlip等transform会自动处理它
        if 'seg_fields' not in results:
            results['seg_fields'] = []
        results['seg_fields'].append('depth')

        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'to_float32={self.to_float32}, '
                f'normalize={self.normalize}, '
                f'depth_scale={self.depth_scale})')


@TRANSFORMS.register_module()
class ConcatRGBD(BaseTransform):
    """
    将RGB图像和深度图拼接为4通道RGBD图像

    将3通道RGB图像 (H, W, 3) 和单通道深度图 (H, W, 1)
    拼接为4通道RGBD图像 (H, W, 4)

    如果深度图和RGB图像尺寸不匹配，会自动resize深度图到RGB图像的尺寸。

    Example:
        >>> transform = ConcatRGBD()
        >>> results = dict(img=np.random.rand(512, 512, 3),
        ...                depth=np.random.rand(512, 512, 1))
        >>> results = transform(results)
        >>> print(results['img'].shape)  # (512, 512, 4)
    """

    def __init__(self):
        super().__init__()

    def transform(self, results: dict) -> dict:
        """
        拼接RGB和深度通道

        Args:
            results: 包含img和depth的字典

        Returns:
            拼接后的results，img字段变为4通道
        """
        if 'img' not in results:
            raise KeyError('img not found in results')
        if 'depth' not in results:
            raise KeyError('depth not found in results')

        img = results['img']
        depth = results['depth']

        # 基本维度检查
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f'Expected RGB image with shape (H, W, 3), got {img.shape}')

        # 深度图可能是 (H, W, 1) 或 (H, W)，统一处理
        if len(depth.shape) == 2:
            depth = depth[:, :, np.newaxis]
        elif len(depth.shape) != 3 or depth.shape[2] != 1:
            raise ValueError(f'Expected depth map with shape (H, W, 1) or (H, W), got {depth.shape}')

        # 如果深度图尺寸与RGB图像不匹配，自动resize深度图
        if img.shape[:2] != depth.shape[:2]:
            import cv2
            # 先squeeze到2D进行resize，避免cv2对单通道3D数组的处理问题
            depth_2d = depth.squeeze()
            depth_2d = cv2.resize(
                depth_2d,
                (img.shape[1], img.shape[0]),  # (width, height)
                interpolation=cv2.INTER_LINEAR
            )
            # 恢复通道维度
            depth = depth_2d[:, :, np.newaxis]

        # 确保深度图和RGB图像的数据类型一致
        if depth.dtype != img.dtype:
            depth = depth.astype(img.dtype)

        # 拼接RGB和深度 -> RGBD (H, W, 4)
        rgbd = np.concatenate([img, depth], axis=2)

        results['img'] = rgbd
        results['img_shape'] = rgbd.shape[:2]

        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

