#!/usr/bin/env python3
"""
训练脚本 - Electronic Component RGBD 语义分割

使用方法:
    # 单GPU训练
    python rgbd_projects/electronic_component/scripts/train_rgbd.py

    # 多GPU训练
    bash tools/dist_train.sh rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_rgbd_512x512_10k.py 2
"""

import os
import sys
import os.path as osp

# 添加项目根目录到路径
project_root = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from mmseg.apis import train_segmentor, init_segmentor
from mmengine.config import Config
from mmengine.runner import set_random_seed


def main():
    # 配置文件路径
    config_file = 'rgbd_projects/electronic_component/configs/electronic_component/pspnet_r50-electronic_rgbd_512x512_10k.py'

    # 检查配置文件是否存在
    if not osp.exists(config_file):
        raise FileNotFoundError(f'配置文件不存在: {config_file}')

    # 加载配置
    cfg = Config.fromfile(config_file)

    # 设置随机种子
    if hasattr(cfg, 'randomness') and cfg.randomness is not None:
        seed = cfg.randomness.get('seed', 0)
        set_random_seed(seed, deterministic=False)
        print(f'随机种子设置为: {seed}')

    # 创建工作目录
    os.makedirs(cfg.work_dir, exist_ok=True)
    print(f'工作目录: {cfg.work_dir}')

    # 显示配置信息
    print('=' * 80)
    print('RGBD 模型训练配置:')
    print(f'  - 配置文件: {config_file}')
    print(f'  - 数据根目录: {cfg.train_dataloader.dataset.data_root}')
    print(f'  - RGB图像路径: {cfg.train_dataloader.dataset.data_prefix["img_path"]}')
    print(f'  - 深度图路径: {cfg.train_dataloader.dataset.data_prefix["depth_path"]}')
    print(f'  - Backbone: {cfg.model.backbone.type} (in_channels={cfg.model.backbone.in_channels})')
    print(f'  - 训练迭代数: {cfg.train_cfg.max_iters}')
    print(f'  - 批次大小: {cfg.train_dataloader.batch_size}')
    print(f'  - 学习率: {cfg.optim_wrapper.optimizer.lr}')
    if 'load_from' in cfg and cfg.load_from:
        print(f'  - 预训练模型: {cfg.load_from}')
    print('=' * 80)

    # 初始化模型
    print('\n正在初始化模型...')
    model = init_segmentor(cfg, device='cuda:0')
    print('模型初始化完成!')

    # 开始训练
    print('\n开始训练...')
    train_segmentor(model, cfg)
    print('\n训练完成!')
    print(f'模型保存在: {cfg.work_dir}')


if __name__ == '__main__':
    main()
