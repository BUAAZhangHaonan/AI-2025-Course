# 训练日志可视化工具使用说明

## 🎯 功能

自动读取MMSegmentation训练日志，生成以下可视化图表：

1. **Loss曲线** - 总Loss、Decode Loss、Auxiliary Loss
2. **学习率曲线** - 学习率变化趋势
3. **mIoU曲线** - 验证集mIoU，标注最佳值
4. **准确率曲线** - 训练准确率、验证aAcc和mAcc
5. **综合图表** - 2x2四合一图表

---

## 📦 依赖

确保已安装matplotlib：
```bash
pip install matplotlib
```

---

## 🚀 使用方法

### 方法1：指定日志文件（推荐）

```bash
python tools/visualize_log.py \
    --log work_dirs/electronic_component_rgbd_pspnet/20251028_231641/20251028_231641.log \
    --output visualizations
```

### 方法2：自动查找最新日志

```bash
python tools/visualize_log.py \
    --work-dir work_dirs/electronic_component_rgbd_pspnet \
    --output visualizations
```

---

## 📊 输出结果

运行后会在输出目录生成5张图片：

```
visualizations/
├── loss_curve.png           # Loss曲线
├── lr_curve.png             # 学习率曲线
├── miou_curve.png           # mIoU曲线（标注最佳值）
├── accuracy_curve.png       # 准确率曲线（训练+验证）
└── training_summary.png     # 综合图表（2x2）
```

---

## 💡 快速示例

### 示例1：可视化RGBD模型训练

```bash
# 方式A：指定具体日志
python tools/visualize_log.py \
    --log work_dirs/electronic_component_rgbd_pspnet/20251028_231641/20251028_231641.log \
    --output results/rgbd_training_curves

# 方式B：自动查找最新日志
python tools/visualize_log.py \
    --work-dir work_dirs/electronic_component_rgbd_pspnet \
    --output results/rgbd_training_curves
```

### 示例2：可视化RGB基线模型

```bash
python tools/visualize_log.py \
    --work-dir work_dirs/electronic_component_pspnet \
    --output results/rgb_training_curves
```

### 示例3：对比多个模型

```bash
# 可视化RGB基线
python tools/visualize_log.py \
    --work-dir work_dirs/electronic_component_pspnet \
    --output results/curves_rgb

# 可视化RGBD拼接
python tools/visualize_log.py \
    --work-dir work_dirs/electronic_component_rgbd_pspnet \
    --output results/curves_rgbd

# 可视化深度注意力V2
python tools/visualize_log.py \
    --work-dir work_dirs/electronic_component_depth_attention_v2_pspnet \
    --output results/curves_attention_v2
```

然后对比三个目录的图表。

---

## 🔍 参数说明

| 参数 | 必需 | 说明 | 示例 |
|------|------|------|------|
| `--log` | 二选一 | 指定日志文件路径 | `--log xxx.log` |
| `--work-dir` | 二选一 | 工作目录，自动查找最新日志 | `--work-dir work_dirs/xxx` |
| `--output` | 否 | 输出目录，默认`visualizations` | `--output my_plots` |

**注意**：`--log` 和 `--work-dir` 必须指定一个。

---

## 📈 图表说明

### 1. loss_curve.png
- **X轴**：训练迭代数
- **Y轴**：Loss值
- **曲线**：
  - Total Loss（蓝色）- 总损失
  - Decode Loss（橙色）- 解码头损失
  - Auxiliary Loss（绿色）- 辅助头损失

**观察要点**：
- ✅ Loss应该平滑下降
- ⚠️  如果出现突增或NaN，说明训练不稳定

### 2. lr_curve.png
- **X轴**：训练迭代数
- **Y轴**：学习率（科学计数法）
- **曲线**：学习率调度策略

**观察要点**：
- Poly LR：逐渐衰减
- Step LR：阶梯式下降

### 3. miou_curve.png
- **X轴**：验证迭代数（每500次）
- **Y轴**：mIoU百分比
- **曲线**：验证mIoU变化
- **红色虚线**：标注最佳mIoU值和对应迭代数

**观察要点**：
- ✅ mIoU应该逐渐上升
- ✅ 找到最佳checkpoint

### 4. accuracy_curve.png
- **左图**：训练分割准确率
- **右图**：验证准确率
  - aAcc：全部像素准确率
  - mAcc：平均类别准确率

### 5. training_summary.png
- **四合一图表**：Loss + LR + mIoU + Accuracy
- 方便整体观察训练过程

---

## ⚠️ 常见问题

### Q1: 提示"未找到学习率数据"

**原因**：某些日志版本可能不记录lr

**解决**：忽略，其他图表仍会生成

### Q2: 图表中文乱码

**解决**：
```bash
# 安装中文字体
sudo apt-get install fonts-wqy-microhei

# 或修改脚本，使用英文标题
```

### Q3: ImportError: No module named 'matplotlib'

**解决**：
```bash
pip install matplotlib
```

### Q4: 只想生成某个图表

**解决**：编辑脚本，注释掉不需要的 `plot_xxx()` 函数调用

---

## 🎨 自定义

### 修改图表样式

编辑 `tools/visualize_log.py`，找到对应的绘图函数：

```python
# 修改线条颜色
ax.plot(..., color='red', ...)  # 改为你喜欢的颜色

# 修改线宽
ax.plot(..., linewidth=3, ...)

# 修改DPI（清晰度）
plt.savefig(..., dpi=300)  # 默认150
```

### 添加新指标

在 `LogParser` 类中：
1. 添加数据字段
2. 在 `_parse_train_line()` 或 `_parse_val_line()` 中解析
3. 在 `LogVisualizer` 中添加绘图函数

---

## 📝 输出示例

运行成功后会看到：

```
📖 正在读取日志文件: work_dirs/xxx/xxx.log
✅ 解析完成:
   - 训练数据点: 10000
   - 验证数据点: 20

📊 正在生成可视化图表...

✅ 所有图表已保存到: visualizations
   - loss_curve.png
   - lr_curve.png
   - miou_curve.png
   - accuracy_curve.png
   - training_summary.png

🎉 完成！
```

---

## 🔧 高级用法

### 批量可视化多个训练

```bash
#!/bin/bash
# 批量可视化所有训练

for work_dir in work_dirs/*/; do
    model_name=$(basename $work_dir)
    echo "Processing $model_name..."

    python tools/visualize_log.py \
        --work-dir "$work_dir" \
        --output "visualizations/$model_name"
done

echo "All done!"
```

### 在训练过程中实时可视化

```bash
# 训练时每隔一段时间自动生成图表
while true; do
    python tools/visualize_log.py \
        --work-dir work_dirs/electronic_component_depth_attention_v2_pspnet \
        --output visualizations/realtime

    echo "Updated at $(date)"
    sleep 300  # 每5分钟更新一次
done
```

---

## 📚 技术细节

### 支持的日志格式

脚本支持MMSegmentation标准日志格式：

**训练行**：
```
Iter(train) [  50/10000]  lr: 9.9500e-03  loss: 0.5123  decode.loss_ce: 0.3234  decode.acc_seg: 85.12
```

**验证行**：
```
Iter(val) [110/110]    aAcc: 93.09  mIoU: 82.13  mAcc: 87.09
```

**Checkpoint行**：
```
Saving checkpoint at 500 iterations
```

### 性能

- 解析速度：~100,000行/秒
- 内存占用：<100MB（对于10万行日志）
- 图表生成：<5秒

---

## ✅ 总结

使用这个脚本，您可以：
- 📊 快速可视化训练过程
- 🔍 发现训练问题（loss突增、不收敛等）
- 📈 找到最佳checkpoint
- 🆚 对比不同模型的训练曲线

**推荐工作流**：
1. 训练完成后立即运行可视化
2. 观察mIoU曲线，确认最佳模型
3. 检查loss曲线，确认训练稳定性
4. 对比不同模型的图表

祝使用愉快！🎉
