# YOLO 消防器材检测模型训练指南

## 一、环境准备

### 1.1 克隆训练框架
```bash
# 使用指定的 ultralytics fork（支持 YOLO26）
git clone https://github.com/liekongxingyu/ultralytics_fork.git
cd ultralytics_fork
pip install -e .
```

### 1.2 验证安装
```bash
yolo version
```

---

## 二、数据集准备

### 2.1 数据集来源

**方式一：从 Roboflow 获取**
1. 访问 [Roboflow Universe](https://universe.roboflow.com/)
2. 搜索关键词：
   - `fire extinguisher` (灭火器)
   - `fire bucket` (消防水桶)
   - `fire blanket` (灭火毯)
   - `welding` / `hot work` (动火作业)
3. 下载 YOLO 格式数据集

**方式二：自行标注**
1. 使用 [LabelImg](https://github.com/heartexlabs/labelImg) 或 [Roboflow Annotate](https://roboflow.com/)
2. 标注类别：
   - `fire_extinguisher` - 灭火器
   - `fire_bucket` - 消防水桶
   - `fire_blanket` - 灭火毯
   - `welding_point` - 动火点/焊接点

### 2.2 数据集目录结构
```
datasets/
└── fire_equipment/
    ├── train/
    │   ├── images/
    │   │   ├── img001.jpg
    │   │   ├── img002.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── img001.txt
    │       ├── img002.txt
    │       └── ...
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

### 2.3 标注格式（YOLO 格式）
每个 `.txt` 文件对应一张图片，每行一个目标：
```
<class_id> <x_center> <y_center> <width> <height>
```
- 坐标值为相对值（0-1 范围）
- class_id: 0=fire_extinguisher, 1=fire_bucket, 2=fire_blanket, 3=welding_point

示例 `img001.txt`:
```
0 0.45 0.62 0.12 0.35
3 0.72 0.48 0.08 0.15
```

### 2.4 创建数据集配置文件

创建 `fire_equipment.yaml`:
```yaml
# 数据集路径（相对于训练脚本或绝对路径）
path: datasets/fire_equipment
train: train/images
val: valid/images
test: test/images

# 类别数量
nc: 4

# 类别名称
names:
  0: fire_extinguisher
  1: fire_bucket
  2: fire_blanket
  3: welding_point
```

---

## 三、模型训练

### 3.1 基础训练命令
```bash
cd ultralytics_fork

# 使用 YOLO26 训练
yolo train \
    model=yolo26.yaml \
    data=fire_equipment.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    project=runs/fire_equipment \
    name=exp1
```

### 3.2 参数说明
| 参数 | 说明 | 建议值 |
|------|------|--------|
| `model` | 模型配置 | yolo26.yaml |
| `data` | 数据集配置 | fire_equipment.yaml |
| `epochs` | 训练轮数 | 100-300 |
| `imgsz` | 输入图像尺寸 | 640 |
| `batch` | 批次大小 | 8-32（根据显存调整）|
| `device` | 训练设备 | 0（GPU）或 cpu |
| `patience` | 早停耐心值 | 50 |
| `lr0` | 初始学习率 | 0.01 |
| `lrf` | 最终学习率比例 | 0.01 |

### 3.3 使用预训练权重（推荐）
```bash
yolo train \
    model=yolo26.pt \
    data=fire_equipment.yaml \
    epochs=100 \
    imgsz=640
```

### 3.4 恢复训练
```bash
yolo train resume model=runs/fire_equipment/exp1/weights/last.pt
```

---

## 四、训练监控

### 4.1 TensorBoard
```bash
tensorboard --logdir runs/fire_equipment
```
访问 http://localhost:6006 查看训练曲线

### 4.2 训练输出
训练完成后，权重保存在：
```
runs/fire_equipment/exp1/
├── weights/
│   ├── best.pt      # 最佳模型（验证集表现最好）
│   └── last.pt      # 最后一轮模型
├── results.csv      # 训练指标
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
└── ...
```

---

## 五、模型验证

### 5.1 验证命令
```bash
yolo val \
    model=runs/fire_equipment/exp1/weights/best.pt \
    data=fire_equipment.yaml
```

### 5.2 关键指标
- **mAP50**: 50% IoU 阈值下的平均精度
- **mAP50-95**: 50%-95% IoU 阈值下的平均精度
- **Precision**: 精确率
- **Recall**: 召回率

建议目标：mAP50 > 0.8

---

## 六、模型导出与部署

### 6.1 复制到项目
```bash
# 将最佳模型复制到项目目录
cp runs/fire_equipment/exp1/weights/best.pt \
   /path/to/SOD_3/fire_detection/weights/fire_equipment.pt
```

### 6.2 测试推理
```python
from ultralytics import YOLO

# 加载模型
model = YOLO("fire_detection/weights/fire_equipment.pt")

# 推理测试
results = model("test_image.jpg", conf=0.5)

# 显示结果
results[0].show()
```

### 6.3 导出其他格式（可选）
```bash
# 导出为 ONNX
yolo export model=best.pt format=onnx

# 导出为 TensorRT（需要 NVIDIA GPU）
yolo export model=best.pt format=engine
```

---

## 七、常见问题

### Q1: 显存不足
```bash
# 减小 batch size
yolo train ... batch=8

# 或使用混合精度训练
yolo train ... amp=True
```

### Q2: 训练不收敛
- 检查数据集标注是否正确
- 尝试降低学习率：`lr0=0.001`
- 增加训练轮数

### Q3: 过拟合
- 增加数据增强
- 使用更小的模型
- 添加正则化

### Q4: 类别不平衡
在 `fire_equipment.yaml` 中添加类别权重：
```yaml
# 类别权重（数量少的类别给更高权重）
class_weights: [1.0, 2.0, 2.0, 1.5]
```

---

## 八、数据增强建议

训练时默认启用的增强：
- 马赛克增强 (mosaic)
- 随机翻转 (fliplr, flipud)
- 色彩抖动 (hsv_h, hsv_s, hsv_v)
- 缩放 (scale)
- 平移 (translate)

可以通过参数调整：
```bash
yolo train ... \
    mosaic=1.0 \
    mixup=0.1 \
    copy_paste=0.1
```

---

## 九、推荐训练流程

1. **小规模测试**：先用少量数据（100张）训练 10 轮，验证流程
2. **完整训练**：使用全部数据训练 100-300 轮
3. **验证评估**：检查 mAP、混淆矩阵
4. **调优迭代**：根据结果调整参数或补充数据
5. **部署测试**：在实际场景中测试效果
