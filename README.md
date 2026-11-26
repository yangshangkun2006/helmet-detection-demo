# 安全帽佩戴检测（YOLO26n）

一个可演示的安全帽佩戴检测项目：用公开数据集微调 YOLO26n，提供**浏览器在线演示**和**本地 Python Web 界面**两种使用方式。

**在线演示（GitHub Pages）：** https://yangshangkun2006.github.io/helmet-detection-demo/

在线版不需要服务器：ONNX 模型与推理运行时都在浏览器本地加载，上传的图片不会离开你的电脑。首次打开需下载约 21 MB 的模型/运行时，之后走浏览器缓存。

## 能做什么

- 上传图片（或点示例图），输出带检测框的结果图
- 区分两类：`hardhat`（戴帽）、`no_hardhat`（没戴帽的人头）
- 统计戴帽/未戴帽数量，未戴帽时给出告警提示
- 可调置信度阈值（默认 0.35）
- 检测框明细：类别、置信度、坐标

## 目录结构

```
docs/                    GitHub Pages 站点根目录（浏览器端推理）
  index.html             在线演示界面
  app.js                 ONNX Runtime Web 推理 + 前处理 + NMS + 绘图
  best.onnx              导出的 ONNX 模型（9.4 MB）
  vendor/                onnxruntime-web 1.17.3（本地存放，不依赖 CDN）
  samples/               三张公开数据集示例图
local-demo/              （见下方"本地运行"，与本仓库根目录的 app.py/index.html 相同）
training/                训练与数据准备脚本、验收报告
  prepare_helmet_data.py COCO → YOLO，分层抽样
  train_helmet.py        YOLO26n 微调
  run_helmet.sh          云端一键复现
  use_helmet_model.py    命令行推理 + 违规计数
  helmet_report.txt      训练与验收报告
  summary.json           结构化指标
best.pt                  训练权重（5.2 MB）
best.onnx                导出的 ONNX 权重（与 docs 内一致）
app.py / index.html      本地 Web 演示（Python 标准库 http.server）
start_demo.bat           本地双击启动入口
```

## 本地运行（可选）

双击 `start_demo.bat`，浏览器会自动打开 `http://127.0.0.1:8765`。
启动器会复用本机 `DeepLearning/infrastructural/pt128` 环境（torch CPU + ultralytics），不需要额外安装依赖，也不需要 GPU。

手动启动：

```bash
python run_demo.py
```

## 模型是怎么来的

- 数据源：`keremberke/hard-hat-detection`（Roboflow 导出，经 hf-mirror 下载）
- 分层抽样子集：train 400 张 / val 100 张 / demo 10 张，全部 640×640
- 类别：`0 = hardhat`，`1 = no_hardhat`
- 起点：COCO 预训练 `yolo26n.pt`
- 训练：30 epoch，imgsz 640，batch 16，SGD，AMD Radeon Pro W7900D / ROCm，约 6 分钟
- 独立验证集：**mAP50 = 0.8115**，mAP50-95 = 0.4396，mAP75 = 0.4054
- 浏览器端后处理与 Python/Ultralytics 推理结果逐框一致（已对三张示例图核对）

## 边界（重要）

- 这是 **400 张公开子集 / 30 epoch 的教学练习模型**，mAP50 0.81 只在该子集上成立。换工地、光照、摄像头角度都会掉。
- **不能用于现场安全合规判定**。
- `no_hardhat` 标注的是「没戴帽的人头」，不是完整人体。要判断「这个人到底戴没戴」，需在下游做 head/helmet 框配对，或改训 with_helmet / without_helmet 两类。
- 模型只认「人头有没有帽」，不认「作业区域」：办公室、教室里的普通人也会触发告警。落地需要加作业区 ROI 过滤、连续 N 帧确认，并用真实场景图片补标注重训。
- 许可：训练/推理使用 Ultralytics（AGPL-3.0 / Enterprise 双许可）与公开数据集，学习研究没问题；商用前请自行核对条款。
