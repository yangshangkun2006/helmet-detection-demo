#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把训练好的 best.pt 接成能干活的「眼睛 + 业务逻辑」。

模型只负责输出 [框坐标, 类别, 置信度]；「没戴帽就告警 / 计数 / 存图」是下面这几行
普通 Python 代码，你可以随意改成写 CSV、推消息、亮灯、控制设备。

用法（云电脑用 Day1 的 venv；本机需先 pip install ultralytics）：
  python use_helmet_model.py --weights best.pt --source photo.jpg --save
  python use_helmet_model.py --weights best.pt --source imgs_dir/ --conf 0.5
  python use_helmet_model.py --weights best.pt --source site.mp4 --save
  python use_helmet_model.py --weights best.pt --source 0          # 本机摄像头，Ctrl+C 退出

类别：0 = hardhat（戴了安全帽）, 1 = no_hardhat（没戴安全帽的人头）
"""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="安全帽检测推理 / 违规计数")
    ap.add_argument("--weights", default="/workspace/helmet_runs/baseline/weights/best.pt",
                    help="训练产出的 best.pt")
    ap.add_argument("--source", required=True, help="图片 / 目录 / 视频 / 摄像头编号")
    ap.add_argument("--conf", type=float, default=0.35, help="置信度阈值；误报多调高，漏检多调低")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--save", action="store_true", help="保存带框图")
    ap.add_argument("--out", default="helmet_output", help="带框图输出目录")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = Path(args.weights)
    assert weights.exists(), f"找不到权重：{weights}（云端默认路径或本机用 --weights 指定）"

    model = YOLO(str(weights))
    print(f"[加载] {weights} | device={device} | conf={args.conf}")
    print(f"[类别] {model.names}")

    results = model.predict(
        source=args.source, conf=args.conf, imgsz=640, device=device,
        save=args.save, project=args.out, name="pred", exist_ok=True,
        verbose=False,
    )

    total = {"frames": 0, "hardhat": 0, "no_hardhat": 0, "alert_frames": 0}
    for r in results:
        n_helmet = n_violation = 0
        for b in r.boxes:
            cls = int(b.cls)
            if cls == 0:
                n_helmet += 1
            elif cls == 1:
                n_violation += 1
        total["frames"] += 1
        total["hardhat"] += n_helmet
        total["no_hardhat"] += n_violation
        if n_violation:
            total["alert_frames"] += 1
        name = Path(getattr(r, "path", "frame")).name
        flag = "  ⚠ 告警" if n_violation else "  OK"
        print(f"{name:52s} hardhat={n_helmet:2d}  no_hardhat={n_violation:2d}{flag}")

    print("\n===== 汇总 =====")
    print(json.dumps(total, ensure_ascii=False, indent=2))
    if total["alert_frames"]:
        print(f"⚠ 共 {total['alert_frames']} 帧检测到未戴安全帽 —— 这里接你的业务动作："
              "写 CSV / 存截图 / 推消息 / 触发告警")
    if args.save:
        print(f"带框图目录：{args.out}/pred/")


if __name__ == "__main__":
    main()
