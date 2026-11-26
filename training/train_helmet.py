#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 AMD Radeon Cloud (W7900 / ROCm) 上微调 YOLO26n 做安全帽检测，
然后评估 val、对官方 test 划分的未见图片做推理并生成带框图。

产出：/workspace/helmet_runs/baseline/weights/best.pt + summary.json + 图表
"""
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path("/workspace")
DATA = ROOT / "helmet_data" / "data.yaml"
INIT = ROOT / "yolo26-rocm-day1-labs" / "yolo26n.pt"
RUNS = ROOT / "helmet_runs"
DEMO = ROOT / "helmet_data" / "images" / "demo"
ASSETS = RUNS / "report_assets"

TRAIN_KWARGS = dict(
    data=str(DATA),
    epochs=30,
    imgsz=640,
    batch=16,
    device="cuda",
    workers=0,          # ROCm 容器里多进程 dataloader 易出问题，课程验证过 0
    amp=False,          # 课程在 ROCm 上关闭混合精度
    optimizer="SGD",
    patience=8,
    seed=26,
    plots=True,
    project=str(RUNS),
    name="baseline",
    exist_ok=True,
)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert DATA.exists(), f"缺少 {DATA}"
    assert INIT.exists(), f"缺少 {INIT}"
    ASSETS.mkdir(parents=True, exist_ok=True)
    print("TRAIN_KWARGS =", json.dumps(TRAIN_KWARGS, ensure_ascii=False), flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(INIT))
    t0 = time.perf_counter()
    model.train(**TRAIN_KWARGS)
    torch.cuda.synchronize()
    elapsed = round(time.perf_counter() - t0, 2)
    peak_gib = round(torch.cuda.max_memory_allocated() / 2**30, 3)

    run_dir = Path(model.trainer.save_dir).resolve()
    best = run_dir / "weights" / "best.pt"
    assert best.exists(), f"best.pt 未生成: {best}"
    print(f"TRAIN_OK  elapsed={elapsed}s peak={peak_gib}GiB best={best}", flush=True)

    # 复制体检图表
    for pat in ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
                "F1_curve.png", "PR_curve.png", "P_curve.png", "R_curve.png",
                "results.csv", "args.yaml", "val_batch0_pred.jpg", "val_batch1_pred.jpg"):
        for src in run_dir.glob(pat):
            shutil.copy2(src, ASSETS / src.name)

    summary = {
        "train": {
            "elapsed_seconds": elapsed,
            "peak_allocated_GiB": peak_gib,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "kwargs": {k: str(v) for k, v in TRAIN_KWARGS.items()},
            "run_dir": str(run_dir),
        },
        "weights": {
            "path": str(best),
            "bytes": best.stat().st_size,
            "sha256": sha256(best),
        },
    }

    # 独立 val 评估
    val_model = YOLO(str(best))
    metrics = val_model.val(
        data=str(DATA), split="val", imgsz=640, batch=16, device="cuda",
        workers=0, amp=False, project=str(RUNS), name="val_final", exist_ok=True,
        plots=True,
    )
    box = metrics.box
    summary["validation"] = {
        "mAP50_95": round(float(box.map), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP75": round(float(box.map75), 4),
        "per_class_mAP50_95": [round(float(v), 4) for v in box.maps],
        "class_names": val_model.names,
    }
    print("VAL_OK", json.dumps(summary["validation"], ensure_ascii=False), flush=True)

    # 官方 test 划分的未见图片 -> 带框图 + 违规计数
    demo_out = RUNS / "demo_pred"
    if demo_out.exists():
        shutil.rmtree(demo_out)
    results = val_model.predict(
        source=str(DEMO), conf=0.35, imgsz=640, device="cuda",
        save=True, project=str(RUNS), name="demo_pred", exist_ok=True,
        workers=0, amp=False,
    )
    per_image = []
    for r in results:
        n_helmet = sum(1 for b in r.boxes if int(b.cls) == 0)
        n_violation = sum(1 for b in r.boxes if int(b.cls) == 1)
        per_image.append({
            "image": Path(r.path).name,
            "hardhat": n_helmet,
            "no_hardhat": n_violation,
            "alert": n_violation > 0,
        })
    summary["demo"] = {
        "images": len(results),
        "images_with_violation": sum(1 for x in per_image if x["alert"]),
        "per_image": per_image,
        "annotated_dir": str(demo_out),
    }
    print("DEMO_OK", json.dumps({
        "images": len(results),
        "images_with_violation": summary["demo"]["images_with_violation"],
    }), flush=True)

    (RUNS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("HELMET_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
