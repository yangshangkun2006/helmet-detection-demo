#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从公开数据集 keremberke/hard-hat-detection (Roboflow COCO json) 生成 YOLO 格式子集。

在 AMD Radeon Cloud 上运行，只用标准库：
  /workspace/helmet_src/{train,valid,test}.zip  ->  /workspace/helmet_data/

类别：0=hardhat（戴了安全帽）, 1=no_hardhat（没戴安全帽的人头）
划分：train 400 / val 100 / demo 10（demo 来自官方 test 划分，完全未参与训练与验证）
"""
import collections
import json
import random
import shutil
import zipfile
from pathlib import Path

SRC = Path("/workspace/helmet_src")
OUT = Path("/workspace/helmet_data")
SEED = 26
NAMES = {0: "hardhat", 1: "no_hardhat"}

# (输出划分, 源 zip, 抽样计划: {分组: 张数})
# 分层抽样保证少数类 no_hardhat 有充分覆盖；train 混入少量空图（对应空 txt）
PLAN = {
    "train": ("train.zip", {"has_nohardhat": 200, "only_hardhat": 180, "empty": 20}),
    "val":   ("valid.zip", {"has_nohardhat": 50,  "only_hardhat": 50,  "empty": 0}),
    "demo":  ("test.zip",  {"has_nohardhat": 8,   "only_hardhat": 2,   "empty": 0}),
}


def main():
    rng = random.Random(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in PLAN:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split, (zname, quotas) in PLAN.items():
        zpath = SRC / zname
        assert zpath.exists(), f"缺少 {zpath}"
        with zipfile.ZipFile(zpath) as zf:
            entries = set(zf.namelist())
            jname = next(n for n in zf.namelist() if n.endswith(".json"))
            d = json.loads(zf.read(jname))
            catmap = {c["id"]: c["name"] for c in d["categories"]}
            assert set(catmap) == {0, 1}, f"类别异常: {catmap}"

            anns = collections.defaultdict(list)
            for a in d["annotations"]:
                if a.get("iscrowd", 0):
                    continue
                anns[a["image_id"]].append(a)
            imgs = {im["id"]: im for im in d["images"]}

            groups = {"has_nohardhat": [], "only_hardhat": [], "empty": []}
            for iid, im in imgs.items():
                a = anns[iid]
                if not a:
                    groups["empty"].append(im)
                elif any(x["category_id"] == 1 for x in a):
                    groups["has_nohardhat"].append(im)
                else:
                    groups["only_hardhat"].append(im)
            for g in groups.values():
                rng.shuffle(g)

            picked = []
            for key, n in quotas.items():
                picked.extend(groups[key][:n])
            rng.shuffle(picked)

            n_img = n_box = n_clipped = n_missing = n_empty = 0
            cls_count = collections.Counter()
            for im in picked:
                fn = im["file_name"]
                if fn not in entries:
                    n_missing += 1
                    continue
                stem = Path(fn).stem
                W, H = float(im["width"]), float(im["height"])
                lines = []
                for a in anns[im["id"]]:
                    x, y, w, h = [float(v) for v in a["bbox"]]
                    x1, y1 = max(0.0, x), max(0.0, y)
                    x2, y2 = min(W, x + w), min(H, y + h)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if (x1, y1, x2, y2) != (x, y, x + w, y + h):
                        n_clipped += 1
                    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                    bw, bh = (x2 - x1) / W, (y2 - y1) / H
                    cls = int(a["category_id"])
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    cls_count[cls] += 1
                (OUT / "images" / split / fn).write_bytes(zf.read(fn))
                (OUT / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                n_img += 1
                n_box += len(lines)
                if not lines:
                    n_empty += 1

            all_stats[split] = {
                "source_zip": zname,
                "source_images": len(imgs),
                "source_annotations": len(d["annotations"]),
                "picked": n_img,
                "boxes": n_box,
                "by_class": {NAMES[k]: v for k, v in sorted(cls_count.items())},
                "empty_images": n_empty,
                "clipped_boxes": n_clipped,
                "missing_files": n_missing,
            }

    # data.yaml
    (OUT / "data.yaml").write_text(
        "path: /workspace/helmet_data\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: hardhat\n"
        "  1: no_hardhat\n",
        encoding="utf-8",
    )

    # 完整性校验
    errors = []
    for split in ("train", "val", "demo"):
        images = sorted((OUT / "images" / split).glob("*.jpg"))
        for p in images:
            lab = OUT / "labels" / split / f"{p.stem}.txt"
            if not lab.exists():
                errors.append(f"缺标注: {lab}")
                continue
            for ln in lab.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if len(parts) != 5 or int(parts[0]) not in (0, 1):
                    errors.append(f"格式错: {lab}: {ln}")
                    continue
                vals = [float(v) for v in parts[1:]]
                if any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                    errors.append(f"归一化越界: {lab}: {ln}")
        all_stats[split]["verified_images"] = len(images)
    # train/val 不得有同名图片（官方划分本身独立）
    tr = {p.name for p in (OUT / "images" / "train").glob("*.jpg")}
    va = {p.name for p in (OUT / "images" / "val").glob("*.jpg")}
    overlap = tr & va
    if overlap:
        errors.append(f"train/val 重名 {len(overlap)} 张: {list(overlap)[:3]}")
    all_stats["train_val_overlap"] = len(overlap)
    all_stats["errors"] = errors

    (OUT / "dataset_stats.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(all_stats, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(f"DATA_CHECK_FAILED: {len(errors)} 项，前 3 条 {errors[:3]}")
    print("DATA_PREP_OK  train=%d val=%d demo=%d" % (
        all_stats["train"]["picked"], all_stats["val"]["picked"], all_stats["demo"]["picked"]
    ))


if __name__ == "__main__":
    main()
