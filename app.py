#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全帽检测 · 本地 Web 演示服务
================================
零额外依赖启动（使用 DeepLearning 自带 pt128 环境里的 ultralytics）：
    python app.py
然后浏览器打开 http://127.0.0.1:8765

功能：上传图片 / 选示例图 -> YOLO26n 推理 -> 返回带框检测图 + 统计结果。
仅用 Python 标准库 http.server 提供 Web 服务，不需要 Flask。
"""
from __future__ import annotations

import cgi
import json
import os
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
RESULT_DIR = BASE / "results"
SAMPLE_DIR = BASE / "samples"
WEIGHTS = BASE / "best.pt"
INDEX_FILE = BASE / "index.html"

PORT = int(os.environ.get("HELMET_DEMO_PORT", "8765"))
MAX_UPLOAD = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".bmp": "image/bmp", ".webp": "image/webp",
}

_model = None
_model_lock = threading.Lock()


def get_model() -> YOLO:
    """模型只加载一次，后续请求复用。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print("[加载模型]", WEIGHTS, flush=True)
                _model = YOLO(str(WEIGHTS))
                print("[模型就绪]", _model.names, flush=True)
    return _model


def detect(src: Path, conf: float) -> dict:
    model = get_model()
    started = time.time()
    with _model_lock:
        result = model.predict(
            source=str(src), conf=conf, imgsz=640, device="cpu", verbose=False
        )[0]
        plotted = result.plot()  # BGR 带框图像
        names = model.names

    out_name = uuid.uuid4().hex + ".jpg"
    out_path = RESULT_DIR / out_name
    cv2.imwrite(str(out_path), plotted, [cv2.IMWRITE_JPEG_QUALITY, 92])

    boxes = []
    for b in result.boxes:
        cls = int(b.cls)
        boxes.append({
            "label": str(names.get(cls, cls)),
            "cls": cls,
            "conf": round(float(b.conf), 3),
            "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
        })
    boxes.sort(key=lambda x: x["conf"], reverse=True)

    n_helmet = sum(1 for x in boxes if x["cls"] == 0)
    n_violation = sum(1 for x in boxes if x["cls"] == 1)
    return {
        "ok": True,
        "image": "/result/" + out_name,
        "source": src.name,
        "elapsed": round(time.time() - started, 2),
        "counts": {"helmet": n_helmet, "no_helmet": n_violation, "total": len(boxes)},
        "alert": n_violation > 0,
        "boxes": boxes,
        "conf": conf,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "HelmetDemo/1.0"

    def log_message(self, fmt, *args):
        print("[HTTP] " + (fmt % args), flush=True)

    # ---------- 通用响应 ----------
    def _send(self, code: int, body: bytes, content_type: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if not cache else "public, max-age=3600")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: dict, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _image(self, folder: Path, raw_name: str):
        name = Path(unquote(raw_name)).name  # 防目录穿越
        path = folder / name
        if not path.is_file() or path.suffix.lower() not in MIME:
            self._json({"ok": False, "error": "图片不存在"}, 404)
            return
        self._send(200, path.read_bytes(), MIME[path.suffix.lower()])

    # ---------- GET ----------
    def do_GET(self):
        path = url_path(self.path)
        if path in ("/", "/index.html"):
            if not INDEX_FILE.is_file():
                self._json({"ok": False, "error": "缺少 index.html"}, 500)
                return
            self._send(200, INDEX_FILE.read_bytes(), "text/html; charset=utf-8")
        elif path.startswith("/result/"):
            self._image(RESULT_DIR, path[len("/result/"):])
        elif path.startswith("/sample/"):
            self._image(SAMPLE_DIR, path[len("/sample/"):])
        elif path == "/api/info":
            self._json({
                "ok": True,
                "weights": WEIGHTS.name,
                "model_loaded": _model is not None,
                "samples": sorted(p.name for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in MIME),
            })
        elif path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._json({"ok": False, "error": "404 Not Found"}, 404)

    # ---------- POST ----------
    def do_POST(self):
        if url_path(self.path) != "/detect":
            self._json({"ok": False, "error": "404 Not Found"}, 404)
            return
        try:
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                self._json({"ok": False, "error": "请使用 multipart/form-data 上传"}, 400)
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype},
                keep_blank_values=True,
            )
            try:
                conf = float(form.getvalue("conf", "0.35"))
            except (TypeError, ValueError):
                conf = 0.35
            conf = max(0.05, min(0.9, conf))

            sample = form.getvalue("sample")
            if sample:
                name = Path(str(sample)).name
                src = SAMPLE_DIR / name
                if not src.is_file() or src.suffix.lower() not in MIME:
                    self._json({"ok": False, "error": "示例图不存在"}, 400)
                    return
                display = name
            else:
                item = form["file"] if "file" in form else None
                if item is None or not getattr(item, "filename", None):
                    self._json({"ok": False, "error": "请先选择一张图片"}, 400)
                    return
                data = item.file.read(MAX_UPLOAD + 1)
                if not data:
                    self._json({"ok": False, "error": "上传内容为空"}, 400)
                    return
                if len(data) > MAX_UPLOAD:
                    self._json({"ok": False, "error": "图片超过 20 MB 限制"}, 413)
                    return
                ext = Path(item.filename).suffix.lower()
                if ext not in MIME:
                    self._json({"ok": False, "error": "仅支持 jpg / png / bmp / webp"}, 400)
                    return
                display = Path(item.filename).name
                src = UPLOAD_DIR / (uuid.uuid4().hex + ext)
                src.write_bytes(data)

            result = detect(src, conf)
            result["source"] = display
            self._json(result)
        except Exception as exc:
            traceback.print_exc()
            self._json({"ok": False, "error": f"推理失败：{exc}"}, 500)


def url_path(raw: str) -> str:
    return unquote(raw.split("?", 1)[0])


def main():
    for d in (UPLOAD_DIR, RESULT_DIR, SAMPLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.is_file():
        raise SystemExit(f"找不到权重文件：{WEIGHTS}")
    print("=" * 58, flush=True)
    print("  安全帽检测 · 本地演示服务", flush=True)
    print(f"  打开浏览器：http://127.0.0.1:{PORT}", flush=True)
    print("  停止服务：在本窗口按 Ctrl+C", flush=True)
    print("=" * 58, flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
