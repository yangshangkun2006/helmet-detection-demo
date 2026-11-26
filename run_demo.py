#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键启动器（只依赖系统 Python 标准库）。
自动找到带 ultralytics 的 Python 解释器 -> 启动 app.py -> 打开浏览器。
"""
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

BASE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8765"

# 本机已验证可用：DeepLearning 程序自带 pt128 环境（torch 2.11 CPU + ultralytics 8.4.71）
CANDIDATES = [
    Path(r"D:\下载\DeepLearning_20261001\DeepLearning\infrastructural\pt128\python.exe"),
]


def find_python():
    for p in CANDIDATES:
        if p.is_file():
            return str(p)
    try:
        import ultralytics  # noqa: F401
        return sys.executable
    except ImportError:
        return None


def main():
    py = find_python()
    if py is None:
        print("未找到带 ultralytics 的 Python 环境，无法启动推理服务。")
        sys.exit(1)
    print("使用解释器：", py)
    print("演示地址：", URL)
    proc = subprocess.Popen([py, str(BASE / "app.py")], cwd=str(BASE))
    time.sleep(1.5)
    webbrowser.open(URL)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
