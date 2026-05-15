"""
独立测试：不需要 LSL 设备，直接用模拟数据验证可视化服务器。

运行:
    cd Visualize
    python test_server.py

然后浏览器打开 http://localhost:8765
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_bridge import DataBridge
from ws_server import WebSocketServer

bridge = DataBridge(window_seconds=10, raw_fs=256)
server = WebSocketServer(bridge, port=8765, push_hz=10)
server.start()

time.sleep(0.5)
print("浏览器打开: http://localhost:8765")
print("按 Ctrl+C 停止\n")

CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
FREQS = [10.0, 12.0, 8.0, 15.0]   # 每通道主频

t = 0.0
epoch_counter = 0

try:
    while True:
        now = time.time()
        n = 13  # 每批约 50ms

        # ── 原始 EEG ──────────────────────────────────────────────────
        timestamps = [now + i / 256.0 for i in range(n)]
        samples = [
            [math.sin(2 * math.pi * FREQS[ch] * (t + i / 256.0)) * 50
             for ch in range(4)]
            for i in range(n)
        ]
        bridge.add_raw(timestamps, samples)

        # ── 预处理 EEG（简单缩放模拟 z-score 后效果）────────────────
        proc_data = {"time": timestamps}
        for idx, ch in enumerate(CHANNELS):
            proc_data[ch] = [
                math.sin(2 * math.pi * FREQS[idx] * (t + i / 256.0))
                for i in range(n)
            ]
        bridge.add_processed(pd.DataFrame(proc_data))

        # ── 特征（每秒更新一次）──────────────────────────────────────
        epoch_counter += n
        if epoch_counter >= 256:
            epoch_counter = 0
            feat = {"timestamp": [now]}
            for ch in CHANNELS:
                total = 100.0
                # 模拟随时间变化的频段功率
                phase = now * 0.3
                powers = {
                    "delta": 10 + 5  * math.sin(phase),
                    "theta": 15 + 8  * math.sin(phase * 1.3),
                    "alpha": 30 + 15 * math.sin(phase * 0.7),
                    "beta":  25 + 10 * math.sin(phase * 1.7),
                    "gamma": 20 + 5  * math.sin(phase * 2.1),
                }
                for band, val in powers.items():
                    feat[f"{ch}_{band}_power"] = [max(0.1, val)]
            bridge.add_features(pd.DataFrame(feat))

        t += n / 256.0
        time.sleep(0.05)

except KeyboardInterrupt:
    server.stop()
    print("已停止")
