"""
线程安全数据缓冲桥接层

在采集线程与 WebSocket 服务线程之间共享最新的 EEG 数据窗口。
三类缓冲：原始 EEG、预处理 EEG、特征（仅频段功率）。
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Dict, List

import numpy as np
import pandas as pd

RAW_CHANNELS = ["ch_1", "ch_2", "ch_3", "ch_4"]
PROC_CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


class DataBridge:
    """
    线程安全环形缓冲区，供采集线程写入、WebSocket 线程读取。

    参数:
        window_seconds : 保留的时间窗口长度（秒）
        raw_fs         : 原始数据采样率（Hz），决定 raw 缓冲区 maxlen
        proc_fs        : 预处理数据采样率（Hz），决定 processed 缓冲区 maxlen
        feat_rate      : 特征更新速率（epoch/秒），决定 features 缓冲区 maxlen
    """

    def __init__(
        self,
        window_seconds: float = 10.0,
        raw_fs: float = 256.0,
        proc_fs: float = 256.0,
        feat_rate: float = 1.0,
    ):
        self._lock = threading.Lock()

        max_raw = max(1, int(window_seconds * raw_fs))
        max_proc = max(1, int(window_seconds * proc_fs))
        max_feat = max(1, int(window_seconds * feat_rate) + 1)

        # 原始 EEG 缓冲
        self._raw_ts: deque = deque(maxlen=max_raw)
        self._raw_ch: Dict[str, deque] = {ch: deque(maxlen=max_raw) for ch in RAW_CHANNELS}

        # 预处理 EEG 缓冲
        self._proc_ts: deque = deque(maxlen=max_proc)
        self._proc_ch: Dict[str, deque] = {ch: deque(maxlen=max_proc) for ch in PROC_CHANNELS}

        # 特征缓冲（仅保留 5 频段绝对功率，每通道一组）
        self._feat_ts: deque = deque(maxlen=max_feat)
        self._feat_band: Dict[str, Dict[str, deque]] = {
            ch: {band: deque(maxlen=max_feat) for band in BANDS}
            for ch in PROC_CHANNELS
        }

        # 注意力指标缓冲
        self._attn_ts: deque = deque(maxlen=max_feat)
        self._attn_score: deque = deque(maxlen=max_feat)
        self._attn_ei: deque = deque(maxlen=max_feat)
        self._attn_tar: deque = deque(maxlen=max_feat)
        self._attn_level: deque = deque(maxlen=max_feat)

    # ── 写入接口（采集线程调用）─────────────────────────────────────────────

    def add_raw(self, timestamps: List[float], samples: List[List[float]]) -> None:
        """写入一批原始 EEG 样本"""
        with self._lock:
            for ts, row in zip(timestamps, samples):
                self._raw_ts.append(ts)
                for i, ch in enumerate(RAW_CHANNELS):
                    self._raw_ch[ch].append(row[i] if i < len(row) else 0.0)

    def add_processed(self, df: pd.DataFrame) -> None:
        """写入预处理后的 EEG DataFrame（列: time, CH1-CH4）"""
        if df is None or df.empty:
            return
        with self._lock:
            for _, row in df.iterrows():
                self._proc_ts.append(float(row.get("time", 0.0)))
                for ch in PROC_CHANNELS:
                    self._proc_ch[ch].append(float(row.get(ch, 0.0)))

    def add_features(self, df: pd.DataFrame) -> None:
        """写入特征 DataFrame，只提取各通道的 5 频段绝对功率"""
        if df is None or df.empty:
            return
        with self._lock:
            for _, row in df.iterrows():
                self._feat_ts.append(float(row.get("timestamp", 0.0)))
                for ch in PROC_CHANNELS:
                    for band in BANDS:
                        col = f"{ch}_{band}_power"
                        self._feat_band[ch][band].append(float(row.get(col, 0.0)))

    def add_attention(self, result: dict) -> None:
        """写入一条注意力检测结果（来自 RealTimeAttentionDetector.add()）"""
        if not result:
            return
        with self._lock:
            self._attn_ts.append(float(result.get("timestamp", 0.0)))
            self._attn_score.append(float(result.get("attention_score", 0.0)))
            self._attn_ei.append(float(result.get("engagement_index", 0.0)))
            self._attn_tar.append(float(result.get("theta_alpha_ratio", 0.0)))
            self._attn_level.append(str(result.get("level", "low")))

    # ── 读取快照（WebSocket 线程调用）────────────────────────────────────────

    def snapshot(self) -> dict:
        """返回当前四类缓冲区的完整快照（可直接 JSON 序列化）"""
        with self._lock:
            attn_ts = list(self._attn_ts)
            return {
                "raw": {
                    "timestamps": _fmt_ts(list(self._raw_ts)),
                    **{ch: list(self._raw_ch[ch]) for ch in RAW_CHANNELS},
                },
                "processed": {
                    "timestamps": _fmt_ts(list(self._proc_ts)),
                    **{ch: list(self._proc_ch[ch]) for ch in PROC_CHANNELS},
                },
                "features": {
                    "timestamps": _fmt_ts(list(self._feat_ts)),
                    **{
                        ch: {band: list(self._feat_band[ch][band]) for band in BANDS}
                        for ch in PROC_CHANNELS
                    },
                },
                "attention": {
                    "timestamps": _fmt_ts(attn_ts),
                    "attention_score": list(self._attn_score),
                    "engagement_index": list(self._attn_ei),
                    "theta_alpha_ratio": list(self._attn_tar),
                    "level": list(self._attn_level),
                },
            }


def _fmt_ts(timestamps: list) -> list:
    """将 Unix 时间戳列表转为 'HH:MM:SS.mmm' 字符串列表供前端显示"""
    if not timestamps:
        return []
    import datetime
    result = []
    for ts in timestamps:
        dt = datetime.datetime.fromtimestamp(ts)
        result.append(dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}")
    return result
