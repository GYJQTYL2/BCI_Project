"""
EEG特征提取核心类

EEGFeatureExtractor 对 DataFrame 按 epoch 分窗提取特征，
支持时域和频域两类特征模块，均可独立启用/禁用。

输入:  预处理后的 EEG DataFrame（列名: time, CH1, CH2, CH3, CH4）
输出:  特征 DataFrame（每行对应一个 epoch，列为各通道各特征）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import time_domain, frequency_domain

EEG_CHANNELS = ["CH1", "CH2", "CH3", "CH4"]


class EEGFeatureExtractor:
    """
    EEG 特征提取器

    用法示例:
        extractor = EEGFeatureExtractor(fs=256.0, epoch_seconds=1.0)

        # 对整段 DataFrame 按 epoch 分窗提取（返回多行 DataFrame）
        df_features = extractor.extract(df_preprocessed)

        # 对单个 epoch 提取（返回单行 DataFrame）
        df_feat = extractor.extract_epoch(df_epoch)

    参数说明:
        fs              : 采样率（Hz），默认 256
        channels        : 要提取特征的通道列表，默认 ['CH1','CH2','CH3','CH4']
        epoch_seconds   : 分窗 epoch 长度（秒），默认 1.0
        enable_time     : 是否提取时域特征，默认 True
        enable_freq     : 是否提取频域特征，默认 True
        min_epoch_ratio : epoch 长度不足时的最小比例阈值，不足则跳过，默认 0.5
    """

    def __init__(
        self,
        fs: float = 256.0,
        channels: Optional[List[str]] = None,
        epoch_seconds: float = 1.0,
        enable_time: bool = True,
        enable_freq: bool = True,
        min_epoch_ratio: float = 0.5,
    ):
        self.fs = fs
        self.channels = channels or EEG_CHANNELS
        self.epoch_size = max(1, int(epoch_seconds * fs))
        self.enable_time = enable_time
        self.enable_freq = enable_freq
        self.min_epoch_ratio = min_epoch_ratio

    # ── 单 epoch 提取 ──────────────────────────────────────────────────────

    def extract_epoch(self, df_epoch: pd.DataFrame) -> pd.DataFrame:
        """
        对单个 epoch DataFrame 提取所有启用的特征

        参数:
            df_epoch : 单个 epoch 的 EEG DataFrame
        返回:
            单行特征 DataFrame
        """
        feat: dict = {}

        if "time" in df_epoch.columns:
            feat["timestamp"] = df_epoch["time"].iloc[0]

        for ch in self.channels:
            if ch not in df_epoch.columns:
                continue
            epoch = df_epoch[ch].to_numpy(dtype=float)
            if self.enable_time:
                feat.update(time_domain.extract(epoch, ch))
            if self.enable_freq:
                feat.update(frequency_domain.extract(epoch, ch, self.fs))

        return pd.DataFrame([feat])

    # ── 整段分窗提取 ──────────────────────────────────────────────────────

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将 DataFrame 按 epoch_seconds 分窗，逐 epoch 提取特征

        参数:
            df : 预处理后的完整 EEG DataFrame
        返回:
            特征 DataFrame，每行对应一个 epoch
        """
        rows: list[pd.DataFrame] = []
        min_len = max(1, int(self.epoch_size * self.min_epoch_ratio))
        total = len(df)

        for start in range(0, total, self.epoch_size):
            epoch_df = df.iloc[start: start + self.epoch_size]
            if len(epoch_df) < min_len:
                continue
            rows.append(self.extract_epoch(epoch_df))

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
