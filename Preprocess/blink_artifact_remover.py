"""
眨眼伪影去除（滚动自适应阈值 + 线性插值）

两层保护：
  1. 线性插值修复：检测到眨眼采样点 → 用前后干净锚点线性插值替换
  2. 整窗口门控：污染比例 > max_contamination → 返回 None，丢弃整窗口

不依赖基线录制，warmup 期间用固定阈值兜底，之后滚动自适应。
作为 ASR 未激活时的 fallback，插在 ASR 之后、IMU 去除之前。

用法：
    remover = BlinkArtifactRemover()
    result = remover.clean(df_proc, sfreq=256.0)
    if result is None:
        pass  # 整窗口被门控（眨眼污染 > 30%）
    else:
        df_clean = result
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_EEG_CHANNELS = ["CH1", "CH2", "CH3", "CH4"]


class BlinkArtifactRemover:
    """
    滚动自适应阈值眨眼检测 + 线性插值修复。

    参数:
        k                 : 阈值倍数，阈值 = 滚动中位数 × k，默认 5
        margin_ms         : 眨眼边缘扩展（ms），覆盖上升/下降沿，默认 100
        max_contamination : 超过此污染比例退化为整窗口丢弃，默认 0.30
        warmup_windows    : 滚动估计稳定所需窗口数，默认 20（≈100s）
        fallback_threshold: warmup 未完成时的固定兜底阈值（µV），默认 200
    """

    def __init__(
        self,
        k: float = 5.0,
        margin_ms: float = 100.0,
        max_contamination: float = 0.30,
        warmup_windows: int = 20,
        fallback_threshold: float = 200.0,
    ):
        self.k = k
        self.margin_ms = margin_ms
        self.max_contamination = max_contamination
        self.fallback_threshold = fallback_threshold
        self._warmup_windows = warmup_windows
        # 只存干净窗口（无眨眼）的各通道 max(abs) 中位数
        self._rolling: deque[float] = deque(maxlen=warmup_windows)
        self._last_blink_detected = False

    @property
    def last_blink_detected(self) -> bool:
        """上一次 clean() 调用是否检测到眨眼"""
        return self._last_blink_detected

    @property
    def is_warmed_up(self) -> bool:
        return len(self._rolling) >= self._warmup_windows

    def _threshold(self) -> float:
        if not self.is_warmed_up:
            return self.fallback_threshold
        return float(np.median(self._rolling)) * self.k

    def clean(
        self, df: pd.DataFrame, sfreq: float = 256.0
    ) -> Optional[pd.DataFrame]:
        """
        对一个 EEG 窗口进行眨眼检测和修复。

        参数:
            df    : 含 CH1–CH4 列的 DataFrame（pipeline 处理后的信号，µV）
            sfreq : 采样率（Hz），用于将 margin_ms 转换为样本数

        返回:
            修复后的 DataFrame，或 None（整窗口门控）
        """
        chs = [c for c in _EEG_CHANNELS if c in df.columns]
        if not chs:
            return df

        n = len(df)
        margin = max(1, int(self.margin_ms * sfreq / 1000))
        threshold = self._threshold()

        # 各通道 max(abs)，跳过全 NaN 通道（bad 通道），避免 NaN 污染滚动估计
        ch_maxabs = [
            float(np.nanmax(np.abs(df[ch].values)))
            for ch in chs
            if not np.all(np.isnan(df[ch].values))
        ]

        # 检测：任一通道超阈的采样点位置
        blink_mask = np.zeros(n, dtype=bool)
        for ch in chs:
            blink_mask |= np.abs(df[ch].values) > threshold

        was_warmed = self.is_warmed_up

        if not blink_mask.any():
            # 只有干净窗口参与滚动估计，避免修复窗口拉低阈值
            self._last_blink_detected = False
            self._rolling.append(float(np.median(ch_maxabs)))
            if not was_warmed and self.is_warmed_up:
                log.info("[BlinkRemover] warmup 完成，切换为自适应阈值")
            return df

        # 向两侧扩展 margin，覆盖眨眼上升/下降沿
        expanded = _expand_mask(blink_mask, margin)

        contamination = expanded.sum() / n
        if contamination > self.max_contamination:
            # 污染过多，整窗口丢弃，不更新滚动估计
            self._last_blink_detected = True
            return None

        # 线性插值修复：每个超阈通道单独插值
        df_out = df.copy()
        indices = np.arange(n)
        good = ~expanded

        if good.sum() < 2:
            # 好锚点不足，无法插值，整窗口丢弃
            self._last_blink_detected = True
            return None

        for ch in chs:
            signal = df_out[ch].values.astype(float)
            # 只对该通道自身超阈的区域做插值，其余通道不动
            ch_mask = _expand_mask(np.abs(signal) > threshold, margin)
            if ch_mask.any():
                ch_good = ~ch_mask
                if ch_good.sum() >= 2:
                    signal[ch_mask] = np.interp(
                        indices[ch_mask], indices[ch_good], signal[ch_good]
                    )
                else:
                    log.warning(f"[BlinkRemover] {ch} 锚点不足，跳过插值，通道标 NaN")
                    signal[:] = np.nan
                df_out[ch] = signal

        self._last_blink_detected = True
        # 修复窗口不参与滚动估计，避免插值幅值偏低导致阈值下漂
        return df_out


def _expand_mask(mask: np.ndarray, margin: int) -> np.ndarray:
    """将布尔掩码向两侧各扩展 margin 个采样点"""
    if not mask.any():
        return mask
    indices = np.where(mask)[0]
    expanded = mask.copy()
    for idx in indices:
        lo = max(0, idx - margin)
        hi = min(len(mask), idx + margin + 1)
        expanded[lo:hi] = True
    return expanded
