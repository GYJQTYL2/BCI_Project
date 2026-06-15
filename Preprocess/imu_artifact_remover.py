"""
IMU 运动伪迹去除（NLMS 自适应滤波 + 门控）

两层保护：
  1. 门控（Gating）：合加速度偏差 > gate_threshold → 本窗口标记为运动污染，
     clean() 返回 None，下游跳过特征提取
  2. NLMS 滤波：偏差在 (nlms_threshold, gate_threshold] 之间 → 自适应修正后继续

用法：
    remover = IMUArtifactRemover()

    # 采集循环中，每次拿到 EEG 窗口和对应时间段的 IMU 数据
    result = remover.clean(eeg_df, imu_df)
    if result is None:
        pass  # 本窗口被门控丢弃
    else:
        eeg_clean = result
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

_EEG_CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
_IMU_CHANNELS = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]

# 静止时合加速度 ≈ 9.8 m/s²（1g），偏差单位与设备输出一致
# Muse 加速度计输出单位为 m/s²，若为 g 则阈值分别调为 0.5 和 0.15
_GRAVITY = 9.8


class NLMSFilter:
    """
    单通道 NLMS 自适应滤波器。

    对每个 EEG 采样点，用 IMU 多轴信号（含延迟抽头）估计运动伪迹并减去。

    参数:
        n_taps : 每轴延迟抽头数，默认 8（覆盖约 154 ms @ 52 Hz IMU）
        mu     : 归一化步长，范围 (0, 2)，默认 0.5
        eps    : 防止除零的小量
        n_ref  : 参考信号维度（IMU 轴数），默认 6
    """

    def __init__(self, n_taps: int = 8, mu: float = 0.5,
                 eps: float = 1e-6, n_ref: int = 6):
        self.n_taps = n_taps
        self.mu = mu
        self.eps = eps
        self.n_ref = n_ref
        self._dim = n_ref * n_taps
        self.W = np.zeros(self._dim)
        self._buf = np.zeros((n_taps, n_ref))

    def filter_sample(self, eeg_val: float, imu_vec: np.ndarray) -> float:
        """处理单个采样点，返回去伪迹后的 EEG 值"""
        self._buf = np.roll(self._buf, 1, axis=0)
        self._buf[0] = imu_vec

        x = self._buf.flatten()
        artifact_est = np.dot(self.W, x)
        e = eeg_val - artifact_est

        norm = np.dot(x, x) + self.eps
        self.W += (self.mu / norm) * e * x
        return e

    def reset(self) -> None:
        """重置权重和缓冲区（大运动后调用，防止错误收敛）"""
        self.W[:] = 0.0
        self._buf[:] = 0.0


class IMUArtifactRemover:
    """
    NLMS 自适应滤波 + 门控，用 IMU 信号去除 EEG 中的运动伪迹。

    参数:
        n_taps          : NLMS 滤波器抽头数，默认 8
        mu              : NLMS 步长，默认 0.5
        gate_threshold  : 合加速度偏差阈值（m/s²），超过则门控丢弃窗口，默认 4.9（≈0.5g）
        nlms_threshold  : 低于此值视为静止，直通不修正，默认 1.47（≈0.15g）
    """

    def __init__(
        self,
        n_taps: int = 8,
        mu: float = 0.5,
        gate_threshold: float = 4.9,
        nlms_threshold: float = 1.47,
    ):
        self.gate_threshold = gate_threshold
        self.nlms_threshold = nlms_threshold
        self._filters = {ch: NLMSFilter(n_taps=n_taps, mu=mu) for ch in _EEG_CHANNELS}
        self._last_motion_level: str = "still"   # still | nlms | gated

    @property
    def last_motion_level(self) -> str:
        """返回上一次窗口的运动级别：still / nlms / gated"""
        return self._last_motion_level

    def clean(
        self, eeg_df: pd.DataFrame, imu_df: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """
        对一个 EEG 窗口应用 IMU 去伪迹。

        参数:
            eeg_df : 含 time + CH1–CH4 列，256 Hz
            imu_df : 含 time + AccX/AccY/AccZ/GyrX/GyrY/GyrZ 列，52 Hz
                     为 None 时直通（IMU 未采集）

        返回:
            去伪迹后的 DataFrame，若被门控则返回 None
        """
        if imu_df is None or imu_df.empty:
            self._last_motion_level = "still"
            return eeg_df

        # 检查 IMU 列是否完整
        missing = [c for c in _IMU_CHANNELS if c not in imu_df.columns]
        if missing:
            return eeg_df

        motion_level = self._assess_motion(imu_df)
        self._last_motion_level = motion_level

        if motion_level == "gated":
            # 大运动：重置所有滤波器权重，丢弃本窗口
            for f in self._filters.values():
                f.reset()
            return None

        if motion_level == "still":
            return eeg_df

        # NLMS 修正
        eeg_times = eeg_df["time"].values
        imu_interp = self._interpolate_imu(eeg_times, imu_df)
        df_out = eeg_df.copy()

        for ch, filt in self._filters.items():
            cleaned = np.empty(len(eeg_df))
            for i in range(len(eeg_df)):
                cleaned[i] = filt.filter_sample(
                    float(eeg_df[ch].iloc[i]), imu_interp[i]
                )
            df_out[ch] = cleaned

        return df_out

    def _assess_motion(self, imu_df: pd.DataFrame) -> str:
        """根据合加速度偏差判断运动级别"""
        acc = imu_df[["AccX", "AccY", "AccZ"]].values
        norm = np.sqrt((acc ** 2).sum(axis=1))
        deviation = np.abs(norm - _GRAVITY).max()

        if deviation > self.gate_threshold:
            return "gated"
        if deviation > self.nlms_threshold:
            return "nlms"
        return "still"

    def _interpolate_imu(
        self, eeg_times: np.ndarray, imu_df: pd.DataFrame
    ) -> np.ndarray:
        """将 52 Hz IMU 线性插值到 EEG 时间轴，返回 (n_samples, 6)"""
        result = np.zeros((len(eeg_times), 6))
        imu_times = imu_df["time"].values
        for i, col in enumerate(_IMU_CHANNELS):
            result[:, i] = np.interp(eeg_times, imu_times, imu_df[col].values)
        return result
