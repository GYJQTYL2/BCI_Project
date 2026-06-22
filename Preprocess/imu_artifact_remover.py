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

# 静止时合加速度 ≈ 1.0g，Muse 加速度计输出单位为 g
_GRAVITY = 1.0


# 陀螺仪阈值（°/s）
_GYR_GATE_THRESHOLD = 30.0   # 超过则门控
_GYR_NLMS_THRESHOLD = 15.0   # 超过则 NLMS 修正


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
        if not (0 < mu < 2):
            raise ValueError(f"NLMS mu 必须在 (0, 2) 内，传入值: {mu}")
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
        gate_threshold  : 合加速度偏差阈值（g），超过则门控丢弃窗口，默认 0.5g
        nlms_threshold  : 低于此值视为静止，直通不修正，默认 0.15g
        陀螺仪阈值由模块常量控制：gated >30°/s，nlms >15°/s
    """

    def __init__(
        self,
        n_taps: int = 8,
        mu: float = 0.5,
        gate_threshold: float = 0.5,
        nlms_threshold: float = 0.15,
    ):
        self.gate_threshold = gate_threshold
        self.nlms_threshold = nlms_threshold
        self._filters = {ch: NLMSFilter(n_taps=n_taps, mu=mu) for ch in _EEG_CHANNELS}
        self._last_motion_level: str = "still"
        self._consecutive_gated: int = 0  # 连续门控窗口计数

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
            self._consecutive_gated += 1
            # 连续3个窗口大幅运动才清零权重，避免短暂运动后立即丢失收敛状态
            if self._consecutive_gated >= 3:
                for f in self._filters.values():
                    f.reset()
            return None

        self._consecutive_gated = 0  # 非 gated 窗口重置计数

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
        """根据合加速度偏差、加速度方差、陀螺仪幅度联合判断运动级别"""
        acc = imu_df[["AccX", "AccY", "AccZ"]].values
        acc_norm = np.sqrt((acc ** 2).sum(axis=1))
        # p95 而非 max，避免单点噪声误触发门控
        acc_dev = np.percentile(np.abs(acc_norm - _GRAVITY), 95)
        acc_std = acc_norm.std()  # 振动/震颤检测（倾斜不变，振动会升高方差）

        gyr = imu_df[["GyrX", "GyrY", "GyrZ"]].values
        # p95 而非 max，避免陀螺仪单点噪声误触发门控
        gyr_norm = np.percentile(np.sqrt((gyr ** 2).sum(axis=1)), 95)

        if acc_dev > self.gate_threshold or gyr_norm > _GYR_GATE_THRESHOLD:
            return "gated"
        if acc_dev > self.nlms_threshold or acc_std > 0.05 or gyr_norm > _GYR_NLMS_THRESHOLD:
            return "nlms"
        return "still"

    def _interpolate_imu(
        self, eeg_times: np.ndarray, imu_df: pd.DataFrame
    ) -> np.ndarray:
        """将 52 Hz IMU 三次样条插值到 EEG 时间轴，返回 (n_samples, 6)"""
        from scipy.interpolate import CubicSpline
        result = np.zeros((len(eeg_times), 6))
        imu_times = imu_df["time"].values

        # LSL 时间戳可能有重复或轻微乱序，CubicSpline 要求严格递增
        order = np.argsort(imu_times, kind="stable")
        imu_times = imu_times[order]
        imu_data = imu_df[_IMU_CHANNELS].values[order]
        # 去除重复时间戳（保留每组第一个）
        unique_mask = np.concatenate(([True], np.diff(imu_times) > 0))
        imu_times = imu_times[unique_mask]
        imu_data = imu_data[unique_mask]

        for i, col in enumerate(_IMU_CHANNELS):
            imu_vals = imu_data[:, i]
            if len(imu_times) >= 3:
                cs = CubicSpline(imu_times, imu_vals, extrapolate=False)
                vals = cs(eeg_times)
                vals = np.where(eeg_times < imu_times[0], imu_vals[0], vals)
                vals = np.where(eeg_times > imu_times[-1], imu_vals[-1], vals)
                # imu_vals 含 NaN 时样条会扩散 NaN 到相邻区间，用线性插值填补
                nan_mask = np.isnan(vals)
                if nan_mask.any():
                    vals[nan_mask] = np.interp(eeg_times[nan_mask], imu_times, imu_vals)
                result[:, i] = vals
            else:
                # IMU 样本太少（<3），回退到线性插值
                result[:, i] = np.interp(eeg_times, imu_times, imu_vals)
        return result
