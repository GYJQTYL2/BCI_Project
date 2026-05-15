"""
TimestampCorrector — LSL 时间戳校正与去抖动

解决原始代码的两个问题：
  1. time_correction() 只在启动时调用一次，长时间采集会出现时钟漂移
  2. 每个 chunk 独立做线性回归，跨 chunk 边界可能出现时间戳跳变

改进点：
  - 定期自动刷新 time_correction（默认每 60 秒一次）
  - 去抖动使用全局样本计数器作为 X，保证跨 chunk 的时间戳连续性
  - 接口简单：correct(lsl_timestamps) → 校正后的 numpy 数组

用法：
    corrector = TimestampCorrector(inlet, dejitter=True)

    # 采集循环中
    samples, lsl_ts = inlet.pull_chunk(...)
    corrected_ts = corrector.correct(lsl_ts)

    # 长时间采集后会自动刷新 time_correction，也可手动刷新：
    corrector.refresh()
"""

import logging
import time
from typing import List, Union

import numpy as np
import pylsl
from sklearn.linear_model import LinearRegression

log = logging.getLogger(__name__)


class TimestampCorrector:
    """
    将 LSL 时间戳转换为本地系统时间，并可选地做线性回归去抖动。

    参数
    ----
    inlet : pylsl.StreamInlet
        已连接的 LSL inlet，用于获取时钟偏移。
    dejitter : bool
        是否对时间戳做线性回归去抖动（默认 True）。
    correction_interval : float
        自动刷新 time_correction 的间隔（秒，默认 60）。
        设为 0 则禁用自动刷新（使用初始值）。
    dejitter_window : int
        去抖动使用的滑动窗口大小（样本数）。
        None 表示对每个 chunk 单独拟合（简单但可能有边界跳变）。
        建议设为 nominal_srate * 10（10 秒），可跨 chunk 保持连续性。
    """

    def __init__(
        self,
        inlet: pylsl.StreamInlet,
        dejitter: bool = True,
        correction_interval: float = 60.0,
        dejitter_window: int = None,
    ):
        self._inlet = inlet
        self._dejitter = dejitter
        self._correction_interval = correction_interval
        self._dejitter_window = dejitter_window

        # 全局样本计数器：作为去抖线性回归的 X 轴
        # 这样跨 chunk 的斜率估计是连续的，不会产生边界跳变
        self._global_sample_idx = 0

        # 滑动窗口缓存（用于 dejitter_window 模式）
        self._window_x: List[float] = []
        self._window_ts: List[float] = []

        # 初始 time_correction
        self._correction: float = self._fetch_correction()
        self._last_refresh: float = time.monotonic()

        log.info(
            f"[TimestampCorrector] 初始化完成 "
            f"correction={self._correction:.4f}s  "
            f"dejitter={dejitter}  "
            f"refresh_interval={correction_interval}s"
        )

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def correct(
        self, lsl_timestamps: Union[List[float], np.ndarray]
    ) -> np.ndarray:
        """
        对一批 LSL 时间戳进行校正（+ 可选去抖动）。

        参数
        ----
        lsl_timestamps : list 或 ndarray
            从 inlet.pull_chunk() 得到的原始 LSL 时间戳。

        返回
        ----
        np.ndarray
            校正到本地时钟的时间戳。
        """
        if len(lsl_timestamps) == 0:
            return np.array([])

        self._maybe_refresh()

        ts = np.asarray(lsl_timestamps, dtype=np.float64) + self._correction

        if self._dejitter and len(ts) > 1:
            ts = self._apply_dejitter(ts)

        self._global_sample_idx += len(ts)
        return ts

    def refresh(self) -> float:
        """立即重新获取 time_correction，返回新的偏移量。"""
        self._correction = self._fetch_correction()
        self._last_refresh = time.monotonic()
        log.info(f"[TimestampCorrector] 刷新 time_correction={self._correction:.4f}s")
        return self._correction

    @property
    def correction(self) -> float:
        """当前使用的时钟偏移量（秒）。"""
        return self._correction

    @property
    def total_samples(self) -> int:
        """已处理的累计样本数。"""
        return self._global_sample_idx

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _maybe_refresh(self) -> None:
        if self._correction_interval <= 0:
            return
        if time.monotonic() - self._last_refresh >= self._correction_interval:
            self.refresh()

    def _fetch_correction(self) -> float:
        try:
            return self._inlet.time_correction()
        except Exception as e:
            log.warning(f"[TimestampCorrector] time_correction() 失败: {e}，使用 0.0")
            return 0.0

    def _apply_dejitter(self, ts: np.ndarray) -> np.ndarray:
        """
        线性回归去抖动。

        X 轴使用全局样本计数器，保证多次调用之间斜率连续。
        若启用 dejitter_window，则用滑动窗口拟合，效果更稳定。
        """
        n = len(ts)
        x_start = self._global_sample_idx
        x_local = np.arange(x_start, x_start + n, dtype=np.float64).reshape(-1, 1)

        if self._dejitter_window is not None:
            # 滑动窗口模式：把当前样本加入窗口，用窗口数据拟合
            self._window_x.extend(x_local.ravel().tolist())
            self._window_ts.extend(ts.tolist())
            # 只保留最近 dejitter_window 个样本
            if len(self._window_x) > self._dejitter_window:
                self._window_x = self._window_x[-self._dejitter_window:]
                self._window_ts = self._window_ts[-self._dejitter_window:]

            wx = np.array(self._window_x).reshape(-1, 1)
            wt = np.array(self._window_ts)

            if len(wx) < 2:
                return ts

            lr = LinearRegression().fit(wx, wt)
            return lr.predict(x_local)
        else:
            # 每 chunk 独立拟合（与原代码行为一致，但 X 从全局计数器开始）
            lr = LinearRegression().fit(x_local, ts)
            return lr.predict(x_local)
