"""
实时认知负荷检测器

接收 RealTimeFeatureExtractor 输出的特征 DataFrame，
计算认知负荷指标并通过滑动窗口平滑输出。

典型集成方式:
    feat_extractor = RealTimeFeatureExtractor(...)
    cl_detector    = RealTimeCognitiveLoadDetector(smooth_window=5)

    for df_features in feat_extractor.add(df_preprocessed):
        result = cl_detector.add(df_features)
        if result is not None:
            print(result)   # {'cog_load_score': 0.62, 'level': 'medium', ...}
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "Xmuse-Connect" / "LSL"))
from data_saver import DataSaver

from cognitive_load_index import CognitiveLoadIndex

log = logging.getLogger(__name__)

_LEVEL_THRESHOLDS = [
    (0.65, "high"),
    (0.40, "medium"),
    (0.0,  "low"),
]

_LEVEL_INT = {"low": 0, "medium": 1, "high": 2}

_CL_COLUMNS = ["cog_load_score", "cognitive_load_index", "level"]


class _IdentityCorrector:
    def correct(self, timestamps):
        return np.asarray(timestamps, dtype=np.float64)


def _score_to_level(score: float) -> str:
    for threshold, level in _LEVEL_THRESHOLDS:
        if score >= threshold:
            return level
    return "low"


class RealTimeCognitiveLoadDetector:
    """
    实时认知负荷检测器（独立模块）

    接收特征 DataFrame → 计算 Theta/Alpha 认知负荷指标 → 滑动窗口平滑 → 输出结果字典

    参数:
        smooth_window   : 平滑滑动窗口大小（epoch 数），默认 5
        channels        : 参与计算的 EEG 通道，默认 CH1-CH4
        output_dir      : 结果保存根目录；None 表示不保存
        segment_seconds : 每个 segment 文件时长（秒），默认 60
    """

    def __init__(
        self,
        smooth_window: int = 5,
        channels: list[str] | None = None,
        output_dir: str | Path | None = None,
        segment_seconds: float = 60.0,
    ):
        self._indexer = CognitiveLoadIndex(channels=channels)
        self._window: int = max(1, smooth_window)
        self._score_buf: Deque[float] = deque(maxlen=self._window)
        self._ci_buf:    Deque[float] = deque(maxlen=self._window)

        self._saver: Optional[DataSaver] = None
        self._save_dir: Optional[Path] = None
        self._segment_seconds = segment_seconds
        if output_dir is not None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._save_dir = Path(output_dir) / session_name
            self._save_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            f"RealTimeCognitiveLoadDetector 初始化: smooth_window={self._window}"
            + (f"  输出→{self._save_dir}" if self._save_dir else "")
        )

    def add(self, df_features: pd.DataFrame) -> Optional[dict]:
        """
        处理一批特征，每行独立产生一个平滑认知负荷结果

        返回字典字段:
            timestamp          : 最新 epoch 时间戳（若有）
            cog_load_score     : 平滑负荷得分 [0, 1]
            cognitive_load_index : 平滑 θ/α 比值
            level              : 负荷等级 "high" / "medium" / "low"
        """
        if df_features is None or df_features.empty:
            return None

        df_cl = self._indexer.compute(df_features)
        if df_cl.empty:
            return None

        last_result: Optional[dict] = None
        has_ts = "timestamp" in df_cl.columns

        for _, row in df_cl.iterrows():
            self._score_buf.append(row["cog_load_score"])
            self._ci_buf.append(row["cognitive_load_index"])

            if len(self._score_buf) < self._window:
                continue

            result: dict = {
                "cog_load_score":      float(np.mean(self._score_buf)),
                "cognitive_load_index": float(np.mean(self._ci_buf)),
            }
            result["level"] = _score_to_level(result["cog_load_score"])

            if has_ts:
                result["timestamp"] = float(row["timestamp"])

            log.debug(
                f"认知负荷={result['cog_load_score']:.3f} "
                f"等级={result['level']} "
                f"CI={result['cognitive_load_index']:.3f}"
            )

            if self._save_dir is not None:
                self._feed_saver(result)

            last_result = result

        return last_result

    def close(self) -> None:
        if self._saver is not None:
            self._saver.close()
        log.info("RealTimeCognitiveLoadDetector 已关闭")

    def reset(self) -> None:
        self._score_buf.clear()
        self._ci_buf.clear()
        log.info("RealTimeCognitiveLoadDetector 缓冲已重置")

    def _feed_saver(self, result: dict) -> None:
        if self._saver is None:
            self._saver = DataSaver(
                save_dir=str(self._save_dir),
                stream_type="cognitive_load",
                ch_names=_CL_COLUMNS,
                corrector=_IdentityCorrector(),
                nominal_srate=1.0,
                segment_seconds=self._segment_seconds,
            )
        sample = [
            result["cog_load_score"],
            result["cognitive_load_index"],
            float(_LEVEL_INT.get(result.get("level", "low"), 0)),
        ]
        ts = result.get("timestamp", 0.0)
        self._saver.add([sample], [ts])

    @property
    def latest_score(self) -> Optional[float]:
        return self._score_buf[-1] if self._score_buf else None
