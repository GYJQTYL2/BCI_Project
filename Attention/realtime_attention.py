"""
实时注意力检测器

接收 RealTimeFeatureExtractor 输出的特征 DataFrame，
计算注意力指标并通过滑动窗口平滑输出。

典型集成方式:
    feat_extractor   = RealTimeFeatureExtractor(...)
    attention_detect = RealTimeAttentionDetector(smooth_window=5)

    for df_features in feat_extractor.add(df_preprocessed):
        result = attention_detect.add(df_features)
        if result is not None:
            print(result)   # {'attention_score': 0.72, 'level': 'high', ...}
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "Xmuse-Connect" / "LSL"))
from data_saver import DataSaver

from attention_index import AttentionIndex

log = logging.getLogger(__name__)

# 注意力等级阈值 (attention_score)
_LEVEL_THRESHOLDS = [
    (0.65, "high"),    # >= 0.65 → 高度集中
    (0.40, "medium"),  # >= 0.40 → 中度集中
    (0.0,  "low"),     # < 0.40  → 低度/走神
]

_LEVEL_INT = {"low": 0, "medium": 1, "high": 2}   # level → 存储编码

_ATTN_COLUMNS = ["attention_score", "engagement_index", "theta_alpha_ratio", "level"]


class _IdentityCorrector:
    """透传时间戳，不做任何校正"""
    def correct(self, timestamps):
        return np.asarray(timestamps, dtype=np.float64)


def _score_to_level(score: float) -> str:
    for threshold, level in _LEVEL_THRESHOLDS:
        if score >= threshold:
            return level
    return "low"


class RealTimeAttentionDetector:
    """
    实时注意力检测器（独立模块）

    接收特征 DataFrame → 计算注意力指标 → 滑动窗口平滑 → 输出结果字典
    可选将结果分 segment 保存到 CSV（格式与特征文件一致）。

    参数:
        smooth_window        : 平滑用滑动窗口大小（epoch 数），默认 5
        channels             : 参与计算的 EEG 通道，默认 CH1-CH4
        attention_output_dir : 结果保存根目录；None 表示不保存
        segment_seconds      : 每个 segment 文件的时长（秒），默认 60
    """

    def __init__(
        self,
        smooth_window: int = 5,
        channels: list[str] | None = None,
        attention_output_dir: str | Path | None = None,
        segment_seconds: float = 60.0,
    ):
        self._indexer = AttentionIndex(channels=channels)
        self._window: int = max(1, smooth_window)
        self._score_buf: Deque[float] = deque(maxlen=self._window)
        self._ei_buf:    Deque[float] = deque(maxlen=self._window)
        self._tar_buf:   Deque[float] = deque(maxlen=self._window)

        # DataSaver（懒初始化，首次输出结果时创建）
        self._saver: Optional[DataSaver] = None
        self._save_dir: Optional[Path] = None
        self._segment_seconds = segment_seconds
        if attention_output_dir is not None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._save_dir = Path(attention_output_dir) / session_name
            self._save_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            f"RealTimeAttentionDetector 初始化: smooth_window={self._window}"
            + (f"  输出→{self._save_dir}" if self._save_dir else "")
        )

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def add(self, df_features: pd.DataFrame) -> Optional[dict]:
        """
        处理一批特征（可含多行），每行独立产生一个平滑注意力结果

        参数:
            df_features : EEGFeatureExtractor 输出的特征 DataFrame（可含多行）
        返回:
            最后一个 epoch 的结果字典（有足够数据时）或 None（缓冲尚未填满）

            字典字段:
                timestamp          : 最新 epoch 时间戳（若有）
                attention_score    : 平滑注意力得分 [0, 1]
                engagement_index   : 平滑参与度指数
                theta_alpha_ratio  : 平滑 θ/α 比值
                level              : 注意力等级 "high" / "medium" / "low"
        """
        if df_features is None or df_features.empty:
            return None

        df_attn = self._indexer.compute(df_features)
        if df_attn.empty:
            return None

        last_result: Optional[dict] = None
        has_ts = "timestamp" in df_attn.columns

        # 每个 epoch 独立推入缓冲并产生一条平滑结果，保证输出率与 epoch 率一致（~1 Hz）
        for _, row in df_attn.iterrows():
            self._score_buf.append(row["attention_score"])
            self._ei_buf.append(row["engagement_index"])
            self._tar_buf.append(row["theta_alpha_ratio"])

            if len(self._score_buf) < self._window:
                continue

            result: dict = {
                "attention_score":   float(np.mean(self._score_buf)),
                "engagement_index":  float(np.mean(self._ei_buf)),
                "theta_alpha_ratio": float(np.mean(self._tar_buf)),
            }
            result["level"] = _score_to_level(result["attention_score"])

            if has_ts:
                result["timestamp"] = float(row["timestamp"])

            log.debug(
                f"注意力得分={result['attention_score']:.3f} "
                f"等级={result['level']} "
                f"EI={result['engagement_index']:.3f}"
            )

            if self._save_dir is not None:
                self._feed_saver(result)

            last_result = result

        return last_result

    def close(self) -> None:
        """刷写剩余缓冲并关闭 DataSaver"""
        if self._saver is not None:
            self._saver.close()
        log.info("RealTimeAttentionDetector 已关闭")

    # ── 内部实现 ──────────────────────────────────────────────────────────

    def _feed_saver(self, result: dict) -> None:
        """将一条注意力结果写入 DataSaver。level 编码为整数（low=0/medium=1/high=2）。"""
        if self._saver is None:
            self._saver = DataSaver(
                save_dir=str(self._save_dir),
                stream_type="attention",
                ch_names=_ATTN_COLUMNS,
                corrector=_IdentityCorrector(),
                nominal_srate=1.0,          # 每 epoch 一条记录，约 1 Hz
                segment_seconds=self._segment_seconds,
            )
            log.info(f"DataSaver 初始化: {len(_ATTN_COLUMNS)} 列")

        sample = [
            result["attention_score"],
            result["engagement_index"],
            result["theta_alpha_ratio"],
            float(_LEVEL_INT.get(result.get("level", "low"), 0)),
        ]
        ts = result.get("timestamp", 0.0)
        self._saver.add([sample], [ts])

    def reset(self) -> None:
        """清空平滑缓冲（更换实验场景时调用）"""
        self._score_buf.clear()
        self._ei_buf.clear()
        self._tar_buf.clear()
        log.info("RealTimeAttentionDetector 缓冲已重置")

    @property
    def latest_score(self) -> Optional[float]:
        """返回当前缓冲中最新的原始（未平滑）注意力得分，缓冲为空时返回 None"""
        return self._score_buf[-1] if self._score_buf else None
