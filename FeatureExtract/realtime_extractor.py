"""
实时特征提取器（独立模块）

接收预处理后的 EEG DataFrame，提取特征，通过 DataSaver 分 segment 保存至 CSV，
并返回特征 DataFrame 供下游使用。与 RealTimeEEGProcessor 完全解耦。

典型集成方式:
    processor    = RealTimeEEGProcessor(output_dir=..., ...)
    feat_extract = RealTimeFeatureExtractor(feature_output_dir=..., ...)

    # 采集循环中
    for df_proc in processor.add(samples, timestamps):
        feat_extract.add(df_proc)

    # 结束时
    processor.close()
    feat_extract.close()
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Xmuse-Connect" / "LSL"))

from extractor import EEGFeatureExtractor
from data_saver import DataSaver

log = logging.getLogger(__name__)


class _IdentityCorrector:
    """透传时间戳，不做任何校正"""
    def correct(self, timestamps):
        return np.asarray(timestamps, dtype=np.float64)


class RealTimeFeatureExtractor:
    """
    实时EEG特征提取器（独立模块，不依赖任何采集或预处理类）

    接收预处理后的 EEG DataFrame（列名: time, CH1, CH2, CH3, CH4），
    提取特征后通过 DataSaver 分 segment 保存，同时返回特征 DataFrame。

    DataSaver 在首次收到特征数据时按实际特征列名懒初始化，
    因此无需预先指定特征名称。

    参数说明:
        feature_output_dir : 特征文件保存根目录
        extractor          : EEGFeatureExtractor 实例，不传则使用默认参数
        fs                 : 采样率（Hz），extractor 未传入时用于构造默认实例，默认 256
        epoch_seconds      : epoch 长度（秒），用于估算 DataSaver 的 nominal_srate，默认 1.0
        segment_seconds    : 每个输出 segment 文件的时长（秒），默认 60
    """

    def __init__(
        self,
        feature_output_dir: str | Path,
        extractor: Optional[EEGFeatureExtractor] = None,
        fs: float = 256.0,
        epoch_seconds: float = 1.0,
        segment_seconds: float = 60.0,
    ):
        self._extractor = extractor or EEGFeatureExtractor(fs=fs, epoch_seconds=epoch_seconds)
        self._epoch_srate = 1.0 / epoch_seconds   # 特征的"采样率"（每秒 epoch 数）
        self._segment_seconds = segment_seconds

        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._feat_dir = Path(feature_output_dir) / session_name
        self._feat_dir.mkdir(parents=True, exist_ok=True)

        # DataSaver 懒初始化：第一次 add() 时按实际特征列名建立
        self._saver: Optional[DataSaver] = None

        log.info(f"RealTimeFeatureExtractor 初始化: 输出→{self._feat_dir}")

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def add(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        接收一个预处理后的 EEG 窗口 DataFrame，提取特征，保存并返回

        参数:
            df : 预处理后的 EEG DataFrame（time, CH1, CH2, CH3, CH4）
        返回:
            特征 DataFrame（每行对应一个 epoch），空 DataFrame 表示处理失败
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df_features = self._extractor.extract(df)
        if df_features.empty:
            return df_features

        self._ensure_saver(df_features)
        self._feed_saver(df_features)

        log.debug(f"提取 {len(df_features)} 个 epoch 特征")
        return df_features

    def close(self) -> None:
        """刷写剩余缓冲并关闭 DataSaver"""
        if self._saver is not None:
            self._saver.close()
        log.info(f"RealTimeFeatureExtractor 已关闭: {self._feat_dir}")

    # ── 内部实现 ──────────────────────────────────────────────────────────

    def _ensure_saver(self, df_features: pd.DataFrame) -> None:
        """首次收到特征数据时，按实际列名懒初始化 DataSaver"""
        if self._saver is not None:
            return
        ch_names = [c for c in df_features.columns if c != "timestamp"]
        self._saver = DataSaver(
            save_dir=str(self._feat_dir),
            stream_type="features",
            ch_names=ch_names,
            corrector=_IdentityCorrector(),
            nominal_srate=self._epoch_srate,
            segment_seconds=self._segment_seconds,
        )
        log.info(f"DataSaver 初始化: {len(ch_names)} 个特征列")

    def _feed_saver(self, df_features: pd.DataFrame) -> None:
        """将特征 DataFrame 转为 DataSaver 期望的格式并写入"""
        ch_names = [c for c in df_features.columns if c != "timestamp"]
        samples: List[List[float]] = df_features[ch_names].values.tolist()

        if "timestamp" in df_features.columns:
            timestamps: List[float] = df_features["timestamp"].tolist()
        else:
            # 无时间戳时用递增占位
            timestamps = [float(i) for i in range(len(samples))]

        self._saver.add(samples, timestamps)
