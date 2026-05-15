"""
批量特征提取

读取 Preprocess/signal_data 中保存的预处理 EEG CSV 文件，
逐文件提取特征，通过 DataSaver 分 segment 保存，同时返回特征 DataFrame。

目录结构（输入）:
    signal_data/
    └── YYYYMMDD_HHMMSS/
        ├── EEG_processed_seg0001_preprocessed.csv
        └── ...

目录结构（输出）:
    feature_output/
    └── YYYYMMDD_HHMMSS/
        ├── EEG_processed_seg0001/
        │   └── features_seg0001.csv
        └── EEG_processed_seg0002/
            └── features_seg0001.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

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


class BatchFeatureExtractor:
    """
    批量EEG特征提取器

    扫描预处理结果目录，对每个 CSV 文件提取特征并通过 DataSaver 保存。

    用法示例:
        extractor = EEGFeatureExtractor(fs=256.0, epoch_seconds=1.0)
        batch = BatchFeatureExtractor(extractor, output_dir="FeatureExtract/features_output")

        # 处理单个文件
        df_feat = batch.run("signal_data/20260428/EEG_seg0001_preprocessed.csv")

        # 批量处理目录
        results = batch.run_dir("signal_data/20260428/")

        # 递归扫描所有 session
        results = batch.run_all("signal_data/")

    参数说明:
        extractor    : EEGFeatureExtractor 实例
        output_dir   : 特征文件保存根目录
        file_suffix  : 要匹配的文件后缀，默认 '_preprocessed.csv'
    """

    def __init__(
        self,
        extractor: EEGFeatureExtractor,
        output_dir: str | Path,
        file_suffix: str = "_preprocessed.csv",
    ):
        self.extractor = extractor
        self.output_dir = Path(output_dir)
        self.file_suffix = file_suffix

    # ── 单文件处理 ─────────────────────────────────────────────────────────

    def run(
        self,
        input_path: str | Path,
        output_dir: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """
        对单个预处理 CSV 文件提取特征

        参数:
            input_path : 预处理后的 CSV 文件路径
            output_dir : 特征文件保存目录，默认使用构造时的 output_dir
        返回:
            特征 DataFrame（每行为一个 epoch）
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"文件不存在: {input_path}")

        stem = input_path.stem.replace("_preprocessed", "")
        out_dir = Path(output_dir) if output_dir else self.output_dir
        # 每个输入文件的特征保存到独立子目录，避免 DataSaver 多文件命名冲突
        save_dir = out_dir / stem
        save_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"提取特征: {input_path.name}")
        df = pd.read_csv(input_path)
        df_features = self.extractor.extract(df)

        if df_features.empty:
            log.warning(f"  特征为空，跳过保存: {input_path.name}")
            return df_features

        self._save(df_features, save_dir)
        log.info(f"  {len(df_features)} 个 epoch → {save_dir}/")
        return df_features

    # ── 目录批量处理 ──────────────────────────────────────────────────────

    def run_dir(
        self,
        input_dir: str | Path,
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        处理单个 session 目录下所有匹配的预处理文件

        参数:
            input_dir  : session 目录（如 signal_data/20260428_100457/）
            output_dir : 特征文件保存目录，默认在 output_dir 下按 session 命名
        返回:
            {文件名: 特征 DataFrame} 字典
        """
        input_dir = Path(input_dir)
        session_name = input_dir.name
        out_dir = Path(output_dir) if output_dir else self.output_dir / session_name

        results: Dict[str, pd.DataFrame] = {}
        files = sorted(input_dir.glob(f"*{self.file_suffix}"))
        if not files:
            log.warning(f"目录中未找到 *{self.file_suffix} 文件: {input_dir}")
            return results

        log.info(f"处理 session: {session_name}  ({len(files)} 个文件)")
        for fp in files:
            try:
                results[fp.name] = self.run(fp, output_dir=out_dir)
            except Exception:
                log.exception(f"  处理失败: {fp.name}")
                results[fp.name] = pd.DataFrame()

        return results

    # ── 递归扫描所有 session ──────────────────────────────────────────────

    def run_all(
        self,
        signal_data_root: str | Path,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        递归扫描 signal_data 根目录下所有 session，批量提取特征

        参数:
            signal_data_root : signal_data 根目录
        返回:
            {session 名: {文件名: 特征 DataFrame}} 嵌套字典
        """
        root = Path(signal_data_root)
        all_results: Dict[str, Dict[str, pd.DataFrame]] = {}

        sessions = sorted(d for d in root.iterdir() if d.is_dir())
        if not sessions:
            log.warning(f"未找到 session 目录: {root}")
            return all_results

        log.info(f"扫描根目录: {root}  ({len(sessions)} 个 session)")
        for session_dir in sessions:
            all_results[session_dir.name] = self.run_dir(session_dir)

        total = sum(len(v) for v in all_results.values())
        log.info(f"批量提取完成：{len(sessions)} 个 session，{total} 个文件")
        return all_results

    # ── 内部实现 ──────────────────────────────────────────────────────────

    def _save(self, df_features: pd.DataFrame, save_dir: Path) -> None:
        """通过 DataSaver 将特征 DataFrame 写入 CSV"""
        ch_names: List[str] = [c for c in df_features.columns if c != "timestamp"]
        nominal_srate = self.extractor.fs / self.extractor.epoch_size

        saver = DataSaver(
            save_dir=str(save_dir),
            stream_type="features",
            ch_names=ch_names,
            corrector=_IdentityCorrector(),
            nominal_srate=nominal_srate,
            segment_seconds=86400,   # 批量模式一次性写入，设超大值确保单 segment
        )

        samples: List[List[float]] = df_features[ch_names].values.tolist()
        timestamps: List[float] = (
            df_features["timestamp"].tolist()
            if "timestamp" in df_features.columns
            else [float(i) for i in range(len(samples))]
        )

        saver.add(samples, timestamps)
        saver.close()
