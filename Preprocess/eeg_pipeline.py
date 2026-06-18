"""
EEG预处理Pipeline，封装各步骤模块，针对Xmuse-Connect采集的EEG数据进行完整预处理。

完整流程:
    Step 1 - 数据清洗与列名标准化  (data_clean.py     → clean_eeg_frame)
    Step 2 - 基线校正              (data_baseline.py  → correct_dc_offset / correct_baseline_channelwise)
    Step 3 - 频域滤波与坏导修复    (data_filter.py    → filter_eeg_frame)
    Step 4 - 极值检测与插值修复    (data_AmpRemove.py → interpolate_outliers)
    Step 5 - 数据缩放              (data_scaler.py    → normalize_channel / standardize_channel)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from data_clean import clean_eeg_frame
from data_baseline import correct_dc_offset, correct_baseline_channelwise
from data_filter import filter_eeg_frame
from data_AmpRemove import interpolate_outliers
from data_scaler import normalize_channel, standardize_channel
from asr_cleaner import ASRCleaner


class EEGPreprocessPipeline:
    """
    Xmuse-Connect EEG数据预处理Pipeline

    用法示例:
        pipeline = EEGPreprocessPipeline()
        df = pipeline.run("subject01.csv", output_dir="processed/")

        # 批量处理
        results = pipeline.run_batch(["s01.csv", "s02.csv"], output_dir="processed/")

    参数说明:
        baseline_method     : 基线校正方法，'channelwise'(默认) 或 'dc_offset'
        baseline_window     : channelwise模式下的基线时间窗口(秒)，默认前200ms
        dc_offset_val       : dc_offset模式下的固定偏移量(µV)，默认800
        hp_cutoff           : 高通截止频率(Hz)，默认0.5
        lp_cutoff           : 低通截止频率(Hz)，默认45
        amplitude_threshold : 极值判定阈值(µV)，默认100
        scale_method        : 缩放方法，'zscore'(默认) 或 'minmax'
        channels            : EEG通道列表，默认['CH1','CH2','CH3','CH4']
        save_intermediates  : 是否保存中间文件，默认True
        output_suffix       : 最终输出文件后缀，默认'_preprocessed'
    """

    def __init__(
        self,
        baseline_method: Literal["channelwise", "dc_offset"] = "channelwise",
        baseline_window: Tuple[float, float] = (0.0, 0.2),
        dc_offset_val: float = 800.0,
        hp_cutoff: float = 1,#0.5,
        lp_cutoff: float = 30.0,
        amplitude_threshold: float = 150.0,
        scale_method: Literal["zscore", "minmax"] = "zscore",
        channels: Optional[List[str]] = None,
        save_intermediates: bool = True,
        output_suffix: str = "_preprocessed",
        asr_cleaner: Optional["ASRCleaner"] = None,
    ):
        self.baseline_method = baseline_method
        self.baseline_window = baseline_window
        self.dc_offset_val = dc_offset_val
        self.hp_cutoff = hp_cutoff
        self.lp_cutoff = lp_cutoff
        self.amplitude_threshold = amplitude_threshold
        self.scale_method = scale_method
        self.channels = channels or ["CH1", "CH2", "CH3", "CH4"]
        self.save_intermediates = save_intermediates
        self.output_suffix = output_suffix
        self.asr_cleaner = asr_cleaner

    # ── Step 1 ────────────────────────────────
    def _step1_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗与列名标准化"""
        return clean_eeg_frame(df)

    # ── Step 2 ────────────────────────────────
    def _step2_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """基线校正"""
        fs = 1.0 / np.mean(np.diff(df["time"]))
        if self.baseline_method == "dc_offset":
            return correct_dc_offset(df, offset_val=self.dc_offset_val, channels=self.channels)
        return correct_baseline_channelwise(df, self.baseline_window, fs, self.channels)

    # ── Step 3 ────────────────────────────────
    def _step3_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """频域滤波与坏导修复"""
        return filter_eeg_frame(df, self.hp_cutoff, self.lp_cutoff, self.channels)

    # ── Step 4 ────────────────────────────────
    def _step4_amp_remove(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """极值检测与插值修复"""
        return interpolate_outliers(df, self.amplitude_threshold)

    # ── Step 5 ────────────────────────────────
    def _step5_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据缩放"""
        df_out = df.copy()
        for ch in self.channels:
            if ch not in df_out.columns:
                continue
            if self.scale_method == "minmax":
                df_out[ch] = normalize_channel(df_out[ch])
            else:
                df_out[ch] = standardize_channel(df_out[ch])
        return df_out

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对 DataFrame 执行完整5步处理，不读写任何文件，直接返回处理后的 DataFrame。
        用于实时处理场景（RealTimeEEGProcessor）。
        """
        df = self._step1_clean(df)
        if df.empty:
            return df
        df = self._step2_baseline(df)
        # 滤波前：高阈值只拦截电极脱落/极端眨眼（原始信号含慢漂移，不能用小阈值）
        df, _ = interpolate_outliers(df, 800.0)
        df = self._step3_filter(df)
        # ASR：去 EOG + EMG（仅在训练好基线后生效）
        if self.asr_cleaner is not None and self.asr_cleaner.is_ready:
            df = self.asr_cleaner.clean(df)
        # 滤波后：低阈值清理残留尖峰（高通已去慢漂移，信号幅值已降至 ±100 µV 级别）
        #df, _ = self._step4_amp_remove(df)
        #df = self._step5_scale(df)
        return df

    def process_for_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        运行 Step1-3（清洗→基线校正→滤波），不含 ASR。
        供 ASR 基线录制使用，确保训练数据与 transform 时的信号频谱一致。
        """
        df = self._step1_clean(df)
        if df.empty:
            return df
        df = self._step2_baseline(df)
        df, _ = interpolate_outliers(df, 800.0)
        df = self._step3_filter(df)
        return df

    # ── 核心运行逻辑 ───────────────────────────
    def _run_pipeline(self, df: pd.DataFrame, stem: str, out_dir: Path) -> pd.DataFrame:
        """内部：对已加载的 DataFrame 执行完整5步处理并保存结果"""
        out_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: 数据清洗
        print("  [1/5] 数据清洗...")
        df = self._step1_clean(df)
        if df.empty:
            print("  警告: 清洗后数据为空，跳过")
            return df
        if self.save_intermediates:
            df.to_csv(out_dir / f"{stem}_cleaned.csv", index=False)

        # Step 2: 基线校正
        print("  [2/5] 基线校正...")
        df = self._step2_baseline(df)
        if self.save_intermediates:
            df.to_csv(out_dir / f"{stem}_baseline.csv", index=False)

        # 滤波前：高阈值只拦截电极脱落/极端眨眼
        print("  [3/5] 极值去除...")
        df, n_fixed = interpolate_outliers(df, 800.0)
        print(f"        共修复 {n_fixed} 个极值点")
        if self.save_intermediates:
            df.to_csv(out_dir / f"{stem}_removed.csv", index=False)

        # Step 3: 频域滤波
        print("  [4/5] 频域滤波...")
        df = self._step3_filter(df)
        if self.save_intermediates:
            df.to_csv(out_dir / f"{stem}_filtered.csv", index=False)

        # Step 5: 数据缩放
        #print("  [5/5] 数据缩放...")
        #df = self._step5_scale(df)

        output_path = out_dir / f"{stem}{self.output_suffix}.csv"
        df.to_csv(output_path, index=False)
        print(f"  完成 → {output_path}")

        return df

    def run(self, input_path: str, output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        从CSV文件路径执行完整预处理Pipeline

        参数:
            input_path : 原始CSV文件路径
            output_dir : 输出目录，默认与input_path同级
        返回:
            预处理完成的DataFrame
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"文件不存在: {input_path}")

        out_dir = Path(output_dir) if output_dir else input_path.parent
        print(f"\n{'='*55}")
        print(f"处理: {input_path.name}")

        df = pd.read_csv(input_path, na_values=[""])
        return self._run_pipeline(df, input_path.stem, out_dir)

    def run_df(
        self,
        df: pd.DataFrame,
        name: str,
        output_dir: str,
    ) -> pd.DataFrame:
        """
        从已加载的 DataFrame 执行完整预处理Pipeline

        参数:
            df         : 原始数据（列名需符合 data_clean 期望格式）
            name       : 输出文件名前缀（不含扩展名）
            output_dir : 输出目录
        返回:
            预处理完成的DataFrame
        """
        out_dir = Path(output_dir)
        print(f"\n{'='*55}")
        print(f"处理: {name}")
        return self._run_pipeline(df, name, out_dir)

    def run_batch(
        self,
        file_list: List[str],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        批量处理多个CSV文件

        参数:
            file_list  : CSV文件路径列表
            output_dir : 统一输出目录，默认与各文件同级
        返回:
            {文件路径: 处理后DataFrame} 的字典，失败时对应值为None
        """
        results: Dict[str, Optional[pd.DataFrame]] = {}
        print(f"批量处理 {len(file_list)} 个文件...")
        for fp in file_list:
            try:
                results[fp] = self.run(fp, output_dir=output_dir)
            except Exception as e:
                print(f"  处理失败 [{fp}]: {e}")
                results[fp] = None

        success = sum(v is not None for v in results.values())
        print(f"\n批量处理完成：{success}/{len(file_list)} 成功")
        return results
