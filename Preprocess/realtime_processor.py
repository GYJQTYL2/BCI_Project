"""
实时EEG数据处理器

从 LSL 采集流实时接收 EEG 样本，积累到处理窗口大小后运行预处理 pipeline，
将处理好的数据通过 DataSaver 按 segment 保存到 Preprocess/signal_data 目录。

接口与 DataSaver.add() 完全兼容，可在 LSL 采集循环中与原始 DataSaver 并行使用:

    raw_saver  = DataSaver(...)        # 保存原始数据
    processor  = RealTimeEEGProcessor(...)  # 保存处理后数据

    # 采集循环中同时调用（pipeline 在后台按窗口批量运行）
    raw_saver.add(samples, lsl_timestamps)
    processor.add(samples, lsl_timestamps)

处理窗口建议 >= 5 秒（filtfilt 需要足够的样本才能稳定工作）。
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
from eeg_pipeline import EEGPreprocessPipeline
from asr_cleaner import ASRCleaner
from imu_artifact_remover import IMUArtifactRemover
from blink_artifact_remover import BlinkArtifactRemover

_QA_DIR = Path(__file__).parent.parent / "QualityAssess"
sys.path.insert(0, str(_QA_DIR))
from channel_quality import assess_window, interpolate_saturated

_LSL_DIR = Path(__file__).parent.parent / "Xmuse-Connect" / "LSL"
sys.path.insert(0, str(_LSL_DIR))
from data_saver import DataSaver

log = logging.getLogger(__name__)

# pipeline 处理后的输出通道名（data_clean.py step1 重命名后）
_PROCESSED_CH = ["CH1", "CH2", "CH3", "CH4"]

# 传入 pipeline 的列名（data_clean.py rename_map 期望的原始列名）
_PIPELINE_INPUT_CH = ["eeg_1", "eeg_2", "eeg_3", "eeg_4"]


class _IdentityCorrector:
    """透传时间戳，不做任何校正（output DataSaver 专用）"""

    def correct(self, timestamps):
        return np.asarray(timestamps, dtype=np.float64)


class RealTimeEEGProcessor:
    """
    实时EEG数据处理器

    接收来自 LSL 采集的原始 EEG 样本，按窗口批量运行预处理 pipeline，
    将处理结果通过 DataSaver 分 segment 保存到 Preprocess/signal_data。

    参数说明:
        output_dir      : 处理结果保存根目录（如 Preprocess/signal_data）
        nominal_srate   : 采样率（Hz），默认 256
        corrector       : 时间戳校正器（与采集端 DataSaver 共用同一个即可），
                          不传则透传时间戳（适合外部已完成校正的场景）
        window_seconds  : 处理窗口大小（秒），建议 >= 5；越大滤波越稳定
        pipeline        : 自定义 EEGPreprocessPipeline，不传则使用默认参数
        segment_seconds : 输出 segment 文件时长（秒），默认 60
    """

    def __init__(
        self,
        output_dir: str | Path,
        nominal_srate: float = 256.0,
        corrector=None,
        window_seconds: float = 5.0,
        pipeline: Optional[EEGPreprocessPipeline] = None,
        segment_seconds: float = 60.0,
    ):
        self.nominal_srate = nominal_srate
        self.window_size = int(window_seconds * nominal_srate)

        # ASR：尝试加载已有基线，没有则先不激活
        self._asr = ASRCleaner(sfreq=nominal_srate)
        loaded = self._asr.load_baseline()
        if not loaded:
            log.info("未找到 ASR 基线文件，请录制基线后 ASR 才会生效")

        self.pipeline = pipeline or EEGPreprocessPipeline(asr_cleaner=self._asr)
        self._input_corrector = corrector or _IdentityCorrector()

        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(output_dir) / session_name

        # 输出 DataSaver：保存处理后数据，时间戳已就绪，直接透传
        self._saver = DataSaver(
            save_dir=str(save_dir),
            stream_type="EEG_processed",
            ch_names=_PROCESSED_CH,
            corrector=_IdentityCorrector(),
            nominal_srate=nominal_srate,
            segment_seconds=segment_seconds,
        )

        self._buf_samples: List[List[float]] = []
        self._buf_timestamps: List[float] = []
        self._window_idx = 0
        self._recording_baseline = False
        self._baseline_fed_idx: int = 0  # 已喂入 ASR 基线缓冲的样本位置

        # 通道质量状态（每窗口更新）
        self._channel_quality: dict = {}   # {ch: assess() 结果}

        # IMU 缓冲（来自 LSL IMU 流，用于运动伪迹去除）
        self._imu_remover = IMUArtifactRemover()
        self._blink_remover = BlinkArtifactRemover()
        self._imu_buf_samples: List[List[float]] = []
        self._imu_buf_timestamps: List[float] = []
        self._imu_channels: List[str] = []

        log.info(
            f"RealTimeEEGProcessor 初始化: "
            f"输出→{save_dir}, 窗口={window_seconds}s ({self.window_size} 样本)"
        )

    # ── 公开接口（与 DataSaver.add / close 兼容）─────────────────────────

    def add_imu(
        self, samples: List[List[float]], lsl_timestamps: List[float],
        ch_names: Optional[List[str]] = None,
    ) -> None:
        """
        接收一批 IMU 样本（AccX/AccY/AccZ/GyrX/GyrY/GyrZ），缓存供 EEG 窗口使用。
        与 add() 并行调用，不需要对齐。
        """
        if not samples:
            return
        if ch_names and not self._imu_channels:
            self._imu_channels = ch_names
        self._imu_buf_samples.extend(samples)
        self._imu_buf_timestamps.extend(lsl_timestamps)
        # 只保留最近 30 秒的 IMU 数据，避免内存无限增长
        max_imu = int(52 * 30)
        if len(self._imu_buf_samples) > max_imu:
            self._imu_buf_samples = self._imu_buf_samples[-max_imu:]
            self._imu_buf_timestamps = self._imu_buf_timestamps[-max_imu:]

    def start_baseline(self) -> None:
        """开始录制 ASR 基线，期间采集到的数据会同时喂入训练缓冲"""
        self._asr.start_recording()
        self._recording_baseline = True
        self._baseline_fed_idx = len(self._buf_samples)  # 从当前位置开始，不重处理历史
        log.info("ASR 基线录制开始，请保持安静闭眼...")

    def stop_baseline(self, save_path: Optional[str] = None) -> bool:
        """
        停止录制并训练 ASR，保存基线文件。
        返回 True = 训练成功。
        """
        self._recording_baseline = False
        try:
            n = self._asr.finish_recording()
            self._asr.save_baseline(save_path)
            log.info(f"ASR 基线训练完成 ({n/self.nominal_srate:.1f}s)，已激活")
            return True
        except Exception as e:
            log.error(f"ASR 基线训练失败: {e}")
            return False

    def add(
        self, samples: List[List[float]], lsl_timestamps: List[float]
    ) -> List[pd.DataFrame]:
        """
        接收一批原始 EEG 样本（接口与 DataSaver.add() 兼容）

        参数:
            samples        : 原始 EEG 样本，每元素为一时刻所有通道的值
            lsl_timestamps : 对应的 LSL 时间戳（raw 或已校正均可）
        返回:
            本次调用中完成处理的窗口 DataFrame 列表（可能为空列表）
            供外部模块（如特征提取器）直接使用，无需继承此类
        """
        if not samples:
            return []

        corrected = self._input_corrector.correct(lsl_timestamps)
        self._buf_samples.extend(samples)
        self._buf_timestamps.extend(corrected.tolist())

        # 基线录制模式：以 window_size 为步进逐窗口喂入，保证 filtfilt 样本充足
        if self._recording_baseline:
            while self._baseline_fed_idx + self.window_size <= len(self._buf_samples):
                start = self._baseline_fed_idx
                end   = start + self.window_size
                df_raw = self._build_df(
                    self._buf_samples[start:end],
                    self._buf_timestamps[start:end],
                )
                df_filtered = self.pipeline.process_for_baseline(df_raw)
                if not df_filtered.empty:
                    self._asr.feed_baseline(df_filtered)
                self._baseline_fed_idx = end

        processed: List[pd.DataFrame] = []
        while len(self._buf_samples) >= self.window_size:
            df = self._process_window(self.window_size)
            if df is not None:
                processed.append(df)
        return processed

    def close(self) -> None:
        """处理缓冲区中剩余数据，关闭 DataSaver"""
        if self._buf_samples:
            self._process_window(len(self._buf_samples))
        self._saver.close()

    # ── 内部实现 ──────────────────────────────────────────────────────────

    def _process_window(self, n: int) -> Optional[pd.DataFrame]:
        """取出 n 个样本，经 pipeline 处理后交给 output DataSaver，返回处理好的 DataFrame"""
        samples = self._buf_samples[:n]
        timestamps = self._buf_timestamps[:n]
        self._buf_samples = self._buf_samples[n:]
        self._buf_timestamps = self._buf_timestamps[n:]
        # 基线索引跟随缓冲区头部移动，保持相对位置正确
        self._baseline_fed_idx = max(0, self._baseline_fed_idx - n)

        self._window_idx += 1

        df = self._build_df(samples, timestamps)
        try:
            # ── 质量评估（基线校正前，在原始 ADC 值上进行）──────────────
            # _build_df 产生 eeg_1~4，rename 到 CH1~4 供 assess_window 使用
            raw_for_qa = df.rename(columns={
                "eeg_1": "CH1", "eeg_2": "CH2", "eeg_3": "CH3", "eeg_4": "CH4"
            })
            self._channel_quality = assess_window(raw_for_qa, _PROCESSED_CH, self.nominal_srate)

            # poor 通道：对零散饱和点做线性插值（在原始列上修复）
            ch_map = {"CH1": "eeg_1", "CH2": "eeg_2", "CH3": "eeg_3", "CH4": "eeg_4"}
            for ch, info in self._channel_quality.items():
                if info["quality"] == "poor":
                    raw_col = ch_map[ch]
                    df[raw_col] = interpolate_saturated(df[raw_col].values)
                elif info["quality"] == "bad":
                    log.debug(
                        f"window_{self._window_idx:04d}: {ch} 质量差（{info['reason']}），标记为坏道"
                    )

            df_proc = self.pipeline.process_df(df)
            if df_proc.empty:
                log.warning(f"window_{self._window_idx:04d}: 处理后数据为空，跳过")
                return None

            # bad 通道预处理后标记为 NaN，防止污染下游特征
            for ch, info in self._channel_quality.items():
                if info["quality"] == "bad" and ch in df_proc.columns:
                    df_proc[ch] = np.nan

            # 眨眼检测 + 线性插值修复（ASR 未激活时的 fallback）
            df_proc = self._blink_remover.clean(df_proc, self.nominal_srate)
            if df_proc is None:
                log.debug(f"window_{self._window_idx:04d}: 眨眼门控丢弃（污染比例过高）")
                return None

            # IMU 门控 + NLMS 去运动伪迹
            imu_df = self._slice_imu(timestamps[0], timestamps[-1])
            df_proc = self._imu_remover.clean(df_proc, imu_df)
            if df_proc is None:
                log.debug(f"window_{self._window_idx:04d}: IMU 门控丢弃（运动幅度过大）")
                return None

            processed_samples = df_proc[_PROCESSED_CH].values.tolist()
            aligned_ts = df_proc["time"].tolist()

            self._saver.add(processed_samples, aligned_ts)
            log.debug(f"window_{self._window_idx:04d}: {len(processed_samples)} 样本已处理")
            return df_proc

        except Exception:
            log.exception(f"window_{self._window_idx:04d}: 处理失败")
            return None

    def _build_df(self, samples: List[List[float]], timestamps: List[float]) -> pd.DataFrame:
        """将样本列表构造为 pipeline 期望的 DataFrame（timestamps + eeg_1~4）"""
        arr = np.array(samples)
        df = pd.DataFrame(arr, columns=_PIPELINE_INPUT_CH)
        df.insert(0, "timestamps", timestamps)
        return df

    def _slice_imu(self, t_start: float, t_end: float) -> Optional[pd.DataFrame]:
        """取出 [t_start, t_end] 时间段内的 IMU 数据，不足时返回 None"""
        if not self._imu_buf_timestamps:
            return None
        ts = np.array(self._imu_buf_timestamps)
        # 扩展 ±100ms，避免 EEG 窗口边界落在两个 IMU 采样点之间时误返回 None
        mask = (ts >= t_start - 0.1) & (ts <= t_end + 0.1)
        if not mask.any():
            return None
        cols = self._imu_channels or ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
        arr = np.array(self._imu_buf_samples)[mask]
        if arr.shape[1] < len(cols):
            return None
        df = pd.DataFrame(arr[:, :len(cols)], columns=cols)
        df.insert(0, "time", ts[mask])
        return df


# ── 独立运行模式（直接连接 LSL 流）────────────────────────────────────────

def _run_lsl(args) -> None:
    """独立模式：连接 LSL EEG 流，实时处理并保存"""
    import time
    import pylsl
    from timestamp_corrector import TimestampCorrector

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("查找 LSL EEG 流...")
    streams = pylsl.resolve_byprop("type", "EEG", timeout=args.lsl_timeout)
    if not streams:
        print("未找到 EEG 流，请确认设备已连接")
        return

    stream_info = streams[0]
    nominal_srate = stream_info.nominal_srate() or 256.0
    channel_count = stream_info.channel_count()
    print(f"已连接: {stream_info.name()}  {channel_count}ch  {nominal_srate:.0f}Hz")

    max_chunk = max(1, int(nominal_srate * 0.05))
    inlet = pylsl.StreamInlet(stream_info, max_chunklen=max_chunk)

    corrector = TimestampCorrector(
        inlet,
        dejitter=True,
        dejitter_window=int(nominal_srate * 5),
    )

    output_dir = Path(args.output)
    processor = RealTimeEEGProcessor(
        output_dir=output_dir,
        nominal_srate=nominal_srate,
        corrector=corrector,
        window_seconds=args.window,
        segment_seconds=args.segment,
    )

    print(f"输出目录 : {output_dir}")
    print(f"处理窗口 : {args.window}s  segment: {args.segment}s")
    print("实时处理中 (Ctrl+C 停止)...\n")

    start = time.time()
    total = 0
    try:
        while True:
            if args.duration and time.time() - start >= args.duration:
                break
            samples, ts = inlet.pull_chunk(timeout=0.01, max_samples=max_chunk)
            if samples:
                processor.add(samples, ts)
                total += len(samples)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        processor.close()
        print(f"共处理 {total} 个样本，已保存至 {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="实时EEG预处理（LSL流 → Preprocess/signal_data）")
    parser.add_argument(
        "--output", "-o",
        default=str(Path(__file__).parent / "signal_data"),
        help="输出根目录（默认: Preprocess/signal_data）",
    )
    parser.add_argument(
        "--window", "-w",
        type=float, default=5.0,
        help="处理窗口大小（秒），建议 >= 5（默认 5）",
    )
    parser.add_argument(
        "--segment", "-s",
        type=float, default=60.0,
        help="输出 segment 文件时长（秒，默认 60）",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float, default=None,
        help="采集时长（秒），不设则持续运行直到 Ctrl+C",
    )
    parser.add_argument(
        "--lsl-timeout",
        type=float, default=5.0,
        help="LSL 流查找超时（秒，默认 5）",
    )
    _run_lsl(parser.parse_args())
