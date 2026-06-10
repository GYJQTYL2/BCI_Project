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

        log.info(
            f"RealTimeEEGProcessor 初始化: "
            f"输出→{save_dir}, 窗口={window_seconds}s ({self.window_size} 样本)"
        )

    # ── 公开接口（与 DataSaver.add / close 兼容）─────────────────────────

    def start_baseline(self) -> None:
        """开始录制 ASR 基线，期间采集到的数据会同时喂入训练缓冲"""
        self._asr.start_recording()
        self._recording_baseline = True
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

        # 基线录制模式：同时喂入 ASR 训练缓冲
        if self._recording_baseline:
            df_raw = self._build_df(samples, corrected.tolist())
            # build_df 用原始列名，需要 clean 后才有 CH1–CH4
            from data_clean import clean_eeg_frame
            df_clean = clean_eeg_frame(df_raw)
            if not df_clean.empty:
                self._asr.feed_baseline(df_clean)

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

        self._window_idx += 1

        df = self._build_df(samples, timestamps)
        try:
            df_proc = self.pipeline.process_df(df)
            if df_proc.empty:
                log.warning(f"window_{self._window_idx:04d}: 处理后数据为空，跳过")
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
