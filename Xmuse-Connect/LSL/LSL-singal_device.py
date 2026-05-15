
import pylsl
import time
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from data_saver import DataSaver
from timestamp_corrector import TimestampCorrector

_PREPROCESS_DIR = Path(__file__).parent.parent.parent / "Preprocess"
sys.path.insert(0, str(_PREPROCESS_DIR))
from realtime_processor import RealTimeEEGProcessor

_FEATURE_DIR = Path(__file__).parent.parent.parent / "FeatureExtract"
sys.path.insert(0, str(_FEATURE_DIR))
from realtime_extractor import RealTimeFeatureExtractor

_VISUALIZE_DIR = Path(__file__).parent.parent.parent / "Visualize"
sys.path.insert(0, str(_VISUALIZE_DIR))
from realtime_visualizer import RealTimeVisualizer

_ATTENTION_DIR = Path(__file__).parent.parent.parent / "Attention"
sys.path.insert(0, str(_ATTENTION_DIR))
from realtime_attention import RealTimeAttentionDetector


def setup_logging(log_dir: str):
    """配置日志系统，同时输出到控制台和文件"""
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    # 日志文件名
    log_filename = os.path.join(log_dir, f"collection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_filename),  # 写入文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    return log_filename


def main():
    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(description='LSL数据采集程序，用于采集EEG等生理信号数据')
    # 数据流设置
    parser.add_argument('--stream-types', nargs='+', default=["EEG"],help='需要采集的数据类型，例如 "EEG ACC GYRO PPG"')
    parser.add_argument('--custom-channels', nargs='+', default=["ch_1", "ch_2", "ch_3", "ch_4"],help='自定义通道名称，数量需与实际通道数匹配')
    # 采集参数
    parser.add_argument('--duration', type=int, default=None,help='数据采集时长（秒），不设置则无限采集直到手动停止')
    parser.add_argument('--lsl-timeout', type=float, default=5.0,help='LSL流查找的超时时间（秒）')
    parser.add_argument('--chunk-duration', type=float, default=0.05,help='数据块拉取的目标时长（秒）')
    parser.add_argument('--chunk-timeout', type=float, default=0.01,help='pull_chunk的超时时间（秒）')
    parser.add_argument('--dejitter', action='store_true', default=True,help='对时间戳进行线性回归去抖动')
    parser.add_argument('--no-dejitter', action='store_false', dest='dejitter',help='不对时间戳进行去抖动处理')
    # 保存设置（三选一，优先级：--segment-seconds > --segment-bytes > --segment-samples，均缺省时默认 60 秒）
    parser.add_argument('--segment-seconds', type=float, default=None,
                        help='每个 segment 文件的时长（秒），默认 60 秒')
    parser.add_argument('--segment-bytes', type=int, default=None,
                        help='每个 segment 文件大小上限（字节），例如 10485760 表示 10 MB')
    parser.add_argument('--segment-samples', type=int, default=None,
                        help='每个 segment 最多包含的样本数')
    parser.add_argument('--output-dir', default="signal_data",help='数据保存的根目录')
    parser.add_argument('--processed-output', default=None,
                        help='实时处理结果保存目录（默认: Preprocess/signal_data）')
    parser.add_argument('--window-seconds', type=float, default=10.0,
                        help='实时处理窗口大小（秒），建议 >= 10（默认 10）')
    parser.add_argument('--no-realtime', action='store_true',
                        help='禁用实时预处理，仅保存原始数据')
    parser.add_argument('--feature-output', default=None,
                        help='实时特征提取结果保存目录（默认: FeatureExtract/features_output）')
    parser.add_argument('--attention-output', default=None,
                        help='注意力检测结果保存目录（默认: Attention/attention_output）')
    parser.add_argument('--epoch-seconds', type=float, default=1.0,
                        help='特征提取 epoch 长度（秒），默认 1.0')
    parser.add_argument('--no-visualize', action='store_true',
                        help='禁用实时可视化面板（默认开启）')
    parser.add_argument('--viz-port', type=int, default=8765,
                        help='可视化 WebSocket 服务端口（默认 8765）')
    parser.add_argument('--viz-window', type=float, default=10.0,
                        help='可视化时间窗口长度（秒，默认 10）')
    args = parser.parse_args()
    # --- 配置参数 ---
    STREAM_TYPES: List[str] = args.stream_types
    COLLECTION_DURATION: Optional[int] = args.duration
    LSL_SCAN_TIMEOUT: float = args.lsl_timeout
    PULL_CHUNK_DURATION: float = args.chunk_duration
    PULL_CHUNK_TIMEOUT: float = args.chunk_timeout
    DEJITTER_TIMESTAMPS: bool = args.dejitter
    SEGMENT_SECONDS: Optional[float] = args.segment_seconds
    SEGMENT_BYTES: Optional[int] = args.segment_bytes
    SEGMENT_SAMPLES: Optional[int] = args.segment_samples
    CUSTOM_CHANNEL_NAMES: List[str] = args.custom_channels
    ENABLE_REALTIME: bool = not args.no_realtime
    PROCESSED_OUTPUT: str = args.processed_output or str(_PREPROCESS_DIR / "signal_data")
    WINDOW_SECONDS: float = args.window_seconds
    FEATURE_OUTPUT: str = args.feature_output or str(_FEATURE_DIR / "features_output")
    ATTENTION_OUTPUT: str = args.attention_output or str(_ATTENTION_DIR / "attention_output")
    EPOCH_SECONDS: float = args.epoch_seconds
    ENABLE_VISUALIZE: bool = not args.no_visualize
    VIZ_PORT: int = args.viz_port
    VIZ_WINDOW: float = args.viz_window
    # --- 数据保存目录设置 ---
    COLLECTION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_SAVE_DIR = os.path.join(args.output_dir, COLLECTION_TIMESTAMP)
    Path(BASE_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    # 配置日志
    log_dir = os.path.join(BASE_SAVE_DIR, "logs")
    log_file = setup_logging(log_dir)
    logging.info(f"程序启动，数据将保存到: {os.path.abspath(BASE_SAVE_DIR)}")
    logging.info(f"日志文件将保存到: {os.path.abspath(log_file)}")
    # --- LSL 流初始化 ---
    inlets: Dict[str, pylsl.StreamInlet] = {}
    stream_details: Dict[str, Dict[str, Any]] = {}
    logging.info("\n--- 开始查找和初始化 LSL 数据流 ---")
    for s_type in STREAM_TYPES:
        logging.info(f"尝试查找 {s_type} 数据流...")
        try:
            streams = pylsl.resolve_byprop("type", s_type, timeout=LSL_SCAN_TIMEOUT)
            if streams:
                logging.info(f"已找到 {s_type} 数据流。")
                stream_info = streams[0]
                inlet = pylsl.StreamInlet(
                    stream_info,
                    max_chunklen=int(stream_info.nominal_srate() * PULL_CHUNK_DURATION)
                )
                inlets[s_type] = inlet
                nominal_srate = stream_info.nominal_srate()
                channel_count = stream_info.channel_count()
                if nominal_srate <= 0:
                    logging.warning(f"{s_type} 的名义采样率为 0 或未指定。默认设置为 256 Hz。")
                    nominal_srate = 256.0

                if len(CUSTOM_CHANNEL_NAMES) == channel_count:
                    ch_names = CUSTOM_CHANNEL_NAMES
                    logging.info(f"  使用自定义通道名称: {ch_names}")
                else:
                    ch_names = []
                    ch = stream_info.desc().child('channels').first_child()
                    for _ in range(channel_count):
                        ch_names.append(ch.child_value('label'))
                        ch = ch.next_sibling()
                    if not ch_names:
                        ch_names = [f'channel_{i + 1}' for i in range(channel_count)]
                    logging.info(f"  使用流定义的通道名称: {ch_names}")

                stream_details[s_type] = {
                    'channel_count': channel_count,
                    'nominal_srate': nominal_srate,
                    'channel_names': ch_names,
                }
                logging.info(f"  {s_type} 流信息：通道数={channel_count}, 采样率={nominal_srate} Hz")
            else:
                logging.warning(f"未找到 {s_type} 数据流。")
        except Exception as e:
            logging.error(f"查找 {s_type} 数据流时发生错误: {str(e)}", exc_info=True)
    if not inlets:
        logging.error("未找到任何主要数据流（如 EEG）。请确保设备已连接并处于运行状态。")
        return
    # --- 创建 TimestampCorrector 和 DataSaver ---
    correctors: Dict[str, TimestampCorrector] = {}
    savers: Dict[str, DataSaver] = {}
    for s_type, inlet in inlets.items():
        correctors[s_type] = TimestampCorrector(
            inlet,
            dejitter=DEJITTER_TIMESTAMPS,
            dejitter_window=int(stream_details[s_type]['nominal_srate'] * 5),
        )
        savers[s_type] = DataSaver(
            save_dir=BASE_SAVE_DIR,
            stream_type=s_type,
            ch_names=stream_details[s_type]['channel_names'],
            corrector=correctors[s_type],
            nominal_srate=stream_details[s_type]['nominal_srate'],
            segment_seconds=SEGMENT_SECONDS,
            segment_bytes=SEGMENT_BYTES,
            segment_samples=SEGMENT_SAMPLES,
        )
        if SEGMENT_SECONDS is not None:
            seg_desc = f"每 {SEGMENT_SECONDS:.0f} 秒一个 segment"
        elif SEGMENT_BYTES is not None:
            seg_desc = f"每 {SEGMENT_BYTES / 1024 / 1024:.1f} MB 一个 segment"
        elif SEGMENT_SAMPLES is not None:
            seg_desc = f"每 {SEGMENT_SAMPLES} 个样本一个 segment"
        else:
            seg_desc = "每 60 秒一个 segment（默认）"
        logging.info(f"  {s_type} 数据将按{seg_desc}保存到: {BASE_SAVE_DIR}")
    # --- 创建实时可视化器 ---
    visualizer = None
    if ENABLE_VISUALIZE:
        visualizer = RealTimeVisualizer(window_seconds=VIZ_WINDOW)

        # 构建历史数据 API（由主入口负责配置目录，传给 visualizer 透传）
        history_api = None
        try:
            from history_reader import HistoryReader
            from history_api    import HistoryAPI
            history_api = HistoryAPI(HistoryReader({
                "raw":       args.output_dir,
                "processed": PROCESSED_OUTPUT,
                "features":  FEATURE_OUTPUT,
                "attention": ATTENTION_OUTPUT,
            }))
        except ImportError:
            logging.warning("history_reader / history_api 未找到，历史数据功能不可用")

        visualizer.start(port=VIZ_PORT, history_api=history_api)
        logging.info(f"可视化面板: http://localhost:{VIZ_PORT}")
        logging.info(f"历史数据:   http://localhost:{VIZ_PORT}/history.html")

    # --- 创建实时处理器（每个4通道流独立创建校正器，避免与 DataSaver 共用同一实例）---
    proc_correctors: Dict[str, TimestampCorrector] = {}
    processors: Dict[str, RealTimeEEGProcessor] = {}
    feat_extractors: Dict[str, RealTimeFeatureExtractor] = {}
    attn_detectors: Dict[str, RealTimeAttentionDetector] = {}
    if ENABLE_REALTIME:
        for s_type, inlet in inlets.items():
            if stream_details[s_type]['channel_count'] != 4:
                logging.info(f"  {s_type} 通道数不为4，跳过实时处理")
                continue
            proc_correctors[s_type] = TimestampCorrector(
                inlet,
                dejitter=DEJITTER_TIMESTAMPS,
                dejitter_window=int(stream_details[s_type]['nominal_srate'] * 5),
            )
            processors[s_type] = RealTimeEEGProcessor(
                output_dir=PROCESSED_OUTPUT,
                nominal_srate=stream_details[s_type]['nominal_srate'],
                corrector=proc_correctors[s_type],
                window_seconds=WINDOW_SECONDS,
            )
            feat_extractors[s_type] = RealTimeFeatureExtractor(
                feature_output_dir=FEATURE_OUTPUT,
                fs=stream_details[s_type]['nominal_srate'],
                epoch_seconds=EPOCH_SECONDS,
            )
            attn_detectors[s_type] = RealTimeAttentionDetector(
                smooth_window=5,
                attention_output_dir=ATTENTION_OUTPUT,
            )
            logging.info(f"  {s_type} 实时处理器已启动，窗口={WINDOW_SECONDS}s → {PROCESSED_OUTPUT}")
            logging.info(f"  {s_type} 实时特征提取器已启动，epoch={EPOCH_SECONDS}s → {FEATURE_OUTPUT}")
            logging.info(f"  {s_type} 注意力检测器已启动，平滑窗口=5 epoch")
    # --- 主数据采集循环 ---
    try:
        logging.info("\n--- 开始记录数据 ---")
        logging.info("按 Ctrl+C 停止记录")
        start_time = time.time()
        total_samples_collected = {s_type: 0 for s_type in STREAM_TYPES}
        while True:
            # 检查是否达到采集时长
            if COLLECTION_DURATION is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time > COLLECTION_DURATION:
                    logging.info(f"\n已达到 {COLLECTION_DURATION} 秒的采集时长。")
                    break
            # 从每个数据流读取数据
            for s_type, inlet_obj in inlets.items():
                try:
                    nominal_srate = stream_details[s_type]['nominal_srate']
                    max_samples_to_pull = max(1, int(nominal_srate * PULL_CHUNK_DURATION))

                    samples, lsl_timestamps = inlet_obj.pull_chunk(timeout=PULL_CHUNK_TIMEOUT, max_samples=max_samples_to_pull)
                    if samples:
                        corrected_timestamps = correctors[s_type].correct(lsl_timestamps)
                        savers[s_type].add(samples, corrected_timestamps.tolist())
                        if visualizer:
                            visualizer.add_raw(s_type, samples, corrected_timestamps.tolist())
                        if s_type in processors:
                            for df_proc in processors[s_type].add(samples, corrected_timestamps.tolist()):
                                if df_proc is not None and not df_proc.empty:
                                    if visualizer:
                                        visualizer.add_processed(s_type, df_proc)
                                    if s_type in feat_extractors:
                                        df_feat = feat_extractors[s_type].add(df_proc)
                                        if s_type in attn_detectors and df_feat is not None and not df_feat.empty:
                                            attn = attn_detectors[s_type].add(df_feat)
                                        if visualizer and df_feat is not None and not df_feat.empty:
                                            visualizer.add_features(s_type, df_feat)
                                        if visualizer and attn is not None:
                                            visualizer.add_attention(attn)
                                        
                        total_samples_collected[s_type] += len(samples)
                        if total_samples_collected[s_type] % 1000 == 0:
                            logging.debug(f"{s_type} 已采集 {total_samples_collected[s_type]} 个样本")
                except pylsl.timeout_error:
                    pass  # 没有新数据，正常现象
                except Exception as e:
                    logging.error(f"处理 {s_type} 数据时出错: {str(e)}", exc_info=True)
                    continue
            time.sleep(0.001)
    except KeyboardInterrupt:
        logging.info("\n用户通过Ctrl+C中断记录。")
    except Exception as e:
        logging.error(f"发生意外错误: {str(e)}", exc_info=True)
    finally:
        logging.info("\n--- 数据采集完成，正在保存剩余数据 ---")
        for s_type in STREAM_TYPES:
            if s_type in savers:
                logging.info(f"  {s_type} 总共采集 {total_samples_collected[s_type]} 个样本")
                savers[s_type].close()
                logging.info(f"  {s_type} 原始数据已保存到: {os.path.abspath(BASE_SAVE_DIR)}")
            if s_type in processors:
                processors[s_type].close()
                logging.info(f"  {s_type} 处理数据已保存到: {os.path.abspath(PROCESSED_OUTPUT)}")
            if s_type in feat_extractors:
                feat_extractors[s_type].close()
                logging.info(f"  {s_type} 特征数据已保存到: {os.path.abspath(FEATURE_OUTPUT)}")
            if s_type in attn_detectors:
                attn_detectors[s_type].close()
                logging.info(f"  {s_type} 注意力数据已保存到: {os.path.abspath(ATTENTION_OUTPUT)}")
        if visualizer:
            visualizer.close()
        # 记录程序结束信息
        elapsed_time = time.time() - start_time
        logging.info(f"所有数据保存完成。总采集时间: {elapsed_time:.2f} 秒")
        logging.info("程序正常退出。")
if __name__ == "__main__":
    main()
