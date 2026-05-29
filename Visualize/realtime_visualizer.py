"""
实时 EEG 可视化集成类

封装 DataBridge + WebSocketServer，供 LSL-singal_device.py 调用。

典型用法:
    visualizer = RealTimeVisualizer(window_seconds=10)
    visualizer.start(port=8765)

    # 采集循环中
    visualizer.add_raw(s_type, samples, timestamps)

    for df_proc in processor.add(samples, timestamps):
        visualizer.add_processed(s_type, df_proc)
        df_feat = feat_extractor.add(df_proc)
        visualizer.add_features(s_type, df_feat)

    # 结束时
    visualizer.close()
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from data_bridge import DataBridge
from ws_server import WebSocketServer

log = logging.getLogger(__name__)


class RealTimeVisualizer:
    """
    实时可视化管理器

    参数:
        window_seconds : 显示时间窗口长度（秒），默认 10
        raw_fs         : 原始数据采样率（Hz），默认 256
        proc_fs        : 预处理数据采样率（Hz），默认 256
        feat_rate      : 特征更新速率（epoch/秒），默认 1.0
        push_hz        : WebSocket 推送频率（Hz），默认 10
    """

    def __init__(
        self,
        window_seconds: float = 10.0,
        raw_fs: float = 256.0,
        proc_fs: float = 256.0,
        feat_rate: float = 1.0,
        push_hz: float = 10.0,
    ):
        self._bridge = DataBridge(
            window_seconds=window_seconds,
            raw_fs=raw_fs,
            proc_fs=proc_fs,
            feat_rate=feat_rate,
        )
        self._push_hz = push_hz
        self._server: WebSocketServer | None = None

    # ── 生命周期 ──────────────────────────────────────────────────────────

    def start(self, port: int = 8765, history_api=None) -> None:
        """启动 WebSocket 服务（daemon 线程，不阻塞主循环）

        参数:
            port        : 监听端口
            history_api : HistoryAPI 实例（可选），由调用方构建后透传给 WebSocketServer
        """
        self._server = WebSocketServer(
            self._bridge, port=port, push_hz=self._push_hz, history_api=history_api
        )
        self._server.start()
        log.info(f"可视化面板已启动: http://localhost:{port}")

    def close(self) -> None:
        """关闭 WebSocket 服务"""
        if self._server:
            self._server.stop()
        log.info("RealTimeVisualizer 已关闭")

    # ── 数据写入接口 ──────────────────────────────────────────────────────

    def add_raw(
        self,
        stream_type: str,
        samples: List[List[float]],
        timestamps: List[float],
    ) -> None:
        """写入一批原始 EEG 样本"""
        self._bridge.add_raw(timestamps, samples)

    def add_processed(self, stream_type: str, df: pd.DataFrame) -> None:
        """写入预处理后的 EEG DataFrame（列: time, CH1-CH4）"""
        self._bridge.add_processed(df)

    def add_features(self, stream_type: str, df: pd.DataFrame) -> None:
        """写入特征 DataFrame"""
        self._bridge.add_features(df)

    def add_attention(self, result: dict) -> None:
        """写入注意力检测结果（来自 RealTimeAttentionDetector.add()）"""
        self._bridge.add_attention(result)

    def add_cognitive_load(self, result: dict) -> None:
        """写入认知负荷检测结果（来自 RealTimeCognitiveLoadDetector.add()）"""
        self._bridge.add_cognitive_load(result)
