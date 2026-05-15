import pandas as pd
import numpy as np
import os

# 基线校正函数定义
def correct_dc_offset(df, offset_val=800, channels=None):
    """
    方法1：DC校正，直接减去固定值800
    """
    df_corrected = df.copy()
    if channels is None:
        channels = df.columns
    for chan in channels:
        if chan in df_corrected.columns:
            df_corrected[chan] = df_corrected[chan] - offset_val
            
    return df_corrected

def correct_baseline_channelwise(df, baseline_window_sec, fs, channels=None):
    """
    方法2：通道独立基线校正，减去各自通道的基线期均值
    """
    df_corrected = df.copy()
    if channels is None:
        channels = df.columns
    # 将时间窗口转换为数据点索引
    start_point = int(baseline_window_sec[0] * fs)
    end_point = int(baseline_window_sec[1] * fs)
    end_point = min(end_point, len(df)) # 确保索引不越界
    for chan in channels:
        if chan in df_corrected.columns:
            # 计算该通道在基线期内的均值
            baseline_mean = df_corrected[chan].iloc[start_point:end_point].mean()
            # 从整个通道减去这个均值
            df_corrected[chan] = df_corrected[chan] - baseline_mean

    return df_corrected

