import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os

# 滤波器函数，最好不要改会出错
def highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def notch_50hz_filter(data, fs, order=5):
    nyq = 0.5 * fs
    low, high = 49 / nyq, 51 / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return filtfilt(b, a, data)


def filter_eeg_frame(df: pd.DataFrame, hp_cutoff=0.5, lp_cutoff=45, channels=None) -> pd.DataFrame:
    """
    对整帧EEG数据进行滤波，并修复坏导

    流程：
        1. 对每个通道插值填补NaN后依次做高通/低通/50Hz陷波
        2. 若滤波后仍有NaN，使用邻近通道均值插值修复坏导
    输入: 含 'time' 列和EEG通道列的DataFrame
    输出: 滤波后的DataFrame
    """
    if channels is None:
        channels = ['CH1', 'CH2', 'CH3', 'CH4']

    df_out = df.copy()
    fs = 1 / np.mean(np.diff(df_out['time'].dropna()))

    # ---Filtering---
    filtered_results = {}
    for chan in channels:
        if chan in df_out.columns and not df_out[chan].isnull().all():
            signal_to_filter = df_out[chan].interpolate(method='linear').bfill().ffill()
            original_signal = signal_to_filter.values

            # 应用滤波器
            filtered_signal = highpass_filter(original_signal, hp_cutoff, fs)
            filtered_signal = lowpass_filter(filtered_signal, lp_cutoff, fs)
            filtered_signal = notch_50hz_filter(filtered_signal, fs)

            filtered_results[chan] = filtered_signal
        else:
            filtered_results[chan] = None  # 标记此通道为空或不存在

    # ---Check：插值滤波失败的坏导---
    for i, chan in enumerate(channels):
        result_signal = filtered_results.get(chan)

        # 检查滤波是否失败: 结果为None或包含NaN
        if result_signal is None or np.isnan(result_signal).any():
            print(f"通道{chan}滤波失败，使用邻近通道插值修复")

            # 寻找有效的邻居通道
            left_neighbor_data = None
            if i > 0:  # 判断坏导是否为第一个通道
                prev_chan = channels[i - 1]
                if filtered_results.get(prev_chan) is not None and not np.isnan(filtered_results[prev_chan]).any():
                    left_neighbor_data = filtered_results[prev_chan]

            right_neighbor_data = None
            if i < len(channels) - 1:  # 判断坏导是否为最后一个通道
                next_chan = channels[i + 1]
                if filtered_results.get(next_chan) is not None and not np.isnan(filtered_results[next_chan]).any():
                    right_neighbor_data = filtered_results[next_chan]

            # 根据邻居情况插值
            if left_neighbor_data is not None and right_neighbor_data is not None:
                df_out[chan] = (left_neighbor_data + right_neighbor_data) / 2
                print(f"使用{channels[i-1]} & {channels[i+1]} 的平均值修复{chan}")
            elif left_neighbor_data is not None:  # 末道
                df_out[chan] = left_neighbor_data
                print(f"使用 {channels[i-1]}的数据修复了{chan}")
            elif right_neighbor_data is not None:  # 首道
                df_out[chan] = right_neighbor_data
                print(f"使用 {channels[i+1]}的数据修复了{chan}")
            else:
                df_out[chan] = np.nan  # 如果没有好邻居，只能置NaN
                print(f"无法修复 {chan}，无可用的邻近通道")
        else:
            # 滤波成功，写回数据
            df_out[chan] = result_signal

    return df_out