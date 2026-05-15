"""
当前的方法仅针对4channel EEG数据，如果改动采集PRESET需要修改代码和函数
"""
import pandas as pd
import numpy as np
import os

def clean_eeg_frame(df: pd.DataFrame) -> pd.DataFrame:
    """EEG数据清理"""

    # 重命名列
    rename_map = {
        'timestamps': 'time',
        'eeg_1': 'CH1',
        'eeg_2': 'CH2',
        'eeg_3': 'CH3',
        'eeg_4': 'CH4'
    }
    required_cols_original = list(rename_map.keys())
    df_cleaned = df[required_cols_original].rename(columns=rename_map)

    # 删除EEG通道数据空行
    eeg_channels = ['CH1', 'CH2', 'CH3', 'CH4']
    df_cleaned.dropna(subset=eeg_channels, how='all', inplace=True)
    if df_cleaned.empty:
        return df_cleaned

    # 时间戳转换为0起始相对时间
    #df_cleaned.reset_index(drop=True, inplace=True)
    #df_cleaned['time'] = df_cleaned['time'] - df_cleaned['time'].iloc[0]
    
    return df_cleaned
