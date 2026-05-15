import pandas as pd
import numpy as np
import os
from typing import Tuple

def interpolate_outliers(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, int]:
    """
    功能: 在给定的表中检测并插值超过阈值的点
    """
    eeg_channels = ['CH1', 'CH2', 'CH3', 'CH4']
    total_interpolated_count = 0

    for chan in eeg_channels:
        if chan not in df.columns:
            continue
        signal = df[chan]
        # 找到绝对值超过阈值的坏点
        bad_indices = np.abs(signal) > threshold
        bad_points_count = bad_indices.sum()

        if bad_points_count > 0:
            print(f"{chan}找到{bad_points_count}个极值点插值...")
            total_interpolated_count += bad_points_count
            
            # 坏点NaN，线性插值填充
            signal[bad_indices] = np.nan
            df[chan] = signal.interpolate(method='linear', limit_direction='both')
        else:
            print(f"{chan}无坏点")
            
    return df, total_interpolated_count

