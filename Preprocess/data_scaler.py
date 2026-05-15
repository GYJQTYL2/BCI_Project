import pandas as pd
import numpy as np
import os

# ---缩放函数---
def normalize_channel(data):
    """(Min-Max) 将数据归一化到 [0, 1] 区间"""
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

def standardize_channel(data):
    """(Z-Score) 将数据标准化为均值0，标准差1"""
    mean_val = data.mean()
    std_val = data.std()
    return (data - mean_val) / std_val

