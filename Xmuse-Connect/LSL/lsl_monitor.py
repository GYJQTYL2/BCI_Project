#!/usr/bin/env python3
from __future__ import annotations
"""
LSL 多路数据监测器（含 SpO2 / 心率实时计算）

自动发现 Muse 设备通过 LSL 发布的全部数据流并实时打印，
对 PPG 流额外计算 SpO2 和心率。

用法:
    python lsl_monitor.py
    python lsl_monitor.py --timeout 8       # 延长发现等待时间
    python lsl_monitor.py --no-throttle     # 高频流打印每个样本
"""

import argparse
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import pylsl
from scipy.signal import butter, filtfilt, find_peaks

# ── ANSI 颜色 ────────────────────────────────────────────────────────────────
R   = "\033[0m"
B   = "\033[1m"
RED = "\033[91m"
GRY = "\033[90m"
_COLORS = [
    "\033[36m", "\033[32m", "\033[33m", "\033[35m", "\033[34m",
    "\033[96m", "\033[92m", "\033[93m", "\033[95m", "\033[94m",
    "\033[91m", "\033[37m",
]

# ── 已知数据流元数据 ─────────────────────────────────────────────────────────
# (名称关键字或类型关键字) → (中文描述, 默认通道标签, 是否整数显示)
STREAM_META: dict[str, tuple[str, list[str], bool]] = {
    # ── EEG ──────────────────────────────────────────────────────────────────
    "EEG":            ("脑电 EEG",          ["TP9","AF7","AF8","TP10","AUX_L","AUX_R"], False),
    # ── PPG（特殊：单独线程处理 SpO2/HR）────────────────────────────────────
    "PPG":            ("光电容积 PPG",       ["Ambient","IR","Red"],                    False),
    # ── IMU ──────────────────────────────────────────────────────────────────
    "IMU":            ("IMU(Acc+Gyro)",     ["AccX","AccY","AccZ","GyrX","GyrY","GyrZ"],False),
    "Accelerometer":  ("加速度计 ACC",       ["AccX","AccY","AccZ"],                    False),
    "Gyroscope":      ("陀螺仪 GYRO",        ["GyrX","GyrY","GyrZ"],                    False),
    # ── 电池 ─────────────────────────────────────────────────────────────────
    "Batt":           ("电池 Batt",          ["Percent%","Voltage_V","Temp_C"],         False),
    "Battery":        ("电池 Batt",          ["Percent%","Voltage_V","Temp_C"],         False),
    # ── 事件（0/1） ──────────────────────────────────────────────────────────
    "Blink":          ("眨眼 Blink",         ["Blink"],                                 True),
    "Jaw_Clench":     ("咀嚼 JawClench",     ["JawClench"],                             True),
    "JawClench":      ("咀嚼 JawClench",     ["JawClench"],                             True),
    # ── 信号质量 ─────────────────────────────────────────────────────────────
    "HSI_PREC":       ("接触指示 HSIPrec",   ["TP9","AF7","AF8","TP10"],                True),
    "HSIPrec":        ("接触指示 HSIPrec",   ["TP9","AF7","AF8","TP10"],                True),
    "HsiPrec":        ("接触指示 HSIPrec",   ["TP9","AF7","AF8","TP10"],                True),
    "IS_GOOD":        ("数据质量 IS_GOOD",   ["TP9","AF7","AF8","TP10"],                True),
    "HeadOn":         ("佩戴标识 HeadOn",    ["HeadOn"],                                True),
    # ── 温度 ─────────────────────────────────────────────────────────────────
    "Therm":          ("温度 Therm",         ["Therm"],                                 False),
    # ── 频段 PSD ─────────────────────────────────────────────────────────────
    "Alpha_Relative": ("α波 相对PSD",        ["TP9","AF7","AF8","TP10"],                False),
    "Alpha_Absolute": ("α波 绝对PSD",        ["TP9","AF7","AF8","TP10"],                False),
    "Beta_Relative":  ("β波 相对PSD",        ["TP9","AF7","AF8","TP10"],                False),
    "Beta_Absolute":  ("β波 绝对PSD",        ["TP9","AF7","AF8","TP10"],                False),
    "Gamma_Relative": ("γ波 相对PSD",        ["TP9","AF7","AF8","TP10"],                False),
    "Gamma_Absolute": ("γ波 绝对PSD",        ["TP9","AF7","AF8","TP10"],                False),
    "Delta_Relative": ("δ波 相对PSD",        ["TP9","AF7","AF8","TP10"],                False),
    "Delta_Absolute": ("δ波 绝对PSD",        ["TP9","AF7","AF8","TP10"],                False),
    "Theta_Relative": ("θ波 相对PSD",        ["TP9","AF7","AF8","TP10"],                False),
    "Theta_Absolute": ("θ波 绝对PSD",        ["TP9","AF7","AF8","TP10"],                False),
}

# 高频流（限速 1 次/秒）
HIGH_FREQ = {"EEG", "PPG", "IMU", "Accelerometer", "Gyroscope"}

# ── 全局打印队列 ─────────────────────────────────────────────────────────────
_Q: queue.SimpleQueue = queue.SimpleQueue()

def _printer():
    while True:
        line = _Q.get()
        if line is None:
            break
        print(line, flush=True)

def _ts_str(lsl_ts: float) -> str:
    unix = time.time() - (pylsl.local_clock() - lsl_ts)
    dt = datetime.fromtimestamp(unix)
    return dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"

# ── 元数据匹配（名称前缀 → 描述/标签/整数显示）────────────────────────────
def _match(stream_name: str, stream_type: str) -> tuple[str, list[str], bool] | None:
    for key, meta in STREAM_META.items():
        if stream_name == key or stream_name.startswith(key) \
                or stream_type == key or stream_type.startswith(key):
            return meta
    return None

# ── PPG 专用：SpO2 + 心率实时计算 ───────────────────────────────────────────
def _ppg_thread(inlet: pylsl.StreamInlet, color: str,
                fs_target: float = 64.0, win_sec: float = 10.0):
    """
    接收 PPG 三通道（Ambient / IR / Red），
    计算 SpO2（AC-DC 法）和心率（峰值检测法），每秒打印一次。
    """
    maxlen = int(win_sec * fs_target)
    ir_buf  = deque(maxlen=maxlen)
    red_buf = deque(maxlen=maxlen)
    ts_buf  = deque(maxlen=maxlen)
    spo2_buf: deque = deque(maxlen=64)

    def _bandpass(data: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
        nyq = 0.5 * fs
        lo_n, hi_n = lo / nyq, hi / nyq
        if not (0 < lo_n < 1 and 0 < hi_n < 1):
            return data
        if len(data) <= 9:
            return data
        b, a = butter(3, [lo_n, hi_n], btype="band")
        return filtfilt(b, a, data)

    def _spo2(ir_raw, red_raw, ir_filt, red_filt):
        ir_dc, red_dc = np.mean(ir_raw), np.mean(red_raw)
        if ir_dc == 0 or red_dc == 0:
            return None
        ir_ac  = np.std(ir_filt)
        red_ac = np.std(red_filt)
        if ir_ac == 0:
            return None
        R = (red_ac / red_dc) / (ir_ac / ir_dc)
        return 100.67 - 9.14 * abs(R)

    def _heart_rate(ir_filt: np.ndarray, fs: float) -> float | None:
        if len(ir_filt) < int(fs * 2):
            return None
        min_dist = int(fs * 0.4)   # 最快 150 bpm
        peaks, _ = find_peaks(ir_filt, distance=min_dist, prominence=np.std(ir_filt) * 0.5)
        if len(peaks) < 2:
            return None
        intervals = np.diff(peaks) / fs       # 秒
        hr = 60.0 / np.mean(intervals)
        return hr if 40 <= hr <= 200 else None

    last_print = 0.0
    while True:
        try:
            samples, timestamps = inlet.pull_chunk(timeout=0.5, max_samples=32)
        except Exception as exc:
            _Q.put(f"{RED}[ERROR]{R} PPG: {exc}")
            time.sleep(1)
            continue

        for sample, ts in zip(samples, timestamps):
            if len(sample) < 3:
                continue
            ambient, ir, red = sample[0], sample[1], sample[2]
            ir_buf.append(ir - ambient)
            red_buf.append(red - ambient)
            ts_buf.append(ts)

        now = time.monotonic()
        if now - last_print < 1.0:
            continue
        last_print = now

        n = len(ir_buf)
        min_fill = int(win_sec * fs_target * 0.5)
        if n < min_fill:
            _Q.put(f"{GRY}[--:--:--]{R} {color}{B}{'光电容积 PPG':<20}{R} "
                   f"缓冲中 {n}/{maxlen} 样本...")
            continue

        ts_arr  = np.array(ts_buf)
        ir_arr  = np.array(ir_buf)
        red_arr = np.array(red_buf)

        fs = n / (ts_arr[-1] - ts_arr[0]) if ts_arr[-1] > ts_arr[0] else fs_target

        ir_filt  = _bandpass(ir_arr,  0.5, 4.0, fs)
        red_filt = _bandpass(red_arr, 0.5, 4.0, fs)

        spo2 = _spo2(ir_arr, red_arr, ir_filt, red_filt)
        hr   = _heart_rate(ir_filt, fs)

        if spo2 is not None:
            spo2_buf.append(spo2)

        spo2_avg = float(np.mean(spo2_buf)) if spo2_buf else float("nan")
        hr_str   = f"{hr:.1f} bpm" if hr is not None else "-- bpm"
        ts_str   = _ts_str(ts_arr[-1])

        _Q.put(
            f"{GRY}[{ts_str}]{R} {color}{B}{'光电容积 PPG':<20}{R} "
            f"SpO2={spo2_avg:5.2f}%  HR={hr_str}  "
            f"{GRY}(IR={ir_arr[-1]:.0f}  Red={red_arr[-1]:.0f}  fs={fs:.1f}Hz){R}"
        )

# ── 通用数据流接收线程 ────────────────────────────────────────────────────────
def _generic_thread(inlet: pylsl.StreamInlet, name: str, desc: str,
                    ch_labels: list[str], color: str,
                    throttle: bool, int_display: bool):
    last_print = 0.0
    while True:
        try:
            samples, timestamps = inlet.pull_chunk(timeout=0.5, max_samples=64)
        except Exception as exc:
            _Q.put(f"{RED}[ERROR]{R} {name}: {exc}")
            time.sleep(1)
            continue

        if not samples:
            continue

        now = time.monotonic()
        if throttle and (now - last_print) < 1.0:
            continue
        last_print = now

        sample = samples[-1]
        lsl_ts = timestamps[-1] if timestamps else pylsl.local_clock()
        n = len(sample)
        labels = ch_labels[:n] + [f"ch{i+1}" for i in range(len(ch_labels), n)]

        if int_display:
            vals = "  ".join(f"{lb}={int(v)}" for lb, v in zip(labels, sample))
        else:
            vals = "  ".join(f"{lb}={v:+.4f}" for lb, v in zip(labels, sample))

        chunk_tag = f"{GRY}(×{len(samples)}){R}" if len(samples) > 1 else "       "
        _Q.put(
            f"{GRY}[{_ts_str(lsl_ts)}]{R} "
            f"{color}{B}{desc:<20}{R} "
            f"{chunk_tag} {vals}"
        )

# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LSL 多路数据监测器")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="流发现超时（秒），默认 5")
    parser.add_argument("--buf", type=float, default=3.0,
                        help="每路 inlet 缓冲时长（秒），默认 3")
    parser.add_argument("--no-throttle", action="store_true",
                        help="关闭高频流限速，打印所有样本")
    args = parser.parse_args()

    print(f"\n{B}LSL 多路数据监测器{R}  发现中（等待 {args.timeout:.0f}s）...\n")
    streams = pylsl.resolve_streams(wait_time=args.timeout)

    if not streams:
        print(f"{RED}未发现任何 LSL 数据流。{R} 请确认 muse_lsl_bridge.py 或 BlueMuse 已启动。")
        sys.exit(1)

    print(f"发现 {B}{len(streams)}{R} 路数据流：\n")
    for i, s in enumerate(streams):
        c = _COLORS[i % len(_COLORS)]
        print(f"  {c}●{R} {s.name():<26} type={s.type():<12} "
              f"ch={s.channel_count():<3} rate={s.nominal_srate():.1f} Hz  "
              f"src={s.source_id()}")

    print(f"\n{'─'*80}")
    print(f"开始接收  {GRY}Ctrl+C 退出{R}\n")

    threading.Thread(target=_printer, daemon=True, name="printer").start()

    active = 0
    for i, s in enumerate(streams):
        name  = s.name()
        stype = s.type()
        color = _COLORS[i % len(_COLORS)]
        meta  = _match(name, stype)

        is_ppg = ("PPG" in name or stype.upper() == "PPG")

        try:
            inlet = pylsl.StreamInlet(
                s,
                max_buflen=int(args.buf),
                processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
            )
        except Exception as exc:
            print(f"{RED}[WARN]{R} 无法连接流 {name}: {exc}")
            continue

        if is_ppg:
            fs = s.nominal_srate() or 64.0
            t = threading.Thread(
                target=_ppg_thread,
                args=(inlet, color, fs),
                daemon=True, name=f"ppg-{name}",
            )
        else:
            if meta:
                desc, labels, int_disp = meta
            else:
                desc     = name
                labels   = [f"ch{j+1}" for j in range(s.channel_count())]
                int_disp = False

            throttle = (not args.no_throttle) and any(
                name.startswith(k) or stype.startswith(k) for k in HIGH_FREQ
            )
            t = threading.Thread(
                target=_generic_thread,
                args=(inlet, name, desc, labels, color, throttle, int_disp),
                daemon=True, name=f"inlet-{name}",
            )

        t.start()
        active += 1

    if active == 0:
        print(f"{RED}所有流连接失败，退出。{R}")
        _Q.put(None)
        sys.exit(1)

    print(f"已连接 {B}{active}{R} 路流。\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\n{GRY}[停止]{R} Ctrl+C 中断。")
        _Q.put(None)

if __name__ == "__main__":
    main()
