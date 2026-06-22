# 眨眼伪影去除方案设计

## 背景

Muse 设备前额 4 通道离眼睛最近，眨眼产生的 EOG 信号幅值（200–500 µV）远大于真实
EEG（10–50 µV），能量主体在 1–8 Hz，与 theta（4–8 Hz）高度重叠。

无 ASR 时眨眼完全不被处理，直接导致 theta 功率虚高 5–50 倍，注意力指数和认知负荷
指数系统性偏高。

---

## 设计目标

1. 不依赖基线录制，启动即可用（作为 ASR 的 fallback）
2. 单次眨眼后 theta 误差 < 5%（当前无保护时误差 > 10 倍）
3. 不整窗口丢弃，保持 RealTimeAttention 滑窗的连续性
4. 频繁眨眼（污染 > 30%）时退化为整窗口丢弃，不产生错误特征

---

## 核心机制

### 两层结构

```
BlinkArtifactRemover.clean(df, sfreq)
  │
  ├─ 1. 阈值估计（滚动自适应）
  │       取最近 N 个干净窗口的信号标准差中位数 × k
  │       前 N 个窗口未满时用固定保守阈值兜底
  │
  ├─ 2. 眨眼检测
  │       对每个通道计算 max(abs(signal))
  │       任一通道超过阈值 → 记录超阈位置
  │       向两侧扩展 ±margin（覆盖上升/下降沿）
  │
  ├─ 3. 污染率判断
  │       被标记样本数 / 总样本数
  │       > max_contamination (0.3) → 返回 None（整窗口门控）
  │
  ├─ 4. 线性插值修复
  │       用眨眼前后干净锚点做 np.interp
  │       和 interpolate_saturated 逻辑完全对称
  │
  └─ 5. 更新滚动估计
          仅用非眨眼窗口（干净窗口）更新 deque
```

### 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `k` | 5 | 阈值倍数（眨眼幅值通常是安静 EEG 的 5–10 倍） |
| `margin_ms` | 100 | 眨眼边缘扩展（ms），覆盖上升/下降沿 |
| `max_contamination` | 0.30 | 超过此比例退化为整窗口丢弃 |
| `warmup_windows` | 20 | 滚动估计稳定所需窗口数（约 100 秒） |
| `fallback_threshold` | 200 | 未 warmup 时的固定兜底阈值（µV，滤波后信号） |

---

## 阈值自适应逻辑

```python
# 每个干净窗口（未被门控）计算各通道 max(abs)
# 取 4 通道中位数存入 deque（最近 20 个）
# 阈值 = median(deque) × k

# warmup 未完成时统一用 fallback_threshold = 200 µV
# 不区分 ASR 状态：
#   ASR 激活时眨眼在进入 BlinkRemover 之前已被处理，
#   滚动估计接管后阈值会自动收敛到正确水平，
#   不需要感知 ASR 状态引入额外耦合
```

---

## 对 Welch PSD 的实际影响（定量）

5 秒窗口（1280 样本），`nperseg = 1280`，1 次眨眼 200ms = 51 样本：

```
原始眨眼混入：
  theta 功率误差 × 10–50 倍（严重高估）

线性插值替换后：
  插值段为斜线，频率成分 < 2 Hz
  theta（4–8 Hz）在插值段中贡献极小
  theta 误差 < 5%（轻微低估，因为插值段不含真实 theta）

连续眨眼（占比 > 30%）退化为门控：
  无数据，不产生错误特征
```

---

## 在流水线中的位置

```
realtime_processor._process_window()

  ChannelQuality（检测硬件故障）
  pipeline.process_df()（滤波、基线校正）
  ASR（若已激活）
  ↓
  BlinkArtifactRemover.clean()   ← 插入位置
  ↓
  IMUArtifactRemover.clean()
  DataSaver.add()
```

放在 ASR 之后：
- ASR 激活时眨眼已在上游处理，BlinkRemover 看到的信号幅值低于其阈值，直接透传
- ASR 未激活时 BlinkRemover 独立承担眨眼处理，这是它的主要使用场景

---

## 边界情况处理

| 情况 | 处理方式 |
|---|---|
| 窗口内无眨眼 | 直接返回原始 df，更新滚动估计 |
| 单次眨眼（< 30%）| 线性插值，返回修复后 df |
| 频繁眨眼（> 30%）| 返回 None，不更新滚动估计 |
| 好锚点 < 2 个 | 无法插值，返回 None（同 interpolate_saturated） |
| warmup 未完成 | 使用 fallback_threshold |
| IMU 已门控窗口 | 不进入 BlinkRemover（上游已返回 None） |

---

## 文件结构

```
Preprocess/
  blink_artifact_remover.py   ← 新文件
  realtime_processor.py       ← 修改：引入 BlinkArtifactRemover，插入调用
```

接口设计：

```python
class BlinkArtifactRemover:
    def __init__(self, k=5, margin_ms=100, max_contamination=0.30,
                 warmup_windows=20, fallback_threshold=200.0): ...

    def clean(self, df: pd.DataFrame, sfreq: float) -> Optional[pd.DataFrame]:
        """
        返回修复后的 df，或 None（整窗口丢弃）。
        接口与 IMUArtifactRemover.clean() 对称。
        """

    @property
    def last_blink_detected(self) -> bool: ...

    @property
    def is_warmed_up(self) -> bool: ...
```

---

## 开放问题（实现前需确认）

1. `fallback_threshold` 的单位是滤波后信号（µV）还是原始 ADC 值？
   → 插入位置在 pipeline.process_df() 之后，单位是滤波后 µV，200 µV 合理。

2. 滚动估计用 `max(abs)` 还是 `std`？
   → 用 `max(abs)`，对眨眼的峰值检测更敏感（std 在 5s 窗口里被稀释约 1.5 倍，不够敏感）。

3. 多通道眨眼检测：用 4 通道中任一超阈，还是需要多通道同时超阈？
   → 任一通道超阈触发，避免单通道接触不良导致漏检。
   → 但插值只对超阈通道做，未超阈通道保持原样。
