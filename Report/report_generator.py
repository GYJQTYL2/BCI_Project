"""
教育学习报告生成器

输入 history_reader.load() 返回的 attention / cognitive_load 数据字典，
生成包含四个模块的结构化报告：注意力分析、认知负荷分析、学习情况与效果、建议。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


class EduReportGenerator:
    """
    教育报告生成器

    典型用法:
        generator = EduReportGenerator()
        report = generator.generate(attn_dict, cl_dict, session_id)
    """

    def generate(self, attn: dict, cl: dict, session_id: str) -> dict:
        """
        生成学习报告

        参数:
            attn : history_reader.load("attention", ...) 的输出
                   字段: timestamps, attention_score, engagement_index,
                         theta_alpha_ratio, level
            cl   : history_reader.load("cognitive_load", ...) 的输出
                   字段: timestamps, cog_load_score, cognitive_load_index, level
            session_id : 会话 ID（YYYYMMDD_HHMMSS 格式）
        """
        has_attn = bool(attn.get("timestamps"))
        has_cl   = bool(cl.get("timestamps"))

        if not has_attn and not has_cl:
            return {"error": "无可用数据，请确认该会话包含注意力或认知负荷数据"}

        attn_analysis = self._analyze_attention(attn)   if has_attn else None
        cl_analysis   = self._analyze_cognitive_load(cl) if has_cl   else None
        learning      = self._analyze_learning(attn, cl, attn_analysis, cl_analysis)
        recommendations = self._generate_recommendations(attn_analysis, cl_analysis, learning)

        # 计算会话时长
        ts_src = attn.get("timestamps") or cl.get("timestamps") or []
        duration = _calc_duration(ts_src)

        return {
            "session_id":   session_id,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration":     duration,
            "attention":    attn_analysis,
            "cognitive_load": cl_analysis,
            "learning":     learning,
            "recommendations": recommendations,
        }

    # ── 注意力分析 ────────────────────────────────────────────────────────────

    def _analyze_attention(self, attn: dict) -> dict:
        scores = attn.get("attention_score", [])
        levels = attn.get("level", [])
        ei     = attn.get("engagement_index", [])
        tar    = attn.get("theta_alpha_ratio", [])

        if not scores:
            return {}

        arr = np.array(scores, dtype=float)
        mean_score = float(np.mean(arr))
        peak_score = float(np.max(arr))

        level_dist = _level_distribution(levels)
        focus_ratio = round(level_dist["high"] + level_dist["medium"], 3)

        trend, slope = _calc_trend(arr)

        return {
            "mean_score":        round(mean_score, 3),
            "peak_score":        round(peak_score, 3),
            "level_distribution": level_dist,
            "focus_ratio":       focus_ratio,
            "trend":             trend,
            "trend_slope":       round(float(slope), 5),
            "mean_ei":           round(float(np.mean(ei)), 3) if ei else None,
            "mean_tar":          round(float(np.mean(tar)), 3) if tar else None,
            "n_epochs":          len(scores),
        }

    # ── 认知负荷分析 ──────────────────────────────────────────────────────────

    def _analyze_cognitive_load(self, cl: dict) -> dict:
        scores = cl.get("cog_load_score", [])
        levels = cl.get("level", [])
        ci     = cl.get("cognitive_load_index", [])

        if not scores:
            return {}

        arr = np.array(scores, dtype=float)
        mean_score = float(np.mean(arr))

        level_dist = _level_distribution(levels)
        optimal_ratio  = level_dist["medium"]
        overload_ratio = level_dist["high"]
        underload_ratio = level_dist["low"]

        trend, slope = _calc_trend(arr)

        return {
            "mean_score":          round(mean_score, 3),
            "level_distribution":  level_dist,
            "optimal_load_ratio":  round(optimal_ratio, 3),
            "overload_ratio":      round(overload_ratio, 3),
            "underload_ratio":     round(underload_ratio, 3),
            "trend":               trend,
            "trend_slope":         round(float(slope), 5),
            "mean_ci":             round(float(np.mean(ci)), 3) if ci else None,
            "n_epochs":            len(scores),
        }

    # ── 学习情况与效果 ────────────────────────────────────────────────────────

    def _analyze_learning(
        self,
        attn:   dict,
        cl:     dict,
        attn_a: Optional[dict],
        cl_a:   Optional[dict],
    ) -> dict:
        # 疲劳检测：后半段注意力均值下降 > 15%，至少 10 个 epoch
        fatigue_detected    = False
        fatigue_onset_ratio: Optional[float] = None

        if attn_a and attn_a.get("n_epochs", 0) >= 10:
            scores = np.array(attn.get("attention_score", []), dtype=float)
            half = len(scores) // 2
            first_mean  = float(np.mean(scores[:half]))
            second_mean = float(np.mean(scores[half:]))
            if first_mean > 0.1 and (first_mean - second_mean) / first_mean > 0.15:
                fatigue_detected = True
                # 找到注意力首次持续低于阈值的位置
                window = max(3, len(scores) // 10)
                threshold = first_mean * 0.85
                for i in range(window, len(scores)):
                    if float(np.mean(scores[i - window: i])) < threshold:
                        fatigue_onset_ratio = round(i / len(scores), 2)
                        break

        # 学习效率指数
        efficiency = 0.0
        if attn_a and cl_a:
            focus   = attn_a.get("focus_ratio", 0.5)
            optimal = cl_a.get("optimal_load_ratio", 0.5)
            overload = cl_a.get("overload_ratio", 0.0)
            efficiency = float(np.clip(focus * (0.5 + optimal) - overload * 0.5, 0.0, 1.0))
        elif attn_a:
            efficiency = float(attn_a.get("focus_ratio", 0.5))
        elif cl_a:
            efficiency = float(cl_a.get("optimal_load_ratio", 0.5))

        efficiency = round(efficiency, 3)
        grade = _efficiency_grade(efficiency)

        return {
            "fatigue_detected":          fatigue_detected,
            "fatigue_onset_ratio":       fatigue_onset_ratio,
            "learning_efficiency_index": efficiency,
            "grade":                     grade,
        }

    # ── 建议生成 ──────────────────────────────────────────────────────────────

    def _generate_recommendations(
        self,
        attn_a:   Optional[dict],
        cl_a:     Optional[dict],
        learning: dict,
    ) -> List[dict]:
        recs: List[dict] = []

        # 注意力相关建议
        if attn_a:
            focus = attn_a.get("focus_ratio", 1.0)
            if focus >= 0.8:
                recs.append(_rec(
                    "attention", "good",
                    f"专注表现优秀，专注率达 {focus*100:.0f}%，保持当前学习节奏"
                ))
            elif focus < 0.5:
                recs.append(_rec(
                    "attention", "warning",
                    f"整体专注率仅 {focus*100:.0f}%，建议减少外界干扰，"
                    "尝试番茄工作法（25 分钟专注 + 5 分钟休息）"
                ))

            if attn_a.get("trend") == "declining":
                recs.append(_rec(
                    "attention", "warning",
                    "注意力呈下降趋势，建议在高注意力时段优先处理核心内容，适当安排休息"
                ))
            elif attn_a.get("trend") == "improving":
                recs.append(_rec(
                    "attention", "good",
                    "注意力呈上升趋势，学习状态持续改善，是处理难点内容的好时机"
                ))

        # 认知负荷相关建议
        if cl_a:
            overload  = cl_a.get("overload_ratio", 0.0)
            underload = cl_a.get("underload_ratio", 0.0)
            optimal   = cl_a.get("optimal_load_ratio", 0.0)

            if overload > 0.4:
                recs.append(_rec(
                    "cognitive_load", "warning",
                    f"高认知负荷占比 {overload*100:.0f}%，学习材料难度可能过高，"
                    "建议分解任务、增加复习或适当降低难度"
                ))
            elif underload > 0.4:
                recs.append(_rec(
                    "cognitive_load", "info",
                    f"低认知负荷占比 {underload*100:.0f}%，当前任务挑战性偏低，"
                    "可适当提升难度以进入更高效的学习状态"
                ))

            if optimal >= 0.5:
                recs.append(_rec(
                    "cognitive_load", "good",
                    f"认知负荷处于最优区间（中等负荷）的时间占 {optimal*100:.0f}%，"
                    "任务难度与能力匹配良好"
                ))

        # 疲劳相关建议
        if learning.get("fatigue_detected"):
            onset = learning.get("fatigue_onset_ratio")
            if onset is not None:
                onset_str = f"约 {onset*100:.0f}% 进度处（第 {int(onset * _get_n_epochs(attn_a, cl_a))} 分钟左右）"
            else:
                onset_str = "学习后半段"
            recs.append(_rec(
                "fatigue", "warning",
                f"检测到疲劳迹象，从{onset_str}开始注意力显著下降。"
                "建议下次在该时间点前安排 5~10 分钟短暂休息"
            ))

        # 兜底建议
        if not recs:
            recs.append(_rec(
                "general", "info",
                "数据点较少，暂时无法生成详细建议。建议采集更长时间的数据后再分析"
            ))

        return recs


# ── 内部辅助函数 ──────────────────────────────────────────────────────────────

def _level_distribution(levels: list) -> Dict[str, float]:
    dist = {"high": 0.0, "medium": 0.0, "low": 0.0}
    if not levels:
        return dist
    total = len(levels)
    for lv in levels:
        if lv in dist:
            dist[lv] += 1
    return {k: round(v / total, 3) for k, v in dist.items()}


def _calc_trend(arr: np.ndarray) -> Tuple[str, float]:
    if len(arr) < 4:
        return "stable", 0.0
    x = np.arange(len(arr), dtype=float)
    slope = float(np.polyfit(x, arr, 1)[0])
    mean_val = float(np.mean(arr))
    if mean_val < 1e-6:
        return "stable", slope
    # 归一化：整个序列的总变化量占均值的比例
    relative = slope * len(arr) / mean_val
    if relative > 0.10:
        return "improving", slope
    elif relative < -0.10:
        return "declining", slope
    return "stable", slope


def _efficiency_grade(score: float) -> str:
    if score >= 0.7:
        return "优秀"
    elif score >= 0.5:
        return "良好"
    elif score >= 0.3:
        return "一般"
    return "待改善"


def _calc_duration(timestamps: list) -> str:
    """从 'HH:MM:SS.mmm' 字符串列表计算时长"""
    if len(timestamps) < 2:
        return "未知"
    try:
        fmt = "%H:%M:%S.%f"
        t0 = datetime.strptime(timestamps[0],  fmt)
        t1 = datetime.strptime(timestamps[-1], fmt)
        delta = (t1 - t0).total_seconds()
        if delta < 0:   # 跨午夜
            delta += 86400
        minutes = int(delta // 60)
        seconds = int(delta % 60)
        return f"{minutes} 分 {seconds} 秒"
    except Exception:
        return "未知"


def _get_n_epochs(attn_a: Optional[dict], cl_a: Optional[dict]) -> int:
    if attn_a:
        return attn_a.get("n_epochs", 60)
    if cl_a:
        return cl_a.get("n_epochs", 60)
    return 60


def _rec(rec_type: str, level: str, message: str) -> dict:
    return {"type": rec_type, "level": level, "message": message}
