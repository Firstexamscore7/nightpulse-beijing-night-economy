#!/usr/bin/env python3
"""Create the Phase 2 descriptive layer and interpretable model outputs.

The script uses only NumPy and pandas from the bundled workspace runtime. It
does not modify the Phase 1 data and does not publish row-level records.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version


SEED = 20261010
TARGET_SOURCE = "expected_visits_after_seasonal_events"
TARGET_NAME = "higher_stated_increase"
LOGISTIC_L2 = 1.0
TREE_MAX_DEPTH = 3
TREE_MIN_LEAF = 40

FEATURES = [
    "night_consumption_frequency_3m",
    "area_core_business_district",
    "area_regional_business_district",
    "area_neighborhood",
    "area_scenic_spot",
    "nearby_accessibility",
    "accessibility_barrier_score",
    "experience_barrier_score",
    "emotional_value_score",
    "social_value_score",
    "functional_quality_score",
    "price_value_for_money",
    "preference_craft",
    "preference_game_ip",
    "preference_wellness",
    "preference_parent_child",
    "need_neighborhood_market",
    "need_metro_extension",
    "need_social_media_promotion",
]

FEATURE_LABELS = {
    "night_consumption_frequency_3m": "近3个月夜间消费频率",
    "area_core_business_district": "消费区域：核心商圈",
    "area_regional_business_district": "消费区域：区域商圈",
    "area_neighborhood": "消费区域：社区周边",
    "area_scenic_spot": "消费区域：景区周边",
    "nearby_accessibility": "周边可达性",
    "accessibility_barrier_score": "可达性堵点",
    "experience_barrier_score": "体验性堵点",
    "emotional_value_score": "情感价值",
    "social_value_score": "社会价值",
    "functional_quality_score": "功能质量",
    "price_value_for_money": "物有所值",
    "preference_craft": "偏好：节气手作",
    "preference_game_ip": "偏好：游戏IP联动",
    "preference_wellness": "偏好：节气养生",
    "preference_parent_child": "偏好：亲子互动",
    "need_neighborhood_market": "需求：社区周边集市",
    "need_metro_extension": "需求：地铁延长运营",
    "need_social_media_promotion": "需求：社交媒体推广",
}

BARRIERS = {
    "barrier_transport_inconvenient": "交通不够方便",
    "barrier_congestion": "拥堵难以接受",
    "barrier_transport_cost": "交通费用偏高",
    "barrier_parking": "停车不够方便",
    "barrier_similar_activities": "活动形式雷同",
    "barrier_similar_products": "商品千篇一律",
    "barrier_limited_formats": "业态类型单一",
    "barrier_unclear_theme": "主题模糊",
    "barrier_environment": "环境单调杂乱",
    "barrier_light_music_mismatch": "灯光音乐割裂",
}

PREFERENCES = {
    "preference_craft": "节气手作",
    "preference_game_ip": "游戏IP联动",
    "preference_wellness": "节气养生体验",
    "preference_parent_child": "亲子节气互动",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Phase 1 全量CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Phase 2 输出目录")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"输入缺少字段: {missing}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    beta: np.ndarray

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        design = np.column_stack([np.ones(len(z)), z])
        return sigmoid(design @ self.beta)


def logistic_loss(design: np.ndarray, y: np.ndarray, beta: np.ndarray, l2: float) -> float:
    probability = sigmoid(design @ beta)
    eps = 1e-12
    data_loss = -np.mean(y * np.log(probability + eps) + (1 - y) * np.log(1 - probability + eps))
    return float(data_loss + l2 * np.sum(beta[1:] ** 2) / (2 * len(y)))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = LOGISTIC_L2) -> LogisticModel:
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), standardized])
    beta = np.zeros(design.shape[1], dtype=float)
    regularizer = np.diag(np.r_[0.0, np.full(design.shape[1] - 1, l2 / len(y))])

    for _ in range(100):
        probability = sigmoid(design @ beta)
        gradient = design.T @ (probability - y) / len(y)
        gradient[1:] += l2 * beta[1:] / len(y)
        weights = np.maximum(probability * (1.0 - probability), 1e-7)
        hessian = (design.T * weights) @ design / len(y) + regularizer
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        old_loss = logistic_loss(design, y, beta, l2)
        step_size = 1.0
        candidate = beta - step
        while logistic_loss(design, y, candidate, l2) > old_loss and step_size > 1e-5:
            step_size *= 0.5
            candidate = beta - step_size * step
        beta = candidate
        if np.max(np.abs(step_size * step)) < 1e-8:
            break
    return LogisticModel(mean=mean, scale=scale, beta=beta)


def auc_score(y: np.ndarray, probability: np.ndarray) -> float:
    positive = probability[y == 1]
    negative = probability[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    tp = int(np.sum((y == 1) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "auc": auc_score(y, probability),
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": float(np.mean(prediction == y)),
        "brier": float(np.mean((probability - y) ** 2)),
        "positive_rate": float(np.mean(y)),
    }


def stratified_folds(y: np.ndarray, fold_count: int = 5, seed: int = SEED) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(fold_count)]
    for class_value in (0, 1):
        indices = np.flatnonzero(y == class_value)
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            buckets[offset % fold_count].append(int(index))
    return [np.array(sorted(bucket), dtype=int) for bucket in buckets]


@dataclass
class TreeNode:
    probability: float
    sample_count: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


@dataclass
class ShallowTree:
    root: TreeNode
    importance: np.ndarray

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        output = np.empty(len(x), dtype=float)
        for row_index, row in enumerate(x):
            node = self.root
            while node.feature_index is not None:
                if row[node.feature_index] <= float(node.threshold):
                    node = node.left  # type: ignore[assignment]
                else:
                    node = node.right  # type: ignore[assignment]
            output[row_index] = node.probability
        return output


def gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p = float(np.mean(y))
    return 2.0 * p * (1.0 - p)


def fit_tree(
    x: np.ndarray,
    y: np.ndarray,
    max_depth: int = TREE_MAX_DEPTH,
    min_leaf: int = TREE_MIN_LEAF,
) -> ShallowTree:
    importance = np.zeros(x.shape[1], dtype=float)
    root_size = len(y)

    def build(indices: np.ndarray, depth: int) -> TreeNode:
        labels = y[indices]
        node = TreeNode(probability=float(np.mean(labels)), sample_count=len(indices))
        parent_gini = gini(labels)
        if depth >= max_depth or len(indices) < 2 * min_leaf or parent_gini < 1e-12:
            return node

        best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
        for feature_index in range(x.shape[1]):
            values = x[indices, feature_index]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            sorted_labels = labels[order]
            cumulative_positive = np.cumsum(sorted_labels)
            total_positive = cumulative_positive[-1]
            for split_position in range(min_leaf, len(indices) - min_leaf + 1):
                if split_position >= len(indices) or sorted_values[split_position - 1] == sorted_values[split_position]:
                    continue
                left_n = split_position
                right_n = len(indices) - split_position
                left_positive = cumulative_positive[split_position - 1]
                right_positive = total_positive - left_positive
                left_p = left_positive / left_n
                right_p = right_positive / right_n
                weighted = (
                    left_n * 2 * left_p * (1 - left_p)
                    + right_n * 2 * right_p * (1 - right_p)
                ) / len(indices)
                if best is None or weighted < best[0] - 1e-12:
                    threshold = float((sorted_values[split_position - 1] + sorted_values[split_position]) / 2)
                    left_indices = indices[order[:split_position]]
                    right_indices = indices[order[split_position:]]
                    best = (float(weighted), feature_index, threshold, left_indices, right_indices)

        if best is None:
            return node
        weighted, feature_index, threshold, left_indices, right_indices = best
        improvement = parent_gini - weighted
        if improvement <= 1e-10:
            return node
        importance[feature_index] += improvement * len(indices) / root_size
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = build(left_indices, depth + 1)
        node.right = build(right_indices, depth + 1)
        return node

    root = build(np.arange(len(y)), 0)
    if importance.sum() > 0:
        importance /= importance.sum()
    return ShallowTree(root=root, importance=importance)


def cross_validated_predictions(x: np.ndarray, y: np.ndarray, model_name: str) -> np.ndarray:
    folds = stratified_folds(y)
    probability = np.empty(len(y), dtype=float)
    all_indices = np.arange(len(y))
    for test_indices in folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_indices] = False
        train_indices = all_indices[train_mask]
        if model_name == "logistic":
            model = fit_logistic(x[train_indices], y[train_indices])
        elif model_name == "shallow_tree":
            model = fit_tree(x[train_indices], y[train_indices])
        elif model_name == "majority_baseline":
            probability[test_indices] = float(np.mean(y[train_indices]))
            continue
        else:
            raise ValueError(model_name)
        probability[test_indices] = model.predict_proba(x[test_indices])
    return probability


def tree_rules(tree: ShallowTree) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []

    def walk(node: TreeNode, conditions: list[str]) -> None:
        if node.feature_index is None:
            rules.append(
                {
                    "rule_id": len(rules) + 1,
                    "conditions": " 且 ".join(conditions) if conditions else "全部样本",
                    "sample_count": node.sample_count,
                    "estimated_positive_share": node.probability,
                }
            )
            return
        label = FEATURE_LABELS[FEATURES[node.feature_index]]
        threshold = float(node.threshold)
        walk(node.left, conditions + [f"{label} ≤ {threshold:.3g}"])  # type: ignore[arg-type]
        walk(node.right, conditions + [f"{label} > {threshold:.3g}"])  # type: ignore[arg-type]

    walk(tree.root, [])
    return rules


def horizontal_bar_svg(
    rows: list[tuple[str, float]],
    title: str,
    subtitle: str,
    path: Path,
    minimum: float = 0.0,
    maximum: float | None = None,
    signed: bool = False,
) -> None:
    width = 1000
    left = 270
    right = 90
    top = 105
    row_height = 42
    bottom = 55
    height = top + len(rows) * row_height + bottom
    values = [value for _, value in rows]
    if maximum is None:
        maximum = max(values) if values else 1.0
    if signed:
        abs_max = max(abs(min(values, default=-1)), abs(max(values, default=1)), 1e-9)
        minimum, maximum = -abs_max, abs_max
    plot_width = width - left - right

    def x_position(value: float) -> float:
        return left + (value - minimum) / (maximum - minimum) * plot_width

    zero_x = x_position(0.0) if minimum <= 0 <= maximum else left
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F7F4EE"/>',
        f'<text x="36" y="42" font-size="25" font-weight="700" fill="#17252A">{html.escape(title)}</text>',
        f'<text x="36" y="70" font-size="14" fill="#50646B">{html.escape(subtitle)}</text>',
        f'<line x1="{zero_x:.1f}" x2="{zero_x:.1f}" y1="{top-12}" y2="{height-bottom+4}" stroke="#8A9BA1" stroke-width="1"/>',
    ]
    for index, (label, value) in enumerate(rows):
        center_y = top + index * row_height
        value_x = x_position(value)
        bar_x = min(zero_x, value_x)
        bar_width = max(abs(value_x - zero_x), 1.5)
        color = "#D86943" if value >= 0 else "#287C8E"
        parts.append(f'<text x="{left-16}" y="{center_y+5}" text-anchor="end" font-size="15" fill="#24383F">{html.escape(label)}</text>')
        parts.append(f'<rect x="{bar_x:.1f}" y="{center_y-12}" width="{bar_width:.1f}" height="24" rx="4" fill="{color}"/>')
        anchor = "start" if value >= 0 else "end"
        text_x = value_x + (8 if value >= 0 else -8)
        parts.append(f'<text x="{text_x:.1f}" y="{center_y+5}" text-anchor="{anchor}" font-size="14" font-weight="600" fill="#17252A">{value:.3f}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")
    horizontal_bar_png(rows, title, subtitle, path.with_suffix(".png"), minimum, maximum, signed)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", size=size)


def horizontal_bar_png(
    rows: list[tuple[str, float]],
    title: str,
    subtitle: str,
    path: Path,
    minimum: float,
    maximum: float,
    signed: bool,
) -> None:
    scale = 1.4
    width = int(1000 * scale)
    left = int(300 * scale)
    right = int(110 * scale)
    top = int(115 * scale)
    row_height = int(44 * scale)
    bottom = int(55 * scale)
    height = top + len(rows) * row_height + bottom
    image = Image.new("RGB", (width, height), "#F7F4EE")
    draw = ImageDraw.Draw(image)
    title_font = load_font(int(25 * scale))
    body_font = load_font(int(14 * scale))
    label_font = load_font(int(15 * scale))
    value_font = load_font(int(14 * scale))
    draw.text((int(36 * scale), int(28 * scale)), title, font=title_font, fill="#17252A")
    draw.text((int(36 * scale), int(66 * scale)), subtitle, font=body_font, fill="#50646B")
    values = [value for _, value in rows]
    if signed:
        abs_max = max(abs(min(values, default=-1)), abs(max(values, default=1)), 1e-9)
        minimum, maximum = -abs_max, abs_max
    plot_width = width - left - right

    def x_position(value: float) -> float:
        return left + (value - minimum) / (maximum - minimum) * plot_width

    zero_x = x_position(0.0) if minimum <= 0 <= maximum else left
    draw.line((zero_x, top - 12, zero_x, height - bottom + 4), fill="#8A9BA1", width=2)
    for index, (label, value) in enumerate(rows):
        center_y = top + index * row_height
        label_box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((left - int(16 * scale) - (label_box[2] - label_box[0]), center_y - int(10 * scale)), label, font=label_font, fill="#24383F")
        value_x = x_position(value)
        bar_x0, bar_x1 = sorted([zero_x, value_x])
        if bar_x1 - bar_x0 < 2:
            bar_x1 = bar_x0 + 2
        color = "#D86943" if value >= 0 else "#287C8E"
        draw.rounded_rectangle((bar_x0, center_y - int(12 * scale), bar_x1, center_y + int(12 * scale)), radius=5, fill=color)
        value_text = f"{value:.3f}"
        value_box = draw.textbbox((0, 0), value_text, font=value_font)
        if value >= 0:
            text_x = value_x + int(8 * scale)
        else:
            text_x = value_x - int(8 * scale) - (value_box[2] - value_box[0])
        draw.text((text_x, center_y - int(9 * scale)), value_text, font=value_font, fill="#17252A")
    image.save(path, format="PNG", optimize=True)


def grouped_bar_svg(rows: list[tuple[str, float, float]], path: Path) -> None:
    width, height = 1000, 430
    left, right, top, bottom = 110, 40, 105, 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F7F4EE"/>',
        '<text x="36" y="42" font-size="25" font-weight="700" fill="#17252A">节气活动偏好选择率</text>',
        '<text x="36" y="70" font-size="14" fill="#50646B">全量样本与较高自报增加意向组；多选题，比例不相加为100%</text>',
        '<rect x="680" y="28" width="16" height="12" fill="#287C8E"/><text x="704" y="39" font-size="13" fill="#24383F">全量样本</text>',
        '<rect x="800" y="28" width="16" height="12" fill="#D86943"/><text x="824" y="39" font-size="13" fill="#24383F">较高意向组</text>',
    ]
    for tick in range(0, 81, 20):
        y = top + plot_height * (1 - tick / 80)
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#D8D3C8"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="13" fill="#50646B">{tick}%</text>')
    for index, (label, overall, high) in enumerate(rows):
        center = left + group_width * (index + 0.5)
        for offset, value, color in [(-20, overall, "#287C8E"), (20, high, "#D86943")]:
            bar_height = plot_height * min(value / 0.8, 1.0)
            x = center + offset - 16
            y = top + plot_height - bar_height
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="32" height="{bar_height:.1f}" rx="4" fill="{color}"/>')
            parts.append(f'<text x="{center+offset:.1f}" y="{y-7:.1f}" text-anchor="middle" font-size="13" font-weight="600" fill="#17252A">{value:.1%}</text>')
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+28}" text-anchor="middle" font-size="15" fill="#24383F">{html.escape(label)}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")
    grouped_bar_png(rows, path.with_suffix(".png"))


def grouped_bar_png(rows: list[tuple[str, float, float]], path: Path) -> None:
    scale = 1.4
    width, height = int(1000 * scale), int(440 * scale)
    left, right, top, bottom = int(110 * scale), int(40 * scale), int(110 * scale), int(90 * scale)
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / len(rows)
    image = Image.new("RGB", (width, height), "#F7F4EE")
    draw = ImageDraw.Draw(image)
    title_font = load_font(int(25 * scale))
    body_font = load_font(int(13 * scale))
    label_font = load_font(int(15 * scale))
    value_font = load_font(int(13 * scale))
    draw.text((int(36 * scale), int(28 * scale)), "节气活动偏好选择率", font=title_font, fill="#17252A")
    draw.text((int(36 * scale), int(66 * scale)), "全量样本与较高自报增加意向组；多选题，比例不相加为100%", font=body_font, fill="#50646B")
    draw.rectangle((int(690 * scale), int(30 * scale), int(706 * scale), int(42 * scale)), fill="#287C8E")
    draw.text((int(714 * scale), int(27 * scale)), "全量样本", font=body_font, fill="#24383F")
    draw.rectangle((int(810 * scale), int(30 * scale), int(826 * scale), int(42 * scale)), fill="#D86943")
    draw.text((int(834 * scale), int(27 * scale)), "较高意向组", font=body_font, fill="#24383F")
    for tick in range(0, 81, 20):
        y = top + plot_height * (1 - tick / 80)
        draw.line((left, y, width - right, y), fill="#D8D3C8", width=2)
        tick_text = f"{tick}%"
        box = draw.textbbox((0, 0), tick_text, font=body_font)
        draw.text((left - int(12 * scale) - (box[2] - box[0]), y - int(8 * scale)), tick_text, font=body_font, fill="#50646B")
    for index, (label, overall, high) in enumerate(rows):
        center = left + group_width * (index + 0.5)
        for offset, value, color in [(-22 * scale, overall, "#287C8E"), (22 * scale, high, "#D86943")]:
            bar_height = plot_height * min(value / 0.8, 1.0)
            x0 = center + offset - int(16 * scale)
            y0 = top + plot_height - bar_height
            draw.rounded_rectangle((x0, y0, x0 + int(32 * scale), top + plot_height), radius=5, fill=color)
            value_text = f"{value:.1%}"
            box = draw.textbbox((0, 0), value_text, font=value_font)
            draw.text((center + offset - (box[2] - box[0]) / 2, y0 - int(22 * scale)), value_text, font=value_font, fill="#17252A")
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((center - (box[2] - box[0]) / 2, height - bottom + int(28 * scale)), label, font=label_font, fill="#24383F")
    image.save(path, format="PNG", optimize=True)


def build_descriptive_outputs(frame: pd.DataFrame, output_dir: Path, threshold: float) -> None:
    high = frame[TARGET_SOURCE] > threshold
    barrier_rows = []
    for field, label in BARRIERS.items():
        barrier_rows.append(
            {
                "field": field,
                "label": label,
                "overall_mean": float(frame[field].mean()),
                "higher_intention_mean": float(frame.loc[high, field].mean()),
                "other_mean": float(frame.loc[~high, field].mean()),
                "mean_difference_high_minus_other": float(frame.loc[high, field].mean() - frame.loc[~high, field].mean()),
            }
        )
    barrier_table = pd.DataFrame(barrier_rows).sort_values("overall_mean", ascending=False)
    barrier_table.to_csv(output_dir / "dashboard_data" / "barrier_summary.csv", index=False, encoding="utf-8-sig")

    preference_rows = []
    for field, label in PREFERENCES.items():
        preference_rows.append(
            {
                "field": field,
                "label": label,
                "overall_selection_rate": float(frame[field].mean()),
                "higher_intention_selection_rate": float(frame.loc[high, field].mean()),
                "other_selection_rate": float(frame.loc[~high, field].mean()),
            }
        )
    preference_table = pd.DataFrame(preference_rows).sort_values("overall_selection_rate", ascending=False)
    preference_table.to_csv(output_dir / "dashboard_data" / "activity_preference_summary.csv", index=False, encoding="utf-8-sig")

    segments = []
    segment_defs = [
        ("当前低频（近3个月少于5次）", frame["night_consumption_frequency_3m"] < 5),
        ("当前高频（近3个月至少5次）", frame["night_consumption_frequency_3m"] >= 5),
    ]
    for segment, mask in segment_defs:
        subset = frame.loc[mask]
        row: dict[str, Any] = {
            "segment": segment,
            "sample_count": int(len(subset)),
            "sample_share": float(len(subset) / len(frame)),
            "higher_stated_increase_share": float((subset[TARGET_SOURCE] > threshold).mean()),
            "mean_current_frequency": float(subset["night_consumption_frequency_3m"].mean()),
            "mean_stated_increase": float(subset[TARGET_SOURCE].mean()),
            "mean_experience_barrier": float(subset["experience_barrier_score"].mean()),
            "mean_accessibility_barrier": float(subset["accessibility_barrier_score"].mean()),
            "mean_need_neighborhood_market": float(subset["need_neighborhood_market"].mean()),
            "mean_need_metro_extension": float(subset["need_metro_extension"].mean()),
            "mean_need_social_media_promotion": float(subset["need_social_media_promotion"].mean()),
        }
        for field, label in PREFERENCES.items():
            row[f"preference_rate_{field.removeprefix('preference_')}"] = float(subset[field].mean())
        segments.append(row)
    pd.DataFrame(segments).to_csv(output_dir / "dashboard_data" / "behavior_segment_summary.csv", index=False, encoding="utf-8-sig")

    horizontal_bar_svg(
        [(row["label"], row["overall_mean"]) for _, row in barrier_table.iterrows()],
        "夜间消费堵点排序",
        "全量982份问卷均值；1–5分，越高表示越认同该问题",
        output_dir / "figures" / "barrier_ranking.svg",
        minimum=0,
        maximum=5,
    )
    grouped_bar_svg(
        [(row["label"], row["overall_selection_rate"], row["higher_intention_selection_rate"]) for row in preference_rows],
        output_dir / "figures" / "activity_preferences.svg",
    )


def write_report(
    output_dir: Path,
    frame: pd.DataFrame,
    threshold: float,
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    tree_rule_table: pd.DataFrame,
) -> None:
    primary_metrics = metrics[metrics["sample"] == "all_raw_982"].set_index("model")
    primary_coefficients = coefficients[coefficients["sample"] == "all_raw_982"].copy()
    positive = primary_coefficients.sort_values("coefficient", ascending=False).head(5)
    negative = primary_coefficients.sort_values("coefficient", ascending=True).head(5)
    barrier_table = pd.read_csv(output_dir / "dashboard_data" / "barrier_summary.csv")
    preference_table = pd.read_csv(output_dir / "dashboard_data" / "activity_preference_summary.csv")
    segment_table = pd.read_csv(output_dir / "dashboard_data" / "behavior_segment_summary.csv")

    def metric_row(name: str) -> str:
        row = primary_metrics.loc[name]
        return f"| {name} | {row['auc']:.3f} | {row['balanced_accuracy']:.3f} | {row['f1']:.3f} | {row['accuracy']:.3f} | {row['brier']:.3f} |"

    lines = [
        "# 初步分析报告 v0.1",
        "",
        "## 结论先行",
        "",
        f"全量982份问卷中，节气活动后预计增加次数的中位数为{threshold:.0f}次；严格高于该值的受访者占{(frame[TARGET_SOURCE] > threshold).mean():.1%}。这一比例只描述问卷中的自报意向，不能当作真实客流增长率。",
        "",
        f"活动形式上，选择率最高的是“{preference_table.iloc[0]['label']}”（{preference_table.iloc[0]['overall_selection_rate']:.1%}），其次是“{preference_table.iloc[1]['label']}”（{preference_table.iloc[1]['overall_selection_rate']:.1%}）。运营试点可以优先把两者做成主方案与备选方案，但仍需结合场地、预算和小样本访谈验证。",
        "",
        f"体验堵点中，均值最高的三项是“{barrier_table.iloc[0]['label']}”（{barrier_table.iloc[0]['overall_mean']:.2f}）、“{barrier_table.iloc[1]['label']}”（{barrier_table.iloc[1]['overall_mean']:.2f}）和“{barrier_table.iloc[2]['label']}”（{barrier_table.iloc[2]['overall_mean']:.2f}）。这支持把内容差异化和业态丰富度作为试点设计检查项。",
        "",
        "## 模型表现",
        "",
        "以下为固定五折分层交叉验证的样本外预测结果。模型目标是检验是否存在可重复的关联结构，不是部署个人级预测。",
        "",
        "| 模型 | AUC | 平衡准确率 | F1 | 准确率 | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        metric_row("majority_baseline"),
        metric_row("frequency_only_logistic"),
        metric_row("without_frequency_logistic"),
        metric_row("full_logistic"),
        metric_row("shallow_tree"),
        "",
        f"仅使用当前夜间消费频率的逻辑回归AUC已达到{primary_metrics.loc['frequency_only_logistic', 'auc']:.3f}；去掉该字段后，其他特征合计的AUC为{primary_metrics.loc['without_frequency_logistic', 'auc']:.3f}；全特征模型AUC为{primary_metrics.loc['full_logistic', 'auc']:.3f}。这说明高表现在很大程度上来自当前频率与未来自报意向之间的强关联，全模型相对频率基线只增加{primary_metrics.loc['full_logistic', 'auc'] - primary_metrics.loc['frequency_only_logistic', 'auc']:.3f} AUC。模型应被视为行为分组和假设整理工具，而不是可靠的个人预测器。",
        "",
        "## 逻辑回归关联方向",
        "",
        "系数基于标准化特征。正值表示与较高自报增加意向同向，负值表示反向；它们不代表因果关系。",
        "",
        "### 较强正向关联",
        "",
        "| 特征 | 标准化系数 | 每1标准差比值比 | 三样本方向是否一致 |",
        "| --- | ---: | ---: | --- |",
    ]
    for _, row in positive.iterrows():
        lines.append(f"| {row['feature_label']} | {row['coefficient']:.3f} | {row['odds_ratio_per_sd']:.3f} | {'是' if row['sign_consistent_all_samples'] else '否'} |")
    lines.extend([
        "",
        "### 较强负向关联",
        "",
        "| 特征 | 标准化系数 | 每1标准差比值比 | 三样本方向是否一致 |",
        "| --- | ---: | ---: | --- |",
    ])
    for _, row in negative.iterrows():
        lines.append(f"| {row['feature_label']} | {row['coefficient']:.3f} | {row['odds_ratio_per_sd']:.3f} | {'是' if row['sign_consistent_all_samples'] else '否'} |")
    lines.extend([
        "",
        "## 行为分组",
        "",
        "| 分组 | 样本数 | 样本占比 | 较高自报增加意向占比 | 节气手作偏好 | 节气养生偏好 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, row in segment_table.iterrows():
        lines.append(
            f"| {row['segment']} | {int(row['sample_count'])} | {row['sample_share']:.1%} | {row['higher_stated_increase_share']:.1%} | {row['preference_rate_craft']:.1%} | {row['preference_rate_wellness']:.1%} |"
        )
    lines.extend([
        "",
        "## 浅层树规则",
        "",
        "浅层树只用于把关联结构翻译成少量可读规则。叶节点的比例是该模型在全量问卷上的组内比例，不是未来活动的转化率。",
        "",
        "| 规则 | 样本数 | 组内较高意向比例 |",
        "| --- | ---: | ---: |",
    ])
    for _, row in tree_rule_table.iterrows():
        lines.append(f"| {row['conditions']} | {int(row['sample_count'])} | {row['estimated_positive_share']:.1%} |")
    lines.extend([
        "",
        "## 敏感性结论",
        "",
        "完整指标见 `model_metrics.csv`，标准化系数见 `feature_coefficients.csv`。同一流程已在全量982、旧版932和旧版去重893三个样本上运行。系数方向一致标记用于提醒哪些关联对样本口径较稳定；它不是统计显著性检验。",
        "",
        "## 可执行的试点假设",
        "",
        "1. 以选择率领先的活动形式为主概念，第二名作为组合模块或备选方案。",
        "2. 方案评审时强制检查商品、活动与业态是否同质化，避免只做视觉包装。",
        "3. 不按模型给个人打分；先用当前消费频率构造两个行为分组，分别进行招募文案和活动时段的小规模测试。",
        "4. 在真实试点中记录曝光、预约、到场、停留、核销和复访，把问卷意向与行为数据分开报告。",
        "",
        "## 局限",
        "",
        "- 问卷为观察性、自报数据，存在共同方法偏差、记忆偏差和社会期许偏差；",
        "- “预计增加次数”是假设场景下的回答，不是实际消费；",
        "- 缺少抽样框、回收渠道、时间和区域配额信息，不能证明样本代表北京市总体；",
        "- 完全重复回答与旧版排除缺少原始处理记录，因此只做敏感性分析；",
        "- 模型未使用收入、职业、区域和学历类别，原因是编码表不完整；",
        "- 对外公开任何聚合结果、代码或问卷材料前，仍需确认团队与数据权利。",
    ])
    (output_dir / "06_初步分析报告_v0.1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    for directory in [output_dir, output_dir / "dashboard_data", output_dir / "figures", output_dir / "metadata"]:
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input)
    require_columns(frame, FEATURES + [TARGET_SOURCE, "legacy_included", "is_duplicate_after_first"] + list(BARRIERS) + list(PREFERENCES))
    frame["legacy_included"] = as_bool(frame["legacy_included"])
    frame["is_duplicate_after_first"] = as_bool(frame["is_duplicate_after_first"])
    threshold = float(frame[TARGET_SOURCE].median())

    samples = {
        "all_raw_982": frame.copy(),
        "legacy_932": frame.loc[frame["legacy_included"]].copy(),
        "legacy_dedup_893": frame.loc[frame["legacy_included"] & ~frame["is_duplicate_after_first"]].copy(),
    }
    expected_counts = {"all_raw_982": 982, "legacy_932": 932, "legacy_dedup_893": 893}
    actual_counts = {name: len(sample) for name, sample in samples.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"样本行数与Phase 1不一致: {actual_counts}")

    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    tree_importance_rows: list[dict[str, Any]] = []
    primary_tree: ShallowTree | None = None

    for sample_name, sample in samples.items():
        x = sample[FEATURES].to_numpy(dtype=float)
        y = (sample[TARGET_SOURCE].to_numpy(dtype=float) > threshold).astype(int)
        if np.isnan(x).any() or np.isnan(y).any():
            raise ValueError(f"{sample_name}含模型缺失值")

        model_specs = [
            ("majority_baseline", x, "majority_baseline"),
            ("frequency_only_logistic", sample[["night_consumption_frequency_3m"]].to_numpy(dtype=float), "logistic"),
            ("without_frequency_logistic", sample[[feature for feature in FEATURES if feature != "night_consumption_frequency_3m"]].to_numpy(dtype=float), "logistic"),
            ("full_logistic", x, "logistic"),
            ("shallow_tree", x, "shallow_tree"),
        ]
        for report_name, model_x, implementation in model_specs:
            probability = cross_validated_predictions(model_x, y, implementation)
            row = {"sample": sample_name, "model": report_name, "n": len(sample), "threshold": threshold}
            row.update(classification_metrics(y, probability))
            metric_rows.append(row)

        logistic = fit_logistic(x, y)
        for feature, coefficient in zip(FEATURES, logistic.beta[1:]):
            coefficient_rows.append(
                {
                    "sample": sample_name,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "coefficient": float(coefficient),
                    "odds_ratio_per_sd": float(math.exp(np.clip(coefficient, -20, 20))),
                    "feature_mean": float(sample[feature].mean()),
                    "feature_sd": float(sample[feature].std(ddof=0)),
                }
            )

        tree = fit_tree(x, y)
        for feature, importance in zip(FEATURES, tree.importance):
            tree_importance_rows.append(
                {
                    "sample": sample_name,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "importance": float(importance),
                }
            )
        if sample_name == "all_raw_982":
            primary_tree = tree

    metrics = pd.DataFrame(metric_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    signs = coefficients.assign(sign=np.sign(coefficients["coefficient"])).groupby("feature")["sign"].nunique()
    coefficients["sign_consistent_all_samples"] = coefficients["feature"].map(signs.eq(1)).astype(bool)
    tree_importance = pd.DataFrame(tree_importance_rows)

    metrics.to_csv(output_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(output_dir / "feature_coefficients.csv", index=False, encoding="utf-8-sig")
    tree_importance.to_csv(output_dir / "tree_feature_importance.csv", index=False, encoding="utf-8-sig")

    assert primary_tree is not None
    primary_rules = pd.DataFrame(tree_rules(primary_tree))
    primary_rules.to_csv(output_dir / "tree_rules_all_raw.csv", index=False, encoding="utf-8-sig")

    build_descriptive_outputs(frame, output_dir, threshold)
    primary_coefficients = coefficients[coefficients["sample"] == "all_raw_982"].copy()
    chart_rows = primary_coefficients.reindex(primary_coefficients["coefficient"].abs().sort_values(ascending=False).index).head(12)
    horizontal_bar_svg(
        [(row["feature_label"], row["coefficient"]) for _, row in chart_rows.iterrows()],
        "较高自报增加意向的关联因素",
        "逻辑回归标准化系数；橙色为同向，蓝色为反向；不代表因果",
        output_dir / "figures" / "logistic_coefficients.svg",
        signed=True,
    )

    write_report(output_dir, frame, threshold, metrics, coefficients, primary_rules)

    manifest = {
        "input_file": str(args.input),
        "input_sha256": file_sha256(args.input),
        "analysis_code_sha256": file_sha256(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": pillow_version,
        },
        "random_seed": SEED,
        "target_source": TARGET_SOURCE,
        "target_definition": f"{TARGET_SOURCE} > {threshold:g}",
        "target_interpretation": "样本内较高的自报节气活动增加意向，不是真实到访增量",
        "features": FEATURES,
        "excluded_direct_target_adjacent_feature": "need_seasonal_events",
        "logistic_l2": LOGISTIC_L2,
        "tree_max_depth": TREE_MAX_DEPTH,
        "tree_min_leaf": TREE_MIN_LEAF,
        "cross_validation": "5-fold stratified, fixed seed",
        "sample_counts": actual_counts,
        "primary_positive_share": float((frame[TARGET_SOURCE] > threshold).mean()),
    }
    (output_dir / "metadata" / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({"status": "ok", **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
