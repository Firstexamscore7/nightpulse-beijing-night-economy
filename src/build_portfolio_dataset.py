#!/usr/bin/env python3
"""Build an auditable analysis dataset from the night-culture survey workbooks.

The script never overwrites either source workbook. It creates three CSV views,
an alignment table, a data dictionary, a quality report, and a run manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SOURCE_FIELDS = [
    ("编号", "source_id", "原始记录编号", "标识", "唯一整数", "可用", "保留原始表编号，不用作模型特征。"),
    ("Q1_近3个月夜间消费频率", "night_consumption_frequency_3m", "近3个月夜间消费次数", "行为", "非负计数", "可用", "自报频率，不等于平台交易记录。"),
    ("Q2_1_核心商圈", "area_core_business_district", "夜间消费区域：核心商圈", "场景", "0/1", "可用", "多选题拆分字段。"),
    ("Q2_2_区域商圈", "area_regional_business_district", "夜间消费区域：区域商圈", "场景", "0/1", "可用", "多选题拆分字段。"),
    ("Q2_3_社区周边", "area_neighborhood", "夜间消费区域：社区周边", "场景", "0/1", "可用", "多选题拆分字段。"),
    ("Q2_4_景区周边", "area_scenic_spot", "夜间消费区域：景区周边", "场景", "0/1", "可用", "多选题拆分字段。"),
    ("Q3_周边可达性", "nearby_accessibility", "周边夜间消费场所可达性", "可达性", "Likert 1-5", "可用", "方向需结合原问卷题干解释。"),
    ("Q4_1_交通不够方便", "barrier_transport_inconvenient", "交通不够方便", "可达性堵点", "Likert 1-5", "可用", "得分越高表示认同程度越高。"),
    ("Q4_2_拥堵难以接受", "barrier_congestion", "拥堵难以接受", "可达性堵点", "Likert 1-5", "可用", "得分越高表示认同程度越高。"),
    ("Q4_3_交通费用偏高", "barrier_transport_cost", "交通费用偏高", "可达性堵点", "Likert 1-5", "可用", "得分越高表示认同程度越高。"),
    ("Q4_4_停车不够方便", "barrier_parking", "停车不够方便", "可达性堵点", "Likert 1-5", "可用", "得分越高表示认同程度越高。"),
    ("Q4_5_活动形式雷同", "barrier_similar_activities", "活动形式雷同", "体验性堵点", "Likert 1-5", "可用", "得分越高表示认同程度越高。"),
    ("Q4_6_商品千篇一律", "barrier_similar_products", "商品千篇一律", "体验性堵点", "Likert 1-5", "可用", "得分越高表示认同程度越高。"),
    ("Q4_7_业态类型单一", "barrier_limited_formats", "业态类型单一", "体验性堵点", "Likert 1-5", "可用", "得分越高表示认同程度越高。"),
    ("Q4_8_主题模糊", "barrier_unclear_theme", "主题模糊", "体验性堵点", "Likert 1-5", "可用", "得分越高表示认同程度越高。"),
    ("Q4_9_环境单调杂乱", "barrier_environment", "环境单调杂乱", "体验性堵点", "Likert 1-5", "可用", "得分越高表示认同程度越高。"),
    ("Q4_10_灯光音乐割裂", "barrier_light_music_mismatch", "灯光音乐割裂", "体验性堵点", "Likert 1-5", "可用", "得分越高表示认同程度越高。"),
    ("Q5_1_传统文化魅力", "value_traditional_culture", "传统文化魅力", "情感价值", "Likert 1-5", "可用", "用于情感价值维度。"),
    ("Q5_2_放松心情愉悦", "value_relaxation", "放松心情愉悦", "情感价值", "Likert 1-5", "可用", "用于情感价值维度。"),
    ("Q5_3_情感共鸣", "value_emotional_resonance", "情感共鸣", "情感价值", "Likert 1-5", "可用", "用于情感价值维度。"),
    ("Q5_4_群体共鸣归属感", "value_group_belonging", "群体共鸣与归属感", "社会价值", "Likert 1-5", "可用", "用于社会价值维度。"),
    ("Q5_5_文化社交方式", "value_cultural_socializing", "文化社交方式", "社会价值", "Likert 1-5", "可用", "用于社会价值维度。"),
    ("Q6_1_文化场景结合自然", "quality_culture_scene_fit", "文化与场景结合自然", "功能质量", "Likert 1-5", "可用", "用于功能质量维度。"),
    ("Q6_2_元素搭配协调", "quality_element_coordination", "元素搭配协调", "功能质量", "Likert 1-5", "可用", "用于功能质量维度。"),
    ("Q6_3_策划执行质量", "quality_execution", "策划执行质量", "功能质量", "Likert 1-5", "可用", "用于功能质量维度。"),
    ("Q6_4_物有所值", "price_value_for_money", "物有所值", "功能价格", "Likert 1-5", "可用", "用于功能价格维度。"),
    ("Q6_5_愿意支付溢价", "price_willingness_to_pay_more", "愿意支付溢价", "功能价格", "Likert 1-5", "可用", "态度题，不等于真实支付行为。"),
    ("Q7_1_节气手作", "preference_craft", "偏好：节气手作", "活动偏好", "0/1", "可用", "多选题拆分字段。"),
    ("Q7_2_游戏IP联动", "preference_game_ip", "偏好：游戏IP联动", "活动偏好", "0/1", "可用", "多选题拆分字段。"),
    ("Q7_3_节气养生体验", "preference_wellness", "偏好：节气养生体验", "活动偏好", "0/1", "可用", "多选题拆分字段。"),
    ("Q7_4_亲子节气互动", "preference_parent_child", "偏好：亲子节气互动", "活动偏好", "0/1", "可用", "多选题拆分字段。"),
    ("Q8_1_社区周边新增集市", "need_neighborhood_market", "改善需求：社区周边新增集市", "改善需求", "Likert 1-5", "可用", "意愿题，不代表措施实施效果。"),
    ("Q8_2_地铁延长至24点", "need_metro_extension", "改善需求：地铁延长至24点", "改善需求", "Likert 1-5", "可用", "意愿题，不代表措施实施效果。"),
    ("Q8_3_推出节气主题活动", "need_seasonal_events", "改善需求：节气主题活动", "改善需求", "Likert 1-5", "可用", "意愿题，不代表措施实施效果。"),
    ("Q8_4_抖音微信推送", "need_social_media_promotion", "改善需求：抖音微信推送", "改善需求", "Likert 1-5", "可用", "意愿题，不代表措施实施效果。"),
    ("Q9_交通解决后增加次数", "expected_visits_after_transport_improvement", "交通改善后预计增加次数", "意向结果", "非负计数", "谨慎使用", "反事实自报意向，不能解释为真实增量。"),
    ("Q10_节气活动后增加次数", "expected_visits_after_seasonal_events", "节气活动后预计增加次数", "意向结果", "非负计数", "谨慎使用", "反事实自报意向，不能解释为真实增量。"),
    ("Q11_溢价支付意愿百分比", "premium_willingness_pct_raw", "溢价支付意愿百分比（原始）", "意向结果", "0-100", "谨慎使用", "旧清洗表对69条记录作了未说明修改；保留原始值并做敏感性分析。"),
    ("Q12_性别(男=1 女=2)", "gender_code", "性别编码", "人口属性", "1=男，2=女", "可用", "只用于描述或公平性检查，不作为默认决策特征。"),
    ("Q13_年龄", "age", "年龄", "人口属性", "整数年龄", "可用", "用于描述、分层及公平性检查。"),
    ("Q14_常住区域", "residence_district_code", "常住区域编码", "人口属性", "类别1-17", "待补编码", "缺少编码与具体区域名称的对应表。"),
    ("Q15_月收入", "income_raw", "月收入原始离散值", "人口属性", "离散数值", "暂不建模", "原问卷类别定义缺失，旧编码存在行级错位风险。"),
    ("Q16_职业", "occupation_code", "职业编码", "人口属性", "类别1-8", "待补编码", "缺少职业编码对应表。"),
    ("Q17_学历", "education_raw", "学历原始编码或文本", "人口属性", "混合", "待补编码", "多数为1-6，另有一条文本值；缺少完整编码表。"),
]


DERIVED_SPECS = {
    "accessibility_barrier_score": [
        "barrier_transport_inconvenient",
        "barrier_congestion",
        "barrier_transport_cost",
        "barrier_parking",
    ],
    "experience_barrier_score": [
        "barrier_similar_activities",
        "barrier_similar_products",
        "barrier_limited_formats",
        "barrier_unclear_theme",
        "barrier_environment",
        "barrier_light_music_mismatch",
    ],
    "accessibility_improvement_need_score": [
        "need_neighborhood_market",
        "need_metro_extension",
    ],
    "content_improvement_need_score": [
        "need_seasonal_events",
        "need_social_media_promotion",
    ],
    "emotional_value_score": [
        "value_traditional_culture",
        "value_relaxation",
        "value_emotional_resonance",
    ],
    "social_value_score": [
        "value_group_belonging",
        "value_cultural_socializing",
    ],
    "functional_quality_score": [
        "quality_culture_scene_fit",
        "quality_element_coordination",
        "quality_execution",
    ],
    "functional_price_score": [
        "price_value_for_money",
        "price_willingness_to_pay_more",
    ],
    "seasonal_scene_overall_score": [
        "value_traditional_culture",
        "value_relaxation",
        "value_emotional_resonance",
        "value_group_belonging",
        "value_cultural_socializing",
        "quality_culture_scene_fit",
        "quality_element_coordination",
        "quality_execution",
        "price_value_for_money",
        "price_willingness_to_pay_more",
    ],
}


LEGACY_DERIVED_NAMES = {
    "accessibility_barrier_score": "可达性堵点",
    "experience_barrier_score": "体验性堵点",
    "accessibility_improvement_need_score": "可达性改善需求",
    "content_improvement_need_score": "内容性改善需求",
    "emotional_value_score": "情感价值",
    "social_value_score": "社会价值",
    "functional_quality_score": "功能质量",
    "functional_price_score": "功能价格",
    "seasonal_scene_overall_score": "节气场景总均分",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path, help="原始数据工作簿")
    parser.add_argument("--legacy-clean", required=True, type=Path, help="旧清洗工作簿")
    parser.add_argument("--output-dir", required=True, type=Path, help="输出目录")
    return parser.parse_args()


def normalize(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        return value.strip()
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{label}缺少字段: {missing}")


def align_legacy_rows(raw: pd.DataFrame, legacy: pd.DataFrame) -> tuple[list[tuple[int, int]], list[int]]:
    """Infer an order-preserving legacy-to-raw row alignment.

    Income is excluded from the matching cost because the legacy column is not
    row-consistent. Other changes remain in the cost so they can be audited.
    """
    if len(raw) < len(legacy):
        raise ValueError("旧清洗数据行数不能超过原始数据行数")
    gap = len(raw) - len(legacy)
    if gap > 200:
        raise ValueError(f"行数差异过大，停止自动对齐: {gap}")

    match_columns = [
        name for name, *_ in SOURCE_FIELDS if name not in {"编号", "Q15_月收入"}
    ]
    raw_rows = [tuple(normalize(v) for v in row) for row in raw[match_columns].itertuples(index=False, name=None)]
    legacy_rows = [tuple(normalize(v) for v in row) for row in legacy[match_columns].itertuples(index=False, name=None)]

    n, m = len(raw_rows), len(legacy_rows)
    inf = 10**9
    cost = np.full((n + 1, gap + 1), inf, dtype=np.int32)
    previous = np.full((n + 1, gap + 1), -1, dtype=np.int8)
    cost[0, 0] = 0

    for i in range(1, n + 1):
        min_gap = max(0, i - m)
        max_gap = min(gap, i)
        for skipped in range(min_gap, max_gap + 1):
            clean_count = i - skipped
            if skipped > 0 and cost[i - 1, skipped - 1] < cost[i, skipped]:
                cost[i, skipped] = cost[i - 1, skipped - 1]
                previous[i, skipped] = 1
            if clean_count > 0 and cost[i - 1, skipped] < inf:
                mismatch = sum(
                    left != right
                    for left, right in zip(raw_rows[i - 1], legacy_rows[clean_count - 1])
                )
                candidate = cost[i - 1, skipped] + mismatch
                if candidate <= cost[i, skipped]:
                    cost[i, skipped] = candidate
                    previous[i, skipped] = 0

    pairs: list[tuple[int, int]] = []
    skipped_rows: list[int] = []
    i, skipped = n, gap
    while i > 0:
        action = int(previous[i, skipped])
        if action == 1:
            skipped_rows.append(i - 1)
            i -= 1
            skipped -= 1
        elif action == 0:
            clean_count = i - skipped
            pairs.append((i - 1, clean_count - 1))
            i -= 1
        else:
            raise RuntimeError(f"对齐回溯失败: row={i}, skipped={skipped}")
    pairs.reverse()
    skipped_rows.reverse()
    if len(pairs) != len(legacy):
        raise RuntimeError("对齐结果行数不一致")
    return pairs, skipped_rows


def canonical_signature(row: Iterable[Any]) -> str:
    payload = json.dumps([normalize(v) for v in row], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cronbach_alpha(frame: pd.DataFrame) -> float:
    numeric = frame.apply(pd.to_numeric, errors="raise").astype(float)
    item_count = numeric.shape[1]
    total_variance = numeric.sum(axis=1).var(ddof=1)
    if item_count < 2 or total_variance == 0:
        return float("nan")
    return float(item_count / (item_count - 1) * (1 - numeric.var(ddof=1).sum() / total_variance))


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(clean(x) for x in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(clean(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def observed_domain(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "无非空值"
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all():
        unique = sorted(numeric.unique().tolist())
        if len(unique) <= 12:
            return ", ".join(str(int(v)) if float(v).is_integer() else str(v) for v in unique)
        return f"{numeric.min():g} 至 {numeric.max():g}；{len(unique)}个不同值"
    unique_text = sorted(str(v) for v in non_null.unique())
    if len(unique_text) <= 12:
        return ", ".join(unique_text)
    return f"{len(unique_text)}个不同值"


def direct_identifier_counts(frames: list[pd.DataFrame]) -> dict[str, int]:
    patterns = {
        "手机号码格式": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        "电子邮箱格式": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
        "身份证号码格式": re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    }
    counts = Counter()
    for frame in frames:
        for value in frame.astype(str).to_numpy().ravel():
            for name, pattern in patterns.items():
                if pattern.search(value):
                    counts[name] += 1
    return {name: int(counts[name]) for name in patterns}


def build_dataset(raw: pd.DataFrame, legacy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_names = [item[0] for item in SOURCE_FIELDS]
    require_columns(raw, source_names, "原始数据")
    require_columns(legacy, source_names + list(LEGACY_DERIVED_NAMES.values()), "旧清洗数据")

    pairs, skipped_raw_rows = align_legacy_rows(raw, legacy)
    rename_map = {source: target for source, target, *_ in SOURCE_FIELDS}
    dataset = raw[source_names].rename(columns=rename_map).copy()
    dataset["source_id"] = pd.to_numeric(dataset["source_id"], errors="raise").astype(int)

    raw_to_legacy = {raw_index: legacy_index for raw_index, legacy_index in pairs}
    dataset["legacy_included"] = [index in raw_to_legacy for index in range(len(raw))]
    dataset["legacy_clean_id"] = pd.array(
        [
            int(legacy.iloc[raw_to_legacy[index]]["编号"])
            if index in raw_to_legacy
            else pd.NA
            for index in range(len(raw))
        ],
        dtype="Int64",
    )
    dataset["legacy_exclusion_reason"] = np.where(
        dataset["legacy_included"], "", "旧版清洗未提供排除依据"
    )

    dataset["premium_willingness_pct_legacy"] = pd.array(
        [
            legacy.iloc[raw_to_legacy[index]]["Q11_溢价支付意愿百分比"]
            if index in raw_to_legacy
            else pd.NA
            for index in range(len(raw))
        ],
        dtype="Float64",
    )
    dataset["premium_willingness_modified_in_legacy"] = [
        bool(
            index in raw_to_legacy
            and normalize(raw.iloc[index]["Q11_溢价支付意愿百分比"])
            != normalize(legacy.iloc[raw_to_legacy[index]]["Q11_溢价支付意愿百分比"])
        )
        for index in range(len(raw))
    ]
    dataset["legacy_income_code_unverified"] = pd.array(
        [
            legacy.iloc[raw_to_legacy[index]]["Q15_月收入"]
            if index in raw_to_legacy
            else pd.NA
            for index in range(len(raw))
        ],
        dtype="Float64",
    )
    dataset["income_use_status"] = "暂不用于建模；等待原问卷编码说明"

    audit_match_columns = [
        name for name in source_names if name not in {"编号", "Q15_月收入"}
    ]
    changed_fields_by_raw: dict[int, list[str]] = defaultdict(list)
    alignment_rows = []
    for raw_index, legacy_index in pairs:
        changed = [
            name
            for name in audit_match_columns
            if normalize(raw.iloc[raw_index][name]) != normalize(legacy.iloc[legacy_index][name])
        ]
        changed_fields_by_raw[raw_index] = changed
        alignment_rows.append(
            {
                "source_id": int(raw.iloc[raw_index]["编号"]),
                "legacy_clean_id": int(legacy.iloc[legacy_index]["编号"]),
                "changed_field_count_excluding_income": len(changed),
                "changed_fields_excluding_income": ";".join(changed),
                "premium_pct_raw": raw.iloc[raw_index]["Q11_溢价支付意愿百分比"],
                "premium_pct_legacy": legacy.iloc[legacy_index]["Q11_溢价支付意愿百分比"],
                "income_raw": raw.iloc[raw_index]["Q15_月收入"],
                "legacy_income_code_unverified": legacy.iloc[legacy_index]["Q15_月收入"],
            }
        )
    alignment = pd.DataFrame(alignment_rows)
    dataset["legacy_change_count_excluding_income"] = [
        len(changed_fields_by_raw[index]) for index in range(len(raw))
    ]
    dataset["legacy_changed_fields_excluding_income"] = [
        ";".join(changed_fields_by_raw[index]) for index in range(len(raw))
    ]

    signature_columns = [target for source, target, *_ in SOURCE_FIELDS if source != "编号"]
    signatures = [
        canonical_signature(row)
        for row in dataset[signature_columns].itertuples(index=False, name=None)
    ]
    signature_counts = Counter(signatures)
    first_seen: dict[str, int] = {}
    duplicate_group_labels: dict[str, str] = {}
    group_number = 0
    for index, signature in enumerate(signatures):
        if signature_counts[signature] > 1 and signature not in duplicate_group_labels:
            group_number += 1
            duplicate_group_labels[signature] = f"DG{group_number:03d}"
        first_seen.setdefault(signature, index)
    dataset["duplicate_group_id"] = [duplicate_group_labels.get(sig, "") for sig in signatures]
    dataset["duplicate_group_size"] = [signature_counts[sig] for sig in signatures]
    dataset["is_exact_duplicate_group"] = [signature_counts[sig] > 1 for sig in signatures]
    dataset["is_duplicate_after_first"] = [index != first_seen[sig] for index, sig in enumerate(signatures)]

    for score_name, component_names in DERIVED_SPECS.items():
        dataset[score_name] = dataset[component_names].apply(pd.to_numeric, errors="raise").mean(axis=1)
    dataset["legacy_high_frequency_group"] = (
        pd.to_numeric(dataset["night_consumption_frequency_3m"], errors="raise") >= 5
    )
    dataset["age_below_16_flag"] = pd.to_numeric(dataset["age"], errors="raise") < 16
    dataset["analysis_record_status"] = "保留并按质量标记分析"
    dataset["public_release_status"] = "未确认公开授权"

    notes = []
    for row in dataset.itertuples(index=False):
        row_notes = []
        if not row.legacy_included:
            row_notes.append("旧版932样本未纳入")
        if row.is_exact_duplicate_group:
            row_notes.append("完全重复回答组")
        if row.premium_willingness_modified_in_legacy:
            row_notes.append("旧版修改溢价意愿")
        if row.age_below_16_flag:
            row_notes.append("年龄低于16岁，需核对纳入标准")
        row_notes.append("收入字段暂不建模")
        notes.append("；".join(row_notes))
    dataset["record_quality_notes"] = notes

    audit_columns = [
        "source_id",
        "legacy_included",
        "legacy_clean_id",
        "legacy_exclusion_reason",
        "duplicate_group_id",
        "duplicate_group_size",
        "is_exact_duplicate_group",
        "is_duplicate_after_first",
        "premium_willingness_modified_in_legacy",
        "premium_willingness_pct_legacy",
        "legacy_income_code_unverified",
        "income_use_status",
        "legacy_change_count_excluding_income",
        "legacy_changed_fields_excluding_income",
        "legacy_high_frequency_group",
        "age_below_16_flag",
        "analysis_record_status",
        "public_release_status",
        "record_quality_notes",
    ]
    source_columns = [target for _, target, *_ in SOURCE_FIELDS if target != "source_id"]
    derived_columns = list(DERIVED_SPECS)
    dataset = dataset[audit_columns + source_columns + derived_columns]

    diagnostics = {
        "alignment_pairs": pairs,
        "skipped_raw_rows": skipped_raw_rows,
        "duplicate_group_count": group_number,
        "duplicate_involved_rows": int(dataset["is_exact_duplicate_group"].sum()),
        "duplicate_extra_rows": int(dataset["is_duplicate_after_first"].sum()),
    }
    return dataset, alignment, diagnostics


def build_dictionary(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, field, description, group, scale, status, notes in SOURCE_FIELDS:
        rows.append(
            {
                "字段名": field,
                "原始字段": source,
                "说明": description,
                "分组": group,
                "预期类型或量表": scale,
                "实际取值": observed_domain(dataset[field]),
                "分析状态": status,
                "备注": notes,
            }
        )

    audit_meta = [
        ("legacy_included", "是否进入旧版932样本", "审计", "布尔", "可用", "不解释为有效或无效。"),
        ("legacy_clean_id", "旧清洗表重新编号", "审计", "可空整数", "可用", "仅用于对齐旧版数据。"),
        ("legacy_exclusion_reason", "旧版排除原因说明", "审计", "文本", "可用", "现有材料只支持标记为依据缺失。"),
        ("duplicate_group_id", "完全重复回答组编号", "审计", "文本", "可用", "仅对组内记录赋值。"),
        ("duplicate_group_size", "完全重复回答组大小", "审计", "整数", "可用", "单独记录取值为1。"),
        ("is_exact_duplicate_group", "是否属于完全重复回答组", "审计", "布尔", "可用", "不能仅凭该标记认定无效。"),
        ("is_duplicate_after_first", "是否为重复组内后续记录", "审计", "布尔", "可用", "用于去重敏感性样本。"),
        ("premium_willingness_modified_in_legacy", "旧版是否修改溢价意愿", "审计", "布尔", "可用", "共69条。"),
        ("premium_willingness_pct_legacy", "旧版溢价意愿值", "审计", "可空数值", "谨慎使用", "与原始值并列保留。"),
        ("legacy_income_code_unverified", "旧版收入编码", "审计", "可空数值", "暂不建模", "行级对应关系存在风险。"),
        ("income_use_status", "收入字段使用状态", "审计", "文本", "可用", "统一提示等待编码表。"),
        ("legacy_change_count_excluding_income", "除收入外旧版改动字段数", "审计", "整数", "可用", "用于追踪旧清洗改动。"),
        ("legacy_changed_fields_excluding_income", "除收入外旧版改动字段", "审计", "文本", "可用", "分号分隔。"),
        ("legacy_high_frequency_group", "旧报告高频组口径", "派生", "布尔", "可用", "近3个月夜间消费频率大于等于5。"),
        ("age_below_16_flag", "年龄是否低于16岁", "审计", "布尔", "待核实", "原始表有3条，旧版均未纳入；缺少原问卷最低年龄标准，不能直接判定为无效。"),
        ("analysis_record_status", "新分析记录状态", "审计", "文本", "可用", "当前不在缺少证据时删除记录。"),
        ("public_release_status", "公开发布状态", "治理", "文本", "可用", "公开前需确认数据权利。"),
        ("record_quality_notes", "行级质量提示", "审计", "文本", "可用", "汇总关键审计标记。"),
    ]
    for field, description, group, scale, status, notes in audit_meta:
        rows.append(
            {
                "字段名": field,
                "原始字段": "新建",
                "说明": description,
                "分组": group,
                "预期类型或量表": scale,
                "实际取值": observed_domain(dataset[field]),
                "分析状态": status,
                "备注": notes,
            }
        )

    score_descriptions = {
        "accessibility_barrier_score": "可达性堵点均值",
        "experience_barrier_score": "体验性堵点均值",
        "accessibility_improvement_need_score": "可达性改善需求均值",
        "content_improvement_need_score": "内容性改善需求均值",
        "emotional_value_score": "情感价值均值",
        "social_value_score": "社会价值均值",
        "functional_quality_score": "功能质量均值",
        "functional_price_score": "功能价格均值",
        "seasonal_scene_overall_score": "节气场景总均分",
    }
    for field, components in DERIVED_SPECS.items():
        rows.append(
            {
                "字段名": field,
                "原始字段": "新建",
                "说明": score_descriptions[field],
                "分组": "派生量表",
                "预期类型或量表": "题项行均值",
                "实际取值": observed_domain(dataset[field]),
                "分析状态": "可用",
                "备注": "组成字段：" + ", ".join(components),
            }
        )
    return pd.DataFrame(rows)


def dictionary_markdown(dictionary: pd.DataFrame) -> str:
    rows = []
    for row in dictionary.itertuples(index=False):
        rows.append([row.字段名, row.原始字段, row.说明, row.预期类型或量表, row.分析状态, row.备注])
    return "\n".join(
        [
            "# 数据字典 v0.1",
            "",
            "本字典依据现有两份 Excel 和答辩材料整理。缺少原问卷编码表的字段保持待核实状态，不补造类别含义。",
            "",
            markdown_table(["字段名", "原始字段", "说明", "类型或量表", "分析状态", "备注"], rows),
            "",
        ]
    )


def build_quality_report(
    raw: pd.DataFrame,
    legacy: pd.DataFrame,
    dataset: pd.DataFrame,
    alignment: pd.DataFrame,
    diagnostics: dict[str, Any],
    source_hashes: dict[str, str],
) -> str:
    legacy_view = dataset[dataset["legacy_included"]].copy()
    dedup_view = legacy_view[~legacy_view["is_duplicate_after_first"]].copy()

    identifier_counts = direct_identifier_counts([raw, legacy])
    changed_fields = Counter()
    for value in alignment["changed_fields_excluding_income"]:
        for field in str(value).split(";"):
            if field:
                changed_fields[field] += 1

    range_checks = []
    binary = [field for source, field, *_ in SOURCE_FIELDS if source.startswith("Q2_") or source.startswith("Q7_")]
    likert = [
        field
        for source, field, *_ in SOURCE_FIELDS
        if source.startswith("Q4_") or source.startswith("Q5_") or source.startswith("Q6_") or source.startswith("Q8_")
    ] + ["nearby_accessibility"]
    for label, fields, valid in [
        ("0/1字段", binary, lambda value: value in {0, 1}),
        ("Likert 1-5字段", likert, lambda value: value in {1, 2, 3, 4, 5}),
    ]:
        invalid = 0
        for field in fields:
            invalid += sum(not valid(normalize(value)) for value in dataset[field].dropna())
        range_checks.append([label, len(fields), invalid])
    range_checks.extend(
        [
            ["年龄低于16岁（需核对纳入标准）", 1, int((pd.to_numeric(dataset["age"]) < 16).sum())],
            ["年龄高于100岁", 1, int((pd.to_numeric(dataset["age"]) > 100).sum())],
            ["溢价意愿0-100", 1, int((~pd.to_numeric(dataset["premium_willingness_pct_raw"]).between(0, 100)).sum())],
            ["消费频率非负", 1, int((pd.to_numeric(dataset["night_consumption_frequency_3m"]) < 0).sum())],
        ]
    )

    score_rows = []
    for score, components in DERIVED_SPECS.items():
        legacy_name = LEGACY_DERIVED_NAMES[score]
        differences = []
        for raw_index, legacy_index in diagnostics["alignment_pairs"]:
            differences.append(abs(float(dataset.iloc[raw_index][score]) - float(legacy.iloc[legacy_index][legacy_name])))
        score_rows.append([score, f"{max(differences):.12g}", int(sum(value > 1e-12 for value in differences))])

    alpha_dimensions = {
        "情感价值": DERIVED_SPECS["emotional_value_score"],
        "社会价值": DERIVED_SPECS["social_value_score"],
        "功能质量": DERIVED_SPECS["functional_quality_score"],
        "功能价格": DERIVED_SPECS["functional_price_score"],
        "可达性堵点": DERIVED_SPECS["accessibility_barrier_score"],
        "体验性堵点": DERIVED_SPECS["experience_barrier_score"],
    }
    alpha_rows = []
    for name, fields in alpha_dimensions.items():
        alpha_rows.append(
            [
                name,
                f"{cronbach_alpha(dataset[fields]):.3f}",
                f"{cronbach_alpha(legacy_view[fields]):.3f}",
                f"{cronbach_alpha(dedup_view[fields]):.3f}",
            ]
        )

    mean_rows = []
    for score in [
        "emotional_value_score",
        "social_value_score",
        "functional_quality_score",
        "functional_price_score",
        "accessibility_barrier_score",
        "experience_barrier_score",
    ]:
        mean_rows.append(
            [
                score,
                f"{dataset[score].mean():.3f}",
                f"{legacy_view[score].mean():.3f}",
                f"{dedup_view[score].mean():.3f}",
            ]
        )

    premium_changes = (
        alignment[alignment["premium_pct_raw"] != alignment["premium_pct_legacy"]]
        .groupby(["premium_pct_raw", "premium_pct_legacy"])
        .size()
        .reset_index(name="记录数")
        .sort_values(["premium_pct_raw", "premium_pct_legacy"])
    )
    premium_rows = premium_changes.astype(object).values.tolist()

    income_cross = pd.crosstab(
        alignment["income_raw"], alignment["legacy_income_code_unverified"]
    ).sort_index()
    income_rows = [[index] + [int(value) for value in income_cross.loc[index]] for index in income_cross.index]
    income_headers = ["原始收入值"] + [f"旧编码{column:g}" for column in income_cross.columns]

    missing_rows = [
        [column, int(count)]
        for column, count in dataset.isna().sum().items()
        if int(count) > 0 and column not in {"legacy_clean_id", "premium_willingness_pct_legacy", "legacy_income_code_unverified"}
    ]

    changed_rows = [[field, count] for field, count in changed_fields.most_common()]
    if not changed_rows:
        changed_rows = [["无", 0]]

    return "\n".join(
        [
            "# 数据质量报告 v0.1",
            "",
            "## 结论",
            "",
            "现有数据可以支持描述性分析、量表检验和新的可解释模型，但旧清洗表不能直接作为唯一分析底表。新流程保留982条原始记录，用质量标记和敏感性分析代替没有证据的删除。收入字段暂不用于建模，溢价支付意愿同时保留原始值与旧版值。",
            "",
            "## 输入与版本",
            "",
            markdown_table(
                ["项目", "结果"],
                [
                    ["原始数据", f"{len(raw)}条记录，{len(raw.columns)}个字段"],
                    ["旧清洗数据", f"{len(legacy)}条记录，{len(legacy.columns)}个字段"],
                    ["新全量视图", f"{len(dataset)}条记录"],
                    ["旧口径视图", f"{len(legacy_view)}条记录"],
                    ["旧口径去重敏感性视图", f"{len(dedup_view)}条记录"],
                    ["原始文件SHA-256", source_hashes["raw"]],
                    ["旧清洗文件SHA-256", source_hashes["legacy_clean"]],
                ],
            ),
            "",
            "## 旧清洗对齐",
            "",
            markdown_table(
                ["检查", "结果"],
                [
                    ["成功对齐", f"{len(alignment)}条"],
                    ["旧版未纳入", f"{len(diagnostics['skipped_raw_rows'])}条"],
                    ["除收入外发生修改", f"{int((alignment['changed_field_count_excluding_income'] > 0).sum())}条"],
                    ["溢价支付意愿发生修改", f"{int(dataset['premium_willingness_modified_in_legacy'].sum())}条"],
                    ["编号处理", "旧清洗表重新编号为1至932"],
                ],
            ),
            "",
            markdown_table(["发生修改的原始字段", "记录数"], changed_rows),
            "",
            "对齐是根据记录顺序和除收入外的问卷答案推断得到。完全重复回答组内可能存在等价匹配，因此对齐表用于审计，不被当作原项目留存的清洗日志。",
            "",
            "## 完全重复回答",
            "",
            markdown_table(
                ["检查", "结果"],
                [
                    ["重复组数量", diagnostics["duplicate_group_count"]],
                    ["涉及记录", diagnostics["duplicate_involved_rows"]],
                    ["每组保留第一条后的额外记录", diagnostics["duplicate_extra_rows"]],
                    ["处理原则", "主视图保留；另建去重敏感性视图"],
                ],
            ),
            "",
            "完全重复不能单独证明问卷无效。后续报告应比较完整样本与去重敏感性样本，而不是直接删除后只报告一种结果。",
            "",
            "## 取值范围和缺失",
            "",
            markdown_table(["检查", "字段数", "范围外单元格"], range_checks),
            "",
            (markdown_table(["存在其他缺失的字段", "缺失数"], missing_rows) if missing_rows else "除审计字段因旧版未纳入而产生的结构性空值外，原始问卷字段未发现空值。"),
            "",
            "原始表有3条年龄为13或14岁的记录，旧版932样本均未纳入。现有材料没有提供最低年龄纳入标准，因此新数据集只做标记，不将其自动定义为无效问卷。",
            "",
            "## 直接标识符格式扫描",
            "",
            markdown_table(["格式", "匹配数"], [[name, value] for name, value in identifier_counts.items()]),
            "",
            "格式扫描只能发现常见号码和邮箱形式，不能替代人工隐私审查。PDF中的姓名、署名、印章及数据使用权仍需在公开发布前确认。",
            "",
            "## 派生指标复算",
            "",
            markdown_table(["派生指标", "与旧表最大绝对差", "差异记录数"], score_rows),
            "",
            "九个派生指标均从原始题项重新计算。与旧清洗表保存的公式结果一致时，说明量表均值口径能够复现；这不等于旧版模型已经复现。",
            "",
            "## 量表内部一致性敏感性",
            "",
            markdown_table(["维度", "全量982", "旧口径932", "旧口径去重"], alpha_rows),
            "",
            "## 关键均值敏感性",
            "",
            markdown_table(["指标", "全量982", "旧口径932", "旧口径去重"], mean_rows),
            "",
            "## 溢价支付意愿的旧版修改",
            "",
            markdown_table(["原始值", "旧版值", "记录数"], premium_rows),
            "",
            "旧材料没有解释变换依据。新分析默认使用原始百分比，并把旧版值作为敏感性对照。",
            "",
            "## 收入字段风险",
            "",
            markdown_table(income_headers, income_rows),
            "",
            "原始收入值与旧编码的总体频数能够对应，但同一行并非稳定的一对一映射。缺少问卷编码表时，旧收入编码不进入第一版聚类或预测模型。",
            "",
            "## 当前可用边界",
            "",
            "- 可直接用于描述：行为频率、场景选择、堵点、价值量表、改善需求、活动偏好和人口结构。",
            "- 需要谨慎解释：交通或活动改善后的预计增加次数、溢价支付意愿。这些是自报意向，不是实际增量。",
            "- 暂不用于建模：收入、常住区域名称、职业名称和学历类别解释，直到补齐编码表。",
            "- 尚未复现：K-means、CART、层次回归、中介效应、随机森林和SHAP。",
            "- 公开限制：三份CSV均为私有工作文件，未确认数据公开授权。",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    private_dir = output_dir / "private"
    metadata_dir = output_dir / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(args.raw, sheet_name="Sheet1", dtype=object)
    legacy = pd.read_excel(args.legacy_clean, sheet_name="Sheet1", dtype=object)
    dataset, alignment, diagnostics = build_dataset(raw, legacy)
    dictionary = build_dictionary(dataset)

    all_path = private_dir / "portfolio_analysis_v1_all_raw.csv"
    legacy_path = private_dir / "portfolio_analysis_v1_legacy932.csv"
    dedup_path = private_dir / "portfolio_analysis_v1_legacy_dedup.csv"
    alignment_path = private_dir / "legacy_alignment_audit.csv"
    dictionary_csv_path = output_dir / "data_dictionary_v0.1.csv"
    dictionary_md_path = output_dir / "03_数据字典_v0.1.md"
    quality_path = output_dir / "04_数据质量报告_v0.1.md"

    dataset.to_csv(all_path, index=False, encoding="utf-8-sig")
    dataset[dataset["legacy_included"]].to_csv(legacy_path, index=False, encoding="utf-8-sig")
    dataset[
        dataset["legacy_included"] & ~dataset["is_duplicate_after_first"]
    ].to_csv(dedup_path, index=False, encoding="utf-8-sig")
    alignment.to_csv(alignment_path, index=False, encoding="utf-8-sig")
    dictionary.to_csv(dictionary_csv_path, index=False, encoding="utf-8-sig")
    dictionary_md_path.write_text(dictionary_markdown(dictionary), encoding="utf-8")

    source_hashes = {"raw": file_sha256(args.raw), "legacy_clean": file_sha256(args.legacy_clean)}
    quality_path.write_text(
        build_quality_report(raw, legacy, dataset, alignment, diagnostics, source_hashes),
        encoding="utf-8",
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "inputs": {
            "raw": {"filename": args.raw.name, "sha256": source_hashes["raw"]},
            "legacy_clean": {"filename": args.legacy_clean.name, "sha256": source_hashes["legacy_clean"]},
        },
        "outputs": {
            "all_raw_rows": len(dataset),
            "legacy_rows": int(dataset["legacy_included"].sum()),
            "legacy_deduplicated_rows": int(
                (dataset["legacy_included"] & ~dataset["is_duplicate_after_first"]).sum()
            ),
        },
        "rules": {
            "row_deletion": "none in all_raw view",
            "legacy_exclusions": "flagged, not relabeled as invalid",
            "duplicates": "flagged; separate sensitivity view",
            "premium_willingness": "raw and legacy values retained",
            "income": "excluded from initial modeling pending codebook",
        },
    }
    (metadata_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest["outputs"], ensure_ascii=False))


if __name__ == "__main__":
    main()
