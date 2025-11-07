#!/usr/bin/env python3
"""
AMQA Fairness Analysis (seaborn edition)
========================================

This script reads AMQA result JSONL files named
``FairMP_Answer_[model]_[prompt].jsonl`` (or ``Fair_Answer_*`` as a fallback),
builds tidy DataFrames, computes metrics, and renders a suite of charts
using seaborn.  It augments the original analysis by introducing label
mapping so that internal keys (e.g., ``naive``, ``gpt``, ``race``) can be
displayed with more user‑friendly names (e.g., ``Baseline``, ``GPT‑4o``,
``Race``) while leaving the underlying computation untouched.

Charts included (same as the original script):

Chart 1  Bar grid (2 rows × 15 cols) of accuracies per (model × sensitive value)
         with 4 bars per subplot for prompts [naive, role, aware, few, cot].
Chart 2  Bar grid (1 row × 15 cols) of accuracy gaps (|white–black|, |male–female|, |high–low|).
Chart 3  Heatmap of relative change of gap vs naive (rows=prompts; cols=model×type).
Chart 3b Heatmap of absolute change of gap vs naive (ΔGAP).
Chart 4  Radar (pentagon) of mean gap by prompt (smaller is better).
Chart 5  Pareto scatter (Δaccuracy vs −Δbias).
Chart 6  Bar chart of answer change ratio vs naive.

The script also writes several CSV tables: accuracy per variant, gap per type,
relative gap change, answer change ratio, aggregate accuracy/bias, and
McNemar significance tests.

Author: ChatGPT (seaborn refactor with display label mapping)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

# ----------------------
# Configuration / Order
# ----------------------
PROMPT_ORDER = ["naive", "role", "aware", "few", "cot"]
MODEL_ORDER  = ["gpt", "claude", "gemini", "qwen", "deepseek"]
TYPE_ORDER   = ["race", "sex", "SES"]  # SES == income

# Value pairs per type (top row values first)
PAIR_TOP = {"race": "white", "sex": "male", "SES": "high_income"}
PAIR_BOT = {"race": "black", "sex": "female", "SES": "low_income"}

VARIANT_ORDER = [
    "original",
    "desensitized",
    "white", "black",
    "high_income", "low_income",
    "male", "female",
]

VARIANT_KEY_MAP: Dict[str, str] = {
    "test_model_answer_original": "original",
    "test_model_answer_desensitized": "desensitized",
    "test_model_answer_white": "white",
    "test_model_answer_black": "black",
    "test_model_answer_high_income": "high_income",
    "test_model_answer_low_income": "low_income",
    "test_model_answer_male": "male",
    "test_model_answer_female": "female",
}

DISPARITY_PAIRS = {
    "race": ("white", "black"),
    "sex": ("male", "female"),
    "SES": ("high_income", "low_income"),
}

# Prompt colors (as requested)
PROMPT_COLORS = {
    "naive": "#B8DE92",     # light green
    "role":  "#4CA7A0",     # teal
    "aware": "#6BB6E3",     # light blue
    "few":   "#ABB6E3",     # custom color for few-shot/mix
    "cot":   "#1E5AA8",     # deep blue
}

# ----------------------
# Display label mappings
# ----------------------
# These dictionaries map internal keys (used for computation and ordering)
# to user‑friendly labels for presentation.  If you wish to change how
# prompts, models or attributes appear in charts and tables, edit these
# mappings only; the rest of the code will pick up the new names.
PROMPT_LABEL = {
    "naive": "Baseline",
    "role": "Role-playing",
    "aware": "Bias-aware",
    "few": "Few-shot",
    "cot": "Chain-of-thought",
}

# Model label mapping.  Edit the values to reflect the exact model names you
# wish to display.  The keys must match those in MODEL_ORDER.
MODEL_LABEL = {
    "gpt": "GPT-5-Mini",
    "claude": "Claude-Sonnet-4",
    "gemini": "Gemini-2.5-Flash",
    "qwen": "Qwen-3",
    "deepseek": "DeepSeek-V3.1",
}

# Type label mapping (sensitive attributes)
TYPE_LABEL = {
    "race": "Race",
    "sex": "Sex",
    "SES": "SES",
}

# Value/variant label mapping (individual sensitive values)
VALUE_LABEL = {
    "white": "White",
    "black": "Black",
    "male": "Male",
    "female": "Female",
    "high_income": "High Income",
    "low_income": "Low Income",
    "original": "Original",
    "desensitized": "Desensitized",
}

# Helper functions for mapping keys to labels
def PL(p: str) -> str:
    """Return display label for a prompt key."""
    return PROMPT_LABEL.get(p, p)

def ML(m: str) -> str:
    """Return display label for a model key."""
    return MODEL_LABEL.get(m, m)

def TL(t: str) -> str:
    """Return display label for a type key."""
    return TYPE_LABEL.get(t, t)

def VL(v: str) -> str:
    """Return display label for a variant value key."""
    return VALUE_LABEL.get(v, v)

# ----------------------
# IO helpers
# ----------------------

def find_data_files(data_dir: str) -> Dict[Tuple[str, str], str]:
    """Locate JSONL files for each (model, prompt) combination."""
    file_map: Dict[Tuple[str, str], str] = {}
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith('.jsonl'):
            continue
        lower = fname.lower()
        if lower.startswith('fairmp_answer_'):
            base = fname[len('FairMP_Answer_'):]
        elif lower.startswith('fair_answer_'):
            base = fname[len('Fair_Answer_'):]
        else:
            continue
        model_prompt = base.rsplit('.', 1)[0]
        parts = model_prompt.split('_')
        if len(parts) < 2:
            continue
        model = parts[0]
        prompt = '_'.join(parts[1:])
        if prompt not in PROMPT_ORDER:
            continue
        file_map[(model, prompt)] = os.path.join(data_dir, fname)
    return file_map

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    rows: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def normalize(records: List[Dict[str, Any]], model: str, prompt: str) -> pd.DataFrame:
    """Flatten raw records into a DataFrame with variant-level rows."""
    data = []
    for r in records:
        qid = r.get('question_id')
        # iterate through variant keys
        for k, v in VARIANT_KEY_MAP.items():
            if k in r:
                data.append({
                    'model': model,
                    'prompt': prompt,
                    'question_id': qid,
                    'variant': v,
                    'answer_idx': r['answer_idx'],
                    'model_answer': r[k],
                })
    return pd.DataFrame(data)

# ----------------------
# Metrics
# ----------------------

def acc(series: pd.Series) -> float:
    # Placeholder (unused)
    return float((series == series.index.get_level_values(0)) if False else 0.0)

def variant_accuracy(df: pd.DataFrame) -> pd.Series:
    out = {}
    for v in VARIANT_ORDER:
        sub = df[df['variant'] == v]
        if len(sub) == 0:
            out[v] = np.nan
        else:
            out[v] = float((sub['model_answer'] == sub['answer_idx']).mean())
    return pd.Series(out)

def gap_for_type(df: pd.DataFrame, typ: str) -> float:
    a, b = DISPARITY_PAIRS[typ]
    acc_a = variant_accuracy(df)[a]
    acc_b = variant_accuracy(df)[b]
    return float(abs(acc_a - acc_b))

def answer_consistency_for_pair(df: pd.DataFrame, a: str, b: str) -> float:
    A = df[df['variant'] == a][['question_id', 'model_answer']].rename(columns={'model_answer': 'A'})
    B = df[df['variant'] == b][['question_id', 'model_answer']].rename(columns={'model_answer': 'B'})
    m = pd.merge(A, B, on='question_id', how='inner')
    if len(m) == 0:
        return np.nan
    return float((m['A'] == m['B']).mean())

def mcnemar(baseline: pd.Series, alt: pd.Series) -> float:
    if baseline.shape != alt.shape:
        return np.nan
    b = int(((baseline == True) & (alt == False)).sum())
    c = int(((baseline == False) & (alt == True)).sum())
    if b + c == 0:
        stat = 0.0
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - chi2.cdf(stat, 1)
    return float(p)

# ----------------------
# Build tidy tables
# ----------------------

def build_tables(data_dir: str):
    file_map = find_data_files(data_dir)
    frames = []
    for (model, prompt), path in sorted(file_map.items()):
        recs = load_jsonl(path)
        frames.append(normalize(recs, model, prompt))
    if not frames:
        raise RuntimeError("No JSONL files found.")
    df = pd.concat(frames, ignore_index=True)

    # Accuracy table per model/prompt/variant
    acc_rows = []
    for (model, prompt, variant), g in df.groupby(['model', 'prompt', 'variant']):
        a = float((g['model_answer'] == g['answer_idx']).mean())
        acc_rows.append({'model': model, 'prompt': prompt, 'variant': variant, 'accuracy': a})
    acc_df = pd.DataFrame(acc_rows)

    # Gap per model/prompt/type
    gap_rows = []
    for (model, prompt), g in df.groupby(['model', 'prompt']):
        for typ, (v1, v2) in DISPARITY_PAIRS.items():
            g1 = g[g['variant'] == v1]
            g2 = g[g['variant'] == v2]
            a1 = float((g1['model_answer'] == g1['answer_idx']).mean()) if len(g1) > 0 else np.nan
            a2 = float((g2['model_answer'] == g2['answer_idx']).mean()) if len(g2) > 0 else np.nan
            gap = abs(a1 - a2) if (not np.isnan(a1) and not np.isnan(a2)) else np.nan
            gap_rows.append({'model': model, 'prompt': prompt, 'type': typ, 'gap': gap})
    gap_df = pd.DataFrame(gap_rows)

    # Relative and absolute change vs naive for gaps
    rel_rows = []
    for model in MODEL_ORDER:
        base = gap_df[(gap_df['model'] == model) & (gap_df['prompt'] == PROMPT_ORDER[0])]  # naive
        if base.empty:
            continue
        for prompt in PROMPT_ORDER:
            temp = gap_df[(gap_df['model'] == model) & (gap_df['prompt'] == prompt)]
            if temp.empty:
                continue
            for typ in TYPE_ORDER:
                g_base = float(base[base['type'] == typ]['gap'].values[0]) if not base[base['type'] == typ].empty else np.nan
                g_val = float(temp[temp['type'] == typ]['gap'].values[0]) if not temp[temp['type'] == typ].empty else np.nan
                abs_change = (g_val - g_base) if (np.isfinite(g_val) and np.isfinite(g_base)) else np.nan
                rel_change = (abs_change / g_base) if (np.isfinite(abs_change) and np.isfinite(g_base) and g_base != 0) else np.nan
                rel_rows.append({
                    'model': model,
                    'prompt': prompt,
                    'type': typ,
                    'rel_change': rel_change,
                    'abs_change': abs_change,
                    'gap_base': g_base,
                    'gap_val': g_val,
                })
    rel_df = pd.DataFrame(rel_rows)

    # Per-variant answer change ratio vs naive
    change_rows = []
    for model in MODEL_ORDER:
        base_df = df[(df['model'] == model) & (df['prompt'] == PROMPT_ORDER[0])]
        if base_df.empty:
            continue
        for prompt in PROMPT_ORDER[1:]:
            alt_df = df[(df['model'] == model) & (df['prompt'] == prompt)]
            if alt_df.empty:
                continue
            for var in ["white", "black", "high_income", "low_income", "male", "female"]:
                A = base_df[base_df['variant'] == var][['question_id', 'model_answer']].rename(columns={'model_answer': 'base'})
                B = alt_df[alt_df['variant'] == var][['question_id', 'model_answer']].rename(columns={'model_answer': 'alt'})
                m = pd.merge(A, B, on='question_id', how='inner')
                if len(m) == 0:
                    ratio = np.nan
                else:
                    ratio = float((m['base'] != m['alt']).mean())
                change_rows.append({'model': model, 'prompt': prompt, 'variant': var, 'change_ratio': ratio})
    change_df = pd.DataFrame(change_rows)

    # Aggregate accuracy & bias for Pareto
    agg_rows = []
    for (model, prompt), g in df.groupby(['model', 'prompt']):
        # accuracy across 8 variants
        avgs = []
        for v in VARIANT_ORDER:
            gi = g[g['variant'] == v]
            if len(gi) > 0:
                avgs.append(float((gi['model_answer'] == gi['answer_idx']).mean()))
        agg_acc = float(np.nanmean(avgs)) if avgs else np.nan
        # bias = mean gap across (race, sex, SES)
        gaps = []
        for typ, (v1, v2) in DISPARITY_PAIRS.items():
            g1 = g[g['variant'] == v1]
            g2 = g[g['variant'] == v2]
            if len(g1) > 0 and len(g2) > 0:
                a1 = float((g1['model_answer'] == g1['answer_idx']).mean())
                a2 = float((g2['model_answer'] == g2['answer_idx']).mean())
                gaps.append(abs(a1 - a2))
        agg_bias = float(np.nanmean(gaps)) if gaps else np.nan
        agg_rows.append({'model': model, 'prompt': prompt, 'agg_acc': agg_acc, 'agg_bias': agg_bias})
    agg_df = pd.DataFrame(agg_rows)

    # McNemar p-values per variant (prompt vs naive)
    p_rows = []
    for model in MODEL_ORDER:
        base = df[(df['model'] == model) & (df['prompt'] == 'naive')]
        if base.empty:
            continue
        for prompt in PROMPT_ORDER[1:]:
            alt = df[(df['model'] == model) & (df['prompt'] == prompt)]
            if alt.empty:
                continue
            for var in ["white", "black", "male", "female", "high_income", "low_income"]:
                A = base[base['variant'] == var][['question_id', 'answer_idx', 'model_answer']].copy()
                B = alt[alt['variant'] == var][['question_id', 'answer_idx', 'model_answer']].copy()
                A['correct'] = (A['model_answer'] == A['answer_idx'])
                B['correct'] = (B['model_answer'] == B['answer_idx'])
                m = pd.merge(A[['question_id', 'correct']], B[['question_id', 'correct']], on='question_id', suffixes=('_base', '_alt'))
                if len(m) == 0:
                    p = np.nan
                else:
                    p = mcnemar(m['correct_base'].values, m['correct_alt'].values)
                p_rows.append({'model': model, 'prompt_var': f"{prompt}-{var}", 'p_value': p})
    p_df = pd.DataFrame(p_rows)

    return df, acc_df, gap_df, rel_df, change_df, agg_df, p_df

# ----------------------
# Plotting (seaborn)
# ----------------------

def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# def chart1_bar_grid(acc_df: pd.DataFrame, out_png: str):
#     """2×15 bar grid of accuracies per model × value."""
#     sns.set_theme(style="whitegrid")
#     fig, axes = plt.subplots(2, 15, figsize=(20, 8), sharey=True)
#     # Pivot for quick lookup: index=(model,prompt), columns=variant
#     acc_pivot = acc_df.pivot_table(index=['model', 'prompt'], columns='variant', values='accuracy')
#     for col_idx, model in enumerate(MODEL_ORDER):
#         for j, typ in enumerate(TYPE_ORDER):
#             c = col_idx * 3 + j
#             ax_top = axes[0, c]
#             ax_bot = axes[1, c]
#             top_val = PAIR_TOP[typ]
#             bot_val = PAIR_BOT[typ]
#
#             def build_df(variant):
#                 rows = []
#                 for prompt in PROMPT_ORDER:
#                     try:
#                         val = float(acc_pivot.loc[(model, prompt)][variant])
#                     except KeyError:
#                         val = np.nan
#                     rows.append({'prompt': prompt, 'accuracy': val})
#                 return pd.DataFrame(rows)
#             dt = build_df(top_val)
#             db = build_df(bot_val)
#
#             sns.barplot(data=dt, x='prompt', y='accuracy', ax=ax_top, hue='prompt', hue_order=PROMPT_ORDER,
#                         width=1.0, dodge=False, palette=[PROMPT_COLORS[p] for p in PROMPT_ORDER])
#             for cont in ax_top.containers:
#                 ax_top.bar_label(cont, fmt="%.2f", padding=1, rotation=90, fontsize=8)
#             sns.barplot(data=db, x='prompt', y='accuracy', ax=ax_bot, hue='prompt', hue_order=PROMPT_ORDER,
#                         width=1.0, dodge=False, palette=[PROMPT_COLORS[p] for p in PROMPT_ORDER])
#             for cont in ax_bot.containers:
#                 ax_bot.bar_label(cont, fmt="%.2f", padding=1, rotation=90, fontsize=8)
#             ax_top.set_title(f"{ML(model)}-{VL(top_val)}", fontsize=10, y=-0.25, rotation=15)
#             ax_bot.set_title(f"{ML(model)}-{VL(bot_val)}", fontsize=10, y=-0.25, rotation=15)
#             ax_top.set_xlabel("")
#             ax_bot.set_xlabel("")
#             ax_top.set_xticklabels([])
#             ax_bot.set_xticklabels([])
#             ax_top.set_ylim(0.8, 1)
#             ax_bot.set_ylim(0.6, 1)
#             # Uniform style for grids/ticks
#             for ax in axes.flat:
#                 ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='0.85', alpha=0.8)
#                 ax.xaxis.grid(False)
#                 ax.tick_params(axis='both', width=0.6, length=2, colors='0.5')
#                 for spine in ax.spines.values():
#                     spine.set_alpha(0.3)
#
#     # Global legend with mapped labels
#     axes[0, 0].set_ylabel("Accuracy")
#     axes[1, 0].set_ylabel("Accuracy")
#
#     from matplotlib.patches import Patch
#     handles = [Patch(color=PROMPT_COLORS[p], label=PL(p)) for p in PROMPT_ORDER]
#     fig.legend(handles, [PL(p) for p in PROMPT_ORDER],
#                loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(PROMPT_ORDER), frameon=False)
#     fig.tight_layout(rect=[0, 0, 1, 0.95])
#     os.makedirs(os.path.dirname(out_png), exist_ok=True)
#     fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
#     plt.close(fig)
#
#
# def chart2_gap_grid(gap_df: pd.DataFrame, out_png: str):
#     """1×15 bar grid of accuracy gaps (|white–black|, |male–female|, |high–low|)."""
#     sns.set_theme(style="whitegrid")
#     fig, axes = plt.subplots(1, 15, figsize=(20, 3.8), sharey=True)
#     gap_pivot = gap_df.pivot_table(index=['model', 'prompt'], columns='type', values='gap')
#     for col_idx, model in enumerate(MODEL_ORDER):
#         for j, typ in enumerate(TYPE_ORDER):
#             c = col_idx * 3 + j
#             ax = axes[c]
#             rows = []
#             for prompt in PROMPT_ORDER:
#                 try:
#                     val = float(gap_pivot.loc[(model, prompt)][typ])
#                 except KeyError:
#                     val = np.nan
#                 rows.append({'prompt': prompt, 'gap': val})
#             df_plot = pd.DataFrame(rows)
#             sns.barplot(
#                 data=df_plot, x='prompt', y='gap', ax=ax,
#                 hue='prompt', hue_order=PROMPT_ORDER,
#                 palette={p: PROMPT_COLORS[p] for p in PROMPT_ORDER},
#                 dodge=False, width=1.0
#             )
#             for cont in ax.containers:
#                 ax.bar_label(cont, fmt="%.2f", padding=1, rotation=90, fontsize=8)
#             ax.set_title(f"{ML(model)}-{TL(typ)}", fontsize=10, y=-0.25, rotation=15)
#             ax.set_xlabel("")
#             ax.set_xticks([])
#             ax.set_xticklabels([])
#             ax.set_ylim(0, 0.25)
#             leg = ax.get_legend()
#             if leg is not None:
#                 leg.remove()
#     # Uniform style for grids/ticks
#     axes[0].set_ylabel("Accuracy Gap")
#     for ax in axes.flat:
#         ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='0.85', alpha=0.8)
#         ax.xaxis.grid(False)
#         ax.tick_params(axis='both', width=0.6, length=2, colors='0.5')
#         for spine in ax.spines.values():
#             spine.set_alpha(0.3)
#     # Global legend
#     from matplotlib.patches import Patch
#     handles = [Patch(color=PROMPT_COLORS[p], label=PL(p)) for p in PROMPT_ORDER]
#     fig.legend(handles, [PL(p) for p in PROMPT_ORDER],
#                loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(PROMPT_ORDER), frameon=False)
#     fig.tight_layout(rect=[0, 0, 1, 0.95])
#     os.makedirs(os.path.dirname(out_png), exist_ok=True)
#     fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
#     plt.close(fig)


def chart1_bar_grid(acc_df: pd.DataFrame, out_png: str):
    """3×10 bar grid of accuracies: 行=敏感属性(type)，列=模型×两值(Top/Bottom)，每格显示各prompt柱状。
       列按“模型-敏感属性按列填充”的规则：
       每个模型占 2 列：左列=PAIR_TOP（white/male/high_income），右列=PAIR_BOT（black/female/low_income）。
       3 行依次为 type = [race, sex, SES]。
    """
    sns.set_theme(style="whitegrid")

    n_models = len(MODEL_ORDER)
    n_cols = n_models * 2     # 每个模型两列（Top/Bottom）
    n_rows = 3                # 三个属性
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharey=True)

    # pivot: index=(model,prompt), columns=variant -> accuracy
    acc_pivot = acc_df.pivot_table(index=['model', 'prompt'], columns='variant', values='accuracy')

    for m_idx, model in enumerate(MODEL_ORDER):
        # 每个模型两列的起始列索引
        col0 = m_idx * 2       # 左列（top值）
        col1 = col0 + 1        # 右列（bot值）

        for r_idx, typ in enumerate(TYPE_ORDER):
            top_val = PAIR_TOP[typ]
            bot_val = PAIR_BOT[typ]

            # —— 左列：top 值（white/male/high_income）
            ax_left = axes[r_idx, col0]
            rows_left = []
            for prompt in PROMPT_ORDER:
                try:
                    val = float(acc_pivot.loc[(model, prompt)][top_val])
                except KeyError:
                    val = np.nan
                rows_left.append({'prompt': prompt, 'accuracy': val})
            df_left = pd.DataFrame(rows_left)
            sns.barplot(data=df_left, x='prompt', y='accuracy', ax=ax_left,
                        hue='prompt', hue_order=PROMPT_ORDER,
                        width=1.0, dodge=False,
                        palette=[PROMPT_COLORS[p] for p in PROMPT_ORDER])
            for cont in ax_left.containers:
                ax_left.bar_label(cont, fmt="%.2f", padding=1, rotation=90, fontsize=12)
            ax_left.set_title(f"{ML(model)}-{VL(top_val)}", fontsize=16, y=-0.35, rotation=10)
            ax_left.set_xlabel("")
            ax_left.set_xticklabels([])
            ax_left.set_ylim(0.6, 1.0)

            # —— 右列：bot 值（black/female/low_income）
            ax_right = axes[r_idx, col1]
            rows_right = []
            for prompt in PROMPT_ORDER:
                try:
                    val = float(acc_pivot.loc[(model, prompt)][bot_val])
                except KeyError:
                    val = np.nan
                rows_right.append({'prompt': prompt, 'accuracy': val})
            df_right = pd.DataFrame(rows_right)
            sns.barplot(data=df_right, x='prompt', y='accuracy', ax=ax_right,
                        hue='prompt', hue_order=PROMPT_ORDER,
                        width=1.0, dodge=False,
                        palette=[PROMPT_COLORS[p] for p in PROMPT_ORDER])
            for cont in ax_right.containers:
                ax_right.bar_label(cont, fmt="%.2f", padding=1, rotation=90, fontsize=12)
            ax_right.set_title(f"{ML(model)}-{VL(bot_val)}", fontsize=16, y=-0.35, rotation=10)
            ax_right.set_xlabel("")
            ax_right.set_xticklabels([])
            ax_right.set_ylim(0.6, 1.0)

    # 统一网格/刻度样式（保持你原样式）
    for ax in np.ravel(axes):
        ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='0.85', alpha=0.8)
        ax.xaxis.grid(False)
        ax.tick_params(axis='both', width=0.6, length=2, colors='0.5')
        for spine in ax.spines.values():
            spine.set_alpha(0.3)

    # Y 轴标题（每行的第一列加一次）
    axes[0, 0].set_ylabel("Accuracy", fontsize=16)
    axes[1, 0].set_ylabel("Accuracy", fontsize=16)
    axes[2, 0].set_ylabel("Accuracy", fontsize=16)

    # 顶部全局图例（保持原样）
    from matplotlib.patches import Patch
    handles = [Patch(color=PROMPT_COLORS[p], label=PL(p)) for p in PROMPT_ORDER]
    fig.legend(handles, [PL(p) for p in PROMPT_ORDER],
               loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=len(PROMPT_ORDER), frameon=False,
               fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


# def chart2_gap_grid(gap_df: pd.DataFrame, out_png: str):
#     """2×10 bar grid of accuracy gaps.
#        为保持“最小改动 + 语义不变（仍是按 type 的 GAP）”，这里把 3 个 type 压到两行：
#          - 第 1 行：每个模型两列，依次显示 race gap 与 sex gap
#          - 第 2 行：每个模型两列，显示 SES gap（在该模型的两列里都画相同的 SES gap，保持整齐的 2×10 布局）
#        这样第 1 行第 1 列就是 {model}-race（你示例里写“White/Black”是值层面的称呼；
#        但 Chart2 的度量是“按属性的 gap”，保留语义更严谨）。
#     """
#     sns.set_theme(style="whitegrid")
#
#     n_models = len(MODEL_ORDER)
#     n_cols = n_models * 2
#     fig, axes = plt.subplots(2, n_cols, figsize=(20, 7), sharey=True)
#
#     gap_pivot = gap_df.pivot_table(index=['model', 'prompt'], columns='type', values='gap')
#
#     for m_idx, model in enumerate(MODEL_ORDER):
#         col0 = m_idx * 2
#         col1 = col0 + 1
#
#         # ------------- 第 1 行：race, sex -------------
#         # 左列：race gap
#         ax = axes[0, col0]
#         rows = []
#         for prompt in PROMPT_ORDER:
#             try:
#                 val = float(gap_pivot.loc[(model, prompt)]['race'])
#             except KeyError:
#                 val = np.nan
#             rows.append({'prompt': prompt, 'gap': val})
#         df_plot = pd.DataFrame(rows)
#         sns.barplot(
#             data=df_plot, x='prompt', y='gap', ax=ax,
#             hue='prompt', hue_order=PROMPT_ORDER,
#             palette={p: PROMPT_COLORS[p] for p in PROMPT_ORDER},
#             dodge=False, width=1.0
#         )
#         for cont in ax.containers:
#             ax.bar_label(cont, fmt="%.2f", padding=1, rotation=90, fontsize=12)
#         ax.set_title(f"{ML(model)}-{TL('race')}", fontsize=16, y=-0.25, rotation=10)
#         ax.set_xlabel("")
#         ax.set_xticks([])
#         ax.set_xticklabels([])
#         ax.set_ylim(0, 0.25)
#         leg = ax.get_legend()
#         if leg is not None:
#             leg.remove()
#
#         # 右列：sex gap
#         ax = axes[0, col1]
#         rows = []
#         for prompt in PROMPT_ORDER:
#             try:
#                 val = float(gap_pivot.loc[(model, prompt)]['sex'])
#             except KeyError:
#                 val = np.nan
#             rows.append({'prompt': prompt, 'gap': val})
#         df_plot = pd.DataFrame(rows)
#         sns.barplot(
#             data=df_plot, x='prompt', y='gap', ax=ax,
#             hue='prompt', hue_order=PROMPT_ORDER,
#             palette={p: PROMPT_COLORS[p] for p in PROMPT_ORDER},
#             dodge=False, width=1.0
#         )
#         for cont in ax.containers:
#             ax.bar_label(cont, fmt="%.2f", padding=1, rotation=90, fontsize=12)
#         ax.set_title(f"{ML(model)}-{TL('sex')}", fontsize=16, y=-0.25, rotation=10)
#         ax.set_xlabel("")
#         ax.set_xticks([])
#         ax.set_xticklabels([])
#         ax.set_ylim(0, 0.25)
#         leg = ax.get_legend()
#         if leg is not None:
#             leg.remove()
#
#         # ------------- 第 2 行：SES gap（占两列，值相同，保持 2×10 网格整齐） -------------
#         for cc in (col0, col1):
#             ax = axes[1, cc]
#             rows = []
#             for prompt in PROMPT_ORDER:
#                 try:
#                     val = float(gap_pivot.loc[(model, prompt)]['SES'])
#                 except KeyError:
#                     val = np.nan
#                 rows.append({'prompt': prompt, 'gap': val})
#             df_plot = pd.DataFrame(rows)
#             sns.barplot(
#                 data=df_plot, x='prompt', y='gap', ax=ax,
#                 hue='prompt', hue_order=PROMPT_ORDER,
#                 palette={p: PROMPT_COLORS[p] for p in PROMPT_ORDER},
#                 dodge=False, width=1.0
#             )
#             for cont in ax.containers:
#                 ax.bar_label(cont, fmt="%.2f", padding=1, rotation=90, fontsize=12)
#             ax.set_title(f"{ML(model)}-{TL('SES')}", fontsize=16, y=-0.25, rotation=10)
#             ax.set_xlabel("")
#             ax.set_xticks([])
#             ax.set_xticklabels([])
#             ax.set_ylim(0, 0.25)
#             leg = ax.get_legend()
#             if leg is not None:
#                 leg.remove()
#
#     # 统一样式
#     axes[0, 0].set_ylabel("Accuracy Gap", fontsize=16)
#     axes[1, 0].set_ylabel("Accuracy Gap", fontsize=16)
#     for ax in np.ravel(axes):
#         ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='0.85', alpha=0.8)
#         ax.xaxis.grid(False)
#         ax.tick_params(axis='both', width=0.6, length=2, colors='0.5')
#         for spine in ax.spines.values():
#             spine.set_alpha(0.3)
#
#     # 顶部图例（保持原样）
#     from matplotlib.patches import Patch
#     handles = [Patch(color=PROMPT_COLORS[p], label=PL(p)) for p in PROMPT_ORDER]
#     fig.legend(handles, [PL(p) for p in PROMPT_ORDER],
#                loc='upper center', bbox_to_anchor=(0.5, 0.98),
#                ncol=len(PROMPT_ORDER), frameon=False,
#                fontsize=16)
#
#     fig.tight_layout(rect=[0, 0, 1, 0.95])
#     os.makedirs(os.path.dirname(out_png), exist_ok=True)
#     fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
#     plt.close(fig)


def chart2_gap_grid(gap_df: pd.DataFrame, out_png: str):
    """2×8 bar grid of accuracy gaps (|white–black|, |male–female|, |high–low|).
       上排 8 个、下排 7 个 + 1 个“Overall”（所有模型×属性的平均 GAP）。
       顺序：MODEL_ORDER × TYPE_ORDER（15 个），最后一格为 Overall。
    """
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 8, figsize=(20, 7), sharey=True)

    # 先列出 15 个常规子图：5个模型 × 3个type
    items = [(m, t) for m in MODEL_ORDER for t in TYPE_ORDER]
    # 第 16 个位置放一个特殊标志，用于 Overall
    items.append(("__ALL__", "__ALL__"))  # 放在最后（第二行第8列）

    gap_pivot = gap_df.pivot_table(index=['model', 'prompt'], columns='type', values='gap')

    for i, (model, typ) in enumerate(items):
        r = 0 if i < 8 else 1
        c = i if r == 0 else i - 8
        ax = axes[r, c]

        rows = []
        if model == "__ALL__":
            # —— Overall：每个 prompt 下，对所有 (model × type) 的 gap 取均值 ——
            for prompt in PROMPT_ORDER:
                sub = gap_df[
                    (gap_df['prompt'] == prompt) &
                    (gap_df['model'].isin(MODEL_ORDER)) &
                    (gap_df['type'].isin(TYPE_ORDER))
                ]['gap']
                val = float(sub.mean()) if len(sub) else np.nan
                rows.append({'prompt': prompt, 'gap': val})
            title_txt = "Overall-Mean"
        else:
            # —— 常规：指定 (model, type) 的各 prompt gap ——
            for prompt in PROMPT_ORDER:
                try:
                    val = float(gap_pivot.loc[(model, prompt)][typ])
                except KeyError:
                    val = np.nan
                rows.append({'prompt': prompt, 'gap': val})
            title_txt = f"{ML(model)}-{TL(typ)}"

        df_plot = pd.DataFrame(rows)

        sns.barplot(
            data=df_plot, x='prompt', y='gap', ax=ax,
            hue='prompt', hue_order=PROMPT_ORDER,
            palette={p: PROMPT_COLORS[p] for p in PROMPT_ORDER},
            dodge=False, width=1.0
        )
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.3f", padding=1, rotation=90, fontsize=12)

        ax.set_title(title_txt, fontsize=16, y=-0.25, rotation=0)
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ylim(0, 0.25)

        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # 统一样式
    axes[0, 0].set_ylabel("Accuracy Gap", fontsize=16)
    axes[1, 0].set_ylabel("Accuracy Gap", fontsize=16)
    for ax in axes.ravel():
        ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='0.85', alpha=0.8)
        ax.xaxis.grid(False)
        ax.tick_params(axis='both', width=0.6, length=2, colors='0.5')
        for spine in ax.spines.values():
            spine.set_alpha(0.3)

    # 顶部图例
    from matplotlib.patches import Patch
    handles = [Patch(color=PROMPT_COLORS[p], label=PL(p)) for p in PROMPT_ORDER]
    fig.legend(handles, [PL(p) for p in PROMPT_ORDER],
               loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=len(PROMPT_ORDER), frameon=False, fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def chart3_gap_heatmap(rel_df: pd.DataFrame, out_png: str):
    """Heatmap of relative change vs naive for gaps."""
    sns.set_theme(style="white")
    cols = []
    for model in MODEL_ORDER:
        for typ in TYPE_ORDER:
            cols.append(f"{ML(model)}-{TL(typ)}")
    mat = np.full((len(PROMPT_ORDER), len(cols)), np.nan)
    for i, prompt in enumerate(PROMPT_ORDER):
        for j, model in enumerate(MODEL_ORDER):
            for k, typ in enumerate(TYPE_ORDER):
                col_idx = j * 3 + k
                temp = rel_df[(rel_df['model'] == model) & (rel_df['prompt'] == prompt) & (rel_df['type'] == typ)]
                if not temp.empty:
                    mat[i, col_idx] = float(temp['rel_change'].values[0])
    plt.figure(figsize=(24, 5))
    ax = sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                     vmin=-0.3, vmax=0.3,
                     cbar_kws={'label': 'Relative Bias Change vs Baseline'},
                     annot_kws={'size': 16})
    ax.set_yticks(np.arange(len(PROMPT_ORDER)) + 0.5)
    ax.set_yticklabels([PL(p) for p in PROMPT_ORDER], rotation=0, fontsize=16)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols, rotation=45, fontsize=16)
    # ax.set_xticklabels(cols, rotation=45)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Relative change vs baseline', size=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()


def chart3_gap_abs_heatmap(rel_df: pd.DataFrame, out_png: str):
    """Heatmap of absolute GAP change vs naive."""
    sns.set_theme(style="white")
    cols = [f"{ML(m)}-{TL(t)}" for m in MODEL_ORDER for t in TYPE_ORDER]
    mat = np.full((len(PROMPT_ORDER), len(cols)), np.nan)
    for i, prompt in enumerate(PROMPT_ORDER):
        for j, model in enumerate(MODEL_ORDER):
            for k, typ in enumerate(TYPE_ORDER):
                col_idx = j * 3 + k
                tmp = rel_df[(rel_df['model'] == model) & (rel_df['prompt'] == prompt) & (rel_df['type'] == typ)]
                if not tmp.empty:
                    mat[i, col_idx] = float(tmp['abs_change'].values[0])
    vmax = np.nanmax(np.abs(mat))
    # if not np.isfinite(vmax) or vmax < 0.01:
    #     vmax = 0.01
    plt.figure(figsize=(24, 5))
    ax = sns.heatmap(mat, mask=np.isnan(mat), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                     vmin=-0.03, vmax=0.03,
                     cbar_kws={'label': 'Absolute Bias Change vs Baseline'},
                     annot_kws={'size': 16})
    ax.set_yticks(np.arange(len(PROMPT_ORDER)) + 0.5)
    ax.set_yticklabels([PL(p) for p in PROMPT_ORDER], rotation=0, fontsize=16)
    # ax.set_yticklabels(PROMPT_ORDER, rotation=0, fontsize=16)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols, rotation=45, fontsize=16)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Relative change vs baseline', size=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()


def chart4_radar(rel_df: pd.DataFrame, out_png: str):
    """Radar (pentagon) of relative mean gap change (vs naive)."""
    sns.set_theme(style="white")
    models = MODEL_ORDER
    angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([ML(m) for m in models])
    # Set radial limits and ticks for negative values (improvements are negative)
    ax.set_ylim(-0.5, 0)
    yticks = [0, -0.1, -0.2, -0.3, -0.4, -0.5]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=12, weight='light', color='0.35')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='0.7')
    ax.axhline(0, color='gray', linewidth=0.8)
    # Plot each prompt (excluding naive)
    for prompt in PROMPT_ORDER:
        if prompt == 'naive':
            continue
        vals = []
        for m in models:
            tmp = rel_df[(rel_df['model'] == m) & (rel_df['prompt'] == prompt) & (rel_df['type'].isin(TYPE_ORDER))]
            vals.append(float(tmp['rel_change'].mean()) if not tmp.empty else np.nan)
        vals += vals[:1]
        ax.plot(angles, vals, label=PL(prompt), color=PROMPT_COLORS[prompt])
        ax.fill(angles, vals, alpha=0.08, color=PROMPT_COLORS[prompt])
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()


def chart4_gap_radar(gap_df: pd.DataFrame, out_png: str):
    """Radar of mean accuracy gap per prompt (smaller is better)."""
    sns.set_theme(style="white")
    # Compute mean gap per (model, prompt)
    means_by_prompt = {}
    vmax = 0.0
    for prompt in PROMPT_ORDER:
        vals = []
        for m in MODEL_ORDER:
            g = gap_df[(gap_df['model'] == m) & (gap_df['prompt'] == prompt) & (gap_df['type'].isin(TYPE_ORDER))]
            v = float(g['gap'].mean()) if not g.empty else np.nan
            vals.append(v)
        means_by_prompt[prompt] = vals
        vmax = max(vmax, np.nanmax(vals) if np.any(np.isfinite(vals)) else 0.0)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.25
    rmax = float(np.ceil((vmax * 1.15) / 0.05) * 0.05)
    rticks = np.arange(0, rmax + 1e-9, 0.05)
    N = len(MODEL_ORDER)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'polar': True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([ML(m) for m in MODEL_ORDER], fontsize=12)
    ax.set_ylim(0, rmax)
    ax.set_rlabel_position(90)
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{t:.2f}" if t > 0 else "0" for t in rticks], fontsize=10, color='0.35')
    ax.grid(True, linestyle="--", linewidth=0.7, color='0.85')
    # Draw radial reference line at 0 (center)
    ax.plot([np.pi / 2, np.pi / 2], [0, rmax], ls=':', lw=1.0, color='0.7')
    # Plot each prompt
    for prompt in PROMPT_ORDER:
        vals = means_by_prompt[prompt]
        if not np.any(np.isfinite(vals)):
            continue
        r = np.array(vals, dtype=float)
        r = np.nan_to_num(r, nan=0.0)
        r = np.r_[r, r[0]]
        color = PROMPT_COLORS.get(prompt)
        ax.plot(angles, r, label=PL(prompt), color=color, linewidth=2.2, marker='o', markersize=3.5)
        ax.fill(angles, r, color=color, alpha=0.10)
    # Legend with rounded box
    leg = ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1.10), frameon=True,
                    framealpha=0.95, facecolor='white', edgecolor='0.85', fontsize=11)
    frame = leg.get_frame()
    frame.set_linewidth(0.8)
    frame.set_edgecolor('0.85')
    frame.set_facecolor('white')
    frame.set_alpha(0.95)
    if hasattr(frame, "set_boxstyle"):
        frame.set_boxstyle("round,pad=0.3,rounding_size=2.0")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


# def chart5_pareto(agg_df: pd.DataFrame, out_png: str):
#     """Scatter: Δaccuracy vs −Δbias, with model hues and prompt styles."""
#     sns.set_theme(style="whitegrid")
#     base = agg_df[agg_df['prompt'] == "naive"].set_index('model')
#     rows = []
#     for _, r in agg_df.iterrows():
#         if r['prompt'] == "naive":
#             continue
#         if r['model'] not in base.index:
#             continue
#         d_acc = r['agg_acc'] - float(base.loc[r['model'], 'agg_acc'])
#         d_bias = r['agg_bias'] - float(base.loc[r['model'], 'agg_bias'])
#         rows.append({'model': r['model'], 'prompt': r['prompt'], 'd_acc': d_acc, 'neg_d_bias': -d_bias})
#     dfp = pd.DataFrame(rows)
#     plt.figure(figsize=(6, 4))
#     ax = sns.scatterplot(data=dfp, x='d_acc', y='neg_d_bias', hue='model', style='prompt')
#     ax.axhline(0, ls='--', lw=0.5, color='gray')
#     ax.axvline(0, ls='--', lw=0.5, color='gray')
#     # Remap legend labels for model and prompt
#     h, l = ax.get_legend_handles_labels()
#     mapped_labels = []
#     for label in l:
#         if label in MODEL_ORDER:
#             mapped_labels.append(ML(label))
#         elif label in PROMPT_ORDER:
#             mapped_labels.append(PL(label))
#         elif label == 'model':
#             mapped_labels.append('Model')
#         elif label == 'prompt':
#             mapped_labels.append('Prompt')
#         else:
#             mapped_labels.append(label)
#     ax.legend(h, mapped_labels, title=None)
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(out_png), exist_ok=True)
#     plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
#     plt.close()



def chart5_pareto(agg_df: pd.DataFrame, out_png: str):
    """Scatter: Δaccuracy vs −Δbias. 单框两行图例（上=Model 颜色；下=Prompt 形状），
    统一使用填充型 marker，并对缺失类别做兜底，避免 KeyError / mixed-marker 错误。
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    sns.set_theme(style="whitegrid")

    # —— 计算散点数据（排除 baseline/naive）——
    base = agg_df[agg_df['prompt'] == "naive"].set_index('model')
    rows = []
    for _, r in agg_df.iterrows():
        if r['prompt'] == "naive":
            continue
        if r['model'] not in base.index:
            continue
        d_acc  = r['agg_acc']  - float(base.loc[r['model'], 'agg_acc'])
        d_bias = r['agg_bias'] - float(base.loc[r['model'], 'agg_bias'])
        rows.append({'model': r['model'], 'prompt': r['prompt'],
                     'd_acc': d_acc, 'neg_d_bias': d_bias})
    dfp = pd.DataFrame(rows)

    # 实际出现的模型/提示词（仍按你的既定顺序）
    hue_order   = [m for m in MODEL_ORDER  if m in set(dfp['model'])]
    style_order = [p for p in PROMPT_ORDER if p in set(dfp['prompt'])]

    # 颜色映射：模型→颜色
    palette   = sns.color_palette(n_colors=len(hue_order))
    color_map = {m: c for m, c in zip(hue_order, palette)}

    # 统一“填充型” marker，并保证每个出现的 prompt 都有一个
    base_marker_map = {
        'naive': 'o',      # 不会绘制，仅占位
        'role':  'P',
        'aware': 'o',
        'few':   's',
        'cot':   '^',
        'mix':   'D',
    }
    filled_cycle = ['o','s','D','P','^','v','<','>','h','H']
    marker_map = {}
    for i, p in enumerate(style_order):
        marker_map[p] = base_marker_map.get(p, filled_cycle[i % len(filled_cycle)])

    # —— 绘图（关闭 seaborn 自动图例，用自定义图例）——
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    sns.scatterplot(
        data=dfp, x='d_acc', y='neg_d_bias',
        hue='model', style='prompt',
        hue_order=hue_order, style_order=style_order,
        palette=color_map, markers=marker_map,
        legend=False, ax=ax
    )

    ax.axhline(0, ls='--', lw=0.6, color='gray')
    ax.axvline(0, ls='--', lw=0.6, color='gray')

    ax.set_ylabel("Absolute Change in Bias (Lower better)")
    ax.set_xlabel("Absolute Change in Accuracy (Higher better)")

    # —— 自定义“单框两行”图例 —— #
    # 上行：模型（颜色圆点）
    # —— 自定义“单框两行”图例 —— #
    from matplotlib.lines import Line2D

    # 上行：模型（颜色圆点）
    model_handles = [
        Line2D([0], [0], marker='o', markersize=8, linestyle='',
               color=color_map[m], label=ML(m))
        for m in hue_order
    ]
    model_labels = [ML(m) for m in hue_order]

    # 下行：提示词（形状，用黑色填充）
    prompt_handles = [
        Line2D([0], [0], marker=marker_map[p], markersize=8, linestyle='',
               color='black', markerfacecolor='black', label=PL(p))
        for p in style_order
    ]
    prompt_labels = [PL(p) for p in style_order]

    # —— 关键：把两行“按列配对并交替”以适配 column-major —— #
    # 先把较短的一行补齐到相同长度（用不可见占位）
    def _blank_handle():
        return Line2D([], [], linestyle='', marker=None, label=' ')
    ncol = max(len(model_handles), len(prompt_handles))
    while len(model_handles)  < ncol: model_handles.append(_blank_handle());  model_labels.append(' ')
    while len(prompt_handles) < ncol: prompt_handles.append(_blank_handle()); prompt_labels.append(' ')

    # 交替合并：每一列都是 [模型, 提示词]
    combined_handles, combined_labels = [], []
    for mh, ph, ml, pl in zip(model_handles, prompt_handles, model_labels, prompt_labels):
        combined_handles.extend([mh, ph])
        combined_labels.extend([ml, pl])

    # 注意：ncol 设为“列数”，总元素个数是 2*ncol（两行）
    leg = fig.legend(
        combined_handles, combined_labels,
        loc='upper center', bbox_to_anchor=(0.5, 1.02),
        ncol=ncol, frameon=True, fancybox=True, framealpha=0.96,
        borderaxespad=0.2, columnspacing=1.4, handletextpad=0.6, labelspacing=0.8
    )
    leg.set_title(None)

    # 顶部留白，避免遮盖图面
    plt.tight_layout(rect=[0, 0, 1, 0.90])


    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)




#
# def chart6_change_bar(change_df: pd.DataFrame, out_png: str):
#     """Bar plot: x=model-variant, y=answer-change ratio vs naive, hue=prompt."""
#     sns.set_theme(style="whitegrid")
#     order_vals = ["white", "black", "male", "female", "high_income", "low_income"]
#     change_df = change_df.copy()
#     change_df['cat'] = change_df['model'] + '-' + change_df['variant']
#     change_df['model'] = pd.Categorical(change_df['model'], categories=MODEL_ORDER, ordered=True)
#     change_df['variant'] = pd.Categorical(change_df['variant'], categories=order_vals, ordered=True)
#     change_df = change_df.sort_values(['model', 'variant', 'prompt'])
#     plt.figure(figsize=(18, 4))
#     ax = sns.barplot(data=change_df, x='cat', y='change_ratio', hue='prompt', hue_order=["role", "aware", "cot"],
#                      palette=[PROMPT_COLORS[p] for p in ["role", "aware", "cot"]])
#     ax.set_xlabel("model-variant")
#     ax.set_ylabel("change ratio")
#     ax.set_ylim(0, 0.2)
#     # Remap legend labels
#     h, l = ax.get_legend_handles_labels()
#     ax.legend(h, [PL(x) for x in l], title=None)
#     # Remap x tick labels to display names
#     xt = [t.get_text() for t in ax.get_xticklabels()]
#     xt = [f"{ML(s.split('-')[0])}-{VL(s.split('-')[1])}" if '-' in s else s for s in xt]
#     ax.set_xticklabels(xt, rotation=90)
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(out_png), exist_ok=True)
#     plt.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
#     plt.close()


from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

def benjamini_hochberg(pvals):
    return multipletests(pvals, method='fdr_bh')[1].tolist()

RNG = np.random.default_rng(42)

def compute_bias_mitigation_sig(df: pd.DataFrame, B: int = 1000) -> pd.DataFrame:
    """Compute Δgap, binomial sign test and effect sizes for each model, prompt, type."""
    rows = []
    df_c = df.copy()
    df_c['correct'] = (df_c['model_answer'] == df_c['answer_idx']).astype(int)
    for model in MODEL_ORDER:
        df_m = df_c[df_c['model'] == model]
        if df_m.empty:
            continue
        base = df_m[df_m['prompt'] == 'naive']
        if base.empty:
            continue
        base_pvt = base.pivot_table(index='question_id', columns='variant', values='correct', aggfunc='first')
        for prompt in PROMPT_ORDER:
            if prompt == 'naive':
                continue
            alt = df_m[df_m['prompt'] == prompt]
            if alt.empty:
                continue
            alt_pvt = alt.pivot_table(index='question_id', columns='variant', values='correct', aggfunc='first')
            qids = base_pvt.index.intersection(alt_pvt.index)
            if len(qids) == 0:
                continue
            for typ, (v1, v2) in DISPARITY_PAIRS.items():
                if v1 not in base_pvt.columns or v2 not in base_pvt.columns:
                    continue
                if v1 not in alt_pvt.columns or v2 not in alt_pvt.columns:
                    continue
                b1 = base_pvt.loc[qids, v1].astype(float).values
                b2 = base_pvt.loc[qids, v2].astype(float).values
                a1 = alt_pvt.loc[qids, v1].astype(float).values
                a2 = alt_pvt.loc[qids, v2].astype(float).values
                d_base = b1 - b2
                d_alt = a1 - a2
                gap_base = abs(np.nanmean(d_base))
                gap_alt = abs(np.nanmean(d_alt))
                delta_gap = gap_alt - gap_base
                s = np.abs(d_alt) - np.abs(d_base)
                dec = int((s < 0).sum())
                inc = int((s > 0).sum())
                ties = int((s == 0).sum())
                n_eff = dec + inc
                p_sign = np.nan
                if n_eff > 0:
                    p_sign = binomtest(dec, n=n_eff, p=0.5, alternative='greater').pvalue
                net_reduction = (dec - inc) / n_eff if n_eff > 0 else np.nan
                b_ = inc + 0.5
                c_ = dec + 0.5
                or_mc = c_ / b_
                se = np.sqrt(1.0 / c_ + 1.0 / b_)
                ci_low, ci_high = np.exp(np.log(or_mc) - 1.96 * se), np.exp(np.log(or_mc) + 1.96 * se)
                boot = []
                if len(qids) > 1:
                    idx = np.arange(len(qids))
                    for _ in range(B):
                        samp = RNG.choice(idx, size=len(idx), replace=True)
                        db = d_base[samp]
                        da = d_alt[samp]
                        boot.append(abs(np.nanmean(da)) - abs(np.nanmean(db)))
                    boot = np.array(boot)
                    ci_lo = float(np.nanpercentile(boot, 2.5))
                    ci_hi = float(np.nanpercentile(boot, 97.5))
                else:
                    ci_lo = ci_hi = np.nan
                rows.append({
                    'model': model,
                    'prompt': prompt,
                    'type': typ,
                    'delta_gap': float(delta_gap),
                    'rel_change': (delta_gap / gap_base) if (np.isfinite(delta_gap) and np.isfinite(gap_base) and gap_base != 0) else np.nan,
                    'gap_base': float(gap_base),
                    'gap_alt': float(gap_alt),
                    'dec': dec, 'inc': inc, 'ties': ties, 'n_eff': n_eff,
                    'p_sign': p_sign,
                    'net_reduction_rate': float(net_reduction),
                    'or_mcnemar': float(or_mc),
                    'or_ci_low': float(ci_low), 'or_ci_high': float(ci_high),
                    'delta_gap_ci_low': ci_lo, 'delta_gap_ci_high': ci_hi,
                })
    out = pd.DataFrame(rows)
    if not out.empty and 'p_sign' in out:
        pvals = out['p_sign'].fillna(1.0).values.tolist()
        adj = benjamini_hochberg(pvals)
        out['p_sign_adj'] = adj
    return out


# ----------------------
# Main
# ----------------------

def main():
    data_dir = "."
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    df, acc_df, gap_df, rel_df, change_df, agg_df, p_df = build_tables(data_dir)
    # Save CSV tables
    save_csv(acc_df, os.path.join(output_dir, 'acc_table.csv'))
    save_csv(gap_df, os.path.join(output_dir, 'gap_table.csv'))
    save_csv(rel_df, os.path.join(output_dir, 'relative_gap_change.csv'))
    save_csv(change_df, os.path.join(output_dir, 'answer_change_ratio.csv'))
    save_csv(agg_df, os.path.join(output_dir, 'aggregate_acc_bias.csv'))
    # Significance table (model × prompt-variant)
    variants_order = ["white", "black", "male", "female", "high_income", "low_income"]
    pv_pivot = p_df.pivot_table(index='model', columns='prompt_var', values='p_value')
    ordered_cols = [f"{p}-{v}" for p in ["role", "aware", "cot"] for v in variants_order]
    pv_pivot = pv_pivot.reindex(index=MODEL_ORDER, columns=ordered_cols)
    save_csv(pv_pivot.reset_index(), os.path.join(output_dir, 'sig_table_variant.csv'))
    # Charts directory
    chart_dir = os.path.join(output_dir, 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    chart1_bar_grid(acc_df, os.path.join(chart_dir, 'chart1_accuracy_grid.png'))
    chart2_gap_grid(gap_df, os.path.join(chart_dir, 'chart2_gap_grid.png'))
    chart3_gap_heatmap(rel_df, os.path.join(chart_dir, 'chart3_gap_heatmap.png'))
    chart3_gap_abs_heatmap(rel_df, os.path.join(chart_dir, 'chart3_gap_abs_heatmap.png'))
    chart4_radar(rel_df, os.path.join(chart_dir, 'chart4_radar.png'))
    chart4_gap_radar(gap_df, os.path.join(chart_dir, 'chart4_radar_gap.png'))
    chart5_pareto(agg_df, os.path.join(chart_dir, 'chart5_pareto.png'))
    # chart6_change_bar(change_df, os.path.join(chart_dir, 'chart6_change_ratio.png'))
    bias_sig_df = compute_bias_mitigation_sig(df)
    save_csv(bias_sig_df, os.path.join(output_dir, 'bias_mitigation_sig.csv'))
    print("Done. Tables and charts saved to:", output_dir)


if __name__ == '__main__':
    main()