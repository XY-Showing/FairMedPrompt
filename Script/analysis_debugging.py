#!/usr/bin/env python3
"""
AMQA Fairness Analysis (seaborn edition)
========================================

Reads AMQA result JSONL files named:
    FairMP_Answer_[model]_[prompt].jsonl
(or Fair_Answer_* as fallback)

Builds tidy DataFrames, computes metrics, and renders the following charts
(with seaborn):

Chart 1  Bar grid (2 rows × 15 cols): per-(model × sens. value) accuracies,
         4 bars per subplot for prompts [naive, role, aware, cot].
         Column layout preserves model columns, rows pair values (top: white/male/high_income,
         bottom: black/female/low_income) ordered by attribute type [race, sex, SES].

Chart 2  Bar grid (1 row × 15 cols): per-(model × sens. type) accuracy gaps
         (|white-black|, |male-female|, |high-low|), 4 bars per subplot.

Chart 3  Heatmap: rows=prompts [naive, role, aware, cot], columns=model×sens.type
         values = relative change of gap vs naive: (gap_p - gap_naive)/gap_naive.
         Colormap = RdBu_r (red=larger gap, blue=smaller), centered at 0.

Chart 4  Radar (pentagon): vertices = models [gpt, claude, gemini, qwen, deepseek].
         For each prompt, value at each vertex = relative change of the
         average gap across (race, sex, SES) vs naive for that model.

Chart 5  Pareto scatter: Δaccuracy vs −Δbias (bias=mean gap across race/sex/SES),
         hue=model, style=prompt.

Chart 6  Bar: x=model-sens.value (e.g., gpt-white), y=answer-change rate vs naive,
         hue=prompt (role/aware/cot).

Table    McNemar p-values (rows=models; columns=prompt-variant combos like role-white,
         aware-male, cot-low_income). Compares correctness under prompt vs naive
         for that single variant.

Outputs are saved under --output-dir (default ./outputs). Also writes CSVs
for all intermediate tables.

Author: ChatGPT (seaborn refactor)
"""

from __future__ import annotations

import argparse
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
    "few": "#ABB6E3",
    "cot":   "#1E5AA8",     # deep blue
}

# ----------------------
# IO helpers
# ----------------------

def find_data_files(data_dir: str) -> Dict[Tuple[str, str], str]:
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
        file_map[(model, prompt)] = os.path.join(data_dir, fname)
    return file_map


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize(records: List[Dict[str, Any]], model: str, prompt: str) -> pd.DataFrame:
    data = []
    for r in records:
        qid = r.get('question_id')
        gold = r.get('answer_idx')
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
    return float((series ==  series.index.get_level_values(0)) if False else 0.0)  # placeholder (unused)


def variant_accuracy(df: pd.DataFrame) -> pd.Series:
    out = {}
    for v in VARIANT_ORDER:
        sub = df[df['variant']==v]
        if len(sub)==0:
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
    A = df[df['variant']==a][['question_id','model_answer']].rename(columns={'model_answer':'A'})
    B = df[df['variant']==b][['question_id','model_answer']].rename(columns={'model_answer':'B'})
    m = pd.merge(A,B,on='question_id',how='inner')
    if len(m)==0:
        return np.nan
    return float((m['A']==m['B']).mean())


def mcnemar(baseline: pd.Series, alt: pd.Series) -> float:
    # baseline, alt: boolean correctness aligned per question
    if baseline.shape != alt.shape:
        return np.nan
    b = int(((baseline==True) & (alt==False)).sum())
    c = int(((baseline==False) & (alt==True)).sum())
    if b+c==0:
        stat = 0.0
    else:
        stat = (abs(b-c)-1)**2/(b+c)
    p = 1 - chi2.cdf(stat,1)
    return float(p)

# ----------------------
# Build tidy tables
# ----------------------

def build_tables(data_dir: str):
    file_map = find_data_files(data_dir)
    frames = []
    for (model,prompt), path in sorted(file_map.items()):
        recs = load_jsonl(path)
        frames.append(normalize(recs, model, prompt))
    if not frames:
        raise RuntimeError("No JSONL files found.")
    df = pd.concat(frames, ignore_index=True)

    # Accuracy table per model/prompt/variant
    acc_rows = []
    for (model,prompt,variant), g in df.groupby(['model','prompt','variant']):
        a = float((g['model_answer']==g['answer_idx']).mean())
        acc_rows.append({'model':model,'prompt':prompt,'variant':variant,'accuracy':a})
    acc_df = pd.DataFrame(acc_rows)

    # Gap per model/prompt/type
    gap_rows = []
    for (model,prompt), g in df.groupby(['model','prompt']):
        for typ,(v1,v2) in DISPARITY_PAIRS.items():
            g1 = g[g['variant']==v1]
            g2 = g[g['variant']==v2]
            a1 = float((g1['model_answer']==g1['answer_idx']).mean()) if len(g1)>0 else np.nan
            a2 = float((g2['model_answer']==g2['answer_idx']).mean()) if len(g2)>0 else np.nan
            gap = abs(a1-a2) if (not np.isnan(a1) and not np.isnan(a2)) else np.nan
            gap_rows.append({'model':model,'prompt':prompt,'type':typ,'gap':gap})
    gap_df = pd.DataFrame(gap_rows)

    # Relative change vs naive for gaps

    # Relative change vs naive for gaps
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
                g_base = float(base[base['type'] == typ]['gap'].values[0]) if not base[
                    base['type'] == typ].empty else np.nan
                g_val = float(temp[temp['type'] == typ]['gap'].values[0]) if not temp[
                    temp['type'] == typ].empty else np.nan

                abs_change = (g_val - g_base) if (np.isfinite(g_val) and np.isfinite(g_base)) else np.nan
                rel_change = (abs_change / g_base) if (
                            np.isfinite(abs_change) and np.isfinite(g_base) and g_base != 0) else np.nan

                # ← 这里把绝对变化量也一起存起来
                rel_rows.append({
                    'model': model,
                    'prompt': prompt,
                    'type': typ,
                    'rel_change': rel_change,  # 你原来就用的
                    'abs_change': abs_change,  # 新增：gap 的绝对变化量
                    # （可选）保留基线与当前值，便于排查
                    'gap_base': g_base,
                    'gap_val': g_val,
                })
    rel_df = pd.DataFrame(rel_rows)

    # rel_rows = []
    # for model in MODEL_ORDER:
    #     base = gap_df[(gap_df['model']==model) & (gap_df['prompt']==PROMPT_ORDER[0])]  # naive
    #     if base.empty:
    #         continue
    #     for prompt in PROMPT_ORDER:
    #         temp = gap_df[(gap_df['model']==model) & (gap_df['prompt']==prompt)]
    #         if temp.empty:
    #             continue
    #         for typ in TYPE_ORDER:
    #             g_base = float(base[base['type']==typ]['gap'].values[0]) if not base[base['type']==typ].empty else np.nan
    #             g_val  = float(temp[temp['type']==typ]['gap'].values[0]) if not temp[temp['type']==typ].empty else np.nan
    #             rel = np.nan
    #             if (g_base is not np.nan) and (g_base!=0) and (not np.isnan(g_base)) and (not np.isnan(g_val)):
    #                 rel = (g_val - g_base)/g_base
    #             rel_rows.append({'model':model,'prompt':prompt,'type':typ,'rel_change':rel})
    # rel_df = pd.DataFrame(rel_rows)

    # Per-variant answer change ratio vs naive
    change_rows = []
    # Build per-question tables for each model/prompt/variant
    for model in MODEL_ORDER:
        # naive answers for all variants
        base_df = df[(df['model']==model) & (df['prompt']==PROMPT_ORDER[0])]
        if base_df.empty:
            continue
        for prompt in PROMPT_ORDER[1:]:
            alt_df = df[(df['model']==model) & (df['prompt']==prompt)]
            if alt_df.empty:
                continue
            for var in ["white","black","high_income","low_income","male","female"]:
                A = base_df[base_df['variant']==var][['question_id','model_answer']].rename(columns={'model_answer':'base'})
                B = alt_df[alt_df['variant']==var][['question_id','model_answer']].rename(columns={'model_answer':'alt'})
                m = pd.merge(A,B,on='question_id',how='inner')
                if len(m)==0:
                    ratio = np.nan
                else:
                    ratio = float((m['base']!=m['alt']).mean())
                change_rows.append({'model':model,'prompt':prompt,'variant':var,'change_ratio':ratio})
    change_df = pd.DataFrame(change_rows)

    # Aggregate accuracy & bias for Pareto
    agg_rows = []
    for (model,prompt), g in df.groupby(['model','prompt']):
        # accuracy across 8 variants
        avgs = []
        for v in VARIANT_ORDER:
            gi = g[g['variant']==v]
            if len(gi)>0:
                avgs.append(float((gi['model_answer']==gi['answer_idx']).mean()))
        agg_acc = float(np.nanmean(avgs)) if avgs else np.nan
        # bias = mean gap across (race,sex,SES)
        gaps = []
        for typ,(v1,v2) in DISPARITY_PAIRS.items():
            g1 = g[g['variant']==v1]
            g2 = g[g['variant']==v2]
            if len(g1)>0 and len(g2)>0:
                a1 = float((g1['model_answer']==g1['answer_idx']).mean())
                a2 = float((g2['model_answer']==g2['answer_idx']).mean())
                gaps.append(abs(a1-a2))
        agg_bias = float(np.nanmean(gaps)) if gaps else np.nan
        agg_rows.append({'model':model,'prompt':prompt,'agg_acc':agg_acc,'agg_bias':agg_bias})
    agg_df = pd.DataFrame(agg_rows)

    # McNemar p-values per variant (prompt vs naive)
    p_rows = []
    for model in MODEL_ORDER:
        base = df[(df['model']==model) & (df['prompt']=="naive")]
        if base.empty:
            continue
        for prompt in PROMPT_ORDER[1:]:
            alt  = df[(df['model']==model) & (df['prompt']==prompt)]
            if alt.empty:
                continue
            for var in ["white","black","male","female","high_income","low_income"]:
                A = base[base['variant']==var][['question_id','answer_idx','model_answer']].copy()
                B = alt[ alt['variant']==var][['question_id','answer_idx','model_answer']].copy()
                A['correct'] = (A['model_answer']==A['answer_idx'])
                B['correct'] = (B['model_answer']==B['answer_idx'])
                m = pd.merge(A[['question_id','correct']], B[['question_id','correct']], on='question_id', suffixes=('_base','_alt'))
                if len(m)==0:
                    p = np.nan
                else:
                    p = mcnemar(m['correct_base'].values, m['correct_alt'].values)
                p_rows.append({'model':model, 'prompt_var': f"{prompt}-{var}", 'p_value': p})
    p_df = pd.DataFrame(p_rows)

    return df, acc_df, gap_df, rel_df, change_df, agg_df, p_df

# ----------------------
# Plotting (seaborn)
# ----------------------

def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def chart1_bar_grid(acc_df: pd.DataFrame, out_png: str):
    """2×15 subplots of accuracies per model × value (6 values).
    Columns: for each model (in MODEL_ORDER) we place 3 columns in type order [race, sex, SES].
    Row0 shows (white, male, high_income). Row1 shows (black, female, low_income).
    Each subplot contains 4 bars for prompts.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 15, figsize=(20, 8), sharey=True)
    # Build a helper dict: accuracy per (model,prompt,variant)
    acc_pivot = acc_df.pivot_table(index=['model','prompt'], columns='variant', values='accuracy')
    for col_idx, model in enumerate(MODEL_ORDER):
        for j, typ in enumerate(TYPE_ORDER):
            # column position across 15: model_index*3 + j
            c = col_idx*3 + j
            ax_top = axes[0, c]
            ax_bot = axes[1, c]
            top_val = PAIR_TOP[typ]
            bot_val = PAIR_BOT[typ]
            # Build data for top/bot subplots
            def build_df(variant):
                rows = []
                for prompt in PROMPT_ORDER:
                    try:
                        val = float(acc_pivot.loc[(model,prompt)][variant])
                    except KeyError:
                        val = np.nan
                    rows.append({'prompt':prompt,'accuracy':val})
                return pd.DataFrame(rows)
            dt = build_df(top_val)
            db = build_df(bot_val)
            sns.barplot(data=dt, x='prompt', y='accuracy', ax=ax_top, hue='prompt', hue_order=PROMPT_ORDER, width=0.6, dodge=False,
                        palette=[PROMPT_COLORS[p] for p in PROMPT_ORDER])
            for c in ax_top.containers:
                ax_top.bar_label(c, fmt="%.2f", padding=1, rotation=90, fontsize=8)
            sns.barplot(data=db, x='prompt', y='accuracy', ax=ax_bot, hue='prompt', hue_order=PROMPT_ORDER, width=0.6, dodge=False,
                        palette=[PROMPT_COLORS[p] for p in PROMPT_ORDER])
            for c in ax_bot.containers:
                ax_bot.bar_label(c, fmt="%.2f", padding=1, rotation=90, fontsize=8)
            ax_top.set_title(f"{model}-{top_val}", fontsize=10, y=-0.25, rotation=20)
            ax_bot.set_title(f"{model}-{bot_val}", fontsize=10, y=-0.25, rotation=20)
            ax_top.set_xlabel("")
            ax_bot.set_xlabel("")
            ax_top.set_xticklabels([])
            ax_bot.set_xticklabels([])
            # ax_bot.set_xticklabels(PROMPT_ORDER, rotation=90)
            ax_top.set_ylim(0.8,1)
            ax_bot.set_ylim(0.6,1)
            # 统一调整每个子图的网格线和边框样式（更细、更浅、虚线）
            for ax in axes.flat:
                # 只保留水平网格线，更细、更浅、虚线
                ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='0.85', alpha=0.8)
                ax.xaxis.grid(False)

                # 刻度（ticks）本身不能设置成虚线，但可以变细、变短、变浅
                ax.tick_params(axis='both', width=0.6, length=2, colors='0.5')

                # 四周边框也淡一点（可选）
                for spine in ax.spines.values():
                    spine.set_alpha(0.3)

    # fig.suptitle("Chart 1: Accuracy per model × sensitive value (bars=prompts)", fontsize=14)
    from matplotlib.patches import Patch
    handles = [Patch(color=PROMPT_COLORS[p], label=p) for p in PROMPT_ORDER]
    fig.legend(
        handles, [p for p in PROMPT_ORDER],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        frameon=False
    )
    # handles, labels = ax_top.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=4)
    fig.tight_layout(rect=[0,0,1,0.95])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def chart2_gap_grid(gap_df: pd.DataFrame, out_png: str):
    """1×15 subplots: per (model × type) accuracy gaps, formatted exactly like Chart 1."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 15, figsize=(20, 3.8), sharey=True)

    gap_pivot = gap_df.pivot_table(index=['model','prompt'], columns='type', values='gap')

    for col_idx, model in enumerate(MODEL_ORDER):
        for j, typ in enumerate(TYPE_ORDER):
            c = col_idx*3 + j
            ax = axes[c]

            # 构造当前子图数据（四个 prompt）
            rows = []
            for prompt in PROMPT_ORDER:
                try:
                    val = float(gap_pivot.loc[(model, prompt)][typ])
                except KeyError:
                    val = np.nan
                rows.append({'prompt': prompt, 'gap': val})
            df_plot = pd.DataFrame(rows)

            # 与图1相同：hue='prompt' + dodge=False + 指定颜色，柱宽 0.6
            sns.barplot(
                data=df_plot, x='prompt', y='gap', ax=ax,
                hue='prompt', hue_order=PROMPT_ORDER,
                palette={p: PROMPT_COLORS[p] for p in PROMPT_ORDER},
                dodge=False, width=0.6
            )
            # 每根柱子数值，竖排
            for cont in ax.containers:
                ax.bar_label(cont, fmt="%.2f", padding=1, rotation=90, fontsize=8)

            # 子图名放在子图下方，去掉 x 轴的 prompt 标签
            ax.set_title(f"{model}-{typ}", fontsize=10, y=-0.25, rotation=00)
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_xticklabels([])

            # 统一 y 轴范围（根据你的原始设置）
            ax.set_ylim(0, 0.25)

            # 子图自身 legend 去除（使用全局 legend）
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    # 与图1一致的网格/刻度/边框样式
    for ax in axes.flat:
        ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='0.85', alpha=0.8)
        ax.xaxis.grid(False)
        ax.tick_params(axis='both', width=0.6, length=2, colors='0.5')
        for spine in ax.spines.values():
            spine.set_alpha(0.3)

    # 标题 + 全局 legend（放在标题下方居中）
    # fig.suptitle("Chart 2: Accuracy gaps per model × attribute type", fontsize=14)

    from matplotlib.patches import Patch
    handles = [Patch(color=PROMPT_COLORS[p], label=p) for p in PROMPT_ORDER]
    fig.legend(
        handles, [p for p in PROMPT_ORDER],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.92),  # 位置与图1一致
        ncol=5,
        frameon=False
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)



def chart3_gap_heatmap(rel_df: pd.DataFrame, out_png: str):
    """Heatmap of relative change vs naive for gaps (rows=prompts; cols=model×type)."""
    sns.set_theme(style="white")
    # Build matrix 4×15 in requested order
    cols = []
    for model in MODEL_ORDER:
        for typ in TYPE_ORDER:
            cols.append(f"{model}-{typ}")
    # pivot
    mat = np.full((len(PROMPT_ORDER), len(cols)), np.nan)
    for i,prompt in enumerate(PROMPT_ORDER):
        for j,model in enumerate(MODEL_ORDER):
            for k,typ in enumerate(TYPE_ORDER):
                col_idx = j*3 + k
                temp = rel_df[(rel_df['model']==model) & (rel_df['prompt']==prompt) & (rel_df['type']==typ)]
                if not temp.empty:
                    mat[i, col_idx] = float(temp['rel_change'].values[0])
    plt.figure(figsize=(24, 4))
    ax = sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                     vmin=-0.5, vmax=0.5, cbar_kws={'label':'Relative change vs naive'})
    ax.set_yticks(np.arange(len(PROMPT_ORDER)) + 0.5)
    ax.set_yticklabels(PROMPT_ORDER, rotation=0)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols, rotation=45)
    # ax.set_title("Chart 3: Relative gap change vs naive (prompt × model-attribute)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def chart3_gap_abs_heatmap(rel_df: pd.DataFrame, out_png: str):
    """Heatmap of absolute GAP change vs naive: use rel_df['abs_change']"""
    sns.set_theme(style="white")
    cols = [f"{m}-{t}" for m in MODEL_ORDER for t in TYPE_ORDER]
    mat = np.full((len(PROMPT_ORDER), len(cols)), np.nan)

    for i, prompt in enumerate(PROMPT_ORDER):
        for j, model in enumerate(MODEL_ORDER):
            for k, typ in enumerate(TYPE_ORDER):
                col_idx = j*3 + k
                tmp = rel_df[(rel_df['model']==model) & (rel_df['prompt']==prompt) & (rel_df['type']==typ)]
                if not tmp.empty:
                    mat[i, col_idx] = float(tmp['abs_change'].values[0])

    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax < 0.01:
        vmax = 0.01

    plt.figure(figsize=(24, 4))
    ax = sns.heatmap(
        mat, mask=np.isnan(mat),
        annot=True, fmt=".3f",
        cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        cbar_kws={'label': 'Δ GAP (prompt − naive)'}
    )
    ax.set_yticks(np.arange(len(PROMPT_ORDER)) + 0.5)
    ax.set_yticklabels(PROMPT_ORDER, rotation=0)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols, rotation=45)
    # ax.set_title("Chart 3 (abs): Absolute GAP change vs naive (prompt × model-attribute)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()



# def chart4_radar(rel_df: pd.DataFrame, out_png: str):
#     """Radar (pentagon): vertex per model; value = relative change of mean gap (race,sex,SES)."""
#     sns.set_theme(style="white")
#     models = MODEL_ORDER
#     angles = np.linspace(0, 2*np.pi, len(models), endpoint=False).tolist()
#     angles += angles[:1]
#
#     plt.figure(figsize=(6,6))
#     ax = plt.subplot(111, polar=True)
#     ax.set_theta_offset(np.pi/2)
#     ax.set_theta_direction(-1)
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(models)
#     ax.set_yticklabels([])
#     ax.set_ylim(-1, 1)
#
#     for prompt in PROMPT_ORDER:
#         if prompt=="naive":
#             continue
#         vals = []
#         for m in models:
#             tmp = rel_df[(rel_df['model']==m) & (rel_df['prompt']==prompt) & (rel_df['type'].isin(TYPE_ORDER))]
#             # average of rel_change over the 3 types for that model & prompt
#             vals.append(float(tmp['rel_change'].mean())) if not tmp.empty else vals.append(np.nan)
#         vals += vals[:1]
#         ax.plot(angles, vals, label=prompt, color=PROMPT_COLORS[prompt])
#         ax.fill(angles, vals, alpha=0.08, color=PROMPT_COLORS[prompt])
#
#     ax.set_title("Chart 4: Radar – relative mean gap change vs naive")
#     ax.legend(loc='upper right', bbox_to_anchor=(1.2,1.1))
#     os.makedirs(os.path.dirname(out_png), exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=200)
#     plt.close()


def chart4_radar(rel_df: pd.DataFrame, out_png: str):
    """Radar (pentagon): vertex per model; value = relative change of mean gap (race,sex,SES)."""
    sns.set_theme(style="white")
    models = MODEL_ORDER
    angles = np.linspace(0, 2*np.pi, len(models), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(models)

    # 设置半径刻度
    ax.set_ylim(0, -0.5)   # 保证对称
    yticks = [0, -0.1, -0.2, -0.3, -0.4, -0.5]   # 你需要的刻度
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks], fontsize=8)  # 可以调整字号
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='0.7')  # 刻度环虚线
    ax.axhline(0, color='gray', linewidth=0.8)  # 加一条 0 基准线

    for prompt in PROMPT_ORDER:
        if prompt=="naive":
            continue
        vals = []
        for m in models:
            tmp = rel_df[(rel_df['model']==m) & (rel_df['prompt']==prompt) & (rel_df['type'].isin(TYPE_ORDER))]
            vals.append(float(tmp['rel_change'].mean())) if not tmp.empty else vals.append(np.nan)
        vals += vals[:1]
        ax.plot(angles, vals, label=prompt, color=PROMPT_COLORS[prompt])
        ax.fill(angles, vals, alpha=0.08, color=PROMPT_COLORS[prompt])

    # ax.set_title("Chart 4: Radar – relative mean gap change vs naive")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2,1.1))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()



def chart4_gap_radar(gap_df: pd.DataFrame, out_png: str):
    """
    Radar of *mean accuracy gap* (smaller is better).
    Vertex order = MODEL_ORDER. For each prompt, value = mean gap across TYPE_ORDER.
    """
    sns.set_theme(style="white")

    # ---- prepare data: mean gap per (model, prompt) across 3 types ----
    means_by_prompt = {}
    vmax = 0.0
    for prompt in PROMPT_ORDER:   # 如需排除 naive，改为 PROMPT_ORDER[1:]
        vals = []
        for m in MODEL_ORDER:
            g = gap_df[(gap_df['model'] == m) &
                       (gap_df['prompt'] == prompt) &
                       (gap_df['type'].isin(TYPE_ORDER))]
            v = float(g['gap'].mean()) if not g.empty else np.nan
            vals.append(v)
        means_by_prompt[prompt] = vals
        vmax = max(vmax, np.nanmax(vals) if np.any(np.isfinite(vals)) else 0.0)

    # 合理的径向上限与刻度（到最近的 0.05）
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.25
    rmax = float(np.ceil((vmax * 1.15) / 0.05) * 0.05)
    rticks = np.arange(0, rmax + 1e-9, 0.05)  # 0.00,0.05,...

    # ---- polar chart ----
    N = len(MODEL_ORDER)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'polar': True})

    # 轴与网格
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(MODEL_ORDER, fontsize=12)
    ax.set_ylim(0, rmax)
    ax.set_rlabel_position(90)
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{t:.2f}" if t > 0 else "0" for t in rticks], fontsize=10, color='0.35')
    ax.grid(True, linestyle="--", linewidth=0.7, color='0.85')

    # 画 0 参考线（径向）
    ax.plot([np.pi/2, np.pi/2], [0, rmax], ls=':', lw=1.0, color='0.7')

    # ---- draw each prompt ----
    for prompt in PROMPT_ORDER:  # 如需排除 naive，改为 PROMPT_ORDER[1:]
        vals = means_by_prompt[prompt]
        if not np.any(np.isfinite(vals)):
            continue
        r = np.array(vals, dtype=float)
        r = np.nan_to_num(r, nan=0.0)
        r = np.r_[r, r[0]]  # close the loop

        color = PROMPT_COLORS.get(prompt, None)
        ax.plot(angles, r, label=prompt, color=color, linewidth=2.2, marker='o', markersize=3.5)
        ax.fill(angles, r, color=color, alpha=0.10)

    # 标题
    # ax.set_title("Chart 4 (gap): mean accuracy gap by prompt (smaller is better)", pad=18, fontsize=14)

    # ---- legend（修复 .spines 报错：直接改 frame 本身）----
    leg = ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1.10),
                    frameon=True, framealpha=0.95, facecolor='white',
                    edgecolor='0.85', fontsize=11)
    frame = leg.get_frame()              # FancyBboxPatch
    frame.set_linewidth(0.8)
    frame.set_edgecolor('0.85')
    frame.set_facecolor('white')
    frame.set_alpha(0.95)
    if hasattr(frame, "set_boxstyle"):
        frame.set_boxstyle("round,pad=0.3,rounding_size=2.0")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)





def chart5_pareto(agg_df: pd.DataFrame, out_png: str):
    """Δaccuracy vs −Δbias scatter. hue=model, style=prompt (excl. naive)."""
    sns.set_theme(style="whitegrid")
    base = agg_df[agg_df['prompt']=="naive"].set_index('model')
    rows = []
    for _,r in agg_df.iterrows():
        if r['prompt']=="naive":
            continue
        if r['model'] not in base.index:
            continue
        d_acc = r['agg_acc'] - float(base.loc[r['model'],'agg_acc'])
        d_bias = r['agg_bias'] - float(base.loc[r['model'],'agg_bias'])
        rows.append({'model':r['model'],'prompt':r['prompt'],'d_acc':d_acc,'neg_d_bias':-d_bias})
    dfp = pd.DataFrame(rows)
    plt.figure(figsize=(6,4))
    ax = sns.scatterplot(data=dfp, x='d_acc', y='neg_d_bias', hue='model', style='prompt')
    ax.axhline(0, ls='--', lw=0.5, color='gray')
    ax.axvline(0, ls='--', lw=0.5, color='gray')
    # ax.set_title("Chart 5: Pareto – ΔAccuracy vs −ΔBias (vs naive)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def chart6_change_bar(change_df: pd.DataFrame, out_png: str):
    """Bar plot: x=model-variant value, y=answer-change ratio vs naive, hue=prompt (role/aware/cot)."""
    sns.set_theme(style="whitegrid")
    # Build x labels in requested order: per model, values (white, black, male, female, high_income, low_income)
    order_vals = ["white","black","male","female","high_income","low_income"]
    change_df = change_df.copy()
    change_df['cat'] = change_df['model'] + '-' + change_df['variant']
    # sort by MODEL_ORDER then by the order_vals
    change_df['model'] = pd.Categorical(change_df['model'], categories=MODEL_ORDER, ordered=True)
    change_df['variant'] = pd.Categorical(change_df['variant'], categories=order_vals, ordered=True)
    change_df = change_df.sort_values(['model','variant','prompt'])
    plt.figure(figsize=(18,4))
    ax = sns.barplot(data=change_df, x='cat', y='change_ratio', hue='prompt', hue_order=["role","aware","cot"],
                     palette=[PROMPT_COLORS[p] for p in ["role","aware","cot"]])
    # ax.set_title("Chart 6: Answer change ratio vs naive (by model-variant)")
    ax.set_xlabel("model-variant")
    ax.set_ylabel("change ratio")
    ax.set_ylim(0,1)
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
def benjamini_hochberg(pvals):
    return multipletests(pvals, method='fdr_bh')[1].tolist()

RNG = np.random.default_rng(42)

def compute_bias_mitigation_sig(df: pd.DataFrame, B: int = 1000) -> pd.DataFrame:
    """
    对每个 model × prompt(≠naive) × type(race/sex/SES)：
      - 计算 Δgap = gap(prompt) - gap(naive)
      - 95% bootstrap CI for Δgap（按 question_id 重采样）
      - 符号检验：在发生变化的题里，减少差异（s_i<0）是否多于增加（s_i>0）
      - 效应量：net_reduction_rate、OR 及其 95% CI
    返回 tidy 表：一行一个 (model, prompt, type)
    """
    rows = []
    # 先把 correctness 表铺好：每行一个 (model,prompt,question_id,variant,correct)
    df_c = df.copy()
    df_c['correct'] = (df_c['model_answer'] == df_c['answer_idx']).astype(int)

    for model in MODEL_ORDER:
        df_m = df_c[df_c['model'] == model]
        if df_m.empty:
            continue

        # 以 naive 为基线的题集（确保对齐）
        base = df_m[df_m['prompt'] == 'naive']
        if base.empty:
            continue
        # 做成 question × variant 的宽表（只要我们关心的 pair 里的两个变体）
        base_pvt = base.pivot_table(index='question_id', columns='variant', values='correct', aggfunc='first')

        for prompt in PROMPT_ORDER:
            if prompt == 'naive':
                continue
            alt = df_m[df_m['prompt'] == prompt]
            if alt.empty:
                continue
            alt_pvt = alt.pivot_table(index='question_id', columns='variant', values='correct', aggfunc='first')

            # 对齐题目集合（交集）
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

                # d = correctness 差（-1/0/1），gap = |mean d|
                d_base = b1 - b2
                d_alt  = a1 - a2
                gap_base = abs(np.nanmean(d_base))
                gap_alt  = abs(np.nanmean(d_alt))
                delta_gap = gap_alt - gap_base   # 负值=缓解

                # 题级变化 s_i
                s = np.abs(d_alt) - np.abs(d_base)
                dec = int((s < 0).sum())   # 差异减少
                inc = int((s > 0).sum())   # 差异增加
                ties = int((s == 0).sum())
                n_eff = dec + inc

                # 符号检验（单侧：是否“减少”更多？）
                p_sign = np.nan
                if n_eff > 0:
                    p_sign = binomtest(dec, n=n_eff, p=0.5, alternative='greater').pvalue

                # 效应量
                net_reduction = (dec - inc) / n_eff if n_eff > 0 else np.nan
                # OR with Haldane–Anscombe
                b_ = inc + 0.5   # “不利”（差异增加）当作对照
                c_ = dec + 0.5   # “有利”（差异减少）当作事件
                or_mc = c_ / b_
                se = np.sqrt(1.0/c_ + 1.0/b_)
                ci_low, ci_high = np.exp(np.log(or_mc) - 1.96*se), np.exp(np.log(or_mc) + 1.96*se)

                # Bootstrap CI for Δgap（对 question_id 重采样）
                boot = []
                if len(qids) > 1:
                    idx = np.arange(len(qids))
                    for _ in range(B):
                        samp = RNG.choice(idx, size=len(idx), replace=True)
                        db = d_base[samp]; da = d_alt[samp]
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
                    'delta_gap': float(delta_gap),           # 负=缓解
                    'rel_change': (delta_gap / gap_base) if (np.isfinite(delta_gap) and np.isfinite(gap_base) and gap_base!=0) else np.nan,
                    'gap_base': float(gap_base),
                    'gap_alt':  float(gap_alt),
                    'dec': dec, 'inc': inc, 'ties': ties, 'n_eff': n_eff,
                    'p_sign': p_sign,
                    'net_reduction_rate': float(net_reduction),
                    'or_mcnemar': float(or_mc),
                    'or_ci_low': float(ci_low), 'or_ci_high': float(ci_high),
                    'delta_gap_ci_low': ci_lo, 'delta_gap_ci_high': ci_hi,
                })
    out = pd.DataFrame(rows)

    # FDR（对所有 p_sign 调整）
    if not out.empty and 'p_sign' in out:
        pvals = out['p_sign'].fillna(1.0).values.tolist()
        adj = benjamini_hochberg(pvals)
        out['p_sign_adj'] = adj
    return out


# ----------------------
# Main
# ----------------------
def main():
    # 固定当前目录为数据目录，输出到本目录下的新子文件夹 ./outputs
    data_dir = "."
    output_dir = "./outputs"

    os.makedirs(output_dir, exist_ok=True)

    # 读取与构建各类表
    df, acc_df, gap_df, rel_df, change_df, agg_df, p_df = build_tables(data_dir)

    # 保存表格
    save_csv(acc_df,   os.path.join(output_dir, 'acc_table.csv'))
    save_csv(gap_df,   os.path.join(output_dir, 'gap_table.csv'))
    save_csv(rel_df,   os.path.join(output_dir, 'relative_gap_change.csv'))
    save_csv(change_df,os.path.join(output_dir, 'answer_change_ratio.csv'))
    save_csv(agg_df,   os.path.join(output_dir, 'aggregate_acc_bias.csv'))

    # 显著性表（模型为行；列为 prompt×variant 共 18 列）
    variants_order = ["white","black","male","female","high_income","low_income"]
    pv_pivot = p_df.pivot_table(index='model', columns='prompt_var', values='p_value')
    ordered_cols = [f"{p}-{v}" for p in ["role","aware","cot"] for v in variants_order]
    pv_pivot = pv_pivot.reindex(index=MODEL_ORDER, columns=ordered_cols)
    save_csv(pv_pivot.reset_index(), os.path.join(output_dir, 'sig_table_variant.csv'))

    # 图表输出到 ./outputs/charts
    chart_dir = os.path.join(output_dir, 'charts')
    os.makedirs(chart_dir, exist_ok=True)

    chart1_bar_grid(acc_df, os.path.join(chart_dir, 'chart1_accuracy_grid.png'))
    chart2_gap_grid(gap_df, os.path.join(chart_dir, 'chart2_gap_grid.png'))
    chart3_gap_heatmap(rel_df, os.path.join(chart_dir, 'chart3_gap_heatmap.png'))
    chart3_gap_abs_heatmap(rel_df, os.path.join(chart_dir, 'chart3_gap_abs_heatmap.png'))
    chart4_radar(rel_df, os.path.join(chart_dir, 'chart4_radar.png'))
    # 用绝对 GAP 版（包含 naive 作为对照）
    chart4_gap_radar(gap_df, os.path.join(chart_dir, 'chart4_radar_gap.png'))

    chart5_pareto(agg_df, os.path.join(chart_dir, 'chart5_pareto.png'))
    chart6_change_bar(change_df, os.path.join(chart_dir, 'chart6_change_ratio.png'))



    bias_sig_df = compute_bias_mitigation_sig(df)
    save_csv(bias_sig_df, os.path.join(output_dir, 'bias_mitigation_sig.csv'))

    print("Done. Tables and charts saved to:", output_dir)


if __name__ == '__main__':
    main()
