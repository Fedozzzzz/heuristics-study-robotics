 # -*- coding: utf-8 -*-
"""Строит диаграмму Ганта расписания выполнения задач доставки (прямая эвристика)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scenario_generator import build_environment, build_cargos_and_pairs
from priority_evaluation import (estimate_task_costs, compute_direct_priority,
                                  run_sequential_by_priority)
from lpt_assignment import assign_by_lpt, apply_assignment

# --- более сложный пример ---
N_ISLANDS, N_PAIRS, N_CARGOS, SEED, L = 25, 6, 18, 0, 6.0

env = build_environment(n_islands=N_ISLANDS, seed=SEED)
cargos, pairs = build_cargos_and_pairs(n_islands=N_ISLANDS, n_pairs=N_PAIRS,
                                        n_cargos=N_CARGOS, seed=SEED)

# --- НАЗНАЧЕНИЕ ГРУЗОВ ПАРАМ по LPT (вместо round-robin из генератора) ---
assignment = assign_by_lpt(cargos, pairs, env, L=L)
cargos = apply_assignment(cargos, assignment)

task_costs = estimate_task_costs(env, cargos, pairs, L=L)
priority = compute_direct_priority(task_costs)
outcome = run_sequential_by_priority(env, cargos, pairs, L=L,
                                      priority=priority, task_costs=task_costs)

# --- раскладываем задачи по парам, накапливая время (duration) ---
pair_ids = sorted({e.pair_id for e in outcome.entries})
pair_row = {pid: i for i, pid in enumerate(pair_ids)}
pair_time = {pid: 0.0 for pid in pair_ids}   # текущее время освобождения пары

# порядок выполнения = порядок в entries (по убыванию приоритета)
blocks = []  # (row, start, duration, cargo_id, Wd, Wb, order)
for order, e in enumerate(outcome.entries, start=1):
    pid = e.pair_id
    start = pair_time[pid]
    dur = e.result.duration
    blocks.append((pair_row[pid], start, dur, e.cargo_id, e.result.W_d,
                   e.result.W_b, order))
    pair_time[pid] += dur

# --- рисуем ---
fig, ax = plt.subplots(figsize=(15, 9))

COL_BUILD = "#2b6cb0"   # строитель (W_b)
COL_DELIV = "#e67e22"   # доставщик (W_d)

bar_h = 0.55
for row, start, dur, cid, wd, wb, order in blocks:
    total_wdb = wd + wb if (wd + wb) > 0 else 1.0
    # ширина блока = duration; делим её пропорционально Wb : Wd для наглядности
    w_build_len = dur * (wb / total_wdb)
    w_deliv_len = dur * (wd / total_wdb)

    # часть строителя (W_b)
    if wb > 0:
        ax.broken_barh([(start, w_build_len)], (row - bar_h/2, bar_h),
                       facecolors=COL_BUILD, edgecolors="white", linewidth=0.5,
                       alpha=0.85, zorder=2)
    # часть доставщика (W_d)
    ax.broken_barh([(start + w_build_len, w_deliv_len)], (row - bar_h/2, bar_h),
                   facecolors=COL_DELIV, edgecolors="white", linewidth=0.5,
                   alpha=0.85, zorder=2)
    # рамка всего блока
    ax.broken_barh([(start, dur)], (row - bar_h/2, bar_h),
                   facecolors="none", edgecolors="#2c3e50", linewidth=1.6, zorder=3)

    # подпись груза и порядок выполнения
    ax.text(start + dur/2, row + 0.02, f"{cid}  (#{order})",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="white", zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc="#2c3e50", ec="none", alpha=0.75))
    # мини-разбивка под блоком
    ax.text(start + dur/2, row - bar_h/2 - 0.13,
            f"W_d={wd:.1f}  W_b={wb:.1f}  |  dur={dur:.1f}",
            ha="center", va="top", fontsize=7.5, color="#555", zorder=4)

# оформление осей
ax.set_yticks(range(len(pair_ids)))
ax.set_yticklabels([f"Пара {pid[-1]}" for pid in pair_ids], fontsize=11)
ax.set_ylim(-0.85, len(pair_ids) - 0.15)
ax.set_xlabel("Условное время (накопленная duration)", fontsize=10)
makespan = max(pair_time.values())
ax.set_xlim(-0.3, makespan * 1.08)
ax.set_title(
    f"Расписание доставки: LPT-назначение грузов парам + прямая эвристика порядка\n"
    f"{N_ISLANDS} островов, {N_PAIRS} пары, {N_CARGOS} грузов",
    fontsize=12, fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# линия makespan
ax.axvline(makespan, color="#c0392b", linestyle=":", linewidth=1.5, zorder=1)
ax.text(makespan, len(pair_ids) - 0.25, f" makespan = {makespan:.1f}",
        color="#c0392b", fontsize=9, va="top", ha="left", fontweight="bold")

# легенда
legend_elems = [
    Patch(facecolor=COL_BUILD, alpha=0.85, label="W_b — работа строителя (мосты + переезды)"),
    Patch(facecolor=COL_DELIV, alpha=0.85, label="W_d — перемещения доставщика"),
]
ax.legend(handles=legend_elems, loc="upper right", fontsize=9, framealpha=0.95)
info = (f"Всего доставлено: {len(blocks)} грузов   |   "
        f"Makespan (посл. пара): {makespan:.1f}   |   "
        f"Σ реальная стоимость: {outcome.real_total:.1f}")
fig.text(0.5, 0.02, info, ha="center", fontsize=9, color="#333")

fig.tight_layout(rect=[0, 0.05, 1, 1])
out = "/home/claude/repo/repo_final/experiments/outputs/schedule_gantt.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved:", out)

# печать текстового расписания
print("\nПорядок выполнения (по убыванию приоритета):")
for row, start, dur, cid, wd, wb, order in blocks:
    print(f"  #{order} {cid} на {pair_ids[row]}: старт={start:.1f} длит={dur:.1f} "
          f"(Wd={wd:.1f}, Wb={wb:.1f})")
