# -*- coding: utf-8 -*-
"""
Сравнение ДИНАМИЧЕСКОГО циклического расписания (Алгоритм 5,
core/algorithm_5_dynamic_rounds.run_dynamic_rounds) со СТАТИЧЕСКИМИ
комбинациями "назначение + эвристика приоритета"
(experiments/evaluate_priority_heuristics.py) по итоговой суммарной
реальной стоимости real_total = Φ.

СТАТИКА (для всех вариантов ниже):
  - назначение груз -> пара фиксируется ОДИН РАЗ, ДО запуска модели
    (lpt_assignment.assign_by_lpt / assign_greedy_nearest);
  - приоритет внутри общей очереди считается ОДИН РАЗ, от НАЧАЛЬНЫХ
    позиций пар (priority_evaluation.estimate_task_costs), и больше не
    пересчитывается по ходу выполнения.

ДИНАМИКА (Алгоритм 5):
  - на каждом раунде выбирается n = |U| доставок (по одной на пару) ЗАНОВО,
    от ТЕКУЩИХ позиций пар - ни назначение груз->пара, ни порядок не
    фиксируются заранее (см. core/algorithm_5_dynamic_rounds.py).

Обе стороны используют ОДНУ И ТУ ЖЕ функцию поиска маршрута
(heuristic_cheapest_bridge.find_route_cheapest_bridge), чтобы разница в
итоговой стоимости объяснялась ТОЛЬКО схемой планирования (статика против
динамики), а не разными алгоритмами построения пути.

Метрика: real_total (Φ) - чем МЕНЬШЕ, тем лучше. Строятся:
  1) График среднего real_total по числу грузов (абсолютный).
  2) График отклонения от лучшего варианта в каждой точке (%).
  3) Win-rate и среднее превышение над лучшим - как в
     evaluate_priority_heuristics.py.

Данные кэшируются (дозапуск при прерывании).

ЗАПУСК: python3 compare_dynamic_rounds.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))

import matplotlib.pyplot as plt

from scenario_generator import build_environment, build_cargos_and_pairs
from priority_evaluation import (estimate_task_costs, compute_ratio_priority,
                                  compute_inverse_priority, run_sequential_by_priority)
from lpt_assignment import assign_by_lpt, assign_greedy_nearest, apply_assignment
from heuristic_cheapest_bridge import find_route_cheapest_bridge
from algorithm_5_dynamic_rounds import run_dynamic_rounds

# ---- Параметры (согласованы с evaluate_priority_heuristics.py) --------------
N_ISLANDS    = 20
N_PAIRS      = 4
N_SEEDS      = 30
N_CARGOS_MAX = 40
L            = 6.0

OUT_DIR    = "outputs"
CACHE_FILE = os.path.join(OUT_DIR, "dynamic_vs_static_cache.json")

# каждая запись: (ключ, отображаемое имя, цвет)
VARIANTS = {
    "lpt_ratio":     ("Статика: LPT + p=W_b/W_d",   "#27ae60"),
    "lpt_inverse":   ("Статика: LPT + p=1/W_T",     "#2b6cb0"),
    "greedy_ratio":  ("Статика: Greedy + p=W_b/W_d", "#8e44ad"),
    "dynamic":       ("Динамика: Алгоритм 5 (nearest)", "#c0392b"),
    "dynamic_lpt":   ("Динамика: Алгоритм 5 (LPT-гибрид)", "#d4ac0d"),
}
# ------------------------------------------------------------------------------


def run_static(assignment_kind, priority_kind, env, cargos, pairs):
    """Один статический прогон: назначение ОДИН РАЗ + приоритет ОДИН РАЗ от
    начальных позиций, затем реальное последовательное выполнение."""
    if assignment_kind == "lpt":
        assignment = assign_by_lpt(cargos, pairs, env, L=L)
    elif assignment_kind == "greedy":
        assignment = assign_greedy_nearest(cargos, pairs, env, L=L)
    else:
        raise ValueError(assignment_kind)
    cargos = apply_assignment(cargos, assignment)

    task_costs = estimate_task_costs(env, cargos, pairs, L=L)
    if priority_kind == "ratio":
        priority = compute_ratio_priority(task_costs)
    elif priority_kind == "inverse":
        priority = compute_inverse_priority(task_costs)
    else:
        raise ValueError(priority_kind)

    outcome = run_sequential_by_priority(env, cargos, pairs, L=L,
                                          priority=priority, task_costs=task_costs)
    if not outcome.all_delivered:
        return None
    return outcome.real_total


def run_dynamic(env, cargos, pairs, selection_strategy="nearest"):
    """Один динамический прогон: Алгоритм 5, полностью пересчитываемый по
    раундам. cargo.assigned_pair игнорируется - route_fn совпадает со
    статикой (find_route_cheapest_bridge), чтобы сравнение было "на равных".

    selection_strategy: "nearest" (жадное по минимальной стоимости раунда)
    или "lpt" (гибрид с LPT - дорогие грузы первыми, минимальная
    накопленная нагрузка пары) - см. core/algorithm_5_dynamic_rounds.py."""
    outcome = run_dynamic_rounds(env, cargos, pairs, L=L,
                                  route_fn=find_route_cheapest_bridge,
                                  selection_strategy=selection_strategy)
    if not outcome.all_delivered:
        return None
    return outcome.real_total


def run_one(key, n_cargos, seed):
    env = build_environment(n_islands=N_ISLANDS, seed=seed)
    cargos, pairs = build_cargos_and_pairs(n_islands=N_ISLANDS, n_pairs=N_PAIRS,
                                            n_cargos=n_cargos, seed=seed)
    if key == "lpt_ratio":
        return run_static("lpt", "ratio", env, cargos, pairs)
    if key == "lpt_inverse":
        return run_static("lpt", "inverse", env, cargos, pairs)
    if key == "greedy_ratio":
        return run_static("greedy", "ratio", env, cargos, pairs)
    if key == "dynamic":
        return run_dynamic(env, cargos, pairs, selection_strategy="nearest")
    if key == "dynamic_lpt":
        return run_dynamic(env, cargos, pairs, selection_strategy="lpt")
    raise ValueError(key)


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def sweep():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = load_cache()

    cargo_range = list(range(N_PAIRS, N_CARGOS_MAX + 1))
    results = {k: [] for k in VARIANTS}
    win_counts = {k: 0 for k in VARIANTS}
    n_comparisons = 0

    for n_cargos in cargo_range:
        per_seed = {k: [] for k in VARIANTS}

        for k in VARIANTS:
            ckey = f"{k}_{n_cargos}"
            if ckey in cache:
                per_seed[k] = cache[ckey]
            else:
                vals = []
                for seed in range(N_SEEDS):
                    rt = run_one(k, n_cargos, seed)
                    vals.append(rt)
                cache[ckey] = vals
                save_cache(cache)
                per_seed[k] = vals

        for k in VARIANTS:
            valid = [v for v in per_seed[k] if v is not None]
            results[k].append(sum(valid) / len(valid) if valid else None)

        for si in range(N_SEEDS):
            row = {k: per_seed[k][si] for k in VARIANTS
                   if per_seed[k][si] is not None}
            if len(row) < len(VARIANTS):
                continue
            n_comparisons += 1
            best_k = min(row, key=row.get)
            win_counts[best_k] += 1

        line = "  ".join(
            f"{k}={results[k][-1]:.1f}" if results[k][-1] is not None else f"{k}=N/A"
            for k in VARIANTS)
        print(f"n={n_cargos:3d}: {line}")

    return cargo_range, results, win_counts, n_comparisons


def plot_absolute(x, results, filename):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for k, (label, color) in VARIANTS.items():
        vx = [n for n, v in zip(x, results[k]) if v is not None]
        vy = [v for v in results[k] if v is not None]
        lw = 2.4 if k == "dynamic" else 1.8
        ax.plot(vx, vy, marker="o", markersize=3, label=label,
                color=color, linewidth=lw)
    ax.set_xlabel("Число грузов")
    ax.set_ylabel("Средняя реальная стоимость Φ (real_total)")
    ax.set_title("Динамическое циклическое расписание vs статические эвристики\n"
                 "(меньше = лучше)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(f"{N_ISLANDS} островов, {N_PAIRS} пары, {N_SEEDS} seed", fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def plot_relative(x, results, filename):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    best = []
    for i in range(len(x)):
        vals = [results[k][i] for k in VARIANTS if results[k][i] is not None]
        best.append(min(vals) if vals else None)

    for k, (label, color) in VARIANTS.items():
        vx, vy = [], []
        for i, n in enumerate(x):
            if results[k][i] is None or best[i] is None or best[i] == 0:
                continue
            vx.append(n)
            vy.append(100 * (results[k][i] / best[i] - 1))
        lw = 2.4 if k == "dynamic" else 1.8
        ax.plot(vx, vy, marker="o", markersize=3, label=label,
                color=color, linewidth=lw)
        ax.fill_between(vx, vy, 0, color=color, alpha=0.08)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Число грузов")
    ax.set_ylabel("Превышение над лучшим вариантом, %")
    ax.set_title("Разрыв между динамикой и статикой\n"
                 "(0% = лучший в точке; выше = хуже)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(f"{N_ISLANDS} островов, {N_PAIRS} пары, {N_SEEDS} seed", fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    t0 = time.time()
    x, results, win_counts, n_comp = sweep()
    print(f"\nСвип завершён за {time.time()-t0:.1f}s")

    plot_absolute(x, results, os.path.join(OUT_DIR, "dynamic_vs_static_absolute.png"))
    plot_relative(x, results, os.path.join(OUT_DIR, "dynamic_vs_static_relative.png"))

    print("\n=== WIN-RATE (доля сценариев, где вариант дал минимальный real_total) ===")
    if n_comp > 0:
        for k, (label, _) in VARIANTS.items():
            pct = 100 * win_counts[k] / n_comp
            print(f"  {label:32s}: {win_counts[k]:4d} / {n_comp}  ({pct:.1f}%)")

    print("\n=== СРЕДНЕЕ ПРЕВЫШЕНИЕ над лучшим вариантом (по всем n_cargos) ===")
    best_per_n = []
    for i in range(len(x)):
        vals = [results[k][i] for k in VARIANTS if results[k][i] is not None]
        best_per_n.append(min(vals) if vals else None)
    for k, (label, _) in VARIANTS.items():
        gaps = [100 * (results[k][i] / best_per_n[i] - 1)
                for i in range(len(x))
                if results[k][i] is not None and best_per_n[i]]
        if gaps:
            print(f"  {label:32s}: +{sum(gaps)/len(gaps):.2f}%")
