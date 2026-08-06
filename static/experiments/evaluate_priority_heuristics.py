"""
Оценка ЭВРИСТИК ПРИОРИТЕТА по итоговому качеству расписания (real_total = Φ).

Назначение грузов парам ФИКСИРОВАНО (round-robin) — так изолируется эффект
ТОЛЬКО порядка выполнения, без влияния стратегии назначения. На одних и тех
же сценариях (одинаковые seed) прогоняются четыре варианта порядка:

  - direct   : p = W_T^i        (дорогие первыми)
  - inverse  : p = 1 / W_T^i     (дешёвые первыми)
  - ratio    : p = W_b^i / W_d^i (дорого строителю первыми)
  - random   : случайный порядок (БЕЙЗЛАЙН - показывает, даёт ли осмысленный
               приоритет выигрыш относительно произвольного порядка вообще)

Метрика: real_total (суммарная реальная стоимость всех задач) = целевая
функция Φ. Чем МЕНЬШЕ, тем лучше эвристика решает задачу.

Строятся:
  1) График среднего real_total по числу грузов (4 линии) - absolute.
  2) График относительно ЛУЧШЕЙ эвристики в каждой точке (%) - показывает
     разрыв между эвристиками нагляднее, чем абсолютные значения.
  3) Печать win-rate: на скольких сценариях каждая эвристика была лучшей.

Данные кэшируются (дозапуск при прерывании).

ЗАПУСК: python3 evaluate_priority_heuristics.py
"""
import os
import sys
import json
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))

import matplotlib.pyplot as plt

from scenario_generator import build_environment, build_cargos_and_pairs
from priority_evaluation import (estimate_task_costs, compute_direct_priority,
                                  compute_inverse_priority, compute_ratio_priority,
                                  run_sequential_by_priority)
from distance_heuristic_evaluation import compute_distance_priority
from lpt_assignment import (assign_by_lpt, assign_greedy_nearest,
                            apply_assignment)

# ---- Параметры ---------------------------------------------------------------
N_ISLANDS    = 20
N_PAIRS      = 4
N_SEEDS      = 30
N_CARGOS_MAX = 60
L            = 6.0

# --- ВЫБОР СТРАТЕГИИ НАЗНАЧЕНИЯ грузов парам ---
# "round_robin"    - циклическое (из генератора), базовое
# "lpt"            - LPT: крупные задачи первыми, паре с мин. нагрузкой
# "greedy_nearest" - каждый груз паре, которой он дешевле всего
ASSIGNMENT = "lpt"   # "round_robin" | "lpt" | "greedy_nearest"

OUT_DIR    = "outputs"
CACHE_FILE = os.path.join(OUT_DIR, f"priority_eval_cache_{ASSIGNMENT}.json")

HEURISTICS = {
    "direct":   ("Прямая  p=W_T^i",        "#c0392b"),
    "inverse":  ("Обратная  p=1/W_T^i",    "#2b6cb0"),
    "ratio":    ("Отношение  p=W_b/W_d",   "#27ae60"),
    "distance": ("Геометрическая (2.4)  p=1/max(dist)", "#8e44ad"),
    "random":   ("Случайный порядок",       "#7f8c8d"),
}
# ------------------------------------------------------------------------------


def apply_chosen_assignment(cargos, pairs, env):
    """Применяет выбранную стратегию назначения (ASSIGNMENT) к грузам.
    Возвращает НОВЫЙ список cargos с проставленным assigned_pair."""
    if ASSIGNMENT == "round_robin":
        return cargos  # генератор уже проставил round-robin
    if ASSIGNMENT == "lpt":
        assignment = assign_by_lpt(cargos, pairs, env, L=L)
        return apply_assignment(cargos, assignment)
    if ASSIGNMENT == "greedy_nearest":
        assignment = assign_greedy_nearest(cargos, pairs, env, L=L)
        return apply_assignment(cargos, assignment)
    raise ValueError(f"Неизвестная ASSIGNMENT: {ASSIGNMENT}")


def compute_distance_priority_static(cargos, pairs, env):
    """
    Геометрическая эвристика (формула 2.4 из постановки задачи):
        p(c_i) = 1 / max(dist(deliverer_pos, v_start^i), dist(builder_pos, v_start^i))
    Считается ОДИН РАЗ от начальной позиции назначенной пары (той же, что
    используется в estimate_task_costs) - как и остальные эвристики здесь,
    чтобы сравнение было честным (без динамического пересчёта на каждом шаге).
    """
    pairs_by_id = {p.id: p for p in pairs}
    priority = {}
    for c in cargos:
        pair = pairs_by_id[c.assigned_pair]
        priority[c.id] = compute_distance_priority(c, env, pair)
    return priority


def make_priority(key, task_costs, seed, cargos=None, pairs=None, env=None):
    if key == "direct":
        return compute_direct_priority(task_costs)
    if key == "inverse":
        return compute_inverse_priority(task_costs)
    if key == "ratio":
        return compute_ratio_priority(task_costs)
    if key == "distance":
        return compute_distance_priority_static(cargos, pairs, env)
    if key == "random":
        # случайный, но детерминированный по seed приоритет
        rnd = random.Random(seed * 7919)
        pr = {}
        for cid, r in task_costs.items():
            pr[cid] = float("-inf") if not r.feasible else rnd.random()
        return pr
    raise ValueError(key)


def run_one(key, n_cargos, seed):
    """real_total для одного сценария с выбранной стратегией назначения."""
    env = build_environment(n_islands=N_ISLANDS, seed=seed)
    cargos, pairs = build_cargos_and_pairs(n_islands=N_ISLANDS, n_pairs=N_PAIRS,
                                            n_cargos=n_cargos, seed=seed)
    # применяем выбранную стратегию назначения грузов парам
    cargos = apply_chosen_assignment(cargos, pairs, env)

    task_costs = estimate_task_costs(env, cargos, pairs, L=L)
    priority = make_priority(key, task_costs, seed, cargos=cargos, pairs=pairs, env=env)
    outcome = run_sequential_by_priority(env, cargos, pairs, L=L,
                                          priority=priority, task_costs=task_costs)
    if not outcome.all_delivered:
        return None
    return outcome.real_total


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
    # results[key] = список средних real_total по n_cargos
    results = {k: [] for k in HEURISTICS}
    # win_counts[key] = на скольких (n_cargos, seed) эвристика была строго лучшей
    win_counts = {k: 0 for k in HEURISTICS}
    n_comparisons = 0

    for n_cargos in cargo_range:
        # для win-rate нужны попарные результаты по каждому seed
        per_seed = {k: [] for k in HEURISTICS}  # real_total по каждому seed

        for k in HEURISTICS:
            ckey = f"{k}_{n_cargos}"
            if ckey in cache:
                per_seed[k] = cache[ckey]
            else:
                vals = []
                for seed in range(N_SEEDS):
                    rt = run_one(k, n_cargos, seed)
                    vals.append(rt)  # может быть None
                cache[ckey] = vals
                save_cache(cache)
                per_seed[k] = vals

        # среднее real_total по валидным сценариям
        for k in HEURISTICS:
            valid = [v for v in per_seed[k] if v is not None]
            results[k].append(sum(valid) / len(valid) if valid else None)

        # win-rate: для каждого seed находим эвристику с минимальным real_total
        for si in range(N_SEEDS):
            row = {k: per_seed[k][si] for k in HEURISTICS
                   if per_seed[k][si] is not None}
            if len(row) < len(HEURISTICS):
                continue  # пропускаем seed, где не все доставили
            n_comparisons += 1
            best_k = min(row, key=row.get)
            win_counts[best_k] += 1

        line = "  ".join(
            f"{k}={results[k][-1]:.1f}" if results[k][-1] is not None else f"{k}=N/A"
            for k in HEURISTICS)
        print(f"n={n_cargos:3d}: {line}")

    return cargo_range, results, win_counts, n_comparisons


def plot_absolute(x, results, filename):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for k, (label, color) in HEURISTICS.items():
        vx = [n for n, v in zip(x, results[k]) if v is not None]
        vy = [v for v in results[k] if v is not None]
        lw = 2.2 if k != "random" else 1.5
        ls = "--" if k == "random" else "-"
        ax.plot(vx, vy, marker="o", markersize=3, label=label,
                color=color, linewidth=lw, linestyle=ls)
    ax.set_xlabel("Число грузов")
    ax.set_ylabel("Средняя реальная стоимость Φ (real_total)")
    ax.set_title("Качество расписания разных эвристик приоритета\n"
                 f"(меньше = лучше; назначение: {ASSIGNMENT})",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(f"{N_ISLANDS} островов, {N_PAIRS} пары, {N_SEEDS} seed",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def plot_relative(x, results, filename):
    """Отклонение каждой эвристики от ЛУЧШЕЙ (минимальной) в каждой точке, %."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    # лучшая (минимум) по каждому n_cargos среди всех эвристик
    best = []
    for i in range(len(x)):
        vals = [results[k][i] for k in HEURISTICS if results[k][i] is not None]
        best.append(min(vals) if vals else None)

    for k, (label, color) in HEURISTICS.items():
        vx, vy = [], []
        for i, n in enumerate(x):
            if results[k][i] is None or best[i] is None or best[i] == 0:
                continue
            vx.append(n)
            vy.append(100 * (results[k][i] / best[i] - 1))
        lw = 2.2 if k != "random" else 1.5
        ls = "--" if k == "random" else "-"
        ax.plot(vx, vy, marker="o", markersize=3, label=label,
                color=color, linewidth=lw, linestyle=ls)
        ax.fill_between(vx, vy, 0, color=color, alpha=0.08)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Число грузов")
    ax.set_ylabel("Превышение над лучшей эвристикой, %")
    ax.set_title("Разрыв между эвристиками приоритета\n"
                 "(0% = лучшая в точке; выше = хуже)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(f"{N_ISLANDS} островов, {N_PAIRS} пары, {N_SEEDS} seed",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    t0 = time.time()
    x, results, win_counts, n_comp = sweep()
    print(f"\nСвип завершён за {time.time()-t0:.1f}s")

    plot_absolute(x, results, os.path.join(OUT_DIR, f"priority_quality_absolute_{ASSIGNMENT}.png"))
    plot_relative(x, results, os.path.join(OUT_DIR, f"priority_quality_relative_{ASSIGNMENT}.png"))

    print(f"\n[Стратегия назначения грузов парам: {ASSIGNMENT}]")

    # сводка win-rate
    print("\n=== WIN-RATE (доля сценариев, где эвристика дала минимальный real_total) ===")
    if n_comp > 0:
        for k, (label, _) in HEURISTICS.items():
            pct = 100 * win_counts[k] / n_comp
            print(f"  {label:28s}: {win_counts[k]:4d} / {n_comp}  ({pct:.1f}%)")

    # средний проигрыш каждой эвристики лучшей (по всем точкам)
    print("\n=== СРЕДНЕЕ ПРЕВЫШЕНИЕ над лучшей эвристикой (по всем n_cargos) ===")
    best_per_n = []
    for i in range(len(x)):
        vals = [results[k][i] for k in HEURISTICS if results[k][i] is not None]
        best_per_n.append(min(vals) if vals else None)
    for k, (label, _) in HEURISTICS.items():
        gaps = [100 * (results[k][i] / best_per_n[i] - 1)
                for i in range(len(x))
                if results[k][i] is not None and best_per_n[i]]
        if gaps:
            print(f"  {label:28s}: +{sum(gaps)/len(gaps):.2f}%")
