"""
Сравнение на одном графике (в % относительно реальной стоимости):
  - НИЖНЯЯ граница  (выбирается: LOWER_BOUND)
  - реальная стоимость (базис, 0%)
  - ВЕРХНЯЯ граница / эвристическая оценка (выбирается: UPPER_BOUND)

Прямая эвристика приоритета (p = W_T^i) во всех случаях.

============================================================
ВЫБОР ГРАНИЦ - меняй константы LOWER_BOUND и UPPER_BOUND ниже:

  LOWER_BOUND (нижняя граница):
    "exact_n4" - compute_lower_bound: max-задача пары + проезд остальных.
                 ТОЧНА при 1 грузе на пару (lb=real в начальной точке).
    "mst"      - compute_lower_bound_mst: идеальный граф + MST.
                 Более консервативна (ниже), не точна при n=4.
    "none"     - не рисовать нижнюю границу.

  UPPER_BOUND (верхняя граница / оценка):
    "pool"     - оценка с коррекцией пула мостов (более плотная сверху).
    "raw"      - сырая оценка estimate_task_costs (без коррекции пула, выше).
    "none"     - не рисовать верхнюю границу.
============================================================

Параметры: 4->N_CARGOS_MAX грузов, шаг 1, N_SEEDS seed на точку.
Кэш в outputs/<CACHE>.json (дозапуск при прерывании). ВАЖНО: при смене
границ имя кэша меняется автоматически, пересчёт не нужен вручную.

ЗАПУСК: python3 compare_lb_pool_real.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))

import matplotlib.pyplot as plt

from scenario_generator import build_environment, build_cargos_and_pairs
from priority_evaluation import (estimate_task_costs, compute_direct_priority,
                                  compute_pool_corrected_costs,
                                  compute_lower_bound, compute_lower_bound_mst,
                                  run_sequential_by_priority)

# ---- Параметры ---------------------------------------------------------------
N_ISLANDS    = 20
N_PAIRS      = 4
N_SEEDS      = 30
N_CARGOS_MAX = 100
L            = 6.0

# --- ВЫБОР ГРАНИЦ (см. docstring выше) ---
LOWER_BOUND = "exact_n4"   # "exact_n4" | "mst" | "none"
UPPER_BOUND = "pool"       # "pool" | "raw" | "none"

OUT_DIR = "outputs"
# ------------------------------------------------------------------------------

# реестры доступных границ
LOWER_LABELS = {
    "exact_n4": "Нижняя граница (max-задача + проезд)",
    "mst":      "Нижняя граница (идеальный граф + MST)",
}
UPPER_LABELS = {
    "pool": "Эвристическая оценка (с пулом)",
    "raw":  "Эвристическая оценка (сырая)",
}

CACHE_FILE = os.path.join(OUT_DIR, f"lbub_cache_{LOWER_BOUND}_{UPPER_BOUND}.json")


def compute_lower(env, cargos, pairs):
    if LOWER_BOUND == "exact_n4":
        return compute_lower_bound(env, cargos, pairs, L=L)
    elif LOWER_BOUND == "mst":
        return compute_lower_bound_mst(env, cargos, pairs, L=L)
    elif LOWER_BOUND == "none":
        return None
    raise ValueError(f"Неизвестная LOWER_BOUND: {LOWER_BOUND}")


def compute_upper_and_real(env, cargos, pairs, task_costs):
    """Возвращает (upper, real). upper зависит от UPPER_BOUND."""
    if UPPER_BOUND == "raw":
        costs_for_prio = task_costs
    else:  # "pool" или "none" - для расписания используем скорректированные
        costs_for_prio = compute_pool_corrected_costs(env, cargos, task_costs)

    priority = compute_direct_priority(costs_for_prio)
    outcome = run_sequential_by_priority(env, cargos, pairs, L=L,
                                          priority=priority,
                                          task_costs=costs_for_prio)
    if not outcome.all_delivered:
        return None, None

    if UPPER_BOUND == "none":
        upper = None
    else:
        upper = outcome.estimated_total
    return upper, outcome.real_total


def run_one(n_cargos: int, seed: int):
    """Возвращает (lower, real, upper) для одного сценария (любое может быть None)."""
    env = build_environment(n_islands=N_ISLANDS, seed=seed)
    cargos, pairs = build_cargos_and_pairs(n_islands=N_ISLANDS, n_pairs=N_PAIRS,
                                            n_cargos=n_cargos, seed=seed)
    task_costs = estimate_task_costs(env, cargos, pairs, L=L)

    lower = compute_lower(env, cargos, pairs)
    upper, real = compute_upper_and_real(env, cargos, pairs, task_costs)
    if real is None:
        return None
    return lower, real, upper


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
    lb_series, real_series, ub_series = [], [], []

    for n_cargos in cargo_range:
        ckey = str(n_cargos)
        if ckey in cache:
            entry = cache[ckey]
        else:
            lb_runs, real_runs, ub_runs = [], [], []
            for seed in range(N_SEEDS):
                res = run_one(n_cargos, seed)
                if res is None:
                    continue
                lb, real, ub = res
                real_runs.append(real)
                if lb is not None:
                    lb_runs.append(lb)
                if ub is not None:
                    ub_runs.append(ub)
            if real_runs:
                entry = {
                    "lb":   (sum(lb_runs) / len(lb_runs)) if lb_runs else None,
                    "real": sum(real_runs) / len(real_runs),
                    "ub":   (sum(ub_runs) / len(ub_runs)) if ub_runs else None,
                }
            else:
                entry = {"lb": None, "real": None, "ub": None}
            cache[ckey] = entry
            save_cache(cache)

        lb_series.append(entry["lb"])
        real_series.append(entry["real"])
        ub_series.append(entry["ub"])

        if entry["real"] is not None:
            lb_s = f"{entry['lb']:.1f}" if entry["lb"] is not None else "-"
            ub_s = f"{entry['ub']:.1f}" if entry["ub"] is not None else "-"
            print(f"n={n_cargos:3d}: lb={lb_s}  real={entry['real']:.1f}  ub={ub_s}")
        else:
            print(f"n={n_cargos:3d}: N/A")

    return cargo_range, lb_series, real_series, ub_series


def plot(x, lb, real, ub, filename):
    fig, ax = plt.subplots(figsize=(10, 6.5))

    def to_pct(series):
        vx, vy = [], []
        for xi, val, r in zip(x, series, real):
            if val is None or r is None or r == 0:
                continue
            vx.append(xi)
            vy.append(100 * (val / r - 1))
        return vx, vy

    ax.axhline(0, color="#c0392b", linewidth=2.0, label="Реальная стоимость (базис)")

    if UPPER_BOUND != "none":
        vx_u, vy_u = to_pct(ub)
        ax.plot(vx_u, vy_u, marker="^", markersize=3,
                label=UPPER_LABELS[UPPER_BOUND], color="#e67e22", linewidth=1.8)
        ax.fill_between(vx_u, vy_u, 0, color="#e67e22", alpha=0.12)

    if LOWER_BOUND != "none":
        vx_l, vy_l = to_pct(lb)
        ax.plot(vx_l, vy_l, marker="s", markersize=3,
                label=LOWER_LABELS[LOWER_BOUND], color="#2b6cb0",
                linewidth=1.8, linestyle="--")
        ax.fill_between(vx_l, vy_l, 0, color="#2b6cb0", alpha=0.12)

    ax.set_xlabel("Число грузов")
    ax.set_ylabel("Отклонение от реальной стоимости, %")
    ax.set_title(
        "Оценки относительно реальной стоимости\n"
        "(реальность = 0%; > 0 переоценка, < 0 недооценка)",
        fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Грузы {N_PAIRS}\u2192{N_CARGOS_MAX}, {N_ISLANDS} островов, "
        f"{N_PAIRS} пары, {N_SEEDS} seed (прямая эвристика)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    t0 = time.time()
    x, lb, real, ub = sweep()
    print(f"\nСвип завершён за {time.time()-t0:.1f}s")
    fname = os.path.join(
        OUT_DIR, f"bounds_{LOWER_BOUND}_{UPPER_BOUND}_{N_PAIRS}to{N_CARGOS_MAX}.png")
    plot(x, lb, real, ub, fname)
