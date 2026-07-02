"""
Сравнение прямой эвристики (p = W_T^i) БЕЗ и С коррекцией пула мостов.

Коррекция пула (compute_pool_corrected_costs): вычитает стоимость повторных
построек одних и тех же мостов из суммарного W_b пары, делая оценку более
точной (более плотная оценка сверху).

График: gap = 100 * (estimated/real - 1), %
  > 0: оценка переоценивает реальность
  Ближе к 0 = лучше

Параметры: 4->N_CARGOS_MAX грузов, шаг 1, N_SEEDS seed на точку.
Данные кэшируются в outputs/pool_correction_cache.json для возможности
дозапуска.

ЗАПУСК: python3 compare_pool_correction.py
"""
import os
import sys
import json
import time
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import matplotlib.pyplot as plt

from scenario_generator import build_environment, build_cargos_and_pairs
from priority_evaluation import (estimate_task_costs, compute_direct_priority,
                                  compute_pool_corrected_costs,
                                  run_sequential_by_priority)

# ---- Параметры ---------------------------------------------------------------
N_ISLANDS    = 20
N_PAIRS      = 4
N_SEEDS      = 30
N_CARGOS_MAX = 100
L            = 6.0

OUT_DIR    = "outputs"
CACHE_FILE = os.path.join(OUT_DIR, "pool_correction_cache.json")

VARIANTS = {
    "direct":          ("Прямая  p=W_T^i  (без коррекции)",          "#c0392b"),
    "direct_poolcorr": ("Прямая  p=W_T^i  (с коррекцией пула)",      "#e67e22"),
}
# ------------------------------------------------------------------------------


def run_one(key: str, n_cargos: int, seed: int):
    env    = build_environment(n_islands=N_ISLANDS, seed=seed)
    cargos, pairs = build_cargos_and_pairs(n_islands=N_ISLANDS, n_pairs=N_PAIRS,
                                            n_cargos=n_cargos, seed=seed)
    task_costs = estimate_task_costs(env, cargos, pairs, L=L)

    if key == "direct_poolcorr":
        task_costs = compute_pool_corrected_costs(env, cargos, task_costs)

    priority = compute_direct_priority(task_costs)
    outcome  = run_sequential_by_priority(env, cargos, pairs, L=L,
                                           priority=priority,
                                           task_costs=task_costs)
    return outcome


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
    results = {key: {"est": [], "real": [], "gap": []} for key in VARIANTS}

    for n_cargos in cargo_range:
        for key in VARIANTS:
            cache_key = f"{key}_{n_cargos}"
            if cache_key in cache:
                entry = cache[cache_key]
            else:
                est_runs, real_runs, gap_runs = [], [], []
                for seed in range(N_SEEDS):
                    outcome = run_one(key, n_cargos, seed)
                    if not outcome.all_delivered:
                        continue
                    est_runs.append(outcome.estimated_total)
                    real_runs.append(outcome.real_total)
                    gap_runs.append(
                        100 * (outcome.estimated_total / outcome.real_total - 1))
                if est_runs:
                    entry = {
                        "est":  sum(est_runs)  / len(est_runs),
                        "real": sum(real_runs) / len(real_runs),
                        "gap":  sum(gap_runs)  / len(gap_runs),
                    }
                else:
                    entry = {"est": None, "real": None, "gap": None}
                cache[cache_key] = entry
                save_cache(cache)

            results[key]["est"].append(entry["est"])
            results[key]["real"].append(entry["real"])
            results[key]["gap"].append(entry["gap"])

        gaps = "  ".join(
            f"{k}={results[k]['gap'][-1]:+.2f}%"
            if results[k]["gap"][-1] is not None else f"{k}=N/A"
            for k in VARIANTS)
        print(f"n={n_cargos:3d}: {gaps}")

    return cargo_range, results


def plot_gap(x_values, results, filename):
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for key, (label, color) in VARIANTS.items():
        vx  = [n for n, g in zip(x_values, results[key]["gap"]) if g is not None]
        gap = [g for g in results[key]["gap"] if g is not None]
        ax.plot(vx, gap, marker="o", markersize=3, label=label,
                color=color, linewidth=1.8)
        ax.fill_between(vx, gap, 0, color=color, alpha=0.13)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Число грузов")
    ax.set_ylabel("Отклонение, %\n(оценка \u2212 реальность)")
    ax.set_title(
        "Эффект коррекции пула мостов на точность оценки\n"
        "(ближе к 0 = точнее; > 0 = переоценка)",
        fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Прямая эвристика: без коррекции vs с коррекцией пула мостов\n"
        f"Грузы {N_PAIRS}\u2192{N_CARGOS_MAX}, {N_ISLANDS} островов, "
        f"{N_PAIRS} пары, {N_SEEDS} seed",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    t0 = time.time()
    cargo_range, results = sweep()
    print(f"\nСвип завершён за {time.time()-t0:.1f}s")

    plot_gap(cargo_range, results,
             os.path.join(OUT_DIR,
                          f"pool_correction_gap_{N_PAIRS}to{N_CARGOS_MAX}.png"))
