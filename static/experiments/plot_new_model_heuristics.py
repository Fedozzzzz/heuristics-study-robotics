"""
Сравнение трёх эвристик приоритета (direct, inverse, ratio) с реальной
стоимостью выполнения на ОБНОВЛЁННОЙ модели (подъезд доставщика включён
в W_d, строитель решает 3 шага последовательно).

Метрика: gap = 100 * (estimated/real - 1), %
  > 0: эвристика переоценивает реальную стоимость
  = 0: точная оценка
  < 0: эвристика недооценивает (в новой модели не должно быть для большинства точек)

Параметры запуска настраиваются константами ниже.

Данные сохраняются в JSON (CACHE_FILE) после каждой точки, что позволяет
прерваться и продолжить с того же места.

ЗАПУСК: python3 plot_new_model_heuristics.py
"""
import os
import sys
import json
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))

import matplotlib.pyplot as plt

from scenario_generator import build_environment, build_cargos_and_pairs
from priority_evaluation import (estimate_task_costs, compute_direct_priority,
                                  compute_inverse_priority, compute_ratio_priority,
                                  run_sequential_by_priority)

# ---- Параметры ----------------------------------------------------------------
N_ISLANDS    = 20
N_PAIRS      = 4
N_SEEDS      = 30        # seed на точку
N_CARGOS_MAX = 100       # верхняя граница свипа
L            = 6.0

OUT_DIR   = "outputs"
CACHE_FILE = os.path.join(OUT_DIR, "new_model_cache.json")

HEURISTICS = {
    "direct":  ("Прямая  p=W_T^i  (дорогие первыми)",      compute_direct_priority,  "#c0392b"),
    "inverse": ("Обратная  p=1/W_T^i  (дешёвые первыми)",  compute_inverse_priority, "#2b6cb0"),
    "ratio":   ("Отношение  p=W_b^i/W_d^i",                compute_ratio_priority,   "#27ae60"),
}
# -------------------------------------------------------------------------------


def run_one(key, n_cargos, seed):
    env = build_environment(n_islands=N_ISLANDS, seed=seed)
    cargos, pairs = build_cargos_and_pairs(n_islands=N_ISLANDS, n_pairs=N_PAIRS,
                                            n_cargos=n_cargos, seed=seed)
    task_costs = estimate_task_costs(env, cargos, pairs, L=L)
    priority = HEURISTICS[key][1](task_costs)
    outcome = run_sequential_by_priority(env, cargos, pairs, L=L,
                                          priority=priority, task_costs=task_costs)
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
    results = {key: {"est": [], "real": [], "gap": []} for key in HEURISTICS}

    for n_cargos in cargo_range:
        for key in HEURISTICS:
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
                    gap_runs.append(100 * (outcome.estimated_total / outcome.real_total - 1))
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

        # прогресс
        gaps = "  ".join(
            f"{k}={results[k]['gap'][-1]:+.2f}%" if results[k]['gap'][-1] is not None
            else f"{k}=N/A"
            for k in HEURISTICS)
        print(f"n={n_cargos:3d}: {gaps}")

    return cargo_range, results


def plot_single(x, results, key, filename):
    label, _, color = HEURISTICS[key]
    vx   = [n for n, g in zip(x, results[key]["gap"])  if g is not None]
    est  = [e for e in results[key]["est"]  if e is not None]
    real = [r for r in results[key]["real"] if r is not None]
    gap  = [g for g in results[key]["gap"]  if g is not None]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 9), sharex=True,
                                           gridspec_kw={"height_ratios": [2, 1]})
    ax_top.plot(vx, real, marker="o", markersize=4, label="Реальная стоимость",
                color=color, linewidth=1.8)
    ax_top.plot(vx, est,  marker="^", markersize=4, label="Эвристическая оценка",
                color=color, linewidth=1.4, linestyle="--", alpha=0.85)
    ax_top.set_ylabel("Суммарная стоимость")
    ax_top.set_title(f"{label}\nРеальная стоимость vs эвристическая оценка",
                      fontweight="bold", fontsize=11)
    ax_top.legend(fontsize=9, loc="upper left")
    ax_top.grid(True, linestyle="--", alpha=0.4)

    ax_bot.plot(vx, gap, marker="D", markersize=4, color=color, linewidth=1.8)
    ax_bot.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax_bot.fill_between(vx, gap, 0, color=color, alpha=0.15)
    ax_bot.set_xlabel("Число грузов")
    ax_bot.set_ylabel("Отклонение, %\n(оценка \u2212 реальность)")
    ax_bot.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Обновлённая модель: {N_ISLANDS} островов, {N_PAIRS} пары, {N_SEEDS} seed",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def plot_gap_all(x, results, filename):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for key, (label, _, color) in HEURISTICS.items():
        vx  = [n for n, g in zip(x, results[key]["gap"]) if g is not None]
        gap = [g for g in results[key]["gap"] if g is not None]
        ax.plot(vx, gap, marker="D", markersize=4, label=label,
                color=color, linewidth=1.8)
        ax.fill_between(vx, gap, 0, color=color, alpha=0.13)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Число грузов")
    ax.set_ylabel("Отклонение, %\n(оценка \u2212 реальность)")
    ax.set_title(
        "Отклонение эвристической оценки от реальной стоимости\n"
        "(> 0: переоценка; < 0: недооценка)",
        fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(
        f"Обновлённая модель: грузы {N_PAIRS}\u2192{N_CARGOS_MAX}, "
        f"{N_ISLANDS} островов, {N_PAIRS} пары, {N_SEEDS} seed",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    t0 = time.time()
    cargo_range, results = sweep()
    print(f"\nСвип завершён за {time.time()-t0:.1f}s")

    plot_single(cargo_range, results, "direct",  f"{OUT_DIR}/new_model_direct_4to{N_CARGOS_MAX}.png")
    plot_single(cargo_range, results, "inverse", f"{OUT_DIR}/new_model_inverse_4to{N_CARGOS_MAX}.png")
    plot_single(cargo_range, results, "ratio",   f"{OUT_DIR}/new_model_ratio_4to{N_CARGOS_MAX}.png")
    plot_gap_all(cargo_range, results,            f"{OUT_DIR}/new_model_gap_all3_4to{N_CARGOS_MAX}.png")
