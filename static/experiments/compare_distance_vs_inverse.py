"""
Сравнение ГЕОМЕТРИЧЕСКОЙ эвристики приоритета (формула 2.4, только расстояние
до груза) с ОБРАТНОЙ эвристикой (p = 1/W_T^i), по итоговому качеству
расписания (real_total = Phi) - т.е. обе эвристики оцениваются на ОДНОЙ и
той же шкале: реальная суммарная стоимость выполнения всех задач.

Эвристики:
  inverse  : p(c_i) = 1 / W_T^i
             (полная стоимость задачи - проезд доставщика + постройка
             мостов строителем; дешёвые задачи выполняются первыми)

  distance : p(c_i) = 1 / max( dist(deliverer_pos, v_start^i),
                                dist(builder_pos,   v_start^i) )
             (формула 2.4 из постановки задачи - "Комбинированный: минимальное
             расстояние до доставщика и строителя одновременно". В отличие
             от inverse, НЕ требует решения подзадачи маршрута/построения
             мостов - использует только прямое (евклидово) расстояние от
             текущей позиции пары до точки погрузки груза. Это дешевле
             вычислительно, но игнорирует реальную стоимость проезда и
             неизбежность строительства мостов.)

ВАЖНО: приоритет для ОБЕИХ эвристик считается ОДИН РАЗ, от НАЧАЛЬНОЙ позиции
назначенной пары (как и direct/inverse/ratio в priority_evaluation.py) - это
даёт честное сравнение, поскольку сам порядок выполнения (run_sequential_by_
priority) у них общий, отличается только формула приоритета. Назначение
грузов парам - round-robin (как в compare_direct_inverse_100cargos.py).

Метрика сравнения - real_total (реальная суммарная стоимость Phi, меньше
= лучше), как функция числа грузов, усреднённая по N_SEEDS случайным
сценариям на каждую точку.

Строятся:
  1) Один график: real_total(n_cargos) для inverse и distance одновременно
     (basic "график относительно реальной стоимости").
  2) Отклонение distance от inverse в %, gap(n) = 100*(distance/inverse - 1),
     чтобы показать разрыв явно (> 0 - distance хуже/дороже, < 0 - дешевле).

ЗАПУСК: python3 compare_distance_vs_inverse.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))

from typing import Dict, List

import matplotlib.pyplot as plt

from delivery_model import IslandGraph, Cargo, Pair, TaskResult
from scenario_generator import build_environment, build_cargos_and_pairs
from priority_evaluation import (estimate_task_costs, compute_inverse_priority,
                                  run_sequential_by_priority)
from distance_heuristic_evaluation import compute_distance_priority

L_FOR_ALL_SCENARIOS = 6.0
N_ISLANDS = 20
N_PAIRS = 4
N_SEEDS = 30
N_CARGOS_MAX = 100

HEURISTICS = {
    "inverse":  ("Обратная  p=1/W_T^i  (дешёвые первыми)", "#2b6cb0"),
    "distance": ("Геометрическая (2.4)  p=1/max(dist до доставщика, "
                 "dist до строителя)", "#8e44ad"),
}


def compute_distance_priority_static(cargos: List[Cargo], pairs: List[Pair],
                                      env: IslandGraph) -> Dict[str, float]:
    """
    Приоритет по формуле 2.4, посчитанный ОДИН РАЗ от НАЧАЛЬНОЙ позиции
    назначенной пары - т.е. так же, как для direct/inverse/ratio (иначе
    сравнение было бы нечестным: там priority фиксирован от начала, а
    здесь бы динамически пересчитывался на каждом шаге, как в
    distance_heuristic_evaluation.run_sequential_by_distance_priority).
    """
    pairs_by_id = {p.id: p for p in pairs}
    priority: Dict[str, float] = {}
    for c in cargos:
        pair = pairs_by_id[c.assigned_pair]
        priority[c.id] = compute_distance_priority(c, env, pair)
    return priority


def make_priority(key: str, task_costs: Dict[str, TaskResult],
                   cargos: List[Cargo], pairs: List[Pair], env: IslandGraph):
    if key == "inverse":
        return compute_inverse_priority(task_costs)
    if key == "distance":
        return compute_distance_priority_static(cargos, pairs, env)
    raise ValueError(key)


def run_one_scenario(heuristic_key: str, n_cargos: int, seed: int):
    env = build_environment(n_islands=N_ISLANDS, seed=seed)
    cargos, pairs = build_cargos_and_pairs(n_islands=N_ISLANDS, n_pairs=N_PAIRS,
                                            n_cargos=n_cargos, seed=seed)

    task_costs = estimate_task_costs(env, cargos, pairs, L=L_FOR_ALL_SCENARIOS)
    priority = make_priority(heuristic_key, task_costs, cargos, pairs, env)

    outcome = run_sequential_by_priority(env, cargos, pairs, L=L_FOR_ALL_SCENARIOS,
                                          priority=priority, task_costs=task_costs)
    return outcome


def sweep_n_cargos(n_seeds: int = N_SEEDS):
    """n_cargos от N_PAIRS до N_CARGOS_MAX включительно, шаг 1."""
    cargo_range = list(range(N_PAIRS, N_CARGOS_MAX + 1))
    results = {key: [] for key in HEURISTICS}

    for n_cargos in cargo_range:
        for key in HEURISTICS:
            real_runs = []
            for seed in range(n_seeds):
                outcome = run_one_scenario(key, n_cargos=n_cargos, seed=seed)
                if not outcome.all_delivered:
                    continue
                real_runs.append(outcome.real_total)

            results[key].append(sum(real_runs) / len(real_runs) if real_runs else None)

        line = "  ".join(
            f"{key}={results[key][-1]:.1f}" if results[key][-1] is not None
            else f"{key}=N/A" for key in HEURISTICS)
        print(f"n_cargos={n_cargos:3d}: {line}")

    return cargo_range, results


def plot_real_cost(x_values, results, filename):
    """Реальная стоимость (Phi) обеих эвристик на одном графике, чем ниже - тем лучше."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for key, (label, color) in HEURISTICS.items():
        vx = [x for x, v in zip(x_values, results[key]) if v is not None]
        vy = [v for v in results[key] if v is not None]
        ax.plot(vx, vy, marker="o", markersize=3.5, label=label,
                color=color, linewidth=2.0)

    ax.set_xlabel("Число грузов")
    ax.set_ylabel("Реальная суммарная стоимость Ф (real_total)")
    ax.set_title("Обратная эвристика (W_T) vs Геометрическая эвристика (2.4)\n"
                 "(меньше = лучше)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(
        f"Грузы {N_PAIRS}\u2192{N_CARGOS_MAX} "
        f"({N_ISLANDS} островов, {N_PAIRS} пары, усреднено по {N_SEEDS} seed)",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def plot_gap(x_values, results, filename):
    """Отклонение distance от inverse в %: gap = 100*(distance/inverse - 1)."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.0))

    vx, vgap = [], []
    for x, inv, dist in zip(x_values, results["inverse"], results["distance"]):
        if inv is None or dist is None or inv == 0:
            continue
        vx.append(x)
        vgap.append(100 * (dist / inv - 1))

    ax.plot(vx, vgap, marker="D", markersize=3.5, color="#8e44ad", linewidth=2.0)
    ax.fill_between(vx, vgap, 0, color="#8e44ad", alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Число грузов")
    ax.set_ylabel("Отклонение геометрической эвристики от обратной, %")
    ax.set_title(
        "Геометрическая (2.4) относительно обратной эвристики\n"
        "(> 0: геометрическая ХУЖЕ (дороже реальное расписание); "
        "< 0: геометрическая ЛУЧШЕ)",
        fontweight="bold", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(
        f"Грузы {N_PAIRS}\u2192{N_CARGOS_MAX} "
        f"({N_ISLANDS} островов, {N_PAIRS} пары, усреднено по {N_SEEDS} seed)",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    print("=" * 70)
    print(f"Обратная vs Геометрическая (2.4): грузы {N_PAIRS}->{N_CARGOS_MAX}, "
          f"{N_ISLANDS} островов, {N_PAIRS} пары, {N_SEEDS} seed на точку")
    print("=" * 70)
    cargo_range, results = sweep_n_cargos(n_seeds=N_SEEDS)

    plot_real_cost(cargo_range, results,
                    f"outputs/distance_vs_inverse_real_cost_{N_PAIRS}to{N_CARGOS_MAX}cargos.png")
    plot_gap(cargo_range, results,
             f"outputs/distance_vs_inverse_gap_{N_PAIRS}to{N_CARGOS_MAX}cargos.png")

    # средний разрыв по всем точкам - краткая сводка в консоль
    gaps = []
    for inv, dist in zip(results["inverse"], results["distance"]):
        if inv is not None and dist is not None and inv != 0:
            gaps.append(100 * (dist / inv - 1))
    if gaps:
        print(f"\nСреднее отклонение геометрической эвристики от обратной: "
              f"{sum(gaps)/len(gaps):+.2f}%")
