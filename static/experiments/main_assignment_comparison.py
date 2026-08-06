"""
СРАВНЕНИЕ ЭВРИСТИК НАЗНАЧЕНИЯ "ГРУЗ -> ПАРА" ДЛЯ МИНИМИЗАЦИИ ОБЩЕЙ
СТОИМОСТИ ВЫПОЛНЕНИЯ ВСЕХ ОПЕРАЦИЙ.

Целевая функция (см. lpt_assignment.py и ALGORITHMS.md, Алгоритм 0):

    Φ(π, σ) = Σ_k Σ_j W_T(U_k, σ_k(j) | v^{(j-1)}_k)  ->  min

Сравниваются три эвристики назначения π (порядок σ_k внутри пары везде
одинаков - динамический приоритет по W_T^i, Алгоритм 2):

  1. ROUND-ROBIN     - циклическое распределение без учёта стоимости (baseline)
  2. LPT             - Longest Processing Time first; минимизирует MAKESPAN
                        (баланс максимальной загрузки), классическая
                        эвристика теории расписаний (Graham, 1969)
  3. GREEDY-NEAREST  - каждый груз назначается паре с минимальной стоимостью
                        обслуживания ПРЯМО СЕЙЧАС; целится непосредственно
                        в минимизацию СУММЫ, а не максимума

КЛЮЧЕВОЙ РЕЗУЛЬТАТ: LPT даёт хорошую БАЛАНСИРОВКУ нагрузки между парами,
но не гарантирует уменьшение СУММЫ - на части сценариев даже проигрывает
наивному round-robin. Причина: в классическом parallel scheduling сумма
стоимостей инвариантна к назначению (если "время обработки" не зависит от
позиции исполнителя) - LPT там влияет только на максимум. В нашей задаче
W_T^i ЗАВИСИТ от текущей позиции пары, поэтому сумма не инвариантна, но
эта зависимость не связана с тем, что оптимизирует LPT. Greedy Nearest
Assignment явно целится в сумму и систематически превосходит обе
альтернативы.

ЗАПУСК: python3 main_assignment_comparison.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))

import matplotlib.pyplot as plt

import scenario_generator as sg
from lpt_assignment import assign_by_lpt, assign_round_robin, assign_greedy_nearest, apply_assignment
from algorithm_2 import run_scheduling


L_FOR_ALL_SCENARIOS = 6.0

METHODS = {
    "round_robin": ("Round-Robin (baseline)", "#7f8c8d"),
    "lpt": ("LPT (баланс максимума)", "#c0392b"),
    "greedy_nearest": ("Greedy Nearest (минимизация суммы)", "#2b6cb0"),
}


def run_assignment(method_key: str, env, cargos, pairs, L: float):
    if method_key == "round_robin":
        assignment = assign_round_robin(cargos, pairs)
    elif method_key == "lpt":
        assignment = assign_by_lpt(cargos, pairs, env, L)
    elif method_key == "greedy_nearest":
        assignment = assign_greedy_nearest(cargos, pairs, env, L)
    else:
        raise ValueError(method_key)

    cargos_assigned = apply_assignment(cargos, assignment)
    outcome = run_scheduling(env, cargos_assigned, pairs, L=L)
    return outcome


def sweep_n_cargos(n_seeds: int = 5, n_points: int = 15):
    """Сравнение трёх методов на растущем числе грузов (острова, пары фиксированы)."""
    n_islands = 20
    n_pairs = 4
    cargo_range = list(range(4, 4 + 2 * n_points, 2))[:n_points]

    results = {key: [] for key in METHODS}

    for n_cargos in cargo_range:
        totals = {key: [] for key in METHODS}
        for seed in range(n_seeds):
            env = sg.build_environment(n_islands=n_islands, seed=seed)
            cargos, pairs = sg.build_cargos_and_pairs(n_islands=n_islands, n_pairs=n_pairs,
                                                       n_cargos=n_cargos, seed=seed)
            for key in METHODS:
                outcome = run_assignment(key, env, cargos, pairs, L_FOR_ALL_SCENARIOS)
                if outcome.all_delivered:
                    totals[key].append(outcome.W_d_total + outcome.W_b_total)

        line = f"n_cargos={n_cargos}: "
        for key in METHODS:
            if totals[key]:
                mean = sum(totals[key]) / len(totals[key])
                results[key].append(mean)
                line += f"{key}={mean:.1f}  "
            else:
                results[key].append(None)
                line += f"{key}=N/A  "
        print(line)

    return cargo_range, results


def plot_assignment_comparison(x_values, results, x_label, title, filename):
    """Два графика: (1) абсолютная суммарная стоимость, (2) улучшение в % относительно baseline."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for key, (label, color) in METHODS.items():
        valid_x = [x for x, v in zip(x_values, results[key]) if v is not None]
        valid_y = [v for v in results[key] if v is not None]
        ax.plot(valid_x, valid_y, marker="o", markersize=4, label=label,
                color=color, linewidth=1.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Суммарная стоимость Φ = W_d_total + W_b_total")
    ax.set_title("Абсолютная суммарная стоимость")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    baseline = results["round_robin"]
    for key in ["lpt", "greedy_nearest"]:
        label, color = METHODS[key]
        improvement = []
        valid_x = []
        for x, v, b in zip(x_values, results[key], baseline):
            if v is not None and b is not None and b > 0:
                improvement.append(100 * (1 - v / b))
                valid_x.append(x)
        ax.plot(valid_x, improvement, marker="o", markersize=4, label=label,
                color=color, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Улучшение относительно Round-Robin, %")
    ax.set_title("Относительное улучшение (положительное = лучше baseline)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    print("=" * 70)
    print("СЕРИЯ: рост числа грузов (15 точек, 20 островов, 4 пары)")
    print("=" * 70)
    cargo_range, results = sweep_n_cargos(n_seeds=5, n_points=15)
    plot_assignment_comparison(
        cargo_range, results, "Число грузов",
        "Сравнение эвристик назначения: рост числа грузов (20 островов, 4 пары)",
        "outputs/assignment_comparison_n_cargos.png")
