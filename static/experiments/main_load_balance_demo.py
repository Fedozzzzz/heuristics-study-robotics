"""
ЭКСПЕРИМЕНТ: почему LPT справляется с БАЛАНСИРОВКОЙ НАГРУЗКИ лучше, чем
Greedy Nearest Assignment - несмотря на то, что Greedy Nearest даёт меньшую
СУММУ (Φ).

Это two-objective trade-off:
  - СУММА (Φ = Σ_k load(U_k))     - то, что мы хотим минимизировать в задаче
  - МАКСИМУМ (Cmax = max_k load(U_k)) - makespan, то, что доказанно
                                          минимизирует LPT (Graham, 1969)

LPT явно ОПТИМИЗИРОВАН для минимизации Cmax (через выравнивание загрузки
между "машинами"), поэтому он систематически даёт более РАВНОМЕРНОЕ
распределение реальной нагрузки между парами. Greedy Nearest же жадно
хватает "выгодные прямо сейчас" грузы для одной и той же пары подряд,
если она оказывается удачно расположена - это даёт меньшую сумму, но может
сильно перегрузить одну конкретную пару относительно других.

Визуализация:
  1. bar_chart_loads.png      - столбчатая диаграмма загрузки каждой пары
                                 для одного представительного сценария (3 метода)
  2. tradeoff_sum_vs_max.png  - точечный график (Φ, Cmax) для всех методов
                                 на серии случайных сценариев - явно показывает
                                 Парето-обмен между двумя целями
  3. sweep_balance_metric.png - как меняется СТАНДАРТНОЕ ОТКЛОНЕНИЕ загрузки
                                 (мера балансировки) с ростом числа грузов
                                 для всех трёх методов

ЗАПУСК: python3 main_load_balance_demo.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))

from collections import defaultdict
import statistics

import matplotlib.pyplot as plt

import scenario_generator as sg
from lpt_assignment import assign_round_robin, assign_by_lpt, assign_greedy_nearest, apply_assignment
from algorithm_2 import run_scheduling


L_FOR_ALL_SCENARIOS = 6.0

METHODS = {
    "round_robin": ("Round-Robin", "#7f8c8d"),
    "lpt": ("LPT (баланс максимума)", "#c0392b"),
    "greedy_nearest": ("Greedy Nearest (минимизация суммы)", "#2b6cb0"),
}


def get_loads(env, cargos, pairs, method_key: str, L: float):
    """Возвращает {pair_id: реальная_загрузка} для метода method_key."""
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
    if not outcome.all_delivered:
        return None

    load = defaultdict(float)
    for e in outcome.schedule:
        load[e.pair_id] += e.result.W_d + e.result.W_b
    # гарантируем, что все пары присутствуют, даже с нулевой загрузкой
    for p in pairs:
        load.setdefault(p.id, 0.0)
    return dict(load)


# ---------------------------------------------------------------------------
# 1. Столбчатая диаграмма загрузки по парам для одного сценария
# ---------------------------------------------------------------------------

def plot_bar_chart_loads(env, cargos, pairs, L: float, filename: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    pair_ids = [p.id for p in pairs]
    x = range(len(pair_ids))

    for ax, key in zip(axes, METHODS):
        label, color = METHODS[key]
        loads = get_loads(env, cargos, pairs, key, L)
        values = [loads[pid] for pid in pair_ids]

        bars = ax.bar(x, values, color=color, alpha=0.85, edgecolor="black", linewidth=0.8)
        mean_load = sum(values) / len(values)
        ax.axhline(mean_load, color="black", linestyle="--", linewidth=1,
                   label=f"среднее = {mean_load:.1f}")

        total = sum(values)
        cmax = max(values)
        std = statistics.pstdev(values)

        for bar, v in zip(bars, values):
            ax.annotate(f"{v:.1f}", (bar.get_x() + bar.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=9)

        ax.set_xticks(list(x))
        ax.set_xticklabels(pair_ids, fontsize=9)
        ax.set_title(f"{label}\nΦ(сумма)={total:.1f}  Cmax(максимум)={cmax:.1f}\n"
                     f"std(разброс)={std:.1f}",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Загрузка пары (W_d + W_b), сумма по её грузам")
    fig.suptitle("Реальная загрузка каждой пары: сумма vs максимум vs разброс",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


# ---------------------------------------------------------------------------
# 2. Trade-off (Φ, Cmax) на серии случайных сценариев
# ---------------------------------------------------------------------------

def collect_tradeoff_points(n_seeds: int = 30, n_islands: int = 20, n_pairs: int = 4,
                             n_cargos: int = 16, L: float = L_FOR_ALL_SCENARIOS):
    """Для каждого seed и каждого метода считает (Φ, Cmax) - точка на графике trade-off."""
    points = {key: {"phi": [], "cmax": []} for key in METHODS}

    for seed in range(n_seeds):
        env = sg.build_environment(n_islands=n_islands, seed=seed)
        cargos, pairs = sg.build_cargos_and_pairs(n_islands=n_islands, n_pairs=n_pairs,
                                                   n_cargos=n_cargos, seed=seed)
        for key in METHODS:
            loads = get_loads(env, cargos, pairs, key, L)
            if loads is None:
                continue
            values = list(loads.values())
            points[key]["phi"].append(sum(values))
            points[key]["cmax"].append(max(values))

    return points


def plot_tradeoff(points, filename: str):
    fig, ax = plt.subplots(figsize=(8, 7))

    for key, (label, color) in METHODS.items():
        phi = points[key]["phi"]
        cmax = points[key]["cmax"]
        ax.scatter(phi, cmax, color=color, alpha=0.6, s=50, label=label,
                   edgecolor="black", linewidth=0.5)
        # отметим среднюю точку метода крупным маркером
        if phi:
            mean_phi = sum(phi) / len(phi)
            mean_cmax = sum(cmax) / len(cmax)
            ax.scatter([mean_phi], [mean_cmax], color=color, s=300, marker="*",
                       edgecolor="black", linewidth=1.5, zorder=5)

    ax.set_xlabel("Φ — сумма загрузок всех пар (то, что хотим минимизировать)")
    ax.set_ylabel("Cmax — максимальная загрузка одной пары (makespan)")
    ax.set_title("Trade-off между минимизацией суммы и минимизацией максимума\n"
                 "(точки - отдельные случайные сценарии, звёзды - средние значения)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


# ---------------------------------------------------------------------------
# 3. Зависимость разброса (std) загрузки от масштаба задачи
# ---------------------------------------------------------------------------

def sweep_balance_metric(n_seeds: int = 5, n_points: int = 15):
    n_islands = 20
    n_pairs = 4
    cargo_range = list(range(4, 4 + 2 * n_points, 2))[:n_points]

    std_results = {key: [] for key in METHODS}
    cmax_results = {key: [] for key in METHODS}
    phi_results = {key: [] for key in METHODS}

    for n_cargos in cargo_range:
        std_runs = {key: [] for key in METHODS}
        cmax_runs = {key: [] for key in METHODS}
        phi_runs = {key: [] for key in METHODS}

        for seed in range(n_seeds):
            env = sg.build_environment(n_islands=n_islands, seed=seed)
            cargos, pairs = sg.build_cargos_and_pairs(n_islands=n_islands, n_pairs=n_pairs,
                                                       n_cargos=n_cargos, seed=seed)
            for key in METHODS:
                loads = get_loads(env, cargos, pairs, key, L_FOR_ALL_SCENARIOS)
                if loads is None:
                    continue
                values = list(loads.values())
                std_runs[key].append(statistics.pstdev(values))
                cmax_runs[key].append(max(values))
                phi_runs[key].append(sum(values))

        for key in METHODS:
            std_results[key].append(sum(std_runs[key]) / len(std_runs[key]) if std_runs[key] else None)
            cmax_results[key].append(sum(cmax_runs[key]) / len(cmax_runs[key]) if cmax_runs[key] else None)
            phi_results[key].append(sum(phi_runs[key]) / len(phi_runs[key]) if phi_runs[key] else None)

        line = f"n_cargos={n_cargos}: "
        for key in METHODS:
            line += f"{key}(std={std_results[key][-1]:.2f}, Cmax={cmax_results[key][-1]:.1f})  "
        print(line)

    return cargo_range, std_results, cmax_results, phi_results


def plot_balance_sweep(x_values, std_results, cmax_results, x_label, filename: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for key, (label, color) in METHODS.items():
        valid_x = [x for x, v in zip(x_values, std_results[key]) if v is not None]
        valid_y = [v for v in std_results[key] if v is not None]
        ax.plot(valid_x, valid_y, marker="o", markersize=4, label=label,
                color=color, linewidth=1.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Стандартное отклонение загрузки между парами")
    ax.set_title("Разброс нагрузки (меньше = равномернее распределение)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    for key, (label, color) in METHODS.items():
        valid_x = [x for x, v in zip(x_values, cmax_results[key]) if v is not None]
        valid_y = [v for v in cmax_results[key] if v is not None]
        ax.plot(valid_x, valid_y, marker="o", markersize=4, label=label,
                color=color, linewidth=1.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cmax — максимальная загрузка (makespan)")
    ax.set_title("Максимальная загрузка одной пары")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Почему LPT лучше балансирует нагрузку, чем Greedy Nearest",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # --- 1. Столбчатая диаграмма для одного представительного сценария ---
    print("=" * 70)
    print("1. Столбчатая диаграмма загрузки (один сценарий)")
    print("=" * 70)
    env_demo = sg.build_environment(n_islands=20, seed=0)
    cargos_demo, pairs_demo = sg.build_cargos_and_pairs(n_islands=20, n_pairs=4,
                                                          n_cargos=16, seed=0)
    plot_bar_chart_loads(env_demo, cargos_demo, pairs_demo, L_FOR_ALL_SCENARIOS,
                          "outputs/bar_chart_loads.png")

    # --- 2. Trade-off на серии случайных сценариев ---
    print("\n" + "=" * 70)
    print("2. Trade-off (Φ, Cmax) на 30 случайных сценариях")
    print("=" * 70)
    points = collect_tradeoff_points(n_seeds=30)
    for key, (label, _) in METHODS.items():
        phi_mean = sum(points[key]["phi"]) / len(points[key]["phi"])
        cmax_mean = sum(points[key]["cmax"]) / len(points[key]["cmax"])
        print(f"{label}: средняя Φ={phi_mean:.2f}, средний Cmax={cmax_mean:.2f}")
    plot_tradeoff(points, "outputs/tradeoff_sum_vs_max.png")

    # --- 3. Серия по масштабу: разброс и максимум от числа грузов ---
    print("\n" + "=" * 70)
    print("3. Разброс и максимум загрузки vs число грузов (15 точек)")
    print("=" * 70)
    cargo_range, std_results, cmax_results, phi_results = sweep_balance_metric(
        n_seeds=5, n_points=15)
    plot_balance_sweep(cargo_range, std_results, cmax_results, "Число грузов",
                        "outputs/sweep_balance_metric.png")
