"""
ОЦЕНКА ЭВРИСТИКИ ПРИОРИТЕТА ПО РАССТОЯНИЮ (формула 2.4):

    p(c_i) = 1 / max( dist(доставщик, v_s^i), dist(строитель, v_s^i) )

Сравнивает ПРЯМОЕ (евклидово) расстояние "по воздуху" от текущей позиции
доставщика до точки погрузки груза с РЕАЛЬНЫМ расстоянием по графу (через
существующие/построенные переправы). Чем ближе "по воздуху" к "по графу",
тем надёжнее эта дешёвая (не требующая решения подзадачи маршрута) эвристика
как предиктор реальной стоимости перемещения.

ЗАПУСК: python3 main_distance_heuristic.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import matplotlib.pyplot as plt

import scenario_generator as sg
from distance_heuristic_evaluation import run_sequential_by_distance_priority
from visualize import plot_priority_graph


L_FOR_ALL_SCENARIOS = 6.0
K_LEVELS = 4


def demo_priority_graph():
    """Граф приоритетов (по формуле 2.4) для одного представительного сценария."""
    os.makedirs("outputs", exist_ok=True)
    env = sg.build_environment(n_islands=15, seed=1)
    cargos, pairs = sg.build_cargos_and_pairs(n_islands=15, n_pairs=3, n_cargos=9, seed=1)

    outcome, levels, priority = run_sequential_by_distance_priority(
        env, cargos, pairs, L=L_FOR_ALL_SCENARIOS, k_levels=K_LEVELS)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_priority_graph(cargos, levels, priority,
                         title="Граф приоритетов (формула 2.4: bottleneck по расстоянию)",
                         ax=ax, show=False)
    fig.tight_layout()
    fig.savefig("outputs/priority_graph_distance.png", dpi=150)
    plt.close(fig)

    print("=" * 70)
    print("Демо-сценарий: 15 островов, 3 пары, 9 грузов")
    print("=" * 70)
    print(f"Все доставлены: {outcome.all_delivered}")
    print(f"Эвристическая сумма расстояний (estimated_total): {outcome.estimated_total:.2f}")
    print(f"Реальная сумма расстояний (real_total):            {outcome.real_total:.2f}")
    gap = 100 * (outcome.real_total / outcome.estimated_total - 1) if outcome.estimated_total > 0 else 0
    print(f"Расхождение: {outcome.real_total - outcome.estimated_total:+.2f} ({gap:+.1f}%)")
    print("\nПорядок выполнения (по убыванию динамического приоритета):")
    for e in outcome.entries:
        diff = e.real_distance - e.estimated_distance
        print(f"  {e.cargo_id} ({e.pair_id}): est={e.estimated_distance:6.2f}  "
              f"real={e.real_distance:6.2f}  diff={diff:+6.2f}")
    print("\nСохранено: priority_graph_distance.png")


def sweep_n_cargos_dense(n_seeds: int = 5, n_points: int = 30):
    """Плотная сетка по числу грузов (острова и пары фиксированы)."""
    n_islands = 20
    n_pairs = 4
    cargo_range = list(range(4, 4 + 2 * n_points, 2))[:n_points]

    est_list, real_list, gap_list = [], [], []
    for n_cargos in cargo_range:
        est_runs, real_runs, gap_runs = [], [], []
        for seed in range(n_seeds):
            env = sg.build_environment(n_islands=n_islands, seed=seed)
            cargos, pairs = sg.build_cargos_and_pairs(n_islands=n_islands, n_pairs=n_pairs,
                                                       n_cargos=n_cargos, seed=seed)
            outcome, _, _ = run_sequential_by_distance_priority(
                env, cargos, pairs, L=L_FOR_ALL_SCENARIOS, k_levels=K_LEVELS)
            if not outcome.all_delivered or outcome.estimated_total <= 0:
                continue
            est_runs.append(outcome.estimated_total)
            real_runs.append(outcome.real_total)
            gap_runs.append(100 * (outcome.real_total / outcome.estimated_total - 1))

        if not est_runs:
            est_list.append(None); real_list.append(None); gap_list.append(None)
            print(f"n_cargos={n_cargos}: ни один seed не дал полной доставки, пропуск")
            continue

        est_mean = sum(est_runs) / len(est_runs)
        real_mean = sum(real_runs) / len(real_runs)
        gap_mean = sum(gap_runs) / len(gap_runs)
        est_list.append(est_mean); real_list.append(real_mean); gap_list.append(gap_mean)
        print(f"n_cargos={n_cargos}: estimated={est_mean:.2f}  real={real_mean:.2f}  "
              f"gap={gap_mean:+.2f}%  (усреднено по {len(est_runs)}/{n_seeds} seed)")

    return cargo_range, est_list, real_list, gap_list


def sweep_n_islands_dense(n_seeds: int = 5, n_points: int = 30):
    """Плотная сетка по числу островов (пары и грузы фиксированы)."""
    n_pairs = 4
    n_cargos = 16
    islands_range = list(range(10, 10 + 2 * n_points, 2))[:n_points]

    est_list, real_list, gap_list = [], [], []
    for n_islands in islands_range:
        est_runs, real_runs, gap_runs = [], [], []
        for seed in range(n_seeds):
            env = sg.build_environment(n_islands=n_islands, seed=seed)
            cargos, pairs = sg.build_cargos_and_pairs(n_islands=n_islands, n_pairs=n_pairs,
                                                       n_cargos=n_cargos, seed=seed)
            outcome, _, _ = run_sequential_by_distance_priority(
                env, cargos, pairs, L=L_FOR_ALL_SCENARIOS, k_levels=K_LEVELS)
            if not outcome.all_delivered or outcome.estimated_total <= 0:
                continue
            est_runs.append(outcome.estimated_total)
            real_runs.append(outcome.real_total)
            gap_runs.append(100 * (outcome.real_total / outcome.estimated_total - 1))

        if not est_runs:
            est_list.append(None); real_list.append(None); gap_list.append(None)
            print(f"n_islands={n_islands}: ни один seed не дал полной доставки, пропуск")
            continue

        est_mean = sum(est_runs) / len(est_runs)
        real_mean = sum(real_runs) / len(real_runs)
        gap_mean = sum(gap_runs) / len(gap_runs)
        est_list.append(est_mean); real_list.append(real_mean); gap_list.append(gap_mean)
        print(f"n_islands={n_islands}: estimated={est_mean:.2f}  real={real_mean:.2f}  "
              f"gap={gap_mean:+.2f}%  (усреднено по {len(est_runs)}/{n_seeds} seed)")

    return islands_range, est_list, real_list, gap_list


def plot_estimated_vs_real(x_values, estimated, real, gaps_pct, x_label, title, filename):
    """Два графика: (1) эвристика vs реальность - расстояние, (2) относительное расхождение."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    valid_x = [x for x, e in zip(x_values, estimated) if e is not None]
    valid_est = [e for e in estimated if e is not None]
    valid_real = [r for r in real if r is not None]
    valid_gap = [g for g in gaps_pct if g is not None]

    ax = axes[0]
    ax.plot(valid_x, valid_est, marker="o", markersize=4,
             label="Эвристическая оценка (прямое расстояние)", color="#2b6cb0", linewidth=1.8)
    ax.plot(valid_x, valid_real, marker="s", markersize=4,
             label="Реальное расстояние (по графу)", color="#c0392b", linewidth=1.8,
             linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Суммарное расстояние")
    ax.set_title("Эвристика vs реальность")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    ax.plot(valid_x, valid_gap, marker="D", markersize=4, color="#8e54a0", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Расхождение, % от эвристической оценки")
    ax.set_title("Относительное расхождение (реальность - эвристика)")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    demo_priority_graph()

    print("\n" + "=" * 70)
    print("СЕРИЯ 1: рост числа грузов (30 точек, 20 островов, 4 пары)")
    print("=" * 70)
    cargo_range, est1, real1, gap1 = sweep_n_cargos_dense(n_seeds=5, n_points=30)
    plot_estimated_vs_real(cargo_range, est1, real1, gap1, "Число грузов",
                            "Эвристика по расстоянию: рост числа грузов (20 островов, 4 пары)",
                            "outputs/distance_heuristic_n_cargos.png")

    print("\n" + "=" * 70)
    print("СЕРИЯ 2: рост числа островов (30 точек, 4 пары, 16 грузов)")
    print("=" * 70)
    islands_range, est2, real2, gap2 = sweep_n_islands_dense(n_seeds=5, n_points=30)
    plot_estimated_vs_real(islands_range, est2, real2, gap2, "Число островов",
                            "Эвристика по расстоянию: рост числа островов (4 пары, 16 грузов)",
                            "outputs/distance_heuristic_n_islands.png")
