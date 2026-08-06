"""
СРАВНЕНИЕ ТРЁХ ЭВРИСТИК ПРИОРИТЕТА:
  1) ПРЯМАЯ:        p(c_i) = W_T^i                  (дорогие первыми)
  2) ОБРАТНАЯ:       p(c_i) = 1 / W_T^i               (дешёвые первыми)
  3) НОРМИРОВАННАЯ:  p(c_i) = W_T^i / Σ_j W_T^j        (доля общего бюджета,
                     дорогие первыми)

ВАЖНОЕ МАТЕМАТИЧЕСКОЕ ЗАМЕЧАНИЕ: нормировка на сумму - это умножение всех
значений приоритета на одну и ту же положительную константу. Такое
преобразование НЕ МЕНЯЕТ порядок сортировки задач. Поэтому порядок
выполнения и итоговые суммы (estimated_total, real_total) для эвристик
"ПРЯМАЯ" и "НОРМИРОВАННАЯ" должны быть ЧИСЛЕННО ИДЕНТИЧНЫ - они дают
одно и то же расписание, отличаются только показанные абсолютные значения
приоритета на графике приоритетов. Это проверяется явно в коде ниже
(assert на совпадение порядка и итоговых сумм).

Для серий по числу грузов/островов используется ПЛОТНАЯ сетка (30+ точек)
по запросу - чтобы увидеть тонкую структуру зависимости расхождения от
масштаба задачи, а не только несколько грубых отсчётов.

ЗАПУСК: python3 main_priority_comparison.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))
import matplotlib.pyplot as plt

from scenario_generator import build_environment, build_cargos_and_pairs
from priority_evaluation import (estimate_task_costs, compute_direct_priority,
                                  compute_inverse_priority, compute_normalized_priority,
                                  assign_priority_levels_from_priority,
                                  run_sequential_by_priority)
from visualize import plot_priority_graph


L_FOR_ALL_SCENARIOS = 6.0
K_LEVELS = 4

HEURISTICS = {
    "direct": ("Прямая  p=W_T^i", lambda tc: compute_direct_priority(tc), "#c0392b"),
    "inverse": ("Обратная  p=1/W_T^i", lambda tc: compute_inverse_priority(tc), "#2b6cb0"),
    "normalized": ("Нормированная  p=W_T^i/ΣW_T^j", lambda tc: compute_normalized_priority(tc, direction="expensive_first"), "#16a085"),
}


def run_one_scenario(heuristic_key: str, n_islands: int, n_pairs: int,
                      n_cargos: int, seed: int = 0):
    env = build_environment(n_islands=n_islands, seed=seed)
    cargos, pairs = build_cargos_and_pairs(n_islands=n_islands, n_pairs=n_pairs,
                                            n_cargos=n_cargos, seed=seed)

    task_costs = estimate_task_costs(env, cargos, pairs, L=L_FOR_ALL_SCENARIOS)
    _, priority_fn, _ = HEURISTICS[heuristic_key]
    priority = priority_fn(task_costs)
    levels = assign_priority_levels_from_priority(cargos, priority, k_levels=K_LEVELS)

    outcome = run_sequential_by_priority(env, cargos, pairs, L=L_FOR_ALL_SCENARIOS,
                                          priority=priority, task_costs=task_costs)

    return {
        "env": env, "cargos": cargos, "pairs": pairs,
        "task_costs": task_costs, "priority": priority, "levels": levels,
        "outcome": outcome,
    }


def verify_direct_equals_normalized():
    """Проверяет математическое утверждение из docstring: порядок и суммы
    для 'direct' и 'normalized' должны совпадать численно."""
    data_d = run_one_scenario("direct", n_islands=15, n_pairs=3, n_cargos=9, seed=1)
    data_n = run_one_scenario("normalized", n_islands=15, n_pairs=3, n_cargos=9, seed=1)

    order_d = [e.cargo_id for e in data_d["outcome"].entries]
    order_n = [e.cargo_id for e in data_n["outcome"].entries]

    print("=" * 70)
    print("ПРОВЕРКА: 'прямая' и 'нормированная' дают идентичный порядок/суммы?")
    print("=" * 70)
    print(f"Порядок (прямая):       {order_d}")
    print(f"Порядок (нормированная): {order_n}")
    print(f"Порядки совпадают: {order_d == order_n}")
    print(f"estimated_total: прямая={data_d['outcome'].estimated_total:.4f}  "
          f"нормированная={data_n['outcome'].estimated_total:.4f}")
    print(f"real_total:      прямая={data_d['outcome'].real_total:.4f}  "
          f"нормированная={data_n['outcome'].real_total:.4f}")
    assert order_d == order_n, "Порядки должны совпадать!"
    assert abs(data_d["outcome"].real_total - data_n["outcome"].real_total) < 1e-6
    print("✓ Подтверждено: расписания идентичны (как и должно быть математически)\n")


def demo_priority_graphs():
    """Граф приоритетов для всех трёх эвристик на одном демо-сценарии."""
    import os
    os.makedirs("outputs", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))

    for ax, key in zip(axes, ["direct", "inverse", "normalized"]):
        label, _, _ = HEURISTICS[key]
        data = run_one_scenario(key, n_islands=15, n_pairs=3, n_cargos=9, seed=1)
        plot_priority_graph(data["cargos"], data["levels"], data["priority"],
                             title=f"{label}", ax=ax, show=False)

    fig.suptitle("Граф приоритетов: сравнение трёх эвристик (15 островов, 3 пары, 9 грузов)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig("outputs/priority_graphs_all_three.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Сохранено: priority_graphs_all_three.png")


def sweep_n_cargos_dense(n_seeds: int = 5, n_points: int = 30):
    """
    ПЛОТНАЯ сетка по числу грузов (n_points точек, по умолчанию 30) для
    всех трёх эвристик одновременно. Острова и пары фиксированы.
    """
    n_islands = 20
    n_pairs = 4
    cargo_range = list(range(4, 4 + 2 * n_points, 2))[:n_points]  # 4,6,8,...,~62

    results = {key: {"est": [], "real": [], "gap": []} for key in HEURISTICS}

    for n_cargos in cargo_range:
        for key in HEURISTICS:
            est_runs, real_runs, gap_runs = [], [], []
            for seed in range(n_seeds):
                data = run_one_scenario(key, n_islands=n_islands, n_pairs=n_pairs,
                                         n_cargos=n_cargos, seed=seed)
                outcome = data["outcome"]
                if not outcome.all_delivered:
                    continue
                est_runs.append(outcome.estimated_total)
                real_runs.append(outcome.real_total)
                gap_runs.append(100 * (outcome.real_total / outcome.estimated_total - 1))

            if not est_runs:
                results[key]["est"].append(None)
                results[key]["real"].append(None)
                results[key]["gap"].append(None)
                continue

            results[key]["est"].append(sum(est_runs) / len(est_runs))
            results[key]["real"].append(sum(real_runs) / len(real_runs))
            results[key]["gap"].append(sum(gap_runs) / len(gap_runs))

        print(f"n_cargos={n_cargos}: " + "  ".join(
            f"{key}={results[key]['gap'][-1]:+.2f}%" if results[key]["gap"][-1] is not None
            else f"{key}=N/A" for key in HEURISTICS))

    return cargo_range, results


def sweep_n_islands_dense(n_seeds: int = 5, n_points: int = 30):
    """ПЛОТНАЯ сетка по числу островов (n_points точек) для всех трёх эвристик."""
    n_pairs = 4
    n_cargos = 16
    islands_range = list(range(10, 10 + 2 * n_points, 2))[:n_points]  # 10,12,...,~68

    results = {key: {"est": [], "real": [], "gap": []} for key in HEURISTICS}

    for n_islands in islands_range:
        for key in HEURISTICS:
            est_runs, real_runs, gap_runs = [], [], []
            for seed in range(n_seeds):
                data = run_one_scenario(key, n_islands=n_islands, n_pairs=n_pairs,
                                         n_cargos=n_cargos, seed=seed)
                outcome = data["outcome"]
                if not outcome.all_delivered:
                    continue
                est_runs.append(outcome.estimated_total)
                real_runs.append(outcome.real_total)
                gap_runs.append(100 * (outcome.real_total / outcome.estimated_total - 1))

            if not est_runs:
                results[key]["est"].append(None)
                results[key]["real"].append(None)
                results[key]["gap"].append(None)
                continue

            results[key]["est"].append(sum(est_runs) / len(est_runs))
            results[key]["real"].append(sum(real_runs) / len(real_runs))
            results[key]["gap"].append(sum(gap_runs) / len(gap_runs))

        print(f"n_islands={n_islands}: " + "  ".join(
            f"{key}={results[key]['gap'][-1]:+.2f}%" if results[key]["gap"][-1] is not None
            else f"{key}=N/A" for key in HEURISTICS))

    return islands_range, results


def plot_comparison(x_values, results, x_label, title, filename):
    """Два графика: (1) эвристика vs реальность для всех трёх героев,
    (2) относительное расхождение для всех трёх героев на одном поле."""
    import os
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    for key in HEURISTICS:
        label, _, color = HEURISTICS[key]
        valid_x = [x for x, e in zip(x_values, results[key]["est"]) if e is not None]
        valid_real = [r for r in results[key]["real"] if r is not None]
        ax.plot(valid_x, valid_real, marker="o", markersize=4, label=label,
                color=color, linewidth=1.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Реальная суммарная стоимость")
    ax.set_title("Реальная стоимость для трёх эвристик")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    for key in HEURISTICS:
        label, _, color = HEURISTICS[key]
        valid_x = [x for x, g in zip(x_values, results[key]["gap"]) if g is not None]
        valid_gap = [g for g in results[key]["gap"] if g is not None]
        ax.plot(valid_x, valid_gap, marker="o", markersize=4, label=label,
                color=color, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Расхождение, % от эвристической оценки")
    ax.set_title("Относительное расхождение (реальность - эвристика)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def plot_estimated_vs_real_single(x_values, results, heuristic_key, x_label,
                                   title, filename):
    """
    Один график с ДВУМЯ линиями для ОДНОЙ эвристики:
      - эвристическая оценка (estimated) - сумма W_T^i, посчитанная от
        начальной позиции пары, ДО фактического выполнения
      - реальная стоимость (real) - сумма, полученная после фактического
        последовательного выполнения с учётом смещения пары
    """
    import os
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    label, _, color = HEURISTICS[heuristic_key]

    valid_x = [x for x, e in zip(x_values, results[heuristic_key]["est"]) if e is not None]
    valid_est = [e for e in results[heuristic_key]["est"] if e is not None]
    valid_real = [r for r in results[heuristic_key]["real"] if r is not None]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(valid_x, valid_est, marker="o", markersize=4,
            label="Эвристическая оценка (estimated)", color=color, linewidth=1.8)
    ax.plot(valid_x, valid_real, marker="s", markersize=4,
            label="Реальная стоимость (real)", color=color, linewidth=1.8,
            linestyle="--", alpha=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Суммарная стоимость")
    ax.set_title(f"{title}\n{label}", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def plot_deviation_single(x_values, results, heuristic_key, x_label, title, filename):
    """
    Отдельный график ОТКЛОНЕНИЯ реальной стоимости от эвристической оценки
    для ОДНОЙ эвристики: gap(x) = 100 * (real(x)/estimated(x) - 1), %.
    Положительное значение - реальность ХУЖЕ (дороже) предсказания;
    отрицательное - реальность ЛУЧШЕ (дешевле) предсказания.
    """
    import os
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    label, _, color = HEURISTICS[heuristic_key]

    valid_x = [x for x, g in zip(x_values, results[heuristic_key]["gap"]) if g is not None]
    valid_gap = [g for g in results[heuristic_key]["gap"] if g is not None]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(valid_x, valid_gap, marker="D", markersize=5, color=color, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.fill_between(valid_x, valid_gap, 0, color=color, alpha=0.15)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Отклонение, % от эвристической оценки\n(реальность - эвристика)")
    ax.set_title(f"{title}\n{label}", fontweight="bold", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    verify_direct_equals_normalized()
    demo_priority_graphs()

    print("\n" + "=" * 70)
    print("СЕРИЯ 1: рост числа грузов (30 точек, 20 островов, 4 пары)")
    print("=" * 70)
    cargo_range, results_cargos = sweep_n_cargos_dense(n_seeds=5, n_points=30)
    plot_comparison(cargo_range, results_cargos, "Число грузов",
                     "Сравнение трёх эвристик: рост числа грузов (20 островов, 4 пары)",
                     "outputs/comparison_3heuristics_n_cargos.png")

    plot_estimated_vs_real_single(
        cargo_range, results_cargos, "direct", "Число грузов",
        "Прямая эвристика: эвристическая оценка vs реальность (рост числа грузов)",
        "outputs/direct_estimated_vs_real_n_cargos.png")
    plot_estimated_vs_real_single(
        cargo_range, results_cargos, "inverse", "Число грузов",
        "Обратная эвристика: эвристическая оценка vs реальность (рост числа грузов)",
        "outputs/inverse_estimated_vs_real_n_cargos.png")

    print("\n" + "=" * 70)
    print("СЕРИЯ 1b: рост числа грузов, УВЕЛИЧЕННАЯ статистика (30 точек, n_seeds=20)")
    print("=" * 70)
    cargo_range_20, results_cargos_20 = sweep_n_cargos_dense(n_seeds=20, n_points=30)
    plot_comparison(cargo_range_20, results_cargos_20, "Число грузов",
                     "Сравнение трёх эвристик: рост числа грузов "
                     "(20 островов, 4 пары, усреднено по 20 seed)",
                     "outputs/comparison_3heuristics_n_cargos_averaged20.png")

    plot_estimated_vs_real_single(
        cargo_range_20, results_cargos_20, "direct", "Число грузов",
        "Прямая эвристика: эвристическая оценка vs реальность\n"
        "(рост числа грузов, усреднено по 20 seed)",
        "outputs/direct_estimated_vs_real_n_cargos_averaged20.png")
    plot_deviation_single(
        cargo_range_20, results_cargos_20, "direct", "Число грузов",
        "Прямая эвристика: отклонение реальности от оценки\n"
        "(рост числа грузов, усреднено по 20 seed)",
        "outputs/direct_deviation_n_cargos_averaged20.png")

    print("\n" + "=" * 70)
    print("СЕРИЯ 2: рост числа островов (30 точек, 4 пары, 16 грузов)")
    print("=" * 70)
    islands_range, results_islands = sweep_n_islands_dense(n_seeds=5, n_points=30)
    plot_comparison(islands_range, results_islands, "Число островов",
                     "Сравнение трёх эвристик: рост числа островов (4 пары, 16 грузов)",
                     "outputs/comparison_3heuristics_n_islands.png")

    plot_estimated_vs_real_single(
        islands_range, results_islands, "direct", "Число островов",
        "Прямая эвристика: эвристическая оценка vs реальность (рост числа островов)",
        "outputs/direct_estimated_vs_real_n_islands.png")
    plot_estimated_vs_real_single(
        islands_range, results_islands, "inverse", "Число островов",
        "Обратная эвристика: эвристическая оценка vs реальность (рост числа островов)",
        "outputs/inverse_estimated_vs_real_n_islands.png")
