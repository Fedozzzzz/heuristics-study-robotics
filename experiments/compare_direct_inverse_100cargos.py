"""
Сравнение ТРЁХ эвристик приоритета (ПРЯМАЯ, ОБРАТНАЯ, ОТНОШЕНИЕ Wb/Wd) с
реальной стоимостью выполнения, при росте числа грузов от n_pairs (=4) до
100 (шаг 1 - каждое целое число грузов своя точка), на фиксированных
20 островах / 4 парах, усреднённое по 30 seed на каждую точку.

Эвристики:
  direct   - p(c_i) = W_T^i              (дорогие первыми)
  inverse  - p(c_i) = 1 / W_T^i           (дешёвые первыми)
  ratio    - p(c_i) = W_b^i / W_d^i       ("приоритет через стоимость по типу
             агента" - см. постановку задачи; высокое значение = задача
             дорогая для строителей, но дешёвая для доставщиков)

Для каждой эвристики строятся:
  1) "Обычный" - реальная стоимость (сплошная) и эвристическая оценка
     (штрихованная) как функция числа грузов (отдельный файл на эвристику).
  2) "Относительный" - отклонение эвристической оценки от реальности,
     gap(n) = 100 * (1 - estimated_total/real_total), %, ВСЕ эвристики
     на одном общем графике.
     Положительное значение - оценка ВЫШЕ реальности (переоценка, "запас сверху").
     Отрицательное значение - оценка НИЖЕ реальности (недооценка).

Дополнительно весь прогон повторяется со ШТРАФОМ ЗА ПЕРЕХОД к следующему
грузу той же пары (compute_transition_penalty/apply_transition_penalty) -
файлы с суффиксом "_penalty".

ЗАПУСК: python3 compare_direct_inverse_100cargos.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
import matplotlib.pyplot as plt

from scenario_generator import build_environment, build_cargos_and_pairs
from priority_evaluation import (estimate_task_costs, compute_direct_priority,
                                  compute_inverse_priority, compute_ratio_priority,
                                  compute_transition_penalty, apply_transition_penalty,
                                  run_sequential_by_priority)

L_FOR_ALL_SCENARIOS = 6.0
N_ISLANDS = 20
N_PAIRS = 4
N_SEEDS = 30
N_CARGOS_MAX = 100

HEURISTICS = {
    "direct": ("Прямая  p=W_T^i  (дорогие первыми)",
               lambda tc: compute_direct_priority(tc), "#c0392b"),
    "inverse": ("Обратная  p=1/W_T^i  (дешёвые первыми)",
                lambda tc: compute_inverse_priority(tc), "#2b6cb0"),
    "ratio": ("Отношение  p=W_b^i/W_d^i  (дорого строителю первыми)",
              lambda tc: compute_ratio_priority(tc), "#27ae60"),
}


def run_one_scenario(heuristic_key: str, n_cargos: int, seed: int, use_penalty: bool = False):
    env = build_environment(n_islands=N_ISLANDS, seed=seed)
    cargos, pairs = build_cargos_and_pairs(n_islands=N_ISLANDS, n_pairs=N_PAIRS,
                                            n_cargos=n_cargos, seed=seed)

    task_costs = estimate_task_costs(env, cargos, pairs, L=L_FOR_ALL_SCENARIOS)
    if use_penalty:
        penalty = compute_transition_penalty(env, cargos, pairs, L=L_FOR_ALL_SCENARIOS)
        task_costs = apply_transition_penalty(task_costs, penalty)

    _, priority_fn, _ = HEURISTICS[heuristic_key]
    priority = priority_fn(task_costs)

    outcome = run_sequential_by_priority(env, cargos, pairs, L=L_FOR_ALL_SCENARIOS,
                                          priority=priority, task_costs=task_costs)
    return outcome


def sweep_n_cargos(n_seeds: int = N_SEEDS, use_penalty: bool = False):
    """
    n_cargos пробегает от N_PAIRS (=4, минимум - по одному грузу на пару)
    до N_CARGOS_MAX включительно, с шагом 1 (каждое целое число грузов -
    своя точка на графике).
    """
    cargo_range = list(range(N_PAIRS, N_CARGOS_MAX + 1))

    results = {key: {"est": [], "real": [], "gap": []} for key in HEURISTICS}

    for n_cargos in cargo_range:
        for key in HEURISTICS:
            est_runs, real_runs, gap_runs = [], [], []
            for seed in range(n_seeds):
                outcome = run_one_scenario(key, n_cargos=n_cargos, seed=seed,
                                            use_penalty=use_penalty)
                if not outcome.all_delivered:
                    continue
                est_runs.append(outcome.estimated_total)
                real_runs.append(outcome.real_total)
                gap_runs.append(100 * (outcome.estimated_total / outcome.real_total - 1))

            if not est_runs:
                results[key]["est"].append(None)
                results[key]["real"].append(None)
                results[key]["gap"].append(None)
                continue

            results[key]["est"].append(sum(est_runs) / len(est_runs))
            results[key]["real"].append(sum(real_runs) / len(real_runs))
            results[key]["gap"].append(sum(gap_runs) / len(gap_runs))

        print(f"n_cargos={n_cargos:3d}: " + "  ".join(
            f"{key}={results[key]['gap'][-1]:+.2f}%" if results[key]["gap"][-1] is not None
            else f"{key}=N/A" for key in HEURISTICS))

    return cargo_range, results


def plot_single_heuristic(x_values, results, heuristic_key, filename):
    """
    Один график для ОДНОЙ эвристики: реальная стоимость (сплошная) vs
    эвристическая оценка (штрихованная), плюс относительное отклонение
    в виде второй панели снизу - одно изображение, полностью описывающее
    качество эвристики.
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    label, _, color = HEURISTICS[heuristic_key]

    valid_x = [x for x, r in zip(x_values, results[heuristic_key]["real"]) if r is not None]
    valid_est = [e for e in results[heuristic_key]["est"] if e is not None]
    valid_real = [r for r in results[heuristic_key]["real"] if r is not None]
    valid_gap = [g for g in results[heuristic_key]["gap"] if g is not None]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})

    ax_top.plot(valid_x, valid_real, marker="o", markersize=4,
                label="Реальная стоимость", color=color, linewidth=1.8)
    ax_top.plot(valid_x, valid_est, marker="^", markersize=4,
                label="Эвристическая оценка", color=color, linewidth=1.6,
                linestyle="--", alpha=0.85)
    ax_top.set_ylabel("Суммарная стоимость")
    ax_top.set_title(f"{label}\nРеальная стоимость vs эвристическая оценка",
                      fontweight="bold", fontsize=11)
    ax_top.legend(fontsize=9, loc="upper left")
    ax_top.grid(True, linestyle="--", alpha=0.4)

    ax_bot.plot(valid_x, valid_gap, marker="D", markersize=4, color=color, linewidth=1.8)
    ax_bot.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax_bot.fill_between(valid_x, valid_gap, 0, color=color, alpha=0.15)
    ax_bot.set_xlabel("Число грузов")
    ax_bot.set_ylabel("Отклонение, %\n(оценка \u2212 реальность)")
    ax_bot.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Грузы {N_PAIRS}\u2192{N_CARGOS_MAX} "
        f"({N_ISLANDS} островов, {N_PAIRS} пары, усреднено по {N_SEEDS} seed)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def plot_gap_both(x_values, results, filename):
    """
    Отдельный график ТОЛЬКО отклонения (оценка - реальность), % - обе
    эвристики на одном поле, чтобы сравнить их поведение напрямую.
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))

    for key in HEURISTICS:
        label, _, color = HEURISTICS[key]
        valid_x = [x for x, g in zip(x_values, results[key]["gap"]) if g is not None]
        valid_gap = [g for g in results[key]["gap"] if g is not None]
        ax.plot(valid_x, valid_gap, marker="D", markersize=4, label=label,
                color=color, linewidth=1.8)
        ax.fill_between(valid_x, valid_gap, 0, color=color, alpha=0.15)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Число грузов")
    ax.set_ylabel("Отклонение, %\n(оценка \u2212 реальность)")
    ax.set_title(
        "Отклонение эвристической оценки от реальной стоимости\n"
        "(> 0: оценка переоценивает; < 0: оценка недооценивает)",
        fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Прямая vs Обратная vs Отношение: грузы {N_PAIRS}\u2192{N_CARGOS_MAX} "
        f"({N_ISLANDS} островов, {N_PAIRS} пары, усреднено по {N_SEEDS} seed)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    print("=" * 70)
    print(f"БЕЗ ШТРАФА: грузы {N_PAIRS}->{N_CARGOS_MAX}, "
          f"{N_ISLANDS} островов, {N_PAIRS} пары, {N_SEEDS} seed на точку")
    print("=" * 70)
    cargo_range, results = sweep_n_cargos(n_seeds=N_SEEDS, use_penalty=False)
    plot_single_heuristic(cargo_range, results, "direct",
                           f"outputs/direct_real_vs_estimate_{N_PAIRS}to{N_CARGOS_MAX}cargos.png")
    plot_single_heuristic(cargo_range, results, "inverse",
                           f"outputs/inverse_real_vs_estimate_{N_PAIRS}to{N_CARGOS_MAX}cargos.png")
    plot_single_heuristic(cargo_range, results, "ratio",
                           f"outputs/ratio_real_vs_estimate_{N_PAIRS}to{N_CARGOS_MAX}cargos.png")
    plot_gap_both(cargo_range, results,
                  f"outputs/gap_direct_vs_inverse_{N_PAIRS}to{N_CARGOS_MAX}cargos.png")

    print()
    print("=" * 70)
    print(f"СО ШТРАФОМ ЗА ПЕРЕХОД: грузы {N_PAIRS}->{N_CARGOS_MAX}, "
          f"{N_ISLANDS} островов, {N_PAIRS} пары, {N_SEEDS} seed на точку")
    print("=" * 70)
    cargo_range_p, results_p = sweep_n_cargos(n_seeds=N_SEEDS, use_penalty=True)
    plot_single_heuristic(cargo_range_p, results_p, "direct",
                           f"outputs/direct_real_vs_estimate_penalty_{N_PAIRS}to{N_CARGOS_MAX}cargos.png")
    plot_single_heuristic(cargo_range_p, results_p, "inverse",
                           f"outputs/inverse_real_vs_estimate_penalty_{N_PAIRS}to{N_CARGOS_MAX}cargos.png")
    plot_single_heuristic(cargo_range_p, results_p, "ratio",
                           f"outputs/ratio_real_vs_estimate_penalty_{N_PAIRS}to{N_CARGOS_MAX}cargos.png")
    plot_gap_both(cargo_range_p, results_p,
                  f"outputs/gap_direct_vs_inverse_penalty_{N_PAIRS}to{N_CARGOS_MAX}cargos.png")
