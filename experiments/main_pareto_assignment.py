"""
ПАРЕТО-ФРОНТ МЕЖДУ Φ (суммой загрузок) И Cmax (максимальной загрузкой):
управление балансом через параметр λ ∈ [0,1] эвристики ASSIGN-WEIGHTED.

ФОРМУЛА (см. lpt_assignment.assign_weighted, Algorithm 0''):

    U*(λ) = argmin_{U_k}  [ λ · ŵ_T(U_k, c_i)  +  (1-λ) · load̂(U_k) ]

где ŵ_T, load̂ ∈ [0,1] - стоимость назначения этого груза этой паре и текущая
накопленная загрузка пары, обе нормированные относительно min/max среди
ВСЕХ пар-кандидатов на ЭТОМ шаге (это позволяет складывать величины разного
масштаба - см. формулы в коде).

λ=1 -> чистая минимизация стоимости назначения (= Greedy Nearest Assignment,
       минимизирует Φ)
λ=0 -> чистая минимизация текущей загрузки (балансировка, аналог LPT,
       минимизирует Cmax)
λ∈(0,1) -> компромисс

Перебор λ по сетке даёт МНОЖЕСТВО точек (Φ(λ), Cmax(λ)); из них выделяется
Парето-фронт (недоминируемое подмножество) - именно эти точки и есть
"эффективные" компромиссы, среди которых стоит выбирать.

ЗАПУСК: python3 main_pareto_assignment.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt

import scenario_generator as sg
from lpt_assignment import assign_weighted, apply_assignment
from algorithm_2 import run_scheduling


L_FOR_ALL_SCENARIOS = 6.0


@dataclass
class LambdaPoint:
    lam: float
    phi: float       # сумма загрузок (Φ)
    cmax: float       # максимальная загрузка (Cmax)


def build_lambda_curve(env, cargos, pairs, L: float, n_points: int = 41) -> List[LambdaPoint]:
    """
    Шаг 1: прогоняет ASSIGN-WEIGHTED на сетке λ ∈ [0,1] из n_points точек,
    для каждой λ считает реальную (Φ, Cmax) после полного выполнения
    расписания (Алгоритм 2).
    """
    points = []
    for i in range(n_points):
        lam = i / (n_points - 1)
        assignment = assign_weighted(cargos, pairs, env, L, lam=lam)
        cargos_a = apply_assignment(cargos, assignment)
        outcome = run_scheduling(env, cargos_a, pairs, L=L)
        if not outcome.all_delivered:
            continue
        load = defaultdict(float)
        for e in outcome.schedule:
            load[e.pair_id] += e.result.W_d + e.result.W_b
        values = list(load.values())
        points.append(LambdaPoint(lam=lam, phi=sum(values), cmax=max(values)))
    return points


def pareto_front_2d(points: List[LambdaPoint]) -> List[LambdaPoint]:
    """
    Шаг 2: выделяет недоминируемое подмножество (минимизация ОБЕИХ осей -
    Φ и Cmax одновременно), аналогично algorithm_4.pareto_front, но для
    пары критериев (Φ, Cmax) вместо (W_d_total, W_b_total).
    """
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q.phi <= p.phi and q.cmax <= p.cmax and (q.phi < p.phi or q.cmax < p.cmax):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda p: p.phi)
    return front


def plot_pareto_lambda(points: List[LambdaPoint], front: List[LambdaPoint],
                        title: str, filename: str):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    all_phi = [p.phi for p in points]
    all_cmax = [p.cmax for p in points]
    ax.scatter(all_phi, all_cmax, color="#b0b0b0", s=40, alpha=0.7,
               label="Все λ ∈ [0,1] (полная сетка)", zorder=2)

    front_phi = [p.phi for p in front]
    front_cmax = [p.cmax for p in front]
    ax.plot(front_phi, front_cmax, color="#d9822b", linewidth=2, zorder=3,
            marker="o", markersize=8, label="Парето-фронт (недоминируемые λ)")

    for p in front:
        ax.annotate(f"λ={p.lam:.2f}", (p.phi, p.cmax), textcoords="offset points",
                    xytext=(8, 6), fontsize=8)

    # отметим крайние случаи явно
    lam0 = min(points, key=lambda p: abs(p.lam - 0.0))
    lam1 = min(points, key=lambda p: abs(p.lam - 1.0))
    ax.scatter([lam0.phi], [lam0.cmax], color="#c0392b", s=250, marker="*",
               edgecolor="black", linewidth=1.2, zorder=5,
               label=f"λ=0 (баланс, ~LPT)")
    ax.scatter([lam1.phi], [lam1.cmax], color="#2b6cb0", s=250, marker="*",
               edgecolor="black", linewidth=1.2, zorder=5,
               label=f"λ=1 (Greedy Nearest)")

    ax.set_xlabel("Φ — сумма загрузок всех пар")
    ax.set_ylabel("Cmax — максимальная загрузка одной пары")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def plot_lambda_sweep(points: List[LambdaPoint], filename: str,
                       title: str = "Φ(λ) и Cmax(λ): как параметр λ перестраивает решение"):
    """Дополнительный график: Φ(λ) и Cmax(λ) как функции λ напрямую -
    показывает, как именно перестройка происходит при изменении λ."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 6))

    lams = [p.lam for p in points]
    phis = [p.phi for p in points]
    cmaxs = [p.cmax for p in points]

    ax1.plot(lams, phis, color="#2b6cb0", marker="o", markersize=3,
             linewidth=1.8, label="Φ (сумма)")
    ax1.set_xlabel("λ  (0 = баланс нагрузки, 1 = минимизация суммы)")
    ax1.set_ylabel("Φ — сумма загрузок", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")

    ax2 = ax1.twinx()
    ax2.plot(lams, cmaxs, color="#c0392b", marker="s", markersize=3,
             linewidth=1.8, label="Cmax (максимум)")
    ax2.set_ylabel("Cmax — максимальная загрузка", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    env = sg.build_environment(n_islands=20, seed=0)
    cargos, pairs = sg.build_cargos_and_pairs(n_islands=20, n_pairs=4, n_cargos=16, seed=0)

    print("=" * 70)
    print("Построение Парето-фронта (Φ, Cmax) через параметр λ")
    print("=" * 70)
    points = build_lambda_curve(env, cargos, pairs, L_FOR_ALL_SCENARIOS, n_points=41)
    print(f"Точек на сетке λ: {len(points)}")

    front = pareto_front_2d(points)
    print(f"Точек на Парето-фронте: {len(front)}")
    for p in front:
        print(f"  λ={p.lam:.3f}: Φ={p.phi:.2f}  Cmax={p.cmax:.2f}")

    plot_pareto_lambda(points, front,
                        "Парето-фронт: сумма загрузок (Φ) vs максимальная загрузка (Cmax)\n"
                        "(20 островов, 4 пары, 16 грузов)",
                        "outputs/pareto_phi_vs_cmax.png")
    plot_lambda_sweep(points, "outputs/lambda_sweep_phi_cmax.png",
                       title="Φ(λ) и Cmax(λ) — один сценарий (seed=0)")

    # --- усреднённый фронт по нескольким случайным сценариям ---
    print("\n" + "=" * 70)
    print("Усреднённый Парето-фронт по 10 случайным сценариям")
    print("=" * 70)
    n_seeds = 10
    all_points_by_lambda = defaultdict(lambda: {"phi": [], "cmax": []})

    for seed in range(n_seeds):
        env_s = sg.build_environment(n_islands=20, seed=seed)
        cargos_s, pairs_s = sg.build_cargos_and_pairs(n_islands=20, n_pairs=4,
                                                       n_cargos=16, seed=seed)
        pts = build_lambda_curve(env_s, cargos_s, pairs_s, L_FOR_ALL_SCENARIOS, n_points=21)
        for p in pts:
            all_points_by_lambda[round(p.lam, 3)]["phi"].append(p.phi)
            all_points_by_lambda[round(p.lam, 3)]["cmax"].append(p.cmax)

    averaged_points = []
    for lam, vals in sorted(all_points_by_lambda.items()):
        if vals["phi"]:
            mean_phi = sum(vals["phi"]) / len(vals["phi"])
            mean_cmax = sum(vals["cmax"]) / len(vals["cmax"])
            averaged_points.append(LambdaPoint(lam=lam, phi=mean_phi, cmax=mean_cmax))
            print(f"  λ={lam:.2f}: Φ_mean={mean_phi:.2f}  Cmax_mean={mean_cmax:.2f}")

    averaged_front = pareto_front_2d(averaged_points)
    plot_pareto_lambda(averaged_points, averaged_front,
                        f"Усреднённый Парето-фронт по {n_seeds} случайным сценариям\n"
                        "(20 островов, 4 пары, 16 грузов)",
                        "outputs/pareto_phi_vs_cmax_averaged.png")

    plot_lambda_sweep(averaged_points, "outputs/lambda_sweep_phi_cmax_averaged.png",
                       title=f"Φ(λ) и Cmax(λ) — усреднено по {n_seeds} случайным сценариям")
