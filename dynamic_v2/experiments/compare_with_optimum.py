"""
Сравнение с нижней границей (оптимумом), а не только эвристик друг с другом.

Все остальные метрики в проекте (real, estimated_raw, gap_raw) сравнивают
эвристики между собой или с их же собственной эвристической оценкой -- нет
независимого ориентира "а насколько это вообще далеко от оптимума". Для
маленьких сценариев (мало островов, мало грузов, мало пар) задачу можно
решить ТОЧНО полным перебором (branch & bound с мемоизацией состояний, см.
`robot_delivery_v2/optimal.py`) и посчитать competitive ratio:

    competitive_ratio = real / optimal   (>= 1, чем ближе к 1 -- тем лучше)

"optimal" здесь -- минимальная суммарная стоимость (W_d + W_b по всем
доставкам), достижимая ЛЮБОЙ последовательностью назначений груз/доставщик/
строитель, без ограничений Шага 2 модели (без деления на раунды с фиксированным
N, без отбора по приоритету p) -- т.е. верхний предел того, чего в принципе
может достичь ЛЮБАЯ эвристика на этой модели.

ВНИМАНИЕ: полный перебор экспоненциален по числу грузов -- держите n_cargos
маленьким (по умолчанию до 5-6). См. --max-cargos.

ЗАПУСК (маленький сценарий по умолчанию):
    python experiments/compare_with_optimum.py \\
        --n-islands 8 --n-pairs 2 --n-cargos-min 2 --n-cargos-max 5 --n-runs 5 --seed 0

Список параметров:
    python experiments/compare_with_optimum.py --help
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_delivery_v2.cargo_priority import CARGO_HEURISTICS
from robot_delivery_v2.experiment import run_single
from robot_delivery_v2.optimal import solve_optimal_brute_force
from robot_delivery_v2.scenario import generate_scenario

DEFAULT_HEURISTICS = ["direct", "inverse", "random"]


@dataclass
class ComparisonRow:
    heuristic: str
    seed: int
    run_index: int
    n_islands: int
    n_cargos: int
    n_pairs: int
    all_delivered: bool
    real: float
    optimal: Optional[float]
    competitive_ratio: Optional[float]
    solve_time_s: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Сравнение эвристик приоритета груза с ТОЧНЫМ оптимумом (полный "
            "перебор) на маленьких сценариях -- competitive ratio = real / optimal."
        )
    )
    p.add_argument("--heuristics", nargs="+", default=DEFAULT_HEURISTICS,
                    help=f"Какие эвристики сравнивать. Доступны: {', '.join(sorted(CARGO_HEURISTICS))}")
    p.add_argument("--n-islands", type=int, default=8, help="Количество островов среды.")
    p.add_argument("--n-pairs", type=int, default=2, help="Количество доставщиков = количество строителей.")
    p.add_argument("--n-cargos-min", type=int, default=2, help="Минимальное число грузов в развёртке.")
    p.add_argument("--n-cargos-max", type=int, default=5,
                    help="Максимальное число грузов в развёртке (ДЕРЖИТЕ МАЛЕНЬКИМ -- "
                         "полный перебор экспоненциален по числу грузов).")
    p.add_argument("--n-runs", type=int, default=5,
                    help="Количество случайных сценариев (рандомайзер) на каждое значение числа грузов.")
    p.add_argument("--seed", type=int, default=0, help="Базовый seed генератора сценариев.")
    p.add_argument("--free-prob", type=float, default=0.45, help="Доля рёбер FREE (остальные BLOCKED).")
    p.add_argument("--build-cost-factor", type=float, default=3.0,
                    help="Множитель стоимости постройки моста относительно длины.")
    p.add_argument("--max-cargos", type=int, default=7,
                    help="Защитный предел на n_cargos для полного перебора (см. optimal.py).")
    p.add_argument("--output-dir", default=None,
                    help="Куда сохранять CSV и графики (по умолчанию -- "
                         "../outputs/vs_optimum относительно этого файла).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    unknown = [h for h in args.heuristics if h not in CARGO_HEURISTICS]
    if unknown:
        print(f"Неизвестные эвристики: {', '.join(unknown)}", file=sys.stderr)
        print(f"Доступные: {', '.join(sorted(CARGO_HEURISTICS))}", file=sys.stderr)
        return 1

    if args.n_cargos_max > args.max_cargos:
        print(
            f"--n-cargos-max={args.n_cargos_max} превышает --max-cargos={args.max_cargos} "
            "(защита от нереализуемого по времени полного перебора). Уменьшите "
            "--n-cargos-max или явно поднимите --max-cargos, если уверены.",
            file=sys.stderr,
        )
        return 1

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "outputs", "vs_optimum"
    )
    os.makedirs(output_dir, exist_ok=True)

    scenario_kwargs = dict(
        free_prob=args.free_prob,
        build_cost_factor=args.build_cost_factor,
    )

    sweep = list(range(args.n_cargos_min, args.n_cargos_max + 1))
    total_scenarios = len(sweep) * args.n_runs
    print(f"Островов среды:        {args.n_islands}")
    print(f"Пар роботов:           {args.n_pairs}")
    print(f"Развёртка по грузам:   {sweep[0]}..{sweep[-1]} ({len(sweep)} точек)")
    print(f"Прогонов на сценарий:  {args.n_runs}")
    print(f"Эвристики:             {', '.join(args.heuristics)}")
    print(f"Всего сценариев (= прогонов полного перебора): {total_scenarios}\n")

    rows: List[ComparisonRow] = []
    n_infeasible = 0
    n_optimal_not_found = 0

    for n_cargos in sweep:
        for run_idx in range(args.n_runs):
            seed = args.seed + run_idx * 1009 + n_cargos * 7919
            scenario = generate_scenario(
                n_islands=args.n_islands, n_cargos=n_cargos, n_pairs=args.n_pairs,
                seed=seed, **scenario_kwargs,
            )

            heuristic_results = {}
            for hname in args.heuristics:
                result, bracket = run_single(scenario, hname)
                heuristic_results[hname] = (result, bracket)

            delivered_reals = [
                bracket.real for result, bracket in heuristic_results.values() if result.all_delivered
            ]
            upper_bound = min(delivered_reals) if delivered_reals else None

            t0 = time.time()
            opt = solve_optimal_brute_force(
                scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
                upper_bound=upper_bound, max_cargos=args.max_cargos,
            )
            solve_time = time.time() - t0

            if not opt.feasible:
                n_infeasible += 1
            elif opt.optimal is None:
                n_optimal_not_found += 1

            for hname in args.heuristics:
                result, bracket = heuristic_results[hname]
                competitive_ratio = None
                optimal_val = opt.optimal if opt.feasible else None
                if result.all_delivered and optimal_val is not None and optimal_val > 1e-9:
                    competitive_ratio = bracket.real / optimal_val
                rows.append(
                    ComparisonRow(
                        heuristic=hname,
                        seed=seed,
                        run_index=run_idx,
                        n_islands=args.n_islands,
                        n_cargos=n_cargos,
                        n_pairs=args.n_pairs,
                        all_delivered=result.all_delivered,
                        real=bracket.real,
                        optimal=optimal_val,
                        competitive_ratio=competitive_ratio,
                        solve_time_s=solve_time,
                    )
                )
        print(f"  n_cargos={n_cargos}: {args.n_runs} сценариев x {len(args.heuristics)} эвристик -- готово")

    print()

    csv_path = os.path.join(output_dir, "vs_optimum_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ComparisonRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    print(f"Сохранено: {csv_path} ({len(rows)} строк)")

    if n_infeasible:
        print(f"Недостижимых сценариев (Шаг 0): {n_infeasible}")
    if n_optimal_not_found:
        print(f"Сценариев, где полный перебор не нашёл решения: {n_optimal_not_found}")

    # --- сводка по competitive ratio ---
    by_h: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r.competitive_ratio is not None:
            by_h[r.heuristic].append(r.competitive_ratio)

    print("\nCompetitive ratio (real / optimal), среднее по всем валидным сценариям:")
    for h in args.heuristics:
        vals = by_h[h]
        if not vals:
            print(f"  {h:10s} нет валидных сценариев (не доставлено/не найден оптимум)")
            continue
        avg = sum(vals) / len(vals)
        best = min(vals)
        worst = max(vals)
        print(f"  {h:10s} avg={avg:.3f}  min={best:.3f}  max={worst:.3f}  (n={len(vals)})")

    # --- график: средний competitive ratio по эвристике ---
    bars_path = os.path.join(output_dir, "competitive_ratio_bars.png")
    labels = [h for h in args.heuristics if by_h[h]]
    avgs = [sum(by_h[h]) / len(by_h[h]) for h in labels]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.8), 5))
    ax.bar(labels, avgs, color="tab:blue")
    ax.axhline(1.0, color="green", linewidth=1.2, linestyle="--", label="оптимум (ratio = 1)")
    ax.set_ylabel("competitive ratio = real / optimal")
    ax.set_title("Эвристики vs точный оптимум (полный перебор)")
    ax.legend()
    plt.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(bars_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено: {bars_path}")

    # --- график: competitive ratio vs n_cargos, линия на эвристику ---
    line_path = os.path.join(output_dir, "competitive_ratio_vs_ncargos.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    for h in args.heuristics:
        by_n: Dict[int, List[float]] = defaultdict(list)
        for r in rows:
            if r.heuristic == h and r.competitive_ratio is not None:
                by_n[r.n_cargos].append(r.competitive_ratio)
        ns = sorted(by_n.keys())
        if not ns:
            continue
        vals = [sum(by_n[n]) / len(by_n[n]) for n in ns]
        ax.plot(ns, vals, marker="o", label=h)
    ax.axhline(1.0, color="green", linewidth=1.2, linestyle="--", label="оптимум (ratio = 1)")
    ax.set_xlabel("n_cargos")
    ax.set_ylabel("competitive ratio = real / optimal")
    ax.set_title("Competitive ratio в зависимости от числа грузов")
    ax.legend(fontsize=8)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(line_path, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {line_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
