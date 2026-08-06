"""
CLI для прогона экспериментов.

Примеры:
    # список доступных эвристик
    python -m robot_delivery.cli --list-heuristics

    # сравнить эвристики: 20 островов, 3 пары роботов, грузов до 20,
    # 10 случайных прогонов на каждое значение числа грузов.
    # Число грузов пробегает диапазон 3..20 (от --n-pairs до -n).
    python -m robot_delivery.cli \\
        --heuristics cost_direct cost_inverse dist_min_deliverer \\
        --n-islands 20 \\
        --n-pairs 3 \\
        -n 20 \\
        --n-runs 10 \\
        --seed 42 \\
        --output-dir outputs

    # плюс детальная трасса по раундам для первой эвристики
    python -m robot_delivery.cli --heuristics cost_direct \\
        --n-islands 16 --n-pairs 2 -n 12 --n-runs 5 --trace --output-dir outputs
"""

from __future__ import annotations

import argparse
import os
import sys

from .experiment import cargo_sweep, run_single, run_suite
from .heuristics import HEURISTICS
from .plotting import (
    export_csv,
    plot_gap_vs_ncargos,
    plot_real_vs_estimated_bars,
    plot_real_vs_ncargos,
    plot_round_trace,
    plot_win_rate,
)
from .scenario import generate_scenario
from .scheduler import ASSIGNMENT_ALGOS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Сравнение эвристической оценки и реальной стоимости в динамической модели многороботной доставки."
    )
    p.add_argument(
        "--list-heuristics", action="store_true",
        help="Вывести список доступных эвристик приоритета p и выйти.",
    )
    p.add_argument(
        "--heuristics", nargs="+", default=["cost_direct"],
        help=f"Эвристики для сравнения. Доступны: {', '.join(sorted(HEURISTICS))}",
    )
    p.add_argument(
        "--assignment-algos", nargs="+", default=["greedy"],
        help="Алгоритмы распределения грузов по парам ВНУТРИ раунда (способ "
             "предъявления таблицы приоритетов циклу назначения), для каждого "
             "прогоняются все --heuristics на одних и тех же сценариях. "
             f"Доступны: {', '.join(sorted(ASSIGNMENT_ALGOS))}. "
             "Если задать несколько -- дополнительно строится сравнительный "
             "график assignment_comparison_real.png.",
    )
    p.add_argument("--n-islands", type=int, default=18,
                    help="Количество островов среды (вершин графа G).")
    p.add_argument("--n-pairs", type=int, default=3,
                    help="Количество пар роботов (коалиция 1 доставщик + 1 строитель).")
    p.add_argument("-n", "--n-cargos", type=int, default=20,
                    help="Количество грузов n (верхняя граница развёртки). Цикл "
                         "проходится по числу грузов от --n-pairs до n включительно.")
    p.add_argument("--n-runs", type=int, default=10,
                    help="Количество прогонов (рандомайзер) на один сценарий, т.е. "
                         "сколько независимых случайных инстансов генерировать на "
                         "каждое значение числа грузов.")
    p.add_argument("--seed", type=int, default=0, help="Базовый seed генератора сценариев.")
    p.add_argument("--free-prob", type=float, default=0.45, help="Доля рёбер FREE (остальные BLOCKED).")
    p.add_argument("--build-cost-factor", type=float, default=3.0,
                    help="Множитель стоимости постройки моста относительно длины.")
    p.add_argument("--output-dir", default="outputs", help="Куда сохранять CSV и графики.")
    p.add_argument("--trace", action="store_true",
                    help="Дополнительно построить график накопления real/estimated по раундам "
                         "для первой эвристики на первом сценарии.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_heuristics:
        for key in sorted(HEURISTICS):
            h = HEURISTICS[key]
            print(f"{key:28s} {h.label}\n{'':28s} {h.description}\n")
        return 0

    unknown = [h for h in args.heuristics if h not in HEURISTICS]
    if unknown:
        print(f"Неизвестные эвристики: {', '.join(unknown)}", file=sys.stderr)
        print(f"Доступные: {', '.join(sorted(HEURISTICS))}", file=sys.stderr)
        return 1

    unknown_algos = [a for a in args.assignment_algos if a not in ASSIGNMENT_ALGOS]
    if unknown_algos:
        print(f"Неизвестные алгоритмы распределения: {', '.join(unknown_algos)}", file=sys.stderr)
        print(f"Доступные: {', '.join(sorted(ASSIGNMENT_ALGOS))}", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    scenario_kwargs = dict(
        free_prob=args.free_prob,
        build_cost_factor=args.build_cost_factor,
    )

    try:
        sweep = cargo_sweep(args.n_pairs, args.n_cargos)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    total_runs = len(args.heuristics) * len(args.assignment_algos) * len(sweep) * args.n_runs
    print(f"Островов среды:        {args.n_islands}")
    print(f"Пар роботов:           {args.n_pairs}")
    print(f"Грузов n:              {args.n_cargos}")
    print(f"Развёртка по грузам:   {sweep[0]}..{sweep[-1]} ({len(sweep)} точек)")
    print(f"Прогонов на сценарий:  {args.n_runs}")
    print(f"Эвристик:              {len(args.heuristics)} ({', '.join(args.heuristics)})")
    print(f"Алгоритмов распределения: {len(args.assignment_algos)} ({', '.join(args.assignment_algos)})")
    print(f"Всего прогонов модели: {total_runs}\n")

    rows = run_suite(
        heuristic_names=args.heuristics,
        n_runs=args.n_runs,
        n_cargos_max=args.n_cargos,
        n_pairs=args.n_pairs,
        n_islands=args.n_islands,
        base_seed=args.seed,
        scenario_kwargs=scenario_kwargs,
        progress=True,
        assignment_names=args.assignment_algos,
    )
    print()

    csv_path = os.path.join(args.output_dir, "experiment_results.csv")
    export_csv(rows, csv_path)
    print(f"Сохранено: {csv_path} ({len(rows)} строк)")

    # если варьируется алгоритм распределения, группируем графики по
    # комбинации (heuristic+assignment) -- иначе прогоны с разными
    # алгоритмами смешались бы под одной меткой эвристики.
    compare_assignment = len(args.assignment_algos) > 1
    group_by = "combo" if compare_assignment else "heuristic"

    bars_path = os.path.join(args.output_dir, "real_vs_estimated_bars.png")
    plot_real_vs_estimated_bars(rows, bars_path, group_by=group_by)
    print(f"Сохранено: {bars_path}")

    gap_raw_path = os.path.join(args.output_dir, "gap_raw_vs_ncargos.png")
    plot_gap_vs_ncargos(rows, gap_raw_path, gap_field="gap_raw", group_by=group_by)
    print(f"Сохранено: {gap_raw_path}")

    gap_pool_path = os.path.join(args.output_dir, "gap_pool_vs_ncargos.png")
    plot_gap_vs_ncargos(rows, gap_pool_path, gap_field="gap_pool", group_by=group_by)
    print(f"Сохранено: {gap_pool_path}")

    gap_prognosis_path = os.path.join(args.output_dir, "gap_prognosis_vs_ncargos.png")
    plot_gap_vs_ncargos(rows, gap_prognosis_path, gap_field="gap_prognosis", group_by=group_by)
    print(f"Сохранено: {gap_prognosis_path}")

    if len(args.heuristics) > 1 or compare_assignment:
        winrate_path = os.path.join(args.output_dir, "win_rate.png")
        plot_win_rate(rows, winrate_path, group_by=group_by)
        print(f"Сохранено: {winrate_path}")

    if compare_assignment:
        # ключевой график задачи: поведение greedy vs round-robin (и т.д.)
        # по фактической стоимости доставки в зависимости от числа грузов,
        # все комбинации -- на одном графике.
        assignment_cmp_path = os.path.join(args.output_dir, "assignment_comparison_real.png")
        plot_real_vs_ncargos(rows, assignment_cmp_path, group_by=group_by)
        print(f"Сохранено: {assignment_cmp_path}")

    n_delivered_ratio = sum(1 for r in rows if r.all_delivered) / len(rows) if rows else 0.0
    print(f"Доля прогонов с полной доставкой всех грузов: {n_delivered_ratio:.1%}")

    n_infeasible = sum(1 for r in rows if not r.feasible)
    if n_infeasible:
        infeasible_ratio = n_infeasible / len(rows)
        print(
            f"Из них недостижимых уже на Шаге 0 (проверка достижимости): "
            f"{n_infeasible} ({infeasible_ratio:.1%})"
        )

    if args.trace:
        scenario = generate_scenario(
            n_islands=args.n_islands,
            n_cargos=args.n_cargos,
            n_pairs=args.n_pairs,
            seed=args.seed,
            **scenario_kwargs,
        )
        result, bracket_obj = run_single(scenario, args.heuristics[0], args.assignment_algos[0])
        trace_path = os.path.join(args.output_dir, "round_trace.png")
        # plot_round_trace(
        #     bracket_obj, trace_path,
        #     title=(
        #         f"{args.heuristics[0]} [{args.assignment_algos[0]}]: real vs estimated_raw "
        #         f"vs estimated_prognosis по раундам (n_cargos={args.n_cargos})"
        #     ),
        # )
        print(f"Сохранено: {trace_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
