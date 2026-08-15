"""
CLI для прогона экспериментов static_v2 (статическая модель: пары и
распределение грузов фиксируются заранее).

Примеры:
    # список доступных эвристик приоритета груза
    python -m robot_delivery_static.cli --list-heuristics

    # все 3 эвристики: 20 островов, 3 пары роботов, грузов до 20, 10 случайных
    # прогонов на каждое значение числа грузов (развёртка 3..20)
    python -m robot_delivery_static.cli \\
        --heuristics direct inverse random \\
        --n-islands 20 \\
        --n-pairs 3 \\
        -n 20 \\
        --n-runs 10 \\
        --seed 42 \\
        --output-dir outputs

    # сравнение режимов назначения груза паре на одних и тех же сценариях
    python -m robot_delivery_static.cli --heuristics direct \\
        --assignment literal cheapest lpt -n 16 --n-runs 10 --output-dir outputs

    # полноценный LPT с поправкой на неидентичность пар (argmin времени
    # завершения вместо argmin загрузки)
    python -m robot_delivery_static.cli --heuristics direct \\
        --assignment lpt --lpt-rule completion --lpt-size mean \\
        -n 16 --n-runs 10 --output-dir outputs

    # плюс детальная трасса по раундам для первого варианта
    python -m robot_delivery_static.cli --heuristics direct \\
        --n-islands 16 --n-pairs 2 -n 12 --n-runs 5 --trace --output-dir outputs
"""

from __future__ import annotations

import argparse
import os
import sys

from .assignment import ASSIGNMENT_MODES, BALANCE_MODES, LPT_RULES, LPT_SIZE_RULES
from .experiment import cargo_sweep, run_single, run_suite, variant_key
from .plotting import (
    export_csv,
    plot_gap_vs_ncargos,
    plot_pair_cost_balance,
    plot_pair_load_balance,
    plot_real_vs_estimated_bars,
    plot_real_vs_ncargos,
    plot_round_trace,
    plot_win_rate,
)
from .priority import CARGO_HEURISTICS
from .scenario import generate_scenario


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Сравнение эвристической оценки и реальной стоимости в СТАТИЧЕСКОЙ "
            "модели многороботной доставки: пары роботов формируются заранее, "
            "грузы распределяются по приоритетам один раз в начале (static_v2)."
        )
    )
    p.add_argument(
        "--list-heuristics", action="store_true",
        help="Вывести список доступных эвристик приоритета груза p и выйти.",
    )
    p.add_argument(
        "--heuristics", nargs="+", default=["direct"],
        help=f"Эвристики приоритета груза. Доступны: {', '.join(sorted(CARGO_HEURISTICS))}",
    )
    p.add_argument(
        "--assignment", nargs="+", default=["literal"],
        help=(
            "Режим назначения груза паре (Шаг 3). literal -- буквально по "
            "постановке: груз достаётся паре из записи с наивысшим приоритетом; "
            "cheapest -- паре с минимальной для неё стоимостью; lpt -- "
            "полноценный LPT (грузы по убыванию размера, каждый -- наименее "
            "загруженной паре). Можно указать несколько для сравнения. "
            f"Доступны: {', '.join(ASSIGNMENT_MODES)}"
        ),
    )
    p.add_argument(
        "--balance", default="load", choices=list(BALANCE_MODES),
        help=(
            "Балансировка загрузки пар (Шаг 3). load -- LPT-подобная по "
            "стоимости (по умолчанию), none -- без ограничения. К режиму "
            "назначения lpt не применяется: он балансирует загрузку сам."
        ),
    )
    p.add_argument(
        "--lpt-size", default="min", choices=list(LPT_SIZE_RULES),
        help=(
            "Только для --assignment lpt: как свернуть стоимости груза у всех "
            "пар в один 'размер' задачи для сортировки. min -- у лучшего "
            "исполнителя (по умолчанию), mean -- средняя, max -- у худшего."
        ),
    )
    p.add_argument(
        "--lpt-rule", default="load", choices=list(LPT_RULES),
        help=(
            "Только для --assignment lpt: правило выбора пары. load -- "
            "классический Грэм, argmin накопленной загрузки (по умолчанию); "
            "completion -- argmin (загрузка + стоимость груза для этой пары), "
            "поправка на неидентичность пар."
        ),
    )
    p.add_argument("--n-islands", type=int, default=18,
                    help="Количество островов среды (вершин графа G).")
    p.add_argument("--n-pairs", type=int, default=3,
                    help="Количество доставщиков = количество строителей "
                         "(пары формируются на Шаге 2 по минимальному расстоянию).")
    p.add_argument("-n", "--n-cargos", type=int, default=20,
                    help="Количество грузов n (верхняя граница развёртки). Цикл "
                         "проходится по числу грузов от --n-pairs до n включительно.")
    p.add_argument("--n-runs", type=int, default=10,
                    help="Количество независимых случайных сценариев на каждое "
                         "значение числа грузов.")
    p.add_argument("--seed", type=int, default=0, help="Базовый seed генератора сценариев.")
    p.add_argument("--free-prob", type=float, default=0.45, help="Доля рёбер FREE (остальные BLOCKED).")
    p.add_argument("--build-cost-factor", type=float, default=3.0,
                    help="Множитель стоимости постройки моста относительно длины.")
    p.add_argument("--output-dir", default="outputs", help="Куда сохранять CSV и графики.")
    p.add_argument("--trace", action="store_true",
                    help="Дополнительно построить график накопления real/estimated по раундам "
                         "для первого варианта на первом сценарии.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_heuristics:
        for key in sorted(CARGO_HEURISTICS):
            h = CARGO_HEURISTICS[key]
            print(f"{key:10s} {h.label}\n{'':10s} {h.description}\n")
        return 0

    unknown = [h for h in args.heuristics if h not in CARGO_HEURISTICS]
    if unknown:
        print(f"Неизвестные эвристики: {', '.join(unknown)}", file=sys.stderr)
        print(f"Доступные: {', '.join(sorted(CARGO_HEURISTICS))}", file=sys.stderr)
        return 1

    unknown_modes = [m for m in args.assignment if m not in ASSIGNMENT_MODES]
    if unknown_modes:
        print(f"Неизвестные режимы назначения: {', '.join(unknown_modes)}", file=sys.stderr)
        print(f"Доступные: {', '.join(ASSIGNMENT_MODES)}", file=sys.stderr)
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

    n_variants = len(args.heuristics) * len(args.assignment)
    total_runs = n_variants * len(sweep) * args.n_runs
    print(f"Островов среды:        {args.n_islands}")
    print(f"Пар роботов:           {args.n_pairs}")
    print(f"Грузов n:              {args.n_cargos}")
    print(f"Развёртка по грузам:   {sweep[0]}..{sweep[-1]} ({len(sweep)} точек)")
    print(f"Прогонов на сценарий:  {args.n_runs}")
    print(f"Эвристики:             {', '.join(args.heuristics)}")
    print(f"Режимы назначения:     {', '.join(args.assignment)}")
    print(f"Балансировка:          {args.balance}")
    if "lpt" in args.assignment:
        print(f"LPT:                   размер задачи {args.lpt_size}, "
              f"выбор пары {args.lpt_rule} (балансировка --balance к нему не применяется)")
    print(f"Всего прогонов модели: {total_runs}\n")

    rows = run_suite(
        heuristic_names=args.heuristics,
        n_runs=args.n_runs,
        n_cargos_max=args.n_cargos,
        n_pairs=args.n_pairs,
        n_islands=args.n_islands,
        base_seed=args.seed,
        assignment_modes=args.assignment,
        balance=args.balance,
        lpt_size=args.lpt_size,
        lpt_rule=args.lpt_rule,
        scenario_kwargs=scenario_kwargs,
        progress=True,
    )
    print()

    csv_path = os.path.join(args.output_dir, "experiment_results.csv")
    export_csv(rows, csv_path)
    print(f"Сохранено: {csv_path} ({len(rows)} строк)")

    bars_path = os.path.join(args.output_dir, "real_vs_estimated_bars.png")
    plot_real_vs_estimated_bars(rows, bars_path)
    print(f"Сохранено: {bars_path}")

    gap_static_path = os.path.join(args.output_dir, "gap_static_vs_ncargos.png")
    plot_gap_vs_ncargos(rows, gap_static_path, gap_field="gap_static")
    print(f"Сохранено: {gap_static_path}")

    gap_raw_path = os.path.join(args.output_dir, "gap_raw_vs_ncargos.png")
    plot_gap_vs_ncargos(rows, gap_raw_path, gap_field="gap_raw")
    print(f"Сохранено: {gap_raw_path}")

    balance_path = os.path.join(args.output_dir, "pair_load_balance.png")
    plot_pair_load_balance(rows, balance_path)
    print(f"Сохранено: {balance_path}")

    cost_balance_path = os.path.join(args.output_dir, "pair_cost_balance.png")
    plot_pair_cost_balance(rows, cost_balance_path)
    print(f"Сохранено: {cost_balance_path}")

    if n_variants > 1:
        real_cmp_path = os.path.join(args.output_dir, "real_vs_ncargos.png")
        plot_real_vs_ncargos(rows, real_cmp_path)
        print(f"Сохранено: {real_cmp_path}")

        winrate_path = os.path.join(args.output_dir, "win_rate.png")
        plot_win_rate(rows, winrate_path)
        print(f"Сохранено: {winrate_path}")

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
        _result, bracket_obj = run_single(
            scenario, args.heuristics[0], args.assignment[0], args.balance,
            args.lpt_size, args.lpt_rule,
        )
        trace_path = os.path.join(args.output_dir, "round_trace.png")
        plot_round_trace(
            bracket_obj, trace_path,
            title=(
                f"{variant_key(args.heuristics[0], args.assignment[0], args.lpt_rule)}: "
                f"real vs оценки по раундам (n_cargos={args.n_cargos})"
            ),
        )
        print(f"Сохранено: {trace_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
