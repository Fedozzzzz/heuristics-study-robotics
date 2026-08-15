"""
Эксперимент: сравнение эвристик приоритета груза (Шаг 3) на ОДНИХ И ТЕХ ЖЕ
случайных сценариях -- direct (p = W_T, "дороже доставка -- выше приоритет",
эвристика постановки) vs inverse (p = 1/W_T) vs random (baseline).

Строит:
  1. compare_real_vs_ncargos.png       -- средняя реальная стоимость (real) в
                                          зависимости от числа грузов.
  2. compare_real_vs_estimated_bars.png -- real рядом с estimated_static и
                                          estimated_raw для каждой эвристики.
  3. compare_gap_static_vs_ncargos.png  -- насколько статическая оценка Шага 3
                                          расходится с фактом.
  4. compare_win_rate.png               -- доля сценариев с минимальным real.

ЗАПУСК:
    python experiments/compare_priority_heuristics.py \\
        --n-islands 18 --n-pairs 3 -n 24 --n-runs 15 --seed 0

Список параметров:
    python experiments/compare_priority_heuristics.py --help
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_delivery_static.assignment import ASSIGNMENT_MODES, BALANCE_MODES
from robot_delivery_static.experiment import cargo_sweep, run_suite
from robot_delivery_static.plotting import (
    export_csv,
    plot_gap_vs_ncargos,
    plot_real_vs_estimated_bars,
    plot_real_vs_ncargos,
    plot_win_rate,
)
from robot_delivery_static.priority import CARGO_HEURISTICS

DEFAULT_HEURISTICS = ["direct", "inverse", "random"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Сравнение эвристик приоритета груза статической модели на одних и тех же сценариях."
    )
    p.add_argument("--heuristics", nargs="+", default=DEFAULT_HEURISTICS,
                    help=f"Какие эвристики сравнивать. Доступны: {', '.join(sorted(CARGO_HEURISTICS))}")
    p.add_argument("--assignment", default="literal", choices=list(ASSIGNMENT_MODES),
                    help="Режим назначения груза паре (Шаг 3), одинаковый для всех эвристик.")
    p.add_argument("--balance", default="load", choices=list(BALANCE_MODES),
                    help="Балансировка загрузки пар (Шаг 3).")
    p.add_argument("--n-islands", type=int, default=18, help="Количество островов среды.")
    p.add_argument("--n-pairs", type=int, default=3,
                    help="Количество доставщиков = количество строителей.")
    p.add_argument("-n", "--n-cargos", type=int, default=24,
                    help="Верхняя граница развёртки по числу грузов (от --n-pairs до n).")
    p.add_argument("--n-runs", type=int, default=15,
                    help="Количество случайных сценариев на каждое значение числа грузов.")
    p.add_argument("--seed", type=int, default=0, help="Базовый seed генератора сценариев.")
    p.add_argument("--free-prob", type=float, default=0.45, help="Доля рёбер FREE.")
    p.add_argument("--build-cost-factor", type=float, default=3.0,
                    help="Множитель стоимости постройки моста относительно длины.")
    p.add_argument("--output-dir", default=None,
                    help="Куда сохранять CSV и графики (по умолчанию -- "
                         "../outputs/priority_heuristics относительно этого файла).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    unknown = [h for h in args.heuristics if h not in CARGO_HEURISTICS]
    if unknown:
        print(f"Неизвестные эвристики: {', '.join(unknown)}", file=sys.stderr)
        print(f"Доступные: {', '.join(sorted(CARGO_HEURISTICS))}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "outputs", "priority_heuristics"
    )
    os.makedirs(output_dir, exist_ok=True)

    try:
        sweep = cargo_sweep(args.n_pairs, args.n_cargos)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Островов среды:        {args.n_islands}")
    print(f"Пар роботов:           {args.n_pairs}")
    print(f"Развёртка по грузам:   {sweep[0]}..{sweep[-1]} ({len(sweep)} точек)")
    print(f"Прогонов на сценарий:  {args.n_runs}")
    print(f"Эвристики:             {', '.join(args.heuristics)}")
    print(f"Режим назначения:      {args.assignment} (балансировка: {args.balance})")
    print(f"Всего прогонов модели: {len(args.heuristics) * len(sweep) * args.n_runs}\n")

    rows = run_suite(
        heuristic_names=args.heuristics,
        n_runs=args.n_runs,
        n_cargos_max=args.n_cargos,
        n_pairs=args.n_pairs,
        n_islands=args.n_islands,
        base_seed=args.seed,
        assignment_modes=[args.assignment],
        balance=args.balance,
        scenario_kwargs=dict(
            free_prob=args.free_prob,
            build_cost_factor=args.build_cost_factor,
        ),
        progress=True,
    )

    csv_path = os.path.join(output_dir, "experiment_results.csv")
    export_csv(rows, csv_path)
    print(f"\nСохранено: {csv_path} ({len(rows)} строк)")

    for name, fn in (
        ("compare_real_vs_ncargos.png", plot_real_vs_ncargos),
        ("compare_real_vs_estimated_bars.png", plot_real_vs_estimated_bars),
        ("compare_win_rate.png", plot_win_rate),
    ):
        path = os.path.join(output_dir, name)
        fn(rows, path)
        print(f"Сохранено: {path}")

    gap_path = os.path.join(output_dir, "compare_gap_static_vs_ncargos.png")
    plot_gap_vs_ncargos(rows, gap_path, gap_field="gap_static")
    print(f"Сохранено: {gap_path}")

    # --- сводка ---
    delivered = [r for r in rows if r.all_delivered]
    print("\nСредние значения (по полностью доставленным прогонам):")
    for h in args.heuristics:
        vals = [r for r in delivered if r.heuristic == h]
        if not vals:
            print(f"  {h:10s} нет полностью доставленных прогонов")
            continue
        avg_real = sum(r.real for r in vals) / len(vals)
        avg_gap = sum(r.gap_static for r in vals) / len(vals)
        print(f"  {h:10s} real={avg_real:9.3f}  gap_static={avg_gap:+7.2f}%  (n={len(vals)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
