"""
Эксперимент: сравнение двух режимов СЛУЧАЙНОГО распределения грузов (Шаг 3) на
ОДНИХ И ТЕХ ЖЕ сценариях и с ОДНИМИ И ТЕМИ ЖЕ сидами модели.

  balanced -- список грузов перемешивается и раздаётся парам по кругу:
              какой груз какой паре достанется -- случайно, но ЧИСЛО грузов у
              пар отличается максимум на 1.
  uniform  -- каждому грузу независимо и равновероятно выбирается пара:
              буквальнее по тексту постановки, но число грузов у пар
              распределено мультиномиально, и при малом n одна пара может
              забрать почти все грузы, а другая остаться вовсе без работы.

Содержательный вопрос эксперимента: во что обходится отсутствие выравнивания
ЧИСЛА грузов, если сама раздача в обоих случаях случайна и никакой информации
о стоимости не использует. Ожидание: по суммарной стоимости real режимы близки
(сумма слабо зависит от того, кто везёт, пока выбор не связан со стоимостью), а
расходятся на метриках баланса -- pair_load_balance, pair_cost_balance и
простое на барьерах раундов idle_total.

ПОВТОРЫ. Модель случайна целиком, поэтому один прогон на сценарий -- одна
реализация случайной величины. По умолчанию каждый сценарий прогоняется
--n-repeats раз с разными сидами модели, и ОБА режима получают одинаковый набор
сидов: сравнение идёт на буквально одной и той же случайности.

ЗАПУСК:
    python experiments/compare_random_assignment.py \\
        --n-islands 18 --n-pairs 3 -n 24 --n-runs 15 --n-repeats 5 --seed 0

Список параметров:
    python experiments/compare_random_assignment.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_delivery_random.assignment import ASSIGNMENT_MODES
from robot_delivery_random.experiment import cargo_sweep, run_suite
from robot_delivery_random.plotting import (
    export_csv,
    plot_idle_vs_ncargos,
    plot_pair_cost_balance,
    plot_pair_load_balance,
    plot_real_bars,
    plot_real_cv_vs_ncargos,
    plot_real_spread,
    plot_real_vs_ncargos,
    plot_win_rate,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Сравнение режимов случайного распределения грузов "
                    "(balanced vs uniform) в модели static_v3."
    )
    p.add_argument("--assignment", nargs="+", default=["balanced", "uniform"],
                   help="Режимы для сравнения. Доступны: "
                        f"{', '.join(ASSIGNMENT_MODES)}")
    p.add_argument("--n-islands", type=int, default=18, help="Количество островов среды.")
    p.add_argument("--n-pairs", type=int, default=3,
                   help="Количество доставщиков = количество строителей.")
    p.add_argument("-n", "--n-cargos", type=int, default=24,
                   help="Верхняя граница развёртки по числу грузов (от --n-pairs до n).")
    p.add_argument("--n-runs", type=int, default=15,
                   help="Количество случайных сценариев на каждое значение числа грузов.")
    p.add_argument("--n-repeats", type=int, default=5,
                   help="Сколько раз прогонять модель на каждом сценарии с разными "
                        "сидами случайности (оба режима получают одинаковые сиды).")
    p.add_argument("--seed", type=int, default=0, help="Базовый seed генератора сценариев.")
    p.add_argument("--free-prob", type=float, default=0.45, help="Доля рёбер FREE.")
    p.add_argument("--build-cost-factor", type=float, default=3.0,
                   help="Множитель стоимости постройки моста относительно длины.")
    p.add_argument("--output-dir", default=None,
                   help="Куда сохранять CSV и графики (по умолчанию -- "
                        "../outputs/random_assignment относительно этого файла).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    unknown_modes = [m for m in args.assignment if m not in ASSIGNMENT_MODES]
    if unknown_modes:
        print(f"Неизвестные режимы распределения: {', '.join(unknown_modes)}", file=sys.stderr)
        return 1
    if args.n_repeats < 1:
        print("--n-repeats должен быть >= 1", file=sys.stderr)
        return 1

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "outputs", "random_assignment"
    )
    os.makedirs(output_dir, exist_ok=True)

    try:
        sweep = cargo_sweep(args.n_pairs, args.n_cargos)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    total = len(args.assignment) * len(sweep) * args.n_runs * args.n_repeats
    print(f"Островов среды:        {args.n_islands}")
    print(f"Пар роботов:           {args.n_pairs}")
    print(f"Развёртка по грузам:   {sweep[0]}..{sweep[-1]} ({len(sweep)} точек)")
    print(f"Сценариев на точку:    {args.n_runs}")
    print(f"Повторов на сценарий:  {args.n_repeats}")
    print(f"Режимы распределения:  {', '.join(args.assignment)}")
    print(f"Всего прогонов модели: {total}\n")

    rows = run_suite(
        n_runs=args.n_runs,
        n_cargos_max=args.n_cargos,
        n_pairs=args.n_pairs,
        n_islands=args.n_islands,
        base_seed=args.seed,
        assignment_modes=args.assignment,
        n_repeats=args.n_repeats,
        scenario_kwargs=dict(
            free_prob=args.free_prob,
            build_cost_factor=args.build_cost_factor,
        ),
        progress=True,
    )

    csv_path = os.path.join(output_dir, "experiment_results.csv")
    export_csv(rows, csv_path)
    print(f"\nСохранено: {csv_path} ({len(rows)} строк)")

    plots = [
        ("compare_real_vs_ncargos.png", plot_real_vs_ncargos),
        ("compare_real_bars.png", plot_real_bars),
        ("compare_real_spread.png", plot_real_spread),
        ("pair_load_balance.png", plot_pair_load_balance),
        ("pair_cost_balance.png", plot_pair_cost_balance),
        ("idle_vs_ncargos.png", plot_idle_vs_ncargos),
    ]
    if args.n_repeats > 1:
        plots.append(("real_cv_vs_ncargos.png", plot_real_cv_vs_ncargos))
    if len(args.assignment) > 1:
        plots.append(("compare_win_rate.png", plot_win_rate))

    for name, fn in plots:
        path = os.path.join(output_dir, name)
        fn(rows, path)
        if os.path.exists(path):
            print(f"Сохранено: {path}")

    # --- сводка по режимам ---
    delivered = [r for r in rows if r.all_delivered]
    print("\nСредние значения (по полностью доставленным прогонам):")
    print(f"  {'режим':10s} {'real':>10s} {'перекос':>9s} {'idle_total':>11s} "
          f"{'max груз.':>10s} {'min груз.':>10s}")
    summary = {}
    for variant in sorted({r.variant for r in rows}):
        vals = [r for r in delivered if r.variant == variant]
        if not vals:
            print(f"  {variant:10s} нет полностью доставленных прогонов")
            continue
        summary[variant] = mean(r.real for r in vals)
        print(f"  {variant:10s} {summary[variant]:10.2f} "
              f"{mean(r.cost_imbalance for r in vals):9.3f} "
              f"{mean(r.idle_total for r in vals):11.2f} "
              f"{mean(r.max_pair_load for r in vals):10.2f} "
              f"{mean(r.min_pair_load for r in vals):10.2f}  (n={len(vals)})")

    if len(summary) > 1:
        best = min(summary, key=lambda v: summary[v])
        worst = max(summary, key=lambda v: summary[v])
        diff = 100.0 * (summary[worst] / summary[best] - 1.0)
        print(f"\nПо суммарной стоимости дешевле: {best} "
              f"(на {diff:.1f}% дешевле, чем {worst})")
        print("Напоминание: сумма real слабо зависит от того, КАКАЯ пара везёт "
              "груз,\nпока выбор не связан со стоимостью; содержательная разница "
              "режимов --\nв балансе (перекос, idle_total, min/max грузов на пару).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
