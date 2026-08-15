"""
CLI для прогона экспериментов static_v3 (статическая модель со СЛУЧАЙНЫМ
формированием пар и СЛУЧАЙНЫМ распределением грузов).

Примеры:
    # базовый прогон: 20 островов, 3 пары роботов, грузов до 20, 10 случайных
    # сценариев на каждое значение числа грузов (развёртка 3..20)
    python -m robot_delivery_random.cli \\
        --n-islands 20 \\
        --n-pairs 3 \\
        -n 20 \\
        --n-runs 10 \\
        --seed 42 \\
        --output-dir outputs

    # сравнение двух режимов случайного распределения на одних сценариях
    python -m robot_delivery_random.cli --assignment balanced uniform \\
        -n 16 --n-runs 10 --output-dir outputs

    # усреднение по 5 реализациям случайности модели на каждом сценарии --
    # модель случайна, и один прогон на сценарий даёт заметный шум
    python -m robot_delivery_random.cli --n-repeats 5 -n 16 --n-runs 10
"""

from __future__ import annotations

import argparse
import os
import sys

from .assignment import ASSIGNMENT_MODES
from .experiment import cargo_sweep, run_suite
from .plotting import (
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
        description=(
            "Фактическая стоимость выполнения всех операций в СТАТИЧЕСКОЙ "
            "модели многороботной доставки со случайным формированием пар "
            "(Шаг 2) и случайным распределением грузов (Шаг 3) -- static_v3. "
            "Модель служит baseline для static_v2, где оба этих шага "
            "выполняются осмысленными правилами."
        )
    )
    p.add_argument(
        "--assignment", nargs="+", default=["balanced"],
        help=(
            "Режим случайного распределения грузов (Шаг 3). balanced -- грузы "
            "перемешиваются и раздаются парам по кругу (число грузов у пар "
            "отличается максимум на 1); uniform -- каждому грузу независимо "
            "выбирается случайная пара (пара может остаться и вовсе без "
            "работы). Можно указать несколько для сравнения. "
            f"Доступны: {', '.join(ASSIGNMENT_MODES)}"
        ),
    )
    p.add_argument("--n-islands", type=int, default=18,
                   help="Количество островов среды (вершин графа G).")
    p.add_argument("--n-pairs", type=int, default=3,
                   help="Количество доставщиков = количество строителей "
                        "(в пары они склеиваются случайно на Шаге 2).")
    p.add_argument("-n", "--n-cargos", type=int, default=20,
                   help="Количество грузов n (верхняя граница развёртки). Цикл "
                        "проходится по числу грузов от --n-pairs до n включительно.")
    p.add_argument("--n-runs", type=int, default=10,
                   help="Количество независимых случайных СЦЕНАРИЕВ на каждое "
                        "значение числа грузов.")
    p.add_argument("--n-repeats", type=int, default=1,
                   help="Сколько раз прогонять модель на КАЖДОМ сценарии с "
                        "разными сидами случайности (Шаги 2/3/4). Модель "
                        "случайна, поэтому один прогон -- одна реализация, а не "
                        "характеристика сценария; при >1 в CSV попадают все "
                        "прогоны, а графики усредняют. По умолчанию 1.")
    p.add_argument("--seed", type=int, default=0, help="Базовый seed генератора сценариев.")
    p.add_argument("--free-prob", type=float, default=0.45,
                   help="Доля рёбер FREE (остальные BLOCKED).")
    p.add_argument("--build-cost-factor", type=float, default=3.0,
                   help="Множитель стоимости постройки моста относительно длины.")
    p.add_argument("--output-dir", default="outputs", help="Куда сохранять CSV и графики.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    unknown_modes = [m for m in args.assignment if m not in ASSIGNMENT_MODES]
    if unknown_modes:
        print(f"Неизвестные режимы распределения: {', '.join(unknown_modes)}", file=sys.stderr)
        print(f"Доступные: {', '.join(ASSIGNMENT_MODES)}", file=sys.stderr)
        return 1

    if args.n_repeats < 1:
        print("--n-repeats должен быть >= 1", file=sys.stderr)
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

    total_runs = len(args.assignment) * len(sweep) * args.n_runs * args.n_repeats
    print(f"Островов среды:        {args.n_islands}")
    print(f"Пар роботов:           {args.n_pairs}")
    print(f"Грузов n:              {args.n_cargos}")
    print(f"Развёртка по грузам:   {sweep[0]}..{sweep[-1]} ({len(sweep)} точек)")
    print(f"Сценариев на точку:    {args.n_runs}")
    print(f"Повторов на сценарий:  {args.n_repeats}")
    print(f"Режимы распределения:  {', '.join(args.assignment)}")
    print(f"Всего прогонов модели: {total_runs}\n")

    rows = run_suite(
        n_runs=args.n_runs,
        n_cargos_max=args.n_cargos,
        n_pairs=args.n_pairs,
        n_islands=args.n_islands,
        base_seed=args.seed,
        assignment_modes=args.assignment,
        n_repeats=args.n_repeats,
        scenario_kwargs=scenario_kwargs,
        progress=True,
    )
    print()

    def save(name: str, fn) -> None:
        path = os.path.join(args.output_dir, name)
        fn(rows, path)
        if os.path.exists(path):
            print(f"Сохранено: {path}")

    csv_path = os.path.join(args.output_dir, "experiment_results.csv")
    export_csv(rows, csv_path)
    print(f"Сохранено: {csv_path} ({len(rows)} строк)")

    save("real_vs_ncargos.png", plot_real_vs_ncargos)
    save("real_bars.png", plot_real_bars)
    save("real_spread.png", plot_real_spread)
    save("pair_load_balance.png", plot_pair_load_balance)
    save("pair_cost_balance.png", plot_pair_cost_balance)
    save("idle_vs_ncargos.png", plot_idle_vs_ncargos)

    if args.n_repeats > 1:
        save("real_cv_vs_ncargos.png", plot_real_cv_vs_ncargos)

    if len(args.assignment) > 1:
        save("win_rate.png", plot_win_rate)

    delivered_ratio = sum(1 for r in rows if r.all_delivered) / len(rows) if rows else 0.0
    print(f"\nДоля прогонов с полной доставкой всех грузов: {delivered_ratio:.1%}")

    n_infeasible = sum(1 for r in rows if not r.feasible)
    if n_infeasible:
        print(
            f"Из них недостижимых уже на Шаге 0 (проверка достижимости): "
            f"{n_infeasible} ({n_infeasible / len(rows):.1%})"
        )

    for mode in args.assignment:
        sub = [r for r in rows if r.variant == mode and r.all_delivered]
        if sub:
            avg = sum(r.real for r in sub) / len(sub)
            print(f"  {mode:9s} средний real = {avg:10.2f}  ({len(sub)} прогонов)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
