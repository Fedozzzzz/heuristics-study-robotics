"""
Эксперимент: насколько эвристика приоритета вообще лучше бессмысленного
(случайного) распределения груза по парам?

Сравнивает ЛЮБЫЕ две (или больше) эвристики из HEURISTICS (heuristics.py) на
ОДНИХ И ТЕХ ЖЕ случайных сценариях (та же развёртка по числу грузов, что и в
cli.py). По умолчанию -- cost_direct vs random_priority:

  - cost_direct     -- эталонный вариант из "Фиксации задачи" (п.2): p = W_T,
                       дороже операция -- выше приоритет.
  - random_priority -- baseline: p ~ U(0,1), не зависит от стоимости задачи
                       (эквивалент случайного распределения груза по парам).

Строит три графика:
  1. compare_real_vs_ncargos.png -- средняя РЕАЛЬНАЯ стоимость (real) для
     сравниваемых эвристик в зависимости от числа грузов, на одном поле --
     прямое сравнение эвристик друг с другом.
  2. compare_real_vs_estimated_bars.png -- для каждой эвристики: её реальная
     стоимость рядом с её же эвристической оценкой (estimated_raw/pool/
     prognosis) -- сравнение с реальной итоговой стоимостью.
  3. compare_win_rate.png -- доля сценариев, где каждая эвристика дала
     минимальную реальную стоимость (только полностью доставленные сценарии).

Параметры развёртки -- те же флаги, что и у robot_delivery.cli (число
островов/пар/грузов/прогонов, seed, параметры генератора графа, куда
сохранять результаты), плюс --heuristics для выбора самих эвристик.

ЗАПУСК (параметры по умолчанию -- cost_direct vs random_priority):
    python3 experiments/compare_direct_vs_random.py \\
        --n-islands 18 --n-pairs 3 -n 24 --n-runs 15 --seed 0

Другая пара эвристик, например cost_inverse vs random_priority:
    python3 experiments/compare_direct_vs_random.py \\
        --heuristics cost_inverse random_priority \\
        --output-dir outputs/inverse_vs_random

Список параметров:
    python3 experiments/compare_direct_vs_random.py --help
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_delivery.experiment import cargo_sweep, run_suite
from robot_delivery.heuristics import HEURISTICS as HEURISTICS_REGISTRY
from robot_delivery.plotting import (
    export_csv,
    plot_real_vs_estimated_bars,
    plot_real_vs_ncargos,
    plot_win_rate,
)
from robot_delivery.scheduler import ASSIGNMENT_ALGOS

DEFAULT_HEURISTICS = ["cost_direct", "random_priority"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Baseline-сравнение эвристик приоритета: по умолчанию cost_direct "
            "vs random_priority (случайное распределение груза по парам), но "
            "через --heuristics можно сравнить любую пару/набор. Флаги "
            "развёртки идентичны robot_delivery.cli."
        )
    )
    p.add_argument("--heuristics", nargs="+", default=DEFAULT_HEURISTICS,
                    help=f"Какие эвристики сравнивать. Доступны: {', '.join(sorted(HEURISTICS_REGISTRY))}")
    p.add_argument("--n-islands", type=int, default=18,
                    help="Количество островов среды (вершин графа G).")
    p.add_argument("--n-pairs", type=int, default=3,
                    help="Количество пар роботов (коалиция 1 доставщик + 1 строитель).")
    p.add_argument("-n", "--n-cargos", type=int, default=24,
                    help="Количество грузов n (верхняя граница развёртки). Цикл "
                         "проходится по числу грузов от --n-pairs до n включительно.")
    p.add_argument("--n-runs", type=int, default=15,
                    help="Количество прогонов (рандомайзер) на каждое значение числа грузов.")
    p.add_argument("--seed", type=int, default=0, help="Базовый seed генератора сценариев.")
    p.add_argument("--free-prob", type=float, default=0.45, help="Доля рёбер FREE (остальные BLOCKED).")
    p.add_argument("--build-cost-factor", type=float, default=3.0,
                    help="Множитель стоимости постройки моста относительно длины.")
    p.add_argument("--assignment", default="greedy", choices=sorted(ASSIGNMENT_ALGOS),
                    help="Алгоритм распределения грузов по парам ВНУТРИ раунда "
                         f"(один и тот же для обеих эвристик). Доступны: {', '.join(sorted(ASSIGNMENT_ALGOS))}.")
    p.add_argument("--output-dir", default=None,
                    help="Куда сохранять CSV и графики (по умолчанию -- "
                         "../outputs/direct_vs_random относительно этого файла).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    unknown = [h for h in args.heuristics if h not in HEURISTICS_REGISTRY]
    if unknown:
        print(f"Неизвестные эвристики: {', '.join(unknown)}", file=sys.stderr)
        print(f"Доступные: {', '.join(sorted(HEURISTICS_REGISTRY))}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "outputs", "direct_vs_random"
    )
    os.makedirs(output_dir, exist_ok=True)

    scenario_kwargs = dict(
        free_prob=args.free_prob,
        build_cost_factor=args.build_cost_factor,
    )

    try:
        sweep = cargo_sweep(args.n_pairs, args.n_cargos)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    total_runs = len(args.heuristics) * len(sweep) * args.n_runs
    print(f"Островов среды:        {args.n_islands}")
    print(f"Пар роботов:           {args.n_pairs}")
    print(f"Грузов n:              {args.n_cargos}")
    print(f"Развёртка по грузам:   {sweep[0]}..{sweep[-1]} ({len(sweep)} точек)")
    print(f"Прогонов на сценарий:  {args.n_runs}")
    print(f"Алгоритм распределения внутри раунда: {args.assignment}")
    print(f"Эвристики:             {', '.join(args.heuristics)}")
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
        assignment_names=[args.assignment],
    )

    csv_path = os.path.join(output_dir, "experiment_results.csv")
    export_csv(rows, csv_path)
    print(f"\nСохранено: {csv_path} ({len(rows)} строк)")

    real_path = os.path.join(output_dir, "compare_real_vs_ncargos.png")
    plot_real_vs_ncargos(rows, real_path, group_by="heuristic")
    print(f"Сохранено: {real_path}")

    bars_path = os.path.join(output_dir, "compare_real_vs_estimated_bars.png")
    plot_real_vs_estimated_bars(rows, bars_path, group_by="heuristic")
    print(f"Сохранено: {bars_path}")

    winrate_path = os.path.join(output_dir, "compare_win_rate.png")
    plot_win_rate(rows, winrate_path, group_by="heuristic")
    print(f"Сохранено: {winrate_path}")

    # --- сводка по среднему реальному расхождению ---
    delivered = [r for r in rows if r.all_delivered]
    by_h = {h: [r.real for r in delivered if r.heuristic == h] for h in args.heuristics}
    avg_by_h = {h: (sum(v) / len(v) if v else float("nan")) for h, v in by_h.items()}
    print("\nСредняя реальная стоимость (по всем полностью доставленным прогонам):")
    for h in args.heuristics:
        print(f"  {h:16s} real={avg_by_h[h]:.3f}  (n={len(by_h[h])})")

    if len(args.heuristics) == 2:
        h1, h2 = args.heuristics
        if by_h[h1] and by_h[h2]:
            diff = 100.0 * (avg_by_h[h2] / avg_by_h[h1] - 1.0)
            print(
                f"\n{h1} {'дешевле' if diff > 0 else 'дороже'} {h2} "
                f"в среднем на {abs(diff):.1f}% "
                f"({h1}_avg={avg_by_h[h1]:.3f}, {h2}_avg={avg_by_h[h2]:.3f})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
