"""
Эксперимент: полноценный LPT против остальных режимов назначения (Шаг 3) на
ОДНИХ И ТЕХ ЖЕ случайных сценариях.

Сравниваются:

  literal        -- буквальное прочтение постановки (груз достаётся паре из
                    записи с наивысшим приоритетом);
  cheapest       -- greedy-nearest (груз -- паре, для которой он дешевле всего);
  lpt-load       -- классический LPT (Graham 1969): грузы по убыванию размера,
                    каждый -- паре с минимальной накопленной загрузкой;
  lpt-completion -- LPT с поправкой на неидентичность пар: пара выбирается по
                    минимуму (загрузка + стоимость груза для неё).

Содержательный вопрос. LPT -- эвристика для МАКСИМУМА (makespan, P||C_max), а
основная метрика модели real = сумма W_d + W_b по всем операциям. Если бы
стоимость груза не зависела от позиции пары, сумма вообще не зависела бы от
распределения -- любое разбиение давало бы одно и то же. Зависимость возникает
только из-за перемещений роботов между доставками и общих мостов, и она никак
не связана с балансировкой максимума. Поэтому LPT здесь проверяется по ДВУМ
метрикам сразу:

  real           -- суммарная стоимость (то, что LPT не оптимизирует напрямую);
  cost_imbalance -- перекос загрузки пар по стоимости, max_pair_cost, делённый
                    на идеально равномерную долю real / n_pairs (то, ради чего
                    LPT и применяется). 1.0 -- идеальный баланс.

Дополнительно на графики выносится idle_total: при барьерной синхронизации
раундов (Шаг 4) плохой баланс превращается в простой пар.

ЗАПУСК:
    python experiments/compare_lpt_assignment.py \\
        --heuristics direct --n-islands 18 --n-pairs 3 -n 24 --n-runs 15

    # влияние правила размера задачи (этап A алгоритма)
    python experiments/compare_lpt_assignment.py --lpt-sizes min mean max

    # LPT (direct) против SPT (inverse) и случайного порядка -- проверка, что
    # выигрыш даёт именно порядок "длинные работы первыми"
    python experiments/compare_lpt_assignment.py \\
        --heuristics direct inverse random --assignment lpt

Список параметров:
    python experiments/compare_lpt_assignment.py --help
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_delivery_static.assignment import (
    ASSIGNMENT_MODES,
    BALANCE_MODES,
    LPT_RULES,
    LPT_SIZE_RULES,
)
from robot_delivery_static.experiment import ExperimentRow, cargo_sweep, run_suite
from robot_delivery_static.plotting import (
    export_csv,
    plot_gap_vs_ncargos,
    plot_pair_cost_balance,
    plot_pair_load_balance,
    plot_real_vs_estimated_bars,
    plot_real_vs_ncargos,
    plot_win_rate,
)
from robot_delivery_static.priority import CARGO_HEURISTICS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Сравнение полноценного LPT с остальными режимами назначения грузов "
            "парам роботов (Шаг 3 статической модели)."
        )
    )
    p.add_argument("--heuristics", nargs="+", default=["direct"],
                    help=f"Эвристики приоритета. Доступны: {', '.join(sorted(CARGO_HEURISTICS))}. "
                         "В режиме lpt эвристика задаёт порядок: direct -- собственно LPT, "
                         "inverse -- SPT, random -- случайный порядок.")
    p.add_argument("--assignment", nargs="+", default=list(ASSIGNMENT_MODES),
                    help=f"Режимы назначения для сравнения. Доступны: {', '.join(ASSIGNMENT_MODES)}")
    p.add_argument("--lpt-rules", nargs="+", default=list(LPT_RULES),
                    help="Правила выбора пары в LPT, каждое даёт отдельную кривую. "
                         f"Доступны: {', '.join(LPT_RULES)}")
    p.add_argument("--lpt-sizes", nargs="+", default=["min"],
                    help="Правила размера задачи в LPT (этап A). При нескольких значениях "
                         f"каждое даёт свой набор прогонов. Доступны: {', '.join(LPT_SIZE_RULES)}")
    p.add_argument("--balance", default="load", choices=list(BALANCE_MODES),
                    help="Балансировка для режимов literal/cheapest (к lpt не применяется).")
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
                         "../outputs/lpt_assignment относительно этого файла).")
    return p


def _rename_variant(row: ExperimentRow, suffix: str) -> None:
    """Дописать в ключ кривой правило размера задачи -- нужно только когда
    сравниваются несколько значений --lpt-sizes сразу."""
    if row.assignment == "lpt" and suffix:
        row.variant = f"{row.variant}-{suffix}"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    for name, values, allowed in (
        ("эвристики", args.heuristics, sorted(CARGO_HEURISTICS)),
        ("режимы назначения", args.assignment, ASSIGNMENT_MODES),
        ("правила LPT", args.lpt_rules, LPT_RULES),
        ("правила размера LPT", args.lpt_sizes, LPT_SIZE_RULES),
    ):
        unknown = [v for v in values if v not in allowed]
        if unknown:
            print(f"Неизвестные {name}: {', '.join(unknown)}", file=sys.stderr)
            print(f"Доступные: {', '.join(allowed)}", file=sys.stderr)
            return 1

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "outputs", "lpt_assignment"
    )
    os.makedirs(output_dir, exist_ok=True)

    try:
        sweep = cargo_sweep(args.n_pairs, args.n_cargos)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    non_lpt = [m for m in args.assignment if m != "lpt"]
    lpt_included = "lpt" in args.assignment

    # прогоны с lpt повторяются на каждое сочетание (правило выбора пары x
    # правило размера задачи); остальные режимы -- ровно один раз, поэтому
    # они добавляются в первый же прогон и дальше не дублируются
    combos = (
        [(rule, size) for size in args.lpt_sizes for rule in args.lpt_rules]
        if lpt_included else [(args.lpt_rules[0], args.lpt_sizes[0])]
    )

    n_variants = len(args.heuristics) * (len(non_lpt) + lpt_included * len(combos))
    print(f"Островов среды:        {args.n_islands}")
    print(f"Пар роботов:           {args.n_pairs}")
    print(f"Развёртка по грузам:   {sweep[0]}..{sweep[-1]} ({len(sweep)} точек)")
    print(f"Прогонов на сценарий:  {args.n_runs}")
    print(f"Эвристики:             {', '.join(args.heuristics)}")
    print(f"Режимы назначения:     {', '.join(args.assignment)} (балансировка: {args.balance})")
    if lpt_included:
        print(f"Правила LPT:           выбор пары {', '.join(args.lpt_rules)}; "
              f"размер задачи {', '.join(args.lpt_sizes)}")
    print(f"Всего прогонов модели: {n_variants * len(sweep) * args.n_runs}\n")

    show_size = len(args.lpt_sizes) > 1
    rows = []
    first = True
    for rule, size in combos:
        modes = list(args.assignment) if first else (["lpt"] if lpt_included else [])
        if not modes:
            break
        batch = run_suite(
            heuristic_names=args.heuristics,
            n_runs=args.n_runs,
            n_cargos_max=args.n_cargos,
            n_pairs=args.n_pairs,
            n_islands=args.n_islands,
            base_seed=args.seed,
            assignment_modes=modes,
            balance=args.balance,
            lpt_size=size,
            lpt_rule=rule,
            scenario_kwargs=dict(
                free_prob=args.free_prob,
                build_cost_factor=args.build_cost_factor,
            ),
            progress=True,
        )
        if show_size:
            for row in batch:
                _rename_variant(row, size)
        rows.extend(batch)
        first = False

    csv_path = os.path.join(output_dir, "experiment_results.csv")
    export_csv(rows, csv_path)
    print(f"\nСохранено: {csv_path} ({len(rows)} строк)")

    for name, fn in (
        ("compare_real_vs_ncargos.png", plot_real_vs_ncargos),
        ("compare_real_vs_estimated_bars.png", plot_real_vs_estimated_bars),
        ("compare_win_rate.png", plot_win_rate),
        ("pair_load_balance.png", plot_pair_load_balance),
        ("pair_cost_balance.png", plot_pair_cost_balance),
    ):
        path = os.path.join(output_dir, name)
        fn(rows, path)
        print(f"Сохранено: {path}")

    gap_path = os.path.join(output_dir, "compare_gap_static_vs_ncargos.png")
    plot_gap_vs_ncargos(rows, gap_path, gap_field="gap_static")
    print(f"Сохранено: {gap_path}")

    # --- сводка по вариантам ---
    delivered = [r for r in rows if r.all_delivered]
    print("\nСредние значения (по полностью доставленным прогонам):")
    print(f"  {'вариант':22s} {'real':>9s} {'gap_static':>11s} "
          f"{'перекос':>9s} {'idle':>9s}")
    real_by_variant = {}
    imbalance_by_variant = {}
    for variant in sorted({r.variant for r in rows}):
        vals = [r for r in delivered if r.variant == variant]
        if not vals:
            print(f"  {variant:22s} нет полностью доставленных прогонов")
            continue
        avg_real = sum(r.real for r in vals) / len(vals)
        avg_gap = sum(r.gap_static for r in vals) / len(vals)
        avg_imbalance = sum(r.cost_imbalance for r in vals) / len(vals)
        avg_idle = sum(r.idle_total for r in vals) / len(vals)
        real_by_variant[variant] = avg_real
        imbalance_by_variant[variant] = avg_imbalance
        print(f"  {variant:22s} {avg_real:9.3f} {avg_gap:+10.2f}% "
              f"{avg_imbalance:9.3f} {avg_idle:9.3f}   (n={len(vals)})")

    if real_by_variant:
        best = min(real_by_variant, key=lambda v: real_by_variant[v])
        worst = max(real_by_variant, key=lambda v: real_by_variant[v])
        if best != worst:
            diff = 100.0 * (real_by_variant[worst] / real_by_variant[best] - 1.0)
            print(f"\nПо суммарной стоимости лучший вариант: {best} "
                  f"(дешевле худшего, {worst}, на {diff:.1f}%)")
    if imbalance_by_variant:
        best_bal = min(imbalance_by_variant, key=lambda v: imbalance_by_variant[v])
        print(f"По балансу загрузки пар лучший вариант: {best_bal} "
              f"(перекос {imbalance_by_variant[best_bal]:.3f} при идеале 1.000)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
