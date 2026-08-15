"""
Эксперимент: сравнение ДВУХ АЛГОРИТМОВ постановки, отличающихся РОВНО ОДНИМ --
учитываются ли уже построенные переправы.

  АЛГОРИТМ 1  (fresh_graph_each_round=True)
      УЖЕ ПОСТРОЕННЫЕ мосты НЕ учитываются. Граф берётся в ИЗНАЧАЛЬНОМ виде на
      каждом раунде и для каждой пары роботов: built всегда пуст, поэтому пара
      строит все переправы на пути к своей цели заново, как будто до этого
      раунда не было построено ничего, а стоимость этой постройки суммируется
      в итоговую оценку. Коалиции одного раунда мостами тоже не делятся
      (Шаг 4 отключён), истории возведённого не ведётся вообще, поэтому и
      приоритет груза на Шаге 1 каждый раунд считается заново с полной
      стоимостью постройки.

  АЛГОРИТМ 2  (fresh_graph_each_round=False)
      built -- глобальное накопительное состояние: однажды построенный мост
      больше не строится и не оплачивается, а внутри раунда работает Шаг 4
      (за общий мост платит один случайно выбранный победитель).

Всё остальное у обоих алгоритмов -- один и тот же код: проверка достижимости
(Шаг 0), приоритет груза (Шаг 1), формирование коалиции под груз (ближайший
доставщик + строитель с минимальным числом новых мостов, Шаг 2), барьерная
синхронизация и обновление позиций (Шаги 5-6). Разница целиком в том, какое
built видят эти шаги.

=============================================================================
ОДИН И ТОТ ЖЕ НАБОР ДАННЫХ ДЛЯ ОБОИХ АЛГОРИТМОВ:
  * сценарий (граф островов, позиции доставщиков и строителей, полный список
    грузов) генерируется РОВНО ОДИН РАЗ на весь свип -- при n_cargos_max;
  * точка свипа берёт ПРЕФИКС списка грузов cargos[:n], поэтому наборы
    вложены друг в друга, а граф и координаты роботов не меняются вообще.

    ВАЖНО: generate_scenario разыгрывает позиции роботов ПОСЛЕ грузов из того
    же потока RNG, поэтому вызвать его отдельно для каждого n_cargos нельзя --
    сдвинулся бы поток и вместе с числом грузов поехали бы и позиции роботов.
    Префикс единственного сценария гарантирует, что меняется РОВНО ОДНА
    переменная -- число доставляемых грузов.
=============================================================================

Строит 6 графиков в одном файле: по 2 на каждый алгоритм (стоимость
выполнения всех операций и реальное время работы расчёта в мс) плюс 2
сравнительных, где обе кривые нанесены на одни оси.

ЧТО СЧИТАЕТСЯ СТОИМОСТЬЮ. Результат работы алгоритма по постановке -- это
план доставок и эвристическая оценка стоимости выполнения всех операций для
этого плана, то есть Phi = sum(W_d + W_b) по всем доставкам построенного
плана. Отдельного "факта" в модели нет: вся модель и есть эвристика, а Phi и
есть её ответ. Внутренние промежуточные величины (W_T_initial -- оценка ДО
скидки Шага 4, estimated_raw -- диагностический пересчёт с built=∅) выходом
алгоритма не являются и на графики не выносятся.

ЗАПУСК:
    python experiments/compare_build_aware_priority.py \\
        --n-islands 18 --n-pairs 3 -n 60 --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from statistics import median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_delivery_v2.experiment import run_single
from robot_delivery_v2.scenario import generate_scenario

# Идентичность серий кодируется НЕ только цветом: у каждой свои маркер и тип
# линии, поэтому кривые различимы в ч/б и при дальтонизме.
COLOR_A1, COLOR_A2 = "#2b6cb0", "#c0392b"
STYLE_A1 = dict(color=COLOR_A1, marker="o", linestyle="-", markersize=3.5)
STYLE_A2 = dict(color=COLOR_A2, marker="s", linestyle="--", markersize=3.5)

LABEL_A1 = "Алгоритм 1: построенные мосты не учитываются (изначальный граф каждый раунд)"
LABEL_A2 = "Алгоритм 2: построенные мосты учитываются (built накапливается)"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Сравнение Алгоритма 1 (построенные мосты не учитываются: "
                    "изначальный граф на каждом раунде для каждой пары, постройка "
                    "оплачивается заново) и Алгоритма 2 (built накапливается) "
                    "на одном наборе данных.")
    p.add_argument("--n-islands", type=int, default=18)
    p.add_argument("--n-pairs", type=int, default=3)
    p.add_argument("-n", "--n-cargos", type=int, default=60,
                   help="верхняя граница развёртки по числу грузов")
    p.add_argument("--cargo-step", type=int, default=2,
                   help="шаг развёртки по числу грузов")
    p.add_argument("--heuristic", default="direct", choices=["direct", "inverse", "random"],
                   help="эвристика приоритета (одна и та же для обоих алгоритмов)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-repeats", type=int, default=5,
                   help="повторов замера времени на точку (берётся медиана)")
    p.add_argument("--free-prob", type=float, default=0.45)
    p.add_argument("--build-cost-factor", type=float, default=3.0)
    p.add_argument("--output-dir", default=os.path.join("outputs", "build_aware_priority"))
    return p


def run_point(scenario, heuristic: str, fresh_graph_each_round: bool, repeats: int):
    """Прогон одного алгоритма на одном наборе.
    Возвращает (стоимость, время_мс, ok, n_rounds).

    СТОИМОСТЬ -- это результат работы алгоритма по постановке: "план доставок
    + эвристическая оценка стоимости выполнения всех операций", то есть
    суммарная стоимость построенного плана

        Phi = sum по всем доставкам (W_d + W_b) = W_d_total + W_b_total.

    Отдельного "факта" в модели не существует: вся модель и есть эвристика,
    а эта сумма и есть её оценка стоимости выполнения операций. Внутренние
    промежуточные величины (W_T_initial -- оценка ДО скидки Шага 4 за
    коллизии, estimated_raw из diagnostics -- пересчёт с built=∅) выходом
    алгоритма НЕ являются и здесь не рисуются.
    """
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result, _bracket = run_single(
            scenario, heuristic,
            fresh_graph_each_round=fresh_graph_each_round)
        times.append((time.perf_counter() - t0) * 1000.0)

    ok = result.feasible and result.all_delivered
    cost = (result.W_d_total + result.W_b_total) if ok else float("nan")
    return cost, median(times), ok, result.n_rounds


def sweep(args):
    # --- единый набор данных на весь свип ---
    scenario_full = generate_scenario(
        n_islands=args.n_islands, n_cargos=args.n_cargos, n_pairs=args.n_pairs,
        seed=args.seed, free_prob=args.free_prob,
        build_cost_factor=args.build_cost_factor,
    )

    cargo_range = list(range(args.n_pairs, args.n_cargos + 1, args.cargo_step))
    cost1, t1, cost2, t2 = [], [], [], []

    for n in cargo_range:
        # префикс единого набора: граф и позиции роботов не меняются
        scenario_n = dataclasses.replace(
            scenario_full, cargos=scenario_full.cargos[:n], n_cargos=n)

        # Алгоритм 1 -- fresh_graph_each_round=True, Алгоритм 2 -- False
        c1, m1, ok1, nr1 = run_point(scenario_n, args.heuristic, True, args.time_repeats)
        c2, m2, ok2, nr2 = run_point(scenario_n, args.heuristic, False, args.time_repeats)

        cost1.append(c1); t1.append(m1)
        cost2.append(c2); t2.append(m2)

        flag = "" if (ok1 and ok2) else "  (!) не все грузы доставлены"
        better = "Алг.2" if c2 < c1 else ("Алг.1" if c1 < c2 else "=")
        print(f"n={n:4d}   Алг.1: Phi={c1:10.2f} t={m1:7.2f}мс раундов={nr1:3d}   |   "
              f"Алг.2: Phi={c2:10.2f} t={m2:7.2f}мс раундов={nr2:3d}   "
              f"дешевле: {better}{flag}")

    return cargo_range, cost1, t1, cost2, t2


def plot(cargo_range, cost1, t1, cost2, t2, args, filename):
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    def decorate(ax, ylabel, title):
        ax.set_xlabel("Число доставляемых грузов")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)

    def single(ax, y, style, ylabel, title):
        ax.plot(cargo_range, y, linewidth=2.0, **style)
        decorate(ax, ylabel, title)

    single(axes[0, 0], cost1, STYLE_A1, "Стоимость Φ",
           "Алгоритм 1: построенные мосты не учитываются\n"
           "(изначальный граф каждый раунд); оценка стоимости всех операций")
    single(axes[0, 1], t1, STYLE_A1, "Время расчёта, мс",
           "Алгоритм 1\nРеальное время работы расчёта стоимости")

    single(axes[1, 0], cost2, STYLE_A2, "Стоимость Φ",
           "Алгоритм 2: построенные мосты учитываются (built накапливается)\n"
           "Оценка стоимости выполнения всех операций")
    single(axes[1, 1], t2, STYLE_A2, "Время расчёта, мс",
           "Алгоритм 2\nРеальное время работы расчёта стоимости")

    ax = axes[2, 0]
    ax.plot(cargo_range, cost1, linewidth=2.0, label=LABEL_A1, **STYLE_A1)
    ax.plot(cargo_range, cost2, linewidth=2.0, label=LABEL_A2, **STYLE_A2)
    ax.fill_between(cargo_range, cost1, cost2, color=COLOR_A1, alpha=0.10)
    decorate(ax, "Стоимость Φ",
             "СРАВНЕНИЕ: оценка стоимости выполнения всех операций\n"
             "(ниже = лучше план)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2, 1]
    ax.plot(cargo_range, t1, linewidth=2.0, label=LABEL_A1, **STYLE_A1)
    ax.plot(cargo_range, t2, linewidth=2.0, label=LABEL_A2, **STYLE_A2)
    decorate(ax, "Время расчёта, мс",
             f"СРАВНЕНИЕ: реальное время работы расчёта\n"
             f"(медиана по {args.time_repeats} повторам)")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"dynamic_v2, {args.n_islands} островов, Rd=Rb={args.n_pairs}, "
        f"seed={args.seed}, эвристика: {args.heuristic}\nОба алгоритма на одном "
        f"наборе данных; отличие — учитываются ли уже построенные мосты",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def export_csv(path, cargo_range, cost1, t1, cost2, t2):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_cargos", "alg1_cost", "alg1_time_ms",
                    "alg2_cost", "alg2_time_ms"])
        for i, n in enumerate(cargo_range):
            w.writerow([n, cost1[i], t1[i], cost2[i], t2[i]])
    print(f"Сохранено: {path}")


def report(cargo_range, cost1, t1, cost2, t2, args):
    print(f"\n=== СВОДКА (dynamic_v2, эвристика: {args.heuristic}) ===")
    print("  Phi -- результат алгоритма: стоимость выполнения всех операций")
    for i in (0, len(cargo_range) - 1):
        n = cargo_range[i]
        d = 100 * (cost1[i] / cost2[i] - 1) if cost2[i] else float("nan")
        print(f"  n={n:4d}:  Алг.1 Phi={cost1[i]:10.2f}   Алг.2 Phi={cost2[i]:10.2f}   "
              f"Алг.1 дороже на {d:+6.2f}%   |   t1={t1[i]:6.2f}мс t2={t2[i]:6.2f}мс")

    wins2 = sum(1 for a, b in zip(cost1, cost2) if b < a - 1e-9)
    wins1 = sum(1 for a, b in zip(cost1, cost2) if a < b - 1e-9)
    ties = len(cargo_range) - wins1 - wins2
    deltas = [100 * (a / b - 1) for a, b in zip(cost1, cost2) if b]
    print(f"\n  по всей развёртке ({len(cargo_range)} точек) стоимость Phi ниже у:")
    print(f"    Алгоритма 1 (мосты не учитываются): {wins1}")
    print(f"    Алгоритма 2 (мосты учитываются):    {wins2}")
    print(f"    совпало:                            {ties}")
    if deltas:
        print(f"    в среднем Алгоритм 1 дороже на {sum(deltas)/len(deltas):+.2f}%")


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    cargo_range, cost1, t1, cost2, t2 = sweep(args)
    print(f"\nСвип завершён за {time.time() - t0:.1f}s")

    plot(cargo_range, cost1, t1, cost2, t2, args,
         os.path.join(args.output_dir, f"build_aware_{args.heuristic}.png"))
    export_csv(os.path.join(args.output_dir, "results.csv"),
               cargo_range, cost1, t1, cost2, t2)
    report(cargo_range, cost1, t1, cost2, t2, args)


if __name__ == "__main__":
    main()
