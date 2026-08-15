"""
СРАВНЕНИЕ ДВУХ АЛГОРИТМОВ РАСЧЁТА СТОИМОСТИ по числу доставляемых грузов:

  Алгоритм 1 "Эвристическая оценка" (estimate_task_costs) - стоимость КАЖДОГО
    груза считается НЕЗАВИСИМО от начальной позиции назначенной пары ("как
    если бы груз был первым"), БЕЗ учёта того, что инфраструктура (уже
    построенные мосты, Pair.built_bridges) меняется по ходу выполнения
    предыдущих грузов. Это одна статическая оценка на весь набор грузов.

  Алгоритм 2 "Реальная работа" (run_sequential_by_priority) - грузы
    выполняются ПОСЛЕДОВАТЕЛЬНО в порядке приоритета; после каждой доставки
    позиция пары обновляется, а построенные мосты добавляются в
    pair.built_bridges и переиспользуются бесплатно следующими грузами той
    же пары (find_route_cheapest_bridge читает already_built = pair.
    built_bridges) - это и есть "изменяемая инфраструктура", максимально
    приближённая к реальному сценарию работы.

Оба алгоритма используют один и тот же маршрутный решатель
(find_route_cheapest_bridge), различие ТОЛЬКО в том, учитывается ли история
уже выполненных грузов (перемещение пары + переиспользование мостов) или
нет - поэтому расхождение между ними отражает именно цену игнорирования
динамики инфраструктуры, а не разницу решателей.

=============================================================================
ОДИН И ТОТ ЖЕ НАБОР ДАННЫХ ДЛЯ ОБОИХ АЛГОРИТМОВ (см. build_dataset):

  * граф островов env, пары роботов pairs и ПОЛНЫЙ список грузов
    cargos_full (N_CARGOS_MAX штук) строятся РОВНО ОДИН РАЗ на весь свип;
  * точка свипа n_cargos берёт ПРЕФИКС cargos_full[:n] - наборы вложены
    друг в друга, при росте n старые грузы не перегенерируются и не
    меняются;
  * оба алгоритма на каждой точке получают ОДИН И ТОТ ЖЕ объект среза
    (через deepcopy, чтобы ни один не мог изменить входные данные другого).

Меняется РОВНО ОДНА переменная - число доставляемых грузов. Граф островов,
координаты роботов, назначение груз->пара, L и seed фиксированы.
=============================================================================
ЭВРИСТИКА ПРИОРИТЕТА (PRIORITY_KIND) задаёт ПОРЯДОК выполнения грузов и
влияет ТОЛЬКО на Алгоритм 2. Оценка Алгоритма 1 - это сумма W_T^i по всем
грузам, посчитанных НЕЗАВИСИМО друг от друга от начальной позиции пары;
сумма не зависит от порядка слагаемых, поэтому кривая "оценка" ИДЕНТИЧНА
для direct/inverse/ratio. Меняется только кривая "реальность" (и, как
следствие, ширина зазора между ними).
=============================================================================

Для КАЖДОГО из двух алгоритмов строятся 2 графика в зависимости от числа
доставляемых грузов:

  1) оценочная/реальная СТОИМОСТЬ выполнения всех операций, Phi(n_cargos);
  2) РЕАЛЬНОЕ время работы (wall-clock, мс) самого расчёта этой стоимости,
     T(n_cargos) - медиана по N_TIME_REPEATS повторов на точку (сценарий не
     меняется между повторами, усредняется только шум замера).

Плюс 2 СРАВНИТЕЛЬНЫХ графика, где обе кривые нанесены одновременно на одни
и те же оси: стоимость Phi(n) и время T(n).

ЗАПУСК: python compare_estimate_vs_real_cost_time.py
"""
import os
import sys
import copy
import time
from statistics import median

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))

import matplotlib.pyplot as plt

from scenario_generator import build_environment, build_cargos_and_pairs
from priority_evaluation import (estimate_task_costs, compute_direct_priority,
                                  compute_inverse_priority, compute_ratio_priority,
                                  run_sequential_by_priority)

# ---- Параметры (фиксированы для всего свипа, кроме n_cargos) -----------------
N_ISLANDS      = 20
N_PAIRS        = 4
SEED           = 0
L              = 6.0
N_CARGOS_MAX   = 150
CARGO_STEP     = 3
N_TIME_REPEATS = 7          # повторов замера времени на точку (берём медиану)

# Эвристика приоритета - задаёт ПОРЯДОК выполнения грузов в Алгоритме 2.
# ВАЖНО: на Алгоритм 1 не влияет вообще (см. примечание в докстринге модуля).
PRIORITY_KIND = "direct"    # "direct" | "inverse" | "ratio"

PRIORITY_FUNCS = {
    "direct":  (compute_direct_priority,  "прямая p=W_T (дорогие первыми)"),
    "inverse": (compute_inverse_priority, "обратная p=1/W_T (дешёвые первыми)"),
    "ratio":   (compute_ratio_priority,   "отношение p=W_b/W_d"),
}

OUT_DIR  = "outputs"

# Идентичность серий кодируется НЕ ТОЛЬКО цветом: у каждой серии свои маркер
# и тип линии, поэтому кривые различимы при ч/б печати и при дальтонизме.
COLOR_ESTIMATE = "#2b6cb0"  # синий - как "оценка" в остальных скриптах пакета
COLOR_REAL     = "#c0392b"  # красный - как "реальная стоимость" в остальных скриптах
STYLE_ESTIMATE = dict(color=COLOR_ESTIMATE, marker="o", linestyle="-",  markersize=3.5)
STYLE_REAL     = dict(color=COLOR_REAL,     marker="s", linestyle="--", markersize=3.5)

LABEL_ESTIMATE = "Алгоритм 1: эвристическая оценка (без изменяемой инфраструктуры)"
LABEL_REAL     = "Алгоритм 2: реальная работа (с изменяемой инфраструктурой)"
# ------------------------------------------------------------------------------


def build_dataset():
    """
    Строит ЕДИНЫЙ набор данных на весь свип: граф островов, пары роботов и
    полный список грузов максимального размера. Точки свипа берут префиксы
    этого списка, поэтому наборы грузов вложены друг в друга, а граф и
    стартовые позиции роботов не меняются вообще.
    """
    env = build_environment(n_islands=N_ISLANDS, seed=SEED)
    cargos_full, pairs = build_cargos_and_pairs(
        n_islands=N_ISLANDS, n_pairs=N_PAIRS, n_cargos=N_CARGOS_MAX, seed=SEED)
    return env, cargos_full, pairs


def run_point(env, cargos, pairs):
    """
    Прогоняет ОБА алгоритма на ОДНОМ И ТОМ ЖЕ наборе (env, cargos, pairs).
    Каждому алгоритму передаются глубокие копии входных данных, чтобы
    возможная мутация внутри одного не повлияла на другой и на следующие
    точки свипа.

    Возвращает (est_cost, est_time_ms, real_cost, real_time_ms, ok).
    """
    # --- Алгоритм 1: эвристическая оценка, БЕЗ изменяемой инфраструктуры ---
    times_est = []
    task_costs = None
    for _ in range(N_TIME_REPEATS):
        cargos_1, pairs_1 = copy.deepcopy(cargos), copy.deepcopy(pairs)
        t0 = time.perf_counter()
        task_costs = estimate_task_costs(env, cargos_1, pairs_1, L=L)
        times_est.append((time.perf_counter() - t0) * 1000.0)

    feasible = [r for r in task_costs.values() if r.feasible]
    all_feasible = len(feasible) == len(task_costs)
    est_cost = sum(r.W_d + r.W_b for r in feasible) if all_feasible else float("nan")

    # --- Алгоритм 2: реальная работа, С изменяемой инфраструктурой ---
    # порядок выполнения задаёт эвристика приоритета PRIORITY_KIND,
    # построенная по оценке Алгоритма 1 (единственная связь между
    # алгоритмами - сам набор данных, расчёт стоимости независим).
    priority_fn, _ = PRIORITY_FUNCS[PRIORITY_KIND]
    priority = priority_fn(task_costs)
    times_real = []
    outcome = None
    for _ in range(N_TIME_REPEATS):
        cargos_2, pairs_2 = copy.deepcopy(cargos), copy.deepcopy(pairs)
        t0 = time.perf_counter()
        outcome = run_sequential_by_priority(env, cargos_2, pairs_2, L=L,
                                              priority=priority, task_costs=task_costs)
        times_real.append((time.perf_counter() - t0) * 1000.0)

    all_delivered = outcome.all_delivered
    real_cost = outcome.real_total if all_delivered else float("nan")

    return (est_cost, median(times_est), real_cost, median(times_real),
            all_feasible and all_delivered)


def sweep():
    env, cargos_full, pairs = build_dataset()
    cargo_range = list(range(N_PAIRS, N_CARGOS_MAX + 1, CARGO_STEP))

    est_cost, est_time_ms, real_cost, real_time_ms = [], [], [], []
    for n in cargo_range:
        cargos = cargos_full[:n]        # префикс единого набора данных
        ec, et, rc, rt, ok = run_point(env, cargos, pairs)
        est_cost.append(ec)
        est_time_ms.append(et)
        real_cost.append(rc)
        real_time_ms.append(rt)
        flag = "" if ok else "  (!) не все грузы доставлены"
        print(f"n={n:4d}   оценка: Phi={ec:9.2f}  t={et:7.3f}мс   "
              f"реальность: Phi={rc:9.2f}  t={rt:7.3f}мс{flag}")

    return cargo_range, est_cost, est_time_ms, real_cost, real_time_ms


def plot(cargo_range, est_cost, est_time_ms, real_cost, real_time_ms, filename):
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    def decorate(ax, ylabel, title):
        ax.set_xlabel("Число доставляемых грузов")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)

    def single(ax, y, style, ylabel, title):
        ax.plot(cargo_range, y, linewidth=2.0, **style)
        decorate(ax, ylabel, title)

    # --- Ряд 1: Алгоритм 1 по отдельности ---
    single(axes[0, 0], est_cost, STYLE_ESTIMATE, "Оценочная стоимость Phi",
           "Алгоритм 1: эвристическая оценка (без изменяемой инфраструктуры)\n"
           "Стоимость выполнения всех операций")
    single(axes[0, 1], est_time_ms, STYLE_ESTIMATE, "Время расчёта, мс",
           "Алгоритм 1: эвристическая оценка\n"
           "Реальное время работы расчёта стоимости")

    # --- Ряд 2: Алгоритм 2 по отдельности ---
    single(axes[1, 0], real_cost, STYLE_REAL, "Реальная стоимость Phi",
           "Алгоритм 2: реальная работа (с изменяемой инфраструктурой)\n"
           "Стоимость выполнения всех операций")
    single(axes[1, 1], real_time_ms, STYLE_REAL, "Время расчёта, мс",
           "Алгоритм 2: реальная работа\n"
           "Реальное время работы расчёта стоимости")

    # --- Ряд 3: СРАВНЕНИЕ обоих алгоритмов на одних осях ---
    ax = axes[2, 0]
    ax.plot(cargo_range, est_cost, linewidth=2.0, label=LABEL_ESTIMATE, **STYLE_ESTIMATE)
    ax.plot(cargo_range, real_cost, linewidth=2.0, label=LABEL_REAL, **STYLE_REAL)
    ax.fill_between(cargo_range, est_cost, real_cost, color=COLOR_ESTIMATE, alpha=0.10)
    decorate(ax, "Стоимость Phi",
             "СРАВНЕНИЕ: стоимость выполнения всех операций\n"
             "(заливка - переоценка от игнорирования изменяемой инфраструктуры)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2, 1]
    ax.plot(cargo_range, est_time_ms, linewidth=2.0, label=LABEL_ESTIMATE, **STYLE_ESTIMATE)
    ax.plot(cargo_range, real_time_ms, linewidth=2.0, label=LABEL_REAL, **STYLE_REAL)
    decorate(ax, "Время расчёта, мс",
             "СРАВНЕНИЕ: реальное время работы расчёта стоимости\n"
             f"(медиана по {N_TIME_REPEATS} повторам)")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"{N_ISLANDS} островов, {N_PAIRS} пары, seed={SEED}, L={L}, эвристика "
        f"приоритета: {PRIORITY_FUNCS[PRIORITY_KIND][1]}\nОба алгоритма на одном "
        f"наборе данных (граф островов, координаты роботов и назначение "
        f"груз→пара фиксированы; меняется только число грузов)",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def report(cargo_range, est_cost, est_time_ms, real_cost, real_time_ms):
    """Итоговая сводка по крайним точкам свипа - переоценка стоимости и
    соотношение времени расчёта."""
    print(f"\n=== СВОДКА (оба алгоритма на одном наборе данных, "
          f"эвристика: {PRIORITY_FUNCS[PRIORITY_KIND][1]}) ===")
    for i in (0, len(cargo_range) - 1):
        n = cargo_range[i]
        ec, rc = est_cost[i], real_cost[i]
        et, rt = est_time_ms[i], real_time_ms[i]
        over = 100 * (ec / rc - 1) if rc else float("nan")
        ratio = et / rt if rt else float("nan")
        print(f"  n={n:4d}:  Phi оценка={ec:9.2f}  Phi реальность={rc:9.2f}  "
              f"переоценка={over:+6.1f}%   |   t оценка={et:7.3f}мс  "
              f"t реальность={rt:7.3f}мс  отношение={ratio:.2f}x")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    cargo_range, est_cost, est_time_ms, real_cost, real_time_ms = sweep()
    print(f"\nСвип завершён за {time.time() - t0:.1f}s")

    fname = os.path.join(
        OUT_DIR,
        f"estimate_vs_real_cost_and_time_{PRIORITY_KIND}_{N_PAIRS}to{N_CARGOS_MAX}.png")
    plot(cargo_range, est_cost, est_time_ms, real_cost, real_time_ms, fname)
    report(cargo_range, est_cost, est_time_ms, real_cost, real_time_ms)
