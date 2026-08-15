"""
СРАВНЕНИЕ ДВУХ МОДЕЛЕЙ -- СТАТИЧЕСКОЙ (static/) И ДИНАМИЧЕСКОЙ (dynamic_v2/) --
по числу доставляемых грузов.

  СТАТИЧЕСКАЯ МОДЕЛЬ (static/)
      Груз ЗАРАНЕЕ закреплён за конкретной парой (доставщик+строитель),
      приоритеты всех грузов считаются ОДИН РАЗ до начала работы
      (estimate_task_costs -- от НАЧАЛЬНЫХ позиций пар), после чего грузы
      выполняются строго по убыванию этого приоритета
      (run_sequential_by_priority). Стоимость каждой задачи в момент
      выполнения пересчитывается заново от ФАКТИЧЕСКОЙ позиции пары, с
      переиспользованием ранее построенных этой парой мостов, -- это и есть
      фактическая стоимость выполнения операций.

  ДИНАМИЧЕСКАЯ МОДЕЛЬ (dynamic_v2/)
      Роботы НЕ склеены в пары заранее: коалиция "доставщик+строитель"
      формируется каждый раунд ПОД выбранный по приоритету груз (Шаг 2).
      Приоритеты грузов пересчитываются в начале КАЖДОГО раунда с учётом уже
      возведённых переправ (built -- глобальное накопительное состояние),
      внутри раунда работает Шаг 4 (за общий мост платит один победитель).
      Фактическая стоимость выполнения операций -- W_d_total + W_b_total по
      фактически исполненному расписанию T (то же, что bracket.real в
      diagnostics.compute_dynamic_cost_bracket).

В ОБЕИХ моделях на графики выносится ТОЛЬКО ФАКТИЧЕСКАЯ стоимость выполнения
всех операций Phi = sum(W_d + W_b) по фактически исполненному расписанию.
Эвристические оценки "до выполнения" (estimated_total / estimated_raw) здесь
не используются вообще.

=============================================================================
ОДИН И ТОТ ЖЕ НАБОР ВХОДНЫХ ДАННЫХ ДЛЯ ОБЕИХ МОДЕЛЕЙ (см. build_common_scenario)

Модели живут в разных пакетах и имеют РАЗНЫЕ структуры данных среды
(static: delivery_model.IslandGraph на networkx; dynamic_v2: graph.Graph на
словарях), поэтому общий сценарий генерируется ОДИН РАЗ в нейтральном виде
(CommonScenario: координаты островов, веса вершин, список рёбер со своим
типом/стоимостями, список грузов, позиции роботов), а затем ПОБУКВЕННО
переносится в структуры каждой модели (to_static_inputs / to_dynamic_inputs).
Ни одна величина при этом не пересчитывается и не разыгрывается заново:

  * граф островов (вершины, их веса w_V, рёбра, их тип free/blocked,
    стоимости проезда w_E, стоимости постройки w_build, длины) -- один и тот же;
  * позиции роботов -- одни и те же: пара k статической модели ставится
    ровно в (deliverer_positions[k], builder_positions[k]) динамической;
  * число пар -- одно и то же: n_pairs пар в статической модели против
    n_pairs доставщиков и n_pairs строителей в динамической (одинаковый
    парк роботов);
  * список грузов -- один и тот же: точка свипа n_cargos берёт ПРЕФИКС
    единого списка cargos[:n], поэтому наборы вложены друг в друга и при
    росте n старые грузы не перегенерируются.

Меняется РОВНО ОДНА переменная -- число доставляемых грузов.

ДВА НЕУСТРАНИМЫХ РАЗЛИЧИЯ МОДЕЛЕЙ, которые НЕ являются различием входных
данных, а являются самой сутью сравнения:
  1) назначение "груз -> робот". Статическая модель требует его на входе
     (Cargo.assigned_pair) -- используется round-robin i % n_pairs, ровно как
     в штатном генераторе static/experiments/scenario_generator.py.
     Динамическая модель назначения на входе не принимает: она формирует
     коалицию под груз сама, каждый раунд.
  2) ограничение L (максимальная длина одного моста). Оно есть только в
     статической модели; в dynamic_v2 любое BLOCKED-ребро можно построить.
     Чтобы среды остались одинаковыми, статическая модель запускается с
     L = inf -- ни одно ребро не отсекается, как и в динамической.

=============================================================================
ЭВРИСТИКА ПРИОРИТЕТА -- ОДНА И ТА ЖЕ ПО СМЫСЛУ В ОБЕИХ МОДЕЛЯХ: ПРЯМАЯ
зависимость приоритета от оценочной стоимости доставки груза (чем дороже --
тем приоритетнее):

  статическая:   p(c_i) = W_T^i = W_d^i + W_b^i   (compute_direct_priority)
  динамическая:  p(c_i) = W_C^i                   (CARGO_HEURISTICS["direct"])

Сама оценка стоимости в каждой модели своя -- ровно та, которую эта модель
умеет считать на своём шаге назначения приоритетов: в статической груз уже
закреплён за парой, поэтому в W_T^i входит и подъезд роботов; в динамической
приоритет по постановке считается НЕЗАВИСИМО от роботов, поэтому W_C^i --
стоимость самого маршрута груза c_start -> c_finish. Подменить одну на другую
нельзя, не сломав модель: это часть самой модели, а не входных данных.

=============================================================================
Строятся 6 графиков в одном файле -- по 2 на каждую модель (фактическая
стоимость выполнения всех операций и реальное время работы расчёта, мс) плюс
2 сравнительных, где обе кривые нанесены на одни и те же оси.

ВРЕМЯ -- это wall-clock время работы САМОГО РАСЧЁТА (мс), медиана по
--time-repeats повторам на точку: для статической модели засекается весь её
пайплайн (оценка стоимостей -> приоритет -> последовательное выполнение), для
динамической -- run_dynamic_rounds целиком (включая пересчёт приоритетов
каждый раунд). Сценарий между повторами не меняется, усредняется только шум
замера.

ЗАПУСК:
    python experiments/compare_static_vs_dynamic_v2.py \\
        --n-islands 18 --n-pairs 3 -n 60 --seed 0
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from statistics import median
from typing import List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# static/: модули импортируются "плоско" (delivery_model, heuristic_cheapest_bridge, ...)
sys.path.insert(0, os.path.join(_ROOT, "static", "core"))
sys.path.insert(0, os.path.join(_ROOT, "static", "experiments"))
# dynamic_v2/: пакет robot_delivery_v2
sys.path.insert(0, os.path.join(_ROOT, "dynamic_v2"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from delivery_model import Cargo as StaticCargo, IslandGraph, Pair as StaticPair
from priority_evaluation import (compute_direct_priority, estimate_task_costs,
                                 run_sequential_by_priority)

from robot_delivery_v2.cargo_priority import get_cargo_heuristic
from robot_delivery_v2.graph import EdgeKind, Graph
from robot_delivery_v2.scheduler import Cargo as DynamicCargo, run_dynamic_rounds

# Ограничение на длину одного моста есть только в статической модели; в
# dynamic_v2 строить можно любое BLOCKED-ребро. Чтобы среда у обеих моделей
# была одинаковой, статическая модель работает без этого ограничения.
L_STATIC = math.inf

# Идентичность серий кодируется НЕ только цветом: у каждой свои маркер и тип
# линии, поэтому кривые различимы в ч/б и при дальтонизме.
COLOR_STATIC, COLOR_DYNAMIC = "#2b6cb0", "#c0392b"
STYLE_STATIC = dict(color=COLOR_STATIC, marker="o", linestyle="-", markersize=3.5)
STYLE_DYNAMIC = dict(color=COLOR_DYNAMIC, marker="s", linestyle="--", markersize=3.5)

LABEL_STATIC = ("Статическая модель (static): груз закреплён за парой,\n"
                "приоритеты считаются один раз до начала работы")
LABEL_DYNAMIC = ("Динамическая модель (dynamic_v2): коалиция под груз,\n"
                 "приоритеты пересчитываются каждый раунд")


# ---------------------------------------------------------------------------
# Общий (нейтральный) сценарий и его перенос в структуры обеих моделей
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EdgeSpec:
    """Ребро среды в нейтральном виде -- ровно те величины, которые обе
    модели понимают одинаково."""
    u: int
    v: int
    is_free: bool     # True -> E_free (готовая переправа), False -> E_blocked (мост нужно построить)
    w_E: float        # стоимость проезда (для blocked -- после постройки)
    w_build: float    # стоимость постройки (0 для free)
    length: float     # физическая длина


@dataclass
class CommonScenario:
    """Единый набор входных данных для ОБЕИХ моделей. Генерируется один раз
    на весь свип; точка свипа берёт префикс cargos[:n]."""
    n_islands: int
    n_pairs: int
    points: List[Tuple[float, float]]
    node_w: List[float]
    edges: List[EdgeSpec]
    cargos: List[Tuple[int, int]]        # (v_start, v_finish), полный список
    deliverer_positions: List[int]
    builder_positions: List[int]
    seed: int


def _mst_edges(points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    """Прим: минимальное остовное дерево на полном графе точек -- гарантирует
    связность базового графа островов (та же схема, что в
    dynamic_v2/robot_delivery_v2/scenario.py)."""
    n = len(points)
    in_tree = [False] * n
    in_tree[0] = True
    dist = [math.dist(points[0], points[j]) for j in range(n)]
    parent = [0] * n
    edges: List[Tuple[int, int]] = []
    for _ in range(n - 1):
        best, best_d = -1, math.inf
        for j in range(n):
            if not in_tree[j] and dist[j] < best_d:
                best, best_d = j, dist[j]
        in_tree[best] = True
        edges.append((min(parent[best], best), max(parent[best], best)))
        for j in range(n):
            if not in_tree[j]:
                d = math.dist(points[best], points[j])
                if d < dist[j]:
                    dist[j] = d
                    parent[j] = best
    return edges


def build_common_scenario(n_islands: int, n_cargos_max: int, n_pairs: int, seed: int,
                          free_prob: float = 0.45, extra_edge_factor: float = 1.3,
                          travel_cost_factor: float = 1.0, build_cost_factor: float = 3.0,
                          node_weight_max: float = 0.5, area: float = 100.0
                          ) -> CommonScenario:
    """Строит связный случайный граф островов (MST + дополнительные рёбра для
    альтернативных маршрутов), грузы и позиции роботов -- в нейтральном виде,
    не привязанном ни к одной из двух моделей. Схема генерации повторяет
    dynamic_v2/robot_delivery_v2/scenario.py.

    Грузы генерируются СРАЗУ в максимальном количестве: точки свипа берут
    префикс этого списка, поэтому при росте n_cargos граф островов, позиции
    роботов и уже сгенерированные грузы не меняются вообще.
    """
    rng = random.Random(seed)

    points = [(rng.uniform(0, area), rng.uniform(0, area)) for _ in range(n_islands)]
    node_w = [rng.uniform(0, node_weight_max) for _ in range(n_islands)]

    edge_set = set(_mst_edges(points))
    n_extra = int(n_islands * extra_edge_factor)
    target = len(edge_set) + n_extra
    attempts = 0
    while len(edge_set) < target and attempts < n_extra * 20:
        attempts += 1
        u, v = rng.randrange(n_islands), rng.randrange(n_islands)
        if u != v:
            edge_set.add((min(u, v), max(u, v)))

    edges: List[EdgeSpec] = []
    for (u, v) in sorted(edge_set):
        d = math.dist(points[u], points[v])
        is_free = rng.random() < free_prob
        edges.append(EdgeSpec(
            u=u, v=v, is_free=is_free,
            w_E=d * travel_cost_factor,
            w_build=0.0 if is_free else d * build_cost_factor,
            length=d,
        ))

    cargos: List[Tuple[int, int]] = []
    for _ in range(n_cargos_max):
        s = rng.randrange(n_islands)
        f = rng.randrange(n_islands)
        while f == s:
            f = rng.randrange(n_islands)
        cargos.append((s, f))

    deliverer_positions = [rng.randrange(n_islands) for _ in range(n_pairs)]
    builder_positions = [rng.randrange(n_islands) for _ in range(n_pairs)]

    return CommonScenario(
        n_islands=n_islands, n_pairs=n_pairs, points=points, node_w=node_w,
        edges=edges, cargos=cargos, deliverer_positions=deliverer_positions,
        builder_positions=builder_positions, seed=seed,
    )


def to_static_inputs(scen: CommonScenario, n_cargos: int):
    """Перенос общего сценария в структуры СТАТИЧЕСКОЙ модели.

    build_cost_multiplier=1.0 -- стоимости постройки берутся ровно такими, как
    в общем сценарии (без штрафного множителя gamma), т.е. численно теми же,
    что видит динамическая модель.

    Назначение "груз -> пара" (обязательный вход статической модели, которого
    у динамической нет) -- round-robin i % n_pairs, как в штатном генераторе
    static/experiments/scenario_generator.py. Оно устойчиво к росту n: у груза
    c_i пара не меняется при добавлении новых грузов.
    """
    env = IslandGraph(build_cost_multiplier=1.0)
    for i in range(scen.n_islands):
        env.add_island(i, w_v=scen.node_w[i], pos=scen.points[i])
    for e in scen.edges:
        if e.is_free:
            env.add_edge(e.u, e.v, kind="free", w_E=e.w_E, length=e.length)
        else:
            env.add_edge(e.u, e.v, kind="blocked", w_E=e.w_E, length=e.length,
                         w_build=e.w_build)

    pairs = [StaticPair(id=f"pair{k + 1}",
                        deliverer_pos=scen.deliverer_positions[k],
                        builder_pos=scen.builder_positions[k])
             for k in range(scen.n_pairs)]

    cargos = [StaticCargo(id=f"c{i + 1}", v_start=s, v_finish=f,
                          assigned_pair=f"pair{i % scen.n_pairs + 1}")
              for i, (s, f) in enumerate(scen.cargos[:n_cargos])]

    return env, cargos, pairs


def to_dynamic_inputs(scen: CommonScenario, n_cargos: int):
    """Перенос ТОГО ЖЕ общего сценария в структуры ДИНАМИЧЕСКОЙ модели.
    Ни одна величина не пересчитывается -- копируются те же числа."""
    G = Graph()
    for i in range(scen.n_islands):
        G.add_node(i, w_V=scen.node_w[i])
    for e in scen.edges:
        if e.is_free:
            G.add_edge(e.u, e.v, EdgeKind.FREE, w_E=e.w_E, length=e.length)
        else:
            G.add_edge(e.u, e.v, EdgeKind.BLOCKED, w_E=e.w_E, w_build=e.w_build,
                       length=e.length)

    cargos = [DynamicCargo(cargo_id=i, start=s, finish=f)
              for i, (s, f) in enumerate(scen.cargos[:n_cargos])]

    return G, cargos, list(scen.deliverer_positions), list(scen.builder_positions)


# ---------------------------------------------------------------------------
# Прогон одной точки свипа
# ---------------------------------------------------------------------------

def run_static_point(scen: CommonScenario, n_cargos: int, repeats: int):
    """Полный пайплайн статической модели: оценка стоимостей всех задач ->
    прямой приоритет (дорогие первыми) -> последовательное выполнение.

    Возвращает (Phi_фактическая, время_мс, ok). Phi -- outcome.real_total, то
    есть сумма ФАКТИЧЕСКИХ W_d + W_b по всем доставкам (каждая пересчитана от
    фактической позиции пары, с переиспользованием уже построенных мостов).
    """
    times: List[float] = []
    outcome = None
    for _ in range(repeats):
        # входные данные пересобираются заново перед каждым повтором (вне
        # замера времени), чтобы повторы были полностью независимы
        env, cargos, pairs = to_static_inputs(scen, n_cargos)
        cargos_run, pairs_run = copy.deepcopy(cargos), copy.deepcopy(pairs)

        t0 = time.perf_counter()
        task_costs = estimate_task_costs(env, cargos, pairs, L=L_STATIC)
        priority = compute_direct_priority(task_costs)   # p = W_T, дорогие первыми
        outcome = run_sequential_by_priority(env, cargos_run, pairs_run, L=L_STATIC,
                                             priority=priority, task_costs=task_costs)
        times.append((time.perf_counter() - t0) * 1000.0)

    ok = outcome.all_delivered
    cost = outcome.real_total if ok else float("nan")
    return cost, median(times), ok


def run_dynamic_point(scen: CommonScenario, n_cargos: int, repeats: int):
    """Полный пайплайн динамической модели: run_dynamic_rounds с накоплением
    built (Алгоритм 2 постановки dynamic_v2) и прямой эвристикой приоритета
    груза (p = W_C, дорогие первыми), пересчитываемой каждый раунд.

    Возвращает (Phi_фактическая, время_мс, ok, n_rounds). Phi = W_d_total +
    W_b_total по фактически исполненному расписанию T -- то же самое, что
    bracket.real в diagnostics.compute_dynamic_cost_bracket.
    """
    heuristic = get_cargo_heuristic("direct")
    times: List[float] = []
    result = None
    for _ in range(repeats):
        G, cargos, d_pos, b_pos = to_dynamic_inputs(scen, n_cargos)

        t0 = time.perf_counter()
        result = run_dynamic_rounds(G, cargos, d_pos, b_pos, heuristic,
                                    rng_seed=scen.seed)
        times.append((time.perf_counter() - t0) * 1000.0)

    ok = result.feasible and result.all_delivered
    cost = (result.W_d_total + result.W_b_total) if ok else float("nan")
    return cost, median(times), ok, result.n_rounds


def sweep(args):
    scen = build_common_scenario(
        n_islands=args.n_islands, n_cargos_max=args.n_cargos, n_pairs=args.n_pairs,
        seed=args.seed, free_prob=args.free_prob,
        build_cost_factor=args.build_cost_factor,
    )

    cargo_range = list(range(args.n_pairs, args.n_cargos + 1, args.cargo_step))
    cost_s, time_s, cost_d, time_d = [], [], [], []

    for n in cargo_range:
        cs, ts, ok_s = run_static_point(scen, n, args.time_repeats)
        cd, td, ok_d, n_rounds = run_dynamic_point(scen, n, args.time_repeats)

        cost_s.append(cs); time_s.append(ts)
        cost_d.append(cd); time_d.append(td)

        flag = "" if (ok_s and ok_d) else "  (!) не все грузы доставлены"
        cheaper = "динам." if cd < cs else ("статич." if cs < cd else "=")
        print(f"n={n:4d}   статич.: Phi={cs:10.2f} t={ts:8.2f}мс   |   "
              f"динам.: Phi={cd:10.2f} t={td:8.2f}мс раундов={n_rounds:3d}   "
              f"дешевле: {cheaper}{flag}")

    return cargo_range, cost_s, time_s, cost_d, time_d


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------

def plot(cargo_range, cost_s, time_s, cost_d, time_d, args, filename):
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    def decorate(ax, ylabel, title):
        ax.set_xlabel("Число доставляемых грузов")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)

    def single(ax, y, style, ylabel, title):
        ax.plot(cargo_range, y, linewidth=2.0, **style)
        decorate(ax, ylabel, title)

    single(axes[0, 0], cost_s, STYLE_STATIC, "Стоимость Φ",
           "Статическая модель (static)\n"
           "Фактическая стоимость выполнения всех операций")
    single(axes[0, 1], time_s, STYLE_STATIC, "Время расчёта, мс",
           "Статическая модель (static)\nРеальное время работы расчёта")

    single(axes[1, 0], cost_d, STYLE_DYNAMIC, "Стоимость Φ",
           "Динамическая модель (dynamic_v2)\n"
           "Фактическая стоимость выполнения всех операций")
    single(axes[1, 1], time_d, STYLE_DYNAMIC, "Время расчёта, мс",
           "Динамическая модель (dynamic_v2)\nРеальное время работы расчёта")

    ax = axes[2, 0]
    ax.plot(cargo_range, cost_s, linewidth=2.0, label=LABEL_STATIC, **STYLE_STATIC)
    ax.plot(cargo_range, cost_d, linewidth=2.0, label=LABEL_DYNAMIC, **STYLE_DYNAMIC)
    ax.fill_between(cargo_range, cost_s, cost_d, color=COLOR_STATIC, alpha=0.10)
    decorate(ax, "Стоимость Φ",
             "СРАВНЕНИЕ: фактическая стоимость выполнения всех операций\n"
             "(ниже = дешевле обошлась доставка всех грузов)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2, 1]
    ax.plot(cargo_range, time_s, linewidth=2.0, label=LABEL_STATIC, **STYLE_STATIC)
    ax.plot(cargo_range, time_d, linewidth=2.0, label=LABEL_DYNAMIC, **STYLE_DYNAMIC)
    decorate(ax, "Время расчёта, мс",
             f"СРАВНЕНИЕ: реальное время работы расчёта\n"
             f"(медиана по {args.time_repeats} повторам)")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"static vs dynamic_v2: {args.n_islands} островов, {args.n_pairs} пар "
        f"(Rd=Rb={args.n_pairs}), seed={args.seed}, эвристика приоритета: прямая "
        f"(дороже груз — выше приоритет)\nОбе модели на ОДНОМ наборе входных "
        f"данных (граф островов, позиции роботов и число пар фиксированы; "
        f"меняется только число грузов)",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def export_csv(path, cargo_range, cost_s, time_s, cost_d, time_d):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_cargos", "static_cost", "static_time_ms",
                    "dynamic_cost", "dynamic_time_ms"])
        for i, n in enumerate(cargo_range):
            w.writerow([n, cost_s[i], time_s[i], cost_d[i], time_d[i]])
    print(f"Сохранено: {path}")


def report(cargo_range, cost_s, time_s, cost_d, time_d):
    print("\n=== СВОДКА (обе модели на одном наборе входных данных) ===")
    print("  Phi -- фактическая стоимость выполнения всех операций, sum(W_d + W_b)")
    for i in (0, len(cargo_range) - 1):
        n = cargo_range[i]
        d = 100 * (cost_s[i] / cost_d[i] - 1) if cost_d[i] else float("nan")
        print(f"  n={n:4d}:  статич. Phi={cost_s[i]:10.2f}   динам. Phi={cost_d[i]:10.2f}   "
              f"статич. дороже на {d:+7.2f}%   |   t_ст={time_s[i]:7.2f}мс "
              f"t_дин={time_d[i]:7.2f}мс")

    wins_d = sum(1 for a, b in zip(cost_s, cost_d) if b < a - 1e-9)
    wins_s = sum(1 for a, b in zip(cost_s, cost_d) if a < b - 1e-9)
    ties = len(cargo_range) - wins_s - wins_d
    deltas = [100 * (a / b - 1) for a, b in zip(cost_s, cost_d) if b]
    print(f"\n  по всей развёртке ({len(cargo_range)} точек) стоимость Phi ниже у:")
    print(f"    статической модели:   {wins_s}")
    print(f"    динамической модели:  {wins_d}")
    print(f"    совпало:              {ties}")
    if deltas:
        print(f"    в среднем статическая дороже динамической на {sum(deltas) / len(deltas):+.2f}%")

    t_ratio = [a / b for a, b in zip(time_s, time_d) if b]
    if t_ratio:
        print(f"    время расчёта: статическая / динамическая в среднем "
              f"{sum(t_ratio) / len(t_ratio):.2f}x")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Сравнение статической (static/) и динамической (dynamic_v2/) "
                    "моделей по числу доставляемых грузов на одном наборе "
                    "входных данных: фактическая стоимость выполнения всех "
                    "операций и реальное время работы расчёта.")
    p.add_argument("--n-islands", type=int, default=18)
    p.add_argument("--n-pairs", type=int, default=3,
                   help="число пар (в dynamic_v2 -- столько же доставщиков и строителей)")
    p.add_argument("-n", "--n-cargos", type=int, default=60,
                   help="верхняя граница развёртки по числу грузов")
    p.add_argument("--cargo-step", type=int, default=2,
                   help="шаг развёртки по числу грузов")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-repeats", type=int, default=5,
                   help="повторов замера времени на точку (берётся медиана)")
    p.add_argument("--free-prob", type=float, default=0.45)
    p.add_argument("--build-cost-factor", type=float, default=3.0)
    p.add_argument("--output-dir",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "outputs"))
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    cargo_range, cost_s, time_s, cost_d, time_d = sweep(args)
    print(f"\nСвип завершён за {time.time() - t0:.1f}s")

    plot(cargo_range, cost_s, time_s, cost_d, time_d, args,
         os.path.join(args.output_dir,
                      f"static_vs_dynamic_v2_{args.n_pairs}to{args.n_cargos}.png"))
    export_csv(os.path.join(args.output_dir, "static_vs_dynamic_v2_results.csv"),
               cargo_range, cost_s, time_s, cost_d, time_d)
    report(cargo_range, cost_s, time_s, cost_d, time_d)


if __name__ == "__main__":
    main()
