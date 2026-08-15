"""
СРАВНЕНИЕ ДВУХ МОДЕЛЕЙ -- СТАТИЧЕСКОЙ (static_v2/) И ДИНАМИЧЕСКОЙ
(dynamic_v2/) -- по числу доставляемых грузов.

  СТАТИЧЕСКАЯ МОДЕЛЬ (static_v2/robot_delivery_static, RUN-STATIC)
      Пары "доставщик+строитель" формируются ОДИН РАЗ до начала работы
      (Шаг 2, pairing.py) и больше не пересобираются. Грузы распределяются
      между парами тоже ОДИН РАЗ (Шаг 3, assignment.py): строится таблица
      пара x груз с оценкой W_T(U_k, c_i) от НАЧАЛЬНЫХ позиций пары при
      built = ∅, по ней считается приоритет, и грузы разбираются одним жадным
      проходом. Дальше (Шаг 4) пары везут свои очереди по раундам, и вот там
      стоимость каждой доставки пересчитывается уже от ФАКТИЧЕСКИХ позиций
      роботов с учётом глобального накопленного built.

      Режим Шага 3 по умолчанию -- LPT (--assignment lpt): грузы разбираются
      по убыванию размера задачи, каждый достаётся наименее загруженной паре
      (Graham 1969). Именно он делает сравнение с динамической моделью
      осмысленным: у динамической модели балансировка встроена в саму схему
      (коалиция собирается под груз каждый раунд, и работа расходится по
      роботам сама собой), тогда как режимы literal/cheapest могут отдать
      почти всё одной паре, и статическая модель проигрывала бы не своей
      статичностью, а перекосом распределения. Правило выбора пары --
      --lpt-rule, правило размера задачи -- --lpt-size (см.
      static_v2/robot_delivery_static/assignment.py, ASSIGN-LPT).

  ДИНАМИЧЕСКАЯ МОДЕЛЬ (dynamic_v2/robot_delivery_v2, RUN-DYNAMIC-ROUNDS)
      Роботы НЕ склеены в пары заранее: коалиция формируется каждый раунд ПОД
      выбранный по приоритету груз (Шаг 2). Приоритеты грузов пересчитываются
      в начале КАЖДОГО раунда с учётом уже возведённых переправ (built --
      глобальное накопительное состояние), внутри раунда работает Шаг 4 (за
      общий мост платит один победитель).

В ОБЕИХ моделях на графики выносится ТОЛЬКО ФАКТИЧЕСКАЯ стоимость выполнения
всех операций Phi = W_d_total + W_b_total по фактически исполненному
расписанию T. Эвристические оценки "до выполнения" (у статической модели это
RunResult.estimated_static -- сумма W_T Шага 3) здесь не используются вообще.

=============================================================================
ОДИН И ТОТ ЖЕ НАБОР ВХОДНЫХ ДАННЫХ ДЛЯ ОБЕИХ МОДЕЛЕЙ (см. build_common_scenario)

Пакеты static_v2 и dynamic_v2 -- независимые копии, у каждой свои модули
graph.py / costs.py / feasibility.py (текстуально совпадающие, но это РАЗНЫЕ
классы: EdgeKind статической модели не равен EdgeKind динамической). Поэтому
общий сценарий генерируется ОДИН РАЗ в нейтральном виде (CommonScenario:
координаты островов, веса вершин, список рёбер со своим типом/стоимостями,
список грузов, позиции роботов), а затем побуквенно переносится в структуры
каждого пакета (to_static_inputs / to_dynamic_inputs). Ни одна величина при
этом не пересчитывается и не разыгрывается заново:

  * граф островов (вершины, их веса w_V, рёбра, их тип free/blocked,
    стоимости проезда w_E, стоимости постройки w_build, длины) -- один и тот же;
  * позиции роботов -- одни и те же: и статическая, и динамическая модель
    получают ровно те же deliverer_positions / builder_positions;
  * число пар -- одно и то же: n_pairs доставщиков и n_pairs строителей;
  * список грузов -- один и тот же: точка свипа n_cargos берёт ПРЕФИКС
    единого списка cargos[:n], поэтому наборы вложены друг в друга и при
    росте n старые грузы не перегенерируются.

Именно из-за префикса грузы генерируются сразу в максимальном количестве:
в штатном scenario.generate_scenario позиции роботов разыгрываются ПОСЛЕ
грузов, так что при генерации "на каждое n заново" смена n сдвигала бы и
позиции роботов -- то есть менялась бы не одна переменная, а три.

Меняется РОВНО ОДНА переменная -- число доставляемых грузов.

Единственное различие моделей на входе -- то, ради чего сравнение и делается:
статическая модель сама формирует пары (Шаг 2) и сама раздаёт грузы по парам
(Шаг 3) один раз в начале, динамическая собирает коалицию под груз каждый
раунд. Никаких дополнительных ограничений (вроде максимальной длины моста) ни
у одной из моделей нет: любое BLOCKED-ребро можно построить в обеих.

=============================================================================
ЭВРИСТИКА ПРИОРИТЕТА (--heuristic) -- ОДНА И ТА ЖЕ ПО СМЫСЛУ В ОБЕИХ МОДЕЛЯХ.
Обе модели получают эвристику с ОДНИМ И ТЕМ ЖЕ ключом, то есть одно и то же
правило пересчёта "оценка стоимости -> приоритет":

  direct  -- ПРЯМАЯ: дороже груз -- выше приоритет (эвристика постановки).
             статическая  p(U_k, c_i) = W_T(U_k, c_i)
             динамическая p(c_i)      = W_C(c_i)
  inverse -- ОБРАТНАЯ: дешевле груз -- выше приоритет.
             статическая  p(U_k, c_i) = 1 / W_T(U_k, c_i)
             динамическая p(c_i)      = 1 / W_C(c_i)
  random  -- baseline: p ~ U(0,1), от стоимости не зависит вообще.

Сама оценка стоимости в каждой модели своя -- ровно та, которую эта модель
умеет считать на своём шаге назначения приоритетов: в статической груз
оценивается для конкретной пары, поэтому в W_T входит и подъезд роботов; в
динамической приоритет по постановке считается НЕЗАВИСИМО от роботов, поэтому
W_C -- стоимость самого маршрута груза c_start -> c_finish. Подменить одну на
другую нельзя, не сломав модель: это часть самой модели, а не входных данных.

Внимание при сравнении прогонов с разными --heuristic: в статической модели
эвристика влияет не только на порядок доставок, но и на распределение грузов
по парам (Шаг 3), причём по-разному в разных режимах:

  lpt (по умолчанию) -- эвристика задаёт только ПОРЯДОК разбора грузов
      (direct -- по убыванию размера задачи, то есть собственно LPT; inverse
      -- SPT, короткие первыми; random -- без сортировки), а пару выбирает
      само правило LPT по загрузке. Это ближе всего к динамической модели, где
      эвристика тоже влияет только на порядок.
  literal -- эвристика определяет ещё и то, КАКОЙ ПАРЕ достанется груз (груз
      уходит паре с максимальным p: при direct -- самой дорогой для неё, при
      inverse -- самой дешёвой), а также порог балансировки.
  cheapest -- эвристика задаёт порядок, пару выбирает минимум стоимости.

В динамической модели -- только на порядок выбора грузов по раундам.

=============================================================================
Строятся 6 графиков в одном файле -- по 2 на каждую модель (фактическая
стоимость выполнения всех операций и реальное время работы расчёта, мс) плюс
2 сравнительных, где обе кривые нанесены на одни и те же оси.

ВРЕМЯ -- это wall-clock время работы САМОГО РАСЧЁТА (мс), медиана по
--time-repeats повторам на точку: для статической модели засекается весь
run_static (Шаг 0 -> формирование пар -> таблица пара x груз -> распределение
-> исполнение по раундам), для динамической -- run_dynamic_rounds целиком
(включая пересчёт приоритетов каждый раунд). Сценарий между повторами не
меняется, усредняется только шум замера.

ЗАПУСК:
    python experiments/compare_static_v2_vs_dynamic_v2.py \\
        --n-islands 20 --n-pairs 5 -n 200 --cargo-step 1 --seed 7 \\
        --heuristic direct

    # прежний режим Шага 3 (буквальное правило постановки) для сравнения
    python experiments/compare_static_v2_vs_dynamic_v2.py --assignment literal
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, os.path.join(_ROOT, "static_v2"))
sys.path.insert(0, os.path.join(_ROOT, "dynamic_v2"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_delivery_static.assignment import (ASSIGNMENT_MODES, BALANCE_MODES,
                                              LPT_RULES, LPT_SIZE_RULES)
from robot_delivery_static.graph import EdgeKind as StaticEdgeKind, Graph as StaticGraph
from robot_delivery_static.model import Cargo as StaticCargo
from robot_delivery_static.priority import (CARGO_HEURISTICS as STATIC_HEURISTICS,
                                            get_cargo_heuristic as get_static_heuristic)
from robot_delivery_static.scheduler import run_static

from robot_delivery_v2.cargo_priority import (CARGO_HEURISTICS as DYNAMIC_HEURISTICS,
                                              get_cargo_heuristic as get_dynamic_heuristic)
from robot_delivery_v2.graph import EdgeKind as DynamicEdgeKind, Graph as DynamicGraph
from robot_delivery_v2.scheduler import Cargo as DynamicCargo, run_dynamic_rounds

# Ключи, которые есть в ОБОИХ пакетах: только по ним сравнение осмысленно
# (одно и то же правило "оценка -> приоритет" у обеих моделей).
COMMON_HEURISTICS = sorted(set(STATIC_HEURISTICS) & set(DYNAMIC_HEURISTICS))

# Подпись эвристики для заголовка -- по смыслу, одинаковая для обеих моделей
# (различие лишь в том, что статическая подставляет W_T, а динамическая W_C).
HEURISTIC_TITLE = {
    "direct": "прямая (дороже груз — выше приоритет)",
    "inverse": "обратная (дешевле груз — выше приоритет)",
    "random": "случайная (baseline, от стоимости не зависит)",
}

# Идентичность серий кодируется НЕ только цветом: у каждой свои маркер и тип
# линии, поэтому кривые различимы в ч/б и при дальтонизме.
COLOR_STATIC, COLOR_DYNAMIC = "#2b6cb0", "#c0392b"
STYLE_STATIC = dict(color=COLOR_STATIC, marker="o", linestyle="-", markersize=3.5)
STYLE_DYNAMIC = dict(color=COLOR_DYNAMIC, marker="s", linestyle="--", markersize=3.5)

LABEL_STATIC = ("Статическая модель (static_v2): пары и распределение грузов\n"
                "фиксируются один раз до начала работы")
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
    cargos: List[Tuple[int, int]]        # (start, finish), полный список
    deliverer_positions: List[int]
    builder_positions: List[int]
    seed: int


def _mst_edges(points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    """Прим: минимальное остовное дерево на полном графе точек -- гарантирует
    связность базового графа островов (та же схема, что в scenario.py обоих
    пакетов)."""
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
    не привязанном ни к одной из двух моделей. Схема генерации и все параметры
    по умолчанию повторяют scenario.generate_scenario обоих пакетов (они там
    посимвольно совпадают).

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

    # sorted(), а не обход set: порядок обхода задаёт порядок разыгрывания
    # free/blocked, и он должен быть воспроизводим от запуска к запуску.
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
    """Перенос общего сценария в структуры СТАТИЧЕСКОЙ модели (static_v2).
    Назначение "груз -> пара" на вход не передаётся: статическая модель делает
    его сама на Шаге 3."""
    G = StaticGraph()
    for i in range(scen.n_islands):
        G.add_node(i, w_V=scen.node_w[i])
    for e in scen.edges:
        if e.is_free:
            G.add_edge(e.u, e.v, StaticEdgeKind.FREE, w_E=e.w_E, length=e.length)
        else:
            G.add_edge(e.u, e.v, StaticEdgeKind.BLOCKED, w_E=e.w_E,
                       w_build=e.w_build, length=e.length)

    cargos = [StaticCargo(cargo_id=i, start=s, finish=f)
              for i, (s, f) in enumerate(scen.cargos[:n_cargos])]

    return G, cargos, list(scen.deliverer_positions), list(scen.builder_positions)


def to_dynamic_inputs(scen: CommonScenario, n_cargos: int):
    """Перенос ТОГО ЖЕ общего сценария в структуры ДИНАМИЧЕСКОЙ модели
    (dynamic_v2). Ни одна величина не пересчитывается -- копируются те же
    числа."""
    G = DynamicGraph()
    for i in range(scen.n_islands):
        G.add_node(i, w_V=scen.node_w[i])
    for e in scen.edges:
        if e.is_free:
            G.add_edge(e.u, e.v, DynamicEdgeKind.FREE, w_E=e.w_E, length=e.length)
        else:
            G.add_edge(e.u, e.v, DynamicEdgeKind.BLOCKED, w_E=e.w_E,
                       w_build=e.w_build, length=e.length)

    cargos = [DynamicCargo(cargo_id=i, start=s, finish=f)
              for i, (s, f) in enumerate(scen.cargos[:n_cargos])]

    return G, cargos, list(scen.deliverer_positions), list(scen.builder_positions)


# ---------------------------------------------------------------------------
# Прогон одной точки свипа
# ---------------------------------------------------------------------------

def run_static_point(scen: CommonScenario, n_cargos: int, repeats: int,
                     heuristic_key: str, assignment_mode: str, balance: str,
                     lpt_size: str = "min", lpt_rule: str = "load"):
    """Полный пайплайн статической модели (RUN-STATIC).

    Возвращает (Phi_фактическая, время_мс, ok, n_rounds). Phi = W_d_total +
    W_b_total по фактически исполненному расписанию T, то есть сумма
    ФАКТИЧЕСКИХ стоимостей операций Шага 4 (каждая пересчитана от фактической
    позиции пары, с учётом уже построенных переправ), а НЕ оценка Шага 3.

    lpt_size / lpt_rule работают только при assignment_mode="lpt" (режим по
    умолчанию), balance -- только при остальных режимах: LPT балансирует
    загрузку пар сам.
    """
    heuristic = get_static_heuristic(heuristic_key)
    times: List[float] = []
    result = None
    for _ in range(repeats):
        # входные данные пересобираются заново перед каждым повтором (вне
        # замера времени), чтобы повторы были полностью независимы
        G, cargos, d_pos, b_pos = to_static_inputs(scen, n_cargos)

        t0 = time.perf_counter()
        result = run_static(G, cargos, d_pos, b_pos, heuristic,
                            assignment_mode=assignment_mode, balance=balance,
                            lpt_size=lpt_size, lpt_rule=lpt_rule,
                            rng_seed=scen.seed)
        times.append((time.perf_counter() - t0) * 1000.0)

    ok = result.feasible and result.all_delivered
    cost = result.real if ok else float("nan")
    return cost, median(times), ok, result.n_rounds


def run_dynamic_point(scen: CommonScenario, n_cargos: int, repeats: int,
                      heuristic_key: str):
    """Полный пайплайн динамической модели: run_dynamic_rounds с накоплением
    built (Алгоритм 2 постановки dynamic_v2) и эвристикой приоритета груза
    (от W_C), пересчитываемой каждый раунд.

    Возвращает (Phi_фактическая, время_мс, ok, n_rounds). Phi = W_d_total +
    W_b_total по фактически исполненному расписанию T.
    """
    heuristic = get_dynamic_heuristic(heuristic_key)
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

    cargo_range = list(range(args.n_cargos_min, args.n_cargos + 1, args.cargo_step))
    cost_s, time_s, cost_d, time_d = [], [], [], []

    for n in cargo_range:
        cs, ts, ok_s, rounds_s = run_static_point(scen, n, args.time_repeats,
                                                  args.heuristic, args.assignment,
                                                  args.balance, args.lpt_size,
                                                  args.lpt_rule)
        cd, td, ok_d, rounds_d = run_dynamic_point(scen, n, args.time_repeats,
                                                   args.heuristic)

        cost_s.append(cs); time_s.append(ts)
        cost_d.append(cd); time_d.append(td)

        flag = "" if (ok_s and ok_d) else "  (!) не все грузы доставлены"
        cheaper = "динам." if cd < cs else ("статич." if cs < cd else "=")
        print(f"n={n:4d}   статич.: Phi={cs:10.2f} t={ts:8.2f}мс раундов={rounds_s:3d}   |   "
              f"динам.: Phi={cd:10.2f} t={td:8.2f}мс раундов={rounds_d:3d}   "
              f"дешевле: {cheaper}{flag}")

    return cargo_range, cost_s, time_s, cost_d, time_d


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------

def assignment_tag(args) -> str:
    """Короткая метка режима Шага 3 для имён файлов: у lpt два разных правила
    выбора пары, и их результаты смешивать нельзя."""
    return f"lpt-{args.lpt_rule}" if args.assignment == "lpt" else args.assignment


def assignment_title(args) -> str:
    """Подпись режима Шага 3 для заголовка графика."""
    if args.assignment == "lpt":
        return (f"LPT (размер задачи: {args.lpt_size}, выбор пары: {args.lpt_rule})")
    return f"{args.assignment}, балансировка: {args.balance}"


def plot(cargo_range, cost_s, time_s, cost_d, time_d, args, filename):
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    # при шаге 1 точек в 2 раза больше, чем при шаге 2, и маркеры дефолтного
    # размера сливаются в сплошную полосу -- уменьшаем их
    ms = 3.5 if len(cargo_range) <= 120 else 2.2
    style_s = dict(STYLE_STATIC, markersize=ms)
    style_d = dict(STYLE_DYNAMIC, markersize=ms)

    def decorate(ax, ylabel, title):
        ax.set_xlabel("Число доставляемых грузов")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)

    def single(ax, y, style, ylabel, title):
        ax.plot(cargo_range, y, linewidth=2.0, **style)
        decorate(ax, ylabel, title)

    single(axes[0, 0], cost_s, style_s, "Стоимость Φ",
           "Статическая модель (static_v2)\n"
           "Фактическая стоимость выполнения всех операций")
    single(axes[0, 1], time_s, style_s, "Время расчёта, мс",
           "Статическая модель (static_v2)\nРеальное время работы расчёта")

    single(axes[1, 0], cost_d, style_d, "Стоимость Φ",
           "Динамическая модель (dynamic_v2)\n"
           "Фактическая стоимость выполнения всех операций")
    single(axes[1, 1], time_d, style_d, "Время расчёта, мс",
           "Динамическая модель (dynamic_v2)\nРеальное время работы расчёта")

    ax = axes[2, 0]
    ax.plot(cargo_range, cost_s, linewidth=2.0, label=LABEL_STATIC, **style_s)
    ax.plot(cargo_range, cost_d, linewidth=2.0, label=LABEL_DYNAMIC, **style_d)
    ax.fill_between(cargo_range, cost_s, cost_d, color=COLOR_STATIC, alpha=0.10)
    decorate(ax, "Стоимость Φ",
             "СРАВНЕНИЕ: фактическая стоимость выполнения всех операций\n"
             "(ниже = дешевле обошлась доставка всех грузов)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2, 1]
    ax.plot(cargo_range, time_s, linewidth=2.0, label=LABEL_STATIC, **style_s)
    ax.plot(cargo_range, time_d, linewidth=2.0, label=LABEL_DYNAMIC, **style_d)
    decorate(ax, "Время расчёта, мс",
             f"СРАВНЕНИЕ: реальное время работы расчёта\n"
             f"(медиана по {args.time_repeats} повторам)")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"static_v2 vs dynamic_v2: {args.n_islands} островов, {args.n_pairs} пар "
        f"(Rd=Rb={args.n_pairs}), seed={args.seed}, эвристика приоритета: "
        f"{HEURISTIC_TITLE.get(args.heuristic, args.heuristic)}\n"
        f"Распределение грузов по парам в статической модели (Шаг 3): "
        f"{assignment_title(args)}\nОбе модели на "
        f"ОДНОМ наборе входных данных (граф островов, позиции роботов и число "
        f"пар фиксированы; меняется только число грузов)",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def export_csv(path, cargo_range, cost_s, time_s, cost_d, time_d):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_cargos", "static_v2_cost", "static_v2_time_ms",
                    "dynamic_v2_cost", "dynamic_v2_time_ms"])
        for i, n in enumerate(cargo_range):
            w.writerow([n, cost_s[i], time_s[i], cost_d[i], time_d[i]])
    print(f"Сохранено: {path}")


def report(cargo_range, cost_s, time_s, cost_d, time_d):
    print("\n=== СВОДКА (обе модели на одном наборе входных данных) ===")
    print("  Phi -- фактическая стоимость выполнения всех операций, W_d_total + W_b_total")
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
        description="Сравнение статической (static_v2/) и динамической "
                    "(dynamic_v2/) моделей по числу доставляемых грузов на "
                    "одном наборе входных данных: фактическая стоимость "
                    "выполнения всех операций и реальное время работы расчёта.")
    p.add_argument("--n-islands", type=int, default=20)
    p.add_argument("--n-pairs", type=int, default=5,
                   help="число пар статической модели (в dynamic_v2 -- столько же "
                        "доставщиков и строителей)")
    p.add_argument("-n", "--n-cargos", type=int, default=200,
                   help="верхняя граница развёртки по числу грузов")
    p.add_argument("--n-cargos-min", type=int, default=None,
                   help="нижняя граница развёртки (по умолчанию -- число пар; "
                        "с --n-cargos-min 1 --cargo-step 1 и -n 200 получается "
                        "ровно 200 точек)")
    p.add_argument("--cargo-step", type=int, default=2,
                   help="шаг развёртки по числу грузов")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--heuristic", default="direct", choices=COMMON_HEURISTICS,
                   help="эвристика приоритета груза, ОДНА И ТА ЖЕ для обеих "
                        "моделей: direct -- дороже груз выше приоритет, "
                        "inverse -- дешевле груз выше приоритет, random -- baseline")
    p.add_argument("--time-repeats", type=int, default=5,
                   help="повторов замера времени на точку (берётся медиана)")
    p.add_argument("--assignment", default="lpt", choices=list(ASSIGNMENT_MODES),
                   help="режим распределения грузов по парам на Шаге 3 статической "
                        "модели (см. static_v2/assignment.py). По умолчанию lpt -- "
                        "полноценный LPT, который балансирует загрузку пар и потому "
                        "сопоставим со встроенной балансировкой динамической модели; "
                        "literal -- буквальное правило постановки, cheapest -- "
                        "greedy-nearest")
    p.add_argument("--balance", default="load", choices=list(BALANCE_MODES),
                   help="балансировка загрузки пар на Шаге 3 статической модели; "
                        "к режиму lpt не применяется (он балансирует сам)")
    p.add_argument("--lpt-size", default="min", choices=list(LPT_SIZE_RULES),
                   help="только для --assignment lpt: как свернуть стоимости груза у "
                        "всех пар в один размер задачи для сортировки (min -- у "
                        "лучшего исполнителя, mean -- средняя, max -- у худшего)")
    p.add_argument("--lpt-rule", default="load", choices=list(LPT_RULES),
                   help="только для --assignment lpt: правило выбора пары. load -- "
                        "классический Грэм (argmin накопленной загрузки), completion "
                        "-- argmin (загрузка + стоимость груза для этой пары)")
    p.add_argument("--free-prob", type=float, default=0.45)
    p.add_argument("--build-cost-factor", type=float, default=3.0)
    p.add_argument("--output-dir",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "outputs"))
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_cargos_min is None:
        args.n_cargos_min = args.n_pairs
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Островов: {args.n_islands}, пар: {args.n_pairs}, seed: {args.seed}, "
          f"эвристика: {args.heuristic}")
    print(f"Шаг 3 статической модели: {assignment_title(args)}\n")

    t0 = time.time()
    cargo_range, cost_s, time_s, cost_d, time_d = sweep(args)
    print(f"\nСвип завершён за {time.time() - t0:.1f}s")

    # эвристика и режим Шага 3 -- в имени файла: прогоны direct/inverse/random
    # и lpt/literal/cheapest не должны затирать друг друга
    stem = (f"static_v2_vs_dynamic_v2_{args.heuristic}_{assignment_tag(args)}_"
            f"{args.n_cargos_min}to{args.n_cargos}_step{args.cargo_step}")
    plot(cargo_range, cost_s, time_s, cost_d, time_d, args,
         os.path.join(args.output_dir, f"{stem}.png"))
    export_csv(os.path.join(args.output_dir, f"{stem}.csv"),
               cargo_range, cost_s, time_s, cost_d, time_d)
    report(cargo_range, cost_s, time_s, cost_d, time_d)


if __name__ == "__main__":
    main()
