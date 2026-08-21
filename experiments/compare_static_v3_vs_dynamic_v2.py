"""
СРАВНЕНИЕ КРАЙНИХ ТОЧЕК СПЕКТРА ПЛАНИРОВАНИЯ -- СЛУЧАЙНОЙ СТАТИЧЕСКОЙ МОДЕЛИ
(static_v3/) И ДИНАМИЧЕСКОЙ МОДЕЛИ (dynamic_v2/) -- по числу доставляемых грузов.

Из трёх моделей репозитория эти две максимально далеки друг от друга: одна не
планирует вообще, вторая пересматривает решение каждый раунд. Поэтому разрыв
между ними -- это ВЕРХНЯЯ ОЦЕНКА того, сколько вообще можно выиграть
планированием на данных сценариях; сравнение static_v3 со static_v2
(compare_static_v3_vs_static_v2.py) показывает, какую часть этого запаса
забирают статические эвристики.

  static_v3 (robot_delivery_random, RUN-RANDOM-STATIC) -- BASELINE, НЕ ПЛАНИРУЕТ.
      Шаг 2: пары "доставщик+строитель" формируются ОДИН РАЗ и СЛУЧАЙНО
             (перемешать доставщиков, перемешать строителей, склеить
             позиционно); граф в этот шаг не передаётся.
      Шаг 3: грузы распределяются между парами ОДИН РАЗ и СЛУЧАЙНО; никакой
             оценки стоимости не считается, таблицы "пара x груз" нет.
      Шаг 4: пары везут свои фиксированные очереди по раундам.
      Режим раздачи -- --v3-assignment: balanced (перемешать и раздать по
      кругу; число грузов у пар отличается максимум на 1) или uniform
      (каждому грузу независимо случайная пара).

  dynamic_v2 (robot_delivery_v2, RUN-DYNAMIC-ROUNDS) -- ПЛАНИРУЕТ КАЖДЫЙ РАУНД.
      Роботы НЕ склеены в пары вообще: коалиция формируется заново каждый
      раунд ПОД выбранный по приоритету груз (Шаг 2). Приоритеты грузов
      пересчитываются в начале КАЖДОГО раунда с учётом уже возведённых
      переправ; в начале раунда берутся N самых приоритетных из ещё не
      доставленных.

Всё остальное у моделей общее и текстуально совпадает: модель графа,
ROUTE-AND-COST, ESTIMATE-TASK-COST, Шаг 0, правило конфликта за общий мост
(платит случайно выбранная коалиция, остальным бесплатно) и глобальность built
(однажды построенный мост никогда не строится повторно).

НА ГРАФИКИ ВЫНОСИТСЯ ТОЛЬКО ФАКТИЧЕСКАЯ стоимость выполнения всех операций
Phi = W_d_total + W_b_total по фактически исполненному расписанию T. У
static_v3 это, по её постановке (Шаг 5), и есть эвристическая оценка работы
модели -- никакой отдельной оценки "до выполнения" у неё не существует.

=============================================================================
ЭВРИСТИКА ПРИОРИТЕТА (--heuristic) ЕСТЬ ТОЛЬКО У ДИНАМИЧЕСКОЙ МОДЕЛИ

Это главная асимметрия сравнения, и её надо держать в голове при чтении
графиков. В dynamic_v2 приоритет груза p(c_i) = f(W_C(c_i)) считается по
стоимости собственного маршрута груза c_start -> c_finish и пересчитывается
каждый раунд:

  inverse -- p(c_i) = 1 / W_T^i: дешевле груз -- выше приоритет. ЗНАЧЕНИЕ ПО
             УМОЛЧАНИЮ этого эксперимента: динамическая модель прогоняется
             именно с обратной эвристикой
  direct  -- p(c_i) = W_T^i: дороже груз -- выше приоритет
  random  -- baseline: p ~ U(0,1), от стоимости не зависит

В static_v3 приоритетов НЕТ ВООБЩЕ: и пары, и распределение грузов случайны,
стоимость доставки ни на что не влияет и потому не вычисляется. Поэтому кривая
static_v3 на графиках ОДНА И ТА ЖЕ при любом --heuristic, а меняется только
кривая dynamic_v2. Прогон с --heuristic random особенно нагляден: он
показывает, сколько даёт одна лишь ДИНАМИЧНОСТЬ (пересборка коалиций и учёт
накопленных мостов), когда порядок грузов у динамической модели тоже случаен.

=============================================================================
ПОВТОРЫ (--repeats) -- ОДИНАКОВОЕ ЧИСЛО ПРОГОНОВ У ОБЕИХ МОДЕЛЕЙ

Случайны ОБЕ модели, поэтому в точке свипа каждая прогоняется --repeats раз с
разными сидами (одна и та же сетка сидов у обеих), и на график идёт СРЕДНЕЕ по
стоимости и МЕДИАНА по времени; полоса вокруг каждой кривой -- min..max по
прогонам, то есть диапазон, в который модель попадает "по везению". Сравнивать
среднее одной модели с единственной реализацией другой было бы некорректно.

  static_v3 случайна целиком: и паросочетание (Шаг 2), и раздача грузов
      (Шаг 3) разыгрываются заново на каждом прогоне.
  dynamic_v2 случайна в одной точке -- Шаг 4, выбор коалиции, оплачивающей
      мост при конфликте за него (rng.choice в scheduler.select_round). Сама
      по себе оплата на суммарную Phi не влияет (мост оплачивается ровно один
      раз в любом случае), но два следствия делают прогон невоспроизводимым
      при другом сиде: проигравшие пересчитывают задачу с уже бесплатным
      мостом и могут выбрать ДРУГОЙ маршрут, а позиция строителя в следующем
      раунде -- это конец ЛИЧНО построенного им моста, то есть зависит от
      исхода жребия. Разброс у dynamic_v2 заметно уже, чем у static_v3, но не
      нулевой -- на прямой эвристике он местами сопоставим с ним.

Если полоса dynamic_v2 целиком лежит ниже полосы static_v3, преимущество
динамической модели не объясняется удачным розыгрышем ни у той, ни у другой --
именно это считает последняя строка сводки.

=============================================================================
ОДИН И ТОТ ЖЕ НАБОР ВХОДНЫХ ДАННЫХ ДЛЯ ОБЕИХ МОДЕЛЕЙ (см. build_common_scenario)

Пакеты static_v3 и dynamic_v2 -- независимые копии, у каждой свои модули
graph.py / costs.py / feasibility.py (текстуально совпадающие, но это РАЗНЫЕ
классы: EdgeKind одной модели не равен EdgeKind другой). Поэтому общий сценарий
генерируется ОДИН РАЗ в нейтральном виде (CommonScenario) и побуквенно
переносится в структуры каждого пакета (to_v3_inputs / to_dynamic_inputs). Ни
одна величина при этом не пересчитывается и не разыгрывается заново: граф
островов, позиции роботов, число пар и список грузов -- одни и те же. Точка
свипа n_cargos берёт ПРЕФИКС единого списка грузов, поэтому наборы вложены друг
в друга и при росте n старые грузы не перегенерируются.

Меняется РОВНО ОДНА переменная -- число доставляемых грузов.

ЗАПУСК:
    python experiments/compare_static_v3_vs_dynamic_v2.py \\
        --n-islands 20 --n-pairs 5 -n 100 --cargo-step 5 --seed 7

    # прямая эвристика вместо обратной (по умолчанию -- inverse, p = 1 / W_T)
    python experiments/compare_static_v3_vs_dynamic_v2.py --heuristic direct

    # сколько даёт одна лишь динамичность, без вклада эвристики приоритета
    python experiments/compare_static_v3_vs_dynamic_v2.py --heuristic random

    # мультиномиальная раздача у случайной модели вместо round-robin
    python experiments/compare_static_v3_vs_dynamic_v2.py --v3-assignment uniform
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
from statistics import mean, median
from typing import List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "static_v3"))
sys.path.insert(0, os.path.join(_ROOT, "dynamic_v2"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_delivery_random.assignment import ASSIGNMENT_MODES as V3_ASSIGNMENT_MODES
from robot_delivery_random.graph import EdgeKind as V3EdgeKind, Graph as V3Graph
from robot_delivery_random.model import Cargo as V3Cargo
from robot_delivery_random.scheduler import run_random_static

from robot_delivery_v2.cargo_priority import (CARGO_HEURISTICS as DYNAMIC_HEURISTICS,
                                              get_cargo_heuristic as get_dynamic_heuristic)
from robot_delivery_v2.graph import EdgeKind as DynamicEdgeKind, Graph as DynamicGraph
from robot_delivery_v2.scheduler import Cargo as DynamicCargo, run_dynamic_rounds

# Идентичность серий кодируется НЕ только цветом: у каждой свои маркер и тип
# линии, поэтому кривые различимы в ч/б и при дальтонизме.
COLOR_V3, COLOR_DYNAMIC = "#1f5fa8", "#c0392b"
STYLE_V3 = dict(color=COLOR_V3, marker="s", linestyle="--", markersize=3.5)
STYLE_DYNAMIC = dict(color=COLOR_DYNAMIC, marker="^", linestyle="-", markersize=3.5)

LABEL_V3 = ("Случайная статическая модель (static_v3, baseline):\n"
            "пары и распределение грузов случайны и фиксированы")
LABEL_DYNAMIC = ("Динамическая модель (dynamic_v2): коалиция под груз,\n"
                 "приоритеты пересчитываются каждый раунд")

HEURISTIC_TITLE = {
    "direct": "прямая, p = W_T (дороже груз — выше приоритет)",
    "inverse": "обратная, p = 1 / W_T (дешевле груз — выше приоритет)",
    "random": "случайная (baseline: остаётся только динамичность)",
}

V3_ASSIGNMENT_TITLE = {
    "balanced": "balanced (перемешать и раздать по кругу)",
    "uniform": "uniform (каждому грузу независимо случайная пара)",
}


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
    по умолчанию повторяют scenario.generate_scenario обоих пакетов.

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


def _to_inputs(scen: CommonScenario, n_cargos: int, GraphCls, EdgeKindCls, CargoCls):
    """Перенос общего сценария в структуры конкретного пакета. Ни одна величина
    не пересчитывается -- копируются те же числа."""
    G = GraphCls()
    for i in range(scen.n_islands):
        G.add_node(i, w_V=scen.node_w[i])
    for e in scen.edges:
        if e.is_free:
            G.add_edge(e.u, e.v, EdgeKindCls.FREE, w_E=e.w_E, length=e.length)
        else:
            G.add_edge(e.u, e.v, EdgeKindCls.BLOCKED, w_E=e.w_E,
                       w_build=e.w_build, length=e.length)

    cargos = [CargoCls(cargo_id=i, start=s, finish=f)
              for i, (s, f) in enumerate(scen.cargos[:n_cargos])]

    return G, cargos, list(scen.deliverer_positions), list(scen.builder_positions)


def to_v3_inputs(scen: CommonScenario, n_cargos: int):
    return _to_inputs(scen, n_cargos, V3Graph, V3EdgeKind, V3Cargo)


def to_dynamic_inputs(scen: CommonScenario, n_cargos: int):
    return _to_inputs(scen, n_cargos, DynamicGraph, DynamicEdgeKind, DynamicCargo)


# ---------------------------------------------------------------------------
# Прогон одной точки свипа
# ---------------------------------------------------------------------------

# Шаг сетки сидов внутри точки свипа. Сиды прогонов -- seed + rep * SEED_STRIDE:
# большое простое число, чтобы прогоны точки не оказались коррелированы и чтобы
# наборы сидов соседних точек свипа не совпадали поэлементно.
SEED_STRIDE = 100_003


@dataclass
class PointResult:
    """Результат точки свипа для одной модели -- агрегат по --repeats прогонам
    с разными сидами. Случайны ОБЕ модели, поэтому структура у обеих одна и та
    же (см. шапку файла).

    cost -- СРЕДНЕЕ по прогонам, lo/hi -- min..max (полоса разброса);
    time_ms -- МЕДИАНА времени (замер времени шумит выбросами планировщика ОС,
    и одиночный выброс не должен утаскивать кривую), time_lo/time_hi --
    min..max по тем же прогонам."""
    cost: float
    lo: float
    hi: float
    time_ms: float
    time_lo: float
    time_hi: float
    ok: bool
    n_rounds: int


def _aggregate(costs: List[float], times: List[float], ok_all: bool,
               n_rounds: int) -> PointResult:
    """Свод прогонов одной точки в PointResult. Прогоны, где не все грузы
    доставлены, дают nan и в агрегат стоимости не попадают; время меряется по
    всем прогонам."""
    good = [c for c in costs if not math.isnan(c)]
    t_ms, t_lo, t_hi = median(times), min(times), max(times)
    if not good:
        nan = float("nan")
        return PointResult(cost=nan, lo=nan, hi=nan, time_ms=t_ms,
                           time_lo=t_lo, time_hi=t_hi, ok=False, n_rounds=n_rounds)
    return PointResult(cost=mean(good), lo=min(good), hi=max(good), time_ms=t_ms,
                       time_lo=t_lo, time_hi=t_hi, ok=ok_all, n_rounds=n_rounds)


def run_v3_point(scen: CommonScenario, n_cargos: int, repeats: int,
                 assignment_mode: str) -> PointResult:
    """Полный пайплайн СЛУЧАЙНОЙ статической модели (RUN-RANDOM-STATIC): Шаг 0
    -> случайное паросочетание -> случайная раздача грузов -> исполнение по
    раундам. repeats прогонов с разными сидами."""
    times: List[float] = []
    costs: List[float] = []
    ok_all = True
    n_rounds = 0
    for rep in range(repeats):
        # входные данные пересобираются заново перед каждым повтором (вне
        # замера времени), чтобы повторы были полностью независимы
        G, cargos, d_pos, b_pos = to_v3_inputs(scen, n_cargos)

        t0 = time.perf_counter()
        result = run_random_static(G, cargos, d_pos, b_pos,
                                   assignment_mode=assignment_mode,
                                   rng_seed=scen.seed + rep * SEED_STRIDE)
        times.append((time.perf_counter() - t0) * 1000.0)

        ok = result.feasible and result.all_delivered
        ok_all = ok_all and ok
        n_rounds = max(n_rounds, result.n_rounds)
        costs.append(result.real if ok else float("nan"))

    return _aggregate(costs, times, ok_all, n_rounds)


def run_dynamic_point(scen: CommonScenario, n_cargos: int, repeats: int,
                      heuristic_key: str) -> PointResult:
    """Полный пайплайн ДИНАМИЧЕСКОЙ модели: run_dynamic_rounds с накоплением
    built (Алгоритм 2 постановки dynamic_v2) и эвристикой приоритета груза (от
    W_C), пересчитываемой каждый раунд.

    repeats прогонов идут по ТОЙ ЖЕ сетке сидов, что и у static_v3. Сид здесь
    управляет Шагом 4 -- жребием, кто из коалиций раунда платит за общий мост;
    через выбор маршрута проигравшими и через позиции строителей это меняет и
    итоговую Phi (см. шапку файла)."""
    heuristic = get_dynamic_heuristic(heuristic_key)
    times: List[float] = []
    costs: List[float] = []
    ok_all = True
    n_rounds = 0
    for rep in range(repeats):
        G, cargos, d_pos, b_pos = to_dynamic_inputs(scen, n_cargos)

        t0 = time.perf_counter()
        result = run_dynamic_rounds(G, cargos, d_pos, b_pos, heuristic,
                                    rng_seed=scen.seed + rep * SEED_STRIDE)
        times.append((time.perf_counter() - t0) * 1000.0)

        ok = result.feasible and result.all_delivered
        ok_all = ok_all and ok
        n_rounds = max(n_rounds, result.n_rounds)
        costs.append((result.W_d_total + result.W_b_total) if ok else float("nan"))

    return _aggregate(costs, times, ok_all, n_rounds)


def sweep(args):
    scen = build_common_scenario(
        n_islands=args.n_islands, n_cargos_max=args.n_cargos, n_pairs=args.n_pairs,
        seed=args.seed, free_prob=args.free_prob,
        build_cost_factor=args.build_cost_factor,
    )

    cargo_range = list(range(args.n_cargos_min, args.n_cargos + 1, args.cargo_step))
    v3_points: List[PointResult] = []
    dyn_points: List[PointResult] = []

    for n in cargo_range:
        p3 = run_v3_point(scen, n, args.repeats, args.v3_assignment)
        pd = run_dynamic_point(scen, n, args.repeats, args.heuristic)

        v3_points.append(p3)
        dyn_points.append(pd)

        flag = "" if (p3.ok and pd.ok) else "  (!) не все грузы доставлены"
        if pd.cost < p3.cost:
            cheaper = "динам."
        elif p3.cost < pd.cost:
            cheaper = "случ."
        else:
            cheaper = "="
        print(f"n={n:4d}   v3: Phi={p3.cost:10.2f} [{p3.lo:9.2f}..{p3.hi:9.2f}] "
              f"t={p3.time_ms:7.2f}мс раундов={p3.n_rounds:3d}   |   "
              f"динам.: Phi={pd.cost:10.2f} [{pd.lo:9.2f}..{pd.hi:9.2f}] "
              f"t={pd.time_ms:8.2f}мс раундов={pd.n_rounds:3d}   "
              f"дешевле: {cheaper}{flag}")

    return cargo_range, v3_points, dyn_points


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------

def plot(cargo_range, v3_points, dyn_points, args, filename):
    cost_3 = [p.cost for p in v3_points]
    lo_3 = [p.lo for p in v3_points]
    hi_3 = [p.hi for p in v3_points]
    time_3 = [p.time_ms for p in v3_points]
    tlo_3 = [p.time_lo for p in v3_points]
    thi_3 = [p.time_hi for p in v3_points]

    cost_d = [p.cost for p in dyn_points]
    lo_d = [p.lo for p in dyn_points]
    hi_d = [p.hi for p in dyn_points]
    time_d = [p.time_ms for p in dyn_points]
    tlo_d = [p.time_lo for p in dyn_points]
    thi_d = [p.time_hi for p in dyn_points]

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    # при шаге 1 точек в 2 раза больше, чем при шаге 2, и маркеры дефолтного
    # размера сливаются в сплошную полосу -- уменьшаем их
    ms = 3.5 if len(cargo_range) <= 120 else 2.2
    style_3 = dict(STYLE_V3, markersize=ms)
    style_d = dict(STYLE_DYNAMIC, markersize=ms)

    def decorate(ax, ylabel, title):
        ax.set_xlabel("Число доставляемых грузов")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)

    def band(ax, y, lo, hi, style, color, alpha=0.15, label=None, band_label=None):
        """Кривая + полоса min..max по прогонам. Полоса есть у ОБЕИХ моделей:
        случайны обе (см. шапку файла)."""
        ax.plot(cargo_range, y, linewidth=2.0, label=label, **style)
        ax.fill_between(cargo_range, lo, hi, color=color, alpha=alpha,
                        linewidth=0, label=band_label)

    # Вторая строка заголовка -- как агрегированы прогоны точки. Стоимость
    # усредняется, время берётся медианой: замер времени шумит выбросами
    # планировщика ОС, и одиночный выброс не должен утаскивать кривую.
    agg_cost = f"Среднее по {args.repeats} прогонам, полоса -- min..max"
    agg_time = f"Время расчёта: медиана по {args.repeats} прогонам, полоса min..max"

    band(axes[0, 0], cost_3, lo_3, hi_3, style_3, COLOR_V3)
    decorate(axes[0, 0], "Стоимость Φ",
             f"Случайная статическая модель (static_v3), baseline\n{agg_cost}")

    band(axes[0, 1], time_3, tlo_3, thi_3, style_3, COLOR_V3)
    decorate(axes[0, 1], "Время расчёта, мс",
             f"Случайная статическая модель (static_v3)\n{agg_time}")

    band(axes[1, 0], cost_d, lo_d, hi_d, style_d, COLOR_DYNAMIC)
    decorate(axes[1, 0], "Стоимость Φ",
             f"Динамическая модель (dynamic_v2)\n{agg_cost}")

    band(axes[1, 1], time_d, tlo_d, thi_d, style_d, COLOR_DYNAMIC)
    decorate(axes[1, 1], "Время расчёта, мс",
             f"Динамическая модель (dynamic_v2)\n{agg_time}")

    # Ряды 1 и 2 -- одна и та же величина у двух моделей, поэтому шкала Y у них
    # ОБЩАЯ (объединение автоматических пределов). Иначе matplotlib растягивает
    # каждый график по своим данным, обе кривые выглядят одинаково и разница
    # между моделями на глаз не читается вообще.
    def share_ylim(ax_a, ax_b):
        lo = min(ax_a.get_ylim()[0], ax_b.get_ylim()[0])
        hi = max(ax_a.get_ylim()[1], ax_b.get_ylim()[1])
        ax_a.set_ylim(lo, hi)
        ax_b.set_ylim(lo, hi)

    share_ylim(axes[0, 0], axes[1, 0])   # стоимость Φ
    share_ylim(axes[0, 1], axes[1, 1])   # время расчёта

    BAND_V3 = "static_v3: разброс по случайности (min..max)"
    BAND_DYNAMIC = "dynamic_v2: разброс по случайности (min..max)"

    ax = axes[2, 0]
    band(ax, cost_3, lo_3, hi_3, style_3, COLOR_V3, alpha=0.12,
         label=LABEL_V3, band_label=BAND_V3)
    band(ax, cost_d, lo_d, hi_d, style_d, COLOR_DYNAMIC, alpha=0.12,
         label=LABEL_DYNAMIC, band_label=BAND_DYNAMIC)
    decorate(ax, "Стоимость Φ",
             "СРАВНЕНИЕ: фактическая стоимость выполнения всех операций\n"
             "(ниже = дешевле обошлась доставка всех грузов)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2, 1]
    band(ax, time_3, tlo_3, thi_3, style_3, COLOR_V3, alpha=0.12,
         label=LABEL_V3, band_label=BAND_V3)
    band(ax, time_d, tlo_d, thi_d, style_d, COLOR_DYNAMIC, alpha=0.12,
         label=LABEL_DYNAMIC, band_label=BAND_DYNAMIC)
    decorate(ax, "Время расчёта, мс",
             "СРАВНЕНИЕ: реальное время работы расчёта\n"
             "(static_v3 не считает приоритетов вообще)")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"static_v3 (не планирует) vs dynamic_v2 (планирует каждый раунд): "
        f"{args.n_islands} островов, {args.n_pairs} пар (Rd=Rb={args.n_pairs}), "
        f"seed={args.seed}\n"
        f"Эвристика приоритета dynamic_v2: "
        f"{HEURISTIC_TITLE.get(args.heuristic, args.heuristic)}; "
        f"у static_v3 приоритетов нет вообще\n"
        f"Шаг 3 static_v3: "
        f"{V3_ASSIGNMENT_TITLE.get(args.v3_assignment, args.v3_assignment)}\n"
        f"Обе модели на ОДНОМ наборе входных данных (граф островов, позиции "
        f"роботов и число роботов фиксированы; меняется только число грузов)\n"
        f"Случайны ОБЕ модели: каждая точка -- {args.repeats} прогонов каждой "
        f"модели по одной и той же сетке сидов, полоса вокруг кривой -- min..max",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def export_csv(path, cargo_range, v3_points, dyn_points):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_cargos",
                    "static_v3_cost_mean", "static_v3_cost_min", "static_v3_cost_max",
                    "static_v3_time_ms_median", "static_v3_time_ms_min",
                    "static_v3_time_ms_max", "static_v3_rounds",
                    "dynamic_v2_cost_mean", "dynamic_v2_cost_min", "dynamic_v2_cost_max",
                    "dynamic_v2_time_ms_median", "dynamic_v2_time_ms_min",
                    "dynamic_v2_time_ms_max", "dynamic_v2_rounds"])
        for i, n in enumerate(cargo_range):
            p3, pd = v3_points[i], dyn_points[i]
            w.writerow([n,
                        p3.cost, p3.lo, p3.hi,
                        p3.time_ms, p3.time_lo, p3.time_hi, p3.n_rounds,
                        pd.cost, pd.lo, pd.hi,
                        pd.time_ms, pd.time_lo, pd.time_hi, pd.n_rounds])
    print(f"Сохранено: {path}")


def report(cargo_range, v3_points, dyn_points):
    cost_3 = [p.cost for p in v3_points]
    cost_d = [p.cost for p in dyn_points]

    print("\n=== СВОДКА (обе модели на одном наборе входных данных) ===")
    print("  Phi -- фактическая стоимость выполнения всех операций, W_d_total + W_b_total")
    print("  у ОБЕИХ моделей -- среднее по прогонам со случайными сидами")
    for i in (0, len(cargo_range) - 1):
        n = cargo_range[i]
        d = 100 * (1 - cost_d[i] / cost_3[i]) if cost_3[i] else float("nan")
        print(f"  n={n:4d}:  случ. Phi={cost_3[i]:10.2f}   динам. Phi={cost_d[i]:10.2f}   "
              f"динамика дешевле случая на {d:+7.2f}%   |   "
              f"t_случ={v3_points[i].time_ms:7.2f}мс t_дин={dyn_points[i].time_ms:8.2f}мс")

    wins_d = sum(1 for a, b in zip(cost_3, cost_d) if b < a - 1e-9)
    wins_3 = sum(1 for a, b in zip(cost_3, cost_d) if a < b - 1e-9)
    ties = len(cargo_range) - wins_3 - wins_d
    gains = [100 * (1 - b / a) for a, b in zip(cost_3, cost_d) if a]
    print(f"\n  по всей развёртке ({len(cargo_range)} точек) стоимость Phi ниже у:")
    print(f"    динамической модели:      {wins_d}")
    print(f"    случайной static_v3:      {wins_3}")
    print(f"    совпало:                  {ties}")
    if gains:
        print(f"    в среднем динамика дешевле случайной модели на {mean(gains):+.2f}%")

    # насколько выигрыш выходит за пределы случайного разброса ОБЕИХ моделей
    beats_band = sum(1 for p3, pd in zip(v3_points, dyn_points) if pd.hi < p3.lo - 1e-9)
    print(f"    точек, где ХУДШИЙ прогон dynamic_v2 дешевле ЛУЧШЕГО прогона "
          f"static_v3: {beats_band} из {len(cargo_range)}")
    print("    (полосы разброса не пересекаются -- выигрыш нельзя объяснить "
          "удачным розыгрышем)")

    # ширина полос: показывает, насколько вообще осмысленно усреднение
    def band_width(points):
        w = [100 * (p.hi - p.lo) / p.cost for p in points if p.cost]
        return mean(w) if w else float("nan")

    def time_band_width(points):
        w = [100 * (p.time_hi - p.time_lo) / p.time_ms for p in points if p.time_ms]
        return mean(w) if w else float("nan")

    print(f"\n  средний размах (max-min) по прогонам точки:")
    print(f"    Phi:    static_v3 {band_width(v3_points):5.2f}%   "
          f"dynamic_v2 {band_width(dyn_points):5.2f}%")
    print(f"    время:  static_v3 {time_band_width(v3_points):5.2f}%   "
          f"dynamic_v2 {time_band_width(dyn_points):5.2f}%")

    t_ratio = [b / a for a, b in zip((p.time_ms for p in v3_points),
                                     (p.time_ms for p in dyn_points)) if a]
    if t_ratio:
        print(f"    время расчёта: dynamic_v2 / static_v3 в среднем {mean(t_ratio):.2f}x")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Сравнение случайной статической (static_v3/) и "
                    "динамической (dynamic_v2/) моделей по числу доставляемых "
                    "грузов на одном наборе входных данных: фактическая "
                    "стоимость выполнения всех операций и время расчёта.")
    p.add_argument("--n-islands", type=int, default=20)
    p.add_argument("--n-pairs", type=int, default=5,
                   help="число пар случайной модели (в dynamic_v2 -- столько же "
                        "доставщиков и строителей)")
    p.add_argument("-n", "--n-cargos", type=int, default=100,
                   help="верхняя граница развёртки по числу грузов")
    p.add_argument("--n-cargos-min", type=int, default=None,
                   help="нижняя граница развёртки (по умолчанию -- число пар)")
    p.add_argument("--cargo-step", type=int, default=2,
                   help="шаг развёртки по числу грузов")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--heuristic", default="inverse", choices=sorted(DYNAMIC_HEURISTICS),
                   help="эвристика приоритета груза ДИНАМИЧЕСКОЙ модели (у "
                        "static_v3 приоритетов нет вообще, её кривая от этого "
                        "параметра не зависит). random особенно нагляден: "
                        "показывает вклад одной лишь динамичности")
    p.add_argument("--v3-assignment", default="balanced", choices=list(V3_ASSIGNMENT_MODES),
                   help="режим случайной раздачи грузов в static_v3 (Шаг 3): "
                        "balanced -- перемешать и раздать по кругу (число грузов у "
                        "пар отличается максимум на 1), uniform -- каждому грузу "
                        "независимо случайная пара")
    # --v3-repeats/--time-repeats -- прежние имена того же параметра: раньше он
    # задавал повторы только static_v3 (и отдельно -- повторы замера времени).
    # Оставлены, чтобы сохранённые командные строки продолжали работать.
    p.add_argument("--repeats", "--v3-repeats", "--time-repeats", type=int,
                   default=7, dest="repeats",
                   help="прогонов КАЖДОЙ из двух моделей на точку свипа, по одной "
                        "и той же сетке сидов: на график идёт среднее (стоимость) "
                        "и медиана (время), полоса -- min..max. Случайны обе "
                        "модели, один прогон любой из них -- это одна реализация, "
                        "а не характеристика сценария")
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
    if args.repeats < 1:
        print("--repeats должен быть >= 1", file=sys.stderr)
        return 1
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Островов: {args.n_islands}, пар: {args.n_pairs}, seed: {args.seed}")
    print(f"Эвристика приоритета dynamic_v2: {args.heuristic} "
          f"(у static_v3 приоритетов нет)")
    print(f"Шаг 3 static_v3: {args.v3_assignment}")
    print(f"Прогонов на точку: {args.repeats} у КАЖДОЙ модели "
          f"(сиды {args.seed} + rep*{SEED_STRIDE})\n")

    t0 = time.time()
    cargo_range, v3_points, dyn_points = sweep(args)
    print(f"\nСвип завершён за {time.time() - t0:.1f}s")

    stem = (f"static_v3_vs_dynamic_v2_{args.heuristic}_v3-{args.v3_assignment}_"
            f"{args.n_cargos_min}to{args.n_cargos}_step{args.cargo_step}")
    plot(cargo_range, v3_points, dyn_points, args,
         os.path.join(args.output_dir, f"{stem}.png"))
    export_csv(os.path.join(args.output_dir, f"{stem}.csv"),
               cargo_range, v3_points, dyn_points)
    report(cargo_range, v3_points, dyn_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
