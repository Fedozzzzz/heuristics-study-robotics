"""
СРАВНЕНИЕ ДВУХ СТАТИЧЕСКИХ МОДЕЛЕЙ -- СЛУЧАЙНОЙ (static_v3/) И ЭВРИСТИЧЕСКОЙ
(static_v2/) -- по числу доставляемых грузов.

Обе модели статические в одном и том же смысле: пары "доставщик+строитель"
формируются ОДИН РАЗ до начала работы (Шаг 2) и не пересобираются, грузы
распределяются между парами тоже ОДИН РАЗ (Шаг 3) и не перераспределяются,
дальше пары везут свои очереди по раундам (Шаг 4) и в конце суммируется общая
стоимость всех операций (Шаг 5).

Различаются они РОВНО ДВУМЯ шагами -- и в этом весь смысл сравнения:

  static_v3 (robot_delivery_random, RUN-RANDOM-STATIC) -- BASELINE.
      Шаг 2: паросочетание СЛУЧАЙНОЕ (перемешать доставщиков, перемешать
             строителей, склеить позиционно). Граф в этот шаг не передаётся.
      Шаг 3: распределение грузов СЛУЧАЙНОЕ. Никакой оценки стоимости не
             считается вообще: таблицы "пара x груз" нет.
      Режим раздачи -- --v3-assignment: balanced (перемешать и раздать по
      кругу; число грузов у пар отличается максимум на 1) или uniform
      (каждому грузу независимо случайная пара).

  static_v2 (robot_delivery_static, RUN-STATIC) -- ЭВРИСТИЧЕСКАЯ МОДЕЛЬ.
      Шаг 2: жадное паросочетание по МИНИМАЛЬНОМУ РАССТОЯНИЮ доставщик-
             строитель (Дейкстра по всем переправам, кроме IMPOSSIBLE).
      Шаг 3: строится таблица "пара x груз" -- |пар| x |грузов| вызовов
             ESTIMATE-TASK-COST от НАЧАЛЬНЫХ позиций пары при built = ∅, -- по
             ней считаются приоритеты, и грузы разбираются одним жадным
             проходом (режим --v2-assignment, по умолчанию lpt).

Всё остальное у моделей общее и текстуально совпадает: модель графа,
ROUTE-AND-COST, ESTIMATE-TASK-COST, Шаг 0, схема раундов Шага 4, правило
конфликта за общий мост (платит случайно выбранная пара, остальным бесплатно) и
глобальность built. Поэтому разница на графиках -- это цена ровно двух
осмысленных правил против случайных.

НА ГРАФИКИ ВЫНОСИТСЯ ТОЛЬКО ФАКТИЧЕСКАЯ стоимость выполнения всех операций
Phi = W_d_total + W_b_total по фактически исполненному расписанию T. Для
static_v2 это RunResult.real, а НЕ её оценка Шага 3 (estimated_static); у
static_v3 оценки "до выполнения" не существует в принципе, и по её постановке
эвристической оценкой работы модели служит именно это итоговое Phi.

=============================================================================
ПОВТОРЫ У СЛУЧАЙНОЙ МОДЕЛИ (--v3-repeats)

static_v3 случайна целиком, поэтому один её прогон -- это одна реализация
случайной величины, а не характеристика сценария: сравнивать с ней
детерминированную static_v2 по одному прогону некорректно. В точке свипа модель
прогоняется --v3-repeats раз с разными сидами, и на график идёт СРЕДНЕЕ; полоса
вокруг кривой -- min..max по этим прогонам, то есть диапазон, в который
случайная модель попадает "по везению". Если кривая static_v2 лежит ниже нижней
границы полосы, её преимущество не объясняется удачным розыгрышем.

=============================================================================
ОДИН И ТОТ ЖЕ НАБОР ВХОДНЫХ ДАННЫХ ДЛЯ ОБЕИХ МОДЕЛЕЙ (см. build_common_scenario)

Пакеты static_v2 и static_v3 -- независимые копии, у каждой свои модули
graph.py / costs.py / feasibility.py (текстуально совпадающие, но это РАЗНЫЕ
классы: EdgeKind одной модели не равен EdgeKind другой). Поэтому общий сценарий
генерируется ОДИН РАЗ в нейтральном виде (CommonScenario) и побуквенно
переносится в структуры каждого пакета (to_v2_inputs / to_v3_inputs). Ни одна
величина при этом не пересчитывается и не разыгрывается заново: граф островов,
позиции роботов, число пар и список грузов -- одни и те же. Точка свипа
n_cargos берёт ПРЕФИКС единого списка грузов, поэтому наборы вложены друг в
друга и при росте n старые грузы не перегенерируются.

Меняется РОВНО ОДНА переменная -- число доставляемых грузов.

ЗАПУСК:
    python experiments/compare_static_v3_vs_static_v2.py \\
        --n-islands 20 --n-pairs 5 -n 100 --cargo-step 2 --seed 7

    # baseline с мультиномиальной раздачей и буквальным правилом static_v2
    python experiments/compare_static_v3_vs_static_v2.py \\
        --v3-assignment uniform --v2-assignment literal
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
sys.path.insert(0, os.path.join(_ROOT, "static_v2"))
sys.path.insert(0, os.path.join(_ROOT, "static_v3"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_delivery_static.assignment import (ASSIGNMENT_MODES as V2_ASSIGNMENT_MODES,
                                              BALANCE_MODES, LPT_RULES, LPT_SIZE_RULES)
from robot_delivery_static.graph import EdgeKind as V2EdgeKind, Graph as V2Graph
from robot_delivery_static.model import Cargo as V2Cargo
from robot_delivery_static.priority import (CARGO_HEURISTICS as V2_HEURISTICS,
                                            get_cargo_heuristic as get_v2_heuristic)
from robot_delivery_static.scheduler import run_static

from robot_delivery_random.assignment import ASSIGNMENT_MODES as V3_ASSIGNMENT_MODES
from robot_delivery_random.graph import EdgeKind as V3EdgeKind, Graph as V3Graph
from robot_delivery_random.model import Cargo as V3Cargo
from robot_delivery_random.scheduler import run_random_static

# Идентичность серий кодируется НЕ только цветом: у каждой свои маркер и тип
# линии, поэтому кривые различимы в ч/б и при дальтонизме.
COLOR_V2, COLOR_V3 = "#2b6cb0", "#c0392b"
STYLE_V2 = dict(color=COLOR_V2, marker="o", linestyle="-", markersize=3.5)
STYLE_V3 = dict(color=COLOR_V3, marker="s", linestyle="--", markersize=3.5)

LABEL_V2 = ("Эвристическая модель (static_v2): пары по минимальному расстоянию,\n"
            "грузы -- по оценке стоимости доставки")
LABEL_V3 = ("Случайная модель (static_v3, baseline): пары случайны,\n"
            "грузы распределены случайно")

HEURISTIC_TITLE = {
    "direct": "прямая (дороже груз — выше приоритет)",
    "inverse": "обратная (дешевле груз — выше приоритет)",
    "random": "случайный приоритет (baseline внутри static_v2)",
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
    не пересчитывается -- копируются те же числа. Назначение "груз -> пара" на
    вход не передаётся: обе модели делают его сами на Шаге 3."""
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


def to_v2_inputs(scen: CommonScenario, n_cargos: int):
    return _to_inputs(scen, n_cargos, V2Graph, V2EdgeKind, V2Cargo)


def to_v3_inputs(scen: CommonScenario, n_cargos: int):
    return _to_inputs(scen, n_cargos, V3Graph, V3EdgeKind, V3Cargo)


# ---------------------------------------------------------------------------
# Прогон одной точки свипа
# ---------------------------------------------------------------------------

@dataclass
class PointResult:
    """Результат точки свипа для одной модели.

    cost -- то, что идёт на график (для static_v3 -- среднее по повторам);
    lo/hi -- min/max по повторам (для static_v2 совпадают с cost: модель
    детерминирована при фиксированном сценарии и сиде)."""
    cost: float
    lo: float
    hi: float
    time_ms: float
    ok: bool
    n_rounds: int


def run_v2_point(scen: CommonScenario, n_cargos: int, repeats: int, heuristic_key: str,
                 assignment_mode: str, balance: str, lpt_size: str, lpt_rule: str
                 ) -> PointResult:
    """Полный пайплайн ЭВРИСТИЧЕСКОЙ статической модели (RUN-STATIC): Шаг 0 ->
    паросочетание по минимальному расстоянию -> таблица пара x груз ->
    распределение по приоритетам -> исполнение по раундам.

    repeats здесь -- только повторы ЗАМЕРА ВРЕМЕНИ: сама модель при фиксированном
    сценарии и сиде детерминирована, и cost от повтора к повтору не меняется."""
    heuristic = get_v2_heuristic(heuristic_key)
    times: List[float] = []
    result = None
    for _ in range(repeats):
        # входные данные пересобираются заново перед каждым повтором (вне
        # замера времени), чтобы повторы были полностью независимы
        G, cargos, d_pos, b_pos = to_v2_inputs(scen, n_cargos)

        t0 = time.perf_counter()
        result = run_static(G, cargos, d_pos, b_pos, heuristic,
                            assignment_mode=assignment_mode, balance=balance,
                            lpt_size=lpt_size, lpt_rule=lpt_rule,
                            rng_seed=scen.seed)
        times.append((time.perf_counter() - t0) * 1000.0)

    ok = result.feasible and result.all_delivered
    cost = result.real if ok else float("nan")
    return PointResult(cost=cost, lo=cost, hi=cost, time_ms=median(times),
                       ok=ok, n_rounds=result.n_rounds)


def run_v3_point(scen: CommonScenario, n_cargos: int, repeats: int,
                 assignment_mode: str) -> PointResult:
    """Полный пайплайн СЛУЧАЙНОЙ статической модели (RUN-RANDOM-STATIC): Шаг 0
    -> случайное паросочетание -> случайная раздача грузов -> исполнение по
    раундам.

    repeats здесь имеет ДРУГОЙ смысл, чем у static_v2: это повторы самой модели
    с разными сидами случайности. На график идёт среднее, полоса -- min..max."""
    times: List[float] = []
    costs: List[float] = []
    ok_all = True
    n_rounds = 0
    for rep in range(repeats):
        G, cargos, d_pos, b_pos = to_v3_inputs(scen, n_cargos)

        t0 = time.perf_counter()
        result = run_random_static(G, cargos, d_pos, b_pos,
                                   assignment_mode=assignment_mode,
                                   rng_seed=scen.seed + rep * 100_003)
        times.append((time.perf_counter() - t0) * 1000.0)

        ok = result.feasible and result.all_delivered
        ok_all = ok_all and ok
        n_rounds = max(n_rounds, result.n_rounds)
        costs.append(result.real if ok else float("nan"))

    good = [c for c in costs if not math.isnan(c)]
    if not good:
        nan = float("nan")
        return PointResult(cost=nan, lo=nan, hi=nan, time_ms=median(times),
                           ok=False, n_rounds=n_rounds)
    return PointResult(cost=mean(good), lo=min(good), hi=max(good),
                       time_ms=median(times), ok=ok_all, n_rounds=n_rounds)


def sweep(args):
    scen = build_common_scenario(
        n_islands=args.n_islands, n_cargos_max=args.n_cargos, n_pairs=args.n_pairs,
        seed=args.seed, free_prob=args.free_prob,
        build_cost_factor=args.build_cost_factor,
    )

    cargo_range = list(range(args.n_cargos_min, args.n_cargos + 1, args.cargo_step))
    v2_points: List[PointResult] = []
    v3_points: List[PointResult] = []

    for n in cargo_range:
        p2 = run_v2_point(scen, n, args.time_repeats, args.heuristic,
                          args.v2_assignment, args.balance, args.lpt_size, args.lpt_rule)
        p3 = run_v3_point(scen, n, args.v3_repeats, args.v3_assignment)

        v2_points.append(p2)
        v3_points.append(p3)

        flag = "" if (p2.ok and p3.ok) else "  (!) не все грузы доставлены"
        if p2.cost < p3.cost:
            cheaper = "static_v2"
        elif p3.cost < p2.cost:
            cheaper = "static_v3"
        else:
            cheaper = "="
        print(f"n={n:4d}   v2: Phi={p2.cost:10.2f} t={p2.time_ms:7.2f}мс раундов={p2.n_rounds:3d}   |   "
              f"v3: Phi={p3.cost:10.2f} [{p3.lo:9.2f}..{p3.hi:9.2f}] t={p3.time_ms:7.2f}мс "
              f"раундов={p3.n_rounds:3d}   дешевле: {cheaper}{flag}")

    return cargo_range, v2_points, v3_points


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------

def v2_assignment_tag(args) -> str:
    """Короткая метка режима Шага 3 static_v2 для имён файлов: у lpt два разных
    правила выбора пары, и их результаты смешивать нельзя."""
    return f"lpt-{args.lpt_rule}" if args.v2_assignment == "lpt" else args.v2_assignment


def v2_assignment_title(args) -> str:
    if args.v2_assignment == "lpt":
        return f"LPT (размер задачи: {args.lpt_size}, выбор пары: {args.lpt_rule})"
    return f"{args.v2_assignment}, балансировка: {args.balance}"


def plot(cargo_range, v2_points, v3_points, args, filename):
    cost_2 = [p.cost for p in v2_points]
    cost_3 = [p.cost for p in v3_points]
    lo_3 = [p.lo for p in v3_points]
    hi_3 = [p.hi for p in v3_points]
    time_2 = [p.time_ms for p in v2_points]
    time_3 = [p.time_ms for p in v3_points]

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    # при шаге 1 точек в 2 раза больше, чем при шаге 2, и маркеры дефолтного
    # размера сливаются в сплошную полосу -- уменьшаем их
    ms = 3.5 if len(cargo_range) <= 120 else 2.2
    style_2 = dict(STYLE_V2, markersize=ms)
    style_3 = dict(STYLE_V3, markersize=ms)

    def decorate(ax, ylabel, title):
        ax.set_xlabel("Число доставляемых грузов")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)

    def single(ax, y, style, ylabel, title):
        ax.plot(cargo_range, y, linewidth=2.0, **style)
        decorate(ax, ylabel, title)

    single(axes[0, 0], cost_2, style_2, "Стоимость Φ",
           "Эвристическая модель (static_v2)\n"
           "Фактическая стоимость выполнения всех операций")
    single(axes[0, 1], time_2, style_2, "Время расчёта, мс",
           "Эвристическая модель (static_v2)\nРеальное время работы расчёта")

    ax = axes[1, 0]
    ax.plot(cargo_range, cost_3, linewidth=2.0, **style_3)
    ax.fill_between(cargo_range, lo_3, hi_3, color=COLOR_V3, alpha=0.15)
    decorate(ax, "Стоимость Φ",
             f"Случайная модель (static_v3), baseline\n"
             f"Среднее по {args.v3_repeats} прогонам, полоса -- min..max")
    single(axes[1, 1], time_3, style_3, "Время расчёта, мс",
           "Случайная модель (static_v3)\nРеальное время работы расчёта")

    ax = axes[2, 0]
    ax.plot(cargo_range, cost_2, linewidth=2.0, label=LABEL_V2, **style_2)
    ax.plot(cargo_range, cost_3, linewidth=2.0, label=LABEL_V3, **style_3)
    ax.fill_between(cargo_range, lo_3, hi_3, color=COLOR_V3, alpha=0.12,
                    label="static_v3: разброс по случайности (min..max)")
    decorate(ax, "Стоимость Φ",
             "СРАВНЕНИЕ: фактическая стоимость выполнения всех операций\n"
             "(ниже = дешевле обошлась доставка всех грузов)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2, 1]
    ax.plot(cargo_range, time_2, linewidth=2.0, label=LABEL_V2, **style_2)
    ax.plot(cargo_range, time_3, linewidth=2.0, label=LABEL_V3, **style_3)
    decorate(ax, "Время расчёта, мс",
             "СРАВНЕНИЕ: реальное время работы расчёта\n"
             "(static_v3 не строит таблицу пара x груз на Шаге 3)")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"static_v3 (случайные пары и раздача) vs static_v2 (эвристики): "
        f"{args.n_islands} островов, {args.n_pairs} пар (Rd=Rb={args.n_pairs}), "
        f"seed={args.seed}\n"
        f"Шаг 3 static_v2: {v2_assignment_title(args)}; приоритет: "
        f"{HEURISTIC_TITLE.get(args.heuristic, args.heuristic)}\n"
        f"Шаг 3 static_v3: "
        f"{V3_ASSIGNMENT_TITLE.get(args.v3_assignment, args.v3_assignment)}\n"
        f"Обе модели на ОДНОМ наборе входных данных (граф островов, позиции "
        f"роботов и число пар фиксированы; меняется только число грузов)",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def export_csv(path, cargo_range, v2_points, v3_points):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_cargos",
                    "static_v2_cost", "static_v2_time_ms", "static_v2_rounds",
                    "static_v3_cost_mean", "static_v3_cost_min", "static_v3_cost_max",
                    "static_v3_time_ms", "static_v3_rounds"])
        for i, n in enumerate(cargo_range):
            p2, p3 = v2_points[i], v3_points[i]
            w.writerow([n, p2.cost, p2.time_ms, p2.n_rounds,
                        p3.cost, p3.lo, p3.hi, p3.time_ms, p3.n_rounds])
    print(f"Сохранено: {path}")


def report(cargo_range, v2_points, v3_points):
    cost_2 = [p.cost for p in v2_points]
    cost_3 = [p.cost for p in v3_points]

    print("\n=== СВОДКА (обе модели на одном наборе входных данных) ===")
    print("  Phi -- фактическая стоимость выполнения всех операций, W_d_total + W_b_total")
    print("  для static_v3 -- среднее по повторам со случайными сидами")
    for i in (0, len(cargo_range) - 1):
        n = cargo_range[i]
        d = 100 * (1 - cost_2[i] / cost_3[i]) if cost_3[i] else float("nan")
        print(f"  n={n:4d}:  v2 Phi={cost_2[i]:10.2f}   v3 Phi={cost_3[i]:10.2f}   "
              f"эвристики дешевле случая на {d:+7.2f}%   |   "
              f"t_v2={v2_points[i].time_ms:7.2f}мс t_v3={v3_points[i].time_ms:7.2f}мс")

    wins_2 = sum(1 for a, b in zip(cost_2, cost_3) if a < b - 1e-9)
    wins_3 = sum(1 for a, b in zip(cost_2, cost_3) if b < a - 1e-9)
    ties = len(cargo_range) - wins_2 - wins_3
    gains = [100 * (1 - a / b) for a, b in zip(cost_2, cost_3) if b]
    print(f"\n  по всей развёртке ({len(cargo_range)} точек) стоимость Phi ниже у:")
    print(f"    static_v2 (эвристики):  {wins_2}")
    print(f"    static_v3 (случай):     {wins_3}")
    print(f"    совпало:                {ties}")
    if gains:
        print(f"    в среднем эвристики дешевле случайной модели на {mean(gains):+.2f}%")

    # насколько выигрыш выходит за пределы случайного разброса
    beats_band = sum(1 for p2, p3 in zip(v2_points, v3_points) if p2.cost < p3.lo - 1e-9)
    print(f"    точек, где static_v2 дешевле ЛУЧШЕГО из прогонов static_v3: "
          f"{beats_band} из {len(cargo_range)}")
    print("    (это и есть выигрыш, который нельзя объяснить удачным розыгрышем)")

    t_ratio = [a / b for a, b in zip((p.time_ms for p in v2_points),
                                     (p.time_ms for p in v3_points)) if b]
    if t_ratio:
        print(f"    время расчёта: static_v2 / static_v3 в среднем {mean(t_ratio):.2f}x")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Сравнение случайной (static_v3/) и эвристической "
                    "(static_v2/) статических моделей по числу доставляемых "
                    "грузов на одном наборе входных данных: фактическая "
                    "стоимость выполнения всех операций и время расчёта.")
    p.add_argument("--n-islands", type=int, default=20)
    p.add_argument("--n-pairs", type=int, default=5,
                   help="число пар роботов (Rd = Rb = n_pairs у обеих моделей)")
    p.add_argument("-n", "--n-cargos", type=int, default=100,
                   help="верхняя граница развёртки по числу грузов")
    p.add_argument("--n-cargos-min", type=int, default=None,
                   help="нижняя граница развёртки (по умолчанию -- число пар)")
    p.add_argument("--cargo-step", type=int, default=2,
                   help="шаг развёртки по числу грузов")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--v3-assignment", default="balanced", choices=list(V3_ASSIGNMENT_MODES),
                   help="режим случайной раздачи грузов в static_v3 (Шаг 3): "
                        "balanced -- перемешать и раздать по кругу (число грузов у "
                        "пар отличается максимум на 1), uniform -- каждому грузу "
                        "независимо случайная пара")
    p.add_argument("--v3-repeats", type=int, default=7,
                   help="повторов СЛУЧАЙНОЙ модели на точку свипа (разные сиды): на "
                        "график идёт среднее, полоса -- min..max. Один прогон "
                        "случайной модели -- это одна реализация, а не "
                        "характеристика сценария")
    p.add_argument("--heuristic", default="direct", choices=sorted(V2_HEURISTICS),
                   help="эвристика приоритета груза в static_v2 (в static_v3 "
                        "приоритетов нет вообще)")
    p.add_argument("--v2-assignment", default="lpt", choices=list(V2_ASSIGNMENT_MODES),
                   help="режим распределения грузов по парам на Шаге 3 static_v2. "
                        "По умолчанию lpt -- он балансирует загрузку пар и потому "
                        "сопоставим с режимом balanced у static_v3")
    p.add_argument("--balance", default="load", choices=list(BALANCE_MODES),
                   help="балансировка загрузки пар на Шаге 3 static_v2; к режиму "
                        "lpt не применяется (он балансирует сам)")
    p.add_argument("--lpt-size", default="min", choices=list(LPT_SIZE_RULES),
                   help="только для --v2-assignment lpt: как свернуть стоимости "
                        "груза у всех пар в один размер задачи для сортировки")
    p.add_argument("--lpt-rule", default="load", choices=list(LPT_RULES),
                   help="только для --v2-assignment lpt: правило выбора пары")
    p.add_argument("--time-repeats", type=int, default=5,
                   help="повторов замера времени static_v2 на точку (берётся медиана)")
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
    if args.v3_repeats < 1:
        print("--v3-repeats должен быть >= 1", file=sys.stderr)
        return 1
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Островов: {args.n_islands}, пар: {args.n_pairs}, seed: {args.seed}")
    print(f"Шаг 3 static_v2: {v2_assignment_title(args)}, приоритет: {args.heuristic}")
    print(f"Шаг 3 static_v3: {args.v3_assignment}, повторов на точку: {args.v3_repeats}\n")

    t0 = time.time()
    cargo_range, v2_points, v3_points = sweep(args)
    print(f"\nСвип завершён за {time.time() - t0:.1f}s")

    stem = (f"static_v3_vs_static_v2_{args.heuristic}_{v2_assignment_tag(args)}_"
            f"v3-{args.v3_assignment}_"
            f"{args.n_cargos_min}to{args.n_cargos}_step{args.cargo_step}")
    plot(cargo_range, v2_points, v3_points, args,
         os.path.join(args.output_dir, f"{stem}.png"))
    export_csv(os.path.join(args.output_dir, f"{stem}.csv"),
               cargo_range, v2_points, v3_points)
    report(cargo_range, v2_points, v3_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
