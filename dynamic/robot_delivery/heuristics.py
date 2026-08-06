"""
Варианты формирования приоритета доставки p из "Постановка задачи".

Каждая эвристика получает HeuristicContext (оценка ESTIMATE-TASK-COST для
конкретной пары кандидата "пара-груз" + вспомогательный контекст) и должна
вернуть скаляр p. Чем больше p -- тем выше приоритет (запись раньше в
отсортированном списке SELECT-ROUND, "прямая" логика ранжирования по
убыванию, как зафиксировано в "Фиксация задачи": п.2).

Некоторые эвристики нормируются НЕ по отдельной ячейке, а по всей текущей
таблице приоритетов (одна оценка на раунд) -- для них задан aggregate(),
применяемый ПОСЛЕ того как per_cell посчитан для всех кандидатов раунда.
По умолчанию aggregate -- тождественная функция (без изменений).
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .costs import EstimateResult
from .graph import EdgeKind, Graph, NodeId

EPS = 1e-9


@dataclass
class HeuristicContext:
    estimate: EstimateResult          # результат ESTIMATE-TASK-COST для этой ячейки
    G: Graph
    built: set
    deliverer_pos: NodeId
    builder_pos: NodeId
    c_start: NodeId
    c_finish: NodeId
    coalition_id: int
    cargo_id: int
    pair_load: float = 0.0            # накопленная нагрузка пары (для LPT/convex, п. "Формирование расписания")
    theta: float = 0.5                # параметр выпуклой комбинации Greedy-Nearest / LPT


def _geometric_distance(G: Graph, built: set, frm: NodeId, to: NodeId) -> Optional[float]:
    """Кратчайшее расстояние только по уже проходимым рёбрам (FREE ∪ built),
    БЕЗ учёта необходимости строительства -- "дешёвый" геометрический признак
    из раздела "2. Эвристический приоритет на основе расстояния"."""
    if frm == to:
        return 0.0
    dist = {frm: 0.0}
    visited = set()
    heap = [(0.0, frm)]
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == to:
            return d
        for e in G.neighbors(node):
            passable = e.kind == EdgeKind.FREE or e.key() in built
            if not passable:
                continue
            nxt = e.other(node)
            nd = d + e.w_E
            if nxt not in dist or nd < dist[nxt]:
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return None  # геометрически недостижимо без строительства


@dataclass
class Heuristic:
    key: str
    label: str
    description: str
    per_cell: Callable[[HeuristicContext], float]
    aggregate: Callable[[Sequence[float]], Sequence[float]] = field(
        default=lambda scores: scores
    )

    def score_table(self, contexts: Sequence[HeuristicContext]) -> List[float]:
        raw = [self.per_cell(ctx) for ctx in contexts]
        return list(self.aggregate(raw))


# ---------------------------------------------------------------------------
# 1. Приоритет через стоимость выполнения (раздел "1.")
# ---------------------------------------------------------------------------

def _cost_direct(ctx: HeuristicContext) -> float:
    # 1.2 Прямая зависимость: p = W_T -- эталонный вариант, зафиксированный
    # в "Фиксация задачи" (п.2): дороже операция -- выше приоритет.
    return ctx.estimate.W_T


def _cost_inverse(ctx: HeuristicContext) -> float:
    # 1.1 Обратная зависимость: p = 1 / W_T -- дешёвые задачи выполняются первыми.
    return 1.0 / (ctx.estimate.W_T + EPS)


def _normalize_share(scores: Sequence[float]) -> Sequence[float]:
    # 1.3 Нормированная стоимость: p_i = W_T_i / Σ W_T_j -- доля общего ресурса,
    # на которую претендует задача (в пределах текущей таблицы раунда).
    total = sum(scores)
    if total <= EPS:
        return [0.0 for _ in scores]
    return [s / total for s in scores]


def _cost_by_agent_type(ctx: HeuristicContext) -> float:
    # 1.4 Приоритет через стоимость по типу агента: p = W_b / W_d.
    # Высокое значение -- задача дорогая для строителей, но дешёвая для
    # доставщиков; низкое -- задачу выгоднее отложить/искать альтернативу.
    return ctx.estimate.W_b / (ctx.estimate.W_d + EPS)


# ---------------------------------------------------------------------------
# 2. Эвристический приоритет на основе расстояния (раздел "2.")
# ---------------------------------------------------------------------------

def _dist_min_deliverer(ctx: HeuristicContext) -> float:
    # 2.1 p = 1 / dist(ближайший доставщик, точка погрузки).
    d = _geometric_distance(ctx.G, ctx.built, ctx.deliverer_pos, ctx.c_start)
    if d is None:
        d = ctx.estimate.W_d  # geometрически недостижимо без построек -- fallback на реальный W_d
    return 1.0 / (d + EPS)


def _dist_max_deliverer(ctx: HeuristicContext) -> float:
    # 2.2 p = dist(ближайший доставщик, точка погрузки) -- "сначала сложное":
    # далёкие грузы запускаются заранее, чтобы не остаться недоставленными.
    d = _geometric_distance(ctx.G, ctx.built, ctx.deliverer_pos, ctx.c_start)
    if d is None:
        d = ctx.estimate.W_d
    return d


def _dist_min_builder(ctx: HeuristicContext) -> float:
    # 2.3 p = 1 / dist(строитель, точка погрузки) -- геометрическая
    # доступность строителя (прокси для "ближайшей вершины моста").
    d = _geometric_distance(ctx.G, ctx.built, ctx.builder_pos, ctx.c_start)
    if d is None:
        d = ctx.estimate.W_b
    return 1.0 / (d + EPS)


def _dist_combined(ctx: HeuristicContext) -> float:
    # 2.4 Комбинированный: p = 1 / (dist_доставщика + dist_строителя).
    dd = _geometric_distance(ctx.G, ctx.built, ctx.deliverer_pos, ctx.c_start)
    db = _geometric_distance(ctx.G, ctx.built, ctx.builder_pos, ctx.c_start)
    if dd is None:
        dd = ctx.estimate.W_d
    if db is None:
        db = ctx.estimate.W_b
    return 1.0 / (dd + db + EPS)


# ---------------------------------------------------------------------------
# Baseline: случайное распределение груза по парам (не зависит от стоимости)
# -- нужен, чтобы оценить, насколько прямая эвристика (p = W_T) вообще лучше
# бессмысленного порядка. См. эксперимент compare_direct_vs_random.py.
# ---------------------------------------------------------------------------

def _random_priority(ctx: HeuristicContext) -> float:
    # p никак не зависит от estimate/стоимости -- чистый шум. Сид берётся из
    # состава ячейки (коалиция, груз, позиции), а не из глобального random,
    # чтобы прогон был воспроизводим при повторном вызове с тем же сценарием.
    seed = (ctx.coalition_id, ctx.cargo_id, ctx.c_start, ctx.c_finish,
             ctx.deliverer_pos, ctx.builder_pos)
    return random.Random(hash(seed)).random()


# ---------------------------------------------------------------------------
# "Формирование расписания": выпуклая комбинация Greedy-Nearest / LPT
# (раздел "Дополнительные корректировки эвристики для p")
# ---------------------------------------------------------------------------

def _make_convex_cost_load(theta: float) -> Heuristic:
    def per_cell(ctx: HeuristicContext) -> float:
        # хранит "сырые" компоненты в самом значении через кортеж -- но
        # интерфейс требует float, поэтому нормировку делаем в aggregate()
        # по накопленным вместе (cost, load) парам через замыкание-таблицу.
        return ctx.estimate.W_T  # placeholder, реальная свёртка -- в aggregate

    # aggregate получает только сырые cost-значения, поэтому загрузку пары
    # инкапсулируем через отдельный список, собираемый вместе с contexts.
    def score_table(contexts: Sequence[HeuristicContext]) -> List[float]:
        costs = [c.estimate.W_T for c in contexts]
        loads = [c.pair_load for c in contexts]
        c_min, c_max = (min(costs), max(costs)) if costs else (0.0, 1.0)
        l_min, l_max = (min(loads), max(loads)) if loads else (0.0, 1.0)

        def norm(x, lo, hi):
            return (x - lo) / (hi - lo) if hi - lo > EPS else 0.0

        out = []
        for c, l in zip(costs, loads):
            # Greedy Nearest минимизирует cost -> берём (1 - norm(cost)) как "близость";
            # LPT минимизирует текущую загрузку пары -> (1 - norm(load)).
            # p = theta * близость_по_стоимости + (1-theta) * незагруженность пары
            score = theta * (1.0 - norm(c, c_min, c_max)) + (1.0 - theta) * (
                1.0 - norm(l, l_min, l_max)
            )
            out.append(score)
        return out

    h = Heuristic(
        key=f"convex_cost_load_theta{theta:g}",
        label=f"Convex Greedy-Nearest/LPT (theta={theta:g})",
        description=(
            "Выпуклая комбинация стоимости назначения и текущей загрузки пары: "
            "theta=1 эквивалентен Greedy Nearest, theta=0 -- жадному аналогу LPT."
        ),
        per_cell=per_cell,
    )
    # переопределяем score_table напрямую, т.к. этой эвристике нужен полный
    # контекст (cost И load) одновременно, а не только предвычисленные per_cell
    h.score_table = score_table  # type: ignore[assignment]
    return h


# ---------------------------------------------------------------------------
# Реестр эвристик
# ---------------------------------------------------------------------------

HEURISTICS: Dict[str, Heuristic] = {
    "cost_direct": Heuristic(
        key="cost_direct",
        label="Прямая стоимость (p = W_T)",
        description="1.2: дороже операция -- выше приоритет. Базовый вариант из фиксации задачи.",
        per_cell=_cost_direct,
    ),
    "cost_inverse": Heuristic(
        key="cost_inverse",
        label="Обратная стоимость (p = 1/W_T)",
        description="1.1: дешевле операция -- выше приоритет.",
        per_cell=_cost_inverse,
    ),
    "cost_normalized": Heuristic(
        key="cost_normalized",
        label="Нормированная стоимость (p = W_T / ΣW_T)",
        description="1.3: приоритет как доля общего ресурса, на который претендует задача.",
        per_cell=_cost_direct,
        aggregate=_normalize_share,
    ),
    "cost_by_agent_type": Heuristic(
        key="cost_by_agent_type",
        label="По типу агента (p = W_b / W_d)",
        description="1.4: высокий приоритет -- дорого для строителей, дёшево для доставщиков.",
        per_cell=_cost_by_agent_type,
    ),
    "dist_min_deliverer": Heuristic(
        key="dist_min_deliverer",
        label="Мин. расстояние до доставщика",
        description="2.1: чем ближе груз к доставщику пары, тем выше приоритет.",
        per_cell=_dist_min_deliverer,
    ),
    "dist_max_deliverer": Heuristic(
        key="dist_max_deliverer",
        label="Макс. расстояние до доставщика ('сначала сложное')",
        description="2.2: далёкие грузы запускаются раньше, чтобы не остаться невыполненными.",
        per_cell=_dist_max_deliverer,
    ),
    "dist_min_builder": Heuristic(
        key="dist_min_builder",
        label="Мин. расстояние до строителя",
        description="2.3: геометрическая доступность строителя пары.",
        per_cell=_dist_min_builder,
    ),
    "dist_combined": Heuristic(
        key="dist_combined",
        label="Комбинированное расстояние (доставщик+строитель)",
        description="2.4: p = 1 / (dist_доставщика + dist_строителя).",
        per_cell=_dist_combined,
    ),
    "convex_cost_load_0.5": _make_convex_cost_load(0.5),
    "convex_cost_load_0.2": _make_convex_cost_load(0.2),
    "convex_cost_load_0.8": _make_convex_cost_load(0.8),
    "random_priority": Heuristic(
        key="random_priority",
        label="Случайный приоритет (baseline)",
        description=(
            "Baseline для оценки качества эвристик: p ~ U(0,1), не зависит от "
            "стоимости задачи -- эквивалент случайного распределения груза по "
            "парам. Сравнивается с cost_direct в compare_direct_vs_random.py."
        ),
        per_cell=_random_priority,
    ),
}


def get_heuristic(name: str) -> Heuristic:
    if name not in HEURISTICS:
        raise KeyError(
            f"Неизвестная эвристика '{name}'. Доступные: {', '.join(sorted(HEURISTICS))}"
        )
    return HEURISTICS[name]
