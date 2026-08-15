"""
ROUTE-AND-COST --- вспомогательная процедура.
ESTIMATE-TASK-COST --- эвристическая оценка стоимости выполнения операции.

Перенесено из dynamic_v2/robot_delivery_v2/costs.py БЕЗ ИЗМЕНЕНИЙ: правила
маршрутизации и стоимости операции у статической и динамической модели одни и
те же, различаются только моменты, в которые эта оценка вызывается. В
static_v2 она вызывается дважды и с разным смыслом:

  Шаг 3 (assignment.py)  -- от НАЧАЛЬНЫХ позиций пары и с built = ∅, один раз
      на всю задачу; именно здесь возникает "потенциально завышенная оценка"
      статической модели: промежуточные перемещения роботов и уже возведённые
      кем-то переправы не учитываются.
  Шаг 4 (scheduler.py)   -- от ФАКТИЧЕСКИХ позиций роботов на начало раунда и
      с накопленным глобальным built; это реальная стоимость исполнения.

Модель взаимодействия внутри коалиции: доставщик и строитель
БОЛЬШЕ НЕ ОБЯЗАНЫ физически встречаться. Вместо этого приоритет отдан
обязанности строителя проложить маршрут для доставщика -- построить все
непостроенные мосты на пути доставщика (v_d -> c_start -> c_finish) раньше,
чем доставщик до них доедет. Доставщик едет по своему маршруту независимо
и самостоятельно; ожидание возникает точечно, только у конкретного
непостроенного моста, если строитель ещё не успел его достроить.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .graph import EdgeKey, EdgeKind, Graph, NodeId

BuiltSet = Set[EdgeKey]


@dataclass
class RouteResult:
    path_nodes: List[NodeId]
    path_edges: List[EdgeKey]
    new_bridges: Set[EdgeKey]     # рёбра BLOCKED на пути, ещё не входящие в built
    travel_cost: float            # реальная стоимость проезда по пути (вершины + рёбра)
    build_cost: float             # стоимость постройки new_bridges


def route_and_cost(
    G: Graph, frm: NodeId, to: NodeId, built: BuiltSet,
) -> Optional[RouteResult]:
    """
    Algorithm ROUTE-AND-COST(G, from, to, built)

    Строит приведённый граф G', где вес ребра ŵ(e) выражает "цену включения
    его в маршрут" (длина моста -- для непостроенных блокированных рёбер;
    обычный вес проезда -- для готовых/свободных). Ищет кратчайший по этому
    весу путь (Dijkstra) и отдельно считает его РЕАЛЬНУЮ стоимость проезда
    и стоимость постройки новых мостов.

    Непостроенная переправа всегда "стоит" свою длину при выборе маршрута
    (Шаг 4 постановки: "каждый мост жадно должен быть минимальной длины"), а
    её постройка всегда попадает в build_cost. Единственный рычаг --- САМ
    АРГУМЕНТ built: на Шаге 3 он пуст (оценка "с чистого поля"), на Шаге 4 в
    нём накоплены все уже возведённые кем-либо переправы.

    Возвращает None, если пути не существует (недостижимо).
    """
    if frm == to:
        return RouteResult([frm], [], set(), 0.0, 0.0)

    # 1-13: строим приведённый граф (неявно, весами при релаксации)
    def reduced_weight(edge_key: EdgeKey) -> Optional[float]:
        e = G.edges[edge_key]
        if e.kind == EdgeKind.IMPOSSIBLE:
            return None
        if edge_key in built:
            return e.w_E
        if e.kind == EdgeKind.BLOCKED:
            return e.length
        # kind == FREE and not built (не имеет смысла для FREE, но на случай)
        return e.w_E

    # 14-15: Dijkstra на G'
    dist: Dict[NodeId, float] = {frm: 0.0}
    prev_edge: Dict[NodeId, EdgeKey] = {}
    prev_node: Dict[NodeId, NodeId] = {}
    visited: Set[NodeId] = set()
    heap: List[Tuple[float, NodeId]] = [(0.0, frm)]

    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == to:
            break
        for e in G.neighbors(node):
            w = reduced_weight(e.key())
            if w is None:
                continue
            nxt = e.other(node)
            nd = d + w
            if nxt not in dist or nd < dist[nxt]:
                dist[nxt] = nd
                prev_edge[nxt] = e.key()
                prev_node[nxt] = node
                heapq.heappush(heap, (nd, nxt))

    if to not in dist:
        return None  # ⊥ недостижимо

    # восстановление пути
    path_nodes = [to]
    path_edges: List[EdgeKey] = []
    cur = to
    while cur != frm:
        pe = prev_edge[cur]
        path_edges.append(pe)
        cur = prev_node[cur]
        path_nodes.append(cur)
    path_nodes.reverse()
    path_edges.reverse()

    # 16: новые мосты на пути
    new_bridges = {
        ek for ek in path_edges
        if G.edges[ek].kind == EdgeKind.BLOCKED and ek not in built
    }

    # 17: реальная стоимость проезда (вершины + рёбра пути), НЕ приведённая
    travel_cost = sum(G.w_V(n) for n in path_nodes) + sum(
        G.edges[ek].w_E for ek in path_edges
    )
    # 18: стоимость постройки новых мостов
    build_cost = sum(G.edges[ek].w_build for ek in new_bridges)

    return RouteResult(path_nodes, path_edges, new_bridges, travel_cost, build_cost)


@dataclass
class EstimateResult:
    W_T: float
    W_d: float
    W_b: float
    duration: float
    path_nodes: List[NodeId]
    bridges: Set[EdgeKey]          # все мосты операции: маршрута доставщика + подъезда строителя
    builder_final_pos: Optional[NodeId] = None  # конечная позиция строителя, None -- если он ничего не строил


def estimate_task_cost(
    G: Graph,
    c_start: NodeId,
    c_finish: NodeId,
    deliverer_pos: NodeId,
    builder_pos: NodeId,
    built: BuiltSet,
) -> Optional[EstimateResult]:
    """
    Algorithm ESTIMATE-TASK-COST(G, c_i, U_k, built)

    Даёт эвристическую оценку (W_T, W_d, W_b, duration) для кандидата
    "пара U_k выполняет груз c_i", исходя из текущих позиций обоих роботов
    пары и текущего состояния built. Без обязательной встречи роботов:
      1. Маршрут доставщика строится напрямую (deliverer_pos -> c_start ->
         c_finish), независимо от позиции строителя.
      2. Строитель обязан построить каждый непостроенный мост на этом
         маршруте -- идёт от своей текущей позиции builder_pos и строит
         мосты по очереди, в порядке их появления вдоль маршрута доставщика.
      3. Доставщик едет по маршруту самостоятельно; у каждого моста он
         ждёт ровно до момента, когда строитель его достроит (если тот ещё
         не успел), после чего пересекает мост уже за собственное время
         проезда.

    Возвращает None (⊥), если задача физически недостижима.
    """
    acc: BuiltSet = set(built)

    # --- маршрут доставщика: подъезд к грузу + доставка ---
    r2 = route_and_cost(G, deliverer_pos, c_start, acc)
    if r2 is None:
        return None
    acc |= r2.new_bridges

    r3 = route_and_cost(G, c_start, c_finish, acc)
    if r3 is None:
        return None

    W_d = r2.travel_cost + r3.travel_cost
    path_nodes = r2.path_nodes[:-1] + r3.path_nodes
    path_edges: List[EdgeKey] = r2.path_edges + r3.path_edges
    new_bridges: Set[EdgeKey] = set(r2.new_bridges) | set(r3.new_bridges)

    # мосты маршрута, упорядоченные по позиции вдоль пути доставщика
    bridges_sorted: List[EdgeKey] = [ek for ek in path_edges if ek in new_bridges]

    # --- строитель: идёт от своей текущей позиции и строит мосты маршрута
    #     по очереди. Между двумя последовательными мостами маршрута строитель
    #     не ограничен уже готовыми переправами -- если для того, чтобы
    #     физически добраться до очередного моста, ему самому нужно построить
    #     что-то ещё (например, он стартует отрезанным от материка), это тоже
    #     его обязанность, и эти дополнительные мосты идут в общий счёт.
    #     ready_time[ek] -- момент завершения постройки моста маршрута (без
    #     учёта времени, которое строитель тратит на его пересечение). ---
    builder_acc: BuiltSet = set(built)
    cur = builder_pos
    W_b = 0.0
    t_builder = 0.0
    ready_time: Dict[EdgeKey, float] = {}
    extra_bridges: Set[EdgeKey] = set()
    for ek in bridges_sorted:
        e = G.edges[ek]
        u, v = e.u, e.v
        # определяем, какой конец моста -- "вход" со стороны уже пройденного
        # доставщиком участка, а какой -- "выход" (куда доставщик едет дальше).
        idx = path_edges.index(ek)
        entry, exit_ = (u, v) if path_nodes[idx] == u else (v, u)
        already_built = ek in builder_acc  # мог быть построен попутно как extra-мост
        seg = route_and_cost(G, cur, entry, builder_acc)
        if seg is None:
            return None
        t_builder += seg.travel_cost + seg.build_cost
        W_b += seg.travel_cost + seg.build_cost
        extra_bridges |= seg.new_bridges
        builder_acc |= seg.new_bridges
        if already_built:
            ready_time[ek] = t_builder
        else:
            ready_time[ek] = t_builder + e.w_build
            W_b += e.w_build
        t_builder = ready_time[ek] + e.w_E + G.w_V(exit_)
        W_b += e.w_E + G.w_V(exit_)
        cur = exit_
        builder_acc.add(ek)

    builder_final_pos: Optional[NodeId] = cur if bridges_sorted else None
    bridges = new_bridges | extra_bridges

    # --- доставщик: едет по маршруту независимо, ждёт готовности моста ---
    t_deliverer = 0.0
    for idx, ek in enumerate(path_edges):
        node_to = path_nodes[idx + 1]
        e = G.edges[ek]
        if ek in new_bridges:
            t_deliverer = max(t_deliverer, ready_time[ek])
        t_deliverer += e.w_E + G.w_V(node_to)

    W_T = W_d + W_b
    duration = t_deliverer

    return EstimateResult(W_T, W_d, W_b, duration, path_nodes, bridges, builder_final_pos)
