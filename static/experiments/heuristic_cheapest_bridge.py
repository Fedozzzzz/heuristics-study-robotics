"""
ЭВРИСТИКА "Самый дешёвый мост": вместо перебора альтернативных простых путей
с минимизацией суммы (W_d+W_b), маршрут строится так, чтобы СУММАРНАЯ
СТОИМОСТЬ СТРОИТЕЛЬСТВА (а не сумма строительство+проезд) была минимальна.

Реализация: модифицированный Дейкстра, где вес ребра:
  - для E_free: 0 (бесплатно использовать, не влияет на выбор)
  - для E_blocked: w_build (ТОЛЬКО стройка, без w_E)
Среди рёбер E_blocked длиннее L - исключаются (как и раньше).

Эта эвристика ПРОЩЕ и ЖАДНЕЕ: она в принципе не смотрит на стоимость проезда
при выборе моста, оптимизируя только W_b.

ТРИ ПОДЗАДАЧИ СТРОИТЕЛЯ (доставщик не умеет строить мосты, поэтому весь путь
должен быть предварительно "проложен" строителем - см. подробное объяснение
в algorithms_1_3.find_route_and_bridges, аналогичная логика здесь применена
к графу этой эвристики):
  1) builder_pos -> deliverer_pos: строитель сам добирается до доставщика.
  2) deliverer_pos -> v_start: строитель строит маршрут, по которому затем
     едет доставщик (входит в W_d).
  3) v_start -> v_finish: сама доставка, строитель продолжает из точки
     после шага 2.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))
from typing import List, Tuple
import networkx as nx

from delivery_model import IslandGraph, Cargo, Pair, TaskResult


def build_cheapest_bridge_graph(env: IslandGraph, L: float) -> nx.Graph:
    """Граф для эвристики 'самый дешёвый мост': вес ребра = w_build
    (для E_free вес = 0 - бесплатно использовать существующую переправу)."""
    Gp = nx.Graph()
    for v, data in env.G.nodes(data=True):
        Gp.add_node(v, **data)
    for u, v, data in env.G.edges(data=True):
        if data["kind"] == "impossible":
            continue
        if data["kind"] == "blocked":
            if data["length"] > L:
                continue
            weight = data["w_build"] * env.build_cost_multiplier
        else:  # free
            weight = 0.0
        Gp.add_edge(u, v, weight=weight, kind=data["kind"],
                    w_E=data["w_E"], w_build=data["w_build"], length=data["length"])
    return Gp


def _solve_segment_cheapest_bridge(env: IslandGraph, Gp: "nx.Graph", v_from: int,
                                    v_to: int, builder_pos: int,
                                    already_built: set):
    """
    Решает один сегмент v_from->v_to на графе Gp эвристики "самый дешёвый
    мост" (вес ребра = w_build, E_free = 0): находит путь минимальной
    стоимости строительства, затем считает W_b строителя, начинающего из
    builder_pos. Аналог IslandGraph.build_segment_with_builder, но на графе
    этой эвристики (минимизация чистой стройки, без альтернативных путей).

    already_built - мосты, уже построенные на более ранних шагах ЭТОЙ ЖЕ
    задачи (или ранее этой же парой) - не учитываются повторно как НОВЫЕ
    мосты и не оплачиваются за постройку повторно (см. подробное объяснение
    в IslandGraph.build_segment_with_builder).

    Возвращает None, если v_from/v_to недостижимы, либо строитель не может
    доехать до нужного моста. Иначе - dict с теми же полями, что у
    build_segment_with_builder (path, bridges, W_path, W_build,
    builder_pos_after).
    """
    if v_from == v_to:
        return {"path": [v_from], "bridges": [],
                "W_path": 0.0, "W_build": 0.0, "builder_pos_after": builder_pos}

    if v_from not in Gp or v_to not in Gp:
        return None

    # уже построенные мосты - убираем вес стройки (граф предпочтёт их как
    # обычную бесплатную-по-стройке переправу при выборе минимального пути)
    if already_built:
        for (u, v) in already_built:
            if Gp.has_edge(u, v) and Gp.edges[u, v]["kind"] == "blocked":
                Gp.edges[u, v]["weight"] = 0.0

    if not nx.has_path(Gp, v_from, v_to):
        return None

    path = nx.dijkstra_path(Gp, v_from, v_to, weight="weight")
    edges_on_path = env.edges_on_path(path)
    bridges = [(u, v) for (u, v) in edges_on_path
               if Gp.edges[u, v]["kind"] == "blocked"
               and (min(u, v), max(u, v)) not in already_built]
    w_path = env.path_cost_for_deliverer(path)

    builder_cost = 0.0
    build_cost_total = 0.0
    current = builder_pos
    bridges_sorted = []
    if bridges:
        order = {(u, v): i for i, (u, v) in enumerate(edges_on_path)}
        order.update({(v, u): i for (u, v), i in list(order.items())})
        bridges_sorted = sorted(bridges, key=lambda e: order[e])
        for (u, v) in bridges_sorted:
            try:
                sub_path = nx.dijkstra_path(Gp, current, u, weight="weight")
            except nx.NetworkXNoPath:
                return None
            builder_cost += env.path_cost_for_deliverer(sub_path)
            build_cost_total += Gp.edges[u, v]["w_build"]
            builder_cost += Gp.edges[u, v]["w_E"] + env.G.nodes[v]["w_V"]
            current = v

    builder_pos_after = bridges_sorted[-1][1] if bridges_sorted else v_to
    return {"path": path, "bridges": bridges,
            "W_path": w_path, "W_build": build_cost_total + builder_cost,
            "builder_pos_after": builder_pos_after}


def find_route_cheapest_bridge(env: IslandGraph, cargo: Cargo, pair: Pair,
                                L: float) -> TaskResult:
    """
    Находит маршрут, минимизирующий СУММАРНУЮ СТОИМОСТЬ СТРОИТЕЛЬСТВА (не
    сумму строительство+проезд). Решает ТРИ подзадачи строителя
    последовательно (доставщик не умеет строить мосты - см.
    algorithms_1_3.find_route_and_bridges для подробного объяснения):

      Шаг 1: builder_pos -> deliverer_pos (строитель сам добирается до
             доставщика; доставщик не участвует, входит только в W_b).
      Шаг 2: deliverer_pos -> v_start (строитель строит маршрут, доставщик
             едет по нему - входит в W_d).
      Шаг 3: v_start -> v_finish (доставка, строитель продолжает из точки
             после шага 2).
    """
    res = TaskResult(cargo_id=cargo.id, pair_id=pair.id, feasible=False)

    Gp = build_cheapest_bridge_graph(env, L)
    already_built = set(pair.built_bridges)

    # --- Шаг 1: строитель сам добирается до доставщика ---
    seg1 = _solve_segment_cheapest_bridge(env, Gp, pair.builder_pos,
                                           pair.deliverer_pos, pair.builder_pos,
                                           already_built)
    if seg1 is None:
        return res
    already_built |= {(min(u, v), max(u, v)) for (u, v) in seg1["bridges"]}

    # --- Шаг 2: строитель строит маршрут до v_start; доставщик едет по нему ---
    seg2 = _solve_segment_cheapest_bridge(env, Gp, pair.deliverer_pos,
                                           cargo.v_start, seg1["builder_pos_after"],
                                           already_built)
    if seg2 is None:
        return res
    already_built |= {(min(u, v), max(u, v)) for (u, v) in seg2["bridges"]}

    usable_bridges_for_approach = set(already_built)
    w_d_approach, approach_path = env.deliverer_cost_with_approach(
        pair.deliverer_pos, [cargo.v_start], usable_bridges_for_approach)
    if w_d_approach is None:
        return res

    # --- Шаг 3: строитель продолжает работу из точки после шага 2 ---
    seg3 = _solve_segment_cheapest_bridge(env, Gp, cargo.v_start,
                                           cargo.v_finish, seg2["builder_pos_after"],
                                           already_built)
    if seg3 is None:
        return res

    # см. algorithms_1_3.find_route_and_bridges - вычитаем w_V(v_start),
    # чтобы не посчитать его дважды (один раз в w_d_approach, другой в seg3)
    w_v_start = env.G.nodes[cargo.v_start]["w_V"]
    w_d_total = w_d_approach + seg3["W_path"] - w_v_start

    duration = seg1["W_build"] + seg2["W_build"] + max(w_d_total, seg3["W_build"])

    res.feasible = True
    res.path = seg3["path"]
    res.approach_path = approach_path
    res.bridges = seg1["bridges"] + seg2["bridges"] + seg3["bridges"]
    res.W_d = w_d_total
    res.W_b = seg1["W_build"] + seg2["W_build"] + seg3["W_build"]
    res.duration = duration
    return res


import copy
from dataclasses import dataclass, field
from typing import Dict, Optional
from algorithms_1_3 import compute_priority
from algorithm_2 import ScheduleEntry, ScheduleOutcome


def run_scheduling_cheapest_bridge(env: IslandGraph, cargos: List[Cargo],
                                    pairs: List[Pair], L: float) -> ScheduleOutcome:
    """
    Версия Алгоритма 2, использующая эвристику 'самый дешёвый мост'
    (find_route_cheapest_bridge) вместо стандартного find_route_and_bridges.
    Логика очередей и приоритета внутри пары - та же, что в основном алгоритме.
    """
    cargos = copy.deepcopy(cargos)
    pairs = copy.deepcopy(pairs)
    pairs_by_id = {p.id: p for p in pairs}

    outcome = ScheduleOutcome(L=L, all_delivered=False)

    pending_by_pair: Dict[str, List[Cargo]] = {p.id: [] for p in pairs}
    for c in cargos:
        pending_by_pair[c.assigned_pair].append(c)

    for pair_id, pair in pairs_by_id.items():
        pending = pending_by_pair[pair_id]
        current_time = 0.0

        while pending:
            scored = [
                (compute_priority(c, env, [pair.deliverer_pos], [pair.builder_pos]), c)
                for c in pending
            ]
            scored.sort(key=lambda t: t[0], reverse=True)
            cargo = scored[0][1]

            result = find_route_cheapest_bridge(env, cargo, pair, L)

            if not result.feasible:
                outcome.all_delivered = False
                return outcome

            pair.deliverer_pos = cargo.v_finish
            if result.bridges:
                pair.builder_pos = result.bridges[-1][1]
                for (u, v) in result.bridges:
                    pair.add_built_bridge(u, v)
            else:
                pair.builder_pos = cargo.v_finish

            cargo.delivered = True
            pending = [c for c in pending if c.id != cargo.id]

            end_time = current_time + result.duration
            outcome.schedule.append(ScheduleEntry(
                cargo_id=cargo.id, pair_id=pair_id, result=result,
                start_time=current_time, end_time=end_time,
            ))
            outcome.W_d_total += result.W_d
            outcome.W_b_total += result.W_b
            current_time = end_time

    outcome.all_delivered = all(c.delivered for c in cargos)
    return outcome


def build_cost_curves_cheapest_bridge(env: IslandGraph, cargos: List[Cargo],
                                       pairs: List[Pair], L_grid: List[float]):
    """Аналог build_cost_curves (Алгоритм 4, шаг 1) для эвристики 'дешёвый мост'."""
    from algorithm_4 import ParetoPoint
    points = []
    for L in L_grid:
        outcome = run_scheduling_cheapest_bridge(env, cargos, pairs, L=L)
        if outcome.all_delivered:
            points.append(ParetoPoint(L=L, W_d_total=outcome.W_d_total,
                                       W_b_total=outcome.W_b_total, outcome=outcome))
    return points
