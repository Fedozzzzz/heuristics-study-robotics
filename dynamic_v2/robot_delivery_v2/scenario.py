"""
Генератор случайных сценариев ("остров-граф" + грузы + пары роботов) для
прогона экспериментов. Не является частью формальной постановки -- это
вспомогательный инструмент, чтобы можно было сравнивать эвристики на
множестве случайных инстансов.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from .graph import EdgeKind, Graph
from .scheduler import Cargo


@dataclass
class Scenario:
    G: Graph
    cargos: List[Cargo]
    deliverer_positions: List[int]
    builder_positions: List[int]
    n_islands: int
    n_cargos: int
    n_pairs: int
    seed: int


def _mst_edges(points: List[Tuple[float, float]], rng: random.Random) -> List[Tuple[int, int, float]]:
    """Прим: минимальное остовное дерево на полном графе точек -- гарантирует
    связность базового графа островов."""
    n = len(points)
    in_tree = [False] * n
    in_tree[0] = True
    dist = [
        math.dist(points[0], points[j]) if j != 0 else 0.0 for j in range(n)
    ]
    parent = [0] * n
    edges: List[Tuple[int, int, float]] = []
    for _ in range(n - 1):
        best, best_d = -1, math.inf
        for j in range(n):
            if not in_tree[j] and dist[j] < best_d:
                best, best_d = j, dist[j]
        in_tree[best] = True
        edges.append((parent[best], best, best_d))
        for j in range(n):
            if not in_tree[j]:
                d = math.dist(points[best], points[j])
                if d < dist[j]:
                    dist[j] = d
                    parent[j] = best
    return edges


def generate_scenario(
    n_islands: int = 18,
    n_cargos: int = 12,
    n_pairs: int = 3,
    seed: int = 0,
    free_prob: float = 0.45,
    extra_edge_factor: float = 1.3,
    travel_cost_factor: float = 1.0,
    build_cost_factor: float = 3.0,
    node_weight_max: float = 0.5,
    area: float = 100.0,
) -> Scenario:
    """Строит связный случайный граф-остров и на нём -- набор грузов и пар
    роботов-доставщиков/строителей.

    free_prob      -- доля рёбер, которые изначально являются готовыми переправами (FREE);
                       остальные -- BLOCKED (требуют постройки моста).
    extra_edge_factor -- во сколько раз больше рёбер добавляется поверх MST
                          (создаёт альтернативные маршруты/циклы).
    """
    rng = random.Random(seed)

    points = [(rng.uniform(0, area), rng.uniform(0, area)) for _ in range(n_islands)]
    G = Graph()
    for i in range(n_islands):
        G.add_node(i, w_V=rng.uniform(0, node_weight_max))

    candidate_edges = set()
    for (u, v, _d) in _mst_edges(points, rng):
        candidate_edges.add((min(u, v), max(u, v)))

    n_extra = int(n_islands * extra_edge_factor)
    attempts = 0
    while len(candidate_edges) < len(_mst_edges(points, rng)) + n_extra and attempts < n_extra * 20:
        attempts += 1
        u = rng.randrange(n_islands)
        v = rng.randrange(n_islands)
        if u == v:
            continue
        candidate_edges.add((min(u, v), max(u, v)))

    edge_specs = []
    for (u, v) in candidate_edges:
        d = math.dist(points[u], points[v])
        is_free = rng.random() < free_prob
        edge_specs.append((u, v, d, is_free))

    for (u, v, d, is_free) in edge_specs:
        if is_free:
            G.add_edge(u, v, EdgeKind.FREE, w_E=d * travel_cost_factor)
        else:
            G.add_edge(
                u, v, EdgeKind.BLOCKED,
                w_E=d * travel_cost_factor,
                w_build=d * build_cost_factor,
                length=d,
            )

    cargos = []
    for i in range(n_cargos):
        start = rng.randrange(n_islands)
        finish = rng.randrange(n_islands)
        while finish == start:
            finish = rng.randrange(n_islands)
        cargos.append(Cargo(cargo_id=i, start=start, finish=finish))

    # Начальные позиции доставщиков и строителей -- независимые случайные
    # точки (не склеены в пары заранее: коалиции формируются заново каждый
    # раунд под выбранный по приоритету груз, Шаг 2 алгоритма dynamic_v2).
    deliverer_positions = [rng.randrange(n_islands) for _ in range(n_pairs)]
    builder_positions = [rng.randrange(n_islands) for _ in range(n_pairs)]

    return Scenario(
        G=G, cargos=cargos,
        deliverer_positions=deliverer_positions, builder_positions=builder_positions,
        n_islands=n_islands, n_cargos=n_cargos, n_pairs=n_pairs, seed=seed,
    )
