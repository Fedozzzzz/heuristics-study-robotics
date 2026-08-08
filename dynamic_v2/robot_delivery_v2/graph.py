"""
Модель среды --- граф G = (V, E) из раздела "1. Обозначения и базовые величины".

V --- острова, каждый с весом прохода w_V.
Рёбра делятся на типы:
    FREE        --- существующие переправы, вес проезда w_E.
    BLOCKED     --- требуют постройки моста: w_build (постройка) + w_E (проезд после).
                     У каждого BLOCKED-ребра также есть length --- физическая длина
                     будущего моста (используется как вес при выборе маршрута).
    IMPOSSIBLE  --- нереализуемо никогда (не участвует в маршрутизации).

Граф неориентированный: доставщики/строители могут ехать в обе стороны.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

NodeId = int
EdgeKey = FrozenSet[NodeId]  # frozenset({u, v}) -- ключ неориентированного ребра


class EdgeKind(Enum):
    FREE = "free"
    BLOCKED = "blocked"
    IMPOSSIBLE = "impossible"


@dataclass
class Edge:
    u: NodeId
    v: NodeId
    kind: EdgeKind
    w_E: float = 0.0        # стоимость проезда (после постройки/для готовых)
    w_build: float = 0.0    # стоимость постройки моста (только для BLOCKED)
    length: float = 0.0     # физическая длина моста (только для BLOCKED)

    def key(self) -> EdgeKey:
        return frozenset((self.u, self.v))

    def other(self, node: NodeId) -> NodeId:
        return self.v if node == self.u else self.u


@dataclass
class Graph:
    """Граф G = (V, E)."""

    node_weight: Dict[NodeId, float] = field(default_factory=dict)
    edges: Dict[EdgeKey, Edge] = field(default_factory=dict)
    _adj: Dict[NodeId, List[EdgeKey]] = field(default_factory=dict)

    # -- построение --------------------------------------------------

    def add_node(self, node: NodeId, w_V: float = 0.0) -> None:
        self.node_weight[node] = w_V
        self._adj.setdefault(node, [])

    def add_edge(
        self,
        u: NodeId,
        v: NodeId,
        kind: EdgeKind,
        w_E: float = 0.0,
        w_build: float = 0.0,
        length: float = 0.0,
    ) -> None:
        if u not in self.node_weight:
            self.add_node(u)
        if v not in self.node_weight:
            self.add_node(v)
        e = Edge(u=u, v=v, kind=kind, w_E=w_E, w_build=w_build, length=length)
        key = e.key()
        self.edges[key] = e
        self._adj[u].append(key)
        self._adj[v].append(key)

    # -- доступ --------------------------------------------------------

    def neighbors(self, node: NodeId) -> Iterable[Edge]:
        for key in self._adj.get(node, []):
            yield self.edges[key]

    def edge_between(self, u: NodeId, v: NodeId) -> Optional[Edge]:
        return self.edges.get(frozenset((u, v)))

    def w_V(self, node: NodeId) -> float:
        return self.node_weight.get(node, 0.0)

    def nodes(self) -> List[NodeId]:
        return list(self.node_weight.keys())

    def copy(self) -> "Graph":
        g = Graph()
        g.node_weight = dict(self.node_weight)
        g.edges = dict(self.edges)
        g._adj = {k: list(v) for k, v in self._adj.items()}
        return g

    # -- связность / расстояния ----------------------------------------

    def connected_components(self, kinds: Set[EdgeKind]) -> List[Set[NodeId]]:
        """Компоненты связности подграфа, состоящего только из рёбер с типом
        из `kinds` (например {FREE} для G_ready, или {FREE, BLOCKED} для
        G' = (V, E_ready ∪ E_blocked) из Шага 0). Изолированные вершины --
        отдельные компоненты из одного узла."""
        visited: Set[NodeId] = set()
        components: List[Set[NodeId]] = []
        for start in self.node_weight:
            if start in visited:
                continue
            comp: Set[NodeId] = {start}
            visited.add(start)
            stack = [start]
            while stack:
                node = stack.pop()
                for e in self.neighbors(node):
                    if e.kind not in kinds:
                        continue
                    nxt = e.other(node)
                    if nxt not in visited:
                        visited.add(nxt)
                        comp.add(nxt)
                        stack.append(nxt)
            components.append(comp)
        return components

    def shortest_distance(
        self,
        frm: NodeId,
        to: NodeId,
        passable: Callable[["Edge"], bool],
        include_vertex_weight: bool = True,
    ) -> Optional[float]:
        """Обобщённый Dijkstra по рёбрам, для которых `passable(edge)` истинно,
        с весами w_E (рёбра) [+ w_V (вершины), если include_vertex_weight].
        Возвращает None, если `to` недостижим из `frm` при данном предикате
        проходимости."""
        if frm == to:
            return 0.0
        dist: Dict[NodeId, float] = {frm: 0.0}
        visited: Set[NodeId] = set()
        heap: List[Tuple[float, NodeId]] = [(0.0, frm)]
        while heap:
            d, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            if node == to:
                return d
            for e in self.neighbors(node):
                if not passable(e):
                    continue
                nxt = e.other(node)
                nd = d + e.w_E + (self.w_V(nxt) if include_vertex_weight else 0.0)
                if nxt not in dist or nd < dist[nxt]:
                    dist[nxt] = nd
                    heapq.heappush(heap, (nd, nxt))
        return None
