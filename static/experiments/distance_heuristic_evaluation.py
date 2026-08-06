"""
ОЦЕНКА ЭВРИСТИКИ ПРИОРИТЕТА ПО РАССТОЯНИЮ (формула 2.4 - комбинированная):

    p(c_i) = 1 / max( min_{d_k in R_d} dist(v_dk, v_s^i),
                       min_{b_l in R_b} dist(v_bl, v_bridge^i) )

В отличие от эвристик W_T^i (priority_evaluation.py), здесь приоритет строится
БЕЗ решения подзадачи поиска маршрута - используется только ПРЯМОЕ (евклидово)
расстояние от текущей позиции пары до точки погрузки груза. Это сильно дешевле
вычислительно, но потенциально менее точно, поскольку "прямая линия" может
сильно отличаться от реального пути по графу (через мосты, в обход
недостроенных переправ).

ЧТО СРАВНИВАЕМ (по договорённости): расстояние ТОЛЬКО до точки погрузки v_s^i
(как в самой формуле p(c_i)), а не весь путь доставки целиком:
  - ЭВРИСТИЧЕСКАЯ оценка (estimated) = straight_distance(текущая позиция
    доставщика, v_s^i) - прямая линия "по воздуху"
  - РЕАЛЬНОЕ расстояние (real) = длина кратчайшего пути ПО ГРАФУ от текущей
    позиции доставщика до v_s^i (через существующие/построенные переправы,
    с учётом ограничения L)

Пайплайн идентичен priority_evaluation.py по структуре:
  1) Для каждого груза считается p(c_i) по формуле 2.4 (от начальной позиции
     назначенной пары - bottleneck доставщик/строитель).
  2) Строится граф приоритетов (k уровней слева-направо).
  3) Грузы выполняются строго по убыванию приоритета, одна пара за другой.
  4) Для КАЖДОГО груза в реальном порядке выполнения считается:
       - эвристическая оценка расстояния ДО НЕЁ (straight_distance от текущей,
         уже сдвинувшейся, позиции доставщика)
       - реальное расстояние по графу (тоже от текущей позиции)
     ВАЖНО: и оценка, и реальность считаются от ТЕКУЩЕЙ позиции в момент
     выполнения (а не от начальной) - это отличие от priority_evaluation.py,
     где W_T^i считался единожды от начальной позиции. Здесь интереснее
     другое сравнение: "прямая линия" против "путь по графу" в КАЖДЫЙ
     конкретный момент, а не эффект смещения пары между задачами.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx

from delivery_model import IslandGraph, Cargo, Pair, TaskResult
from heuristic_cheapest_bridge import find_route_cheapest_bridge


# ---------------------------------------------------------------------------
# Расстояние ПО ГРАФУ до точки погрузки (реальное)
# ---------------------------------------------------------------------------

def real_distance_to_pickup(env: IslandGraph, from_node: int, v_s: int,
                             L: float) -> Optional[float]:
    """
    Длина кратчайшего пути ПО ГРАФУ от from_node до v_s (точка погрузки),
    с учётом ограничения L (рёбра E_blocked длиннее L недоступны).
    Используется тот же приведённый граф, что и в Алгоритме 3 (вес ребра
    blocked = w_build+w_E, free = w_E), поскольку доставщик физически едет
    по этому же графу - его маршрут зависит от того, какие мосты доступны.
    Возвращает None, если путь недостижим.
    """
    Gp = env.build_weighted_graph(L)
    if from_node not in Gp or v_s not in Gp:
        return None
    if not nx.has_path(Gp, from_node, v_s):
        return None
    path = nx.dijkstra_path(Gp, from_node, v_s, weight="weight")
    # длина пути - это сумма ФИЗИЧЕСКИХ длин рёбер (а не приведённого веса),
    # т.к. нас интересует "сколько проедет" доставщик, не стоимость
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += Gp.edges[u, v]["length"]
    return total


# ---------------------------------------------------------------------------
# Приоритет по формуле 2.4 (bottleneck доставщик/строитель, по прямой)
# ---------------------------------------------------------------------------

def compute_distance_priority(cargo: Cargo, env: IslandGraph, pair: Pair) -> float:
    """
    p(c_i) = 1 / max( dist(v_d, v_s^i), dist(v_b, v_s^i) )

    Упрощение относительно общей формулы 2.4: поскольку коалиция жёстко
    закреплена 1 доставщик : 1 строитель (cargo.assigned_pair), min по
    множеству R_d/R_b вырождается в единственное расстояние от позиции
    именно ЭТОЙ пары (как и в priority_evaluation.compute_priority_for_assigned_pair,
    но здесь bridge-точка не используется отдельно - для простоты и в
    соответствии с упрощённой моделью используется v_s^i для обоих агентов,
    т.к. требуемый мост в общем случае неизвестен до решения подзадачи маршрута).
    """
    dist_d = env.straight_distance(pair.deliverer_pos, cargo.v_start)
    dist_b = env.straight_distance(pair.builder_pos, cargo.v_start)
    bottleneck = max(dist_d, dist_b)
    if bottleneck == 0:
        return float("inf")
    return 1.0 / bottleneck


def assign_priority_levels_from_priority(cargos: List[Cargo], priority: Dict[str, float],
                                          k_levels: int) -> Dict[str, int]:
    """Квантильное разбиение приоритета на k уровней (1=самый приоритетный/левый)."""
    scored = sorted(cargos, key=lambda c: priority[c.id], reverse=True)
    n = len(scored)
    k_levels = max(1, min(k_levels, n))
    levels: Dict[str, int] = {}
    for idx, c in enumerate(scored):
        level = min(k_levels, idx * k_levels // n + 1)
        levels[c.id] = level
    return levels


# ---------------------------------------------------------------------------
# Последовательное выполнение с пересчётом приоритета на каждом шаге
# ---------------------------------------------------------------------------

@dataclass
class DistanceEntry:
    cargo_id: str
    pair_id: str
    estimated_distance: float   # straight_distance в момент выбора этого груза
    real_distance: float        # длина пути по графу в тот же момент
    result: TaskResult = None


@dataclass
class DistanceOutcome:
    L: float
    all_delivered: bool
    entries: List[DistanceEntry] = field(default_factory=list)
    estimated_total: float = 0.0
    real_total: float = 0.0


def run_sequential_by_distance_priority(env: IslandGraph, cargos: List[Cargo],
                                         pairs: List[Pair], L: float,
                                         k_levels: int = 4):
    """
    На каждом шаге для ВСЕХ ещё не выполненных грузов ПЕРЕСЧИТЫВАЕТСЯ
    приоритет (формула 2.4) от ТЕКУЩЕЙ позиции их назначенной пары - это
    отражает динамическую природу эвристики (она дешёвая именно потому,
    что её можно пересчитывать на каждом шаге без значительных затрат).
    Выбирается груз с максимальным приоритетом среди оставшихся в очереди
    каждой пары; пары работают независимо (как и раньше).

    Возвращает (outcome, levels, initial_priority) - levels/initial_priority
    нужны для построения графа приоритетов по НАЧАЛЬНОМУ состоянию (для
    наглядности на графике), сам процесс выполнения пересчитывает приоритет
    динамически.
    """
    cargos_copy = copy.deepcopy(cargos)
    pairs_state = {p.id: copy.deepcopy(p) for p in pairs}

    # приоритет в начальный момент - для построения графа приоритетов
    initial_priority = {
        c.id: compute_distance_priority(c, env, pairs_state[c.assigned_pair])
        for c in cargos_copy
    }
    levels = assign_priority_levels_from_priority(cargos_copy, initial_priority, k_levels)

    pending_by_pair: Dict[str, List[Cargo]] = {p.id: [] for p in pairs}
    for c in cargos_copy:
        pending_by_pair[c.assigned_pair].append(c)

    outcome = DistanceOutcome(L=L, all_delivered=False)

    for pair_id, pair in pairs_state.items():
        pending = pending_by_pair[pair_id]

        while pending:
            # динамический пересчёт приоритета от ТЕКУЩЕЙ позиции пары
            scored = [(compute_distance_priority(c, env, pair), c) for c in pending]
            scored.sort(key=lambda t: t[0], reverse=True)
            cargo = scored[0][1]

            estimated = env.straight_distance(pair.deliverer_pos, cargo.v_start)
            real = real_distance_to_pickup(env, pair.deliverer_pos, cargo.v_start, L)

            if real is None:
                outcome.all_delivered = False
                return outcome, levels, initial_priority

            result = find_route_cheapest_bridge(env, cargo, pair, L)
            if not result.feasible:
                outcome.all_delivered = False
                return outcome, levels, initial_priority

            pair.deliverer_pos = cargo.v_finish
            if result.bridges:
                pair.builder_pos = result.bridges[-1][1]
            else:
                pair.builder_pos = cargo.v_finish

            outcome.entries.append(DistanceEntry(
                cargo_id=cargo.id, pair_id=pair_id,
                estimated_distance=estimated, real_distance=real, result=result))
            outcome.estimated_total += estimated
            outcome.real_total += real

            pending = [c for c in pending if c.id != cargo.id]

    outcome.all_delivered = True
    return outcome, levels, initial_priority
