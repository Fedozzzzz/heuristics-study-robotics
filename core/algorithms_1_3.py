"""
Алгоритм 1: приоритет груза через прямое (евклидово) расстояние до ближайшего
            доставчика и ближайшего строителя одновременно (bottleneck-эвристика).
Алгоритм 3: поиск маршрута доставки P* и минимального набора мостов E_b для
            конкретной пары при заданном ограничении L (модифицированный Дейкстра).
"""

import math
from typing import List, Dict, Tuple

from delivery_model import IslandGraph, Cargo, Pair, TaskResult


# ---------------------------------------------------------------------------
# Алгоритм 1 — приоритет по прямому расстоянию (bottleneck по типам агентов)
# ---------------------------------------------------------------------------

def compute_priority(cargo: Cargo, env: IslandGraph,
                      deliverer_positions: List[int],
                      builder_positions: List[int]) -> float:
    """
    p(c_i) = 1 / max( min_k dist(v_dk, v_s^i),  min_l dist(v_bl, v_s^i) )

    dist - прямое (евклидово) расстояние, как если бы существовал прямой путь
    (эвристика для построения графа приоритетов, без обращения к графу G).
    """
    if not deliverer_positions or not builder_positions:
        return 0.0

    dist_d = min(env.straight_distance(v_s, cargo.v_start) for v_s in deliverer_positions)
    dist_b = min(env.straight_distance(v_s, cargo.v_start) for v_s in builder_positions)

    bottleneck = max(dist_d, dist_b)
    if bottleneck == 0:
        return math.inf  # агент уже стоит на точке погрузки - максимальный приоритет
    return 1.0 / bottleneck


def compute_priority_for_assigned_pair(cargo: Cargo, env: IslandGraph,
                                        pairs: List[Pair]) -> float:
    """
    Версия приоритета для случая, когда груз ЗАРАНЕЕ закреплён за конкретной
    парой (cargo.assigned_pair). В отличие от compute_priority (который берёт
    минимум по ВСЕМ доставщикам/строителям сразу - корректно только когда пара
    ещё предстоит выбрать), здесь расстояние считается относительно позиции
    ИМЕННО назначенной пары. Это даёт различающиеся, содержательные значения
    приоритета даже когда у разных грузов разные назначенные пары, расположенные
    в разных точках карты.
    """
    pair = next((p for p in pairs if p.id == cargo.assigned_pair), None)
    if pair is None:
        return 0.0

    dist_d = env.straight_distance(pair.deliverer_pos, cargo.v_start)
    dist_b = env.straight_distance(pair.builder_pos, cargo.v_start)

    bottleneck = max(dist_d, dist_b)
    if bottleneck == 0:
        return math.inf
    return 1.0 / bottleneck


def rank_cargo_by_priority(cargos: List[Cargo], env: IslandGraph,
                            deliverer_positions: List[int],
                            builder_positions: List[int]) -> List[Cargo]:
    """Возвращает список грузов, отсортированный по убыванию приоритета p(c_i)."""
    scored = [
        (compute_priority(c, env, deliverer_positions, builder_positions), c)
        for c in cargos if not c.delivered
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored]


def assign_priority_levels(cargos: List[Cargo], env: IslandGraph,
                            deliverer_positions: List[int],
                            builder_positions: List[int],
                            k_levels: int) -> Dict[str, int]:
    """
    Строит граф приоритетов слева-направо по k дискретным уровням
    (уровень 1 - самый приоритетный/левый, k - наименее приоритетный/правый).
    Разбиение - квантильное (равное число грузов в каждом уровне), устойчиво
    к выбросам в значениях p(c_i).

    Возвращает словарь {cargo_id: level}, где level in {1, ..., k_levels}.
    """
    scored = [
        (compute_priority(c, env, deliverer_positions, builder_positions), c)
        for c in cargos
    ]
    return _levels_from_scored(scored, k_levels)


def assign_priority_levels_for_assigned_pairs(cargos: List[Cargo], env: IslandGraph,
                                               pairs: List[Pair],
                                               k_levels: int) -> Dict[str, int]:
    """
    Версия assign_priority_levels для груза с ЗАРАНЕЕ закреплённой парой
    (cargo.assigned_pair) - использует compute_priority_for_assigned_pair,
    поэтому корректно различает грузы, назначенные разным, географически
    разнесённым парам.
    """
    scored = [
        (compute_priority_for_assigned_pair(c, env, pairs), c)
        for c in cargos
    ]
    return _levels_from_scored(scored, k_levels)


def _levels_from_scored(scored: List[Tuple[float, Cargo]], k_levels: int) -> Dict[str, int]:
    """Общая часть квантильного разбиения на уровни по уже посчитанным (p_val, cargo)."""
    scored.sort(key=lambda t: t[0], reverse=True)  # убывание приоритета

    n = len(scored)
    k_levels = max(1, min(k_levels, n))  # не больше уровней, чем грузов

    levels: Dict[str, int] = {}
    for idx, (p_val, cargo) in enumerate(scored):
        # квантильное разбиение: idx пробегает 0..n-1, делим на k_levels групп
        level = min(k_levels, idx * k_levels // n + 1)
        levels[cargo.id] = level

    return levels


# ---------------------------------------------------------------------------
# Алгоритм 3 — построение маршрута P* и мостов E_b для одной пары при данном L
# ---------------------------------------------------------------------------

def find_route_and_bridges(env: IslandGraph, cargo: Cargo, pair: Pair,
                            L: float, k_alternatives: int = 12) -> TaskResult:
    """
    Решает ТРИ подзадачи строителя последовательно (доставщик НЕ умеет
    строить мосты, поэтому весь путь, включая подъезд доставщика к v_start,
    должен быть предварительно "проложен" строителем):

      Шаг 1 (подход строителя): builder_pos -> deliverer_pos. Строитель сам
        добирается до текущей позиции доставщика, при необходимости строя
        мосты. Доставщик в этом шаге не участвует - это работа только
        строителя, её стоимость входит в W_b, но НЕ в W_d.

      Шаг 2 (подъезд): deliverer_pos -> v_start. Строитель (теперь уже
        находящийся в deliverer_pos после шага 1) строит маршрут до точки
        погрузки. Доставщик едет ПО ЭТОМУ ЖЕ маршруту (т.е. по мостам шага 2,
        плюс E_free, плюс pair.built_bridges с предыдущих задач) - весь
        набор мостов шага 2 доступен сразу, без учёта порядка постройки.

      Шаг 3 (доставка): v_start -> v_finish, строитель продолжает работу
        ИЗ ТОЧКИ, где оказался после шага 2 (а не из исходного builder_pos).

    W_b^i = W_build(шаг 1) + W_build(шаг 2) + W_build(шаг 3).
    W_d^i = approach_cost(шаг 2 - маршрут deliverer_pos->v_start, по которому
            едет доставщик) + W_path(шаг 3, т.е. v_start->v_finish).
            Шаг 1 НЕ входит в W_d - доставщик не участвует в этом шаге.

    Если хотя бы один из трёх шагов недостижим для строителя - вся задача
    помечается feasible=False (включая случай, когда сам строитель не может
    физически добраться до доставщика).
    """
    res = TaskResult(cargo_id=cargo.id, pair_id=pair.id, feasible=False)

    # already_built накопительно растёт по мере прохождения шагов 1->2->3,
    # чтобы один и тот же физический мост не оплачивался за постройку
    # повторно, если несколько сегментов этой задачи проходят через него.
    already_built = set(pair.built_bridges)

    # --- Шаг 1: строитель сам добирается до доставщика ---
    seg1 = env.build_segment_with_builder(
        pair.builder_pos, pair.deliverer_pos, pair.builder_pos, L,
        already_built=already_built, k_alternatives=k_alternatives)
    if seg1 is None:
        return res  # строитель не может физически добраться до доставщика
    already_built |= {(min(u, v), max(u, v)) for (u, v) in seg1["bridges"]}

    # --- Шаг 2: строитель строит маршрут до v_start; доставщик едет по нему ---
    seg2 = env.build_segment_with_builder(
        pair.deliverer_pos, cargo.v_start, seg1["builder_pos_after"], L,
        already_built=already_built, k_alternatives=k_alternatives)
    if seg2 is None:
        return res  # маршрут до v_start не может быть построен
    already_built |= {(min(u, v), max(u, v)) for (u, v) in seg2["bridges"]}

    usable_bridges_for_approach = set(already_built)
    w_d_approach, approach_path = env.deliverer_cost_with_approach(
        pair.deliverer_pos, [cargo.v_start], usable_bridges_for_approach)
    if w_d_approach is None:
        return res  # доставщик не может доехать до v_start даже по мостам шагов 1+2

    # --- Шаг 3: строитель продолжает работу из точки после шага 2 ---
    seg3 = env.build_segment_with_builder(
        cargo.v_start, cargo.v_finish, seg2["builder_pos_after"], L,
        already_built=already_built, k_alternatives=k_alternatives)
    if seg3 is None:
        return res  # доставка v_start->v_finish невозможна для этого строителя

    # w_d_approach уже включает w_V(v_start) ровно один раз (см.
    # deliverer_cost_with_approach: approach без последней вершины + main_path
    # из одной вершины v_start). seg3["W_path"] = path_cost_for_deliverer(seg3
    # ["path"]), где seg3["path"][0] == v_start - тоже включает w_V(v_start).
    # Вычитаем его здесь, чтобы не посчитать дважды при сложении.
    w_v_start = env.G.nodes[cargo.v_start]["w_V"]
    w_d_total = w_d_approach + seg3["W_path"] - w_v_start

    # duration = время занятости пары с момента начала задачи до освобождения:
    # шаги 1 и 2 строитель работает один (доставщик ждёт у deliverer_pos),
    # шаг 3 строитель и доставщик работают параллельно - задача завершается
    # когда оба заканчивают шаг 3.
    duration = seg1["W_build"] + seg2["W_build"] + max(w_d_total, seg3["W_build"])

    res.feasible = True
    res.path = seg3["path"]
    res.bridges = seg1["bridges"] + seg2["bridges"] + seg3["bridges"]
    res.approach_path = approach_path
    res.W_d = w_d_total
    res.W_b = seg1["W_build"] + seg2["W_build"] + seg3["W_build"]
    res.duration = duration
    return res

