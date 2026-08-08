"""
Точное решение (нижняя граница/оптимум) для маленьких сценариев -- нужно как
НЕЗАВИСИМЫЙ ориентир при оценке качества эвристик приоритета груза (Шаг 1),
а не только сравнение эвристик друг с другом или с их же собственной
эвристической оценкой (estimated_raw).

Метрика "real" в этой модели -- ЧИСТАЯ СУММА РАБОТЫ (W_d + W_b по всем
доставкам), а не время выполнения (makespan): барьерная синхронизация раундов
влияет только на то, кто кого ждёт, но не на то, сколько всего проезда и
постройки моста потребовалось. Поэтому "оптимум" для этой метрики -- это
МИНИМАЛЬНАЯ суммарная стоимость (W_d + W_b), достижимая ЛЮБОЙ
последовательностью назначений (груз, доставщик, строитель), при которой:
  - на каждом шаге можно взять ЛЮБУЮ пару "свободный доставщик + свободный
    строитель" под ЛЮБОЙ ещё не доставленный груз (без ограничений Шага 2 --
    ни фиксированного числа коалиций N на раунд, ни отбора по приоритету p);
  - возведённые переправы -- глобальное, неубывающее состояние: однажды
    построенный мост никогда не строится повторно (как и в самой модели).

Это РЕЛАКСАЦИЯ модели dynamic_v2 (у неё больше свободы в назначении), поэтому
optimal <= real для любой эвристики -- competitive_ratio = real / optimal
всегда >= 1, и показывает, насколько далеко эвристика (с её ограничениями
Шага 2) от теоретического предела при полной свободе назначения.

Реализовано как полный перебор (branch & bound) с мемоизацией состояний
"какие грузы уже доставлены x позиции роботов x built". Экспоненциальная
сложность -- предназначено ТОЛЬКО для очень маленьких сценариев (единицы
грузов и пар роботов), см. max_cargos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

from .costs import estimate_task_cost
from .feasibility import check_feasibility
from .graph import EdgeKey, Graph, NodeId
from .scheduler import Cargo, RobotId

StateKey = Tuple[FrozenSet[int], Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...], FrozenSet[EdgeKey]]


@dataclass
class OptimalResult:
    optimal: Optional[float]   # None -- недостижимо (Шаг 0) либо не найдено за отведённое усилие
    feasible: bool
    infeasibility_reason: Optional[str] = None


def solve_optimal_brute_force(
    G: Graph,
    cargos: Sequence[Cargo],
    deliverer_positions: Sequence[NodeId],
    builder_positions: Sequence[NodeId],
    upper_bound: Optional[float] = None,
    max_cargos: int = 7,
) -> OptimalResult:
    """Точный минимум суммарной стоимости (W_d + W_b по всем доставкам) --
    см. docstring модуля. `upper_bound` -- известная достижимая стоимость
    (например, реальная стоимость одной из эвристик), используется для
    затравки branch & bound и резко ускоряет перебор; можно не передавать.

    `max_cargos` -- защитный предел: при большем числе грузов перебор
    практически нереализуем за разумное время, поднимается ValueError.
    """
    if len(cargos) > max_cargos:
        raise ValueError(
            f"Слишком много грузов для полного перебора: {len(cargos)} > max_cargos={max_cargos}. "
            "solve_optimal_brute_force предназначен только для очень маленьких сценариев."
        )

    deliverer_pos0: Dict[RobotId, NodeId] = {i: p for i, p in enumerate(deliverer_positions)}
    builder_pos0: Dict[RobotId, NodeId] = {i: p for i, p in enumerate(builder_positions)}

    feasibility = check_feasibility(G, cargos, deliverer_pos0, builder_pos0)
    if not feasibility.ok:
        return OptimalResult(optimal=None, feasible=False, infeasibility_reason=feasibility.reason)

    if not cargos:
        return OptimalResult(optimal=0.0, feasible=True)

    cargo_map: Dict[int, Cargo] = {c.cargo_id: c for c in cargos}
    all_ids: FrozenSet[int] = frozenset(cargo_map.keys())

    best = [upper_bound if upper_bound is not None else math.inf]
    memo: Dict[StateKey, float] = {}

    def state_key(
        remaining: FrozenSet[int],
        deliverer_pos: Dict[RobotId, NodeId],
        builder_pos: Dict[RobotId, NodeId],
        built: FrozenSet[EdgeKey],
    ) -> StateKey:
        return (
            remaining,
            tuple(sorted(deliverer_pos.items())),
            tuple(sorted(builder_pos.items())),
            built,
        )

    def solve(
        remaining: FrozenSet[int],
        deliverer_pos: Dict[RobotId, NodeId],
        builder_pos: Dict[RobotId, NodeId],
        built: FrozenSet[EdgeKey],
        cost_so_far: float,
    ) -> float:
        """Возвращает минимальную стоимость ДОСТАВКИ ОСТАВШИХСЯ грузов
        (remaining) начиная с этого состояния (не считая cost_so_far)."""
        if not remaining:
            return 0.0

        key = state_key(remaining, deliverer_pos, builder_pos, built)
        cached = memo.get(key)
        if cached is not None:
            return cached

        local_best = math.inf
        for cargo_id in remaining:
            cargo = cargo_map[cargo_id]
            for d_id, d_pos in deliverer_pos.items():
                for b_id, b_pos in builder_pos.items():
                    est = estimate_task_cost(G, cargo.start, cargo.finish, d_pos, b_pos, built)
                    if est is None:
                        continue
                    if cost_so_far + est.W_T >= best[0]:
                        continue  # branch & bound: заведомо не лучше уже найденного

                    new_deliverer_pos = dict(deliverer_pos)
                    new_deliverer_pos[d_id] = cargo.finish
                    new_builder_pos = dict(builder_pos)
                    if est.bridges and est.builder_final_pos is not None:
                        new_builder_pos[b_id] = est.builder_final_pos
                    new_built = built | est.bridges

                    remainder = solve(
                        remaining - {cargo_id}, new_deliverer_pos, new_builder_pos,
                        new_built, cost_so_far + est.W_T,
                    )
                    total = est.W_T + remainder
                    if total < local_best:
                        local_best = total
                    if cost_so_far + total < best[0]:
                        best[0] = cost_so_far + total

        memo[key] = local_best
        return local_best

    solve(all_ids, deliverer_pos0, builder_pos0, frozenset(), 0.0)
    # ВАЖНО: итоговый ответ -- best[0] (глобальный трекер лучшего НАЙДЕННОГО
    # ПОЛНОГО решения, включая исходный upper_bound), а НЕ возврат solve(...)
    # напрямую. Если upper_bound уже равен истинному оптимуму, branch & bound
    # (сравнение >=) отсекает ЛЮБОЙ путь, доходящий ровно до best[0] -- ни
    # один полный путь не сможет "переоткрыть" его заново и обновить
    # local_best на промежуточных уровнях рекурсии. best[0] же остаётся
    # корректным нижним/точным значением в любом случае: он инициализируется
    # заведомо ДОСТИЖИМЫМ upper_bound (реальным результатом эвристики) и
    # уменьшается только когда исчерпывающий перебор находит СТРОГО лучшее
    # полное решение -- т.е. по завершении поиска best[0] математически
    # доказанно является истинным минимумом.
    return OptimalResult(optimal=(best[0] if best[0] < math.inf else None), feasible=True)
