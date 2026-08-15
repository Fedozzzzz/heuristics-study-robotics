"""
Шаг 1. Оценка приоритетов грузов --- НЕЗАВИСИМО от роботов.

Главное отличие dynamic_v2 от dynamic/: приоритет груза считается САМ ПО
СЕБЕ, до и без формирования коалиций. Для каждого груза оценивается
ROUTE-AND-COST(G, c_start, c_finish, built) --- стоимость маршрута от точки
погрузки до точки разгрузки с учётом уже готовых переправ (built) и с учётом
ещё не построенных на этом маршруте (их стоимость постройки включается в
оценку), но БЕЗ учёта перемещений роботов до точки погрузки и БЕЗ учёта
достраивания переправ для этого подъезда (см. постановку, Шаг 1: "без учета
перемещений роботов и достраивания переправ для доезда до начала погрузки").

built передаётся снаружи и на практике меняется от раунда к раунду (переправы
накапливаются и не строятся повторно), поэтому приоритеты грузов
пересчитываются заново в начале КАЖДОГО раунда (см. scheduler.select_round).

В Алгоритме 1 (см. scheduler.run_dynamic_rounds, fresh_graph_each_round=True)
built сюда всегда приходит ПУСТЫМ: там построенное не помнится вообще, поэтому
приоритет груза каждый раунд считается заново с полной стоимостью постройки
всех E_blocked его маршрута. Отдельного флага здесь для этого не нужно.

Приоритет p назначается одной из эвристик, зафиксированных в постановке
задачи ("Рассматриваемые эвристики -- прямая и обратная"), плюс baseline для
оценки их качества:
  direct  (p = W_C)     --- дороже маршрут груза -- выше приоритет.
  inverse (p = 1 / W_C) --- дешевле маршрут груза -- выше приоритет.
  random  (p ~ U(0,1))  --- baseline: не зависит от стоимости, эквивалент
                             случайного распределения приоритета грузов
                             (перенесено из dynamic/, см. random_priority
                             в dynamic/robot_delivery/heuristics.py).
"""

from __future__ import annotations

import random as _random_module
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

from .costs import route_and_cost
from .graph import EdgeKey, Graph, NodeId

EPS = 1e-9


def route_cost_for_cargo(G: Graph, start: NodeId, finish: NodeId,
                         built: Set[EdgeKey]) -> float:
    """W_C --- стоимость маршрута c_start -> c_finish, без участия роботов:
    проезд + стоимость постройки тех E_blocked маршрута, которых нет в built.

    При built = ∅ (Алгоритм 1 -- на любом раунде) это даёт "цену с чистого
    поля": оплачивается постройка ВСЕХ заблокированных переправ маршрута.
    """
    r = route_and_cost(G, start, finish, built)
    if r is None:
        # Не должно происходить: Шаг 0 (check_feasibility) уже гарантирует,
        # что все точки погрузки/разгрузки лежат в одной компоненте связности
        # G' = (V, E_ready ∪ E_blocked), т.е. каждый груз достижим при
        # каком-то built.
        raise RuntimeError(
            f"груз {start}->{finish} недостижим даже с учётом всех потенциальных "
            "построек -- должно быть отсечено Шагом 0 (check_feasibility)"
        )
    return r.travel_cost + r.build_cost


@dataclass
class CargoPriorityHeuristic:
    key: str
    label: str
    description: str
    score: Callable[[Any, float], float]   # (cargo, W_C) -> p


def _direct(cargo: Any, w: float) -> float:
    # p = W_C -- дороже маршрут -- выше приоритет.
    return w


def _inverse(cargo: Any, w: float) -> float:
    # p = 1 / W_C -- дешевле маршрут -- выше приоритет.
    return 1.0 / (w + EPS)


def _random(cargo: Any, w: float) -> float:
    # p ~ U(0,1), не зависит от W_C -- чистый шум. Сид берётся из состава
    # груза (id + точки погрузки/разгрузки), а не из глобального random,
    # чтобы прогон был воспроизводим при повторном вызове с тем же сценарием
    # (та же техника, что и random_priority в dynamic/robot_delivery/heuristics.py).
    seed = (cargo.cargo_id, cargo.start, cargo.finish)
    return _random_module.Random(hash(seed)).random()


CARGO_HEURISTICS: Dict[str, CargoPriorityHeuristic] = {
    "direct": CargoPriorityHeuristic(
        key="direct",
        label="Прямая (p = W_C)",
        description="Дороже маршрут груза -- выше приоритет.",
        score=_direct,
    ),
    "inverse": CargoPriorityHeuristic(
        key="inverse",
        label="Обратная (p = 1 / W_C)",
        description="Дешевле маршрут груза -- выше приоритет.",
        score=_inverse,
    ),
    "random": CargoPriorityHeuristic(
        key="random",
        label="Случайный приоритет (baseline)",
        description=(
            "Baseline для оценки качества direct/inverse: p ~ U(0,1), не "
            "зависит от стоимости маршрута груза -- эквивалент случайного "
            "распределения приоритета грузов."
        ),
        score=_random,
    ),
}


def get_cargo_heuristic(name: str) -> CargoPriorityHeuristic:
    if name not in CARGO_HEURISTICS:
        raise KeyError(
            f"Неизвестная эвристика приоритета груза '{name}'. Доступные: "
            f"{', '.join(sorted(CARGO_HEURISTICS))}"
        )
    return CARGO_HEURISTICS[name]


def rank_cargos(
    G: Graph, cargos: Sequence, built: Set[EdgeKey], heuristic: CargoPriorityHeuristic,
) -> List[Tuple[float, float, object]]:
    """Список грузов, отсортированный по убыванию приоритета p (Шаг 1).
    Возвращает [(p, W_C, cargo), ...]. Tie-break -- по cargo_id (воспроизводимость)."""
    scored: List[Tuple[float, float, object]] = []
    for cargo in cargos:
        w = route_cost_for_cargo(G, cargo.start, cargo.finish, built)
        p = heuristic.score(cargo, w)
        scored.append((p, w, cargo))
    scored.sort(key=lambda t: (-t[0], t[2].cargo_id))
    return scored
