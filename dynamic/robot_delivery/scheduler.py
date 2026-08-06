"""
Шаг 0 (проверка достижимости, feasibility.py) + Шаги 1-6 динамического цикла
распределения грузов по раундам с ГЛОБАЛЬНО доступными построенными мостами
(built -- единое состояние графа).

Шаг 1 (FORM-COALITIONS): на каждом раунде ещё свободные (для
зафиксированного варианта "1 доставщик - 1 строитель" -- это ВСЕ доставщики
и строители, т.к. раунд синхронизирован барьером и к его началу все роботы
свободны) доставщики и строители пересобираются в коалиции жадным
паросочетанием по минимальному расстоянию dist() на графе G (с учётом
переправ, доступных к возведению).

SELECT-ROUND:
  2. Строится таблица "коалиция x груз": для каждой достижимой комбинации
     считается ESTIMATE-TASK-COST и приоритет p (эвристика подставляется
     снаружи).
  3. Таблица разворачивается в список и сортируется по приоритету по
     убыванию (tie-break: id коалиции, затем id груза), список проходится
     один раз сверху вниз -- присваивание БЕЗ пересчёта под внутрираундовые
     коллизии (коллизии не могут сделать уже достижимую по глобальному
     built ячейку недостижимой, они лишь удешевляют её).
  4. Для каждого нового моста, нужного более чем одной коалиции из
     назначений раунда A, ОДНА коалиция выбирается СЛУЧАЙНО как та, что его
     реально строит (`won_bridges`); остальным ребро в рамках раунда --
     бесплатно. Финальная стоимость/маршрут каждого назначения пересчитываются
     с учётом этой скидки.

RUN-DYNAMIC-ROUNDS: сначала проверяет достижимость (Шаг 0); если недостижимо
-- возвращает RunResult с feasible=False без захода в цикл. Иначе вызывает
SELECT-ROUND, пока не опустеет pending; после каждого раунда синхронизируется
по барьеру (самая долгая доставка раунда), обновляет позиции роботов и built,
накапливает историю T для диагностики. Позиция доставщика после раунда --
точка разгрузки груза; позиция строителя -- конец последнего ЛИЧНО
построенного им моста (won_bridges), либо прежняя позиция, если он сам
ничего не строил в этом раунде.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .costs import EstimateResult, estimate_task_cost
from .feasibility import check_feasibility
from .graph import EdgeKey, EdgeKind, Graph, NodeId
from .heuristics import Heuristic, HeuristicContext

CargoId = int
RobotId = int


@dataclass
class Cargo:
    cargo_id: CargoId
    start: NodeId
    finish: NodeId


@dataclass
class Coalition:
    coalition_id: int          # == deliverer_id (паросочетание 1-1 биективно)
    deliverer_id: RobotId
    builder_id: RobotId
    deliverer_pos: NodeId
    builder_pos: NodeId


@dataclass
class ScheduleRecord:
    coalition_id: int
    deliverer_id: RobotId
    builder_id: RobotId
    cargo_id: CargoId
    path_nodes: List[NodeId]
    bridges: Set[EdgeKey]          # все мосты операции (в т.ч. оплаченные другой коалицией)
    won_bridges: Set[EdgeKey]      # мосты, которые ЛИЧНО построил строитель этой коалиции
    W_d: float
    W_b: float
    p_rank: float                 # приоритет, по которому запись была отсортирована
    W_T_initial: float            # W_T на момент построения таблицы (до скидки за коллизии)
    pos_round: Tuple[NodeId, NodeId]  # (deliverer_pos, builder_pos) НА НАЧАЛО раунда
    round_index: int
    duration: float               # длительность именно этой доставки (без барьера)
    builder_final_pos: Optional[NodeId]  # конечная позиция строителя, None -- если он ничего не строил


Cell = Tuple[Coalition, Cargo, EstimateResult]


def form_coalitions(
    G: Graph, deliverer_pos: Dict[RobotId, NodeId], builder_pos: Dict[RobotId, NodeId]
) -> List[Coalition]:
    """Шаг 1 (FORM-COALITIONS), зафиксированный вариант "1 доставщик - 1
    строитель": жадное паросочетание по минимальному dist(v_d, v_b) --
    кратчайшему пути на графе G с весами w_E, w_V, учитывающему в том числе
    переправы, доступные к возведению (E_blocked), но без учёта постройки.

    Реализовано как одна сортировка по возрастанию расстояния для ВСЕХ пар
    (d, b) с последующим однопроходным жадным отбором -- эквивалентно
    повторному argmin с удалением, но без пересчёта расстояний на каждой
    итерации."""

    def passable(e) -> bool:
        return e.kind != EdgeKind.IMPOSSIBLE

    candidates: List[Tuple[float, RobotId, RobotId]] = []
    for d_id, d_pos in deliverer_pos.items():
        for b_id, b_pos in builder_pos.items():
            dist = G.shortest_distance(d_pos, b_pos, passable)
            if dist is not None:
                candidates.append((dist, d_id, b_id))
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))

    used_d: Set[RobotId] = set()
    used_b: Set[RobotId] = set()
    coalitions: List[Coalition] = []
    for _dist, d_id, b_id in candidates:
        if d_id in used_d or b_id in used_b:
            continue
        used_d.add(d_id)
        used_b.add(b_id)
        coalitions.append(
            Coalition(
                coalition_id=d_id,
                deliverer_id=d_id,
                builder_id=b_id,
                deliverer_pos=deliverer_pos[d_id],
                builder_pos=builder_pos[b_id],
            )
        )
    coalitions.sort(key=lambda c: c.coalition_id)
    return coalitions


def _order_greedy(cells: List[Cell], scores: List[float]) -> List[int]:
    """Глобальный жадный порядок: вся таблица "коалиция x груз" сортируется
    по убыванию приоритета один раз, и самая приоритетная ячейка во всей
    таблице получает назначение первой -- независимо от того, какой
    коалиции она принадлежит. Ровно поведение, описанное в Шаге 3."""
    return sorted(
        range(len(cells)),
        key=lambda i: (-scores[i], cells[i][0].coalition_id, cells[i][1].cargo_id),
    )


def _order_round_robin(cells: List[Cell], scores: List[float], round_index: int) -> List[int]:
    """Round-robin порядок: коалиции получают "ход" по очереди (порядок --
    по coalition_id, со сдвигом на round_index). На своём ходу коалиция
    выбирает лучший ДЛЯ СЕБЯ ещё доступный груз (по тем же скорам, что и
    greedy)."""
    by_coalition: Dict[int, List[int]] = {}
    for idx, (coalition, _cargo, _est) in enumerate(cells):
        by_coalition.setdefault(coalition.coalition_id, []).append(idx)
    for idxs in by_coalition.values():
        idxs.sort(key=lambda i: (-scores[i], cells[i][1].cargo_id))

    coalition_ids = sorted(by_coalition.keys())
    if coalition_ids:
        shift = round_index % len(coalition_ids)
        coalition_ids = coalition_ids[shift:] + coalition_ids[:shift]

    order: List[int] = []
    for cid in coalition_ids:
        order.extend(by_coalition[cid])
    return order


ASSIGNMENT_ALGOS: Dict[str, str] = {
    "greedy": "Глобальный жадный проход по отсортированной таблице приоритетов "
              "(текущее поведение SELECT-ROUND).",
    "round_robin": "Коалиции выбирают груз по очереди (round-robin, со сдвигом по "
                    "раундам): каждая -- лучший ещё доступный ей груз.",
}


def select_round(
    G: Graph,
    pending: Sequence[Cargo],
    deliverer_pos: Dict[RobotId, NodeId],
    builder_pos: Dict[RobotId, NodeId],
    built: Set[EdgeKey],
    heuristic: Heuristic,
    pair_loads: Dict[int, float],
    round_index: int,
    rng: random.Random,
    assignment: str = "greedy",
) -> Tuple[List[ScheduleRecord], Set[EdgeKey], List[Coalition]]:
    """Один раунд: Шаг 1 (формирование коалиций) + Шаги 2-4 (таблица
    приоритетов, присваивание, разрешение коллизий за мосты). Возвращает
    (назначения, обновлённый built, коалиции этого раунда)."""

    # --- Шаг 1: коалиции этого раунда ---
    coalitions = form_coalitions(G, deliverer_pos, builder_pos)
    if not coalitions:
        return [], set(built), []

    # --- Шаг 2: таблица "коалиция x груз" ---
    contexts: List[HeuristicContext] = []
    cells: List[Cell] = []

    for coalition in coalitions:
        for cargo in pending:
            est = estimate_task_cost(
                G, cargo.start, cargo.finish, coalition.deliverer_pos, coalition.builder_pos, built
            )
            if est is None:
                continue
            ctx = HeuristicContext(
                estimate=est,
                G=G,
                built=built,
                deliverer_pos=coalition.deliverer_pos,
                builder_pos=coalition.builder_pos,
                c_start=cargo.start,
                c_finish=cargo.finish,
                coalition_id=coalition.coalition_id,
                cargo_id=cargo.cargo_id,
                pair_load=pair_loads.get(coalition.coalition_id, 0.0),
            )
            contexts.append(ctx)
            cells.append((coalition, cargo, est))

    if not cells:
        return [], set(built), coalitions

    scores = heuristic.score_table(contexts)
    if assignment == "greedy":
        order = _order_greedy(cells, scores)
    elif assignment == "round_robin":
        order = _order_round_robin(cells, scores, round_index)
    else:
        raise KeyError(
            f"Неизвестный алгоритм распределения '{assignment}'. "
            f"Доступные: {', '.join(sorted(ASSIGNMENT_ALGOS))}"
        )

    # --- Шаг 3: присваивание, БЕЗ пересчёта под внутрираундовые коллизии ---
    assigned_coalitions: Set[int] = set()
    assigned_cargo: Set[CargoId] = set()
    assigned: List[Tuple[Coalition, Cargo, EstimateResult, float]] = []

    for i in order:
        coalition, cargo, est = cells[i]
        if coalition.coalition_id in assigned_coalitions or cargo.cargo_id in assigned_cargo:
            continue
        assigned_coalitions.add(coalition.coalition_id)
        assigned_cargo.add(cargo.cargo_id)
        assigned.append((coalition, cargo, est, scores[i]))

    if not assigned:
        return [], set(built), coalitions

    # --- Шаг 4: конфликт за один и тот же мост -- случайный победитель ---
    edge_users: Dict[EdgeKey, List[int]] = {}
    for idx, (_coalition, _cargo, est, _score) in enumerate(assigned):
        for ek in est.bridges:
            if ek in built:
                continue
            edge_users.setdefault(ek, []).append(idx)

    winner_of_edge: Dict[EdgeKey, int] = {}
    for ek, users in edge_users.items():
        winner_of_edge[ek] = users[0] if len(users) == 1 else rng.choice(users)

    built_after: Set[EdgeKey] = set(built)
    records: List[ScheduleRecord] = []

    for idx, (coalition, cargo, est_initial, score) in enumerate(assigned):
        free_for_me = {
            ek for ek in est_initial.bridges
            if ek not in built and winner_of_edge.get(ek, idx) != idx
        }
        if free_for_me:
            est_final = estimate_task_cost(
                G, cargo.start, cargo.finish, coalition.deliverer_pos, coalition.builder_pos,
                built | free_for_me,
            )
            assert est_final is not None, "скидка за коллизии не может сделать маршрут недостижимым"
        else:
            est_final = est_initial

        won_bridges = {
            ek for ek in est_final.bridges
            if ek not in built and winner_of_edge.get(ek, idx) == idx
        }
        built_after |= set(est_final.bridges)

        records.append(
            ScheduleRecord(
                coalition_id=coalition.coalition_id,
                deliverer_id=coalition.deliverer_id,
                builder_id=coalition.builder_id,
                cargo_id=cargo.cargo_id,
                path_nodes=est_final.path_nodes,
                bridges=set(est_final.bridges),
                won_bridges=won_bridges,
                W_d=est_final.W_d,
                W_b=est_final.W_b,
                p_rank=score,
                W_T_initial=est_initial.W_T,
                pos_round=(coalition.deliverer_pos, coalition.builder_pos),
                round_index=round_index,
                duration=est_final.duration,
                builder_final_pos=est_final.builder_final_pos,
            )
        )

    return records, built_after, coalitions


@dataclass
class RunResult:
    T: List[ScheduleRecord]
    W_d_total: float
    W_b_total: float
    idle_total: float
    n_rounds: int
    all_delivered: bool
    delivered_cargo: Set[CargoId]
    feasible: bool = True
    infeasibility_reason: Optional[str] = None


def run_dynamic_rounds(
    G: Graph,
    cargos: Sequence[Cargo],
    deliverer_positions: Sequence[NodeId],
    builder_positions: Sequence[NodeId],
    heuristic: Heuristic,
    max_rounds: int = 10_000,
    assignment: str = "greedy",
    rng_seed: int = 0,
) -> RunResult:
    """Algorithm RUN-DYNAMIC-ROUNDS(G, C, R_d, R_b). assignment -- см.
    select_round. rng_seed -- сид генератора случайности для Шага 4
    (случайный выбор коалиции, оплачивающей постройку моста при коллизии);
    прогон с одинаковым rng_seed полностью воспроизводим."""

    deliverer_pos: Dict[RobotId, NodeId] = {i: p for i, p in enumerate(deliverer_positions)}
    builder_pos: Dict[RobotId, NodeId] = {i: p for i, p in enumerate(builder_positions)}

    # --- Шаг 0: проверка достижимости ---
    feasibility = check_feasibility(G, cargos, deliverer_pos, builder_pos)
    if not feasibility.ok:
        return RunResult(
            T=[],
            W_d_total=0.0,
            W_b_total=0.0,
            idle_total=0.0,
            n_rounds=0,
            all_delivered=False,
            delivered_cargo=set(),
            feasible=False,
            infeasibility_reason=feasibility.reason,
        )

    pending: Dict[CargoId, Cargo] = {c.cargo_id: c for c in cargos}
    built: Set[EdgeKey] = set()
    pair_loads: Dict[int, float] = {i: 0.0 for i in deliverer_pos}
    rng = random.Random(rng_seed)

    T: List[ScheduleRecord] = []
    W_d_total = 0.0
    W_b_total = 0.0
    idle_total = 0.0
    round_index = 0

    while pending and round_index < max_rounds:
        cargo_snapshot = list(pending.values())
        records, built, _coalitions = select_round(
            G, cargo_snapshot, dict(deliverer_pos), dict(builder_pos), built, heuristic,
            pair_loads, round_index, rng, assignment=assignment,
        )
        if not records:
            break  # остаток недостижим при текущем состоянии роботов

        round_duration = max(r.duration for r in records)
        idle_total += sum(round_duration - r.duration for r in records)

        for r in records:
            T.append(r)
            W_d_total += r.W_d
            W_b_total += r.W_b
            pair_loads[r.coalition_id] = pair_loads.get(r.coalition_id, 0.0) + (r.W_d + r.W_b)

            # Шаг 5: позиция доставщика -- точка разгрузки
            deliverer_pos[r.deliverer_id] = pending[r.cargo_id].finish
            # позиция строителя -- конец последнего ЛИЧНО построенного им
            # моста; если он сам ничего не строил в этом раунде -- прежняя
            # позиция не меняется
            if r.won_bridges and r.builder_final_pos is not None:
                builder_pos[r.builder_id] = r.builder_final_pos

            del pending[r.cargo_id]

        round_index += 1

    return RunResult(
        T=T,
        W_d_total=W_d_total,
        W_b_total=W_b_total,
        idle_total=idle_total,
        n_rounds=round_index,
        all_delivered=(len(pending) == 0),
        delivered_cargo={r.cargo_id for r in T},
        feasible=True,
        infeasibility_reason=None,
    )
