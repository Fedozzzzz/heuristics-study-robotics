"""
Шаг 0 (проверка достижимости, feasibility.py) + Шаги 1-6 динамического цикла
распределения грузов по раундам, dynamic_v2 --- см. README.md об отличиях от
dynamic/. built (возведённые переправы) --- ГЛОБАЛЬНОЕ состояние, общее для
всех раундов: однажды построенный мост никогда не строится повторно (Шаг 4 в
постановке задачи).

Шаг 1 (RANK-CARGOS, cargo_priority.py): приоритет груза p считается ТОЛЬКО по
стоимости его собственного маршрута c_start -> c_finish, независимо от того,
где сейчас находятся роботы. Список грузов пересчитывается заново в начале
КАЖДОГО раунда (built меняется между раундами -- построенные переправы
удешевляют маршруты).

Шаг 2 (SELECT-ROUND, здесь): виртуальная пара доставщик-строитель
("коалиция") формируется ПОД ВЫБРАННЫЙ груз, а не наоборот. Число коалиций
раунда:
    N = min( floor((|R_d| + |R_b|) / 2), |R_d|, |R_b|, |pending| )
При |R_d| == |R_b| (обычный случай для этой модели) это сводится к N =
|R_d| = |R_b| -- "число пар роботов". Общий min() не даёт формуле выйти за
физически возможное число биективных пар доставщик-строитель, если
доставщиков и строителей не поровну.

Далее грузы разбираются В ПОРЯДКЕ УБЫВАНИЯ ПРИОРИТЕТА (жадно, один проход),
пока не набрано N коалиций:
  - доставщик: ближайший ЕЩЁ СВОБОДНЫЙ доставщик к точке погрузки груза
    (dist() на графе G, с учётом переправ, доступных к возведению, но без
    учёта их постройки -- как и в dynamic/);
  - строитель: среди ЕЩЁ СВОБОДНЫХ строителей выбирается тот, для кого
    ESTIMATE-TASK-COST(груз, выбранный доставщик, строитель, built) требует
    построить МИНИМАЛЬНОЕ число НОВЫХ мостов; при равенстве -- минимальную
    суммарную стоимость их постройки (см. README про выбор tie-break).
  - если для груза не находится ни доставщика, ни строителя (все свободные
    роботы исчерпаны, либо маршрут для всех комбинаций недостижим) -- груз
    пропускается В ЭТОМ раунде и остаётся pending для следующего.

Шаг 3 (эвристическая оценка стоимости раунда без учёта возводимых переправ):
это W_T_initial каждой сформированной коалиции -- ESTIMATE-TASK-COST ДО
скидки за внутрираундовое резервирование мостов (Шаг 4). Суммируется в
diagnostics.compute_dynamic_cost_bracket.

Шаг 4 (конфликт за один и тот же ещё не построенный мост между несколькими
коалициями раунда): случайный победитель строит мост за свой счёт, для
остальных в рамках раунда (и всех последующих, т.к. built глобален) проезд
по нему бесплатен.

Шаг 5/6 (RUN-DYNAMIC-ROUNDS): барьерная синхронизация раунда по самой долгой
доставке, обновление позиций (доставщик -- в точке разгрузки, строитель -- в
конце последнего ЛИЧНО построенного им моста, либо на прежнем месте), цикл
до тех пор, пока остаются недоставленные грузы.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .cargo_priority import CargoPriorityHeuristic, rank_cargos
from .costs import EstimateResult, estimate_task_cost
from .feasibility import check_feasibility
from .graph import EdgeKey, EdgeKind, Graph, NodeId

CargoId = int
RobotId = int


@dataclass
class Cargo:
    cargo_id: CargoId
    start: NodeId
    finish: NodeId


@dataclass
class Coalition:
    coalition_id: int          # == deliverer_id
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
    p_rank: float                  # приоритет груза p (Шаг 1), по которому груз был отобран
    W_T_initial: float             # W_T на момент формирования коалиции (до скидки за коллизии, Шаг 3)
    pos_round: Tuple[NodeId, NodeId]  # (deliverer_pos, builder_pos) НА НАЧАЛО раунда
    round_index: int
    duration: float                # длительность именно этой доставки (без барьера)
    builder_final_pos: Optional[NodeId]  # конечная позиция строителя, None -- если он ничего не строил


def _passable(e) -> bool:
    return e.kind != EdgeKind.IMPOSSIBLE


def _nearest_available_deliverer(
    G: Graph, cargo_start: NodeId, deliverer_pos: Dict[RobotId, NodeId], available: Set[RobotId]
) -> Optional[RobotId]:
    """Критерий доставщика (Шаг 2): минимальное расстояние до точки
    отправления груза, среди ещё свободных доставщиков."""
    best_id: Optional[RobotId] = None
    best_d: Optional[float] = None
    for d_id in sorted(available):
        d = G.shortest_distance(deliverer_pos[d_id], cargo_start, _passable)
        if d is None:
            continue
        if best_d is None or d < best_d:
            best_id, best_d = d_id, d
    return best_id


def _best_available_builder(
    G: Graph,
    cargo: Cargo,
    deliverer_position: NodeId,
    builder_pos: Dict[RobotId, NodeId],
    available: Set[RobotId],
    built: Set[EdgeKey],
) -> Tuple[Optional[RobotId], Optional[EstimateResult]]:
    """Критерий строителя (Шаг 2): минимальное число НОВЫХ мостов, нужных для
    выполнения операции целиком (маршрут доставщика + подъезд строителя).
    Tie-break -- минимальная суммарная стоимость их постройки."""
    best_id: Optional[RobotId] = None
    best_key: Optional[Tuple[int, float, RobotId]] = None
    best_est: Optional[EstimateResult] = None
    for b_id in sorted(available):
        est = estimate_task_cost(G, cargo.start, cargo.finish, deliverer_position, builder_pos[b_id], built)
        if est is None:
            continue
        new_bridges = est.bridges - built
        n_new = len(new_bridges)
        build_cost = sum(G.edges[ek].w_build for ek in new_bridges)
        key = (n_new, build_cost, b_id)
        if best_key is None or key < best_key:
            best_key, best_id, best_est = key, b_id, est
    return best_id, best_est


def select_round(
    G: Graph,
    pending: Sequence[Cargo],
    deliverer_pos: Dict[RobotId, NodeId],
    builder_pos: Dict[RobotId, NodeId],
    built: Set[EdgeKey],
    cargo_heuristic: CargoPriorityHeuristic,
    round_index: int,
    rng: random.Random,
) -> Tuple[List[ScheduleRecord], Set[EdgeKey]]:
    """Один раунд: Шаг 1 (приоритет грузов) + Шаг 2 (отбор N грузов и
    формирование коалиций под них) + Шаги 3-4 (оценка раунда, разрешение
    коллизий за общие мосты). Возвращает (назначения, обновлённый built)."""

    n_deliverers = len(deliverer_pos)
    n_builders = len(builder_pos)
    if n_deliverers == 0 or n_builders == 0 or not pending:
        return [], set(built)

    N = min((n_deliverers + n_builders) // 2, n_deliverers, n_builders, len(pending))
    if N <= 0:
        return [], set(built)

    # --- Шаг 1: приоритет грузов, независимый от роботов ---
    ranked = rank_cargos(G, pending, built, cargo_heuristic)

    # --- Шаг 2: жадный отбор N грузов + формирование коалиций под них ---
    available_d: Set[RobotId] = set(deliverer_pos.keys())
    available_b: Set[RobotId] = set(builder_pos.keys())
    formed: List[Tuple[Coalition, Cargo, EstimateResult, float]] = []

    for p, _w_c, cargo in ranked:
        if len(formed) >= N:
            break
        if not available_d or not available_b:
            break
        d_id = _nearest_available_deliverer(G, cargo.start, deliverer_pos, available_d)
        if d_id is None:
            continue
        b_id, est = _best_available_builder(
            G, cargo, deliverer_pos[d_id], builder_pos, available_b, built
        )
        if b_id is None:
            continue
        available_d.discard(d_id)
        available_b.discard(b_id)
        coalition = Coalition(
            coalition_id=d_id,
            deliverer_id=d_id,
            builder_id=b_id,
            deliverer_pos=deliverer_pos[d_id],
            builder_pos=builder_pos[b_id],
        )
        formed.append((coalition, cargo, est, p))

    if not formed:
        return [], set(built)

    # --- Шаг 4: конфликт за один и тот же ещё не построенный мост между
    #     несколькими коалициями раунда -- случайный победитель ---
    edge_users: Dict[EdgeKey, List[int]] = {}
    for idx, (_c, _cargo, est, _p) in enumerate(formed):
        for ek in est.bridges:
            if ek in built:
                continue
            edge_users.setdefault(ek, []).append(idx)

    winner_of_edge: Dict[EdgeKey, int] = {}
    for ek, users in edge_users.items():
        winner_of_edge[ek] = users[0] if len(users) == 1 else rng.choice(users)

    built_after: Set[EdgeKey] = set(built)
    records: List[ScheduleRecord] = []

    for idx, (coalition, cargo, est_initial, p) in enumerate(formed):
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
                p_rank=p,
                W_T_initial=est_initial.W_T,
                pos_round=(coalition.deliverer_pos, coalition.builder_pos),
                round_index=round_index,
                duration=est_final.duration,
                builder_final_pos=est_final.builder_final_pos,
            )
        )

    return records, built_after


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
    cargo_heuristic: CargoPriorityHeuristic,
    max_rounds: int = 10_000,
    rng_seed: int = 0,
) -> RunResult:
    """Algorithm RUN-DYNAMIC-ROUNDS(G, C, R_d, R_b). rng_seed -- сид
    генератора случайности для Шага 4 (случайный выбор коалиции, оплачивающей
    постройку моста при коллизии); прогон с одинаковым rng_seed полностью
    воспроизводим."""

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
    rng = random.Random(rng_seed)

    T: List[ScheduleRecord] = []
    W_d_total = 0.0
    W_b_total = 0.0
    idle_total = 0.0
    round_index = 0

    while pending and round_index < max_rounds:
        cargo_snapshot = list(pending.values())
        records, built = select_round(
            G, cargo_snapshot, dict(deliverer_pos), dict(builder_pos), built,
            cargo_heuristic, round_index, rng,
        )
        if not records:
            break  # остаток недостижим/неразрешим при текущем состоянии роботов

        round_duration = max(r.duration for r in records)
        idle_total += sum(round_duration - r.duration for r in records)

        for r in records:
            T.append(r)
            W_d_total += r.W_d
            W_b_total += r.W_b

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
