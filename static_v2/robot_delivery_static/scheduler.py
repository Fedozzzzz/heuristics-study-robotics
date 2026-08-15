"""
Шаги 0, 2, 3, 4, 5 статической модели целиком (RUN-STATIC).

Отличие от dynamic_v2: пары формируются ЗАРАНЕЕ и не пересобираются
(pairing.py, Шаг 2), грузы распределяются между парами ОДИН РАЗ в самом начале
(assignment.py, Шаг 3) -- при оценке приоритета и распределении не учитываются
промежуточные перемещения роботов, что и даёт потенциально завышенную оценку
стоимости выполнения всех операций. Динамическая модель, наоборот,
пересчитывает приоритеты и собирает коалиции заново в начале каждого раунда.

Шаг 4 (выполнение доставки, здесь). Все назначенные доставки выполняются
параллельно, по раундам: в раунде r каждая пара везёт r-й груз из своего
списка (сперва все доставили первый груз, затем второй, и так далее, пока у
всех пар не кончатся грузы). Пары с более коротким списком в поздних раундах
просто не участвуют.

Внутри раунда для каждой пары:
  1. ESTIMATE-TASK-COST от ФАКТИЧЕСКИХ позиций её роботов с учётом
     ГЛОБАЛЬНОГО built (все уже возведённые кем угодно переправы). Маршруты --
     минимальные в графе, мосты выбираются жадно по минимальной длине
     (см. costs.route_and_cost); строитель строит только те переправы,
     которые нужны для доезда доставщика до точки погрузки и до точки
     разгрузки, плюс те, что нужны ему самому, чтобы добраться до этих
     мостов.
  2. Если один и тот же ещё не построенный мост нужен нескольким парам этого
     раунда -- его строит СЛУЧАЙНО выбранная пара, а для остальных проезд по
     нему в этом (и всех последующих) раунде бесплатен. Воспроизводимость --
     через rng_seed.
  3. Барьер: длительность раунда -- по самой долгой доставке; разница идёт в
     idle_total.
  4. Позиции: доставщик -- в точке разгрузки, строитель -- в конце последнего
     ЛИЧНО построенного им моста (если сам ничего не строил -- остаётся на
     месте).

Шаг 5 (остановка). Когда у всех пар закончились грузы, суммируется общая
стоимость выполнения всех операций W_d_total + W_b_total (это и есть
эвристическая оценка работы модели); план доставок -- список ScheduleRecord.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .assignment import Assignment, CostTable, assign_cargos, build_cost_table
from .costs import EstimateResult, estimate_task_cost
from .feasibility import check_feasibility
from .graph import EdgeKey, Graph, NodeId
from .model import Cargo, CargoId
from .pairing import Pair, PairId, RobotId, form_pairs
from .priority import CargoPriorityHeuristic


@dataclass
class ScheduleRecord:
    """Одна выполненная доставка (строка плана доставок)."""

    pair_id: PairId
    deliverer_id: RobotId
    builder_id: RobotId
    cargo_id: CargoId
    round_index: int
    path_nodes: List[NodeId]
    bridges: Set[EdgeKey]          # все мосты операции (в т.ч. оплаченные другой парой)
    won_bridges: Set[EdgeKey]      # мосты, которые ЛИЧНО построил строитель этой пары
    W_d: float
    W_b: float
    duration: float                # длительность именно этой доставки (без барьера)
    p_rank: float                  # приоритет груза (Шаг 3), по которому он был назначен
    W_T_static: float              # оценка Шага 3 (от НАЧАЛЬНЫХ позиций пары, built = ∅)
    pos_round: Tuple[NodeId, NodeId]     # (deliverer_pos, builder_pos) на начало раунда
    builder_final_pos: Optional[NodeId]  # None -- если строитель ничего не строил


@dataclass
class RunResult:
    T: List[ScheduleRecord]
    W_d_total: float
    W_b_total: float
    idle_total: float
    n_rounds: int
    all_delivered: bool
    delivered_cargo: Set[CargoId]
    pairs: List[Pair] = field(default_factory=list)
    plan: Dict[PairId, List[CargoId]] = field(default_factory=dict)
    estimated_static: float = 0.0     # сумма оценок Шага 3 по всем ВЫПОЛНЕННЫМ доставкам
    undelivered: List[CargoId] = field(default_factory=list)
    feasible: bool = True
    infeasibility_reason: Optional[str] = None

    @property
    def real(self) -> float:
        """Шаг 5: общая стоимость выполнения всех операций."""
        return self.W_d_total + self.W_b_total


def run_round(
    G: Graph,
    round_index: int,
    tasks: Sequence[Tuple[Pair, Cargo]],
    deliverer_pos: Dict[RobotId, NodeId],
    builder_pos: Dict[RobotId, NodeId],
    built: Set[EdgeKey],
    assignment: Assignment,
    rng: random.Random,
) -> Tuple[List[ScheduleRecord], Set[EdgeKey], List[CargoId]]:
    """Шаг 4, один раунд: параллельное выполнение по одной доставке на пару.

    Возвращает (записи плана, обновлённый built, грузы, которые выполнить не
    удалось). Последнее возможно только при недостижимости маршрута -- Шаг 0
    такие сценарии отсекает заранее, поэтому в норме список пуст."""

    estimates: List[Tuple[Pair, Cargo, EstimateResult]] = []
    failed: List[CargoId] = []
    for pair, cargo in tasks:
        est = estimate_task_cost(
            G, cargo.start, cargo.finish,
            deliverer_pos=deliverer_pos[pair.deliverer_id],
            builder_pos=builder_pos[pair.builder_id],
            built=built,
        )
        if est is None:
            failed.append(cargo.cargo_id)
            continue
        estimates.append((pair, cargo, est))

    if not estimates:
        return [], set(built), failed

    # --- один и тот же ещё не построенный мост у нескольких пар раунда:
    #     платит случайно выбранная пара, остальным проезд бесплатен ---
    edge_users: Dict[EdgeKey, List[int]] = {}
    for idx, (_pair, _cargo, est) in enumerate(estimates):
        for ek in est.bridges:
            if ek in built:
                continue
            edge_users.setdefault(ek, []).append(idx)

    winner_of_edge: Dict[EdgeKey, int] = {}
    for ek, users in edge_users.items():
        winner_of_edge[ek] = users[0] if len(users) == 1 else rng.choice(users)

    built_after: Set[EdgeKey] = set(built)
    records: List[ScheduleRecord] = []

    for idx, (pair, cargo, est_initial) in enumerate(estimates):
        free_for_me = {
            ek for ek in est_initial.bridges
            if ek not in built and winner_of_edge.get(ek, idx) != idx
        }
        if free_for_me:
            est_final = estimate_task_cost(
                G, cargo.start, cargo.finish,
                deliverer_pos=deliverer_pos[pair.deliverer_id],
                builder_pos=builder_pos[pair.builder_id],
                built=built | free_for_me,
            )
            assert est_final is not None, "бесплатный проезд не может сделать маршрут недостижимым"
        else:
            est_final = est_initial

        won_bridges = {
            ek for ek in est_final.bridges
            if ek not in built and winner_of_edge.get(ek, idx) == idx
        }
        built_after |= set(est_final.bridges)

        records.append(
            ScheduleRecord(
                pair_id=pair.pair_id,
                deliverer_id=pair.deliverer_id,
                builder_id=pair.builder_id,
                cargo_id=cargo.cargo_id,
                round_index=round_index,
                path_nodes=est_final.path_nodes,
                bridges=set(est_final.bridges),
                won_bridges=won_bridges,
                W_d=est_final.W_d,
                W_b=est_final.W_b,
                duration=est_final.duration,
                p_rank=assignment.priority.get(cargo.cargo_id, 0.0),
                W_T_static=assignment.static_estimate.get(cargo.cargo_id, 0.0),
                pos_round=(deliverer_pos[pair.deliverer_id], builder_pos[pair.builder_id]),
                builder_final_pos=est_final.builder_final_pos,
            )
        )

    return records, built_after, failed


def run_static(
    G: Graph,
    cargos: Sequence[Cargo],
    deliverer_positions: Sequence[NodeId],
    builder_positions: Sequence[NodeId],
    cargo_heuristic: CargoPriorityHeuristic,
    assignment_mode: str = "literal",
    balance: str = "load",
    lpt_size: str = "min",
    lpt_rule: str = "load",
    rng_seed: int = 0,
    max_rounds: int = 10_000,
) -> RunResult:
    """Algorithm RUN-STATIC(G, C, R_d, R_b) --- статическая модель целиком.

    assignment_mode / balance -- см. assignment.py (какой паре достаётся груз
    и ограничивается ли её загрузка); lpt_size / lpt_rule -- параметры режима
    назначения "lpt" (см. assignment.assign_lpt), в остальных режимах не
    используются. rng_seed -- сид случайного выбора пары, оплачивающей общий
    мост (Шаг 4); прогон с одинаковым rng_seed полностью воспроизводим."""

    deliverer_pos: Dict[RobotId, NodeId] = {i: p for i, p in enumerate(deliverer_positions)}
    builder_pos: Dict[RobotId, NodeId] = {i: p for i, p in enumerate(builder_positions)}

    # --- Шаг 0: проверка достижимости ---
    feasibility = check_feasibility(G, cargos, deliverer_pos, builder_pos)
    if not feasibility.ok:
        return RunResult(
            T=[], W_d_total=0.0, W_b_total=0.0, idle_total=0.0, n_rounds=0,
            all_delivered=False, delivered_cargo=set(),
            undelivered=[c.cargo_id for c in cargos],
            feasible=False, infeasibility_reason=feasibility.reason,
        )

    # --- Шаг 2: формирование пар ---
    pairing = form_pairs(G, deliverer_positions, builder_positions)
    if not pairing.pairs:
        return RunResult(
            T=[], W_d_total=0.0, W_b_total=0.0, idle_total=0.0, n_rounds=0,
            all_delivered=False, delivered_cargo=set(),
            undelivered=[c.cargo_id for c in cargos],
            feasible=False,
            infeasibility_reason=(
                "Не удалось сформировать ни одной пары доставщик-строитель "
                "(нет достижимых друг для друга роботов). Задача невыполнима."
            ),
        )

    # --- Шаг 3: распределение грузов между парами (один раз, в самом начале) ---
    table: CostTable = build_cost_table(G, pairing.pairs, cargos)
    assignment = assign_cargos(
        table, pairing.pairs, cargos, cargo_heuristic,
        mode=assignment_mode, balance=balance,
        lpt_size=lpt_size, lpt_rule=lpt_rule,
    )

    # --- Шаг 4: выполнение доставок по раундам ---
    cargo_by_id = {c.cargo_id: c for c in cargos}
    queues: Dict[PairId, List[CargoId]] = {
        k: list(v) for k, v in assignment.per_pair.items()
    }
    built: Set[EdgeKey] = set()
    rng = random.Random(rng_seed)

    T: List[ScheduleRecord] = []
    W_d_total = 0.0
    W_b_total = 0.0
    idle_total = 0.0
    undelivered: List[CargoId] = list(assignment.unassigned)
    round_index = 0

    n_rounds_needed = max((len(q) for q in queues.values()), default=0)
    while round_index < n_rounds_needed and round_index < max_rounds:
        tasks: List[Tuple[Pair, Cargo]] = [
            (pair, cargo_by_id[queues[pair.pair_id][round_index]])
            for pair in pairing.pairs
            if len(queues[pair.pair_id]) > round_index
        ]
        records, built, failed = run_round(
            G, round_index, tasks, deliverer_pos, builder_pos, built, assignment, rng,
        )
        undelivered.extend(failed)

        if records:
            round_duration = max(r.duration for r in records)
            idle_total += sum(round_duration - r.duration for r in records)

            for r in records:
                T.append(r)
                W_d_total += r.W_d
                W_b_total += r.W_b

                # Шаг 4.4: позиция доставщика -- точка разгрузки; позиция
                # строителя -- конец последнего ЛИЧНО построенного им моста
                deliverer_pos[r.deliverer_id] = cargo_by_id[r.cargo_id].finish
                if r.won_bridges and r.builder_final_pos is not None:
                    builder_pos[r.builder_id] = r.builder_final_pos

        round_index += 1

    delivered = {r.cargo_id for r in T}

    # --- Шаг 5: остановка, итоговые суммы ---
    return RunResult(
        T=T,
        W_d_total=W_d_total,
        W_b_total=W_b_total,
        idle_total=idle_total,
        n_rounds=round_index,
        all_delivered=(len(delivered) == len(cargos)),
        delivered_cargo=delivered,
        pairs=list(pairing.pairs),
        plan={k: list(v) for k, v in assignment.per_pair.items()},
        estimated_static=sum(r.W_T_static for r in T),
        undelivered=sorted(set(undelivered)),
        feasible=True,
        infeasibility_reason=None,
    )
