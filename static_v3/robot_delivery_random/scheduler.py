"""
Шаги 0, 2, 3, 4, 5 модели static_v3 целиком (RUN-RANDOM-STATIC).

Модель СТАТИЧЕСКАЯ: пары формируются один раз (Шаг 2) и не пересобираются,
грузы распределяются между парами один раз (Шаг 3) и не перераспределяются.
Оба этих шага выполняются СЛУЧАЙНО -- ни расстояния между роботами, ни
стоимости доставки в них не участвуют. Тем самым static_v3 -- это baseline,
относительно которого измеряется, сколько на самом деле даёт осмысленное
правило: static_v2 отличается от него ровно двумя шагами (паросочетание по
минимальному расстоянию и распределение грузов по приоритетам), а всё
остальное -- граф, ROUTE-AND-COST, ESTIMATE-TASK-COST, Шаг 0, схема раундов
Шага 4 и правило конфликта за общий мост -- у них общее.

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
     мостов. Встречаться роботам не требуется.
  2. Если один и тот же ещё не построенный мост нужен нескольким парам этого
     раунда -- его строит СЛУЧАЙНО выбранная пара, а для остальных проезд по
     нему в этом (и всех последующих) раунде бесплатен.
  3. Барьер: длительность раунда -- по самой долгой доставке; разница идёт в
     idle_total.
  4. Позиции: доставщик -- в точке разгрузки, строитель -- в конце последнего
     ЛИЧНО построенного им моста (если сам ничего не строил -- остаётся на
     месте).

built -- ГЛОБАЛЬНОЕ состояние: однажды построенный мост никогда не строится
повторно, ни этой парой в следующем раунде, ни любой другой парой.

Шаг 5 (остановка). Когда у всех пар закончились грузы, суммируется общая
стоимость выполнения всех операций real = W_d_total + W_b_total. По постановке
это и есть эвристическая оценка работы модели; отдельной оценки "до
выполнения" (estimated_static из static_v2) у static_v3 нет, поскольку
распределение случайно и никакой предварительной оценки стоимости не считает.
Результат прогона -- план доставок (RunResult.T, по записи на доставку) плюс
эта итоговая стоимость.

ВОСПРОИЗВОДИМОСТЬ. Случайность входит в модель в трёх независимых местах:
паросочетание (Шаг 2), раздача грузов (Шаг 3) и выбор пары, оплачивающей общий
мост (Шаг 4). Каждое получает СВОЙ поток random.Random, порождённый из
rng_seed, -- иначе изменение числа грузов сдвигало бы и паросочетание, и
розыгрыш мостов, и сравнить два прогона было бы нельзя. Прогон с одинаковым
rng_seed полностью воспроизводим.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .assignment import Assignment, assign_cargos_random
from .costs import EstimateResult, estimate_task_cost
from .feasibility import check_feasibility
from .graph import EdgeKey, Graph, NodeId
from .model import Cargo, CargoId
from .pairing import Pair, PairId, RobotId, form_pairs_random


def _rng_streams(rng_seed: int) -> Tuple[random.Random, random.Random, random.Random]:
    """Три независимых потока случайности из одного сида: Шаг 2, Шаг 3, Шаг 4.

    Разводятся по разным сидам, а не берутся из одного генератора подряд,
    чтобы изменение одного шага (например, числа грузов, которое меняет
    количество вызовов на Шаге 3) не сдвигало розыгрыш остальных."""
    return (
        random.Random(rng_seed * 3 + 1),   # Шаг 2: паросочетание
        random.Random(rng_seed * 3 + 2),   # Шаг 3: раздача грузов
        random.Random(rng_seed * 3 + 3),   # Шаг 4: кто платит за общий мост
    )


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
    undelivered: List[CargoId] = field(default_factory=list)
    feasible: bool = True
    infeasibility_reason: Optional[str] = None

    @property
    def real(self) -> float:
        """Шаг 5: общая стоимость выполнения всех операций. По постановке это
        и есть эвристическая оценка работы модели."""
        return self.W_d_total + self.W_b_total


def run_round(
    G: Graph,
    round_index: int,
    tasks: Sequence[Tuple[Pair, Cargo]],
    deliverer_pos: Dict[RobotId, NodeId],
    builder_pos: Dict[RobotId, NodeId],
    built: Set[EdgeKey],
    rng: random.Random,
) -> Tuple[List[ScheduleRecord], Set[EdgeKey], List[CargoId]]:
    """Шаг 4, один раунд: параллельное выполнение по одной доставке на пару.

    Возвращает (записи плана, обновлённый built, грузы, которые выполнить не
    удалось). Последнее возможно только при недостижимости маршрута -- Шаг 0
    отсекает такие сценарии заранее, но не ловит случай, когда случайное
    паросочетание Шага 2 отдало груз паре с отрезанным строителем."""

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
                pos_round=(deliverer_pos[pair.deliverer_id], builder_pos[pair.builder_id]),
                builder_final_pos=est_final.builder_final_pos,
            )
        )

    return records, built_after, failed


def run_random_static(
    G: Graph,
    cargos: Sequence[Cargo],
    deliverer_positions: Sequence[NodeId],
    builder_positions: Sequence[NodeId],
    assignment_mode: str = "balanced",
    rng_seed: int = 0,
    max_rounds: int = 10_000,
) -> RunResult:
    """Algorithm RUN-RANDOM-STATIC(G, C, R_d, R_b) --- модель static_v3 целиком.

    assignment_mode -- как именно разыгрывается случайное распределение грузов
    (balanced / uniform, см. assignment.py). rng_seed -- сид всех трёх потоков
    случайности модели; прогон с одинаковым rng_seed полностью воспроизводим.
    """

    deliverer_pos: Dict[RobotId, NodeId] = {i: p for i, p in enumerate(deliverer_positions)}
    builder_pos: Dict[RobotId, NodeId] = {i: p for i, p in enumerate(builder_positions)}

    rng_pairing, rng_assign, rng_bridges = _rng_streams(rng_seed)

    # --- Шаг 0: проверка достижимости ---
    feasibility = check_feasibility(G, cargos, deliverer_pos, builder_pos)
    if not feasibility.ok:
        return RunResult(
            T=[], W_d_total=0.0, W_b_total=0.0, idle_total=0.0, n_rounds=0,
            all_delivered=False, delivered_cargo=set(),
            undelivered=[c.cargo_id for c in cargos],
            feasible=False, infeasibility_reason=feasibility.reason,
        )

    # --- Шаг 2: формирование пар (случайно) ---
    pairing = form_pairs_random(deliverer_positions, builder_positions, rng_pairing)
    if not pairing.pairs:
        return RunResult(
            T=[], W_d_total=0.0, W_b_total=0.0, idle_total=0.0, n_rounds=0,
            all_delivered=False, delivered_cargo=set(),
            undelivered=[c.cargo_id for c in cargos],
            feasible=False,
            infeasibility_reason=(
                "Не удалось сформировать ни одной пары доставщик-строитель "
                "(нет ни доставщиков, ни строителей). Задача невыполнима."
            ),
        )

    # --- Шаг 3: распределение грузов между парами (случайно, один раз) ---
    assignment: Assignment = assign_cargos_random(
        pairing.pairs, cargos, rng_assign, mode=assignment_mode,
    )

    # --- Шаг 4: выполнение доставок по раундам ---
    cargo_by_id = {c.cargo_id: c for c in cargos}
    queues: Dict[PairId, List[CargoId]] = {
        k: list(v) for k, v in assignment.per_pair.items()
    }
    built: Set[EdgeKey] = set()

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
            G, round_index, tasks, deliverer_pos, builder_pos, built, rng_bridges,
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

    # --- Шаг 5: остановка, итоговая стоимость всех операций ---
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
        undelivered=sorted(set(undelivered)),
        feasible=True,
        infeasibility_reason=None,
    )
