"""
Алгоритм 2 (новая версия): назначение пар (доставчик, строитель) на грузы
ФИКСИРУЕТСЯ ДО ЗАПУСКА МОДЕЛИ - каждый груз имеет заранее заданное поле
cargo.assigned_pair, указывающее, какая пара его обслуживает. Внутри одной
пары, если ей назначено несколько грузов, порядок выполнения определяется
приоритетом p(c_i) (Алгоритм 1), который для очереди ОДНОЙ пары пересчитывается
динамически по её текущей позиции после каждой выполненной задачи.

Это устраняет нестабильность предыдущей версии, где предназначение груза
паре определялось на лету (через "ближайшую свободную пару") и могло
неожиданно меняться в зависимости от порядка освобождения пар.
"""

import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from delivery_model import IslandGraph, Cargo, Pair, TaskResult
from algorithms_1_3 import compute_priority, find_route_and_bridges


@dataclass
class ScheduleEntry:
    cargo_id: str
    pair_id: str
    result: TaskResult
    start_time: float
    end_time: float


@dataclass
class ScheduleOutcome:
    """Итог одного полного прогона Алгоритма 2 при фиксированном L."""
    L: float
    all_delivered: bool
    schedule: List[ScheduleEntry] = field(default_factory=list)
    W_d_total: float = 0.0
    W_b_total: float = 0.0


def validate_assignment(cargos: List[Cargo], pairs: List[Pair]):
    """Проверка, что у каждого груза задано существующее назначение паре."""
    pair_ids = {p.id for p in pairs}
    for c in cargos:
        if c.assigned_pair is None:
            raise ValueError(f"Груз {c.id} не имеет назначенной пары "
                              f"(задайте cargo.assigned_pair до запуска модели)")
        if c.assigned_pair not in pair_ids:
            raise ValueError(f"Груз {c.id} назначен паре {c.assigned_pair}, "
                              f"которой не существует среди {pair_ids}")


def run_scheduling(env: IslandGraph, cargos: List[Cargo], pairs: List[Pair],
                    L: float, k_alternatives: int = 12) -> ScheduleOutcome:
    """
    Главный цикл: для каждой пары своя очередь грузов (предопределённая через
    cargo.assigned_pair). Очередь упорядочивается по приоритету p(c_i),
    который пересчитывается перед выбором каждого следующего груза в очереди
    ЭТОЙ пары - по её текущей (уже изменившейся после доставок) позиции.
    Пары работают независимо друг от друга и параллельно по времени.
    """
    validate_assignment(cargos, pairs)

    cargos = copy.deepcopy(cargos)
    pairs = copy.deepcopy(pairs)
    pairs_by_id = {p.id: p for p in pairs}

    outcome = ScheduleOutcome(L=L, all_delivered=False)

    # независимая очередь грузов для каждой пары
    pending_by_pair: Dict[str, List[Cargo]] = {p.id: [] for p in pairs}
    for c in cargos:
        pending_by_pair[c.assigned_pair].append(c)

    for pair_id, pair in pairs_by_id.items():
        pending = pending_by_pair[pair_id]
        current_time = 0.0

        while pending:
            # приоритет пересчитывается по ТЕКУЩЕЙ позиции этой пары (динамически)
            scored = [
                (compute_priority(c, env, [pair.deliverer_pos], [pair.builder_pos]), c)
                for c in pending
            ]
            scored.sort(key=lambda t: t[0], reverse=True)
            cargo = scored[0][1]

            result = find_route_and_bridges(env, cargo, pair, L, k_alternatives=k_alternatives)

            if not result.feasible:
                # при данном L задачу для этого груза выполнить нельзя -> L недопустимо
                outcome.all_delivered = False
                return outcome

            pair.deliverer_pos = cargo.v_finish
            if result.bridges:
                pair.builder_pos = result.bridges[-1][1]
                for (u, v) in result.bridges:
                    pair.add_built_bridge(u, v)
            else:
                pair.builder_pos = cargo.v_finish

            cargo.delivered = True
            pending = [c for c in pending if c.id != cargo.id]

            end_time = current_time + result.duration
            outcome.schedule.append(ScheduleEntry(
                cargo_id=cargo.id, pair_id=pair_id, result=result,
                start_time=current_time, end_time=end_time,
            ))
            outcome.W_d_total += result.W_d
            outcome.W_b_total += result.W_b
            current_time = end_time

    outcome.all_delivered = all(c.delivered for c in cargos)
    return outcome
