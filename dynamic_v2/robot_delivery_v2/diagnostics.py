"""
2.5. COMPUTE-DYNAMIC-COST-BRACKET --- диагностическая вилка оценка/факт.

Оценивает, насколько эвристическая оценка стоимости расходится с фактически
исполненной стоимостью расписания. Строит real, estimated_raw и метрику
разрыва gap_raw (в процентах).

estimated_raw -- полный пересчёт КАЖДОЙ доставки изолированно (от pos_round,
но забыв про ВСЕ накопленные мосты, built = ∅) -- верхняя оценка "что было бы,
если бы каждая коалиция строила с нуля, не переиспользуя ничьи переправы".
estimated_raw >= real -- гарантированное свойство модели: real использует
накопленный за все раунды built плюс скидку за внутрираундовые коллизии
(Шаг 4), т.е. дополнительная информация, которая не может увеличить стоимость.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

from .costs import estimate_task_cost
from .graph import Graph
from .scheduler import Cargo, ScheduleRecord


@dataclass
class CostBracket:
    real: float
    estimated_raw: float
    gap_raw: float
    # разбивка по раундам -- для графиков "оценка vs факт по ходу выполнения"
    per_round_real: Dict[int, float] = field(default_factory=dict)
    per_round_estimated_raw: Dict[int, float] = field(default_factory=dict)


def compute_dynamic_cost_bracket(
    G: Graph, T: Sequence[ScheduleRecord], cargos: Dict[int, Cargo]
) -> CostBracket:
    """Algorithm COMPUTE-DYNAMIC-COST-BRACKET(G, T, U)."""

    if not T:
        return CostBracket(real=0.0, estimated_raw=0.0, gap_raw=0.0)

    # --- real: то, что фактически произошло ---
    real = sum(r.W_d + r.W_b for r in T)

    # --- estimated_raw: полный трёхшаговый пересчёт КАЖДОЙ доставки
    #     изолированно -- от pos_round, но забыв про ВСЕ накопленные мосты
    #     (built = ∅) ---
    estimated_raw = 0.0
    per_round_real: Dict[int, float] = {}
    per_round_estimated_raw: Dict[int, float] = {}

    for r in T:
        cargo = cargos[r.cargo_id]
        deliverer_pos, builder_pos = r.pos_round
        result = estimate_task_cost(G, cargo.start, cargo.finish, deliverer_pos, builder_pos, built=set())
        if result is None:
            raise RuntimeError(
                f"pos_round несовместим с исполнением (запись coalition={r.coalition_id}, cargo={r.cargo_id})"
            )
        estimated_raw += result.W_T

        per_round_real[r.round_index] = per_round_real.get(r.round_index, 0.0) + r.W_d + r.W_b
        per_round_estimated_raw[r.round_index] = (
            per_round_estimated_raw.get(r.round_index, 0.0) + result.W_T
        )

    gap_raw = 100.0 * (estimated_raw / real - 1.0) if real > 1e-9 else 0.0

    return CostBracket(
        real=real,
        estimated_raw=estimated_raw,
        gap_raw=gap_raw,
        per_round_real=per_round_real,
        per_round_estimated_raw=per_round_estimated_raw,
    )
