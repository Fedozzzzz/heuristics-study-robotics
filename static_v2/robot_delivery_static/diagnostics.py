"""
COMPUTE-STATIC-COST-BRACKET --- диагностическая вилка оценка/факт.

Насколько эвристическая оценка расходится с фактически исполненным
расписанием. Считаются три величины:

  real             -- то, что фактически произошло: сумма W_d + W_b по всем
                      выполненным доставкам (Шаг 5 постановки).
  estimated_static -- ОЦЕНКА СТАТИЧЕСКОЙ МОДЕЛИ: сумма W_T из таблицы Шага 3,
                      т.е. от НАЧАЛЬНЫХ позиций пары и с built = ∅, без учёта
                      промежуточных перемещений роботов. Это ровно та оценка,
                      по которой модель распределяла грузы, и именно про неё в
                      постановке сказано, что она "потенциально даёт завышенную
                      оценку стоимости выполнения всех операций".
  estimated_raw    -- контрольная величина: каждая доставка пересчитывается
                      изолированно от ФАКТИЧЕСКИХ позиций роботов на начало её
                      раунда, но с built = ∅ (забыв про все накопленные мосты).
                      Показывает, сколько стоило бы то же самое расписание, если
                      бы переправы не переиспользовались.

estimated_raw >= real --- гарантированное свойство модели (real видит
накопленный built и скидку за общие мосты внутри раунда, т.е. дополнительную
информацию, которая не может увеличить стоимость).

estimated_static такой гарантии НЕ имеет: начальная позиция пары может
оказаться дальше или ближе к грузу, чем фактическая позиция того момента,
когда пара за него берётся. Отрицательный gap_static на отдельных сценариях --
не ошибка, а измеряемое свойство статического распределения.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

from .costs import estimate_task_cost
from .graph import Graph
from .model import Cargo
from .scheduler import ScheduleRecord


@dataclass
class CostBracket:
    real: float
    estimated_static: float
    estimated_raw: float
    gap_static: float
    gap_raw: float
    # разбивка по раундам -- для графика "оценка vs факт по ходу выполнения"
    per_round_real: Dict[int, float] = field(default_factory=dict)
    per_round_estimated_static: Dict[int, float] = field(default_factory=dict)
    per_round_estimated_raw: Dict[int, float] = field(default_factory=dict)


def compute_static_cost_bracket(
    G: Graph, T: Sequence[ScheduleRecord], cargos: Dict[int, Cargo]
) -> CostBracket:
    """Algorithm COMPUTE-STATIC-COST-BRACKET(G, T, C)."""

    if not T:
        return CostBracket(0.0, 0.0, 0.0, 0.0, 0.0)

    real = 0.0
    estimated_static = 0.0
    estimated_raw = 0.0
    per_round_real: Dict[int, float] = {}
    per_round_estimated_static: Dict[int, float] = {}
    per_round_estimated_raw: Dict[int, float] = {}

    for r in T:
        cargo = cargos[r.cargo_id]
        deliverer_pos, builder_pos = r.pos_round
        result = estimate_task_cost(
            G, cargo.start, cargo.finish, deliverer_pos, builder_pos, built=set()
        )
        if result is None:
            raise RuntimeError(
                f"pos_round несовместим с исполнением (пара={r.pair_id}, груз={r.cargo_id})"
            )

        r_real = r.W_d + r.W_b
        real += r_real
        estimated_static += r.W_T_static
        estimated_raw += result.W_T

        per_round_real[r.round_index] = per_round_real.get(r.round_index, 0.0) + r_real
        per_round_estimated_static[r.round_index] = (
            per_round_estimated_static.get(r.round_index, 0.0) + r.W_T_static
        )
        per_round_estimated_raw[r.round_index] = (
            per_round_estimated_raw.get(r.round_index, 0.0) + result.W_T
        )

    gap_static = 100.0 * (estimated_static / real - 1.0) if real > 1e-9 else 0.0
    gap_raw = 100.0 * (estimated_raw / real - 1.0) if real > 1e-9 else 0.0

    return CostBracket(
        real=real,
        estimated_static=estimated_static,
        estimated_raw=estimated_raw,
        gap_static=gap_static,
        gap_raw=gap_raw,
        per_round_real=per_round_real,
        per_round_estimated_static=per_round_estimated_static,
        per_round_estimated_raw=per_round_estimated_raw,
    )
