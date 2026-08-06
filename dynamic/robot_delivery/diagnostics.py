"""
2.5. COMPUTE-DYNAMIC-COST-BRACKET --- диагностическая вилка оценка/факт.

Оценивает, насколько эвристическая оценка стоимости, использованная для
приоритезации, расходится с фактически исполненной стоимостью расписания.
Строит real, estimated_raw, estimated_pool, estimated_precollision,
estimated_prognosis и метрики разрыва gap_raw / gap_pool / gap_precollision /
gap_prognosis (в процентах).

estimated_precollision в данной реализации считается через W_T_initial
(оценка ESTIMATE-TASK-COST на момент построения таблицы приоритетов, ДО
скидки за внутрираундовое резервирование мостов) -- а не через "сырой" ранг
p_rank, потому что p_rank для геометрических/дистанционных эвристик не
находится в тех же единицах, что реальная стоимость, и несравним напрямую.
W_T_initial, напротив, всегда является полноценной оценкой стоимости в тех
же единицах, что real, независимо от того, какая эвристика использовалась
для ранжирования -- это и есть искомая "цена коллизий", обобщённая на любую
эвристику.

estimated_prognosis -- прогнозируемая эвристическая оценка стоимости прямой
эвристики (p = W_T, "чем дороже доставка -- тем первее её нужно выполнить"),
пересчитываемая на каждом раунде для каждой пары (коалиции): мосты, уже
построенные на предыдущих раундах, учитываются как бесплатные (по стоимости
проезда w_E), а мосты, которые ещё НЕОБХОДИМО построить для этого маршрута,
учитываются по полной стоимости (постройка w_build + проезд w_E) -- то есть
ровно так же, как считается сама таблица приоритетов на Шаге 2. Это то же
значение, что уже сохраняется как W_T_initial (см. estimated_precollision
ниже) -- estimated_prognosis суммирует его для отдельного графика "прогноз vs
факт по раундам". Отсюда estimated_prognosis >= real (как и
estimated_precollision): реальная стоимость может оказаться ниже за счёт
скидки при разрешении коллизий за мосты внутри раунда (Шаг 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .costs import estimate_task_cost
from .graph import EdgeKey, Graph
from .scheduler import Cargo, ScheduleRecord


@dataclass
class CostBracket:
    real: float
    estimated_raw: float
    estimated_pool: float
    estimated_precollision: float
    estimated_prognosis: float
    gap_raw: float
    gap_pool: float
    gap_precollision: float
    gap_prognosis: float
    # разбивка по раундам -- для графиков "оценка vs факт по ходу выполнения"
    per_round_real: Dict[int, float] = field(default_factory=dict)
    per_round_estimated_raw: Dict[int, float] = field(default_factory=dict)
    per_round_estimated_prognosis: Dict[int, float] = field(default_factory=dict)


def compute_dynamic_cost_bracket(
    G: Graph, T: Sequence[ScheduleRecord], cargos: Dict[int, Cargo]
) -> CostBracket:
    """Algorithm COMPUTE-DYNAMIC-COST-BRACKET(G, T, U)."""

    if not T:
        return CostBracket(
            real=0.0, estimated_raw=0.0, estimated_pool=0.0, estimated_precollision=0.0,
            estimated_prognosis=0.0, gap_raw=0.0, gap_pool=0.0, gap_precollision=0.0, gap_prognosis=0.0,
        )

    # --- real: то, что фактически произошло ---
    real = sum(r.W_d + r.W_b for r in T)

    # --- estimated_precollision: сумма оценок ДО скидки за резервирование ---
    estimated_precollision = sum(r.W_T_initial for r in T)

    # --- estimated_prognosis: прямая эвристика (p = W_T), пересчитанная на
    #     каждом раунде для каждой пары -- уже построенные на предыдущих
    #     раундах мосты бесплатны, мосты, которые ещё нужно построить,
    #     учитываются по полной стоимости. Это ровно W_T_initial (значение
    #     уже посчитано в select_round на Шаге 2, пересчитывать не нужно) ---
    estimated_prognosis = sum(r.W_T_initial for r in T)
    per_round_estimated_prognosis: Dict[int, float] = {}
    for r in T:
        per_round_estimated_prognosis[r.round_index] = (
            per_round_estimated_prognosis.get(r.round_index, 0.0) + r.W_T_initial
        )

    # --- estimated_raw: полный трёхшаговый пересчёт КАЖДОЙ доставки
    #     изолированно -- от pos_round, но забыв про ВСЕ накопленные мосты
    #     (built = ∅) ---
    estimated_raw = 0.0
    raw_Wd: List[float] = []
    raw_Wb: List[float] = []
    raw_bridges: List[set] = []

    for r in T:
        cargo = cargos[r.cargo_id]
        deliverer_pos, builder_pos = r.pos_round
        result = estimate_task_cost(G, cargo.start, cargo.finish, deliverer_pos, builder_pos, built=set())
        if result is None:
            raise RuntimeError(
                f"pos_round несовместим с исполнением (запись coalition={r.coalition_id}, cargo={r.cargo_id})"
            )
        estimated_raw += result.W_T
        raw_Wd.append(result.W_d)
        raw_Wb.append(result.W_b)
        raw_bridges.append(set(result.bridges))

    # --- estimated_pool: коррекция за переиспользование мостов, ГЛОБАЛЬНАЯ
    #     по всему T ---
    edge_count: Dict[EdgeKey, int] = {}
    for bridges in raw_bridges:
        for e in bridges:
            edge_count[e] = edge_count.get(e, 0) + 1

    total_Wb_raw = sum(raw_Wb)
    reuse_discount = sum((c - 1) * G.edges[e].w_build for e, c in edge_count.items())
    W_b_pool_total = total_Wb_raw - reuse_discount

    estimated_pool = 0.0
    for wd_i, wb_i in zip(raw_Wd, raw_Wb):
        share = wb_i / total_Wb_raw if total_Wb_raw > 1e-9 else 0.0
        Wb_hat_i = W_b_pool_total * share
        estimated_pool += wd_i + Wb_hat_i

    def pct_gap(estimated: float) -> float:
        return 100.0 * (estimated / real - 1.0) if real > 1e-9 else 0.0

    gap_raw = pct_gap(estimated_raw)
    gap_pool = pct_gap(estimated_pool)
    gap_precollision = pct_gap(estimated_precollision)
    gap_prognosis = pct_gap(estimated_prognosis)

    # --- разбивка по раундам (для построения графика по ходу выполнения) ---
    per_round_real: Dict[int, float] = {}
    per_round_estimated_raw: Dict[int, float] = {}
    for r, wd_raw, wb_raw in zip(T, raw_Wd, raw_Wb):
        per_round_real[r.round_index] = per_round_real.get(r.round_index, 0.0) + r.W_d + r.W_b
        per_round_estimated_raw[r.round_index] = per_round_estimated_raw.get(r.round_index, 0.0) + wd_raw + wb_raw

    return CostBracket(
        real=real,
        estimated_raw=estimated_raw,
        estimated_pool=estimated_pool,
        estimated_precollision=estimated_precollision,
        estimated_prognosis=estimated_prognosis,
        gap_raw=gap_raw,
        gap_pool=gap_pool,
        gap_precollision=gap_precollision,
        gap_prognosis=gap_prognosis,
        per_round_real=per_round_real,
        per_round_estimated_raw=per_round_estimated_raw,
        per_round_estimated_prognosis=per_round_estimated_prognosis,
    )
