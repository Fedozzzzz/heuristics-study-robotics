"""
Шаг 2. Формирование пар роботов (FORM-PAIRS).

Пары формируются ОДИН РАЗ, до начала выполнения, по принципу минимального
расстояния между доставщиком и строителем: обоим роботам так или иначе
придётся добираться до точки погрузки, и минимальное расстояние между ними
означает, что для старта доставки им в сумме придётся преодолевать меньшее
расстояние.

Реализация -- жадное паросочетание по возрастанию расстояний: все пары
(доставщик, строитель) сортируются по расстоянию в графе, список проходится
один раз, пара берётся, если ОБА робота ещё свободны. Это буквальное прочтение
"по принципу минимального расстояния" (в отличие от оптимального по сумме
паросочетания -- венгерского алгоритма, который здесь намеренно не
используется).

Расстояние считается по графу G с учётом всех переправ, ДОСТУПНЫХ к возведению
(проходимо всё, кроме IMPOSSIBLE), но БЕЗ учёта стоимости их постройки -- так
же, как критерий выбора доставщика в dynamic_v2 (Шаг 2). Пары роботов, между
которыми пути нет вообще (разные компоненты связности даже с учётом
строительства), в паросочетание не допускаются.

Роботы, оставшиеся без пары (неравное число доставщиков и строителей либо
недостижимость), в работе модели не участвуют -- они возвращаются в
PairingResult.unpaired_* для диагностики.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .graph import EdgeKind, Graph, NodeId

RobotId = int
PairId = int


@dataclass
class Pair:
    """Виртуальная пара (коалиция) из 1 доставщика и 1 строителя.

    deliverer_pos/builder_pos -- НАЧАЛЬНЫЕ позиции роботов (на момент
    формирования пары). Именно от них считается статическая оценка стоимости
    доставки на Шаге 3; фактические позиции по ходу исполнения ведёт
    scheduler.py отдельно."""

    pair_id: PairId
    deliverer_id: RobotId
    builder_id: RobotId
    deliverer_pos: NodeId
    builder_pos: NodeId
    distance: float  # расстояние доставщик-строитель, по которому пара была выбрана


@dataclass
class PairingResult:
    pairs: List[Pair] = field(default_factory=list)
    unpaired_deliverers: List[RobotId] = field(default_factory=list)
    unpaired_builders: List[RobotId] = field(default_factory=list)


def _passable(e) -> bool:
    return e.kind != EdgeKind.IMPOSSIBLE


def form_pairs(
    G: Graph,
    deliverer_positions: Sequence[NodeId],
    builder_positions: Sequence[NodeId],
) -> PairingResult:
    """Algorithm FORM-PAIRS(G, R_d, R_b) --- Шаг 2.

    Возвращает список пар (по одному доставщику и строителю в каждой) и
    роботов, которым пары не нашлось. Порядок пар (и их pair_id) --
    в порядке формирования, т.е. по возрастанию расстояния внутри пары."""

    candidates: List[Tuple[float, RobotId, RobotId]] = []
    for d_id, d_pos in enumerate(deliverer_positions):
        for b_id, b_pos in enumerate(builder_positions):
            dist: Optional[float] = G.shortest_distance(d_pos, b_pos, _passable)
            if dist is None:
                continue  # робот-строитель физически недостижим для доставщика
            candidates.append((dist, d_id, b_id))

    # tie-break по (d_id, b_id) -- воспроизводимость при равных расстояниях
    candidates.sort()

    free_d: Set[RobotId] = set(range(len(deliverer_positions)))
    free_b: Set[RobotId] = set(range(len(builder_positions)))
    pairs: List[Pair] = []

    for dist, d_id, b_id in candidates:
        if d_id not in free_d or b_id not in free_b:
            continue
        free_d.discard(d_id)
        free_b.discard(b_id)
        pairs.append(
            Pair(
                pair_id=len(pairs),
                deliverer_id=d_id,
                builder_id=b_id,
                deliverer_pos=deliverer_positions[d_id],
                builder_pos=builder_positions[b_id],
                distance=dist,
            )
        )

    return PairingResult(
        pairs=pairs,
        unpaired_deliverers=sorted(free_d),
        unpaired_builders=sorted(free_b),
    )


def pairs_by_id(pairs: Sequence[Pair]) -> Dict[PairId, Pair]:
    return {p.pair_id: p for p in pairs}
