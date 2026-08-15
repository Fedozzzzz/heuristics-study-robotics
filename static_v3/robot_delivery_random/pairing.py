"""
Шаг 2. Формирование пар роботов (FORM-PAIRS-RANDOM).

    "Пары формируются случайным образом. Результат алгоритма -- виртуальные
     пары роботов, состоящие из 1 доставщика и 1 строителя."

Пары формируются ОДИН РАЗ, до начала выполнения, и не пересобираются до конца
прогона (в этом static_v3 совпадает со static_v2 и отличается от dynamic_v2,
где коалиция собирается заново каждый раунд под выбранный груз).

Реализация -- случайное совершенное паросочетание: список доставщиков и список
строителей независимо перемешиваются и склеиваются позиционно (zip). Это даёт
равномерное распределение на всех паросочетаниях: каждое из min(|R_d|, |R_b|)!
сопоставлений равновероятно. Роботы, оставшиеся без пары (при неравном числе
доставщиков и строителей), в работе модели не участвуют и возвращаются в
PairingResult.unpaired_* для диагностики.

ОТЛИЧИЕ ОТ static_v2. Там Шаг 2 -- жадное паросочетание по ВОЗРАСТАНИЮ
расстояния "доставщик-строитель" (Дейкстра по всем переправам, кроме
IMPOSSIBLE): пары подбираются так, чтобы роботам в сумме пришлось меньше
добираться до старта. Здесь расстояние не считается вообще -- граф в этот шаг
даже не передаётся. Разница между двумя правилами и есть то, что измеряет
experiments/compare_static_v3_vs_static_v2.py.

О ДОСТИЖИМОСТИ. static_v2 не допускал в паросочетание роботов, между которыми
пути нет вообще (разные компоненты связности даже с учётом строительства);
случайное паросочетание такую пару составить может. Это НЕ обрабатывается
специально: пара, чей строитель отрезан от точек доставки, просто не сможет
выполнить назначенные ей доставки, груз останется в RunResult.undelivered, и
прогон завершится с all_delivered = False. Фильтрация по достижимости была бы
уже не случайным выбором, а эвристикой, то есть другой моделью.

На сценариях scenario.generate_scenario этот случай не возникает: базовый граф
островов там строится на MST и связен, рёбер IMPOSSIBLE в нём нет, поэтому все
роботы всегда лежат в одной компоненте G'.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .graph import NodeId

RobotId = int
PairId = int


@dataclass
class Pair:
    """Виртуальная пара (коалиция) из 1 доставщика и 1 строителя.

    deliverer_pos/builder_pos -- НАЧАЛЬНЫЕ позиции роботов на момент
    формирования пары. В static_v3 они нужны только для диагностики и как
    стартовая точка Шага 4: никакой предварительной оценки стоимости от них
    не считается (в отличие от static_v2, где именно от них строилась таблица
    "пара x груз"). Фактические позиции по ходу исполнения ведёт scheduler.py
    отдельно."""

    pair_id: PairId
    deliverer_id: RobotId
    builder_id: RobotId
    deliverer_pos: NodeId
    builder_pos: NodeId


@dataclass
class PairingResult:
    pairs: List[Pair] = field(default_factory=list)
    unpaired_deliverers: List[RobotId] = field(default_factory=list)
    unpaired_builders: List[RobotId] = field(default_factory=list)


def form_pairs_random(
    deliverer_positions: Sequence[NodeId],
    builder_positions: Sequence[NodeId],
    rng: random.Random,
) -> PairingResult:
    """Algorithm FORM-PAIRS-RANDOM(R_d, R_b) --- Шаг 2.

    Граф на вход не подаётся: правило случайное и от среды не зависит.
    Возвращает список пар (по одному доставщику и строителю в каждой) и
    роботов, которым пары не нашлось. Прогон полностью воспроизводим при
    одинаковом состоянии rng."""

    d_ids: List[RobotId] = list(range(len(deliverer_positions)))
    b_ids: List[RobotId] = list(range(len(builder_positions)))
    rng.shuffle(d_ids)
    rng.shuffle(b_ids)

    n_pairs = min(len(d_ids), len(b_ids))
    pairs = [
        Pair(
            pair_id=i,
            deliverer_id=d_ids[i],
            builder_id=b_ids[i],
            deliverer_pos=deliverer_positions[d_ids[i]],
            builder_pos=builder_positions[b_ids[i]],
        )
        for i in range(n_pairs)
    ]

    return PairingResult(
        pairs=pairs,
        unpaired_deliverers=sorted(d_ids[n_pairs:]),
        unpaired_builders=sorted(b_ids[n_pairs:]),
    )


def pairs_by_id(pairs: Sequence[Pair]) -> Dict[PairId, Pair]:
    return {p.pair_id: p for p in pairs}
