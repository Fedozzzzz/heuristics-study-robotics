"""
Шаг 3. Распределение грузов между коалициями (ASSIGN-CARGOS-RANDOM).

    "Распределение грузов производится случайным образом. В результате
     получаем список грузов для каждой коалиции роботов."

Выполняется ОДИН РАЗ, до начала исполнения, и больше не пересматривается:
список грузов пары фиксирован на весь прогон. Порядок внутри списка -- это и
порядок раундов Шага 4 (в раунде r пара везёт r-й груз своего списка), поэтому
он тоже разыгрывается случайно.

Никакой оценки стоимости здесь НЕ считается. В static_v2 этот шаг начинался с
таблицы "пара x груз" -- |пар| x |грузов| вызовов ESTIMATE-TASK-COST от
начальных позиций пары при built = ∅, -- по которой считались приоритеты. В
static_v3 распределение случайное, приоритет ни на что не влияет, и таблица не
строится вовсе: Шаг 3 линеен по числу грузов и не обращается к графу.

Отсюда же и ответ на вопрос, что считать эвристической оценкой работы модели:
величины "оценка до выполнения" (estimated_static из static_v2) у static_v3
просто нет. Согласно Шагу 5 постановки, эвристическая оценка -- это
ФАКТИЧЕСКАЯ общая стоимость выполнения всех операций, полученная в конце
прогона (RunResult.real).

Реализованы два режима (--assignment), различающиеся тем, как именно
разыгрывается "случайным образом":

  balanced (по умолчанию) -- список грузов перемешивается и раздаётся парам по
      кругу (round-robin). Какой груз какой паре достанется -- полностью
      случайно, но ЧИСЛО грузов у пар отличается максимум на 1. Раунды Шага 4
      при этом не вырождаются: все пары работают примерно одинаковое число
      раундов.
  uniform -- каждому грузу независимо и равновероятно назначается пара
      (rng.choice). Буквальнее по тексту постановки, но число грузов у пар
      распределено мультиномиально: при малом числе грузов одна пара может
      забрать почти все, а другая остаться вовсе без работы (и тогда
      "параллельное выполнение по раундам" фактически сводится к
      последовательной работе одной пары).

Оба режима сравниваются на одних и тех же сценариях --
experiments/compare_random_assignment.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .model import Cargo, CargoId
from .pairing import Pair, PairId

ASSIGNMENT_MODES = ("balanced", "uniform")


@dataclass
class Assignment:
    """Результат Шага 3: список грузов для каждой коалиции.

    per_pair -- очередь пары; её порядок и есть порядок раундов Шага 4.
    unassigned в static_v3 всегда пуст (случайное правило раздаёт все грузы) и
    сохранён только ради единообразия с RunResult.undelivered, куда сваливаются
    грузы, которые пара не смогла выполнить физически."""

    per_pair: Dict[PairId, List[CargoId]] = field(default_factory=dict)
    unassigned: List[CargoId] = field(default_factory=list)
    mode: str = "balanced"


def assign_cargos_random(
    pairs: Sequence[Pair],
    cargos: Sequence[Cargo],
    rng: random.Random,
    mode: str = "balanced",
) -> Assignment:
    """Algorithm ASSIGN-CARGOS-RANDOM(U, C) --- Шаг 3.

    Ни граф, ни позиции роботов на вход не подаются: правило случайное и от
    среды не зависит. Прогон полностью воспроизводим при одинаковом состоянии
    rng."""

    if mode not in ASSIGNMENT_MODES:
        raise ValueError(
            f"Неизвестный режим распределения грузов '{mode}'. "
            f"Доступные: {', '.join(ASSIGNMENT_MODES)}"
        )

    per_pair: Dict[PairId, List[CargoId]] = {p.pair_id: [] for p in pairs}
    if not pairs:
        return Assignment(per_pair=per_pair,
                          unassigned=[c.cargo_id for c in cargos], mode=mode)

    # Перемешиваются в обоих режимах: в balanced этим и задаётся случайность
    # раздачи, в uniform -- случайность ПОРЯДКА внутри очереди пары (сам выбор
    # пары там разыгрывается отдельно).
    order: List[CargoId] = [c.cargo_id for c in cargos]
    rng.shuffle(order)

    if mode == "balanced":
        for i, cargo_id in enumerate(order):
            per_pair[pairs[i % len(pairs)].pair_id].append(cargo_id)
    else:  # uniform
        for cargo_id in order:
            per_pair[rng.choice(pairs).pair_id].append(cargo_id)

    return Assignment(per_pair=per_pair, unassigned=[], mode=mode)
