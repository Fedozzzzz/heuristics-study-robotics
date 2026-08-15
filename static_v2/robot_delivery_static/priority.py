"""
Шаг 3 (часть 1). Приоритет груза для КОНКРЕТНОЙ пары.

Отличие от dynamic_v2: там приоритет груза считается сам по себе, только по
стоимости его собственного маршрута c_start -> c_finish. Здесь, по постановке
статической модели, "для каждой коалиции оценивается стоимость доставки
каждого груза", т.е. приоритет считается для ЯЧЕЙКИ таблицы пара x груз:

    p(U_k, c_i) = f( W_T(U_k, c_i) ),

где W_T -- полная оценка ESTIMATE-TASK-COST от НАЧАЛЬНЫХ позиций пары U_k при
built = ∅ (см. assignment.build_cost_table).

Базовая эвристика постановки -- прямая: "приоритет напрямую зависит от
стоимости доставки груза текущей парой: чем выше стоимость, тем выше
приоритет". Для сравнения перенесены остальные две эвристики из dynamic_v2:

  direct  (p = W_T)     --- дороже доставка -- выше приоритет (постановка).
  inverse (p = 1 / W_T) --- дешевле доставка -- выше приоритет.
  random  (p ~ U(0,1))  --- baseline: не зависит от стоимости вообще.
"""

from __future__ import annotations

import random as _random_module
from dataclasses import dataclass
from typing import Any, Callable, Dict

EPS = 1e-9


@dataclass
class CargoPriorityHeuristic:
    key: str
    label: str
    description: str
    score: Callable[[Any, float], float]   # (cargo, W_T) -> p


def _direct(cargo: Any, w: float) -> float:
    # p = W_T -- дороже доставка -- выше приоритет.
    return w


def _inverse(cargo: Any, w: float) -> float:
    # p = 1 / W_T -- дешевле доставка -- выше приоритет.
    return 1.0 / (w + EPS)


def _random(cargo: Any, w: float) -> float:
    # p ~ U(0,1), не зависит от W_T -- чистый шум. Сид берётся из состава груза
    # (id + точки погрузки/разгрузки), а не из глобального random, чтобы
    # прогон был воспроизводим (та же техника, что в dynamic_v2).
    #
    # ВНИМАНИЕ: значение одинаково для ВСЕХ пар одного груза (приоритет груза,
    # а не ячейки таблицы). В режиме назначения literal это означает, что
    # выбор пары для такого груза определяется tie-break'ом по pair_id и
    # балансировкой загрузки, а не стоимостью -- что и требуется от baseline.
    seed = (cargo.cargo_id, cargo.start, cargo.finish)
    return _random_module.Random(hash(seed)).random()


CARGO_HEURISTICS: Dict[str, CargoPriorityHeuristic] = {
    "direct": CargoPriorityHeuristic(
        key="direct",
        label="Прямая (p = W_T)",
        description="Дороже доставка груза данной парой -- выше приоритет (эвристика постановки).",
        score=_direct,
    ),
    "inverse": CargoPriorityHeuristic(
        key="inverse",
        label="Обратная (p = 1 / W_T)",
        description="Дешевле доставка груза данной парой -- выше приоритет.",
        score=_inverse,
    ),
    "random": CargoPriorityHeuristic(
        key="random",
        label="Случайный приоритет (baseline)",
        description=(
            "Baseline для оценки качества direct/inverse: p ~ U(0,1), не "
            "зависит от стоимости доставки."
        ),
        score=_random,
    ),
}


def get_cargo_heuristic(name: str) -> CargoPriorityHeuristic:
    if name not in CARGO_HEURISTICS:
        raise KeyError(
            f"Неизвестная эвристика приоритета груза '{name}'. Доступные: "
            f"{', '.join(sorted(CARGO_HEURISTICS))}"
        )
    return CARGO_HEURISTICS[name]
