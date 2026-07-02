"""
Алгоритм 4: построение кривых W_d_total(L), W_b_total(L) на сетке значений L,
выделение Парето-фронта недоминируемых точек, и (опционально) выбор единственной
компромиссной точки L* методом идеальной точки (ideal point method).
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from delivery_model import IslandGraph, Cargo, Pair
from algorithm_2 import run_scheduling, ScheduleOutcome


@dataclass
class ParetoPoint:
    L: float
    W_d_total: float
    W_b_total: float
    outcome: ScheduleOutcome = None


def build_cost_curves(env: IslandGraph, cargos: List[Cargo], pairs: List[Pair],
                       L_grid: List[float], k_alternatives: int = 12) -> List[ParetoPoint]:
    """Шаг 4.1: прогоняет Алгоритм 2 для каждого L из сетки, отбирает допустимые L."""
    points = []
    for L in L_grid:
        outcome = run_scheduling(env, cargos, pairs, L=L, k_alternatives=k_alternatives)
        if outcome.all_delivered:
            points.append(ParetoPoint(L=L, W_d_total=outcome.W_d_total,
                                       W_b_total=outcome.W_b_total, outcome=outcome))
    return points


def pareto_front(points: List[ParetoPoint]) -> List[ParetoPoint]:
    """Шаг 4.2: отбирает недоминируемые точки (минимизация по обеим осям)."""
    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if (q.W_d_total <= p.W_d_total and q.W_b_total <= p.W_b_total and
                    (q.W_d_total < p.W_d_total or q.W_b_total < p.W_b_total)):
                dominated = True
                break
        if not dominated:
            front.append(p)
    front.sort(key=lambda p: p.L)
    return front


def is_concave_front(front: List[ParetoPoint]) -> bool:
    """
    Проверяет, является ли Парето-фронт (при минимизации обеих осей)
    ВОГНУТЫМ к началу координат - то есть "правильной" формы, аналогичной
    классической кривой производственных возможностей (но для минимизации,
    а не максимизации, поэтому ориентация противоположная).

    Геометрический смысл (минимизация): фронт вогнут, если |наклон| между
    соседними точками УБЫВАЕТ по мере движения в сторону роста W_d (то есть
    каждая дополнительная единица снижения W_d обходится ВСЁ ДОРОЖЕ по W_b -
    классический "убывающий эффект масштаба"). Если вместо этого |наклон|
    РАСТЁТ - фронт выпуклый, и любая точка "идеальной близости" будет
    систематически совпадать с одним из КРАЙНИХ узлов фронта, а не с
    промежуточной точкой - именно это наблюдается, когда метод
    choose_compromise(method="euclidean") выбирает крайность, а не середину.

    Возвращает True, если фронт вогнут (или содержит <3 точек - тривиально
    вогнут), False, если обнаружен хотя бы один выпуклый излом.
    """
    pts = sorted(front, key=lambda p: p.W_d_total)
    if len(pts) < 3:
        return True
    slopes = []
    for i in range(len(pts) - 1):
        dwd = pts[i + 1].W_d_total - pts[i].W_d_total
        dwb = pts[i + 1].W_b_total - pts[i].W_b_total
        if dwd == 0:
            continue
        slopes.append(abs(dwb / dwd))
    # вогнутость (при минимизации) <=> |наклон| НЕ возрастает вдоль фронта
    return all(slopes[i] >= slopes[i + 1] - 1e-9 for i in range(len(slopes) - 1))


def choose_compromise(front: List[ParetoPoint], method: str = "euclidean",
                       weight_d: float = 0.5) -> ParetoPoint:
    """
    Шаг 4.3: выбор единственной компромиссной точки L* из Парето-фронта.

    method:
      "euclidean"  - минимизация евклидова расстояния до идеальной точки
                     (min W_d, min W_b) в нормированных координатах.
      "chebyshev"  - минимизация МАКСИМАЛЬНОГО из двух нормированных отклонений
                     (минимакс) - на сильно изогнутых фронтах чаще выбирает
                     промежуточную точку, чем euclidean.
      "weighted_sum" - явная линейная свёртка с весом weight_d (вес W_d,
                     1-weight_d - вес W_b) - подходит, когда есть явное
                     предпочтение одного критерия над другим.

    ПОЧЕМУ КОМПРОМИСС МОЖЕТ ОКАЗАТЬСЯ КРАЙНЕЙ ТОЧКОЙ ФРОНТА:
    Две независимые причины:

    1) ФОРМА ФРОНТА. Если фронт ВЫПУКЛЫЙ (|наклон| между соседними точками
       РАСТЁТ при движении вправо - см. is_concave_front), любая промежуточная
       точка геометрически дальше от угла (wd_min, wb_min), чем хотя бы одна
       из крайних - метод систематически выбирает край. Если фронт ВОГНУТЫЙ
       (как классическая кривая производственных возможностей, но для
       минимизации - выгнут К началу координат), середина может оказаться
       ближе к идеалу, чем оба конца.

    2) МАСШТАБ ОСЕЙ. Даже на вогнутом фронте крайняя точка может победить,
       если нормировка искажает относительную значимость осей. Поэтому здесь
       используется нормировка ПО ДИАПАЗОНУ (x-min)/(max-min), а не по
       минимуму (x-min)/min - это устраняет искажение, когда одна ось
       варьируется на десятки процентов, а другая - на единицы.

    Если нужна гарантированно промежуточная точка независимо от формы фронта -
    используйте method="weighted_sum" с weight_d=0.5.
    """
    if not front:
        raise ValueError("Парето-фронт пуст: ни одно L не дало допустимого решения")

    wd_min = min(p.W_d_total for p in front)
    wb_min = min(p.W_b_total for p in front)
    wd_max = max(p.W_d_total for p in front)
    wb_max = max(p.W_b_total for p in front)
    range_d = max(wd_max - wd_min, 1e-9)
    range_b = max(wb_max - wb_min, 1e-9)

    # ВАЖНО: нормировка по ДИАПАЗОНУ (max-min), а не по минимуму. Если оси
    # имеют сильно разные масштабы изменения (например, W_d варьируется на
    # десятки процентов, а W_b - всего на несколько), нормировка (x-min)/min
    # искажает геометрию: малое в абсолютных числах изменение W_b может
    # выглядеть "незначимым" просто из-за масштаба, и идеальная точка
    # перестаёт корректно отражать относительную важность осей. Нормировка
    # по диапазону приводит обе оси к единому масштабу [0,1] независимо от
    # абсолютных величин - стандартная практика для метода идеальной точки.
    def norm_a(p: ParetoPoint) -> float:
        return (p.W_d_total - wd_min) / range_d

    def norm_b(p: ParetoPoint) -> float:
        return (p.W_b_total - wb_min) / range_b

    if method == "euclidean":
        score = lambda p: (norm_a(p) ** 2 + norm_b(p) ** 2) ** 0.5
    elif method == "chebyshev":
        score = lambda p: max(norm_a(p), norm_b(p))
    elif method == "weighted_sum":
        score = lambda p: weight_d * norm_a(p) + (1 - weight_d) * norm_b(p)
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    return min(front, key=score)
