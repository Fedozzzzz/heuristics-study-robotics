"""
robot_delivery
==============

Реализация динамической модели многороботной доставки на графе-острове:
ROUTE-AND-COST, ESTIMATE-TASK-COST, SELECT-ROUND, RUN-DYNAMIC-ROUNDS,
COMPUTE-DYNAMIC-COST-BRACKET, а также набор сменных эвристик приоритета p
и инструменты для прогона экспериментов (эвристика vs реальная стоимость).

См. README.md для инструкций по запуску.
"""

from .graph import Graph, EdgeKind
from .costs import route_and_cost, estimate_task_cost, RouteResult, EstimateResult
from .scheduler import run_dynamic_rounds, select_round, ScheduleRecord
from .diagnostics import compute_dynamic_cost_bracket, CostBracket
from .heuristics import HEURISTICS, get_heuristic

__all__ = [
    "Graph",
    "EdgeKind",
    "route_and_cost",
    "estimate_task_cost",
    "RouteResult",
    "EstimateResult",
    "run_dynamic_rounds",
    "select_round",
    "ScheduleRecord",
    "compute_dynamic_cost_bracket",
    "CostBracket",
    "HEURISTICS",
    "get_heuristic",
]
