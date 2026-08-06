"""
Прогон экспериментов: сравнение реальной стоимости исполнения (real) с
эвристической оценкой (estimated_raw / estimated_pool / estimated_precollision)
для разных эвристик приоритета p и разных размеров сценария (n_cargos).

Не привязан к конкретному способу вывода -- возвращает список "плоских"
словарей (по одной строке на прогон), которые cli.py использует для графиков.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from .diagnostics import compute_dynamic_cost_bracket
from .heuristics import get_heuristic
from .scenario import Scenario, generate_scenario
from .scheduler import RunResult, run_dynamic_rounds


@dataclass
class ExperimentRow:
    heuristic: str
    assignment: str
    seed: int
    run_index: int
    n_islands: int
    n_cargos: int
    n_pairs: int
    feasible: bool
    all_delivered: bool
    n_delivered: int
    real: float
    estimated_raw: float
    estimated_pool: float
    estimated_precollision: float
    estimated_prognosis: float
    gap_raw: float
    gap_pool: float
    gap_precollision: float
    gap_prognosis: float
    idle_total: float
    n_rounds: int


def run_single(
    scenario: Scenario, heuristic_name: str, assignment: str = "greedy"
) -> "tuple[RunResult, Dict]":
    heuristic = get_heuristic(heuristic_name)
    result = run_dynamic_rounds(
        scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
        heuristic, assignment=assignment, rng_seed=scenario.seed,
    )
    cargo_map = {c.cargo_id: c for c in scenario.cargos}
    bracket = compute_dynamic_cost_bracket(scenario.G, result.T, cargo_map)
    return result, bracket


def cargo_sweep(n_pairs: int, n_cargos_max: int) -> List[int]:
    """Диапазон числа грузов для развёртки: от числа пар роботов до заданного
    максимума n включительно.

    Нижняя граница -- n_pairs: при меньшем числе грузов часть пар заведомо
    остаётся без назначения уже в первом раунде, и сценарий вырождается.
    Верхняя граница -- n (заданное пользователем количество грузов), что
    соответствует условию |C| >> |U| из постановки задачи.
    """
    if n_cargos_max < n_pairs:
        raise ValueError(
            f"Количество грузов n={n_cargos_max} меньше количества пар роботов "
            f"({n_pairs}); развёртка невозможна."
        )
    return list(range(n_pairs, n_cargos_max + 1))


def run_suite(
    heuristic_names: Sequence[str],
    n_runs: int,
    n_cargos_max: int,
    n_pairs: int = 3,
    n_islands: int = 18,
    base_seed: int = 0,
    scenario_kwargs: Dict | None = None,
    progress: bool = False,
    assignment_names: Sequence[str] = ("greedy",),
) -> List[ExperimentRow]:
    """Полная развёртка эксперимента.

    Число грузов пробегает диапазон от n_pairs до n_cargos_max включительно
    (см. cargo_sweep). Для КАЖДОГО значения числа грузов генерируется n_runs
    независимых случайных сценариев (рандомайзер), и на каждом из них
    прогоняются ВСЕ комбинации (эвристика приоритета x алгоритм распределения
    грузов по раундам) -- на одних и тех же инстансах, что делает сравнение
    честным (см. ASSIGNMENT_ALGOS в scheduler.py: "greedy" -- текущий жадный
    проход по таблице приоритетов, "round_robin" -- пары выбирают груз по
    очереди).

    Итого строк в результате: |heuristics| x |assignment_names| x |sweep| x n_runs.
    """
    scenario_kwargs = dict(scenario_kwargs or {})
    rows: List[ExperimentRow] = []
    sweep = cargo_sweep(n_pairs, n_cargos_max)

    for n_cargos in sweep:
        for run_idx in range(n_runs):
            # воспроизводимый, но различный seed на каждую точку развёртки
            seed = base_seed + run_idx * 1009 + n_cargos * 7919
            scenario = generate_scenario(
                n_islands=n_islands,
                n_cargos=n_cargos,
                n_pairs=n_pairs,
                seed=seed,
                **scenario_kwargs,
            )
            for hname in heuristic_names:
                for aname in assignment_names:
                    result, bracket = run_single(scenario, hname, aname)
                    rows.append(
                        ExperimentRow(
                            heuristic=hname,
                            assignment=aname,
                            seed=seed,
                            run_index=run_idx,
                            n_islands=n_islands,
                            n_cargos=n_cargos,
                            n_pairs=n_pairs,
                            feasible=result.feasible,
                            all_delivered=result.all_delivered,
                            n_delivered=len(result.delivered_cargo),
                            real=bracket.real,
                            estimated_raw=bracket.estimated_raw,
                            estimated_pool=bracket.estimated_pool,
                            estimated_precollision=bracket.estimated_precollision,
                            estimated_prognosis=bracket.estimated_prognosis,
                            gap_raw=bracket.gap_raw,
                            gap_pool=bracket.gap_pool,
                            gap_precollision=bracket.gap_precollision,
                            gap_prognosis=bracket.gap_prognosis,
                            idle_total=result.idle_total,
                            n_rounds=result.n_rounds,
                        )
                    )
        if progress:
            print(
                f"  n_cargos={n_cargos}: {n_runs} прогонов x {len(heuristic_names)} эвристик "
                f"x {len(assignment_names)} алгоритмов распределения -- готово"
            )

    return rows


def rows_to_dicts(rows: Sequence[ExperimentRow]) -> List[Dict]:
    return [asdict(r) for r in rows]
