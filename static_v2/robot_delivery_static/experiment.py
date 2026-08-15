"""
Прогон экспериментов: реальная стоимость исполнения (real) против
эвристических оценок (estimated_static -- оценка Шага 3 статической модели,
estimated_raw -- контрольная) для эвристик приоритета груза (direct / inverse
/ random), режимов назначения (literal / cheapest) и разных размеров сценария.

Не привязан к способу вывода -- возвращает список "плоских" строк (по одной на
прогон), которые plotting.py/cli.py используют для графиков и CSV.

variant = "<эвристика>/<режим назначения>" -- ключ, по которому графики
разделяют кривые: так на одном поле можно сравнивать и эвристики приоритета
между собой, и режимы назначения. Для режима lpt в ключ добавляется ещё и
правило выбора пары ("direct/lpt-load"), чтобы оба его варианта можно было
сравнивать в одной выборке.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .diagnostics import CostBracket, compute_static_cost_bracket
from .priority import get_cargo_heuristic
from .scenario import Scenario, generate_scenario
from .scheduler import RunResult, run_static


@dataclass
class ExperimentRow:
    variant: str
    heuristic: str
    assignment: str
    balance: str
    lpt_size: str
    lpt_rule: str
    seed: int
    run_index: int
    n_islands: int
    n_cargos: int
    n_pairs: int
    feasible: bool
    all_delivered: bool
    n_delivered: int
    real: float
    estimated_static: float
    estimated_raw: float
    gap_static: float
    gap_raw: float
    idle_total: float
    n_rounds: int
    max_pair_load: int      # сколько грузов досталось самой загруженной паре (баланс)
    max_pair_cost: float    # фактическая стоимость работы самой загруженной пары
    cost_imbalance: float   # max_pair_cost / (real / n_pairs); 1.0 -- идеальный баланс


def variant_key(
    heuristic_name: str, assignment_mode: str, lpt_rule: Optional[str] = None,
) -> str:
    """Ключ кривой на графиках. Для lpt дописывается правило выбора пары:
    "direct/lpt-load" и "direct/lpt-completion" -- разные алгоритмы, и в одной
    выборке они не должны сливаться в одну кривую."""
    if assignment_mode == "lpt" and lpt_rule:
        return f"{heuristic_name}/lpt-{lpt_rule}"
    return f"{heuristic_name}/{assignment_mode}"


def pair_cost_stats(result: RunResult) -> Tuple[float, float]:
    """Баланс загрузки пар в ЕДИНИЦАХ СТОИМОСТИ (а не числа грузов): сколько
    работы фактически выпало самой загруженной паре и во сколько раз это
    больше идеально равномерной доли real / n_pairs.

    Именно эту величину минимизирует LPT (makespan), поэтому она нужна как
    отдельная метрика: одинаковое ЧИСЛО грузов у пар ещё не значит одинаковую
    работу."""
    per_pair: Dict[int, float] = defaultdict(float)
    for r in result.T:
        per_pair[r.pair_id] += r.W_d + r.W_b
    n_pairs = len(result.plan) or len(result.pairs)
    if not n_pairs or not per_pair:
        return 0.0, 1.0
    max_cost = max(per_pair.values())
    ideal = result.real / n_pairs
    return max_cost, (max_cost / ideal if ideal > 0 else 1.0)


def run_single(
    scenario: Scenario,
    heuristic_name: str,
    assignment_mode: str = "literal",
    balance: str = "load",
    lpt_size: str = "min",
    lpt_rule: str = "load",
) -> "tuple[RunResult, CostBracket]":
    """Один прогон модели на готовом сценарии."""
    heuristic = get_cargo_heuristic(heuristic_name)
    result = run_static(
        scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
        heuristic,
        assignment_mode=assignment_mode,
        balance=balance,
        lpt_size=lpt_size,
        lpt_rule=lpt_rule,
        rng_seed=scenario.seed,
    )
    cargo_map = {c.cargo_id: c for c in scenario.cargos}
    bracket = compute_static_cost_bracket(scenario.G, result.T, cargo_map)
    return result, bracket


def cargo_sweep(n_pairs: int, n_cargos_max: int) -> List[int]:
    """Диапазон числа грузов для развёртки: от числа пар роботов до заданного
    максимума включительно (нижняя граница -- n_pairs: при меньшем числе
    грузов часть пар заведомо остаётся без работы и сценарий вырождается)."""
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
    assignment_modes: Sequence[str] = ("literal",),
    balance: str = "load",
    lpt_size: str = "min",
    lpt_rule: str = "load",
    scenario_kwargs: Dict | None = None,
    progress: bool = False,
) -> List[ExperimentRow]:
    """Полная развёртка эксперимента.

    Число грузов пробегает диапазон от n_pairs до n_cargos_max включительно.
    Для КАЖДОГО значения генерируется n_runs независимых случайных сценариев, и
    на каждом из них прогоняются ВСЕ комбинации (эвристика x режим назначения)
    -- на одних и тех же инстансах, что делает сравнение честным.

    Итого строк: |heuristics| x |assignment_modes| x |sweep| x n_runs.
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
                for mode in assignment_modes:
                    result, bracket = run_single(
                        scenario, hname, mode, balance, lpt_size, lpt_rule
                    )
                    max_pair_cost, cost_imbalance = pair_cost_stats(result)
                    rows.append(
                        ExperimentRow(
                            variant=variant_key(hname, mode, lpt_rule),
                            heuristic=hname,
                            assignment=mode,
                            balance=("n/a" if mode == "lpt" else balance),
                            lpt_size=(lpt_size if mode == "lpt" else "n/a"),
                            lpt_rule=(lpt_rule if mode == "lpt" else "n/a"),
                            seed=seed,
                            run_index=run_idx,
                            n_islands=n_islands,
                            n_cargos=n_cargos,
                            n_pairs=n_pairs,
                            feasible=result.feasible,
                            all_delivered=result.all_delivered,
                            n_delivered=len(result.delivered_cargo),
                            real=bracket.real,
                            estimated_static=bracket.estimated_static,
                            estimated_raw=bracket.estimated_raw,
                            gap_static=bracket.gap_static,
                            gap_raw=bracket.gap_raw,
                            idle_total=result.idle_total,
                            n_rounds=result.n_rounds,
                            max_pair_load=max(
                                (len(q) for q in result.plan.values()), default=0
                            ),
                            max_pair_cost=max_pair_cost,
                            cost_imbalance=cost_imbalance,
                        )
                    )
        if progress:
            n_variants = len(heuristic_names) * len(assignment_modes)
            print(
                f"  n_cargos={n_cargos}: {n_runs} прогонов x {n_variants} вариантов -- готово"
            )

    return rows


def rows_to_dicts(rows: Sequence[ExperimentRow]) -> List[Dict]:
    return [asdict(r) for r in rows]
