"""
Прогон экспериментов static_v3.

На графики и в CSV выносится ТОЛЬКО фактическая стоимость выполнения всех
операций real = W_d_total + W_b_total (Шаг 5). Величины "оценка до выполнения"
у этой модели нет: и пары, и распределение грузов разыгрываются случайно, так
что никакой предварительной оценки стоимости модель не считает -- см.
assignment.py.

ПОВТОРЫ (n_repeats). Модель случайна в трёх местах (паросочетание, раздача
грузов, выбор плательщика за общий мост), поэтому один прогон на сценарии --
это одна реализация случайной величины, а не характеристика сценария. Чтобы
кривые не состояли из шума, каждый сгенерированный сценарий можно прогнать
n_repeats раз с разными сидами модели; в CSV пишутся ВСЕ прогоны (строка на
пару "сценарий x повтор"), а усреднение делают уже графики. При n_repeats = 1
поведение совпадает со static_v2: один прогон на сценарий, дисперсия гасится
числом сценариев (--n-runs).

Не привязан к способу вывода -- возвращает список "плоских" строк, которые
plotting.py/cli.py используют для графиков и CSV.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

from .scenario import Scenario, generate_scenario
from .scheduler import RunResult, run_random_static

# Сдвиг сида модели между повторами на одном и том же сценарии. Взаимно простое
# с шагами сидов сценариев в run_suite, чтобы серии не накладывались.
REPEAT_SEED_STRIDE = 100_003


@dataclass
class ExperimentRow:
    variant: str            # режим случайного распределения (balanced / uniform)
    assignment: str
    seed: int               # сид сценария
    rng_seed: int           # сид самой модели (различается между повторами)
    run_index: int          # номер случайного сценария
    repeat_index: int       # номер повтора модели на этом сценарии
    n_islands: int
    n_cargos: int
    n_pairs: int
    feasible: bool
    all_delivered: bool
    n_delivered: int
    real: float             # Шаг 5: общая стоимость всех операций
    idle_total: float
    n_rounds: int
    max_pair_load: int      # сколько грузов досталось самой загруженной паре
    min_pair_load: int      # сколько -- самой незагруженной (для uniform бывает 0)
    max_pair_cost: float    # фактическая стоимость работы самой загруженной пары
    cost_imbalance: float   # max_pair_cost / (real / n_pairs); 1.0 -- идеальный баланс


def pair_cost_stats(result: RunResult) -> Tuple[float, float]:
    """Баланс загрузки пар в ЕДИНИЦАХ СТОИМОСТИ (а не числа грузов): сколько
    работы фактически выпало самой загруженной паре и во сколько раз это
    больше идеально равномерной доли real / n_pairs."""
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
    assignment_mode: str = "balanced",
    rng_seed: int | None = None,
) -> RunResult:
    """Один прогон модели на готовом сценарии.

    rng_seed -- сид случайности САМОЙ МОДЕЛИ (Шаги 2/3/4). По умолчанию берётся
    сид сценария, то есть один сценарий -> один воспроизводимый прогон."""
    return run_random_static(
        scenario.G, scenario.cargos,
        scenario.deliverer_positions, scenario.builder_positions,
        assignment_mode=assignment_mode,
        rng_seed=scenario.seed if rng_seed is None else rng_seed,
    )


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
    n_runs: int,
    n_cargos_max: int,
    n_pairs: int = 3,
    n_islands: int = 18,
    base_seed: int = 0,
    assignment_modes: Sequence[str] = ("balanced",),
    n_repeats: int = 1,
    scenario_kwargs: Dict | None = None,
    progress: bool = False,
) -> List[ExperimentRow]:
    """Полная развёртка эксперимента.

    Число грузов пробегает диапазон от n_pairs до n_cargos_max включительно.
    Для КАЖДОГО значения генерируется n_runs независимых случайных сценариев;
    на каждом из них модель прогоняется n_repeats раз (разные сиды модели) для
    ВСЕХ режимов распределения -- на одних и тех же инстансах и с одними и теми
    же сидами модели, что делает сравнение режимов честным.

    Итого строк: |assignment_modes| x |sweep| x n_runs x n_repeats.
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
            for repeat_idx in range(n_repeats):
                rng_seed = seed + repeat_idx * REPEAT_SEED_STRIDE
                for mode in assignment_modes:
                    result = run_single(scenario, mode, rng_seed=rng_seed)
                    max_pair_cost, cost_imbalance = pair_cost_stats(result)
                    loads = [len(q) for q in result.plan.values()]
                    rows.append(
                        ExperimentRow(
                            variant=mode,
                            assignment=mode,
                            seed=seed,
                            rng_seed=rng_seed,
                            run_index=run_idx,
                            repeat_index=repeat_idx,
                            n_islands=n_islands,
                            n_cargos=n_cargos,
                            n_pairs=n_pairs,
                            feasible=result.feasible,
                            all_delivered=result.all_delivered,
                            n_delivered=len(result.delivered_cargo),
                            real=result.real,
                            idle_total=result.idle_total,
                            n_rounds=result.n_rounds,
                            max_pair_load=max(loads, default=0),
                            min_pair_load=min(loads, default=0),
                            max_pair_cost=max_pair_cost,
                            cost_imbalance=cost_imbalance,
                        )
                    )
        if progress:
            n_variants = len(assignment_modes) * n_repeats
            print(
                f"  n_cargos={n_cargos}: {n_runs} сценариев x {n_variants} прогонов -- готово"
            )

    return rows


def rows_to_dicts(rows: Sequence[ExperimentRow]) -> List[Dict]:
    return [asdict(r) for r in rows]
