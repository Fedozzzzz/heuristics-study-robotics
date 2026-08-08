"""
Базовые тесты dynamic_v2: маршрутизация, оценка стоимости (те же ROUTE-AND-COST
и ESTIMATE-TASK-COST, что в dynamic/), Шаг 0 (проверка достижимости), Шаг 1
(приоритет груза не зависит от роботов), Шаг 2 (размер раунда N, критерии
выбора доставщика/строителя, tie-break строителя по стоимости постройки),
инвариант estimated_raw >= real, воспроизводимость случайного разрешения
коллизий за мосты, отсутствие повторной постройки одного и того же моста.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_delivery_v2.graph import EdgeKind, Graph
from robot_delivery_v2.costs import route_and_cost, estimate_task_cost
from robot_delivery_v2.cargo_priority import CARGO_HEURISTICS, get_cargo_heuristic, rank_cargos
from robot_delivery_v2.diagnostics import compute_dynamic_cost_bracket
from robot_delivery_v2.optimal import solve_optimal_brute_force
from robot_delivery_v2.scenario import generate_scenario
from robot_delivery_v2.scheduler import Cargo, select_round, run_dynamic_rounds


def make_simple_graph():
    """0 --free(w=1)-- 1 --blocked(len=5,w_build=10,w_E=2)-- 2"""
    G = Graph()
    G.add_node(0, w_V=0.0)
    G.add_node(1, w_V=0.0)
    G.add_node(2, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(1, 2, EdgeKind.BLOCKED, w_E=2.0, w_build=10.0, length=5.0)
    return G


def test_route_and_cost_free_edge():
    G = make_simple_graph()
    r = route_and_cost(G, 0, 1, built=set())
    assert r is not None
    assert r.travel_cost == 1.0
    assert r.build_cost == 0.0


def test_route_and_cost_blocked_edge_needs_build():
    G = make_simple_graph()
    r = route_and_cost(G, 1, 2, built=set())
    assert r is not None
    assert r.build_cost == 10.0
    assert r.travel_cost == 2.0


def test_route_and_cost_reuses_built_bridge():
    G = make_simple_graph()
    built = {G.edge_between(1, 2).key()}
    r = route_and_cost(G, 1, 2, built=built)
    assert r is not None
    assert r.build_cost == 0.0  # мост уже построен -- бесплатно


def test_estimate_task_cost_three_steps():
    G = make_simple_graph()
    est = estimate_task_cost(G, c_start=1, c_finish=2, deliverer_pos=0, builder_pos=0, built=set())
    assert est is not None
    assert est.W_T == est.W_d + est.W_b
    assert est.W_b > 0  # мост нужно строить


def test_cargo_priority_independent_of_robots():
    """Шаг 1: приоритет груза считается только по маршруту c_start -> c_finish,
    сигнатура rank_cargos не принимает никаких позиций роботов -- поэтому
    порядок ранжирования для фиксированного built не может зависеть от них."""
    G = make_simple_graph()
    cargos = [Cargo(cargo_id=0, start=0, finish=2), Cargo(cargo_id=1, start=0, finish=1)]
    ranked = rank_cargos(G, cargos, built=set(), heuristic=CARGO_HEURISTICS["direct"])
    # груз 0->2 (проезд+постройка моста) должен быть дороже и приоритетнее груза 0->1 (просто проезд)
    assert ranked[0][2].cargo_id == 0
    assert ranked[0][1] > ranked[1][1]


def test_direct_and_inverse_are_opposite_orders():
    G = make_simple_graph()
    cargos = [Cargo(cargo_id=0, start=0, finish=2), Cargo(cargo_id=1, start=0, finish=1)]
    direct = rank_cargos(G, cargos, built=set(), heuristic=CARGO_HEURISTICS["direct"])
    inverse = rank_cargos(G, cargos, built=set(), heuristic=CARGO_HEURISTICS["inverse"])
    assert [c.cargo_id for _p, _w, c in direct] == [0, 1]
    assert [c.cargo_id for _p, _w, c in inverse] == [1, 0]


def test_random_priority_reproducible_and_ignores_cost():
    """Эвристика random (baseline): не зависит от W_C, но воспроизводима при
    повторном ранжировании одного и того же набора грузов (сид берётся из
    состава груза, а не из глобального random)."""
    G = make_simple_graph()
    cargos = [Cargo(cargo_id=0, start=0, finish=2), Cargo(cargo_id=1, start=0, finish=1)]
    heuristic = CARGO_HEURISTICS["random"]
    r1 = rank_cargos(G, cargos, built=set(), heuristic=heuristic)
    r2 = rank_cargos(G, cargos, built=set(), heuristic=heuristic)
    assert [(p, c.cargo_id) for p, _w, c in r1] == [(p, c.cargo_id) for p, _w, c in r2]


def test_all_heuristics_deliver_small_scenario():
    for name, heuristic in CARGO_HEURISTICS.items():
        scenario = generate_scenario(n_islands=10, n_cargos=6, n_pairs=2, seed=123)
        result = run_dynamic_rounds(
            scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
            heuristic,
        )
        assert result.feasible
        assert result.all_delivered, f"heuristic {name} failed to deliver all cargo"


def test_unknown_cargo_heuristic_raises():
    try:
        get_cargo_heuristic("bogus")
        assert False, "expected KeyError for unknown cargo heuristic"
    except KeyError:
        pass


def test_step0_infeasible_when_cargo_endpoints_disconnected():
    """Два острова графа физически не связаны никаким ребром (даже
    BLOCKED) -- груз между ними недостижим ни при каком построении мостов,
    Шаг 0 должен остановить алгоритм без единого раунда."""
    G = Graph()
    for i in range(4):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(2, 3, EdgeKind.FREE, w_E=1.0)
    cargos = [Cargo(cargo_id=0, start=0, finish=2)]
    heuristic = CARGO_HEURISTICS["direct"]

    result = run_dynamic_rounds(G, cargos, deliverer_positions=[0], builder_positions=[1], cargo_heuristic=heuristic)

    assert not result.feasible
    assert not result.all_delivered
    assert result.n_rounds == 0
    assert result.T == []
    assert result.infeasibility_reason


def test_step0_infeasible_when_no_builder_in_reachable_component():
    """Груз достижим через BLOCKED-ребро, но ни одного строителя нет в
    достижимой компоненте -- задача невыполнима."""
    G = Graph()
    for i in range(3):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(1, 2, EdgeKind.BLOCKED, w_E=2.0, w_build=10.0, length=5.0)
    cargos = [Cargo(cargo_id=0, start=0, finish=2)]
    heuristic = CARGO_HEURISTICS["direct"]

    G.add_node(3, w_V=0.0)
    result = run_dynamic_rounds(G, cargos, deliverer_positions=[0], builder_positions=[3], cargo_heuristic=heuristic)

    assert not result.feasible
    assert not result.all_delivered
    assert result.infeasibility_reason


def test_coalitions_formed_each_round_no_duplicate_assignment():
    """Шаг 2: внутри одного раунда ни один доставщик и ни один строитель не
    может попасть в две коалиции одновременно."""
    scenario = generate_scenario(n_islands=12, n_cargos=8, n_pairs=3, seed=9)
    heuristic = CARGO_HEURISTICS["direct"]
    result = run_dynamic_rounds(
        scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
        heuristic,
    )
    assert result.feasible
    assert result.all_delivered

    by_round = {}
    for r in result.T:
        by_round.setdefault(r.round_index, []).append((r.deliverer_id, r.builder_id))
    for round_index, coalitions in by_round.items():
        deliverers = [d for d, _b in coalitions]
        builders = [b for _d, b in coalitions]
        assert len(deliverers) == len(set(deliverers)), f"round {round_index}: deliverer reused"
        assert len(builders) == len(set(builders)), f"round {round_index}: builder reused"


def test_round_size_equals_min_deliverers_builders():
    """N = min(floor((Rd+Rb)/2), Rd, Rb, |pending|); при Rd == Rb == 2 и
    достаточном числе грузов первый раунд должен сформировать ровно 2
    коалиции (граф связный по построению генератора -- все комбинации
    доставщик/строитель/груз достижимы)."""
    scenario = generate_scenario(n_islands=12, n_cargos=8, n_pairs=2, seed=3)
    heuristic = CARGO_HEURISTICS["direct"]
    rng = random.Random(scenario.seed)
    records, _built = select_round(
        scenario.G, scenario.cargos,
        deliverer_pos={i: p for i, p in enumerate(scenario.deliverer_positions)},
        builder_pos={i: p for i, p in enumerate(scenario.builder_positions)},
        built=set(), cargo_heuristic=heuristic, round_index=0, rng=rng,
    )
    assert len(records) == 2


def test_builder_tiebreak_by_min_build_cost():
    """Шаг 2: если у нескольких строителей одинаковое минимальное число
    новых мостов, побеждает тот, у кого их суммарная постройка дешевле."""
    G = Graph()
    for i in range(5):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)                                   # доставщик(0) -> c_start(1)
    G.add_edge(1, 2, EdgeKind.BLOCKED, w_E=2.0, w_build=10.0, length=5.0)      # мост маршрута доставки
    G.add_edge(3, 1, EdgeKind.BLOCKED, w_E=1.0, w_build=1.0, length=1.0)       # дешёвый подъезд строителя A (id 0)
    G.add_edge(4, 1, EdgeKind.BLOCKED, w_E=1.0, w_build=100.0, length=1.0)     # дорогой подъезд строителя B (id 1)
    cargo = Cargo(cargo_id=0, start=1, finish=2)

    heuristic = CARGO_HEURISTICS["direct"]
    rng = random.Random(0)
    records, _built = select_round(
        G, [cargo], deliverer_pos={0: 0}, builder_pos={0: 3, 1: 4}, built=set(),
        cargo_heuristic=heuristic, round_index=0, rng=rng,
    )
    assert len(records) == 1
    # оба строителя нуждаются в 2 новых мостах (свой подъезд + мост маршрута) --
    # выбирается тот, у кого дешевле постройка (builder id 0, узел 3)
    assert records[0].builder_id == 0


def test_bridges_never_rebuilt_across_rounds():
    """Возведённые переправы сохраняются между раундами -- ни один мост не
    строится дважды за весь прогон."""
    scenario = generate_scenario(n_islands=16, n_cargos=14, n_pairs=3, seed=7)
    heuristic = CARGO_HEURISTICS["direct"]
    result = run_dynamic_rounds(
        scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
        heuristic,
    )
    assert result.all_delivered
    all_won = [ek for r in result.T for ek in r.won_bridges]
    assert len(all_won) == len(set(all_won)), "один и тот же мост был построен более одного раза"


def test_estimated_raw_never_below_real():
    scenario = generate_scenario(n_islands=16, n_cargos=14, n_pairs=3, seed=7)
    heuristic = CARGO_HEURISTICS["direct"]
    result = run_dynamic_rounds(
        scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
        heuristic,
    )
    assert result.all_delivered
    cargo_map = {c.cargo_id: c for c in scenario.cargos}
    bracket = compute_dynamic_cost_bracket(scenario.G, result.T, cargo_map)
    assert bracket.estimated_raw >= bracket.real - 1e-6
    assert bracket.gap_raw >= -1e-6


def test_optimal_never_exceeds_heuristic_real():
    """optimal.solve_optimal_brute_force -- релаксация модели (любая пара
    доставщик/строитель на любой груз, без ограничений Шага 2), поэтому
    optimal <= real для ЛЮБОЙ эвристики -- competitive ratio >= 1."""
    scenario = generate_scenario(n_islands=8, n_cargos=4, n_pairs=1, seed=1)
    heuristic = CARGO_HEURISTICS["direct"]
    result = run_dynamic_rounds(
        scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
        heuristic,
    )
    assert result.all_delivered
    real = result.W_d_total + result.W_b_total

    opt = solve_optimal_brute_force(
        scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
        upper_bound=real,
    )
    assert opt.feasible
    assert opt.optimal is not None
    assert opt.optimal <= real + 1e-6


def test_optimal_infeasible_returns_none():
    G = Graph()
    for i in range(4):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(2, 3, EdgeKind.FREE, w_E=1.0)
    cargos = [Cargo(cargo_id=0, start=0, finish=2)]

    opt = solve_optimal_brute_force(G, cargos, deliverer_positions=[0], builder_positions=[1])
    assert not opt.feasible
    assert opt.optimal is None
    assert opt.infeasibility_reason


def test_optimal_trivial_single_free_edge():
    """Единственный груз, единственный маршрут без построек -- оптимум точно
    известен заранее: проезд туда деливером напрямую (мост строить не нужно,
    т.к. груз уже на маршруте FREE-рёбер)."""
    G = Graph()
    for i in range(3):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(1, 2, EdgeKind.FREE, w_E=1.0)
    cargos = [Cargo(cargo_id=0, start=0, finish=2)]

    opt = solve_optimal_brute_force(G, cargos, deliverer_positions=[0], builder_positions=[2])
    assert opt.feasible
    assert opt.optimal is not None
    assert abs(opt.optimal - 2.0) < 1e-9  # доставщик проезжает 0->1->2, вес рёбер 1+1


def test_optimal_confirms_upper_bound_when_it_is_already_the_true_optimum():
    """Регрессия: если переданный upper_bound уже РАВЕН истинному оптимуму,
    branch & bound (сравнение >=) отсекает единственный путь, доходящий ровно
    до best[0], и не может "переоткрыть" его через local_best/total -- баг
    приводил к тому, что solve_optimal_brute_force возвращал None вместо
    заведомо достижимого и доказанно оптимального значения. Итоговый ответ
    должен браться из best[0] (глобального трекера), а не из total."""
    G = Graph()
    for i in range(3):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(1, 2, EdgeKind.FREE, w_E=1.0)
    cargos = [Cargo(cargo_id=0, start=0, finish=2)]

    opt = solve_optimal_brute_force(
        G, cargos, deliverer_positions=[0], builder_positions=[2], upper_bound=2.0,
    )
    assert opt.feasible
    assert opt.optimal is not None
    assert abs(opt.optimal - 2.0) < 1e-9


def test_optimal_max_cargos_guard_raises():
    scenario = generate_scenario(n_islands=6, n_cargos=3, n_pairs=1, seed=1)
    try:
        solve_optimal_brute_force(
            scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
            max_cargos=2,
        )
        assert False, "expected ValueError when n_cargos > max_cargos"
    except ValueError:
        pass


def test_reproducible_with_same_rng_seed():
    """Шаг 4: случайный выбор коалиции, оплачивающей постройку моста при
    коллизии, должен быть воспроизводим при одинаковом rng_seed."""
    scenario = generate_scenario(n_islands=14, n_cargos=10, n_pairs=3, seed=5)
    heuristic = CARGO_HEURISTICS["direct"]

    def run():
        result = run_dynamic_rounds(
            scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
            heuristic, rng_seed=42,
        )
        return [
            (r.round_index, r.cargo_id, r.deliverer_id, r.builder_id, r.W_d, r.W_b)
            for r in result.T
        ]

    assert run() == run()


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\nВсе тесты пройдены ({len(tests)}).")
