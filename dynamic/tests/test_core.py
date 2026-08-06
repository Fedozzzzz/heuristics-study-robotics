"""
Базовые тесты: маршрутизация, оценка стоимости, инвариант estimated_raw >= real,
Шаг 0 (проверка достижимости), пересборка коалиций по раундам, правило
"строитель не двигается, если сам не строил", воспроизводимость случайного
разрешения коллизий за мосты, и то, что каждая зарегистрированная эвристика
способна довезти все грузы хотя бы на паре случайных сценариев.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_delivery.graph import EdgeKind, Graph
from robot_delivery.costs import route_and_cost, estimate_task_cost
from robot_delivery.diagnostics import compute_dynamic_cost_bracket
from robot_delivery.heuristics import HEURISTICS
from robot_delivery.scenario import generate_scenario
from robot_delivery.scheduler import Cargo, run_dynamic_rounds


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


def test_all_heuristics_deliver_small_scenario():
    for name, heuristic in HEURISTICS.items():
        scenario = generate_scenario(n_islands=10, n_cargos=6, n_pairs=2, seed=123)
        result = run_dynamic_rounds(
            scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
            heuristic,
        )
        assert result.feasible
        assert result.all_delivered, f"heuristic {name} failed to deliver all cargo"


def test_round_robin_delivers_small_scenario():
    scenario = generate_scenario(n_islands=10, n_cargos=6, n_pairs=2, seed=123)
    heuristic = HEURISTICS["cost_direct"]
    result = run_dynamic_rounds(
        scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
        heuristic, assignment="round_robin",
    )
    assert result.all_delivered


def test_round_robin_assigns_at_most_one_cargo_per_coalition_per_round():
    scenario = generate_scenario(n_islands=14, n_cargos=10, n_pairs=3, seed=5)
    heuristic = HEURISTICS["cost_direct"]
    result = run_dynamic_rounds(
        scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
        heuristic, assignment="round_robin",
    )
    by_round = {}
    for r in result.T:
        by_round.setdefault(r.round_index, []).append(r.coalition_id)
    for round_index, coalition_ids in by_round.items():
        assert len(coalition_ids) == len(set(coalition_ids)), f"round {round_index}: coalition assigned twice"


def test_unknown_assignment_algo_raises():
    scenario = generate_scenario(n_islands=8, n_cargos=4, n_pairs=2, seed=1)
    heuristic = HEURISTICS["cost_direct"]
    try:
        run_dynamic_rounds(
            scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
            heuristic, assignment="bogus",
        )
        assert False, "expected KeyError for unknown assignment algo"
    except KeyError:
        pass


def test_estimated_raw_never_below_real():
    scenario = generate_scenario(n_islands=16, n_cargos=14, n_pairs=3, seed=7)
    heuristic = HEURISTICS["cost_direct"]
    result = run_dynamic_rounds(
        scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
        heuristic,
    )
    assert result.all_delivered
    cargo_map = {c.cargo_id: c for c in scenario.cargos}
    bracket = compute_dynamic_cost_bracket(scenario.G, result.T, cargo_map)
    assert bracket.estimated_raw >= bracket.real - 1e-6
    assert bracket.gap_raw >= -1e-6


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
    heuristic = HEURISTICS["cost_direct"]

    result = run_dynamic_rounds(G, cargos, deliverer_positions=[0], builder_positions=[1], heuristic=heuristic)

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
    heuristic = HEURISTICS["cost_direct"]

    # и доставщик, и строитель стоят вне достижимой компоненты (её тут нет:
    # единственная компонента -- {0,1,2}, поэтому строителя размещаем так,
    # будто он не входит в граф вовсе -- имитируем через отдельный
    # изолированный узел)
    G.add_node(3, w_V=0.0)
    result = run_dynamic_rounds(G, cargos, deliverer_positions=[0], builder_positions=[3], heuristic=heuristic)

    assert not result.feasible
    assert not result.all_delivered
    assert result.infeasibility_reason


def test_coalitions_reformed_each_round_no_duplicate_assignment():
    """Коалиции пересобираются заново на каждом раунде (Шаг 1): проверяем,
    что внутри одного раунда ни один доставщик и ни один строитель не
    попадает в две коалиции одновременно."""
    scenario = generate_scenario(n_islands=12, n_cargos=8, n_pairs=3, seed=9)
    heuristic = HEURISTICS["cost_direct"]
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


def test_builder_position_unchanged_when_no_building_needed():
    """Шаг 5: если строитель коалиции сам ничего не построил в раунде, его
    позиция не меняется. На полностью FREE-графе строить никогда не нужно,
    поэтому позиция единственного строителя должна оставаться неизменной
    во всех раундах."""
    G = Graph()
    for i in range(5):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(1, 2, EdgeKind.FREE, w_E=1.0)
    G.add_edge(2, 3, EdgeKind.FREE, w_E=1.0)
    G.add_edge(3, 4, EdgeKind.FREE, w_E=1.0)
    G.add_edge(0, 4, EdgeKind.FREE, w_E=1.0)
    cargos = [
        Cargo(cargo_id=0, start=0, finish=2),
        Cargo(cargo_id=1, start=1, finish=3),
        Cargo(cargo_id=2, start=2, finish=4),
    ]
    heuristic = HEURISTICS["cost_direct"]

    result = run_dynamic_rounds(G, cargos, deliverer_positions=[0], builder_positions=[4], heuristic=heuristic)

    assert result.feasible
    assert result.all_delivered
    assert all(not r.won_bridges for r in result.T)
    builder_positions_seen = {r.pos_round[1] for r in result.T}
    assert builder_positions_seen == {4}


def test_estimated_prognosis_equals_precollision_and_usually_not_below_real():
    """estimated_prognosis учитывает уже построенные (на предыдущих раундах)
    мосты как бесплатные, но заряжает ПОЛНУЮ стоимость (постройка + проезд)
    за мосты, которые ещё нужно построить -- по построению это то же самое,
    что уже посчитано как W_T_initial/estimated_precollision (см. Шаг 2).
    Для дефолтных параметров генератора сценариев (w_E совпадает с length)
    это, как правило, не ниже real -- реальная стоимость может оказаться
    ниже только за счёт скидки при разрешении коллизий за мосты (Шаг 4);
    строгой гарантии на этот счёт нет (аналогично estimated_pool), поэтому
    здесь не тестируется как инвариант модели."""
    scenario = generate_scenario(n_islands=16, n_cargos=14, n_pairs=3, seed=7)
    heuristic = HEURISTICS["cost_direct"]
    result = run_dynamic_rounds(
        scenario.G, scenario.cargos, scenario.deliverer_positions, scenario.builder_positions,
        heuristic,
    )
    assert result.all_delivered
    # хотя бы один мост реально построен -- иначе сравнение было бы тривиальным
    assert any(r.won_bridges for r in result.T)
    cargo_map = {c.cargo_id: c for c in scenario.cargos}
    bracket = compute_dynamic_cost_bracket(scenario.G, result.T, cargo_map)
    assert abs(bracket.estimated_prognosis - bracket.estimated_precollision) < 1e-9
    assert abs(bracket.gap_prognosis - bracket.gap_precollision) < 1e-9
    assert bracket.estimated_prognosis >= bracket.real - 1e-6


def test_reproducible_with_same_rng_seed():
    """Шаг 4: случайный выбор коалиции, оплачивающей постройку моста при
    коллизии, должен быть воспроизводим при одинаковом rng_seed."""
    scenario = generate_scenario(n_islands=14, n_cargos=10, n_pairs=3, seed=5)
    heuristic = HEURISTICS["cost_direct"]

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
