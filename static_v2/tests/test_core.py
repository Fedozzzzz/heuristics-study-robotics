"""
Базовые тесты static_v2: маршрутизация и оценка стоимости (те же
ROUTE-AND-COST и ESTIMATE-TASK-COST, что в dynamic_v2), Шаг 0 (проверка
достижимости), Шаг 2 (жадное паросочетание по минимальному расстоянию),
Шаг 3 (таблица пара x груз, оба режима назначения, балансировка загрузки,
статичность распределения), Шаг 4 (структура раундов, общие мосты,
воспроизводимость), инвариант estimated_raw >= real.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_delivery_static.assignment import assign_cargos, assign_lpt, build_cost_table
from robot_delivery_static.costs import estimate_task_cost, route_and_cost
from robot_delivery_static.diagnostics import compute_static_cost_bracket
from robot_delivery_static.graph import EdgeKind, Graph
from robot_delivery_static.model import Cargo
from robot_delivery_static.pairing import form_pairs
from robot_delivery_static.priority import CARGO_HEURISTICS, get_cargo_heuristic
from robot_delivery_static.scenario import generate_scenario
from robot_delivery_static.scheduler import run_static


def make_simple_graph():
    """0 --free(w=1)-- 1 --blocked(len=5,w_build=10,w_E=2)-- 2"""
    G = Graph()
    G.add_node(0, w_V=0.0)
    G.add_node(1, w_V=0.0)
    G.add_node(2, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(1, 2, EdgeKind.BLOCKED, w_E=2.0, w_build=10.0, length=5.0)
    return G


# --- ядро (перенесено из dynamic_v2) -------------------------------------

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


def test_estimate_task_cost_three_steps():
    G = make_simple_graph()
    est = estimate_task_cost(G, c_start=1, c_finish=2, deliverer_pos=0, builder_pos=0, built=set())
    assert est is not None
    assert est.W_T == est.W_d + est.W_b
    assert est.W_b > 0  # мост нужно строить


# --- Шаг 0 ----------------------------------------------------------------

def test_step0_infeasible_when_cargo_endpoints_disconnected():
    """Два острова физически не связаны никаким ребром (даже BLOCKED) -- груз
    между ними недостижим ни при каком строительстве."""
    G = Graph()
    for i in range(4):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(2, 3, EdgeKind.FREE, w_E=1.0)
    cargos = [Cargo(cargo_id=0, start=0, finish=2)]

    result = run_static(G, cargos, [0], [1], CARGO_HEURISTICS["direct"])

    assert not result.feasible
    assert not result.all_delivered
    assert result.n_rounds == 0
    assert result.T == []
    assert result.infeasibility_reason


def test_step0_infeasible_when_no_builder_in_reachable_component():
    G = Graph()
    for i in range(3):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(1, 2, EdgeKind.BLOCKED, w_E=2.0, w_build=10.0, length=5.0)
    G.add_node(3, w_V=0.0)  # изолированный остров со строителем
    cargos = [Cargo(cargo_id=0, start=0, finish=2)]

    result = run_static(G, cargos, [0], [3], CARGO_HEURISTICS["direct"])

    assert not result.feasible
    assert result.infeasibility_reason


# --- Шаг 2: формирование пар ----------------------------------------------

def test_pairing_greedy_by_min_distance():
    """Линейный граф 0-1-2-3: доставщики стоят в 0 и 3, строители -- в 1 и 2.
    Жадное паросочетание должно дать (0,1) и (3,2), а не перекрёстное."""
    G = Graph()
    for i in range(4):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(1, 2, EdgeKind.FREE, w_E=5.0)
    G.add_edge(2, 3, EdgeKind.FREE, w_E=1.0)

    pairing = form_pairs(G, deliverer_positions=[0, 3], builder_positions=[1, 2])

    assert len(pairing.pairs) == 2
    matched = {(p.deliverer_pos, p.builder_pos) for p in pairing.pairs}
    assert matched == {(0, 1), (3, 2)}
    assert not pairing.unpaired_deliverers and not pairing.unpaired_builders


def test_pairing_leaves_extra_robots_unpaired():
    G = make_simple_graph()
    pairing = form_pairs(G, deliverer_positions=[0, 1], builder_positions=[0])
    assert len(pairing.pairs) == 1
    assert len(pairing.unpaired_deliverers) == 1
    assert not pairing.unpaired_builders


def test_pairing_is_fixed_for_the_whole_run():
    """Статическая модель: пара (доставщик, строитель) не пересобирается --
    один и тот же доставщик за весь прогон работает с одним и тем же
    строителем."""
    scenario = generate_scenario(n_islands=14, n_cargos=10, n_pairs=3, seed=11)
    result = run_static(
        scenario.G, scenario.cargos, scenario.deliverer_positions,
        scenario.builder_positions, CARGO_HEURISTICS["direct"],
    )
    assert result.all_delivered
    partner = {}
    for r in result.T:
        partner.setdefault(r.deliverer_id, r.builder_id)
        assert partner[r.deliverer_id] == r.builder_id


# --- Шаг 3: распределение грузов ------------------------------------------

def _two_pair_setup():
    """Два "куста" по разные стороны длинного моста: пара 0 стоит у острова 0,
    пара 1 -- у острова 3. Груз A (0->1) дёшев для пары 0 и дорог для пары 1,
    груз B (3->4) -- наоборот."""
    G = Graph()
    for i in range(5):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(1, 2, EdgeKind.FREE, w_E=20.0)
    G.add_edge(2, 3, EdgeKind.FREE, w_E=20.0)
    G.add_edge(3, 4, EdgeKind.FREE, w_E=1.0)
    cargos = [Cargo(cargo_id=0, start=0, finish=1), Cargo(cargo_id=1, start=3, finish=4)]
    pairing = form_pairs(G, deliverer_positions=[0, 3], builder_positions=[0, 3])
    return G, cargos, pairing


def test_cost_table_covers_all_pairs_and_cargos():
    G, cargos, pairing = _two_pair_setup()
    table = build_cost_table(G, pairing.pairs, cargos)
    assert len(table) == len(pairing.pairs) * len(cargos)
    # для каждой пары "свой" груз дешевле чужого
    near = {p.pair_id: p.deliverer_pos for p in pairing.pairs}
    for pair_id, pos in near.items():
        own, foreign = (0, 1) if pos == 0 else (1, 0)
        assert table[(pair_id, own)].W_T < table[(pair_id, foreign)].W_T


def test_assignment_literal_gives_cargo_to_most_expensive_pair():
    """Режим literal + эвристика direct: список свёрнут по убыванию стоимости,
    поэтому первый же груз достаётся паре, для которой он САМЫЙ ДОРОГОЙ."""
    G, cargos, pairing = _two_pair_setup()
    table = build_cost_table(G, pairing.pairs, cargos)
    a = assign_cargos(table, pairing.pairs, cargos, CARGO_HEURISTICS["direct"],
                      mode="literal", balance="none")
    for cargo_id, pair_id in a.pair_of_cargo.items():
        costs = {k: table[(k, cargo_id)].W_T for k in a.per_pair}
        assert pair_id == max(costs, key=lambda k: costs[k])


def test_assignment_cheapest_gives_cargo_to_cheapest_pair():
    G, cargos, pairing = _two_pair_setup()
    table = build_cost_table(G, pairing.pairs, cargos)
    a = assign_cargos(table, pairing.pairs, cargos, CARGO_HEURISTICS["direct"],
                      mode="cheapest", balance="none")
    for cargo_id, pair_id in a.pair_of_cargo.items():
        costs = {k: table[(k, cargo_id)].W_T for k in a.per_pair}
        assert pair_id == min(costs, key=lambda k: costs[k])


def test_assignment_inverse_in_literal_mode_mirrors_cheapest():
    """p = 1/W_T максимален там, где стоимость минимальна, поэтому буквальный
    проход по списку при обратной эвристике отдаёт груз самой ДЕШЁВОЙ паре."""
    G, cargos, pairing = _two_pair_setup()
    table = build_cost_table(G, pairing.pairs, cargos)
    a = assign_cargos(table, pairing.pairs, cargos, CARGO_HEURISTICS["inverse"],
                      mode="literal", balance="none")
    for cargo_id, pair_id in a.pair_of_cargo.items():
        costs = {k: table[(k, cargo_id)].W_T for k in a.per_pair}
        assert pair_id == min(costs, key=lambda k: costs[k])


def test_assignment_covers_every_cargo_exactly_once():
    scenario = generate_scenario(n_islands=14, n_cargos=12, n_pairs=3, seed=4)
    pairing = form_pairs(scenario.G, scenario.deliverer_positions, scenario.builder_positions)
    table = build_cost_table(scenario.G, pairing.pairs, scenario.cargos)
    for mode in ("literal", "cheapest", "lpt"):
        for balance in ("load", "none"):
            a = assign_cargos(table, pairing.pairs, scenario.cargos,
                              CARGO_HEURISTICS["direct"], mode=mode, balance=balance)
            flat = [c for lst in a.per_pair.values() for c in lst]
            assert sorted(flat) == [c.cargo_id for c in scenario.cargos], (mode, balance)
            assert not a.unassigned


def test_balance_load_prevents_one_pair_taking_everything():
    """Балансировка по стоимости: ни одна пара не забирает заметно больше
    грузов, чем при равномерном распределении, и ни одна не простаивает.
    Проверяется для ОБЕИХ эвристик: порог загрузки считается по стоимости той
    пары, которую правило реально выберет, поэтому для inverse (где выигрывает
    самая дешёвая пара) он должен работать так же, как для direct."""
    scenario = generate_scenario(n_islands=16, n_cargos=15, n_pairs=3, seed=21)
    pairing = form_pairs(scenario.G, scenario.deliverer_positions, scenario.builder_positions)
    table = build_cost_table(scenario.G, pairing.pairs, scenario.cargos)
    ideal = len(scenario.cargos) / len(pairing.pairs)

    for hname in ("direct", "inverse"):
        heuristic = CARGO_HEURISTICS[hname]
        balanced = assign_cargos(table, pairing.pairs, scenario.cargos, heuristic,
                                 mode="literal", balance="load")
        unbalanced = assign_cargos(table, pairing.pairs, scenario.cargos, heuristic,
                                   mode="literal", balance="none")

        max_balanced = max(len(v) for v in balanced.per_pair.values())
        max_unbalanced = max(len(v) for v in unbalanced.per_pair.values())
        assert max_balanced <= max_unbalanced, hname
        assert max_balanced < len(scenario.cargos), hname
        # балансировка по стоимости не обязана выравнивать ЧИСЛО грузов
        # в точности, но перекос вдвое относительно равномерного -- уже нет
        assert max_balanced <= 2 * ideal, hname
        # ни одна пара не простаивает полностью
        assert all(len(v) > 0 for v in balanced.per_pair.values()), hname


# --- Шаг 3, режим lpt ------------------------------------------------------

def _pair_loads(assignment, table):
    """Загрузка каждой пары в оценках Шага 3 (то, что балансирует LPT)."""
    loads = {k: 0.0 for k in assignment.per_pair}
    for cargo_id, pair_id in assignment.pair_of_cargo.items():
        loads[pair_id] += table[(pair_id, cargo_id)].W_T
    return loads


def _lpt_two_cargos_near_one_pair():
    """Оба груза стоят вплотную к паре 0 и очень далеко от пары 1 (см.
    _two_pair_setup: между "кустами" 40 единиц пути)."""
    G = Graph()
    for i in range(5):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(1, 2, EdgeKind.FREE, w_E=20.0)
    G.add_edge(2, 3, EdgeKind.FREE, w_E=20.0)
    G.add_edge(3, 4, EdgeKind.FREE, w_E=1.0)
    cargos = [Cargo(cargo_id=0, start=0, finish=1), Cargo(cargo_id=1, start=0, finish=1)]
    pairing = form_pairs(G, deliverer_positions=[0, 3], builder_positions=[0, 3])
    return G, cargos, pairing


def test_lpt_processes_long_jobs_first():
    """Ядро LPT -- порядок: грузы разбираются по УБЫВАНИЮ размера задачи.
    Очередь каждой пары -- подпоследовательность этого общего порядка, значит
    приоритеты внутри очереди не возрастают."""
    scenario = generate_scenario(n_islands=16, n_cargos=15, n_pairs=3, seed=21)
    pairing = form_pairs(scenario.G, scenario.deliverer_positions, scenario.builder_positions)
    table = build_cost_table(scenario.G, pairing.pairs, scenario.cargos)
    a = assign_lpt(table, pairing.pairs, scenario.cargos, CARGO_HEURISTICS["direct"])

    # размер задачи при size_rule="min" -- стоимость у лучшего исполнителя
    sizes = {
        c.cargo_id: min(
            table[(k, c.cargo_id)].W_T for k in a.per_pair if (k, c.cargo_id) in table
        )
        for c in scenario.cargos
    }
    for queue in a.per_pair.values():
        ps = [a.priority[cid] for cid in queue]
        assert ps == sorted(ps, reverse=True), "очередь пары нарушает порядок LPT"
        assert all(abs(a.priority[cid] - sizes[cid]) < 1e-9 for cid in queue)

    # самый крупный груз распределяется первым -- он стоит в начале очереди
    # своей пары
    biggest = max(sizes, key=lambda cid: sizes[cid])
    assert a.per_pair[a.pair_of_cargo[biggest]][0] == biggest


def test_lpt_gives_cargo_to_least_loaded_pair():
    """Правило load -- классический Грэм: пока пара 1 простаивает, второй груз
    уходит ей, хотя для неё он в 42 раза дороже."""
    _G, cargos, pairing = _lpt_two_cargos_near_one_pair()
    table = build_cost_table(_G, pairing.pairs, cargos)
    a = assign_lpt(table, pairing.pairs, cargos, CARGO_HEURISTICS["direct"], rule="load")
    assert sorted(len(q) for q in a.per_pair.values()) == [1, 1]


def test_lpt_completion_rule_accounts_for_pair_specific_cost():
    """Правило completion -- поправка на неидентичность пар: тот же груз
    остаётся у пары 0, потому что её момент ЗАВЕРШЕНИЯ (загрузка + стоимость)
    всё равно меньше, чем у простаивающей, но далёкой пары 1."""
    _G, cargos, pairing = _lpt_two_cargos_near_one_pair()
    table = build_cost_table(_G, pairing.pairs, cargos)
    a = assign_lpt(table, pairing.pairs, cargos, CARGO_HEURISTICS["direct"], rule="completion")
    assert sorted(len(q) for q in a.per_pair.values()) == [0, 2]

    loads = _pair_loads(a, table)
    lpt_load = assign_lpt(table, pairing.pairs, cargos, CARGO_HEURISTICS["direct"], rule="load")
    assert max(loads.values()) < max(_pair_loads(lpt_load, table).values())


def test_lpt_balances_cost_better_than_literal():
    """LPT минимизирует максимальную загрузку пары (makespan). Проверяется
    против буквального правила без балансировки -- на каждом сценарии, и
    против него же с балансировкой -- в среднем."""
    seeds = (4, 7, 11, 17, 21, 33)
    lpt_avg, balanced_avg = 0.0, 0.0
    for seed in seeds:
        scenario = generate_scenario(n_islands=16, n_cargos=15, n_pairs=3, seed=seed)
        pairing = form_pairs(scenario.G, scenario.deliverer_positions,
                             scenario.builder_positions)
        table = build_cost_table(scenario.G, pairing.pairs, scenario.cargos)

        def max_load(**kwargs):
            a = assign_cargos(table, pairing.pairs, scenario.cargos,
                              CARGO_HEURISTICS["direct"], **kwargs)
            return max(_pair_loads(a, table).values())

        lpt = max_load(mode="lpt", lpt_rule="completion")
        plain = max_load(mode="literal", balance="none")
        balanced = max_load(mode="literal", balance="load")

        assert lpt < plain, f"seed={seed}: LPT хуже нераспределённого literal"
        lpt_avg += lpt / len(seeds)
        balanced_avg += balanced / len(seeds)

    assert lpt_avg < balanced_avg


def test_lpt_target_load_is_reference_not_threshold():
    """В режиме lpt --balance не применяется: результат не зависит от него, а
    target_load возвращается как СПРАВОЧНАЯ идеальная загрузка (нижняя оценка
    makespan), а не как порог насыщения."""
    scenario = generate_scenario(n_islands=16, n_cargos=14, n_pairs=3, seed=7)
    pairing = form_pairs(scenario.G, scenario.deliverer_positions, scenario.builder_positions)
    table = build_cost_table(scenario.G, pairing.pairs, scenario.cargos)

    with_balance = assign_cargos(table, pairing.pairs, scenario.cargos,
                                 CARGO_HEURISTICS["direct"], mode="lpt", balance="load")
    without = assign_cargos(table, pairing.pairs, scenario.cargos,
                            CARGO_HEURISTICS["direct"], mode="lpt", balance="none")
    assert with_balance.per_pair == without.per_pair

    sizes_total = sum(
        min(table[(k, c.cargo_id)].W_T for k in with_balance.per_pair
            if (k, c.cargo_id) in table)
        for c in scenario.cargos
    )
    assert abs(with_balance.target_load - sizes_total / len(pairing.pairs)) < 1e-9
    # порога нет: пары получают грузы и после превышения target_load
    assert max(_pair_loads(with_balance, table).values()) > with_balance.target_load


def test_lpt_size_rules_change_only_the_order():
    """Правило размера задачи (min/mean/max) влияет на ПОРЯДОК разбора, но не
    на состав результата: каждый груз всё равно назначен ровно один раз."""
    scenario = generate_scenario(n_islands=16, n_cargos=15, n_pairs=3, seed=11)
    pairing = form_pairs(scenario.G, scenario.deliverer_positions, scenario.builder_positions)
    table = build_cost_table(scenario.G, pairing.pairs, scenario.cargos)

    results = {}
    for size_rule in ("min", "mean", "max"):
        a = assign_lpt(table, pairing.pairs, scenario.cargos,
                       CARGO_HEURISTICS["direct"], size_rule=size_rule)
        flat = [c for q in a.per_pair.values() for c in q]
        assert sorted(flat) == [c.cargo_id for c in scenario.cargos], size_rule
        results[size_rule] = a.priority

    # min и max дают разные размеры задач (пары неидентичны), значит и разные
    # ключи сортировки
    assert results["min"] != results["max"]


def test_lpt_inverse_heuristic_gives_reversed_order():
    """direct в режиме lpt -- собственно LPT (длинные работы первыми), inverse
    -- SPT (короткие первыми): порядки строго противоположны."""
    scenario = generate_scenario(n_islands=16, n_cargos=12, n_pairs=3, seed=17)
    pairing = form_pairs(scenario.G, scenario.deliverer_positions, scenario.builder_positions)
    table = build_cost_table(scenario.G, pairing.pairs, scenario.cargos)

    lpt = assign_lpt(table, pairing.pairs, scenario.cargos, CARGO_HEURISTICS["direct"])
    spt = assign_lpt(table, pairing.pairs, scenario.cargos, CARGO_HEURISTICS["inverse"])

    order_lpt = sorted(lpt.priority, key=lambda cid: -lpt.priority[cid])
    order_spt = sorted(spt.priority, key=lambda cid: -spt.priority[cid])
    assert order_lpt == list(reversed(order_spt))


def test_lpt_unknown_rules_raise():
    scenario = generate_scenario(n_islands=10, n_cargos=6, n_pairs=2, seed=3)
    pairing = form_pairs(scenario.G, scenario.deliverer_positions, scenario.builder_positions)
    table = build_cost_table(scenario.G, pairing.pairs, scenario.cargos)

    for kwargs in ({"rule": "bogus"}, {"size_rule": "bogus"}):
        try:
            assign_lpt(table, pairing.pairs, scenario.cargos,
                       CARGO_HEURISTICS["direct"], **kwargs)
            assert False, f"ожидался KeyError для {kwargs}"
        except KeyError:
            pass


def test_assignment_is_static_and_ignores_intermediate_moves():
    """Оценки Шага 3 берутся от НАЧАЛЬНЫХ позиций пары: W_T_static каждой
    записи расписания совпадает с ячейкой таблицы, посчитанной до старта, --
    т.е. промежуточные перемещения роботов в распределении не учтены."""
    scenario = generate_scenario(n_islands=14, n_cargos=9, n_pairs=3, seed=33)
    result = run_static(
        scenario.G, scenario.cargos, scenario.deliverer_positions,
        scenario.builder_positions, CARGO_HEURISTICS["direct"],
    )
    assert result.all_delivered

    pairing = form_pairs(scenario.G, scenario.deliverer_positions, scenario.builder_positions)
    table = build_cost_table(scenario.G, pairing.pairs, scenario.cargos)
    for r in result.T:
        assert abs(r.W_T_static - table[(r.pair_id, r.cargo_id)].W_T) < 1e-9


# --- Шаг 4: исполнение по раундам -----------------------------------------

def test_rounds_follow_per_pair_queues():
    """В раунде r каждая пара везёт ровно r-й груз своего списка, и ни одна
    пара не выполняет в раунде больше одной доставки."""
    scenario = generate_scenario(n_islands=16, n_cargos=13, n_pairs=3, seed=8)
    result = run_static(
        scenario.G, scenario.cargos, scenario.deliverer_positions,
        scenario.builder_positions, CARGO_HEURISTICS["direct"],
    )
    assert result.all_delivered

    by_round = {}
    for r in result.T:
        by_round.setdefault(r.round_index, []).append(r)
    for round_index, records in by_round.items():
        pair_ids = [r.pair_id for r in records]
        assert len(pair_ids) == len(set(pair_ids)), f"раунд {round_index}: пара везёт два груза"
        for r in records:
            assert result.plan[r.pair_id][round_index] == r.cargo_id

    assert result.n_rounds == max(len(q) for q in result.plan.values())


def test_bridges_never_rebuilt_across_rounds():
    scenario = generate_scenario(n_islands=16, n_cargos=14, n_pairs=3, seed=7)
    result = run_static(
        scenario.G, scenario.cargos, scenario.deliverer_positions,
        scenario.builder_positions, CARGO_HEURISTICS["direct"],
    )
    assert result.all_delivered
    all_won = [ek for r in result.T for ek in r.won_bridges]
    assert len(all_won) == len(set(all_won)), "один и тот же мост построен более одного раза"


def test_shared_bridge_is_paid_by_one_pair_only():
    """Обе пары обязаны построить один и тот же мост 1-2, чтобы довезти свои
    грузы: платит одна (случайно выбранная), для второй проезд бесплатен."""
    G = Graph()
    for i in range(4):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(1, 2, EdgeKind.BLOCKED, w_E=1.0, w_build=50.0, length=1.0)
    G.add_edge(2, 3, EdgeKind.FREE, w_E=1.0)
    cargos = [Cargo(cargo_id=0, start=0, finish=3), Cargo(cargo_id=1, start=1, finish=2)]

    result = run_static(
        G, cargos, deliverer_positions=[0, 1], builder_positions=[0, 1],
        cargo_heuristic=CARGO_HEURISTICS["direct"], balance="none",
    )
    assert result.all_delivered
    first_round = [r for r in result.T if r.round_index == 0]
    if len(first_round) == 2:
        bridge = G.edge_between(1, 2).key()
        owners = [r.pair_id for r in first_round if bridge in r.won_bridges]
        assert len(owners) == 1, "мост оплачен более чем одной парой"


def test_all_heuristics_and_modes_deliver_small_scenario():
    scenario = generate_scenario(n_islands=10, n_cargos=6, n_pairs=2, seed=123)
    for name, heuristic in CARGO_HEURISTICS.items():
        for mode in ("literal", "cheapest", "lpt"):
            for lpt_rule in ("load", "completion"):
                result = run_static(
                    scenario.G, scenario.cargos, scenario.deliverer_positions,
                    scenario.builder_positions, heuristic, assignment_mode=mode,
                    lpt_rule=lpt_rule,
                )
                assert result.feasible
                assert result.all_delivered, f"{name}/{mode} не доставил все грузы"
                if mode != "lpt":
                    break  # lpt_rule на остальные режимы не влияет


def test_unknown_heuristic_and_mode_raise():
    try:
        get_cargo_heuristic("bogus")
        assert False, "ожидался KeyError для неизвестной эвристики"
    except KeyError:
        pass

    scenario = generate_scenario(n_islands=8, n_cargos=4, n_pairs=2, seed=1)
    try:
        run_static(
            scenario.G, scenario.cargos, scenario.deliverer_positions,
            scenario.builder_positions, CARGO_HEURISTICS["direct"],
            assignment_mode="bogus",
        )
        assert False, "ожидался KeyError для неизвестного режима назначения"
    except KeyError:
        pass


def test_reproducible_with_same_rng_seed():
    """Случайный выбор пары, оплачивающей общий мост (Шаг 4), воспроизводим
    при одинаковом rng_seed."""
    scenario = generate_scenario(n_islands=14, n_cargos=10, n_pairs=3, seed=5)

    def run():
        result = run_static(
            scenario.G, scenario.cargos, scenario.deliverer_positions,
            scenario.builder_positions, CARGO_HEURISTICS["direct"], rng_seed=42,
        )
        return [
            (r.round_index, r.cargo_id, r.pair_id, r.W_d, r.W_b) for r in result.T
        ]

    assert run() == run()


# --- диагностика ----------------------------------------------------------

def test_estimated_raw_never_below_real():
    scenario = generate_scenario(n_islands=16, n_cargos=14, n_pairs=3, seed=7)
    result = run_static(
        scenario.G, scenario.cargos, scenario.deliverer_positions,
        scenario.builder_positions, CARGO_HEURISTICS["direct"],
    )
    assert result.all_delivered
    cargo_map = {c.cargo_id: c for c in scenario.cargos}
    bracket = compute_static_cost_bracket(scenario.G, result.T, cargo_map)
    assert bracket.estimated_raw >= bracket.real - 1e-6
    assert bracket.gap_raw >= -1e-6


def test_estimated_static_matches_sum_of_step3_estimates():
    scenario = generate_scenario(n_islands=16, n_cargos=12, n_pairs=3, seed=17)
    result = run_static(
        scenario.G, scenario.cargos, scenario.deliverer_positions,
        scenario.builder_positions, CARGO_HEURISTICS["direct"],
    )
    assert result.all_delivered
    cargo_map = {c.cargo_id: c for c in scenario.cargos}
    bracket = compute_static_cost_bracket(scenario.G, result.T, cargo_map)
    assert abs(bracket.estimated_static - result.estimated_static) < 1e-9
    assert abs(bracket.real - result.real) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\nВсе тесты пройдены ({len(tests)}).")
