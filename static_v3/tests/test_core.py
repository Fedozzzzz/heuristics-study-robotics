"""
Базовые тесты static_v3: маршрутизация и оценка стоимости (те же
ROUTE-AND-COST и ESTIMATE-TASK-COST, что в static_v2/dynamic_v2), Шаг 0
(проверка достижимости), Шаг 2 (СЛУЧАЙНОЕ паросочетание), Шаг 3 (СЛУЧАЙНОЕ
распределение грузов, режимы balanced/uniform), Шаг 4 (структура раундов,
общие мосты, статичность плана), Шаг 5 (итоговая стоимость) и
воспроизводимость всей случайности по rng_seed.
"""

import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot_delivery_random.assignment import ASSIGNMENT_MODES, assign_cargos_random
from robot_delivery_random.costs import estimate_task_cost, route_and_cost
from robot_delivery_random.graph import EdgeKind, Graph
from robot_delivery_random.model import Cargo
from robot_delivery_random.pairing import form_pairs_random
from robot_delivery_random.scenario import generate_scenario
from robot_delivery_random.scheduler import run_random_static


def make_simple_graph():
    """0 --free(w=1)-- 1 --blocked(len=5,w_build=10,w_E=2)-- 2"""
    G = Graph()
    G.add_node(0, w_V=0.0)
    G.add_node(1, w_V=0.0)
    G.add_node(2, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(1, 2, EdgeKind.BLOCKED, w_E=2.0, w_build=10.0, length=5.0)
    return G


def run_scenario(scenario, **kwargs):
    return run_random_static(
        scenario.G, scenario.cargos,
        scenario.deliverer_positions, scenario.builder_positions,
        **kwargs,
    )


# --- ядро (перенесено из static_v2 без изменений) -------------------------

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

    result = run_random_static(G, cargos, [0], [1])

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

    result = run_random_static(G, cargos, [0], [3])

    assert not result.feasible
    assert result.infeasibility_reason


# --- Шаг 2: случайное формирование пар ------------------------------------

def test_pairing_is_a_perfect_matching():
    """Каждый робот входит максимум в одну пару, пар ровно min(|R_d|, |R_b|)."""
    pairing = form_pairs_random([0, 1, 2], [3, 4, 5], random.Random(0))
    assert len(pairing.pairs) == 3
    assert len({p.deliverer_id for p in pairing.pairs}) == 3
    assert len({p.builder_id for p in pairing.pairs}) == 3
    assert not pairing.unpaired_deliverers and not pairing.unpaired_builders


def test_pairing_leaves_extra_robots_unpaired():
    pairing = form_pairs_random([0, 1], [0], random.Random(0))
    assert len(pairing.pairs) == 1
    assert len(pairing.unpaired_deliverers) == 1
    assert not pairing.unpaired_builders


def test_pairing_positions_match_robot_ids():
    """deliverer_pos/builder_pos пары -- это позиции именно тех роботов, чьи id
    в ней записаны (перемешивание не должно рассогласовать id и позицию)."""
    d_pos, b_pos = [10, 11, 12], [20, 21, 22]
    pairing = form_pairs_random(d_pos, b_pos, random.Random(7))
    for p in pairing.pairs:
        assert p.deliverer_pos == d_pos[p.deliverer_id]
        assert p.builder_pos == b_pos[p.builder_id]


def test_pairing_is_reproducible_and_actually_random():
    """Одинаковый сид -- одно и то же паросочетание; по разным сидам
    встречаются РАЗНЫЕ паросочетания (иначе правило не случайное)."""
    def matching(seed):
        pairing = form_pairs_random([0, 1, 2], [0, 1, 2], random.Random(seed))
        return tuple(sorted((p.deliverer_id, p.builder_id) for p in pairing.pairs))

    assert matching(3) == matching(3)
    assert len({matching(s) for s in range(60)}) > 1


def test_pairing_covers_all_permutations():
    """Случайное паросочетание не должно быть смещено: на 3x3 роботах за
    достаточное число сидов должны встретиться все 3! = 6 сопоставлений."""
    seen = set()
    for seed in range(300):
        pairing = form_pairs_random([0, 1, 2], [0, 1, 2], random.Random(seed))
        seen.add(tuple(sorted((p.deliverer_id, p.builder_id) for p in pairing.pairs)))
    assert len(seen) == 6, f"встретилось только {len(seen)} из 6 паросочетаний"


def test_pairing_is_fixed_for_the_whole_run():
    """Модель статическая: пара не пересобирается -- один и тот же доставщик за
    весь прогон работает с одним и тем же строителем."""
    scenario = generate_scenario(n_islands=14, n_cargos=10, n_pairs=3, seed=11)
    result = run_scenario(scenario)
    assert result.all_delivered
    partner = {}
    for r in result.T:
        partner.setdefault(r.deliverer_id, r.builder_id)
        assert partner[r.deliverer_id] == r.builder_id


# --- Шаг 3: случайное распределение грузов --------------------------------

def _cargos(n):
    return [Cargo(cargo_id=i, start=0, finish=1) for i in range(n)]


def _pairs(n):
    return form_pairs_random(list(range(n)), list(range(n)), random.Random(0)).pairs


def test_assignment_covers_every_cargo_exactly_once():
    cargos = _cargos(12)
    pairs = _pairs(3)
    for mode in ASSIGNMENT_MODES:
        a = assign_cargos_random(pairs, cargos, random.Random(1), mode=mode)
        flat = [c for lst in a.per_pair.values() for c in lst]
        assert sorted(flat) == [c.cargo_id for c in cargos], mode
        assert not a.unassigned, mode
        assert set(a.per_pair) == {p.pair_id for p in pairs}, mode


def test_assignment_balanced_equalizes_cargo_counts():
    """balanced: число грузов у пар отличается максимум на 1 при любом сиде."""
    pairs = _pairs(3)
    for n_cargos in (3, 7, 12, 20):
        cargos = _cargos(n_cargos)
        for seed in range(20):
            a = assign_cargos_random(pairs, cargos, random.Random(seed), mode="balanced")
            loads = [len(q) for q in a.per_pair.values()]
            assert max(loads) - min(loads) <= 1, (n_cargos, seed, loads)


def test_assignment_uniform_can_be_unbalanced():
    """uniform: раздача мультиномиальная, поэтому перекос числа грузов больше
    единицы обязан встречаться (иначе режим ничем не отличался бы от
    balanced)."""
    pairs = _pairs(3)
    cargos = _cargos(12)
    spreads = [
        max(len(q) for q in a.per_pair.values()) - min(len(q) for q in a.per_pair.values())
        for a in (
            assign_cargos_random(pairs, cargos, random.Random(seed), mode="uniform")
            for seed in range(50)
        )
    ]
    assert max(spreads) > 1


def test_assignment_is_random_not_by_cargo_id():
    """Распределение не должно вырождаться в детерминированное "по порядку
    id": у одного и того же груза при разных сидах должны встречаться разные
    пары."""
    pairs = _pairs(3)
    cargos = _cargos(9)
    for mode in ASSIGNMENT_MODES:
        pair_of_first = set()
        for seed in range(40):
            a = assign_cargos_random(pairs, cargos, random.Random(seed), mode=mode)
            for pair_id, queue in a.per_pair.items():
                if cargos[0].cargo_id in queue:
                    pair_of_first.add(pair_id)
        assert len(pair_of_first) > 1, mode


def test_assignment_order_within_queue_is_shuffled():
    """Порядок внутри очереди пары -- это порядок раундов Шага 4, и он тоже
    разыгрывается: очередь не обязана быть возрастающей по cargo_id."""
    pairs = _pairs(2)
    cargos = _cargos(20)
    for mode in ASSIGNMENT_MODES:
        unsorted_seen = False
        for seed in range(20):
            a = assign_cargos_random(pairs, cargos, random.Random(seed), mode=mode)
            if any(q != sorted(q) for q in a.per_pair.values()):
                unsorted_seen = True
                break
        assert unsorted_seen, mode


def test_assignment_is_reproducible():
    pairs = _pairs(3)
    cargos = _cargos(15)
    for mode in ASSIGNMENT_MODES:
        a = assign_cargos_random(pairs, cargos, random.Random(9), mode=mode)
        b = assign_cargos_random(pairs, cargos, random.Random(9), mode=mode)
        assert a.per_pair == b.per_pair, mode


def test_assignment_unknown_mode_raises():
    try:
        assign_cargos_random(_pairs(2), _cargos(4), random.Random(0), mode="bogus")
        assert False, "ожидался ValueError для неизвестного режима распределения"
    except ValueError:
        pass


def test_assignment_does_not_touch_the_graph():
    """Шаг 3 в static_v3 не обращается к среде вообще: ни графа, ни позиций
    роботов в сигнатуре нет, поэтому распределение на разных графах при одном
    сиде совпадает."""
    cargos = _cargos(10)
    pairs = _pairs(3)
    a = assign_cargos_random(pairs, cargos, random.Random(4), mode="balanced")
    b = assign_cargos_random(pairs, cargos, random.Random(4), mode="balanced")
    assert a.per_pair == b.per_pair


# --- Шаг 4: исполнение по раундам -----------------------------------------

def test_rounds_follow_per_pair_queues():
    """В раунде r каждая пара везёт ровно r-й груз своего списка, и ни одна
    пара не выполняет в раунде больше одной доставки."""
    scenario = generate_scenario(n_islands=16, n_cargos=13, n_pairs=3, seed=8)
    result = run_scenario(scenario)
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


def test_plan_is_static_and_matches_execution():
    """План Шага 3 не пересматривается: множество грузов, фактически
    выполненных парой, в точности совпадает с её очередью."""
    scenario = generate_scenario(n_islands=16, n_cargos=14, n_pairs=3, seed=29)
    for mode in ASSIGNMENT_MODES:
        result = run_scenario(scenario, assignment_mode=mode)
        assert result.all_delivered, mode
        done = {}
        for r in result.T:
            done.setdefault(r.pair_id, []).append(r.cargo_id)
        for pair_id, queue in result.plan.items():
            assert sorted(done.get(pair_id, [])) == sorted(queue), (mode, pair_id)


def test_bridges_never_rebuilt_across_rounds():
    scenario = generate_scenario(n_islands=16, n_cargos=14, n_pairs=3, seed=7)
    result = run_scenario(scenario)
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
    bridge = G.edge_between(1, 2).key()

    # балансировка balanced гарантирует по одному грузу на пару, то есть обе
    # доставки попадают в один раунд и конфликтуют за мост
    for rng_seed in range(10):
        result = run_random_static(
            G, cargos, deliverer_positions=[0, 1], builder_positions=[0, 1],
            assignment_mode="balanced", rng_seed=rng_seed,
        )
        assert result.all_delivered
        first_round = [r for r in result.T if r.round_index == 0]
        assert len(first_round) == 2
        owners = [r.pair_id for r in first_round if bridge in r.won_bridges]
        assert len(owners) == 1, "мост оплачен более чем одной парой"


def test_shared_bridge_winner_is_random():
    """Плательщик за общий мост выбирается случайно: по разным rng_seed
    выигрывают разные пары."""
    G = Graph()
    for i in range(4):
        G.add_node(i, w_V=0.0)
    G.add_edge(0, 1, EdgeKind.FREE, w_E=1.0)
    G.add_edge(1, 2, EdgeKind.BLOCKED, w_E=1.0, w_build=50.0, length=1.0)
    G.add_edge(2, 3, EdgeKind.FREE, w_E=1.0)
    cargos = [Cargo(cargo_id=0, start=0, finish=3), Cargo(cargo_id=1, start=1, finish=2)]
    bridge = G.edge_between(1, 2).key()

    winners = Counter()
    for rng_seed in range(40):
        result = run_random_static(
            G, cargos, deliverer_positions=[0, 1], builder_positions=[0, 1],
            assignment_mode="balanced", rng_seed=rng_seed,
        )
        for r in result.T:
            if r.round_index == 0 and bridge in r.won_bridges:
                winners[r.deliverer_id] += 1
    assert len(winners) > 1, f"мост всегда строит один и тот же робот: {winners}"


# --- Шаг 5: итоговая стоимость и воспроизводимость -------------------------

def test_real_is_sum_of_operation_costs():
    """Шаг 5: real = W_d_total + W_b_total и совпадает с суммой по записям
    плана доставок."""
    scenario = generate_scenario(n_islands=16, n_cargos=12, n_pairs=3, seed=17)
    result = run_scenario(scenario)
    assert result.all_delivered
    assert abs(result.real - (result.W_d_total + result.W_b_total)) < 1e-9
    assert abs(result.real - sum(r.W_d + r.W_b for r in result.T)) < 1e-9
    assert result.real > 0


def test_both_modes_deliver_small_scenario():
    scenario = generate_scenario(n_islands=10, n_cargos=6, n_pairs=2, seed=123)
    for mode in ASSIGNMENT_MODES:
        for rng_seed in range(5):
            result = run_scenario(scenario, assignment_mode=mode, rng_seed=rng_seed)
            assert result.feasible, (mode, rng_seed)
            assert result.all_delivered, f"{mode}/{rng_seed} не доставил все грузы"


def test_unknown_assignment_mode_raises():
    scenario = generate_scenario(n_islands=8, n_cargos=4, n_pairs=2, seed=1)
    try:
        run_scenario(scenario, assignment_mode="bogus")
        assert False, "ожидался ValueError для неизвестного режима распределения"
    except ValueError:
        pass


def test_reproducible_with_same_rng_seed():
    """ВСЯ случайность модели (Шаги 2, 3 и 4) воспроизводима по rng_seed."""
    scenario = generate_scenario(n_islands=14, n_cargos=10, n_pairs=3, seed=5)

    def run():
        result = run_scenario(scenario, rng_seed=42)
        return (
            [(r.round_index, r.cargo_id, r.pair_id, r.W_d, r.W_b) for r in result.T],
            {k: list(v) for k, v in result.plan.items()},
            result.real,
        )

    assert run() == run()


def test_different_rng_seed_changes_the_plan():
    """Разные сиды дают разные планы и, как правило, разную итоговую стоимость
    -- это и есть дисперсия, ради которой в экспериментах существует
    --n-repeats."""
    scenario = generate_scenario(n_islands=16, n_cargos=12, n_pairs=3, seed=13)
    plans, costs = set(), set()
    for rng_seed in range(15):
        result = run_scenario(scenario, rng_seed=rng_seed)
        assert result.all_delivered
        plans.add(tuple(sorted((k, tuple(v)) for k, v in result.plan.items())))
        costs.add(round(result.real, 6))
    assert len(plans) > 1
    assert len(costs) > 1


def test_rng_streams_are_independent():
    """Потоки случайности Шагов 2/3/4 разведены: сценарии с разным числом
    грузов при одном rng_seed дают ОДНО И ТО ЖЕ паросочетание (иначе число
    вызовов на Шаге 3 сдвигало бы розыгрыш Шага 2, и сравнивать точки развёртки
    было бы нельзя)."""
    matchings = set()
    for n_cargos in (4, 8, 16):
        scenario = generate_scenario(n_islands=14, n_cargos=n_cargos, n_pairs=3, seed=2)
        result = run_scenario(scenario, rng_seed=99)
        matchings.add(
            tuple(sorted((p.deliverer_id, p.builder_id) for p in result.pairs))
        )
    assert len(matchings) == 1, f"паросочетание поехало от числа грузов: {matchings}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\nВсе тесты пройдены ({len(tests)}).")
