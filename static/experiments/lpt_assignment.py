"""
АЛГОРИТМ 0 — Назначение грузов парам по эвристике LPT (Longest Processing
Time first), формализующее исходную задачу как задачу составления расписания
для МИНИМИЗАЦИИ ОБЩЕЙ СТОИМОСТИ ВЫПОЛНЕНИЯ ВСЕХ ОПЕРАЦИЙ.

ПОСТАНОВКА ЗАДАЧИ (см. обсуждение в чате):

  Дано: граф среды G, грузы C = {c_1,...,c_n} (n >> m), коалиции
        U = {U_1,...,U_m} (каждая - 1 доставщик + 1 строитель), жадный
        алгоритм построения мостов (Алгоритм A), дающий детерминированную
        W_T(U_k, c_i | текущая позиция U_k).

  Переменные решения:
    1) π: C -> U      - назначение "груз -> пара"
    2) σ_k            - порядок обслуживания очереди C_k = π^{-1}(U_k)

  Целевая функция:
    min_{π, {σ_k}}  Φ(π,σ) = Σ_k Σ_j W_T(U_k, σ_k(j) | v^{(j-1)}_k)

  где v^{(j-1)}_k - позиция коалиции U_k после выполнения (j-1)-й задачи
  в порядке σ_k (рекурсивно).

Это обобщение задачи параллельного машинного расписания (parallel machine
scheduling) с временами обработки, зависящими от последовательности
(sequence-dependent processing times) - NP-трудная задача даже при
фиксированном π. Полный перебор неподъёмен при n >> m, поэтому применяется
ДВУХУРОВНЕВАЯ ЖАДНАЯ ДЕКОМПОЗИЦИЯ:

  Шаг 1 (этот модуль)        - π через LPT по убыванию W_T^i (оценённого
                                от начальных позиций пар жадным методом A)
  Шаг 2 (core/algorithm_2.py) - σ_k через динамический приоритет внутри
                                каждой пары (уже реализовано)

LPT (Longest Processing Time first) - классическая эвристика теории
расписаний для P||C_max и related задач, с доказанной гарантией
приближения (4/3 - 1/(3m)) от оптимума для параллельных идентичных машин
(Graham, 1969). Здесь "время обработки" - это W_T^i, а "машины" - пары;
формальная гарантия Грэма строго применима к P||C_max (минимакс), но
эвристика переносится по аналогии и на min ΣW_T (минимизацию суммы),
что и проверяется здесь эмпирически.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))

from typing import Dict, List

from delivery_model import IslandGraph, Cargo, Pair
from heuristic_cheapest_bridge import find_route_cheapest_bridge


def estimate_initial_cost(cargo: Cargo, env: IslandGraph, pair: Pair, L: float) -> float:
    """
    W_T^i, оценённая жадным алгоритмом построения мостов (Алгоритм A) от
    НАЧАЛЬНОЙ позиции пары - используется только для ранжирования при
    назначении (шаг 1), не для итогового расписания (там стоимость
    пересчитывается от актуальной позиции на каждом шаге).
    """
    result = find_route_cheapest_bridge(env, cargo, pair, L)
    if not result.feasible:
        return float("inf")
    return result.W_d + result.W_b


def assign_by_lpt(cargos: List[Cargo], pairs: List[Pair], env: IslandGraph,
                   L: float) -> Dict[str, str]:
    """
    Algorithm 0  ASSIGN-LPT(C, 𝒰, G, L)
    Input:  грузы C, коалиции 𝒰 (на начальных позициях), граф G, ограничение L
    Output: назначение π: C -> 𝒰 (словарь {cargo_id: pair_id})

    1: for each c_i in C:
    2:     W_T^i <- ESTIMATE-INITIAL-COST(c_i, U_k=любая) -- здесь: оценка
            относительно НАЗНАЧАЕМОЙ на лету пары, см. ниже
    3: sort C by W_T^i in descending order            -- LPT: long jobs first
    4: load[U_k] <- 0 for all U_k                       -- текущая суммарная нагрузка пары
    5: for each c_i in sorted C:
    6:     U* <- argmin_{U_k} load[U_k]                -- пара с минимальной текущей нагрузкой
    7:     π(c_i) <- U*
    8:     load[U*] <- load[U*] + W_T^i(U*, c_i)        -- учёт назначенной стоимости
    9: return π

    ПРИМЕЧАНИЕ: на шаге 2 W_T^i технически зависит от пары (расстояние от
    позиции пары до груза разное для разных пар), поэтому здесь используется
    приближение: ранжирование производится по W_T^i ОТНОСИТЕЛЬНО БЛИЖАЙШЕЙ
    пары (это и определяет "размер" задачи безотносительно конкретного
    исполнителя), а назначение на шаге 6 - по минимальной текущей загрузке
    среди ВСЕХ пар (включая стоимость для конкретно этой пары).
    """
    # шаг 2: оцениваем "размер" задачи как минимум W_T^i по всем парам
    # (более точная версия, чем W_T от единственной произвольной пары)
    cost_estimates: Dict[str, float] = {}
    for c in cargos:
        costs = [estimate_initial_cost(c, env, p, L) for p in pairs]
        cost_estimates[c.id] = min(costs)

    # шаг 3: сортировка по убыванию (LPT)
    sorted_cargos = sorted(cargos, key=lambda c: cost_estimates[c.id], reverse=True)

    # шаги 4-8: жадное назначение паре с минимальной текущей нагрузкой
    load: Dict[str, float] = {p.id: 0.0 for p in pairs}
    assignment: Dict[str, str] = {}

    for c in sorted_cargos:
        best_pair = min(pairs, key=lambda p: load[p.id])
        cost_for_best = estimate_initial_cost(c, env, best_pair, L)
        assignment[c.id] = best_pair.id
        load[best_pair.id] += cost_for_best if cost_for_best != float("inf") else 0.0

    return assignment


def assign_weighted(cargos: List[Cargo], pairs: List[Pair], env: IslandGraph,
                     L: float, lam: float) -> Dict[str, str]:
    """
    Algorithm 0''  ASSIGN-WEIGHTED(C, 𝒰, G, L, λ)
    Input:  грузы C, коалиции 𝒰 (начальные позиции), граф G, ограничение L,
            параметр баланса λ ∈ [0, 1]
    Output: назначение π: C → 𝒰

    Обобщает Greedy Nearest (λ=1, чистая минимизация стоимости назначения)
    и жадную балансировку нагрузки (λ=0, чистая минимизация максимальной
    загрузки) через выпуклую комбинацию НОРМИРОВАННЫХ критериев:

        U*(λ) = argmin_{U_k} [ λ · ŵ_T(U_k, c_i) + (1-λ) · load̂(U_k) ]

    где ŵ_T, load̂ ∈ [0,1] - критерии, нормированные ПО ТЕКУЩЕМУ шагу
    (минимум/максимум среди пар-кандидатов прямо сейчас), что позволяет
    складывать величины разного масштаба и физического смысла.

    1: v_δ[U_k], v_β[U_k] ← начальные позиции,  for all U_k ∈ 𝒰
    2: load[U_k] ← 0,  for all U_k ∈ 𝒰
    3: π ← ∅
    4: for each c_i ∈ C (в порядке поступления) do
    5:     W ← { W_T(U_k, c_i | v_δ[U_k], v_β[U_k]) : U_k ∈ 𝒰 }
    6:     w_min, w_max ← min(W), max(W)
    7:     l_min, l_max ← min(load), max(load)
    8:     for each U_k ∈ 𝒰 do
    9:         ŵ_T(U_k) ← (W[U_k] - w_min) / (w_max - w_min + ε)
    10:        load̂(U_k) ← (load[U_k] - l_min) / (l_max - l_min + ε)
    11:        score(U_k) ← λ·ŵ_T(U_k) + (1-λ)·load̂(U_k)
    12:    U* ← argmin_{U_k} score(U_k)
    13:    π(c_i) ← U*
    14:    load[U*] ← load[U*] + W[U*]
    15:    обновить v_δ[U*], v_β[U*] (виртуальное продвижение позиции)
    16: return π
    """
    import copy
    EPS = 1e-9
    virtual_pairs = {p.id: copy.deepcopy(p) for p in pairs}
    load: Dict[str, float] = {p.id: 0.0 for p in pairs}
    assignment: Dict[str, str] = {}

    for c in cargos:
        costs = {pid: estimate_initial_cost(c, env, vp, L)
                 for pid, vp in virtual_pairs.items()}
        w_min, w_max = min(costs.values()), max(costs.values())
        l_min, l_max = min(load.values()), max(load.values())

        def score(pid: str) -> float:
            w_hat = (costs[pid] - w_min) / (w_max - w_min + EPS)
            l_hat = (load[pid] - l_min) / (l_max - l_min + EPS)
            return lam * w_hat + (1 - lam) * l_hat

        best_pid = min(virtual_pairs.keys(), key=score)
        assignment[c.id] = best_pid
        load[best_pid] += costs[best_pid] if costs[best_pid] != float("inf") else 0.0

        result = find_route_cheapest_bridge(env, c, virtual_pairs[best_pid], L)
        if result.feasible:
            virtual_pairs[best_pid].deliverer_pos = c.v_finish
            if result.bridges:
                virtual_pairs[best_pid].builder_pos = result.bridges[-1][1]
            else:
                virtual_pairs[best_pid].builder_pos = c.v_finish

    return assignment


def assign_round_robin(cargos: List[Cargo], pairs: List[Pair]) -> Dict[str, str]:
    """
    БАЗОВОЕ (наивное) назначение для сравнения - циклическое распределение
    без учёта стоимости (текущая реализация scenario_generator.py).
    """
    n_pairs = len(pairs)
    return {c.id: pairs[i % n_pairs].id for i, c in enumerate(cargos)}


def assign_greedy_nearest(cargos: List[Cargo], pairs: List[Pair], env: IslandGraph,
                           L: float) -> Dict[str, str]:
    """
    Algorithm 0'  ASSIGN-GREEDY-NEAREST(C, 𝒰, G, L)
    Input:  грузы C, коалиции 𝒰 (на начальных позициях), граф G, ограничение L
    Output: назначение π: C -> 𝒰

    В ОТЛИЧИЕ ОТ LPT: LPT (assign_by_lpt) минимизирует MAKESPAN (максимальную
    загрузку среди пар) - это классический критерий Грэма для P||C_max, с
    доказанной гарантией приближения. НО наша целевая функция - это СУММА
    Φ = Σ_k Σ_i W_T^i, а не максимум. Если бы W_T^i не зависела от текущей
    позиции пары (как в классическом parallel scheduling), сумма была бы
    ИНВАРИАНТНА к назначению π вообще - любое разбиение даёт одну и ту же
    сумму, просто перегруппированную. У нас W_T^i ЗАВИСИТ от позиции пары
    в момент выполнения, и именно это создаёт зависимость суммы от π - но
    эта зависимость никак не связана с балансировкой максимума, поэтому
    LPT для неё не оптимизирован и может даже проигрывать наивному RR
    (эмпирически подтверждено).

    Greedy nearest assignment - простая альтернатива, явно целящаяся в
    минимизацию СУММЫ: каждый груз (в произвольном порядке обхода)
    назначается паре, для которой его обслуживание ОБОЙДЁТСЯ ДЕШЕВЛЕ ВСЕГО
    ПРЯМО СЕЙЧАС (с учётом уже сделанных назначений - позиция пары
    "виртуально" обновляется после каждого назначения, как и в реальном
    исполнении).

    1: load_pos[U_k] <- начальная позиция U_k,  for all U_k
    2: π <- {}
    3: for each c_i in C (в порядке поступления):
    4:     U* <- argmin_{U_k} W_T(U_k, c_i | load_pos[U_k])
    5:     π(c_i) <- U*
    6:     load_pos[U*] <- v_f^{c_i}   (виртуально продвигаем позицию пары)
    7: return π
    """
    import copy
    virtual_pairs = {p.id: copy.deepcopy(p) for p in pairs}
    assignment: Dict[str, str] = {}

    for c in cargos:
        best_pair_id, best_cost = None, float("inf")
        for pid, vp in virtual_pairs.items():
            cost = estimate_initial_cost(c, env, vp, L)
            if cost < best_cost:
                best_cost, best_pair_id = cost, pid

        assignment[c.id] = best_pair_id

        # виртуально продвигаем позицию выбранной пары к финишу груза
        # (приближение - не учитывает построенные мосты, только конечную точку)
        result = find_route_cheapest_bridge(env, c, virtual_pairs[best_pair_id], L)
        if result.feasible:
            virtual_pairs[best_pair_id].deliverer_pos = c.v_finish
            if result.bridges:
                virtual_pairs[best_pair_id].builder_pos = result.bridges[-1][1]
            else:
                virtual_pairs[best_pair_id].builder_pos = c.v_finish

    return assignment


def apply_assignment(cargos: List[Cargo], assignment: Dict[str, str]) -> List[Cargo]:
    """Возвращает КОПИЮ списка грузов с применённым полем assigned_pair."""
    import copy
    new_cargos = copy.deepcopy(cargos)
    for c in new_cargos:
        c.assigned_pair = assignment[c.id]
    return new_cargos
