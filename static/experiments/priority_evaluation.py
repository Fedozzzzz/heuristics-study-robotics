"""
Оценка эвристики приоритета "нормированная стоимость" W_T^i / sum(W_T^j).

Пайплайн:
  1) Для каждого груза c_i считается W_T^i = W_d^i + W_b^i ОДНОЙ И ТОЙ ЖЕ
     парой (доставщик+строитель), стоящей в СВОЕЙ НАЧАЛЬНОЙ позиции
     (т.е. как если бы каждый груз был "первым" - это и есть ЭВРИСТИЧЕСКАЯ,
     предварительная оценка, не учитывающая, что пара после доставки i-го
     груза физически окажется в другом месте перед (i+1)-м).
     Маршрут строится ЖАДНЫМ алгоритмом построения мостов (минимум
     суммарной стоимости строительства, без учёта стоимости проезда при
     выборе пути) - heuristic_cheapest_bridge.find_route_cheapest_bridge.

  1b) (опционально) Штраф за переход к следующему грузу той же пары -
     compute_transition_penalty + apply_transition_penalty. Порядок
     выполнения на шаге 1 ещё не известен, поэтому штраф для груза c_i
     берётся как СРЕДНЕЕ графовое расстояние от v_finish^i до v_start^j
     по всем остальным грузам c_j, закреплённым за той же парой - грубая,
     не зависящая от конкретного порядка оценка "куда ехать дальше".
     Прибавляется ОТДЕЛЬНО к W_d^i и к W_b^i (оба робота в паре "платят"
     за переход).

  2) Приоритет: p(c_i) = W_T^i / sum_j(W_T^j) - нормированная стоимость,
     доля общего "бюджета" стоимости (см. обсуждение в начале разговора).
     Чем больше доля - тем выше приоритет (выполняется раньше) ИЛИ чем
     меньше доля - тем выше приоритет (выполняется раньше); оба варианта
     поддержаны через параметр direction.

  3) Строится граф приоритетов (k уровней слева-направо, как раньше).

  4) РЕАЛЬНОЕ расписание: пара выполняет грузы СТРОГО по убыванию приоритета,
     одна за другой. После каждой доставки позиция пары (доставчик,
     строитель) обновляется на фактическое место завершения. Стоимость
     КАЖДОЙ задачи в реальном расписании пересчитывается ЗАНОВО (тем же
     жадным алгоритмом, но уже от ТЕКУЩЕЙ, а не начальной позиции пары) -
     эта реальная стоимость штраф НЕ включает, так как она и без него уже
     точно учитывает фактическое перемещение.

  5) Сравнение: ЭВРИСТИЧЕСКАЯ оценка суммы (сумма W_T^i, посчитанных в
     шаге 1/1b, от начальной позиции) против РЕАЛЬНОЙ суммы (сумма
     стоимостей из шага 4, с учётом фактического смещения пары). Разница
     показывает, насколько предсказание "как если бы каждый груз был
     первым" (плюс штраф за переход) далеко от истинной стоимости
     последовательного выполнения.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))
import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import networkx as nx

from delivery_model import IslandGraph, Cargo, Pair, TaskResult
from heuristic_cheapest_bridge import find_route_cheapest_bridge


# ---------------------------------------------------------------------------
# Шаг 1 - эвристическая оценка стоимости каждой задачи (от начальной позиции)
# ---------------------------------------------------------------------------

def estimate_task_costs(env: IslandGraph, cargos: List[Cargo], pairs: List[Pair],
                         L: float) -> Dict[str, TaskResult]:
    """
    Для каждого груза считает W_T^i = W_d^i + W_b^i, используя ЖАДНЫЙ
    алгоритм построения мостов, от НАЧАЛЬНОЙ позиции назначенной пары
    (т.е. без учёта порядка выполнения - "как если бы груз был первым").
    """
    pairs_by_id = {p.id: p for p in pairs}
    results: Dict[str, TaskResult] = {}
    for c in cargos:
        pair = pairs_by_id[c.assigned_pair]
        res = find_route_cheapest_bridge(env, c, pair, L)
        results[c.id] = res
    return results


# ---------------------------------------------------------------------------
# Шаг 1b - штраф за переход к следующему грузу той же пары (без знания порядка)
# ---------------------------------------------------------------------------

def compute_transition_penalty(env: IslandGraph, cargos: List[Cargo],
                                pairs: List[Pair], L: float) -> Dict[str, float]:
    """
    Штраф за подъезд/возврат, который estimate_task_costs (шаг 1) не
    учитывает: оценка W_T^i там считается "как если бы груз был первым",
    то есть от НАЧАЛЬНОЙ позиции пары - а после выполнения предыдущего
    груза пара физически окажется не в начальной позиции, а в v_finish
    какого-то ДРУГОГО груза той же пары. Порядок выполнения на этом шаге
    ещё не известен (он сам зависит от приоритета, который мы только
    собираемся построить), поэтому штраф для груза c_i не привязывается к
    конкретному "предыдущему" грузу, а усредняется по всем грузам,
    потенциально закреплённым за той же парой:

        penalty(c_i) = mean_{j in C_k, j != i} dist_graph(v_finish^i, v_start^j)

    где C_k - множество грузов, назначенных той же паре, что и c_i.
    Физический смысл: "куда в среднем нужно будет ехать дальше после
    этого груза, если смотреть на все задачи, которые эта пара должна
    выполнить".

    Расстояние - кратчайший путь по ТОМУ ЖЕ приведённому графу G' (вес
    w_hat = w_build + w_E для непостроенных мостов, длиннее L исключены),
    что используется при расчёте самих W_d/W_b (env.build_weighted_graph),
    чтобы штраф был в тех же единицах стоимости, а не в "сыром" w_E без
    учёта необходимости строительства. Если для груза нет других грузов в
    его паре, либо v_finish^i не достижим ни до одного v_start^j в этом
    графе, штраф равен 0.0 (нет дополнительной информации о следующем шаге).

    Возвращает {cargo_id: penalty}, ОДНО значение на груз (применяется
    отдельно к W_d и к W_b - см. apply_transition_penalty).
    """
    Gp = env.build_weighted_graph(L)

    cargos_by_pair: Dict[str, List[Cargo]] = {}
    for c in cargos:
        cargos_by_pair.setdefault(c.assigned_pair, []).append(c)

    # кэш кратчайших расстояний от каждого v_finish до всех остальных вершин,
    # чтобы не пересчитывать Дейкстру много раз для одной и той же вершины
    dist_cache: Dict[int, Dict[int, float]] = {}

    def shortest_dist(v_from: int, v_to: int) -> float:
        if v_from not in dist_cache:
            if v_from in Gp:
                dist_cache[v_from] = nx.single_source_dijkstra_path_length(
                    Gp, v_from, weight="weight")
            else:
                dist_cache[v_from] = {}
        d = dist_cache[v_from].get(v_to)
        return d if d is not None else math.inf

    penalty: Dict[str, float] = {}
    for c_i in cargos:
        others = [c_j for c_j in cargos_by_pair.get(c_i.assigned_pair, [])
                  if c_j.id != c_i.id]
        if not others:
            penalty[c_i.id] = 0.0
            continue

        dists = [shortest_dist(c_i.v_finish, c_j.v_start) for c_j in others]
        finite = [d for d in dists if math.isfinite(d)]
        penalty[c_i.id] = (sum(finite) / len(finite)) if finite else 0.0

    return penalty


def apply_transition_penalty(task_costs: Dict[str, TaskResult],
                              penalty: Dict[str, float]) -> Dict[str, TaskResult]:
    """
    Возвращает НОВЫЙ словарь {cargo_id: TaskResult} (исходный task_costs не
    модифицируется), где у каждого результата W_d и W_b увеличены на
    penalty(c_i) каждый - штраф применяется ОТДЕЛЬНО к доставщику и к
    строителю (оба робота в паре "платят" за переход к следующему грузу).

    Недостижимые задачи (feasible=False) не штрафуются - они и так не
    участвуют в дальнейших расчётах приоритета.
    """
    penalized: Dict[str, TaskResult] = {}
    for cid, r in task_costs.items():
        r2 = copy.deepcopy(r)
        if r2.feasible:
            p = penalty.get(cid, 0.0)
            r2.W_d = r2.W_d + p
            r2.W_b = r2.W_b + p
        penalized[cid] = r2
    return penalized


def compute_pool_corrected_costs(env: IslandGraph, cargos: List[Cargo],
                                  task_costs: Dict[str, TaskResult]
                                  ) -> Dict[str, TaskResult]:
    """
    Корректировка оценки W_b^i с учётом переиспользования мостов внутри пары
    (Вариант B: вычитаем стоимость повторных построек из уже посчитанных W_b^i).

    Для каждой пары k считается "пул стоимости строительства":

        W_b^pool(k) = sum_i W_b^i  -  sum_e (c_e - 1) * w_build(e)

    где c_e - число грузов пары k, требующих моста e (из result.bridges всех
    трёх шагов), w_build(e) - стоимость строительства этого моста.
    Смысл: каждый уникальный мост платится ровно один раз, а не c_e раз.
    Переезды строителя между мостами не пересчитываются (берутся из исходных
    W_b^i) - это приближение.

    Скорректированная оценка для груза c_i:

        W_b_hat^i = W_b^pool(k) * W_b^i / sum_j W_b^j

    то есть пул распределяется пропорционально индивидуальным W_b^i.

    Возвращает новый словарь TaskResult с обновлёнными W_b (W_d не меняется).
    Если у пары все грузы имеют W_b=0 (мостов нет), скорректированные W_b
    тоже равны 0.
    """
    # собираем грузы по парам
    cargos_by_pair: Dict[str, List[Cargo]] = {}
    for c in cargos:
        cargos_by_pair.setdefault(c.assigned_pair, []).append(c)

    corrected: Dict[str, TaskResult] = {cid: copy.deepcopy(r)
                                         for cid, r in task_costs.items()}

    for pair_id, pair_cargos in cargos_by_pair.items():
        feasible_cargos = [c for c in pair_cargos
                           if task_costs[c.id].feasible]
        if not feasible_cargos:
            continue

        # считаем c_e - сколько грузов пары требуют каждого моста
        bridge_count: Dict[Tuple[int, int], int] = {}
        for c in feasible_cargos:
            r = task_costs[c.id]
            for (u, v) in r.bridges:
                key = (min(u, v), max(u, v))
                bridge_count[key] = bridge_count.get(key, 0) + 1

        # стоимость повторных построек, которую нужно вычесть
        duplicate_cost = sum(
            (count - 1) * env.G.edges[u, v]["w_build"]
            for (u, v), count in bridge_count.items()
            if env.G.has_edge(u, v)
        )

        # W_b^pool = sum(W_b^i) - duplicate_cost
        wb_sum = sum(task_costs[c.id].W_b for c in feasible_cargos)
        wb_pool = wb_sum - duplicate_cost

        # распределяем пул пропорционально W_b^i
        if wb_sum > 0:
            for c in feasible_cargos:
                r = corrected[c.id]
                r.W_b = wb_pool * (task_costs[c.id].W_b / wb_sum)
        # если wb_sum == 0, мостов нет, W_b уже равны 0 — ничего не меняем

    return corrected


def compute_lower_bound(env: IslandGraph, cargos: List[Cargo], pairs: List[Pair],
                        L: float) -> float:
    """
    НИЖНЯЯ оценка суммарной стоимости, ТОЧНАЯ при 1 грузе на пару.

    Для каждой пары k:
      - берём САМУЮ ДОРОГУЮ её задачу целиком: max_i W_T^i (i in C_k), где
        W_T^i = W_d^i + W_b^i посчитаны от ИСХОДНОЙ позиции пары
        (estimate_task_costs). Эту задачу всё равно придётся выполнить со
        всем её строительством, поэтому её полная стоимость неизбежна;
      - для ОСТАЛЬНЫХ грузов пары добавляем только проезд доставщика W_d^j
        (перевезти каждый груз физически необходимо), БЕЗ их строительства -
        мосты могут быть переиспользованы от самой дорогой задачи.

        lower_bound = sum_k [ max_{i in C_k} W_T^i
                              + sum_{j in C_k, j != i*} W_d^j ]

    где i* = argmax W_T^i в паре.

    Свойства:
      - при 1 грузе на пару (n_cargos = число пар): в каждой паре ровно одна
        задача, max = W_T этого груза, "остальных" нет -> lower_bound точно
        равна сумме W_T = РЕАЛЬНОЙ стоимости (реальное выполнение единственной
        задачи от исходной позиции). Граница ТОЧНА;
      - при нескольких грузах на пару: реальное выполнение дороже (пара
        смещается, каждая следующая задача считается от новой позиции и
        добавляет своё строительство), а мы берём стройку только самой
        дорогой задачи + голый проезд остальных -> граница <= real.

    Возвращает одно число. Недостижимые грузы (feasible=False) пропускаются.
    """
    task_costs = estimate_task_costs(env, cargos, pairs, L=L)

    cargos_by_pair: Dict[str, List[Cargo]] = {}
    for c in cargos:
        if task_costs[c.id].feasible:
            cargos_by_pair.setdefault(c.assigned_pair, []).append(c)

    total = 0.0
    for pair_id, pair_cargos in cargos_by_pair.items():
        wt = {c.id: task_costs[c.id].W_d + task_costs[c.id].W_b for c in pair_cargos}
        # груз с максимальным W_T - берётся целиком
        i_star = max(pair_cargos, key=lambda c: wt[c.id])
        total += wt[i_star.id]
        # остальные грузы - только проезд доставщика W_d
        for c in pair_cargos:
            if c.id != i_star.id:
                total += task_costs[c.id].W_d

    return total


def compute_lower_bound_mst(env: IslandGraph, cargos: List[Cargo], pairs: List[Pair],
                             L: float) -> float:
    """
    Альтернативная нижняя граница (идеальный граф + MST). Строго нижняя во
    всех случаях, но НЕ точна при 1 грузе на пару (идеализирует проезд и
    строительство). Сохранена как более консервативная оценка снизу.

    Состоит из ДВУХ слагаемых:

    1) Неизбежный ПРОЕЗД ДОСТАВЩИКА по идеальному графу (все мосты уже
       построены, бесплатны по стройке):
         sum_i [ dist_full(deliverer_pos_k, v_start^i)
                 + dist_full(v_start^i, v_finish^i) ]

    2) Минимальное НЕИЗБЕЖНОЕ СТРОИТЕЛЬСТВО для связности терминалов каждой
       пары (MST по blocked-мостам, E_free бесплатны):
         sum_k MST_build(terminals_k)
       где terminals_k = { deliverer_pos_k, builder_pos_k } ∪
                         { v_start^i, v_finish^i : i назначен паре k }.
    """
    import networkx as nx

    pairs_by_id = {p.id: p for p in pairs}
    Gf = env.build_full_graph_no_build(L)

    def path_cost(v_from: int, v_to: int) -> float:
        if v_from not in Gf or v_to not in Gf:
            return math.inf
        try:
            path = nx.dijkstra_path(Gf, v_from, v_to, weight="weight")
        except nx.NetworkXNoPath:
            return math.inf
        return env.path_cost_for_deliverer(path)

    total_travel = 0.0
    for c in cargos:
        pair = pairs_by_id[c.assigned_pair]
        approach = path_cost(pair.deliverer_pos, c.v_start)
        delivery = path_cost(c.v_start, c.v_finish)
        if not math.isfinite(approach) or not math.isfinite(delivery):
            continue
        w_v_start = env.G.nodes[c.v_start]["w_V"]
        total_travel += approach + delivery - w_v_start

    cargos_by_pair: Dict[str, List[Cargo]] = {}
    for c in cargos:
        cargos_by_pair.setdefault(c.assigned_pair, []).append(c)

    total_build = 0.0
    for pair_id, pair_cargos in cargos_by_pair.items():
        pair = pairs_by_id[pair_id]
        terminals = {pair.deliverer_pos, pair.builder_pos}
        for c in pair_cargos:
            terminals.add(c.v_start)
            terminals.add(c.v_finish)
        mst_cost = env.mst_build_cost_for_terminals(terminals, L)
        if math.isfinite(mst_cost):
            total_build += mst_cost

    return total_travel + total_build


# ---------------------------------------------------------------------------
# Шаг 2 - приоритет через нормированную стоимость
# ---------------------------------------------------------------------------

def compute_direct_priority(task_costs: Dict[str, TaskResult]) -> Dict[str, float]:
    """
    p(c_i) = W_T^i  -  ПРЯМАЯ зависимость от стоимости выполнения (без
    нормировки на сумму).

    Чем БОЛЬШЕ стоимость выполнения задачи, тем ВЫШЕ её приоритет (выполняется
    раньше). Логика: дорогие задачи стоит начинать заранее, иначе они рискуют
    не успеть быть выполненными при ограниченном ресурсе времени.

    ВАЖНО: поскольку нормировка на сумму (как в compute_normalized_priority с
    direction="expensive_first") - это умножение всех значений на одну и ту
    же положительную константу 1/ΣW_T^j, она НЕ МЕНЯЕТ порядок сортировки
    задач по приоритету. Поэтому порядок выполнения и итоговые суммы
    (estimated_total, real_total) при использовании compute_direct_priority
    будут ЧИСЛЕННО ИДЕНТИЧНЫ результату compute_normalized_priority(...,
    direction="expensive_first") - отличаются только показанные на графике
    приоритетов абсолютные значения p (W_T^i вместо его доли в сумме).
    """
    priority: Dict[str, float] = {}
    for cid, r in task_costs.items():
        if not r.feasible:
            priority[cid] = float("-inf")
            continue
        priority[cid] = r.W_d + r.W_b
    return priority


def compute_inverse_priority(task_costs: Dict[str, TaskResult]) -> Dict[str, float]:
    """
    p(c_i) = 1 / W_T^i  -  ОБРАТНАЯ зависимость от стоимости выполнения.

    Чем МЕНЬШЕ стоимость выполнения задачи, тем ВЫШЕ её приоритет (выполняется
    раньше). Логика: сначала выполнить дешёвые задачи, чтобы быстро набрать
    число успешных доставок при ограниченном ресурсе времени/бюджета.

    В отличие от compute_normalized_priority(direction="cheap_first")
    (которая берёт 1 - нормированную долю - линейное преобразование),
    здесь используется ТОЧНАЯ формула 1/W_T^i из исходного списка эвристик -
    нелинейное преобразование, которое сильнее выделяет самые дёшевые задачи
    (значения приоритета растут гиперболически по мере уменьшения W_T^i,
    а не линейно).
    """
    priority: Dict[str, float] = {}
    for cid, r in task_costs.items():
        if not r.feasible:
            priority[cid] = float("-inf")
            continue
        w_total = r.W_d + r.W_b
        priority[cid] = (1.0 / w_total) if w_total > 0 else float("inf")
    return priority


def compute_ratio_priority(task_costs: Dict[str, TaskResult]) -> Dict[str, float]:
    """
    p(c_i) = W_b^i / W_d^i  -  "Приоритет через стоимость по типу агента"
    (см. постановку задачи, раздел "Варианты формирования приоритета доставки").

    W_b^i и W_d^i - конкурирующие критерии (дешевле для строителей часто
    означает дороже для доставщиков, и наоборот), поэтому приоритет строится
    на их ОТНОШЕНИИ, а не на сумме (как в direct/inverse/normalized).

    Высокое значение приоритета (большой W_b^i относительно W_d^i): задача
    дорогая для строителей, но дешёвая для доставщиков - выполняется раньше,
    логика "разгрузить дорогое строительство заранее, пока доставщик и так
    почти ничего не платит за маршрут".
    Низкое значение (W_b^i << W_d^i): задача дёшева для строителей -
    сигнал, что её можно отложить или искать альтернативный маршрут без
    строительства, не теряя в общей стоимости.

    Если W_d^i == 0 (вырожденный случай - груз с нулевой стоимостью маршрута
    для доставщика), приоритет считается +inf при W_b^i > 0, и 0.0 если
    W_b^i тоже равен 0 (отношение не определено, но строить такую задачу
    нет смысла откладывать или торопить).
    """
    priority: Dict[str, float] = {}
    for cid, r in task_costs.items():
        if not r.feasible:
            priority[cid] = float("-inf")
            continue
        if r.W_d > 0:
            priority[cid] = r.W_b / r.W_d
        else:
            priority[cid] = float("inf") if r.W_b > 0 else 0.0
    return priority


def compute_normalized_priority(task_costs: Dict[str, TaskResult],
                                 direction: str = "expensive_first") -> Dict[str, float]:
    """
    p(c_i) = W_T^i / sum_j(W_T^j)

    direction:
      "expensive_first" - больший W_T^i -> больший приоритет (выполняется
                           раньше). Логика: дорогие задачи стоит начинать
                           заранее, иначе рискуют не успеть.
      "cheap_first"      - больший W_T^i -> МЕНЬШИЙ приоритет (выполняется
                           позже). Логика: сначала набрать число успешных
                           доставок дешёвыми задачами.
    Возвращает {cargo_id: p_value}, где p_value - уже сама величина
    приоритета (выше = выполняется раньше), готовая для сортировки.
    """
    feasible_costs = {cid: (r.W_d + r.W_b) for cid, r in task_costs.items() if r.feasible}
    total = sum(feasible_costs.values())
    if total <= 0:
        return {cid: 0.0 for cid in task_costs}

    normalized = {cid: w / total for cid, w in feasible_costs.items()}

    if direction == "expensive_first":
        priority = normalized
    elif direction == "cheap_first":
        priority = {cid: 1.0 - p for cid, p in normalized.items()}
    else:
        raise ValueError(direction)

    # недостижимые грузы получают приоритет -inf (никогда не выбираются первыми,
    # расписание всё равно сообщит об ошибке при попытке их выполнить)
    for cid, r in task_costs.items():
        if not r.feasible:
            priority[cid] = float("-inf")

    return priority


def assign_priority_levels_from_priority(cargos: List[Cargo], priority: Dict[str, float],
                                          k_levels: int) -> Dict[str, int]:
    """Квантильное разбиение готовых значений приоритета на k уровней
    (уровень 1 = самый приоритетный/левый, k = наименее приоритетный/правый)."""
    scored = sorted(cargos, key=lambda c: priority[c.id], reverse=True)
    n = len(scored)
    k_levels = max(1, min(k_levels, n))
    levels: Dict[str, int] = {}
    for idx, c in enumerate(scored):
        level = min(k_levels, idx * k_levels // n + 1)
        levels[c.id] = level
    return levels


# ---------------------------------------------------------------------------
# Шаг 3 - реальное последовательное выполнение в порядке приоритета
# ---------------------------------------------------------------------------

@dataclass
class SequentialEntry:
    cargo_id: str
    pair_id: str
    estimated_cost: float   # эвристическая оценка (шаг 1, от начальной позиции)
    real_cost: float        # реальная стоимость (от текущей позиции пары)
    result: TaskResult = None


@dataclass
class SequentialOutcome:
    L: float
    all_delivered: bool
    entries: List[SequentialEntry] = field(default_factory=list)
    estimated_total: float = 0.0
    real_total: float = 0.0


def run_sequential_by_priority(env: IslandGraph, cargos: List[Cargo], pairs: List[Pair],
                                L: float, priority: Dict[str, float],
                                task_costs: Dict[str, TaskResult]) -> SequentialOutcome:
    """
    Выполняет грузы СТРОГО в порядке убывания приоритета (одна общая
    очередь по всем парам сразу - порядок определяется приоритетом, а не
    тем, какая пара свободна). Каждая пара обслуживает только СВОИ грузы
    (через cargo.assigned_pair), но порядок ВНУТРИ очереди задаётся общим
    приоритетом.
    """
    cargos_sorted = sorted(cargos, key=lambda c: priority[c.id], reverse=True)
    pairs_state = {p.id: copy.deepcopy(p) for p in pairs}

    outcome = SequentialOutcome(L=L, all_delivered=False)

    for c in cargos_sorted:
        pair = pairs_state[c.assigned_pair]
        result = find_route_cheapest_bridge(env, c, pair, L)

        estimated = task_costs[c.id]
        estimated_cost = (estimated.W_d + estimated.W_b) if estimated.feasible else float("inf")

        if not result.feasible:
            outcome.all_delivered = False
            outcome.entries.append(SequentialEntry(
                cargo_id=c.id, pair_id=pair.id,
                estimated_cost=estimated_cost, real_cost=float("inf"), result=result))
            return outcome

        real_cost = result.W_d + result.W_b

        pair.deliverer_pos = c.v_finish
        if result.bridges:
            pair.builder_pos = result.bridges[-1][1]
            for (u, v) in result.bridges:
                pair.add_built_bridge(u, v)
        else:
            pair.builder_pos = c.v_finish

        outcome.entries.append(SequentialEntry(
            cargo_id=c.id, pair_id=pair.id,
            estimated_cost=estimated_cost, real_cost=real_cost, result=result))
        outcome.estimated_total += estimated_cost
        outcome.real_total += real_cost

    outcome.all_delivered = True
    return outcome
