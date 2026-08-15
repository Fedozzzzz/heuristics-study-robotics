"""
Шаг 3. Распределение грузов между коалициями (ASSIGN-CARGOS).

Выполняется ОДИН РАЗ, до начала исполнения, в два этапа.

Этап 1 -- таблица пара x груз. Для каждой пары U_k и каждого груза c_i
считается ESTIMATE-TASK-COST от НАЧАЛЬНЫХ позиций пары при built = ∅:

    W_T(U_k, c_i) = W_d + W_b   (подъезд доставщика + доставка + работа строителя)

Именно здесь заключено главное отличие статической модели от динамической:
оценка не учитывает ни промежуточных перемещений роботов (после первой же
доставки пара стоит уже не там, где стартовала), ни переправ, возведённых к
этому моменту кем-либо ещё. Поэтому суммарная оценка стоимости всех операций
заведомо может быть завышена относительно фактического исполнения
(см. diagnostics.estimated_static).

Приоритет ячейки p(U_k, c_i) = heuristic(W_T(U_k, c_i)) -- см. priority.py; в
эвристике постановки (direct) приоритет прямо равен стоимости, т.е. "сначала
дорогие грузы".

Этап 2 -- свёртка таблицы в список и один жадный проход. Реализованы два
режима (--assignment), различающиеся тем, КАКОЙ ПАРЕ достаётся груз:

  literal  -- буквально по постановке. Таблица сворачивается в один список
              записей (p, груз, пара), отсортированный по убыванию p, и
              проходится ОДИН раз: первая встреченная запись с ещё не
              назначенным грузом отдаёт этот груз своей паре. Так как p прямо
              зависит от стоимости, при эвристике direct груз достаётся паре,
              для которой он САМЫЙ ДОРОГОЙ (а при inverse -- наоборот, самой
              дешёвой: 1/W_T максимален там, где W_T минимален).
  cheapest -- грузы разбираются в том же порядке убывания приоритета (p
              считается по стоимости той пары, которая груз и получит), но
              каждый груз отдаётся паре с МИНИМАЛЬНОЙ для неё стоимостью
              доставки (классический greedy-nearest).
  lpt      -- полноценный LPT (Longest Processing Time first), классическая
              эвристика теории расписаний для параллельных машин. Порядок
              задаёт сам алгоритм (по убыванию РАЗМЕРА задачи, не зависящего
              от исполнителя), а груз достаётся НАИМЕНЕЕ ЗАГРУЖЕННОЙ паре.
              Подробно -- в docstring assign_lpt ниже.

Все режимы сравнимы на одних и тех же сценариях -- это и есть предмет
экспериментов experiments/compare_assignment_modes.py и
experiments/compare_lpt_assignment.py.

Балансировка (--balance) не даёт одной паре забрать почти все грузы (иначе
раунды Шага 4 вырождаются: одна пара везёт, остальные простаивают):

  load -- балансировка по СТОИМОСТИ (LPT-подобная). Считается целевая
          загрузка target = (сумма оценок всех грузов) / (число пар); пара,
          набравшая загрузку >= target, дальше грузы не получает, и груз
          уходит следующей допустимой паре по правилу режима. Если все пары
          насыщены (последние остатки), груз достаётся паре с минимальной
          накопленной загрузкой.
  none -- без ограничения, строго по правилу режима.

Для режима lpt балансировка не применяется: он балансирует загрузку сам, по
построению, и порог его правилу выбора пары только мешал бы (см. assign_lpt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .costs import estimate_task_cost
from .graph import EdgeKey, Graph
from .model import Cargo, CargoId
from .pairing import Pair, PairId
from .priority import CargoPriorityHeuristic

ASSIGNMENT_MODES = ("literal", "cheapest", "lpt")
BALANCE_MODES = ("load", "none")

# Как свернуть строку таблицы (стоимости груза у всех пар) в ОДИН "размер"
# задачи, по которому LPT сортирует грузы (см. assign_lpt, этап A).
LPT_SIZE_RULES = ("min", "mean", "max")

# Правило выбора пары в LPT (этап C).
LPT_RULES = ("load", "completion")

EPS = 1e-9


@dataclass
class TaskEstimate:
    """Ячейка таблицы Шага 3: во что паре U_k обойдётся груз c_i, если браться
    за него из НАЧАЛЬНОЙ позиции пары и с нуля построенных переправ."""

    pair_id: PairId
    cargo_id: CargoId
    W_T: float
    W_d: float
    W_b: float
    bridges: Set[EdgeKey] = field(default_factory=set)


CostTable = Dict[Tuple[PairId, CargoId], TaskEstimate]


def build_cost_table(G: Graph, pairs: Sequence[Pair], cargos: Sequence[Cargo]) -> CostTable:
    """Этап 1 Шага 3: полная таблица пара x груз. Недостижимые для пары грузы
    в таблицу не попадают (ячейки просто нет)."""
    table: CostTable = {}
    for pair in pairs:
        for cargo in cargos:
            est = estimate_task_cost(
                G, cargo.start, cargo.finish,
                deliverer_pos=pair.deliverer_pos,
                builder_pos=pair.builder_pos,
                built=set(),
            )
            if est is None:
                continue
            table[(pair.pair_id, cargo.cargo_id)] = TaskEstimate(
                pair_id=pair.pair_id,
                cargo_id=cargo.cargo_id,
                W_T=est.W_T,
                W_d=est.W_d,
                W_b=est.W_b,
                bridges=set(est.bridges),
            )
    return table


@dataclass
class Assignment:
    """Результат Шага 3: у каждой пары -- свой список грузов в порядке
    убывания приоритета (именно в этом порядке они и будут развозиться по
    раундам на Шаге 4)."""

    per_pair: Dict[PairId, List[CargoId]]
    pair_of_cargo: Dict[CargoId, PairId]
    priority: Dict[CargoId, float]          # p, по которому груз был назначен
    static_estimate: Dict[CargoId, float]   # W_T назначенной пары (оценка Шага 3)
    unassigned: List[CargoId]               # грузы, недостижимые для ВСЕХ пар
    # порог балансировки (None при balance="none"); в режиме lpt порога нет, и
    # здесь лежит СПРАВОЧНАЯ идеальная загрузка sum s(c_i) / m -- нижняя оценка
    # makespan, см. assign_lpt
    target_load: Optional[float]


def _pairs_for_cargo(table: CostTable, pair_ids: Sequence[PairId], cargo_id: CargoId) -> List[PairId]:
    return [k for k in pair_ids if (k, cargo_id) in table]


def _compute_target_load(
    table: CostTable,
    pair_ids: Sequence[PairId],
    cargos: Sequence[Cargo],
    heuristic: CargoPriorityHeuristic,
    mode: str,
) -> float:
    """Целевая загрузка одной пары: суммарная ожидаемая работа, поделённая на
    число пар.

    "Ожидаемая работа" одного груза -- стоимость ТОЙ пары, которую правило
    режима выбрало бы для него без ограничений: в режиме literal это пара с
    максимальным приоритетом p (для direct -- самая дорогая, для inverse --
    самая дешёвая), в режиме cheapest -- всегда самая дешёвая. Порог обязан
    считаться именно так: если брать, например, максимальную стоимость при
    обратной эвристике, порог оказывается многократно выше реально
    назначаемых стоимостей и балансировка не срабатывает вовсе."""
    total = 0.0
    for cargo in cargos:
        candidates = _pairs_for_cargo(table, pair_ids, cargo.cargo_id)
        if not candidates:
            continue
        if mode == "literal":
            # та же функция сравнения, что и при сортировке списка записей:
            # максимум p, tie-break -- меньший pair_id
            winner = max(
                candidates,
                key=lambda k: (heuristic.score(cargo, table[(k, cargo.cargo_id)].W_T), -k),
            )
            total += table[(winner, cargo.cargo_id)].W_T
        else:
            total += min(table[(k, cargo.cargo_id)].W_T for k in candidates)
    return total / len(pair_ids) if pair_ids else 0.0


def _job_size(
    table: CostTable, cargo_id: CargoId, candidates: Sequence[PairId], size_rule: str,
) -> float:
    """Этап A алгоритма LPT: "размер" задачи c_i -- ОДНО число, характеризующее
    груз безотносительно исполнителя.

    В классической постановке P||C_max время обработки p_i от машины не
    зависит. Здесь зависит: W_T(U_k, c_i) у разных пар разное (разные позиции
    роботов, разные мосты по дороге), то есть машины неидентичные. Чтобы
    сортировка LPT вообще имела смысл, строка таблицы сворачивается в скаляр:

      min  -- размер задачи = стоимость у лучшего исполнителя. Наиболее
              честная мера "внутренней" трудоёмкости груза: она не наказывает
              груз за то, что какая-то далёкая пара повезла бы его дорого.
      mean -- средняя стоимость по всем способным парам; учитывает и то,
              насколько груз неудобен "в среднем по флоту".
      max  -- пессимистичная оценка (стоимость у худшего исполнителя).
    """
    costs = [table[(k, cargo_id)].W_T for k in candidates]
    if size_rule == "min":
        return min(costs)
    if size_rule == "max":
        return max(costs)
    return sum(costs) / len(costs)


def assign_lpt(
    table: CostTable,
    pairs: Sequence[Pair],
    cargos: Sequence[Cargo],
    heuristic: CargoPriorityHeuristic,
    size_rule: str = "min",
    rule: str = "load",
) -> Assignment:
    """
    Algorithm ASSIGN-LPT(таблица, U, C, эвристика) --- Шаг 3, этап 2, режим lpt.

    Полноценный LPT (Longest Processing Time first, Graham 1969) -- список-
    планировщик для параллельных машин: длинные работы раскладываются первыми,
    каждая -- на наименее загруженную машину. Для P||C_max даёт гарантию
    C_max^LPT / C_max^opt <= 4/3 - 1/(3m). Здесь "машины" -- пары роботов U_k,
    "работы" -- грузы c_i, "время обработки" -- оценка W_T(U_k, c_i) Шага 3.

    Вход:  таблица {(U_k, c_i) -> W_T} (Шаг 3, этап 1), пары U, грузы C,
           эвристика приоритета, правило размера задачи, правило выбора пары.
    Выход: назначение pi: C -> U и очередь каждой пары.

     1: for each c_i in C do                                   -- этап A
     2:     K(c_i) <- { U_k : (U_k, c_i) in таблица }           -- кто вообще может
     3:     if K(c_i) = 0 then c_i -> unassigned; continue      -- недостижим для всех
     4:     s(c_i) <- REDUCE_{U_k in K(c_i)} W_T(U_k, c_i)      -- размер задачи
     5: target <- (sum_i s(c_i)) / m                            -- ссылочная загрузка
     6: order <- сортировка C по убыванию p(c_i) = heuristic(s(c_i))   -- этап B
     7: load[U_k] <- 0 for all U_k                              -- этап C
     8: for each c_i in order do
     9:     U* <- argmin_{U_k in K(c_i)} key(U_k, c_i)
    10:     pi(c_i) <- U*
    11:     load[U*] <- load[U*] + W_T(U*, c_i)
    12: return pi

    Этап B (строка 6). У классического LPT порядок жёстко задан: по УБЫВАНИЮ
    размера. Здесь ключ сортировки пропущен через ту же эвристику приоритета,
    что и в остальных режимах, поэтому

        direct  (p = s)     -- в точности LPT, длинные работы первыми;
        inverse (p = 1 / s) -- SPT (Shortest Processing Time first), зеркальный
                               порядок -- контрольный вариант: именно на нём
                               видно, что выигрыш LPT даёт порядок, а не сама
                               по себе балансировка;
        random  (p ~ U(0,1)) -- list scheduling в случайном порядке, baseline
                               Грэма без сортировки.

    Важно, что эвристика применяется к РАЗМЕРУ ЗАДАЧИ s(c_i), а не к стоимости
    конкретной пары: в LPT порядок обязан быть общим для всех машин.

    Этап C (строка 9), правило выбора пары:

      load       -- классический Грэм: U* = argmin load[U_k], пара с наименьшей
                    накопленной загрузкой, стоимость самого груза для неё в
                    выборе не участвует (при равных загрузках -- та, которой
                    груз дешевле; в начале, пока все загрузки нулевые, это
                    и определяет старт).
      completion -- поправка на НЕИДЕНТИЧНОСТЬ машин (R||C_max):
                    U* = argmin (load[U_k] + W_T(U_k, c_i)), то есть минимум
                    момента ЗАВЕРШЕНИЯ груза, а не момента начала. Пара, для
                    которой груз втрое дороже, его не получит, даже будучи
                    сейчас самой свободной.

    Балансировка --balance к режиму не применяется: LPT балансирует загрузку
    сам. Assignment.target_load возвращается как СПРАВОЧНАЯ идеальная загрузка
    (sum s / m) -- нижняя оценка makespan, с которой полезно сравнивать
    фактический max load, а не как порог отсечения.
    """
    if size_rule not in LPT_SIZE_RULES:
        raise KeyError(
            f"Неизвестное правило размера задачи LPT '{size_rule}'. "
            f"Доступные: {', '.join(LPT_SIZE_RULES)}"
        )
    if rule not in LPT_RULES:
        raise KeyError(
            f"Неизвестное правило выбора пары LPT '{rule}'. "
            f"Доступные: {', '.join(LPT_RULES)}"
        )

    pair_ids = [p.pair_id for p in pairs]
    cargo_by_id = {c.cargo_id: c for c in cargos}

    per_pair: Dict[PairId, List[CargoId]] = {k: [] for k in pair_ids}
    pair_of_cargo: Dict[CargoId, PairId] = {}
    priority: Dict[CargoId, float] = {}
    static_estimate: Dict[CargoId, float] = {}
    loads: Dict[PairId, float] = {k: 0.0 for k in pair_ids}

    if not pair_ids:
        return Assignment(per_pair, pair_of_cargo, priority, static_estimate,
                          [c.cargo_id for c in cargos], None)

    # --- этап A: размер задачи + грузы, недостижимые для всех пар ---
    sizes: Dict[CargoId, float] = {}
    unassigned: List[CargoId] = []
    for cargo in cargos:
        candidates = _pairs_for_cargo(table, pair_ids, cargo.cargo_id)
        if not candidates:
            unassigned.append(cargo.cargo_id)
            continue
        sizes[cargo.cargo_id] = _job_size(table, cargo.cargo_id, candidates, size_rule)

    # ссылочная идеальная загрузка (нижняя оценка makespan), не порог
    target_load = sum(sizes.values()) / len(pair_ids)

    # --- этап B: порядок обработки (LPT -- по убыванию размера) ---
    ranked: List[Tuple[float, CargoId]] = [
        (heuristic.score(cargo_by_id[cid], s), cid) for cid, s in sizes.items()
    ]
    ranked.sort(key=lambda t: (-t[0], t[1]))   # tie-break по cargo_id -- воспроизводимость

    # --- этап C: список-планировщик ---
    for p, cargo_id in ranked:
        candidates = _pairs_for_cargo(table, pair_ids, cargo_id)
        if rule == "load":
            def key(k: PairId) -> Tuple[float, float, PairId]:
                return (loads[k], table[(k, cargo_id)].W_T, k)
        else:  # rule == "completion"
            def key(k: PairId) -> Tuple[float, float, PairId]:
                w = table[(k, cargo_id)].W_T
                return (loads[k] + w, w, k)

        pair_id = min(candidates, key=key)
        per_pair[pair_id].append(cargo_id)
        pair_of_cargo[cargo_id] = pair_id
        priority[cargo_id] = p
        static_estimate[cargo_id] = table[(pair_id, cargo_id)].W_T
        loads[pair_id] += table[(pair_id, cargo_id)].W_T

    return Assignment(
        per_pair=per_pair,
        pair_of_cargo=pair_of_cargo,
        priority=priority,
        static_estimate=static_estimate,
        unassigned=unassigned,
        target_load=target_load,
    )


def assign_cargos(
    table: CostTable,
    pairs: Sequence[Pair],
    cargos: Sequence[Cargo],
    heuristic: CargoPriorityHeuristic,
    mode: str = "literal",
    balance: str = "load",
    lpt_size: str = "min",
    lpt_rule: str = "load",
) -> Assignment:
    """Algorithm ASSIGN-CARGOS(таблица, U, C, эвристика) --- Шаг 3, этап 2.

    lpt_size / lpt_rule используются только при mode="lpt" (см. assign_lpt);
    balance при mode="lpt" игнорируется -- LPT балансирует загрузку сам."""

    if mode not in ASSIGNMENT_MODES:
        raise KeyError(
            f"Неизвестный режим назначения '{mode}'. Доступные: {', '.join(ASSIGNMENT_MODES)}"
        )
    if balance not in BALANCE_MODES:
        raise KeyError(
            f"Неизвестный режим балансировки '{balance}'. Доступные: {', '.join(BALANCE_MODES)}"
        )

    if mode == "lpt":
        return assign_lpt(table, pairs, cargos, heuristic,
                          size_rule=lpt_size, rule=lpt_rule)

    pair_ids = [p.pair_id for p in pairs]
    cargo_by_id = {c.cargo_id: c for c in cargos}

    per_pair: Dict[PairId, List[CargoId]] = {k: [] for k in pair_ids}
    pair_of_cargo: Dict[CargoId, PairId] = {}
    priority: Dict[CargoId, float] = {}
    static_estimate: Dict[CargoId, float] = {}
    loads: Dict[PairId, float] = {k: 0.0 for k in pair_ids}

    if not pair_ids:
        return Assignment(per_pair, pair_of_cargo, priority, static_estimate,
                          [c.cargo_id for c in cargos], None)

    target_load: Optional[float] = (
        _compute_target_load(table, pair_ids, cargos, heuristic, mode)
        if balance == "load" else None
    )

    def saturated(pair_id: PairId) -> bool:
        return target_load is not None and loads[pair_id] >= target_load - EPS

    def do_assign(pair_id: PairId, cargo_id: CargoId, p: float) -> None:
        per_pair[pair_id].append(cargo_id)
        pair_of_cargo[cargo_id] = pair_id
        priority[cargo_id] = p
        static_estimate[cargo_id] = table[(pair_id, cargo_id)].W_T
        loads[pair_id] += table[(pair_id, cargo_id)].W_T

    if mode == "literal":
        # --- свёртка таблицы в один список, отсортированный по убыванию p ---
        entries: List[Tuple[float, CargoId, PairId]] = []
        for (pair_id, cargo_id), cell in table.items():
            p = heuristic.score(cargo_by_id[cargo_id], cell.W_T)
            entries.append((p, cargo_id, pair_id))
        # tie-break по (cargo_id, pair_id) -- воспроизводимость
        entries.sort(key=lambda t: (-t[0], t[1], t[2]))

        # --- один жадный проход по списку ---
        for p, cargo_id, pair_id in entries:
            if cargo_id in pair_of_cargo:
                continue
            if saturated(pair_id):
                continue
            do_assign(pair_id, cargo_id, p)
    else:  # mode == "cheapest"
        # приоритет груза считается по стоимости той пары, которая его и
        # получит (минимальной), после чего грузы разбираются по убыванию p
        ranked: List[Tuple[float, CargoId]] = []
        for cargo in cargos:
            candidates = _pairs_for_cargo(table, pair_ids, cargo.cargo_id)
            if not candidates:
                continue
            best_cost = min(table[(k, cargo.cargo_id)].W_T for k in candidates)
            ranked.append((heuristic.score(cargo, best_cost), cargo.cargo_id))
        ranked.sort(key=lambda t: (-t[0], t[1]))

        for p, cargo_id in ranked:
            candidates = _pairs_for_cargo(table, pair_ids, cargo_id)
            eligible = [k for k in candidates if not saturated(k)] or candidates
            pair_id = min(eligible, key=lambda k: (table[(k, cargo_id)].W_T, k))
            do_assign(pair_id, cargo_id, p)

    # --- остаток: грузы, все допустимые пары которых оказались насыщены на
    #     момент прохода. Отдаём их наименее загруженной паре из способных
    #     выполнить груз (LPT-подобное правило без порога). ---
    leftovers = [c for c in cargos if c.cargo_id not in pair_of_cargo]
    unassigned: List[CargoId] = []
    for cargo in leftovers:
        candidates = _pairs_for_cargo(table, pair_ids, cargo.cargo_id)
        if not candidates:
            unassigned.append(cargo.cargo_id)  # недостижим для всех пар
            continue
        pair_id = min(candidates, key=lambda k: (loads[k], k))
        p = heuristic.score(cargo, table[(pair_id, cargo.cargo_id)].W_T)
        do_assign(pair_id, cargo.cargo_id, p)

    return Assignment(
        per_pair=per_pair,
        pair_of_cargo=pair_of_cargo,
        priority=priority,
        static_estimate=static_estimate,
        unassigned=unassigned,
        target_load=target_load,
    )
