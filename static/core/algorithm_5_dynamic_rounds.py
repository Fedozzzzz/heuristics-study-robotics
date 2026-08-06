"""
Алгоритм 5: ДИНАМИЧЕСКОЕ циклическое расписание с барьером синхронизации
по раундам (round-synchronized dynamic scheduling).

===========================================================================
ОТЛИЧИЕ ОТ ВСЕГО, ЧТО БЫЛО РАНЬШЕ
===========================================================================
Алгоритм 2 (core/algorithm_2.py) и оценка эвристик приоритета
(experiments/priority_evaluation.py) СТАТИЧНЫ в следующем смысле:
  - назначение груз -> пара (cargo.assigned_pair) фиксируется ОДИН РАЗ,
    ДО запуска модели (round-robin / LPT / greedy-nearest - не важно, все
    они считают стоимость по НАЧАЛЬНЫМ позициям пар и больше к этому
    решению не возвращаются);
  - очередь ВНУТРИ пары обрабатывается последовательно и НЕЗАВИСИМО от
    остальных пар - пары не синхронизируются по времени и не "видят" друг
    друга при выборе следующего груза.

Здесь используется принципиально другая, ПОЛНОСТЬЮ ДИНАМИЧЕСКАЯ схема
(отвечает на вопрос протокола из Постановка_задачи, п. "Вопросы для
протокола", №2 - про соотношение модели приоритета с динамическим
перепланированием):

  1. На каждом ЦИКЛЕ (раунде) из пула ЕЩЁ НЕ ДОСТАВЛЕННЫХ грузов выбирается
     n = |U| доставок - по одной на каждую пару - ЗАНОВО, от ТЕКУЩИХ
     (уже сдвинутых предыдущими раундами) позиций всех пар. Никакого
     заранее фиксированного cargo.assigned_pair не используется вообще -
     решение "какая пара повезёт какой груз" принимается на лету, отдельно
     для каждого раунда, глобальным жадным паросочетанием (см.
     _select_round_assignments).
  2. Выбранные n доставок выполняются "параллельно": для каждой пары
     считается её собственная длительность (TaskResult.duration) на
     доставшуюся ей задачу.
  3. Раунд завершается, когда ПОСЛЕДНЯЯ из n параллельных доставок
     закончена - барьер синхронизации по max(duration) среди участников
     раунда. Только после этого момента позиции всех пар считаются
     актуальными, и шаг 1 повторяется заново для оставшихся грузов.

Важное следствие барьера: пара, закончившая свою задачу раньше остальных,
в этой модели ПРОСТАИВАЕТ до конца раунда, а не берёт следующий груз сразу
(в отличие от асинхронного Алгоритма 2). Это плата за то, что стоимость и
приоритет каждой доставки пересчитываются на графе с АКТУАЛЬНЫМ состоянием
(позиции + уже построенные мосты), а не на изначально построенном
статическом графе.

ВЫБОР ПУЛА НА РАУНД - ДВЕ СТРАТЕГИИ (см. run_dynamic_rounds(..., selection_strategy=...)):
  "nearest" - глобальное жадное паросочетание по минимальной стоимости
              комбинации (пара, груз) на этом раунде (_select_round_assignments).
  "lpt"     - ГИБРИД с Algorithm 0 ASSIGN-LPT (lpt_assignment.assign_by_lpt):
              на каждом раунде дорогие грузы обслуживаются первыми и
              отдаются паре с минимальной НАКОПЛЕННОЙ ЗА ВСЕ РАУНДЫ
              нагрузкой, а не той, для которой этот груз дешевле всего
              (_select_round_assignments_lpt). Сохраняет балансировку
              нагрузки, характерную для LPT, но остаётся динамическим -
              решение принимается заново каждый раунд от актуальных
              позиций, а не один раз в начале, как в обычном LPT-назначении.
===========================================================================
"""

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from delivery_model import IslandGraph, Cargo, Pair, TaskResult
from algorithms_1_3 import find_route_and_bridges


# route_fn: то, чем решается подзадача (пара, груз) -> TaskResult.
# По умолчанию - "точный" Алгоритм 3 (find_route_and_bridges, тот же, что
# использует core/algorithm_2.py). Можно передать
# heuristic_cheapest_bridge.find_route_cheapest_bridge, чтобы сравнение со
# experiments/evaluate_priority_heuristics.py было "на равных" (та же
# эвристика поиска маршрута с обеих сторон).
RouteFn = Callable[[IslandGraph, Cargo, Pair, float], TaskResult]


@dataclass
class RoundAssignment:
    """Одно назначение (пара, груз) внутри раунда, с уже посчитанным результатом."""
    pair_id: str
    cargo_id: str
    result: TaskResult


@dataclass
class DynamicRound:
    """Один раунд динамического расписания."""
    index: int
    round_start: float = 0.0
    round_end: float = 0.0          # round_start + max(duration) среди назначений раунда
    assignments: List[RoundAssignment] = field(default_factory=list)


@dataclass
class DynamicOutcome:
    """Итог полного прогона Алгоритма 5."""
    L: float
    all_delivered: bool
    rounds: List[DynamicRound] = field(default_factory=list)
    W_d_total: float = 0.0
    W_b_total: float = 0.0
    makespan: float = 0.0           # момент завершения последнего раунда

    @property
    def real_total(self) -> float:
        """Φ = сумма W_d + W_b по всем выполненным доставкам - метрика,
        напрямую сопоставимая с SequentialOutcome.real_total из
        experiments/priority_evaluation.py."""
        return self.W_d_total + self.W_b_total

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    @property
    def idle_total(self) -> float:
        """
        Суммарный ПРОСТОЙ пар на барьерах синхронизации:

            idle = Σ_раунд Σ_пара (round_duration - duration этой пары)

        Метрика, которой в СТАТИЧЕСКОЙ модели не существует в принципе:
        там пары независимы во времени и никогда друг друга не ждут
        (см. run_sequential_by_priority - времени нет вообще;
        run_scheduling - m независимых временных линий из нуля).
        Здесь же простой физически реален: пара, закончившая раньше,
        стоит до конца раунда. Чем РОВНЕЕ эвристика подбирает грузы по
        длительности внутри раунда, тем меньше idle - это отдельный,
        независимый от Φ критерий качества эвристики приоритета.
        """
        total = 0.0
        for r in self.rounds:
            round_duration = r.round_end - r.round_start
            for a in r.assignments:
                total += round_duration - a.result.duration
        return total

    @property
    def idle_fraction(self) -> float:
        """Доля простоя от общего оплаченного времени пар (idle / (idle + работа)).
        Нормированная версия idle_total - сопоставима между сценариями
        разного размера."""
        work = sum(a.result.duration for r in self.rounds for a in r.assignments)
        denom = work + self.idle_total
        return (self.idle_total / denom) if denom > 0 else 0.0


def _select_round_assignments(env: IslandGraph, pending: List[Cargo],
                               pairs_state: Dict[str, Pair], L: float,
                               route_fn: RouteFn) -> List[RoundAssignment]:
    """
    Шаг 1 раунда: динамически выбирает ПО ОДНОМУ грузу на каждую пару из
    пула `pending`, от текущих позиций `pairs_state`. Ни один груз заранее
    ни за одной парой не закреплён - решение принимается заново каждый
    раунд.

    Алгоритм - ГЛОБАЛЬНОЕ жадное паросочетание (не "по очереди пар" и не
    "по очереди грузов", а по минимальной стоимости среди ВСЕХ ещё не
    занятых комбинаций сразу, что не зависит от порядка обхода):

      1) для каждой пары и каждого ещё не доставленного груза считается
         W_T(пара, груз) = W_d + W_b (route_fn, от текущей позиции пары);
      2) пока остались свободные пары и груза в пуле:
           выбрать (пара*, груз*) с МИНИМАЛЬНЫМ W_T среди ещё не занятых
           комбинаций -> назначить -> исключить пару* и груз* из
           дальнейшего рассмотрения ЭТОГО раунда;
      3) недостижимые комбинации (feasible=False) в рассмотрение не
         попадают вовсе.

    Если для оставшейся пары нет ни одного достижимого груза в пуле (или
    наоборот) - эта пара/груз просто остаются без назначения в этом раунде
    (пара продолжит просто ждать со следующего раунда, если на ней вообще
    имеет смысл настаивать - на практике при feasible-графе такое означает,
    что при данном L конкретный груз в принципе недостижим ни для кого).
    """
    costs: Dict[Tuple[str, str], TaskResult] = {}
    for pid, pair in pairs_state.items():
        for cargo in pending:
            res = route_fn(env, cargo, pair, L)
            if res.feasible:
                costs[(pid, cargo.id)] = res

    remaining_pair_ids = set(pairs_state.keys())
    remaining_cargo_ids = {c.id for c in pending}

    assignments: List[RoundAssignment] = []
    while remaining_pair_ids and remaining_cargo_ids:
        candidates = [
            (pid, cid, res) for (pid, cid), res in costs.items()
            if pid in remaining_pair_ids and cid in remaining_cargo_ids
        ]
        if not candidates:
            break  # ни одна оставшаяся пара не достаёт ни один оставшийся груз
        best_pid, best_cid, best_res = min(
            candidates, key=lambda t: t[2].W_d + t[2].W_b)
        assignments.append(RoundAssignment(pair_id=best_pid, cargo_id=best_cid,
                                            result=best_res))
        remaining_pair_ids.discard(best_pid)
        remaining_cargo_ids.discard(best_cid)

    return assignments


def _select_round_assignments_lpt(env: IslandGraph, pending: List[Cargo],
                                   pairs_state: Dict[str, Pair], L: float,
                                   route_fn: RouteFn,
                                   pair_load: Dict[str, float]) -> List[RoundAssignment]:
    """
    ГИБРИД: LPT-стиль выбора пула на раунд вместо глобального жадного
    минимума стоимости (_select_round_assignments). Комбинирует идею
    lpt_assignment.assign_by_lpt (дорогие грузы - в первую очередь, паре с
    минимальной накопленной нагрузкой) с тем, что здесь это решение
    принимается ЗАНОВО каждый раунд от ТЕКУЩИХ позиций пар, а не один раз
    в начале.

    Шаги (аналог Algorithm 0 ASSIGN-LPT, но на один раунд):
      1) для каждого груза пула оценивается его "размер" как МИНИМАЛЬНАЯ
         W_T среди всех ещё свободных пар (при их текущих позициях) - та
         же логика, что в assign_by_lpt.estimate_initial_cost, просто не
         от начальной, а от текущей позиции;
      2) грузы сортируются по УБЫВАНИЮ размера (LPT: дорогие первыми);
      3) каждый груз по очереди отдаётся ещё свободной паре с МИНИМАЛЬНОЙ
         накопленной нагрузкой pair_load среди пар, которые вообще могут
         его выполнить - НЕ паре, для которой он дешевле всего (это и есть
         отличие от _select_round_assignments: там критерий - минимальная
         стоимость данной комбинации, здесь - баланс суммарной загрузки).

    pair_load - {pair_id: накопленная реальная стоимость (W_d+W_b) за ВСЕ
    предыдущие раунды}; передаётся и читается вызывающим кодом
    (run_dynamic_rounds), обновляется ПОСЛЕ выполнения раунда, поэтому
    здесь используется как есть - "нагрузка на начало этого раунда".
    """
    remaining_pair_ids = set(pairs_state.keys())
    remaining_cargos = {c.id: c for c in pending}

    # 1) стоимость каждого груза для каждой ещё свободной пары + "размер"
    #    груза (минимум по парам) для сортировки LPT
    costs: Dict[str, Dict[str, TaskResult]] = {}
    best_cost: Dict[str, float] = {}
    for cid, cargo in remaining_cargos.items():
        row: Dict[str, TaskResult] = {}
        for pid, pair in pairs_state.items():
            res = route_fn(env, cargo, pair, L)
            if res.feasible:
                row[pid] = res
        if row:
            costs[cid] = row
            best_cost[cid] = min(r.W_d + r.W_b for r in row.values())

    # 2) LPT-сортировка: дорогие грузы обслуживаются в первую очередь
    sorted_cids = sorted(costs.keys(), key=lambda cid: best_cost[cid], reverse=True)

    # 3) жадное распределение по минимальной накопленной нагрузке
    assignments: List[RoundAssignment] = []
    for cid in sorted_cids:
        if not remaining_pair_ids:
            break
        candidates = {pid: r for pid, r in costs[cid].items()
                      if pid in remaining_pair_ids}
        if not candidates:
            continue  # ни одна из ещё свободных пар не достаёт этот груз
        best_pid = min(candidates.keys(), key=lambda pid: pair_load.get(pid, 0.0))
        assignments.append(RoundAssignment(pair_id=best_pid, cargo_id=cid,
                                            result=candidates[best_pid]))
        remaining_pair_ids.discard(best_pid)

    return assignments


def _priority_value(kind: str, res: TaskResult) -> float:
    """
    Значение приоритета p(пара, груз) по TaskResult этой комбинации.
    Формулы ТЕ ЖЕ, что в статике (experiments/priority_evaluation.py:
    compute_direct_priority / compute_inverse_priority / compute_ratio_priority),
    включая обработку вырожденных случаев - чтобы сравнение "статика против
    динамики при одной и той же эвристике" было корректным.

    ПРИНЦИПИАЛЬНОЕ ОТЛИЧИЕ ОТ СТАТИКИ: там p(c_i) считается ОДИН РАЗ, от
    НАЧАЛЬНОЙ позиции закреплённой пары, и это ОДНО число на груз. Здесь
    p зависит от ПАРЫ (у каждой своя текущая позиция и свой набор уже
    построенных мостов), то есть это МАТРИЦА p(пара, груз), и она
    пересчитывается заново каждый раунд.
    """
    w_t = res.W_d + res.W_b
    if kind == "direct":            # дорогие первыми
        return w_t
    if kind == "inverse":           # дешёвые первыми
        return (1.0 / w_t) if w_t > 0 else float("inf")
    if kind == "ratio":             # дорого строителю первыми
        if res.W_d > 0:
            return res.W_b / res.W_d
        return float("inf") if res.W_b > 0 else 0.0
    raise ValueError(f"Неизвестная эвристика приоритета: {kind}")


def _select_round_assignments_priority(env: IslandGraph, pending: List[Cargo],
                                        pairs_state: Dict[str, Pair], L: float,
                                        route_fn: RouteFn, priority_kind: str,
                                        rnd) -> List[RoundAssignment]:
    """
    Шаг 1 раунда по ЭВРИСТИКЕ ПРИОРИТЕТА: каждой паре достаётся груз с
    НАИВЫСШИМ приоритетом (а не с минимальной стоимостью, как в
    _select_round_assignments, и не по балансу нагрузки, как в
    _select_round_assignments_lpt).

    Разрешение конфликтов (две пары хотят один и тот же груз) - глобальным
    жадным argmax, симметрично _select_round_assignments:
      пока есть свободные пары и грузы в пуле:
        взять комбинацию (пара, груз) с МАКСИМАЛЬНЫМ p среди ещё не
        занятых -> назначить -> исключить эту пару и этот груз из
        рассмотрения ЭТОГО раунда.
    Порядок обхода пар/грузов на результат не влияет.

    priority_kind: "direct" | "inverse" | "ratio" | "random".
    "random" - БЕЙЗЛАЙН: p присваивается случайно каждой комбинации
    (пара, груз), то есть раунд получает случайное паросочетание. Именно
    случайное ПАРОСОЧЕТАНИЕ, а не случайный порядок одной очереди - в
    динамической модели выбор "кто везёт" и "что везём" неразделимы, и
    бейзлайн должен рандомизировать оба решения сразу. rnd - экземпляр
    random.Random, детерминированный по seed сценария.

    Недостижимые комбинации (feasible=False) в рассмотрение не попадают.
    """
    scored: Dict[Tuple[str, str], Tuple[float, TaskResult]] = {}
    for pid, pair in pairs_state.items():
        for cargo in pending:
            res = route_fn(env, cargo, pair, L)
            if not res.feasible:
                continue
            if priority_kind == "random":
                p = rnd.random()
            else:
                p = _priority_value(priority_kind, res)
            scored[(pid, cargo.id)] = (p, res)

    remaining_pair_ids = set(pairs_state.keys())
    remaining_cargo_ids = {c.id for c in pending}

    assignments: List[RoundAssignment] = []
    while remaining_pair_ids and remaining_cargo_ids:
        candidates = [
            (pid, cid, p, res) for (pid, cid), (p, res) in scored.items()
            if pid in remaining_pair_ids and cid in remaining_cargo_ids
        ]
        if not candidates:
            break
        best_pid, best_cid, _, best_res = max(candidates, key=lambda t: t[2])
        assignments.append(RoundAssignment(pair_id=best_pid, cargo_id=best_cid,
                                            result=best_res))
        remaining_pair_ids.discard(best_pid)
        remaining_cargo_ids.discard(best_cid)

    return assignments


def run_dynamic_rounds(env: IslandGraph, cargos: List[Cargo], pairs: List[Pair],
                        L: float, route_fn: RouteFn = find_route_and_bridges,
                        max_rounds: Optional[int] = None,
                        selection_strategy: str = "nearest",
                        priority_kind: str = "ratio",
                        seed: int = 0) -> DynamicOutcome:
    """
    Главный цикл Алгоритма 5 (см. описание модуля).

    cargo.assigned_pair ИГНОРИРУЕТСЯ полностью - назначение динамическое,
    пересчитывается на каждом раунде.

    selection_strategy - как выбирается пул из n доставок на раунд:
      "nearest" (по умолчанию) - _select_round_assignments: глобальное
                 жадное паросочетание по МИНИМАЛЬНОЙ стоимости комбинации
                 (пара, груз) - минимизирует стоимость КАЖДОГО раунда
                 локально, не заботясь о балансе нагрузки между парами.
      "lpt"    - _select_round_assignments_lpt: ГИБРИД с LPT
                 (lpt_assignment.assign_by_lpt) - на каждом раунде дорогие
                 грузы обслуживаются в первую очередь и отдаются паре с
                 минимальной НАКОПЛЕННОЙ ЗА ВСЕ РАУНДЫ нагрузкой, а не той,
                 которой этот конкретный груз обойдётся дешевле всего.
                 Балансирует суммарную загрузку пар (как обычный LPT), но
                 остаётся динамическим - назначение и позиции пересчитываются
                 от АКТУАЛЬНОГО состояния на каждом раунде, а не один раз
                 в начале, как в lpt_assignment.assign_by_lpt.
      "priority" - _select_round_assignments_priority: каждой паре достаётся
                 груз с НАИВЫСШИМ приоритетом p, посчитанным по эвристике
                 priority_kind ("direct"/"inverse"/"ratio"/"random") - те же
                 формулы, что в статике, но матрица p(пара, груз)
                 пересчитывается каждый раунд от актуальных позиций.

    priority_kind - какая эвристика приоритета используется; имеет смысл
    только при selection_strategy="priority".
    seed - зерно для priority_kind="random" (детерминированный бейзлайн).

    max_rounds - опциональный предохранитель от бесконечного цикла в
    вырожденных сценариях (по умолчанию не ограничено - цикл естественно
    заканчивается, когда pending пуст либо ни одна комбинация не feasible).
    """
    if selection_strategy not in ("nearest", "lpt", "priority"):
        raise ValueError(f"Неизвестная selection_strategy: {selection_strategy}")
    if selection_strategy == "priority" and priority_kind not in (
            "direct", "inverse", "ratio", "random"):
        raise ValueError(f"Неизвестная priority_kind: {priority_kind}")

    import random as _random
    rnd = _random.Random(seed * 7919)   # тот же множитель, что в статике
                                        # (evaluate_priority_heuristics.make_priority)

    cargos = copy.deepcopy(cargos)
    pairs_state: Dict[str, Pair] = {p.id: copy.deepcopy(p) for p in pairs}
    pending: Dict[str, Cargo] = {c.id: c for c in cargos if not c.delivered}
    pair_load: Dict[str, float] = {pid: 0.0 for pid in pairs_state}

    outcome = DynamicOutcome(L=L, all_delivered=False)
    current_time = 0.0
    round_idx = 0

    while pending:
        if max_rounds is not None and round_idx >= max_rounds:
            break

        if selection_strategy == "priority":
            assignments = _select_round_assignments_priority(
                env, list(pending.values()), pairs_state, L, route_fn,
                priority_kind, rnd)
        elif selection_strategy == "lpt":
            assignments = _select_round_assignments_lpt(
                env, list(pending.values()), pairs_state, L, route_fn, pair_load)
        else:
            assignments = _select_round_assignments(
                env, list(pending.values()), pairs_state, L, route_fn)
        if not assignments:
            # оставшиеся грузы недостижимы ни для одной пары при данном L
            break

        round_duration = max(a.result.duration for a in assignments)
        round_start = current_time
        round_end = round_start + round_duration
        dround = DynamicRound(index=round_idx, round_start=round_start,
                               round_end=round_end)

        for a in assignments:
            pair = pairs_state[a.pair_id]
            cargo = pending[a.cargo_id]
            result = a.result

            pair.deliverer_pos = cargo.v_finish
            if result.bridges:
                pair.builder_pos = result.bridges[-1][1]
                for (u, v) in result.bridges:
                    pair.add_built_bridge(u, v)
            else:
                pair.builder_pos = cargo.v_finish

            cargo.delivered = True
            del pending[a.cargo_id]

            dround.assignments.append(a)
            outcome.W_d_total += result.W_d
            outcome.W_b_total += result.W_b
            pair_load[a.pair_id] = pair_load.get(a.pair_id, 0.0) + result.W_d + result.W_b

        outcome.rounds.append(dround)
        current_time = round_end
        round_idx += 1

    outcome.all_delivered = (len(pending) == 0)
    outcome.makespan = current_time
    return outcome


def to_schedule_entries(outcome: DynamicOutcome):
    """
    Конвертирует DynamicOutcome в список algorithm_2.ScheduleEntry, чтобы
    переиспользовать core/visualize.py:plot_schedule_routes без изменений.

    ВАЖНО: start_time/end_time каждой записи берутся как
    [round_start, round_start + result.duration] - то есть РЕАЛЬНЫЙ момент
    освобождения пары, а не round_end (барьер). Из-за этого на диаграмме
    Ганта у пар, закончивших раньше остальных в раунде, будет виден
    промежуток простоя до начала следующего раунда - это осознанная,
    содержательная часть модели (см. описание модуля), а не артефакт.
    """
    from algorithm_2 import ScheduleEntry  # локальный импорт - избегаем цикла

    entries = []
    for dround in outcome.rounds:
        for a in dround.assignments:
            entries.append(ScheduleEntry(
                cargo_id=a.cargo_id, pair_id=a.pair_id, result=a.result,
                start_time=dround.round_start,
                end_time=dround.round_start + a.result.duration,
            ))
    return entries


# ===========================================================================
# ВИЛКА "ОЦЕНКА vs ФАКТ" ВНУТРИ ДИНАМИКИ (real <= estimated_pool <= estimated_raw)
# ===========================================================================

@dataclass
class DynamicCostBracket:
    """
    Диагностическая "вилка" качества оценки стоимости ДЛЯ ОДНОГО прогона
    динамической модели. Аналог статической вилки из
    priority_evaluation / Формулы_модели.docx (W̲T <= real <= ŴT_pool <=
    ŴT_raw), но пересчитанной для фактической последовательности раундов.

    Все три величины считаются по УЖЕ ВЫПОЛНЕННОМУ расписанию (DynamicOutcome),
    то есть по фактическому разбиению грузов на пары и фактическому порядку
    раундов - НЕ по гипотетическому назначению.
    """
    real_total: float          # факт: сумма W_d + W_b по прогону
    estimated_raw: float       # наивная оценка "каждый груз с чистого листа"
    estimated_pool: float      # raw с коррекцией пула мостов (по факту)
    gap_raw_pct: float         # 100*(raw/real - 1) - завышение наивной оценки
    gap_pool_pct: float        # 100*(pool/real - 1) - завышение после коррекции


def compute_dynamic_cost_bracket(env: IslandGraph, outcome: DynamicOutcome,
                                  route_fn: RouteFn = find_route_and_bridges,
                                  L: Optional[float] = None) -> DynamicCostBracket:
    """
    Считает вилку real <= estimated_pool <= estimated_raw для завершённого
    прогона динамической модели.

    ---------------------------------------------------------------------
    ПОЧЕМУ ТРИ РАЗНЫЕ ВЕЛИЧИНЫ (и чем отличаются от статики)
    ---------------------------------------------------------------------
    В динамике фактический result.W_b каждого груза УЖЕ учитывает мосты,
    построенные этой парой на ПРЕДЫДУЩИХ раундах (route_fn стартует с
    pair.built_bridges), поэтому переиспользование "назад по времени" уже
    сидит в real. Наивная же оценка так не умеет - она смотрит на груз
    изолированно. Отсюда:

    real_total
        Факт из outcome (W_d_total + W_b_total). Мосты переиспользуются
        естественно по ходу раундов.

    estimated_raw
        Каждый ФАКТИЧЕСКИ выполненный груз пересчитывается ЗАНОВО от
        позиции пары НА НАЧАЛО ЕГО РАУНДА, но с ПУСТЫМ набором уже
        построенных мостов (built_bridges = ∅) - то есть "как если бы этот
        груз был единственным у пары". Это прямой аналог статической
        estimate_task_costs, наложенный на фактическую последовательность:
        систематически ПЕРЕОЦЕНИВАЕТ, потому что заставляет каждый груз
        заново оплачивать мосты, которые в реальности уже стояли.

    estimated_pool
        estimated_raw с коррекцией пула мостов (формула из
        priority_evaluation.compute_pool_corrected_costs и
        Формулы_модели.docx):
            W_b^pool(k) = Σ_i W_b_raw^i  -  Σ_e (c_e - 1)·w_build(e)
        где c_e - сколько ФАКТИЧЕСКИ выполненных грузов пары k требовали
        мост e (по result.bridges), группировка - по ФАКТИЧЕСКОЙ паре из
        прогона (в динамике нет assigned_pair, поэтому берётся то, как
        реально легло расписание). Каждый уникальный мост оплачивается
        один раз - оценка приближается к real сверху.

    Ожидаемый порядок: real <= estimated_pool <= estimated_raw. gap_pool
    должен быть заметно меньше gap_raw - это и показывает, СКОЛЬКО
    завышения наивной оценки объясняется именно переиспользованием мостов
    (а не смещением позиции пары, которое коррекция пула не трогает).

    route_fn / L: та же функция и то же L, что в прогоне (для корректного
    пересчёта estimated_raw). Если L не задан, берётся outcome.L.
    """
    if L is None:
        L = outcome.L

    real_total = outcome.real_total

    # --- estimated_raw: каждый груз "с чистого листа" от позиции на начало
    #     его раунда. Позицию восстанавливаем, проигрывая раунды заново, но
    #     считаем груз с ПУСТЫМИ built_bridges (изолированно). ---
    # чтобы получить позицию пары на начало каждого раунда, проигрываем
    # фактическую последовательность и запоминаем позиции ПЕРЕД применением.
    from delivery_model import Pair as _Pair
    pair_state: Dict[str, _Pair] = {}
    # инициализация: позиции пар до первого раунда неизвестны из outcome
    # напрямую, поэтому восстанавливаем их из первого появления пары -
    # берём deliverer_pos/builder_pos ПЕРЕД первым её грузом. Для этого
    # проигрываем раунды и на первом появлении пары фиксируем её позицию
    # как позицию, от которой был посчитан result (она уже "зашита" в
    # том, что мы храним approach_path/bridges, но проще пересчитать от
    # известной стартовой позиции). Здесь используем прямой способ: raw
    # для груза = W_d + W_b, пересчитанные route_fn от позиции пары на
    # начало раунда с built_bridges=∅.
    #
    # ВАЖНО: позиция пары на начало раунда = её позиция после ПРЕДЫДУЩЕГО
    # её груза (deliverer в v_finish предыдущего, builder в конце его
    # мостов). До первого груза позиция = исходная, которую мы знаем из
    # того, что result первого груза был посчитан именно от неё. Мы её
    # не храним явно, поэтому берём из built-in инварианта: на первом
    # раунде built_bridges пусты, значит result первого груза УЖЕ и есть
    # его raw. Для последующих - пересчитываем.
    estimated_raw = 0.0

    # сгруппируем назначения по парам в порядке раундов
    from collections import defaultdict
    seq_by_pair = defaultdict(list)  # pair_id -> [RoundAssignment ...] по порядку
    for dround in outcome.rounds:
        for a in dround.assignments:
            seq_by_pair[a.pair_id].append(a)

    # raw_bridges_by_pair[pid] = список наборов мостов КАЖДОГО груза,
    # посчитанных ИЗОЛИРОВАННО (с чистого листа) - именно по ним считается
    # переиспользование c_e для коррекции пула (в фактическом result.bridges
    # мосты с прошлых раундов уже дедуплицированы и c_e всегда = 1).
    raw_bridges_by_pair: Dict[str, List[List[Tuple[int, int]]]] = defaultdict(list)

    for pid, seq in seq_by_pair.items():
        prev_deliverer = None
        prev_builder = None
        for i, a in enumerate(seq):
            if i == 0:
                # первый груз пары: built_bridges были пусты -> raw == факт,
                # и мосты факта = мосты изолированного расчёта
                estimated_raw += a.result.W_d + a.result.W_b
                raw_bridges_by_pair[pid].append(
                    [(min(u, v), max(u, v)) for (u, v) in a.result.bridges])
            else:
                tmp = _Pair(id=pid, deliverer_pos=prev_deliverer,
                            builder_pos=prev_builder)
                # built_bridges НЕ переносим - это и есть "с чистого листа"
                raw_res = route_fn(env, _cargo_of(a, outcome), tmp, L)
                if raw_res.feasible:
                    estimated_raw += raw_res.W_d + raw_res.W_b
                    raw_bridges_by_pair[pid].append(
                        [(min(u, v), max(u, v)) for (u, v) in raw_res.bridges])
                else:
                    estimated_raw += a.result.W_d + a.result.W_b
                    raw_bridges_by_pair[pid].append(
                        [(min(u, v), max(u, v)) for (u, v) in a.result.bridges])
            prev_deliverer, prev_builder = _pos_after(a)

    # --- estimated_pool: raw минус повторные постройки (по ИЗОЛИРОВАННЫМ
    #     наборам мостов - там переиспользование ещё видно) ---
    duplicate_cost_total = 0.0
    for pid, per_cargo_bridges in raw_bridges_by_pair.items():
        bridge_count: Dict[Tuple[int, int], int] = {}
        for bridges in per_cargo_bridges:
            for key in bridges:
                bridge_count[key] = bridge_count.get(key, 0) + 1
        for (u, v), count in bridge_count.items():
            if count > 1 and env.G.has_edge(u, v):
                duplicate_cost_total += (count - 1) * env.G.edges[u, v]["w_build"]

    estimated_pool = estimated_raw - duplicate_cost_total

    gap_raw = 100 * (estimated_raw / real_total - 1) if real_total > 0 else 0.0
    gap_pool = 100 * (estimated_pool / real_total - 1) if real_total > 0 else 0.0

    return DynamicCostBracket(
        real_total=real_total, estimated_raw=estimated_raw,
        estimated_pool=estimated_pool, gap_raw_pct=gap_raw, gap_pool_pct=gap_pool)


def _cargo_of(assignment: RoundAssignment, outcome: DynamicOutcome) -> Cargo:
    """Восстанавливает объект Cargo (v_start, v_finish) по id из результата.
    result хранит path (seg3) и cargo_id - v_start/v_finish берём из
    approach/path результата, но надёжнее пересобрать минимальный Cargo из
    сохранённых в result концов маршрута."""
    # v_start = первая вершина основного маршрута доставки (seg3.path[0]),
    # v_finish = последняя. result.path - это seg3["path"] (v_start->v_finish).
    r = assignment.result
    v_start = r.path[0] if r.path else None
    v_finish = r.path[-1] if r.path else None
    return Cargo(id=assignment.cargo_id, v_start=v_start, v_finish=v_finish)


def _pos_after(assignment: RoundAssignment) -> Tuple[int, int]:
    """Позиция пары ПОСЛЕ выполнения этого груза (как в основном цикле):
    deliverer в v_finish, builder в конце последнего построенного моста
    (или тоже в v_finish, если мостов не было)."""
    r = assignment.result
    v_finish = r.path[-1] if r.path else None
    if r.bridges:
        builder_pos = r.bridges[-1][1]
    else:
        builder_pos = v_finish
    return v_finish, builder_pos
