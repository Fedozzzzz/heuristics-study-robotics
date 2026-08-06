"""
Модель среды: граф островов G=(V,E), грузы, роботы.
Соответствует постановке задачи: вершины-острова, рёбра-переправы
(существующие/построенные E_free, доступные к строительству E_blocked).
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import networkx as nx


# ---------------------------------------------------------------------------
# Базовые структуры данных
# ---------------------------------------------------------------------------

@dataclass
class Cargo:
    """Груз c_i с точкой отправления и точкой назначения (вариант Точка-Точка)."""
    id: str
    v_start: int   # P_s^i
    v_finish: int  # P_f^i
    assigned_pair: Optional[str] = None  # фиксированное назначение паре (id пары),
                                          # задаётся ДО запуска модели
    delivered: bool = False


@dataclass
class Pair:
    """Коалиция: одна пара (доставщик, строитель), 1-к-1 по условию задачи."""
    id: str
    deliverer_pos: int   # текущая позиция доставщика (обновляется после доставки)
    builder_pos: int      # текущая позиция строителя (обновляется после доставки)
    busy_until: float = 0.0   # момент времени освобождения (для list scheduling)
    free: bool = True
    built_bridges: Set[Tuple[int, int]] = field(default_factory=set)
    # мосты (рёбра), уже построенные ЭТОЙ парой ранее - после постройки мост
    # становится бесплатным для проезда (включая для доставщика при подъезде
    # deliverer_pos -> v_start, см. IslandGraph.path_for_deliverer_to_start).
    # Хранится как неориентированные пары (u, v) с u < v - см. add_built_bridge.

    def add_built_bridge(self, u: int, v: int) -> None:
        """Регистрирует мост (u, v) как построенный этой парой (порядок вершин
        не важен - хранится в нормализованном виде (min, max))."""
        self.built_bridges.add((min(u, v), max(u, v)))


@dataclass
class TaskResult:
    """Результат решения подзадачи строителя для одной (груз, пара, L)."""
    cargo_id: str
    pair_id: str
    feasible: bool
    path: List[int] = field(default_factory=list)        # P* - маршрут доставки v_start->v_finish
    bridges: List[Tuple[int, int]] = field(default_factory=list)  # E_b - все построенные мосты (все 3 шага)
    approach_path: List[int] = field(default_factory=list)  # подъезд доставщика deliverer_pos->v_start
    W_d: float = math.inf
    W_b: float = math.inf
    duration: float = math.inf   # время занятости пары: W_b(шаг1)+W_b(шаг2)+max(W_d, W_b(шаг3))


class IslandGraph:
    """
    Среда G=(V,E). Реализована на networkx.Graph с атрибутами ребра:
        kind: 'free'    -> e in E_free      (уже можно использовать, вес w_E)
              'blocked' -> e in E_blocked    (можно построить, вес w_E + стоимость стройки)
              'impossible' -> e in E_impossible (k=-1, никогда нельзя использовать)
        w_E: стоимость проезда по ребру (для free и blocked после постройки)
        w_build: стоимость строительства (только для blocked)
        length: длина ребра l_i (для проверки l_i <= L)
    Атрибут вершины:
        w_V: стоимость проезда через остров (Wv)
        pos: (x, y) координаты для эвристики "прямого расстояния"

    build_cost_multiplier: коэффициент γ ≥ 0, на который домножается СТОИМОСТЬ
    СТРОИТЕЛЬСТВА (w_build) каждого моста при расчёте маршрута. Позволяет сделать
    строительство относительно дороже (γ > 1) или дешевле (γ < 1) по сравнению
    с обычным перемещением по уже готовым островам/переправам, не трогая
    исходные "физические" значения w_build, заданные в среде. Применяется
    только внутри build_weighted_graph (т.е. влияет на выбор маршрута и на
    итоговую стоимость W_b), не модифицирует сами данные графа.

    ===========================
    МОДЕЛЬ ВЫПОЛНЕНИЯ ЗАДАЧИ (расширение постановки, зафиксировано здесь)
    ===========================
    Каждая задача доставки T_i выполняется парой (доставщик, строитель) в
    THREE ПОСЛЕДОВАТЕЛЬНЫХ ШАГА. Доставщик не умеет строить мосты, поэтому
    строитель должен предварительно проложить весь путь для доставщика:

    Шаг 1 (подход строителя к доставщику):
      builder_pos -> deliverer_pos, строитель едет один, при необходимости
      строя мосты. Доставщик ждёт на месте. Стоимость входит в W_b, НЕ в W_d.
      Формально: W_b^(1) = стоимость маршрута строителя builder_pos->deliverer_pos.
      Обоснование: строитель и доставщик начинают задачу из разных позиций
      (после предыдущей задачи). Чтобы проложить путь для доставщика к v_start,
      строитель должен сначала физически добраться до позиции доставщика, т.к.
      строительство мостов возможно только в зоне присутствия строителя.

    Шаг 2 (подъезд доставщика к точке погрузки):
      deliverer_pos -> v_start, строитель прокладывает маршрут из позиции
      доставщика до точки погрузки (строя мосты при необходимости). После
      завершения шага 2 доставщик едет по проложенному маршруту к v_start.
      Стоимость маршрута deliverer_pos->v_start входит ОДНОВРЕМЕННО в:
        - W_b^(2): стоимость работы строителя (подъезд к мостам + стройка)
        - W_d^(approach): стоимость проезда доставщика по проложенному пути

    Шаг 3 (доставка):
      v_start -> v_finish, строитель продолжает работу из точки, где оказался
      после шага 2. Доставщик и строитель работают ПАРАЛЛЕЛЬНО на этом шаге.
      W_b^(3): стоимость работы строителя на шаге 3.
      W_d^(delivery): стоимость проезда доставщика v_start->v_finish.

    Итоговые стоимости:
      W_d^i = W_d^(approach) + W_d^(delivery)   [полная стоимость доставщика]
      W_b^i = W_b^(1) + W_b^(2) + W_b^(3)       [полная стоимость строителя]
      duration = W_b^(1) + W_b^(2) + max(W_d^i, W_b^(3))
               [время занятости пары: шаги 1+2 строго последовательны,
                шаг 3 - параллельно, завершается когда оба заканчивают]
    """

    def __init__(self, build_cost_multiplier: float = 1.0):
        self.G = nx.Graph()
        self.build_cost_multiplier = build_cost_multiplier

    def add_island(self, v: int, w_v: float, pos: Tuple[float, float]):
        self.G.add_node(v, w_V=w_v, pos=pos)

    def add_edge(self, u: int, v: int, kind: str, w_E: float,
                 length: float, w_build: float = 0.0):
        assert kind in ("free", "blocked", "impossible")
        self.G.add_edge(u, v, kind=kind, w_E=w_E, length=length, w_build=w_build)

    def node_pos(self, v: int) -> Tuple[float, float]:
        return self.G.nodes[v]["pos"]

    def straight_distance(self, v1: int, v2: int) -> float:
        """Эвристика расстояния 'по прямой' (как если бы существовал прямой путь)."""
        x1, y1 = self.node_pos(v1)
        x2, y2 = self.node_pos(v2)
        return math.hypot(x2 - x1, y2 - y1)

    def free_subgraph(self) -> nx.Graph:
        """Подграф, состоящий ТОЛЬКО из уже существующих переправ (E_free) —
        используется для проверки, действительно ли нужно строительство."""
        Gf = nx.Graph()
        for v, data in self.G.nodes(data=True):
            Gf.add_node(v, **data)
        for u, v, data in self.G.edges(data=True):
            if data["kind"] == "free":
                Gf.add_edge(u, v, weight=data["w_E"], w_E=data["w_E"])
        return Gf

    def shortest_free_path(self, v_s: int, v_f: int) -> Optional[List[int]]:
        """
        Возвращает кратчайший путь от v_s до v_f, использующий ТОЛЬКО уже
        существующие переправы (E_free), если такой путь существует, иначе None.

        Реализует требуемое ограничение строительства: если вершина (и весь
        маршрут целиком) уже достижима через существующую инфраструктуру -
        строить дополнительные мосты не нужно вообще, независимо от L.
        """
        Gf = self.free_subgraph()
        if v_s not in Gf or v_f not in Gf:
            return None
        if not nx.has_path(Gf, v_s, v_f):
            return None
        return nx.dijkstra_path(Gf, v_s, v_f, weight="weight")

    def path_for_deliverer_to_start(self, deliverer_pos: int, v_start: int,
                                     usable_bridges: Set[Tuple[int, int]]
                                     ) -> Optional[List[int]]:
        """
        Кратчайший путь доставщика ОТ ТЕКУЩЕЙ ПОЗИЦИИ до точки погрузки
        груза (v_start), используя только уже существующие переправы
        (E_free) и мосты из usable_bridges (уже построенные парой ранее,
        ЛИБО только что построенные строителем для подзадачи подъезда этой
        же задачи - см. build_segment_with_builder). Такой мост проезжается
        бесплатно по w_E (стройка не платится повторно).

        Возвращает None, если deliverer_pos или v_start отсутствуют в этом
        подграфе, либо между ними нет пути даже с учётом usable_bridges.
        """
        Gf = self.free_subgraph()
        for (u, v) in usable_bridges:
            if self.G.has_edge(u, v):
                w_E = self.G.edges[u, v]["w_E"]
                Gf.add_edge(u, v, weight=w_E, w_E=w_E)

        if deliverer_pos not in Gf or v_start not in Gf:
            return None
        if not nx.has_path(Gf, deliverer_pos, v_start):
            return None
        return nx.dijkstra_path(Gf, deliverer_pos, v_start, weight="weight")

    def build_segment_with_builder(self, v_from: int, v_to: int,
                                    builder_pos: int, L: float,
                                    already_built: Optional[Set[Tuple[int, int]]] = None,
                                    k_alternatives: int = 12
                                    ) -> Optional[Dict]:
        """
        ОБЩАЯ подзадача строителя: найти маршрут v_from -> v_to (с
        возможностью строить новые мосты, через build_weighted_graph(L)) и
        посчитать стоимость W_b этого сегмента, считая, что строитель
        начинает работу из своей ТЕКУЩЕЙ позиции builder_pos и должен сам
        доехать/построить путь до первого нужного моста на сегменте, затем
        пройти по всем мостам сегмента в порядке появления на пути.

        already_built - мосты, УЖЕ построенные (этой же парой, на более
        ранних шагах ЭТОЙ ЖЕ задачи или на предыдущих задачах - см.
        Pair.built_bridges) ДО начала этого сегмента. Такие мосты
        проезжаются как уже готовая переправа (вес w_E, БЕЗ повторной
        оплаты w_build) - предотвращает двойной счёт стоимости постройки
        одного и того же физического моста, если несколько сегментов одной
        задачи проходят через него (например, шаг 1 строит мост, который
        затем оказывается и на маршруте шага 2).

        Используется ТРИ РАЗА для одной задачи доставки (см.
        find_route_and_bridges/find_route_cheapest_bridge):
          1) builder_pos_исходный -> deliverer_pos (подход строителя)
          2) deliverer_pos -> v_start (подъезд, по нему едет доставщик)
          3) v_start -> v_finish (сама доставка)
        Каждый следующий вызов должен передавать already_built, включающий
        ВСЕ мосты, построенные на предыдущих шагах (накопительно).

        Возвращает None, если v_from/v_to недостижимы вовсе (даже с учётом
        строительства) ИЛИ если строитель не может доехать до необходимого
        моста (нет пути builder_pos -> мост даже со стройкой). В противном
        случае возвращает dict с полями:
          path           - List[int], выбранный маршрут v_from -> v_to
          bridges        - List[Tuple[int,int]], НОВЫЕ мосты НА ЭТОМ
                            маршруте (уже учтённые в already_built не
                            считаются повторно построенными), упорядоченные
                            по появлению на пути
          W_path         - float, w_E+w_V по path (без учёта builder_pos)
          W_build        - float, суммарная стоимость W_b ЭТОГО сегмента
                            (подъезд строителя к НОВЫМ мостам + сами НОВЫЕ
                            мосты; уже построенные мосты проезжаются бесплатно
                            по стройке, только за w_E/w_V)
          builder_pos_after - int, позиция строителя ПОСЛЕ этого сегмента
                            (конец последнего НОВОГО построенного моста на
                            пути, либо v_to, если новых мостов не было)

        Если v_from == v_to, сегмент считается тривиальным: path=[v_from],
        bridges=[], W_path=0.0 (вершина учитывается вызывающим кодом - см.
        path_cost_for_deliverer/approach_cost_for_deliverer), W_build=0.0,
        builder_pos_after = builder_pos (строитель никуда не сдвигается).
        """
        already_built = already_built or set()

        if v_from == v_to:
            return {
                "path": [v_from], "bridges": [],
                "W_path": 0.0, "W_build": 0.0, "builder_pos_after": builder_pos,
            }

        # сначала пробуем без строительства вовсе: E_free + уже построенные
        # ранее мосты (already_built) считаются готовой переправой
        free_or_built = self._free_plus_built_subgraph(already_built)
        if v_from in free_or_built and v_to in free_or_built \
                and nx.has_path(free_or_built, v_from, v_to):
            free_path = nx.dijkstra_path(free_or_built, v_from, v_to, weight="weight")
            return {
                "path": free_path, "bridges": [],
                "W_path": self.path_cost_for_deliverer(free_path),
                "W_build": 0.0, "builder_pos_after": builder_pos,
            }

        Gp = self.build_weighted_graph(L)
        # уже построенные мосты - убираем стоимость стройки из веса (граф
        # должен предпочитать их как обычную бесплатную-по-стройке переправу)
        for (u, v) in already_built:
            if Gp.has_edge(u, v) and Gp.edges[u, v]["kind"] == "blocked":
                Gp.edges[u, v]["weight"] = Gp.edges[u, v]["w_E"]

        if v_from not in Gp or v_to not in Gp:
            return None
        if not nx.has_path(Gp, v_from, v_to):
            return None

        candidates = []
        try:
            gen = nx.shortest_simple_paths(Gp, v_from, v_to, weight="weight")
            for i, path in enumerate(gen):
                if i >= k_alternatives:
                    break
                candidates.append(path)
        except nx.NetworkXNoPath:
            return None
        if not candidates:
            return None

        scored = []
        for path in candidates:
            edges_on_path = self.edges_on_path(path)
            # НОВЫЕ мосты на этом пути - только blocked-рёбра, КОТОРЫЕ ЕЩЁ
            # НЕ были построены ранее (already_built исключаются из bridges,
            # они уже готовы и не требуют (повторной) постройки)
            bridges = [(u, v) for (u, v) in edges_on_path
                       if Gp.edges[u, v]["kind"] == "blocked"
                       and (min(u, v), max(u, v)) not in already_built]
            w_path = self.path_cost_for_deliverer(path)

            builder_cost = 0.0
            build_cost_total = 0.0
            current = builder_pos
            bridges_sorted = []
            ok = True
            if bridges:
                order = {(u, v): i for i, (u, v) in enumerate(edges_on_path)}
                order.update({(v, u): i for (u, v), i in list(order.items())})
                bridges_sorted = sorted(bridges, key=lambda e: order[e])
                for (u, v) in bridges_sorted:
                    try:
                        sub_path = nx.dijkstra_path(Gp, current, u, weight="weight")
                        builder_cost += self.path_cost_for_deliverer(sub_path)
                    except nx.NetworkXNoPath:
                        ok = False
                        break
                    build_cost_total += Gp.edges[u, v]["w_build"]
                    builder_cost += Gp.edges[u, v]["w_E"] + self.G.nodes[v]["w_V"]
                    current = v
            if not ok:
                continue
            w_build = build_cost_total + builder_cost
            builder_pos_after = bridges_sorted[-1][1] if bridges_sorted else v_to
            scored.append((path, w_path, w_build, bridges, bridges_sorted, builder_pos_after))

        if not scored:
            return None

        def is_dominated(item, others):
            _, wp, wb, _, _, _ = item
            for _, wp2, wb2, _, _, _ in others:
                if wp2 <= wp and wb2 <= wb and (wp2 < wp or wb2 < wb):
                    return True
            return False

        non_dominated = [it for it in scored if not is_dominated(it, scored)]
        best = min(non_dominated, key=lambda it: it[1] + it[2])
        path, w_path, w_build, bridges, bridges_sorted, builder_pos_after = best

        return {
            "path": path, "bridges": bridges,
            "W_path": w_path, "W_build": w_build,
            "builder_pos_after": builder_pos_after,
        }

    def _free_plus_built_subgraph(self, extra_bridges: Set[Tuple[int, int]]) -> nx.Graph:
        """Подграф E_free + дополнительные мосты (уже построенные где-то
        ранее), используется как 'бесплатная' инфраструктура для проверки
        достижимости без новой стройки."""
        Gf = self.free_subgraph()
        for (u, v) in extra_bridges:
            if self.G.has_edge(u, v):
                w_E = self.G.edges[u, v]["w_E"]
                Gf.add_edge(u, v, weight=w_E, w_E=w_E)
        return Gf

    def mst_build_cost_for_terminals(self, terminals, L: float) -> float:
        """
        Минимальная СТОИМОСТЬ СТРОИТЕЛЬСТВА мостов, необходимая для связности
        всех вершин terminals между собой (нижняя граница на W_b для набора
        задач с этими терминалами).

        Модель: E_free-рёбра уже существуют и бесплатны, поэтому вершины,
        связанные через E_free, объединяются в компоненты (суперузлы) - внутри
        компоненты строить ничего не нужно. Затем строится MST по компонентам,
        где ребро между двумя компонентами - самый дешёвый по w_build мост
        (blocked, length <= L), соединяющий их. Вес MST = минимальная суммарная
        стоимость строительства для связывания всех терминалов.

        Строго нижняя граница на реальную W_b: чтобы выполнить грузы, их
        терминалы обязаны быть связаны, а дешевле веса MST это невозможно;
        переезды строителя и проезд по мостам (w_E) отброшены (>= 0).

        Возвращает 0.0, если все терминалы уже в одной E_free-компоненте;
        inf, если связать их мостами длины <= L невозможно.
        """
        terminals = set(terminals)
        if len(terminals) <= 1:
            return 0.0

        free_components = list(nx.connected_components(self.free_subgraph()))
        comp_of = {}
        for i, comp in enumerate(free_components):
            for v in comp:
                comp_of[v] = i

        terminal_comps = set(comp_of[v] for v in terminals if v in comp_of)
        if len(terminal_comps) <= 1:
            return 0.0

        # граф компонент: ребро = самый дешёвый мост между компонентами
        comp_graph = nx.Graph()
        comp_graph.add_nodes_from(terminal_comps)
        for u, v, data in self.G.edges(data=True):
            if data["kind"] != "blocked" or data["length"] > L:
                continue
            cu, cv = comp_of.get(u), comp_of.get(v)
            if cu is None or cv is None or cu == cv:
                continue
            w = data["w_build"] * self.build_cost_multiplier
            if comp_graph.has_edge(cu, cv):
                if w < comp_graph.edges[cu, cv]["weight"]:
                    comp_graph.edges[cu, cv]["weight"] = w
            else:
                comp_graph.add_edge(cu, cv, weight=w)

        for tc in terminal_comps:
            if tc not in comp_graph:
                comp_graph.add_node(tc)

        # терминальные компоненты должны оказаться в одной компоненте связности
        membership = {}
        for i, cc in enumerate(nx.connected_components(comp_graph)):
            for node in cc:
                membership[node] = i
        memberships = set(membership.get(tc) for tc in terminal_comps)
        if len(memberships) > 1 or None in memberships:
            return math.inf

        mst = nx.minimum_spanning_tree(comp_graph, weight="weight")
        return sum(data["weight"] for _, _, data in mst.edges(data=True))

    def build_full_graph_no_build(self, L: float) -> nx.Graph:
        """
        "Идеальный" граф для НИЖНЕЙ оценки: все проходимые рёбра доступны с
        весом ТОЛЬКО w_E (строительство бесплатно, w_build = 0), как если бы
        все мосты уже были построены заранее. Используется для оценки снизу
        "когда строить ничего не нужно".

        Исключаются:
          - рёбра kind='impossible' (никогда нельзя использовать);
          - рёбра kind='blocked' с length > L (мост физически нельзя
            построить, даже бесплатно - ограничение L сохраняется).
        Оставшиеся рёбра (free и blocked с length <= L) получают вес w_E.
        Вес вершины w_V добавляется отдельно при подсчёте стоимости пути
        (path_cost_for_deliverer).
        """
        Gf = nx.Graph()
        for v, data in self.G.nodes(data=True):
            Gf.add_node(v, **data)
        for u, v, data in self.G.edges(data=True):
            if data["kind"] == "impossible":
                continue
            if data["kind"] == "blocked" and data["length"] > L:
                continue
            Gf.add_edge(u, v, weight=data["w_E"], w_E=data["w_E"],
                        kind=data["kind"], length=data["length"])
        return Gf

    def build_weighted_graph(self, L: float, skip_redundant_bridges: bool = True) -> nx.Graph:
        """
        Алгоритм 3, шаг 1: строит вспомогательный граф G' для заданного L:
          - рёбра kind='impossible' исключаются всегда;
          - рёбра kind='blocked' с length > L исключаются (мост физически не построить);
          - оставшимся blocked рёбрам присваивается приведённый вес w_hat = w_build + w_E;
          - рёбрам free присваивается вес w_E.
        Вес вершины добавляется отдельно при подсчёте стоимости пути (не как вес ребра,
        чтобы не считать его дважды при заходе в вершину с разных рёбер).

        Если skip_redundant_bridges=True (по умолчанию), дополнительно исключаются
        ИЗБЫТОЧНЫЕ blocked-рёбра: если обе вершины ребра уже находятся в одной
        компоненте связности подграфа E_free, строительство этого моста не имеет
        смысла - между вершинами уже существует обходной путь без строительства.
        Реализует ограничение: "если вершина достижима через существующую
        инфраструктуру, строить мост к ней не нужно".
        """
        free_components = None
        if skip_redundant_bridges:
            free_components = list(nx.connected_components(self.free_subgraph()))

        def in_same_free_component(u, v) -> bool:
            if free_components is None:
                return False
            for comp in free_components:
                if u in comp and v in comp:
                    return True
            return False

        Gp = nx.Graph()
        for v, data in self.G.nodes(data=True):
            Gp.add_node(v, **data)
        for u, v, data in self.G.edges(data=True):
            if data["kind"] == "impossible":
                continue
            if data["kind"] == "blocked":
                if data["length"] > L:
                    continue
                if skip_redundant_bridges and in_same_free_component(u, v):
                    continue  # избыточный мост - u и v уже связаны через E_free
                effective_build = data["w_build"] * self.build_cost_multiplier
                w_hat = effective_build + data["w_E"]
            else:  # free
                w_hat = data["w_E"]
            Gp.add_edge(u, v, weight=w_hat, kind=data["kind"],
                        w_E=data["w_E"],
                        w_build=data["w_build"] * self.build_cost_multiplier
                                if data["kind"] == "blocked" else data["w_build"],
                        length=data["length"])
        return Gp

    def path_cost_for_deliverer(self, path: List[int]) -> float:
        """W_d = сумма w_E по рёбрам пути + сумма w_V по вершинам пути."""
        if len(path) < 1:
            return 0.0
        cost = 0.0
        for v in path:
            cost += self.G.nodes[v]["w_V"]
        for u, v in zip(path[:-1], path[1:]):
            cost += self.G.edges[u, v]["w_E"]
        return cost

    def approach_cost_for_deliverer(self, approach_path: List[int]) -> float:
        """
        Стоимость подъезда доставщика deliverer_pos -> v_start, ПОДГОТОВЛЕННАЯ
        для склейки с основным маршрутом груза без двойного счёта стыковой
        вершины v_start: считает w_E по рёбрам подъезда + w_V только по
        вершинам СТРОГО ДО последней (последняя вершина approach_path - это
        v_start, её w_V будет посчитан как первая вершина основного маршрута
        в path_cost_for_deliverer и не должен повторяться здесь).

        Если approach_path состоит из одной вершины (deliverer_pos == v_start,
        нет рёбер для проезда), возвращает 0.0 - подъезд физически не нужен,
        вся стоимость v_start уже учтена основным маршрутом.
        """
        if len(approach_path) < 2:
            return 0.0
        cost = 0.0
        for v in approach_path[:-1]:
            cost += self.G.nodes[v]["w_V"]
        for u, v in zip(approach_path[:-1], approach_path[1:]):
            cost += self.G.edges[u, v]["w_E"]
        return cost

    def edges_on_path(self, path: List[int]) -> List[Tuple[int, int]]:
        return list(zip(path[:-1], path[1:]))

    def deliverer_cost_with_approach(self, deliverer_pos: int, main_path: List[int],
                                      usable_bridges: Set[Tuple[int, int]]
                                      ) -> Tuple[Optional[float], Optional[List[int]]]:
        """
        Полная стоимость W_d ДЛЯ ДОСТАВЩИКА с учётом подъезда от текущей
        позиции (deliverer_pos) до начала груза (main_path[0] == v_start),
        используя E_free + usable_bridges (мосты, уже построенные парой
        ранее ИЛИ построенные строителем для подзадачи подъезда этой же
        задачи - см. build_segment_with_builder и find_route_and_bridges).

        Возвращает (W_d_total, approach_path):
          - W_d_total = approach_cost_for_deliverer(approach_path)
                        + path_cost_for_deliverer(main_path)
          - approach_path - найденный путь подъезда (для диагностики/отрисовки)

        Если подъезд недостижим даже с учётом usable_bridges, возвращает
        (None, None) - вызывающий код должен пометить задачу как
        feasible=False (подъезд физически невозможен для этой пары).
        """
        if not main_path:
            return None, None
        v_start = main_path[0]
        approach_path = self.path_for_deliverer_to_start(deliverer_pos, v_start,
                                                          usable_bridges)
        if approach_path is None:
            return None, None
        w_d = (self.approach_cost_for_deliverer(approach_path)
               + self.path_cost_for_deliverer(main_path))
        return w_d, approach_path
