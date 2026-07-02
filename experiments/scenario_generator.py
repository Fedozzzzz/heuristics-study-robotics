"""
Параметризованный генератор сценария "полносвязный граф" для оценки
эвристики приоритета. Позволяет задать произвольное число островов,
пар роботов и грузов (число грузов может превышать число пар - тогда
несколько грузов последовательно назначаются одной паре, что и создаёт
содержательную разницу между эвристической оценкой стоимости "от начала"
и реальной стоимостью "с учётом уже сделанных доставок").
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
import itertools
import math
import random

from delivery_model import IslandGraph, Cargo, Pair


def build_environment(n_islands: int, seed: int = 0,
                       w_build_coef: float = 0.35,
                       w_E_base: float = 0.3, w_E_coef: float = 0.08,
                       n_free_edges: int = 4) -> IslandGraph:
    """
    n_islands островов в виде зигзага (2 ряда), полносвязный граф по рёбрам
    E_blocked (кроме n_free_edges штук, которые - E_free). Квадратичная
    калибровка w_build от длины (необходима для содержательного решения
    при полной связности - см. предыдущие эксперименты).
    """
    env = IslandGraph()

    coords = {}
    for i in range(n_islands):
        row = i % 2
        col = i // 2
        coords[i] = (col * 1.6, row * 1.4)
    rnd = random.Random(seed)
    for v, pos in coords.items():
        env.add_island(v, w_v=round(0.3 + 0.2 * rnd.random(), 2), pos=pos)

    def dist(u, v):
        x1, y1 = coords[u]; x2, y2 = coords[v]
        return math.hypot(x2 - x1, y2 - y1)

    all_pairs_edges = list(itertools.combinations(range(n_islands), 2))
    rnd.shuffle(all_pairs_edges)
    # выбираем n_free_edges КОРОТКИХ рёбер как готовые мосты (более реалистично,
    # чем случайные - готовая инфраструктура обычно соединяет близкие острова)
    sorted_by_len = sorted(all_pairs_edges, key=lambda e: dist(*e))
    free_edges = set(sorted_by_len[:n_free_edges])

    for u, v in itertools.combinations(range(n_islands), 2):
        d = dist(u, v)
        if (u, v) in free_edges:
            env.add_edge(u, v, kind="free", w_E=1.0, length=d)
            continue
        w_E = w_E_base + w_E_coef * d
        w_build = w_build_coef * d * d
        env.add_edge(u, v, kind="blocked", w_E=w_E, length=d, w_build=w_build)

    return env


def build_cargos_and_pairs(n_islands: int, n_pairs: int, n_cargos: int,
                            seed: int = 0):
    """
    n_pairs пар роботов, размещённых на случайных (но фиксированных по seed)
    островах. n_cargos грузов со случайными старт/финиш, распределённых
    МАКСИМАЛЬНО РАВНОМЕРНО по парам (round-robin), чтобы при n_cargos > n_pairs
    каждая пара получала несколько последовательных задач.
    """
    rnd = random.Random(seed + 1000)

    pair_nodes = rnd.sample(range(n_islands), min(2 * n_pairs, n_islands))
    pairs = []
    for i in range(n_pairs):
        d_pos = pair_nodes[(2 * i) % len(pair_nodes)]
        b_pos = pair_nodes[(2 * i + 1) % len(pair_nodes)]
        pairs.append(Pair(id=f"pair{i+1}", deliverer_pos=d_pos, builder_pos=b_pos))

    cargos = []
    for i in range(n_cargos):
        s, f = rnd.sample(range(n_islands), 2)
        assigned_pair = f"pair{(i % n_pairs) + 1}"  # round-robin распределение
        cargos.append(Cargo(id=f"c{i+1}", v_start=s, v_finish=f, assigned_pair=assigned_pair))

    return cargos, pairs
