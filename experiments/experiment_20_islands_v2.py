"""
ЗАДАЧА (версия 2, с контролируемой калибровкой):
20 островов, 8 пар роботов, 8 грузов, 4 готовых моста, остальные рёбра -
ПОЛНОСВЯЗНЫЙ граф (190-4=186 доступных к строительству).

Урок из experiment_more_points.py: чтобы получить N содержательных точек на
фронте, нужны N НЕЗАВИСИМЫХ порогов переключения - то есть для каждого груза
индивидуально откалиброванная пара (длина прямого маршрута, стоимость его
строительства), такая что для ЭТОГО конкретного груза:
  1) Wb(прямой) > Wb(альтернатива) - настоящий trade-off, не доминирование
  2) (Wd+Wb)(прямой) < (Wd+Wb)(альтернатива) при достаточном L - чтобы
     алгоритм (минимизирующий сумму среди недоминируемых) реально его выбрал

При ПОЛНОЙ связности этого добиться сложнее, чем в блочной структуре, потому
что каждый груз имеет МНОЖЕСТВО доступных промежуточных путей, а не один
"эталонный" обход - поэтому здесь используется единая формула w_build(d),
а индивидуальность порогов достигается через РАЗНУЮ ГЕОМЕТРИЮ (разные
расстояния между стартом/финишем каждого груза), а не через ручную правку
каждого ребра.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
import itertools
import math

import matplotlib.pyplot as plt

from delivery_model import IslandGraph, Cargo, Pair
from algorithm_2 import run_scheduling
from algorithm_4 import build_cost_curves, pareto_front, choose_compromise, is_concave_front
from visualize import plot_pareto


N_ISLANDS = 20


def build_environment(w_build_coef: float = 0.35, w_E_base: float = 0.3,
                       w_E_coef: float = 0.08,
                       build_cost_multiplier: float = 1.0) -> IslandGraph:
    """
    20 островов в виде зигзага (2 ряда по 10), полносвязный граф по рёбрам
    E_blocked, 4 готовых моста (E_free). w_build = w_build_coef * length^2
    (квадратичная зависимость - необходима для содержательного trade-off
    при полной связности, см. предыдущий эксперимент с 10 островами).
    """
    env = IslandGraph(build_cost_multiplier=build_cost_multiplier)

    coords = {}
    for i in range(N_ISLANDS):
        row = i % 2
        col = i // 2
        coords[i] = (col * 1.6, row * 1.4)
    for v, pos in coords.items():
        env.add_island(v, w_v=round(0.35 + 0.05 * (v % 4), 2), pos=pos)

    def dist(u, v):
        x1, y1 = coords[u]; x2, y2 = coords[v]
        return math.hypot(x2 - x1, y2 - y1)

    free_edges = {(0, 1), (4, 5), (8, 9), (12, 13)}

    for u, v in itertools.combinations(range(N_ISLANDS), 2):
        d = dist(u, v)
        if (u, v) in free_edges or (v, u) in free_edges:
            env.add_edge(u, v, kind="free", w_E=1.0, length=d)
            continue
        w_E = w_E_base + w_E_coef * d
        w_build = w_build_coef * d * d
        env.add_edge(u, v, kind="blocked", w_E=w_E, length=d, w_build=w_build)

    return env


def build_cargos_and_pairs():
    """
    8 пар и 8 грузов. Старт/финиш каждого груза подобраны так, чтобы
    РАССТОЯНИЯ между ними были РАЗНЫМИ (это и создаёт разные индивидуальные
    пороги переключения обход/прямой маршрут у разных грузов, без необходимости
    вручную калибровать каждое ребро отдельно).
    """
    cargo_routes = [
        (0, 3),    # короткая дистанция
        (2, 7),
        (1, 10),
        (4, 13),
        (3, 16),
        (6, 17),
        (5, 18),
        (8, 19),   # самая длинная дистанция
    ]
    cargos = [
        Cargo(id=f"c{i+1}", v_start=s, v_finish=f, assigned_pair=f"pair{i+1}")
        for i, (s, f) in enumerate(cargo_routes)
    ]

    pair_positions = [
        (0, 1), (2, 3), (1, 4), (4, 6),
        (3, 7), (6, 9), (5, 10), (8, 12),
    ]
    pairs = [
        Pair(id=f"pair{i+1}", deliverer_pos=d, builder_pos=b)
        for i, (d, b) in enumerate(pair_positions)
    ]
    return cargos, pairs


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    cargos, pairs = build_cargos_and_pairs()

    print("=" * 70)
    print(f"ЗАДАЧА: {N_ISLANDS} островов, {len(pairs)} пар, {len(cargos)} грузов, "
          f"полносвязный граф, квадратичная стройка")
    print("=" * 70)

    # покажем расстояния старт-финиш у каждого груза - чтобы убедиться,
    # что они действительно разные (и значит дадут разные пороги)
    env_probe = build_environment()
    for c in cargos:
        d = env_probe.straight_distance(c.v_start, c.v_finish)
        print(f"  {c.id}: {c.v_start}->{c.v_finish}, прямое расстояние={d:.2f}")

    env = build_environment()
    max_len = max(d["length"] for _, _, d in env.G.edges(data=True))
    L_grid = [round(0.3 + 0.05 * i, 2) for i in range(int(max_len / 0.05) + 10)]
    print(f"\nСетка L: {len(L_grid)} точек, от {L_grid[0]} до {L_grid[-1]}")

    points = build_cost_curves(env, cargos, pairs, L_grid)
    front = pareto_front(points)
    unique = sorted(set((round(p.W_d_total, 2), round(p.W_b_total, 2)) for p in front))

    print(f"\nДопустимых точек на сетке: {len(points)}")
    print(f"Уникальных недоминируемых точек на фронте: {len(unique)}")
    for wd, wb in unique:
        print(f"  Wd={wd:8.2f}  Wb={wb:8.2f}")

    concave = is_concave_front(front)
    print(f"Форма фронта: {'вогнутая' if concave else 'выпуклая'}")

    best = choose_compromise(front)
    print(f"Компромисс: L*={best.L}, Wd={best.W_d_total:.2f}, Wb={best.W_b_total:.2f}")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    plot_pareto(points, front, best,
                title=f"20 островов, полносвязный граф, 8 грузов\n"
                      f"{len(unique)} точек на Парето-фронте",
                ax=ax, show=False)
    fig.tight_layout()
    fig.savefig("outputs/pareto_20_islands.png", dpi=150)
    plt.close(fig)
    print("\nСохранено: pareto_20_islands.png")
