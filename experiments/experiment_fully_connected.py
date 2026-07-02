"""
ЭКСПЕРИМЕНТ: что происходит с Парето-фронтом, если граф сделать полносвязным
по рёбрам, доступным к строительству?

Запускает ДВА варианта полносвязного графа и сравнивает с базовым (разреженным):
  A) build_build_proportional - стоимость строительства РАСТЁТ ЛИНЕЙНО с длиной
     моста (физически естественно, но недостаточно для конкуренции с обходом)
  B) build_quadratic          - стоимость строительства растёт КВАДРАТИЧНО
     с длиной (длинные мосты резко дороже - восстанавливает trade-off)

Вывод сохраняется в текстовый отчёт и три PNG для сравнения.
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


COORDS = {
    0: (0, 1.4), 1: (1.6, 0), 2: (3.2, 1.4), 3: (4.8, 0), 4: (6.4, 1.4),
    5: (8.0, 0), 6: (9.6, 1.4), 7: (11.2, 0), 8: (12.8, 1.4), 9: (14.4, 0),
}
W_V = {0: 0.4, 1: 0.5, 2: 0.4, 3: 0.6, 4: 0.4, 5: 0.5, 6: 0.4, 7: 0.6, 8: 0.4, 9: 0.5}
FREE_EDGES = {(1, 2), (3, 4), (5, 6), (7, 8)}


def _dist(u, v):
    x1, y1 = COORDS[u]; x2, y2 = COORDS[v]
    return math.hypot(x2 - x1, y2 - y1)


def build_fully_connected(build_cost_mode: str = "linear",
                           build_cost_multiplier: float = 1.0) -> IslandGraph:
    """
    build_cost_mode:
      "linear"    - w_build = 1.5 * length          (стройка пропорциональна длине)
      "quadratic" - w_build = 0.3 * length**2        (длинные мосты резко дороже)
    """
    env = IslandGraph(build_cost_multiplier=build_cost_multiplier)
    for v, pos in COORDS.items():
        env.add_island(v, w_v=W_V[v], pos=pos)

    for u, v in itertools.combinations(range(10), 2):
        d = _dist(u, v)
        if (u, v) in FREE_EDGES or (v, u) in FREE_EDGES:
            env.add_edge(u, v, kind="free", w_E=1.0, length=d)
            continue
        if build_cost_mode == "linear":
            w_E = 0.3 + 0.15 * d
            w_build = 1.5 * d
        elif build_cost_mode == "quadratic":
            w_E = 0.3 + 0.1 * d
            w_build = 0.3 * d * d
        else:
            raise ValueError(build_cost_mode)
        env.add_edge(u, v, kind="blocked", w_E=w_E, length=d, w_build=w_build)

    return env


def build_cargos_and_pairs():
    cargos = [
        Cargo(id="c1", v_start=0, v_finish=9, assigned_pair="pair1"),
        Cargo(id="c2", v_start=2, v_finish=7, assigned_pair="pair1"),
        Cargo(id="c3", v_start=1, v_finish=9, assigned_pair="pair2"),
        Cargo(id="c4", v_start=4, v_finish=9, assigned_pair="pair2"),
    ]
    pairs = [
        Pair(id="pair1", deliverer_pos=0, builder_pos=1),
        Pair(id="pair2", deliverer_pos=3, builder_pos=4),
    ]
    return cargos, pairs


def analyze(env: IslandGraph, cargos, pairs, label: str, L_grid):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"Рёбер всего: {env.G.number_of_edges()} "
          f"(E_free: {sum(1 for _,_,d in env.G.edges(data=True) if d['kind']=='free')}, "
          f"E_blocked: {sum(1 for _,_,d in env.G.edges(data=True) if d['kind']=='blocked')})")

    points = build_cost_curves(env, cargos, pairs, L_grid)
    front = pareto_front(points)
    unique = sorted(set((round(p.W_d_total, 2), round(p.W_b_total, 2)) for p in front))

    print(f"Допустимых точек на сетке: {len(points)}, уникальных решений на фронте: {len(unique)}")
    for wd, wb in unique:
        print(f"  Wd={wd:7.2f}  Wb={wb:7.2f}")

    concave = is_concave_front(front)
    print(f"Форма фронта: {'вогнутая' if concave else 'выпуклая'}")

    if front:
        best = choose_compromise(front)
        print(f"Компромисс: L*={best.L}, Wd={best.W_d_total:.2f}, Wb={best.W_b_total:.2f}")
        return points, front, best
    return points, front, None


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    cargos, pairs = build_cargos_and_pairs()
    L_grid = [round(0.5 + 0.1 * i, 1) for i in range(140)]  # 0.5..14.4

    # --- A: базовый разреженный граф (для сравнения) ---
    import main as base_main
    base_env = base_main.build_environment()
    base_points, base_front, base_best = analyze(
        base_env, cargos, pairs, "БАЗОВЫЙ ГРАФ (17 рёбер, разреженный)",
        [round(1.0 + 0.1 * i, 1) for i in range(41)])

    # --- B: полносвязный, линейная стоимость строительства ---
    env_linear = build_fully_connected(build_cost_mode="linear")
    points_linear, front_linear, best_linear = analyze(
        env_linear, cargos, pairs,
        "ПОЛНОСВЯЗНЫЙ ГРАФ (45 рёбер), w_build ЛИНЕЙНО от длины", L_grid)

    # --- C: полносвязный, квадратичная стоимость строительства ---
    env_quad = build_fully_connected(build_cost_mode="quadratic")
    points_quad, front_quad, best_quad = analyze(
        env_quad, cargos, pairs,
        "ПОЛНОСВЯЗНЫЙ ГРАФ (45 рёбер), w_build КВАДРАТИЧНО от длины", L_grid)

    # --- визуализация сравнения трёх фронтов ---
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    plot_pareto(base_points, base_front, base_best,
                title="A. Базовый разреженный граф\n(17 рёбер)",
                ax=axes[0], show=False)
    plot_pareto(points_linear, front_linear, best_linear,
                title="B. Полносвязный граф (45 рёбер)\nw_build ЛИНЕЙНО от длины",
                ax=axes[1], show=False)
    plot_pareto(points_quad, front_quad, best_quad,
                title="C. Полносвязный граф (45 рёбер)\nw_build КВАДРАТИЧНО от длины",
                ax=axes[2], show=False)

    fig.tight_layout()
    fig.savefig("outputs/experiment_fully_connected_comparison.png", dpi=150)
    plt.close(fig)
    print("\nСохранено: experiment_fully_connected_comparison.png")
