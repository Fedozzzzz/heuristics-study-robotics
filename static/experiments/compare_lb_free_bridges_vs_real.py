"""
СРАВНЕНИЕ на одном графике (в % относительно реальной стоимости):
  - ОПТИМИСТИЧНАЯ нижняя граница "все мосты уже построены" (строительство
    мостов вообще НЕ учитывается - только проезд по идеальному графу)
  - реальная стоимость (базис, 0%)
  - ВЕРХНЯЯ граница / эвристическая оценка (с коррекцией пула, как в
    compare_lb_pool_real.py)

Прямая эвристика приоритета (p = W_T^i) - как и в compare_lb_pool_real.py,
чтобы график получился сопоставимым (по построению должен выглядеть похоже:
верхняя оценка растёт вверх с ростом числа грузов на пару, нижняя граница
уходит вниз и стабилизируется).

============================================================
ЧЕМ ЭТА НИЖНЯЯ ГРАНИЦА ОТЛИЧАЕТСЯ ОТ compute_lower_bound_mst:

  compute_lower_bound_mst (LOWER_BOUND="mst" в compare_lb_pool_real.py):
      неизбежный проезд по идеальному графу
    + МИНИМАЛЬНОЕ построение (MST) для связности терминалов пары
    -> учитывает, что СВЯЗНОСТЬ всё равно придётся обеспечить, но по
       минимуму (не переплачивая за жадный неоптимальный выбор мостов).

  compute_lower_bound_free_bridges (эта функция):
      неизбежный проезд по идеальному графу
    + 0 (строительство НЕ учитывается вообще, будто мосты уже есть даром)
    -> абсолютно оптимистичная граница: ЧИСТО нижняя оценка стоимости
       перемещения, без какого-либо учёта необходимости связности.
       Строго ниже (или равна) compute_lower_bound_mst, поскольку та
       включает дополнительное неотрицательное слагаемое MST_build.
============================================================

ЗАПУСК: python3 compare_lb_free_bridges_vs_real.py
"""
import os
import sys
import json
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))

import networkx as nx
import matplotlib.pyplot as plt

from delivery_model import IslandGraph, Cargo, Pair
from scenario_generator import build_environment, build_cargos_and_pairs
from priority_evaluation import (estimate_task_costs, compute_direct_priority,
                                  compute_pool_corrected_costs,
                                  run_sequential_by_priority)

# ---- Параметры ---------------------------------------------------------------
N_ISLANDS    = 20
N_PAIRS      = 4
N_SEEDS      = 30
N_CARGOS_MAX = 50
L            = 6.0

OUT_DIR    = "outputs"
CACHE_FILE = os.path.join(OUT_DIR, "lb_free_bridges_cache.json")
# ------------------------------------------------------------------------------


def compute_lower_bound_free_bridges(env: IslandGraph, cargos, pairs, L: float) -> float:
    """
    ОПТИМИСТИЧНАЯ нижняя граница: строительство мостов вообще НЕ учитывается
    (как если бы все переправы E_blocked уже были построены заранее и стоили
    только проезд w_E, без w_build). Считается ТОЛЬКО неизбежный проезд
    доставщика по этому идеальному графу - подъезд к точке погрузки плюс
    сам путь доставки:

        LB_free = sum_i [ dist_full(deliverer_pos_i, v_start^i)
                           + dist_full(v_start^i, v_finish^i) ]

    где dist_full считается на графе build_full_graph_no_build(L) (все
    проходимые рёбра с весом w_E, w_build=0 - см. IslandGraph). Позиция
    доставщика берётся НАЧАЛЬНАЯ (deliverer_pos пары, БЕЗ учёта того, что
    пара физически смещается после каждой доставки) - это ещё одно
    упрощение в пользу оптимистичности границы (аналогично compute_lower_
    bound_mst, для сопоставимости).

    В отличие от compute_lower_bound_mst, здесь НЕТ второго слагаемого
    (MST-стоимость связности терминалов) - граница строго ниже или равна ей.

    Недостижимые по идеальному графу задачи (нет пути даже без учёта
    строительства - вершина в принципе изолирована) пропускаются, как и в
    compute_lower_bound_mst.

    Возвращает одно число (сумму по всем достижимым грузам).
    """
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

    total = 0.0
    for c in cargos:
        pair = pairs_by_id[c.assigned_pair]
        approach = path_cost(pair.deliverer_pos, c.v_start)
        delivery = path_cost(c.v_start, c.v_finish)
        if not math.isfinite(approach) or not math.isfinite(delivery):
            continue
        # v_start считается и в конце approach, и в начале delivery -
        # вычитаем вес вершины один раз, чтобы не считать её дважды
        # (тот же приём, что в compute_lower_bound_mst)
        w_v_start = env.G.nodes[c.v_start]["w_V"]
        total += approach + delivery - w_v_start

    return total


def compute_upper_and_real(env, cargos, pairs, task_costs):
    """Верхняя оценка (с коррекцией пула) + реальная стоимость, прямая эвристика."""
    costs_for_prio = compute_pool_corrected_costs(env, cargos, task_costs)
    priority = compute_direct_priority(costs_for_prio)
    outcome = run_sequential_by_priority(env, cargos, pairs, L=L,
                                          priority=priority,
                                          task_costs=costs_for_prio)
    if not outcome.all_delivered:
        return None, None
    return outcome.estimated_total, outcome.real_total


def run_one(n_cargos: int, seed: int):
    """Возвращает (lower, real, upper) для одного сценария (любое может быть None)."""
    env = build_environment(n_islands=N_ISLANDS, seed=seed)
    cargos, pairs = build_cargos_and_pairs(n_islands=N_ISLANDS, n_pairs=N_PAIRS,
                                            n_cargos=n_cargos, seed=seed)
    task_costs = estimate_task_costs(env, cargos, pairs, L=L)

    lower = compute_lower_bound_free_bridges(env, cargos, pairs, L=L)
    upper, real = compute_upper_and_real(env, cargos, pairs, task_costs)
    if real is None:
        return None
    return lower, real, upper


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def sweep():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = load_cache()

    cargo_range = list(range(N_PAIRS, N_CARGOS_MAX + 1))
    lb_series, real_series, ub_series = [], [], []

    for n_cargos in cargo_range:
        ckey = str(n_cargos)
        if ckey in cache:
            entry = cache[ckey]
        else:
            lb_runs, real_runs, ub_runs = [], [], []
            for seed in range(N_SEEDS):
                res = run_one(n_cargos, seed)
                if res is None:
                    continue
                lb, real, ub = res
                real_runs.append(real)
                lb_runs.append(lb)
                if ub is not None:
                    ub_runs.append(ub)
            if real_runs:
                entry = {
                    "lb":   (sum(lb_runs) / len(lb_runs)) if lb_runs else None,
                    "real": sum(real_runs) / len(real_runs),
                    "ub":   (sum(ub_runs) / len(ub_runs)) if ub_runs else None,
                }
            else:
                entry = {"lb": None, "real": None, "ub": None}
            cache[ckey] = entry
            save_cache(cache)

        lb_series.append(entry["lb"])
        real_series.append(entry["real"])
        ub_series.append(entry["ub"])

        if entry["real"] is not None:
            lb_s = f"{entry['lb']:.1f}" if entry["lb"] is not None else "-"
            ub_s = f"{entry['ub']:.1f}" if entry["ub"] is not None else "-"
            print(f"n={n_cargos:3d}: lb_free={lb_s}  real={entry['real']:.1f}  ub={ub_s}")
        else:
            print(f"n={n_cargos:3d}: N/A")

    return cargo_range, lb_series, real_series, ub_series


def plot(x, lb, real, ub, filename):
    fig, ax = plt.subplots(figsize=(10, 6.5))

    def to_pct(series):
        vx, vy = [], []
        for xi, val, r in zip(x, series, real):
            if val is None or r is None or r == 0:
                continue
            vx.append(xi)
            vy.append(100 * (val / r - 1))
        return vx, vy

    ax.axhline(0, color="#c0392b", linewidth=2.0, label="Реальная стоимость (базис)")

    vx_u, vy_u = to_pct(ub)
    ax.plot(vx_u, vy_u, marker="^", markersize=3,
            label="Эвристическая оценка (с пулом)", color="#e67e22", linewidth=1.8)
    ax.fill_between(vx_u, vy_u, 0, color="#e67e22", alpha=0.12)

    vx_l, vy_l = to_pct(lb)
    ax.plot(vx_l, vy_l, marker="s", markersize=3,
            label="Нижняя граница (мосты уже построены, строительство не учитывается)",
            color="#2b6cb0", linewidth=1.8, linestyle="--")
    ax.fill_between(vx_l, vy_l, 0, color="#2b6cb0", alpha=0.12)

    ax.set_xlabel("Число грузов")
    ax.set_ylabel("Отклонение от реальной стоимости, %")
    ax.set_title(
        "Оценки относительно реальной стоимости\n"
        "(реальность = 0%; > 0 переоценка, < 0 недооценка)",
        fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Грузы {N_PAIRS}\u2192{N_CARGOS_MAX}, {N_ISLANDS} островов, "
        f"{N_PAIRS} пары, {N_SEEDS} seed (прямая эвристика)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


if __name__ == "__main__":
    t0 = time.time()
    x, lb, real, ub = sweep()
    print(f"\nСвип завершён за {time.time()-t0:.1f}s")
    fname = os.path.join(OUT_DIR, f"bounds_free_bridges_pool_{N_PAIRS}to{N_CARGOS_MAX}.png")
    plot(x, lb, real, ub, fname)
