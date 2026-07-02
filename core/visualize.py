"""
Визуализация:
  1) plot_environment       - граф среды G: острова, существующие переправы (E_free),
                               доступные к строительству (E_blocked), с подсветкой
                               маршрута доставки P* и построенных мостов E_b для
                               конкретного результата TaskResult (если передан).
  2) plot_schedule_routes    - тот же граф, но сразу со всеми маршрутами расписания
                               (каждая пара своим цветом).
  3) plot_pareto             - график W_d_total(L) vs W_b_total(L): все точки сетки,
                               выделенный Парето-фронт, отмеченная компромиссная L*.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from delivery_model import IslandGraph, Cargo, Pair, TaskResult
from algorithm_2 import ScheduleOutcome
from algorithm_4 import ParetoPoint


_COLOR_FREE = "#4a7c59"        # существующие переправы - зелёный
_COLOR_BLOCKED = "#b0b0b0"     # доступные к строительству, не использованные - серый
_COLOR_BRIDGE_BUILT = "#d9822b"  # построенный мост (использован в маршруте) - оранжевый
_COLOR_ROUTE = "#2b6cb0"       # маршрут доставки - синий (для одиночного режима)
_COLOR_ISLAND = "#3d3d3d"
_PAIR_PALETTE = ["#2b6cb0", "#c0392b", "#8e44ad", "#16a085", "#d4ac0d", "#7f8c8d"]


def _draw_base_graph(ax, env: IslandGraph, show_costs: bool = True):
    """
    Рисует острова и все рёбра в их 'нейтральном' состоянии (free/blocked).
    Если show_costs=True, подписывает:
      - на каждом ребре: стоимость проезда w_E (и стоимость строительства w_build
        для blocked-рёбер, в скобках);
      - на каждой вершине: стоимость проезда острова w_V (под номером острова).
    """
    pos = {v: env.node_pos(v) for v in env.G.nodes}

    free_edges = [(u, v) for u, v, d in env.G.edges(data=True) if d["kind"] == "free"]
    blocked_edges = [(u, v) for u, v, d in env.G.edges(data=True) if d["kind"] == "blocked"]

    nx.draw_networkx_edges(env.G, pos, edgelist=free_edges, ax=ax,
                            edge_color=_COLOR_FREE, width=2.5, style="solid")
    nx.draw_networkx_edges(env.G, pos, edgelist=blocked_edges, ax=ax,
                            edge_color=_COLOR_BLOCKED, width=1.5, style="dashed")

    nx.draw_networkx_nodes(env.G, pos, ax=ax, node_color="white",
                            edgecolors=_COLOR_ISLAND, linewidths=1.8, node_size=550)
    nx.draw_networkx_labels(env.G, pos, ax=ax, font_size=10, font_weight="bold",
                             font_color=_COLOR_ISLAND)

    if show_costs:
        # масштаб смещения подписей - пропорционален размеру графа, а не
        # фиксированное число, чтобы корректно работать на графах любого масштаба
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        graph_scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        v_offset = graph_scale * 0.06
        e_offset = graph_scale * 0.045

        # подписи стоимости проезда острова (w_V) - чуть ниже узла
        for v, data in env.G.nodes(data=True):
            x, y = pos[v]
            ax.annotate(f"w_V={data['w_V']:.1f}", (x, y - v_offset),
                        ha="center", va="top", fontsize=7, color="#666666")

        # подписи стоимости ребра (w_E, и w_build для blocked) - у середины ребра,
        # со смещением перпендикулярно ребру, пропорциональным масштабу графа
        for u, v, data in env.G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            norm = max((dx**2 + dy**2) ** 0.5, 1e-6)
            off_x, off_y = -dy / norm * e_offset, dx / norm * e_offset
            if data["kind"] == "free":
                label = f"w_E={data['w_E']:.1f}"
                color = "#2f5a3a"
            else:
                label = f"w_E={data['w_E']:.1f}\nстройка={data['w_build']:.1f}"
                color = "#6b6b6b"
            ax.annotate(label, (mx + off_x, my + off_y), ha="center", va="center",
                        fontsize=7, color=color,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.8))

    return pos


def _mark_initial_positions(ax, env: IslandGraph, cargos: Optional[List[Cargo]] = None,
                             pairs: Optional[List[Pair]] = None):
    """
    Отмечает на графе начальные позиции грузов (треугольник=старт, звезда=финиш)
    и роботов (квадрат с подписью id паpы для доставщика, ромб для строителя).
    Маркеры рисуются СО СМЕЩЕНИЕМ от центра узла (не на самом узле), чтобы не
    закрывать номер острова; если несколько меток приходятся на один узел,
    они дополнительно разносятся по кругу вокруг него.
    """
    pos = {v: env.node_pos(v) for v in env.G.nodes}
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    marker_r = scale * 0.032   # расстояние маркера от центра узла
    label_step = scale * 0.038  # дополнительный шаг подписи над/под маркером

    # для каждого узла считаем, сколько меток уже размещено - чтобы раскладывать
    # их по кругу вокруг узла, а не друг на друге
    placed_at_node: dict = {}

    def place(node, marker, color, label, label_above: bool):
        cx, cy = pos[node]
        k = placed_at_node.get(node, 0)
        placed_at_node[node] = k + 1
        # раскладываем последовательные метки этого узла по компасным направлениям
        angles = [90, 270, 0, 180, 45, 135, 225, 315]
        angle = angles[k % len(angles)]
        import math as _m
        mx = cx + marker_r * _m.cos(_m.radians(angle))
        my = cy + marker_r * _m.sin(_m.radians(angle))
        ax.scatter(mx, my, s=300, marker=marker,
                   facecolors=color if color else "none",
                   edgecolors="#1a1a1a", linewidths=1.6, zorder=6,
                   alpha=0.95 if color else 1.0)
        ly = my + (label_step if label_above else -label_step)
        ax.annotate(label, (mx, ly), ha="center",
                    va="bottom" if label_above else "top",
                    fontsize=7.2, fontweight="bold",
                    color=color if color else "#1a1a1a",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88))

    if cargos:
        for c in cargos:
            place(c.v_start, "^", None, f"старт {c.id}", label_above=True)
            place(c.v_finish, "*", None, f"финиш {c.id}", label_above=True)

    if pairs:
        for i, p in enumerate(pairs):
            color = _PAIR_PALETTE[i % len(_PAIR_PALETTE)]
            place(p.deliverer_pos, "s", color, f"{p.id} (доставщик)", label_above=False)
            place(p.builder_pos, "D", color, f"{p.id} (строитель)", label_above=False)


def plot_priority_graph(cargos: List[Cargo], levels: dict, priorities: dict,
                         title: str = "Граф приоритетов задач доставки",
                         ax=None, show: bool = True):
    """
    Рисует граф приоритетов 'слева-направо' по дискретным уровням:
    самые приоритетные узлы - слева (уровень 1), наименее приоритетные - справа
    (уровень k). Узлы одного уровня располагаются в один столбец.

    levels: {cargo_id: level}  (1 = самый приоритетный)
    priorities: {cargo_id: p_value} - для подписи числового значения приоритета
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))

    max_level = max(levels.values())
    # группируем грузы по уровням
    by_level = {lv: [] for lv in range(1, max_level + 1)}
    for c in cargos:
        by_level[levels[c.id]].append(c)

    node_pos = {}
    x_step = 3.0  # горизонтальное расстояние между уровнями (увеличено для читаемости подписей)
    for lv in range(1, max_level + 1):
        items = by_level[lv]
        n = len(items)
        for i, c in enumerate(items):
            # уровень 1 - крайний левый (x маленький), уровень k - крайний правый
            x = lv * x_step
            y = (i - (n - 1) / 2.0) * 1.8
            node_pos[c.id] = (x, y)

    # рёбра между соседними уровнями (полный двудольный граф между слоями) -
    # отражают структуру 'дерева приоритетов': переход от более приоритетных
    # задач к менее приоритетным
    for lv in range(1, max_level):
        for c_from in by_level[lv]:
            for c_to in by_level[lv + 1]:
                x1, y1 = node_pos[c_from.id]
                x2, y2 = node_pos[c_to.id]
                ax.plot([x1, x2], [y1, y2], color="#c9c9c9", linewidth=1.0,
                        zorder=1, alpha=0.6)

    # цветовая градация по уровню: тёмный (высокий приоритет) -> светлый (низкий)
    cmap = plt.cm.YlOrRd_r
    for lv in range(1, max_level + 1):
        for c in by_level[lv]:
            x, y = node_pos[c.id]
            color = cmap(0.15 + 0.65 * (lv - 1) / max(1, max_level - 1))
            ax.scatter(x, y, s=1400, color=color, edgecolors="#2d2d2d",
                       linewidths=1.8, zorder=3)
            p_val = priorities.get(c.id, None)
            p_label = f"\np={p_val:.3f}" if p_val is not None and p_val != float("inf") else "\np=inf"
            ax.annotate(f"{c.id}{p_label}", (x, y), ha="center", va="center",
                       fontsize=9, fontweight="bold", zorder=4)

    y_values_all = [y for x, y in node_pos.values()]
    y_min = min(y_values_all) if y_values_all else -1.0
    y_max = max(y_values_all) if y_values_all else 1.0
    label_y = y_min - 2.0
    arrow_y = y_max + 1.5

    for lv in range(1, max_level + 1):
        ax.annotate(f"Уровень {lv}", (lv * x_step, label_y), ha="center", fontsize=10,
                   fontweight="bold", color="#555555")

    ax.annotate("← выше приоритет", (x_step * 0.5, arrow_y), ha="left", fontsize=9,
               style="italic", color="#777777")
    ax.annotate("ниже приоритет →", (max_level * x_step + x_step * 0.5, arrow_y),
               ha="right", fontsize=9, style="italic", color="#777777")

    ax.set_xlim(x_step * 0.3, max_level * x_step + x_step * 0.7)
    ax.set_ylim(label_y - 1.0, arrow_y + 1.0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    if standalone:
        plt.tight_layout()
        if show:
            plt.show()
        return fig


def plot_environment(env: IslandGraph, result: Optional[TaskResult] = None,
                      cargo: Optional[Cargo] = None,
                      all_cargos: Optional[List[Cargo]] = None,
                      all_pairs: Optional[List[Pair]] = None,
                      show_costs: bool = True,
                      title: str = "Среда G и маршрут доставки",
                      ax=None, show: bool = True):
    """
    Рисует граф среды. Если передан result (TaskResult) - подсвечивает маршрут
    доставки P* синим и построенные мосты оранжевым, остальные blocked-рёбра
    остаются серыми пунктирными ("доступны, но не построены").

    all_cargos/all_pairs (если заданы) - отмечают НАЧАЛЬНЫЕ позиции всех грузов
    (старт/финиш) и пар роботов (доставщик/строитель) на графе сразу, для
    обзорной картины "карта в начальный момент".
    cargo (если задан, без all_cargos) - отмечает старт/финиш ОДНОГО груза,
    для использования в поэтапной (per-cargo) визуализации маршрута.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    pos = _draw_base_graph(ax, env, show_costs=show_costs)

    if result is not None and result.feasible:
        route_edges = list(zip(result.path[:-1], result.path[1:]))
        bridge_set = set(result.bridges) | set((v, u) for (u, v) in result.bridges)

        non_bridge_route = [(u, v) for (u, v) in route_edges if (u, v) not in bridge_set]
        bridge_route = [(u, v) for (u, v) in route_edges if (u, v) in bridge_set]

        nx.draw_networkx_edges(env.G, pos, edgelist=non_bridge_route, ax=ax,
                                edge_color=_COLOR_ROUTE, width=4.0, style="solid")
        nx.draw_networkx_edges(env.G, pos, edgelist=bridge_route, ax=ax,
                                edge_color=_COLOR_BRIDGE_BUILT, width=4.0, style="solid")

        nx.draw_networkx_nodes(env.G, pos, nodelist=result.path, ax=ax,
                                node_color=_COLOR_ROUTE, edgecolors=_COLOR_ISLAND,
                                linewidths=1.8, node_size=600)
        nx.draw_networkx_labels(env.G, pos, ax=ax,
                                 labels={v: str(v) for v in result.path},
                                 font_size=10, font_weight="bold", font_color="white")

    if all_cargos is not None or all_pairs is not None:
        _mark_initial_positions(ax, env, cargos=all_cargos, pairs=all_pairs)
    elif cargo is not None:
        ax.scatter(*env.node_pos(cargo.v_start), s=260, marker="^",
                   facecolors="none", edgecolors="#1a1a1a", linewidths=2, zorder=5)
        ax.scatter(*env.node_pos(cargo.v_finish), s=260, marker="*",
                   facecolors="none", edgecolors="#1a1a1a", linewidths=2, zorder=5)

    legend_items = [
        mpatches.Patch(color=_COLOR_FREE, label="Существующая переправа (E_free)"),
        mpatches.Patch(color=_COLOR_BLOCKED, label="Доступна к строительству (E_blocked)"),
    ]
    if result is not None and result.feasible:
        legend_items += [
            mpatches.Patch(color=_COLOR_ROUTE, label="Маршрут доставки P*"),
            mpatches.Patch(color=_COLOR_BRIDGE_BUILT, label="Построенный мост (E_b)"),
        ]
    if all_cargos is not None:
        legend_items.append(mpatches.Patch(facecolor="none", edgecolor="#1a1a1a",
                                            label="▲ старт груза   ★ финиш груза"))
    if all_pairs is not None:
        legend_items.append(mpatches.Patch(facecolor="none", edgecolor="#1a1a1a",
                                            label="■ доставщик   ◆ строитель"))

    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              fontsize=8, framealpha=0.9, ncol=2)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    ax.set_aspect("equal")
    ax.margins(x=0.1, y=0.35)
    ax.axis("off")


    if standalone:
        plt.tight_layout()
        if show:
            plt.show()
        return fig


def plot_schedule_routes(env: IslandGraph, outcome: ScheduleOutcome,
                          show_costs: bool = False,
                          title: str = "Итоговое расписание: маршруты всех пар",
                          ax=None, show: bool = True):
    """Рисует граф среды со всеми маршрутами расписания одновременно, каждая пара - свой цвет."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5.5))

    pos = _draw_base_graph(ax, env, show_costs=show_costs)

    pair_color = {}
    color_idx = 0

    for entry in outcome.schedule:
        if entry.pair_id not in pair_color:
            pair_color[entry.pair_id] = _PAIR_PALETTE[color_idx % len(_PAIR_PALETTE)]
            color_idx += 1
        color = pair_color[entry.pair_id]

        path = entry.result.path
        route_edges = list(zip(path[:-1], path[1:]))
        bridge_set = set(entry.result.bridges) | set((v, u) for (u, v) in entry.result.bridges)

        non_bridge_route = [(u, v) for (u, v) in route_edges if (u, v) not in bridge_set]
        bridge_route = [(u, v) for (u, v) in route_edges if (u, v) in bridge_set]

        nx.draw_networkx_edges(env.G, pos, edgelist=non_bridge_route, ax=ax,
                                edge_color=color, width=3.5, style="solid", alpha=0.85)
        nx.draw_networkx_edges(env.G, pos, edgelist=bridge_route, ax=ax,
                                edge_color=color, width=3.5, style="solid", alpha=0.85)
        # построенные мосты дополнительно маркируем точками поверх линии
        for (u, v) in bridge_route:
            mx = (env.node_pos(u)[0] + env.node_pos(v)[0]) / 2
            my = (env.node_pos(u)[1] + env.node_pos(v)[1]) / 2
            ax.scatter(mx, my, s=90, marker="s", color=color,
                       edgecolors="black", linewidths=1.0, zorder=6)

    legend_items = [
        mpatches.Patch(color=_COLOR_FREE, label="Существующая переправа (E_free)"),
        mpatches.Patch(color=_COLOR_BLOCKED, label="Доступна к строительству (не использована)"),
    ]
    for pair_id, color in pair_color.items():
        legend_items.append(mpatches.Patch(color=color, label=f"Маршрут пары {pair_id}"))
    legend_items.append(mpatches.Patch(facecolor="white", edgecolor="black",
                                        label="■ построенный мост"))

    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              fontsize=8, framealpha=0.9, ncol=3)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    if standalone:
        plt.tight_layout()
        if show:
            plt.show()
        return fig


def plot_pareto(points: List[ParetoPoint], front: List[ParetoPoint],
                 best: Optional[ParetoPoint] = None,
                 title: str = "Парето-фронт: W_d_total(L) vs W_b_total(L)",
                 ax=None, show: bool = True):
    """
    Точечная диаграмма всех допустимых L (серые точки), выделенный Парето-фронт
    (оранжевая линия с маркерами), и отдельно отмеченная компромиссная точка L*.
    Рядом с каждой точкой фронта подписано значение L.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 6))

    all_wd = [p.W_d_total for p in points]
    all_wb = [p.W_b_total for p in points]
    ax.scatter(all_wd, all_wb, color="#b0b0b0", s=70, zorder=2, label="Все допустимые L")

    front_sorted = sorted(front, key=lambda p: p.W_d_total)
    front_wd = [p.W_d_total for p in front_sorted]
    front_wb = [p.W_b_total for p in front_sorted]
    ax.plot(front_wd, front_wb, color=_COLOR_BRIDGE_BUILT, linewidth=2, zorder=3,
            marker="o", markersize=9, label="Парето-фронт")

    # группируем точки фронта по совпадающим (W_d, W_b) - несколько L часто
    # дают идентичный результат (один и тот же маршрут остаётся оптимальным
    # на целом интервале L); подписываем диапазон, а не каждое значение отдельно
    grouped = {}
    for p in front_sorted:
        key = (round(p.W_d_total, 6), round(p.W_b_total, 6))
        grouped.setdefault(key, []).append(p.L)

    for (wd, wb), Ls in grouped.items():
        Ls_sorted = sorted(Ls)
        label = f"L={Ls_sorted[0]}" if len(Ls_sorted) == 1 else f"L={Ls_sorted[0]}–{Ls_sorted[-1]}"
        ax.annotate(label, (wd, wb), textcoords="offset points", xytext=(10, 8), fontsize=9)

    if best is not None:
        ax.scatter([best.W_d_total], [best.W_b_total], color="#c0392b", s=220,
                  marker="*", zorder=5, label=f"Компромисс L*={best.L}")

    ax.set_xlabel("W_d_total  (суммарная стоимость доставщиков)", fontsize=11)
    ax.set_ylabel("W_b_total  (суммарная стоимость строителей)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    if standalone:
        plt.tight_layout()
        if show:
            plt.show()
        return fig
