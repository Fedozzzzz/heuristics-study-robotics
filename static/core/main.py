"""
ГЛАВНЫЙ ДЕМОНСТРАЦИОННЫЙ СКРИПТ
================================
Сценарий: 10 островов в виде планарного "зигзага" (без пересекающихся рёбер),
4 готовых моста (E_free), 2 пары роботов (доставщик+строитель), 4 груза.
Назначение груз -> пара задано заранее (до запуска модели).

Пайплайн:
  Алгоритм 1 -> приоритет груза по прямому расстоянию + граф приоритетов
  Алгоритм 2 -> выполнение задач каждой парой по своей очереди
  Алгоритм 3 -> поиск маршрута P* и минимального набора мостов E_b
  Алгоритм 4 -> построение Парето-фронта (W_d_total, W_b_total) по сетке L

На выходе:
  - текстовый отчёт в консоль
  - 0_overview.png            - граф среды со стоимостями, стартом грузов и роботов
  - 1_priority_graph.png      - граф приоритетов задач
  - cargo_X_route.png (x4)    - маршрут КАЖДОГО груза отдельно, разными цветами
  - schedule_combined.png     - итоговое решение, все маршруты на одной карте
  - pareto_front.png          - Парето-фронт W_d_total(L) vs W_b_total(L)

ЗАПУСК:  python3 main.py
Зависимости: pip install networkx matplotlib
"""

import matplotlib.pyplot as plt

from delivery_model import IslandGraph, Cargo, Pair
from algorithm_2 import run_scheduling
from algorithm_4 import build_cost_curves, pareto_front, choose_compromise, is_concave_front
from algorithms_1_3 import (assign_priority_levels_for_assigned_pairs,
                             compute_priority_for_assigned_pair,
                             find_route_and_bridges)
from visualize import (plot_environment, plot_priority_graph,
                        plot_schedule_routes, plot_pareto)


# ===========================================================================
# 1. ОПИСАНИЕ СРЕДЫ: планарный "зигзаг" без пересекающихся рёбер.
#    Острова чередуются верх/низ: 0(в)-1(н)-2(в)-3(н)-...-9(н).
#    Основные связи - звенья цепи (соседние острова), альтернативные -
#    короткие "хорды" через один остров (тоже не пересекаются при такой
#    укладке - проверено геометрически, см. check_planar.py).
# ===========================================================================

def build_environment(build_cost_multiplier: float = 1.0) -> IslandGraph:
    """
    build_cost_multiplier - константа γ, домножающая стоимость строительства
    КАЖДОГО моста (w_build) при расчёте маршрута, не изменяя сами исходные
    значения, заданные ниже в add_edge(...). Используйте γ > 1, чтобы сделать
    строительство относительно дороже простого перемещения по уже готовым
    островам/переправам (например, γ=1.5 увеличивает все w_build в полтора
    раза), и γ < 1 - чтобы сделать строительство относительно выгоднее.
    """
    env = IslandGraph(build_cost_multiplier=build_cost_multiplier)

    coords = {
        0: (0, 1.4), 1: (1.6, 0), 2: (3.2, 1.4), 3: (4.8, 0), 4: (6.4, 1.4),
        5: (8.0, 0), 6: (9.6, 1.4), 7: (11.2, 0), 8: (12.8, 1.4), 9: (14.4, 0),
    }
    # стоимость проезда каждого острова (условный "размер"/сложность острова)
    w_v = {0: 0.4, 1: 0.5, 2: 0.4, 3: 0.6, 4: 0.4,
           5: 0.5, 6: 0.4, 7: 0.6, 8: 0.4, 9: 0.5}
    for v, pos in coords.items():
        env.add_island(v, w_v=w_v[v], pos=pos)

    # 4 уже построенных моста (E_free) - чередуются со звеньями, требующими
    # строительства, чтобы ни один маршрут вдоль цепи не был полностью готов
    env.add_edge(1, 2, kind="free", w_E=1.0, length=1.33)
    env.add_edge(3, 4, kind="free", w_E=1.0, length=1.33)
    env.add_edge(5, 6, kind="free", w_E=1.0, length=1.33)
    env.add_edge(7, 8, kind="free", w_E=1.0, length=1.33)

    # доступные к строительству звенья цепи (короткие, дёшево строить);
    # w_build - "физическая" базовая стоимость, build_cost_multiplier
    # применяется поверх неё при расчёте маршрута (см. build_weighted_graph)
    env.add_edge(0, 1, kind="blocked", w_E=1.0, length=1.33, w_build=2.0)
    env.add_edge(2, 3, kind="blocked", w_E=1.0, length=1.33, w_build=2.0)
    env.add_edge(4, 5, kind="blocked", w_E=1.0, length=1.33, w_build=2.0)
    env.add_edge(6, 7, kind="blocked", w_E=1.0, length=1.33, w_build=2.0)
    env.add_edge(8, 9, kind="blocked", w_E=1.0, length=1.33, w_build=2.0)

    # доступные к строительству "верхние" хорды (через чётные острова) -
    # открываются при L>=2.6; ПЕРВЫЙ (и самый "дорогой за единицу выигрыша")
    # шаг вдоль фронта: скромное снижение W_d ценой заметного роста W_b
    env.add_edge(0, 2, kind="blocked", w_E=1.0, length=2.6, w_build=4.0)
    env.add_edge(2, 4, kind="blocked", w_E=1.0, length=2.6, w_build=4.0)
    env.add_edge(4, 6, kind="blocked", w_E=1.0, length=2.6, w_build=4.0)
    env.add_edge(6, 8, kind="blocked", w_E=1.0, length=2.6, w_build=4.0)

    # доступные к строительству "нижние" хорды (через нечётные острова) -
    # открываются позже (L>=3.5); ВТОРОЙ шаг вдоль фронта, "дешевле за единицу
    # выигрыша" чем первый - именно это даёт ВОГНУТЫЙ (геометрически корректный
    # при минимизации двух издержек) Парето-фронт, а не выпуклый. Подробное
    # объяснение разницы между выпуклым/вогнутым фронтом при минимизации -
    # см. комментарий в algorithm_4.py и README.
    env.add_edge(1, 3, kind="blocked", w_E=0.6, length=3.5, w_build=6.0)
    env.add_edge(3, 5, kind="blocked", w_E=0.6, length=3.5, w_build=6.0)
    env.add_edge(5, 7, kind="blocked", w_E=0.6, length=3.5, w_build=6.0)
    env.add_edge(7, 9, kind="blocked", w_E=0.6, length=3.5, w_build=6.0)

    return env


# ===========================================================================
# 2. ГРУЗЫ И ПАРЫ РОБОТОВ
#    Назначение груз -> пара задаётся ЗАРАНЕЕ. Роботы стоят на разных
#    островах, отдельно от старта грузов.
# ===========================================================================

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


CARGO_COLORS = {
    "c1": "#2b6cb0",  # синий
    "c2": "#c0392b",  # красный
    "c3": "#16a085",  # бирюзовый
    "c4": "#8e54a0",  # фиолетовый
}


# ===========================================================================
# 3. ОСНОВНОЙ ПАЙПЛАЙН
# ===========================================================================

def main():
    # Константа γ: домножает стоимость строительства КАЖДОГО моста (w_build).
    # γ > 1 делает строительство дороже относительно перемещения по уже готовым
    # островам/переправам - увеличьте, чтобы модель сильнее "не любила" строить
    # новые мосты без явной необходимости. γ = 1.0 - без изменений (по умолчанию).
    BUILD_COST_MULTIPLIER = 1.0

    env = build_environment(build_cost_multiplier=BUILD_COST_MULTIPLIER)
    cargos, pairs = build_cargos_and_pairs()

    # --- Обзорная картина: граф среды + стоимости + начальные позиции ---
    print("=" * 70)
    print("ОБЗОР СРЕДЫ: острова, переправы, стоимости, начальные позиции")
    print("=" * 70)
    fig0, ax0 = plt.subplots(figsize=(15, 8))
    plot_environment(env, all_cargos=cargos, all_pairs=pairs, show_costs=True,
                      title="0. Граф среды: стоимости, старт грузов и роботов",
                      ax=ax0, show=False)
    fig0.tight_layout()
    fig0.savefig("outputs/0_overview.png", dpi=150)
    plt.close(fig0)
    print("Сохранено: 0_overview.png")

    # --- Алгоритм 1: граф приоритетов ---
    print("\n" + "=" * 70)
    print("АЛГОРИТМ 1: приоритет грузов и граф приоритетов")
    print("=" * 70)
    priorities = {c.id: compute_priority_for_assigned_pair(c, env, pairs) for c in cargos}
    K_LEVELS = 3
    levels = assign_priority_levels_for_assigned_pairs(cargos, env, pairs, k_levels=K_LEVELS)

    for c in sorted(cargos, key=lambda c: levels[c.id]):
        p = priorities[c.id]
        p_str = f"{p:.4f}" if p != float("inf") else "inf"
        print(f"  {c.id}: p={p_str:>8}  ->  уровень {levels[c.id]}")

    fig1, ax1 = plt.subplots(figsize=(10, 5.5))
    plot_priority_graph(cargos, levels, priorities,
                         title="1. Граф приоритетов задач доставки",
                         ax=ax1, show=False)
    fig1.tight_layout()
    fig1.savefig("outputs/1_priority_graph.png", dpi=150)
    plt.close(fig1)
    print("Сохранено: 1_priority_graph.png")

    # --- Алгоритм 4: Парето-фронт по L ---
    # более плотная сетка (шаг 0.1) - даёт больше точек-кандидатов, точнее
    # определяет границы интервалов, на которых меняется оптимальный маршрут
    L_grid = [round(1.0 + 0.1 * i, 1) for i in range(41)]  # 1.0..5.0, шаг 0.1

    print("\n" + "=" * 70)
    print("АЛГОРИТМ 4: построение кривых W_d_total(L) и W_b_total(L)")
    print("=" * 70)
    points = build_cost_curves(env, cargos, pairs, L_grid)

    if not points:
        print("Ни одно значение L из сетки не дало допустимого решения.")
        return

    for p in points:
        print(f"  L={p.L:<5} W_d_total={p.W_d_total:7.2f}   W_b_total={p.W_b_total:7.2f}")

    front = pareto_front(points)
    print("\nПарето-фронт (недоминируемые значения L):")
    for p in front:
        print(f"  L={p.L:<5} W_d_total={p.W_d_total:7.2f}   W_b_total={p.W_b_total:7.2f}")

    concave = is_concave_front(front)
    print(f"\nФорма фронта: {'вогнутая (как классическая КПВ)' if concave else 'выпуклая'}"
          f" {'✓' if concave else '— компромисс почти наверняка совпадёт с краем фронта'}")

    best = choose_compromise(front)
    print(f"\nКомпромиссное L* (метод идеальной точки) = {best.L}")
    print(f"  -> W_d_total = {best.W_d_total:.2f}, W_b_total = {best.W_b_total:.2f}")

    fig_p, ax_p = plt.subplots(figsize=(7.5, 6))
    plot_pareto(points, front, best,
                title="Парето-фронт: W_d_total(L) vs W_b_total(L)",
                ax=ax_p, show=False)
    fig_p.tight_layout()
    fig_p.savefig("outputs/pareto_front.png", dpi=150)
    plt.close(fig_p)
    print("Сохранено: pareto_front.png")

    # --- Финальный прогон расписания при выбранном L* ---
    print("\n" + "=" * 70)
    print(f"ИТОГОВОЕ РАСПИСАНИЕ при L* = {best.L}")
    print("=" * 70)
    final_outcome = run_scheduling(env, cargos, pairs, L=best.L)
    print("Все грузы доставлены:", final_outcome.all_delivered)
    for e in final_outcome.schedule:
        print(f"  Груз {e.cargo_id} -> пара {e.pair_id}: "
              f"путь={e.result.path}, мосты={e.result.bridges}, "
              f"W_d={e.result.W_d:.2f}, W_b={e.result.W_b:.2f}, "
              f"время [{e.start_time:.2f} -> {e.end_time:.2f}]")

    # --- Поэтапные картинки: маршрут КАЖДОГО груза отдельно ---
    print("\n" + "=" * 70)
    print("ПОЭТАПНАЯ ВИЗУАЛИЗАЦИЯ: маршрут каждого груза отдельно")
    print("=" * 70)
    cargo_by_id = {c.id: c for c in cargos}
    for entry in final_outcome.schedule:
        c = cargo_by_id[entry.cargo_id]
        fig_c, ax_c = plt.subplots(figsize=(9, 5))
        plot_environment(env, result=entry.result, cargo=c, show_costs=False,
                          title=f"Маршрут груза {c.id} (пара {entry.pair_id}): "
                                f"W_d={entry.result.W_d:.1f}, W_b={entry.result.W_b:.1f}",
                          ax=ax_c, show=False)
        fig_c.tight_layout()
        fname = f"outputs/cargo_{c.id}_route.png"
        fig_c.savefig(fname, dpi=150)
        plt.close(fig_c)
        print(f"Сохранено: cargo_{c.id}_route.png")

    # --- Итоговая карта со всеми маршрутами вместе (разными цветами по парам) ---
    fig_s, ax_s = plt.subplots(figsize=(11, 6))
    plot_schedule_routes(env, final_outcome, show_costs=False,
                          title=f"Итоговое решение при L*={best.L}: все маршруты",
                          ax=ax_s, show=False)
    fig_s.tight_layout()
    fig_s.savefig("outputs/schedule_combined.png", dpi=150)
    plt.close(fig_s)
    print("Сохранено: schedule_combined.png")

    print("\nГотово. Все файлы сохранены в outputs/")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    main()
