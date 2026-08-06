"""
ЭКСПЕРИМЕНТ: как получить БОЛЬШЕ точек на Парето-фронте.

Ключевой вывод предыдущих экспериментов: число точек на фронте определяется
НЕ числом островов и не числом грузов само по себе, а числом НЕЗАВИСИМЫХ
"переломных" порогов длины - то есть числом РАЗНЫХ ситуаций, где альтернатива
"один длинный мост" начинает конкурировать с альтернативой "обход из нескольких
коротких мостов" на разных, не совпадающих друг с другом значениях L.

Конструкция: K независимых "блоков", каждый - это треугольник из 3 островов
(старт-промежуток-финиш), где есть выбор между:
  а) ОБХОДОМ через промежуточный остров (2 коротких моста, ФИКСИРОВАННАЯ
     стоимость, не зависящая от L)
  б) ПРЯМЫМ мостом (1 мост, длина и стоимость строительства РАЗНЫЕ для
     каждого блока) - открывается на своём пороге L

Блоки физически НЕ связаны переправами, требующими строительства - они
соединены друг с другом ТОЛЬКО узкими бесплатными (E_free) переходами,
чтобы один доставщик мог последовательно обслужить все грузы (общий ресурс),
но при этом сам выбор маршрута в каждом блоке был полностью независим от
остальных - что и даёт контролируемое число точек фронта.

КАЛИБРОВКА (для каждого блока должно выполняться одновременно):
  Wb_direct > Wb_detour   - чтобы был настоящий trade-off, а не доминирование
  Wd_direct + Wb_direct < Wd_detour + Wb_detour
                           - чтобы алгоритм (минимизирующий сумму среди
                             недоминируемых) реально выбрал прямой мост,
                             когда он становится доступен по L
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))
import matplotlib.pyplot as plt

from delivery_model import IslandGraph, Cargo, Pair
from algorithm_2 import run_scheduling
from algorithm_4 import build_cost_curves, pareto_front, choose_compromise, is_concave_front
from visualize import plot_pareto


def build_environment(configs):
    """
    configs: список (short_len, direct_len, build_direct) - один на каждый блок.
    short_len   - длина каждого из двух коротких сегментов обхода (фиксирована)
    direct_len  - длина прямого моста (свой порог L для каждого блока)
    build_direct - стоимость строительства прямого моста (откалибрована так,
                   чтобы выполнялись оба условия выше)
    """
    env = IslandGraph()
    cargos = []
    finish_nodes = []
    node_id = 0

    for i, (short_len, direct_len, build_direct) in enumerate(configs):
        s, mid, f = node_id, node_id + 1, node_id + 2
        env.add_island(s, w_v=0.2, pos=(i * 6, 0))
        env.add_island(mid, w_v=0.2, pos=(i * 6 + direct_len / 2, 1.2))
        env.add_island(f, w_v=0.2, pos=(i * 6 + direct_len, 0))

        # обход: фиксированная стоимость, не зависящая от L (всегда доступен)
        env.add_edge(s, mid, kind="blocked", w_E=0.5, length=short_len, w_build=1.5)
        env.add_edge(mid, f, kind="blocked", w_E=0.5, length=short_len, w_build=1.5)
        # прямой мост: свой порог длины и своя калибровка стоимости стройки
        env.add_edge(s, f, kind="blocked", w_E=0.2, length=direct_len, w_build=build_direct)

        cargos.append(Cargo(id=f"c{i + 1}", v_start=s, v_finish=f, assigned_pair="pair1"))

        if finish_nodes:
            # узкий бесплатный переход между блоками - НЕ требует строительства
            # и не влияет на конкуренцию внутри блока
            env.add_edge(finish_nodes[-1], s, kind="free", w_E=0.1, length=0.1)
        finish_nodes.append(f)
        node_id += 3

    pairs = [Pair(id="pair1", deliverer_pos=0, builder_pos=0)]
    return env, cargos, pairs


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    # 5 блоков, build_direct подобран в "окне" (4.4, 5.4) для данных значений
    # short_len=1.0, w_E_direct=0.2, w_V=0.2 (см. вывод калибровки в чате) -
    # это гарантирует одновременно trade-off И реальный выбор прямого моста
    configs = [
        (1.0, 2.0, 4.5),
        (1.0, 2.5, 4.7),
        (1.0, 3.0, 4.9),
        (1.0, 3.5, 5.1),
        (1.0, 4.0, 5.3),
    ]

    env, cargos, pairs = build_environment(configs)

    L_grid = [round(0.5 + 0.05 * i, 2) for i in range(120)]
    points = build_cost_curves(env, cargos, pairs, L_grid)
    front = pareto_front(points)
    unique = sorted(set((round(p.W_d_total, 2), round(p.W_b_total, 2)) for p in front))

    print(f"Точек на сетке: {len(points)}")
    print(f"Уникальных недоминируемых точек на фронте: {len(unique)}")
    for wd, wb in unique:
        print(f"  Wd={wd:.2f}  Wb={wb:.2f}")

    concave = is_concave_front(front)
    print(f"Форма фронта: {'вогнутая' if concave else 'выпуклая'}")

    best = choose_compromise(front)
    print(f"Компромисс: L*={best.L}, Wd={best.W_d_total:.2f}, Wb={best.W_b_total:.2f}")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    plot_pareto(points, front, best,
                title=f"5 независимых блоков -> {len(unique)} точек фронта",
                ax=ax, show=False)
    fig.tight_layout()
    fig.savefig("outputs/experiment_more_points.png", dpi=150)
    plt.close(fig)
    print("\nСохранено: experiment_more_points.png")
