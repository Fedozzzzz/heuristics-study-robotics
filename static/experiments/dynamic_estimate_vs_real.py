# -*- coding: utf-8 -*-
"""
ВИЛКА "ОЦЕНКА vs ФАКТ" ВНУТРИ ДИНАМИЧЕСКОЙ МОДЕЛИ, по эвристикам приоритета.

Уходим от сравнения со статикой. Здесь эталон - НЕ другая модель, а
ФАКТИЧЕСКАЯ стоимость выполнения самого динамического расписания
(real_total). Вопрос: насколько ОЦЕНКА стоимости расходится с ФАКТОМ, и
какая эвристика приоритета даёт наименьший разрыв.

Для каждого прогона динамики (core/algorithm_5_dynamic_rounds) считается
вилка (compute_dynamic_cost_bracket):

    real  <=  estimated_pool  <=  estimated_raw

  real           - фактическая сумма W_d+W_b по прогону (мосты
                   переиспользуются естественно между раундами одной пары);
  estimated_raw  - наивная оценка: каждый ФАКТИЧЕСКИ выполненный груз
                   пересчитан от позиции пары НА НАЧАЛО ЕГО РАУНДА, но
                   ИЗОЛИРОВАННО (built_bridges=∅) - "как если бы груз был
                   единственным". Систематически переоценивает;
  estimated_pool - raw с коррекцией пула мостов (формула из
                   Формулы_модели.docx: Σ_i W_b^i − Σ_e (c_e−1)·w_build(e),
                   c_e - по фактически выполненным грузам пары). Убирает
                   завышение от повторной оплаты одних и тех же мостов.

Две метрики разрыва (обе - "на сколько % оценка выше факта"):
  gap_raw  = 100·(estimated_raw  / real − 1)
  gap_pool = 100·(estimated_pool / real − 1)

Разность (gap_raw − gap_pool) показывает, какую часть ошибки наивной оценки
объясняет ПЕРЕИСПОЛЬЗОВАНИЕ МОСТОВ. Остаток gap_pool - это ошибка от
СМЕЩЕНИЯ ПОЗИЦИИ пары между раундами (коррекция пула её не трогает).

ЛУЧШАЯ эвристика по этому критерию - та, у которой оценка (особенно pool)
ближе всего к факту: её приоритет строится на числах, которым можно
доверять, а не на систематически завышенном прогнозе.

Обе величины считаются по ОДНОМУ И ТОМУ ЖЕ прогону, поэтому сравнение
эвристик здесь ПАРНОЕ по сценариям автоматически.

ЗАПУСК: python3 dynamic_estimate_vs_real.py
"""
import os
import sys
import json
import time
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))

import matplotlib.pyplot as plt

from scenario_generator import build_environment, build_cargos_and_pairs
from heuristic_cheapest_bridge import find_route_cheapest_bridge
from algorithm_5_dynamic_rounds import (run_dynamic_rounds,
                                         compute_dynamic_cost_bracket)

# ---- Параметры ---------------------------------------------------------------
N_ISLANDS    = 20
N_PAIRS      = 4
N_SEEDS      = 15
N_CARGOS_MAX = 44
CARGO_STEP   = 8          # динамика квадратична по грузам - сетка разрежена
L            = 6.0

OUT_DIR    = "outputs"
CACHE_FILE = os.path.join(OUT_DIR, "dyn_estimate_vs_real_cache.json")

HEURISTICS = {
    "direct":  ("Прямая  p=W_T",       "#c0392b"),
    "inverse": ("Обратная  p=1/W_T",   "#2b6cb0"),
    "ratio":   ("Отношение  p=W_b/W_d", "#27ae60"),
    "random":  ("Случайный (бейзлайн)", "#7f8c8d"),
}
# ------------------------------------------------------------------------------


def run_one(kind, n_cargos, seed):
    env = build_environment(n_islands=N_ISLANDS, seed=seed)
    cargos, pairs = build_cargos_and_pairs(n_islands=N_ISLANDS, n_pairs=N_PAIRS,
                                            n_cargos=n_cargos, seed=seed)
    outcome = run_dynamic_rounds(env, cargos, pairs, L=L,
                                  route_fn=find_route_cheapest_bridge,
                                  selection_strategy="priority",
                                  priority_kind=kind, seed=seed)
    if not outcome.all_delivered:
        return None
    b = compute_dynamic_cost_bracket(env, outcome,
                                     route_fn=find_route_cheapest_bridge, L=L)
    return {"real": b.real_total, "pool": b.estimated_pool, "raw": b.estimated_raw,
            "gap_pool": b.gap_pool_pct, "gap_raw": b.gap_raw_pct}


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
    cargo_range = list(range(N_PAIRS, N_CARGOS_MAX + 1, CARGO_STEP))

    # raw_data[kind][i] = список dict-метрик по seed (или None)
    raw_data = {k: [] for k in HEURISTICS}

    for n_cargos in cargo_range:
        for k in HEURISTICS:
            ckey = f"{k}_{n_cargos}"
            if ckey in cache:
                vals = cache[ckey]
            else:
                vals = [run_one(k, n_cargos, seed) for seed in range(N_SEEDS)]
                cache[ckey] = vals
                save_cache(cache)
            raw_data[k].append(vals)

        line = "  ".join(
            f"{k[:3]}:{_avg(raw_data[k][-1], 'gap_pool'):+.0f}/"
            f"{_avg(raw_data[k][-1], 'gap_raw'):+.0f}"
            for k in HEURISTICS)
        print(f"n={n_cargos:3d}  gap%(pool/raw)  {line}")

    return cargo_range, raw_data


def _avg(vals, field):
    got = [v[field] for v in vals if v is not None]
    return mean(got) if got else float("nan")


def plot_gaps(cargo_range, raw_data, filename):
    """gap_pool (сплошная) и gap_raw (пунктир) по числу грузов, цвет=эвристика."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for k, (label, color) in HEURISTICS.items():
        pool = [_avg(raw_data[k][i], "gap_pool") for i in range(len(cargo_range))]
        raw = [_avg(raw_data[k][i], "gap_raw") for i in range(len(cargo_range))]
        ax.plot(cargo_range, pool, marker="o", markersize=3, color=color,
                linewidth=2.2, label=f"{label} [pool]")
        ax.plot(cargo_range, raw, marker="s", markersize=3, color=color,
                linewidth=1.4, linestyle="--", label=f"{label} [raw]")
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Число грузов")
    ax.set_ylabel("Превышение оценки над фактом, %  (0 = идеальная оценка)")
    ax.set_title("Разрыв оценки и факта внутри динамики\n"
                 "(pool - сплошная, raw - пунктир; ближе к 0 = точнее)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(f"{N_ISLANDS} островов, {N_PAIRS} пары, {N_SEEDS} seed", fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def plot_bracket_bars(cargo_range, raw_data, filename):
    """Столбцы real/pool/raw в самой большой точке по числу грузов - наглядная
    вилка для каждой эвристики."""
    i = len(cargo_range) - 1  # последняя (самая крупная) точка
    fig, ax = plt.subplots(figsize=(10, 6))
    import numpy as np
    labels = [HEURISTICS[k][0] for k in HEURISTICS]
    reals = [_avg(raw_data[k][i], "real") for k in HEURISTICS]
    pools = [_avg(raw_data[k][i], "pool") for k in HEURISTICS]
    raws = [_avg(raw_data[k][i], "raw") for k in HEURISTICS]

    x = np.arange(len(labels))
    w = 0.26
    ax.bar(x - w, reals, w, label="real (факт)", color="#2c3e50")
    ax.bar(x, pools, w, label="pool (оценка с коррекцией)", color="#2b6cb0")
    ax.bar(x + w, raws, w, label="raw (наивная оценка)", color="#c0392b", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Стоимость Φ")
    ax.set_title(f"Вилка real ≤ pool ≤ raw по эвристикам "
                 f"(n_cargos = {cargo_range[i]})", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {filename}")


def report(cargo_range, raw_data):
    print("\n=== СРЕДНИЙ РАЗРЫВ ОЦЕНКА/ФАКТ по эвристикам (по всем n_cargos) ===")
    print("    gap_pool - оценка с коррекцией пула; gap_raw - наивная оценка")
    print("    (меньше = оценка ближе к факту = эвристике можно доверять)")
    rows = []
    for k, (label, _) in HEURISTICS.items():
        pool_all, raw_all = [], []
        for i in range(len(cargo_range)):
            for v in raw_data[k][i]:
                if v is not None:
                    pool_all.append(v["gap_pool"])
                    raw_all.append(v["gap_raw"])
        if pool_all:
            gp, gr = mean(pool_all), mean(raw_all)
            rows.append((k, label, gp, gr))
            print(f"  {label:24s}: gap_pool={gp:+6.2f}%   gap_raw={gr:+6.2f}%   "
                  f"вклад переиспользования={gr-gp:5.2f}пп")

    if rows:
        best = min(rows, key=lambda r: r[2])
        print(f"\n  ЛУЧШАЯ по близости оценки к факту (min gap_pool): "
              f"{best[1]} ({best[2]:+.2f}%)")

    # win-rate по минимальному |gap_pool| на сценарии
    print("\n=== WIN-RATE: у какой эвристики оценка pool ЧАЩЕ ВСЕГО ближе к факту ===")
    wins = {k: 0 for k in HEURISTICS}
    n_comp = 0
    for i in range(len(cargo_range)):
        for si in range(N_SEEDS):
            row = {}
            for k in HEURISTICS:
                v = raw_data[k][i][si]
                if v is not None:
                    row[k] = abs(v["gap_pool"])
            if len(row) < len(HEURISTICS):
                continue
            n_comp += 1
            wins[min(row, key=row.get)] += 1
    for k, (label, _) in HEURISTICS.items():
        pct = 100 * wins[k] / n_comp if n_comp else 0
        print(f"  {label:24s}: {wins[k]:4d} / {n_comp}  ({pct:.1f}%)")


if __name__ == "__main__":
    t0 = time.time()
    cargo_range, raw_data = sweep()
    print(f"\nСвип завершён за {time.time()-t0:.1f}s")

    plot_gaps(cargo_range, raw_data, os.path.join(OUT_DIR, "dyn_gap_pool_raw.png"))
    plot_bracket_bars(cargo_range, raw_data, os.path.join(OUT_DIR, "dyn_bracket_bars.png"))
    report(cargo_range, raw_data)
