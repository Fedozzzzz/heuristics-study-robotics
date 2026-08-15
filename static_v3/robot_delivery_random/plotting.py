"""
Графики и экспорт CSV для static_v3.

Все графики строятся по ФАКТИЧЕСКОЙ стоимости выполнения всех операций
(real = W_d_total + W_b_total, Шаг 5) и производным от неё метрикам баланса:
никакой оценки "до выполнения" у этой модели нет.

Кривые разделяются по variant -- режиму случайного распределения грузов
(balanced / uniform).

Используется matplotlib (без специфичного стиля, чтобы не тянуть лишние
зависимости).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .experiment import ExperimentRow


def export_csv(rows: Sequence[ExperimentRow], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)


def _group_by_variant(rows: Sequence[ExperimentRow]) -> Dict[str, List[ExperimentRow]]:
    out = defaultdict(list)
    for r in rows:
        out[r.variant].append(r)
    return out


def _by_ncargos(rows: Sequence[ExperimentRow]) -> Dict[int, List[ExperimentRow]]:
    out = defaultdict(list)
    for r in rows:
        out[r.n_cargos].append(r)
    return out


def plot_real_vs_ncargos(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """Главный график модели: средняя фактическая стоимость выполнения всех
    операций (real) в зависимости от числа грузов, отдельная линия на режим."""
    by_v = _group_by_variant(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in sorted(by_v.keys()):
        by_n = _by_ncargos(by_v[v])
        ns = sorted(by_n.keys())
        vals = [mean(r.real for r in by_n[n]) for n in ns]
        ax.plot(ns, vals, marker="o", label=v)
    ax.set_xlabel("n_cargos")
    ax.set_ylabel("средняя стоимость всех операций (real)")
    ax.set_title("static_v3: фактическая стоимость доставки всех грузов")
    ax.legend(fontsize=8)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_real_spread(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """Разброс real, порождённый самой случайностью модели.

    Смысл графика специфичен именно для static_v3: пары и распределение грузов
    разыгрываются случайно, поэтому на ОДНОМ И ТОМ ЖЕ сценарии разные сиды
    модели дают разную стоимость. Полоса -- min..max по всем прогонам точки,
    линия -- среднее. Чем шире полоса, тем сильнее результат зависит от
    везения, а не от входных данных (осмысленно при --n-repeats > 1)."""
    by_v = _group_by_variant(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in sorted(by_v.keys()):
        by_n = _by_ncargos(by_v[v])
        ns = sorted(by_n.keys())
        avg = [mean(r.real for r in by_n[n]) for n in ns]
        lo = [min(r.real for r in by_n[n]) for n in ns]
        hi = [max(r.real for r in by_n[n]) for n in ns]
        line, = ax.plot(ns, avg, marker="o", label=f"{v} (среднее)")
        ax.fill_between(ns, lo, hi, alpha=0.15, color=line.get_color(),
                        label=f"{v} (min..max)")
    ax.set_xlabel("n_cargos")
    ax.set_ylabel("стоимость всех операций (real)")
    ax.set_title("Разброс стоимости, порождённый случайностью модели")
    ax.legend(fontsize=8)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_real_cv_vs_ncargos(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """Коэффициент вариации real (стандартное отклонение / среднее, %) внутри
    ОДНОГО сценария -- то есть чистый вклад случайности Шагов 2-4, без вклада
    разброса самих сценариев. Считается по повторам и усредняется по
    сценариям; требует --n-repeats > 1."""
    by_v = _group_by_variant(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for v in sorted(by_v.keys()):
        by_n = _by_ncargos(by_v[v])
        ns, vals = [], []
        for n in sorted(by_n.keys()):
            per_scenario = defaultdict(list)
            for r in by_n[n]:
                per_scenario[r.seed].append(r.real)
            cvs = [
                100.0 * pstdev(v_list) / mean(v_list)
                for v_list in per_scenario.values()
                if len(v_list) > 1 and mean(v_list) > 1e-9
            ]
            if cvs:
                ns.append(n)
                vals.append(mean(cvs))
        if ns:
            ax.plot(ns, vals, marker="o", label=v)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("n_cargos")
    ax.set_ylabel("коэффициент вариации real внутри сценария, %")
    ax.set_title("Насколько результат зависит от везения, а не от входных данных")
    ax.legend(fontsize=8)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pair_load_balance(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """Балансировка по ЧИСЛУ грузов: сколько грузов достаётся самой загруженной
    и самой незагруженной паре. Пунктир -- идеально равномерное
    n_cargos / n_pairs. Именно здесь balanced и uniform расходятся сильнее
    всего: у balanced обе кривые прижаты к пунктиру, у uniform верхняя уходит
    вверх, а нижняя может лежать на нуле."""
    by_v = _group_by_variant(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ideal_ns: List[int] = []
    ideal_vals: List[float] = []
    for v in sorted(by_v.keys()):
        by_n = _by_ncargos(by_v[v])
        ns = sorted(by_n.keys())
        hi = [mean(r.max_pair_load for r in by_n[n]) for n in ns]
        lo = [mean(r.min_pair_load for r in by_n[n]) for n in ns]
        line, = ax.plot(ns, hi, marker="o", label=f"{v}: самая загруженная пара")
        ax.plot(ns, lo, marker="v", linestyle=":", color=line.get_color(),
                label=f"{v}: самая незагруженная пара")
        if not ideal_ns:
            ideal_ns = ns
            ideal_vals = [
                n / by_n[n][0].n_pairs if by_n[n][0].n_pairs else 0.0 for n in ns
            ]
    if ideal_ns:
        ax.plot(ideal_ns, ideal_vals, linestyle="--", color="grey",
                label="равномерно (n_cargos / n_pairs)")
    ax.set_xlabel("n_cargos")
    ax.set_ylabel("грузов у пары")
    ax.set_title("Балансировка распределения грузов между парами")
    ax.legend(fontsize=8)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pair_cost_balance(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """Балансировка в ЕДИНИЦАХ СТОИМОСТИ: во сколько раз работа самой
    загруженной пары превышает идеально равномерную долю real / n_pairs
    (cost_imbalance). 1.0 -- идеальный баланс, поэтому чем ниже кривая, тем
    лучше. Одинаковое ЧИСЛО грузов у пар ещё не значит одинаковую работу,
    поэтому даже у balanced эта кривая заметно выше единицы."""
    by_v = _group_by_variant(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in sorted(by_v.keys()):
        by_n = _by_ncargos([r for r in by_v[v] if r.all_delivered])
        ns = sorted(by_n.keys())
        if not ns:
            continue
        vals = [mean(r.cost_imbalance for r in by_n[n]) for n in ns]
        ax.plot(ns, vals, marker="o", label=v)
    ax.axhline(1.0, linestyle="--", color="grey", label="идеальный баланс (1.0)")
    ax.set_xlabel("n_cargos")
    ax.set_ylabel("max_pair_cost / (real / n_pairs)")
    ax.set_title("Перекос загрузки пар по стоимости")
    ax.legend(fontsize=8)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_idle_vs_ncargos(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """Суммарный простой на барьерах раундов (Шаг 4.3): пары, управившиеся
    раньше всех, ждут самую долгую доставку раунда."""
    by_v = _group_by_variant(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in sorted(by_v.keys()):
        by_n = _by_ncargos(by_v[v])
        ns = sorted(by_n.keys())
        vals = [mean(r.idle_total for r in by_n[n]) for n in ns]
        ax.plot(ns, vals, marker="o", label=v)
    ax.set_xlabel("n_cargos")
    ax.set_ylabel("idle_total")
    ax.set_title("Простой пар на барьерах раундов")
    ax.legend(fontsize=8)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_real_bars(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """Средний real по каждому режиму -- сводка одним столбиком на режим."""
    by_v = _group_by_variant(rows)
    variants = sorted(by_v.keys())
    vals = [mean(r.real for r in by_v[v]) for v in variants]

    fig, ax = plt.subplots(figsize=(max(6, len(variants) * 2.2), 5))
    ax.bar(variants, vals)
    ax.set_ylabel("средняя стоимость всех операций (real)")
    ax.set_title("static_v3: стоимость по режимам случайного распределения")
    plt.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_win_rate(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """WIN-RATE: доля прогонов, где режим дал минимальный real среди
    сравниваемых. Сравнение идёт внутри группы с ОДИНАКОВЫМИ сценарием и сидом
    модели (n_cargos, seed, rng_seed), то есть режимы сопоставляются на
    буквально одной и той же случайности."""
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r.n_cargos, r.seed, r.rng_seed)].append(r)

    wins: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    for _key, group in by_key.items():
        delivered = [r for r in group if r.all_delivered]
        if not delivered:
            continue
        best = min(r.real for r in delivered)
        for r in group:
            total[r.variant] += 1
        for r in delivered:
            if abs(r.real - best) < 1e-6:
                wins[r.variant] += 1

    labels = sorted(total.keys())
    rates = [wins[g] / total[g] if total[g] else 0.0 for g in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2.0), 5))
    ax.bar(labels, rates)
    ax.set_ylabel("win-rate (доля прогонов с минимальным real)")
    ax.set_title("WIN-RATE по фактической стоимости")
    ax.set_ylim(0, 1)
    plt.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
