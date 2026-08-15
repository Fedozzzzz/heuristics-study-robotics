"""
Графики и экспорт CSV для static_v2. Кривые разделяются по variant
("<эвристика>/<режим назначения>"), поэтому одни и те же функции годятся и
для сравнения эвристик приоритета, и для сравнения режимов назначения.

Используется matplotlib (без специфичного стиля, чтобы не тянуть лишние
зависимости).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .diagnostics import CostBracket
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


def _group_by_variant(rows: Sequence[ExperimentRow]) -> Dict[str, list]:
    out = defaultdict(list)
    for r in rows:
        out[r.variant].append(r)
    return out


def plot_real_vs_estimated_bars(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """Столбчатая диаграмма: средние real / estimated_static / estimated_raw
    по каждому варианту (усреднение по всем прогонам)."""
    by_v = _group_by_variant(rows)
    variants = sorted(by_v.keys())

    real_avg = [sum(r.real for r in by_v[v]) / len(by_v[v]) for v in variants]
    static_avg = [sum(r.estimated_static for r in by_v[v]) / len(by_v[v]) for v in variants]
    raw_avg = [sum(r.estimated_raw for r in by_v[v]) / len(by_v[v]) for v in variants]

    x = range(len(variants))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(7, len(variants) * 2.0), 5))
    ax.bar([i - width for i in x], real_avg, width=width, label="real (факт)")
    ax.bar(list(x), static_avg, width=width, label="estimated_static (оценка Шага 3)")
    ax.bar([i + width for i in x], raw_avg, width=width, label="estimated_raw")
    ax.set_xticks(list(x))
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_ylabel("суммарная стоимость (W_d + W_b)")
    ax.set_title("Реальная стоимость vs эвристические оценки (статическая модель)")
    ax.legend()
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gap_vs_ncargos(
    rows: Sequence[ExperimentRow], out_path: str, gap_field: str = "gap_static"
) -> None:
    """Линейный график: средний gap (%) в зависимости от n_cargos, отдельная
    линия на каждый вариант."""
    by_v = _group_by_variant(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in sorted(by_v.keys()):
        by_n = defaultdict(list)
        for r in by_v[v]:
            by_n[r.n_cargos].append(r)
        ns = sorted(by_n.keys())
        gaps = [sum(getattr(r, gap_field) for r in by_n[n]) / len(by_n[n]) for n in ns]
        ax.plot(ns, gaps, marker="o", label=v)
    ax.set_xlabel("n_cargos")
    ax.set_ylabel(f"{gap_field}, %")
    ax.set_title(f"Разрыв оценка/факт ({gap_field}) в зависимости от числа грузов")
    ax.legend(fontsize=8)
    plt.grid(True)
    ax.axhline(0, color="grey", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_real_vs_ncargos(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """Линейный график: средняя ФАКТИЧЕСКАЯ стоимость доставки (real) в
    зависимости от n_cargos, отдельная линия на вариант."""
    by_v = _group_by_variant(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in sorted(by_v.keys()):
        by_n = defaultdict(list)
        for r in by_v[v]:
            by_n[r.n_cargos].append(r)
        ns = sorted(by_n.keys())
        vals = [sum(r.real for r in by_n[n]) / len(by_n[n]) for n in ns]
        ax.plot(ns, vals, marker="o", label=v)
    ax.set_xlabel("n_cargos")
    ax.set_ylabel("средняя фактическая стоимость (real)")
    ax.set_title("Фактическая стоимость доставки в зависимости от числа грузов")
    ax.legend(fontsize=8)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_win_rate(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """WIN-RATE: доля сценариев, где вариант дал минимальный real среди
    сравниваемых (только полностью доставленные сценарии, сравнение по
    одинаковым (n_cargos, seed))."""
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r.n_cargos, r.seed)].append(r)

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

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.6), 5))
    ax.bar(labels, rates)
    ax.set_ylabel("win-rate (доля сценариев с минимальным real)")
    ax.set_title("WIN-RATE по фактической стоимости")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pair_load_balance(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """Балансировка: сколько грузов достаётся самой загруженной паре
    (max_pair_load) в зависимости от n_cargos. Пунктиром -- идеально
    равномерное распределение n_cargos / n_pairs."""
    by_v = _group_by_variant(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ideal_ns, ideal_vals = [], []
    for v in sorted(by_v.keys()):
        by_n = defaultdict(list)
        for r in by_v[v]:
            by_n[r.n_cargos].append(r)
        ns = sorted(by_n.keys())
        vals = [sum(r.max_pair_load for r in by_n[n]) / len(by_n[n]) for n in ns]
        ax.plot(ns, vals, marker="o", label=v)
        if not ideal_ns:
            ideal_ns = ns
            ideal_vals = [
                n / by_n[n][0].n_pairs if by_n[n][0].n_pairs else 0.0 for n in ns
            ]
    if ideal_ns:
        ax.plot(ideal_ns, ideal_vals, linestyle="--", color="grey",
                label="равномерно (n_cargos / n_pairs)")
    ax.set_xlabel("n_cargos")
    ax.set_ylabel("грузов у самой загруженной пары")
    ax.set_title("Балансировка распределения грузов между парами")
    ax.legend(fontsize=8)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pair_cost_balance(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """Балансировка в ЕДИНИЦАХ СТОИМОСТИ: во сколько раз работа самой
    загруженной пары превышает идеально равномерную долю real / n_pairs
    (cost_imbalance). Это и есть величина, которую минимизирует LPT; 1.0 --
    идеальный баланс, поэтому чем ниже кривая, тем лучше."""
    by_v = _group_by_variant(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in sorted(by_v.keys()):
        by_n = defaultdict(list)
        for r in by_v[v]:
            if r.all_delivered:
                by_n[r.n_cargos].append(r)
        ns = sorted(by_n.keys())
        if not ns:
            continue
        vals = [sum(r.cost_imbalance for r in by_n[n]) / len(by_n[n]) for n in ns]
        ax.plot(ns, vals, marker="o", label=v)
    ax.axhline(1.0, linestyle="--", color="grey", label="идеальный баланс (1.0)")
    ax.set_xlabel("n_cargos")
    ax.set_ylabel("max_pair_cost / (real / n_pairs)")
    ax.set_title("Перекос загрузки пар по стоимости (makespan / идеал)")
    ax.legend(fontsize=8)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_round_trace(bracket: CostBracket, out_path: str, title: str = "") -> None:
    """Накопительный график real vs estimated_static vs estimated_raw по
    раундам одного прогона."""
    rounds = sorted(bracket.per_round_real.keys())
    cum_real, cum_static, cum_raw = [], [], []
    r_acc, s_acc, raw_acc = 0.0, 0.0, 0.0
    for rnd in rounds:
        r_acc += bracket.per_round_real.get(rnd, 0.0)
        s_acc += bracket.per_round_estimated_static.get(rnd, 0.0)
        raw_acc += bracket.per_round_estimated_raw.get(rnd, 0.0)
        cum_real.append(r_acc)
        cum_static.append(s_acc)
        cum_raw.append(raw_acc)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, cum_real, marker="o", label="real (накопительно)")
    ax.plot(rounds, cum_static, marker="s", label="estimated_static (накопительно)")
    ax.plot(rounds, cum_raw, marker="^", label="estimated_raw (накопительно)")
    ax.set_xlabel("раунд")
    ax.set_ylabel("накопленная стоимость")
    ax.set_title(title or "Реальная и оценочная стоимость по раундам")
    ax.legend()
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
