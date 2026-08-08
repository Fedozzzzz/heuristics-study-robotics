"""
Графики для сравнения эвристик приоритета груза (direct / inverse): реальная
стоимость исполнения vs эвристическая оценка. Используется matplotlib (без
специфичного стиля, чтобы не тянуть лишние зависимости).
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


def _group_by_heuristic(rows: Sequence[ExperimentRow]) -> Dict[str, list]:
    out = defaultdict(list)
    for r in rows:
        out[r.heuristic].append(r)
    return out


def plot_real_vs_estimated_bars(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """Столбчатая диаграмма: средние real / estimated_raw по каждой эвристике
    (усреднение по всем прогонам)."""
    by_h = _group_by_heuristic(rows)
    heuristics = sorted(by_h.keys())

    real_avg = [sum(r.real for r in by_h[h]) / len(by_h[h]) for h in heuristics]
    raw_avg = [sum(r.estimated_raw for r in by_h[h]) / len(by_h[h]) for h in heuristics]

    x = range(len(heuristics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(heuristics) * 1.8), 5))
    ax.bar([i - 0.5 * width for i in x], real_avg, width=width, label="real (факт)")
    ax.bar([i + 0.5 * width for i in x], raw_avg, width=width, label="estimated_raw")
    ax.set_xticks(list(x))
    ax.set_xticklabels(heuristics, rotation=30, ha="right")
    ax.set_ylabel("суммарная стоимость (W_d + W_b)")
    ax.set_title("Реальная стоимость vs эвристическая оценка, по эвристикам приоритета груза")
    ax.legend()
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gap_vs_ncargos(rows: Sequence[ExperimentRow], out_path: str, gap_field: str = "gap_raw") -> None:
    """Линейный график: средний gap (%) в зависимости от n_cargos, отдельная
    линия на каждую эвристику."""
    by_h = _group_by_heuristic(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for h in sorted(by_h.keys()):
        by_n = defaultdict(list)
        for r in by_h[h]:
            by_n[r.n_cargos].append(r)
        ns = sorted(by_n.keys())
        gaps = [sum(getattr(r, gap_field) for r in by_n[n]) / len(by_n[n]) for n in ns]
        ax.plot(ns, gaps, marker="o", label=h)
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
    зависимости от n_cargos, отдельная линия на эвристику (direct vs inverse)."""
    by_h = _group_by_heuristic(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for h in sorted(by_h.keys()):
        by_n = defaultdict(list)
        for r in by_h[h]:
            by_n[r.n_cargos].append(r)
        ns = sorted(by_n.keys())
        vals = [sum(r.real for r in by_n[n]) / len(by_n[n]) for n in ns]
        ax.plot(ns, vals, marker="o", label=h)
    ax.set_xlabel("n_cargos")
    ax.set_ylabel("средняя фактическая стоимость (real)")
    ax.set_title("Фактическая стоимость доставки в зависимости от числа грузов")
    ax.legend(fontsize=8)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_win_rate(rows: Sequence[ExperimentRow], out_path: str) -> None:
    """WIN-RATE: доля сценариев, где эвристика дала минимальный real среди
    сравниваемых эвристик (только полностью доставленные сценарии, сравнение
    по одинаковым (n_cargos, seed))."""
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r.n_cargos, r.seed)].append(r)

    wins: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    for key, group in by_key.items():
        delivered = [r for r in group if r.all_delivered]
        if not delivered:
            continue
        best = min(r.real for r in delivered)
        for r in group:
            total[r.heuristic] += 1
        for r in delivered:
            if abs(r.real - best) < 1e-6:
                wins[r.heuristic] += 1

    labels = sorted(total.keys())
    rates = [wins[g] / total[g] if total[g] else 0.0 for g in labels]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
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


def plot_round_trace(bracket: CostBracket, out_path: str, title: str = "") -> None:
    """Накопительный график real vs estimated_raw по раундам одного прогона
    (детализация одного сценария)."""
    rounds = sorted(bracket.per_round_real.keys())
    cum_real, cum_raw = [], []
    r_acc, raw_acc = 0.0, 0.0
    for rnd in rounds:
        r_acc += bracket.per_round_real.get(rnd, 0.0)
        raw_acc += bracket.per_round_estimated_raw.get(rnd, 0.0)
        cum_real.append(r_acc)
        cum_raw.append(raw_acc)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, cum_real, marker="o", label="real (накопительно)")
    ax.plot(rounds, cum_raw, marker="s", label="estimated_raw (накопительно)")
    ax.set_xlabel("раунд")
    ax.set_ylabel("накопленная стоимость")
    ax.set_title(title or "Реальная и оценочная стоимость по раундам")
    ax.legend()
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
