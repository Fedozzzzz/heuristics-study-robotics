"""
Шаг 0. Проверка достижимости решения (перенесено из dynamic_v2 без изменений:
формулировка шага у обеих моделей идентична).

Строится G' = (V, E_ready ∪ E_blocked) (то есть весь граф G, кроме рёбер
IMPOSSIBLE) и проверяется, что все точки погрузки/разгрузки грузов лежат в
одной компоненте связности G' -- условия 1 и 2 из постановки эквивалентны
единой проверке на G', так как компонента E_ready всегда вложена в
компоненту E_ready ∪ E_blocked (если условие 1 выполняется без построек --
условие 2 на объединённом графе выполняется автоматически). Если такой
компоненты нет -- задача невыполнима.

Если общая компонента V* найдена, дополнительно проверяется, что в ней есть
хотя бы один доставщик и хотя бы один строитель (иначе даже потенциально
достижимые грузы некому доставлять/для кого строить).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Set

from .graph import EdgeKind, Graph, NodeId

_READY_AND_BLOCKED = {EdgeKind.FREE, EdgeKind.BLOCKED}


@dataclass
class FeasibilityResult:
    ok: bool
    reason: str
    reachable_nodes: Optional[Set[NodeId]] = None


def check_feasibility(
    G: Graph,
    cargos: Sequence,
    deliverer_positions: Dict[int, NodeId],
    builder_positions: Dict[int, NodeId],
) -> FeasibilityResult:
    """Algorithm-Шаг-0(G, C, R_d, R_b)."""

    required_nodes: Set[NodeId] = set()
    for cargo in cargos:
        required_nodes.add(cargo.start)
        required_nodes.add(cargo.finish)

    if not required_nodes:
        return FeasibilityResult(ok=True, reason="Нет грузов -- задача тривиально достижима.")

    components = G.connected_components(_READY_AND_BLOCKED)
    target_component: Optional[Set[NodeId]] = None
    for comp in components:
        if required_nodes <= comp:
            target_component = comp
            break

    if target_component is None:
        return FeasibilityResult(
            ok=False,
            reason=(
                "Точки погрузки/разгрузки грузов не лежат в одной компоненте "
                "связности даже с учётом всех переправ, доступных к возведению "
                "(E_ready ∪ E_blocked). Задача невыполнима."
            ),
        )

    has_deliverer = any(pos in target_component for pos in deliverer_positions.values())
    has_builder = any(pos in target_component for pos in builder_positions.values())

    if not has_deliverer or not has_builder:
        missing = []
        if not has_deliverer:
            missing.append("доставщика")
        if not has_builder:
            missing.append("строителя")
        return FeasibilityResult(
            ok=False,
            reason=(
                f"В достижимой компоненте V* нет ни одного робота-"
                f"{' и '.join(missing)}. Задача невыполнима."
            ),
            reachable_nodes=target_component,
        )

    return FeasibilityResult(ok=True, reason="Достижимо.", reachable_nodes=target_component)
