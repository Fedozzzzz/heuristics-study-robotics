"""Общие сущности предметной области, не привязанные к конкретному шагу."""

from __future__ import annotations

from dataclasses import dataclass

from .graph import NodeId

CargoId = int


@dataclass
class Cargo:
    """Груз c_i: точка погрузки (start) и точка разгрузки (finish)."""

    cargo_id: CargoId
    start: NodeId
    finish: NodeId
