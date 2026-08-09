"""Immutable authenticated principal contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Principal:
    subject: str
    roles: frozenset[str]
    facilities: frozenset[str]

    def __init__(self, subject: str, roles: Iterable[str], facilities: Iterable[str]) -> None:
        if not subject.strip():
            raise ValueError("Principal subject is required.")
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "roles", frozenset(str(role) for role in roles))
        object.__setattr__(self, "facilities", frozenset(str(facility) for facility in facilities))

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def can_access_facility(self, facility: str) -> bool:
        return "*" in self.facilities or facility in self.facilities
