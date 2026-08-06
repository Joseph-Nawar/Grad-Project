"""Replaceable persistence protocol."""

from __future__ import annotations

from typing import Protocol, Sequence

from rural_stroke_assist.cases.contracts import Case, CaseStatus


class CaseRepository(Protocol):
    def initialize(self) -> None: ...
    def save(self, case: Case) -> None: ...
    def get(self, case_id: str) -> Case: ...
    def list(self, *, statuses: Sequence[CaseStatus] | None = None) -> list[Case]: ...
