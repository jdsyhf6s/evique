"""Evidence sufficiency states shared by validation, packing, and prompts."""
from __future__ import annotations

from enum import Enum


class EvidenceStatus(str, Enum):
    COMPLETE = 'COMPLETE'
    PARTIAL = 'PARTIAL'
    INSUFFICIENT = 'INSUFFICIENT'
    UNAVAILABLE = 'UNAVAILABLE'

    def __str__(self) -> str:
        return self.value


COMPLETE = EvidenceStatus.COMPLETE
PARTIAL = EvidenceStatus.PARTIAL
INSUFFICIENT = EvidenceStatus.INSUFFICIENT
UNAVAILABLE = EvidenceStatus.UNAVAILABLE


__all__ = [
    'EvidenceStatus', 'COMPLETE', 'PARTIAL', 'INSUFFICIENT', 'UNAVAILABLE',
]
