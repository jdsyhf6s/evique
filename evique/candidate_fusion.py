"""Rank-only fusion of retrieval channels without changing canonical evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite


CHANNELS = frozenset({'keyword', 'embedding', 'visual', 'event'})


@dataclass
class CandidateEvidence:
    """A channel result whose raw score is retained only for tracing."""

    id: str
    source: str
    score: float
    rank: int
    timestamp: float | None
    video_id: str
    evidence: dict

    def to_dict(self) -> dict:
        """Return an independent trace payload, including the full evidence."""
        return asdict(self)


def _has_consistent_video(candidate: CandidateEvidence) -> bool:
    if not isinstance(candidate.video_id, str) or not candidate.video_id.strip():
        return False
    provenance = candidate.evidence.get('provenance')
    provenance = provenance if isinstance(provenance, dict) else {}
    identities = [candidate.evidence.get('video_id'), provenance.get('video_id')]
    present = [value for value in identities if value is not None and value != '']
    return bool(present) and all(value == candidate.video_id for value in present)


def reciprocal_rank_fusion(
    channels: dict[str, list[CandidateEvidence]],
    *,
    k: int = 60,
    limit: int | None = None,
    video_ids: set[str] | None = None,
) -> list[dict]:
    """Fuse supplied 1-based ranks, deduplicating within each channel.

    Identity is (video_id, evidence.view, id). Missing or conflicting video
    identity is excluded before fusion and limiting. Filtering does not rerank:
    callers should rank each channel within their intended video scope.
    Invalid channel names, source mismatches, ranks, and scores raise ValueError.
    Raw scores never affect fusion or cross-channel representative selection.
    """
    if type(k) is not int or k <= 0:
        raise ValueError('k must be a positive integer')
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError('limit must be a nonnegative integer or None')
    if any(source not in CHANNELS for source in channels):
        raise ValueError('unknown retrieval channel')

    fused: dict[tuple[str, str, str], dict] = {}
    # Fixed traversal also makes representative payloads and float sums stable
    # when callers construct their channel dictionaries in different orders.
    for source in sorted(channels):
        unique: dict[tuple[str, str, str], CandidateEvidence] = {}
        for candidate in channels[source]:
            if candidate.source != source:
                raise ValueError('candidate source must match its channel')
            if type(candidate.rank) is not int or candidate.rank <= 0:
                raise ValueError('candidate rank must be a positive integer')
            if not isinstance(candidate.score, (int, float)) or not isfinite(candidate.score):
                raise ValueError('candidate score must be finite')
            if not _has_consistent_video(candidate):
                continue
            if video_ids is not None and candidate.video_id not in video_ids:
                continue
            identity = (candidate.video_id, str(candidate.evidence.get('view') or ''), candidate.id)
            previous = unique.get(identity)
            if previous is None or candidate.rank < previous.rank:
                unique[identity] = candidate

        for identity, candidate in unique.items():
            if identity not in fused:
                fused[identity] = {
                    'id': candidate.id,
                    'video_id': candidate.video_id,
                    'timestamp': candidate.timestamp,
                    'evidence': deepcopy(candidate.evidence),
                    'fusion_score': 0.0,
                    'original_ranks': {},
                    'original_scores': {},
                    'sources': [],
                }
            result = fused[identity]
            result['fusion_score'] += 1 / (k + candidate.rank)
            result['original_ranks'][source] = candidate.rank
            result['original_scores'][source] = candidate.score
            result['sources'].append(source)

    ordered = sorted(fused, key=lambda identity: (-fused[identity]['fusion_score'], identity))
    return [fused[identity] for identity in ordered[:limit]]
