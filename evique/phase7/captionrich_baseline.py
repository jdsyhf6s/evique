from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

from .schema import TimeRange
from .scope_index import RankedScope, SharedPhase7Index


RRF_K = 60


def _normalized_hash(text: str) -> str:
    normalized = " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FusedScope:
    evidence_id: str
    scope_id: str
    video_id: str
    source_kind: str
    text: str
    fusion_score: float
    original_ranks: dict[str, int]
    original_scores: dict[str, float]
    time_range: TimeRange
    source_frame_ids: tuple[str, ...]
    provenance: dict[str, Any]
    normalized_text_hash: str

    @property
    def dedup_identity(self) -> tuple[str, str, str, str]:
        return self.video_id, self.scope_id, self.source_kind, self.normalized_text_hash


@dataclass(frozen=True)
class BaselinePackage:
    visible_items: tuple[FusedScope, ...]
    visible_ids: list[str]
    omitted_ids: list[str]
    serialized_items: list[str]
    visible_context: str
    truncated: bool
    context_chars_before_limit: int
    context_chars_visible: int
    estimated_tokens_before_limit: int
    estimated_tokens_visible: int
    evidence_items_before_limit: int
    evidence_items_visible: int
    trace: dict[str, Any]


def _serialize(item: FusedScope) -> str:
    return json.dumps({
        "evidence_id": item.evidence_id,
        "video_id": item.video_id,
        "time_range": {"start": item.time_range.start, "end": item.time_range.end},
        "source_kind": item.source_kind,
        "source_frame_ids": list(item.source_frame_ids),
        "text": item.text,
        "provenance": item.provenance,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _candidate_trace(item: RankedScope) -> dict[str, Any]:
    return {
        "evidence_id": item.evidence_id,
        "scope_id": item.scope_id,
        "video_id": item.video_id,
        "source_kind": item.source_kind,
        "text": item.text,
        "score": item.score,
        "rank": item.rank,
        "time_range": {"start": item.time_range.start, "end": item.time_range.end},
        "source_frame_ids": list(item.source_frame_ids),
        "provenance": item.provenance,
    }


class CaptionRichHybrid:
    def __init__(self, index: SharedPhase7Index, *, rrf_k: int = RRF_K, channel_limit: int = 100):
        if rrf_k <= 0 or channel_limit <= 0:
            raise ValueError("rrf_k and channel_limit must be positive")
        self.index = index
        self.rrf_k = rrf_k
        self.channel_limit = channel_limit

    def fuse(self, keyword: list[RankedScope], dense: list[RankedScope]) -> list[FusedScope]:
        fused: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        order = 0
        for channel, candidates in (("keyword", keyword), ("dense", dense)):
            seen = set()
            for candidate in candidates:
                identity = (
                    candidate.video_id,
                    candidate.scope_id,
                    candidate.source_kind,
                    _normalized_hash(candidate.text),
                )
                if identity in seen:
                    continue
                seen.add(identity)
                if identity not in fused:
                    fused[identity] = {
                        "candidate": candidate,
                        "fusion_score": 0.0,
                        "ranks": {},
                        "scores": {},
                        "order": order,
                    }
                    order += 1
                entry = fused[identity]
                entry["fusion_score"] += 1 / (self.rrf_k + candidate.rank)
                entry["ranks"][channel] = candidate.rank
                entry["scores"][channel] = candidate.score
        ordered = sorted(fused.values(), key=lambda entry: (-entry["fusion_score"], entry["order"]))
        result = []
        for entry in ordered:
            item = entry["candidate"]
            result.append(FusedScope(
                evidence_id=item.evidence_id,
                scope_id=item.scope_id,
                video_id=item.video_id,
                source_kind=item.source_kind,
                text=item.text,
                fusion_score=entry["fusion_score"],
                original_ranks=dict(entry["ranks"]),
                original_scores=dict(entry["scores"]),
                time_range=item.time_range,
                source_frame_ids=item.source_frame_ids,
                provenance=dict(item.provenance),
                normalized_text_hash=_normalized_hash(item.text),
            ))
        return result

    def retrieve(self, query: str, video_id: str, *, context_limit_chars: int) -> BaselinePackage:
        if context_limit_chars < 0:
            raise ValueError("context_limit_chars must be nonnegative")
        keyword = self.index.search_scope_keyword(query, video_id, self.channel_limit)
        dense = self.index.search_scope_dense(query, video_id, self.channel_limit)
        fused = self.fuse(keyword, dense)
        serialized = [_serialize(item) for item in fused]
        visible: list[FusedScope] = []
        visible_serialized: list[str] = []
        stop = len(fused)
        used = 0
        for position, (item, payload) in enumerate(zip(fused, serialized)):
            added = len(payload) + (1 if visible_serialized else 0)
            if used + added > context_limit_chars:
                stop = position
                break
            visible.append(item)
            visible_serialized.append(payload)
            used += added
        omitted = fused[stop:]
        before_chars = sum(len(item) for item in serialized) + max(0, len(serialized) - 1)
        trace = {
            "query": query,
            "video_id": video_id,
            "keyword": [_candidate_trace(candidate) for candidate in keyword],
            "dense": [_candidate_trace(candidate) for candidate in dense],
            "fusion": [{
                "evidence_id": item.evidence_id,
                "fusion_score": item.fusion_score,
                "original_ranks": item.original_ranks,
                "original_scores": item.original_scores,
            } for item in fused],
            "context_limit_chars": context_limit_chars,
            "context_chars_before_limit": before_chars,
            "context_chars_visible": used,
            "evidence_items_before_limit": len(fused),
            "evidence_items_visible": len(visible),
            "truncated": bool(omitted),
            "visible_ids": [item.evidence_id for item in visible],
            "omitted_ids": [item.evidence_id for item in omitted],
        }
        return BaselinePackage(
            visible_items=tuple(visible),
            visible_ids=[item.evidence_id for item in visible],
            omitted_ids=[item.evidence_id for item in omitted],
            serialized_items=visible_serialized,
            visible_context="\n".join(visible_serialized),
            truncated=bool(omitted),
            context_chars_before_limit=before_chars,
            context_chars_visible=used,
            estimated_tokens_before_limit=math.ceil(before_chars / 4),
            estimated_tokens_visible=math.ceil(used / 4),
            evidence_items_before_limit=len(fused),
            evidence_items_visible=len(visible),
            trace=trace,
        )
