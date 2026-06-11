from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RetrievalMethod = Literal["vector", "vector_naive", "lightrag", "hipporag", "mixed"]


@dataclass
class EvaluationQuestion:
    question: str
    reference: str
    difficulty: str = "unknown"
    expected_terms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    methods: list[RetrievalMethod] = field(
        default_factory=lambda: ["vector_naive", "lightrag", "hipporag", "mixed"]
    )
    async_mixed: bool = False
    max_questions: int = 0
    max_context_chunks: int = 5
    chunk_top_k: int = 10
    entity_top_k: int = 20
    relation_top_k: int = 20
    hippo_seed_top_k: int = 12
    hippo_ppr_entities: int = 40
    hippo_hops: int = 2
    max_chunks: int = 12
    max_concurrent_methods: int = 4
    ragas_generated_questions: int = 3
    output_dir: str = "data/evaluation/ragas_style_small"


@dataclass
class RetrievalMetricResult:
    """Simple retrieval-only checks used alongside RAGAS-style metrics.

    These are not RAGAS metrics. They help catch cases where the generator or
    judge knows the answer even though retrieval missed the expected disease or
    evidence terms.
    """

    expected_terms: list[str] = field(default_factory=list)
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    mrr: float = 0.0
    first_hit_rank: int | None = None
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_terms": self.expected_terms,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "mrr": self.mrr,
            "first_hit_rank": self.first_hit_rank,
            "matched_terms": self.matched_terms,
        }


@dataclass
class RagasStyleResult:
    """RAGAS-style reference-free scores.

    Main metrics follow the RAGAS paper structure:
    - faithfulness: answer -> statements -> context-supported statements ratio
    - answer_relevance: answer -> generated questions -> average embedding cosine
    - context_relevance: context -> relevant sentence extraction ratio

    medical_safety and answer_correctness are domain/reference auxiliaries and
    are not included in ragas_style_overall.
    """

    faithfulness: float
    answer_relevance: float
    context_relevance: float
    ragas_style_overall: float

    medical_safety: float = 0.0
    answer_correctness: float | None = None
    verdict: str = "fail"
    rationale: str = ""

    statements: list[str] = field(default_factory=list)
    supported_statements: list[str] = field(default_factory=list)
    unsupported_statements: list[str] = field(default_factory=list)
    generated_questions: list[str] = field(default_factory=list)
    relevant_sentences: list[str] = field(default_factory=list)
    total_context_sentences: int = 0
    raw_outputs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_relevance": self.context_relevance,
            "ragas_style_overall": self.ragas_style_overall,
            "medical_safety": self.medical_safety,
            "answer_correctness": self.answer_correctness,
            "verdict": self.verdict,
            "rationale": self.rationale,
            "statements": self.statements,
            "supported_statements": self.supported_statements,
            "unsupported_statements": self.unsupported_statements,
            "generated_questions": self.generated_questions,
            "relevant_sentences": self.relevant_sentences,
            "total_context_sentences": self.total_context_sentences,
            "raw_outputs": self.raw_outputs,
        }


# Backward-compatible aliases for old imports. New code should use
# RagasStyleResult and RagasStyleEvaluator.
JudgeResult = RagasStyleResult


@dataclass
class EvaluationRow:
    difficulty: str
    method: str
    question: str
    reference: str
    answer: str
    contexts: list[str]
    latency_ms: float
    retrieval: dict[str, Any]
    evaluation: RagasStyleResult | None = None
    retrieval_metrics: RetrievalMetricResult | None = None

    @property
    def judge(self) -> RagasStyleResult | None:
        """Backward-compatible access for older scripts."""
        return self.evaluation

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "difficulty": self.difficulty,
            "method": self.method,
            "question": self.question,
            "reference": self.reference,
            "answer": self.answer,
            "contexts": self.contexts,
            "latency_ms": self.latency_ms,
            "retrieval": self.retrieval,
            "ragas_style": self.evaluation.to_dict() if self.evaluation else None,
            "retrieval_metrics": self.retrieval_metrics.to_dict() if self.retrieval_metrics else None,
            # Backward-compatible key; contains the same payload as ragas_style.
            "judge": self.evaluation.to_dict() if self.evaluation else None,
        }
