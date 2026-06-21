from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RetrievalMethod = Literal["vector", "vector_naive", "vector_ce", "lightrag", "hipporag", "mixed", "mixed_ce"]


@dataclass
class EvaluationQuestion:
    question: str
    reference: str
    difficulty: str = "unknown"
    expected_terms: list[str] = field(default_factory=list)
    expected_answer_points: list[str] = field(default_factory=list)
    source_links: list[str] = field(default_factory=list)
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
    answer_relevance_embedding_provider: str = "openai"
    answer_relevance_embedding_model: str = "text-embedding-3-large"

@dataclass
class RetrievalMetricResult:
    """Retrieval-only checks used alongside RAGAS-style metrics.

    These are not RAGAS metrics. They are descriptive retrieval metrics, without
    pass/fail thresholds:
    - term Hit@K/MRR checks expected disease/entity terms in ranked contexts.
    - source Hit@K/MRR checks whether retrieved chunks come from expected source URLs.
    """

    expected_terms: list[str] = field(default_factory=list)
    expected_source_links: list[str] = field(default_factory=list)

    term_hit_at_1: float = 0.0
    term_hit_at_3: float = 0.0
    term_hit_at_5: float = 0.0
    term_mrr: float = 0.0
    term_first_hit_rank: int | None = None
    matched_terms: list[str] = field(default_factory=list)

    source_hit_at_1: float = 0.0
    source_hit_at_3: float = 0.0
    source_hit_at_5: float = 0.0
    source_mrr: float = 0.0
    source_first_hit_rank: int | None = None
    matched_source_links: list[str] = field(default_factory=list)

    # Backward-compatible aliases used by earlier summary code.
    @property
    def hit_at_1(self) -> float:
        return self.term_hit_at_1

    @property
    def hit_at_3(self) -> float:
        return self.term_hit_at_3

    @property
    def hit_at_5(self) -> float:
        return self.term_hit_at_5

    @property
    def mrr(self) -> float:
        return self.term_mrr

    @property
    def first_hit_rank(self) -> int | None:
        return self.term_first_hit_rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_terms": self.expected_terms,
            "expected_source_links": self.expected_source_links,
            "term_hit_at_1": self.term_hit_at_1,
            "term_hit_at_3": self.term_hit_at_3,
            "term_hit_at_5": self.term_hit_at_5,
            "term_mrr": self.term_mrr,
            "term_first_hit_rank": self.term_first_hit_rank,
            "matched_terms": self.matched_terms,
            "source_hit_at_1": self.source_hit_at_1,
            "source_hit_at_3": self.source_hit_at_3,
            "source_hit_at_5": self.source_hit_at_5,
            "source_mrr": self.source_mrr,
            "source_first_hit_rank": self.source_first_hit_rank,
            "matched_source_links": self.matched_source_links,
        }


@dataclass
class RagasStyleResult:
    """RAGAS-style scores plus answer quality metrics.

    Main RAGAS-style metrics follow the RAGAS paper structure:
    - faithfulness: answer -> statements -> context-supported statements ratio
    - answer_relevance: answer -> generated questions -> average embedding cosine
    - context_relevance: context -> relevant sentence extraction ratio

    Domain/reference auxiliaries are reported separately and are not used to set
    pass/fail thresholds.
    """

    faithfulness: float
    answer_relevance: float
    context_relevance: float
    ragas_style_overall: float

    medical_safety: float = 0.0
    answer_point_coverage: float | None = None
    answer_correctness: float | None = None
    answer_correctness_rationale: str = ""
    rationale: str = ""

    statements: list[str] = field(default_factory=list)
    supported_statements: list[str] = field(default_factory=list)
    unsupported_statements: list[str] = field(default_factory=list)
    generated_questions: list[str] = field(default_factory=list)
    relevant_sentences: list[str] = field(default_factory=list)
    total_context_sentences: int = 0

    expected_answer_points: list[str] = field(default_factory=list)
    covered_answer_points: list[str] = field(default_factory=list)
    missing_answer_points: list[str] = field(default_factory=list)

    raw_outputs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_relevance": self.context_relevance,
            "ragas_style_overall": self.ragas_style_overall,
            "medical_safety": self.medical_safety,
            "answer_point_coverage": self.answer_point_coverage,
            "answer_correctness": self.answer_correctness,
            "answer_correctness_rationale": self.answer_correctness_rationale,
            "rationale": self.rationale,
            "statements": self.statements,
            "supported_statements": self.supported_statements,
            "unsupported_statements": self.unsupported_statements,
            "generated_questions": self.generated_questions,
            "relevant_sentences": self.relevant_sentences,
            "total_context_sentences": self.total_context_sentences,
            "expected_answer_points": self.expected_answer_points,
            "covered_answer_points": self.covered_answer_points,
            "missing_answer_points": self.missing_answer_points,
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