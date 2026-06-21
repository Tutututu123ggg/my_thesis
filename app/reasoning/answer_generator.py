from __future__ import annotations

from app.reasoning.llm_clients import ChatLLMClient
from app.reasoning.models import GeneratedAnswer, MedicalAnswerConfig
from app.reasoning.prompts import build_medical_answer_prompt
from app.retrieval import EvidenceSelectionConfig, RetrievalResult, RetrievedChunk, SentenceEvidenceSelector


def build_clean_chunk_context(chunks: list[RetrievedChunk], max_chunks: int) -> str:
    """Format only retrieved source chunks for the RAG prompt.

    Deliberately avoids keywords/entities/relations/debug metadata so the answer
    generator sees clean evidence instead of noisy retrieval internals. The raw
    chunk text is preferred over contextualized_text to keep evaluation fair.
    If sentence-level evidence selection has already compacted a chunk, that
    compact text is stored in ``chunk.text`` and is used here transparently.
    """
    sections: list[str] = []
    for idx, chunk in enumerate(chunks[:max_chunks], start=1):
        header_parts = [f"[Context {idx}]", f"chunk_id={chunk.chunk_id}"]
        if chunk.title:
            header_parts.append(f"title={chunk.title}")
        if chunk.section:
            header_parts.append(f"section={chunk.section}")
        if chunk.source_url:
            header_parts.append(f"source_url={chunk.source_url}")
        body = (chunk.text or chunk.contextualized_text or "").strip()
        if not body:
            continue
        sections.append(" | ".join(header_parts) + "\n" + body)
    return "\n\n".join(sections).strip()


class MedicalAnswerGenerator:
    """Generate final Vietnamese medical answers from retrieved chunks."""

    def __init__(
        self,
        llm_client: ChatLLMClient,
        config: MedicalAnswerConfig | None = None,
    ):
        self.llm_client = llm_client
        self.config = config or MedicalAnswerConfig()
        self.evidence_selector = SentenceEvidenceSelector(
            EvidenceSelectionConfig(
                enabled=self.config.enable_evidence_selection,
                max_sentences_per_chunk=self.config.evidence_max_sentences_per_chunk,
                sentence_window_size=self.config.evidence_sentence_window_size,
                max_compact_chars_per_chunk=self.config.evidence_max_compact_chars_per_chunk,
            )
        )

    def generate(self, question: str, retrieval_result: RetrievalResult) -> GeneratedAnswer:
        chunks_for_context, evidence_debug = self._prepare_context_chunks(question, retrieval_result)
        if self.config.context_mode == "chunks_only":
            context = build_clean_chunk_context(
                chunks_for_context,
                max_chunks=self.config.max_chunks,
            )
        else:
            compact_result = RetrievalResult(
                query=retrieval_result.query,
                method=retrieval_result.method,
                chunks=chunks_for_context,
                entities=retrieval_result.entities,
                relations=retrieval_result.relations,
                keywords=retrieval_result.keywords,
                debug=retrieval_result.debug,
            )
            context = compact_result.assemble_context(
                max_entities=self.config.max_entities,
                max_relations=self.config.max_relations,
                max_chunks=self.config.max_chunks,
            )
        prompt = build_medical_answer_prompt(question, context)
        answer = self.llm_client.generate(
            system_prompt=self.config.system_prompt,
            user_prompt=prompt,
        )
        return GeneratedAnswer(
            question=question,
            answer=answer.strip(),
            method=retrieval_result.method,
            context=context,
            context_chunks=[
                build_clean_chunk_context([chunk], max_chunks=1)
                for chunk in chunks_for_context[: self.config.max_chunks]
            ],
            metadata={
                "retrieval_debug": retrieval_result.debug,
                "evidence_selection_debug": evidence_debug,
            },
        )

    def _prepare_context_chunks(
        self,
        question: str,
        retrieval_result: RetrievalResult,
    ) -> tuple[list[RetrievedChunk], dict]:
        chunks = list(retrieval_result.chunks[: self.config.max_chunks] or [])
        if not self.config.enable_evidence_selection:
            return chunks, {"enabled": False, "reason": "disabled"}
        return self.evidence_selector.select_for_chunks(question, chunks)
