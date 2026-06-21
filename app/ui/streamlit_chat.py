from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# Allow running with: streamlit run app/ui/streamlit_chat.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.retriever_factory import RetrieverBuildConfig, build_retriever_bundle
from app.reasoning import ChatLLMClient, MedicalAnswerConfig, MedicalAnswerGenerator
from app.retrieval import RetrievalResult, RetrievedChunk, RetrievedEntity, RetrievedRelation


DEFAULT_QUESTIONS = [
    "Vì sao cùng là ngứa và bong tróc, viêm da tiết bã nên được truy hồi khác với viêm da cơ địa?",
    "Tôi bị ngứa, bong da, thỉnh thoảng nặng hơn khi gặp bụi hoặc dùng xà phòng, nên tìm hiểu nhóm bệnh nào?",
    "Yếu tố môi trường nào nối các bệnh dị ứng ở da, mũi và hô hấp?",
]


st.set_page_config(
    page_title="Vietnamese Medical Hybrid RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
.block-container { max-width: 1080px; padding-top: 1.4rem; }
.chat-title { font-size: 1.55rem; font-weight: 700; margin-bottom: 0.1rem; }
.chat-subtitle { color: #666; margin-bottom: 1.2rem; }
.small-muted { color: #777; font-size: 0.86rem; }
.evidence-card {
    border: 1px solid rgba(49, 51, 63, 0.16);
    border-radius: 14px;
    padding: 0.85rem 0.95rem;
    margin: 0.55rem 0;
    background: rgba(250, 250, 250, 0.72);
}
.evidence-title { font-weight: 700; margin-bottom: 0.18rem; }
.evidence-meta { color: #666; font-size: 0.86rem; margin-bottom: 0.4rem; }
.preview { white-space: pre-wrap; line-height: 1.45; }
.code-lite {
    background: rgba(49, 51, 63, 0.06);
    border-radius: 8px;
    padding: 0.15rem 0.35rem;
    font-size: 0.85rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_retriever_bundle(
    no_llm_keywords: bool,
    chunk_top_k: int,
    entity_top_k: int,
    relation_top_k: int,
    hippo_seed_top_k: int,
    hippo_ppr_entities: int,
    hippo_hops: int,
    max_chunks: int,
):
    config = RetrieverBuildConfig(
        no_llm_keywords=no_llm_keywords,
        chunk_top_k=chunk_top_k,
        entity_top_k=entity_top_k,
        relation_top_k=relation_top_k,
        hippo_seed_top_k=hippo_seed_top_k,
        hippo_ppr_entities=hippo_ppr_entities,
        hippo_hops=hippo_hops,
        max_chunks=max_chunks,
    )
    return build_retriever_bundle(config=config)


@st.cache_resource(show_spinner=False)
def get_answer_generator(
    provider: str,
    model: str,
    context_mode: str,
    max_contexts: int,
):
    llm = ChatLLMClient.from_env(
        prefix="REASONING",
        default_provider=provider,
        default_model=model,
    )
    # Sidebar values should override env-derived defaults for the demo session.
    llm.config.provider = provider  # type: ignore[assignment]
    llm.config.model = model
    return MedicalAnswerGenerator(
        llm_client=llm,
        config=MedicalAnswerConfig(
            max_chunks=max_contexts,
            context_mode=context_mode,  # type: ignore[arg-type]
        ),
    )


def short_text(text: str, max_chars: int = 700) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(value)


def render_keywords(result: RetrievalResult) -> None:
    if not result.keywords:
        return
    low = result.keywords.low_level_keywords or []
    high = result.keywords.high_level_keywords or []
    if not low and not high:
        return
    with st.expander("🔑 Query keywords", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Low-level**")
            st.write(", ".join(low) if low else "—")
        with c2:
            st.markdown("**High-level**")
            st.write(", ".join(high) if high else "—")


def render_chunk_card(chunk: RetrievedChunk, rank: int) -> None:
    title = chunk.title or "Không rõ tiêu đề"
    url = chunk.source_url or chunk.metadata.get("source_url") or ""
    section = " / ".join(x for x in [chunk.section, chunk.subsection] if x) or "Không rõ section"
    text = chunk.text or chunk.contextualized_text or ""
    mixed_sources = chunk.metadata.get("mixed_sources")
    extra_badges = []
    if chunk.metadata.get("evidence_expansion"):
        extra_badges.append("expanded")
    if mixed_sources:
        extra_badges.append("mixed: " + ", ".join(map(str, mixed_sources)))
    badge_text = f" · {' · '.join(extra_badges)}" if extra_badges else ""

    st.markdown(
        f"""
<div class=\"evidence-card\">
  <div class=\"evidence-title\">#{rank} · {title}</div>
  <div class=\"evidence-meta\">
    score=<span class=\"code-lite\">{chunk.score:.4f}</span> · source=<span class=\"code-lite\">{chunk.source}</span>{badge_text}<br/>
    section: {section}<br/>
    chunk_id: <span class=\"code-lite\">{chunk.chunk_id}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    if url:
        st.markdown(f"Nguồn: {url}")
    st.markdown(f"<div class='preview'>{short_text(text)}</div>", unsafe_allow_html=True)


def render_entities(entities: list[RetrievedEntity], limit: int = 12) -> None:
    if not entities:
        st.info("Không có entity được trả về.")
        return
    rows = []
    for e in entities[:limit]:
        rows.append(
            {
                "score": round(float(e.score or 0.0), 4),
                "name": e.name,
                "type": e.entity_type,
                "source": e.source,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_relations(relations: list[RetrievedRelation], limit: int = 12) -> None:
    if not relations:
        st.info("Không có relation được trả về.")
        return
    rows = []
    for r in relations[:limit]:
        rows.append(
            {
                "score": round(float(r.score or 0.0), 4),
                "type": r.relation_type,
                "subject": r.subject_name,
                "object": r.object_name,
                "section": r.section,
                "source": r.source,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def compact_debug(debug: dict[str, Any]) -> dict[str, Any]:
    """Keep the debug useful but not wall-of-log long."""
    if not debug:
        return {}

    keep_keys = [
        "collection",
        "vector_text_mode",
        "top_k",
        "seed_query",
        "intent",
        "intent_confidence",
        "intent_cues",
        "intent_relation_boosts",
        "seed_count",
        "adjacency_edge_count",
        "ppr_entity_count",
        "vector_chunk_count",
        "lightrag_chunk_count",
        "hipporag_chunk_count",
        "fusion",
        "evidence_expansion",
        "branch_debug",
    ]
    result: dict[str, Any] = {}
    for key in keep_keys:
        if key not in debug:
            continue
        value = debug[key]
        if key == "branch_debug" and isinstance(value, dict):
            result[key] = {name: compact_debug(branch) for name, branch in value.items()}
        else:
            result[key] = value
    return result


def render_evidence(result: RetrievalResult, show_debug: bool, max_contexts: int) -> None:
    with st.expander("📚 Evidence retrieval", expanded=False):
        render_keywords(result)

        st.markdown("### Top chunks")
        if not result.chunks:
            st.warning("Retriever không trả về chunk nào.")
        for idx, chunk in enumerate(result.chunks[:max_contexts], start=1):
            render_chunk_card(chunk, idx)

        tab_entities, tab_relations = st.tabs(["Entities", "Relations"])
        with tab_entities:
            render_entities(result.entities)
        with tab_relations:
            render_relations(result.relations)

        if show_debug:
            st.markdown("### Debug gọn")
            st.code(safe_json(compact_debug(result.debug)), language="json")


def retrieve(method: str, query: str, retrievers: dict[str, Any], chunk_top_k: int) -> RetrievalResult:
    retriever = retrievers[method]
    if method in {"vector", "vector_naive"}:
        return retriever.retrieve(query, top_k=chunk_top_k)
    return retriever.retrieve(query)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_answer", None)


def add_sample_question(question: str) -> None:
    st.session_state["pending_question"] = question


def sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("⚙️ Retrieval config")
        method = st.selectbox(
            "Method",
            options=["vector_naive", "lightrag", "hipporag", "mixed"],
            index=3,
            help="vector_naive vẫn dùng collection medical_chunks_naive và raw chunk text.",
        )
        no_llm_keywords = st.checkbox(
            "No LLM keywords",
            value=False,
            help="Bật để dùng keyword extractor fallback/rule-based nếu không muốn gọi LLM cho LightRAG/HippoRAG.",
        )
        chunk_top_k = st.slider("chunk_top_k", 1, 20, 5)
        entity_top_k = st.slider("entity_top_k", 1, 50, 5)
        relation_top_k = st.slider("relation_top_k", 1, 50, 5)
        hippo_seed_top_k = st.slider("hippo_seed_top_k", 1, 30, 5)
        hippo_ppr_entities = st.slider("hippo_ppr_entities", 10, 120, 60, step=5)
        hippo_hops = st.slider("hippo_hops", 1, 5, 3)
        max_chunks = st.slider("max_chunks", 1, 20, 5)
        max_contexts = st.slider("max_contexts for answer/evidence", 1, 12, 5)

        st.divider()
        st.header("🧠 Answer LLM")
        generate_answer = st.checkbox("Generate answer", value=True)
        reasoning_provider = st.selectbox("Provider", ["openai", "ollama"], index=0)
        default_model = "gpt-4o-mini" if reasoning_provider == "openai" else "qwen2.5:3b"
        reasoning_model = st.text_input("Model", value=os.getenv("REASONING_MODEL", default_model))
        context_mode = st.selectbox("Context mode", ["chunks_only", "full_graph"], index=0)

        st.divider()
        show_debug = st.checkbox("Show compact debug", value=False)
        if st.button("Clear chat"):
            st.session_state["messages"] = []
            st.session_state["last_result"] = None
            st.session_state["last_answer"] = None
            st.rerun()
        if st.button("Clear cached retrievers"):
            bundle = st.session_state.get("bundle")
            if bundle is not None:
                close = getattr(bundle, "close", None)
                if close:
                    close()
            st.cache_resource.clear()
            st.session_state.pop("bundle", None)
            st.success("Đã clear cache. Lần hỏi tiếp theo sẽ build lại retriever bundle.")

    return {
        "method": method,
        "no_llm_keywords": no_llm_keywords,
        "chunk_top_k": chunk_top_k,
        "entity_top_k": entity_top_k,
        "relation_top_k": relation_top_k,
        "hippo_seed_top_k": hippo_seed_top_k,
        "hippo_ppr_entities": hippo_ppr_entities,
        "hippo_hops": hippo_hops,
        "max_chunks": max_chunks,
        "max_contexts": max_contexts,
        "generate_answer": generate_answer,
        "reasoning_provider": reasoning_provider,
        "reasoning_model": reasoning_model,
        "context_mode": context_mode,
        "show_debug": show_debug,
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    init_state()
    cfg = sidebar()

    st.markdown("<div class='chat-title'>Vietnamese Medical Hybrid RAG</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='chat-subtitle'>Chat demo đọc Neo4j/Qdrant, trả lời bằng context truy xuất và giấu evidence trong dropdown.</div>",
        unsafe_allow_html=True,
    )

    st.caption("Câu test nhanh")
    cols = st.columns(len(DEFAULT_QUESTIONS))
    for col, question in zip(cols, DEFAULT_QUESTIONS):
        with col:
            if st.button(short_text(question, 90), key=f"sample-{question}"):
                add_sample_question(question)

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("result") is not None:
                render_evidence(message["result"], cfg["show_debug"], cfg["max_contexts"])

    pending = st.session_state.pop("pending_question", None)
    user_query = pending or st.chat_input("Nhập câu hỏi y khoa tiếng Việt...")
    if not user_query:
        return

    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    try:
        with st.spinner("Đang build retriever bundle..." if "bundle" not in st.session_state else "Đang retrieval..."):
            bundle = get_retriever_bundle(
                no_llm_keywords=cfg["no_llm_keywords"],
                chunk_top_k=cfg["chunk_top_k"],
                entity_top_k=cfg["entity_top_k"],
                relation_top_k=cfg["relation_top_k"],
                hippo_seed_top_k=cfg["hippo_seed_top_k"],
                hippo_ppr_entities=cfg["hippo_ppr_entities"],
                hippo_hops=cfg["hippo_hops"],
                max_chunks=cfg["max_chunks"],
            )
            st.session_state["bundle"] = bundle
            result = retrieve(
                method=cfg["method"],
                query=user_query,
                retrievers=bundle.retrievers,
                chunk_top_k=cfg["chunk_top_k"],
            )

        if cfg["generate_answer"]:
            with st.spinner("Đang sinh câu trả lời từ retrieved chunks..."):
                answer_generator = get_answer_generator(
                    provider=cfg["reasoning_provider"],
                    model=cfg["reasoning_model"],
                    context_mode=cfg["context_mode"],
                    max_contexts=cfg["max_contexts"],
                )
                generated = answer_generator.generate(user_query, result)
                answer_text = generated.answer or "Không sinh được câu trả lời."
        else:
            answer_text = (
                f"Đã retrieval bằng `{cfg['method']}`. "
                f"Tìm thấy {len(result.chunks)} chunks, {len(result.entities)} entities, {len(result.relations)} relations."
            )

        with st.chat_message("assistant"):
            st.markdown(answer_text)
            render_evidence(result, cfg["show_debug"], cfg["max_contexts"])

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer_text, "result": result}
        )
        st.session_state["last_result"] = result
        st.session_state["last_answer"] = answer_text

    except Exception as exc:
        error_text = (
            "Demo bị lỗi khi build/retrieve/generate. Kiểm tra Neo4j, Qdrant, `.env`, "
            "collection `medical_chunks_naive`, và API key/model nếu bật Generate answer.\n\n"
            f"**{type(exc).__name__}:** {exc}"
        )
        with st.chat_message("assistant"):
            st.error(error_text)
            with st.expander("Traceback", expanded=False):
                st.code(traceback.format_exc())
        st.session_state["messages"].append({"role": "assistant", "content": error_text})


if __name__ == "__main__":
    main()
