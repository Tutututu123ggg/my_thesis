from __future__ import annotations

import csv
import json
import os
import queue
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.graph_database import GraphRepository, Neo4jClient
from app.infrastructure.vector_database import QdrantClientWrapper, VectorRepository
from app.infrastructure.vector_database.vector_collections import (
    CHUNK_COLLECTION,
    ENTITY_COLLECTION,
    NAIVE_CHUNK_COLLECTION,
    RELATION_COLLECTION,
)
from app.ingestion.builders.graph_builder import GraphBuilder
from app.ingestion.builders.medical_graph_builder import MedicalGraphBuilder
from app.ingestion.builders.vector_graph_builder import VectorGraphBuilder
from app.ingestion.extraction.entity.medical_extractor import MedicalExtractor
from app.ingestion.extraction.keyword.keyword_profiler import ArticleKeywordProfiler
try:
    from app.ingestion.extraction.pipeline.article_extraction_pipeline import (
        ArticleExtractionPipeline,
        ArticleExtractionResult,
    )
except ModuleNotFoundError:  # repo variant using plural `pipelines`
    from app.ingestion.extraction.pipelines.article_extraction_pipeline import (
        ArticleExtractionPipeline,
        ArticleExtractionResult,
    )
from app.ingestion.extraction.relation.extraction_validator import ExtractionValidator
from app.ingestion.processing import ChunkDocument, chunk_article, load_markdown_article
from app.ingestion.resolution import EntityResolver


CONFIRM_CLEAR_TOKEN = "medical-vietnamese-hybrid-rag-457"


@dataclass(frozen=True)
class ArticleSpec:
    index: int
    md_path: Path
    row: dict[str, str] = field(default_factory=dict)


@dataclass
class FullCorpusIngestConfig:
    input_csv: Path
    md_root: Path = Path(".")
    out_dir: Path = Path("data/ingest_runs/ingest_457_v1")
    run_id: str = "ingest_457_v1"

    start_index: int = 0
    limit: int = 0
    resume: bool = True
    dry_run: bool = False

    max_tokens: int = 900
    fact_window_max_tokens: int = 450
    fact_window_min_tokens: int = 80
    fact_window_overlap_sentences: int = 1

    entity_max_tokens: int = 2500
    relation_max_tokens: int = 4000
    keyword_llm: bool = False
    keyword_model: str | None = "gpt-4o-mini"

    embedding_batch_size: int = 16
    llm_workers: int = 3
    llm_retries: int = 3

    checkpoint_jsonl: Path | None = None

    recreate: bool = False
    confirm_clear: str = ""


@dataclass
class ArticleExtractedJob:
    spec: ArticleSpec
    article: Any | None = None
    result: ArticleExtractionResult | None = None
    artifact_path: Path | None = None
    error: str = ""


class IngestCheckpoint:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completed: set[str] = set()
        self.latest_by_key: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                article_id = str(row.get("article_id") or "")
                key = article_id or str(row.get("md_path") or "")
                if not key:
                    continue
                self.latest_by_key[key] = row
                if article_id and row.get("status") == "completed":
                    self.completed.add(article_id)

    def is_completed(self, article_id: str) -> bool:
        return bool(article_id and article_id in self.completed)

    def append(self, row: dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            article_id = str(row.get("article_id") or "")
            key = article_id or str(row.get("md_path") or "")
            if key:
                self.latest_by_key[key] = row
            if article_id and row.get("status") == "completed":
                self.completed.add(article_id)


def project_root() -> Path:
    """Return the repository root, not the app/ package directory.

    The first patch used ``parents[3]`` from this file, which resolves to
    ``<repo>/app`` for the plural package path. That made CLI paths such as
    ``data/...`` become ``app/data/...``. Prefer the current working directory
    when it is the repo root, then walk upward from this file and look for the
    actual repo shape.
    """
    cwd = Path.cwd().resolve()
    if (cwd / "app").is_dir() and (cwd / "scripts").is_dir():
        return cwd

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "app").is_dir() and (parent / "scripts").is_dir():
            return parent
        if (parent / "pyproject.toml").exists() or (parent / "requirements.txt").exists():
            return parent

    # Fallback for app/ingestion/extraction/pipelines/full_corpus_ingest_pipeline.py
    return current.parents[4]


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or project_root()) / p


def safe_slug(value: str, max_len: int = 80) -> str:
    text = str(value or "article").strip().lower()
    text = re.sub(r'[\\/:*?"<>|]+', " ", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return (text[:max_len].strip("-") or "article")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {path}. "
            f"Run from repo root or pass an absolute --input-csv path."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_md_path(row: dict[str, str], md_root: Path) -> Path | None:
    raw = (
        row.get("md_path")
        or row.get("markdown_path")
        or row.get("article_markdown_path")
        or row.get("path")
        or row.get("file_path")
        or ""
    ).strip()
    if not raw:
        return None

    candidates: list[Path] = []
    p = Path(raw)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(md_root / p)
        candidates.append(project_root() / p)

    normalized = raw.replace("\\", "/")
    if "data/" in normalized:
        suffix = normalized[normalized.index("data/"):]
        candidates.append(project_root() / suffix)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def load_article_specs(input_csv: Path, md_root: Path) -> list[ArticleSpec]:
    rows = read_csv_rows(input_csv)
    specs: list[ArticleSpec] = []
    for idx, row in enumerate(rows):
        md_path = find_md_path(row, md_root)
        if md_path is None:
            continue
        specs.append(ArticleSpec(index=idx, md_path=md_path, row=row))
    return specs


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class FullCorpusIngestPipeline:
    def __init__(self, config: FullCorpusIngestConfig):
        self.config = config
        self.config.input_csv = resolve_path(config.input_csv)
        self.config.md_root = resolve_path(config.md_root)
        self.config.out_dir = resolve_path(config.out_dir)
        if config.checkpoint_jsonl is not None:
            self.config.checkpoint_jsonl = resolve_path(config.checkpoint_jsonl)

        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[PATH] repo_root={project_root()}")
        print(f"[PATH] input_csv={self.config.input_csv}")
        print(f"[PATH] md_root={self.config.md_root}")
        print(f"[PATH] out_dir={self.config.out_dir}")
        checkpoint_path = self.config.checkpoint_jsonl or (self.config.out_dir / "checkpoints.jsonl")
        self.checkpoint = IngestCheckpoint(checkpoint_path)

        self.extracted_dir = self.config.out_dir / "extracted_articles"
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

        self.raw_jsonl = self.config.out_dir / "raw_and_validated_chunk_extractions.jsonl"
        self.article_status_csv = self.config.out_dir / "article_status.csv"
        self.failed_articles_csv = self.config.out_dir / "failed_articles.csv"
        self.neo4j_counts_csv = self.config.out_dir / "neo4j_write_counts.csv"
        self.qdrant_counts_csv = self.config.out_dir / "qdrant_write_counts.csv"
        self.summary_json = self.config.out_dir / "ingest_summary.json"

        self.status_by_key: dict[str, dict[str, Any]] = {}
        self.failed_rows: list[dict[str, Any]] = []
        self.neo4j_rows: list[dict[str, Any]] = []
        self.qdrant_rows: list[dict[str, Any]] = []

        self.completed_count = 0
        self.failed_count = 0
        self.extracted_count = 0

        self.embedding: EmbeddingService | None = None
        self.neo4j_client: Neo4jClient | None = None
        self.qdrant_client: QdrantClientWrapper | None = None
        self.graph_repo: GraphRepository | None = None
        self.vector_repo: VectorRepository | None = None
        self.graph_builder: GraphBuilder | None = None
        self.medical_graph_builder: MedicalGraphBuilder | None = None
        self.vector_graph_builder: VectorGraphBuilder | None = None

    def run(self) -> dict[str, Any]:
        specs = self._select_specs()
        if self.config.dry_run:
            return self._dry_run(specs)

        self._connect()
        self._setup_targets()

        jobs: queue.Queue[ArticleExtractedJob | None] = queue.Queue()
        writer = threading.Thread(target=self._writer_loop, args=(jobs,), name="full-ingest-writer", daemon=False)
        writer.start()

        submitted = 0
        with ThreadPoolExecutor(max_workers=max(1, int(self.config.llm_workers))) as pool:
            futures = [pool.submit(self._extract_with_retry, spec) for spec in specs]
            submitted = len(futures)
            for future in as_completed(futures):
                try:
                    jobs.put(future.result())
                except Exception as exc:  # defensive; normal failures are encoded in ArticleExtractedJob
                    jobs.put(ArticleExtractedJob(spec=ArticleSpec(index=-1, md_path=Path("UNKNOWN")), error=f"{type(exc).__name__}: {exc}"))

        jobs.put(None)
        writer.join()

        summary = self._build_summary(selected=len(specs), submitted=submitted)
        self.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        self._print_post_ingest_checks()
        self._close()
        return summary

    def _select_specs(self) -> list[ArticleSpec]:
        specs = load_article_specs(self.config.input_csv, self.config.md_root)
        if self.config.start_index > 0:
            specs = [spec for spec in specs if spec.index >= self.config.start_index]
        if self.config.limit and self.config.limit > 0:
            specs = specs[: self.config.limit]

        if self.config.resume:
            kept: list[ArticleSpec] = []
            for spec in specs:
                try:
                    article = load_markdown_article(spec.md_path)
                    if self.checkpoint.is_completed(article.article_id):
                        print(f"[RESUME SKIP] {article.article_id} | {article.title}")
                        continue
                except Exception:
                    pass
                kept.append(spec)
            specs = kept

        print(f"[SELECTED] {len(specs)} articles")
        return specs

    def _dry_run(self, specs: list[ArticleSpec]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for spec in specs:
            try:
                article = load_markdown_article(spec.md_path)
                rows.append({
                    "index": spec.index,
                    "article_id": article.article_id,
                    "title": article.title,
                    "url": article.url,
                    "md_path": str(spec.md_path),
                    "status": "dry_run_selected",
                    "error": "",
                })
            except Exception as exc:
                rows.append({
                    "index": spec.index,
                    "article_id": "",
                    "title": "",
                    "url": "",
                    "md_path": str(spec.md_path),
                    "status": "dry_run_error",
                    "error": f"{type(exc).__name__}: {exc}",
                })

        out_csv = self.config.out_dir / "dry_run_articles.csv"
        write_csv(out_csv, rows, ["index", "article_id", "title", "url", "md_path", "status", "error"])
        summary = {"dry_run": True, "selected": len(rows), "out_csv": str(out_csv)}
        self.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    def _connect(self) -> None:
        self.embedding = EmbeddingService()
        self.neo4j_client = Neo4jClient(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )
        self.graph_repo = GraphRepository(self.neo4j_client)

        self.qdrant_client = QdrantClientWrapper(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY") or None,
        )
        self.vector_repo = VectorRepository(
            client=self.qdrant_client,
            vector_dim=self.embedding.vector_dim,
            distance=os.getenv("VECTOR_DISTANCE", "cosine"),
        )

        if not self.graph_repo.health_check():
            raise RuntimeError("Neo4j health check failed. Check NEO4J_* env and Docker.")
        if not self.vector_repo.health_check():
            raise RuntimeError("Qdrant health check failed. Check QDRANT_* env and Docker.")

        resolver = EntityResolver(self.graph_repo)
        self.graph_builder = GraphBuilder(self.graph_repo)
        self.medical_graph_builder = MedicalGraphBuilder(self.graph_repo, resolver=resolver)
        self.vector_graph_builder = VectorGraphBuilder(
            vector_repo=self.vector_repo,
            embedder=self.embedding,
            resolver=resolver,
            batch_size=self.config.embedding_batch_size,
        )

    def _setup_targets(self) -> None:
        assert self.graph_repo is not None
        assert self.vector_repo is not None
        assert self.vector_graph_builder is not None

        if self.config.recreate:
            if self.config.confirm_clear != CONFIRM_CLEAR_TOKEN:
                raise RuntimeError(
                    "Refuse to clear Neo4j/Qdrant. Required token: "
                    f"--recreate --confirm-clear {CONFIRM_CLEAR_TOKEN}"
                )
            print("[CLEAR] Neo4j graph + Qdrant collections")
            self.graph_repo.delete_all_data()
            self.vector_repo.delete_all_collections()

        self.graph_repo.setup_schema()
        self.vector_graph_builder.setup_collections(recreate=False)

    def _build_extraction_pipeline(self) -> ArticleExtractionPipeline:
        extractor = MedicalExtractor(
            entity_max_tokens=self.config.entity_max_tokens,
            relation_max_tokens=self.config.relation_max_tokens,
        )
        validator = ExtractionValidator()
        keyword_profiler = ArticleKeywordProfiler(
            model=(self.config.keyword_model or None) if self.config.keyword_llm else ""
        )
        return ArticleExtractionPipeline(
            extractor=extractor,
            validator=validator,
            keyword_profiler=keyword_profiler,
            use_fact_windows=True,
            fact_window_max_tokens=self.config.fact_window_max_tokens,
            fact_window_min_tokens=self.config.fact_window_min_tokens,
            fact_window_overlap_sentences=self.config.fact_window_overlap_sentences,
        )

    def _extract_with_retry(self, spec: ArticleSpec) -> ArticleExtractedJob:
        last_error = ""
        article = None
        for attempt in range(1, max(1, self.config.llm_retries) + 1):
            try:
                # Build a separate pipeline/client per worker attempt. This avoids
                # sharing OpenAI parser/client state across threads.
                pipeline = self._build_extraction_pipeline()
                article = load_markdown_article(spec.md_path)
                chunks = chunk_article(article, max_tokens=self.config.max_tokens)

                def progress(i: int, total: int, window: ChunkDocument) -> None:
                    print(f"[LLM] article_index={spec.index} window={i}/{total} chunk={window.chunk_id}")

                result = pipeline.extract_article(
                    article=article,
                    retrieval_chunks=chunks,
                    max_windows=0,
                    progress_callback=progress,
                )
                artifact_path = self._write_extracted_artifact(spec, article, result)
                return ArticleExtractedJob(spec=spec, article=article, result=result, artifact_path=artifact_path)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                sleep_s = min(60, 5 * attempt * attempt)
                print(f"[RETRY] article_index={spec.index} attempt={attempt} error={last_error}")
                time.sleep(sleep_s)

        return ArticleExtractedJob(spec=spec, article=article, error=last_error)

    def _write_extracted_artifact(
        self,
        spec: ArticleSpec,
        article: Any,
        result: ArticleExtractionResult,
    ) -> Path:
        path = self.extracted_dir / f"{spec.index:04d}_{safe_slug(article.article_id)}.json"
        payload = {
            "run_id": self.config.run_id,
            "article_index": spec.index,
            "article_id": article.article_id,
            "title": article.title,
            "url": article.url,
            "md_path": str(spec.md_path),
            "retrieval_chunks": [chunk.model_dump() for chunk in result.retrieval_chunks],
            "extraction_windows": [chunk.model_dump() for chunk in result.extraction_windows],
            "raw_graphs": [graph.model_dump() for graph in result.raw_graphs],
            "validated_graphs": [graph.model_dump() for graph in result.validated_graphs],
            "keyword_profile_debug": result.keyword_profile_debug,
            "windowing_debug": result.windowing_debug,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def _writer_loop(self, jobs: queue.Queue[ArticleExtractedJob | None]) -> None:
        while True:
            job = jobs.get()
            if job is None:
                jobs.task_done()
                break
            try:
                if job.error:
                    self._record_failed(job.spec, job.error, article=job.article)
                else:
                    self._record_extracted(job)
                    self._write_job(job)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                self._record_failed(job.spec, error, article=job.article)
            finally:
                self._flush_csvs()
                jobs.task_done()

    def _record_extracted(self, job: ArticleExtractedJob) -> None:
        article = job.article
        result = job.result
        if article is None or result is None:
            return
        row = self._base_row(
            job=job,
            status="extracted",
            retrieval_chunk_count=len(result.retrieval_chunks),
            extraction_window_count=len(result.extraction_windows),
            entity_count=sum(len(g.entities) for g in result.validated_graphs),
            relation_count=sum(len(g.relations) for g in result.validated_graphs),
        )
        self.checkpoint.append(row)
        self.status_by_key[article.article_id] = row
        self.extracted_count += 1

    def _write_job(self, job: ArticleExtractedJob) -> None:
        assert self.graph_builder is not None
        assert self.medical_graph_builder is not None
        assert self.vector_graph_builder is not None
        article = job.article
        result = job.result
        if article is None or result is None:
            raise RuntimeError("Cannot write empty extraction job.")

        extra_windows = result.extra_extraction_windows
        all_chunks_for_graph = list(result.retrieval_chunks) + list(extra_windows)

        # Parent retrieval chunks are the only chunks written to vector collections.
        # Child extraction windows are graph evidence nodes, not Vector RAG context.
        qdrant_chunk_count = self.vector_graph_builder.upsert_chunks(result.retrieval_chunks)
        qdrant_naive_chunk_count = self.vector_graph_builder.upsert_naive_chunks(result.retrieval_chunks)

        # Article + parent chunks + child fact windows + parent-child edges.
        self.graph_builder.ingest_article_with_chunks(article, all_chunks_for_graph)

        neo4j_entity_write_count = 0
        neo4j_relation_write_count = 0
        qdrant_entity_count = 0
        qdrant_relation_count = 0
        entity_count = 0
        relation_count = 0

        for window_index, (window, raw_graph, graph) in enumerate(
            zip(result.extraction_windows, result.raw_graphs, result.validated_graphs, strict=True),
            start=1,
        ):
            entity_count += len(graph.entities)
            relation_count += len(graph.relations)
            parent_chunk_id = dict(window.metadata or {}).get("parent_chunk_id")

            append_jsonl(
                self.raw_jsonl,
                {
                    "run_id": self.config.run_id,
                    "article_index": job.spec.index,
                    "article_id": article.article_id,
                    "article_title": article.title,
                    "title": article.title,
                    "url": article.url,
                    "md_path": str(job.spec.md_path),
                    "window_index": window_index,
                    "chunk_id": window.chunk_id,
                    "parent_chunk_id": parent_chunk_id,
                    "section": window.section,
                    "subsection": window.subsection,
                    "text": window.text,
                    "raw_graph": raw_graph.model_dump(),
                    "validated_graph": graph.model_dump(),
                    "validated_entities": [e.model_dump() for e in graph.entities],
                    "validated_relations": [r.model_dump() for r in graph.relations],
                },
            )

            neo4j_stats = self.medical_graph_builder.write_chunk_extraction(window, graph)
            qdrant_stats = self.vector_graph_builder.upsert_chunk_extraction(window, graph)

            neo4j_entity_write_count += int(neo4j_stats.get("entity_write_count") or 0)
            neo4j_relation_write_count += int(neo4j_stats.get("relation_write_count") or 0)
            qdrant_entity_count += int(qdrant_stats.get("entity_vector_count") or 0)
            qdrant_relation_count += int(qdrant_stats.get("relation_vector_count") or 0)

            self.neo4j_rows.append({"article_id": article.article_id, **neo4j_stats})
            self.qdrant_rows.append({"article_id": article.article_id, **qdrant_stats})

        completed = self._base_row(
            job=job,
            status="completed",
            retrieval_chunk_count=len(result.retrieval_chunks),
            extraction_window_count=len(result.extraction_windows),
            extra_extraction_window_count=len(extra_windows),
            entity_count=entity_count,
            relation_count=relation_count,
            neo4j_chunk_count=len(all_chunks_for_graph),
            neo4j_entity_write_count=neo4j_entity_write_count,
            neo4j_relation_write_count=neo4j_relation_write_count,
            qdrant_chunk_count=qdrant_chunk_count,
            qdrant_naive_chunk_count=qdrant_naive_chunk_count,
            qdrant_entity_count=qdrant_entity_count,
            qdrant_relation_count=qdrant_relation_count,
        )
        self.checkpoint.append(completed)
        self.status_by_key[article.article_id] = completed
        self.completed_count += 1
        print(f"[COMPLETED] {article.article_id} | windows={len(result.extraction_windows)} rel={relation_count} | {article.title}")

    def _base_row(self, *, job: ArticleExtractedJob, status: str, **updates: Any) -> dict[str, Any]:
        article = job.article
        base = {
            "run_id": self.config.run_id,
            "article_index": job.spec.index,
            "article_id": getattr(article, "article_id", "") or "",
            "title": getattr(article, "title", "") or "",
            "url": getattr(article, "url", "") or "",
            "md_path": str(job.spec.md_path),
            "artifact_path": str(job.artifact_path or ""),
            "status": status,
            "retrieval_chunk_count": 0,
            "extraction_window_count": 0,
            "extra_extraction_window_count": 0,
            "entity_count": 0,
            "relation_count": 0,
            "neo4j_chunk_count": 0,
            "neo4j_entity_write_count": 0,
            "neo4j_relation_write_count": 0,
            "qdrant_chunk_count": 0,
            "qdrant_naive_chunk_count": 0,
            "qdrant_entity_count": 0,
            "qdrant_relation_count": 0,
            "error": "",
        }
        base.update(updates)
        return base

    def _record_failed(self, spec: ArticleSpec, error: str, article: Any | None = None) -> None:
        job = ArticleExtractedJob(spec=spec, article=article, error=error)
        row = self._base_row(job=job, status="failed", error=error)
        self.checkpoint.append(row)
        key = row.get("article_id") or row.get("md_path") or str(spec.index)
        self.status_by_key[str(key)] = row
        self.failed_rows.append(row)
        self.failed_count += 1
        print(f"[FAILED] index={spec.index} path={spec.md_path} error={error}")

    def _flush_csvs(self) -> None:
        fields = [
            "run_id",
            "article_index",
            "article_id",
            "title",
            "url",
            "md_path",
            "artifact_path",
            "status",
            "retrieval_chunk_count",
            "extraction_window_count",
            "extra_extraction_window_count",
            "entity_count",
            "relation_count",
            "neo4j_chunk_count",
            "neo4j_entity_write_count",
            "neo4j_relation_write_count",
            "qdrant_chunk_count",
            "qdrant_naive_chunk_count",
            "qdrant_entity_count",
            "qdrant_relation_count",
            "error",
        ]
        write_csv(self.article_status_csv, list(self.status_by_key.values()), fields)
        write_csv(self.failed_articles_csv, self.failed_rows, fields)
        write_csv(
            self.neo4j_counts_csv,
            self.neo4j_rows,
            [
                "article_id",
                "chunk_id",
                "entity_write_count",
                "mention_write_count",
                "relation_write_count",
                "candidate_entity_count",
                "selected_entity_count",
                "endpoint_entity_count",
                "kept_anchor_entity_count",
                "pruned_orphan_candidate_count",
                "skipped_graph_ingest",
                "skip_reason",
            ],
        )
        write_csv(
            self.qdrant_counts_csv,
            self.qdrant_rows,
            [
                "article_id",
                "chunk_id",
                "entity_vector_count",
                "relation_vector_count",
                "skipped_graph_vector_ingest",
                "skip_reason",
            ],
        )

    def _build_summary(self, *, selected: int, submitted: int) -> dict[str, Any]:
        graph_stats = self.graph_repo.get_graph_stats() if self.graph_repo else {}
        parent_child_edges = self.graph_repo.count_parent_child_edges() if self.graph_repo else 0
        qdrant_counts: dict[str, int] = {}
        if self.vector_repo:
            for name in [CHUNK_COLLECTION, NAIVE_CHUNK_COLLECTION, ENTITY_COLLECTION, RELATION_COLLECTION]:
                qdrant_counts[name] = self.vector_repo.count_points(name)
        return {
            "run_id": self.config.run_id,
            "selected_articles": selected,
            "submitted_articles": submitted,
            "extracted_articles": self.extracted_count,
            "completed_articles": self.completed_count,
            "failed_articles": self.failed_count,
            "neo4j": {**graph_stats, "parent_child_edges": parent_child_edges},
            "qdrant": qdrant_counts,
            "config": asdict(self.config),
        }

    def _print_post_ingest_checks(self) -> None:
        print("\n[POST-INGEST NEO4J CHECKS]")
        print("""
MATCH (a:Article)
WITH count(a) AS articles
MATCH (c:Chunk)
WITH articles, count(c) AS chunks
MATCH (e:Entity)
WITH articles, chunks, count(e) AS entities
MATCH ()-[r:MEDICAL_RELATION]->()
RETURN articles, chunks, entities, count(r) AS relations;
""".strip())
        print("""
MATCH ()-[r:MEDICAL_RELATION]->()
RETURN
  count(r) AS rels,
  sum(CASE WHEN r.evidence_text IS NULL OR r.evidence_text = '' THEN 1 ELSE 0 END) AS missing_evidence,
  sum(CASE WHEN r.keywords IS NULL OR size(r.keywords) = 0 THEN 1 ELSE 0 END) AS missing_keywords,
  sum(CASE WHEN r.evidence_chunk_ids IS NULL OR size(r.evidence_chunk_ids) = 0 THEN 1 ELSE 0 END) AS missing_evidence_chunks;
""".strip())
        print("""
MATCH (parent:Chunk)-[:HAS_FACT_WINDOW]->(child:Chunk)
RETURN count(*) AS parent_child_edges;
""".strip())
        print("\n[POST-INGEST QDRANT CHECKS]")
        print(f"- count {CHUNK_COLLECTION}")
        print(f"- count {NAIVE_CHUNK_COLLECTION}")
        print(f"- count {ENTITY_COLLECTION}")
        print(f"- count {RELATION_COLLECTION}")
        print("- scroll sample medical_relations payload: verify relation_id, chunk_id, evidence_text, keywords")

    def _close(self) -> None:
        if self.neo4j_client:
            self.neo4j_client.close()
        if self.qdrant_client:
            self.qdrant_client.close()


def run_full_corpus_ingest(config: FullCorpusIngestConfig) -> dict[str, Any]:
    if load_dotenv is not None:
        load_dotenv(project_root() / ".env")
    pipeline = FullCorpusIngestPipeline(config)
    return pipeline.run()
