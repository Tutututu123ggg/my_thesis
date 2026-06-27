#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze whether 4 same-anchor medical articles produce a connected graph
or just a star/flower around one disease hub.

Usage example:

python scripts/analyze_same_anchor_graph.py ^
  --entities data/debug/unbiased_entities.csv ^
  --relations data/debug/unbiased_relations.csv ^
  --anchor "viêm da cơ địa" ^
  --main-entity "viêm da cơ địa" ^
  --out-dir data/debug/anchor_viem_da_co_dia_graph_audit

Dependency-light: Python stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Set

h
ALLOWED_ENTITY_TYPES = {
    "BENH_LY",
    "BIEU_HIEN_LAM_SANG",
    "YEU_TO_BENH_SINH",
    "YEU_TO_NGOAI_SINH",
    "PHUONG_PHAP_CHAN_DOAN",
    "CAN_THIEP_Y_TE",
    "THUOC_VA_HOAT_CHAT",
    "DOI_TUONG",
    "VI_TRI_GIAI_PHAU",
}

ALLOWED_RELATION_TYPES = {
    "CO_BIEU_HIEN",
    "LIEN_QUAN_YEU_TO",
    "ANH_HUONG_DEN",
    "GAY_BIEN_CHUNG",
    "DIEU_TRI_BANG",
    "PHONG_NGUA_BANG",
    "CHAN_DOAN_BANG",
    "DONG_NGHIA_VOI",
    "LA_DANG_CUA",
    "CO_LIEN_HE_VOI",
}

GENERIC_ENTITY_TERMS = {
    "bệnh",
    "bệnh lý",
    "bệnh lý da liễu",
    "triệu chứng",
    "biểu hiện",
    "dấu hiệu",
    "tình trạng",
    "vấn đề",
    "nguyên nhân",
    "yếu tố",
    "thuốc",
    "điều trị",
    "phương pháp",
    "cơ sở y tế",
    "cơ sở y tế uy tín",
    "sản phẩm",
    "sản phẩm chăm sóc da",
    "sản phẩm làm sạch da",
    "nước",
    "vệ sinh",
    "chăm sóc",
    "da",
    "vùng da",
    "tổn thương",
    "thương tổn",
}

GENERIC_KEYWORDS = {
    "triệu chứng",
    "biểu hiện",
    "nguyên nhân",
    "yếu tố liên quan",
    "điều trị",
    "phòng ngừa",
    "chẩn đoán",
    "bệnh da liễu",
    "bệnh lý",
    "liên quan",
    "biến chứng",
    "thuốc",
}


def norm_text(s: object) -> str:
    if s is None:
        return ""
    s = str(s).replace("\ufeff", "")
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    return s


def key_text(s: object) -> str:
    return norm_text(s).lower()


def canonical_header(h: str) -> str:
    return key_text(h).replace(" ", "_").replace("-", "_")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [{canonical_header(k): norm_text(v) for k, v in row.items()} for row in reader]


def pick_col(rows: Sequence[Dict[str, str]], candidates: Sequence[str], required: bool = True) -> Optional[str]:
    if not rows:
        if required:
            raise ValueError("CSV has no rows.")
        return None
    cols = set(rows[0].keys())
    for c in [canonical_header(x) for x in candidates]:
        if c in cols:
            return c
    if required:
        raise ValueError(f"Cannot find any column among {candidates}. Available columns: {sorted(cols)}")
    return None


def parse_keywords(value: str) -> List[str]:
    value = norm_text(value)
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [norm_text(x) for x in parsed if norm_text(x)]
    except Exception:
        pass
    parts = re.split(r"\s*(?:\||;|,|\n)\s*", value)
    return [p for p in (norm_text(x) for x in parts) if p]


def article_key(row: Dict[str, str], article_col: Optional[str], title_col: Optional[str]) -> str:
    if article_col and row.get(article_col):
        return row[article_col]
    if title_col and row.get(title_col):
        return row[title_col]
    return "__unknown_article__"


def contains_anchor(row: Dict[str, str], anchor: str) -> bool:
    if not anchor:
        return True
    anchor_k = key_text(anchor)
    blob = " | ".join(key_text(v) for v in row.values())
    return anchor_k in blob


def connected_components(nodes: Set[str], adj: Dict[str, Set[str]]) -> List[Set[str]]:
    seen: Set[str] = set()
    comps: List[Set[str]] = []
    for start in sorted(nodes):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = {start}
        while q:
            u = q.popleft()
            for v in adj.get(u, set()):
                if v not in seen:
                    seen.add(v)
                    comp.add(v)
                    q.append(v)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def count_triangles(nodes: Set[str], adj: Dict[str, Set[str]]) -> int:
    ordered = sorted(nodes)
    rank = {n: i for i, n in enumerate(ordered)}
    triangles = 0
    for u in ordered:
        nbrs = [v for v in adj.get(u, set()) if rank[v] > rank[u]]
        nbrs_set = set(nbrs)
        for v in nbrs:
            later = {w for w in adj.get(v, set()) if rank[w] > rank[v]}
            triangles += len(nbrs_set.intersection(later))
    return triangles


def build_graph(edges: Sequence[Tuple[str, str]]) -> Tuple[Set[str], Dict[str, Set[str]], Set[Tuple[str, str]]]:
    nodes: Set[str] = set()
    adj: Dict[str, Set[str]] = defaultdict(set)
    undirected_edges: Set[Tuple[str, str]] = set()
    for a, b in edges:
        a, b = key_text(a), key_text(b)
        if not a or not b or a == b:
            continue
        x, y = sorted([a, b])
        nodes.update([x, y])
        adj[x].add(y)
        adj[y].add(x)
        undirected_edges.add((x, y))
    return nodes, adj, undirected_edges


def graph_metrics(article: str, edges: Sequence[Tuple[str, str]], main_entity: Optional[str] = None) -> Dict[str, object]:
    nodes, adj, undirected_edges = build_graph(edges)
    degree = {n: len(adj.get(n, set())) for n in nodes}
    comps = connected_components(nodes, adj) if nodes else []
    triangles = count_triangles(nodes, adj) if nodes else 0

    if main_entity:
        main_k = key_text(main_entity)
        if main_k in nodes:
            hub = main_k
            hub_source = "main_entity"
        else:
            hub = max(degree, key=degree.get) if degree else ""
            hub_source = "top_degree_fallback"
    else:
        hub = max(degree, key=degree.get) if degree else ""
        hub_source = "top_degree"

    total_edges = len(undirected_edges)
    hub_edges = 0
    nonhub_edges: List[Tuple[str, str]] = []
    for a, b in undirected_edges:
        if hub and (a == hub or b == hub):
            hub_edges += 1
        else:
            nonhub_edges.append((a, b))

    leaves = sum(1 for n, d in degree.items() if d == 1)
    largest_component = len(comps[0]) if comps else 0
    hub_edge_pct = hub_edges / total_edges if total_edges else 0.0
    leaf_pct = leaves / len(nodes) if nodes else 0.0
    star_score = 0.65 * hub_edge_pct + 0.35 * leaf_pct

    return {
        "article": article,
        "nodes": len(nodes),
        "edges": total_edges,
        "components": len(comps),
        "largest_component_nodes": largest_component,
        "largest_component_pct": round(largest_component / len(nodes), 4) if nodes else 0.0,
        "hub": hub,
        "hub_source": hub_source,
        "hub_degree": degree.get(hub, 0),
        "hub_edge_pct": round(hub_edge_pct, 4),
        "nonhub_edges": len(nonhub_edges),
        "leaf_nodes": leaves,
        "leaf_pct": round(leaf_pct, 4),
        "triangles": triangles,
        "star_score": round(star_score, 4),
        "top_nodes": sorted(degree.items(), key=lambda x: (-x[1], x[0]))[:20],
        "nonhub_edge_list": nonhub_edges,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", required=True, help="Path to entity CSV")
    ap.add_argument("--relations", required=True, help="Path to relation CSV")
    ap.add_argument("--anchor", default="", help='Anchor phrase, e.g. "viêm da cơ địa"')
    ap.add_argument("--main-entity", default="", help='Main hub entity to test, e.g. "viêm da cơ địa"')
    ap.add_argument("--out-dir", required=True, help="Output report directory")
    ap.add_argument("--filter-anchor", action="store_true", help="Filter rows containing --anchor anywhere.")
    args = ap.parse_args()

    entity_rows = read_csv_rows(Path(args.entities))
    relation_rows = read_csv_rows(Path(args.relations))

    ent_name_col = pick_col(entity_rows, ["entity_name", "name", "canonical_name", "entity", "text"], required=True)
    ent_type_col = pick_col(entity_rows, ["entity_type", "type", "label"], required=False)
    ent_article_col = pick_col(entity_rows, ["article_url", "source_url", "url", "article_id", "source", "file", "path"], required=False)
    ent_title_col = pick_col(entity_rows, ["title", "article_title", "document_title"], required=False)

    subj_col = pick_col(relation_rows, ["subject", "source", "head", "from", "source_entity", "subj"], required=True)
    obj_col = pick_col(relation_rows, ["object", "target", "tail", "to", "target_entity", "obj"], required=True)
    rel_type_col = pick_col(relation_rows, ["relation_type", "type", "predicate", "relation", "label"], required=False)
    kw_col = pick_col(relation_rows, ["keywords", "keyword", "relation_keywords", "search_keywords"], required=False)
    ev_col = pick_col(relation_rows, ["evidence", "description", "sentence", "support", "text"], required=False)
    rel_article_col = pick_col(relation_rows, ["article_url", "source_url", "url", "article_id", "source", "file", "path"], required=False)
    rel_title_col = pick_col(relation_rows, ["title", "article_title", "document_title"], required=False)

    if args.filter_anchor and args.anchor:
        entity_rows = [r for r in entity_rows if contains_anchor(r, args.anchor)]
        relation_rows = [r for r in relation_rows if contains_anchor(r, args.anchor)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entity_audit_rows: List[Dict[str, object]] = []
    entity_type_counter = Counter()
    entity_articles: Dict[str, Set[str]] = defaultdict(set)
    for r in entity_rows:
        name = r.get(ent_name_col, "")
        name_k = key_text(name)
        etype = r.get(ent_type_col, "") if ent_type_col else ""
        article = article_key(r, ent_article_col, ent_title_col)
        entity_articles[name_k].add(article)
        if etype:
            entity_type_counter[etype] += 1
        issues = []
        if name_k in GENERIC_ENTITY_TERMS:
            issues.append("generic_entity")
        if etype and etype not in ALLOWED_ENTITY_TYPES:
            issues.append("entity_type_outside_schema")
        if len(name_k) <= 2:
            issues.append("too_short")
        if issues:
            entity_audit_rows.append({"article": article, "entity": name, "entity_type": etype, "issues": "; ".join(issues)})

    repeated_entities = [
        {"entity": ent, "article_count": len(arts), "articles": " | ".join(sorted(arts))}
        for ent, arts in sorted(entity_articles.items(), key=lambda x: (-len(x[1]), x[0]))
        if len(arts) >= 2
    ]

    per_article_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    all_edges: List[Tuple[str, str]] = []
    combined_edge_rows: List[Dict[str, object]] = []
    keyword_audit_rows: List[Dict[str, object]] = []
    relation_type_counter = Counter()
    duplicate_edge_counter = Counter()

    for idx, r in enumerate(relation_rows, start=1):
        article = article_key(r, rel_article_col, rel_title_col)
        subj = r.get(subj_col, "")
        obj = r.get(obj_col, "")
        subj_k, obj_k = key_text(subj), key_text(obj)
        rel_type = r.get(rel_type_col, "") if rel_type_col else ""
        kws = parse_keywords(r.get(kw_col, "") if kw_col else "")
        evidence = r.get(ev_col, "") if ev_col else ""

        if rel_type:
            relation_type_counter[rel_type] += 1
        if subj_k and obj_k and subj_k != obj_k:
            per_article_edges[article].append((subj, obj))
            all_edges.append((subj, obj))
            a, b = sorted([subj_k, obj_k])
            duplicate_edge_counter[(article, a, b, rel_type)] += 1

        combined_edge_rows.append({
            "row_id": idx, "article": article, "subject": subj, "relation_type": rel_type,
            "object": obj, "keywords": " | ".join(kws), "evidence": evidence,
        })

        issues = []
        if not kws:
            issues.append("missing_keywords")
        if rel_type and rel_type not in ALLOWED_RELATION_TYPES:
            issues.append("relation_type_outside_base_schema")
        if subj_k == obj_k and subj_k:
            issues.append("self_loop")
        if subj_k in GENERIC_ENTITY_TERMS or obj_k in GENERIC_ENTITY_TERMS:
            issues.append("edge_has_generic_entity")
        kw_blob = " ".join(key_text(k) for k in kws)
        if kws and subj_k and subj_k not in kw_blob:
            issues.append("keyword_missing_subject")
        if kws and obj_k and obj_k not in kw_blob:
            issues.append("keyword_missing_object")
        if kws and all(key_text(k) in GENERIC_KEYWORDS for k in kws):
            issues.append("keywords_too_generic_only")
        if kws and len(kws) < 3:
            issues.append("few_keywords")
        if issues:
            keyword_audit_rows.append({
                "row_id": idx, "article": article, "subject": subj, "relation_type": rel_type,
                "object": obj, "keywords": " | ".join(kws), "issues": "; ".join(issues),
            })

    duplicate_rows = [
        {"article": article, "subject_norm": a, "object_norm": b, "relation_type": rel_type, "count": count}
        for (article, a, b, rel_type), count in duplicate_edge_counter.items()
        if count > 1
    ]

    metrics_rows: List[Dict[str, object]] = []
    top_nodes_rows: List[Dict[str, object]] = []
    nonhub_rows: List[Dict[str, object]] = []

    for article, edges in sorted(per_article_edges.items()):
        m = graph_metrics(article, edges, args.main_entity or None)
        metrics_rows.append({k: v for k, v in m.items() if k not in {"top_nodes", "nonhub_edge_list"}})
        for rank, (node, degree) in enumerate(m["top_nodes"], start=1):
            top_nodes_rows.append({"article": article, "rank": rank, "node": node, "degree": degree})
        for a, b in m["nonhub_edge_list"]:
            nonhub_rows.append({"article": article, "endpoint_a": a, "endpoint_b": b})

    combined = graph_metrics("__COMBINED__", all_edges, args.main_entity or None)
    metrics_rows.append({k: v for k, v in combined.items() if k not in {"top_nodes", "nonhub_edge_list"}})
    for rank, (node, degree) in enumerate(combined["top_nodes"], start=1):
        top_nodes_rows.append({"article": "__COMBINED__", "rank": rank, "node": node, "degree": degree})
    for a, b in combined["nonhub_edge_list"]:
        nonhub_rows.append({"article": "__COMBINED__", "endpoint_a": a, "endpoint_b": b})

    rel_entity_articles: Dict[str, Set[str]] = defaultdict(set)
    for article, edges in per_article_edges.items():
        for a, b in edges:
            rel_entity_articles[key_text(a)].add(article)
            rel_entity_articles[key_text(b)].add(article)
    repeated_relation_nodes = [
        {"node": n, "article_count": len(arts), "articles": " | ".join(sorted(arts))}
        for n, arts in sorted(rel_entity_articles.items(), key=lambda x: (-len(x[1]), x[0]))
        if len(arts) >= 2
    ]

    write_csv(out_dir / "per_article_metrics.csv", metrics_rows, [
        "article", "nodes", "edges", "components", "largest_component_nodes",
        "largest_component_pct", "hub", "hub_source", "hub_degree",
        "hub_edge_pct", "nonhub_edges", "leaf_nodes", "leaf_pct",
        "triangles", "star_score",
    ])
    write_csv(out_dir / "top_nodes.csv", top_nodes_rows, ["article", "rank", "node", "degree"])
    write_csv(out_dir / "nonhub_edges.csv", nonhub_rows, ["article", "endpoint_a", "endpoint_b"])
    write_csv(out_dir / "combined_edges.csv", combined_edge_rows, ["row_id", "article", "subject", "relation_type", "object", "keywords", "evidence"])
    write_csv(out_dir / "keyword_audit.csv", keyword_audit_rows, ["row_id", "article", "subject", "relation_type", "object", "keywords", "issues"])
    write_csv(out_dir / "entity_audit.csv", entity_audit_rows, ["article", "entity", "entity_type", "issues"])
    write_csv(out_dir / "duplicate_edges.csv", duplicate_rows, ["article", "subject_norm", "object_norm", "relation_type", "count"])
    write_csv(out_dir / "repeated_entities.csv", repeated_entities, ["entity", "article_count", "articles"])
    write_csv(out_dir / "repeated_relation_nodes.csv", repeated_relation_nodes, ["node", "article_count", "articles"])
    write_csv(out_dir / "entity_type_counts.csv", [{"entity_type": k, "count": v} for k, v in entity_type_counter.most_common()], ["entity_type", "count"])
    write_csv(out_dir / "relation_type_counts.csv", [{"relation_type": k, "count": v} for k, v in relation_type_counter.most_common()], ["relation_type", "count"])

    star_label = (
        "RAT_HOA_THI" if combined["star_score"] >= 0.75 else
        "KHA_HOA_THI" if combined["star_score"] >= 0.60 else
        "TRUNG_BINH" if combined["star_score"] >= 0.45 else
        "LIEN_THONG_NGANG_TOT"
    )

    lines = [
        "# Same-anchor graph audit",
        "",
        f"- Anchor: `{args.anchor or '(not filtered)'}`",
        f"- Main entity: `{args.main_entity or '(auto top-degree)'}`",
        f"- Entity rows analyzed: `{len(entity_rows)}`",
        f"- Relation rows analyzed: `{len(relation_rows)}`",
        "",
        "## Combined graph",
    ]
    for k, v in {kk: vv for kk, vv in combined.items() if kk not in {"top_nodes", "nonhub_edge_list"}}.items():
        lines.append(f"- `{k}`: `{v}`")
    lines += [
        f"- Verdict by star_score: **{star_label}**",
        "",
        "## How to read",
        "- `hub_edge_pct` cao nghĩa là phần lớn cạnh dính vào bệnh chính/hub.",
        "- `nonhub_edges` cao nghĩa là có nhiều cạnh phụ-phụ, tốt hơn cho HippoRAG/PPR.",
        "- `leaf_pct` cao nghĩa là nhiều node lá, graph giống hoa thị.",
        "- `triangles` > 0 nghĩa là có vòng tam giác/cluster nhỏ, thường tốt hơn graph hoa thị thuần.",
        "- `star_score` càng cao càng giống hoa thị.",
        "",
        "## Output files",
        "- `per_article_metrics.csv`: chấm hoa thị theo từng bài và combined.",
        "- `nonhub_edges.csv`: các cạnh phụ-phụ không đi qua hub chính.",
        "- `repeated_relation_nodes.csv`: node xuất hiện trong >=2 bài, cầu nối cross-source.",
        "- `keyword_audit.csv`: keyword rỗng/quá generic/thiếu subject-object.",
        "- `entity_audit.csv`: entity generic hoặc type ngoài schema.",
        "",
        "## Quick thresholds",
        "- Tốt cho HippoRAG: `hub_edge_pct < 0.70`, `leaf_pct < 0.65`, `nonhub_edges` nhiều, có `triangles`.",
        "- Chấp nhận cho LightRAG: keyword tốt + evidence đúng, dù `hub_edge_pct` còn cao.",
        "- Đáng sửa prompt: `hub_edge_pct > 0.85` và `nonhub_edges` rất thấp ở cả 4 bài.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Wrote audit report to: {out_dir}")
    print(f"[SUMMARY] combined star_score={combined['star_score']} hub_edge_pct={combined['hub_edge_pct']} "
          f"leaf_pct={combined['leaf_pct']} nonhub_edges={combined['nonhub_edges']} triangles={combined['triangles']}")
    print(f"[VERDICT] {star_label}")


if __name__ == "__main__":
    main()
