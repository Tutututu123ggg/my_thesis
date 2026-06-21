from __future__ import annotations

import argparse
import itertools
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class DiseaseGroup:
    concept: str
    synonyms: list[str]
    hard_negatives: list[str]


GROUPS: list[DiseaseGroup] = [
    DiseaseGroup(
        concept="viêm da cơ địa",
        synonyms=["viêm da cơ địa", "chàm thể tạng", "eczema", "atopic dermatitis", "bệnh chàm"],
        hard_negatives=["viêm da tiếp xúc", "viêm da tiết bã", "vảy nến", "nấm da", "mày đay", "chàm sữa"],
    ),
    DiseaseGroup(
        concept="chàm sữa",
        synonyms=["chàm sữa", "lác sữa", "eczema ở trẻ sơ sinh", "infantile eczema"],
        hard_negatives=["bệnh chàm", "viêm da cơ địa", "viêm da tiếp xúc", "hăm tã", "rôm sảy"],
    ),
    DiseaseGroup(
        concept="mày đay",
        synonyms=["mày đay", "mề đay", "nổi mề đay", "urticaria", "hives"],
        hard_negatives=["phát ban", "dị ứng da", "viêm da cơ địa", "mụn trứng cá", "zona thần kinh"],
    ),
    DiseaseGroup(
        concept="hen phế quản",
        synonyms=["hen phế quản", "hen suyễn", "bệnh hen", "asthma", "bronchial asthma"],
        hard_negatives=["COPD", "bệnh phổi tắc nghẽn mạn tính", "viêm phế quản", "viêm phổi", "giãn phế quản"],
    ),
    DiseaseGroup(
        concept="COPD",
        synonyms=["COPD", "bệnh phổi tắc nghẽn mạn tính", "phổi tắc nghẽn mạn tính", "chronic obstructive pulmonary disease"],
        hard_negatives=["hen phế quản", "viêm phế quản mạn tính", "khí phế thũng", "viêm phổi", "lao phổi"],
    ),
    DiseaseGroup(
        concept="viêm phổi",
        synonyms=["viêm phổi", "pneumonia", "nhiễm trùng phổi", "lung infection"],
        hard_negatives=["viêm phế quản", "viêm họng", "hen phế quản", "lao phổi", "COVID-19"],
    ),
    DiseaseGroup(
        concept="viêm mũi dị ứng",
        synonyms=["viêm mũi dị ứng", "dị ứng mũi", "allergic rhinitis", "hay fever"],
        hard_negatives=["viêm xoang", "cảm lạnh", "cúm", "viêm họng", "hen phế quản"],
    ),
    DiseaseGroup(
        concept="đái tháo đường",
        synonyms=["đái tháo đường", "tiểu đường", "diabetes", "diabetes mellitus"],
        hard_negatives=["hạ đường huyết", "rối loạn lipid máu", "tăng huyết áp", "béo phì", "hội chứng chuyển hóa"],
    ),
    DiseaseGroup(
        concept="tăng huyết áp",
        synonyms=["tăng huyết áp", "cao huyết áp", "hypertension", "high blood pressure"],
        hard_negatives=["hạ huyết áp", "rối loạn nhịp tim", "suy tim", "đột quỵ", "xơ vữa động mạch"],
    ),
    DiseaseGroup(
        concept="trào ngược dạ dày thực quản",
        synonyms=["trào ngược dạ dày thực quản", "trào ngược dạ dày", "GERD", "gastroesophageal reflux disease"],
        hard_negatives=["viêm loét dạ dày", "đau dạ dày", "viêm đại tràng", "hội chứng ruột kích thích", "khó tiêu"],
    ),
    DiseaseGroup(
        concept="thủy đậu",
        synonyms=["thủy đậu", "bệnh thủy đậu", "varicella", "chickenpox"],
        hard_negatives=["zona thần kinh", "sởi", "rubella", "tay chân miệng", "mụn nước"],
    ),
    DiseaseGroup(
        concept="sởi",
        synonyms=["sởi", "bệnh sởi", "measles", "rubeola"],
        hard_negatives=["rubella", "thủy đậu", "tay chân miệng", "sốt phát ban", "dị ứng da"],
    ),
    DiseaseGroup(
        concept="lao phổi",
        synonyms=["lao phổi", "bệnh lao phổi", "pulmonary tuberculosis", "TB phổi"],
        hard_negatives=["viêm phổi", "ung thư phổi", "COPD", "viêm phế quản", "ho kéo dài"],
    ),
    DiseaseGroup(
        concept="mụn trứng cá",
        synonyms=["mụn trứng cá", "trứng cá", "acne", "acne vulgaris"],
        hard_negatives=["phát ban mụn trứng cá", "viêm nang lông", "mụn nhọt", "viêm da tiết bã", "rosacea"],
    ),
    DiseaseGroup(
        concept="lang ben",
        synonyms=["lang ben", "nấm lang ben", "pityriasis versicolor", "tinea versicolor"],
        hard_negatives=["nấm da", "hắc lào", "viêm da cơ địa", "bạch biến", "vảy nến"],
    ),
    DiseaseGroup(
        concept="hắc lào",
        synonyms=["hắc lào", "lác đồng tiền", "nấm da thân", "tinea corporis", "ringworm"],
        hard_negatives=["lang ben", "nấm móng", "viêm da tiếp xúc", "vảy nến", "chàm"],
    ),
]


def maybe_prefix(texts: list[str], model_name: str) -> list[str]:
    # E5 models are trained with query/passsage prefixes.
    # For short entity-to-entity similarity, using query prefix for both sides is acceptable.
    if "e5" in model_name.lower():
        return [f"query: {t}" for t in texts]
    return texts


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    return vectors @ vectors.T


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.array(values), q))


def unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def run_one_model(model_name: str, output_dir: Path, batch_size: int) -> None:
    print("\n" + "=" * 100)
    print(f"MODEL: {model_name}")
    print("=" * 100)

    all_terms: list[str] = []
    for group in GROUPS:
        all_terms.extend(group.synonyms)
        all_terms.extend(group.hard_negatives)
    all_terms = unique_keep_order(all_terms)

    model = SentenceTransformer(model_name, trust_remote_code=True)

    encoded_texts = maybe_prefix(all_terms, model_name)
    vectors = model.encode(
        encoded_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    idx = {term: i for i, term in enumerate(all_terms)}
    sims = cosine_matrix(vectors)

    rows: list[dict] = []

    for group in GROUPS:
        for a, b in itertools.combinations(group.synonyms, 2):
            rows.append({
                "model": model_name,
                "concept": group.concept,
                "kind": "synonym",
                "a": a,
                "b": b,
                "similarity": float(sims[idx[a], idx[b]]),
            })

        for a in group.synonyms:
            for b in group.hard_negatives:
                if a in idx and b in idx:
                    rows.append({
                        "model": model_name,
                        "concept": group.concept,
                        "kind": "hard_negative",
                        "a": a,
                        "b": b,
                        "similarity": float(sims[idx[a], idx[b]]),
                    })

    df = pd.DataFrame(rows)

    pos = df[df["kind"] == "synonym"]["similarity"].tolist()
    neg = df[df["kind"] == "hard_negative"]["similarity"].tolist()

    print("\nSUMMARY")
    print("-" * 100)
    print(f"positive synonym pairs: {len(pos)}")
    print(f"hard-negative pairs    : {len(neg)}")
    print(f"POS mean={statistics.mean(pos):.4f} median={statistics.median(pos):.4f} p10={percentile(pos, 0.10):.4f} min={min(pos):.4f} max={max(pos):.4f}")
    print(f"NEG mean={statistics.mean(neg):.4f} median={statistics.median(neg):.4f} p90={percentile(neg, 0.90):.4f} max={max(neg):.4f}")

    print("\nLOWEST SYNONYM PAIRS")
    print("-" * 100)
    for _, r in df[df["kind"] == "synonym"].sort_values("similarity").head(40).iterrows():
        print(f'{r["similarity"]:.4f} | {r["concept"]} | {r["a"]} <> {r["b"]}')

    print("\nHIGHEST HARD NEGATIVES")
    print("-" * 100)
    for _, r in df[df["kind"] == "hard_negative"].sort_values("similarity", ascending=False).head(50).iterrows():
        print(f'{r["similarity"]:.4f} | {r["concept"]} | {r["a"]} <> {r["b"]}')

    print("\nTHRESHOLD SWEEP")
    print("-" * 100)
    threshold_rows = []
    for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        tp = sum(s >= th for s in pos)
        fn = sum(s < th for s in pos)
        fp = sum(s >= th for s in neg)
        tn = sum(s < th for s in neg)

        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0

        threshold_rows.append({
            "model": model_name,
            "threshold": th,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fpr,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

        print(
            f"th={th:.2f} | "
            f"precision={precision:.3f} recall={recall:.3f} fpr={fpr:.3f} "
            f"| tp={tp} fp={fp} fn={fn} tn={tn}"
        )

    safe_name = model_name.replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = output_dir / f"{safe_name}_pairs.csv"
    thresholds_path = output_dir / f"{safe_name}_thresholds.csv"

    df.sort_values(["kind", "similarity"], ascending=[True, False]).to_csv(
        pairs_path,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(threshold_rows).to_csv(
        thresholds_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved:")
    print("-", pairs_path)
    print("-", thresholds_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "BAAI/bge-m3",
            "intfloat/multilingual-e5-large",
        ],
    )
    parser.add_argument("--output-dir", default="data/debug/synonym_similarity")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    for model_name in args.models:
        run_one_model(
            model_name=model_name,
            output_dir=Path(args.output_dir),
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()