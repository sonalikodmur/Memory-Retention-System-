import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
import os

NOW = datetime(2026, 7, 29)

FEATURE_COLS = [
    "days_since_created", "days_since_accessed", "feedback_encoded", "access_count_log",
    "type_event", "type_fact", "type_goal", "type_preference", "type_task"
]

MIN_WORDS_FOR_COMPRESS = 15
SCORE_LOW = 0.35
SCORE_HIGH = 0.70
SUMMARY_RATIO = 0.5
MIN_SUMMARY_WORDS = 5
MIN_REDUCTION_PERCENT = 10.0
MIN_INFORMATION_RETENTION = 0.40

LONG_VERBOSE_MEMORIES = [
    {
        "memory_id": "long_001",
        "text": "User mentioned during a long conversation last month that they're planning a trip to Goa in December with two friends, they're looking at hotels near Baga beach, budget is around 30k, and they want recommendations for water sports.",
        "type": "goal",
        "created_at": "2026-06-15",
        "last_accessed_at": "2026-07-10",
        "access_count": 3,
        "feedback": "none",
    },
    {
        "memory_id": "long_002",
        "text": "User said last Tuesday during the team sync that they prefer using the dark theme on all their applications because it reduces eye strain, especially when working late at night on coding projects and reading documentation.",
        "type": "preference",
        "created_at": "2026-06-20",
        "last_accessed_at": "2026-07-12",
        "access_count": 4,
        "feedback": "positive",
    },
    {
        "memory_id": "long_003",
        "text": "Last week the user talked about their weekend hiking trip where they went to the mountains with their family, saw some beautiful waterfalls, had a picnic lunch by the river, and got back home really tired but happy on Sunday evening.",
        "type": "event",
        "created_at": "2026-07-15",
        "last_accessed_at": "2026-07-18",
        "access_count": 1,
        "feedback": "none",
    },
    {
        "memory_id": "long_004",
        "text": "The user is currently working on upgrading the backend infrastructure by migrating all the microservices from the old monolithic repository running on Python 2.7 to a new containerized Kubernetes cluster using Docker and FastAPI.",
        "type": "task",
        "created_at": "2026-05-01",
        "last_accessed_at": "2026-07-20",
        "access_count": 8,
        "feedback": "positive",
    },
    {
        "memory_id": "long_005",
        "text": "User's workstation setup consists of a 16-inch MacBook Pro with 32GB RAM connected to a 4K ultrawide external monitor, a mechanical keyboard with Cherry MX Brown switches, and a vertical ergonomic mouse to prevent wrist pain.",
        "type": "fact",
        "created_at": "2026-04-10",
        "last_accessed_at": "2026-06-01",
        "access_count": 2,
        "feedback": "none",
    },
    {
        "memory_id": "long_006",
        "text": "User mentioned they want to learn Rust programming language over the next six months by taking an online course, building three small CLI tools, and contributing to at least two open source projects to get real-world experience before switching jobs.",
        "type": "goal",
        "created_at": "2026-06-01",
        "last_accessed_at": "2026-07-15",
        "access_count": 5,
        "feedback": "positive",
    },
    {
        "memory_id": "long_007",
        "text": "During yesterday's onboarding call, the new intern Sam mentioned he lives in Brooklyn, recently graduated from NYU with a degree in Computer Science, likes playing basketball on weekends, and is excited to work on the data pipeline team this summer.",
        "type": "fact",
        "created_at": "2026-07-25",
        "last_accessed_at": "2026-07-26",
        "access_count": 1,
        "feedback": "none",
    },
    {
        "memory_id": "long_008",
        "text": "User is organizing a team offsite for mid-September, they're looking at venues in upstate New York that can accommodate 25 people overnight, have conference rooms for workshops, and offer team activities like hiking, kayaking, or a ropes course.",
        "type": "task",
        "created_at": "2026-07-01",
        "last_accessed_at": "2026-07-22",
        "access_count": 6,
        "feedback": "positive",
    },
    {
        "memory_id": "long_009",
        "text": "The user does not drink coffee or any caffeinated beverages after 2 pm because they find it affects their sleep quality, makes them wake up frequently during the night, and leaves them feeling groggy and unproductive the entire next day.",
        "type": "preference",
        "created_at": "2026-03-15",
        "last_accessed_at": "2026-05-10",
        "access_count": 1,
        "feedback": "none",
    },
    {
        "memory_id": "long_010",
        "text": "User completed their three-year professional certification in cloud architecture last week after passing the final exam on the second attempt, they studied for about four hours every weekend for six months and are now eligible for a promotion and salary bump.",
        "type": "event",
        "created_at": "2026-07-28",
        "last_accessed_at": "2026-07-29",
        "access_count": 2,
        "feedback": "positive",
    },
    {
        "memory_id": "long_011",
        "text": "User told me last Friday they usually order lunch from the Thai place around the corner on Wednesdays, they get the green curry with extra vegetables and tofu, spicy level 3 out of 5, and a side of sticky rice to share with their desk mate.",
        "type": "preference",
        "created_at": "2026-05-20",
        "last_accessed_at": "2026-06-15",
        "access_count": 2,
        "feedback": "none",
    },
    {
        "memory_id": "long_012",
        "text": "The database migration scheduled for this weekend involves backing up 4 terabytes of production customer data, upgrading the PostgreSQL engine from version 12 to 15, rebuilding all indexes for performance, and testing failover and rollback procedures before Monday morning.",
        "type": "task",
        "created_at": "2026-07-10",
        "last_accessed_at": "2026-07-27",
        "access_count": 7,
        "feedback": "none",
    },
]


def build_features_for_memory(row):
    features = {col: 0 for col in FEATURE_COLS}
    created = pd.to_datetime(row["created_at"])
    accessed = pd.to_datetime(row["last_accessed_at"])
    features["days_since_created"] = (NOW - created).days
    features["days_since_accessed"] = (NOW - accessed).days
    feedback_map = {"positive": 1, "none": 0, "negative": -1}
    features["feedback_encoded"] = feedback_map.get(row["feedback"], 0)
    features["access_count_log"] = np.log1p(row["access_count"])
    features[f"type_{row['type']}"] = 1
    return pd.DataFrame([features])[FEATURE_COLS]


def load_and_score_all_memories():
    df = pd.read_csv("data/raw/synthetic_memories.csv")
    verbose_df = pd.DataFrame(LONG_VERBOSE_MEMORIES)
    combined = pd.concat([df, verbose_df], ignore_index=True)

    model = joblib.load("src/model/saved_models/retention_model.pkl")
    scores = []
    for _, row in combined.iterrows():
        feat = build_features_for_memory(row)
        scores.append(model.predict(feat)[0])
    combined["retention_score"] = scores
    combined["retention_score"] = combined["retention_score"].round(3)
    return combined


def word_count(text):
    return len(str(text).split())


def split_clauses(text):
    clauses = []
    buffer = ""
    for ch in text:
        if ch in (",", ".", "!", "?"):
            buffer += ch
            clauses.append(buffer.strip())
            buffer = ""
        elif ch == ";" or (ch == ":" and len(buffer.split()) > 5):
            clauses.append(buffer.strip())
            buffer = ""
        else:
            buffer += ch
    if buffer.strip():
        clauses.append(buffer.strip())

    split_result = []
    for c in clauses:
        cleaned = c.rstrip(".,!?;: ").strip()
        if not cleaned:
            continue
        for conj in [" and they", " and we", " and he", " and she", " and it",
                     " and the", " and is", " and are", " and have", " and has",
                     " and want", " and now", " and got", " and offer"]:
            idx = cleaned.lower().find(conj)
            if idx > 10 and len(cleaned) - idx > 10:
                split_result.append(cleaned[:idx].rstrip(" ,;:"))
                cleaned = cleaned[idx + 1:].strip()
        if cleaned:
            split_result.append(cleaned)
    return split_result if split_result else [text.strip()]


def score_chunk(chunk, vectorizer):
    try:
        vec = vectorizer.transform([chunk])
        return vec.sum()
    except Exception:
        return len(chunk.split())


def chunk_information_density(chunk, vectorizer, position_bonus=1.0):
    base = score_chunk(chunk, vectorizer)
    text = chunk
    digits = len(re.findall(r'\d+', text))
    capitals = len(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text))
    numbers_penalty = 1.0 + 0.35 * digits
    proper_penalty = 1.0 + 0.3 * capitals
    wc = max(1, word_count(text))
    mentions_subject = bool(re.search(r'\b(User|user|They|they)\b', text))
    subject_bonus = 1.25 if mentions_subject else 1.0
    score = (base / wc ** 0.5) * numbers_penalty * proper_penalty * subject_bonus * position_bonus
    return score


def extractive_summarize(text, corpus_texts, target_ratio=SUMMARY_RATIO, min_words=MIN_SUMMARY_WORDS):
    original_words = str(text).split()
    original_len = len(original_words)
    if original_len <= min_words:
        return text

    target_len = max(min_words, int(original_len * target_ratio))

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectorizer.fit(corpus_texts)
    except Exception:
        return text

    if len(sentences) > 1:
        try:
            sent_scores = []
            for i, sent in enumerate(sentences):
                pos_bonus = 1.15 if i == 0 else 1.0
                score = chunk_information_density(sent, vectorizer, pos_bonus)
                sent_scores.append((score, sent, i))
            sent_scores.sort(reverse=True, key=lambda x: x[0])
            top_indices = []
            word_total = 0
            for _, sent, idx in sent_scores:
                sent_words = word_count(sent)
                if word_total + sent_words <= int(target_len * 1.2) or not top_indices:
                    top_indices.append(idx)
                    word_total += sent_words
                if word_total >= target_len:
                    break
            top_indices.sort()
            ordered_top = [sentences[i] for i in top_indices]
            result = " ".join(ordered_top)
            result_words = word_count(result)
            if min_words <= result_words:
                return result
        except Exception:
            pass

    clauses = split_clauses(text)
    if len(clauses) > 1:
        try:
            clause_scores = []
            for i, cl in enumerate(clauses):
                pos_bonus = 1.55 if i == 0 else 1.0
                score = chunk_information_density(cl, vectorizer, pos_bonus)
                clause_scores.append((score, cl, i))
            clause_scores.sort(reverse=True, key=lambda x: x[0])
            kept_indices = []
            word_total = 0
            budget = target_len + 5
            for _, cl, idx in clause_scores:
                cl_words = word_count(cl)
                if word_total + cl_words <= budget or idx == 0 or not kept_indices:
                    kept_indices.append(idx)
                    word_total += cl_words
                if word_total >= target_len and len(kept_indices) >= 2:
                    break
            if 0 not in kept_indices:
                kept_indices.append(0)
                kept_indices = sorted(set(kept_indices))
                kept_indices = kept_indices[:4]
            kept_indices.sort()
            ordered_kept = [clauses[i] for i in kept_indices]
            cleaned = [c.rstrip(".,!?;: ").strip() for c in ordered_kept if c.rstrip(".,!?;: ").strip()]
            if cleaned:
                result = ", ".join(cleaned)
                if not result.endswith("."):
                    result += "."
                result_words = word_count(result)
                if min_words <= result_words:
                    return result
        except Exception:
            pass

    try:
        vocab = vectorizer.vocabulary_
        idf = vectorizer.idf_
    except Exception:
        return text

    scored_words = []
    for i, w in enumerate(original_words):
        clean = re.sub(r'[^\w\s]', '', w.lower())
        if clean in vocab:
            score = idf[vocab[clean]]
        else:
            score = 0.1
        is_stop = len(clean) <= 3
        if is_stop and score < 1.0:
            score *= 0.3
        if i < 3:
            score += 0.5
        scored_words.append((i, w, score))

    scored_words.sort(key=lambda x: -x[2])
    keep_indices = set()
    for idx, w, s in scored_words[:target_len]:
        keep_indices.add(idx)

    result_words = []
    for i, w in enumerate(original_words):
        if i in keep_indices:
            result_words.append(w)

    if len(result_words) < min_words:
        for idx, w, s in scored_words:
            if idx not in keep_indices:
                keep_indices.add(idx)
                if len(keep_indices) >= min(min_words, original_len):
                    break
        result_words = [original_words[i] for i in sorted(keep_indices)]

    return " ".join(result_words)



def important_terms(text, vectorizer, top_n=20):
    """Return the most important TF-IDF terms in a memory."""
    try:
        vector = vectorizer.transform([str(text)])
        scores = vector.toarray()[0]
        terms = vectorizer.get_feature_names_out()
        indices = np.argsort(scores)[::-1]

        result = []
        for idx in indices:
            if scores[idx] <= 0:
                continue
            result.append(terms[idx])
            if len(result) >= top_n:
                break
        return set(result)
    except Exception:
        return set()


def calculate_information_retention(original, compressed, vectorizer):
    """Measure how many important original terms survive compression."""
    original_terms = important_terms(original, vectorizer)
    compressed_terms = important_terms(compressed, vectorizer)

    if not original_terms:
        return 1.0

    return len(original_terms & compressed_terms) / len(original_terms)


def load_previous_compressions(out_path="data/processed/compressed_memories.csv"):
    """Prevent the same memory from being compressed repeatedly."""
    if not os.path.exists(out_path):
        return set()

    try:
        previous = pd.read_csv(out_path)
        if "memory_id" not in previous.columns:
            return set()
        return set(previous["memory_id"].dropna().astype(str))
    except Exception:
        return set()


def needs_compression(row):
    wc = word_count(row["text"])
    score = row["retention_score"]
    already = row.get("is_compressed_before", False)
    return wc >= MIN_WORDS_FOR_COMPRESS and SCORE_LOW <= score <= SCORE_HIGH and not already


def run_compression_pipeline():
    print("=== COMPRESSION PIPELINE ===")
    print(f"\nStep 1: Scoring all memories with ML model...")
    scored = load_and_score_all_memories()
    print(f"  Total memories scored: {len(scored)}")
    print(f"  Score distribution: min={scored['retention_score'].min():.3f}, "
          f"mean={scored['retention_score'].mean():.3f}, max={scored['retention_score'].max():.3f}")

    scored["word_count"] = scored["text"].apply(word_count)
    previous_ids = load_previous_compressions()
    scored["is_compressed_before"] = scored["memory_id"].astype(str).isin(previous_ids)

    print(f"\nStep 2: Identifying compression candidates...")
    print(f"  Condition 1 (>= {MIN_WORDS_FOR_COMPRESS} words): {(scored['word_count'] >= MIN_WORDS_FOR_COMPRESS).sum()}")
    print(f"  Condition 2 (score in [{SCORE_LOW}, {SCORE_HIGH}]): "
          f"{((scored['retention_score'] >= SCORE_LOW) & (scored['retention_score'] <= SCORE_HIGH)).sum()}")

    candidates = scored[scored.apply(needs_compression, axis=1)].copy()
    print(f"  Compression candidates (both conditions): {len(candidates)}")

    if len(candidates) == 0:
        print("\n  No candidates found. Printing verbose-memory stats to show filters in action:")
        verbose_ids = [m["memory_id"] for m in LONG_VERBOSE_MEMORIES]
        verbose_rows = scored[scored["memory_id"].isin(verbose_ids)]
        for _, r in verbose_rows.iterrows():
            print(f"    id={r['memory_id']} wc={r['word_count']} score={r['retention_score']:.3f} "
                  f"type={r['type']} -> compress={needs_compression(r)}")

    corpus_texts = scored["text"].tolist()

    print(f"\nStep 3: Running extractive summarization on {len(candidates)} candidates...")

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        vectorizer.fit(corpus_texts)
    except Exception:
        vectorizer = None

    results = []

    for _, row in candidates.iterrows():
        original = row["text"]
        compressed = extractive_summarize(original, corpus_texts)

        original_wc = word_count(original)
        compressed_wc = word_count(compressed)

        reduction_pct = (
            (1 - compressed_wc / original_wc) * 100
            if original_wc else 0
        )

        if vectorizer is not None:
            information_retention = calculate_information_retention(
                original, compressed, vectorizer
            )
        else:
            information_retention = 0.0

        quality_pass = (
            compressed_wc >= MIN_SUMMARY_WORDS
            and compressed_wc < original_wc
            and reduction_pct >= MIN_REDUCTION_PERCENT
            and information_retention >= MIN_INFORMATION_RETENTION
        )

        action = "COMPRESS" if quality_pass else "KEEP"
        final_text = compressed if quality_pass else original
        final_wc = word_count(final_text)

        results.append({
            "memory_id": row["memory_id"],
            "type": row["type"],
            "original_text": original,
            "compressed_text": final_text,
            "retention_score": row["retention_score"],
            "original_word_count": original_wc,
            "compressed_word_count": final_wc,
            "compression_ratio": round(final_wc / original_wc, 3) if original_wc else 1.0,
            "reduction_percent": round((1 - final_wc / original_wc) * 100, 2) if original_wc else 0,
            "information_retention": round(information_retention, 3),
            "quality_check": "PASS" if quality_pass else "FAIL",
            "action_taken": action,
            "action_reason": (
                "LONG_AND_MEDIUM_RETENTION"
                if quality_pass
                else "COMPRESSION_QUALITY_CHECK_FAILED"
            ),
            "compressed_on": NOW.strftime("%Y-%m-%d") if quality_pass else "",
            "created_at": row["created_at"],
            "last_accessed_at": row["last_accessed_at"],
            "access_count": row["access_count"],
            "feedback": row["feedback"],
        })

    compressed_df = pd.DataFrame(results)

    out_path = "data/processed/compressed_memories.csv"
    compressed_df.to_csv(out_path, index=False)
    print(f"\nStep 4: Saved results to {out_path} ({len(compressed_df)} rows)")

    compressed_only = compressed_df[
        compressed_df["action_taken"] == "COMPRESS"
    ]

    total_compressed = len(compressed_only)
    total_quality_failed = len(compressed_df) - total_compressed

    if total_compressed > 0:
        avg_original = compressed_only["original_word_count"].mean()
        avg_compressed = compressed_only["compressed_word_count"].mean()
        reduction_pct = compressed_only["reduction_percent"].mean()
        avg_information_retention = compressed_only["information_retention"].mean()
    else:
        avg_original = avg_compressed = reduction_pct = 0
        avg_information_retention = 0

    print(f"\nStep 5: Summary statistics")
    print(f"  Total candidates: {len(compressed_df)}")
    print(f"  Successfully compressed: {total_compressed}")
    print(f"  Failed quality check: {total_quality_failed}")
    print(f"  Average original length: {avg_original:.1f} words")
    print(f"  Average compressed length: {avg_compressed:.1f} words")
    print(f"  Average reduction: {reduction_pct:.1f}%")
    print(f"  Average information retention: {avg_information_retention:.3f}")

    report = pd.DataFrame([{
        "run_date": NOW.strftime("%Y-%m-%d"),
        "total_memories_scored": len(scored),
        "compression_candidates": len(compressed_df),
        "memories_compressed": total_compressed,
        "quality_check_failures": total_quality_failed,
        "average_original_words": round(avg_original, 2),
        "average_compressed_words": round(avg_compressed, 2),
        "average_reduction_percent": round(reduction_pct, 2),
        "average_information_retention": round(avg_information_retention, 3),
        "minimum_words_threshold": MIN_WORDS_FOR_COMPRESS,
        "score_low": SCORE_LOW,
        "score_high": SCORE_HIGH,
    }])

    report_path = "data/processed/compression_report.csv"
    report.to_csv(report_path, index=False)
    print(f"  Evaluation report: {report_path}")

        print(f"\nStep 6: Manual sanity check — sample compressions:")
        sample = compressed_df.head(min(15, len(compressed_df)))
        for i, (_, r) in enumerate(sample.iterrows(), 1):
            print(f"\n  --- Example {i} (id={r['memory_id']}, score={r['retention_score']:.3f}, "
                  f"{r['original_word_count']} -> {r['compressed_word_count']} words) ---")
            print(f"  BEFORE: {r['original_text']}")
            print(f"  AFTER:  {r['compressed_text']}")

    return compressed_df, report


if __name__ == "__main__":
    run_compression_pipeline()
