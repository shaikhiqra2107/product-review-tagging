#!/usr/bin/env python3
"""
product_review_tagging_pipeline_full.py

Rule-based product review tagging pipeline (command-line tool).

Features:
 - Load reviews from CSV (auto-detect review column: review/review_text/text/comment/content)
 - Preprocess text (lowercase, remove punctuation/emojis, simple stemming)
 - Load tag dictionary (JSON mapping tag -> list of phrases) or use built-in defaults
 - Pattern-based matching using regex and optional spaCy PhraseMatcher (if spaCy is installed)
 - Basic negation handling (window-based + optional dependency-based using spaCy)
 - Multi-label tagging (a review may receive multiple tags)
 - Export tagged CSV, produce tag-frequency PNG, optional JSON report
 - Optional small evaluation against gold CSV (column: gold/gold_tags/labels/tags)
 - Can be used on up to 1000 scraped reviews (or more) — use --maxrows to limit

Usage example:
  python product_review_tagging_pipeline_full.py --input reviews.csv --output tagged.csv --chart tag_freq.png --report summary.json --tagdict my_tag_dict.json --maxrows 1000

Requirements:
  Python 3.8+
  pip install pandas matplotlib
  (optional) pip install spacy
    python -m spacy download en_core_web_sm
"""

import argparse, json, os, re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# spaCy optional
USE_SPACY = False
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    USE_SPACY = True
except Exception:
    USE_SPACY = False

# Basic negation tokens (extend as needed)
NEGATION_WORDS = {"not", "no", "never", "n't", "dont", "don't", "didnt", "didn't", "none", "without", "hardly", "rarely"}

# ---------------------------
# Preprocessing utilities
# ---------------------------
def remove_emojis_and_symbols(text: str) -> str:
    """Remove URLs and non-word symbols but keep hyphens and apostrophes."""
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"http\S+|www\.\S+", " ", s)  # remove URLs
    s = re.sub(r"[^\w\s\-\']", " ", s)      # keep letters, digits, whitespace, hyphen, apostrophe
    s = re.sub(r"\s+", " ", s).strip()
    return s

def simple_stem(word: str) -> str:
    """Lightweight rule-based stemmer to increase phrase matching robustness."""
    for suf in ("ing", "ly", "ed", "es", "s", "ment"):
        if word.endswith(suf) and len(word) - len(suf) > 2:
            return word[:-len(suf)]
    return word

def preprocess_text(text: str, do_stem: bool = True) -> str:
    """Lowercase, remove emojis/symbols and optionally stem tokens."""
    txt = "" if text is None else str(text)
    txt = txt.lower()
    txt = remove_emojis_and_symbols(txt)
    tokens = txt.split()
    if do_stem:
        tokens = [simple_stem(t) for t in tokens]
    return " ".join(tokens)

# ---------------------------
# Tag dictionary (default)
# ---------------------------
DEFAULT_TAG_DICT = {
    "Late Delivery": ["delayed", "came late", "took too long", "late delivery", "delivery was late", "delivery delay", "shipment delayed", "arrived late", "not delivered on time"],
    "Fast Delivery": ["fast delivery", "delivered quickly", "came early", "arrived quickly", "quick delivery", "delivered fast", "on time delivery", "on time"],
    "Battery Issue": ["battery drains", "poor battery", "doesnt last", "doesn't last", "battery problem", "battery issue", "battery heating", "battery died", "battery backup poor", "battery backup"],
    "Good Packaging": ["good packaging", "well packed", "properly packed", "packed well", "excellent packaging", "nice packaging", "package was sealed", "box was sealed"],
    "Bad Sound Quality": ["bad sound quality", "poor sound", "distorted sound", "sound is bad", "low volume", "muffled sound", "tinny sound", "no bass", "sound issue"]
}

def load_tag_dict(path: str = None) -> dict:
    """Load tag dictionary from JSON, or return the default dictionary."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf8") as f:
            raw = json.load(f)
        return {str(k): [str(p) for p in v] for k, v in raw.items()}
    return DEFAULT_TAG_DICT.copy()

def save_tag_dict(tag_dict: dict, path: str):
    with open(path, "w", encoding="utf8") as f:
        json.dump(tag_dict, f, indent=2, ensure_ascii=False)

# ---------------------------
# Pattern construction
# ---------------------------
def build_regex_patterns(tag_dict: dict):
    patterns = {}
    for tag, phrases in tag_dict.items():
        esc = []
        for p in phrases:
            s = str(p).lower()
            s = re.sub(r"[^\w\s]", " ", s)        # remove punctuation for matching flexibility
            s = re.sub(r"\s+", r"\\s+", s.strip())  # allow flexible whitespace
            esc.append(s)
        if esc:
            patt = r"(" + r"|".join(esc) + r")"
            patterns[tag] = re.compile(patt, flags=re.IGNORECASE)
        else:
            patterns[tag] = re.compile(r"$^")
    return patterns

def setup_spacy_phrase_matcher(tag_dict: dict):
    """Return (nlp, matcher) if spaCy available, otherwise (None, None)."""
    if not USE_SPACY:
        return None, None
    try:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except Exception:
            nlp = spacy.blank("en")
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        for tag, phrases in tag_dict.items():
            docs = [nlp.make_doc(p) for p in phrases]
            matcher.add(tag, docs)
        return nlp, matcher
    except Exception:
        return None, None

# ---------------------------
# Negation detection
# ---------------------------
def window_negation_check(preprocessed_text: str, match_start_char_idx: int, window_chars: int = 40) -> bool:
    """Return True if a negation token appears in a small character window before match."""
    start = max(0, match_start_char_idx - window_chars)
    window = preprocessed_text[start:match_start_char_idx]
    tokens = window.split()
    for w in tokens[-6:]:
        if w.strip(".,!?;:") in NEGATION_WORDS:
            return True
    return False

def dependency_negation_check_spacy(nlp, doc, match_start_token_idx):
    """Basic spaCy-based check of tokens/children for negation (if spaCy available)."""
    if nlp is None:
        return False
    start = max(0, match_start_token_idx - 4)
    for tok in doc[start:match_start_token_idx]:
        if tok.lower_ in NEGATION_WORDS:
            return True
        for child in tok.children:
            if child.lower_ in NEGATION_WORDS:
                return True
    return False

# ---------------------------
# Tagging logic
# ---------------------------
def tag_review(text: str, tag_patterns, spa_nlp=None, spa_matcher=None, do_negation_dep: bool = True):
    """Tag a single review text. Returns sorted list of tags (may be empty)."""
    cleaned = preprocess_text(text)
    tags = set()

    # 1) spaCy-based PhraseMatcher (if available)
    if USE_SPACY and spa_nlp and spa_matcher:
        try:
            doc = spa_nlp(cleaned)
            matches = spa_matcher(doc)
            for mid, start, end in matches:
                tag = spa_nlp.vocab.strings[mid]
                # dependency negation check (if enabled)
                if do_negation_dep and spa_nlp:
                    if dependency_negation_check_spacy(spa_nlp, doc, start):
                        continue
                start_char = doc[start].idx
                if window_negation_check(cleaned, start_char):
                    continue
                tags.add(tag)
        except Exception:
            pass

    # 2) regex patterns fallback / supplement
    for tag, patt in tag_patterns.items():
        for m in patt.finditer(cleaned):
            start_char = m.start()
            if window_negation_check(cleaned, start_char):
                continue
            tags.add(tag)

    return sorted(tags)

# ---------------------------
# Evaluation helper
# ---------------------------
def multilabel_metrics(gold_list, pred_list):
    """Compute micro-precision/recall/F1 for small gold sets."""
    tp = fp = fn = 0
    for g, p in zip(gold_list, pred_list):
        gset = set(g)
        pset = set(p)
        tp += len(gset & pset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

# ---------------------------
# Plotting & reporting
# ---------------------------
def plot_tag_frequencies(tag_counter: Counter, outpath: str, title: str = "Tag Frequency"):
    if not tag_counter:
        print("No tags found to plot.")
        return
    tags = list(tag_counter.keys())
    counts = [tag_counter[t] for t in tags]
    plt.figure(figsize=(8,5))
    plt.bar(tags, counts)
    plt.xlabel("Tag")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline(input_csv, output_csv, tagdict_path=None, chart_path=None, report_path=None, max_rows=None, gold_csv=None):
    df = pd.read_csv(input_csv, dtype=str)
    # detect review text column
    col_candidates = [c for c in df.columns if c.lower() in ("review", "review_text", "text", "comment", "content")]
    text_col = col_candidates[0] if col_candidates else df.columns[0]
    if max_rows and isinstance(max_rows, int):
        df = df.head(max_rows)

    tag_dict = load_tag_dict(tagdict_path)
    tag_patterns = build_regex_patterns(tag_dict)
    spa_nlp, spa_matcher = setup_spacy_phrase_matcher(tag_dict)

    preds = []
    cleaned_texts = []
    for _, row in df.iterrows():
        txt = row.get(text_col, "") if isinstance(row, pd.Series) else ""
        cleaned = preprocess_text(txt)
        cleaned_texts.append(cleaned)
        p = tag_review(txt, tag_patterns, spa_nlp, spa_matcher)
        preds.append(p)

    df["cleaned_review"] = cleaned_texts
    df["predicted_tags"] = preds
    df.to_csv(output_csv, index=False, encoding="utf8")

    all_tags = Counter([t for tags in preds for t in tags])
    if chart_path:
        plot_tag_frequencies(all_tags, chart_path)

    report = {"input_rows": len(df), "unique_tags_found": len(all_tags), "tag_counts": dict(all_tags)}

    # optional gold evaluation
    if gold_csv and os.path.exists(gold_csv):
        gold_df = pd.read_csv(gold_csv, dtype=str)
        gold_col = None
        for c in gold_df.columns:
            if c.lower() in ("gold", "gold_tags", "labels", "tags"):
                gold_col = c; break
        if gold_col is None:
            gold_col = gold_df.columns[-1]
        gold_labels = gold_df[gold_col].fillna("").apply(lambda x: [s.strip() for s in str(x).split(",") if s.strip()])
        pred_for_eval = preds[:len(gold_labels)]
        metrics = multilabel_metrics(gold_labels.tolist(), pred_for_eval)
        report["evaluation"] = metrics

    if report_path:
        with open(report_path, "w", encoding="utf8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return {"output_csv": output_csv, "chart": chart_path, "report": report, "tag_dict_used": tagdict_path or "DEFAULT"}

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Rule-based Product Review Tagging Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path (must contain a review text column)")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path (tagged reviews)")
    parser.add_argument("--tagdict", "-t", required=False, help="Path to tag dictionary JSON (optional)")
    parser.add_argument("--chart", "-c", required=False, help="Path to save tag frequency chart (PNG)")
    parser.add_argument("--report", "-r", required=False, help="Path to save JSON report summary")
    parser.add_argument("--maxrows", type=int, required=False, help="Process only first N rows (optional, e.g., 1000)")
    parser.add_argument("--gold", required=False, help="Optional gold CSV for small evaluation (column: gold/gold_tags/labels/tags)")
    args = parser.parse_args()

    out = run_pipeline(args.input, args.output, tagdict_path=args.tagdict, chart_path=args.chart, report_path=args.report, max_rows=args.maxrows, gold_csv=args.gold)
    print("Pipeline completed. Output:")
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()