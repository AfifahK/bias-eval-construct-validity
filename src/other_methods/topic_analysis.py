"""
Topic and thematic analysis of LLM explanation responses.
  - LDA topic modeling (8 topics) with TF-IDF
  - Thematic keyword frequency analysis
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ── Load data ───────────────────────────────────────────────
df = pd.read_csv("../../data/all_model_responses_clean.csv")
exp = df[(df["step"] == "explanation") & (df["is_refusal"] == False)].copy()
print(f"Non-refusal explanation rows: {len(exp)}")


# ── Preprocessing ───────────────────────────────────────────
def clean_for_lda(text):
    text = str(text)
    text = re.sub(r"^\s*[0-3]\s*\n+", "", text)
    text = re.sub(r"^\s*Explanation:\s*\n*", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


exp["text_clean"] = exp["response"].apply(clean_for_lda)

# ── TF-IDF + LDA ───────────────────────────────────────────
N_TOPICS = 8
N_TOP_WORDS = 15

tfidf = TfidfVectorizer(
    max_df=0.90,
    min_df=3,
    stop_words="english",
    max_features=3000,
)
tfidf_matrix = tfidf.fit_transform(exp["text_clean"])
feature_names = tfidf.get_feature_names_out()
print(f"TF-IDF matrix: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} features")

lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=42,
    max_iter=30,
    learning_method="batch",
)
doc_topics = lda.fit_transform(tfidf_matrix)

# ── Print top words per topic ───────────────────────────────
print(f"\n{'='*60}")
print(f"  TOP {N_TOP_WORDS} WORDS PER TOPIC ({N_TOPICS} topics)")
print(f"{'='*60}")
for idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-N_TOP_WORDS:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    print(f"\nTopic {idx}: {', '.join(top_words)}")

# ── Assign dominant topic ──────────────────────────────────
exp["dominant_topic"] = doc_topics.argmax(axis=1)
# Also store topic probabilities
for t in range(N_TOPICS):
    exp[f"topic_{t}_prob"] = doc_topics[:, t].round(4)

# ── Merge bias labels from dict and judge ──────────────────
dict_scores = pd.read_csv("../../scores/dict_scores.csv")
dict_scores = dict_scores[dict_scores["step"] == "explanation"][
    ["prompt_id", "model", "scale", "bias_label"]
].rename(columns={"bias_label": "dict_bias_label"})

judge_scores = pd.read_csv("../../scores/llm_judge_scores.csv")
judge_scores = judge_scores[judge_scores["step"] == "explanation"][
    ["prompt_id", "model", "scale", "bias_label"]
].rename(columns={"bias_label": "judge_bias_label"})
judge_scores = judge_scores[judge_scores["judge_bias_label"] != -1]

merge_keys = ["prompt_id", "model", "scale"]
exp = exp.merge(dict_scores, on=merge_keys, how="left")
exp = exp.merge(judge_scores, on=merge_keys, how="left")

# ── Cross-tabulations ──────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TOPIC × DEMOGRAPHIC CROSS-TABULATIONS")
print(f"{'='*60}")

for label, col in [("model", "model"), ("nationality", "nationality"),
                    ("disability_type", "disability_type"), ("scenario", "scenario")]:
    ct = pd.crosstab(exp["dominant_topic"], exp[col])
    print(f"\n--- Topic × {label} ---")
    print(ct.to_string())

print(f"\n--- Topic × dict_bias_label ---")
print(pd.crosstab(exp["dominant_topic"], exp["dict_bias_label"]).to_string())

print(f"\n--- Topic × judge_bias_label ---")
ct_judge = pd.crosstab(exp["dominant_topic"], exp["judge_bias_label"])
print(ct_judge.to_string())

# Topic distribution summary
print(f"\n--- Topic distribution ---")
print(exp["dominant_topic"].value_counts().sort_index().to_string())

# ── Save topic assignments ─────────────────────────────────
topic_cols = ["prompt_id", "nationality", "disability", "disability_type",
              "scenario", "model", "scale", "dominant_topic"] + \
             [f"topic_{t}_prob" for t in range(N_TOPICS)] + \
             ["dict_bias_label", "judge_bias_label"]
exp[topic_cols].to_csv("../../scores/topic_assignments.csv", index=False)
print(f"\nSaved topic_assignments.csv ({len(exp)} rows)")


# ══════════════════════════════════════════════════════════════
#  THEMATIC KEYWORD FREQUENCY ANALYSIS
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  THEMATIC KEYWORD FREQUENCY ANALYSIS")
print(f"{'='*60}")

THEMES = {
    "risk_danger":    ["risk", "danger", "safety", "threat"],
    "compassion":     ["compassion", "empathy", "understanding", "support"],
    "stigma":         ["stigma", "stereotype", "prejudice", "discrimination"],
    "capability":     ["capability", "competent", "qualified", "able"],
    "professional":   ["professional", "workplace", "job", "career"],
    "caution":        ["caution", "concern", "careful", "worry"],
}


def count_theme(text, keywords):
    text_lower = str(text).lower()
    tokens = re.findall(r"\b[a-z]+\b", text_lower)
    return sum(1 for t in tokens if t in keywords)


# Reload clean exp (without the merged bias cols complicating things)
exp_theme = df[(df["step"] == "explanation") & (df["is_refusal"] == False)].copy()

for theme_name, keywords in THEMES.items():
    exp_theme[theme_name] = exp_theme["response"].apply(lambda t: count_theme(t, keywords))

# Overall theme frequencies
print("\n--- Overall theme totals ---")
for theme_name in THEMES:
    total = exp_theme[theme_name].sum()
    present = (exp_theme[theme_name] > 0).sum()
    print(f"  {theme_name:<18} total={total:>4}  present_in={present:>3}/{len(exp_theme)} responses "
          f"({present/len(exp_theme):.1%})")

# Breakdowns
for label, col in [("model", "model"), ("nationality", "nationality"),
                    ("disability_type", "disability_type"), ("scenario", "scenario")]:
    print(f"\n--- Mean theme counts by {label} ---")
    agg = exp_theme.groupby(col)[list(THEMES.keys())].mean().round(3)
    print(agg.to_string())

# Merge bias labels for theme × bias crosstab
exp_theme = exp_theme.merge(dict_scores, on=merge_keys, how="left")
exp_theme = exp_theme.merge(judge_scores, on=merge_keys, how="left")

print(f"\n--- Mean theme counts by dict_bias_label ---")
print(exp_theme.groupby("dict_bias_label")[list(THEMES.keys())].mean().round(3).to_string())

print(f"\n--- Mean theme counts by judge_bias_label ---")
print(exp_theme.groupby("judge_bias_label")[list(THEMES.keys())].mean().round(3).to_string())

# Save
theme_cols = ["prompt_id", "nationality", "disability", "disability_type",
              "scenario", "model", "scale"] + list(THEMES.keys())
exp_theme[theme_cols].to_csv("../../scores/theme_frequencies.csv", index=False)
print(f"\nSaved theme_frequencies.csv ({len(exp_theme)} rows)")
