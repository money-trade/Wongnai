"""Wongnai review rating prediction with Naive Bayes (word table -> word cloud -> model -> heat map)."""
import os
import re
from collections import Counter
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from wordcloud import WordCloud

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, "review_dataset")
TRAIN = os.environ.get("WONGNAI_TRAIN", os.path.join(DATA_DIR, "w_review_train.csv"))
TEST_FILE = os.path.join(DATA_DIR, "test_file.csv")
# Thai-capable font for the word cloud / heat map (first one found is used)
FONT_CANDIDATES = [r"C:\Windows\Fonts\tahoma.ttf",
                   "/System/Library/Fonts/Supplemental/Thonburi.ttc",
                   "/Library/Fonts/Thonburi.ttc",
                   "/usr/share/fonts/opentype/tlwg/Loma.otf",
                   "/usr/share/fonts/truetype/tlwg/Loma.ttf",
                   "/usr/share/fonts/truetype/tlwg/Garuda.ttf"]
FONT = os.environ.get("WONGNAI_FONT") or next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)
FONT_PROP = font_manager.FontProperties(fname=FONT) if FONT else None

# model settings (chosen on the validation split)
ALPHA = 0.5          # Laplace smoothing
MIN_COUNT = 5        # drop features seen fewer times than this in training
OBJECTIVE = "macro_f1"  # what the per-star bias is tuned for: "macro_f1" or "accuracy"
USE_BIGRAMS = True   # add word pairs such as "ไม่_อร่อย" to the model features

# Stop words are removed for the word table / pictures, but sentiment-bearing ones (negation / intensity) are kept
KEEP = {"ไม่", "มาก", "น้อย", "สุด", "ที่สุด", "เกิน", "จริง", "เลย"}
STOP = set(thai_stopwords()) - KEEP
# extra filler words hidden from the word clouds only
CLOUD_STOP = {"ร้าน", "ๆ", "ไม่", "มี", "มา", "ไป", "ก็", "ที่", "ได้", "เป็น", "แต่", "นี้", "จะ",
              "ว่า", "แล้ว", "กับ", "และ", "ให้", "อยู่", "ของ", "ใน", "กิน", "ทาน", "เมนู",
              "อาหาร", "ผม", "เรา", "อีก", "หรือ", "คน", "ดู", "ทาง", "นั่ง", "สั่ง"}
THAI_OR_LATIN = re.compile(r"[\u0E00-\u0E7Fa-zA-Z0-9]")


def tokenize(text):
    """All Thai/Latin/digit tokens, stop words included (the model uses them)."""
    toks = word_tokenize(str(text).lower(), engine="newmm", keep_whitespace=False)
    return [t for t in toks if THAI_OR_LATIN.search(t)]


def features(toks):
    """Set of unigrams + bigrams: each feature counts once per review."""
    feats = set(toks)
    if USE_BIGRAMS:
        feats.update(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
    return feats


def stratified_split(labels, stars, frac, seed):
    """Boolean mask that picks `frac` of each star at random."""
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.permutation(np.where(labels == s)[0])[: int(frac * (labels == s).sum())]
                          for s in stars])
    mask = np.zeros(len(labels), dtype=bool)
    mask[idx] = True
    return mask


def train_nb(token_lists, labels, stars):
    """Count features per star and return (feature -> row, log P(feature | star) matrix)."""
    counts = {s: Counter() for s in stars}
    for toks, y in zip(token_lists, labels):
        counts[y].update(toks)
    total = Counter()
    for c in counts.values():
        total.update(c)
    vocab = [f for f, c in total.items() if c >= MIN_COUNT]
    m = np.array([[counts[s][w] for s in stars] for w in vocab], dtype=float)
    log_like = np.log((m + ALPHA) / (m.sum(axis=0) + ALPHA * len(vocab)))
    return {w: i for i, w in enumerate(vocab)}, log_like


def nb_scores(token_lists, row, log_like):
    """Average log P(feature | star) of each review (unknown features are skipped)."""
    out = np.zeros((len(token_lists), log_like.shape[1]))
    for i, toks in enumerate(token_lists):
        ix = [row[w] for w in toks if w in row]
        if ix:
            out[i] = log_like[ix].mean(axis=0)
    return out


def metrics(actual, pred, stars):
    """Confusion matrix (rows = actual, columns = predicted) and per-star precision / recall / F1."""
    k = len(stars)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (np.searchsorted(stars, actual), np.searchsorted(stars, pred)), 1)
    tp, support = np.diag(cm), cm.sum(axis=1)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(k), where=cm.sum(axis=0) > 0)
    recall = tp / support
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=(precision + recall) > 0)
    return cm, precision, recall, f1, support, tp.sum() / cm.sum()


def tune_bias(scores, actual, stars):
    """Coordinate search for one bias per star (it replaces the log prior)."""
    stars = np.array(stars)
    def objective(b):
        _, _, _, f1, _, acc = metrics(actual, stars[np.argmax(scores + b, axis=1)], stars)
        return f1.mean() if OBJECTIVE == "macro_f1" else acc
    bias = np.zeros(len(stars))
    grid = np.linspace(-1, 1, 161)
    for _ in range(3):
        for k in range(len(stars)):
            vals = []
            for v in grid:
                b = bias.copy()
                b[k] = v
                vals.append(objective(b))
            bias[k] = grid[int(np.argmax(vals))]
    return bias


def main():
    # 1) load data: no header, ';' separated, last column = star rating
    df = pd.read_csv(TRAIN, sep=";", header=None, names=["review", "rating"], engine="python")
    df = df.dropna().astype({"rating": int})
    stars = [int(s) for s in sorted(df.rating.unique())]
    print(f"Loaded df: {len(df)} reviews, stars = {stars}")

    with Pool() as p:
        df["all_tokens"] = p.map(tokenize, df["review"], chunksize=200)
    df["tokens"] = [[t for t in toks if t not in STOP] for toks in df.all_tokens]  # for table/pictures
    df["feats"] = [features(toks) for toks in df.all_tokens]                         # for the model

    # 2) split 80/20 per star, then count words in the train part only
    ratings = df.rating.values
    is_test = stratified_split(ratings, stars, 0.2, seed=42)
    train, test = df[~is_test], df[is_test]

    # word table: rows = words, columns = counts per star + P(word | star) with Laplace smoothing
    counts = {s: Counter(w for toks in train.tokens[train.rating == s] for w in toks) for s in stars}
    table = pd.DataFrame(counts).fillna(0).astype(int)
    table.columns = [f"star_{s}" for s in stars]
    table["total"] = table.sum(axis=1)
    for s in stars:
        table[f"P(word|{s})"] = (table[f"star_{s}"] + 1) / (table[f"star_{s}"].sum() + len(table))
    table = table.sort_values("total", ascending=False).rename_axis("word")
    table.to_csv(os.path.join(OUT_DIR, "word_table_by_star.csv"), encoding="utf-8-sig")

    # 3) word count per star -> console
    print(f"\nWord table: {len(table)} unique words")
    print("Total words per star:")
    print(table[[f"star_{s}" for s in stars]].sum().to_string())
    for s in stars:
        print(f"\nTop 15 words for {s} star:")
        print(table[f"star_{s}"].nlargest(15).to_string())

    # 4) word clouds: one window, one image per star
    fig, axes = plt.subplots(1, len(stars), figsize=(5 * len(stars), 5))
    for ax, s in zip(axes, stars):
        freq = table.loc[~table.index.isin(CLOUD_STOP), f"star_{s}"].nlargest(150).to_dict()
        ax.imshow(WordCloud(font_path=FONT, background_color="white", width=500, height=500)
                  .generate_from_frequencies(freq))
        ax.set_title(f"{s} star")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "wordcloud_by_star.png"), dpi=110)

    # 4b) heat map of the word table: P(word | star) for the 30 most frequent words
    top = table[~table.index.isin(CLOUD_STOP)].head(30)[[f"P(word|{s})" for s in stars]]
    fig3 = plt.figure(figsize=(7, 10))
    sns.heatmap(top, annot=True, fmt=".4f", cmap="YlOrRd", xticklabels=stars,
                annot_kws={"size": 7})
    plt.xlabel("Star")
    plt.title("P(word | star) heat map")
    plt.yticks(fontproperties=FONT_PROP)
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUT_DIR, "word_heatmap.png"), dpi=110)

    # 5) Naive Bayes by hand: score(star) = average log P(feature | star) + bias(star)
    # 5a) tune the per-star bias on a validation split taken from the train part
    tr_labels = train.rating.values
    is_val = stratified_split(tr_labels, stars, 0.2, seed=7)
    tr_feats = list(train.feats)
    row, log_like = train_nb([f for f, v in zip(tr_feats, is_val) if not v], tr_labels[~is_val], stars)
    val_scores = nb_scores([f for f, v in zip(tr_feats, is_val) if v], row, log_like)
    bias = tune_bias(val_scores, tr_labels[is_val], stars)
    print(f"\nPer-star bias (tuned for {OBJECTIVE}): {np.round(bias, 3).tolist()}")

    # 5b) retrain on the whole train part and evaluate on the test part
    row, log_like = train_nb(tr_feats, tr_labels, stars)
    print(f"\nModel features: {len(row)} (unigrams{' + bigrams' if USE_BIGRAMS else ''}, min count {MIN_COUNT})")
    pred = np.array(stars)[np.argmax(nb_scores(list(test.feats), row, log_like) + bias, axis=1)]
    cm, precision, recall, f1, support, acc = metrics(test.rating.values, pred, stars)
    print(f"\nTest Accuracy: {acc:.4f}   Macro F1: {f1.mean():.4f}\n")
    print(f"{'star':>8}{'precision':>11}{'recall':>9}{'f1-score':>10}{'support':>9}")
    for i, s in enumerate(stars):
        print(f"{s:>8}{precision[i]:>11.2f}{recall[i]:>9.2f}{f1[i]:>10.2f}{support[i]:>9}")
    print(f"{'accuracy':>8}{'':>20}{acc:>10.2f}{support.sum():>9}")
    print(f"{'macro':>8}{precision.mean():>11.2f}{recall.mean():>9.2f}{f1.mean():>10.2f}{support.sum():>9}")
    w = support / support.sum()
    print(f"{'weighted':>8}{(precision * w).sum():>11.2f}{(recall * w).sum():>9.2f}{(f1 * w).sum():>10.2f}{support.sum():>9}")

    # 6) confusion matrix heat map
    fig2 = plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=stars, yticklabels=stars)
    plt.xlabel("Predicted star")
    plt.ylabel("Actual star")
    plt.title(f"Confusion Matrix (accuracy {acc:.3f}, macro F1 {f1.mean():.3f})")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=110)

    # 7) retrain on all 40,000 reviews and predict review_dataset/test_file.csv -> submission.csv
    if os.path.exists(TEST_FILE):
        sub = pd.read_csv(TEST_FILE, sep=";", engine="python")
        with Pool() as p:
            sub_feats = [features(t) for t in p.map(tokenize, sub["review"], chunksize=200)]
        row, log_like = train_nb(list(df.feats), ratings, stars)
        sub["rating"] = np.array(stars)[np.argmax(nb_scores(sub_feats, row, log_like) + bias, axis=1)]
        sub[["reviewID", "rating"]].to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)
        print(f"\nsubmission.csv: {len(sub)} predictions, "
              f"stars = {sub.rating.value_counts().sort_index().to_dict()}")

    plt.show()


if __name__ == "__main__":
    main()
