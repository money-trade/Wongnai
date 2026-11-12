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
# Thai-capable font for the word cloud / heat map (first one found is used)
FONT_CANDIDATES = [r"C:\Windows\Fonts\tahoma.ttf",
                   "/System/Library/Fonts/Supplemental/Thonburi.ttc",
                   "/Library/Fonts/Thonburi.ttc",
                   "/usr/share/fonts/opentype/tlwg/Loma.otf",
                   "/usr/share/fonts/truetype/tlwg/Loma.ttf",
                   "/usr/share/fonts/truetype/tlwg/Garuda.ttf"]
FONT = os.environ.get("WONGNAI_FONT") or next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)
FONT_PROP = font_manager.FontProperties(fname=FONT) if FONT else None

# Stop words are removed, but sentiment-bearing ones (negation / intensity) are kept
KEEP = {"ไม่", "มาก", "น้อย", "สุด", "ที่สุด", "เกิน", "จริง", "เลย"}
STOP = set(thai_stopwords()) - KEEP
# extra filler words hidden from the word clouds only
CLOUD_STOP = {"ร้าน", "ๆ", "ไม่", "มี", "มา", "ไป", "ก็", "ที่", "ได้", "เป็น", "แต่", "นี้", "จะ",
              "ว่า", "แล้ว", "กับ", "และ", "ให้", "อยู่", "ของ", "ใน", "กิน", "ทาน", "เมนู",
              "อาหาร", "ผม", "เรา", "อีก", "หรือ", "คน", "ดู", "ทาง", "นั่ง", "สั่ง"}
THAI_OR_LATIN = re.compile(r"[\u0E00-\u0E7Fa-zA-Z0-9]")


def tokenize(text):
    toks = word_tokenize(str(text).lower(), engine="newmm", keep_whitespace=False)
    return [t for t in toks if THAI_OR_LATIN.search(t) and t not in STOP]


def stratified_split(labels, stars, frac, seed):
    """Boolean mask that picks `frac` of each star at random."""
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.permutation(np.where(labels == s)[0])[: int(frac * (labels == s).sum())]
                          for s in stars])
    mask = np.zeros(len(labels), dtype=bool)
    mask[idx] = True
    return mask


def main():
    # 1) load data: no header, ';' separated, last column = star rating
    df = pd.read_csv(TRAIN, sep=";", header=None, names=["review", "rating"], engine="python")
    df = df.dropna().astype({"rating": int})
    stars = [int(s) for s in sorted(df.rating.unique())]
    print(f"Loaded df: {len(df)} reviews, stars = {stars}")

    with Pool() as p:
        df["tokens"] = p.map(tokenize, df["review"], chunksize=200)

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

    # 5) Naive Bayes by hand: score(star) = log P(star) + sum of log P(word | star) over the review's words
    log_like = np.log(table[[f"P(word|{s})" for s in stars]].values)  # words x stars
    log_prior = np.log([(train.rating == s).mean() for s in stars])
    row = {w: i for i, w in enumerate(table.index)}  # words not in the table (OOV) are skipped
    pred = np.array([stars[np.argmax(log_prior + log_like[[row[w] for w in toks if w in row]].sum(axis=0))]
                     for toks in test.tokens])
    actual = test.rating.values

    # confusion matrix (rows = actual, columns = predicted) and metrics
    k = len(stars)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (np.searchsorted(stars, actual), np.searchsorted(stars, pred)), 1)
    tp, support = np.diag(cm), cm.sum(axis=1)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(k), where=cm.sum(axis=0) > 0)
    recall = tp / support
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=(precision + recall) > 0)
    acc = tp.sum() / cm.sum()
    print(f"\nTest Accuracy: {acc:.4f}\n")
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
    plt.title(f"Confusion Matrix (accuracy {acc:.3f})")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=110)

    plt.show()


if __name__ == "__main__":
    main()
