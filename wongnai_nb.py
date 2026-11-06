"""Wongnai review rating prediction with Naive Bayes (word table -> word cloud -> model -> heat map)."""
import os
import re
import sys
from multiprocessing import Pool

import matplotlib

if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

TRAIN = r"D:\Download\review_dataset\w_review_train.csv"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT = r"C:\Windows\Fonts\tahoma.ttf"  # Thai-capable font for the word cloud
FONT_PROP = font_manager.FontProperties(fname=FONT)

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
    return " ".join(t for t in toks if THAI_OR_LATIN.search(t) and t not in STOP)


def main():
    # 1) load data: no header, ';' separated, last column = star rating
    df = pd.read_csv(TRAIN, sep=";", header=None, names=["review", "rating"], engine="python")
    df = df.dropna().astype({"rating": int})
    stars = sorted(df.rating.unique())
    print(f"Loaded df: {len(df)} reviews, stars = {stars}")

    with Pool() as p:
        df["tokens"] = p.map(tokenize, df["review"], chunksize=200)

    # 2) word table: rows = words, columns = counts per star + Naive Bayes probabilities
    cv = CountVectorizer(token_pattern=r"\S+", lowercase=False)
    X = cv.fit_transform(df["tokens"])
    table = pd.DataFrame({f"star_{s}": X[(df.rating == s).values].sum(axis=0).A1 for s in stars},
                         index=cv.get_feature_names_out())
    table["total"] = table.sum(axis=1)
    for s in stars:  # P(word | star), Laplace smoothing
        table[f"P(word|{s})"] = (table[f"star_{s}"] + 1) / (table[f"star_{s}"].sum() + len(table))
    table["P(word)"] = table["total"] / table["total"].sum()
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

    # 5) Naive Bayes: test accuracy + classification report
    Xtr, Xte, ytr, yte = train_test_split(X, df["rating"], test_size=0.2, random_state=42,
                                          stratify=df["rating"])
    pred = MultinomialNB().fit(Xtr, ytr).predict(Xte)
    acc = (pred == yte).mean()
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print(classification_report(yte, pred, zero_division=0))

    # 6) confusion matrix heat map
    fig2 = plt.figure(figsize=(7, 6))
    sns.heatmap(confusion_matrix(yte, pred, labels=stars), annot=True, fmt="d", cmap="Blues",
                xticklabels=stars, yticklabels=stars)
    plt.xlabel("Predicted star")
    plt.ylabel("Actual star")
    plt.title(f"Confusion Matrix (accuracy {acc:.3f})")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=110)

    if "--no-show" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
