"""PHASE 1 - Khám phá dữ liệu (EDA). Chạy: python src/eda.py

Ghi chú phương pháp: EDA mang tính MÔ TẢ, chạy trên toàn bộ dữ liệu thô.
Không có tham số mô hình/tiền xử lý nào được học từ đây; mọi tham số
học được nằm trong Pipeline và chỉ fit trên Train (xem preprocessing.py).
Các quyết định dựa trên EDA (bỏ 'education', '?' -> 'Unknown') là quy tắc
xác định, không phụ thuộc phân phối Validation/Test.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_loader import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, ROOT, TARGET,
                         clean, load_raw)

FIG = ROOT / "results" / "figures" / "eda"
TAB = ROOT / "results" / "eda"
# Bảng màu tham chiếu (slot 1-2, đã kiểm tra phân biệt cho người mù màu)
C_LOW, C_HIGH = "#2a78d6", "#eb6834"
INK, MUTED = "#2b2b2b", "#6b6b6b"
LABELS = {0: "<=50K", 1: ">50K"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#bdbdbd", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#ececec", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)


def basic_tables(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Kích thước, kiểu dữ liệu, thiếu, trùng lặp, phân phối nhãn."""
    TAB.mkdir(parents=True, exist_ok=True)
    info = pd.DataFrame({
        "dtype": raw.dtypes.astype(str),
        "n_unique": raw.nunique(),
        "n_question_mark": (raw.astype(str).apply(lambda s: s.str.strip()) == "?").sum(),
        "n_nan": raw.isna().sum(),
    })
    info["pct_question_mark"] = (100 * info["n_question_mark"] / len(raw)).round(2)
    info.to_csv(TAB / "columns_overview.csv")

    label = df[TARGET].map(LABELS).value_counts().rename("count").to_frame()
    label["pct"] = (100 * label["count"] / len(df)).round(2)
    label.to_csv(TAB / "label_distribution.csv")

    num_desc = df[NUMERIC_FEATURES].describe().T
    num_desc["skew"] = df[NUMERIC_FEATURES].skew()
    num_desc["pct_zero"] = (100 * (df[NUMERIC_FEATURES] == 0).mean()).round(2)
    num_desc.round(3).to_csv(TAB / "numeric_summary.csv")

    by_class = df.groupby(TARGET)[NUMERIC_FEATURES].agg(["mean", "median"]).T
    by_class.columns = [LABELS[c] for c in by_class.columns]
    by_class.round(2).to_csv(TAB / "numeric_by_class.csv")

    missing_rows = df[["workclass", "occupation", "native.country"]].isna().any(axis=1)
    miss = pd.DataFrame({
        "n_rows": [int(missing_rows.sum()), int((~missing_rows).sum())],
        "rate_gt50K": [df.loc[missing_rows, TARGET].mean(), df.loc[~missing_rows, TARGET].mean()],
    }, index=["có '?'", "không có '?'"]).round(4)
    miss.to_csv(TAB / "missing_vs_income.csv")
    # workclass='?' và occupation='?' có trùng dòng không?
    both = int((df["workclass"].isna() & df["occupation"].isna()).sum())

    return {"shape": raw.shape, "info": info, "label": label, "num_desc": num_desc,
            "by_class": by_class, "miss": miss,
            "dups": int(raw.duplicated().sum()), "wc_and_occ_missing": both}


def category_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Tỷ lệ >50K theo từng hạng mục của mọi cột phân loại (kể cả education)."""
    rows = []
    for c in CATEGORICAL_FEATURES + ["education"]:
        g = df.assign(**{c: df[c].fillna("Unknown")}).groupby(c)[TARGET].agg(["size", "mean"])
        for k, r in g.iterrows():
            rows.append({"feature": c, "category": k, "count": int(r["size"]),
                         "rate_gt50K": round(r["mean"], 4)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "category_income_rates.csv", index=False)
    return out


def cramers_v(x, y) -> float:
    """Cramér's V (có hiệu chỉnh độ chệch Bergsma 2013) giữa hai biến phân loại."""
    from scipy.stats import chi2_contingency
    t = pd.crosstab(x, y)
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.values.sum()
    r, k = t.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(kc - 1, rc - 1))))


def associations(df: pd.DataFrame):
    """Liên hệ thuộc tính - nhãn và giữa các thuộc tính (phát hiện dư thừa)."""
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import OrdinalEncoder
    cats = CATEGORICAL_FEATURES + ["education"]
    filled = df[cats].fillna("Unknown")
    # Tương quan Pearson & Spearman giữa các thuộc tính số (+ nhãn)
    pearson = df[NUMERIC_FEATURES + [TARGET]].corr(method="pearson")
    spearman = df[NUMERIC_FEATURES + [TARGET]].corr(method="spearman")
    pearson.round(3).to_csv(TAB / "corr_pearson.csv")
    spearman.round(3).to_csv(TAB / "corr_spearman.csv")
    # Cramér's V giữa các thuộc tính phân loại (+ nhãn)
    cv_cols = cats + [TARGET]
    cv = pd.DataFrame(index=cv_cols, columns=cv_cols, dtype=float)
    for a in cv_cols:
        for b in cv_cols:
            xa = filled[a] if a in cats else df[a]
            xb = filled[b] if b in cats else df[b]
            cv.loc[a, b] = 1.0 if a == b else cramers_v(xa, xb)
    cv.round(3).to_csv(TAB / "cramers_v.csv")
    # Mutual information với nhãn (mô tả, seed cố định)
    X = pd.concat([df[NUMERIC_FEATURES],
                   pd.DataFrame(OrdinalEncoder().fit_transform(filled), columns=cats,
                                index=df.index)], axis=1)
    discrete = [c in cats for c in X.columns]
    mi = pd.Series(mutual_info_classif(X, df[TARGET], discrete_features=discrete,
                                       random_state=42), index=X.columns)
    mi = mi.sort_values(ascending=False).rename("mutual_info")
    mi.round(4).to_csv(TAB / "mutual_info.csv")
    return pearson, spearman, cv, mi


# ----------------------------- BIỂU ĐỒ ---------------------------------------
def plot_income(df):
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[C_LOW, C_HIGH], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.annotate(f"{v:,}  ({100 * v / len(df):.1f}%)", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
    ax.set_ylabel("Số mẫu")
    ax.set_title("Phân phối nhãn thu nhập")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="x", visible=False)
    _save(fig, "01_income_distribution.png")


def plot_numeric_by_class(df, col, fname, title, bins, xlabel):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for k, color in [(0, C_LOW), (1, C_HIGH)]:
        ax.hist(df.loc[df[TARGET] == k, col], bins=bins, color=color, alpha=0.55,
                density=True, label=LABELS[k], edgecolor="white", linewidth=0.5)
        ax.axvline(df.loc[df[TARGET] == k, col].median(), color=color, lw=2, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mật độ (chuẩn hóa theo lớp)")
    ax.set_title(title)
    ax.legend(title="Thu nhập (đường đứt = trung vị)", frameon=False)
    _save(fig, fname)


def plot_rate_bar(rates, feature, fname, title, order_by="rate_gt50K", min_count=0):
    """Thanh ngang: tỷ lệ >50K mỗi hạng mục, kèm số mẫu (n)."""
    r = rates[(rates.feature == feature) & (rates["count"] >= min_count)]
    r = r.sort_values(order_by)
    overall = None
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(r) + 1.2))
    ax.barh(r["category"].astype(str), 100 * r["rate_gt50K"], color=C_HIGH, height=0.6)
    for i, (rate, n) in enumerate(zip(r["rate_gt50K"], r["count"])):
        ax.annotate(f"{100 * rate:.1f}%  (n={n:,})", (100 * rate, i), va="center",
                    xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_xlabel("Tỷ lệ thu nhập >50K (%)")
    ax.set_title(title)
    ax.set_xlim(0, max(100 * r["rate_gt50K"].max() * 1.35, 10))
    ax.grid(axis="y", visible=False)
    return fig, ax


def plot_education(df):
    g = df.groupby(["education.num", "education"])[TARGET].agg(["size", "mean"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(g["education"], 100 * g["mean"], color=C_HIGH, height=0.6)
    for i, (rate, n) in enumerate(zip(g["mean"], g["size"])):
        ax.annotate(f"{100 * rate:.1f}%  (n={n:,})", (100 * rate, i), va="center",
                    xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{e}  [{n}]" for e, n in zip(g["education"], g["education.num"])])
    ax.set_xlabel("Tỷ lệ thu nhập >50K (%)")
    ax.set_title("Trình độ học vấn (sắp theo education.num) và tỷ lệ >50K")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    _save(fig, "04_education.png")


def plot_capital(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, col in zip(axes, ["capital.gain", "capital.loss"]):
        grp = np.where(df[col] > 0, f"{col} > 0", f"{col} = 0")
        t = pd.crosstab(grp, df[TARGET], normalize="index")[1] * 100
        n = pd.Series(grp).value_counts()
        ax.bar(t.index, t.values, color=C_HIGH, width=0.5)
        for i, k in enumerate(t.index):
            ax.annotate(f"{t[k]:.1f}%  (n={n[k]:,})", (i, t[k]), ha="center",
                        xytext=(0, 3), textcoords="offset points", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Tỷ lệ >50K (%)")
        ax.set_title(col)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Thuộc tính lệch về 0: tỷ lệ >50K khi có / không có lãi-lỗ vốn",
                 fontweight="bold")
    _save(fig, "07_capital_zero_vs_nonzero.png")


def plot_heatmap(mat, fname, title, fmt=".2f", vmin=-1, cmap="RdBu_r"):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(mat.astype(float), annot=True, fmt=fmt, cmap=cmap, vmin=vmin, vmax=1,
                center=0 if vmin < 0 else None, square=True, linewidths=1,
                linecolor="white", cbar_kws={"shrink": 0.75}, ax=ax,
                annot_kws={"fontsize": 8})
    ax.set_title(title)
    ax.grid(False)
    _save(fig, fname)


def plot_mi(mi):
    m = mi.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(m.index, m.values, color=C_LOW, height=0.6)
    for i, v in enumerate(m.values):
        ax.annotate(f"{v:.3f}", (v, i), va="center", xytext=(4, 0),
                    textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_xlabel("Mutual information với nhãn (nats)")
    ax.set_title("Mức liên hệ của từng thuộc tính với nhãn thu nhập")
    ax.grid(axis="y", visible=False)
    _save(fig, "10_mutual_information.png")


def run():
    raw = load_raw()
    df = clean(raw)
    res = basic_tables(raw, df)
    rates = category_rates(df)
    pearson, spearman, cv, mi = associations(df)

    plot_income(df)
    plot_numeric_by_class(df, "age", "02_age.png", "Phân phối độ tuổi theo lớp thu nhập",
                          bins=range(17, 92, 2), xlabel="Tuổi")
    plot_numeric_by_class(df, "hours.per.week", "03_hours_per_week.png",
                          "Số giờ làm việc mỗi tuần theo lớp thu nhập",
                          bins=range(0, 101, 4), xlabel="Giờ/tuần")
    plot_education(df)
    fig, _ = plot_rate_bar(rates, "occupation", None, "Nghề nghiệp và tỷ lệ >50K")
    _save(fig, "05_occupation.png")
    for i, (col, title) in enumerate([("marital.status", "Tình trạng hôn nhân"),
                                      ("relationship", "Quan hệ trong hộ gia đình"),
                                      ("workclass", "Loại hình công việc"),
                                      ("sex", "Giới tính"), ("race", "Chủng tộc")]):
        fig, _ = plot_rate_bar(rates, col, None, f"{title} và tỷ lệ >50K")
        _save(fig, f"06{chr(97 + i)}_{col.replace('.', '_')}.png")
    plot_capital(df)
    plot_heatmap(pearson, "08_corr_numeric_pearson.png",
                 "Tương quan Pearson giữa thuộc tính số và nhãn")
    plot_heatmap(cv, "09_cramers_v_categorical.png",
                 "Cramér's V giữa thuộc tính phân loại và nhãn", vmin=0, cmap="Blues")
    plot_mi(mi)

    res.update(rates=rates, pearson=pearson, spearman=spearman, cv=cv, mi=mi)
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import warnings
    warnings.filterwarnings("ignore")
    r = run()
    pd.set_option("display.width", 200)
    print("shape:", r["shape"], "| dòng trùng:", r["dups"],
          "| wc&occ cùng '?':", r["wc_and_occ_missing"])
    print(r["info"]); print(r["label"]); print(r["num_desc"].round(2))
    print(r["by_class"]); print(r["miss"]); print(r["mi"].round(4))
    print(r["pearson"].round(3)); print(r["cv"].round(3))
    rt = r["rates"]
    for f in ["occupation", "marital.status", "relationship", "sex", "workclass", "race"]:
        print(rt[rt.feature == f].sort_values("rate_gt50K", ascending=False).to_string(index=False))
    nc = rt[rt.feature == "native.country"].sort_values("count", ascending=False)
    print(nc.head(5).to_string(index=False), "\n# countries:", len(nc),
          "| US share:", round(nc.iloc[0]["count"] / nc["count"].sum(), 4))
