"""PHASE 1 - Khám phá dữ liệu (EDA). Chạy: python src/eda.py

Ghi chú phương pháp:
- EDA mang tính MÔ TẢ trên dữ liệu thô để hiểu phân phối, dữ liệu thiếu và tương quan.
- Không học tham số mô hình/tiền xử lý ở đây; mọi bước học tham số nằm trong
  Pipeline ở preprocessing.py và chỉ fit trên Train.
"""
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

from data_loader import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ROOT,
    TARGET,
    clean,
    load_raw,
)

FIG_DIR = ROOT / "results" / "figures" / "eda"
TABLE_DIR = ROOT / "results" / "eda"

COLOR_LOW = "#2a78d6"
COLOR_HIGH = "#eb6834"
COLOR_TEXT = "#2b2b2b"
COLOR_MUTED = "#6b6b6b"
LABELS = {0: "<=50K", 1: ">50K"}

# Cấu hình vẽ cho 5 biểu đồ tỷ lệ >50K theo từng thuộc tính phân loại (06a..06e)
RATE_PLOT_SPECS = [
    ("06a_marital_status.png", "marital.status", "Tình trạng hôn nhân và tỷ lệ >50K"),
    ("06b_relationship.png", "relationship", "Quan hệ trong hộ gia đình và tỷ lệ >50K"),
    ("06c_workclass.png", "workclass", "Loại hình công việc và tỷ lệ >50K"),
    ("06d_sex.png", "sex", "Giới tính và tỷ lệ >50K"),
    ("06e_race.png", "race", "Chủng tộc và tỷ lệ >50K"),
]

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#bdbdbd",
    "axes.labelcolor": COLOR_TEXT,
    "text.color": COLOR_TEXT,
    "xtick.color": COLOR_MUTED,
    "ytick.color": COLOR_MUTED,
    "axes.grid": True,
    "grid.color": "#ececec",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def _save_fig(fig, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


# ========================== PHẦN 1: TÍNH BẢNG THỐNG KÊ ==========================
def basic_tables(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Tính bảng tổng quan cột, phân phối nhãn, thống kê số, missing và trùng lặp."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Tổng quan kiểu dữ liệu và giá trị '?'
    stripped_raw = raw.astype(str).apply(lambda col: col.str.strip())
    info = pd.DataFrame({
        "dtype": raw.dtypes.astype(str),
        "n_unique": raw.nunique(),
        "n_question_mark": (stripped_raw == "?").sum(),
        "n_nan": raw.isna().sum(),
    })
    info["pct_question_mark"] = (100 * info["n_question_mark"] / len(raw)).round(2)
    info.to_csv(TABLE_DIR / "columns_overview.csv")

    # 2. Phân phối nhãn <=50K / >50K
    label_dist = df[TARGET].map(LABELS).value_counts().rename("count").to_frame()
    label_dist["pct"] = (100 * label_dist["count"] / len(df)).round(2)
    label_dist.to_csv(TABLE_DIR / "label_distribution.csv")

    # 3. Thống kê các cột số (kèm độ lệch skew và tỷ lệ giá trị 0)
    num_desc = df[NUMERIC_FEATURES].describe().T
    num_desc["skew"] = df[NUMERIC_FEATURES].skew()
    num_desc["pct_zero"] = (100 * (df[NUMERIC_FEATURES] == 0).mean()).round(2)
    num_desc.round(3).to_csv(TABLE_DIR / "numeric_summary.csv")

    by_class = df.groupby(TARGET)[NUMERIC_FEATURES].agg(["mean", "median"]).T
    by_class.columns = [LABELS[c] for c in by_class.columns]
    by_class.round(2).to_csv(TABLE_DIR / "numeric_by_class.csv")

    # 4. So sánh tỷ lệ >50K giữa nhóm có '?' và nhóm không có '?'
    has_missing = df[["workclass", "occupation", "native.country"]].isna().any(axis=1)
    miss_summary = pd.DataFrame({
        "n_rows": [int(has_missing.sum()), int((~has_missing).sum())],
        "rate_gt50K": [
            df.loc[has_missing, TARGET].mean(),
            df.loc[~has_missing, TARGET].mean(),
        ],
    }, index=["có '?'", "không có '?'"]).round(4)
    miss_summary.to_csv(TABLE_DIR / "missing_vs_income.csv")

    both_missing = int((df["workclass"].isna() & df["occupation"].isna()).sum())

    return {
        "shape": raw.shape,
        "info": info,
        "label": label_dist,
        "num_desc": num_desc,
        "by_class": by_class,
        "miss": miss_summary,
        "dups": int(raw.duplicated().sum()),
        "wc_and_occ_missing": both_missing,
    }


def category_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Tính tỷ lệ thu nhập >50K theo từng giá trị của mỗi cột phân loại."""
    rows = []
    for col in CATEGORICAL_FEATURES + ["education"]:
        filled_col = df[col].fillna("Unknown")
        grouped = df.assign(**{col: filled_col}).groupby(col)[TARGET].agg(["size", "mean"])
        for category, stats in grouped.iterrows():
            rows.append({
                "feature": col,
                "category": category,
                "count": int(stats["size"]),
                "rate_gt50K": round(stats["mean"], 4),
            })

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "category_income_rates.csv", index=False)
    return out


def cramers_v(x, y) -> float:
    """Tính Cramér's V (hiệu chỉnh độ chệch Bergsma 2013) giữa hai biến phân loại."""
    table = pd.crosstab(x, y)
    chi2_stat = chi2_contingency(table, correction=False)[0]
    n_samples = table.values.sum()
    n_rows, n_cols = table.shape

    phi2_corrected = max(0.0, chi2_stat / n_samples - (n_cols - 1) * (n_rows - 1) / (n_samples - 1))
    rows_corrected = n_rows - (n_rows - 1) ** 2 / (n_samples - 1)
    cols_corrected = n_cols - (n_cols - 1) ** 2 / (n_samples - 1)
    denom = max(1e-12, min(cols_corrected - 1, rows_corrected - 1))
    return float(np.sqrt(phi2_corrected / denom))


def associations(df: pd.DataFrame):
    """Đo mức liên hệ giữa các đặc trưng và nhãn (Pearson, Spearman, Cramér's V, MI)."""
    cat_cols = CATEGORICAL_FEATURES + ["education"]
    filled_cats = df[cat_cols].fillna("Unknown")

    # 1. Tương quan giữa các cột số và nhãn
    pearson = df[NUMERIC_FEATURES + [TARGET]].corr(method="pearson")
    spearman = df[NUMERIC_FEATURES + [TARGET]].corr(method="spearman")
    pearson.round(3).to_csv(TABLE_DIR / "corr_pearson.csv")
    spearman.round(3).to_csv(TABLE_DIR / "corr_spearman.csv")

    # 2. Ma trận Cramér's V giữa các cột phân loại và nhãn
    cv_cols = cat_cols + [TARGET]
    cv_matrix = pd.DataFrame(index=cv_cols, columns=cv_cols, dtype=float)
    for col_a in cv_cols:
        for col_b in cv_cols:
            if col_a == col_b:
                cv_matrix.loc[col_a, col_b] = 1.0
                continue
            series_a = filled_cats[col_a] if col_a in cat_cols else df[col_a]
            series_b = filled_cats[col_b] if col_b in cat_cols else df[col_b]
            cv_matrix.loc[col_a, col_b] = cramers_v(series_a, series_b)
    cv_matrix.round(3).to_csv(TABLE_DIR / "cramers_v.csv")

    # 3. Mutual Information với nhãn (đo cả quan hệ phi tuyến)
    encoded_cats = pd.DataFrame(
        OrdinalEncoder().fit_transform(filled_cats),
        columns=cat_cols,
        index=df.index,
    )
    X_all = pd.concat([df[NUMERIC_FEATURES], encoded_cats], axis=1)
    discrete_mask = [col in cat_cols for col in X_all.columns]

    mi_scores = mutual_info_classif(
        X_all,
        df[TARGET],
        discrete_features=discrete_mask,
        random_state=42,
    )
    mi = pd.Series(mi_scores, index=X_all.columns).sort_values(ascending=False).rename("mutual_info")
    mi.round(4).to_csv(TABLE_DIR / "mutual_info.csv")

    return pearson, spearman, cv_matrix, mi


# ========================== PHẦN 2: VẼ BIỂU ĐỒ ===============================
def plot_income(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values, color=[COLOR_LOW, COLOR_HIGH], width=0.55)

    for bar, val in zip(bars, counts.values):
        ax.annotate(
            f"{val:,}  ({100 * val / len(df):.1f}%)",
            (bar.get_x() + bar.get_width() / 2, val),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
        )

    ax.set_ylabel("Số mẫu")
    ax.set_title("Phân phối nhãn thu nhập")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="x", visible=False)
    _save_fig(fig, "01_income_distribution.png")


def plot_numeric_by_class(df: pd.DataFrame, col: str, fname: str, title: str, bins, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for label_val, color in [(0, COLOR_LOW), (1, COLOR_HIGH)]:
        values = df.loc[df[TARGET] == label_val, col]
        ax.hist(
            values,
            bins=bins,
            color=color,
            alpha=0.55,
            density=True,
            label=LABELS[label_val],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axvline(values.median(), color=color, lw=2, ls="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mật độ (chuẩn hóa theo lớp)")
    ax.set_title(title)
    ax.legend(title="Thu nhập (đường đứt = trung vị)", frameon=False)
    _save_fig(fig, fname)


def plot_rate_bar(rates: pd.DataFrame, feature: str, fname: str, title: str) -> None:
    """Vẽ biểu đồ cột ngang hiển thị tỷ lệ >50K theo từng hạng mục."""
    subset = rates[rates.feature == feature].sort_values("rate_gt50K")
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(subset) + 1.2))
    ax.barh(subset["category"].astype(str), 100 * subset["rate_gt50K"], color=COLOR_HIGH, height=0.6)

    for idx, (rate, count) in enumerate(zip(subset["rate_gt50K"], subset["count"])):
        ax.annotate(
            f"{100 * rate:.1f}%  (n={count:,})",
            (100 * rate, idx),
            va="center",
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=COLOR_MUTED,
        )

    ax.set_xlabel("Tỷ lệ thu nhập >50K (%)")
    ax.set_title(title)
    ax.set_xlim(0, max(100 * subset["rate_gt50K"].max() * 1.35, 10))
    ax.grid(axis="y", visible=False)
    _save_fig(fig, fname)


def plot_education(df: pd.DataFrame) -> None:
    grouped = df.groupby(["education.num", "education"])[TARGET].agg(["size", "mean"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(grouped["education"], 100 * grouped["mean"], color=COLOR_HIGH, height=0.6)

    for idx, (rate, count) in enumerate(zip(grouped["mean"], grouped["size"])):
        ax.annotate(
            f"{100 * rate:.1f}%  (n={count:,})",
            (100 * rate, idx),
            va="center",
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=COLOR_MUTED,
        )

    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels([f"{edu}  [{num}]" for edu, num in zip(grouped["education"], grouped["education.num"])])
    ax.set_xlabel("Tỷ lệ thu nhập >50K (%)")
    ax.set_title("Trình độ học vấn (sắp theo education.num) và tỷ lệ >50K")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    _save_fig(fig, "04_education.png")


def plot_capital(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, col in zip(axes, ["capital.gain", "capital.loss"]):
        group_labels = np.where(df[col] > 0, f"{col} > 0", f"{col} = 0")
        rates_pct = pd.crosstab(group_labels, df[TARGET], normalize="index")[1] * 100
        counts = pd.Series(group_labels).value_counts()

        ax.bar(rates_pct.index, rates_pct.values, color=COLOR_HIGH, width=0.5)
        for idx, key in enumerate(rates_pct.index):
            ax.annotate(
                f"{rates_pct[key]:.1f}%  (n={counts[key]:,})",
                (idx, rates_pct[key]),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=9,
            )
        ax.set_ylim(0, 100)
        ax.set_ylabel("Tỷ lệ >50K (%)")
        ax.set_title(col)
        ax.grid(axis="x", visible=False)

    fig.suptitle("Thuộc tính lệch về 0: tỷ lệ >50K khi có / không có lãi-lỗ vốn", fontweight="bold")
    _save_fig(fig, "07_capital_zero_vs_nonzero.png")


def plot_heatmap(matrix: pd.DataFrame, fname: str, title: str, fmt: str = ".2f", vmin: float = -1, cmap: str = "RdBu_r") -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=1,
        center=0 if vmin < 0 else None,
        square=True,
        linewidths=1,
        linecolor="white",
        cbar_kws={"shrink": 0.75},
        ax=ax,
        annot_kws={"fontsize": 8},
    )
    ax.set_title(title)
    ax.grid(False)
    _save_fig(fig, fname)


def plot_mi(mi: pd.Series) -> None:
    sorted_mi = mi.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(sorted_mi.index, sorted_mi.values, color=COLOR_LOW, height=0.6)

    for idx, val in enumerate(sorted_mi.values):
        ax.annotate(
            f"{val:.3f}",
            (val, idx),
            va="center",
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=COLOR_MUTED,
        )

    ax.set_xlabel("Mutual information với nhãn (nats)")
    ax.set_title("Mức liên hệ của từng thuộc tính với nhãn thu nhập")
    ax.grid(axis="y", visible=False)
    _save_fig(fig, "10_mutual_information.png")


# ========================== PHẦN 3: ĐIỀU PHỐI CHÍNH ===========================
def run() -> dict:
    """Chạy toàn bộ EDA: tính bảng CSV, vẽ 14 hình và trả về dict kết quả."""
    raw = load_raw()
    df = clean(raw)

    results = basic_tables(raw, df)
    rates = category_rates(df)
    pearson, spearman, cramers, mi = associations(df)

    plot_income(df)
    plot_numeric_by_class(
        df, "age", "02_age.png", "Phân phối độ tuổi theo lớp thu nhập",
        bins=range(17, 92, 2), xlabel="Tuổi",
    )
    plot_numeric_by_class(
        df, "hours.per.week", "03_hours_per_week.png",
        "Số giờ làm việc mỗi tuần theo lớp thu nhập",
        bins=range(0, 101, 4), xlabel="Giờ/tuần",
    )
    plot_education(df)
    plot_rate_bar(rates, "occupation", "05_occupation.png", "Nghề nghiệp và tỷ lệ >50K")

    for fname, col, title in RATE_PLOT_SPECS:
        plot_rate_bar(rates, col, fname, title)

    plot_capital(df)
    plot_heatmap(pearson, "08_corr_numeric_pearson.png", "Tương quan Pearson giữa thuộc tính số và nhãn")
    plot_heatmap(cramers, "09_cramers_v_categorical.png", "Cramér's V giữa thuộc tính phân loại và nhãn", vmin=0, cmap="Blues")
    plot_mi(mi)

    results.update(rates=rates, pearson=pearson, spearman=spearman, cv=cramers, mi=mi)
    return results


def _print_summary(results: dict) -> None:
    pd.set_option("display.width", 200)
    print(
        "shape:", results["shape"],
        "| dòng trùng:", results["dups"],
        "| wc&occ cùng '?':", results["wc_and_occ_missing"],
    )
    for key in ("info", "label"):
        print(results[key])
    print(results["num_desc"].round(2))
    for key in ("by_class", "miss"):
        print(results[key])
    print(results["mi"].round(4))
    print(results["pearson"].round(3))
    print(results["cv"].round(3))

    rates = results["rates"]
    for col in ("occupation", "marital.status", "relationship", "sex", "workclass", "race"):
        sub = rates[rates.feature == col].sort_values("rate_gt50K", ascending=False)
        print(sub.to_string(index=False))

    countries = rates[rates.feature == "native.country"].sort_values("count", ascending=False)
    us_share = round(countries.iloc[0]["count"] / countries["count"].sum(), 4)
    print(countries.head(5).to_string(index=False))
    print(f"# countries: {len(countries)} | US share: {us_share}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    eda_results = run()
    _print_summary(eda_results)
