"""PHASE 5–6 – Huấn luyện cuối trên Train+Validation và đánh giá Test MỘT LẦN.

Chạy (từ thư mục gốc dự án):
    python src/final_evaluation.py

Nguyên tắc bảo vệ tập Test:
- Cấu hình bị khóa cứng (LOCKED_CONFIGS) và phải khớp best_configs.json từ PHASE 4.
- Preprocessing + mô hình chỉ fit trên Train+Validation (27.677 mẫu).
- Dự đoán trên Test đúng MỘT lần với ngưỡng cố định 0.5.
- Nếu final_predictions.csv đã tồn tại, script tự dừng để tránh đánh giá Test nhiều lần.
"""
import hashlib
import json
import logging
import platform
import sys
import time
import warnings
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import binomtest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from data_loader import (
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    RAW_PATH,
    ROOT,
    TARGET,
    clean,
    duplicate_group_ids,
    load_raw,
    split_data,
    xy,
)
from decision_tree import build_dt, dt_complexity
from evaluation import POS_LABEL, compute_metrics, positive_proba
from naive_bayes import build_cnb, build_gnb

RES_DIR = ROOT / "results"
MODEL_DIR = RES_DIR / "models" / "final"
METRICS_DIR = RES_DIR / "metrics"
FIG_DIR = RES_DIR / "figures" / "final"
LOGS_DIR = RES_DIR / "logs"
REPORTS_DIR = RES_DIR / "reports"

LOCKED_CONFIGS = {
    "DecisionTree": {
        "criterion": "entropy",
        "max_depth": 12,
        "min_samples_leaf": 10,
        "min_samples_split": 2,
        "class_weight": "balanced",
    },
    "CategoricalNB": {
        "n_bins": 20,
        "alpha": 0.1,
        "fit_prior": True,
    },
    "GaussianNB": {
        "var_smoothing": 1e-7,
    },
}

BUILDERS = {
    "DecisionTree": build_dt,
    "CategoricalNB": build_cnb,
    "GaussianNB": build_gnb,
}
ROLES = {"DecisionTree": "chính", "CategoricalNB": "chính", "GaussianNB": "phụ"}
MAIN_MODELS = ["DecisionTree", "CategoricalNB"]
COLORS = {"DecisionTree": "#2a78d6", "CategoricalNB": "#eb6834", "GaussianNB": "#1baf7a"}

N_BOOT = 5000
BOOT_SEED = 42
EXPECTED_COUNTS = {"trainval": 27677, "test": 4884, "test_pos": 1176}

log = logging.getLogger("final")


def setup_logging(timestamp: str) -> None:
    for d in (MODEL_DIR, METRICS_DIR, FIG_DIR, LOGS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(LOGS_DIR / f"phase5_6_final_{timestamp}.log", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, stream_handler):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def sha256_file(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ========================== BOOTSTRAP HELPERS =================================
def stratified_bootstrap_indices(y, n_boot, seed):
    """Sinh chỉ số lấy mẫu lại có hoàn lại, giữ nguyên số lượng mẫu mỗi lớp."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([
            rng.choice(neg, len(neg), replace=True),
            rng.choice(pos, len(pos), replace=True),
        ])


def ci(a, level=0.95):
    """Khoảng tin cậy percentile (mặc định 95%: phân vị 2.5% đến 97.5%)."""
    lo, hi = np.percentile(a, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


# ========================== VẼ BIỂU ĐỒ TEST ==================================
def plot_cm(name: str, cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(f"{name} – Test ({ROLES[name]})", fontweight="bold")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    fig.savefig(FIG_DIR / f"cm_test_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_true, scores: dict, metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, score in scores.items():
        line_style = "--" if ROLES[name] == "phụ" else "-"
        fpr, tpr, _ = roc_curve(y_true, score, pos_label=POS_LABEL)
        precision, recall, _ = precision_recall_curve(y_true, score, pos_label=POS_LABEL)

        axes[0].plot(fpr, tpr, color=COLORS[name], lw=2, ls=line_style, label=f"{name} (AUC {metrics[name]['roc_auc']:.3f})")
        axes[1].plot(recall, precision, color=COLORS[name], lw=2, ls=line_style, label=f"{name} (AP {metrics[name]['pr_auc']:.3f})")
        axes[1].plot(
            metrics[name]["recall"],
            metrics[name]["precision"],
            "o",
            ms=8,
            color=COLORS[name],
            mec="white",
            mew=2,
        )

    axes[0].plot([0, 1], [0, 1], color="#9e9e9e", lw=1, ls=":")
    axes[1].axhline(y_true.mean(), color="#9e9e9e", lw=1, ls=":", label=f"Ngẫu nhiên ({y_true.mean():.3f})")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC – Test")
    axes[1].set(xlabel="Recall (>50K)", ylabel="Precision (>50K)", title="Precision-Recall – Test (chấm = ngưỡng 0,5)")

    for ax, loc in zip(axes, ("lower right", "upper right")):
        ax.legend(frameon=False, loc=loc, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#ececec")

    fig.savefig(FIG_DIR / "roc_pr_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_val_vs_test(tbl: pd.DataFrame) -> None:
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    x_pos = np.arange(len(metric_keys))

    for ax, name in zip(axes, MAIN_MODELS):
        val_vals = [tbl.loc[(name, "validation"), m] for m in metric_keys]
        test_vals = [tbl.loc[(name, "test"), m] for m in metric_keys]

        ax.bar(x_pos - 0.2, val_vals, 0.38, color="#c9c9c9", label="Validation (fit trên Train)")
        test_bars = ax.bar(x_pos + 0.2, test_vals, 0.38, color=COLORS[name], label="Test (fit trên Train+Val)")
        for rect, val in zip(test_bars, test_vals):
            ax.annotate(
                f"{val:.3f}",
                (rect.get_x() + rect.get_width() / 2, val),
                ha="center",
                va="bottom",
                fontsize=7.5,
                xytext=(0, 2),
                textcoords="offset points",
            )
        ax.set_xticks(x_pos, metric_labels, fontsize=9)
        ax.set_title(name, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#ececec")
        ax.legend(frameon=False, fontsize=8, loc="lower left")

    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Giá trị")
    fig.suptitle("Validation và Test – khả năng tổng quát hóa", fontweight="bold")
    fig.savefig(FIG_DIR / "validation_vs_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap(diffs: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    specs = [("f1", "F1 (>50K)"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]

    for ax, metric_key, title in zip(axes, *zip(*specs)):
        diff_vals = diffs[metric_key]
        lo, hi = ci(diff_vals)
        ax.hist(diff_vals, bins=60, color="#7aa7e0", edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="#2b2b2b", lw=1.5)
        ax.axvline(lo, color="#eb6834", lw=1.5, ls="--")
        ax.axvline(hi, color="#eb6834", lw=1.5, ls="--")
        ax.set_title(f"Δ {title}: DT − CNB\nCI95% [{lo:+.4f}, {hi:+.4f}]", fontsize=10, fontweight="bold")
        ax.set_xlabel("Chênh lệch")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(f"Số lần lặp (/{N_BOOT})")
    fig.suptitle("Paired stratified bootstrap trên Test (đường đứt = CI 95%, đường đen = 0)", fontweight="bold", y=1.04)
    fig.savefig(FIG_DIR / "bootstrap_differences.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ========================== 5 BƯỚC ĐÁNH GIÁ CUỐI ==============================
def load_locked_data():
    """Bước A: Xác nhận cấu hình khóa khớp PHASE 4 và đọc split đã khóa."""
    best = json.loads((METRICS_DIR / "best_configs.json").read_text(encoding="utf-8"))["best_configs"]
    for name, cfg in LOCKED_CONFIGS.items():
        assert best[name]["config"] == cfg, f"{name}: cấu hình khác PHASE 4"
    log.info("Cấu hình khóa trùng với best_configs.json (PHASE 4)")

    df = clean(load_raw())
    splits = split_data(df)
    ref_indices = pd.read_csv(ROOT / "data" / "processed" / "split_indices.csv")
    for split_name, part in splits.items():
        expected_idx = set(ref_indices.loc[ref_indices.split == split_name, "row_index"])
        assert expected_idx == set(part.index), f"split {split_name} lệch"

    trainval_df = pd.concat([splits["train"], splits["val"]]).sort_index()
    test_df = splits["test"]

    groups = duplicate_group_ids(df)
    shared_groups = set(groups.loc[trainval_df.index]) & set(groups.loc[test_df.index])
    assert not shared_groups, "Nhóm trùng lặp dùng chung giữa Train+Val và Test"
    assert len(trainval_df) == EXPECTED_COUNTS["trainval"]
    assert len(test_df) == EXPECTED_COUNTS["test"]
    assert int(test_df[TARGET].sum()) == EXPECTED_COUNTS["test_pos"]

    log.info(
        f"Split khớp split_indices.csv | Train+Val {len(trainval_df)} | "
        f"Test {len(test_df)} (>50K: {int(test_df[TARGET].sum())}) | nhóm trùng dùng chung: 0"
    )
    return trainval_df, test_df


def train_and_verify_models(Xtv, ytv) -> tuple[dict, dict]:
    """Bước B: Huấn luyện mô hình cuối trên Train+Val và kiểm tra tái lập."""
    models, fit_info = {}, {}

    for name, cfg in LOCKED_CONFIGS.items():
        t0 = time.perf_counter()
        model = BUILDERS[name](**cfg).fit(Xtv, ytv)
        fit_time = time.perf_counter() - t0

        model_path = MODEL_DIR / f"{name}_final.joblib"
        joblib.dump(model, model_path)

        # Kiểm tra tái lập và load lại trên Train+Val (không dùng Test)
        model_refit = BUILDERS[name](**cfg).fit(Xtv, ytv)
        model_loaded = joblib.load(model_path)
        proba_tv = positive_proba(model, Xtv)
        assert np.array_equal(proba_tv, positive_proba(model_refit, Xtv)), f"{name}: fit lại không tái lập"
        assert np.array_equal(proba_tv, positive_proba(model_loaded, Xtv)), f"{name}: load lại khác"
        assert np.array_equal(model.predict(Xtv), model_loaded.predict(Xtv))

        info_entry = {
            "fit_time_s": fit_time,
            "model_file": str(model_path.relative_to(ROOT)),
            "model_sha256": sha256_file(model_path)[:16],
            "n_fit": len(Xtv),
            "trainval_f1": float(f1_score(ytv, model.predict(Xtv), pos_label=1)),
        }
        if name == "DecisionTree":
            info_entry.update(dt_complexity(model))

        fit_info[name] = info_entry
        models[name] = model
        log.info(f"{name}: fit Train+Val {fit_time:.3f}s | lưu {model_path.name} | tái lập & load lại: OK")

    _verify_preprocessing_fit_on_trainval(models, Xtv, ytv)
    return models, fit_info


def _verify_preprocessing_fit_on_trainval(models: dict, Xtv, ytv) -> None:
    """Đảm bảo encoder và các mốc rời rạc hóa được học từ Train+Validation."""
    cnb_pre = models["CategoricalNB"].pre_
    cnb_enc = cnb_pre.named_transformers_["cat"].named_steps["ordinal"].enc_
    dt_ohe = models["DecisionTree"].named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]

    for col, cats_cnb, cats_dt in zip(CATEGORICAL_FEATURES, cnb_enc.categories_, dt_ohe.categories_):
        expected_cats = set(Xtv[col].fillna("Unknown"))
        assert set(cats_cnb) == expected_cats, f"CNB encoder {col} không khớp Train+Val"
        assert set(cats_dt) == expected_cats, f"DT one-hot {col} không khớp Train+Val"

    fresh_pre = build_cnb(**LOCKED_CONFIGS["CategoricalNB"]).fit(Xtv, ytv).pre_
    for edges_a, edges_b in zip(fresh_pre.named_transformers_["num"].bin_edges_, cnb_pre.named_transformers_["num"].bin_edges_):
        assert np.array_equal(edges_a, edges_b)
    log.info("Preprocessing: hạng mục và biên khoảng học từ Train+Val (không từ Test)")


def predict_test_once(models: dict, Xtv, Xte, test_df: pd.DataFrame, pred_path) -> dict:
    """Bước C: Dự đoán trên Test ĐÚNG MỘT LẦN cho mỗi mô hình và lưu kết quả."""
    pred_table = pd.DataFrame({"row_index": test_df.index})
    pred_time = {}

    for name, m in models.items():
        t0 = time.perf_counter()
        proba = positive_proba(m, Xte)
        pred_time[name] = time.perf_counter() - t0
        pred_table[f"{name}_proba_gt50K"] = proba
        pred_table[f"{name}_pred"] = (proba >= 0.5).astype(int)

    for name, model in models.items():
        same_rate = np.mean(model.predict(Xtv) == (positive_proba(model, Xtv) >= 0.5).astype(int))
        log.info(f"{name}: predict() khớp proba>=0.5 trên Train+Val: {same_rate:.6f}")

    # Nhãn thật của Test chỉ được gắn vào bảng SAU khi dự đoán xong
    pred_table.insert(1, "y_true", test_df[TARGET].to_numpy())
    pred_table.to_csv(pred_path, index=False)
    log.info(f"Đã dự đoán Test và lưu {pred_path.name} ({len(pred_table)} dòng)")
    return pred_time


def compute_test_metrics(pred_path, test_df: pd.DataFrame, models: dict, fit_info: dict, pred_time: dict):
    """Đọc lại final_predictions.csv và tính toàn bộ metric + Confusion Matrix."""
    P = pd.read_csv(pred_path)
    assert P["row_index"].tolist() == test_df.index.tolist()
    y = P["y_true"].to_numpy()

    rows, metrics, scores = [], {}, {}
    for name in models:
        pred = P[f"{name}_pred"].to_numpy()
        score = P[f"{name}_proba_gt50K"].to_numpy()
        m = compute_metrics(y, pred, score)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

        assert np.isclose(m["precision"], tp / (tp + fp))
        assert np.isclose(m["recall"], tp / (tp + fn))
        assert np.isclose(m["f1"], 2 * tp / (2 * tp + fp + fn))
        assert np.isclose(m["accuracy"], (tp + tn) / len(y)) and tn + fp + fn + tp == len(y)

        metrics[name] = m
        scores[name] = score
        rows.append({
            "model": name,
            "role": ROLES[name],
            "config": json.dumps(LOCKED_CONFIGS[name]),
            **m,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "fit_time_s": fit_info[name]["fit_time_s"],
            "test_predict_time_s": pred_time[name],
            "n_test": len(y),
            "n_fit": fit_info[name]["n_fit"],
        })
        plot_cm(name, np.array([[tn, fp], [fn, tp]]))
        m_str = ", ".join(f"{k}={v:.4f}" for k, v in m.items())
        log.info(f"TEST {name}: {m_str} | TN={tn} FP={fp} FN={fn} TP={tp}")

    pd.DataFrame(rows).to_csv(METRICS_DIR / "final_test_metrics.csv", index=False)
    log.info("Metrics tính lại từ final_predictions.csv và khớp confusion matrix")
    return P, y, metrics, scores


def run_bootstrap_and_mcnemar(P: pd.DataFrame, y: np.ndarray, models: dict, metrics: dict, scores: dict):
    """Bước D: Paired stratified bootstrap (5.000 lần) và kiểm định McNemar."""
    t0 = time.perf_counter()
    stats = {n: {"f1": [], "roc_auc": [], "pr_auc": []} for n in models}

    for idx in stratified_bootstrap_indices(y, N_BOOT, BOOT_SEED):
        yb = y[idx]
        for n in models:
            prb = P[f"{n}_pred"].to_numpy()[idx]
            scb = scores[n][idx]
            stats[n]["f1"].append(f1_score(yb, prb, pos_label=1))
            stats[n]["roc_auc"].append(roc_auc_score(yb, scb))
            stats[n]["pr_auc"].append(average_precision_score(yb, scb, pos_label=1))

    stats = {n: {k: np.asarray(v) for k, v in d.items()} for n, d in stats.items()}
    log.info(f"Bootstrap {N_BOOT} lần (seed {BOOT_SEED}) xong trong {time.perf_counter() - t0:.1f}s")

    ci_rows = []
    for n in models:
        for k in ("f1", "roc_auc", "pr_auc"):
            lo, hi = ci(stats[n][k])
            ci_rows.append({
                "model": n,
                "role": ROLES[n],
                "metric": k,
                "point_estimate": metrics[n][k],
                "bootstrap_mean": stats[n][k].mean(),
                "bootstrap_std": stats[n][k].std(ddof=1),
                "ci95_low": lo,
                "ci95_high": hi,
                "n_boot": N_BOOT,
                "seed": BOOT_SEED,
            })
    pd.DataFrame(ci_rows).to_csv(METRICS_DIR / "bootstrap_confidence_intervals.csv", index=False)

    diffs = {k: stats["DecisionTree"][k] - stats["CategoricalNB"][k] for k in ("f1", "roc_auc", "pr_auc")}
    dt_ok = P["DecisionTree_pred"].to_numpy() == y
    cnb_ok = P["CategoricalNB_pred"].to_numpy() == y
    b_ = int((dt_ok & ~cnb_ok).sum())
    c_ = int((~dt_ok & cnb_ok).sum())
    mcnemar_p = binomtest(min(b_, c_), b_ + c_, 0.5).pvalue if b_ + c_ else 1.0

    cmp_rows = []
    for k in ("f1", "roc_auc", "pr_auc"):
        lo, hi = ci(diffs[k])
        point_diff = metrics["DecisionTree"][k] - metrics["CategoricalNB"][k]
        if lo <= 0 <= hi:
            conclusion = "không đủ bằng chứng về khác biệt"
        elif lo > 0:
            conclusion = "DecisionTree cao hơn"
        else:
            conclusion = "CategoricalNB cao hơn"

        cmp_rows.append({
            "metric": k,
            "comparison": "DecisionTree - CategoricalNB",
            "point_difference": point_diff,
            "bootstrap_mean_diff": diffs[k].mean(),
            "ci95_low": lo,
            "ci95_high": hi,
            "ci_contains_0": bool(lo <= 0 <= hi),
            "share_boot_diff_gt0": float((diffs[k] > 0).mean()),
            "conclusion": conclusion,
            "n_boot": N_BOOT,
            "seed": BOOT_SEED,
        })

    cmp_rows.append({
        "metric": "accuracy (McNemar exact, bổ sung)",
        "comparison": "DecisionTree vs CategoricalNB",
        "point_difference": metrics["DecisionTree"]["accuracy"] - metrics["CategoricalNB"]["accuracy"],
        "conclusion": f"DT đúng/CNB sai={b_}, DT sai/CNB đúng={c_}, p={mcnemar_p:.4g}",
    })
    pd.DataFrame(cmp_rows).to_csv(METRICS_DIR / "paired_model_comparison.csv", index=False)

    for row in cmp_rows:
        ci_text = f"CI95 [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | " if "ci95_low" in row else ""
        log.info(f"Δ {row['metric']}: {row['point_difference']:+.4f} | {ci_text}{row['conclusion']}")

    return diffs, b_, c_, mcnemar_p


def export_summary_and_plots(env, raw_hash, fit_info, pred_time, P, y, models, metrics, scores, diffs, b_, c_, mcnemar_p):
    """Bước E: Lưu bảng so sánh Validation vs Test, vẽ các hình cuối và xuất final_run_info.json."""
    val_df = pd.read_csv(METRICS_DIR / "validation_comparison.csv", index_col=0)
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    vt_rows = []
    for n in models:
        vt_rows.append({"model": n, "set": "validation", **{k: float(val_df.loc[n, f"val_{k}"]) for k in metric_keys}})
        vt_rows.append({"model": n, "set": "test", **{k: metrics[n][k] for k in metric_keys}})

    vt_df = pd.DataFrame(vt_rows).set_index(["model", "set"])
    vt_df.to_csv(METRICS_DIR / "validation_vs_test.csv")

    plot_roc_pr(y, scores, metrics)
    plot_val_vs_test(vt_df)
    plot_bootstrap(diffs)

    assert sha256_file(RAW_PATH) == raw_hash
    agree = float((P["DecisionTree_pred"] == P["CategoricalNB_pred"]).mean())
    summary = {
        "environment": env,
        "locked_configs": LOCKED_CONFIGS,
        "fit_info": fit_info,
        "test_predict_time_s": pred_time,
        "dt_cnb_test_agreement": agree,
        "mcnemar": {"dt_right_cnb_wrong": b_, "dt_wrong_cnb_right": c_, "p_value": mcnemar_p},
        "checks": {
            "split_matches_locked_indices": True,
            "dup_groups_shared_with_test": 0,
            "preprocessing_fit_on": "Train+Validation",
            "test_predictions_per_model": 1,
            "threshold": 0.5,
            "metrics_recomputed_from_saved_predictions": True,
            "cm_metric_consistency": True,
            "final_models_reproducible": True,
            "raw_file_unchanged": True,
        },
    }
    (METRICS_DIR / "final_run_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    log.info(f"Đồng thuận dự đoán DT–CNB trên Test: {agree:.4f}")
    log.info("PHASE 5–6 HOÀN TẤT")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=UserWarning)

    pred_path = METRICS_DIR / "final_predictions.csv"
    if pred_path.exists():
        sys.exit(f"DỪNG: {pred_path} đã tồn tại – Test đã được đánh giá. Không đánh giá lại.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(timestamp)
    env = {
        "run_timestamp": timestamp,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "device": "CPU (local), 1 tiến trình",
        "random_state": RANDOM_STATE,
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "stratified": True, "paired": True},
        "threshold": 0.5,
    }
    log.info("Môi trường: " + json.dumps(env, ensure_ascii=False))
    raw_hash = sha256_file(RAW_PATH)

    trainval_df, test_df = load_locked_data()
    Xtv, ytv = xy(trainval_df)
    Xte = xy(test_df)[0]

    models, fit_info = train_and_verify_models(Xtv, ytv)
    pred_time = predict_test_once(models, Xtv, Xte, test_df, pred_path)
    P, y, metrics, scores = compute_test_metrics(pred_path, test_df, models, fit_info, pred_time)
    diffs, b_, c_, mcnemar_p = run_bootstrap_and_mcnemar(P, y, models, metrics, scores)
    export_summary_and_plots(
        env, raw_hash, fit_info, pred_time, P, y, models, metrics, scores, diffs, b_, c_, mcnemar_p
    )


if __name__ == "__main__":
    main()
