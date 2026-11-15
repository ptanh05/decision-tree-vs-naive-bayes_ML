"""PHASE 5–6 – Huấn luyện cuối trên Train+Validation và đánh giá Test MỘT LẦN.

Chạy (từ thư mục gốc dự án):  python src/final_evaluation.py

Bảo vệ Test:
- Cấu hình bị KHÓA (LOCKED_CONFIGS) và phải trùng với results/metrics/best_configs.json.
- Preprocessing + mô hình chỉ fit trên Train+Validation (27.677 mẫu).
- Mỗi mô hình gọi dự đoán trên Test đúng MỘT lần; ngưỡng cố định 0,5.
- Nếu results/metrics/final_predictions.csv đã tồn tại, script dừng (không đánh giá lại Test).
- Mọi kiểm tra tái lập / load lại mô hình dùng dữ liệu Train+Validation, không dùng Test.
- Bootstrap chỉ ước lượng độ bất định của dự đoán ĐÃ KHÓA (không fit lại gì).
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
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from data_loader import (CATEGORICAL_FEATURES, RANDOM_STATE, RAW_PATH, ROOT, TARGET, clean,
                         duplicate_group_ids, load_raw, split_data, xy)
from decision_tree import build_dt, dt_complexity
from evaluation import POS_LABEL, compute_metrics, positive_proba
from naive_bayes import build_cnb, build_gnb

RES = ROOT / "results"
MODEL_DIR = RES / "models" / "final"
METRICS = RES / "metrics"
FIG = RES / "figures" / "final"
LOGS = RES / "logs"
REPORTS = RES / "reports"

LOCKED_CONFIGS = {
    "DecisionTree": {"criterion": "entropy", "max_depth": 12, "min_samples_leaf": 10,
                     "min_samples_split": 2, "class_weight": "balanced"},
    "CategoricalNB": {"n_bins": 20, "alpha": 0.1, "fit_prior": True},
    "GaussianNB": {"var_smoothing": 1e-7},
}
BUILDERS = {"DecisionTree": build_dt, "CategoricalNB": build_cnb, "GaussianNB": build_gnb}
ROLES = {"DecisionTree": "chính", "CategoricalNB": "chính", "GaussianNB": "phụ"}
MAIN = ["DecisionTree", "CategoricalNB"]
COLORS = {"DecisionTree": "#2a78d6", "CategoricalNB": "#eb6834", "GaussianNB": "#1baf7a"}
N_BOOT = 5000
BOOT_SEED = 42
EXPECTED = {"trainval": 27677, "test": 4884, "test_pos": 1176}

log = logging.getLogger("final")


def setup(ts):
    for d in (MODEL_DIR, METRICS, FIG, LOGS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.FileHandler(LOGS / f"phase5_6_final_{ts}.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# --------------------------- BOOTSTRAP -------------------------------------
def stratified_bootstrap_indices(y, n_boot, seed):
    """Mỗi lần lặp: lấy có hoàn lại TRONG từng lớp, giữ nguyên số mẫu mỗi lớp."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(neg, len(neg), replace=True),
                              rng.choice(pos, len(pos), replace=True)])


def ci(a, level=0.95):
    lo, hi = np.percentile(a, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


# ------------------------------ BIỂU ĐỒ ------------------------------------
def plot_cm(name, cm):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{name} – Test ({ROLES[name]})", fontweight="bold")
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    fig.savefig(FIG / f"cm_test_{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_roc_pr(y, scores, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, s in scores.items():
        ls = "--" if ROLES[name] == "phụ" else "-"
        fpr, tpr, _ = roc_curve(y, s, pos_label=POS_LABEL)
        p, r, _ = precision_recall_curve(y, s, pos_label=POS_LABEL)
        axes[0].plot(fpr, tpr, color=COLORS[name], lw=2, ls=ls,
                     label=f"{name} (AUC {metrics[name]['roc_auc']:.3f})")
        axes[1].plot(r, p, color=COLORS[name], lw=2, ls=ls,
                     label=f"{name} (AP {metrics[name]['pr_auc']:.3f})")
        # đánh dấu điểm làm việc tại ngưỡng 0,5
        axes[1].plot(metrics[name]["recall"], metrics[name]["precision"], "o", ms=8,
                     color=COLORS[name], mec="white", mew=2)
    axes[0].plot([0, 1], [0, 1], color="#9e9e9e", lw=1, ls=":")
    axes[1].axhline(y.mean(), color="#9e9e9e", lw=1, ls=":", label=f"Ngẫu nhiên ({y.mean():.3f})")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC – Test")
    axes[1].set(xlabel="Recall (>50K)", ylabel="Precision (>50K)",
                title="Precision-Recall – Test (chấm = ngưỡng 0,5)")
    for ax, loc in zip(axes, ["lower right", "upper right"]):
        ax.legend(frameon=False, loc=loc, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#ececec")
    fig.savefig(FIG / "roc_pr_test.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_val_vs_test(tbl):
    mets = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    x = np.arange(len(mets))
    for ax, name in zip(axes, MAIN):
        v = [tbl.loc[(name, "validation"), m] for m in mets]
        t = [tbl.loc[(name, "test"), m] for m in mets]
        ax.bar(x - 0.2, v, 0.38, color="#c9c9c9", label="Validation (fit trên Train)")
        b = ax.bar(x + 0.2, t, 0.38, color=COLORS[name], label="Test (fit trên Train+Val)")
        for rect, val in zip(b, t):
            ax.annotate(f"{val:.3f}", (rect.get_x() + rect.get_width() / 2, val), ha="center",
                        va="bottom", fontsize=7.5, xytext=(0, 2), textcoords="offset points")
        ax.set_xticks(x, labels, fontsize=9); ax.set_title(name, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#ececec")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Giá trị")
    fig.suptitle("Validation và Test – khả năng tổng quát hóa", fontweight="bold")
    fig.savefig(FIG / "validation_vs_test.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_bootstrap(boot, diffs):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, m, title in zip(axes, ["f1", "roc_auc", "pr_auc"], ["F1 (>50K)", "ROC-AUC", "PR-AUC"]):
        d = diffs[m]
        lo, hi = ci(d)
        ax.hist(d, bins=60, color="#7aa7e0", edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="#2b2b2b", lw=1.5)
        for v in (lo, hi):
            ax.axvline(v, color="#eb6834", lw=1.5, ls="--")
        ax.set_title(f"Δ {title}: DT − CNB\nCI95% [{lo:+.4f}, {hi:+.4f}]", fontsize=10,
                     fontweight="bold")
        ax.set_xlabel("Chênh lệch"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"Số lần lặp (/{N_BOOT})")
    fig.suptitle("Paired stratified bootstrap trên Test (đường đứt = CI 95%, đường đen = 0)",
                 fontweight="bold", y=1.04)
    fig.savefig(FIG / "bootstrap_differences.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# -------------------------------- MAIN -------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=UserWarning)
    pred_path = METRICS / "final_predictions.csv"
    if pred_path.exists():
        sys.exit(f"DỪNG: {pred_path} đã tồn tại – Test đã được đánh giá. Không đánh giá lại.")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup(ts)
    env = {"run_timestamp": ts, "platform": platform.platform(), "python": sys.version.split()[0],
           "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
           "device": "CPU (local), 1 tiến trình", "random_state": RANDOM_STATE,
           "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "stratified": True, "paired": True},
           "threshold": 0.5}
    log.info("Môi trường: " + json.dumps(env, ensure_ascii=False))
    raw_hash = sha256_file(RAW_PATH)

    # ---- A. Khóa cấu hình ----
    best = json.loads((METRICS / "best_configs.json").read_text(encoding="utf-8"))["best_configs"]
    for name, cfg in LOCKED_CONFIGS.items():
        assert best[name]["config"] == cfg, f"{name}: cấu hình khác PHASE 4"
    log.info("Cấu hình khóa trùng với best_configs.json (PHASE 4)")

    # ---- Dữ liệu & split đã khóa ----
    df = clean(load_raw())
    s = split_data(df)
    ref = pd.read_csv(ROOT / "data" / "processed" / "split_indices.csv")
    for k, v in s.items():
        assert set(ref.loc[ref.split == k, "row_index"]) == set(v.index), f"split {k} lệch"
    trval = pd.concat([s["train"], s["val"]]).sort_index()
    test = s["test"]
    groups = duplicate_group_ids(df)
    shared = set(groups.loc[trval.index]) & set(groups.loc[test.index])
    assert not shared, "Nhóm trùng lặp dùng chung giữa Train+Val và Test"
    assert len(trval) == EXPECTED["trainval"] and len(test) == EXPECTED["test"]
    assert int(test[TARGET].sum()) == EXPECTED["test_pos"]
    log.info(f"Split khớp split_indices.csv | Train+Val {len(trval)} | Test {len(test)} "
             f"(>50K: {int(test[TARGET].sum())}) | nhóm trùng dùng chung: 0")
    Xtv, ytv = xy(trval)
    Xte = xy(test)[0]           # nhãn Test chỉ được lấy SAU khi dự đoán xong

    # ---- B. Final training (chỉ Train+Val) ----
    models, fit_info = {}, {}
    for name, cfg in LOCKED_CONFIGS.items():
        t0 = time.perf_counter()
        m = BUILDERS[name](**cfg).fit(Xtv, ytv)
        fit_time = time.perf_counter() - t0
        path = MODEL_DIR / f"{name}_final.joblib"
        joblib.dump(m, path)
        # Tái lập + load lại: so sánh trên Train+Val (KHÔNG dùng Test)
        m2 = BUILDERS[name](**cfg).fit(Xtv, ytv)
        loaded = joblib.load(path)
        p_ref = positive_proba(m, Xtv)
        assert np.array_equal(p_ref, positive_proba(m2, Xtv)), f"{name}: fit lại không tái lập"
        assert np.array_equal(p_ref, positive_proba(loaded, Xtv)), f"{name}: load lại khác"
        assert np.array_equal(m.predict(Xtv), loaded.predict(Xtv))
        fit_info[name] = {"fit_time_s": fit_time, "model_file": str(path.relative_to(ROOT)),
                          "model_sha256": sha256_file(path)[:16], "n_fit": len(Xtv),
                          "trainval_f1": float(f1_score(ytv, m.predict(Xtv), pos_label=1))}
        if name == "DecisionTree":
            fit_info[name].update(dt_complexity(m))
        models[name] = m
        log.info(f"{name}: fit Train+Val {fit_time:.3f}s | lưu {path.name} | tái lập & load lại: OK")

    # Kiểm tra preprocessing chỉ học từ Train+Val
    cnb_pre = models["CategoricalNB"].pre_
    enc = cnb_pre.named_transformers_["cat"].named_steps["ordinal"].enc_
    for c, cats in zip(CATEGORICAL_FEATURES, enc.categories_):
        assert set(cats) == set(Xtv[c].fillna("Unknown")), f"CNB encoder {c} không khớp Train+Val"
    ohe = models["DecisionTree"].named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
    for c, cats in zip(CATEGORICAL_FEATURES, ohe.categories_):
        assert set(cats) == set(Xtv[c].fillna("Unknown")), f"DT one-hot {c} không khớp Train+Val"
    fresh = build_cnb(**LOCKED_CONFIGS["CategoricalNB"]).fit(Xtv, ytv).pre_
    for a, b in zip(fresh.named_transformers_["num"].bin_edges_,
                    cnb_pre.named_transformers_["num"].bin_edges_):
        assert np.array_equal(a, b)
    log.info("Preprocessing: hạng mục và biên khoảng học từ Train+Val (không từ Test)")

    # ---- C. Dự đoán Test – ĐÚNG MỘT LẦN mỗi mô hình ----
    pred_tbl = pd.DataFrame({"row_index": test.index})
    pred_time = {}
    for name, m in models.items():
        t0 = time.perf_counter()
        proba = positive_proba(m, Xte)                 # lần duy nhất chạm vào Test
        pred_time[name] = time.perf_counter() - t0
        pred_tbl[f"{name}_proba_gt50K"] = proba
        pred_tbl[f"{name}_pred"] = (proba >= 0.5).astype(int)   # ngưỡng cố định 0,5
    # Ghi nhận (không assert) mức khớp giữa predict() và proba>=0.5 trên Train+Val
    for name, m in models.items():
        same = np.mean(m.predict(Xtv) == (positive_proba(m, Xtv) >= 0.5).astype(int))
        log.info(f"{name}: predict() khớp proba>=0.5 trên Train+Val: {same:.6f}")
    yte = test[TARGET].to_numpy()                     # đọc nhãn Test sau khi đã dự đoán
    pred_tbl.insert(1, "y_true", yte)
    pred_tbl.to_csv(pred_path, index=False)
    log.info(f"Đã dự đoán Test và lưu {pred_path.name} ({len(pred_tbl)} dòng)")

    # ---- Metrics tính LẠI từ file dự đoán đã lưu ----
    P = pd.read_csv(pred_path)
    assert P["row_index"].tolist() == test.index.tolist()
    y = P["y_true"].to_numpy()
    rows, metrics, scores = [], {}, {}
    for name in models:
        pr, sc = P[f"{name}_pred"].to_numpy(), P[f"{name}_proba_gt50K"].to_numpy()
        m = compute_metrics(y, pr, sc)
        tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
        # Nhất quán CM <-> metric
        assert np.isclose(m["precision"], tp / (tp + fp)) and np.isclose(m["recall"], tp / (tp + fn))
        assert np.isclose(m["f1"], 2 * tp / (2 * tp + fp + fn))
        assert np.isclose(m["accuracy"], (tp + tn) / len(y)) and tn + fp + fn + tp == len(y)
        metrics[name], scores[name] = m, sc
        rows.append({"model": name, "role": ROLES[name], "config": json.dumps(LOCKED_CONFIGS[name]),
                     **m, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                     "fit_time_s": fit_info[name]["fit_time_s"],
                     "test_predict_time_s": pred_time[name], "n_test": len(y),
                     "n_fit": fit_info[name]["n_fit"]})
        plot_cm(name, np.array([[tn, fp], [fn, tp]]))
        log.info(f"TEST {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items())
                 + f" | TN={tn} FP={fp} FN={fn} TP={tp}")
    final = pd.DataFrame(rows)
    final.to_csv(METRICS / "final_test_metrics.csv", index=False)
    log.info("Metrics tính lại từ final_predictions.csv và khớp confusion matrix")

    # Độ nhạy (không thay đổi mô hình): Test bỏ các dòng có nội dung đặc trưng
    # trùng với Train+Val nhưng khác/không khác nhãn – đã 0 nhóm trùng đầy đủ.
    # ---- D. Paired stratified bootstrap ----
    t0 = time.perf_counter()
    stats = {n: {"f1": [], "roc_auc": [], "pr_auc": []} for n in models}
    for idx in stratified_bootstrap_indices(y, N_BOOT, BOOT_SEED):
        yb = y[idx]
        for n in models:
            prb, scb = P[f"{n}_pred"].to_numpy()[idx], scores[n][idx]
            stats[n]["f1"].append(f1_score(yb, prb, pos_label=1))
            stats[n]["roc_auc"].append(roc_auc_score(yb, scb))
            stats[n]["pr_auc"].append(average_precision_score(yb, scb, pos_label=1))
    stats = {n: {k: np.asarray(v) for k, v in d.items()} for n, d in stats.items()}
    log.info(f"Bootstrap {N_BOOT} lần (seed {BOOT_SEED}) xong trong {time.perf_counter() - t0:.1f}s")
    ci_rows = []
    for n in models:
        for k in ("f1", "roc_auc", "pr_auc"):
            lo, hi = ci(stats[n][k])
            ci_rows.append({"model": n, "role": ROLES[n], "metric": k, "point_estimate": metrics[n][k],
                            "bootstrap_mean": stats[n][k].mean(), "bootstrap_std": stats[n][k].std(ddof=1),
                            "ci95_low": lo, "ci95_high": hi, "n_boot": N_BOOT, "seed": BOOT_SEED})
    pd.DataFrame(ci_rows).to_csv(METRICS / "bootstrap_confidence_intervals.csv", index=False)

    diffs = {k: stats["DecisionTree"][k] - stats["CategoricalNB"][k] for k in ("f1", "roc_auc", "pr_auc")}
    dt_ok = P["DecisionTree_pred"].to_numpy() == y
    cnb_ok = P["CategoricalNB_pred"].to_numpy() == y
    b_, c_ = int((dt_ok & ~cnb_ok).sum()), int((~dt_ok & cnb_ok).sum())
    mcn = binomtest(min(b_, c_), b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
    cmp_rows = []
    for k in ("f1", "roc_auc", "pr_auc"):
        lo, hi = ci(diffs[k])
        point = metrics["DecisionTree"][k] - metrics["CategoricalNB"][k]
        cmp_rows.append({"metric": k, "comparison": "DecisionTree - CategoricalNB",
                         "point_difference": point, "bootstrap_mean_diff": diffs[k].mean(),
                         "ci95_low": lo, "ci95_high": hi, "ci_contains_0": bool(lo <= 0 <= hi),
                         "share_boot_diff_gt0": float((diffs[k] > 0).mean()),
                         "conclusion": ("không đủ bằng chứng về khác biệt" if lo <= 0 <= hi else
                                        ("DecisionTree cao hơn" if lo > 0 else "CategoricalNB cao hơn")),
                         "n_boot": N_BOOT, "seed": BOOT_SEED})
    cmp_rows.append({"metric": "accuracy (McNemar exact, bổ sung)",
                     "comparison": "DecisionTree vs CategoricalNB",
                     "point_difference": metrics["DecisionTree"]["accuracy"] - metrics["CategoricalNB"]["accuracy"],
                     "conclusion": f"DT đúng/CNB sai={b_}, DT sai/CNB đúng={c_}, p={mcn:.4g}"})
    pd.DataFrame(cmp_rows).to_csv(METRICS / "paired_model_comparison.csv", index=False)
    for r in cmp_rows:
        log.info(f"Δ {r['metric']}: {r['point_difference']:+.4f} | "
                 + (f"CI95 [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | " if "ci95_low" in r else "")
                 + r["conclusion"])

    # ---- Validation vs Test ----
    val = pd.read_csv(METRICS / "validation_comparison.csv", index_col=0)
    mets = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    vt = []
    for n in models:
        vt.append({"model": n, "set": "validation", **{k: float(val.loc[n, f"val_{k}"]) for k in mets}})
        vt.append({"model": n, "set": "test", **{k: metrics[n][k] for k in mets}})
    vt = pd.DataFrame(vt).set_index(["model", "set"])
    vt.to_csv(METRICS / "validation_vs_test.csv")

    plot_roc_pr(y, scores, metrics)
    plot_val_vs_test(vt)
    plot_bootstrap(stats, diffs)

    # ---- E. Kiểm tra cuối ----
    assert sha256_file(RAW_PATH) == raw_hash
    agree = float((P["DecisionTree_pred"] == P["CategoricalNB_pred"]).mean())
    summary = {"environment": env, "locked_configs": LOCKED_CONFIGS, "fit_info": fit_info,
               "test_predict_time_s": pred_time, "dt_cnb_test_agreement": agree,
               "mcnemar": {"dt_right_cnb_wrong": b_, "dt_wrong_cnb_right": c_, "p_value": mcn},
               "checks": {"split_matches_locked_indices": True, "dup_groups_shared_with_test": 0,
                          "preprocessing_fit_on": "Train+Validation", "test_predictions_per_model": 1,
                          "threshold": 0.5, "metrics_recomputed_from_saved_predictions": True,
                          "cm_metric_consistency": True, "final_models_reproducible": True,
                          "raw_file_unchanged": True}}
    (METRICS / "final_run_info.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                            default=str), encoding="utf-8")
    log.info(f"Đồng thuận dự đoán DT–CNB trên Test: {agree:.4f}")
    log.info("PHASE 5–6 HOÀN TẤT")


if __name__ == "__main__":
    main()
