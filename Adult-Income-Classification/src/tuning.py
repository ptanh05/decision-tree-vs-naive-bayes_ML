"""PHASE 4 – Lựa chọn siêu tham số trên VALIDATION (không dùng Test).

Chạy (từ thư mục gốc dự án):
    python src/tuning.py            # chạy mới; kết quả cũ (nếu có) được lưu sang archive/
    python src/tuning.py --resume   # tiếp tục từ checkpoint

Quy tắc:
- Mỗi cấu hình: fit pipeline trên TRAIN, đánh giá trên VALIDATION.
- Ngưỡng phân loại cố định 0.5; tiêu chí chọn: F1 lớp >50K (pos_label=1).
- Phá hòa: F1 giảm dần -> PR-AUC giảm dần -> mô hình đơn giản hơn (DT: ít lá hơn) -> config_id.
- Tập Test bị loại khỏi bộ nhớ ngay sau khi chia; module này không bao giờ chạm vào Test.
"""
import argparse
import json
import logging
import platform
import shutil
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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from data_loader import RANDOM_STATE, ROOT, get_splits, xy
from decision_tree import build_dt, dt_complexity, dt_configs
from evaluation import POS_LABEL, compute_metrics, positive_proba
from naive_bayes import build_cnb, build_gnb, cnb_configs, gnb_configs

RES_DIR = ROOT / "results"
DIRS = {k: RES_DIR / k for k in ("models", "metrics", "figures", "logs", "reports")}
FIG_DIR = DIRS["figures"] / "validation"

MODELS = {
    "DecisionTree": ("chính", build_dt, dt_configs),
    "CategoricalNB": ("chính", build_cnb, cnb_configs),
    "GaussianNB": ("phụ", build_gnb, gnb_configs),
}
COLORS = {
    "DecisionTree": "#2a78d6",
    "CategoricalNB": "#eb6834",
    "GaussianNB": "#1baf7a",
}

log = logging.getLogger("phase4")


# ========================== HÀM HỖ TRỢ CƠ BẢN =================================
def setup_logging(timestamp: str) -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(DIRS["logs"] / f"phase4_tuning_{timestamp}.log", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, stream_handler):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def archive_old_results(timestamp: str, resume: bool) -> None:
    """Chuyển file CSV tuning cũ sang archive/<timestamp>/ để không ghi đè."""
    if resume or not DIRS["metrics"].exists():
        return
    old_files = list(DIRS["metrics"].glob("*_validation_grid*.csv"))
    if not old_files:
        return

    archive_dir = DIRS["metrics"] / "archive" / timestamp
    archive_dir.mkdir(parents=True)
    for old_file in old_files:
        shutil.move(str(old_file), archive_dir / old_file.name)
    log.info(f"Đã lưu trữ {len(old_files)} file kết quả cũ sang {archive_dir}")


def env_info(timestamp: str) -> dict:
    return {
        "run_timestamp": timestamp,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "random_state": RANDOM_STATE,
        "threshold": 0.5,
        "selection_metric": "f1 (pos_label=1, >50K) trên Validation",
        "device": "CPU, 1 tiến trình",
    }


def cfg_key(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, default=str)


def evaluate(model, X, y):
    t0 = time.perf_counter()
    score = positive_proba(model, X)
    pred = model.predict(X)
    predict_time = time.perf_counter() - t0
    metrics = compute_metrics(y, pred, score)
    return metrics, pred, score, predict_time


# ========================== GRID SEARCH & XẾP HẠNG ============================
def evaluate_single_config(name: str, build_fn, config_id: int, cfg: dict, Xtr, ytr, Xv, yv) -> dict:
    """Huấn luyện 1 cấu hình trên Train và chấm điểm trên Validation."""
    model = build_fn(**cfg)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0

    val_metrics, _, _, pred_time = evaluate(model, Xv, yv)
    train_metrics, _, _, _ = evaluate(model, Xtr, ytr)

    row = {
        "config_id": config_id,
        "config": cfg_key(cfg),
    }
    for param_name, param_val in cfg.items():
        row[f"param_{param_name}"] = param_val
    for metric_name, metric_val in val_metrics.items():
        row[f"val_{metric_name}"] = metric_val

    row["train_f1"] = train_metrics["f1"]
    row["train_accuracy"] = train_metrics["accuracy"]
    row["f1_gap_train_minus_val"] = train_metrics["f1"] - val_metrics["f1"]
    row["fit_time_s"] = fit_time
    row["val_predict_time_s"] = pred_time

    if name == "DecisionTree":
        row.update(dt_complexity(model))
    return row


def rank_grid(name: str, grid: pd.DataFrame) -> pd.DataFrame:
    """Xếp hạng các cấu hình theo val_f1 giảm dần, phá hòa bằng val_pr_auc và độ đơn giản."""
    sort_cols = ["val_f1", "val_pr_auc"]
    ascending = [False, False]
    if name == "DecisionTree":
        sort_cols.append("n_leaves")
        ascending.append(True)
    sort_cols.append("config_id")
    ascending.append(True)

    ranked = grid.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked.to_csv(DIRS["metrics"] / f"{name}_validation_grid_ranked.csv", index=False)
    return ranked


def run_grid(name: str, Xtr, ytr, Xv, yv, resume: bool) -> pd.DataFrame:
    """Chạy toàn bộ lưới siêu tham số cho 1 mô hình (hỗ trợ resume checkpoint)."""
    role, build_fn, get_configs = MODELS[name]
    grid_path = DIRS["metrics"] / f"{name}_validation_grid.csv"

    done_keys = set()
    if resume and grid_path.exists():
        done_keys = set(pd.read_csv(grid_path)["config"])
        log.info(f"{name}: resume – đã có {len(done_keys)} cấu hình trong checkpoint")

    configs = get_configs()
    log.info(f"{name} (mô hình {role}): {len(configs)} cấu hình")
    t_start = time.perf_counter()

    for idx, cfg in enumerate(configs):
        if cfg_key(cfg) in done_keys:
            continue

        row = evaluate_single_config(name, build_fn, idx, cfg, Xtr, ytr, Xv, yv)
        pd.DataFrame([row]).to_csv(
            grid_path,
            mode="a",
            header=not grid_path.exists(),
            index=False,
        )

        if (idx + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            log.info(f"  {name}: {idx + 1}/{len(configs)} cấu hình ({elapsed:.1f}s)")

    grid = pd.read_csv(grid_path)
    assert grid["config"].is_unique and len(grid) == len(configs), f"{name}: checkpoint không khớp lưới"
    log.info(f"{name}: xong {len(grid)} cấu hình trong {time.perf_counter() - t_start:.1f}s")
    return rank_grid(name, grid)


def time_repeated(build_fn, cfg: dict, Xtr, ytr, Xv, n_runs: int = 5) -> tuple[float, float]:
    """Đo lại thời gian fit/predict n lần rồi lấy trung vị cho ổn định."""
    fit_times, pred_times = [], []
    for _ in range(n_runs):
        model = build_fn(**cfg)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        positive_proba(model, Xv)
        model.predict(Xv)
        pred_times.append(time.perf_counter() - t0)

    return float(np.median(fit_times)), float(np.median(pred_times))


# ========================== VẼ BIỂU ĐỒ VALIDATION =============================
def plot_cm(name: str, y_true, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(f"{name} – Validation", fontweight="bold")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    fig.savefig(FIG_DIR / f"cm_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curves(scores: dict, y_true) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, score, pos_label=POS_LABEL)
        precision, recall, _ = precision_recall_curve(y_true, score, pos_label=POS_LABEL)
        metrics = compute_metrics(y_true, (score >= 0.5).astype(int), score)
        line_style = "--" if name == "GaussianNB" else "-"
        axes[0].plot(fpr, tpr, color=COLORS[name], lw=2, ls=line_style, label=f"{name} (AUC {metrics['roc_auc']:.3f})")
        axes[1].plot(recall, precision, color=COLORS[name], lw=2, ls=line_style, label=f"{name} (AP {metrics['pr_auc']:.3f})")

    axes[0].plot([0, 1], [0, 1], color="#9e9e9e", lw=1, ls=":")
    axes[1].axhline(np.mean(y_true), color="#9e9e9e", lw=1, ls=":", label=f"Ngẫu nhiên ({np.mean(y_true):.3f})")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC – Validation")
    axes[1].set(xlabel="Recall (>50K)", ylabel="Precision (>50K)", title="Precision-Recall – Validation")

    for ax, loc in zip(axes, ("lower right", "upper right")):
        ax.legend(frameon=False, loc=loc, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#ececec")

    fig.savefig(FIG_DIR / "roc_pr_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dt_depth(grid: pd.DataFrame) -> None:
    """Vẽ F1 Train vs Validation theo max_depth để quan sát overfitting của Decision Tree."""
    df = grid.copy()
    df["depth_label"] = df["param_max_depth"].fillna(-1).astype(int)
    best_by_depth = (
        df.sort_values("val_f1", ascending=False)
        .groupby("depth_label")
        .head(1)
        .sort_values("depth_label")
    )
    order = [d for d in best_by_depth["depth_label"] if d != -1]
    if -1 in set(best_by_depth["depth_label"]):
        order.append(-1)

    best_by_depth = best_by_depth.set_index("depth_label").loc[order]
    x_pos = np.arange(len(best_by_depth))
    x_labels = ["None" if d == -1 else str(d) for d in order]

    def _save_depth_plot(sub_df: pd.DataFrame, title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(x_pos, sub_df["train_f1"], "o-", color="#9e9e9e", lw=2, ms=6, label="Train F1")
        ax.plot(x_pos, sub_df["val_f1"], "o-", color=COLORS["DecisionTree"], lw=2, ms=6, label="Validation F1")
        ax.set_xticks(x_pos, x_labels)
        ax.set_xlabel("max_depth")
        ax.set_ylabel("F1 lớp >50K")
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#ececec")
        fig.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _save_depth_plot(
        best_by_depth,
        "Decision Tree: F1 theo độ sâu (tốt nhất mỗi độ sâu)",
        "dt_f1_vs_depth.png",
    )

    unconstrained = df[
        (df.param_min_samples_split == 2)
        & (df.param_min_samples_leaf == 1)
        & (df.param_class_weight.isna())
        & (df.param_criterion == "gini")
    ].set_index("depth_label").loc[order]
    _save_depth_plot(
        unconstrained,
        "Decision Tree (gini, không ràng buộc lá): overfitting theo độ sâu",
        "dt_overfitting_default_stopping.png",
    )


def plot_cnb_alpha(grid: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, fit_prior in zip(axes, [True, False]):
        sub = grid[grid.param_fit_prior == fit_prior]
        for j, n_bins in enumerate(sorted(sub.param_n_bins.unique())):
            line_df = sub[sub.param_n_bins == n_bins].sort_values("param_alpha")
            ax.plot(
                line_df.param_alpha,
                line_df.val_f1,
                "o-",
                lw=1.8,
                ms=5,
                label=f"n_bins={n_bins}",
                color=plt.cm.Oranges(0.35 + 0.15 * j),
            )
        ax.set_xscale("log")
        ax.set_xlabel("alpha (log)")
        ax.set_title(f"fit_prior={fit_prior}", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#ececec")

    axes[0].set_ylabel("Validation F1 lớp >50K")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("CategoricalNB: F1 Validation theo alpha và n_bins", fontweight="bold")
    fig.savefig(FIG_DIR / "cnb_f1_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(cmp_df: pd.DataFrame) -> None:
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    names = [n for n in cmp_df.index if n in COLORS]
    x_pos = np.arange(len(metric_keys))
    bar_width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for j, name in enumerate(names):
        vals = cmp_df.loc[name, [f"val_{m}" for m in metric_keys]].astype(float).values
        offset = (j - (len(names) - 1) / 2) * bar_width
        bars = ax.bar(
            x_pos + offset,
            vals,
            bar_width * 0.92,
            color=COLORS[name],
            label=f"{name} ({MODELS[name][0]})",
            hatch="//" if name == "GaussianNB" else None,
            edgecolor="white",
        )
        for rect, val in zip(bars, vals):
            ax.annotate(
                f"{val:.3f}",
                (rect.get_x() + rect.get_width() / 2, val),
                ha="center",
                va="bottom",
                fontsize=7,
                xytext=(0, 2),
                textcoords="offset points",
            )

    ax.set_xticks(x_pos, metric_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Giá trị trên Validation")
    ax.set_title("So sánh cấu hình tốt nhất trên Validation (lớp dương = >50K, ngưỡng 0.5)", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#ececec")
    fig.savefig(FIG_DIR / "validation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ========================== LUỒNG CHÍNH PHASE 4 ===============================
def select_and_verify_best(name: str, grid: pd.DataFrame, Xtr, ytr, Xv, yv) -> dict:
    """Lấy cấu hình đứng đầu, kiểm tra tái lập, lưu model tốt nhất trên Train."""
    role, build_fn, _ = MODELS[name]
    top = grid.iloc[0]
    cfg = json.loads(top["config"])

    model = build_fn(**cfg).fit(Xtr, ytr)
    val_metrics, pred, score, _ = evaluate(model, Xv, yv)

    # Kiểm tra tái lập: fit lại một lần nữa phải ra dự đoán và metric giống hệt
    model_check = build_fn(**cfg).fit(Xtr, ytr)
    assert np.array_equal(pred, model_check.predict(Xv)), f"{name}: dự đoán không tái lập"
    assert np.allclose(score, positive_proba(model_check, Xv), rtol=0, atol=1e-12), f"{name}: xác suất không tái lập"
    assert all(np.isclose(val_metrics[k], top[f"val_{k}"], atol=1e-12) for k in val_metrics), f"{name}: metric lệch"

    fit_t, pred_t = time_repeated(build_fn, cfg, Xtr, ytr, Xv)
    tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
    joblib.dump(model, DIRS["models"] / f"{name}_best_trainonly.joblib")

    best_entry = {
        "role": role,
        "config": cfg,
        "rank1_config_id": int(top["config_id"]),
        "n_configs": len(grid),
        "validation_metrics": {k: float(v) for k, v in val_metrics.items()},
        "train_f1": float(top["train_f1"]),
        "confusion_matrix_val": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fit_time_s_median5": fit_t,
        "val_predict_time_s_median5": pred_t,
        "reproducible": True,
    }
    if name == "DecisionTree":
        best_entry.update(dt_complexity(model))

    table_row = {
        "role": role,
        "config": json.dumps(cfg),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        "train_f1": top["train_f1"],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fit_time_s_median5": fit_t,
        "val_predict_time_s_median5": pred_t,
    }

    plot_cm(name, yv, pred)
    if name == "DecisionTree":
        plot_dt_depth(grid)
    elif name == "CategoricalNB":
        plot_cnb_alpha(grid)

    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
    log.info(f"{name} tốt nhất: {cfg} | {metrics_str} | tái lập: OK | fit {fit_t:.3f}s, predict {pred_t:.4f}s")

    return {"best_entry": best_entry, "table_row": table_row, "score": score, "pred": pred}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=UserWarning)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    archive_old_results(timestamp, args.resume)
    setup_logging(timestamp)
    env = env_info(timestamp)
    log.info("Môi trường: " + json.dumps(env, ensure_ascii=False))

    splits = get_splits()
    n_test = len(splits.pop("test"))  # BẢO VỆ TEST: loại khỏi bộ nhớ ngay lập tức
    assert set(splits) == {"train", "val"}

    Xtr, ytr = xy(splits["train"])
    Xv, yv = xy(splits["val"])
    log.info(f"Train {len(Xtr)} | Validation {len(Xv)} | Test {n_test} (không dùng trong PHASE 4)")
    log.info(f"Tỷ lệ >50K – Train {ytr.mean():.4f} | Validation {yv.mean():.4f}")

    best_configs, comparison_rows, scores, preds = {}, {}, {}, {}
    for name in MODELS:
        grid = run_grid(name, Xtr, ytr, Xv, yv, args.resume)
        outcome = select_and_verify_best(name, grid, Xtr, ytr, Xv, yv)
        best_configs[name] = outcome["best_entry"]
        comparison_rows[name] = outcome["table_row"]
        scores[name] = outcome["score"]
        preds[name] = outcome["pred"]

    # Thêm dòng mốc tham chiếu "luôn đoán <=50K"
    cmp_df = pd.DataFrame(comparison_rows).T
    always_zero = np.zeros(len(yv), dtype=int)
    ref_metrics = compute_metrics(yv, always_zero, np.zeros(len(yv)))
    cmp_df.loc["Reference: luôn đoán <=50K"] = {
        "role": "mốc tham chiếu",
        "config": "-",
        **{f"val_{k}": v for k, v in ref_metrics.items()},
        "tn": int((yv == 0).sum()),
        "fp": 0,
        "fn": int((yv == 1).sum()),
        "tp": 0,
    }
    cmp_df.to_csv(DIRS["metrics"] / "validation_comparison.csv", index_label="model")

    plot_curves(scores, yv.values)
    plot_comparison(cmp_df)

    agreement = float(np.mean(preds["DecisionTree"] == preds["CategoricalNB"]))
    output = {
        "environment": env,
        "best_configs": best_configs,
        "dt_cnb_val_prediction_agreement": agreement,
        "test_policy": (
            "Test chưa được dùng. Bước tiếp theo: refit cấu hình trên Train+Validation "
            "(27.677 mẫu), đánh giá Test (4.884 mẫu) đúng một lần cho mỗi mô hình."
        ),
    }
    (DIRS["metrics"] / "best_configs.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Đã lưu best_configs.json, validation_comparison.csv, biểu đồ và mô hình (fit trên Train)")
    log.info("PHASE 4 HOÀN TẤT – CHƯA ĐÁNH GIÁ TEST")


if __name__ == "__main__":
    main()
