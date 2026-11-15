"""PHASE 4 – Lựa chọn siêu tham số trên VALIDATION (không dùng Test).

Chạy (từ thư mục gốc dự án):
    python src/tuning.py            # chạy mới; kết quả cũ (nếu có) được lưu trữ sang archive/
    python src/tuning.py --resume   # tiếp tục từ checkpoint (bỏ qua cấu hình đã có)

Quy tắc:
- Mỗi cấu hình: fit pipeline (preprocessing + mô hình) trên TRAIN, đánh giá trên VALIDATION.
- Ngưỡng cố định 0.5 (predict mặc định). Tiêu chí chọn: F1 lớp >50K (pos_label=1).
- Phá hòa (deterministic): F1 giảm dần -> PR-AUC giảm dần -> mô hình đơn giản hơn
  (DT: ít lá hơn) -> thứ tự cấu hình trong lưới.
- Tập Test bị loại khỏi bộ nhớ ngay sau khi chia; module này không bao giờ chạm vào Test.
- Chạy tuần tự (1 tiến trình) để thời gian đo không bị nhiễu và tránh oversubscription.
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
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_curve, roc_curve)

from data_loader import RANDOM_STATE, ROOT, get_splits, xy
from decision_tree import build_dt, dt_complexity, dt_configs
from evaluation import POS_LABEL, compute_metrics, positive_proba
from naive_bayes import build_cnb, build_gnb, cnb_configs, gnb_configs

RES = ROOT / "results"
DIRS = {k: RES / k for k in ("models", "metrics", "figures", "logs", "reports")}
FIG = DIRS["figures"] / "validation"

MODELS = {
    # tên: (vai trò, hàm tạo, danh sách cấu hình)
    "DecisionTree": ("chính", build_dt, dt_configs),
    "CategoricalNB": ("chính", build_cnb, cnb_configs),
    "GaussianNB": ("phụ", build_gnb, gnb_configs),
}
C_DT, C_CNB, C_GNB = "#2a78d6", "#eb6834", "#1baf7a"
COLORS = {"DecisionTree": C_DT, "CategoricalNB": C_CNB, "GaussianNB": C_GNB}

log = logging.getLogger("phase4")


def setup_logging(ts):
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(DIRS["logs"] / f"phase4_tuning_{ts}.log", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        log.addHandler(h)


def env_info(ts):
    return {"run_timestamp": ts, "platform": platform.platform(),
            "python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
            "random_state": RANDOM_STATE, "threshold": 0.5,
            "selection_metric": "f1 (pos_label=1, >50K) trên Validation",
            "device": "CPU, 1 tiến trình"}


def cfg_key(cfg):
    return json.dumps(cfg, sort_keys=True, default=str)


def evaluate(model, X, y):
    t0 = time.perf_counter()
    score = positive_proba(model, X)
    pred = model.predict(X)
    dt = time.perf_counter() - t0
    return compute_metrics(y, pred, score), pred, score, dt


def run_grid(name, Xtr, ytr, Xv, yv, resume):
    role, build, configs = MODELS[name]
    path = DIRS["metrics"] / f"{name}_validation_grid.csv"
    done = set()
    if resume and path.exists():
        done = set(pd.read_csv(path)["config"])
        log.info(f"{name}: resume – đã có {len(done)} cấu hình trong checkpoint")
    cfgs = configs()
    log.info(f"{name} (mô hình {role}): {len(cfgs)} cấu hình")
    t_start = time.perf_counter()
    for i, cfg in enumerate(cfgs):
        key = cfg_key(cfg)
        if key in done:
            continue
        model = build(**cfg)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_time = time.perf_counter() - t0
        m_val, _, _, pred_time = evaluate(model, Xv, yv)
        m_tr, _, _, _ = evaluate(model, Xtr, ytr)
        row = {"config_id": i, "config": key, **{f"param_{k}": v for k, v in cfg.items()},
               **{f"val_{k}": v for k, v in m_val.items()},
               "train_f1": m_tr["f1"], "train_accuracy": m_tr["accuracy"],
               "f1_gap_train_minus_val": m_tr["f1"] - m_val["f1"],
               "fit_time_s": fit_time, "val_predict_time_s": pred_time}
        if name == "DecisionTree":
            row.update(dt_complexity(model))
        # checkpoint: ghi nối từng dòng
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
        if (i + 1) % 100 == 0:
            log.info(f"  {name}: {i + 1}/{len(cfgs)} cấu hình "
                     f"({time.perf_counter() - t_start:.1f}s)")
    grid = pd.read_csv(path)
    assert grid["config"].is_unique and len(grid) == len(cfgs), f"{name}: checkpoint không khớp lưới"
    log.info(f"{name}: xong {len(grid)} cấu hình trong {time.perf_counter() - t_start:.1f}s")
    return rank(name, grid)


def rank(name, grid):
    keys = ["val_f1", "val_pr_auc"]
    asc = [False, False]
    if name == "DecisionTree":
        keys.append("n_leaves"); asc.append(True)
    keys.append("config_id"); asc.append(True)
    grid = grid.sort_values(keys, ascending=asc).reset_index(drop=True)
    grid.insert(0, "rank", np.arange(1, len(grid) + 1))
    grid.to_csv(DIRS["metrics"] / f"{name}_validation_grid_ranked.csv", index=False)
    return grid


def time_repeated(model_fn, Xtr, ytr, Xv, n=5):
    """Đo lại thời gian của cấu hình tốt nhất n lần, lấy trung vị (ổn định hơn 1 lần đo)."""
    fits, preds = [], []
    for _ in range(n):
        m = model_fn()
        t0 = time.perf_counter(); m.fit(Xtr, ytr); fits.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); positive_proba(m, Xv); m.predict(Xv)
        preds.append(time.perf_counter() - t0)
    return float(np.median(fits)), float(np.median(preds))


# ------------------------------- BIỂU ĐỒ -----------------------------------
def plot_cm(name, y, pred):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1]),
                           display_labels=["<=50K", ">50K"]).plot(ax=ax, cmap="Blues",
                                                                  colorbar=False, values_format="d")
    ax.set_title(f"{name} – Validation", fontweight="bold")
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    fig.savefig(FIG / f"cm_{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_curves(scores, y):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s, pos_label=POS_LABEL)
        p, r, _ = precision_recall_curve(y, s, pos_label=POS_LABEL)
        m = compute_metrics(y, (s >= 0.5).astype(int), s)
        ls = "--" if name == "GaussianNB" else "-"
        axes[0].plot(fpr, tpr, color=COLORS[name], lw=2, ls=ls, label=f"{name} (AUC {m['roc_auc']:.3f})")
        axes[1].plot(r, p, color=COLORS[name], lw=2, ls=ls, label=f"{name} (AP {m['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#9e9e9e", lw=1, ls=":")
    axes[1].axhline(np.mean(y), color="#9e9e9e", lw=1, ls=":", label=f"Ngẫu nhiên ({np.mean(y):.3f})")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC – Validation")
    axes[1].set(xlabel="Recall (>50K)", ylabel="Precision (>50K)", title="Precision-Recall – Validation")
    for ax in axes:
        ax.legend(frameon=False, loc="lower right" if ax is axes[0] else "upper right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#ececec")
    fig.savefig(FIG / "roc_pr_validation.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_dt_depth(grid):
    """F1 Train vs Validation theo max_depth (mỗi độ sâu: cấu hình tốt nhất theo val F1)."""
    g = grid.copy()
    g["depth_label"] = g["param_max_depth"].fillna(-1).astype(int)
    best = g.sort_values("val_f1", ascending=False).groupby("depth_label").head(1).sort_values("depth_label")
    order = [d for d in best["depth_label"] if d != -1] + ([-1] if -1 in set(best["depth_label"]) else [])
    best = best.set_index("depth_label").loc[order]
    x = np.arange(len(best)); lbl = ["None" if d == -1 else str(d) for d in order]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(x, best["train_f1"], "o-", color="#9e9e9e", lw=2, ms=6, label="Train F1")
    ax.plot(x, best["val_f1"], "o-", color=C_DT, lw=2, ms=6, label="Validation F1")
    ax.set_xticks(x, lbl); ax.set_xlabel("max_depth"); ax.set_ylabel("F1 lớp >50K")
    ax.set_title("Decision Tree: F1 theo độ sâu (tốt nhất mỗi độ sâu)", fontweight="bold")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#ececec")
    fig.savefig(FIG / "dt_f1_vs_depth.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    # Cùng cấu hình điều kiện dừng mặc định (split=2, leaf=1) để thấy overfitting thuần
    raw = g[(g.param_min_samples_split == 2) & (g.param_min_samples_leaf == 1)
            & (g.param_class_weight.isna()) & (g.param_criterion == "gini")]
    raw = raw.set_index("depth_label").loc[order]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(x, raw["train_f1"], "o-", color="#9e9e9e", lw=2, ms=6, label="Train F1")
    ax.plot(x, raw["val_f1"], "o-", color=C_DT, lw=2, ms=6, label="Validation F1")
    ax.set_xticks(x, lbl); ax.set_xlabel("max_depth"); ax.set_ylabel("F1 lớp >50K")
    ax.set_title("Decision Tree (gini, không ràng buộc lá): overfitting theo độ sâu", fontweight="bold")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#ececec")
    fig.savefig(FIG / "dt_overfitting_default_stopping.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_cnb_alpha(grid):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, fp in zip(axes, [True, False]):
        g = grid[grid.param_fit_prior == fp]
        for j, nb in enumerate(sorted(g.param_n_bins.unique())):
            s = g[g.param_n_bins == nb].sort_values("param_alpha")
            ax.plot(s.param_alpha, s.val_f1, "o-", lw=1.8, ms=5, label=f"n_bins={nb}",
                    color=plt.cm.Oranges(0.35 + 0.15 * j))
        ax.set_xscale("log"); ax.set_xlabel("alpha (log)")
        ax.set_title(f"fit_prior={fp}", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#ececec")
    axes[0].set_ylabel("Validation F1 lớp >50K"); axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("CategoricalNB: F1 Validation theo alpha và n_bins", fontweight="bold")
    fig.savefig(FIG / "cnb_f1_vs_alpha.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_comparison(cmp):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    names = [n for n in cmp.index if n in COLORS]
    x = np.arange(len(metrics)); w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for j, n in enumerate(names):
        vals = cmp.loc[n, [f"val_{m}" for m in metrics]].astype(float).values
        b = ax.bar(x + (j - (len(names) - 1) / 2) * w, vals, w * 0.92, color=COLORS[n],
                   label=f"{n} ({MODELS[n][0]})", hatch="//" if n == "GaussianNB" else None,
                   edgecolor="white")
        for rect, v in zip(b, vals):
            ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("Giá trị trên Validation")
    ax.set_title("So sánh cấu hình tốt nhất trên Validation (lớp dương = >50K, ngưỡng 0.5)",
                 fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#ececec")
    fig.savefig(FIG / "validation_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------- MAIN ------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=UserWarning)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Không ghi đè kết quả cũ: lưu trữ sang archive/<ts>/ nếu chạy mới
    old = list(DIRS["metrics"].glob("*_validation_grid*.csv")) if DIRS["metrics"].exists() else []
    if old and not args.resume:
        arch = DIRS["metrics"] / "archive" / ts
        arch.mkdir(parents=True)
        for f in old:
            shutil.move(str(f), arch / f.name)
    setup_logging(ts)
    env = env_info(ts)
    log.info("Môi trường: " + json.dumps(env, ensure_ascii=False))
    if old and not args.resume:
        log.info(f"Đã lưu trữ {len(old)} file kết quả cũ sang {arch}")

    splits = get_splits()
    n_test = len(splits.pop("test"))  # BẢO VỆ TEST: loại khỏi bộ nhớ ngay lập tức
    assert set(splits) == {"train", "val"}
    Xtr, ytr = xy(splits["train"]); Xv, yv = xy(splits["val"])
    log.info(f"Train {len(Xtr)} | Validation {len(Xv)} | Test {n_test} (không dùng trong PHASE 4)")
    log.info(f"Tỷ lệ >50K – Train {ytr.mean():.4f} | Validation {yv.mean():.4f}")

    best, best_rows, scores, preds = {}, {}, {}, {}
    for name in MODELS:
        grid = run_grid(name, Xtr, ytr, Xv, yv, args.resume)
        top = grid.iloc[0]
        cfg = json.loads(top["config"])
        build = MODELS[name][1]
        model = build(**cfg).fit(Xtr, ytr)
        m, pred, score, _ = evaluate(model, Xv, yv)
        # Kiểm tra tái lập: fit lại cho đúng dự đoán & metric như lúc chạy lưới
        model2 = build(**cfg).fit(Xtr, ytr)
        same_pred = np.array_equal(pred, model2.predict(Xv))
        same_prob = np.allclose(score, positive_proba(model2, Xv), rtol=0, atol=1e-12)
        same_metric = all(np.isclose(m[k], top[f"val_{k}"], atol=1e-12) for k in m)
        assert same_pred and same_prob and same_metric, f"{name}: không tái lập"
        fit_t, pred_t = time_repeated(lambda: build(**cfg), Xtr, ytr, Xv)
        tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
        joblib.dump(model, DIRS["models"] / f"{name}_best_trainonly.joblib")
        best[name] = {"role": MODELS[name][0], "config": cfg, "rank1_config_id": int(top["config_id"]),
                      "n_configs": len(grid),
                      "validation_metrics": {k: float(v) for k, v in m.items()},
                      "train_f1": float(top["train_f1"]),
                      "confusion_matrix_val": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                      "fit_time_s_median5": fit_t, "val_predict_time_s_median5": pred_t,
                      "reproducible": True}
        if name == "DecisionTree":
            best[name].update(dt_complexity(model))
        best_rows[name] = {"role": MODELS[name][0], "config": json.dumps(cfg),
                           **{f"val_{k}": v for k, v in m.items()},
                           "train_f1": top["train_f1"], "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                           "fit_time_s_median5": fit_t, "val_predict_time_s_median5": pred_t}
        scores[name], preds[name] = score, pred
        plot_cm(name, yv, pred)
        log.info(f"{name} tốt nhất: {cfg} | " + ", ".join(f"{k}={v:.4f}" for k, v in m.items())
                 + f" | tái lập: OK | fit {fit_t:.3f}s, predict {pred_t:.4f}s (trung vị 5 lần)")
        if name == "DecisionTree":
            plot_dt_depth(grid)
        if name == "CategoricalNB":
            plot_cnb_alpha(grid)

    # Bảng so sánh Validation (+ mốc tham chiếu: luôn đoán lớp đa số – KHÔNG phải mô hình)
    cmp = pd.DataFrame(best_rows).T
    maj = np.zeros(len(yv), dtype=int)
    ref = compute_metrics(yv, maj, np.zeros(len(yv)))
    cmp.loc["Reference: luôn đoán <=50K"] = {"role": "mốc tham chiếu", "config": "-",
                                             **{f"val_{k}": v for k, v in ref.items()},
                                             "tn": int((yv == 0).sum()), "fp": 0,
                                             "fn": int((yv == 1).sum()), "tp": 0}
    cmp.to_csv(DIRS["metrics"] / "validation_comparison.csv", index_label="model")
    plot_curves(scores, yv.values)
    plot_comparison(cmp)

    # Mức độ đồng thuận dự đoán giữa 2 mô hình chính (phục vụ phân tích sai số sau này)
    agree = float(np.mean(preds["DecisionTree"] == preds["CategoricalNB"]))

    out = {"environment": env, "best_configs": best,
           "dt_cnb_val_prediction_agreement": agree,
           "test_policy": "Test chưa được dùng. Bước tiếp theo: refit cấu hình trên Train+Validation "
                          "(27.677 mẫu), đánh giá Test (4.884 mẫu) đúng một lần cho mỗi mô hình."}
    (DIRS["metrics"] / "best_configs.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info("Đã lưu best_configs.json, validation_comparison.csv, biểu đồ và mô hình (fit trên Train)")
    log.info("PHASE 4 HOÀN TẤT – CHƯA ĐÁNH GIÁ TEST")


if __name__ == "__main__":
    main()
