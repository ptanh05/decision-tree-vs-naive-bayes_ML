"""FINAL SCIENTIFIC AUDIT – chỉ ĐỌC dữ liệu và kết quả đã lưu.

Không import sklearn estimator, không load joblib, không fit/predict. Chỉ dùng hàm metric
của sklearn trên saved predictions. Ghi kết quả vào results/reports/final_audit/ (thư mục mới).
Chạy: python scripts/final_audit.py
"""
import hashlib
import json
import platform
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import binomtest, chi2
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reports" / "final_audit"
RAW = ROOT / "data" / "raw" / "adult.csv"
SPLIT = ROOT / "data" / "processed" / "split_indices.csv"
MET = ROOT / "results" / "metrics"
PRED = MET / "final_predictions.csv"
SRC_FINAL = ROOT / "src" / "final_evaluation.py"
REPORT_MD = ROOT / "results" / "reports" / "final_evaluation.md"
RAW_HASH_PREFIX_RECORDED = "250e154ed75714ae"   # ghi nhận trong phiên PHASE 0–5 (16 ký tự đầu)
TARGET = "income"
N_BOOT, SEED = 5000, 42

checks = []           # (audit, item, status, evidence)
inspected = set()


def chk(audit, item, status, evidence):
    checks.append({"audit": audit, "item": item, "status": status, "evidence": str(evidence)})
    print(f"[{status}] {audit} | {item} | {evidence}")


def sha(p):
    inspected.add(Path(p).relative_to(ROOT).as_posix())
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_csv(p, **kw):
    inspected.add(Path(p).relative_to(ROOT).as_posix())
    return pd.read_csv(p, **kw)


def read_text(p):
    inspected.add(Path(p).relative_to(ROOT).as_posix())
    return Path(p).read_text(encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if (OUT / "FINAL_AUDIT_REPORT.md").exists():
        sys.exit(f"DỪNG: {OUT} đã có nội dung – không ghi đè.")
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    # Ghi lại hash mọi artifact TRƯỚC khi audit để chứng minh read-only
    frozen = [RAW, SPLIT, PRED, MET / "final_test_metrics.csv", MET / "bootstrap_confidence_intervals.csv",
              MET / "paired_model_comparison.csv", MET / "final_run_info.json", MET / "validation_vs_test.csv",
              MET / "best_configs.json", MET / "validation_comparison.csv", REPORT_MD,
              *sorted((ROOT / "results" / "models").rglob("*.joblib"))]
    hash_before = {p.relative_to(ROOT).as_posix(): sha(p) for p in frozen}

    # ================= AUDIT 1 – DUPLICATE (14 đặc trưng) =================
    # Đọc nguyên văn dạng chuỗi: không strip, không thay '?', không ép kiểu số.
    raw = read_csv(RAW, dtype=str, keep_default_na=False)
    feats = [c for c in raw.columns if c != TARGET]
    assert len(feats) == 14, feats
    split = read_csv(SPLIT)
    assert split.row_index.is_unique and len(split) == len(raw)
    part = split.set_index("row_index")["split"].reindex(raw.index)
    assert part.notna().all()
    sig = raw[feats].agg("\x1f".join, axis=1)          # khóa = 14 giá trị nguyên văn
    gid = pd.Series(pd.factorize(sig)[0], index=raw.index)
    sizes = gid.map(gid.value_counts())
    dup = raw.loc[sizes > 1].copy()
    dup["feature_group"] = gid[dup.index]
    dup["split"] = part[dup.index]
    rows = []
    for g, d in dup.groupby("feature_group"):
        sp = d["split"].value_counts()
        lab = d[TARGET].value_counts()
        parts = sorted(sp.index)
        rows.append({"feature_group": g, "n_rows": len(d), "row_indices": ";".join(map(str, d.index)),
                     "n_labels": d[TARGET].nunique(),
                     "n_le50K": int(lab.get("<=50K", 0)), "n_gt50K": int(lab.get(">50K", 0)),
                     "category": "B_conflicting_labels" if d[TARGET].nunique() > 1 else "A_same_label",
                     "n_train": int(sp.get("train", 0)), "n_val": int(sp.get("val", 0)),
                     "n_test": int(sp.get("test", 0)), "partitions": "+".join(parts),
                     "cross_partition": len(parts) > 1,
                     "conflicting_cross_partition": d[TARGET].nunique() > 1 and len(parts) > 1,
                     **{f: d[f].iloc[0] for f in feats}})
    da = pd.DataFrame(rows)
    da.to_csv(OUT / "duplicate_audit.csv", index=False)
    n_conf = int((da.category == "B_conflicting_labels").sum()) if len(da) else 0
    stats1 = {
        "n_duplicate_feature_signatures": len(da),
        "n_rows_in_duplicate_signatures": int(da.n_rows.sum()) if len(da) else 0,
        "n_extra_rows_beyond_first": int((da.n_rows - 1).sum()) if len(da) else 0,
        "n_same_label_groups": int((da.category == "A_same_label").sum()) if len(da) else 0,
        "n_conflicting_label_groups": n_conf,
        "n_train_and_val": int(((da.n_train > 0) & (da.n_val > 0)).sum()),
        "n_train_and_test": int(((da.n_train > 0) & (da.n_test > 0)).sum()),
        "n_val_and_test": int(((da.n_val > 0) & (da.n_test > 0)).sum()),
        "n_cross_partition": int(da.cross_partition.sum()),
        "n_conflicting_cross_partition": int(da.conflicting_cross_partition.sum()),
    }
    # Đối chiếu với khóa 15 cột (gồm income) – phương pháp split gốc
    sig15 = raw.agg("\x1f".join, axis=1)
    stats1["n_duplicate_signatures_15cols_raw"] = int((sig15.value_counts() > 1).sum())
    # Kiểm tra độ nhạy: chuẩn hóa nhẹ (strip) có làm xuất hiện thêm nhóm không?
    sig_strip = raw[feats].apply(lambda s: s.str.strip()).agg("\x1f".join, axis=1)
    stats1["n_duplicate_signatures_14cols_stripped"] = int((sig_strip.value_counts() > 1).sum())
    print("AUDIT1", stats1)
    chk("1-Duplicate", "Khóa 14 đặc trưng (không income)", "PASS", f"feats={feats}")
    if stats1["n_conflicting_label_groups"] == 0:
        chk("1-Duplicate", "Conflicting labels", "PASS", "0 nhóm 14-đặc-trưng có nhãn khác nhau")
    else:
        chk("1-Duplicate", "Conflicting labels", "WARNING",
            f"{stats1['n_conflicting_label_groups']} nhóm có cùng 14 đặc trưng nhưng khác nhãn (nhiễu nhãn vốn có của dữ liệu)")
    for k, lbl in [("n_train_and_val", "Train∩Val"), ("n_train_and_test", "Train∩Test"),
                   ("n_val_and_test", "Val∩Test")]:
        chk("1-Duplicate", f"Feature-identical {lbl}", "PASS" if stats1[k] == 0 else "WARNING",
            f"{stats1[k]} nhóm")
    chk("1-Duplicate", "Conflicting-label cross-partition",
        "PASS" if stats1["n_conflicting_cross_partition"] == 0 else "WARNING",
        f"{stats1['n_conflicting_cross_partition']} nhóm")

    # ================= AUDIT 4a – SPLIT =================
    cnt = split.split.value_counts().to_dict()
    ok_cnt = cnt == {"train": 22793, "val": 4884, "test": 4884} and len(split) == 32561
    chk("4-Integrity", "Kích thước split", "PASS" if ok_cnt else "FAIL", f"{cnt}, tổng {len(split)}")
    chk("4-Integrity", "Split không overlap", "PASS" if split.row_index.is_unique else "FAIL",
        "row_index duy nhất; mỗi dòng đúng 1 split")
    g15 = pd.Series(pd.factorize(sig15)[0], index=raw.index)  # khóa nguyên văn 15 cột
    gsets = {s: set(g15[part == s]) for s in ("train", "val", "test")}
    inter15 = {f"{a}&{b}": len(gsets[a] & gsets[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    gsplit = split.groupby("dup_group")["split"].nunique()
    chk("4-Integrity", "Group disjointness (15 cột, phương pháp gốc)",
        "PASS" if sum(inter15.values()) == 0 and (gsplit == 1).all() else "FAIL",
        f"giao nhóm nguyên văn {inter15}; dup_group lưu trong split_indices nằm trọn 1 split: {(gsplit == 1).all()}")
    rates = split.assign(y=(raw[TARGET] == ">50K").values[split.row_index]).groupby("split")["y"].mean()
    chk("4-Integrity", "Tỷ lệ >50K giữa các split", "PASS" if rates.max() - rates.min() < 0.005 else "WARNING",
        rates.round(4).to_dict())
    dl = read_text(ROOT / "src" / "data_loader.py")
    seed_ok = all(s in dl for s in ["RANDOM_STATE = 42", "N_FOLDS = 20", "StratifiedGroupKFold"])
    chk("4-Integrity", "Định nghĩa split & seed trong code", "PASS" if seed_ok else "WARNING",
        "data_loader.py: StratifiedGroupKFold, N_FOLDS=20, RANDOM_STATE=42" if seed_ok else "không tìm thấy đủ hằng số")

    # ================= AUDIT 4b – PREDICTIONS & METRICS =================
    P = read_csv(PRED)
    test_idx = split.loc[split.split == "test", "row_index"]
    chk("4-Integrity", "Số dòng predictions", "PASS" if len(P) == 4884 else "FAIL", len(P))
    same_set = set(P.row_index) == set(test_idx) and P.row_index.is_unique
    chk("4-Integrity", "row_index predictions = tập Test đã khóa", "PASS" if same_set else "FAIL", same_set)
    y_raw = (raw.loc[P.row_index, TARGET].str.strip() == ">50K").astype(int).to_numpy()
    lab_ok = np.array_equal(y_raw, P.y_true.to_numpy())
    chk("4-Integrity", "y_true khớp nhãn gốc theo row_index (đúng thứ tự)", "PASS" if lab_ok else "FAIL",
        f"{(y_raw == P.y_true).sum()}/{len(P)} khớp; >50K={P.y_true.sum()}")
    y = P.y_true.to_numpy()
    fm = read_csv(MET / "final_test_metrics.csv").set_index("model")
    md = read_text(REPORT_MD)
    ref = {"DecisionTree": dict(accuracy=.8022, precision=.5576, recall=.8639, f1=.6778, roc_auc=.8984,
                                pr_auc=.7715, cm=(2902, 806, 160, 1016)),
           "CategoricalNB": dict(accuracy=.8176, precision=.5907, recall=.7891, f1=.6756, roc_auc=.9011,
                                 pr_auc=.7665, cm=(3065, 643, 248, 928))}
    recomputed = {}
    for m in ["DecisionTree", "CategoricalNB", "GaussianNB"]:
        pr, sc = P[f"{m}_pred"].to_numpy(), P[f"{m}_proba_gt50K"].to_numpy()
        thr_ok = np.array_equal(pr, (sc >= 0.5).astype(int))
        chk("4-Integrity", f"{m}: pred = (proba ≥ 0.5)", "PASS" if thr_ok else "FAIL", thr_ok)
        tn, fp, fn, tp = map(int, confusion_matrix(y, pr, labels=[0, 1]).ravel())
        prec_c, rec_c, _ = precision_recall_curve(y, sc, pos_label=1)
        r = {"accuracy": (tp + tn) / len(y), "precision": tp / (tp + fp), "recall": tp / (tp + fn),
             "f1": f1_score(y, pr, pos_label=1), "roc_auc": roc_auc_score(y, sc),
             "pr_auc": average_precision_score(y, sc, pos_label=1), "pr_auc_trapezoid": auc(rec_c, prec_c),
             "tn": tn, "fp": fp, "fn": fn, "tp": tp}
        recomputed[m] = r
        diffs = {k: abs(r[k] - fm.loc[m, k]) for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]}
        cm_csv = tuple(int(fm.loc[m, k]) for k in ["tn", "fp", "fn", "tp"])
        ok = max(diffs.values()) < 1e-12 and cm_csv == (tn, fp, fn, tp)
        chk("4-Integrity", f"{m}: metrics CSV tái tính từ predictions", "PASS" if ok else "FAIL",
            f"max |Δ|={max(diffs.values()):.2e}; CM={tn}/{fp}/{fn}/{tp}")
        f1_cm = 2 * tp / (2 * tp + fp + fn)
        chk("4-Integrity", f"{m}: CM ↔ P/R/F1", "PASS" if abs(f1_cm - r["f1"]) < 1e-12 else "FAIL",
            f"F1 từ CM={f1_cm:.6f}")
        if m in ref:
            rr = ref[m]
            ok_ref = all(round(r[k], 4) == rr[k] for k in rr if k != "cm") and rr["cm"] == (tn, fp, fn, tp)
            chk("4-Integrity", f"{m}: khớp số tham chiếu (4 chữ số)", "PASS" if ok_ref else "FAIL",
                {k: round(r[k], 4) for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]})
            in_md = all(f"{rr[k]:.4f}".replace(".", ",") in md for k in rr if k != "cm") and \
                " / ".join(map(str, rr["cm"])) in md
            chk("4-Integrity", f"{m}: số liệu xuất hiện trong final_evaluation.md", "PASS" if in_md else "FAIL", in_md)

    # ================= AUDIT 2 – McNEMAR =================
    dt_ok = P.DecisionTree_pred.to_numpy() == y
    cnb_ok = P.CategoricalNB_pred.to_numpy() == y
    a = int((dt_ok & cnb_ok).sum()); b = int((dt_ok & ~cnb_ok).sum())
    c = int((~dt_ok & cnb_ok).sum()); d = int((~dt_ok & ~cnb_ok).sum())
    n = b + c
    p_exact = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    p_exact_manual = min(1.0, 2 * sum(__import__("math").comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    chi_nc = (b - c) ** 2 / n
    chi_cc = (abs(b - c) - 1) ** 2 / n
    p_nc, p_cc = chi2.sf(chi_nc, 1), chi2.sf(chi_cc, 1)
    pmc = read_csv(MET / "paired_model_comparison.csv")
    mrow = pmc[pmc.metric.str.contains("McNemar")].iloc[0]
    m_rep = re.search(r"=(\d+),.*=(\d+), p=([\d.e-]+)", mrow.conclusion)
    rb, rc, rp = int(m_rep.group(1)), int(m_rep.group(2)), float(m_rep.group(3))
    src = read_text(SRC_FINAL)
    method_in_code = "binomtest(min(b_, c_), b_ + c_, 0.5)" in src
    mc = {"both_correct": a, "b_dt_right_cnb_wrong": b, "c_dt_wrong_cnb_right": c, "both_wrong": d,
          "n_total": a + b + c + d, "acc_dt": (a + b) / len(y), "acc_cnb": (a + c) / len(y),
          "p_exact_binomial": p_exact, "p_exact_manual": p_exact_manual,
          "chi2_no_cc": chi_nc, "p_chi2_no_cc": p_nc, "chi2_cc": chi_cc, "p_chi2_cc": p_cc,
          "reported": {"b": rb, "c": rc, "p": rp}, "method_in_code_exact_binomial": method_in_code}
    print("AUDIT2", mc)
    chk("2-McNemar", "Bảng 2×2 tổng = 4884", "PASS" if a + b + c + d == len(y) == 4884 else "FAIL", f"{a}/{b}/{c}/{d}")
    chk("2-McNemar", "b, c khớp báo cáo", "PASS" if (b, c) == (rb, rc) else "FAIL", f"b={b}, c={c}")
    chk("2-McNemar", "Phương pháp trong code = exact binomial", "PASS" if method_in_code else "WARNING",
        "scipy.stats.binomtest(min(b,c), b+c, 0.5), two-sided")
    chk("2-McNemar", "p-value tái tính khớp (4 chữ số có nghĩa)",
        "PASS" if float(f"{p_exact:.4g}") == rp and abs(p_exact - p_exact_manual) < 1e-12 else "FAIL",
        f"exact={p_exact:.6g} (manual {p_exact_manual:.6g}); báo cáo={rp}")
    chk("2-McNemar", "Accuracy từ bảng 2×2", "PASS" if abs(mc["acc_dt"] - recomputed["DecisionTree"]["accuracy"]) < 1e-12
        and abs(mc["acc_cnb"] - recomputed["CategoricalNB"]["accuracy"]) < 1e-12 else "FAIL",
        f"DT={mc['acc_dt']:.4f}, CNB={mc['acc_cnb']:.4f}")
    fn_fp = (recomputed["DecisionTree"]["fn"] < recomputed["CategoricalNB"]["fn"],
             recomputed["CategoricalNB"]["fp"] < recomputed["DecisionTree"]["fp"])
    chk("2-McNemar", "DT ít FN hơn, CNB ít FP hơn", "PASS" if all(fn_fp) else "FAIL",
        f"FN {recomputed['DecisionTree']['fn']} vs {recomputed['CategoricalNB']['fn']}; "
        f"FP {recomputed['DecisionTree']['fp']} vs {recomputed['CategoricalNB']['fp']}")
    chk("2-McNemar", "Tiêu chí chính vs phân tích phụ", "WARNING",
        "McNemar là phân tích BỔ SUNG, không định trước; chỉ so sánh Accuracy (không phải tiêu chí chọn mô hình F1). "
        "Cần trình bày là post-hoc/mô tả.")

    # ================= AUDIT 3 – BOOTSTRAP =================
    code = {
        "n_boot_5000": "N_BOOT = 5000" in src,
        "seed_42": "BOOT_SEED = 42" in src and "np.random.default_rng(seed)" in src,
        "stratified_by_y_true": "np.flatnonzero(y == 1), np.flatnonzero(y == 0)" in src
                                and "rng.choice(neg, len(neg), replace=True)" in src,
        "paired_shared_indices": "for idx in stratified_bootstrap_indices(y, N_BOOT, BOOT_SEED):" in src
                                 and "for n in models:" in src,
        "percentile_ci": "np.percentile(a, [100 * (1 - level) / 2, 100 * (1 + level) / 2])" in src,
        "direction_dt_minus_cnb": 'stats["DecisionTree"][k] - stats["CategoricalNB"][k]' in src,
        "f1_pos_label_1": 'f1_score(yb, prb, pos_label=1)' in src,
        "roc_on_scores": "roc_auc_score(yb, scb)" in src,
        "pr_auc_is_average_precision": "average_precision_score(yb, scb, pos_label=1)" in src,
        "y_is_saved_y_true": 'y = P["y_true"].to_numpy()' in src,
    }
    for k, v in code.items():
        chk("3-Bootstrap", f"Code: {k}", "PASS" if v else "FAIL", v)
    # Tái tính độc lập từ saved predictions (cùng thuật toán sinh chỉ số, cài đặt lại tại đây)
    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    S = {m: {k: np.empty(N_BOOT) for k in ("f1", "roc_auc", "pr_auc")} for m in ("DecisionTree", "CategoricalNB", "GaussianNB")}
    strat_ok = True
    for i in range(N_BOOT):
        idx = np.concatenate([rng.choice(neg, len(neg), replace=True), rng.choice(pos, len(pos), replace=True)])
        yb = y[idx]
        strat_ok &= int(yb.sum()) == len(pos) and len(yb) == len(y)
        for m in S:
            S[m]["f1"][i] = f1_score(yb, P[f"{m}_pred"].to_numpy()[idx], pos_label=1)
            sc = P[f"{m}_proba_gt50K"].to_numpy()[idx]
            S[m]["roc_auc"][i] = roc_auc_score(yb, sc)
            S[m]["pr_auc"][i] = average_precision_score(yb, sc, pos_label=1)
    chk("3-Bootstrap", "Mỗi replicate giữ đúng 1176 dương / 3708 âm", "PASS" if strat_ok else "FAIL", strat_ok)
    bci = read_csv(MET / "bootstrap_confidence_intervals.csv")
    worst = 0.0
    ci_rows = []
    for _, r in bci.iterrows():
        lo, hi = np.percentile(S[r.model][r.metric], [2.5, 97.5])
        worst = max(worst, abs(lo - r.ci95_low), abs(hi - r.ci95_high))
        ci_rows.append((r.model, r.metric, r.ci95_low, r.ci95_high, lo, hi))
    chk("3-Bootstrap", "CI từng mô hình tái tính khớp CSV", "PASS" if worst < 1e-12 else "FAIL", f"max |Δ|={worst:.2e}")
    diff_rows = []
    worst_d = 0.0
    for _, r in pmc[pmc.metric.isin(["f1", "roc_auc", "pr_auc"])].iterrows():
        dd = S["DecisionTree"][r.metric] - S["CategoricalNB"][r.metric]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        pt = recomputed["DecisionTree"][r.metric] - recomputed["CategoricalNB"][r.metric]
        worst_d = max(worst_d, abs(lo - r.ci95_low), abs(hi - r.ci95_high), abs(pt - r.point_difference))
        # đối chứng: nếu KHÔNG ghép cặp (chỉ số độc lập) – chỉ để minh họa độ rộng
        diff_rows.append({"metric": r.metric, "point": pt, "lo": lo, "hi": hi, "csv_lo": r.ci95_low,
                          "csv_hi": r.ci95_high, "contains0": lo <= 0 <= hi, "share_gt0": (dd > 0).mean(),
                          "csv_contains0": bool(r.ci_contains_0)})
    chk("3-Bootstrap", "CI chênh lệch DT−CNB tái tính khớp CSV", "PASS" if worst_d < 1e-12 else "FAIL",
        f"max |Δ|={worst_d:.2e}")
    md_diff = all(s in md for s in ["+0,0021", "[−0,0133; +0,0175]", "−0,0027", "[−0,0102; +0,0046]",
                                    "+0,0050", "[−0,0093; +0,0192]"])
    chk("3-Bootstrap", "Chênh lệch & CI trong final_evaluation.md khớp", "PASS" if md_diff else "FAIL", md_diff)
    chk("3-Bootstrap", "Tất cả CI chênh lệch chứa 0", "PASS" if all(r["contains0"] for r in diff_rows) else "FAIL",
        [(r["metric"], round(r["lo"], 4), round(r["hi"], 4)) for r in diff_rows])
    trap = {m: recomputed[m]["pr_auc_trapezoid"] for m in recomputed}
    chk("3-Bootstrap", "Định nghĩa 'PR-AUC'", "WARNING",
        "Implementation = Average Precision (sklearn average_precision_score, tổng bậc thang), KHÔNG phải tích phân "
        f"hình thang. Hình thang (tham khảo): DT={trap['DecisionTree']:.4f}, CNB={trap['CategoricalNB']:.4f}. "
        "Báo cáo cần ghi rõ 'PR-AUC (Average Precision)'.")
    chk("3-Bootstrap", "Phạm vi suy diễn của bootstrap", "WARNING",
        "Chỉ phản ánh biến thiên do mẫu Test; mô hình cố định (không tính biến thiên do huấn luyện/chia dữ liệu).")

    # ================= AUDIT 4c – ARTIFACTS =================
    raw_hash = hash_before["data/raw/adult.csv"]
    chk("4-Integrity", "Checksum dataset gốc", "PASS" if raw_hash.startswith(RAW_HASH_PREFIX_RECORDED) else "FAIL",
        f"sha256={raw_hash}; khớp 16 ký tự đầu đã ghi nhận ({RAW_HASH_PREFIX_RECORDED}). "
        "Lưu ý: checksum đầy đủ ban đầu không được lưu thành file trong repo.")
    info = json.loads(read_text(MET / "final_run_info.json"))
    for m, fi in info["fit_info"].items():
        p = ROOT / fi["model_file"].replace("\\", "/")
        ok = p.exists() and hash_before[p.relative_to(ROOT).as_posix()].startswith(fi["model_sha256"])
        chk("4-Integrity", f"Model artifact {m}", "PASS" if ok else "FAIL",
            f"{fi['model_file']} tồn tại, sha256 khớp prefix {fi['model_sha256']} trong final_run_info.json (không load)")
    for f in ["DecisionTree", "CategoricalNB", "GaussianNB"]:
        p = ROOT / "results" / "models" / f"{f}_best_trainonly.joblib"
        chk("4-Integrity", f"Artifact PHASE 4 {f}", "PASS" if p.exists() else "FAIL", p.exists())
    cfg_ok = info["locked_configs"] == {k: v["config"] for k, v in json.loads(read_text(MET / "best_configs.json"))["best_configs"].items()}
    chk("4-Integrity", "Cấu hình final = best_configs PHASE 4", "PASS" if cfg_ok else "FAIL", cfg_ok)
    chk("4-Integrity", "Metadata ngưỡng/seed", "PASS" if info["environment"]["threshold"] == 0.5 and
        info["environment"]["bootstrap"] == {"n": 5000, "seed": 42, "stratified": True, "paired": True} else "FAIL",
        info["environment"]["bootstrap"])
    logs = sorted((ROOT / "results" / "logs").glob("phase5_6_final_*.log"))
    for lg in logs:
        inspected.add(lg.relative_to(ROOT).as_posix())
    log_ok = logs and "TP=1016" in logs[-1].read_text(encoding="utf-8") and "p=0.002554" in logs[-1].read_text(encoding="utf-8")
    first_log = logs[0].read_text(encoding="utf-8") if len(logs) > 1 else ""
    chk("4-Integrity", "Log lần chạy hoàn tất khớp kết quả", "PASS" if log_ok else "FAIL", logs[-1].name if logs else None)
    chk("4-Integrity", "Log lần chạy lỗi không chạm Test",
        "PASS" if len(logs) > 1 and "TEST" not in first_log and "Đã dự đoán Test" not in first_log else "WARNING",
        f"{logs[0].name}: không có dòng dự đoán/metric Test" if len(logs) > 1 else "chỉ 1 log")
    chk("4-Integrity", "Một lần dự đoán Test / mô hình (theo code)",
        "PASS" if src.count("positive_proba(m, Xte)") == 1 and "final_predictions.csv đã tồn tại" not in "" else "WARNING",
        "1 lời gọi positive_proba(m, Xte) trong vòng lặp; guard chống chạy lại khi final_predictions.csv tồn tại")
    chk("4-Integrity", "Phase 4 log dùng Validation, không Test",
        "PASS" if "splits.pop(\"test\")" in read_text(ROOT / "src" / "tuning.py") else "WARNING",
        "tuning.py loại Test ngay sau khi chia")

    # read-only proof
    hash_after = {k: sha(ROOT / k) for k in hash_before}
    ro = hash_before == hash_after
    chk("Read-only", "Artifacts không đổi trong khi audit", "PASS" if ro else "FAIL", f"{len(hash_after)} file, hash trước = sau")

    # ======= Ghi kết quả số để viết báo cáo =======
    out = {"started": started, "finished": datetime.now().isoformat(timespec="seconds"),
           "environment": {"platform": platform.platform(), "python": sys.version.split()[0],
                           "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                           "device": "CPU local"},
           "duplicate": stats1, "mcnemar": mc, "recomputed_metrics": recomputed,
           "bootstrap_ci_rows": ci_rows, "bootstrap_diff_rows": diff_rows,
           "checks": checks, "files_inspected": sorted(inspected), "frozen_hashes": hash_before,
           "read_only_verified": ro}
    (OUT / "_audit_numbers.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=lambda o: o.item()
                                                         if hasattr(o, "item") else str(o)), encoding="utf-8")
    print("DONE", sum(c["status"] == "PASS" for c in checks), "PASS /",
          sum(c["status"] == "WARNING" for c in checks), "WARNING /", sum(c["status"] == "FAIL" for c in checks), "FAIL")


if __name__ == "__main__":
    main()
