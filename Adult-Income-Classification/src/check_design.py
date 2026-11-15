"""Kiểm tra thiết kế chia dữ liệu và preprocessing (chạy: python src/check_design.py).

Ghi ra data/processed/:
  split_indices.csv          - chỉ số dòng gốc, tập, ID nhóm trùng lặp
  split_summary.csv          - số mẫu, tỷ lệ chia, phân phối nhãn
  cross_split_duplicates.csv - số nội dung trùng giữa từng cặp tập (phải = 0)
  split_old_vs_new.csv       - so sánh cách chia cũ (không xét nhóm) và mới
Mọi assert thất bại nghĩa là thiết kế bị vi phạm.
"""
import hashlib
import os
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import CategoricalNB

from data_loader import (CATEGORICAL_FEATURES, PROCESSED_DIR, RAW_PATH, TARGET,
                         clean, duplicate_group_ids, load_raw, save_splits,
                         split_data, split_data_legacy, split_summary, xy)
from preprocessing import (build_cnb_preprocessor, build_dt_preprocessor,
                           build_gnb_preprocessor, cnb_min_categories)

sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252
warnings.filterwarnings("ignore", category=UserWarning)
PAIRS = [("train", "val"), ("train", "test"), ("val", "test")]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def content_overlap(s):
    """Số nội dung bản ghi (mọi cột kể cả nhãn) xuất hiện ở cả hai tập."""
    key = lambda d: set(d.fillna("?").astype(str).agg("|".join, axis=1))
    K = {n: key(d) for n, d in s.items()}
    return {f"{a}&{b}": len(K[a] & K[b]) for a, b in PAIRS}


def fitted_state(obj):
    """Ảnh chụp các thuộc tính đã học (kết thúc bằng '_') để so sánh trước/sau transform."""
    import pickle
    return pickle.dumps(obj)


def main():
    print(f"Môi trường: {platform.platform()} | Python {sys.version.split()[0]} | "
          f"scikit-learn {sklearn.__version__} | pandas {pd.__version__} | CPUs {os.cpu_count()}")
    raw_hash_before = sha256(RAW_PATH)

    raw = load_raw()
    df = clean(raw)
    N = len(df)
    groups = duplicate_group_ids(df)
    sizes = groups.value_counts()
    print(f"Tổng bản ghi: {N} | số nhóm: {groups.nunique()} | "
          f"nhóm có >1 bản ghi: {(sizes > 1).sum()} (chứa {sizes[sizes > 1].sum()} bản ghi, "
          f"{int((sizes[sizes > 1] - 1).sum())} bản sao thừa)")
    # Nhóm trùng phải có nhãn đồng nhất (nhãn nằm trong khóa)
    assert (df.groupby(groups)[TARGET].nunique() == 1).all()

    # ------------------------ CHIA MỚI (group-aware) -------------------------
    s = split_data(df)
    summary = save_splits(s)
    print("\nKết quả chia MỚI (StratifiedGroupKFold, seed 42):")
    print(summary.to_string(index=False))

    # 1. Rời nhau theo chỉ số, phủ đủ, tổng = 32.561
    I = {k: set(v.index) for k, v in s.items()}
    assert not any(I[a] & I[b] for a, b in PAIRS)
    assert set().union(*I.values()) == set(df.index) and sum(map(len, s.values())) == N == 32561
    # 2. Không nhóm trùng nào nằm ở nhiều tập
    g_by_split = {k: set(groups.loc[v.index]) for k, v in s.items()}
    groups_shared = {f"{a}&{b}": len(g_by_split[a] & g_by_split[b]) for a, b in PAIRS}
    assert all(v == 0 for v in groups_shared.values()), groups_shared
    # 3. Không nội dung bản ghi nào giao nhau giữa các tập
    overlap = content_overlap(s)
    assert all(v == 0 for v in overlap.values()), overlap
    pd.Series(overlap, name="n_shared_rows").to_csv(
        PROCESSED_DIR / "cross_split_duplicates.csv", index_label="pair")
    # 4. Tỷ lệ chia gần 70/15/15 (sai lệch < 0.5 điểm %), nhãn phân tầng (< 0.5 điểm %)
    p_all = df[TARGET].mean()
    for row in summary.itertuples():
        target = {"train": 0.70, "val": 0.15, "test": 0.15}[row.split]
        assert abs(row.fraction - target) < 0.005, row
        assert abs(row.ratio_gt50K - p_all) < 0.005, row
    # 5. Tái lập
    s2 = split_data(clean(load_raw()))
    assert all(s[k].index.equals(s2[k].index) for k in s)
    # 6. education <-> education.num ánh xạ 1-1 (lý do bỏ cột education)
    assert df.groupby("education")["education.num"].nunique().max() == 1
    assert df.groupby("education.num")["education"].nunique().max() == 1
    print(f"OK: tổng {N} | nhóm dùng chung giữa các tập {groups_shared} | "
          f"nội dung trùng giữa các tập {overlap} | tái lập | phân tầng")

    # ------------------------ SO SÁNH CŨ vs MỚI ------------------------------
    old = split_data_legacy(df)
    old_sum = split_summary(old).set_index("split")
    new_sum = summary.set_index("split")
    cmp = pd.concat({"old": old_sum, "new": new_sum}, axis=1)
    cmp.columns = [f"{a}_{b}" for a, b in cmp.columns]
    cols = [c for c in cmp.columns if c.split("_", 1)[1] in
            ("n_samples", "fraction", "n_gt50K", "ratio_gt50K")]
    cmp = cmp[sorted(cols, key=lambda c: (c.split("_", 1)[1], c))]
    old_overlap = content_overlap(old)
    moved = {k: len(set(old[k].index) - set(s[k].index)) for k in s}
    cmp["rows_moved_out_vs_old"] = pd.Series(moved)
    cmp.to_csv(PROCESSED_DIR / "split_old_vs_new.csv")
    print("\nSo sánh CŨ (train_test_split) và MỚI (StratifiedGroupKFold):")
    print(cmp.to_string())
    print("Nội dung trùng giữa các tập – cũ:", old_overlap, "| mới:", overlap)

    # ------------------------ PREPROCESSING ---------------------------------
    Xtr, ytr = xy(s["train"]); Xv, _ = xy(s["val"]); Xte, _ = xy(s["test"])
    print()
    for name, b in [("DT", build_dt_preprocessor), ("CNB", build_cnb_preprocessor),
                    ("GNB", build_gnb_preprocessor)]:
        p = b().fit(Xtr)
        before = fitted_state(p)
        shapes = [p.transform(X).shape for X in (Xtr, Xv, Xte)]
        # 7. transform KHÔNG thay đổi trạng thái đã học (so sánh toàn bộ đối tượng)
        assert fitted_state(p) == before, f"{name}: transform làm thay đổi trạng thái"
        print(f"{name}: shapes train/val/test = {shapes} | trạng thái không đổi sau transform")

    p = build_cnb_preprocessor().fit(Xtr)
    print("Số khoảng mỗi cột số:",
          dict(zip(p.named_transformers_["num"].feature_names_in_,
                   p.named_transformers_["num"].n_bins_.tolist())),
          dict(zip(["capital.gain", "capital.loss"],
                   p.named_transformers_["capital"].n_bins_.tolist())))
    assert min(p.named_transformers_["capital"].n_bins_) > 1
    # 8. Encoder chỉ học hạng mục từ Train
    enc = p.named_transformers_["cat"].named_steps["ordinal"].enc_
    for c, cats in zip(CATEGORICAL_FEATURES, enc.categories_):
        tr = set(Xtr[c].fillna("Unknown"))
        assert set(cats) == tr
        unseen = (set(Xv[c].fillna("Unknown")) | set(Xte[c].fillna("Unknown"))) - tr
        if unseen:
            print(f"  Hạng mục chỉ có ở Val/Test (mã dành riêng): {c} {unseen}")
    m = CategoricalNB(min_categories=cnb_min_categories(p)).fit(p.transform(Xtr), ytr)
    m.predict(p.transform(Xv)); m.predict(p.transform(Xte))

    # 9. Fit lại trên Train+Validation: preprocessing học lại từ đầu
    trv = pd.concat([s["train"], s["val"]])
    Xtv, ytv = xy(trv)
    p2 = build_cnb_preprocessor().fit(Xtv)
    m2 = CategoricalNB(min_categories=cnb_min_categories(p2)).fit(p2.transform(Xtv), ytv)
    m2.predict(p2.transform(Xte))
    print(f"OK: refit Train+Val ({len(trv)} mẫu) -> dự đoán Test ({len(Xte)} mẫu)")

    # 10. File raw không bị sửa
    assert sha256(RAW_PATH) == raw_hash_before
    print(f"OK: file raw không đổi (sha256 {raw_hash_before[:16]}…)")
    print("TẤT CẢ KIỂM TRA ĐẠT")


if __name__ == "__main__":
    main()
