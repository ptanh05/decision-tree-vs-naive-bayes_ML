"""Kiểm tra thiết kế chia dữ liệu và preprocessing (chạy: python src/check_design.py).

Cảnh báo bảo toàn artifact:
- Script này ghi ra `data/processed/` gồm `split_indices.csv` (artifact đã đóng băng SHA256),
  `split_summary.csv`, `cross_split_duplicates.csv` và `split_old_vs_new.csv`.
- Chỉ chạy lại khi cần tái lập từ đầu; để học dự án, chỉ nên đọc các file đã lưu.

Mọi `assert` thất bại đồng nghĩa một giả định thiết kế chống rò rỉ bị vi phạm.
"""
import hashlib
import os
import pickle
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import CategoricalNB

from data_loader import (
    CATEGORICAL_FEATURES,
    PROCESSED_DIR,
    RAW_PATH,
    TARGET,
    clean,
    duplicate_group_ids,
    load_raw,
    save_splits,
    split_data,
    split_data_legacy,
    split_summary,
    xy,
)
from preprocessing import (
    build_cnb_preprocessor,
    build_dt_preprocessor,
    build_gnb_preprocessor,
    cnb_min_categories,
)

sys.stdout.reconfigure(encoding="utf-8")  # Console Windows mặc định cp1252
warnings.filterwarnings("ignore", category=UserWarning)
PAIRS = [("train", "val"), ("train", "test"), ("val", "test")]


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def content_overlap(splits: dict) -> dict:
    """Đếm số nội dung bản ghi (cả 15 cột kể cả nhãn) xuất hiện ở cả hai tập."""
    def make_keys(df: pd.DataFrame) -> set:
        return set(df.fillna("?").astype(str).agg("|".join, axis=1))

    keys_by_split = {name: make_keys(part) for name, part in splits.items()}
    return {f"{a}&{b}": len(keys_by_split[a] & keys_by_split[b]) for a, b in PAIRS}


def fitted_state(obj) -> bytes:
    """Chụp ảnh trạng thái đã học của preprocessor để kiểm tra bất biến trước/sau transform."""
    return pickle.dumps(obj)


def main() -> None:
    print(
        f"Môi trường: {platform.platform()} | Python {sys.version.split()[0]} | "
        f"scikit-learn {sklearn.__version__} | pandas {pd.__version__} | CPUs {os.cpu_count()}"
    )
    raw_hash_before = sha256(RAW_PATH)

    raw = load_raw()
    df = clean(raw)
    total_rows = len(df)
    groups = duplicate_group_ids(df)
    sizes = groups.value_counts()
    print(
        f"Tổng bản ghi: {total_rows} | số nhóm: {groups.nunique()} | "
        f"nhóm có >1 bản ghi: {(sizes > 1).sum()} (chứa {sizes[sizes > 1].sum()} bản ghi, "
        f"{int((sizes[sizes > 1] - 1).sum())} bản sao thừa)"
    )
    # Do nhãn nằm trong khóa nhóm 15 cột, mọi bản ghi cùng nhóm phải có cùng nhãn
    assert (df.groupby(groups)[TARGET].nunique() == 1).all()

    # ------------------------ CHIA MỚI (group-aware) -------------------------
    splits = split_data(df)
    summary = save_splits(splits)
    print("\nKết quả chia MỚI (StratifiedGroupKFold, seed 42):")
    print(summary.to_string(index=False))

    # Kiểm tra 1: Rời nhau theo chỉ số dòng, phủ đủ 32.561 mẫu
    indices_by_split = {k: set(v.index) for k, v in splits.items()}
    assert not any(indices_by_split[a] & indices_by_split[b] for a, b in PAIRS)
    assert set().union(*indices_by_split.values()) == set(df.index)
    assert sum(map(len, splits.values())) == total_rows == 32561

    # Kiểm tra 2: Không có nhóm trùng lặp nào bị tách sang 2 tập khác nhau
    groups_by_split = {k: set(groups.loc[v.index]) for k, v in splits.items()}
    groups_shared = {f"{a}&{b}": len(groups_by_split[a] & groups_by_split[b]) for a, b in PAIRS}
    assert all(v == 0 for v in groups_shared.values()), groups_shared

    # Kiểm tra 3: Giao nội dung bản ghi (15 cột) giữa mọi cặp tập phải bằng 0
    overlap = content_overlap(splits)
    assert all(v == 0 for v in overlap.values()), overlap
    pd.Series(overlap, name="n_shared_rows").to_csv(
        PROCESSED_DIR / "cross_split_duplicates.csv",
        index_label="pair",
    )

    # Kiểm tra 4: Tỷ lệ chia ~70/15/15 (sai lệch < 0.5%) và phân tầng nhãn (< 0.5%)
    p_all = df[TARGET].mean()
    for row in summary.itertuples():
        target = {"train": 0.70, "val": 0.15, "test": 0.15}[row.split]
        assert abs(row.fraction - target) < 0.005, row
        assert abs(row.ratio_gt50K - p_all) < 0.005, row

    # Kiểm tra 5: Chia dữ liệu tái lập hoàn toàn khi gọi lại với cùng seed 42
    splits_repeat = split_data(clean(load_raw()))
    assert all(splits[k].index.equals(splits_repeat[k].index) for k in splits)

    # Kiểm tra 6: Cột 'education' và 'education.num' ánh xạ 1-1 -> căn cứ loại bỏ 'education'
    assert df.groupby("education")["education.num"].nunique().max() == 1
    assert df.groupby("education.num")["education"].nunique().max() == 1
    print(
        f"OK: tổng {total_rows} | nhóm dùng chung giữa các tập {groups_shared} | "
        f"nội dung trùng giữa các tập {overlap} | tái lập | phân tầng"
    )

    # ------------------------ SO SÁNH CŨ vs MỚI ------------------------------
    old = split_data_legacy(df)
    old_sum = split_summary(old).set_index("split")
    new_sum = summary.set_index("split")
    cmp = pd.concat({"old": old_sum, "new": new_sum}, axis=1)
    cmp.columns = [f"{a}_{b}" for a, b in cmp.columns]
    cols = [
        c for c in cmp.columns
        if c.split("_", 1)[1] in ("n_samples", "fraction", "n_gt50K", "ratio_gt50K")
    ]
    cmp = cmp[sorted(cols, key=lambda c: (c.split("_", 1)[1], c))]
    old_overlap = content_overlap(old)
    moved = {k: len(set(old[k].index) - set(splits[k].index)) for k in splits}
    cmp["rows_moved_out_vs_old"] = pd.Series(moved)
    cmp.to_csv(PROCESSED_DIR / "split_old_vs_new.csv")
    print("\nSo sánh CŨ (train_test_split) và MỚI (StratifiedGroupKFold):")
    print(cmp.to_string())
    print("Nội dung trùng giữa các tập – cũ:", old_overlap, "| mới:", overlap)

    # ------------------------ PREPROCESSING ---------------------------------
    Xtr, ytr = xy(splits["train"])
    Xv, _ = xy(splits["val"])
    Xte, _ = xy(splits["test"])
    print()

    # Kiểm tra 7: transform trên Val/Test KHÔNG làm thay đổi tham số đã học từ Train
    for name, builder in [
        ("DT", build_dt_preprocessor),
        ("CNB", build_cnb_preprocessor),
        ("GNB", build_gnb_preprocessor),
    ]:
        pre = builder().fit(Xtr)
        before = fitted_state(pre)
        shapes = [pre.transform(X).shape for X in (Xtr, Xv, Xte)]
        assert fitted_state(pre) == before, f"{name}: transform làm thay đổi trạng thái"
        print(f"{name}: shapes train/val/test = {shapes} | trạng thái không đổi sau transform")

    pre_cnb = build_cnb_preprocessor().fit(Xtr)
    print(
        "Số khoảng mỗi cột số:",
        dict(zip(pre_cnb.named_transformers_["num"].feature_names_in_,
                 pre_cnb.named_transformers_["num"].n_bins_.tolist())),
        dict(zip(["capital.gain", "capital.loss"],
                 pre_cnb.named_transformers_["capital"].n_bins_.tolist())),
    )
    assert min(pre_cnb.named_transformers_["capital"].n_bins_) > 1

    # Kiểm tra 8: Encoder của CNB chỉ học tập hạng mục có trong Train; xử lý mượt hạng mục lạ
    enc = pre_cnb.named_transformers_["cat"].named_steps["ordinal"].enc_
    for col, cats in zip(CATEGORICAL_FEATURES, enc.categories_):
        train_cats = set(Xtr[col].fillna("Unknown"))
        assert set(cats) == train_cats
        unseen = (set(Xv[col].fillna("Unknown")) | set(Xte[col].fillna("Unknown"))) - train_cats
        if unseen:
            print(f"  Hạng mục chỉ có ở Val/Test (mã dành riêng): {col} {unseen}")

    model = CategoricalNB(min_categories=cnb_min_categories(pre_cnb)).fit(pre_cnb.transform(Xtr), ytr)
    model.predict(pre_cnb.transform(Xv))
    model.predict(pre_cnb.transform(Xte))

    # Kiểm tra 9: Refit toàn bộ pipeline trên Train+Validation hoạt động và dự đoán được trên Test
    trainval_df = pd.concat([splits["train"], splits["val"]])
    Xtv, ytv = xy(trainval_df)
    pre_tv = build_cnb_preprocessor().fit(Xtv)
    model_tv = CategoricalNB(min_categories=cnb_min_categories(pre_tv)).fit(pre_tv.transform(Xtv), ytv)
    model_tv.predict(pre_tv.transform(Xte))
    print(f"OK: refit Train+Val ({len(trainval_df)} mẫu) -> dự đoán Test ({len(Xte)} mẫu)")

    # Kiểm tra 10: File dữ liệu thô adult.csv hoàn toàn không bị thay đổi (SHA256 giữ nguyên)
    assert sha256(RAW_PATH) == raw_hash_before
    print(f"OK: file raw không đổi (sha256 {raw_hash_before[:16]}…)")
    print("TẤT CẢ KIỂM TRA ĐẠT")


if __name__ == "__main__":
    main()
