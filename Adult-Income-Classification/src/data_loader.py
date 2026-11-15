"""Đọc dữ liệu thô và chia Train / Validation / Test (70/15/15, stratified, seed 42).

Lưu ý: bước làm sạch ở đây chỉ gồm các phép biến đổi KHÔNG học tham số
(strip khoảng trắng, đổi '?' -> NaN, mã hóa nhãn). Mọi bước có học tham số
(imputer, encoder, discretizer) nằm trong Pipeline ở preprocessing.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "adult.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

RANDOM_STATE = 42
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.70, 0.15, 0.15
TARGET = "income"

NUMERIC_FEATURES = ["age", "fnlwgt", "education.num",
                    "capital.gain", "capital.loss", "hours.per.week"]
CATEGORICAL_FEATURES = ["workclass", "marital.status", "occupation",
                        "relationship", "race", "sex", "native.country"]
# 'education' bị loại vì trùng thông tin với 'education.num' (kiểm tra ở EDA).
DROPPED_FEATURES = ["education"]


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    """Đọc CSV gốc, không sửa file trên đĩa."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch không học tham số: strip chuỗi, '?' -> NaN, nhãn -> 0/1."""
    df = df.copy()
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip()
    df = df.replace("?", np.nan)
    mapping = {"<=50K": 0, ">50K": 1}
    unknown = set(df[TARGET].unique()) - set(mapping)
    if unknown:
        raise ValueError(f"Nhãn không mong đợi: {unknown}")
    df[TARGET] = df[TARGET].map(mapping).astype(int)
    return df


N_FOLDS = 20                      # mỗi fold ~5% dữ liệu
TEST_FOLDS, VAL_FOLDS = (0, 1, 2), (3, 4, 5)   # 3 fold = 15%, 3 fold = 15%, còn lại 70%


def duplicate_group_ids(df: pd.DataFrame) -> pd.Series:
    """ID nhóm trùng lặp theo NỘI DUNG ĐẦY ĐỦ của bản ghi (mọi cột, kể cả nhãn).

    Bản ghi duy nhất -> nhóm 1 phần tử. ID đánh số theo thứ tự xuất hiện
    đầu tiên trong file gốc nên hoàn toàn xác định (tái lập).
    """
    key = df.fillna("?").astype(str).agg("|".join, axis=1)
    return pd.Series(pd.factorize(key)[0], index=df.index, name="dup_group")


def split_data(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """Chia Train/Validation/Test ~70/15/15 theo nhóm, có phân tầng theo nhãn.

    Dùng StratifiedGroupKFold(20 fold, shuffle, seed 42): mỗi nhóm trùng lặp
    nằm trọn trong MỘT fold; fold 0-2 -> Test, 3-5 -> Validation, 6-19 -> Train.
    Vì nhãn là một phần của khóa nhóm, mọi bản ghi trong nhóm có cùng nhãn,
    nên phân tầng theo nhãn không xung đột với ràng buộc nhóm.
    Trả về dict {'train','val','test'} -> DataFrame (giữ index gốc).
    """
    from sklearn.model_selection import StratifiedGroupKFold
    groups = duplicate_group_ids(df)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df[TARGET], groups)):
        fold[idx] = k
    part = np.where(np.isin(fold, TEST_FOLDS), "test",
                    np.where(np.isin(fold, VAL_FOLDS), "val", "train"))
    return {name: df[part == name] for name in ("train", "val", "test")}


def split_data_legacy(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """[CŨ - chỉ để so sánh] Chia 2 bước train_test_split, không xét nhóm trùng.
    Chia 2 bước để được 70/15/15 có phân tầng theo nhãn.

    Bước 1: tách Test 15%. Bước 2: từ 85% còn lại tách Validation = 15/85.
    Trả về dict {'train','val','test'} -> DataFrame (giữ index gốc).
    """
    rest, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET],
                                  random_state=random_state)
    train, val = train_test_split(rest, test_size=VAL_SIZE / (TRAIN_SIZE + VAL_SIZE),
                                  stratify=rest[TARGET], random_state=random_state)
    return {"train": train, "val": val, "test": test}


def xy(part: pd.DataFrame):
    """Tách X (chỉ các đặc trưng được dùng) và y."""
    return part[NUMERIC_FEATURES + CATEGORICAL_FEATURES], part[TARGET]


def split_summary(splits: dict) -> pd.DataFrame:
    rows = []
    total = sum(len(p) for p in splits.values())
    for name, p in splits.items():
        n1 = int(p[TARGET].sum())
        rows.append({"split": name, "n_samples": len(p),
                     "fraction": round(len(p) / total, 4),
                     "n_le50K": len(p) - n1, "n_gt50K": n1,
                     "ratio_gt50K": round(n1 / len(p), 4)})
    return pd.DataFrame(rows)


def save_splits(splits: dict, out_dir: Path = PROCESSED_DIR):
    """Lưu chỉ số dòng (index trong file gốc) của từng tập + bảng tóm tắt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    full = pd.concat(splits.values()).sort_index()
    groups = duplicate_group_ids(full)
    idx = pd.concat([pd.DataFrame({"row_index": p.index, "split": name})
                     for name, p in splits.items()]).sort_values("row_index")
    idx["dup_group"] = groups.loc[idx["row_index"]].values
    idx["dup_group_size"] = idx["dup_group"].map(idx["dup_group"].value_counts())
    idx.to_csv(out_dir / "split_indices.csv", index=False)
    summary = split_summary(splits)
    summary.to_csv(out_dir / "split_summary.csv", index=False)
    return summary


def get_splits():
    """Điểm vào duy nhất cho mọi module: luôn cho ra cùng 3 tập."""
    return split_data(clean(load_raw()))
