"""Đọc dữ liệu thô và chia Train / Validation / Test (70/15/15, stratified theo nhóm, seed 42).

Lưu ý quan trọng:
- Bước làm sạch ở đây chỉ gồm các phép biến đổi KHÔNG học tham số
  (xóa khoảng trắng thừa, đổi '?' -> NaN, mã hóa nhãn).
- Mọi bước có học tham số (imputer, encoder, discretizer) nằm trong Pipeline
  ở preprocessing.py và chỉ được fit trên Train (hoặc Train+Val khi refit).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "adult.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
TARGET = "income"

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]

# Bỏ cột 'education' vì ánh xạ 1-1 với 'education.num' (tránh trùng lặp thông tin).
DROPPED_FEATURES = ["education"]

N_FOLDS = 20               # Mỗi fold chiếm ~5% dữ liệu
TEST_FOLDS = (0, 1, 2)     # 3 fold = 15% cho Test
VAL_FOLDS = (3, 4, 5)      # 3 fold = 15% cho Validation (14 fold còn lại = 70% cho Train)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Đọc file CSV gốc, không chỉnh sửa file trên đĩa."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch không học tham số: strip khoảng trắng, '?' -> NaN, nhãn -> 0/1."""
    cleaned = df.copy()

    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    cleaned = cleaned.replace("?", np.nan)

    label_mapping = {"<=50K": 0, ">50K": 1}
    unknown_labels = set(cleaned[TARGET].unique()) - set(label_mapping)
    if unknown_labels:
        raise ValueError(f"Nhãn không mong đợi trong cột {TARGET}: {unknown_labels}")

    cleaned[TARGET] = cleaned[TARGET].map(label_mapping).astype(int)
    return cleaned


def duplicate_group_ids(df: pd.DataFrame) -> pd.Series:
    """Gán ID nhóm cho các dòng giống hệt nhau ở cả 15 cột (kể cả nhãn).

    Mỗi dòng duy nhất là một nhóm 1 phần tử; các dòng trùng nhau có chung ID
    để đảm bảo khi chia tập chúng luôn nằm cùng một chỗ.
    """
    row_signature = df.fillna("?").astype(str).agg("|".join, axis=1)
    group_ids, _ = pd.factorize(row_signature)
    return pd.Series(group_ids, index=df.index, name="dup_group")


def split_data(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> dict:
    """Chia dữ liệu thành Train / Validation / Test (~70/15/15) có phân tầng theo nhóm.

    Dùng StratifiedGroupKFold(20 fold, shuffle, seed 42):
    - Các bản ghi trùng lặp luôn nằm trọn trong cùng một fold.
    - Fold 0-2 -> Test, fold 3-5 -> Validation, fold 6-19 -> Train.
    """
    groups = duplicate_group_ids(df)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)

    fold_ids = np.empty(len(df), dtype=int)
    for fold_idx, (_, val_indices) in enumerate(splitter.split(df, df[TARGET], groups)):
        fold_ids[val_indices] = fold_idx

    split_labels = pd.Series("train", index=df.index)
    split_labels[np.isin(fold_ids, TEST_FOLDS)] = "test"
    split_labels[np.isin(fold_ids, VAL_FOLDS)] = "val"

    return {
        "train": df[split_labels == "train"],
        "val": df[split_labels == "val"],
        "test": df[split_labels == "test"],
    }


def split_data_legacy(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> dict:
    """[Cũ - chỉ dùng để đối chiếu] Chia 2 bước train_test_split, không xét nhóm trùng."""
    rest, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=random_state,
    )
    val_ratio_in_rest = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    train, val = train_test_split(
        rest,
        test_size=val_ratio_in_rest,
        stratify=rest[TARGET],
        random_state=random_state,
    )
    return {"train": train, "val": val, "test": test}


def xy(part: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Tách ma trận đặc trưng X (13 cột được dùng) và nhãn y."""
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return part[feature_cols], part[TARGET]


def split_summary(splits: dict) -> pd.DataFrame:
    """Tạo bảng thống kê số lượng mẫu và tỷ lệ lớp >50K của từng tập."""
    total_rows = sum(len(part) for part in splits.values())
    rows = []

    for split_name, part in splits.items():
        n_pos = int(part[TARGET].sum())
        n_total = len(part)
        rows.append({
            "split": split_name,
            "n_samples": n_total,
            "fraction": round(n_total / total_rows, 4),
            "n_le50K": n_total - n_pos,
            "n_gt50K": n_pos,
            "ratio_gt50K": round(n_pos / n_total, 4),
        })

    return pd.DataFrame(rows)


def save_splits(splits: dict, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Lưu chỉ số dòng của từng tập (split_indices.csv) và bảng tóm tắt (split_summary.csv)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.concat(splits.values()).sort_index()
    groups = duplicate_group_ids(full_df)

    index_frames = [
        pd.DataFrame({"row_index": part.index, "split": split_name})
        for split_name, part in splits.items()
    ]
    split_index_df = pd.concat(index_frames).sort_values("row_index")
    split_index_df["dup_group"] = groups.loc[split_index_df["row_index"]].values

    group_sizes = split_index_df["dup_group"].value_counts()
    split_index_df["dup_group_size"] = split_index_df["dup_group"].map(group_sizes)
    split_index_df.to_csv(out_dir / "split_indices.csv", index=False)

    summary = split_summary(splits)
    summary.to_csv(out_dir / "split_summary.csv", index=False)
    return summary


def get_splits() -> dict:
    """Điểm vào duy nhất cho mọi module: đọc, làm sạch và chia cùng 3 tập."""
    raw_df = load_raw()
    cleaned_df = clean(raw_df)
    return split_data(cleaned_df)
