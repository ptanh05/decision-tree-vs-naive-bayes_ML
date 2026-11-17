"""Decision Tree (thuật toán CART của scikit-learn) – mô hình chính số 1.

Pipeline gồm:
1. build_dt_preprocessor(): giữ nguyên số + One-Hot biến phân loại.
2. DecisionTreeClassifier: học quy tắc chia nhánh từ dữ liệu huấn luyện.
"""
from itertools import product

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data_loader import RANDOM_STATE
from preprocessing import build_dt_preprocessor

# Lưới siêu tham số: 2 x 10 x 4 x 5 x 2 = 800 cấu hình
DT_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15, 20, None],
    "min_samples_split": [2, 10, 50, 100],
    "min_samples_leaf": [1, 5, 10, 20, 50],
    "class_weight": [None, "balanced"],
}


def dt_configs() -> list[dict]:
    """Sinh toàn bộ 800 tổ hợp siêu tham số từ DT_GRID."""
    keys = list(DT_GRID)
    return [dict(zip(keys, values)) for values in product(*DT_GRID.values())]


def build_dt(
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str | None = None,
) -> Pipeline:
    """Tạo Pipeline hoàn chỉnh cho Decision Tree."""
    classifier = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("pre", build_dt_preprocessor()),
        ("model", classifier),
    ])


def dt_complexity(pipe: Pipeline) -> dict:
    """Đo độ phức tạp của cây sau khi fit: số lá và độ sâu thực tế."""
    tree = pipe.named_steps["model"]
    return {
        "n_leaves": int(tree.get_n_leaves()),
        "depth": int(tree.get_depth()),
    }
