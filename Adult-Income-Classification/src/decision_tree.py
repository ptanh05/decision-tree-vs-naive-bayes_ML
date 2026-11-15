"""Decision Tree (CART của scikit-learn) – mô hình chính số 1.

Pipeline = build_dt_preprocessor() (fit trên dữ liệu được đưa vào) + DecisionTreeClassifier.
Không gộp với Naive Bayes dưới bất kỳ hình thức nào.
"""
from itertools import product

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data_loader import RANDOM_STATE
from preprocessing import build_dt_preprocessor

# Không gian tìm kiếm (2 x 10 x 4 x 5 x 2 = 800 cấu hình).
# - criterion: hai thước đo tạp chất đã học (Gini, Entropy).
# - max_depth: từ cây nông dễ đọc (3) tới không giới hạn (None) để thấy rõ overfitting.
# - min_samples_split / min_samples_leaf: điều kiện dừng (tiền cắt tỉa) chống overfitting.
# - class_weight: None hoặc 'balanced' – cách chuẩn của DT để xử lý mất cân bằng lớp
#   (đổi trọng số mẫu khi tính tạp chất), vẫn là cùng một thuật toán Decision Tree.
DT_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15, 20, None],
    "min_samples_split": [2, 10, 50, 100],
    "min_samples_leaf": [1, 5, 10, 20, 50],
    "class_weight": [None, "balanced"],
}


def dt_configs():
    keys = list(DT_GRID)
    return [dict(zip(keys, v)) for v in product(*DT_GRID.values())]


def build_dt(criterion="gini", max_depth=None, min_samples_split=2,
             min_samples_leaf=1, class_weight=None):
    return Pipeline([
        ("pre", build_dt_preprocessor()),
        ("model", DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                         min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf,
                                         class_weight=class_weight,
                                         random_state=RANDOM_STATE)),
    ])


def dt_complexity(pipe) -> dict:
    t = pipe.named_steps["model"]
    return {"n_leaves": int(t.get_n_leaves()), "depth": int(t.get_depth())}
