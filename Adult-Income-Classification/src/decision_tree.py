"""Decision Tree (thuật toán CART của scikit-learn) – mô hình chính số 1.

Pipeline gồm:
1. build_dt_preprocessor(): giữ nguyên 6 cột số + One-Hot 7 cột phân loại (92 đặc trưng).
2. DecisionTreeClassifier: chia không gian đặc trưng theo ngưỡng nhị phân để giảm độ hỗn tạp.
   Không lai ghép với Naive Bayes dưới bất kỳ hình thức nào.

Giải thích các siêu tham số trong DT_GRID:
- criterion        : Thước đo độ hỗn tạp khi chia nút ("gini" hoặc "entropy" / Information Gain).
- max_depth        : Độ sâu tối đa của cây (tiền cắt tỉa); None = không giới hạn để quan sát overfitting.
- min_samples_split: Số mẫu tối thiểu để một nút trong được phép chia tiếp.
- min_samples_leaf : Số mẫu tối thiểu phải có ở mỗi nút lá (giúp làm mượt xác suất lá và chống nhiễu).
- class_weight     : None = trọng số đều; "balanced" = gán trọng số tỉ lệ nghịch tần suất lớp
                     (trên Train, mỗi mẫu >50K nặng ~3.15 lần mẫu <=50K), giúp tăng Recall lớp >50K.
- random_state     : Cố định seed 42 để chọn nhánh tái lập được khi các phép chia có điểm bằng nhau.
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
    """Đo độ phức tạp của cây sau khi fit: số lá (n_leaves) và độ sâu thực tế (depth)."""
    tree = pipe.named_steps["model"]
    return {
        "n_leaves": int(tree.get_n_leaves()),
        "depth": int(tree.get_depth()),
    }
