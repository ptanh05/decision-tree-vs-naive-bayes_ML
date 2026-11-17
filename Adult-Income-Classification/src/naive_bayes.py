"""Naive Bayes – mô hình chính số 2 (CategoricalNB) và thí nghiệm phụ (GaussianNB).

Hai mô hình Naive Bayes hoạt động hoàn toàn độc lập với Decision Tree.
"""
from itertools import product

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.pipeline import Pipeline

from preprocessing import (
    build_cnb_preprocessor,
    build_gnb_preprocessor,
    cnb_min_categories,
)

# Lưới siêu tham số cho CategoricalNB: 9 x 7 x 2 = 126 cấu hình
# Lưu ý: lưới ban đầu n_bins={5..20} đạt đỉnh tại biên 20 nên được mở rộng thêm {25..50}.
CNB_GRID = {
    "n_bins": [5, 8, 10, 15, 20, 25, 30, 40, 50],
    "alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "fit_prior": [True, False],
}

# Lưới siêu tham số cho GaussianNB (thí nghiệm phụ chỉ dùng 6 cột số): 7 cấu hình
GNB_GRID = {
    "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
}


def _expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*grid.values())]


def cnb_configs() -> list[dict]:
    """Sinh toàn bộ 126 tổ hợp siêu tham số cho CategoricalNB."""
    return _expand_grid(CNB_GRID)


def gnb_configs() -> list[dict]:
    """Sinh toàn bộ 7 cấu hình siêu tham số cho GaussianNB."""
    return _expand_grid(GNB_GRID)


class CategoricalNBPipeline(BaseEstimator, ClassifierMixin):
    """Bọc bộ tiền xử lý và CategoricalNB thành một estimator thống nhất.

    Lý do phải viết lớp bọc: tham số min_categories của CategoricalNB phụ thuộc vào
    số khoảng và số hạng mục mà bộ tiền xử lý học được từ chính tập dữ liệu fit.
    """

    def __init__(self, n_bins: int = 10, alpha: float = 1.0, fit_prior: bool = True):
        self.n_bins = n_bins
        self.alpha = alpha
        self.fit_prior = fit_prior

    def fit(self, X, y):
        self.pre_ = build_cnb_preprocessor(n_bins=self.n_bins).fit(X)
        X_encoded = self.pre_.transform(X)

        self.model_ = CategoricalNB(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
            min_categories=cnb_min_categories(self.pre_),
        )
        self.model_.fit(X_encoded, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        X_encoded = self.pre_.transform(X)
        return self.model_.predict_proba(X_encoded)

    def predict(self, X):
        proba = self.predict_proba(X)
        best_class_idx = np.argmax(proba, axis=1)
        return self.classes_[best_class_idx]


def build_cnb(n_bins: int = 10, alpha: float = 1.0, fit_prior: bool = True) -> CategoricalNBPipeline:
    """Tạo Pipeline hoàn chỉnh cho CategoricalNB."""
    return CategoricalNBPipeline(n_bins=n_bins, alpha=alpha, fit_prior=fit_prior)


def build_gnb(var_smoothing: float = 1e-9) -> Pipeline:
    """Tạo Pipeline hoàn chỉnh cho GaussianNB (thí nghiệm phụ)."""
    return Pipeline([
        ("pre", build_gnb_preprocessor()),
        ("model", GaussianNB(var_smoothing=var_smoothing)),
    ])
