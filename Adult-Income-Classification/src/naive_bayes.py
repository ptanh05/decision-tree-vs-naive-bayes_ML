"""Naive Bayes – mô hình chính số 2 (CategoricalNB) và thí nghiệm phụ (GaussianNB).

Hai mô hình NB độc lập hoàn toàn với Decision Tree (không NBTree, không lai).
"""
from itertools import product

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.pipeline import Pipeline

from preprocessing import (build_cnb_preprocessor, build_gnb_preprocessor,
                           cnb_min_categories)

# CategoricalNB (9 x 7 x 2 = 126 cấu hình)
# - n_bins: số khoảng rời rạc hóa cho age, fnlwgt, education.num, hours.per.week.
#   Lưới ban đầu {5..20} cho cấu hình tốt nhất tại biên n_bins=20 (F1 tăng đơn điệu),
#   nên được mở rộng thêm {25, 30, 40, 50} – vẫn chỉ đánh giá trên Validation.
#   Kết quả lưới ban đầu lưu tại results/metrics/archive/.
# - alpha: hệ số làm trơn Laplace/Lidstone cho P(x_j = v | y).
# - fit_prior: True = prior theo tần suất lớp trong Train (~76/24);
#   False = prior đồng đều (cách tương ứng của NB để xử lý mất cân bằng).
CNB_GRID = {
    "n_bins": [5, 8, 10, 15, 20, 25, 30, 40, 50],
    "alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "fit_prior": [True, False],
}

# GaussianNB – THÍ NGHIỆM PHỤ (7 cấu hình), chỉ 6 thuộc tính số.
GNB_GRID = {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]}


def _configs(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in product(*grid.values())]


cnb_configs = lambda: _configs(CNB_GRID)
gnb_configs = lambda: _configs(GNB_GRID)


class CategoricalNBPipeline(BaseEstimator, ClassifierMixin):
    """Preprocessing (rời rạc hóa + mã hóa) + CategoricalNB trong một estimator.

    Cần lớp bọc vì `min_categories` của CategoricalNB phụ thuộc vào số khoảng/hạng mục
    mà preprocessor học được từ CHÍNH dữ liệu fit (Train, hoặc Train+Val khi refit).
    """

    def __init__(self, n_bins=10, alpha=1.0, fit_prior=True):
        self.n_bins = n_bins
        self.alpha = alpha
        self.fit_prior = fit_prior

    def fit(self, X, y):
        self.pre_ = build_cnb_preprocessor(n_bins=self.n_bins).fit(X)
        self.model_ = CategoricalNB(alpha=self.alpha, fit_prior=self.fit_prior,
                                    min_categories=cnb_min_categories(self.pre_))
        self.model_.fit(self.pre_.transform(X), y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self.pre_.transform(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_cnb(n_bins=10, alpha=1.0, fit_prior=True):
    return CategoricalNBPipeline(n_bins=n_bins, alpha=alpha, fit_prior=fit_prior)


def build_gnb(var_smoothing=1e-9):
    return Pipeline([("pre", build_gnb_preprocessor()),
                     ("model", GaussianNB(var_smoothing=var_smoothing))])
