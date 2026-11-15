"""Pipeline tiền xử lý cho từng thuật toán.

Nguyên tắc chống rò rỉ dữ liệu: mọi bước có tham số học được đều nằm trong
sklearn Pipeline. Gọi pipeline.fit(X_train) -> chỉ học từ Train; gọi
predict(X_val / X_test) -> chỉ transform. Khi huấn luyện lại trên
Train+Validation, toàn bộ pipeline (kể cả preprocessing) được fit lại.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder

from data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES

UNKNOWN = "Unknown"


def _cat_imputer():
    # '?' đã thành NaN -> thay bằng hạng mục riêng "Unknown" (không xóa dòng).
    return SimpleImputer(strategy="constant", fill_value=UNKNOWN)


class SafeOrdinalEncoder(BaseEstimator, TransformerMixin):
    """OrdinalEncoder cho CategoricalNB.

    Giá trị chưa thấy trong Train (ví dụ quốc gia hiếm chỉ xuất hiện ở Test)
    được ánh xạ vào một mã dành riêng = số hạng mục của cột, để đầu vào luôn
    là số nguyên không âm như CategoricalNB yêu cầu.
    """

    def fit(self, X, y=None):
        self.enc_ = OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1, dtype=np.int64).fit(X)
        self.n_categories_ = np.array([len(c) for c in self.enc_.categories_])
        return self

    def transform(self, X):
        Z = self.enc_.transform(X).astype(np.int64)
        for j in range(Z.shape[1]):
            Z[Z[:, j] == -1, j] = self.n_categories_[j]
        return Z

    def get_feature_names_out(self, input_features=None):
        return self.enc_.get_feature_names_out(input_features)


class ZeroInflatedDiscretizer(BaseEstimator, TransformerMixin):
    """Rời rạc hóa cho cột lệch về 0 (capital.gain, capital.loss).

    KBinsDiscretizer(quantile) gộp các cột này còn 1 khoảng vì >90% giá trị = 0.
    Ở đây: mã 0 = giá trị 0; giá trị > 0 chia theo quantile của các giá trị
    dương TRONG TRAIN thành tối đa n_bins khoảng (mã 1..k).
    """

    def __init__(self, n_bins: int = 4):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.edges_, self.n_bins_ = [], []
        for j in range(X.shape[1]):
            pos = X[X[:, j] > 0, j]
            qs = np.unique(np.quantile(pos, np.linspace(0, 1, self.n_bins + 1)[1:-1])) \
                if len(pos) else np.array([])
            self.edges_.append(qs)
            self.n_bins_.append(len(qs) + 2)  # khoảng "=0" + (len(qs)+1) khoảng dương
        self.n_bins_ = np.array(self.n_bins_)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Z = np.zeros(X.shape, dtype=np.int64)
        for j, e in enumerate(self.edges_):
            pos = X[:, j] > 0
            Z[pos, j] = 1 + np.searchsorted(e, X[pos, j], side="right")
        return Z

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


CAPITAL_FEATURES = ["capital.gain", "capital.loss"]
SMOOTH_NUMERIC = [c for c in NUMERIC_FEATURES if c not in CAPITAL_FEATURES]


def build_dt_preprocessor():
    """Decision Tree: số giữ nguyên (cây không cần chuẩn hóa),
    phân loại -> One-Hot (tránh áp đặt thứ tự giả cho hạng mục)."""
    cat = Pipeline([("impute", _cat_imputer()),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", "passthrough", NUMERIC_FEATURES),
                              ("cat", cat, CATEGORICAL_FEATURES)])


def build_cnb_preprocessor(n_bins: int = 10):
    """CategoricalNB (thí nghiệm chính): mọi đặc trưng thành mã nguyên không âm.

    - age, fnlwgt, education.num, hours.per.week -> KBinsDiscretizer(quantile),
      biên khoảng học từ Train (biên trùng được gộp tự động).
    - capital.gain/loss -> ZeroInflatedDiscretizer (0 riêng + quantile phần dương).
    - Phân loại -> impute "Unknown" -> SafeOrdinalEncoder.
    """
    num = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile",
                           quantile_method="averaged_inverted_cdf", dtype=np.float64)
    cat = Pipeline([("impute", _cat_imputer()), ("ordinal", SafeOrdinalEncoder())])
    return ColumnTransformer([("num", num, SMOOTH_NUMERIC),
                              ("capital", ZeroInflatedDiscretizer(), CAPITAL_FEATURES),
                              ("cat", cat, CATEGORICAL_FEATURES)])


def build_gnb_preprocessor():
    """GaussianNB (thí nghiệm phụ): chỉ 6 đặc trưng số, giữ nguyên giá trị.
    GaussianNB tự ước lượng mean/var theo lớp từ Train; chuẩn hóa không đổi kết quả."""
    return ColumnTransformer([("num", "passthrough", NUMERIC_FEATURES)])


def cnb_min_categories(fitted_preprocessor):
    """Số hạng mục tối đa mỗi cột (kể cả mã 'unseen') để khai báo cho
    CategoricalNB, tránh lỗi khi Test có mã không xuất hiện trong Train."""
    kb = fitted_preprocessor.named_transformers_["num"]
    cap = fitted_preprocessor.named_transformers_["capital"]
    enc = fitted_preprocessor.named_transformers_["cat"].named_steps["ordinal"]
    return np.concatenate([kb.n_bins_, cap.n_bins_, enc.n_categories_ + 1]).astype(int)
