"""Pipeline tiền xử lý riêng cho từng thuật toán.

Nguyên tắc chống rò rỉ dữ liệu:
- Mọi bước có học tham số đều nằm trong sklearn Pipeline / ColumnTransformer.
- Khi gọi fit(X_train), tham số chỉ được học từ Train.
- Khi gọi transform(X_val / X_test), chỉ áp dụng lại tham số đã học.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder

from data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES

UNKNOWN_CATEGORY = "Unknown"
CAPITAL_FEATURES = ["capital.gain", "capital.loss"]
SMOOTH_NUMERIC_FEATURES = [col for col in NUMERIC_FEATURES if col not in CAPITAL_FEATURES]

# Alias tương thích ngược cho notebooks/02_Preprocessing.ipynb và scripts/make_preprocessing_notebook.py
UNKNOWN = UNKNOWN_CATEGORY
SMOOTH_NUMERIC = SMOOTH_NUMERIC_FEATURES


def _make_categorical_imputer() -> SimpleImputer:
    """Điền giá trị thiếu (NaN) bằng hạng mục riêng 'Unknown', không xóa dòng."""
    return SimpleImputer(strategy="constant", fill_value=UNKNOWN_CATEGORY)


class SafeOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Mã hóa cột phân loại thành số nguyên không âm (0, 1, 2, ...) cho CategoricalNB.

    Nếu ở Validation/Test xuất hiện hạng mục chưa từng thấy trong Train,
    nó được gán mã riêng = số lượng hạng mục đã biết (thay vì -1 gây lỗi cho CategoricalNB).

    Ví dụ:
        Train có ['Private', 'Self-emp', 'Unknown'] -> gán mã 0, 1, 2 (n_categories_ = 3).
        Test gặp 'Never-worked' (chưa thấy) -> OrdinalEncoder trả -1 -> đổi thành mã 3.
    """

    def fit(self, X, y=None):
        self.enc_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
        self.enc_.fit(X)
        self.n_categories_ = np.array([len(cats) for cats in self.enc_.categories_])
        return self

    def transform(self, X):
        encoded = self.enc_.transform(X).astype(np.int64)
        for col_idx in range(encoded.shape[1]):
            unseen_mask = encoded[:, col_idx] == -1
            encoded[unseen_mask, col_idx] = self.n_categories_[col_idx]
        return encoded

    def get_feature_names_out(self, input_features=None):
        return self.enc_.get_feature_names_out(input_features)


class ZeroInflatedDiscretizer(BaseEstimator, TransformerMixin):
    """Rời rạc hóa cho các cột có trên 90% giá trị bằng 0 (capital.gain, capital.loss).

    Lý do: KBinsDiscretizer(strategy="quantile") thông thường bị trùng mốc phân vị về 0
    và gộp cả cột thành 1 khoảng duy nhất, làm mất toàn bộ tín hiệu.

    Cách hoạt động:
    - Giá trị == 0 được giữ riêng ở mã 0.
    - Các giá trị > 0 được chia theo phân vị (chỉ học trên Train) thành các mã 1..k.

    Ví dụ (n_bins=4, edges_=[3425, 7298, 14084]):
        0 -> 0 | 2000 -> 1 | 5000 -> 2 | 10000 -> 3 | 99999 -> 4.
    """

    def __init__(self, n_bins: int = 4):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.edges_ = []
        n_bins_per_col = []

        # Các mốc phân vị nằm trong khoảng (0, 1), ví dụ với n_bins=4 là [0.25, 0.50, 0.75]
        quantile_probs = np.linspace(0, 1, self.n_bins + 1)[1:-1]

        for col_idx in range(arr.shape[1]):
            positive_vals = arr[arr[:, col_idx] > 0, col_idx]
            if len(positive_vals) > 0:
                edges = np.unique(np.quantile(positive_vals, quantile_probs))
            else:
                edges = np.array([])

            self.edges_.append(edges)
            # Tổng số khoảng = 1 khoảng cho giá trị 0 + (len(edges) + 1) khoảng cho giá trị > 0
            n_bins_per_col.append(len(edges) + 2)

        self.n_bins_ = np.array(n_bins_per_col)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        result = np.zeros(arr.shape, dtype=np.int64)

        for col_idx, edges in enumerate(self.edges_):
            positive_mask = arr[:, col_idx] > 0
            bin_ids = 1 + np.searchsorted(edges, arr[positive_mask, col_idx], side="right")
            result[positive_mask, col_idx] = bin_ids

        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_dt_preprocessor() -> ColumnTransformer:
    """Tiền xử lý cho Decision Tree: giữ nguyên cột số, One-Hot cột phân loại."""
    categorical_pipe = Pipeline([
        ("impute", _make_categorical_imputer()),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def build_cnb_preprocessor(n_bins: int = 10) -> ColumnTransformer:
    """Tiền xử lý cho CategoricalNB (mô hình chính): đưa mọi cột về mã nguyên không âm."""
    # Từ scikit-learn 1.9.0, mặc định của strategy="quantile" đã là quantile_method="averaged_inverted_cdf"
    numeric_discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="quantile",
        dtype=np.float64,
    )
    categorical_pipe = Pipeline([
        ("impute", _make_categorical_imputer()),
        ("ordinal", SafeOrdinalEncoder()),
    ])
    return ColumnTransformer([
        ("num", numeric_discretizer, SMOOTH_NUMERIC_FEATURES),
        ("capital", ZeroInflatedDiscretizer(), CAPITAL_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def build_gnb_preprocessor() -> ColumnTransformer:
    """Tiền xử lý cho GaussianNB (thí nghiệm phụ): chỉ giữ lại 6 đặc trưng số."""
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
    ])


def cnb_min_categories(fitted_preprocessor: ColumnTransformer) -> np.ndarray:
    """Trả về số hạng mục tối đa của từng cột để khai báo cho CategoricalNB."""
    num_bins = fitted_preprocessor.named_transformers_["num"].n_bins_
    capital_bins = fitted_preprocessor.named_transformers_["capital"].n_bins_
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["ordinal"]
    # Cộng thêm 1 cho mỗi cột phân loại để chừa chỗ cho mã 'unseen'
    cat_bins = cat_encoder.n_categories_ + 1
    return np.concatenate([num_bins, capital_bins, cat_bins]).astype(int)
