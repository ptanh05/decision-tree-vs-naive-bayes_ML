"""Các chỉ số đánh giá dùng chung cho mọi mô hình.

Quy ước cố định của thực nghiệm:
- Lớp dương (positive) = 1 (thu nhập >50K); lớp âm = 0 (<=50K).
- Tiêu chí chọn siêu tham số: F1 của lớp 1 trên VALIDATION.
- Ngưỡng phân loại cố định 0.5 (predict mặc định); KHÔNG tinh chỉnh ngưỡng.
- Test chỉ dùng một lần sau khi đã chốt cấu hình và refit trên Train+Val.

Giải thích các chỉ số trong compute_metrics():
- accuracy : (TP + TN) / N — dễ gây hiểu lầm khi mất cân bằng (baseline <=50K đạt ~0.7592).
- precision: TP / (TP + FP) — trong các ca đoán >50K, bao nhiêu phần trăm là đúng.
- recall   : TP / (TP + FN) — trong toàn bộ người thật sự >50K, tìm được bao nhiêu phần trăm.
- f1       : 2 * P * R / (P + R) — trung bình điều hòa cân bằng Precision và Recall (tiêu chí chọn).
- roc_auc  : Diện tích dưới đường ROC (tính trên xác suất y_score, độc lập ngưỡng 0.5).
- pr_auc   : Lưu ý thuật ngữ — thực chất tính bằng `average_precision_score` (Average Precision,
             tổng bậc thang trên đường Precision-Recall), KHÔNG phải tích phân hình thang.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

POS_LABEL = 1
SELECTION_METRIC = "f1"


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """Tính 6 chỉ số đánh giá từ nhãn thật (y_true), nhãn dự đoán (y_pred) và xác suất lớp 1 (y_score).

    Lưu ý: key 'pr_auc' lưu giá trị Average Precision (`average_precision_score`), nhạy với lớp hiếm >50K.
    """
    y_true = np.asarray(y_true)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "roc_auc": roc_auc_score(y_true == POS_LABEL, y_score),
        # PR-AUC = Average Precision (tổng bậc thang) cho lớp dương 1
        "pr_auc": average_precision_score(y_true, y_score, pos_label=POS_LABEL),
    }


def positive_proba(model, X):
    """Lấy cột xác suất P(y = 1 | x) theo model.classes_ (không giả định vị trí cột 1)."""
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(POS_LABEL)]
