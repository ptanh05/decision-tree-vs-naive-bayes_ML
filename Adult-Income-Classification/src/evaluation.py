"""Các chỉ số đánh giá dùng chung cho mọi mô hình.

Quy ước cố định của thực nghiệm:
- Lớp dương (positive) = 1 (thu nhập >50K).
- Tiêu chí chọn siêu tham số: F1 của lớp 1 trên VALIDATION.
- Ngưỡng phân loại cố định 0.5 (predict mặc định); KHÔNG tinh chỉnh ngưỡng.
- Test chỉ dùng một lần sau khi đã chốt cấu hình và refit trên Train+Val.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

POS_LABEL = 1
SELECTION_METRIC = "f1"


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """y_score = xác suất dự đoán cho lớp 1 (dùng cho ROC-AUC, PR-AUC)."""
    y_true = np.asarray(y_true)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "roc_auc": roc_auc_score(y_true == POS_LABEL, y_score),
        # PR-AUC = Average Precision cho lớp 1
        "pr_auc": average_precision_score(y_true, y_score, pos_label=POS_LABEL),
    }


def positive_proba(model, X):
    """Lấy cột xác suất của lớp 1 theo model.classes_ (không giả định vị trí cột)."""
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(POS_LABEL)]
