# PHASE 4 – Lựa chọn siêu tham số trên Validation

- Thời điểm chạy: 2026-11-15 (lần chạy đầy đủ `20261115_232648`, chạy tiếp lưới CNB mở rộng `20261115_233334`; log trong `results/logs/`).
- Môi trường: Windows 11 (10.0.26300), Python 3.12.10, scikit-learn 1.9.0, pandas 3.1.1, numpy 2.4.4; CPU, 1 tiến trình; `random_state=42`.
- Lệnh: `python src/tuning.py` rồi `python src/tuning.py --resume` (sau khi mở rộng lưới CNB).
- Dữ liệu: Train 22.793 / Validation 4.884 (chia theo nhóm, StratifiedGroupKFold). **Test (4.884) không được dùng.**
- Tiêu chí chọn: F1 lớp >50K (pos_label=1) trên Validation, ngưỡng 0,5. Phá hòa: PR-AUC → số lá ít hơn (DT) → thứ tự cấu hình.

## Không gian tìm kiếm
| Mô hình | Vai trò | Lưới | Số cấu hình |
|---|---|---|---|
| DecisionTree | chính | criterion {gini, entropy} × max_depth {3,4,5,6,8,10,12,15,20,None} × min_samples_split {2,10,50,100} × min_samples_leaf {1,5,10,20,50} × class_weight {None, balanced} | 800 |
| CategoricalNB | chính | n_bins {5,8,10,15,20,25,30,40,50} × alpha {0.01,0.1,0.5,1,2,5,10} × fit_prior {True, False} | 126 |
| GaussianNB | phụ | var_smoothing {1e-9 … 1e-3} | 7 |

Lưới CNB ban đầu chỉ có n_bins {5..20}; cấu hình tốt nhất rơi đúng biên 20 nên lưới được mở rộng (chỉ dùng Validation). Kết quả lưới ban đầu được lưu trữ ở `results/metrics/archive/20261115_232648_cnb_grid_n_bins_5-20/`. Cấu hình tốt nhất không đổi (n_bins=20 giờ nằm trong lưới).

## Cấu hình tốt nhất (mô hình fit trên Train, đánh giá Validation)
| | DecisionTree | CategoricalNB | GaussianNB (phụ) | Luôn đoán <=50K (tham chiếu) |
|---|---|---|---|---|
| Cấu hình | entropy, max_depth=12, min_samples_split=2, min_samples_leaf=10, class_weight=balanced | n_bins=20, alpha=0.1, fit_prior=True | var_smoothing=1e-7 | – |
| Accuracy | 0,8073 | 0,8172 | 0,7920 | 0,7592 |
| Precision (>50K) | 0,5667 | 0,5909 | 0,6562 | 0 |
| Recall (>50K) | 0,8486 | 0,7823 | 0,2857 | 0 |
| **F1 (>50K)** | **0,6796** | **0,6733** | 0,3981 | 0 |
| ROC-AUC | 0,8927 | 0,9004 | 0,7517 | 0,5 |
| PR-AUC | 0,7623 | 0,7584 | 0,5515 | 0,2408 |
| TN / FP / FN / TP | 2945 / 763 / 178 / 998 | 3071 / 637 / 256 / 920 | 3532 / 176 / 840 / 336 | 3708 / 0 / 1176 / 0 |
| Train F1 | 0,7094 | 0,6811 | 0,4229 | – |
| Fit time (trung vị 5 lần) | 0,165 s | 0,122 s | 0,003 s | – |
| Predict time Validation | 0,025 s | 0,026 s | 0,002 s | – |

Nguồn: `results/metrics/validation_comparison.csv`, `best_configs.json`.

## Nhận xét (chỉ trên Validation – chưa phải kết luận cuối)
- Hai mô hình chính rất sát nhau: chênh F1 0,006 (DT cao hơn), ROC-AUC 0,008 (CNB cao hơn). Mức chênh này nhỏ; cần đánh giá trên Test và khoảng tin cậy (bootstrap) ở PHASE 6 trước khi kết luận.
- DT chọn `class_weight=balanced` → Recall cao (0,849) đổi lấy Precision thấp hơn; không cân bằng trọng số, F1 tốt nhất của DT là 0,6738.
- CNB: `fit_prior=True` tốt hơn prior đồng đều (0,6733 so với 0,6542). F1 gần như không nhạy với alpha (0,01–5) và n_bins ≥ 15 (0,6708–0,6733).
- Overfitting của DT: cây không giới hạn (3.146–3.222 lá) đạt Train F1 0,9999 nhưng Validation F1 chỉ ~0,60; cấu hình chọn có 324 lá, độ sâu 12, chênh Train–Val F1 = 0,030.
- GaussianNB (phụ, chỉ 6 thuộc tính số) kém rõ rệt (F1 0,398), phù hợp với EDA: capital.gain/loss lệch mạnh, không Gaussian, và thiếu toàn bộ thuộc tính phân loại.
- Cấu hình top-1 và top-2 của DT cho dự đoán giống hệt nhau (min_samples_split 2 và 10 không có tác dụng khi min_samples_leaf=10); phá hòa theo thứ tự lưới.

## Kiểm tra
- Tái lập: cấu hình tốt nhất của mỗi mô hình được fit lại 2 lần → dự đoán, xác suất và metric trùng khớp với kết quả lưới (sai khác ≤ 1e-12).
- Test: `tuning.py` loại Test khỏi bộ nhớ ngay sau khi chia; không đọc nhãn Test. Trước PHASE 4, `check_design.py` chỉ gọi `predict` trên Test để kiểm tra kỹ thuật (không tính metric, không đọc nhãn).
- Thời gian đo có dao động giữa các lần chạy (DT fit 0,230 s ở lần 1 so với 0,165 s ở lần 2) → báo cáo trung vị 5 lần.

## Bước tiếp theo (chờ duyệt)
Refit từng cấu hình đã khóa trên Train+Validation (27.677 mẫu) – fit lại toàn bộ preprocessing – rồi đánh giá Test (4.884 mẫu) đúng một lần cho mỗi mô hình.
