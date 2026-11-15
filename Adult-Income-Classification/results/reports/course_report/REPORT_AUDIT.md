# Kiểm toán số liệu trong báo cáo cuối kỳ

Mọi số liệu trong `BAO_CAO_CUOI_KY_ML.md` đã được đối chiếu với các file kết quả thực tế. Việc đối chiếu chỉ đọc file: không fit, không predict, không sửa artifact nào.

| Nội dung | File nguồn | Trạng thái xác minh | Ghi chú |
|---|---|---|---|
| 32.561 dòng; 24.720 / 7.841 (75,92% / 24,08%) | `results/eda/label_distribution.csv` | Khớp | |
| Số giá trị '?' 1.836 / 1.843 / 583; 2.399 dòng có '?'; tỉ lệ >50K 13,9% so với 24,9% | `results/eda/columns_overview.csv`, `missing_vs_income.csv` | Khớp | |
| capital.gain: 91,67% bằng 0, skew 11,95; capital.loss: 95,33% bằng 0 | `results/eda/numeric_summary.csv` | Khớp | |
| 159 dòng có capital.gain = 99999, tất cả là >50K | `data/raw/adult.csv` (truy vấn chỉ đọc) | Khớp | |
| Giá trị trung bình theo lớp: age, education.num, hours | `results/eda/numeric_by_class.csv` | Khớp | |
| Tỉ lệ >50K: Married-civ-spouse 44,7%; Never-married 4,6%; Self-emp-inc 55,7%; Private 21,9% | `results/eda/category_income_rates.csv` | Khớp | |
| Pearson giữa các thuộc tính số: \|r\| ≤ 0,15 (lớn nhất 0,148) | `results/eda/corr_pearson.csv` | Khớp | |
| Cramér's V: relationship–sex 0,649; marital–relationship 0,488 | `results/eda/cramers_v.csv` | Khớp | |
| Mutual Information: 0,1146 / 0,1085 / 0,0833; race 0,0058; native.country 0,0060 | `results/eda/mutual_info.csv` | Khớp | |
| education và education.num ánh xạ 1-1 | `src/check_design.py` | Khớp | |
| Kích thước Train/Val/Test 22.793 / 4.884 / 4.884; tỉ lệ >50K 0,2408 | `data/processed/split_summary.csv` | Khớp | |
| 23 nhóm trùng theo 15 cột | `results/reports/final_audit/FINAL_AUDIT_REPORT.md` | Khớp | |
| Số đặc trưng: 92 (DT), 13 (CNB) | `src/preprocessing.py`, log PHASE 4 | Khớp | |
| Lưới tuning 800 / 126 / 7 cấu hình; mở rộng n_bins | `results/metrics/*_validation_grid.csv`, `results/metrics/archive/` | Khớp | |
| F1 theo depth / criterion / class_weight / n_bins / alpha / fit_prior | `results/metrics/dt_validation_grid.csv`, `cnb_validation_grid.csv` | Khớp | Groupby theo các cột `param_*` |
| Cây không giới hạn: train F1 0,9999, val khoảng 0,60 | `results/reports/phase4_summary.md`, Hình 4.2 | Khớp | |
| Cấu hình được chọn | `results/metrics/best_configs.json` | Khớp | |
| Bảng 4.2 (Validation, CM, baseline) | `results/metrics/validation_comparison.csv` | Khớp | |
| Bảng 4.3 (Test, CM, thời gian) | `results/metrics/final_test_metrics.csv` | Khớp | Đã tính lại từ `final_predictions.csv` (xem `integrity_check.md`) |
| Cây cuối có 365 lá; F1 trên Train+Val 0,707 / 0,678 | `results/metrics/final_run_info.json` | Khớp | |
| DT có 161 ngưỡng; AP so với PR-AUC hình thang (0,7840 / 0,7664) | `results/reports/final_audit/bootstrap_audit.md` | Khớp | |
| CI bootstrap của từng mô hình | `results/metrics/bootstrap_confidence_intervals.csv` | Khớp | |
| Chênh lệch DT − CNB và CI tương ứng | `results/metrics/paired_model_comparison.csv` | Khớp | Tái lập được với sai lệch ≤ 2e-18 |
| McNemar 3654 / 264 / 339 / 627; p = 0,0025538 | `results/reports/final_audit/mcnemar_audit.md` | Khớp | |
| Trùng theo 14 đặc trưng: 24 nhóm / 49 dòng / 0 giao giữa các tập; dòng 4963 và 24061 | `results/reports/final_audit/duplicate_audit.csv`, `FINAL_AUDIT_REPORT.md` | Khớp | |
| Kết quả audit: 61 PASS / 4 WARNING / 0 FAIL | `results/reports/final_audit/FINAL_AUDIT_REPORT.md` | Khớp | |
| sha256 của dữ liệu gốc | `results/reports/final_audit/integrity_check.md` | Khớp | Băm lại sau khi viết báo cáo (xem bên dưới) |
| Môi trường chạy (Windows 11, Python 3.12.10, sklearn 1.9.0, pandas 3.1.1, numpy 2.4.4) | `results/metrics/final_run_info.json` | Khớp | |
| Thông tin trang bìa (trường, khoa, sinh viên, giảng viên) | Không có trong repository | [CẦN BỔ SUNG] | |
| Nguồn tải và ngày tải `adult.csv` | Không có trong repository | [CẦN BỔ SUNG] | |
| DOI của UCI Adult; số mục trong tài liệu sklearn | `REFERENCES.md` | [CẦN XÁC MINH NGUỒN] | Phiên làm việc không dùng Internet |

## Kiểm tra tính bất biến của artifact
Sau khi tạo báo cáo, sha256 của các artifact đã đóng băng được so với `frozen_artifact_sha256` trong `results/reports/final_audit/audit_manifest.json`. Script `build_docx.py` ghi kết quả so sánh vào cuối file này.

## Kết quả hash
**17/17 artifact khớp sha256 với audit_manifest.json.**

- `data/raw/adult.csv`: khớp
- `data/processed/split_indices.csv`: khớp
- `results/metrics/final_predictions.csv`: khớp
- `results/metrics/final_test_metrics.csv`: khớp
- `results/metrics/bootstrap_confidence_intervals.csv`: khớp
- `results/metrics/paired_model_comparison.csv`: khớp
- `results/metrics/final_run_info.json`: khớp
- `results/metrics/validation_vs_test.csv`: khớp
- `results/metrics/best_configs.json`: khớp
- `results/metrics/validation_comparison.csv`: khớp
- `results/reports/final_evaluation.md`: khớp
- `results/models/CategoricalNB_best_trainonly.joblib`: khớp
- `results/models/DecisionTree_best_trainonly.joblib`: khớp
- `results/models/final/CategoricalNB_final.joblib`: khớp
- `results/models/final/DecisionTree_final.joblib`: khớp
- `results/models/final/GaussianNB_final.joblib`: khớp
- `results/models/GaussianNB_best_trainonly.joblib`: khớp
