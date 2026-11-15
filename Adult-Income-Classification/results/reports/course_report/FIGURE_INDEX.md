# Danh mục hình

Tất cả các hình đều là hình **có sẵn** trong repository, do các script của dự án sinh ra. Không có hình nào được vẽ mới hoặc chỉnh sửa.

| Hình | Tiêu đề | File | Mục |
|---|---|---|---|
| 3.1 | Phân phối nhãn income | `results/figures/eda/01_income_distribution.png` | 3.2 |
| 3.2 | Phân phối tuổi theo lớp | `results/figures/eda/02_age.png` | 3.2 |
| 3.3 | Tỉ lệ >50K theo capital = 0 / ≠ 0 | `results/figures/eda/07_capital_zero_vs_nonzero.png` | 3.2 |
| 3.4 | Tương quan Pearson giữa thuộc tính số | `results/figures/eda/08_corr_numeric_pearson.png` | 3.2 |
| 3.5 | Cramér's V giữa thuộc tính phân loại | `results/figures/eda/09_cramers_v_categorical.png` | 3.2 |
| 3.6 | Mutual Information với nhãn | `results/figures/eda/10_mutual_information.png` | 3.2 |
| 3.7 | Phân phối nhãn trên Train/Val/Test | `results/figures/preprocessing/split_label_distribution.png` | 3.4 |
| 4.1 | F1 tốt nhất theo max_depth (DT) | `results/figures/validation/dt_f1_vs_depth.png` | 4.4 |
| 4.2 | Quá khớp của DT theo độ sâu | `results/figures/validation/dt_overfitting_default_stopping.png` | 4.4 |
| 4.3 | So sánh Validation và Test | `results/figures/final/validation_vs_test.png` | 4.6 |
| 4.4 | Đường ROC và PR trên Test | `results/figures/final/roc_pr_test.png` | 4.6 |
| 4.5a | Confusion Matrix của DT trên Test | `results/figures/final/cm_test_DecisionTree.png` | 4.7 |
| 4.5b | Confusion Matrix của CNB trên Test | `results/figures/final/cm_test_CategoricalNB.png` | 4.7 |
| 5.1 | Phân phối bootstrap của chênh lệch DT − CNB | `results/figures/final/bootstrap_differences.png` | 5.2 |

Lưu ý: nhãn "PR-AUC" trong Hình 4.3 và Hình 5.1 thực chất là **Average Precision**. Điều này đã được ghi rõ trong chú thích của hình trong báo cáo.

Các hình có sẵn nhưng không dùng trong báo cáo: `eda/03`–`06e`, `validation/cm_*`, `validation/cnb_f1_vs_alpha.png`, `validation/roc_pr_validation.png`, `validation/validation_comparison.png`, `final/cm_test_GaussianNB.png`.
