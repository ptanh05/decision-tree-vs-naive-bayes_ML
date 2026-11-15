# PHASE 5–6 – Huấn luyện cuối và đánh giá Test

- Lệnh: `python src/final_evaluation.py`, chạy local trên CPU, 1 tiến trình. Môi trường: Windows 11 (10.0.26300), Python 3.12.10, scikit-learn 1.9.0, pandas 3.1.1, numpy 2.4.4.
- Lần chạy 1 (`20261115_234406`) dừng do lỗi `AttributeError` trong bước kiểm tra encoder. Lỗi xảy ra **trước** khi dự đoán Test: chưa có dự đoán nào trên Test và chưa ghi file kết quả nào. Sau khi sửa đúng dòng kiểm tra đó, lần chạy 2 (`20261115_234444`) hoàn tất. Log của cả hai lần nằm trong `results/logs/`.
- Cấu hình đã khóa từ PHASE 4, không tuning lại:
  - DT: entropy, max_depth=12, min_samples_leaf=10, min_samples_split=2, class_weight=balanced.
  - CNB: n_bins=20, alpha=0.1, fit_prior=True.
  - GNB (thí nghiệm phụ): var_smoothing=1e-7.
- Preprocessing và mô hình fit trên Train+Val (27.677 mẫu). Đánh giá trên Test (4.884 mẫu, 1.176 mẫu >50K) với ngưỡng 0,5. Mỗi mô hình chỉ dự đoán Test một lần.

## 1. Kết quả Test
| | DecisionTree | CategoricalNB | GaussianNB (phụ) |
|---|---|---|---|
| Accuracy | 0,8022 | 0,8176 | 0,8006 |
| Precision (>50K) | 0,5576 | 0,5907 | 0,6741 |
| Recall (>50K) | 0,8639 | 0,7891 | 0,3325 |
| **F1 (>50K)** | **0,6778** | **0,6756** | 0,4453 |
| ROC-AUC | 0,8984 | 0,9011 | 0,7840 |
| PR-AUC | 0,7715 | 0,7665 | 0,6013 |
| TN / FP / FN / TP | 2902 / 806 / 160 / 1016 | 3065 / 643 / 248 / 928 | 3519 / 189 / 785 / 391 |
| Fit trên Train+Val | 0,250 s | 0,134 s | 0,006 s |
| Predict trên Test | 0,017 s | 0,014 s | 0,001 s |

Thời gian là số đo của một lần chạy duy nhất, nên chỉ mang tính tham khảo. Bảng tham chiếu: mô hình luôn đoán <=50K có Accuracy 0,7592.

## 2. Khoảng tin cậy 95%
Phương pháp: paired stratified bootstrap, 5000 lần lặp, seed 42, CI theo percentile.

| Mô hình | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| DecisionTree | 0,678 [0,663; 0,694] | 0,898 [0,888; 0,909] | 0,772 [0,751; 0,791] |
| CategoricalNB | 0,676 [0,659; 0,693] | 0,901 [0,892; 0,910] | 0,767 [0,747; 0,786] |
| GaussianNB (phụ) | 0,445 [0,415; 0,474] | 0,784 [0,769; 0,799] | 0,601 [0,577; 0,625] |

## 3. So sánh ghép cặp DT − CNB
| Metric | Chênh lệch | CI 95% | CI chứa 0? |
|---|---|---|---|
| F1 | +0,0021 | [−0,0133; +0,0175] | Có |
| ROC-AUC | −0,0027 | [−0,0102; +0,0046] | Có |
| PR-AUC | +0,0050 | [−0,0093; +0,0192] | Có |

Phân tích bổ sung, ngoài tiêu chí đã thống nhất: kiểm định McNemar exact trên Accuracy. Có 264 mẫu DT đúng còn CNB sai, và 339 mẫu DT sai còn CNB đúng; p = 0,0026. Như vậy CNB có Accuracy cao hơn với ý nghĩa thống kê (0,8176 so với 0,8022). Nguyên nhân chủ yếu là DT dùng class_weight=balanced nên dự đoán >50K nhiều hơn (FP 806 so với 643). Accuracy không phải tiêu chí chọn mô hình, nên kết quả này chỉ dùng để mô tả sự đánh đổi. Hai mô hình đưa ra cùng dự đoán trên 87,65% mẫu Test.

## 4. Validation và Test, tổng quát hóa
| | Val F1 | Test F1 | Val ROC | Test ROC | Val PR | Test PR |
|---|---|---|---|---|---|---|
| DT | 0,6796 | 0,6778 | 0,8927 | 0,8984 | 0,7623 | 0,7715 |
| CNB | 0,6733 | 0,6756 | 0,9004 | 0,9011 | 0,7584 | 0,7665 |

- Mọi chênh lệch giữa Val và Test đều nhỏ hơn 0,01, và giá trị Validation nằm trong CI bootstrap của Test. Không thấy dấu hiệu chọn siêu tham số bị overfit vào Validation.
- Overfitting: F1 của DT trên Train+Val là 0,707 so với 0,678 trên Test (cây final có 365 lá). Với CNB, hai giá trị là 0,678 và 0,676. Khoảng cách của DT lớn hơn một chút nhưng vẫn nhỏ, nhờ max_depth=12 và min_samples_leaf=10. Để so sánh, ở PHASE 4 cây không giới hạn đạt Train F1 ≈ 1,0 nhưng Val F1 chỉ ≈ 0,60.
- Mất cân bằng lớp (24% là >50K): ở ngưỡng 0,5, DT với class_weight=balanced thiên về Recall (0,864), còn CNB với prior theo tần suất cân bằng hơn (Recall 0,789, Precision 0,591). F1 của hai mô hình gần như bằng nhau, nhưng chúng nằm ở hai điểm làm việc khác nhau. Lựa chọn giữa hai mô hình phụ thuộc vào chi phí của FP so với FN, và phần này nằm ngoài phạm vi đánh giá.
- GNB chỉ dùng 6 thuộc tính số và kém rõ rệt so với hai mô hình chính, đúng như dự đoán từ EDA.

## 5. Kết luận khoa học (giới hạn)
- Trên Test, CI 95% của chênh lệch DT − CNB chứa 0 ở cả F1, ROC-AUC và PR-AUC. **Không có đủ bằng chứng để kết luận mô hình nào tốt hơn** theo tiêu chí đã định trước. Kết quả này chỉ cho thấy hai mô hình tương đương trong phạm vi độ chính xác mà một tập Test 4.884 mẫu có thể phân biệt; nó không chứng minh hai mô hình tương đương tuyệt đối.
- Hai mô hình khác nhau rõ ở sự đánh đổi Precision/Recall tại ngưỡng 0,5, và vì vậy khác nhau ở Accuracy (McNemar p = 0,0026).
- Hạn chế:
  - Chỉ có một lần chia dữ liệu và một tập Test.
  - Dữ liệu là điều tra dân số Mỹ năm 1994, không đại diện cho thị trường lao động hiện nay hay Việt Nam.
  - Dữ liệu có thuộc tính nhạy cảm (sex, race), và relationship tương quan mạnh với sex.
  - Giá trị '?' mang tính hệ thống.
  - Không được dùng mô hình này để tuyển dụng hoặc đánh giá năng lực cá nhân.

## 6. Kiểm tra (mục E) – tất cả đạt
- Split khớp `split_indices.csv`; Train+Val 27.677 / Test 4.884 / 1.176 mẫu >50K; không có duplicate group nào dùng chung giữa Test và Train+Val.
- Hạng mục của encoder (CNB ordinal, DT one-hot) và biên các khoảng KBins khớp với một lần fit mới trên Train+Val. Không có thành phần nào fit trên Test.
- Không tuning lại và không chọn ngưỡng. Mỗi mô hình gọi dự đoán Test đúng một lần, và nhãn Test chỉ được đọc sau khi dự đoán xong. Script sẽ dừng nếu `final_predictions.csv` đã tồn tại.
- Metric được tính lại từ `final_predictions.csv` và khớp với TN/FP/FN/TP (Precision, Recall, F1, Accuracy).
- Final model: fit lại hai lần và load lại từ joblib đều cho xác suất trùng khớp từng bit trên Train+Val.
- sha256 của `data/raw/adult.csv` không đổi. Kết quả PHASE 4 không bị ghi đè.

## File đầu ra
- `src/final_evaluation.py`
- `results/models/final/{DecisionTree,CategoricalNB,GaussianNB}_final.joblib`
- `results/metrics/`: `final_test_metrics.csv`, `final_predictions.csv`, `bootstrap_confidence_intervals.csv`, `paired_model_comparison.csv`, `validation_vs_test.csv`, `final_run_info.json`
- `results/figures/final/`: `cm_test_*.png` (3 file), `roc_pr_test.png`, `validation_vs_test.png`, `bootstrap_differences.png`
- `results/logs/phase5_6_final_20261115_234406.log` (lần chạy lỗi, trước Test), `phase5_6_final_20261115_234444.log`
