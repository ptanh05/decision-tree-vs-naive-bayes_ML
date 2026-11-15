# AUDIT 2: Kiểm định McNemar (DecisionTree so với CategoricalNB)

**Trạng thái: PASS, kèm 1 WARNING về cách diễn giải**

## Dữ liệu dùng
Kiểm toán chỉ dùng file `results/metrics/final_predictions.csv`, gồm các cột `y_true`, `DecisionTree_pred` và `CategoricalNB_pred`. Không mô hình nào được chạy lại.

## Bảng 2×2 (n = 4.884)
| | CNB đúng | CNB sai |
|---|---:|---:|
| **DT đúng** | 3.654 (cả hai đúng) | **b = 264** |
| **DT sai** | **c = 339** | 627 (cả hai sai) |

Tổng 4 ô: 3.654 + 264 + 339 + 627 = 4.884, đúng bằng số mẫu Test.

Accuracy tính lại từ bảng:
- DT: (3.654 + 264) / 4.884 = 0,80221
- CNB: (3.654 + 339) / 4.884 = 0,81757

Hai giá trị này khớp với `final_test_metrics.csv`.

So sánh lỗi: DT có ít FN hơn (160 so với 248), còn CNB có ít FP hơn (643 so với 806). Cả hai nhận định trong báo cáo đều đúng.

## Phương pháp và p-value tính lại
| Phương pháp | Thống kê | Hiệu chỉnh liên tục | p (hai phía) |
|---|---|---|---:|
| **Exact binomial** (phương pháp dùng trong báo cáo gốc) | min(b,c) = 264 ~ Binomial(603; 0,5) | không áp dụng | **0,0025538** |
| Chi-square tiệm cận | (b−c)²/(b+c) = 9,3284 | Không | 0,0022563 |
| Chi-square tiệm cận | (\|b−c\|−1)²/(b+c) = 9,0813 | Có (Edwards) | 0,0025824 |

- Code gốc ở [src/final_evaluation.py](../../../src/final_evaluation.py) dùng `scipy.stats.binomtest(min(b,c), b+c, 0.5).pvalue`, mặc định là kiểm định hai phía. Đây là **exact binomial McNemar**.
- Tôi tính lại bằng hai cách: dùng scipy, và cộng tay các hệ số tổ hợp theo công thức 2·Σ_{k≤264} C(603,k)/2^603. Cả hai đều cho 0,00255377. Giá trị này khớp với số đã báo cáo (`p=0.002554`) khi làm tròn đến 4 chữ số có nghĩa.
- Một agent xác minh độc lập, không được biết số liệu của tôi, cũng ra đúng các con số trên.
- Cả ba biến thể kiểm định đều cho p < 0,003, nên kết luận không phụ thuộc vào việc chọn phương pháp nào.

## Đối chiếu với kết quả gốc
Kết quả gốc **khớp** với tính toán độc lập: b = 264, c = 339, p = 0,002554, phương pháp exact binomial hai phía.

## WARNING: cách diễn giải
- McNemar **không** nằm trong tiêu chí so sánh đã định trước. Tiêu chí đã định trước là CI bootstrap cho F1, ROC-AUC và PR-AUC. McNemar được thêm vào sau, như một phân tích bổ sung; điều này được ghi rõ trong `paired_model_comparison.csv`.
- Kiểm định này chỉ so sánh **Accuracy tại ngưỡng 0,5**. Accuracy khác nhau chủ yếu là do DT dùng `class_weight=balanced`, nên DT dự đoán lớp >50K nhiều hơn. Đây là một sự đánh đổi giữa Precision và Recall, không có nghĩa là CNB tốt hơn về tổng thể.
- Cách viết đề xuất cho báo cáo: *"Phân tích bổ sung (không định trước): kiểm định McNemar chính xác cho thấy Accuracy của CNB cao hơn DT có ý nghĩa thống kê (b=264, c=339, p=0,0026). Khác biệt này phản ánh điểm làm việc khác nhau tại ngưỡng 0,5, không thay đổi kết luận chính về F1, ROC-AUC và PR-AUC."*
