# AUDIT 3: Bootstrap paired stratified

**Trạng thái: PASS về cài đặt và về khả năng tái lập; có 2 WARNING về cách đặt tên metric và cách diễn giải**

## 1. Rà soát code ([src/final_evaluation.py](../../../src/final_evaluation.py))
| # | Nội dung kiểm tra | Bằng chứng trong code | Kết quả |
|---|---|---|---|
| 1 | Số lần lặp là 5.000 | `N_BOOT = 5000` | PASS |
| 2 | Seed là 42 | `BOOT_SEED = 42`, `np.random.default_rng(seed)` | PASS |
| 3 | Lấy mẫu phân tầng theo nhãn thật | `pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)`; mỗi lớp được lấy có hoàn lại với đúng số mẫu của lớp đó | PASS |
| 4 | Hai mô hình dùng chung chỉ số | mỗi lần lặp sinh **một** mảng `idx` rồi dùng nó cho mọi mô hình (`for n in models`) | PASS |
| 5 | Giữ được tính ghép cặp | chênh lệch được tính theo từng lần lặp trên cùng một mẫu: `stats["DecisionTree"][k] - stats["CategoricalNB"][k]` | PASS |
| 6 | CI theo phương pháp percentile | `np.percentile(a, [2.5, 97.5])` (không dùng BCa) | PASS |
| 7 | Chiều của chênh lệch | DT − CNB | PASS |
| 8a | F1 tính cho lớp >50K | `f1_score(yb, prb, pos_label=1)`, nhãn 1 tương ứng với ">50K" (hàm `clean` trong data_loader) | PASS |
| 8b | ROC-AUC tính trên xác suất | `roc_auc_score(yb, scb)`, trong đó `scb` = cột `*_proba_gt50K` | PASS |
| 8c | PR-AUC | `average_precision_score(yb, scb, pos_label=1)`, tức là **Average Precision** | WARNING (xem mục 3) |

Thêm hai điểm:
- Nhãn `y` dùng trong bootstrap được đọc lại từ file `final_predictions.csv` đã lưu.
- `bootstrap_std` dùng `ddof=1`.

## 2. Tái tính độc lập từ dự đoán đã lưu
Tôi cài đặt lại thuật toán sinh chỉ số trong script [scripts/final_audit.py](../../../scripts/final_audit.py). Script này chỉ đọc file `final_predictions.csv`, không chạy mô hình nào. Thứ tự gọi RNG giữ như code gốc: lấy mẫu lớp âm trước, lớp dương sau.

- Mọi lần lặp đều giữ đúng 3.708 mẫu âm và 1.176 mẫu dương.
- CI của từng mô hình khớp `bootstrap_confidence_intervals.csv` với sai lệch lớn nhất = 0.
- CI của chênh lệch khớp `paired_model_comparison.csv` với sai lệch lớn nhất = 1,7e-18, chỉ là sai số dấu phẩy động.
- Một agent xác minh độc lập, viết script riêng, cũng tái lập được tới hơn 10 chữ số thập phân.

| Chênh lệch DT − CNB | Giá trị điểm | CI 95% (tính lại) | CI 95% (CSV và báo cáo) | CI chứa 0 | Tỉ lệ lần lặp có Δ>0 |
|---|---:|---|---|---|---:|
| F1 | +0,0021 | [−0,0133; +0,0175] | [−0,0133; +0,0175] | Có | 60,2% |
| ROC-AUC | −0,0027 | [−0,0102; +0,0046] | [−0,0102; +0,0046] | Có | 23,3% |
| PR-AUC (AP) | +0,0050 | [−0,0093; +0,0192] | [−0,0093; +0,0192] | Có | 74,7% |

## 3. WARNING: "PR-AUC" thực chất là Average Precision
Code dùng `average_precision_score`, tức là tổng theo bậc thang Σ(Rₙ−Rₙ₋₁)·Pₙ. Đây **không** phải diện tích tính bằng tích phân hình thang dưới đường Precision-Recall. Hai cách tính cho kết quả như sau:

| Mô hình | AP (giá trị đã báo cáo) | PR-AUC hình thang `auc(recall, precision)` (chỉ để tham khảo) | Số ngưỡng phân biệt |
|---|---:|---:|---:|
| DecisionTree | 0,7715 | 0,7840 | 161 |
| CategoricalNB | 0,7665 | 0,7664 | 4.843 |

- Với DT, hai cách tính lệch nhau 0,0125. Lý do: cây quyết định chỉ cho ra rất ít giá trị xác suất khác nhau (mỗi lá một giá trị), nên khi nội suy tuyến tính giữa các điểm thưa, phép hình thang cho kết quả lạc quan quá mức. Đây là lý do sklearn khuyến nghị dùng AP thay vì hình thang.
- Code nhất quán: dùng AP ở mọi chỗ. Vì vậy **không có lỗi tính toán**. Vấn đề chỉ nằm ở **tên gọi**.
- Đề xuất: trong báo cáo môn học, ghi metric này là "PR-AUC (Average Precision)". Không nên đổi sang cách hình thang vì đó là thay đổi metric sau khi đã xem Test.

## 4. WARNING: phạm vi suy diễn
- CI chỉ phản ánh độ bất định do **mẫu Test**: mô hình được giữ cố định và không fit lại. CI không bao gồm biến động do dữ liệu huấn luyện, do cách chia dữ liệu hay do việc chọn siêu tham số.
- Cách diễn giải đúng: CI của chênh lệch chứa 0 nghĩa là **chưa có đủ bằng chứng thống kê ở mức 95%** để nói hai mô hình khác nhau về F1, ROC-AUC hay PR-AUC trên tập Test này. Điều đó **không** chứng minh hai mô hình tương đương, và cũng không cho phép kết luận mô hình nào tốt hơn.
