# FINAL SCIENTIFIC AUDIT: Decision Tree và Naive Bayes trên Adult Census Income

- **Ngày thực hiện:** 2026-11-15/16.
- **Môi trường:** máy local, chỉ dùng CPU. Windows 11, Python 3.12.10, scikit-learn 1.9.0, pandas 3.1.1, numpy 2.4.4.
- **Phạm vi:** chỉ đọc. Không train, không tuning, không gọi `fit` hay `predict` lại trên Test, không load file `.joblib`, không sửa hay xóa file cũ.
- **Kết luận: READY WITH CAVEATS.**

## 1. Phương pháp
1. Script kiểm toán chính là [scripts/final_audit.py](../../../scripts/final_audit.py). Script này chỉ đọc dữ liệu gốc, `split_indices.csv`, các dự đoán đã lưu, các file CSV/JSON/MD/log và mã nguồn. Metric được tính bằng các hàm của `sklearn.metrics` trên dự đoán đã lưu. Script băm sha256 của 17 artifact trước và sau khi audit để chứng minh không file nào bị thay đổi.
2. Bốn agent kiểm tra độc lập, mỗi agent phụ trách một mảng, chạy script riêng trong thư mục tạm của hệ thống và **không được cung cấp số liệu của tôi**. Kết quả của cả bốn agent khớp với script chính.
3. Mỗi nội dung được gán một trong ba trạng thái PASS, WARNING hoặc FAIL dựa trên bằng chứng. Chi tiết xem các file con và [audit_manifest.json](audit_manifest.json).

## 2. Tóm tắt kết quả
| Mảng kiểm tra | Trạng thái | Kết quả chính | Chi tiết |
|---|---|---|---|
| 1. Trùng lặp theo 14 đặc trưng | **PASS** + 1 WARNING | Có 24 tổ hợp đặc trưng bị trùng (49 dòng). **Không** tổ hợp nào xuất hiện ở hai tập khác nhau. Có 1 nhóm khác nhãn, nằm hoàn toàn trong Train | [duplicate_audit.csv](duplicate_audit.csv), mục 3 |
| 2. McNemar | **PASS** + 1 WARNING | b=264, c=339. Exact binomial cho p=0,0025538, khớp với giá trị đã báo cáo (0,002554) | [mcnemar_audit.md](mcnemar_audit.md) |
| 3. Bootstrap | **PASS** + 2 WARNING | Cài đặt đúng cả 8 điểm kiểm tra. Mọi CI tái lập chính xác (sai lệch ≤ 2e-18) | [bootstrap_audit.md](bootstrap_audit.md) |
| 4. Toàn vẹn | **PASS** | Split, dự đoán, metric, confusion matrix, checksum và artifact đều khớp | [integrity_check.md](integrity_check.md) |

Tổng cộng có 61 PASS, 4 WARNING và **0 FAIL**.

## 3. AUDIT 1: Trùng lặp theo 14 đặc trưng (không tính `income`)
Cách làm: đọc file `data/raw/adult.csv` dưới dạng chuỗi nguyên văn, không strip, không đổi '?' thành NaN. Khóa nhận diện gồm 14 cột: age, workclass, fnlwgt, education, education.num, marital.status, occupation, relationship, race, sex, capital.gain, capital.loss, hours.per.week, native.country.

| Thống kê | Giá trị |
|---|---:|
| Số tổ hợp đặc trưng bị trùng | **24** (23 nhóm 2 dòng, 1 nhóm 3 dòng; tổng 49 dòng) |
| A. Cùng đặc trưng, cùng nhãn | 23 nhóm |
| B. Cùng đặc trưng, khác nhãn | **1 nhóm** |
| Nhóm có mặt ở cả Train và Val | **0** |
| Nhóm có mặt ở cả Train và Test | **0** |
| Nhóm có mặt ở cả Val và Test | **0** |
| Nhóm khác nhãn nằm ở hai tập trở lên | **0** |
| Đối chiếu: số nhóm trùng khi tính cả 15 cột (cách chia gốc) | 23 |
| Kiểm tra độ nhạy: số nhóm sau khi strip khoảng trắng | 24 (không đổi) |

Các nhóm nằm hoàn toàn trong một tập: 17 nhóm trong Train, 3 trong Val, 4 trong Test.

**Nhóm khác nhãn:** dòng 4963 (<=50K) và dòng 24061 (>50K). Cả hai có cùng thông tin: 39 tuổi, Private, fnlwgt 138192, Bachelors/13, Married-civ-spouse, Craft-repair, Husband, White, Male, 0/0, 40 giờ, United-States. Cả hai dòng đều thuộc **Train**.

**WARNING:**
- Đây là nhiễu nhãn có sẵn trong dữ liệu, không phải rò rỉ dữ liệu.
- Cách chia gốc gom nhóm theo 15 cột (kể cả nhãn), nên **về thiết kế** cách chia này không đảm bảo các dòng cùng đặc trưng nhưng khác nhãn sẽ nằm chung một tập. Lần chia hiện tại không gặp vấn đề này là **do ngẫu nhiên**. Bằng chứng cho thấy kết quả thực nghiệm không bị ảnh hưởng.
- Báo cáo nên ghi nhận hạn chế này. Không cần chia lại dữ liệu vì giao giữa các tập đã được kiểm chứng là 0.

## 4. Các sai lệch phát hiện được
Không phát hiện sai lệch số liệu nào giữa các artifact. Các điểm cần lưu ý (WARNING):

| # | Vấn đề | Mức ảnh hưởng | Khuyến nghị |
|---|---|---|---|
| W1 | Có 1 cặp dòng cùng đặc trưng nhưng khác nhãn; cách gom nhóm theo 15 cột không bảo đảm chúng ở chung một tập | Không có (cả hai đều trong Train) | Nêu trong mục hạn chế của dữ liệu và phương pháp chia |
| W2 | McNemar là phân tích bổ sung, không định trước, và chỉ so sánh Accuracy | Về diễn giải | Ghi rõ là phân tích bổ sung; phản ánh khác biệt điểm làm việc tại ngưỡng 0,5 |
| W3 | Metric gọi là "PR-AUC" thực chất là Average Precision. Nếu tính bằng hình thang, DT đạt 0,7840 thay vì 0,7715 | Về tên gọi; số liệu vẫn đúng và nhất quán | Ghi là "PR-AUC (Average Precision)". **Không** đổi metric sau khi đã xem Test |
| W4 | CI bootstrap chỉ phản ánh độ bất định do mẫu Test | Về diễn giải | Nêu trong mục hạn chế |
| Lưu ý | Checksum đầy đủ ban đầu của dữ liệu gốc chưa được lưu thành file | Thấp (16 ký tự đầu đã khớp) | Lưu checksum đầy đủ vào tài liệu dự án |

## 5. Ảnh hưởng đến báo cáo môn học
- Không có vấn đề nào làm thay đổi kết luận chính: **CI 95% của chênh lệch DT − CNB đều chứa 0 ở cả F1, ROC-AUC và PR-AUC (AP)**. Nghĩa là chưa có đủ bằng chứng để nói mô hình nào tốt hơn theo tiêu chí đã định trước. Điều này không có nghĩa hai mô hình tương đương.
- Kết quả McNemar (Accuracy của CNB cao hơn, p=0,0026) là đúng về số liệu, nhưng phải trình bày như một phân tích bổ sung mô tả sự đánh đổi giữa FP và FN.

## 6. Khuyến nghị
1. Trong báo cáo cuối kỳ, dùng tên "PR-AUC (Average Precision)", ghi McNemar là "phân tích bổ sung", và thêm W1 cùng W4 vào mục hạn chế.
2. Lưu checksum đầy đủ của file `adult.csv` vào README hoặc tài liệu dự án.
3. Theo bằng chứng hiện có, **không cần train lại**, không cần chia lại dữ liệu và không cần đánh giá lại Test.

## 7. Kết luận
### READY WITH CAVEATS
Kết quả đủ tin cậy để đưa vào báo cáo cuối kỳ, với điều kiện trình bày minh bạch 4 WARNING ở trên. Không có FAIL nào.
