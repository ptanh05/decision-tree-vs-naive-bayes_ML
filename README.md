# decision-tree-vs-naive-bayes_ML

Repository đồ án môn **Machine Learning**: nghiên cứu và so sánh **Decision Tree (CART)** và **Categorical Naive Bayes** trên bài toán phân loại mức thu nhập cá nhân (*Adult Census Income*, `32.561` mẫu, `>50K` vs `<=50K`).

> Toàn bộ source code, dữ liệu và kết quả thực nghiệm nằm trong thư mục **[`Adult-Income-Classification/`](Adult-Income-Classification/README.md)**.

## Tóm tắt nhanh
- **Phân loại nhị phân (mất cân bằng `75,92% / 24,08%`):** lấy **F1 lớp `>50K`** tại ngưỡng cố định `0,5` làm tiêu chí chọn mô hình.
- **Chống rò rỉ dữ liệu:** chia `70/15/15` (`22.793 / 4.884 / 4.884`) bằng `StratifiedGroupKFold` (`20` fold, `seed=42`) để giữ các bản ghi trùng trong cùng một tập; mọi bước tiền xử lý học tham số đặt trong `Pipeline`.
- **Kết quả Test (`4.884` mẫu):**
  - `DecisionTree`: Accuracy `0,8022` · Precision `0,5576` · Recall `0,8639` · **F1 `0,6778`** · ROC-AUC `0,8984` · AP `0,7715`
  - `CategoricalNB`: Accuracy `0,8176` · Precision `0,5907` · Recall `0,7891` · **F1 `0,6756`** · ROC-AUC `0,9011` · AP `0,7665`
  - Paired stratified bootstrap `5.000` lần: CI 95% của chênh lệch `DT − CNB` chứa `0` (F1 `[−0,0133; +0,0175]`) → chưa đủ bằng chứng kết luận mô hình nào tốt hơn; hai mô hình đánh đổi khác nhau giữa Recall (DT) và Precision/Accuracy (CNB).
- **Kiểm toán độc lập:** `scripts/final_audit.py` xác nhận `61 PASS / 4 WARNING / 0 FAIL` và `17/17` artifact đóng băng khớp SHA256.

## Tài liệu đọc tiếp
- Hướng dẫn dự án và thứ tự đọc code: [`Adult-Income-Classification/README.md`](Adult-Income-Classification/README.md)
- Lộ trình học chi tiết qua 12 commit + 30 câu hỏi bảo vệ: [`Adult-Income-Classification/results/reports/course_report/LEARNING_ROADMAP.md`](Adult-Income-Classification/results/reports/course_report/LEARNING_ROADMAP.md)
- Báo cáo cuối kỳ: [`Adult-Income-Classification/results/reports/course_report/BAO_CAO_CUOI_KY_ML.md`](Adult-Income-Classification/results/reports/course_report/BAO_CAO_CUOI_KY_ML.md)
