# AUDIT 4: Kiểm tra toàn vẹn cuối cùng

**Trạng thái: PASS** (có 1 lưu ý nhỏ về checksum)

## 1. Toàn vẹn split (`data/processed/split_indices.csv`)
| Nội dung kiểm tra | Kết quả | Trạng thái |
|---|---|---|
| Kích thước các tập | Train 22.793, Val 4.884, Test 4.884, tổng 32.561 | PASS |
| Chồng lấn chỉ số | `row_index` không trùng, mỗi dòng chỉ thuộc 1 tập | PASS |
| Nhóm trùng bị tách giữa các tập (theo 15 cột, đúng phương pháp chia gốc) | giao giữa train&val, train&test, val&test đều = 0; mỗi `dup_group` nằm trọn trong 1 tập | PASS |
| Tỷ lệ lớp >50K | 0,2408 ở cả ba tập | PASS |
| Định nghĩa split trong code | `StratifiedGroupKFold`, `N_FOLDS=20`, `RANDOM_STATE=42` trong `src/data_loader.py` | PASS |

## 2. Toàn vẹn predictions (`results/metrics/final_predictions.csv`)
| Nội dung kiểm tra | Kết quả | Trạng thái |
|---|---|---|
| Số dòng | 4.884 | PASS |
| `row_index` trùng với tập Test đã khóa | trùng chính xác | PASS |
| `y_true` khớp nhãn gốc tại cùng `row_index` | 4.884/4.884 dòng khớp; có 1.176 mẫu >50K | PASS |
| Cột pred bằng (proba ≥ 0,5) | đúng với cả 3 mô hình | PASS |

## 3. Nhất quán giữa metric, CSV và báo cáo
Tôi tính lại các metric từ dự đoán đã lưu và so với `final_test_metrics.csv`. Sai lệch lớn nhất là 0.

| Metric | DT (tính lại) | CNB (tính lại) |
|---|---:|---:|
| Accuracy | 0,8022 | 0,8176 |
| Precision | 0,5576 | 0,5907 |
| Recall | 0,8639 | 0,7891 |
| F1 | 0,6778 | 0,6756 |
| ROC-AUC | 0,8984 | 0,9011 |
| PR-AUC (AP) | 0,7715 | 0,7665 |
| TN/FP/FN/TP | 2902/806/160/1016 | 3065/643/248/928 |

- Tất cả khớp với bảng tham chiếu và với `results/reports/final_evaluation.md`. Tôi đã kiểm tra bằng cách tìm chuỗi số tương ứng trong file .md.
- F1 tính từ confusion matrix (2TP/(2TP+FP+FN)) khớp với `f1_score`.
- GaussianNB (thí nghiệm phụ) cũng khớp: 3519/189/785/391.

## 4. Checksum dữ liệu và artifact
| Nội dung kiểm tra | Kết quả | Trạng thái |
|---|---|---|
| `data/raw/adult.csv` | sha256 = `250e154ed75714ae57a564926d66c6319cd6aac1bcd32774cc76841a88d74e53`; 16 ký tự đầu trùng với giá trị đã ghi nhận ở các phase trước (`250e154ed75714ae`) | PASS (có lưu ý) |
| Model final (3 file `.joblib`) | đều tồn tại; 16 ký tự đầu của sha256 khớp với `final_run_info.json` (DT `bba7da207acd0545`, CNB `6c8122c2bcd83a4b`, GNB `20cfd974c0d4b2d0`). Chỉ băm nội dung file, **không load** | PASS |
| Model PHASE 4 (`*_best_trainonly.joblib`) | còn đủ 3 file | PASS |
| Cấu hình final | trùng với `best_configs.json` của PHASE 4 | PASS |
| Metadata | threshold 0,5; bootstrap {n 5000, seed 42, stratified, paired} | PASS |
| Log lần chạy hoàn tất (`phase5_6_final_20261115_234444.log`) | khớp số liệu (TP=1016, p=0.002554) | PASS |
| Log lần chạy lỗi (`phase5_6_final_20261115_234406.log`) | dừng ngay sau khi fit GaussianNB; không có dòng dự đoán hay metric Test nào | PASS |
| Tính read-only của chính đợt audit | sha256 của 17 file artifact trước và sau audit giống hệt nhau | PASS |

**Lưu ý:** checksum **đầy đủ** của file dữ liệu gốc chưa từng được lưu thành file trong repo. Giá trị dùng để đối chiếu chỉ là 16 ký tự đầu, ghi trong log làm việc. Mức trùng 64 bit như vậy đủ để khẳng định file không bị thay đổi. Dù vậy, nên lưu checksum đầy đủ ở trên vào tài liệu dự án để dùng cho các lần kiểm tra sau.

## 5. Rà soát luồng dữ liệu Test trong code
Phần này do agent độc lập thực hiện và tôi đã đối chiếu lại.
- Mọi lệnh `fit` đều chỉ dùng Train+Val (`Xtv`, `ytv`).
- Mỗi mô hình có đúng 1 lời gọi `positive_proba(m, Xte)`.
- Nhãn Test chỉ được đọc **sau khi** đã ghi xong dự đoán.
- Các bước kiểm tra tái lập và load lại mô hình chỉ dùng Train+Val.
- Script tự dừng nếu `final_predictions.csv` đã tồn tại.
- `src/tuning.py` loại tập Test ngay sau khi chia dữ liệu.
