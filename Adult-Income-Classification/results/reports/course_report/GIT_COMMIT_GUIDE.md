# Hướng dẫn học lại dự án qua Git History

Dự án: *Nghiên cứu và so sánh Decision Tree và Naive Bayes trong bài toán dự đoán mức thu nhập (Adult Census Income).*

Lịch sử Git được sắp xếp theo đúng trình tự của một dự án Machine Learning. Đọc từ commit cũ nhất đến commit mới nhất là đi lại toàn bộ quy trình, từ dữ liệu đến báo cáo.

```bash
git log --oneline --reverse          # xem danh sách commit từ cũ đến mới
git show --stat <commit_hash>        # xem commit thêm những file nào
git show <commit_hash> -- <file>     # xem nội dung một file trong commit
git checkout <commit_hash>           # mở lại trạng thái dự án tại commit đó
git checkout main                    # quay về nhánh chính
```

> **Lưu ý:** `git checkout <commit_hash>` đưa repo vào trạng thái **detached HEAD**. Trạng thái này chỉ nên dùng để xem và học, không nên sửa code hay commit ở đó. Nếu muốn thử nghiệm, hãy tạo nhánh riêng: `git switch -c thu-nghiem <commit_hash>`.

## Danh sách commit

| # | Hash | Message | Commit trước → sau |
|---|---|---|---|
| 0 | `4f56da8` | Initial commit (có sẵn, chỉ có README gốc) | – → `454322b` |
| 1 | `454322b` | chore: initialize machine learning project | `4f56da8` → `1468f55` |
| 2 | `1468f55` | data: add adult census income dataset pipeline | `454322b` → `ce98eb3` |
| 3 | `ce98eb3` | feat: perform exploratory data analysis | `1468f55` → `f254108` |
| 4 | `f254108` | feat: implement machine learning preprocessing pipelines | `ce98eb3` → `b9550a4` |
| 5 | `b9550a4` | feat: implement decision tree classifier | `f254108` → `1acb45f` |
| 6 | `1acb45f` | feat: implement categorical naive bayes classifier | `b9550a4` → `ece6056` |
| 7 | `ece6056` | feat: add model evaluation utilities | `1acb45f` → `6d7d9c1` |
| 8 | `6d7d9c1` | feat: add hyperparameter tuning experiments | `ece6056` → `46c8f16` |
| 9 | `46c8f16` | feat: add reproducible final model evaluation | `6d7d9c1` → `ebd9b6c` |
| 10 | `ebd9b6c` | test: add final experiment integrity audit | `46c8f16` → `9ee0847` |
| 11 | `9ee0847` | docs: add final machine learning course report | `ebd9b6c` → commit chứa file này |
| 12 | (commit chứa chính file này) | docs: add git commit learning guide | `9ee0847` → – |

Đường dẫn file bên dưới tính từ thư mục `Adult-Income-Classification/`.

## Chi tiết từng giai đoạn

### 1. `454322b`: Khởi tạo dự án
- **File:** `requirements.txt`, `.gitignore`.
- **Mục đích:** cố định phiên bản thư viện để có thể tái lập kết quả (Python 3.12, chỉ dùng CPU), và bỏ qua file cache khi commit.
- **Kiến thức cần học:** môi trường ảo, quản lý phiên bản thư viện, tính tái lập (reproducibility).
- **Xem lại:** `git checkout 454322b`

### 2. `1468f55`: Dữ liệu
- **File:** `data/raw/adult.csv`, `src/data_loader.py`.
- **Mục đích:** đưa dữ liệu gốc vào repo (sha256 `250e154e…`) và viết hàm đọc dữ liệu. Hàm này xử lý schema, giá trị `?`, mã hóa nhãn (`>50K` = 1) và định nghĩa cách chia tập.
- **Kiến thức cần học:** học có giám sát, phân loại nhị phân, kiểm tra chất lượng dữ liệu, giá trị thiếu ẩn (ghi bằng `?`).
- **Xem lại:** `git checkout 1468f55`

### 3. `ce98eb3`: Khám phá dữ liệu (EDA)
- **File:** `src/eda.py`, `notebooks/01_EDA.ipynb`, `scripts/make_eda_notebook.py`, `scripts/run_notebook.py`, `results/eda/*.csv`, `results/figures/eda/*.png`.
- **Mục đích:** hiểu dữ liệu trước khi xây mô hình, gồm: mất cân bằng lớp (24% `>50K`), phân phối lệch của các cột capital, quan hệ giữa các thuộc tính.
- **Kiến thức cần học:** thống kê mô tả, tương quan Pearson, Cramér's V, Mutual Information.
- **Xem lại:** `git checkout ce98eb3`

### 4. `f254108`: Tiền xử lý và chia dữ liệu
- **File:** `src/preprocessing.py`, `src/check_design.py`, `notebooks/02_Preprocessing.ipynb`, `scripts/make_preprocessing_notebook.py`, `data/processed/*`, `results/figures/preprocessing/*`.
- **Mục đích:** xây pipeline riêng cho từng mô hình. Decision Tree dùng One-Hot. CategoricalNB dùng rời rạc hóa theo phân vị, xử lý riêng giá trị 0 của capital, và Ordinal encoding. Dữ liệu được chia 70/15/15 bằng `StratifiedGroupKFold` (seed 42) để các bản ghi trùng nhau không bị tách sang các tập khác nhau.
- **Kiến thức cần học:** data leakage, `Pipeline`/`ColumnTransformer`, encoding, discretization, vai trò của các tập Train/Validation/Test.
- **Xem lại:** `git checkout f254108`

### 5. `b9550a4`: Decision Tree
- **File:** `src/decision_tree.py`.
- **Mục đích:** hàm dựng pipeline Decision Tree và lưới siêu tham số gồm 800 cấu hình.
- **Kiến thức cần học:** entropy, Information Gain, Gini, `max_depth`, `min_samples_leaf`, `class_weight`, overfitting.

### 6. `1acb45f`: Naive Bayes
- **File:** `src/naive_bayes.py`.
- **Mục đích:** pipeline CategoricalNB (mô hình chính) và GaussianNB (thí nghiệm bổ sung).
- **Kiến thức cần học:** định lý Bayes, giả định độc lập có điều kiện, prior và posterior, làm trơn Laplace (`alpha`), `fit_prior`.

### 7. `ece6056`: Hàm đánh giá
- **File:** `src/evaluation.py`.
- **Mục đích:** các metric dùng chung cho lớp `>50K`.
- **Kiến thức cần học:** Confusion Matrix, Precision, Recall, F1, ROC-AUC, Average Precision. Cần phân biệt AP với diện tích PR tính bằng phương pháp hình thang.

### 8. `6d7d9c1`: Tuning siêu tham số
- **File:** `src/tuning.py`, `results/metrics/*_validation_grid*.csv`, `results/metrics/archive/`, `best_configs.json`, `validation_comparison.csv`, `results/models/*_best_trainonly.joblib`, `results/figures/validation/*`, `results/logs/phase4_*`, `results/reports/phase4_summary.md`.
- **Mục đích:** grid search chỉ trên tập Validation, tiêu chí là F1 của lớp `>50K` với ngưỡng 0,5. Lưới n_bins của CNB được mở rộng; lưới cũ được lưu trong `archive/`.
- **Kiến thức cần học:** chọn mô hình, đánh đổi bias–variance, vì sao không được tuning trên Test.
- **Xem lại:** `git checkout 6d7d9c1`

### 9. `46c8f16`: Huấn luyện cuối và đánh giá Test
- **File:** `src/final_evaluation.py`, `results/models/final/*.joblib`, `results/metrics/final_*`, `bootstrap_confidence_intervals.csv`, `paired_model_comparison.csv`, `validation_vs_test.csv`, `results/figures/final/*`, `results/logs/phase5_6_*`, `results/reports/final_evaluation.md`.
- **Mục đích:** fit lại mô hình trên Train+Validation, dự đoán Test **một lần**, tính khoảng tin cậy bằng paired stratified bootstrap (5.000 lần lặp) và chạy kiểm định McNemar. Log của lần chạy lỗi đầu tiên (lỗi xảy ra trước khi chạm vào Test) được giữ lại để minh bạch.
- **Kiến thức cần học:** đánh giá trên dữ liệu chưa thấy, bootstrap confidence interval, so sánh ghép cặp, kiểm định McNemar.
- **Xem lại:** `git checkout 46c8f16`

### 10. `ebd9b6c`: Kiểm toán khoa học
- **File:** `scripts/final_audit.py`, `results/reports/final_audit/*`.
- **Mục đích:** kiểm toán ở chế độ chỉ đọc: kiểm tra trùng lặp theo 14 đặc trưng, tái tính McNemar và bootstrap, kiểm tra tính toàn vẹn của split, predictions và metric, kiểm tra sha256 của 17 artifact. Kết quả: 61 PASS, 4 WARNING, 0 FAIL.
- **Kiến thức cần học:** tính tái lập, kiểm tra rò rỉ dữ liệu, diễn giải đúng kết quả thống kê.

### 11. `9ee0847`: Báo cáo cuối kỳ
- **File:** `results/reports/course_report/` (`BAO_CAO_CUOI_KY_ML.md`/`.docx`, `REFERENCES.md`, `FIGURE_INDEX.md`, `REPORT_AUDIT.md`, `build_docx.py`).
- **Mục đích:** báo cáo học thuật. Mọi số liệu trong báo cáo đều truy được về file kết quả. Các placeholder `[CẦN BỔ SUNG]` và `[CẦN XÁC MINH NGUỒN]` vẫn được giữ nguyên.
- **Kiến thức cần học:** cách trình bày và diễn giải kết quả ML một cách trung thực.

## Ghi chú về tính trung thực của lịch sử
- Toàn bộ code và kết quả đã có sẵn trước khi tạo các commit này. Lịch sử Git **phân nhóm file theo giai đoạn**, chứ không ghi lại thời điểm thực tế từng file được viết. Thời gian thực của từng giai đoạn nằm trong tên file log (`results/logs/`).
- Hai lỗi đã được sửa trong `src/final_evaluation.py` (một khối assert giả và lỗi `feature_names_in_`) được sửa trước khi file được commit, nên **không** có commit `fix:` riêng cho chúng. Tôi không tạo commit giả để mô phỏng lại các lỗi này.
- Các commit ở bước 5–7 chỉ chứa code. Ở thời điểm checkout các commit đó, chưa có kết quả nào được tạo ra.
- Repo đang dùng `core.autocrlf=true`. File được lưu trong Git ở dạng LF, giống file gốc. Nếu clone trên Windows với autocrlf bật, file văn bản (kể cả `adult.csv`) có thể bị đổi sang CRLF, làm sha256 khác với audit. Khi cần kiểm tra hash, nên clone với `git -c core.autocrlf=false clone …`.
