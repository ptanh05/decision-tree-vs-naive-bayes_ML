# LỘ TRÌNH HỌC LẠI DỰ ÁN QUA 12 COMMIT

**Đề tài:** Nghiên cứu và so sánh Decision Tree và Naive Bayes trong bài toán dự đoán mức thu nhập cá nhân (Adult Census Income).

**Đối tượng:** sinh viên năm cuối CNTT chuẩn bị bảo vệ đồ án môn Machine Learning.

**Cách dùng tài liệu:** học lần lượt từng commit. Ở mỗi commit, chạy `git show --stat <hash>` và mở các file được liệt kê, đọc code song song với phần giải thích bên dưới. Làm phần bài tập rồi tự trả lời các câu hỏi bảo vệ **trước khi** đọc đáp án. Chỉ chuyển sang commit tiếp theo khi đã đạt "điều kiện chuyển tiếp".

> **Nguyên tắc về số liệu:** mọi con số trong tài liệu này được lấy từ các kết quả đã đóng băng: `results/metrics/*.csv`, `results/reports/final_audit/*`, `BAO_CAO_CUOI_KY_ML.md`. Tài liệu không chạy lại mô hình và không tạo số liệu mới. Nếu giảng viên hỏi một số liệu không có ở đây, hãy trả lời trung thực: "Em chưa đo chỉ số đó."

> **Lưu ý về bài tập:** một số bài tập yêu cầu viết code nhỏ. Hãy làm trong một notebook **riêng** (ví dụ `scratch.ipynb`, không commit) và **không ghi đè** file nào trong `results/`. Tuyệt đối không gọi `predict` trên tập Test trong các bài tập.

```bash
git log --oneline --reverse
git show --stat ce98eb3
git switch -c hoc-lai ce98eb3   # tạo nhánh riêng nếu muốn thử nghiệm; chỉ checkout hash thì sẽ vào detached HEAD
git switch main                 # quay về
```

---

## Bức tranh toàn cảnh (đọc trước)

```
adult.csv (32.561 dòng)
   │  clean(): strip, '?'→NaN, nhãn→0/1               [commit 2]
   ▼
EDA: hiểu phân phối, thiếu dữ liệu, mất cân bằng     [commit 3]
   ▼
StratifiedGroupKFold → Train 22.793 | Val 4.884 | Test 4.884 (khóa lại)   [commit 4]
   ▼
Pipeline = Preprocessor + Model (DT [5] / CNB [6]), metric [7]
   ▼
Tuning: fit trên Train, chấm trên Val, chọn theo F1 lớp >50K   [commit 8]
   ▼
Refit trên Train+Val → predict Test MỘT lần → bootstrap, McNemar   [commit 9]
   ▼
Audit chỉ đọc (61 PASS / 4 WARNING / 0 FAIL)   [commit 10]  →  Báo cáo [11]  →  Hướng dẫn Git [12]
```

**Ba câu chốt cần thuộc lòng và hiểu:**
1. Tập Test **không được dùng** cho tuning, chọn mô hình, chọn siêu tham số hay chỉnh ngưỡng. Mô hình chỉ dự đoán trên Test **một lần**, sau khi mọi lựa chọn (mô hình, siêu tham số, ngưỡng 0,5) đã đóng băng. Lưu ý trung thực: EDA ban đầu chạy trên toàn bộ 32.561 mẫu **trước khi chia tập**, nên thống kê mô tả (phân phối đặc trưng và nhãn) của các mẫu sau này thuộc Test đã được quan sát. Vì vậy không thể nói Test "hoàn toàn chưa từng được tiếp cận"; đây là một hạn chế phương pháp luận được công khai.
2. Hai mô hình có F1 gần bằng nhau: DT 0,6778, CNB 0,6756. Khoảng tin cậy 95% của chênh lệch là [−0,0133; +0,0175], **chứa 0**. Vì vậy chưa đủ bằng chứng để nói mô hình nào tốt hơn. Kết quả này cũng **không** chứng minh hai mô hình tương đương.
3. Hai mô hình đánh đổi khác nhau. DT có Recall cao hơn (0,8639 so với 0,7891) nhưng nhiều FP hơn. CNB có Accuracy cao hơn (0,8176 so với 0,8022) và Precision cao hơn, nhưng bỏ sót nhiều người `>50K` hơn.

---

## COMMIT 0 – `4f56da8` Initial commit
Đây là commit gốc của repository trên GitHub, chỉ chứa `README.md` ở thư mục gốc. Commit này được giữ nguyên, không có nội dung ML nào cần học.

---

## COMMIT 1 – `454322b` chore: initialize machine learning project

1. **Hash / message:** `454322b`, `chore: initialize machine learning project`.
2. **Mục tiêu:** cố định môi trường để người khác chạy lại cho ra cùng kết quả.
3. **Vấn đề cần giải quyết:** các phiên bản thư viện khác nhau có thể cho kết quả khác nhau. Ví dụ, biên các khoảng của `KBinsDiscretizer` phụ thuộc vào cách thư viện tính phân vị; dự án cấu hình rõ `strategy="quantile"` và phương pháp `averaged_inverted_cdf`, và chạy trên scikit-learn 1.9.0, nên ghim phiên bản giúp tái lập đúng các biên này.
4. **File quan trọng:**
   - `requirements.txt`: ghim phiên bản numpy 2.4.4, pandas 3.1.1, scikit-learn 1.9.0, matplotlib, seaborn, joblib, nbformat/nbclient.
   - `.gitignore`: bỏ qua `__pycache__`, `.venv`, `.env`, `.ipynb_checkpoints`.
5. **Lý thuyết:** tính tái lập (reproducibility) là một yêu cầu của nghiên cứu khoa học, không phải một tùy chọn.
6. **Kỹ thuật:** môi trường ảo (`python -m venv .venv`), `pip install -r requirements.txt`.
7. **Luồng xử lý:** `requirements.txt` → `pip` → môi trường giống hệt môi trường đã chạy thí nghiệm (Python 3.12.10, chỉ dùng CPU).
8. **Quyết định thiết kế:** ghim phiên bản cụ thể cho các thư viện ảnh hưởng đến kết quả. `scipy` và `ipykernel` chưa được ghim; đây là một điểm có thể cải thiện.
9. **Lỗi thường gặp:** commit `.env` có chứa API key; commit thư mục `.venv` nặng hàng trăm MB; không ghi rõ phiên bản thư viện.
10. **Câu hỏi bảo vệ:** "Làm sao để người khác chạy lại được thí nghiệm của em?"
11. **Trả lời mẫu:** "Em ghim phiên bản thư viện trong `requirements.txt` và cố định seed 42 cho việc chia dữ liệu, cho Decision Tree và cho bootstrap. Em lưu chỉ số chia tập vào `split_indices.csv` và lưu sha256 của dữ liệu cùng các artifact để kiểm tra rằng không có gì bị thay đổi."
12. **Bài tập:** tạo một môi trường ảo mới, cài từ `requirements.txt`, rồi chạy `python -c "import sklearn; print(sklearn.__version__)"`. Kết quả phải là `1.9.0`.
13. **Điều kiện chuyển tiếp:** giải thích được vì sao cần ghim phiên bản thư viện và vì sao cần đặt seed.

---

## COMMIT 2 – `1468f55` data: add adult census income dataset pipeline

1. **Hash / message:** `1468f55`, `data: add adult census income dataset pipeline`.
2. **Mục tiêu:** đọc dữ liệu gốc một cách an toàn, chuẩn hóa nhãn và định nghĩa danh sách đặc trưng.
3. **Vấn đề cần giải quyết:**
   - Giá trị thiếu được ghi bằng chuỗi `?` chứ không phải ô trống.
   - Chuỗi có thể chứa khoảng trắng thừa.
   - Nhãn là chuỗi `<=50K`/`>50K`.
   - Cột `education` trùng thông tin với `education.num`.
4. **File quan trọng:**
   - `data/raw/adult.csv`: 32.561 dòng × 15 cột; sha256 `250e154e…`. **Không bao giờ được sửa file này.**
   - `src/data_loader.py`:
     - `load_raw()`: chỉ đọc file.
     - `clean()`: strip khoảng trắng, đổi `?` thành NaN, ánh xạ nhãn (`<=50K` → 0, `>50K` → 1). Hàm báo lỗi nếu gặp nhãn lạ.
     - Khai báo `NUMERIC_FEATURES` (6 cột), `CATEGORICAL_FEATURES` (7 cột), `DROPPED_FEATURES = ["education"]`.
     - Định nghĩa `split_data()` (học kỹ ở commit 4) và `get_splits()`, là "điểm vào duy nhất" để mọi module nhận cùng 3 tập.
5. **Lý thuyết:** học có giám sát, bài toán phân loại nhị phân, lớp dương (positive class) = `>50K`.
6. **Kỹ thuật:** tách bạch hai loại biến đổi:
   - Biến đổi **không học tham số** (strip, đổi `?`, ánh xạ nhãn) có thể làm trên toàn bộ dữ liệu vì không rò rỉ thông tin gì.
   - Biến đổi **có học tham số** (imputer, encoder, discretizer) phải nằm trong Pipeline và chỉ fit trên Train.
7. **Luồng xử lý:** CSV → `load_raw` → `clean` → DataFrame (nhãn kiểu int) → `xy()` → X gồm 13 đặc trưng, y là nhãn.
8. **Quyết định thiết kế:**
   - Bỏ `education` vì 16 trình độ ánh xạ 1-1 với 16 giá trị của `education.num`. Theo giả định độc lập có điều kiện, nếu giữ cả hai thì Naive Bayes sẽ coi chúng là hai bằng chứng độc lập và có **nguy cơ** "đếm đôi" cùng một thông tin. Dự án không chạy thí nghiệm giữ cả hai cột, nên mức tác động này chưa được đo trực tiếp.
   - Giữ `?` dưới dạng NaN để xử lý sau, không xóa dòng.
9. **Lỗi thường gặp:**
   - Dùng `dropna()` để xóa 2.399 dòng có `?`. Đây là lỗi vì nhóm này có tỉ lệ `>50K` chỉ 13,9%, so với 24,9% ở phần còn lại. Xóa các dòng này sẽ làm lệch phân phối dữ liệu.
   - Mã hóa nhãn bằng `LabelEncoder` mà không kiểm tra lớp nào thành 1.
10. **Câu hỏi:** "Tại sao em bỏ cột `education`?"
11. **Trả lời:** "Vì `education` và `education.num` ánh xạ một-một với nhau, em đã kiểm tra trong `check_design.py`. Giữ cả hai không thêm thông tin mới. Với Naive Bayes còn có nguy cơ cùng một bằng chứng bị nhân hai lần vào tích xác suất, vì mô hình giả định các đặc trưng độc lập khi biết lớp. Em chưa đo trực tiếp mức tác động này; đây là lập luận dựa trên giả định của mô hình."
12. **Bài tập:** viết 3 dòng pandas kiểm tra rằng mỗi giá trị `education` chỉ ứng với đúng một giá trị `education.num` (gợi ý: `groupby('education')['education.num'].nunique().max() == 1`).
13. **Điều kiện chuyển tiếp:** phân biệt được biến đổi "không học tham số" với biến đổi "học tham số", và giải thích được vì sao không xóa các dòng có `?`.

---

## COMMIT 3 – `ce98eb3` feat: perform exploratory data analysis

1. **Hash / message:** `ce98eb3`, `feat: perform exploratory data analysis`.
2. **Mục tiêu:** hiểu dữ liệu trước khi chọn cách tiền xử lý và chọn mô hình.
3. **Vấn đề cần giải quyết:** dữ liệu mất cân bằng đến mức nào? Thiếu dữ liệu ở đâu và có liên quan đến nhãn không? Phân phối có lệch không? Các đặc trưng phụ thuộc nhau ra sao?
4. **File quan trọng:**
   - `src/eda.py`: toàn bộ phép tính EDA.
   - `scripts/make_eda_notebook.py`, `scripts/run_notebook.py`: sinh notebook và chạy notebook tự động.
   - `notebooks/01_EDA.ipynb`: notebook đã chạy, có sẵn output.
   - `results/eda/*.csv`: các bảng thống kê.
   - `results/figures/eda/*.png`: 14 hình.
5. **Lý thuyết và kết quả thực tế:**
   - **Phân phối nhãn:** 24.720 mẫu `<=50K` (75,92%) và 7.841 mẫu `>50K` (24,08%). Đây là **mất cân bằng lớp** (class imbalance). Một mô hình luôn đoán `<=50K` vẫn đạt Accuracy khoảng 0,7592.
   - **Missing values:** workclass có 1.836 giá trị thiếu, occupation 1.843, native.country 583, tổng cộng 2.399 dòng. Tình trạng thiếu dữ liệu **có liên hệ với biến mục tiêu** (tỉ lệ `>50K` là 13,9% ở nhóm thiếu so với 24,9% ở nhóm đủ). Tuy nhiên, phân tích hiện tại **chưa đủ để xác định** cơ chế thiếu thuộc MCAR, MAR hay MNAR. Quan sát một mối liên hệ chỉ cho biết việc thiếu không độc lập với nhãn (nên khó là MCAR). Còn phân biệt MAR (thiếu phụ thuộc vào các biến *quan sát được*) với MNAR (thiếu phụ thuộc vào chính giá trị *bị thiếu*) đòi hỏi mô hình hóa cơ chế thiếu hoặc thông tin ngoài dữ liệu; dự án không làm việc này.
   - **Phân phối lệch:**
     - 91,67% giá trị `capital.gain` bằng 0, độ lệch (skew) 11,95.
     - 95,33% giá trị `capital.loss` bằng 0.
     - Có 159 dòng `capital.gain = 99999`, và toàn bộ thuộc lớp `>50K`. Giá trị này nhiều khả năng là mã chặn trên (top-code).
   - **Pearson:** đo quan hệ tuyến tính giữa hai biến số. Giữa các thuộc tính số, |r| ≤ 0,15. Tương quan với nhãn: education.num 0,335, age 0,234.
   - **Cramér's V:** đo mức liên hệ giữa hai biến phân loại, nhận giá trị trong [0, 1], dựa trên thống kê χ². Kết quả: relationship–sex 0,649; marital.status–relationship 0,488. Naive Bayes giả định các đặc trưng **độc lập có điều kiện khi biết lớp Y**, tức P(x_i, x_j | y) = P(x_i | y)·P(x_j | y). Cramér's V ở đây tính trên **phân phối chung** (không tách theo lớp), nên chỉ cho thấy mối liên hệ đáng kể giữa các đặc trưng phân loại. **Đây là dấu hiệu cảnh báo rằng giả định độc lập có điều kiện có thể không phù hợp, nhưng chưa phải kiểm định trực tiếp giả định đó.** Muốn kiểm tra trực tiếp cần đo liên hệ trong từng lớp (ví dụ Cramér's V riêng trong nhóm `<=50K` và nhóm `>50K`); dự án chưa làm phép đo này.
   - **Mutual Information (MI):** đo lượng thông tin một đặc trưng mang về nhãn, bắt được cả quan hệ phi tuyến. Cao nhất: relationship 0,1146, marital.status 0,1085, capital.gain 0,0833. Thấp nhất: race 0,0058.
   - **Dữ liệu trùng lặp:** có 23 nhóm bản ghi trùng toàn bộ 15 cột. Thông tin này dẫn tới thiết kế chia tập theo nhóm ở commit 4.
6. **Kỹ thuật:** histogram và boxplot theo lớp, heatmap tương quan, bảng tỉ lệ `>50K` theo từng hạng mục. Ví dụ, nhóm Married-civ-spouse có tỉ lệ 44,7%, nhóm Never-married chỉ 4,6%.
7. **Luồng xử lý:** `clean(load_raw())` → `eda.py` → file CSV và PNG → notebook trình bày.
8. **Quyết định thiết kế:** EDA chỉ dùng để **hiểu dữ liệu** và **chọn kiểu tiền xử lý**, ví dụ quyết định xử lý riêng giá trị 0 của capital. EDA **không** được dùng để chọn mô hình hay siêu tham số. **Hạn chế cần công khai:** EDA chạy trên toàn bộ 32.561 mẫu trước khi chia tập, nên thống kê mô tả của phần dữ liệu sau này thành Test đã được quan sát.
9. **Lỗi thường gặp:**
   - Chọn đặc trưng dựa vào MI tính trên toàn bộ dữ liệu, kể cả Test, rồi báo cáo kết quả Test như thể chưa từng nhìn thấy Test. Đây là rò rỉ dữ liệu ở mức "quyết định".
   - Đọc tương quan là quan hệ nhân quả, ví dụ "đã kết hôn **gây ra** thu nhập cao".
10. **Câu hỏi:** "EDA của em có nhìn vào Test không? Có làm rò rỉ dữ liệu không?"
11. **Trả lời:** "Có, ở mức thống kê mô tả. EDA được làm trên toàn bộ 32.561 mẫu trước khi chia tập, nên phân phối đặc trưng và nhãn của các mẫu sau này thuộc Test đã được quan sát; em không tuyên bố Test hoàn toàn chưa từng được tiếp cận. Tuy nhiên EDA chỉ dẫn đến các quyết định mang tính cấu trúc, như cách mã hóa và cách rời rạc hóa. Test không được dùng để tuning, chọn mô hình, chọn siêu tham số hay chỉnh ngưỡng. Mọi tham số có học (biên khoảng, bộ mã hóa) đều fit trên Train. Việc chọn mô hình và siêu tham số chỉ dựa trên Validation. Cách làm chặt chẽ nhất là chỉ làm EDA trên Train, và em ghi nhận đó là một điểm có thể cải thiện."
12. **Bài tập:** tính lại tỉ lệ `>50K` ở nhóm có `?` và nhóm không có `?` từ `adult.csv`. Kết quả phải là 13,88% và 24,89%.
13. **Điều kiện chuyển tiếp:** giải thích được sự khác nhau giữa Pearson, Cramér's V và MI, và nêu ra được 3 phát hiện EDA đã ảnh hưởng tới cách tiền xử lý.

---

## COMMIT 4 – `f254108` feat: implement machine learning preprocessing pipelines

1. **Hash / message:** `f254108`, `feat: implement machine learning preprocessing pipelines`.
2. **Mục tiêu:** chia tập dữ liệu chống rò rỉ, và xây pipeline tiền xử lý riêng cho từng thuật toán.
3. **Vấn đề cần giải quyết:**
   - Bản ghi trùng lặp có thể rơi vào cả Train lẫn Test, khi đó mô hình được "nhìn trước" đáp án.
   - Decision Tree cần đầu vào là số. CategoricalNB cần đầu vào là **mã nguyên rời rạc**.
   - Hạng mục chưa từng xuất hiện trong Train (ví dụ một quốc gia hiếm) có thể làm mô hình lỗi khi dự đoán.
4. **File quan trọng:**
   - `src/preprocessing.py`: `build_dt_preprocessor`, `build_cnb_preprocessor`, `ZeroInflatedDiscretizer`, `SafeOrdinalEncoder`, `cnb_min_categories`.
   - `src/check_design.py`: kiểm tra các giả định thiết kế, ví dụ quan hệ 1-1 giữa `education` và `education.num`.
   - `data/processed/split_indices.csv`: tập của từng dòng (khóa cố định, sha256 được kiểm tra trong audit).
   - `data/processed/split_summary.csv`: kích thước và tỉ lệ lớp của từng tập.
   - `data/processed/split_old_vs_new.csv`: so sánh cách chia cũ (`train_test_split` hai bước, không xét trùng lặp) với cách chia mới.
   - `notebooks/02_Preprocessing.ipynb`: notebook minh họa.
5. **Lý thuyết:**
   - **Train / Validation / Test:**
     - Train dùng để học tham số mô hình.
     - Validation dùng để chọn siêu tham số.
     - Test dùng để ước lượng hiệu năng cuối: mô hình chỉ dự đoán trên Test **một lần**, và Test không tham gia bất kỳ lựa chọn nào. (Thống kê mô tả của toàn bộ dữ liệu, gồm cả phần Test, đã được xem trong EDA; xem commit 3.)
   - **Data leakage (rò rỉ dữ liệu):** thông tin từ dữ liệu đánh giá lọt vào quá trình huấn luyện, làm kết quả lạc quan giả tạo.
   - **Stratification (phân tầng):** giữ tỉ lệ lớp như nhau ở mọi tập. Kết quả thực tế: tỉ lệ `>50K` bằng 0,2408 ở cả ba tập.
   - **Group split (chia theo nhóm):** các bản ghi trong cùng một nhóm phải nằm cùng một tập.
6. **Kỹ thuật:**
   - **StratifiedGroupKFold(20 fold, shuffle, seed 42)**, mỗi fold khoảng 5% dữ liệu. Fold 0–2 thành Test (15%), fold 3–5 thành Validation (15%), 14 fold còn lại thành Train (70%). Nhóm được xác định bởi `duplicate_group_ids`, gom các dòng giống nhau ở **cả 15 cột**. Kết quả: Train 22.793, Val 4.884, Test 4.884 mẫu.
   - **SimpleImputer(constant="Unknown"):** biến NaN thành một hạng mục riêng, vì việc thiếu dữ liệu cũng mang thông tin về nhãn.
   - **OneHotEncoder(handle_unknown="ignore")** cho Decision Tree: mỗi hạng mục thành một cột 0/1, nên không áp đặt thứ tự giả lên các hạng mục. Một hạng mục chưa xuất hiện lúc fit được mã hóa thành **vectơ toàn 0 trên các cột one-hot của đặc trưng đó** (các đặc trưng khác không bị ảnh hưởng), thay vì gây lỗi. Tổng cộng 92 đặc trưng.
   - **KBinsDiscretizer(strategy="quantile")** cho CNB: cắt age, fnlwgt, education.num, hours.per.week thành `n_bins` khoảng, mỗi khoảng có số mẫu gần bằng nhau. Biên các khoảng được học **từ Train**.
   - **ZeroInflatedDiscretizer** (tự viết): với cột capital, mã 0 dành cho giá trị = 0; phần giá trị dương được chia theo phân vị thành tối đa 4 khoảng. Nếu dùng KBins thông thường thì >90% giá trị bằng 0 sẽ gộp cả cột thành 1 khoảng, làm mất thông tin.
   - **SafeOrdinalEncoder** (tự viết): hạng mục lạ được gán một mã riêng (bằng số hạng mục đã biết), vì CategoricalNB không chấp nhận giá trị −1.
   - **`cnb_min_categories`:** khai báo cho CategoricalNB số giá trị tối đa của mỗi cột, để mô hình không lỗi khi gặp mã chưa từng thấy.
   - Kết quả: CNB nhận 13 đặc trưng rời rạc.
7. **Luồng xử lý:** DataFrame → ColumnTransformer gồm nhánh số (passthrough hoặc binning) và nhánh phân loại (impute rồi encode) → ma trận số → mô hình.
8. **Quyết định thiết kế và lý do:**
   - Thay `train_test_split` bằng StratifiedGroupKFold để các bản ghi trùng không bị tách sang các tập khác nhau. File `split_old_vs_new.csv` lưu lại sự khác biệt giữa hai cách chia.
   - Hai mô hình có pipeline riêng, vì mỗi thuật toán có yêu cầu đầu vào khác nhau.
   - Mọi bước có học tham số đều nằm trong `Pipeline`. Khi gọi `fit(X_train)` thì chỉ học từ Train; khi refit trên Train+Val thì toàn bộ pipeline được học lại.
   - Không dùng SMOTE. Mất cân bằng lớp được xử lý bằng cơ chế sẵn có của từng thuật toán (`class_weight` cho DT, `fit_prior` cho NB).
9. **Lỗi thường gặp:**
   - **Fit encoder hoặc discretizer trên toàn bộ dữ liệu**, sau đó mới chia tập. Khi đó biên phân vị đã "nhìn thấy" Test, gây rò rỉ dữ liệu.
   - Dùng OrdinalEncoder cho Decision Tree với biến không có thứ tự, ví dụ đặt Private=3 > Federal=1. Cây sẽ chia theo thứ tự giả này.
   - Quên xử lý hạng mục lạ, khiến mô hình lỗi khi chạy trên Test.
10. **Câu hỏi:** "Tại sao preprocessing phải fit trên Train mà không fit trên toàn bộ dữ liệu?"
11. **Trả lời:** "Vì các tham số như biên phân vị hay danh sách hạng mục là thứ mô hình 'học'. Nếu học chúng từ cả Test thì thông tin của Test đã lọt vào mô hình, và điểm số trên Test không còn là ước lượng trung thực cho dữ liệu mới. Đặt các bước này trong `Pipeline` bảo đảm `fit` chỉ nhìn thấy dữ liệu huấn luyện."
12. **Bài tập:** đọc `split_indices.csv` và kiểm tra: (a) mỗi `row_index` xuất hiện đúng 1 lần; (b) mỗi `dup_group` chỉ thuộc 1 tập (`groupby('dup_group')['split'].nunique().max() == 1`).
13. **Điều kiện chuyển tiếp:** tự vẽ được sơ đồ hai pipeline (DT và CNB), và giải thích được vì sao cần `ZeroInflatedDiscretizer` và vì sao cần `SafeOrdinalEncoder`.

---

## COMMIT 5 – `b9550a4` feat: implement decision tree classifier

1. **Hash / message:** `b9550a4`, `feat: implement decision tree classifier`.
2. **Mục tiêu:** dựng pipeline Decision Tree và định nghĩa không gian siêu tham số.
3. **Vấn đề cần giải quyết:** nếu không bị giới hạn, cây sẽ học thuộc dữ liệu (overfitting). Ngoài ra lớp `>50K` là lớp thiểu số.
4. **File quan trọng:** `src/decision_tree.py`:
   - `DT_GRID`: 2 × 10 × 4 × 5 × 2 = 800 cấu hình.
   - `build_dt()`: tạo Pipeline gồm preprocessor của DT và `DecisionTreeClassifier`.
   - `dt_complexity()`: trả về số lá và độ sâu của cây.
5. **Lý thuyết:**
   - **Entropy:** H(S) = −Σ p_k log₂ p_k. H = 0 khi tập chỉ có một lớp, H = 1 khi hai lớp chia 50/50.
   - **Information Gain:** IG = H(cha) − Σ (|S_v|/|S|)·H(S_v). Thuật toán chọn phép chia có IG lớn nhất.
   - **Gini:** 1 − Σ p_k². Gini tính nhanh hơn entropy, và thường cho kết quả gần giống. Trong dự án, F1 tốt nhất với entropy là 0,6796, với gini là 0,6786.
   - **Overfitting:** cây sâu khớp cả nhiễu trong dữ liệu. Bằng chứng thực tế: cây gini không giới hạn có F1 trên Train là 0,9999 nhưng F1 trên Val chỉ khoảng 0,60, với hơn 3.000 lá.
6. **Giải thích kỹ thuật (thuật toán CART):** tại mỗi nút, thử mọi đặc trưng và mọi ngưỡng chia nhị phân, chọn phép chia làm giảm độ hỗn tạp nhiều nhất, rồi lặp lại cho các nút con. Thuật toán dừng khi:
   - đạt `max_depth`;
   - nút có ít hơn `min_samples_split` mẫu;
   - một phép chia sẽ tạo ra lá có ít hơn `min_samples_leaf` mẫu;
   - hoặc nút đã thuần nhất.

   `class_weight="balanced"` gán trọng số mỗi lớp bằng n / (2·n_lớp). Như vậy mỗi mẫu `>50K` nặng khoảng **3,15 lần** mẫu `<=50K` khi tính độ hỗn tạp (Train: 17.304/5.489 ≈ 3,15; Train+Val: 21.012/6.665 ≈ 3,15). Cụ thể trên Train (n = 22.793): w₀ = 22.793/(2·17.304) ≈ 0,659 và w₁ = 22.793/(2·5.489) ≈ 2,076. Cần phân biệt: **trọng số trên mỗi mẫu** chênh nhau ≈ 3,15:1, còn **tổng trọng số của mỗi lớp** bằng nhau (mỗi lớp = n/2), nghĩa là hai lớp được "cân" như nhau khi tính độ hỗn tạp. `predict_proba` trả về tỉ lệ (có trọng số) của lớp 1 trong lá.
7. **Luồng xử lý:** X (13 cột) → One-Hot (92 cột) → cây → xác suất của lá → ngưỡng 0,5 → nhãn dự đoán.
8. **Quyết định thiết kế:**
   - Thêm `max_depth=None` vào lưới để quan sát rõ hiện tượng overfitting.
   - `random_state=42` để kết quả tái lập được, vì khi hai phép chia tốt ngang nhau, thứ tự xét đặc trưng có yếu tố ngẫu nhiên.
   - Không chuẩn hóa dữ liệu số, vì phép chia theo ngưỡng không phụ thuộc vào thang đo.
9. **Lỗi thường gặp:**
   - Chọn cây sâu vì F1 trên Train cao.
   - Nghĩ rằng `class_weight` là một thuật toán khác. Thực chất đó vẫn là Decision Tree, chỉ thay đổi trọng số của mẫu.
   - Cho rằng cây là ổn định. Thực tế dữ liệu thay đổi nhỏ có thể làm cấu trúc cây thay đổi lớn.
10. **Câu hỏi:** "Cây cuối của em có bao nhiêu lá? Em kiểm soát overfitting thế nào?"
11. **Trả lời:** "Cây cuối có 365 lá, độ sâu 12, với cấu hình entropy, `min_samples_leaf=10`, `class_weight=balanced`. Em kiểm soát độ phức tạp bằng tiền cắt tỉa (`max_depth`, `min_samples_leaf`, `min_samples_split`) và chọn cấu hình theo F1 trên Validation. F1 trên Val đạt đỉnh 0,6796 ở độ sâu 12, rồi giảm khi cây sâu hơn."
12. **Bài tập:** tính bằng tay entropy của tập Train (5.489 mẫu dương trên 22.793 mẫu, p ≈ 0,2408). Kết quả khoảng 0,796 bit.
13. **Điều kiện chuyển tiếp:** tự tính được IG cho một ví dụ nhỏ, và giải thích được tác dụng của từng siêu tham số trong `DT_GRID`.

---

## COMMIT 6 – `1acb45f` feat: implement categorical naive bayes classifier

1. **Hash / message:** `1acb45f`, `feat: implement categorical naive bayes classifier`.
2. **Mục tiêu:** xây mô hình xác suất CategoricalNB, và GaussianNB làm đối chứng phụ.
3. **Vấn đề cần giải quyết:** CategoricalNB chỉ nhận giá trị rời rạc. Số khoảng rời rạc hóa và số hạng mục được học từ chính dữ liệu fit.
4. **File quan trọng:** `src/naive_bayes.py`:
   - `CNB_GRID`: 9 × 7 × 2 = 126 cấu hình.
   - `CategoricalNBPipeline`: lớp bọc. Khi `fit`, lớp này fit preprocessor trước, tính `min_categories`, rồi mới fit CategoricalNB.
   - `GNB_GRID`: 7 cấu hình, chỉ dùng 6 thuộc tính số.
5. **Lý thuyết:**
   - **Định lý Bayes:** P(y|x) = P(y)·P(x|y) / P(x).
   - **Giả định độc lập có điều kiện:** P(x|y) = Π_j P(x_j|y). Nhờ giả định này, mô hình chỉ cần ước lượng 13 bảng tần suất nhỏ thay vì một bảng kết hợp khổng lồ.
   - **Laplace / Lidstone smoothing:** P(x_j=v | y) = (N_yjv + α) / (N_y + α·K_j). Hệ số α tránh xác suất bằng 0 cho các tổ hợp chưa từng gặp. α = 1 gọi là Laplace, α < 1 gọi là Lidstone.
   - **Prior:** `fit_prior=True` dùng P(y) theo tần suất thực (khoảng 76/24). `False` dùng prior đều 50/50.
6. **Giải thích kỹ thuật:**
   - **Fit** = đếm tần suất có làm trơn.
   - **Dự đoán** = tính log P(y) + Σ log P(x_j|y) cho từng lớp, rồi chuẩn hóa thành xác suất.
   - Phải rời rạc hóa vì CategoricalNB mô hình hóa mỗi đặc trưng bằng phân phối rời rạc. GaussianNB có thể nhận số thực, nhưng giả định phân phối chuẩn, trong khi `capital.*` lệch rất mạnh.
7. **Luồng xử lý:** X → binning và ordinal (13 mã nguyên) → bảng tần suất theo từng lớp → posterior → ngưỡng 0,5.
8. **Quyết định thiết kế:**
   - Mở rộng lưới `n_bins`: cấu hình tốt nhất ban đầu nằm đúng biên 20, nên lưới được thêm các giá trị 25–50. Kết quả 20 vẫn tốt nhất. Lưới cũ được lưu trong `archive/`.
   - GaussianNB chỉ là thí nghiệm phụ, để minh họa hậu quả khi giả định phân phối sai.
9. **Lỗi thường gặp:**
   - Đưa trực tiếp số thực vào CategoricalNB: thuật toán sẽ coi mỗi giá trị khác nhau là một hạng mục.
   - Đặt α = 0.
   - Tin rằng xác suất NB đầu ra là chính xác. Do các đặc trưng phụ thuộc nhau, xác suất thường bị "cực đoan", dồn về gần 0 hoặc 1.
10. **Câu hỏi:** "Relationship và sex liên hệ mạnh (Cramér's V 0,649), giả định độc lập có thể không đúng, sao NB vẫn chạy tốt?"
11. **Trả lời:** "Trước hết, Cramér's V 0,649 đo trên phân phối chung, nên chỉ là dấu hiệu cảnh báo, chưa phải kiểm định trực tiếp giả định độc lập *có điều kiện khi biết lớp*. Dù giả định có bị vi phạm thì để phân loại đúng chỉ cần **thứ tự** posterior giữa các lớp đúng, không cần giá trị xác suất chính xác (Domingos & Pazzani, 1997). Vi phạm giả định chủ yếu làm xác suất kém hiệu chỉnh. Dù vậy, CNB của em vẫn có ROC-AUC 0,9011 trên Test. Hạn chế là các đặc trưng tương quan bị đếm trùng bằng chứng."
12. **Bài tập:** với α = 1, một đặc trưng có K = 5 giá trị, lớp y có N_y = 100 mẫu và giá trị v chưa từng xuất hiện cùng y, hãy tính P(v|y). Đáp án: 1/105 ≈ 0,0095.
13. **Điều kiện chuyển tiếp:** tự viết được công thức dự đoán của NB, và giải thích được vai trò của α, `fit_prior` và `n_bins`.

---

## COMMIT 7 – `ece6056` feat: add model evaluation utilities

1. **Hash / message:** `ece6056`, `feat: add model evaluation utilities`.
2. **Mục tiêu:** dùng một bộ metric thống nhất cho mọi mô hình.
3. **Vấn đề cần giải quyết:** dữ liệu mất cân bằng, nên chỉ dùng Accuracy sẽ gây hiểu lầm. Ngoài ra cần lấy đúng cột xác suất của lớp 1.
4. **File quan trọng:** `src/evaluation.py`:
   - `compute_metrics()` tính Accuracy, Precision, Recall, F1, ROC-AUC, `pr_auc`. Cột `pr_auc` thực chất là **Average Precision** (`average_precision_score`); đây là tên cột chưa chính xác về thuật ngữ, được giữ nguyên vì là artifact đã đóng băng.
   - `positive_proba()` lấy cột xác suất theo `classes_`, không giả định vị trí cột.
5. **Lý thuyết:**
   - **Confusion Matrix:** TN, FP, FN, TP. Trong bài toán này:
     - **FP**: người có thu nhập `<=50K` bị đoán nhầm thành `>50K`.
     - **FN**: người có thu nhập `>50K` bị bỏ sót.
   - **Precision** = TP/(TP+FP): trong số người được đoán là `>50K`, bao nhiêu phần trăm đúng.
   - **Recall** = TP/(TP+FN): trong số người thật sự `>50K`, mô hình tìm được bao nhiêu phần trăm.
   - **F1** = 2TP/(2TP+FP+FN): trung bình điều hòa của Precision và Recall. Đây là tiêu chí chọn mô hình.
   - **ROC-AUC:** xác suất một mẫu dương ngẫu nhiên được chấm điểm cao hơn một mẫu âm ngẫu nhiên. Không phụ thuộc ngưỡng, và phải tính trên **score/xác suất**, không phải nhãn dự đoán. TPR được chuẩn hóa theo số mẫu dương, FPR theo số mẫu âm, nên ROC-AUC không thay đổi trực tiếp khi chỉ thay tỉ lệ lớp (prevalence) mà giữ nguyên phân phối điểm trong từng lớp. Tuy vậy, điều đó không có nghĩa ROC-AUC hoàn toàn "miễn nhiễm" với mất cân bằng trong mọi tình huống (ví dụ khi lớp hiếm có ít mẫu, ước lượng sẽ dao động nhiều hơn). Mô hình đoán ngẫu nhiên có ROC-AUC = 0,5.
   - **Average Precision:** AP = Σ (R_n − R_{n−1})·P_n, tức tổng theo bậc thang trên đường PR. AP **không** đồng nhất với diện tích dưới đường PR tính bằng phương pháp hình thang (nội suy tuyến tính); xem commit 9. Mô hình đoán ngẫu nhiên có AP = tỉ lệ lớp dương (0,2408).
6. **Vì sao Accuracy không đủ:** mô hình "luôn đoán `<=50K`" đạt Accuracy 0,7592 nhưng F1 = 0, vì không tìm được một người `>50K` nào.
7. **Luồng xử lý:** (y_true, y_pred tại ngưỡng 0,5, y_score) → dict 6 chỉ số.
8. **Quyết định thiết kế:**
   - Cố định `pos_label=1` (lớp `>50K`).
   - Cố định ngưỡng 0,5, không tinh chỉnh ngưỡng.
   - ROC-AUC và AP được tính trên **xác suất**, không tính trên nhãn dự đoán.
9. **Lỗi thường gặp:**
   - Tính ROC-AUC trên nhãn 0/1 thay vì trên xác suất.
   - Lấy nhầm `predict_proba[:, 0]`, tức xác suất lớp 0.
   - Gọi AP là "diện tích dưới đường PR" mà không nói rõ phương pháp tính.
10. **Câu hỏi:** "FP và FN cái nào nghiêm trọng hơn trong bài toán của em?"
11. **Trả lời:** "Điều đó tùy ứng dụng. Ví dụ, nếu mục đích là tìm khách hàng tiềm năng thu nhập cao thì FN (bỏ sót khách hàng) tốn kém hơn, nên nên ưu tiên Recall. Nếu đoán nhầm gây ra chi phí cao thì FP nghiêm trọng hơn. Đề tài không có hàm chi phí thực tế, nên em dùng F1 để cân bằng hai loại lỗi, và trình bày sự đánh đổi chứ không tuyên bố mô hình nào tốt nhất cho mọi ứng dụng."
12. **Bài tập:** từ CM Test của DT (2902/806/160/1016), tính bằng tay P, R, F1. Kết quả phải là 0,5576; 0,8639; 0,6778.
13. **Điều kiện chuyển tiếp:** tính được mọi metric từ CM bằng tay, và giải thích được vì sao baseline có Accuracy 0,76.

---

## COMMIT 8 – `6d7d9c1` feat: add hyperparameter tuning experiments

1. **Hash / message:** `6d7d9c1`, `feat: add hyperparameter tuning experiments`.
2. **Mục tiêu:** chọn siêu tham số tốt nhất cho từng mô hình, **chỉ dùng Validation**.
3. **Vấn đề cần giải quyết:** có tổng cộng 933 cấu hình (800 + 126 + 7). Cần một tiêu chí chọn rõ ràng và ghi lại mọi lần thử.
4. **File quan trọng:**
   - `src/tuning.py`: loại tập Test ngay sau khi chia, fit từng cấu hình trên Train, chấm điểm trên Val, rồi xếp hạng.
   - `results/metrics/*_validation_grid*.csv`: toàn bộ cấu hình đã thử (cả bản thô và bản đã xếp hạng).
   - `best_configs.json`, `validation_comparison.csv`: cấu hình được chọn và bảng so sánh.
   - `results/models/*_best_trainonly.joblib`: mô hình fit trên Train.
   - `results/figures/validation/*`, `results/logs/phase4_*`, `results/reports/phase4_summary.md`.
5. **Lý thuyết:**
   - **Grid Search:** thử tất cả tổ hợp trong lưới.
   - **Validation và Test khác nhau ở chỗ:**
     - Validation được dùng **nhiều lần** để so sánh các cấu hình, nên điểm của cấu hình thắng trên Validation thường hơi lạc quan. Đây là **selection bias / optimism**: điểm Val của mỗi cấu hình = hiệu năng thật + nhiễu ngẫu nhiên; khi thử 800 cấu hình rồi lấy cấu hình có điểm cao nhất, ta có xu hướng chọn đúng cấu hình gặp nhiễu dương, nên điểm Val của nó là ước lượng thiên cao.
     - Test không tham gia lựa chọn và mô hình chỉ dự đoán trên Test **một lần**, nên không chịu selection bias này. (Hạn chế: EDA đã xem thống kê mô tả của toàn bộ dữ liệu, xem commit 3.)
   - Nếu tuning dựa trên Test thì Test trở thành Validation thứ hai, và không còn ước lượng nào trung thực nữa.
6. **Kết quả thực tế:**
   - DT tốt nhất: entropy, depth 12, `min_samples_leaf=10`, `min_samples_split=2`, `class_weight=balanced`. Val F1 = 0,6796 (Train F1 = 0,7094).
   - CNB tốt nhất: `n_bins=20`, `alpha=0,1`, `fit_prior=True`. Val F1 = 0,6733.
   - Độ nhạy: CNB gần như không nhạy với α (F1 dao động 0,6711–0,6733). `fit_prior=False` làm F1 giảm còn 0,6542. Với DT, `class_weight=balanced` cho F1 tốt nhất 0,6796, so với 0,6738 khi không dùng trọng số.
   - Validation (fit trên Train):

     | | Acc | P | R | F1 | ROC-AUC | AP |
     |---|---|---|---|---|---|---|
     | DT | 0,8073 | 0,5667 | 0,8486 | 0,6796 | 0,8927 | 0,7623 |
     | CNB | 0,8172 | 0,5909 | 0,7823 | 0,6733 | 0,9004 | 0,7584 |
     | GNB (phụ) | 0,7920 | 0,6562 | 0,2857 | 0,3981 | 0,7517 | 0,5515 |
7. **Luồng xử lý:** `get_splits()` → bỏ tập Test → với mỗi cấu hình: `fit(Train)` → `predict(Val)` → metric → CSV → xếp hạng theo F1 (khi bằng điểm thì xét AP, rồi đến số lá) → `best_configs.json`.
8. **Quyết định thiết kế:**
   - Chọn theo **F1 lớp `>50K`** vì dữ liệu mất cân bằng và mục tiêu là phát hiện lớp thiểu số. Accuracy sẽ ưu tiên mô hình "lười", ít khi đoán `>50K`.
   - Ghi lại **mọi** cấu hình đã thử, không chỉ cấu hình thắng, để có thể kiểm tra lại.
   - Chỉ dùng một tập Validation cố định, không dùng cross-validation. Cách này đơn giản và rẻ, nhưng chịu dao động do cách chia. Đây là một hạn chế.
9. **Lỗi thường gặp:**
   - Thấy kết quả Test rồi quay lại sửa lưới tham số.
   - Báo cáo F1 trên Validation như thể đó là hiệu năng cuối cùng.
   - Cấu hình tốt nhất nằm ở biên lưới mà không mở rộng lưới.
10. **Câu hỏi:** "Tại sao không chọn cấu hình theo Accuracy?"
11. **Trả lời:** "Vì 76% dữ liệu là `<=50K`, Accuracy sẽ ưu tiên mô hình ít đoán `>50K`. Baseline luôn đoán `<=50K` đã đạt 0,76. F1 của lớp `>50K` buộc mô hình phải vừa tìm được người `>50K` (Recall), vừa đoán đúng khi nói `>50K` (Precision)."
12. **Bài tập:** mở `DecisionTree_validation_grid.csv`, nhóm theo `param_max_depth` và lấy max của F1 trên Val. Vẽ lại đường cong và so sánh với Hình `dt_f1_vs_depth.png`. Chỉ đọc file CSV, không chạy lại tuning.
13. **Điều kiện chuyển tiếp:** giải thích được vì sao điểm Validation của cấu hình thắng thường hơi lạc quan, và vì sao không được dùng Test để tuning.

---

## COMMIT 9 – `46c8f16` feat: add reproducible final model evaluation

1. **Hash / message:** `46c8f16`, `feat: add reproducible final model evaluation`.
2. **Mục tiêu:** đánh giá cuối cùng, trung thực, trên tập Test, kèm định lượng độ bất định.
3. **Vấn đề cần giải quyết:**
   - Bảo đảm mô hình chỉ dự đoán trên Test một lần và Test không tham gia lựa chọn nào.
   - Cho biết chênh lệch giữa hai mô hình là thật hay chỉ do may rủi của mẫu Test.
4. **File quan trọng:**
   - `src/final_evaluation.py`:
     - `LOCKED_CONFIGS` lấy cấu hình từ `best_configs.json`.
     - Script **tự dừng** nếu `final_predictions.csv` đã tồn tại.
     - Fit trên Train+Val (27.677 mẫu).
     - Gọi `positive_proba(model, X_test)` **đúng một lần** cho mỗi mô hình.
     - Chỉ đọc nhãn Test **sau khi** đã lưu dự đoán.
   - `results/metrics/final_predictions.csv` (4.884 dòng), `final_test_metrics.csv`, `bootstrap_confidence_intervals.csv`, `paired_model_comparison.csv`, `validation_vs_test.csv`, `final_run_info.json`.
   - `results/models/final/*.joblib`, `results/figures/final/*`.
   - `results/logs/phase5_6_*`: gồm cả log của **lần chạy lỗi** (lỗi xảy ra trước khi chạm vào Test), được giữ lại để minh bạch.
5. **Lý thuyết:**
   - **Fit cuối trên Train+Val:** cấu hình đã chốt, nên dùng thêm dữ liệu Validation để mô hình học tốt hơn. Toàn bộ pipeline, kể cả preprocessing, được fit lại.
   - **Bootstrap CI:** lấy mẫu lại có hoàn lại từ Test 5.000 lần (seed 42). Mỗi lần tính lại metric. CI 95% là khoảng từ phân vị 2,5% đến phân vị 97,5% (phương pháp percentile).
     - **Phân tầng:** mỗi mẫu bootstrap giữ đúng 3.708 mẫu âm và 1.176 mẫu dương.
     - **Ghép cặp:** trong mỗi lần lặp, hai mô hình được chấm trên **cùng** các chỉ số mẫu, rồi lấy hiệu DT − CNB. Nhờ vậy phần dao động chung (các mẫu "khó" với cả hai mô hình) bị triệt tiêu.
   - **McNemar:** chỉ xét các mẫu mà hai mô hình bất đồng:
     - b = số mẫu DT đúng, CNB sai;
     - c = số mẫu DT sai, CNB đúng.

     Dưới H₀ (hai mô hình có tỉ lệ lỗi như nhau), min(b, c) ~ Binomial(b+c; 0,5). Dự án dùng phiên bản exact binomial.
6. **Kết quả Test (đã đóng băng):**

   | | Acc | P | R | F1 | ROC-AUC | AP | TN/FP/FN/TP |
   |---|---|---|---|---|---|---|---|
   | DT | **0,8022** | 0,5576 | **0,8639** | **0,6778** | 0,8984 | 0,7715 | 2902/806/160/1016 |
   | CNB | **0,8176** | 0,5907 | **0,7891** | **0,6756** | 0,9011 | 0,7665 | 3065/643/248/928 |

   - **Chênh lệch DT − CNB (CI 95%):**
     - F1: +0,0021 [−0,0133; +0,0175].
     - ROC-AUC: −0,0027 [−0,0102; +0,0046].
     - AP: +0,0050 [−0,0093; +0,0192].
     - **Cả ba khoảng đều chứa 0.**
   - **McNemar:** cả hai cùng đúng 3.654 mẫu, b = 264, c = 339, cả hai cùng sai 627 mẫu. Exact p = 0,0025538. Như vậy tỉ lệ dự đoán đúng (Accuracy) của CNB cao hơn DT tại ngưỡng 0,5 và khác biệt có ý nghĩa thống kê. Đây là phân tích **bổ sung**; không được biến nó thành tiêu chí chọn mô hình sau khi đã xem kết quả, vì tiêu chí đã định trước là F1.
   - **Validation và Test:** mọi chênh lệch đều nhỏ hơn 0,01. Ví dụ F1 của DT: 0,6796 (Val) và 0,6778 (Test). Đây là kết quả trên **một lần chia dữ liệu**, chưa phải bằng chứng tổng quát hóa trên nhiều lần chia.
7. **Luồng xử lý:** `best_configs.json` → build → fit(Train+Val) → kiểm tra tái lập trên Train+Val → `predict_proba(Test)` một lần → lưu CSV → đọc lại CSV → metric, bootstrap, McNemar, hình.
8. **Cách diễn giải trung thực (phần quan trọng nhất khi bảo vệ):**
   - **CI chứa 0 ≠ hai mô hình tương đương.** CI chứa 0 chỉ có nghĩa là với 4.884 mẫu Test, ta **không đủ bằng chứng** để phân biệt hai mô hình. Dữ liệu tương thích với chênh lệch F1 âm, bằng 0 hoặc dương trong khoảng từ −0,013 đến +0,018. Cũng không được nói "chắc chắn hai mô hình chỉ khác nhau do ngẫu nhiên". Muốn kết luận "tương đương" cần một phép kiểm định khác (kiểm định tương đương với ngưỡng định trước), và dự án không làm kiểm định đó.
   - **McNemar không mâu thuẫn với bootstrap.** Hai phép kiểm định đo hai thứ khác nhau:
     - McNemar đo **Accuracy tại ngưỡng 0,5**. CNB đúng nhiều hơn chủ yếu vì có ít FP hơn (643 so với 806).
     - Bootstrap đo **F1, ROC-AUC, AP**.

     DT dùng `class_weight=balanced` nên chọn điểm làm việc thiên về Recall. Đây là **khác biệt về điểm làm việc** (sự đánh đổi Precision–Recall), không phải bằng chứng CNB tốt hơn về mọi mặt. Ngoài ra McNemar là phân tích **bổ sung**, không nằm trong tiêu chí so sánh đã định trước.
   - **CI chỉ phản ánh dao động do mẫu Test.** CI được tính trên **dự đoán Test cố định** (đọc từ `final_predictions.csv`); mô hình không được huấn luyện lại khi bootstrap. CI không bao gồm dao động do dữ liệu huấn luyện, cách chia tập hay việc chọn siêu tham số.
   - **AP khác diện tích PR tính bằng hình thang:**
     - DT: AP = 0,7715, hình thang = 0,7840.
     - CNB: AP = 0,7665, hình thang = 0,7664.

     DT chỉ có 161 giá trị xác suất khác nhau (mỗi lá một giá trị), nên phép nội suy tuyến tính của cách hình thang cho kết quả lạc quan. Dự án giữ AP vì đó là metric đã định trước; đổi metric sau khi đã xem Test là sai phương pháp.
9. **Lỗi thường gặp:**
   - Chạy đánh giá Test nhiều lần, mỗi lần chỉnh một chút.
   - Bootstrap riêng từng mô hình rồi so xem hai CI có chồng nhau không. Cách này không chính xác bằng bootstrap ghép cặp trên hiệu số.
   - Kết luận "p > 0,05 nên hai mô hình bằng nhau".
10. **Câu hỏi:** "F1 của DT cao hơn, sao em không kết luận DT tốt hơn?"
11. **Trả lời:** "Vì chênh lệch chỉ 0,0021, và CI 95% ghép cặp là [−0,0133; +0,0175], chứa 0. Với cỡ mẫu Test hiện tại, chênh lệch này nằm trong mức dao động ngẫu nhiên. Em cũng không kết luận hai mô hình tương đương. Điều em khẳng định được là hai mô hình có cách đánh đổi khác nhau: DT có Recall cao hơn, CNB có Precision và Accuracy cao hơn."
12. **Bài tập:** từ `final_predictions.csv`, tính lại b và c (DT đúng/CNB sai và ngược lại), sau đó tính `scipy.stats.binomtest(264, 603, 0.5).pvalue`. Chỉ đọc file đã lưu, không gọi mô hình.
13. **Điều kiện chuyển tiếp:** tự giải thích được sự khác nhau giữa "CI chứa 0" và "tương đương", và vì sao kết quả McNemar không mâu thuẫn với kết quả bootstrap.

---

## COMMIT 10 – `ebd9b6c` test: add final experiment integrity audit

1. **Hash / message:** `ebd9b6c`, `test: add final experiment integrity audit`.
2. **Mục tiêu:** kiểm tra độc lập, ở chế độ chỉ đọc, rằng mọi kết quả là đúng và tái lập được.
3. **Vấn đề cần giải quyết:** làm sao chứng minh với người khác rằng: Test không bị dùng sai, số liệu trong báo cáo khớp với dự đoán đã lưu, và các file không bị sửa?
4. **File quan trọng:**
   - `scripts/final_audit.py`: không load `.joblib`, không fit, không predict.
   - `results/reports/final_audit/`:
     - `FINAL_AUDIT_REPORT.md`: kết luận READY WITH CAVEATS.
     - `duplicate_audit.csv`, `mcnemar_audit.md`, `bootstrap_audit.md`, `integrity_check.md`.
     - `audit_manifest.json`: sha256 của 17 artifact.
5. **Lý thuyết:**
   - **Tính tái lập:** tính lại metric từ dự đoán đã lưu, và cài đặt lại bootstrap cho ra cùng CI (sai lệch ≤ 2·10⁻¹⁸).
   - **Hàm băm SHA256:** chỉ cần thay đổi 1 byte trong file là mã băm thay đổi hoàn toàn. Vì vậy so sánh mã băm giúp kiểm tra **nội dung file hiện tại có khớp với hash tham chiếu hay không**. Hash tự nó không chứng minh file chưa từng bị sửa hay chứng minh nguồn gốc file: giá trị chứng minh phụ thuộc vào độ tin cậy của hash tham chiếu (ở đây là `audit_manifest.json`, cũng nằm trong cùng repo) và quy trình lưu giữ nó (ví dụ lịch sử Git).
6. **Các phát hiện (61 PASS, 4 WARNING, 0 FAIL):**
   - **Hai định nghĩa trùng lặp, hai con số:** theo **15 cột (gồm nhãn)**, dùng khi chia tập, có **23 nhóm**; theo **14 đặc trưng (không gồm nhãn)**, dùng trong audit, có **24 nhóm** (49 dòng). Chênh 1 nhóm chính là cặp cùng đặc trưng nhưng khác nhãn: theo 15 cột hai dòng này không được coi là trùng. Kết quả audit: **không có nhóm nào** (theo 14 đặc trưng) nằm ở hai tập khác nhau. Điều này cho thấy không có rò rỉ do bản ghi trùng **trong lần chia hiện tại**, nhưng không loại bỏ mọi khả năng rò rỉ khác (ví dụ bản ghi gần trùng, hoặc EDA trên toàn bộ dữ liệu). Có 1 cặp dòng cùng đặc trưng nhưng khác nhãn (dòng 4963 và 24061), cả hai đều ở Train. Đây là nhiễu nhãn, không đủ căn cứ để khẳng định là lỗi gán nhãn.
   - W1: cách gom nhóm theo 15 cột (có tính nhãn) **về thiết kế** không bảo đảm cặp dòng mâu thuẫn nhãn sẽ nằm cùng một tập. Lần chia hiện tại không gặp vấn đề này.
   - W2: McNemar là phân tích bổ sung. W3: tên "PR-AUC" thực chất là AP. W4: CI chỉ phản ánh dao động do mẫu Test.
7. **Luồng xử lý:** đọc file → tính lại → so sánh với file kết quả → băm trước và sau → viết báo cáo.
8. **Quyết định thiết kế:**
   - Audit chỉ đọc. Không sửa kết quả ngay cả khi phát hiện vấn đề, chỉ ghi nhận thành WARNING.
   - Bốn agent kiểm tra độc lập tính lại mà không được biết trước đáp án, và kết quả khớp với script chính. Đây là kiểm tra tính toán lại (re-computation) trên cùng dữ liệu và artifact, không phải phản biện khoa học độc lập hay bằng chứng rằng thiết kế thí nghiệm là đúng.
9. **Lỗi thường gặp:**
   - "Sửa cho đẹp" kết quả sau khi audit.
   - Coi WARNING là FAIL, hoặc ngược lại bỏ qua WARNING.
10. **Câu hỏi:** "Dữ liệu có bản ghi trùng, em có chắc không bị rò rỉ không?"
11. **Trả lời:** "Có. Em chia tập theo nhóm trùng lặp bằng StratifiedGroupKFold, và audit kiểm tra lại theo 14 đặc trưng (không tính nhãn): 24 nhóm trùng, không nhóm nào nằm ở hai tập khác nhau. Có một hạn chế về thiết kế là gom nhóm theo 15 cột. Em đã ghi nhận hạn chế này, và kiểm chứng rằng ở lần chia hiện tại nó không gây ra rò rỉ."
12. **Bài tập:** tính sha256 của `data/raw/adult.csv` (`sha256sum` hoặc `hashlib`) và so với giá trị trong `integrity_check.md`.
13. **Điều kiện chuyển tiếp:** giải thích được 4 WARNING và vì sao không cần huấn luyện lại.

---

## COMMIT 11 – `9ee0847` docs: add final machine learning course report

1. **Hash / message:** `9ee0847`, `docs: add final machine learning course report`.
2. **Mục tiêu:** trình bày toàn bộ nghiên cứu một cách học thuật và trung thực.
3. **Vấn đề cần giải quyết:** mọi số liệu phải truy được về file nguồn; không được phóng đại kết luận.
4. **File quan trọng:**
   - `BAO_CAO_CUOI_KY_ML.md` và `.docx`: báo cáo gồm 5 chương, 14 hình.
   - `REFERENCES.md`: tài liệu tham khảo, có cột trạng thái xác minh.
   - `FIGURE_INDEX.md`: danh mục hình.
   - `REPORT_AUDIT.md`: đối chiếu từng số liệu với file nguồn, kèm kết quả hash 17/17.
   - `build_docx.py`: tạo file docx từ md.
5. **Lý thuyết:** cách viết khoa học. Phân biệt rõ "kết quả đo được" với "diễn giải"; nêu rõ hạn chế (dữ liệu Mỹ năm 1994, fairness, chỉ một lần chia tập); không suy luận nhân quả.
6. **Kỹ thuật:** dùng python-docx để sinh file Word, chèn hình có sẵn, không vẽ hình mới.
7. **Luồng xử lý:** các file kết quả → viết md → audit số liệu → docx.
8. **Quyết định thiết kế:** giữ nguyên các placeholder `[CẦN BỔ SUNG]` và `[CẦN XÁC MINH NGUỒN]` thay vì tự điền thông tin.
9. **Lỗi thường gặp:** viết "mô hình X tốt hơn" khi CI chứa 0; dùng tài liệu tham khảo không xác minh được.
10. **Câu hỏi:** "Hạn chế lớn nhất của nghiên cứu là gì?"
11. **Trả lời:** "Thứ nhất, chỉ có một lần chia dữ liệu, nên kết quả chưa phản ánh dao động do cách chia. Hướng khắc phục là dùng repeated group-aware cross-validation. Thứ hai, dữ liệu là của Mỹ năm 1994 và chứa thuộc tính nhạy cảm, nên mô hình không được dùng cho các quyết định thực tế như tuyển dụng hay cho vay."
12. **Bài tập:** chọn 5 con số bất kỳ trong báo cáo và tự tìm file nguồn của chúng bằng `REPORT_AUDIT.md`.
13. **Điều kiện chuyển tiếp:** trình bày được mục Kết luận bằng lời của mình.

## COMMIT 12 – `1145e12` docs: add git commit learning guide
Commit này thêm file `GIT_COMMIT_GUIDE.md`: bản đồ từ commit sang giai đoạn ML, kèm lệnh checkout. Đây là phần học về **quy trình làm việc với Git**, không có nội dung ML mới. Lưu ý quan trọng: lịch sử Git phân nhóm file theo giai đoạn, không phản ánh thời điểm thật mỗi file được viết.

---

# PHẦN D. CHUẨN BỊ BẢO VỆ

## A. Bộ 30 câu hỏi bảo vệ

### A1. Lý thuyết Machine Learning (10 câu)

1. **Học có giám sát là gì? Bài toán của em thuộc loại nào?**
   *Trả lời:* học hàm f(x) → y từ các cặp (x, y) có nhãn. Bài toán của em là phân loại nhị phân: `>50K` (1) và `<=50K` (0).
   *Vì sao đúng:* dữ liệu có nhãn `income`, và nhãn chỉ có hai giá trị.

2. **Entropy và Information Gain là gì?**
   *Trả lời:* entropy đo độ hỗn tạp, H = −Σ p log₂ p. IG là phần entropy giảm đi sau khi chia. Cây chọn phép chia có IG lớn nhất.
   *Vì sao đúng:* đây là định nghĩa của Quinlan (1986). `criterion="entropy"` trong sklearn tối ưu đúng đại lượng này.

3. **Gini khác Entropy thế nào?**
   *Trả lời:* Gini = 1 − Σ p². Cả hai đều bằng 0 khi nút thuần nhất. Gini không dùng logarit nên tính nhanh hơn. Thực tế hai tiêu chí cho kết quả gần nhau (trong dự án, F1 tốt nhất là 0,6786 với gini và 0,6796 với entropy).
   *Vì sao đúng:* hai hàm có hình dạng tương tự và cùng đạt cực đại tại phân phối đều.

4. **Overfitting là gì? Làm sao phát hiện?**
   *Trả lời:* mô hình khớp cả nhiễu của Train, nên tổng quát hóa kém. Phát hiện bằng khoảng cách giữa điểm Train và điểm Validation. Ví dụ: cây không giới hạn có F1 trên Train 0,9999 nhưng trên Val khoảng 0,60.
   *Vì sao đúng:* hiệu năng trên dữ liệu chưa thấy mới là mục tiêu thật.

5. **Phát biểu định lý Bayes và giả định "naive".**
   *Trả lời:* P(y|x) ∝ P(y)·P(x|y). "Naive" nghĩa là giả định các đặc trưng độc lập khi biết lớp, tức P(x|y) = Π P(x_j|y).
   *Vì sao đúng:* nhờ giả định này, mô hình chỉ cần ước lượng các phân phối một chiều thay vì một phân phối kết hợp khổng lồ.

6. **Laplace smoothing để làm gì?**
   *Trả lời:* cộng α vào mọi số đếm để không có xác suất nào bằng 0. Nếu một xác suất bằng 0 thì cả tích bằng 0, và một đặc trưng duy nhất "phủ quyết" toàn bộ các đặc trưng còn lại.
   *Vì sao đúng:* công thức (N + α) / (N_y + αK) luôn dương khi α > 0.

7. **Vì sao Accuracy không phù hợp khi dữ liệu mất cân bằng?**
   *Trả lời:* mô hình luôn đoán lớp đa số vẫn đạt Accuracy 0,7592 nhưng F1 = 0.
   *Vì sao đúng:* Accuracy bị lớp đa số chi phối.

8. **ROC-AUC và Average Precision khác nhau thế nào?**
   *Trả lời:* ROC-AUC xét TPR theo FPR; vì mỗi trục được chuẩn hóa theo số mẫu của chính lớp đó, ROC-AUC ít chịu ảnh hưởng trực tiếp của tỉ lệ lớp. AP tóm tắt đường Precision–Recall, nhạy hơn với lớp dương hiếm; giá trị của mô hình ngẫu nhiên bằng tỉ lệ lớp dương (0,24).
   *Vì sao đúng:* Precision phụ thuộc số FP, mà số FP tăng theo số mẫu âm; FPR thì đã chuẩn hóa theo số mẫu âm.

9. **Bootstrap là gì?**
   *Trả lời:* lấy mẫu lại có hoàn lại từ dữ liệu đang có để mô phỏng phân phối của một thống kê, rồi lấy phân vị 2,5% và 97,5% làm CI 95%.
   *Vì sao đúng:* đây là phương pháp của Efron (1979). Mẫu bootstrap đóng vai trò xấp xỉ việc lấy mẫu mới từ tổng thể.

10. **Kiểm định McNemar kiểm định điều gì?**
    *Trả lời:* kiểm định xem hai bộ phân loại có tỉ lệ lỗi như nhau trên cùng các mẫu hay không, chỉ dựa vào các mẫu hai mô hình bất đồng (b, c).
    *Vì sao đúng:* các mẫu cả hai cùng đúng hoặc cùng sai không chứa thông tin về sự khác biệt giữa hai mô hình.

### A2. Triển khai dự án (10 câu)

11. **Em chia dữ liệu thế nào?**
    *Trả lời:* dùng StratifiedGroupKFold với 20 fold, seed 42. Fold 0–2 là Test, 3–5 là Val, còn lại là Train, cho ra 22.793 / 4.884 / 4.884 mẫu. Tỉ lệ `>50K` bằng 0,2408 ở cả ba tập.
    *Vì sao đúng:* có trong `data_loader.split_data` và `split_summary.csv`.

12. **Tại sao dùng StratifiedGroupKFold thay vì train_test_split?**
    *Trả lời:* để vừa giữ tỉ lệ lớp, vừa giữ các bản ghi trùng nhau trong cùng một tập, tránh mô hình "thấy trước" bản sao của mẫu Test.
    *Vì sao đúng:* `train_test_split` không biết khái niệm nhóm. File `split_old_vs_new.csv` lưu lại so sánh giữa hai cách chia.

13. **Xử lý giá trị thiếu thế nào?**
    *Trả lời:* đổi `?` thành NaN, rồi impute thành hạng mục "Unknown". Không xóa dòng nào.
    *Vì sao đúng:* trong dữ liệu quan sát, nhóm có `?` có tỉ lệ `>50K` khác (13,9% so với 24,9%), nên việc thiếu có liên hệ với nhãn. Giữ thành hạng mục `Unknown` là **lựa chọn thiết kế** để mô hình có thể học thông tin này từ Train; đây không phải bằng chứng xác định cơ chế thiếu (chưa đủ để nói là MCAR, MAR hay MNAR).

14. **DT và CNB tiền xử lý khác nhau ra sao?**
    *Trả lời:* DT giữ nguyên thuộc tính số và One-Hot thuộc tính phân loại (92 đặc trưng). CNB rời rạc hóa thuộc tính số theo phân vị và dùng Ordinal cho thuộc tính phân loại (13 đặc trưng).
    *Vì sao đúng:* cây chia theo ngưỡng nên dùng được số thực; CategoricalNB chỉ nhận mã rời rạc.

15. **ZeroInflatedDiscretizer giải quyết vấn đề gì?**
    *Trả lời:* hơn 90% giá trị capital bằng 0, nên rời rạc hóa phân vị thông thường gộp cả cột thành 1 khoảng. Bộ rời rạc hóa này tách riêng "= 0" và chia phần giá trị dương theo phân vị.
    *Vì sao đúng:* nếu cả cột chỉ có một khoảng thì đặc trưng không còn mang thông tin gì.

16. **Làm sao tránh rò rỉ dữ liệu khi tiền xử lý?**
    *Trả lời:* đặt mọi bước có học tham số trong `Pipeline`, và chỉ gọi `fit` trên Train (hoặc trên Train+Val khi refit).
    *Vì sao đúng:* như vậy biên phân vị và danh sách hạng mục không bao giờ được học từ Test.

17. **Xử lý mất cân bằng lớp thế nào? Sao không dùng SMOTE?**
    *Trả lời:* DT dùng `class_weight`, NB dùng `fit_prior`, cả hai đều được tuning. Không dùng SMOTE vì SMOTE nội suy trong không gian số, không tự nhiên với dữ liệu nhiều thuộc tính phân loại, dễ gây rò rỉ nếu áp dụng sai chỗ, và em muốn giữ phép so sánh "thuần" giữa hai thuật toán.
    *Vì sao đúng:* đây là lựa chọn thiết kế đã ghi trong báo cáo, không phải kết luận rằng SMOTE kém.

18. **Tiêu chí chọn siêu tham số là gì?**
    *Trả lời:* F1 của lớp `>50K` trên Validation, với ngưỡng 0,5. Khi bằng điểm thì xét AP, rồi đến số lá.
    *Vì sao đúng:* có trong `evaluation.py` (`SELECTION_METRIC = "f1"`) và `tuning.py`.

19. **Vì sao refit trên Train+Val trước khi đánh giá Test?**
    *Trả lời:* cấu hình đã chốt, nên dùng thêm 4.884 mẫu giúp mô hình học tốt hơn. Test vẫn chưa được dùng cho dự đoán hay lựa chọn nào (chỉ thống kê mô tả toàn bộ dữ liệu đã được xem trong EDA).
    *Vì sao đúng:* refit không dùng thông tin gì từ Test.

20. **Làm sao chứng minh Test chỉ dùng một lần?**
    *Trả lời:* cần nói chính xác: mô hình chỉ dự đoán trên Test một lần và Test không dùng cho lựa chọn nào; còn thống kê mô tả của toàn bộ dữ liệu (gồm phần Test) đã được xem trong EDA trước khi chia. Bằng chứng cho phần dự đoán: script tự dừng nếu `final_predictions.csv` đã tồn tại; trong code chỉ có một lời gọi `positive_proba(m, Xte)`; log ghi lại cả lần chạy lỗi (lỗi xảy ra trước khi chạm vào Test); audit đã xác nhận các điểm này.
    *Vì sao đúng:* có trong `integrity_check.md`, mục 5.

### A3. Phân tích kết quả và phản biện (10 câu)

21. **Mô hình nào tốt hơn?**
    *Trả lời:* theo tiêu chí đã định trước thì chưa đủ bằng chứng để kết luận. F1 0,6778 so với 0,6756; CI của chênh lệch là [−0,0133; +0,0175], chứa 0. Hai mô hình đánh đổi khác nhau.
    *Vì sao đúng:* CI chứa 0 nghĩa là dữ liệu Test tương thích với chênh lệch âm, bằng 0 hoặc dương; không đủ để khẳng định chiều của chênh lệch.

22. **Vậy hai mô hình tương đương?**
    *Trả lời:* không. "Không phát hiện được khác biệt" khác với "chứng minh không có khác biệt". Muốn kết luận tương đương cần kiểm định tương đương với ngưỡng định trước.
    *Vì sao đúng:* thiếu bằng chứng cho một điều không phải là bằng chứng cho điều ngược lại.

23. **McNemar p = 0,0026, sao không nói CNB tốt hơn?**
    *Trả lời:* McNemar chỉ so sánh Accuracy tại ngưỡng 0,5, và là phân tích bổ sung. Accuracy của CNB cao hơn chủ yếu vì có ít FP hơn (643 so với 806). Đây là khác biệt về điểm làm việc. Trên F1, ROC-AUC và AP, chưa có bằng chứng nào về khác biệt.
    *Vì sao đúng:* hai phép kiểm định đo hai đại lượng khác nhau.

24. **Vì sao DT có Recall cao hơn?**
    *Trả lời:* `class_weight=balanced` làm mỗi mẫu `>50K` nặng hơn khoảng 3,15 lần mẫu `<=50K`, nên xác suất trong lá bị đẩy về phía lớp 1, và ở ngưỡng 0,5 cây đoán `>50K` nhiều hơn. Kết quả: Recall 0,8639, nhưng có 806 FP.
    *Vì sao đúng:* nhất quán với CM và với cơ chế trọng số. Lưu ý: đây là giải thích dựa trên cơ chế, chưa được tách riêng bằng thí nghiệm trên Test; trong tuning, cấu hình không trọng số tốt nhất có F1 Val 0,6738 so với 0,6796.

25. **Vì sao CNB có Accuracy cao hơn?**
    *Trả lời:* **Quan sát được:** tại ngưỡng 0,5, CNB có ít FP hơn DT (643 so với 806) nhưng nhiều FN hơn (248 so với 160), cho Precision 0,5907 và Accuracy 0,8176. **Giả thuyết giải thích:** CNB dùng prior theo tần suất lớp khi `fit_prior=True` (khoảng 76/24), có thể góp phần làm mô hình dè dặt hơn khi đoán `>50K`; còn DT dùng `class_weight=balanced` nên thiên về Recall. Tuy nhiên không nên quy toàn bộ sự khác biệt cho prior, vì hai mô hình còn khác nhau ở cấu trúc, cách biểu diễn đặc trưng và cách ước lượng xác suất.
    *Vì sao đúng:* phân biệt số liệu đo được (CM) với nguyên nhân chưa được kiểm chứng riêng. Bằng chứng gián tiếp: trong tuning, `fit_prior=True` cho F1 Val 0,6733, cao hơn `fit_prior=False` (0,6542), nhưng thí nghiệm đó không tách riêng tác động lên FP của mô hình cuối.

26. **Có dấu hiệu overfitting Validation không?**
    *Trả lời:* trên lần chia dữ liệu hiện tại, hiệu năng Validation và Test tương đối nhất quán: mọi chênh lệch đều nhỏ hơn 0,01 (F1 của DT 0,6796 → 0,6778; F1 của CNB 0,6733 → 0,6756). Tuy nhiên điều này không chứng minh mô hình không overfit hoặc có khả năng tổng quát hóa ổn định trên mọi tập dữ liệu. Muốn đánh giá chắc chắn hơn cần repeated group-aware cross-validation.
    *Vì sao đúng:* số liệu có trong `validation_vs_test.csv`. Với một lần chia, Val và Test mỗi tập chỉ là một mẫu duy nhất; sự nhất quán có thể một phần do cách chia, và ta không đo được độ dao động giữa các lần chia. Ngoài ra DT vẫn có khoảng cách Train–Val (F1 Train 0,7094 so với Val 0,6796).

27. **"PR-AUC" của em là gì?**
    *Trả lời:* là Average Precision, tính bằng `average_precision_score`. Nếu tính bằng phương pháp hình thang thì DT đạt 0,7840 thay vì 0,7715, vì cây chỉ có 161 giá trị xác suất khác nhau. Em giữ AP vì đó là metric đã định trước.
    *Vì sao đúng:* có trong `bootstrap_audit.md`, mục 3.

28. **Khoảng tin cậy bootstrap của em có hạn chế gì?**
    *Trả lời:* CI chỉ phản ánh dao động do mẫu Test. Mô hình được giữ cố định, nên CI không tính đến biến động do dữ liệu huấn luyện, cách chia tập hay việc chọn siêu tham số.
    *Vì sao đúng:* trong bootstrap không có bước nào fit lại mô hình.

29. **Kết quả có dùng được để xét tuyển dụng hay cho vay không?**
    *Trả lời:* không. Dữ liệu là của Mỹ năm 1994, có các thuộc tính nhạy cảm (sex, race), và relationship liên hệ mạnh với sex (V = 0,649). Em chưa phân tích fairness, và mô hình chỉ học tương quan chứ không học quan hệ nhân quả.
    *Vì sao đúng:* có trong mục Hạn chế của báo cáo.

30. **Nếu có thêm thời gian, em sẽ cải thiện gì?**
    *Trả lời:* trong tương lai, lặp lại việc chia theo nhóm có phân tầng nhiều lần (ví dụ chạy `StratifiedGroupKFold` với nhiều seed khác nhau, vì scikit-learn không có sẵn lớp "repeated" cho trường hợp này) để đo dao động do cách chia tập; làm EDA chỉ trên Train; gom nhóm theo 14 đặc trưng; phân tích fairness theo sex và race; chọn ngưỡng trên Validation theo mục tiêu ứng dụng; phân tích feature importance.
    *Vì sao đúng:* mỗi hướng tương ứng với một hạn chế đã được ghi nhận.

## B. Dàn ý thuyết trình 5–7 phút

| Phút | Phần | Nội dung chính | Hình hoặc bảng gợi ý |
|---|---|---|---|
| 0:00–0:45 | 1. Bài toán | Dự đoán thu nhập `>50K` (phân loại nhị phân). So sánh hai thuật toán đã học: DT và CNB, không lai ghép. | – |
| 0:45–1:45 | 2. Dataset & EDA | 32.561 mẫu, 24% `>50K` (mất cân bằng). Có `?` ở 3 cột và việc thiếu có liên quan đến nhãn. capital lệch mạnh về 0. relationship và sex liên hệ mạnh. | Hình 3.1, 3.5 |
| 1:45–2:45 | 3. Preprocessing | StratifiedGroupKFold 70/15/15 theo nhóm trùng. Pipeline riêng: One-Hot cho DT; quantile binning, xử lý riêng giá trị 0 của capital và Ordinal cho CNB. Fit trên Train. | Bảng 3.2, Hình 3.7 |
| 2:45–3:30 | 4. Hai thuật toán | DT: entropy/IG, kiểm soát độ sâu, `class_weight`. CNB: Bayes, giả định độc lập, α, prior. | Hình 4.2 (overfitting) |
| 3:30–4:15 | 5. Thiết kế thực nghiệm | Grid search 800 + 126 cấu hình trên Val, chọn theo F1 lớp `>50K`, ngưỡng 0,5. Refit trên Train+Val, predict Test một lần (Test không dùng để chọn). Bootstrap 5.000 lần, McNemar. | Sơ đồ luồng |
| 4:15–5:45 | 6. Kết quả | F1 0,6778 so với 0,6756, CI chênh lệch chứa 0. DT có Recall 0,864; CNB có Accuracy 0,818. McNemar (bổ sung) chỉ nói về Accuracy. | Bảng 4.3, Hình 4.5, 5.1 |
| 5:45–6:45 | 7. Kết luận & hạn chế | Chưa đủ bằng chứng để nói mô hình nào tốt hơn; hai mô hình đánh đổi khác nhau. Hạn chế: một lần chia, EDA trên toàn bộ dữ liệu trước khi chia, dữ liệu 1994, fairness. Hướng phát triển. | – |

**Câu mở đầu gợi ý:** "Em không đi tìm mô hình có con số cao nhất. Em tìm cách so sánh hai thuật toán một cách công bằng và trung thực."

**Câu kết gợi ý:** "Hai mô hình có F1 gần như bằng nhau, và khác biệt không vượt quá mức dao động ngẫu nhiên. Điều có ý nghĩa thực tế là cách đánh đổi: DT tìm được nhiều người thu nhập cao hơn, còn CNB đoán chính xác hơn khi nói một người có thu nhập cao."

## C. Checklist tự đánh giá

Đánh dấu `x` vào cột tương ứng.

| Chủ đề | Chưa hiểu | Hiểu lý thuyết | Hiểu code | Giải thích không cần tài liệu |
|---|:-:|:-:|:-:|:-:|
| Môi trường, seed, tái lập | [ ] | [ ] | [ ] | [ ] |
| Đọc dữ liệu, `clean()`, mã hóa nhãn | [ ] | [ ] | [ ] | [ ] |
| Lý do bỏ `education` | [ ] | [ ] | [ ] | [ ] |
| Phân phối nhãn, mất cân bằng lớp | [ ] | [ ] | [ ] | [ ] |
| Missing values `?` và quan hệ với nhãn | [ ] | [ ] | [ ] | [ ] |
| Dữ liệu trùng lặp (23 nhóm theo 15 cột / 24 nhóm theo 14 đặc trưng) | [ ] | [ ] | [ ] | [ ] |
| Pearson / Cramér's V / Mutual Information | [ ] | [ ] | [ ] | [ ] |
| Vì sao EDA không dùng Test để chọn mô hình | [ ] | [ ] | [ ] | [ ] |
| Vai trò Train / Validation / Test | [ ] | [ ] | [ ] | [ ] |
| StratifiedGroupKFold | [ ] | [ ] | [ ] | [ ] |
| Data leakage và Pipeline | [ ] | [ ] | [ ] | [ ] |
| OneHotEncoder, `handle_unknown` | [ ] | [ ] | [ ] | [ ] |
| Quantile binning (KBinsDiscretizer) | [ ] | [ ] | [ ] | [ ] |
| ZeroInflatedDiscretizer | [ ] | [ ] | [ ] | [ ] |
| SafeOrdinalEncoder và `min_categories` | [ ] | [ ] | [ ] | [ ] |
| Entropy, Information Gain | [ ] | [ ] | [ ] | [ ] |
| Gini Impurity | [ ] | [ ] | [ ] | [ ] |
| `max_depth`, `min_samples_split`, `min_samples_leaf` | [ ] | [ ] | [ ] | [ ] |
| `class_weight="balanced"` | [ ] | [ ] | [ ] | [ ] |
| Overfitting của cây | [ ] | [ ] | [ ] | [ ] |
| Định lý Bayes, prior/posterior | [ ] | [ ] | [ ] | [ ] |
| Giả định độc lập và hạn chế | [ ] | [ ] | [ ] | [ ] |
| Alpha / Laplace smoothing | [ ] | [ ] | [ ] | [ ] |
| Vì sao rời rạc hóa cho CNB | [ ] | [ ] | [ ] | [ ] |
| Confusion Matrix, FP/FN trong bài toán | [ ] | [ ] | [ ] | [ ] |
| Accuracy, Precision, Recall, F1 | [ ] | [ ] | [ ] | [ ] |
| ROC-AUC và Average Precision | [ ] | [ ] | [ ] | [ ] |
| AP so với diện tích PR hình thang | [ ] | [ ] | [ ] | [ ] |
| Grid Search, tiêu chí F1 trên Validation | [ ] | [ ] | [ ] | [ ] |
| Vì sao không tuning trên Test | [ ] | [ ] | [ ] | [ ] |
| Refit trên Train+Val, predict Test một lần (EDA đã xem toàn bộ dữ liệu) | [ ] | [ ] | [ ] | [ ] |
| Bootstrap CI (phân tầng, ghép cặp, percentile) | [ ] | [ ] | [ ] | [ ] |
| CI chứa 0 ≠ tương đương | [ ] | [ ] | [ ] | [ ] |
| McNemar exact và giới hạn diễn giải | [ ] | [ ] | [ ] | [ ] |
| Sự đánh đổi Recall (DT) và Precision/Accuracy (CNB) | [ ] | [ ] | [ ] | [ ] |
| SHA256 (kiểm tra khớp hash tham chiếu) và audit chỉ đọc | [ ] | [ ] | [ ] | [ ] |
| Hạn chế, fairness, không suy luận nhân quả | [ ] | [ ] | [ ] | [ ] |
| Trình bày 5–7 phút trôi chảy | [ ] | [ ] | [ ] | [ ] |
