"""Sinh notebooks/02_Preprocessing.ipynb (PHASE 2-3: tiền xử lý + chia dữ liệu)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# 02 – Tiền xử lý và chia dữ liệu (PHASE 2–3)
Mọi bước dùng code trong `src/`; notebook chỉ gọi và hiển thị kết quả thực tế.
Không huấn luyện mô hình phân lớp trong notebook này.""")

md("## 0. Kiểm tra môi trường và đường dẫn dữ liệu")
code("""import sys, os, platform, hashlib, warnings
sys.path.insert(0, '../src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sklearn
from data_loader import *
from preprocessing import *
print(platform.platform(), '| Python', sys.version.split()[0], '| sklearn', sklearn.__version__,
      '| pandas', pd.__version__, '| CPUs', os.cpu_count())
print('Dataset:', RAW_PATH, '| tồn tại:', RAW_PATH.exists(),
      '| sha256:', hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()[:16])""")

md("""## 1. Làm sạch không học tham số
- Bỏ khoảng trắng đầu/cuối chuỗi; `'?'` → NaN; nhãn `<=50K` → 0, `>50K` → 1.
- Các phép này là quy tắc cố định, không phụ thuộc phân phối dữ liệu → áp dụng trước khi chia không gây rò rỉ.""")
code("""raw = load_raw(); df = clean(raw)
print(df.shape); print('NaN mỗi cột:', df.isna().sum()[lambda s: s > 0].to_dict())
print('Nhãn:', df[TARGET].value_counts().to_dict())
df.head()""")

md("""## 2. Nhóm trùng lặp theo nội dung (gồm cả nhãn)
Mỗi nội dung bản ghi đầy đủ là một nhóm; bản ghi duy nhất là nhóm 1 phần tử.""")
code("""groups = duplicate_group_ids(df); sizes = groups.value_counts()
print('Số nhóm:', groups.nunique(), '| nhóm >1 bản ghi:', (sizes > 1).sum(),
      '| bản ghi trong các nhóm đó:', sizes[sizes > 1].sum())
print('Phân bố kích thước nhóm:', sizes.value_counts().sort_index().to_dict())
print('Mọi nhóm có nhãn đồng nhất:', (df.groupby(groups)[TARGET].nunique() == 1).all())""")

md("""## 3. Chia Train / Validation / Test theo nhóm, có phân tầng (PHASE 3)
`StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)` → 20 fold (~5%/fold):
fold 0–2 → Test (15%), fold 3–5 → Validation (15%), fold 6–19 → Train (70%).
Mỗi nhóm trùng nằm trọn trong một fold, nên không bao giờ bị tách sang hai tập.""")
code("""splits = split_data(df)
summary = split_summary(splits)
summary['fraction_exact'] = (summary['n_samples'] / len(df)).round(5)
summary""")
code("""g = {k: set(groups.loc[v.index]) for k, v in splits.items()}
pairs = [('train','val'), ('train','test'), ('val','test')]
print('Nhóm dùng chung giữa các tập:', {f'{a}&{b}': len(g[a] & g[b]) for a, b in pairs})
print('Tổng bản ghi:', sum(len(v) for v in splits.values()))
pd.read_csv('../data/processed/split_old_vs_new.csv', index_col=0)""")
code("""import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3))
names = ['train', 'val', 'test']; s = summary.set_index('split').loc[names]
ax.bar(names, 100 * (1 - s['ratio_gt50K']), color='#2a78d6', label='<=50K', width=0.5)
ax.bar(names, 100 * s['ratio_gt50K'], bottom=100 * (1 - s['ratio_gt50K']), color='#eb6834',
       label='>50K', width=0.5, edgecolor='white', linewidth=2)
for i, (n, r) in enumerate(zip(s['n_samples'], s['ratio_gt50K'])):
    ax.annotate(f'n={n:,}\\n>50K {100*r:.2f}%', (i, 50), ha='center', va='center', color='white', fontsize=9)
ax.set_ylabel('%'); ax.set_title('Phân phối nhãn trong từng tập', fontweight='bold')
ax.legend(frameon=False, bbox_to_anchor=(1, 1)); ax.spines[['top', 'right']].set_visible(False)
os.makedirs('../results/figures/preprocessing', exist_ok=True)
fig.savefig('../results/figures/preprocessing/split_label_distribution.png', dpi=150, bbox_inches='tight')
plt.show()""")

md("""## 4. Pipeline tiền xử lý (PHASE 2) – chỉ fit trên Train
| Mô hình | Thuộc tính số | Thuộc tính phân loại |
|---|---|---|
| Decision Tree | giữ nguyên (cây bất biến với phép biến đổi đơn điệu) | `"Unknown"` → One-Hot (`handle_unknown='ignore'`) |
| CategoricalNB (chính) | KBinsDiscretizer quantile (age, fnlwgt, education.num, hours.per.week); ZeroInflatedDiscretizer (capital.gain/loss) | `"Unknown"` → mã nguyên; hạng mục chưa thấy → mã dành riêng |
| GaussianNB (phụ) | giữ nguyên 6 thuộc tính số | không dùng |

**Vì sao không dùng GaussianNB trên One-Hot cho thí nghiệm chính:** biến One-Hot chỉ nhận 0/1, không thể mô hình bằng phân phối Gaussian; GaussianNB sẽ ước lượng phương sai rất nhỏ cho các cột hiếm và cho mật độ cực lớn/nhỏ, làm xác suất méo mó. CategoricalNB ước lượng trực tiếp P(x_j = giá trị | lớp) bằng tần suất có làm trơn – đúng bản chất dữ liệu phân loại.""")
code("""Xtr, ytr = xy(splits['train']); Xv, yv = xy(splits['val']); Xte, yte = xy(splits['test'])
pre = {'DT': build_dt_preprocessor().fit(Xtr),
       'CNB': build_cnb_preprocessor().fit(Xtr),
       'GNB': build_gnb_preprocessor().fit(Xtr)}
pd.DataFrame({k: {s: p.transform(X).shape for s, X in [('train', Xtr), ('val', Xv), ('test', Xte)]}
              for k, p in pre.items()})""")
code("""cnb = pre['CNB']
kb = cnb.named_transformers_['num']; cap = cnb.named_transformers_['capital']
for f, e in zip(kb.feature_names_in_, kb.bin_edges_):
    print(f'{f:15s} biên khoảng (học từ Train):', np.round(e, 1).tolist())
for f, e in zip(CAPITAL_FEATURES, cap.edges_):
    print(f'{f:15s} 0 | các mốc phần dương (học từ Train):', e.tolist())
print('min_categories cho CategoricalNB:', cnb_min_categories(cnb).tolist())""")
code("""pd.DataFrame(cnb.transform(Xtr.head()), columns=SMOOTH_NUMERIC + CAPITAL_FEATURES + CATEGORICAL_FEATURES)""")
code("""names = pre['DT'].get_feature_names_out()
print(len(names), 'đặc trưng cho Decision Tree, ví dụ:', list(names[:8]), '...')""")

md("## 5. Kiểm tra chống rò rỉ")
code("""enc = cnb.named_transformers_['cat'].named_steps['ordinal'].enc_
for c, cats in zip(CATEGORICAL_FEATURES, enc.categories_):
    tr = set(Xtr[c].fillna('Unknown'))
    unseen = (set(Xv[c].fillna('Unknown')) | set(Xte[c].fillna('Unknown'))) - tr
    print(f'{c:15s} hạng mục học được == hạng mục của Train: {set(cats) == tr} | chỉ có ở Val/Test: {unseen or "-"}')
# Biên khoảng tính lại chỉ từ Train phải trùng với biên đã học
ref = build_cnb_preprocessor().fit(Xtr).named_transformers_['num'].bin_edges_
print('Biên khoảng tái lập từ Train:', all(np.allclose(a, b) for a, b in zip(ref, kb.bin_edges_)))
alt = build_cnb_preprocessor().fit(pd.concat([Xtr, Xv, Xte])).named_transformers_['num'].bin_edges_
print('Biên khoảng nếu (sai) fit trên toàn bộ dữ liệu có khác không:',
      any(len(a) != len(b) or not np.allclose(a, b) for a, b in zip(alt, kb.bin_edges_)))""")

md("## 6. Lưu kết quả chia dữ liệu")
code("""print(save_splits(splits).to_string(index=False))
idx = pd.read_csv('../data/processed/split_indices.csv')
print(idx.shape); idx[idx.dup_group_size > 1].sort_values('dup_group').head(10)""")
md("""**Tóm tắt.** Ba tập rời nhau theo cả chỉ số lẫn nội dung bản ghi, tổng 32.561, tỷ lệ ~70/15/15, tỷ lệ >50K ≈ 24,08% ở cả ba tập.
Pipeline được định nghĩa trong `src/preprocessing.py` và sẽ được fit lại bên trong từng thí nghiệm (Train khi chọn siêu tham số, Train+Validation khi đánh giá cuối).""")

nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/02_Preprocessing.ipynb")
print("Đã ghi notebooks/02_Preprocessing.ipynb")
