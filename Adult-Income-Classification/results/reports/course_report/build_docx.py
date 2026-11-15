"""Chuyển BAO_CAO_CUOI_KY_ML.md sang .docx (python-docx) và kiểm tra hash artifact.
Chỉ đọc kết quả thực nghiệm; không fit/predict; không sửa file cũ."""
import hashlib, json, re
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
md = (HERE / "BAO_CAO_CUOI_KY_ML.md").read_text(encoding="utf-8").splitlines()

doc = Document()
doc.styles["Normal"].font.name = "Times New Roman"
doc.styles["Normal"].font.size = Pt(13)


def add_runs(par, text):
    # **bold**, *italic*, `code`
    for tok in re.split(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("`"):
            r = par.add_run(tok[1:-1]); r.font.name = "Consolas"
        elif tok.startswith("*") and len(tok) > 2:
            par.add_run(tok[1:-1]).italic = True
        else:
            par.add_run(tok)


i = 0
while i < len(md):
    line = md[i]
    s = line.strip()
    if not s or s.startswith("<!--") or s == "---":
        i += 1; continue
    m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", s)
    if m:
        img = (HERE / m.group(1)).resolve()
        if img.exists():
            doc.add_picture(str(img), width=Inches(6))
        else:
            doc.add_paragraph(f"[THIẾU HÌNH: {m.group(1)}]")
        i += 1; continue
    if s.startswith("|"):
        rows = []
        while i < len(md) and md[i].strip().startswith("|"):
            cells = [c.strip() for c in md[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) or c == "" for c in cells):
                rows.append(cells)
            i += 1
        rows = [r for r in rows if any(r)]
        if rows:
            n = max(len(r) for r in rows)
            t = doc.add_table(rows=len(rows), cols=n); t.style = "Table Grid"
            for ri, r in enumerate(rows):
                for ci, c in enumerate(r):
                    add_runs(t.cell(ri, ci).paragraphs[0], c.replace("\\|", "|"))
        continue
    h = re.match(r"(#+)\s+(.*)", s)
    if h:
        doc.add_heading(h.group(2), level=min(len(h.group(1)), 4)); i += 1; continue
    if s.startswith(">"):
        p = doc.add_paragraph(style="Intense Quote"); add_runs(p, s.lstrip("> ")); i += 1; continue
    b = re.match(r"^(\s*)[-*]\s+(.*)", line)
    if b:
        p = doc.add_paragraph(style="List Bullet" if not b.group(1) else "List Bullet 2"); add_runs(p, b.group(2)); i += 1; continue
    n_ = re.match(r"^(\s*)\d+\.\s+(.*)", line)
    if n_:
        p = doc.add_paragraph(style="List Number"); add_runs(p, n_.group(2)); i += 1; continue
    add_runs(doc.add_paragraph(), s); i += 1

out = HERE / "BAO_CAO_CUOI_KY_ML.docx"
doc.save(out)
print("Đã ghi", out)

# Kiểm tra bất biến của artifact
man = json.loads((ROOT / "results/reports/final_audit/audit_manifest.json").read_text(encoding="utf-8"))
res = []
for rel, h0 in man["frozen_artifact_sha256"].items():
    p = ROOT / rel
    h1 = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
    res.append((rel, h1 == h0))
ok = sum(r[1] for r in res)
print(f"Artifact khớp hash: {ok}/{len(res)}")
audit = HERE / "REPORT_AUDIT.md"
txt = audit.read_text(encoding="utf-8").split("\n## Kết quả hash")[0]
txt += f"\n## Kết quả hash\n**{ok}/{len(res)} artifact khớp sha256 với audit_manifest.json.**\n\n" + \
       "\n".join(f"- `{r}`: {'khớp' if k else 'KHÔNG KHỚP'}" for r, k in res) + "\n"
audit.write_text(txt, encoding="utf-8")
