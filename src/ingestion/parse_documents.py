import argparse, hashlib, json, os, re, sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import re

# --- optional deps (all in your requirements.txt) ---
import ftfy
from bs4 import BeautifulSoup
import frontmatter
from langdetect import detect as detect_lang

# PDF extraction libs
import fitz  # PyMuPDF (fast, layout-aware)
import pdfplumber  # (good for tables)

# HTML main-content extractor
from trafilatura import extract as trafi_extract

# ---------- helpers ----------

def sha1_of_path(p: Path) -> str:
    try:
        return hashlib.sha1(p.read_bytes()).hexdigest()[:16]
    except Exception:
        # fallback to name-based
        return hashlib.sha1(str(p).encode()).hexdigest()[:16]

def normalize_whitespace(text: str) -> str:
    # remove trailing spaces, collapse multiple spaces, normalize newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def dehyphenate(text: str) -> str:
    # join words broken by line-wrap hyphenation: "inter-\nesting" -> "interesting"
    return re.sub(r"(?<=\w)-\n(?=\w)", "", text)

def fix_wrapped_lines(text: str) -> str:
    # join single linebreaks inside paragraphs but keep blank-line paragraph breaks
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

def remove_page_headers_footers(page_texts: List[str]) -> List[str]:
    """
    Heuristic: find top N/bottom N lines that repeat on >60% of pages and strip them.
    Works best when you first split each page into lines.
    """
    if not page_texts:
        return page_texts
    top_counts, bottom_counts = {}, {}
    def top_line(t): return t.splitlines()[0].strip() if t.splitlines() else ""
    def bottom_line(t): return t.splitlines()[-1].strip() if t.splitlines() else ""

    tops  = [top_line(t) for t in page_texts if t.strip()]
    bots  = [bottom_line(t) for t in page_texts if t.strip()]
    for s in tops:  top_counts[s]  = top_counts.get(s, 0) + 1
    for s in bots:  bottom_counts[s]= bottom_counts.get(s, 0) + 1

    threshold = max(2, int(0.6 * len(page_texts)))

    def strip_header_footer(t):
        lines = t.splitlines()
        if lines:
            if lines[0].strip() in top_counts and top_counts[lines[0].strip()] >= threshold:
                lines = lines[1:]
        if lines:
            if lines[-1].strip() in bottom_counts and bottom_counts[lines[-1].strip()] >= threshold:
                lines = lines[:-1]
        return "\n".join(lines)

    return [strip_header_footer(t) for t in page_texts]

def clean_text_basic(text: str) -> str:
    text = ftfy.fix_text(text)
    text = dehyphenate(text)
    text = fix_wrapped_lines(text)
    text = normalize_whitespace(text)

    # Remove LaTeX-style math markers like $\mathbf{8}$ or $8$
    text = re.sub(r"\$\\?mathbf\{([^}]+)\}\$", r"\1", text)  # -> 8
    text = re.sub(r"\$([^$]+)\$", r"\1", text)  # -> remove plain $...$ math delimiters

    # Optionally clean stray backslashes or curly braces
    text = text.replace("\\", "").replace("{", "").replace("}", "")

    return text.strip()

# ---------- schema ----------

@dataclass
class Block:
    doc_id: str
    source_path: str
    mime: str
    title: Optional[str]
    section_id: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    text: str
    lang: Optional[str]
    ocr: bool = False
    tables_md: Optional[List[str]] = None

# ---------- extractors ----------

def extract_pdf_blocks(path: Path) -> Iterable[Block]:
    doc_id = sha1_of_path(path)
    title = path.stem
    pdf = fitz.open(path)
    # page-wise text with PyMuPDF (fast)
    page_texts = []
    for i, page in enumerate(pdf):
        t = page.get_text("text") or ""
        page_texts.append(t)

    # if all pages are empty (image-only), try OCR if enabled
    ocr_enabled = os.getenv("ENABLE_OCR", "0") == "1"
    did_ocr = False
    if all(not t.strip() for t in page_texts) and ocr_enabled:
        try:
            # lightweight OCR path: render page images and OCR via pytesseract
            import pytesseract
            from PIL import Image
            did_ocr = True
            page_texts = []
            for i, page in enumerate(pdf):
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                t = pytesseract.image_to_string(img)
                page_texts.append(t)
        except Exception as e:
            # keep empty texts; we'll just skip them later
            sys.stderr.write(f"[WARN] OCR failed for {path}: {e}\n")

    # strip repeating headers/footers
    page_texts = remove_page_headers_footers(page_texts)

    # tables (best-effort) with pdfplumber
    tables_by_page: Dict[int, List[str]] = {}
    try:
        with pdfplumber.open(path) as pl:
            for i, p in enumerate(pl.pages):
                md_tables = []
                try:
                    for table in p.extract_tables() or []:
                        if table and any(any(cell for cell in row) for row in table):
                            # convert to Markdown
                            heads = table[0]
                            if heads and all(isinstance(h, str) for h in heads):
                                # build markdown table
                                md = "|" + "|".join(h or "" for h in heads) + "|\n"
                                md += "|" + "|".join("---" for _ in heads) + "|\n"
                                for row in table[1:]:
                                    md += "|" + "|".join((c or "") for c in row) + "|\n"
                                md_tables.append(md)
                except Exception:
                    pass
                if md_tables:
                    tables_by_page[i + 1] = md_tables
    except Exception:
        pass

    for i, raw in enumerate(page_texts, start=1):
        if not raw.strip():
            continue
        text = clean_text_basic(raw)
        try:
            lang = detect_lang(text)
        except Exception:
            lang = None
        yield Block(
            doc_id=doc_id,
            source_path=str(path),
            mime="application/pdf",
            title=title,
            section_id=f"page-{i}",
            page_start=i,
            page_end=i,
            text=text,
            lang=lang,
            ocr=did_ocr,
            tables_md=tables_by_page.get(i),
        )

def extract_html_blocks(path: Path) -> Iterable[Block]:
    doc_id = sha1_of_path(path)
    title = path.stem
    html = path.read_text(encoding="utf-8", errors="ignore")

    # use trafilatura to get main content (boilerplate removal)
    main = trafi_extract(html) or html
    soup = BeautifulSoup(main, "lxml")

    # drop leftover non-content
    for bad in soup(["script", "style", "nav", "footer", "aside", "form"]):
        bad.decompose()

    # keep tables as Markdown (simple)
    tables_md = []
    for t in soup.find_all("table"):
        rows = []
        for tr in t.find_all("tr"):
            row = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
            if row:
                rows.append(row)
        if rows:
            heads = rows[0]
            md = "|" + "|".join(heads) + "|\n"
            md += "|" + "|".join("---" for _ in heads) + "|\n"
            for r in rows[1:]:
                md += "|" + "|".join(r) + "|\n"
            tables_md.append(md)
        t.decompose()  # remove table from text so it isn't duplicated

    text = soup.get_text("\n")
    text = clean_text_basic(text)
    try:
        lang = detect_lang(text)
    except Exception:
        lang = None

    yield Block(
        doc_id=doc_id,
        source_path=str(path),
        mime="text/html",
        title=title,
        section_id=None,
        page_start=None,
        page_end=None,
        text=text,
        lang=lang,
        ocr=False,
        tables_md=tables_md or None,
    )

def extract_md_blocks(path: Path) -> Iterable[Block]:
    doc_id = sha1_of_path(path)
    post = frontmatter.load(path)
    body = str(post).strip()
    # optional: split by top-level headings as sections
    sections = re.split(r"(?m)^(?=# )", body)
    for idx, sec in enumerate(sections, start=1):
        if not sec.strip():
            continue
        title_match = re.search(r"(?m)^#\s+(.*)", sec)
        title = title_match.group(1).strip() if title_match else path.stem
        text = clean_text_basic(sec)
        try:
            lang = detect_lang(text)
        except Exception:
            lang = None
        yield Block(
            doc_id=doc_id,
            source_path=str(path),
            mime="text/markdown",
            title=title,
            section_id=f"h1-{idx}",
            page_start=None,
            page_end=None,
            text=text,
            lang=lang,
        )

def extract_txt_blocks(path: Path) -> Iterable[Block]:
    doc_id = sha1_of_path(path)
    title = path.stem
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = clean_text_basic(raw)
    try:
        lang = detect_lang(text)
    except Exception:
        lang = None
    yield Block(
        doc_id=doc_id,
        source_path=str(path),
        mime="text/plain",
        title=title,
        section_id=None,
        page_start=None,
        page_end=None,
        text=text,
        lang=lang,
    )

# ---------- driver ----------

def discover_files(root: Path) -> List[Path]:
    exts = {".pdf", ".html", ".htm", ".md", ".txt"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]

def to_jsonl(blocks: Iterable[Block], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for b in blocks:
            rec = asdict(b)
            # drop empty fields
            rec = {k: v for k, v in rec.items() if v not in (None, [], "")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Parse & clean docs for RAG.")
    ap.add_argument("--in", dest="inp", default="data/raw_docs", help="Input folder")
    ap.add_argument("--out", dest="out", default="data/processed/corpus_clean.jsonl", help="Output JSONL")
    args = ap.parse_args()

    root = Path(args.inp)
    files = discover_files(root)
    if not files:
        print(f"[WARN] No files found under {root}", file=sys.stderr)

    all_blocks: List[Block] = []
    for p in files:
        try:
            if p.suffix.lower() == ".pdf":
                all_blocks.extend(list(extract_pdf_blocks(p)))
            elif p.suffix.lower() in {".html", ".htm"}:
                all_blocks.extend(list(extract_html_blocks(p)))
            elif p.suffix.lower() == ".md":
                all_blocks.extend(list(extract_md_blocks(p)))
            else:
                all_blocks.extend(list(extract_txt_blocks(p)))
        except Exception as e:
            print(f"[ERROR] Failed on {p}: {e}", file=sys.stderr)

    # deduplicate near-identical blocks (cheap cosine-ish hashing)
    # Here: exact duplicate pruning; swap with SimHash/MinHash if needed.
    seen = set()
    unique_blocks = []
    for b in all_blocks:
        key = hashlib.sha1(b.text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique_blocks.append(b)

    to_jsonl(unique_blocks, Path(args.out))
    print(f"[OK] Wrote {len(unique_blocks)} cleaned blocks -> {args.out}")

if __name__ == "__main__":
    main()