"""
Chunk cleaned corpus into retrieval-ready chunks using semantic boundaries.

Input  : JSONL from parse step (one record per block/page/section)
Output : JSONL of chunks with propagated metadata + chunk ids

Usage:
  python -m src.embeddings.chunk_text \
    --in data/processed/corpus_clean.jsonl \
    --out data/processed/corpus_chunks.jsonl \
    --max_tokens 450 --overlap_tokens 60 --min_tokens 60 --respect_pages
"""

import argparse, json, re, sys, hashlib, time
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Optional

# --- Tokenizer (tiktoken if available; fallback to words) ---
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(s: str) -> int:
        return len(_enc.encode(s or ""))
    def slice_by_tokens(tokens: int) -> int:
        return tokens
except Exception:
    _enc = None
    def count_tokens(s: str) -> int:
        # crude but stable fallback
        return max(1, len((s or "").split()))
    def slice_by_tokens(tokens: int) -> int:
        return tokens

# --- Heuristics for semantic segmentation ---
# Detects markdown-style headings, numbered headings, and ALL CAPS lines.
HEADING_RX = re.compile(
    r"(?m)("
    r"^\s{0,3}#{1,6}\s+.+$"            # Markdown headings (#, ##, etc.)
    r"|^[A-Z][A-Z0-9 &/,:-]{3,}$"      # ALL CAPS section titles
    r"|^\d{1,2}(?:\.\d+)*[.)]\s+.+$"   # Numbered or outline headings
    r")"
)

LIST_RX = re.compile(r"(?m)^\s*(?:[-*•]|[a-zA-Z]\)|\d+[.)])\s+")

def normalize_text(s: str) -> str:
    # Preserve paragraph breaks, normalize spacing
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_into_paragraphs(s: str) -> List[str]:
    # Split at blank lines; keep list items grouped with following wrapped lines
    paras = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    return paras

def is_heading(p: str) -> bool:
    # Short strong line or matches heading regex
    line0 = p.splitlines()[0].strip()
    if HEADING_RX.match(line0):
        return True
    # very short line ending with ":" often a label
    if len(line0) <= 60 and line0.endswith(":"):
        return True
    return False

def coalesce_by_headings(paras: List[str]) -> List[Tuple[Optional[str], str]]:
    """
    Convert [para, para, ...] into [(heading, body), ...]
    If a paragraph looks like a heading, start a new section.
    Subsequent non-heading paragraphs attach to the current heading.
    """
    sections: List[Tuple[Optional[str], List[str]]] = []
    current_head: Optional[str] = None
    current_body: List[str] = []

    def _flush():
        if current_body or current_head:
            sections.append((current_head, "\n\n".join(current_body).strip()))

    for p in paras:
        if is_heading(p):
            # start new section
            _flush()
            current_head = p.strip()
            current_body = []
        else:
            # keep list bullets intact; avoid joining separate bullets
            if LIST_RX.match(p) and current_body and not current_body[-1].endswith("\n"):
                current_body.append(p)
            else:
                current_body.append(p)

    _flush()
    # Expand into list of (heading, body) where body may be empty
    return [(h, b) for (h, b) in sections if (h or b)]

def segment_text_semantically(text: str) -> List[str]:
    """
    Returns fine-grained segments:
    - Each section becomes: [heading (optional), paragraph blocks...]
    - Headings are kept as their own tiny segment to stick to the next chunk.
    """
    text = normalize_text(text)
    paras = split_into_paragraphs(text)
    if not paras:
        return []
    sections = coalesce_by_headings(paras)

    segments: List[str] = []
    for head, body in sections:
        if head:
            segments.append(head.strip())
        if body:
            # Further split body on list items to avoid fusing unrelated bullets
            body_paras = [bp.strip() for bp in re.split(r"\n(?=\s*(?:[-*•]|\d+[.)]|[a-zA-Z]\)))", body) if bp.strip()]
            segments.extend(body_paras)
    return segments

# --- Packing segments into token windows with overlap ---

def pack_segments(
    segments: List[str],
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int
) -> List[List[str]]:
    """
    Greedy packer that keeps segment boundaries; starts a new chunk if adding the next
    segment would exceed max_tokens. Adds a token-overlap window between consecutive chunks.
    """
    chunks: List[List[str]] = []
    cur: List[str] = []
    cur_tok = 0

    for seg in segments:
        seg_tok = count_tokens(seg)
        # If a single segment is larger than max_tokens, hard-split it.
        if seg_tok > max_tokens:
            # flush current
            if cur:
                chunks.append(cur)
                cur, cur_tok = [], 0
            # slice long segment into pieces
            words = seg.split()
            # approximate tokens ~ words in fallback; for tiktoken we just pack by count
            step = max(1, max_tokens - 5)
            piece = []
            piece_tok = 0
            for w in words:
                t = count_tokens(w)
                if piece_tok + t + 1 > step:
                    chunks.append([" ".join(piece)])
                    piece, piece_tok = [], 0
                piece.append(w)
                piece_tok += t + 1
            if piece:
                chunks.append([" ".join(piece)])
            # add overlap from previous chunk only handled in normal flow
            continue

        # normal case: try to add seg
        if cur_tok + seg_tok <= max_tokens:
            cur.append(seg)
            cur_tok += seg_tok
        else:
            # finalize current
            if cur:
                chunks.append(cur)
            # create next with overlap tail from previous
            if overlap_tokens > 0 and chunks:
                tail = tail_for_overlap(chunks[-1], overlap_tokens)
                cur = tail + [seg]
                cur_tok = sum(count_tokens(x) for x in cur)
            else:
                cur = [seg]
                cur_tok = seg_tok

    if cur:
        chunks.append(cur)

    # ensure min_tokens by borrowing from neighbors when possible
    chunks = enforce_min_tokens(chunks, min_tokens, max_tokens)
    return chunks

def tail_for_overlap(prev_chunk: List[str], overlap_tokens: int) -> List[str]:
    chosen: List[str] = []
    total = 0
    for seg in reversed(prev_chunk):
        t = count_tokens(seg)
        if total + t > overlap_tokens:
            break
        chosen.append(seg)
        total += t
    return list(reversed(chosen))

def enforce_min_tokens(chunks: List[List[str]], min_tokens: int, max_tokens: int) -> List[List[str]]:
    if not chunks or min_tokens <= 0:
        return chunks
    out: List[List[str]] = []
    buf: List[str] = []
    for ch in chunks:
        if count_tokens("\n\n".join(ch)) >= min_tokens:
            # flush any buffer first if it exists
            if buf:
                merged = buf + ch
                if count_tokens("\n\n".join(merged)) <= max_tokens:
                    out.append(merged)
                    buf = []
                else:
                    out.append(buf)
                    out.append(ch)
                    buf = []
            else:
                out.append(ch)
        else:
            # accumulate until we reach min_tokens
            candidate = (buf + ch) if buf else ch
            if count_tokens("\n\n".join(candidate)) >= min_tokens:
                out.append(candidate)
                buf = []
            else:
                buf = candidate
    if buf:
        out.append(buf)
    return out

# --- JSONL IO ---

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Bad JSON line {i}: {e}", file=sys.stderr)

def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# --- Chunk driver per record ---

def chunk_record(
    rec: Dict[str, Any],
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
    forbid_cross_page: bool
) -> List[Dict[str, Any]]:
    """
    Make chunks from a single cleaned record. If forbid_cross_page is True and the
    record has page markers, we chunk within each page independently.
    """
    text = (rec.get("text") or "").strip()
    if not text:
        return []

    # If the record already represents one page (page_start present), we don't
    # have to split across pages—just segment the text. The forbid_cross_page
    # flag matters when a record could contain multiple pages (rare in our pipeline).
    units: List[Tuple[Optional[int], Optional[int], str]] = []
    if forbid_cross_page and rec.get("page_start") is not None and rec.get("page_end") is not None:
        # Treat the whole record as a single-page unit
        units.append((rec.get("page_start"), rec.get("page_end"), text))
    else:
        units.append((rec.get("page_start"), rec.get("page_end"), text))

    out_chunks: List[Dict[str, Any]] = []
    for (ps, pe, unit_text) in units:
        segs = segment_text_semantically(unit_text)
        if not segs:
            continue
        windows = pack_segments(segs, max_tokens=max_tokens, min_tokens=min_tokens, overlap_tokens=overlap_tokens)

        # emit records
        for idx, segs_in_chunk in enumerate(windows, start=1):
            chunk_text = "\n\n".join(segs_in_chunk).strip()
            if not chunk_text:
                continue
            chunk_tokens = count_tokens(chunk_text)
            chunk_id = make_chunk_id(rec, idx, chunk_text)
            out_chunks.append({
                # identity
                "chunk_id": chunk_id,
                "parent_doc_id": rec.get("doc_id"),
                "parent_section_id": rec.get("section_id"),
                # location/meta
                "source_path": rec.get("source_path"),
                "mime": rec.get("mime"),
                "title": rec.get("title"),
                "page_start": ps,
                "page_end": pe,
                "lang": rec.get("lang"),
                "ocr": rec.get("ocr", False),
                # content
                "text": chunk_text,
                "num_tokens": chunk_tokens,
                # book-keeping
                "created_at": int(time.time()),
                "generator": "semantic_chunker_v1",
            })
    return out_chunks

def make_chunk_id(rec: Dict[str, Any], idx: int, text: str) -> str:
    base = f"{rec.get('doc_id','')}|{rec.get('section_id','')}|{idx}|{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:24]

# --- CLI ---

def main():
    ap = argparse.ArgumentParser(description="Semantic chunker for cleaned corpus JSONL.")
    ap.add_argument("--in", dest="inp", required=True, help="Input cleaned JSONL")
    ap.add_argument("--out", dest="out", required=True, help="Output chunks JSONL")
    ap.add_argument("--max_tokens", type=int, default=450, help="Max tokens per chunk")
    ap.add_argument("--min_tokens", type=int, default=60, help="Min tokens per chunk after packing")
    ap.add_argument("--overlap_tokens", type=int, default=60, help="Token overlap between consecutive chunks")
    ap.add_argument("--respect_pages", action="store_true", help="Avoid crossing page boundaries if present")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)
    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    total_in = 0
    total_out = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in read_jsonl(in_path):
            total_in += 1
            chunks = chunk_record(
                rec,
                max_tokens=args.max_tokens,
                min_tokens=args.min_tokens,
                overlap_tokens=args.overlap_tokens,
                forbid_cross_page=args.respect_pages
            )
            total_out += len(chunks)
            for ch in chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"[OK] Chunked {total_in} records into {total_out} chunks -> {out_path}")

if __name__ == "__main__":
    main()