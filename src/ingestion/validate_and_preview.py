import argparse, json, sys, hashlib, os, time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any
from dataclasses import dataclass, asdict

# Optional: pretty preview
import pandas as pd

# Optional: token estimates for budgeting chunk sizes later
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def est_tokens(txt: str) -> int: return len(_enc.encode(txt or ""))
except Exception:
    _enc = None
    def est_tokens(txt: str) -> int: return max(1, len((txt or "").split()))

REQUIRED_FIELDS = ["doc_id", "source_path", "mime", "text"]
OPTIONAL_FIELDS = ["title", "section_id", "page_start", "page_end", "lang", "ocr", "tables_md"]

@dataclass
class NormalizedRecord:
    # original fields
    doc_id: str
    source_path: str
    mime: str
    text: str
    title: str = ""
    section_id: str = ""
    page_start: int = None
    page_end: int = None
    lang: str = ""
    ocr: bool = False
    tables_md: List[str] = None

    # additional derived metadata
    text_hash: str = ""
    char_count: int = 0
    token_est: int = 0
    created_at: float = 0.0       # unix timestamp when normalized
    source_exists: bool = False

def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode at line {i}: {e}", file=sys.stderr)

def _validate_record(rec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    for k in REQUIRED_FIELDS:
        if k not in rec:
            errors.append(f"missing required field: {k}")
    # basic type/logic checks
    if "text" in rec and (not isinstance(rec["text"], str) or not rec["text"].strip()):
        errors.append("text is empty or not a string")
    if "mime" in rec and not isinstance(rec["mime"], str):
        errors.append("mime must be a string")
    # page_start/page_end consistency
    ps, pe = rec.get("page_start"), rec.get("page_end")
    if (ps is not None or pe is not None) and (not isinstance(ps, int) or not isinstance(pe, int) or ps > pe):
        errors.append("page_start/page_end invalid (must be ints and page_start <= page_end)")
    return (len(errors) == 0, errors)

def _normalize_record(rec: Dict[str, Any], root: Path) -> NormalizedRecord:
    text = (rec.get("text") or "").strip()
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
    source_path = rec.get("source_path", "")
    source_exists = (root.parent / source_path).exists() if source_path else False

    return NormalizedRecord(
        doc_id=str(rec.get("doc_id", ""))[:256],
        source_path=source_path,
        mime=rec.get("mime", ""),
        text=text,
        title=str(rec.get("title", ""))[:512],
        section_id=str(rec.get("section_id", ""))[:256],
        page_start=rec.get("page_start"),
        page_end=rec.get("page_end"),
        lang=str(rec.get("lang", ""))[:16],
        ocr=bool(rec.get("ocr", False)),
        tables_md=rec.get("tables_md"),
        text_hash=text_hash,
        char_count=len(text),
        token_est=est_tokens(text),
        created_at=time.time(),
        source_exists=source_exists,
    )

def _summarize(records: List[NormalizedRecord]) -> Dict[str, Any]:
    by_mime: Dict[str, int] = {}
    by_lang: Dict[str, int] = {}
    src_missing = 0
    total_chars = 0
    total_tokens = 0
    pdf_page_blocks = 0

    for r in records:
        by_mime[r.mime] = by_mime.get(r.mime, 0) + 1
        if r.lang:
            by_lang[r.lang] = by_lang.get(r.lang, 0) + 1
        if not r.source_exists:
            src_missing += 1
        total_chars += r.char_count
        total_tokens += r.token_est
        if r.mime == "application/pdf" and r.page_start is not None:
            pdf_page_blocks += 1

    return dict(
        total_records=len(records),
        unique_docs=len({r.doc_id for r in records}),
        mimes=by_mime,
        languages=by_lang,
        source_files_missing=src_missing,
        total_characters=total_chars,
        total_token_estimate=total_tokens,
        avg_tokens_per_block=round(total_tokens / max(1, len(records)), 2),
        pdf_page_blocks=pdf_page_blocks,
        duplicates_removed=0,  # filled later
        too_short_removed=0,   # filled later
    )

def _dedupe_and_filter(records: List[NormalizedRecord], min_chars: int) -> Tuple[List[NormalizedRecord], int, int]:
    seen = set()
    deduped: List[NormalizedRecord] = []
    dup = 0
    short = 0
    for r in records:
        if r.char_count < min_chars:
            short += 1
            continue
        if r.text_hash in seen:
            dup += 1
            continue
        seen.add(r.text_hash)
        deduped.append(r)
    return deduped, dup, short

def main():
    ap = argparse.ArgumentParser(description="Validate, preview, and optionally normalize a cleaned corpus JSONL.")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL (from your parse step)")
    ap.add_argument("--out", dest="out", default="", help="Optional normalized JSONL output (writes if provided)")
    ap.add_argument("--report", dest="report", default="data/processed/validation_report.json", help="Validation report path")
    ap.add_argument("--preview_csv", dest="preview_csv", default="data/processed/preview_sample.csv", help="Preview CSV path")
    ap.add_argument("--sample", type=int, default=20, help="How many rows to include in the preview CSV")
    ap.add_argument("--min_chars", type=int, default=20, help="Drop blocks shorter than this when normalizing")
    args = ap.parse_args()

    in_path = Path(args.inp)
    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    raw: List[Dict[str, Any]] = list(_load_jsonl(in_path))
    if not raw:
        print("[WARN] Input JSONL appears empty.", file=sys.stderr)

    # Validate
    val_errors = 0
    for i, rec in enumerate(raw, start=1):
        ok, errs = _validate_record(rec)
        if not ok:
            val_errors += 1
            print(f"[INVALID] line {i}: {errs}", file=sys.stderr)

    # Normalize + enrich
    norm: List[NormalizedRecord] = [_normalize_record(r, in_path) for r in raw]

    # Summary before filtering
    summary = _summarize(norm)

    # Dedupe & filter
    filtered, dup, short = _dedupe_and_filter(norm, args.min_chars)
    summary["duplicates_removed"] = dup
    summary["too_short_removed"] = short

    # Build small preview table
    df = pd.DataFrame([
        dict(
            doc_id=r.doc_id,
            title=r.title,
            section_id=r.section_id,
            mime=r.mime,
            lang=r.lang,
            char_count=r.char_count,
            token_est=r.token_est,
            page=r.page_start,
            source_ok=r.source_exists,
            text_preview=(r.text[:200] + "…") if len(r.text) > 200 else r.text
        )
        for r in filtered[: max(1, args.sample)]
    ])

    # Write artifacts
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(dict(
            input=str(in_path),
            validation_errors=val_errors,
            summary=summary,
        ), f, ensure_ascii=False, indent=2)

    Path(args.preview_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.preview_csv, index=False, encoding="utf-8")

    if args.out:
        # Emit normalized JSONL
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in filtered:
                rec = asdict(r)
                # Drop the full text if you wanted a "meta only" file later—here we keep it.
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Console summary
    print("\n=== CORPUS VALIDATION REPORT ===")
    print(f"Input file:     {in_path}")
    print(f"Records:        {len(raw)} (after filter: {len(filtered)})")
    print(f"Invalid lines:  {val_errors}")
    print(f"Duplicates rm:  {summary['duplicates_removed']}")
    print(f"Too-short rm:   {summary['too_short_removed']}")
    print(f"Unique docs:    {summary['unique_docs']}")
    print(f"MIME counts:    {summary['mimes']}")
    print(f"Lang counts:    {summary['languages']}")
    print(f"Missing files:  {summary['source_files_missing']}")
    print(f"Avg tokens/bl:  {summary['avg_tokens_per_block']}")
    print(f"Preview CSV:    {args.preview_csv}")
    print(f"Report JSON:    {args.report}")
    if args.out:
        print(f"Normalized JSONL written -> {args.out}")

if __name__ == "__main__":
    main()