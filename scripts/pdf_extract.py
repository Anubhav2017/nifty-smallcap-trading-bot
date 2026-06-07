#!/usr/bin/env python3
"""
Process BS announcement PDFs and create consolidated files with
date, important information, and key numbers.

Filename convention in attachments/:
  {newsid_variant}_{attachment_uuid}.pdf
  → matched by searching for the ATTACHMENTNAME UUID inside the filename.
"""

import json
import os
import re
import csv
import glob
import sys
from multiprocessing import Pool, cpu_count
from datetime import datetime

try:
    import fitz  # pymupdf
except ImportError:
    print("pymupdf not found. Install with: pip install pymupdf")
    sys.exit(1)


# ── helpers ──────────────────────────────────────────────────────────────────

def build_attachment_index(attachments_dir: str) -> dict:
    """
    Build a dict mapping attachment UUID (lower-case, no .pdf) → full path.
    Actual filenames look like: {newsid_variant}_{attach_uuid}.pdf
    We index by the attach_uuid part (everything after the last '_').
    """
    index = {}
    if not os.path.isdir(attachments_dir):
        return index
    for fn in os.listdir(attachments_dir):
        if not fn.lower().endswith('.pdf'):
            continue
        # The attachment UUID is the part after the last underscore
        base = fn[:-4]  # strip .pdf
        if '_' in base:
            attach_uuid = base.split('_', 1)[1].lower()
        else:
            attach_uuid = base.lower()
        index[attach_uuid] = os.path.join(attachments_dir, fn)
    return index


def find_pdf(attach_name: str, index: dict) -> str:
    """Return the full path for an attachment, or '' if not found."""
    if not attach_name:
        return ''
    key = attach_name.lower().replace('.pdf', '')
    return index.get(key, '')


def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        texts = []
        for page in doc:
            texts.append(page.get_text())
        doc.close()
        return "\n".join(texts)
    except Exception as e:
        return f"[PDF_ERROR: {e}]"


def extract_numbers(text: str) -> str:
    """
    Extract key financial numbers / metrics from announcement text.
    Returns a compact string of found key-value pairs.
    """
    found = []
    patterns = [
        (r'(?:total\s+)?(?:revenue|income|turnover)[^\n:]{0,40}?[:\s]\s*([\d,]+(?:\.\d+)?)\s*(?:crore|cr\.?|lakh|mn|million|billion)?',
         'Revenue'),
        (r'(?:net\s+)?(?:profit|loss)[^\n:]{0,40}?[:\s]\s*([\d,]+(?:\.\d+)?)\s*(?:crore|cr\.?|lakh|mn|million|billion)?',
         'Net Profit/Loss'),
        (r'(?:basic\s+)?(?:diluted\s+)?eps[^\n:]{0,30}?[:\s]\s*([\d,]+(?:\.\d+)?)',
         'EPS'),
        (r'ebitda[^\n:]{0,40}?[:\s]\s*([\d,]+(?:\.\d+)?)\s*(?:crore|cr\.?|lakh|mn|million|billion)?',
         'EBITDA'),
        (r'dividend[^\n:]{0,40}?[:\s]\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+share|%)?',
         'Dividend'),
        (r'total\s+assets[^\n:]{0,40}?[:\s]\s*([\d,]+(?:\.\d+)?)\s*(?:crore|cr\.?|lakh|mn|million|billion)?',
         'Total Assets'),
        (r'\bpat\b[^\n:]{0,40}?[:\s]\s*([\d,]+(?:\.\d+)?)\s*(?:crore|cr\.?|lakh|mn|million|billion)?',
         'PAT'),
        (r'operating\s+(?:profit|income)[^\n:]{0,40}?[:\s]\s*([\d,]+(?:\.\d+)?)\s*(?:crore|cr\.?|lakh|mn|million|billion)?',
         'Operating Profit'),
    ]
    text_lower = text.lower()
    for pattern, label in patterns:
        m = re.search(pattern, text_lower)
        if m:
            found.append(f"{label}: {m.group(1)}")
    return " | ".join(found) if found else ""


def clean_text(text: str) -> str:
    """Clean extracted text (no truncation)."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── per-stock worker ──────────────────────────────────────────────────────────

def process_stock(args):
    """Process all announcements for a single stock. Returns list of row dicts."""
    dataset_name, stock_dir = args
    stock = os.path.basename(stock_dir)
    ann_json = os.path.join(stock_dir, 'announcements.json')
    attachments_dir = os.path.join(stock_dir, 'attachments')

    if not os.path.exists(ann_json):
        return []

    try:
        with open(ann_json, 'r', encoding='utf-8') as f:
            announcements = json.load(f)
    except Exception as e:
        return []

    # Build attachment index once per stock
    attach_index = build_attachment_index(attachments_dir)

    rows = []
    for ann in announcements:
        pdf_name   = ann.get('ATTACHMENTNAME', '')
        date_str   = ann.get('DT_TM', ann.get('NEWS_DT', ''))
        headline   = ann.get('HEADLINE', ann.get('NEWSSUB', ''))
        category   = ann.get('CATEGORYNAME', '')
        subcategory = ann.get('SUBCATNAME', '')
        company    = ann.get('SLONGNAME', stock)
        newsid     = ann.get('NEWSID', '')

        # Parse date
        try:
            dt = datetime.fromisoformat(date_str.split('.')[0])
            date_fmt = dt.strftime('%Y-%m-%d')
            time_fmt = dt.strftime('%H:%M:%S')
        except Exception:
            date_fmt = date_str[:10] if date_str else ''
            time_fmt = ''

        # Locate PDF
        pdf_path   = find_pdf(pdf_name, attach_index)
        pdf_text   = ''
        key_numbers = ''

        if pdf_path:
            raw_text    = extract_pdf_text(pdf_path)
            pdf_text    = clean_text(raw_text)
            key_numbers = extract_numbers(raw_text)

        rows.append({
            'dataset':          dataset_name,
            'stock':            stock,
            'company':          company,
            'date':             date_fmt,
            'time':             time_fmt,
            'category':         category,
            'subcategory':      subcategory,
            'headline':         headline,
            'pdf_filename':     pdf_name,
            'pdf_found':        '1' if pdf_path else '0',
            'key_numbers':      key_numbers,
            'pdf_text_summary': pdf_text,
            'newsid':           newsid,
        })

    return rows


# ── main ──────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    'dataset', 'stock', 'company', 'date', 'time',
    'category', 'subcategory', 'headline',
    'pdf_filename', 'pdf_found', 'key_numbers', 'pdf_text_summary', 'newsid',
]


def process_dataset(dataset_path: str, output_csv: str):
    dataset_name = os.path.basename(dataset_path)
    ann_base = os.path.join(dataset_path, 'bse_announcements')

    if not os.path.exists(ann_base):
        print(f"[SKIP] {ann_base} not found")
        return 0

    stock_dirs = sorted([
        d for d in glob.glob(os.path.join(ann_base, '*'))
        if os.path.isdir(d) and os.path.basename(d) != 'meta'
    ])

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name}")
    print(f"Stocks  : {len(stock_dirs)}")
    print(f"Output  : {output_csv}")
    print(f"Workers : {min(cpu_count(), 8)}")
    print('='*60, flush=True)

    args = [(dataset_name, d) for d in stock_dirs]
    all_rows = []
    workers = min(cpu_count(), 8)

    with Pool(workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_stock, args), 1):
            all_rows.extend(rows)
            stock_name = os.path.basename(args[i-1][1])
            print(f"  [{i:3d}/{len(stock_dirs)}] {stock_name:20s} "
                  f"-> {len(rows):4d} rows  (total: {len(all_rows)})", flush=True)

    # Sort by stock, then date
    all_rows.sort(key=lambda r: (r['stock'], r['date']))

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    pdf_found = sum(1 for r in all_rows if r['pdf_found'] == '1')
    print(f"\n✓ Written {len(all_rows)} rows to {output_csv}")
    print(f"  PDFs extracted: {pdf_found}/{len(all_rows)}", flush=True)
    return len(all_rows)


if __name__ == '__main__':
    datasets = [
        ('dataset_nifty50',     'output/nifty50_announcements_consolidated.csv'),
        ('dataset_smallcap250', 'output/smallcap250_announcements_consolidated.csv'),
    ]

    os.makedirs('output', exist_ok=True)

    total_rows = 0
    for ds_path, out_csv in datasets:
        if os.path.exists(ds_path):
            n = process_dataset(ds_path, out_csv)
            total_rows += n
        else:
            print(f"[SKIP] {ds_path} not found")

    print(f"\n{'='*60}")
    print(f"ALL DONE — total rows written: {total_rows}")
    print('='*60)