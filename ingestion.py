"""
Loading logic for every supported input type. The original app only handled
PDF/txt/md/docx and treated everything as an unstructured text blob - a CSV
of transactions or an XLSX of line items retrieves terribly that way, since
a chunk boundary can slice a row in half or merge two unrelated rows. Tabular
files are instead loaded row-by-row with column headers folded into the text
so each chunk is a self-contained, labeled fact.
"""
import logging
import os

from langchain_core.documents import Document as LC_Doc

from config import Config


def _load_pdf_with_ocr_fallback(filepath: str):
    from langchain_community.document_loaders import PyPDFLoader
    docs = PyPDFLoader(filepath).load()

    if not Config.ENABLE_OCR_FALLBACK:
        return docs

    needs_ocr = any(
        len((d.page_content or "").strip()) < Config.OCR_MIN_CHARS_PER_PAGE for d in docs
    )
    if not needs_ocr:
        return docs

    logging.info(f"'{filepath}': some pages look image-only, retrying with OCR (hi_res + tesseract).")
    try:
        from langchain_unstructured import UnstructuredLoader
        ocr_docs = UnstructuredLoader(
            filepath, mode="elements", strategy="hi_res", languages=["eng"]
        ).load()
        if ocr_docs and any((d.page_content or "").strip() for d in ocr_docs):
            return ocr_docs
        logging.warning(f"OCR pass on '{filepath}' produced no text either; keeping original extraction.")
        return docs
    except Exception as e:
        logging.error(f"OCR fallback failed for '{filepath}': {e}", exc_info=True)
        return docs


def _load_tabular(filepath: str):
    """CSV/XLSX -> one Document per row, with 'column: value' pairs so the
    row is self-describing rather than a bare comma-separated string."""
    import pandas as pd

    if filepath.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(filepath, sheet_name=None)
    else:
        sheets = {"sheet1": pd.read_csv(filepath)}

    docs = []
    for sheet_name, df in sheets.items():
        df = df.fillna("")
        for row_idx, row in df.iterrows():
            text = "\n".join(f"{col}: {val}" for col, val in row.items() if str(val).strip() != "")
            if not text.strip():
                continue
            docs.append(
                LC_Doc(
                    page_content=text,
                    metadata={"sheet": sheet_name, "row": int(row_idx)},
                )
            )
    return docs


def _load_url(url: str):
    from langchain_community.document_loaders import WebBaseLoader
    return WebBaseLoader(url).load()


def load_source(path_or_url: str):
    """
    Dispatches to the right loader based on extension, or treats the input
    as a URL if it looks like one. Returns a list of langchain Documents
    with page_content already non-empty (blank pages/rows are dropped).
    """
    is_url = path_or_url.lower().startswith(("http://", "https://"))

    if is_url:
        raw_docs = _load_url(path_or_url)
    elif path_or_url.lower().endswith(".pdf"):
        raw_docs = _load_pdf_with_ocr_fallback(path_or_url)
    elif path_or_url.lower().endswith((".csv", ".xlsx", ".xls")):
        raw_docs = _load_tabular(path_or_url)
    else:
        from langchain_community.document_loaders import UnstructuredFileLoader
        raw_docs = UnstructuredFileLoader(path_or_url, mode="single").load()

    return [d for d in raw_docs if d.page_content and not d.page_content.isspace()]
