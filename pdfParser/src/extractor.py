"""Unified Document Extractor for PDF and DOCX files preserving layout and typography."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from pdfParser.src.models import DocumentBlock
except ImportError:
    try:
        from ml.src.models import DocumentBlock
    except ImportError:
        from models import DocumentBlock


class DocumentExtractor:
    """Reads PDF and DOCX documents into layout-aware DocumentBlock instances."""

    def extract(self, file_path: str) -> list[DocumentBlock]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._extract_pdf(file_path)
        elif ext in (".docx", ".doc"):
            return self._extract_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported: .pdf, .docx")

    def _extract_pdf(self, file_path: str) -> list[DocumentBlock]:
        try:
            import fitz  # PyMuPDF
            return self._extract_fitz(file_path)
        except ImportError:
            return self._extract_pypdf(file_path)

    def _extract_fitz(self, file_path: str) -> list[DocumentBlock]:
        import fitz

        doc = fitz.open(file_path)
        blocks: list[DocumentBlock] = []

        for page_num, page in enumerate(doc, start=1):
            page_height = page.rect.height
            text_page = page.get_text("dict")

            for b in text_page.get("blocks", []):
                if b.get("type") != 0:
                    continue

                bbox = tuple(b.get("bbox", (0, 0, 0, 0)))
                y0, y1 = bbox[1], bbox[3]

                # Header/footer filter (top/bottom 35pt margin)
                if y0 < 35 or y1 > page_height - 35:
                    continue

                block_text_lines: list[str] = []
                max_font_size = 0.0
                is_bold = False
                is_italic = False

                for line in b.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                        font_size = span.get("size", 0.0)
                        if font_size > max_font_size:
                            max_font_size = font_size
                        flags = span.get("flags", 0)
                        font_name = span.get("font", "").lower()
                        if (flags & 2**4) or "bold" in font_name:
                            is_bold = True
                        if (flags & 2**1) or "italic" in font_name or "oblique" in font_name:
                            is_italic = True

                    if line_text.strip():
                        block_text_lines.append(line_text.strip())

                full_text = " ".join(block_text_lines).strip()
                if not full_text:
                    continue

                blocks.append(
                    DocumentBlock(
                        text=full_text,
                        page=page_num,
                        font_size=round(max_font_size, 1),
                        bold=is_bold,
                        italic=is_italic,
                        indent_left=round(bbox[0], 1),
                    )
                )

        doc.close()
        return blocks

    def _extract_pypdf(self, file_path: str) -> list[DocumentBlock]:
        import pypdf

        reader = pypdf.PdfReader(file_path)
        blocks: list[DocumentBlock] = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text:
                continue
            for line in text.splitlines():
                clean_line = line.strip()
                if clean_line:
                    blocks.append(DocumentBlock(text=clean_line, page=page_num, font_size=12.0))

        return blocks

    def _extract_docx(self, file_path: str) -> list[DocumentBlock]:
        import docx

        doc = docx.Document(file_path)
        blocks: list[DocumentBlock] = []

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue

            style_name = p.style.name.lower() if p.style else ""
            is_bold = any(run.bold for run in p.runs if run.bold is not None) or "heading" in style_name
            is_italic = any(run.italic for run in p.runs if run.italic is not None)

            font_sizes = [run.font.size.pt for run in p.runs if run.font and run.font.size]
            font_size = font_sizes[0] if font_sizes else 12.0

            blocks.append(
                DocumentBlock(
                    text=text,
                    page=1,
                    font_size=round(font_size, 1),
                    bold=is_bold,
                    italic=is_italic,
                )
            )

        return blocks
