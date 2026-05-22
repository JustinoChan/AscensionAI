"""Generate AscensionAI_Technical_Writeup.pdf from the markdown source.

Lightweight markdown -> PDF rendering using fpdf2. Handles:
  - Headings (#, ##, ###, ####)
  - Bullet lists (-)
  - Numbered lists (1. 2. 3.)
  - Pipe tables (| col | col |)
  - Fenced code blocks (```)
  - Horizontal rules (---)
  - Inline `code` and **bold**
  - ASCII charts inside code blocks (monospace, preserves whitespace)
"""

from __future__ import annotations

import argparse
import io
import os
import re

try:
    from fpdf import FPDF
    _FPDF_IMPORT_ERROR = None
except ImportError as exc:
    FPDF = object
    _FPDF_IMPORT_ERROR = exc

try:
    from reportlab.lib.pagesizes import A4 as RL_A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen import canvas as rl_canvas
    _REPORTLAB_IMPORT_ERROR = None
except ImportError as exc:
    RL_A4 = None
    pdfmetrics = None
    rl_canvas = None
    _REPORTLAB_IMPORT_ERROR = exc

DOC_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(DOC_DIR, "AscensionAI_Technical_Writeup.md")
PDF_PATH = os.path.join(DOC_DIR, "AscensionAI_Technical_Writeup.pdf")

PAGE_W = 210      # A4 width mm
PAGE_H = 297
MARGIN = 18

FONT_BODY = "Helvetica"
FONT_MONO = "Courier"


if _FPDF_IMPORT_ERROR is None:
    class WriteupPDF(FPDF):
        def __init__(self, title: str = "AscensionAI Technical Writeup"):
            super().__init__(orientation="P", unit="mm", format="A4")
            self.document_title = title
            self.set_margins(MARGIN, MARGIN, MARGIN)
            self.set_auto_page_break(auto=True, margin=MARGIN)
            self.add_page()

        def header(self):
            if self.page_no() == 1:
                return
            self.set_font(FONT_BODY, "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, _ascii_safe(self.document_title), align="L")
            self.cell(0, 6, f"Page {self.page_no()}", align="R")
            self.set_text_color(0, 0, 0)
            self.ln(8)

        def footer(self):
            pass
else:
    class WriteupPDF:
        """Small ReportLab-backed adapter for the subset of FPDF used here."""

        _MM_TO_PT = 72 / 25.4

        def __init__(self, title: str = "AscensionAI Technical Writeup"):
            if _REPORTLAB_IMPORT_ERROR is not None:
                raise RuntimeError(
                    "Missing dependency: install fpdf2 or reportlab to render PDFs."
                ) from _REPORTLAB_IMPORT_ERROR
            self.document_title = title
            self._buffer = io.BytesIO()
            self._canvas = rl_canvas.Canvas(self._buffer, pagesize=RL_A4)
            self._page_w_pt, self._page_h_pt = RL_A4
            self._page_w = self._page_w_pt / self._MM_TO_PT
            self._page_h = self._page_h_pt / self._MM_TO_PT
            self._left = MARGIN
            self._top = MARGIN
            self._right = MARGIN
            self._bottom = MARGIN
            self._x = self._left
            self._y = self._top
            self._font_family = FONT_BODY
            self._font_style = ""
            self._font_size = 10
            self._text_color = (0, 0, 0)
            self._fill_color = (255, 255, 255)
            self._draw_color = (0, 0, 0)
            self._page_no = 0
            self.add_page()

        def _pt(self, mm_value: float) -> float:
            return mm_value * self._MM_TO_PT

        def _font_name(self) -> str:
            family = "Courier" if self._font_family == FONT_MONO else "Helvetica"
            bold = "B" in self._font_style
            italic = "I" in self._font_style
            if family == "Courier":
                if bold and italic:
                    return "Courier-BoldOblique"
                if bold:
                    return "Courier-Bold"
                if italic:
                    return "Courier-Oblique"
                return "Courier"
            if bold and italic:
                return "Helvetica-BoldOblique"
            if bold:
                return "Helvetica-Bold"
            if italic:
                return "Helvetica-Oblique"
            return "Helvetica"

        def _apply_font(self) -> None:
            self._canvas.setFont(self._font_name(), self._font_size)

        def _rgb(self, color: tuple[int, int, int]) -> tuple[float, float, float]:
            return tuple(max(0, min(255, c)) / 255 for c in color)

        def _canvas_y(self, y_mm: float) -> float:
            return self._page_h_pt - self._pt(y_mm)

        def _text_baseline(self, y_mm: float, h_mm: float) -> float:
            return self._canvas_y(y_mm + h_mm * 0.72)

        def _ensure_room(self, h_mm: float) -> None:
            if self._y + h_mm > self._page_h - self._bottom:
                self.add_page()

        def _draw_text(self, text: str, x_mm: float, y_mm: float, h_mm: float) -> None:
            self._apply_font()
            self._canvas.setFillColorRGB(*self._rgb(self._text_color))
            self._canvas.drawString(
                self._pt(x_mm),
                self._text_baseline(y_mm, h_mm),
                _ascii_safe(text),
            )

        def _wrap_text(self, text: str, w_mm: float) -> list[str]:
            text = _ascii_safe(text)
            if w_mm <= 0:
                w_mm = self._page_w - self._right - self._x
            max_w = max(1, w_mm)
            lines: list[str] = []
            for raw_line in text.splitlines() or [""]:
                words = raw_line.split()
                if not words:
                    lines.append("")
                    continue
                line = words[0]
                for word in words[1:]:
                    candidate = f"{line} {word}"
                    if self.get_string_width(candidate) <= max_w:
                        line = candidate
                    else:
                        lines.append(line)
                        line = word
                lines.append(line)
            return lines

        def set_margins(self, left: float, top: float, right: float) -> None:
            self._left = left
            self._top = top
            self._right = right
            self._x = left
            self._y = top

        def set_auto_page_break(self, auto: bool = True, margin: float = MARGIN) -> None:
            self._bottom = margin

        def add_page(self) -> None:
            if self._page_no:
                self._canvas.showPage()
            self._page_no += 1
            self._x = self._left
            self._y = self._top
            self.header()

        def page_no(self) -> int:
            return self._page_no

        def header(self) -> None:
            if self.page_no() == 1:
                return
            self.set_font(FONT_BODY, "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, _ascii_safe(self.document_title), align="L")
            self.set_xy(self._page_w - self._right - 24, self._top)
            self.cell(24, 6, f"Page {self.page_no()}", align="R")
            self.set_text_color(0, 0, 0)
            self.ln(8)

        def footer(self) -> None:
            pass

        def set_font(self, family: str, style: str = "", size: float = 10) -> None:
            self._font_family = family
            self._font_style = style or ""
            self._font_size = size

        def set_text_color(self, r: int, g: int, b: int) -> None:
            self._text_color = (r, g, b)

        def set_fill_color(self, r: int, g: int, b: int) -> None:
            self._fill_color = (r, g, b)

        def set_draw_color(self, r: int, g: int, b: int) -> None:
            self._draw_color = (r, g, b)

        def get_y(self) -> float:
            return self._y

        def get_x(self) -> float:
            return self._x

        def set_x(self, x: float) -> None:
            self._x = x

        def set_xy(self, x: float, y: float) -> None:
            self._x = x
            self._y = y

        def get_string_width(self, text: str) -> float:
            width_pt = pdfmetrics.stringWidth(
                _ascii_safe(text),
                self._font_name(),
                self._font_size,
            )
            return width_pt / self._MM_TO_PT

        def ln(self, h: float | None = None) -> None:
            self._y += h if h is not None else self._font_size * 0.35
            self._x = self._left
            self._ensure_room(0)

        def write(self, h: float, text: str) -> None:
            for token in re.findall(r"\S+\s*|\s+", _ascii_safe(text)):
                token_w = self.get_string_width(token)
                if self._x + token_w > self._page_w - self._right and token.strip():
                    self.ln(h)
                self._ensure_room(h)
                self._draw_text(token, self._x, self._y, h)
                self._x += token_w

        def cell(
            self,
            w: float,
            h: float,
            txt: str = "",
            border: int = 0,
            ln: int = 0,
            align: str = "",
            fill: bool = False,
        ) -> None:
            if w == 0:
                w = self._page_w - self._right - self._x
            self._ensure_room(h)
            x_pt = self._pt(self._x)
            y_pt = self._canvas_y(self._y + h)
            w_pt = self._pt(w)
            h_pt = self._pt(h)
            if fill:
                self._canvas.setFillColorRGB(*self._rgb(self._fill_color))
                self._canvas.rect(x_pt, y_pt, w_pt, h_pt, stroke=0, fill=1)
            if border:
                self._canvas.setStrokeColorRGB(*self._rgb(self._draw_color))
                self._canvas.rect(x_pt, y_pt, w_pt, h_pt, stroke=1, fill=0)
            if txt:
                text_w = self.get_string_width(txt)
                text_x = self._x + 1
                if align == "R":
                    text_x = self._x + max(1, w - text_w - 1)
                elif align == "C":
                    text_x = self._x + max(1, (w - text_w) / 2)
                self._draw_text(txt, text_x, self._y, h)
            if ln:
                self.ln(h)
            else:
                self._x += w

        def multi_cell(self, w: float, h: float, txt: str, *args, **kwargs) -> None:
            if w == 0:
                w = self._page_w - self._right - self._x
            start_x = self._x
            for line in self._wrap_text(txt, w):
                self._ensure_room(h)
                self._draw_text(line, start_x, self._y, h)
                self._y += h
            self._x = self._left

        def line(self, x1: float, y1: float, x2: float, y2: float) -> None:
            self._canvas.setStrokeColorRGB(*self._rgb(self._draw_color))
            self._canvas.line(
                self._pt(x1),
                self._canvas_y(y1),
                self._pt(x2),
                self._canvas_y(y2),
            )

        def output(self, path: str) -> None:
            self._canvas.save()
            with open(path, "wb") as f:
                f.write(self._buffer.getvalue())


def usable_width(pdf: FPDF) -> float:
    return PAGE_W - 2 * MARGIN


def render_inline(pdf: FPDF, text: str, font_size: float = 10):
    """Render a single line of text with **bold** and `code` segments."""
    parts = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text)
    line_h = 5
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            pdf.set_font(FONT_BODY, "B", font_size)
            pdf.write(line_h, _ascii_safe(part[2:-2]))
        elif part.startswith("`") and part.endswith("`"):
            pdf.set_font(FONT_MONO, "", font_size)
            pdf.write(line_h, _ascii_safe(part[1:-1]))
        else:
            pdf.set_font(FONT_BODY, "", font_size)
            pdf.write(line_h, _ascii_safe(part))
    pdf.ln(line_h)


def render_heading(pdf: FPDF, text: str, level: int):
    sizes = {1: 18, 2: 14, 3: 12, 4: 11}
    sz = sizes.get(level, 10)
    spacing_above = {1: 6, 2: 6, 3: 4, 4: 3}.get(level, 2)
    spacing_below = {1: 4, 2: 3, 3: 2, 4: 2}.get(level, 1)
    # Avoid orphan headings: if there isn't enough room for the heading plus
    # a couple of lines of body content, push to the next page.
    min_room_below = {1: 60, 2: 40, 3: 30, 4: 25}.get(level, 20)
    if pdf.get_y() + spacing_above + sz * 0.5 + min_room_below > PAGE_H - MARGIN:
        pdf.add_page()
    elif pdf.get_y() > MARGIN + 4:
        pdf.ln(spacing_above)
    pdf.set_font(FONT_BODY, "B", sz)
    clean = _ascii_safe(text.replace("**", ""))
    pdf.multi_cell(0, sz * 0.5 + 2, clean)
    pdf.ln(spacing_below)


def render_bullet(pdf: FPDF, text: str, indent: int = 0):
    pdf.set_font(FONT_BODY, "", 10)
    indent_mm = 4 + indent * 5
    pdf.set_x(MARGIN + indent_mm)
    bullet_w = 4
    pdf.cell(bullet_w, 5, "-")
    # Use multi_cell for the bullet text body so wrapping works
    avail = usable_width(pdf) - indent_mm - bullet_w
    x_start = pdf.get_x()
    y_start = pdf.get_y()
    pdf.multi_cell(avail, 5, _strip_inline_md(text))
    # Restore left margin for next line
    pdf.set_x(MARGIN)


def render_numbered(pdf: FPDF, num: str, text: str, indent: int = 0):
    pdf.set_font(FONT_BODY, "", 10)
    indent_mm = 4 + indent * 5
    pdf.set_x(MARGIN + indent_mm)
    num_w = 7
    pdf.cell(num_w, 5, f"{num}.")
    avail = usable_width(pdf) - indent_mm - num_w
    pdf.multi_cell(avail, 5, _strip_inline_md(text))
    pdf.set_x(MARGIN)


_UNICODE_FALLBACKS = {
    "—": "-",   # em dash
    "–": "-",   # en dash
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "…": "...",
    "→": "->", "←": "<-",
    "↑": "^", "↓": "v",
    "×": "x",   # multiplication
    "·": "*",   # middle dot
    "±": "+/-",
    " ": " ",
    "°": " deg",
    "µ": "u",
    "≤": "<=", "≥": ">=",
    "≈": "~",
    "«": '"', "»": '"',
    "•": "-",
    "′": "'",
    "γ": "gamma", "λ": "lambda", "π": "pi",
    "ε": "epsilon", "α": "alpha", "β": "beta",
}


def _ascii_safe(text: str) -> str:
    for k, v in _UNICODE_FALLBACKS.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _strip_inline_md(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return _ascii_safe(text)


def render_table(pdf: FPDF, header_cells, rows):
    pdf.set_font(FONT_BODY, "", 9)
    n_cols = len(header_cells)
    if n_cols == 0:
        return
    avail = usable_width(pdf)
    # Equal-width columns; fpdf2 will wrap inside cells
    col_w = avail / n_cols
    line_h = 5

    def estimate_row_height(cells):
        max_lines = 1
        for c in cells:
            text = _strip_inline_md(c)
            # Estimate wrapping
            approx_chars_per_line = max(1, int(col_w / 1.6))
            lines = max(1, sum(
                max(1, (len(seg) + approx_chars_per_line - 1) // approx_chars_per_line)
                for seg in text.split("\n")
            ))
            max_lines = max(max_lines, lines)
        return max_lines * line_h

    # If the header + at least one row wouldn't fit on this page, push to next
    header_h = estimate_row_height(header_cells)
    first_row_h = estimate_row_height(rows[0]) if rows else 0
    if pdf.get_y() + header_h + first_row_h > PAGE_H - MARGIN:
        pdf.add_page()

    # Header
    pdf.set_font(FONT_BODY, "B", 9)
    pdf.set_fill_color(225, 225, 230)
    h = header_h
    _draw_table_row(pdf, header_cells, col_w, h, fill=True)

    pdf.set_font(FONT_BODY, "", 9)
    fill = False
    for row in rows:
        h = estimate_row_height(row)
        if pdf.get_y() + h > PAGE_H - MARGIN:
            pdf.add_page()
            pdf.set_font(FONT_BODY, "B", 9)
            pdf.set_fill_color(225, 225, 230)
            _draw_table_row(pdf, header_cells, col_w, estimate_row_height(header_cells), fill=True)
            pdf.set_font(FONT_BODY, "", 9)
            fill = False
        if fill:
            pdf.set_fill_color(245, 245, 248)
        _draw_table_row(pdf, row, col_w, h, fill=fill)
        fill = not fill
    pdf.ln(2)


def _draw_table_row(pdf: FPDF, cells, col_w: float, row_h: float, fill: bool):
    x_start = pdf.get_x()
    y_start = pdf.get_y()
    for i, cell in enumerate(cells):
        x_cell = x_start + i * col_w
        pdf.set_xy(x_cell, y_start)
        # Border + optional fill
        pdf.cell(col_w, row_h, "", border=1, fill=fill)
        # Render text on top
        pdf.set_xy(x_cell + 1, y_start + 1)
        pdf.multi_cell(col_w - 2, 5, _strip_inline_md(cell))
    pdf.set_xy(x_start, y_start + row_h)


def render_code_block(pdf: FPDF, lines):
    pdf.set_font(FONT_MONO, "", 8)
    pdf.set_fill_color(240, 240, 240)
    line_h = 4
    pad = 1.5
    block_h = len(lines) * line_h + 2 * pad
    if pdf.get_y() + block_h > PAGE_H - MARGIN:
        pdf.add_page()
    avail = usable_width(pdf)
    x_start = MARGIN
    y_start = pdf.get_y()
    pdf.set_xy(x_start, y_start)
    pdf.cell(avail, block_h, "", fill=True)
    pdf.set_xy(x_start + pad, y_start + pad)
    for line in lines:
        safe = _ascii_safe(line.replace("\t", "    "))
        pdf.set_x(x_start + pad)
        pdf.cell(avail - 2 * pad, line_h, safe)
        pdf.ln(line_h)
    pdf.ln(2)


def render_hr(pdf: FPDF):
    y = pdf.get_y() + 1
    pdf.set_draw_color(180, 180, 180)
    pdf.line(MARGIN, y, PAGE_W - MARGIN, y)
    pdf.set_draw_color(0, 0, 0)
    pdf.ln(4)


def parse_table_block(lines):
    """Parse a sequence of lines forming a markdown pipe table. Returns (header, rows)."""
    if len(lines) < 2:
        return None
    if not re.match(r"^\s*\|.*\|\s*$", lines[0]):
        return None
    if not re.match(r"^\s*\|?\s*[-:|\s]+\|?\s*$", lines[1]):
        return None

    def split_row(raw):
        raw = raw.strip()
        if raw.startswith("|"):
            raw = raw[1:]
        if raw.endswith("|"):
            raw = raw[:-1]
        return [c.strip() for c in raw.split("|")]

    header = split_row(lines[0])
    rows = [split_row(l) for l in lines[2:]]
    return header, rows


def parse_markdown(md_text: str):
    """Yield render directives as a stream of dicts."""
    lines = md_text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Fenced code block
        if stripped.startswith("```"):
            block = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            yield {"type": "code", "lines": block}
            continue

        # Heading
        m = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if m:
            level = len(m.group(1))
            yield {"type": "heading", "level": level, "text": m.group(2)}
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^-{3,}\s*$", stripped):
            yield {"type": "hr"}
            i += 1
            continue

        # Pipe table
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\s*\|?[-:|\s]+\|?\s*$", lines[i + 1].strip()):
            tbl_lines = []
            while i < n and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            parsed = parse_table_block(tbl_lines)
            if parsed:
                header, rows = parsed
                yield {"type": "table", "header": header, "rows": rows}
                continue

        # Bullet
        m = re.match(r"^(\s*)-\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            yield {"type": "bullet", "indent": indent, "text": m.group(2)}
            i += 1
            continue

        # Numbered list
        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 3
            yield {"type": "numbered", "indent": indent, "num": m.group(2), "text": m.group(3)}
            i += 1
            continue

        # Blank line
        if stripped == "":
            yield {"type": "blank"}
            i += 1
            continue

        # Paragraph: gather contiguous non-empty lines until blank/structure
        para_lines = [line]
        i += 1
        while i < n:
            nxt = lines[i]
            ns = nxt.strip()
            if (ns == ""
                    or ns.startswith("#")
                    or ns.startswith("```")
                    or ns.startswith("|")
                    or re.match(r"^-{3,}\s*$", ns)
                    or re.match(r"^\s*-\s+", nxt)
                    or re.match(r"^\s*\d+\.\s+", nxt)):
                break
            para_lines.append(nxt)
            i += 1
        yield {"type": "paragraph", "text": " ".join(p.strip() for p in para_lines)}


def render_paragraph(pdf: FPDF, text: str):
    pdf.set_font(FONT_BODY, "", 10)
    avail = usable_width(pdf)
    pdf.set_x(MARGIN)

    parts = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text)
    line_h = 5
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            pdf.set_font(FONT_BODY, "B", 10)
            pdf.write(line_h, _ascii_safe(part[2:-2]))
        elif part.startswith("`") and part.endswith("`"):
            pdf.set_font(FONT_MONO, "", 9)
            pdf.write(line_h, _ascii_safe(part[1:-1]))
        else:
            pdf.set_font(FONT_BODY, "", 10)
            pdf.write(line_h, _ascii_safe(part))
    pdf.ln(line_h)
    pdf.ln(1)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Render an AscensionAI markdown writeup to PDF."
    )
    parser.add_argument(
        "--input",
        "-i",
        default=MD_PATH,
        help="Markdown input path. Defaults to docs/AscensionAI_Technical_Writeup.md.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=PDF_PATH,
        help="PDF output path. Defaults to docs/AscensionAI_Technical_Writeup.pdf.",
    )
    parser.add_argument(
        "--title",
        default="AscensionAI Technical Writeup",
        help="Header title to show on pages after the first.",
    )
    args = parser.parse_args(argv)
    if _FPDF_IMPORT_ERROR is not None and _REPORTLAB_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing dependency: install fpdf2 or reportlab to render PDFs."
        ) from _FPDF_IMPORT_ERROR

    md_path = os.path.abspath(args.input)
    pdf_path = os.path.abspath(args.output)

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    pdf = WriteupPDF(title=args.title)
    pdf.set_font(FONT_BODY, "", 10)

    for directive in parse_markdown(md_text):
        t = directive["type"]
        if t == "heading":
            render_heading(pdf, directive["text"], directive["level"])
        elif t == "paragraph":
            render_paragraph(pdf, directive["text"])
        elif t == "bullet":
            render_bullet(pdf, directive["text"], directive.get("indent", 0))
        elif t == "numbered":
            render_numbered(pdf, directive["num"], directive["text"], directive.get("indent", 0))
        elif t == "table":
            render_table(pdf, directive["header"], directive["rows"])
        elif t == "code":
            render_code_block(pdf, directive["lines"])
        elif t == "hr":
            render_hr(pdf)
        elif t == "blank":
            pdf.ln(2)

    output_dir = os.path.dirname(pdf_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    pdf.output(pdf_path)
    size = os.path.getsize(pdf_path)
    print(f"Wrote {pdf_path} ({size:,} bytes)")


if __name__ == "__main__":
    main()
