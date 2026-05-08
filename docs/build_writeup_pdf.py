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

import os
import re
from fpdf import FPDF

DOC_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(DOC_DIR, "AscensionAI_Technical_Writeup.md")
PDF_PATH = os.path.join(DOC_DIR, "AscensionAI_Technical_Writeup.pdf")

PAGE_W = 210      # A4 width mm
PAGE_H = 297
MARGIN = 18

FONT_BODY = "Helvetica"
FONT_MONO = "Courier"


class WriteupPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(auto=True, margin=MARGIN)
        self.add_page()

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font(FONT_BODY, "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "AscensionAI Technical Writeup", align="L")
        self.cell(0, 6, f"Page {self.page_no()}", align="R")
        self.set_text_color(0, 0, 0)
        self.ln(8)

    def footer(self):
        pass


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
    if pdf.get_y() > MARGIN + 4:
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

    # Header
    pdf.set_font(FONT_BODY, "B", 9)
    pdf.set_fill_color(225, 225, 230)
    h = estimate_row_height(header_cells)
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


def main():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    pdf = WriteupPDF()
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

    pdf.output(PDF_PATH)
    size = os.path.getsize(PDF_PATH)
    print(f"Wrote {PDF_PATH} ({size:,} bytes)")


if __name__ == "__main__":
    main()
