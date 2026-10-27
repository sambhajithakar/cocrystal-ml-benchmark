"""Build submission-ready .docx files from the markdown sources.

Applies the formatting ACS asks for in a submitted manuscript: Times New Roman
12 pt, double spaced, 1 in margins, continuous line numbers (so reviewers can
cite a line), figures embedded at a width that fits the text block, and tables
and captions kept with their content.

HTML super/subscript tags in the source are rewritten to pandoc's own
`^x^` / `~x~` syntax first, because pandoc's docx writer drops raw HTML.

Usage:
    python3 build_docx.py --src ../benchmark-paper/submission/manuscript.md \
                          --out ../benchmark-paper/submission/manuscript.docx
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile

import docx
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

TEXT_WIDTH_IN = 6.0
BLACK = RGBColor(0, 0, 0)
BODY_INDENT_IN = 0.3

# Heading sizes in points. ACS asks for a plain manuscript, so headings are the
# body face in bold at a modest step up, never a theme colour and never a
# different font.
# Pandoc maps the single `#` line to Heading 1, so that is the title and the
# numbered sections arrive as Heading 2.
HEADING_PT = {"Title": 14, "Heading 1": 14, "Heading 2": 13,
              "Heading 3": 12, "Heading 4": 12}


def style_headings(document: docx.Document) -> None:
    """Make every heading style Times New Roman, bold and black."""
    for name, size in HEADING_PT.items():
        try:
            style = document.styles[name]
        except KeyError:
            continue
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = BLACK
        rpr = style.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rfonts.set(qn(attr), "Times New Roman")
        # Word stores a theme colour separately; clear it or it wins.
        for tag in ("w:color",):
            el = rpr.find(qn(tag))
            if el is not None:
                rpr.remove(el)
        color = OxmlElement("w:color")
        color.set(qn("w:val"), "000000")
        rpr.append(color)


def to_pandoc_markdown(text: str) -> str:
    text = re.sub(r"<sup>(.*?)</sup>", r"^\1^", text, flags=re.S)
    text = re.sub(r"<sub>(.*?)</sub>", r"~\1~", text, flags=re.S)
    # Section separators are a reading aid in the markdown source; in Word they
    # become a full-width rule across the page, which looks like a stray line.
    text = re.sub(r"^-{3,}\s*$", "", text, flags=re.M)
    return text


def add_line_numbers(document: docx.Document) -> None:
    """Continuous line numbering, restarting nowhere, counted every line."""
    for section in document.sections:
        sect_pr = section._sectPr
        existing = sect_pr.find(qn("w:lnNumType"))
        if existing is not None:
            sect_pr.remove(existing)
        ln = OxmlElement("w:lnNumType")
        ln.set(qn("w:countBy"), "1")
        ln.set(qn("w:start"), "1")
        ln.set(qn("w:restart"), "continuous")
        sect_pr.append(ln)


def keep_together(text: str) -> str:
    """Stop Word breaking a measurement or a protocol name across two lines."""
    text = text.replace(" \u00b1 ", "\u00a0\u00b1\u00a0")
    # Only short phrases that must stay on one line. A long multi-word heading
    # is left breakable, or Word splits it inside a word instead.
    for phrase in ("anchor held out", "Anchor held out", "top decile",
                   "Top decile", "log P", "log *P*"):
        text = text.replace(phrase, phrase.replace(" ", "\u00a0"))
    return text


def set_table_borders(table) -> None:
    """ACS tables carry horizontal rules only: above and below the header row,
    and under the last row. No vertical rules, no grid."""
    tbl_pr = table._tbl.tblPr
    for existing in tbl_pr.findall(qn("w:tblBorders")):
        tbl_pr.remove(existing)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:" + edge)
        if edge in ("top", "bottom"):
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "8")
            el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "none")
            el.set(qn("w:sz"), "0")
        el.set(qn("w:space"), "0")
        borders.append(el)
    tbl_pr.append(borders)

    # A single rule under the header row.
    if not table.rows:
        return
    for cell in table.rows[0].cells:
        tc_pr = cell._tc.get_or_add_tcPr()
        for existing in tc_pr.findall(qn("w:tcBorders")):
            tc_pr.remove(existing)
        tc_borders = OxmlElement("w:tcBorders")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "8")
        bottom.set(qn("w:color"), "000000")
        bottom.set(qn("w:space"), "0")
        tc_borders.append(bottom)
        tc_pr.append(tc_borders)


# Times New Roman sets digits at exactly half an em, and table cells here are
# mostly digits, so half an em per character is the right estimate.
CHAR_EM = 0.50
CELL_PAD_IN = 0.07


def merge_runs(para) -> None:
    """Put a cell paragraph's text into one run.

    Pandoc splits a cell such as "0.785 ± 0.010" into several runs, so a
    no-break space inserted run by run never spans the gap. Within a table cell
    every run carries the same formatting, so merging them loses nothing.
    """
    if len(para.runs) < 2:
        return
    first = para.runs[0]
    first.text = "".join(run.text for run in para.runs)
    for run in para.runs[1:]:
        run._r.getparent().remove(run._r)


def column_tokens(table, index: int) -> list[str]:
    tokens = []
    for row in table.rows:
        try:
            text = keep_together(row.cells[index].text.strip())
        except IndexError:
            continue
        # Split on ordinary spaces only: str.split() also splits on the
        # no-break space, which would hide exactly the tokens that must not
        # be broken.
        tokens.extend(tok for tok in text.replace(chr(10), ' ').split(' ') if tok)
    return tokens or ["---"]


def format_table(table) -> None:
    """Fit the table to the text block without breaking a word.

    Each column is given at least the width of its longest unbreakable token,
    so a measurement such as "0.785 ± 0.010" is never split across two lines.
    If the columns together need more room than the text block has, the whole
    table steps down in type size instead.
    """
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.left_indent = Inches(0)

    columns = len(table.columns)
    if not columns:
        return

    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                merge_runs(para)

    longest = [max(len(tok) for tok in column_tokens(table, i))
               for i in range(columns)]

    def needed(size_pt: float) -> list[float]:
        # A small safety factor, because a capital or a wide glyph runs above
        # the half-em average and one overrun character breaks the word.
        return [n * size_pt * CHAR_EM * 1.08 / 72.0 + CELL_PAD_IN
                for n in longest]

    size_pt = 9.0
    while sum(needed(size_pt)) > TEXT_WIDTH_IN and size_pt > 6.5:
        size_pt -= 0.25

    req = needed(size_pt)
    spare = TEXT_WIDTH_IN - sum(req)
    if spare > 0:
        # Share the leftover room out in proportion to what each column holds.
        total = float(sum(longest))
        req = [w + spare * n / total for w, n in zip(req, longest)]
    widths = [Inches(w) for w in req]

    for index, column in enumerate(table.columns):
        column.width = widths[index]

    for row_index, row in enumerate(table.rows):
        header = row_index == 0
        if header:
            tr_pr = row._tr.get_or_add_trPr()
            if not tr_pr.findall(qn("w:tblHeader")):
                tr_pr.append(OxmlElement("w:tblHeader"))
        for cell_index, cell in enumerate(row.cells):
            cell.width = widths[min(cell_index, columns - 1)]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.BOTTOM if header \
                else WD_ALIGN_VERTICAL.TOP
            for para in cell.paragraphs:
                pf = para.paragraph_format
                pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
                pf.space_before = pf.space_after = Pt(2)
                pf.first_line_indent = Inches(0)
                pf.left_indent = Inches(0)
                pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in para.runs:
                    run.text = keep_together(run.text)
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(size_pt)
                    run.font.color.rgb = BLACK
                    if header:
                        run.font.bold = True

    set_table_borders(table)


def format_document(path: str, line_numbers: bool = True) -> dict:
    d = docx.Document(path)

    for section in d.sections:
        section.top_margin = section.bottom_margin = Inches(1)
        section.left_margin = section.right_margin = Inches(1)

    normal = d.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(attr), "Times New Roman")

    style_headings(d)

    # Pandoc maps the document's single `#` line to Heading 1, so the first
    # Heading 1 is the title and everything before the first Heading 2 is the
    # author and affiliation block.
    paragraphs = d.paragraphs
    title_index = next((i for i, q in enumerate(paragraphs)
                        if q.style.name == "Heading 1"), None)
    front_end = next((i for i, q in enumerate(paragraphs)
                      if q.style.name.startswith("Heading")
                      and i != title_index), len(paragraphs))

    after_heading = True
    for index, para in enumerate(paragraphs):
        pf = para.paragraph_format
        name = para.style.name
        heading = name.startswith("Heading")
        title = name == "Title"
        has_image = bool(para._p.findall(".//" + qn("w:drawing")))
        listed = bool(para._p.findall(".//" + qn("w:numPr")))
        text = para.text.strip()
        # A caption is the bold "Figure N." or "Table N." line under a float.
        caption = text[:7].rstrip(".").rstrip() in ("Figure", "Table") and (
            text.startswith("Figure") or text.startswith("Table"))

        if index == title_index:
            title, heading = True, False
        front = title_index is not None and title_index < index < front_end

        if title:
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            pf.space_before, pf.space_after = Pt(0), Pt(12)
            pf.keep_with_next = True
        elif heading:
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            pf.space_before, pf.space_after = Pt(14), Pt(6)
            pf.first_line_indent = Inches(0)
            pf.keep_with_next = True
        elif listed:
            # A numbered or bulleted item carries its own indent from the list
            # definition; adding the body indent on top of it pushes the text
            # out of line with everything around it.
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
            pf.space_before = pf.space_after = Pt(0)
            pf.left_indent = Inches(0.5)
            pf.first_line_indent = Inches(-0.25)
        elif front:
            # Authors, affiliations and the corresponding-author line.
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            pf.space_before, pf.space_after = Pt(0), Pt(6)
            pf.first_line_indent = Inches(0)
        elif has_image:
            pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            pf.space_before, pf.space_after = Pt(12), Pt(6)
            pf.first_line_indent = Inches(0)
            pf.keep_with_next = True
        elif caption:
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            pf.space_before, pf.space_after = Pt(0), Pt(12)
            pf.first_line_indent = Inches(0)
        else:
            # Justifying a line that holds a URL, a DOI or a commit hash opens
            # wide gaps between the other words, because the long token cannot
            # be broken. Such a paragraph is set flush left instead.
            long_token = any(len(tok) > 26 for tok in text.split())
            pf.alignment = (WD_ALIGN_PARAGRAPH.LEFT if long_token
                            else WD_ALIGN_PARAGRAPH.JUSTIFY)
            pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
            pf.space_before, pf.space_after = Pt(0), Pt(0)
            # Indent the first line of a running paragraph, so a reader can see
            # where one paragraph ends in double-spaced text. The paragraph that
            # opens a section is not indented, which is the usual convention.
            label = text.startswith(("Keywords", "Note", "Figure", "Table"))
            pf.first_line_indent = Inches(0 if after_heading or not text or label
                                          else BODY_INDENT_IN)

        # Latent heading styles are not always in the style table, so the
        # heading face is also set on the runs themselves.
        if title:
            size, bold = Pt(HEADING_PT["Title"]), True
        elif heading:
            size, bold = Pt(HEADING_PT.get(name, 12)), True
        else:
            size, bold = Pt(12), None

        for run in para.runs:
            run.font.name = "Times New Roman"
            run.font.color.rgb = BLACK
            run.font.size = size
            if bold:
                run.font.bold = True

        if text:
            after_heading = heading or title or front or has_image or caption

    for table in d.tables:
        format_table(table)

    # Give the paragraph that follows a table room to breathe, so the closing
    # rule does not sit on the next line of text.
    body = d.element.body
    for index, child in enumerate(body):
        if child.tag == qn("w:p") and index and body[index - 1].tag == qn("w:tbl"):
            docx.text.paragraph.Paragraph(child, d).paragraph_format.space_before = Pt(12)

    for shape in d.inline_shapes:
        if shape.width > Inches(TEXT_WIDTH_IN):
            ratio = Inches(TEXT_WIDTH_IN) / shape.width
            shape.width = int(shape.width * ratio)
            shape.height = int(shape.height * ratio)

    if line_numbers:
        add_line_numbers(d)

    d.save(path)
    return {
        "paragraphs": len(d.paragraphs),
        "tables": len(d.tables),
        "images": len(d.inline_shapes),
        "words": sum(len(p.text.split()) for p in d.paragraphs),
        "placeholders": sum(1 for p in d.paragraphs if "[" in p.text and "]" in p.text
                            and p.text.strip().startswith("[")),
    }


def build(src: str, out: str, line_numbers: bool = True) -> dict:
    with open(src, encoding="utf-8") as fh:
        text = to_pandoc_markdown(fh.read())
    workdir = os.path.dirname(os.path.abspath(src))
    with tempfile.NamedTemporaryFile("w", suffix=".md", dir=workdir,
                                     delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    try:
        subprocess.run(["pandoc", os.path.basename(tmp_path), "-o",
                        os.path.abspath(out)], cwd=workdir, check=True)
    finally:
        os.unlink(tmp_path)
    return format_document(out, line_numbers=line_numbers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-line-numbers", action="store_true")
    args = ap.parse_args()
    stats = build(args.src, args.out, line_numbers=not args.no_line_numbers)
    print(f"[done] {args.out}")
    for k, val in stats.items():
        print(f"       {k}: {val}")


if __name__ == "__main__":
    main()
