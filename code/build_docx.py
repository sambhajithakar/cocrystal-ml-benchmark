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
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
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

    # Tables single-spaced and 10 pt so a wide table still fits the text block.
    for table in d.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    pf = para.paragraph_format
                    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
                    pf.space_before = pf.space_after = Pt(0)
                    pf.first_line_indent = Inches(0)
                    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for run in para.runs:
                        run.font.name = "Times New Roman"
                        run.font.size = Pt(10)
                        run.font.color.rgb = BLACK

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
