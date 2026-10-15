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
from docx.enum.text import WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt

TEXT_WIDTH_IN = 6.0


def to_pandoc_markdown(text: str) -> str:
    text = re.sub(r"<sup>(.*?)</sup>", r"^\1^", text, flags=re.S)
    text = re.sub(r"<sub>(.*?)</sub>", r"~\1~", text, flags=re.S)
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

    for para in d.paragraphs:
        pf = para.paragraph_format
        heading = para.style.name.startswith("Heading")
        if heading:
            pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            pf.space_before, pf.space_after = Pt(12), Pt(6)
            pf.keep_with_next = True
        else:
            pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
            pf.space_after = Pt(0)
        for run in para.runs:
            run.font.name = "Times New Roman"
            if not heading:
                run.font.size = Pt(12)

    # Tables single-spaced and 10 pt so a wide table still fits the text block.
    for table in d.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                    para.paragraph_format.space_after = Pt(0)
                    for run in para.runs:
                        run.font.name = "Times New Roman"
                        run.font.size = Pt(10)

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
