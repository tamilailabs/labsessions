import os
import re
import subprocess
from bs4 import BeautifulSoup
from lxml import etree
import torch
from transformers import MarianMTModel, MarianTokenizer

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r""
PDF_FILE = ""
HTML_FILE = ""
XML_FILE = ""
OUT_PDF  = ""

FONTS_DIR = os.path.join(BASE_DIR, "fonts")
PDF_W, PDF_H = 612, 792

TARGET_LANG = "es"          # "de", "fr", "es"
PAGES_TO_TRANSLATE = 15
TRANSLATE_PAGE_LIST = None
# ============================================================

LANG_MODELS = {
    "de": "Helsinki-NLP/opus-mt-en-de",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
}

# ============================================================
# UTILS
# ============================================================
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)

def pages_to_translate():
    if TRANSLATE_PAGE_LIST:
        return set(TRANSLATE_PAGE_LIST)
    return set(range(1, PAGES_TO_TRANSLATE + 1))

# ============================================================
# TRANSLATOR
# ============================================================
def load_translator():
    name = LANG_MODELS[TARGET_LANG]
    tok = MarianTokenizer.from_pretrained(name)
    mdl = MarianMTModel.from_pretrained(name)
    mdl.eval()
    return tok, mdl

def translate_lines(lines, tok, mdl):
    if not lines:
        return lines
    with torch.no_grad():
        batch = tok(lines, return_tensors="pt", padding=True, truncation=True)
        out = mdl.generate(**batch)
        return [tok.decode(o, skip_special_tokens=True) for o in out]

# ============================================================
# PDF → HTML
# ============================================================
def pdf_to_html():
    os.chdir(BASE_DIR)
    run(["pdftohtml", "-c", "-hidden", "-noframes", PDF_FILE, HTML_FILE])

# ============================================================
# HTML → XML (SINGLE XML, ALL PAGES)
# ============================================================
RE_SIZE   = re.compile(r"font-size:([0-9.]+)px")
RE_COLOR  = re.compile(r"color:(#[0-9a-fA-F]{6})")
RE_MATRIX = re.compile(r"matrix\(([^)]+)\)")

def build_single_xml():
    soup = BeautifulSoup(open(os.path.join(BASE_DIR, HTML_FILE), encoding="utf-8"), "html.parser")

    style_block = soup.find_all("style")[1].text
    styles = {}
    for blk in style_block.split("}"):
        if not blk.strip().startswith(".ft"):
            continue
        cls, body = blk.split("{")
        cls = cls.strip()[1:]
        styles[cls] = {
            "size": float(RE_SIZE.search(body).group(1)),
            "color": RE_COLOR.search(body).group(1),
            "matrix": tuple(map(float, RE_MATRIX.search(body).group(1).split(",")))
                     if RE_MATRIX.search(body) else None
        }

    root = etree.Element("document", source=PDF_FILE)

    pages = soup.find_all("div", id=lambda x: x and x.endswith("-div"))
    for page_no, page in enumerate(pages, start=1):
        w = float(page["style"].split("width:")[1].split("px")[0])
        h = float(page["style"].split("height:")[1].split("px")[0])

        p_el = etree.SubElement(root, "page",
                                number=str(page_no),
                                width=str(w),
                                height=str(h))

        img = page.find("img")
        if img:
            etree.SubElement(
                p_el, "image",
                src=img["src"],
                width=img["width"],
                height=img["height"]
            )

        for p in page.find_all("p"):
            cls = p.get("class", [""])[0]
            if cls not in styles:
                continue
            style = p["style"]
            top  = style.split("top:")[1].split("px")[0]
            left = style.split("left:")[1].split("px")[0]

            t_el = etree.SubElement(
                p_el, "text",
                cls=cls,
                top=top,
                left=left
            )
            for ln in p.decode_contents().replace("&nbsp;", " ").split("<br/>"):
                etree.SubElement(t_el, "line").text = ln

    etree.ElementTree(root).write(
        os.path.join(BASE_DIR, XML_FILE),
        pretty_print=True,
        encoding="utf-8",
        xml_declaration=True
    )
    return styles

# ============================================================
# XML → SINGLE PDF
# ============================================================
def register_fonts():
    pdfmetrics.registerFont(TTFont("Serif", os.path.join(FONTS_DIR, "LiberationSerif-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("SerifBold", os.path.join(FONTS_DIR, "LiberationSerif-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", os.path.join(FONTS_DIR, "LiberationMono-Regular.ttf")))

def font_for(cls):
    if cls in {"ft01", "ft02", "ft06"}:
        return "SerifBold"
    if cls == "ft05":
        return "Mono"
    return "Serif"

def baseline(font, size):
    a = pdfmetrics.getAscent(font) / 1000 * size
    d = abs(pdfmetrics.getDescent(font)) / 1000 * size
    return a + d

def render_single_pdf(styles, tok, mdl):
    tree = etree.parse(os.path.join(BASE_DIR, XML_FILE))
    root = tree.getroot()

    translate_set = pages_to_translate()
    c = canvas.Canvas(os.path.join(BASE_DIR, OUT_PDF), pagesize=(PDF_W, PDF_H))

    for page in root.findall("page"):
        page_no = int(page.get("number"))
        translate = page_no in translate_set

        html_w = float(page.get("width"))
        html_h = float(page.get("height"))
        sx = PDF_W / html_w
        sy = PDF_H / html_h

        img = page.find("image")
        if img is not None:
            img_path = os.path.join(BASE_DIR, img.get("src"))
            iw = float(img.get("width")) * sx
            ih = float(img.get("height")) * sy
            c.drawImage(ImageReader(img_path), 0, PDF_H - ih, iw, ih)

        for t in page.findall("text"):
            cls = t.get("cls")
            st = styles[cls]
            font = font_for(cls)
            size = st["size"] * sx
            step = baseline(font, size)

            x = float(t.get("left")) * sx
            y = PDF_H - (float(t.get("top")) * sy)

            lines = [ln.text or "" for ln in t.findall("line")]
            if translate:
                lines = translate_lines(lines, tok, mdl)

            c.setFont(font, size)
            c.setFillColor(HexColor(st["color"]))

            if st["matrix"]:
                a,b,c_,d,e,f = st["matrix"]
                c.saveState()
                c.translate(x, y)
                c.transform(a, b, c_, d, e * sx, f * sy)
                w = max(c.stringWidth(l, font, size) for l in lines)
                if abs(b + 1) < 1e-3:
                    c.translate(-w, 0)
                for i,l in enumerate(lines):
                    c.drawString(0, -i * step, l)
                c.restoreState()
            else:
                for i,l in enumerate(lines):
                    c.drawString(x, y - i * step, l)

        c.showPage()

    c.save()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    register_fonts()
    pdf_to_html()
    styles = build_single_xml()
    tok, mdl = load_translator()
    render_single_pdf(styles, tok, mdl)

    print("✅ Outputs created:")
    print(" -", XML_FILE)
    print(" -", OUT_PDF)
