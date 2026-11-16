#!/usr/bin/env python
"""Build the README hero figure of `auditable` as an editable PowerPoint file.

Run from anywhere:

    python figure-src/hero/build_hero.py

Outputs (next to this file unless noted):
    hero.pptx             editable master (native shapes, connectors, chart)
    hero.pdf              exported through desktop PowerPoint (macOS)
    ../../assets/hero.png README display export (about 2600 px wide, white)

Requirements: python-pptx 1.0.x, PyMuPDF, Pillow, and desktop PowerPoint on macOS for
the PDF export (the build still writes hero.pptx without it; pass --no-export).

Composition: primary layout from concept A, shallow-3D trace plate and the
Leaderboards panel borrowed from concept B (see README.md in this folder).
All coordinates are inches on a 13.333 x 6.15 canvas.
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.dml import MSO_LINE, MSO_PATTERN
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PPTX = HERE / "hero.pptx"
PDF = HERE / "hero.pdf"
PNG = REPO / "assets" / "hero.png"
GOLDEN = REPO / "src" / "auditable" / "bench" / "golden" / "detect__swegym.json"

NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
NSDECL = f'xmlns:a="{NS_A}" xmlns:p="{NS_P}"'

# --------------------------------------------------------------------------
# Palette (lab default) and fonts
# --------------------------------------------------------------------------
PRE, PRE_S = "ECB66C", "B8862F"
LIVE, LIVE_S = "4198AC", "2B6E7E"
POST, POST_S = "ED8D5A", "B66037"
MINT, MINT_S = "BFDFD2", "365E49"
INK, SEC, RULE = "202923", "67756D", "DCE6DF"
WHITE = "FFFFFF"
OK_GREEN = "4C8A6E"
GRAY_BAR = "9AA5A0"

FONT = "Arial"
MONO = "Consolas"  # bundled with Office for Mac and Windows


def mix(hex_color: str, t: float, other: str = WHITE) -> str:
    """Blend hex_color toward `other` by fraction t (0 = color, 1 = other)."""
    a = [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(other[i:i + 2], 16) for i in (0, 2, 4)]
    return "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


PRE_T1, PRE_T2 = mix(PRE, 0.78), mix(PRE, 0.55)
LIVE_T1, LIVE_T2 = mix(LIVE, 0.82), mix(LIVE, 0.55)
POST_T1, POST_T2 = mix(POST, 0.84), mix(POST, 0.55)
MINT_T1 = mix(MINT, 0.5)
CODE_FILL, CODE_LINE = "F1F5F3", "C5D3CA"

STAGE = {
    "PRE": dict(c=PRE, s=PRE_S, t1=PRE_T1, t2=PRE_T2),
    "LIVE": dict(c=LIVE, s=LIVE_S, t1=LIVE_T1, t2=LIVE_T2),
    "POST": dict(c=POST, s=POST_S, t1=POST_T1, t2=POST_T2),
}


# --------------------------------------------------------------------------
# Primitive helpers
# --------------------------------------------------------------------------
def E(v: float) -> Emu:
    return Emu(int(round(v * 914400)))


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h.upper())


def strip_style(shape) -> None:
    """Remove the theme style reference so no theme shadow or outline leaks in."""
    el = shape._element
    for st in el.findall(f"{{{NS_P}}}style"):
        el.remove(st)


def spPr_of(shape):
    return shape._element.spPr


def _geom(spPr):
    for tag in ("prstGeom", "custGeom"):
        g = spPr.find(f"{{{NS_A}}}{tag}")
        if g is not None:
            return g
    return None


def set_fill_xml(shape, xml: str) -> None:
    spPr = spPr_of(shape)
    for tag in ("noFill", "solidFill", "gradFill", "pattFill", "blipFill"):
        for el in spPr.findall(f"{{{NS_A}}}{tag}"):
            spPr.remove(el)
    new = parse_xml(xml.replace("<a:", "<a:", 1))
    _geom(spPr).addnext(new)


def fill_solid(shape, color: str | None, alpha: float | None = None) -> None:
    if color is None:
        set_fill_xml(shape, f'<a:noFill {NSDECL}/>')
        return
    a = f'<a:alpha val="{int(alpha * 100000)}"/>' if alpha is not None else ""
    set_fill_xml(shape, f'<a:solidFill {NSDECL}><a:srgbClr val="{color}">{a}</a:srgbClr></a:solidFill>')


def fill_linear(shape, c1: str, c2: str, angle_deg: float = 90) -> None:
    set_fill_xml(
        shape,
        f'<a:gradFill {NSDECL} rotWithShape="1"><a:gsLst>'
        f'<a:gs pos="0"><a:srgbClr val="{c1}"/></a:gs>'
        f'<a:gs pos="100000"><a:srgbClr val="{c2}"/></a:gs></a:gsLst>'
        f'<a:lin ang="{int(angle_deg * 60000)}" scaled="0"/></a:gradFill>',
    )


def fill_sphere(shape, light: str, base: str) -> None:
    """Radial gradient with the highlight at the upper left (a shaded bead)."""
    set_fill_xml(
        shape,
        f'<a:gradFill {NSDECL} rotWithShape="1"><a:gsLst>'
        f'<a:gs pos="0"><a:srgbClr val="{light}"/></a:gs>'
        f'<a:gs pos="100000"><a:srgbClr val="{base}"/></a:gs></a:gsLst>'
        f'<a:path path="circle"><a:fillToRect l="32000" t="28000" r="68000" b="72000"/></a:path></a:gradFill>',
    )


def line(shape, color: str | None, width: float = 1.0, dash: str | None = None,
         head: str | None = None, tail: str | None = None, cap: str | None = None) -> None:
    """Explicit outline. `tail` is the arrowhead at the END of a connector or arc."""
    if color is None:
        shape.line.fill.background()
        return
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(width)
    if dash:
        shape.line.dash_style = getattr(MSO_LINE, dash)
    ln = shape.line._get_or_add_ln()
    if cap:
        ln.set("cap", cap)
    for tag, kind in (("headEnd", head), ("tailEnd", tail)):
        for el in ln.findall(f"{{{NS_A}}}{tag}"):
            ln.remove(el)
        if kind:
            ln.append(parse_xml(f'<a:{tag} {NSDECL} type="{kind}" w="med" len="med"/>'))


def add_shape(shapes, kind, x, y, w, h, name, fill=None, outline=None, lw=1.0, dash=None,
              adj=None, rot=None, flip_h=False, flip_v=False):
    shp = shapes.add_shape(kind, E(x), E(y), E(w), E(h))
    strip_style(shp)
    shp.name = name
    if adj:
        for i, v in enumerate(adj):
            shp.adjustments[i] = v
    fill_solid(shp, fill)
    line(shp, outline, lw, dash)
    if rot:
        shp.rotation = rot
    if flip_h or flip_v:
        xfrm = spPr_of(shp).find(f"{{{NS_A}}}xfrm")
        if flip_h:
            xfrm.set("flipH", "1")
        if flip_v:
            xfrm.set("flipV", "1")
    return shp


def rrect(shapes, x, y, w, h, name, fill=None, outline=None, lw=1.0, r=0.1, dash=None):
    rr = min(0.5, r / min(w, h))
    return add_shape(shapes, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, name, fill, outline, lw,
                     dash, adj=[rr])


def ellipse(shapes, cx, cy, d, name, fill=None, outline=None, lw=1.0, dash=None):
    return add_shape(shapes, MSO_SHAPE.OVAL, cx - d / 2, cy - d / 2, d, d, name, fill, outline, lw, dash)


Runs = list  # a paragraph is a str or a list of (text, opts) runs


def set_text(shape, paras, size=14, color=INK, bold=False, italic=False, font=FONT,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margins=(0, 0, 0, 0), wrap=True,
             spacing=None):
    tf = shape.text_frame
    tf.word_wrap = wrap
    tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = (E(m) for m in margins)
    tf.vertical_anchor = anchor
    bodyPr = tf._txBody.find(f"{{{NS_A}}}bodyPr")
    for el in list(bodyPr):
        bodyPr.remove(el)  # no autofit element: fixed geometry
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.line_spacing = spacing
        runs = [(para, {})] if isinstance(para, str) else para
        for text, o in runs:
            r = p.add_run()
            r.text = text
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
            rPr = r._r.get_or_add_rPr()
            if o.get("sub"):
                rPr.set("baseline", "-25000")
            # East Asian / complex-script fallbacks use the same face
            for tag in ("ea", "cs"):
                rPr.append(parse_xml(f'<a:{tag} {NSDECL} typeface="{o.get("font", font)}"/>'))
    return shape


def textbox(shapes, x, y, w, h, paras, name, **kw):
    tb = shapes.add_textbox(E(x), E(y), E(w), E(h))
    tb.name = name
    set_text(tb, paras, **kw)
    return tb


def pill(shapes, x, y, w, h, text, name, fill, outline, size=13, font=FONT, bold=False,
         color=INK, lw=1.25, r=None):
    p = rrect(shapes, x, y, w, h, name, fill, outline, lw, r=h / 2 if r is None else r)
    set_text(p, text, size=size, font=font, bold=bold, color=color, wrap=False,
             margins=(0.05, 0, 0.05, 0))
    return p


# Connection-site indices per preset (OOXML presetShapeDefinitions order).
SITE = {
    "rect": dict(top=0, left=1, bottom=2, right=3),
    "ellipse": dict(top=0, tl=1, left=2, bl=3, bottom=4, br=5, right=6, tr=7),
}


def site_xy(shape, which: str):
    x, y, w, h = (v / 914400 for v in (shape.left, shape.top, shape.width, shape.height))
    cx, cy = x + w / 2, y + h / 2
    if which == "top":
        return cx, y
    if which == "bottom":
        return cx, y + h
    if which == "left":
        return x, cy
    if which == "right":
        return x + w, cy
    r = min(w, h) / 2 * math.sqrt(0.5)
    return {"tl": (cx - r, cy - r), "tr": (cx + r, cy - r),
            "bl": (cx - r, cy + r), "br": (cx + r, cy + r)}[which]


def site_idx(shape, which):
    try:
        kind = "ellipse" if shape.auto_shape_type == MSO_SHAPE.OVAL else "rect"
    except ValueError:  # a text box is a rectangle
        kind = "rect"
    return SITE[kind][which]


def connect(shapes, a, a_site, b, b_site, name, color=INK, lw=1.5, tail="triangle", dash=None,
            head=None):
    """Straight connector glued to a site on each shape (follows the nodes when moved)."""
    x1, y1 = site_xy(a, a_site)
    x2, y2 = site_xy(b, b_site)
    c = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, E(x1), E(y1), E(x2), E(y2))
    strip_style(c)
    c.name = name
    line(c, color, lw, dash, head=head, tail=tail)
    c._connect_begin_to(a, site_idx(a, a_site))
    c._connect_end_to(b, site_idx(b, b_site))
    return c


def polyline(shapes, pts, name, color, lw=1.5, closed=False, fill=None, cap="rnd", tail=None):
    """Native freeform through straight segments (points in inches)."""
    fb = shapes.build_freeform(E(pts[0][0]), E(pts[0][1]), scale=1.0)
    fb.add_line_segments([(E(x), E(y)) for x, y in pts[1:]], close=closed)
    shp = fb.convert_to_shape()
    strip_style(shp)
    shp.name = name
    fill_solid(shp, fill)
    line(shp, color, lw, cap=cap, tail=tail)
    if color:
        ln = shp.line._get_or_add_ln()
        ln.append(parse_xml(f'<a:round {NSDECL}/>')) if ln.find(f"{{{NS_A}}}round") is None else None
    return shp


def replace_geometry(shape, path_xml: str, w_emu: int, h_emu: int):
    spPr = spPr_of(shape)
    g = _geom(spPr)
    cust = parse_xml(
        f'<a:custGeom {NSDECL}><a:avLst/><a:gdLst/><a:ahLst/><a:cxnLst/>'
        f'<a:rect l="0" t="0" r="r" b="b"/><a:pathLst>{path_xml}</a:pathLst></a:custGeom>'
    )
    g.addprevious(cust)
    spPr.remove(g)


def bezier(shapes, p0, c1, c2, p1, name, color, lw=2.0, tail=None, dash=None):
    """Open cubic Bezier as a custom-geometry shape (four editable control points)."""
    xs = [p[0] for p in (p0, c1, c2, p1)]
    ys = [p[1] for p in (p0, c1, c2, p1)]
    x0, y0 = min(xs), min(ys)
    w, h = max(max(xs) - x0, 0.01), max(max(ys) - y0, 0.01)
    shp = add_shape(shapes, MSO_SHAPE.RECTANGLE, x0, y0, w, h, name)
    pt = lambda p: f'<a:pt x="{int((p[0] - x0) * 914400)}" y="{int((p[1] - y0) * 914400)}"/>'
    path = (f'<a:path w="{int(w * 914400)}" h="{int(h * 914400)}" fill="none"><a:moveTo>{pt(p0)}</a:moveTo>'
            f'<a:cubicBezTo>{pt(c1)}{pt(c2)}{pt(p1)}</a:cubicBezTo></a:path>')
    replace_geometry(shp, path, int(w * 914400), int(h * 914400))
    fill_solid(shp, None)
    line(shp, color, lw, dash, tail=tail, cap="rnd")
    return shp


def area_under(shapes, pts_curve, base_y, name, fill):
    """Closed shape under a smooth curve given as [p0, c1, c2, p1]; used for the alarm curve."""
    p0, c1, c2, p1 = pts_curve
    xs = [p[0] for p in pts_curve]
    ys = [p[1] for p in pts_curve] + [base_y]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0, max(ys) - y0
    shp = add_shape(shapes, MSO_SHAPE.RECTANGLE, x0, y0, w, h, name)
    pt = lambda p: f'<a:pt x="{int((p[0] - x0) * 914400)}" y="{int((p[1] - y0) * 914400)}"/>'
    path = (f'<a:path w="{int(w * 914400)}" h="{int(h * 914400)}"><a:moveTo>{pt(p0)}</a:moveTo>'
            f'<a:cubicBezTo>{pt(c1)}{pt(c2)}{pt(p1)}</a:cubicBezTo>'
            f'<a:lnTo>{pt((p1[0], base_y))}</a:lnTo><a:lnTo>{pt((p0[0], base_y))}</a:lnTo><a:close/></a:path>')
    replace_geometry(shp, path, int(w * 914400), int(h * 914400))
    fill_solid(shp, fill)
    line(shp, None)
    return shp


def ring_segment(shapes, cx, cy, R, r, a0, a1, name, fill, outline):
    """Annulus sector (angles in degrees, clockwise from +x on screen) as custom geometry
    with a tight bounding box, built from true arcs."""
    pts = [(cx + R * math.cos(math.radians(a)), cy + R * math.sin(math.radians(a))) for a in (a0, a1)]
    pts += [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in (a0, a1)]
    arc_pts = []
    for rad in (R, r):
        for a in range(int(a0), int(a1) + 1, 1):
            arc_pts.append((cx + rad * math.cos(math.radians(a)), cy + rad * math.sin(math.radians(a))))
    xs, ys = [p[0] for p in arc_pts], [p[1] for p in arc_pts]
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0, max(ys) - y0
    shp = add_shape(shapes, MSO_SHAPE.RECTANGLE, x0, y0, w, h, name)
    P = lambda x, y: f'<a:pt x="{int(round((x - x0) * 914400))}" y="{int(round((y - y0) * 914400))}"/>'
    sw = int(round((a1 - a0) * 60000))
    path = (
        f'<a:path w="{int(round(w * 914400))}" h="{int(round(h * 914400))}">'
        f'<a:moveTo>{P(*pts[0])}</a:moveTo>'
        f'<a:arcTo wR="{int(R * 914400)}" hR="{int(R * 914400)}" stAng="{int(a0 * 60000)}" swAng="{sw}"/>'
        f'<a:lnTo>{P(*pts[3])}</a:lnTo>'
        f'<a:arcTo wR="{int(r * 914400)}" hR="{int(r * 914400)}" stAng="{int(a1 * 60000)}" swAng="{-sw}"/>'
        f'<a:close/></a:path>'
    )
    replace_geometry(shp, path, int(w * 914400), int(h * 914400))
    fill_solid(shp, fill)
    line(shp, outline, 1.0)
    return shp


def dep_curve(shapes, p0, p1, h, name, color, lw=1.5):
    """Dependency curve from p0 to p1 bulging up by about 0.75 h, arrowhead at p1.
    A freeform cubic Bezier (custom geometry): four editable points, not attached to the beads."""
    dx = p1[0] - p0[0]
    c1 = (p0[0] + 0.15 * dx, p0[1] - h)
    c2 = (p1[0] - 0.15 * dx, p1[1] - h)
    return bezier(shapes, p0, c1, c2, p1, name, color, lw, tail="triangle")


def bezier_crossing(curve, y):
    """Point where the cubic Bezier [p0, c1, c2, p1] first reaches height y (screen y grows downward)."""
    p0, c1, c2, p1 = curve
    pt = lambda t: tuple((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d
                         for a, b, c, d in zip(p0, c1, c2, p1))
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if pt(mid)[1] > y:      # still below the threshold line
            lo = mid
        else:
            hi = mid
    return pt((lo + hi) / 2)


def freeform_poly(shapes, pts, name, fill, outline, lw=1.25):
    fb = shapes.build_freeform(E(pts[0][0]), E(pts[0][1]), scale=1.0)
    fb.add_line_segments([(E(x), E(y)) for x, y in pts[1:]], close=True)
    shp = fb.convert_to_shape()
    strip_style(shp)
    shp.name = name
    fill_solid(shp, fill)
    line(shp, outline, lw)
    return shp


def group(slide, name):
    g = slide.shapes.add_group_shape()
    g.name = name
    return g


def group_in(shapes, name):
    """Nested group inside a region group."""
    g = shapes.add_group_shape()
    g.name = name
    return g


# --------------------------------------------------------------------------
# Geometry (inches)
# --------------------------------------------------------------------------
W, H = 13.333, 6.15          # H is 0.15 above the first draft: air under the footer pill
YC = 2.90                     # trace centerline (beads, run_shell chip, fork root)
PLATE_X0, PLATE_X1 = 0.12, 6.92
PLATE_Y0, PLATE_Y1 = 2.50, 3.50
PLATE_DEPTH = 0.13

CARD = dict(x=0.35, y=1.05, w=2.75, h=2.33)
PRE_CX = CARD["x"] + CARD["w"] / 2
LIVE_X0, LIVE_X1 = 3.10, 6.75
LIVE_CX = (LIVE_X0 + LIVE_X1) / 2
POST_PANEL = dict(x=7.15, y=1.00, w=3.00, h=4.45)
POST_CX = POST_PANEL["x"] + POST_PANEL["w"] / 2
LB = dict(x=10.27, y=0.08, w=2.94, h=5.37)
Y_LABEL = 3.76               # top of the PRE and LIVE label rows
PAD_PX = 30                  # white margin added around the README PNG


# --------------------------------------------------------------------------
# Regions
# --------------------------------------------------------------------------
def build_banners(slide):
    g = group(slide, "Banners")
    S = g.shapes
    bh, by = 0.82, 0.08
    specs = [
        ("PRE", "before the run", 0.12, 3.05, MSO_SHAPE.PENTAGON, INK),
        ("LIVE", "while it runs", 2.99, 6.95, MSO_SHAPE.CHEVRON, WHITE),
        ("POST", "after it ends", 6.89, 10.15, MSO_SHAPE.CHEVRON, INK),
    ]
    tip = 0.30
    for key, sub, x0, x1, kind, tcolor in specs:
        st = STAGE[key]
        b = add_shape(S, kind, x0, by, x1 - x0, bh, f"banner_{key}", st["c"], st["s"], 1.5,
                      adj=[tip / bh])
        badge_cx = x0 + (0.40 if key == "PRE" else 0.58)
        text_x = badge_cx + 0.37 + 0.12
        # a chevron already insets its text rectangle by the notch depth
        left = text_x - x0 - (tip if kind == MSO_SHAPE.CHEVRON else 0)
        set_text(b, [[(key, dict(size=24, bold=True))], [(sub, dict(size=20))]], color=tcolor,
                 align=PP_ALIGN.LEFT, margins=(left, 0, 0.1, 0), wrap=False)
        b.text_frame.paragraphs[0].line_spacing = 0.9
        b.text_frame.paragraphs[1].line_spacing = 0.9
        cy = by + bh / 2
        ellipse(S, badge_cx, cy, 0.74, f"badge_{key}", WHITE, st["s"], 3.0)
        {"PRE": icon_clipboard, "LIVE": icon_pulse, "POST": icon_target}[key](S, badge_cx, cy)
    return g


def icon_clipboard(S, cx, cy):
    rrect(S, cx - 0.15, cy - 0.17, 0.30, 0.37, "icon_PRE_board", WHITE, PRE_S, 1.75, r=0.04)
    rrect(S, cx - 0.07, cy - 0.21, 0.14, 0.08, "icon_PRE_clip", PRE, PRE_S, 1.25, r=0.025)
    for i, yy in enumerate((cy - 0.07, cy + 0.04, cy + 0.15)):
        ellipse(S, cx - 0.08, yy, 0.04, f"icon_PRE_dot{i}", PRE_S)
        add_shape(S, MSO_SHAPE.RECTANGLE, cx - 0.04, yy - 0.0125, 0.13, 0.025, f"icon_PRE_row{i}", mix(PRE_S, 0.35))
    ellipse(S, cx + 0.12, cy + 0.15, 0.19, "icon_PRE_check_disc", OK_GREEN, WHITE, 1.25)
    polyline(S, [(cx + 0.07, cy + 0.15), (cx + 0.105, cy + 0.185), (cx + 0.17, cy + 0.115)],
             "icon_PRE_check", WHITE, 1.75)


def icon_pulse(S, cx, cy):
    pts = [(-0.24, 0.0), (-0.12, 0.0), (-0.07, -0.12), (-0.01, 0.15), (0.05, -0.19), (0.10, 0.04), (0.14, 0.0), (0.24, 0.0)]
    polyline(S, [(cx + x, cy + y) for x, y in pts], "icon_LIVE_pulse", LIVE_S, 2.5)


def icon_target(S, cx, cy):
    ellipse(S, cx, cy, 0.46, "icon_POST_ring1", WHITE, POST_S, 2.0)
    ellipse(S, cx, cy, 0.31, "icon_POST_ring2", POST, POST_S, 1.0)
    ellipse(S, cx, cy, 0.17, "icon_POST_ring3", WHITE, POST_S, 1.0)
    ellipse(S, cx, cy, 0.07, "icon_POST_bullseye", POST_S)
    polyline(S, [(cx, cy), (cx + 0.21, cy - 0.21)], "icon_POST_dart", INK, 1.75)
    polyline(S, [(cx + 0.16, cy - 0.27), (cx + 0.24, cy - 0.19), (cx + 0.28, cy - 0.28), (cx + 0.16, cy - 0.27)],
             "icon_POST_fletch", INK, 1.0, closed=True, fill=INK)


def build_plate(slide):
    """Shared structure: the 3D trace plate and the two fork arrows toward POST."""
    g = group(slide, "Plate")
    S = g.shapes
    # fork arrows first so the plate overlaps their roots
    xs, hw, hh, hl = 6.78, 0.085, 0.17, 0.15
    tip = POST_PANEL["x"] - 0.07            # tips stop short of the dashed panel border
    for tag, y_end, y_root, sgn in (("up", 2.20, YC, -1), ("down", 4.30, YC, 1)):
        c = 0.07  # chamfer on the outer corner
        if sgn < 0:
            pts = [(xs - hw, y_root), (xs - hw, y_end - hw + c), (xs - hw + c, y_end - hw),
                   (tip - hl, y_end - hw), (tip - hl, y_end - hh), (tip, y_end), (tip - hl, y_end + hh),
                   (tip - hl, y_end + hw), (xs + hw, y_end + hw), (xs + hw, y_root)]
        else:
            pts = [(xs - hw, y_root), (xs - hw, y_end + hw - c), (xs - hw + c, y_end + hw),
                   (tip - hl, y_end + hw), (tip - hl, y_end + hh), (tip, y_end), (tip - hl, y_end - hh),
                   (tip - hl, y_end - hw), (xs + hw, y_end - hw), (xs + hw, y_root)]
        freeform_poly(S, pts, f"plate_fork_{tag}", MINT, MINT_S, 1.5)
    # the plate: darker offset edge, then the lighter top face
    slab = rrect(S, PLATE_X0, PLATE_Y0 + PLATE_DEPTH, PLATE_X1 - PLATE_X0, PLATE_Y1 - PLATE_Y0, "plate_edge",
                 mix(MINT_S, 0.55, MINT), MINT_S, 1.5, r=0.30)
    top = rrect(S, PLATE_X0, PLATE_Y0, PLATE_X1 - PLATE_X0, PLATE_Y1 - PLATE_Y0, "plate_top",
                MINT, MINT_S, 1.5, r=0.30)
    fill_linear(top, mix(MINT, 0.55), MINT, 90)
    return g


def sphere(S, cx, cy, d, name, base, outline, light=None, lw=1.5):
    b = ellipse(S, cx, cy, d, name, base, outline, lw)
    fill_sphere(b, light or mix(base, 0.75), base)
    return b


def build_pre(slide):
    g = group(slide, "PRE")
    S = g.shapes
    c = CARD
    # 3D: a darker slab offset to the lower right, then the card
    rrect(S, c["x"] + 0.07, c["y"] + 0.07, c["w"], c["h"], "card_depth", mix(PRE, 0.15, PRE_S), PRE_S, 1.25, r=0.12)
    rrect(S, c["x"], c["y"], c["w"], c["h"], "card_body", WHITE, PRE_S, 2.0, r=0.12)
    hdr = add_shape(S, MSO_SHAPE.ROUND_2_SAME_RECTANGLE, c["x"], c["y"], c["w"], 0.40, "card_header",
                    PRE_T2, PRE_S, 2.0, adj=[0.30, 0.0])
    set_text(hdr, "agent configuration", size=15, bold=True, color=INK, margins=(0.1, 0, 0.1, 0))
    textbox(S, c["x"] + 0.14, c["y"] + 0.48, 2.25, 0.72,
            "Read the quarterly report and summarize the revenue section", "card_task",
            size=14, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=0.95)
    # tool chips: status mark on the left, so the arrow to the plate leaves the chip cleanly
    cx0, cw = c["x"] + 0.46, c["w"] - 0.46 - 0.12
    y_ok = 2.34
    ellipse(S, c["x"] + 0.27, y_ok + 0.16, 0.24, "mark_ok_disc", OK_GREEN, None)
    polyline(S, [(c["x"] + 0.205, y_ok + 0.165), (c["x"] + 0.255, y_ok + 0.215), (c["x"] + 0.34, y_ok + 0.115)],
             "mark_ok_check", WHITE, 2.0)
    pill(S, cx0, y_ok, cw, 0.32, "read_report (read)", "chip_read_report", MINT_T1, MINT_S, size=13, font=MONO)
    tri = add_shape(S, MSO_SHAPE.ISOSCELES_TRIANGLE, c["x"] + 0.10, YC - 0.19, 0.34, 0.30, "mark_warning",
                    PRE, PRE_S, 1.5)
    set_text(tri, "!", size=13, bold=True, color=INK, margins=(0, 0, 0, 0), wrap=False)
    pill(S, cx0, YC - 0.16, cw, 0.32, "run_shell (execute)", "chip_run_shell", PRE_T2, PRE_S, size=13, font=MONO)
    pill(S, c["x"] + c["w"] - 1.98, c["y"] + c["h"] - 0.15, 1.80, 0.28, "excess capability", "tag_excess_capability",
         PRE, PRE_S, size=14, color=INK, lw=1.5)
    labels(S, "PRE", PRE_CX, "Over-privileged", "before it runs?", ("config · 8 methods", 1.90), "audit(config)",
           [("pre/over_privilege", 1.74)], y=Y_LABEL, qw=2.9)
    return g


def code_pill(S, x, y, w, h, text, name, key):
    """Code box tinted with its stage color (about 10 percent) and a stage-tinted outline."""
    st = STAGE[key]
    return pill(S, x, y, w, h, text, name, mix(st["c"], 0.90), mix(st["s"], 0.55), size=13, font=MONO,
                lw=1.0, r=0.06)


def labels(S, key, cx, q1, q2, family, call, ids, y, qw, call_w=None):
    st = STAGE[key]
    textbox(S, cx - qw / 2, y, qw, 0.50, [q1, q2] if q2 else [q1], f"{key}_question", size=14, bold=True,
            anchor=MSO_ANCHOR.TOP, spacing=0.95)
    family, fw = family
    pill(S, cx - fw / 2, y + 0.57, fw, 0.28, family, f"{key}_family", st["t2"], st["s"], size=14)
    cw = call_w or (0.28 + len(call) * 0.0993)
    code_pill(S, cx - cw / 2, y + 0.93, cw, 0.33, call, f"{key}_call", key)
    total = sum(w for _, w in ids) + 0.1 * (len(ids) - 1)
    xx = cx - total / 2
    for text, w in ids:
        pill(S, xx, y + 1.33, w, 0.28, text, f"{key}_board_{text}", WHITE, st["s"], size=13, lw=1.25)
        xx += w + 0.1


def build_live(slide):
    g = group(slide, "LIVE")
    S = g.shapes
    st = STAGE["LIVE"]
    xs = [3.56, 4.26, 4.96]
    t_x, t_d = 5.97, 0.50
    r_small, r_halo = 0.19, 0.33
    # each bead is grouped with its own label, so a bead moved as a group takes its label along;
    # the glued connectors still follow a bead inside its group
    beads = []
    for i, x in enumerate(xs):
        bg = group_in(S, f"step_{i}")
        beads.append(sphere(bg.shapes, x, YC, 2 * r_small, f"bead_step_{i}", MINT, MINT_S, light="F4FBF7"))
        textbox(bg.shapes, x - 0.36, YC + 0.265, 0.72, 0.24, f"step {i}", f"label_step_{i}", size=14, color=INK,
                wrap=False)
    dots = textbox(S, 5.30, YC - 0.16, 0.30, 0.32, "···", "bead_gap_dots", size=18, bold=True, color=SEC)
    # newest step: dashed halo, the emphasized bead, and its label
    tg = group_in(S, "step_t")
    ellipse(tg.shapes, t_x, YC, 2 * r_halo, "bead_step_t_halo", LIVE_T1, LIVE, 1.25, dash="DASH")
    bead_t = sphere(tg.shapes, t_x, YC, t_d, "bead_step_t", LIVE, LIVE_S, light=mix(LIVE, 0.55), lw=2.0)
    textbox(tg.shapes, t_x - 0.36, YC + 0.30, 0.72, 0.24, "step t", "label_step_t", size=14, bold=True, color=LIVE_S,
            wrap=False)
    # sequence links, attached to the beads
    connect(S, beads[0], "right", beads[1], "left", "link_0_1", MINT_S, 1.75)
    connect(S, beads[1], "right", beads[2], "left", "link_1_2", MINT_S, 1.75)
    connect(S, beads[2], "right", dots, "left", "link_2_gap", MINT_S, 1.75)
    connect(S, dots, "right", bead_t, "left", "link_gap_t", MINT_S, 1.75)
    # dependency curves above the beads: non-adjacent dependencies, each leaving its bead at its own point.
    # Static freeform Bezier curves (not attached; reroute by hand when a bead moves). The longest ends on
    # the dashed halo of step t.
    on = lambda cx, r, deg: (cx + r * math.sin(math.radians(deg)), YC - r * math.cos(math.radians(deg)))
    dep_curve(S, on(xs[0], r_small, 32), on(xs[2], r_small, -32), 0.30, "dep_curve_0_2", MINT_S, 1.5)
    dep_curve(S, on(xs[1], r_small, 32), on(t_x, r_halo + 0.01, -58), 0.40, "dep_curve_1_t", MINT_S, 1.5)
    dep_curve(S, on(xs[0], r_small, -30), on(t_x, r_halo + 0.01, -20), 0.62, "dep_curve_0_t", LIVE_S, 1.75)
    textbox(S, 3.70, 1.78, 1.5, 0.26, "dependencies", "label_dependencies", size=14, italic=True,
            color=SEC, wrap=False)
    # monitor above step t
    mw, mh, my = 1.72, 0.95, 1.00
    mx = t_x - mw / 2
    bez = rrect(S, mx, my, mw, mh, "monitor_bezel", LIVE_S, LIVE_S, 1.25, r=0.09)
    fill_linear(bez, mix(LIVE_S, 0.12, LIVE), LIVE_S, 90)
    rrect(S, mx + 0.05, my + 0.05, mw - 0.10, mh - 0.10, "monitor_screen", "F2F9FB", None, r=0.05)
    textbox(S, mx + 0.11, my + 0.06, mw - 0.22, 0.32,
            [[("s", dict(italic=True)), ("t", dict(italic=True, sub=True)), (" = ", {}),
              ("f", dict(italic=True)), ("(", {}), ("x", dict(italic=True)),
              ("0..t", dict(italic=True, sub=True)), (")", {})]],
            "monitor_formula", size=18, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    # alarm curve: flat, then rising across the dashed threshold; the marker sits on the crossing
    x0, x1, yb, yt = mx + 0.13, mx + mw - 0.13, my + 0.83, my + 0.50
    curve = [(x0, yb - 0.02), (x0 + 0.68, yb - 0.02), (x0 + 0.90, yt + 0.02), (x1, yt - 0.03)]
    y_thr = my + 0.64
    area_under(S, curve, yb, "monitor_alarm_area", mix(LIVE, 0.70))
    bezier(S, *curve, "monitor_alarm_curve", LIVE_S, 2.0)
    thr = polyline(S, [(x0, y_thr), (x1, y_thr)], "monitor_threshold", POST_S, 1.0)
    thr.line.dash_style = MSO_LINE.DASH
    cross = bezier_crossing(curve, y_thr)
    ellipse(S, cross[0], cross[1], 0.10, "monitor_alarm_marker", POST, POST_S, 1.0)
    # stand and base
    add_shape(S, MSO_SHAPE.TRAPEZOID, t_x - 0.14, my + mh, 0.28, 0.12, "monitor_stand", LIVE_S, LIVE_S, 1.0,
              adj=[0.35])
    base = rrect(S, t_x - 0.40, my + mh + 0.11, 0.80, 0.06, "monitor_base", LIVE_S, LIVE_S, 1.0, r=0.03)
    # monitor reads the newest step: attached connector from bead t up to the monitor base
    connect(S, bead_t, "top", base, "bottom", "link_step_t_to_monitor", LIVE_S, 1.5, tail="triangle")
    labels(S, "LIVE", LIVE_CX, "Can the failure be caught", "while it runs?", ("online · 5 methods", 1.90),
           "step_alarm_scores(trace)", [("live/swegym", 1.28), ("live/tau", 0.87)], y=Y_LABEL, qw=3.4,
           call_w=2.68)
    return g, beads[0]


def build_post(slide):
    g = group(slide, "POST")
    S = g.shapes
    st = STAGE["POST"]
    p = POST_PANEL
    rrect(S, p["x"], p["y"], p["w"], p["h"], "post_panel", mix(POST, 0.93), POST_S, 1.75, r=0.16, dash="DASH")
    mid = 3.13
    d = polyline(S, [(p["x"] + 0.12, mid), (p["x"] + p["w"] - 0.12, mid)], "post_divider", POST_S, 1.25)
    d.line.dash_style = MSO_LINE.DASH

    # ---- top: failure dial, question, family, call, boards ----
    R, r = 0.68, 0.41
    gcx, gcy = POST_CX, 1.76
    # mint (healthy) then a coral ramp from pale to saturated; no numbers. A straight RGB blend of mint and
    # coral passes through khaki, so the ramp runs through coral tints instead.
    seg_cols = [MINT, mix(POST, 0.68), mix(POST, 0.40), POST, mix(POST, 0.45, POST_S)]
    n = len(seg_cols)
    for i, col in enumerate(seg_cols):
        ring_segment(S, gcx, gcy, R, r, 180 + 180 * i / n, 180 + 180 * (i + 1) / n, f"gauge_segment_{i}", col, WHITE)
    ang = math.radians(305)
    tipx, tipy = gcx + 0.57 * math.cos(ang), gcy + 0.57 * math.sin(ang)
    nx, ny = -math.sin(ang), math.cos(ang)
    freeform_poly(S, [(gcx + nx * 0.045, gcy + ny * 0.045), (tipx, tipy), (gcx - nx * 0.045, gcy - ny * 0.045)],
                  "gauge_needle", INK, INK, 0.75)
    ellipse(S, gcx, gcy, 0.15, "gauge_hub", INK, WHITE, 1.5)
    textbox(S, p["x"] + 0.05, 1.85, p["w"] - 0.10, 0.26, "Did the run fail?", "POST_question_run", size=14, bold=True)
    pill(S, POST_CX - 0.89, 2.13, 1.77, 0.27, "run · 17 methods", "POST_family_run", st["t2"], POST_S, size=14)
    code_pill(S, POST_CX - 1.33, 2.45, 2.66, 0.32, "decision_function(traces)", "POST_call_run", "POST")
    xx = POST_CX - (1.50 + 1.09 + 0.10) / 2
    for t, w in (("detect/swegym", 1.50), ("detect/tau", 1.09)):
        pill(S, xx, 2.83, w, 0.27, t, f"POST_board_{t}", WHITE, POST_S, size=13)
        xx += w + 0.10

    # ---- bottom: finished trace, magnifier, ranked list ----
    ty = 3.56
    yb = ty + 0.05                                  # bead centerline
    bx = [p["x"] + 0.38, p["x"] + 0.78, p["x"] + 1.20, p["x"] + 1.62]
    rrect(S, p["x"] + 0.16, ty - 0.13 + 0.09, 1.64, 0.36, "mini_plate_edge", mix(MINT_S, 0.55, MINT), MINT_S, 1.25, r=0.17)
    mp = rrect(S, p["x"] + 0.16, ty - 0.13, 1.64, 0.36, "mini_plate_top", MINT, MINT_S, 1.25, r=0.17)
    fill_linear(mp, mix(MINT, 0.55), MINT, 90)
    for i, x in enumerate(bx):
        if i == 2:
            continue
        b = sphere(S, x, yb, 0.31, f"post_bead_step_{i}", MINT, MINT_S, light="F4FBF7", lw=1.25)
        set_text(b, str(i), size=14, bold=True, color=INK)
    bad = sphere(S, bx[2], yb, 0.34, "post_bead_step_2_mistake", POST, POST_S, light=mix(POST, 0.6), lw=1.75)
    set_text(bad, "2", size=14, bold=True, color=INK)
    # magnifier over the coral bead
    ld = 0.58
    lens = ellipse(S, bx[2], yb, ld, "magnifier_lens", WHITE, INK, 3.0)
    fill_solid(lens, WHITE, alpha=0.08)
    hl, hth = 0.24, 0.085
    hc = (bx[2] + (ld / 2 + hl / 2 - 0.01) * math.cos(math.radians(45)),
          yb + (ld / 2 + hl / 2 - 0.01) * math.sin(math.radians(45)))
    add_shape(S, MSO_SHAPE.ROUNDED_RECTANGLE, hc[0] - hl / 2, hc[1] - hth / 2, hl, hth, "magnifier_handle", INK, INK, 1.0,
              adj=[0.5], rot=45)
    # ranked list (illustrative): #1 matches the magnified bead
    lx = p["x"] + 1.95
    for i, (t, bold) in enumerate((("#1 step 2", True), ("#2 step 1", False), ("#3 step 0", False))):
        textbox(S, lx, yb - 0.39 + 0.26 * i, 0.86, 0.26, t, f"rank_{i + 1}", size=14, bold=bold,
                color=POST_S if bold else INK, align=PP_ALIGN.LEFT, wrap=False)
    textbox(S, p["x"] + 0.05, 4.08, p["w"] - 0.10, 0.26, "Which step was the mistake?", "POST_question_step", size=14, bold=True)
    pill(S, POST_CX - 0.93, 4.37, 1.86, 0.27, "step · 14 methods", "POST_family_step", st["t2"], POST_S, size=14)
    code_pill(S, POST_CX - 0.78, 4.69, 1.56, 0.32, "rank(trace)", "POST_call_step", "POST")
    pill(S, POST_CX - 1.01, 5.07, 2.02, 0.27, "localize/whoandwhen", "POST_board_localize", WHITE, POST_S, size=13)
    return g


def hatch(fill, fore="BAC4BF", back=WHITE):
    """Control rows are hatched, as in assets/leaderboard-swegym.png."""
    fill.patterned()
    fill.pattern = MSO_PATTERN.WIDE_UPWARD_DIAGONAL
    fill.fore_color.rgb = rgb(fore)
    fill.back_color.rgb = rgb(back)


def build_leaderboards(slide, rows):
    """rows: [(label, value, is_control)] in drawing order, top to bottom."""
    g = group(slide, "Leaderboards")
    S = g.shapes
    p = LB
    tx = p["x"] + 0.20                          # left edge of the captions
    rrect(S, p["x"], p["y"], p["w"], p["h"], "lb_panel", "F6F8F7", RULE, 1.5, r=0.14)
    # small three-bar icon in the three stage colors, then the title
    for i, (col, hh) in enumerate(((PRE, 0.17), (LIVE, 0.26), (POST, 0.21))):
        add_shape(S, MSO_SHAPE.RECTANGLE, p["x"] + 0.22 + i * 0.105, p["y"] + 0.52 - hh, 0.075, hh, f"lb_icon_bar_{i}", col)
    textbox(S, p["x"] + 0.58, p["y"] + 0.18, 2.1, 0.40, "Leaderboards", "lb_title", size=18, bold=True,
            align=PP_ALIGN.LEFT)
    textbox(S, tx, p["y"] + 0.72, p["w"] - 0.40, 0.26, "detect/swegym · ROC-AUC", "lb_caption",
            size=14, color=SEC, align=PP_ALIGN.LEFT)
    # the chart shows 6 of the 17 board rows; the hatched swatch is its only legend
    textbox(S, tx, p["y"] + 1.00, 1.15, 0.26, "6 of 17 rows", "lb_scope", size=14, color=SEC,
            align=PP_ALIGN.LEFT, wrap=False)
    sw = add_shape(S, MSO_SHAPE.RECTANGLE, tx + 1.30, p["y"] + 1.07, 0.26, 0.13, "lb_legend_swatch_control",
                   None, GRAY_BAR, 1.0)
    hatch(sw.fill)
    textbox(S, tx + 1.64, p["y"] + 1.00, 0.85, 0.26, "control", "lb_legend_control", size=14, color=SEC,
            align=PP_ALIGN.LEFT, wrap=False)

    # a name with a hyphen or a space may break onto a second line, which keeps the category column narrow
    wrapped = {"spline size control": "spline size\ncontrol", "G-Safeguard-Lite": "G-Safeguard-\nLite",
               "GRADE full": "GRADE\nfull", "GRADE flatdep": "GRADE\nflatdep"}
    cd = CategoryChartData(number_format="0.000")
    cd.categories = [wrapped.get(n, n) for n, _, _ in rows]
    cd.add_series("ROC-AUC", [v for _, v, _ in rows])
    gf = S.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, E(p["x"] + 0.06), E(p["y"] + 1.30), E(p["w"] - 0.12), E(3.30), cd)
    gf.name = "chart_detect_swegym_roc_auc"
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = False
    ch.font.name = FONT
    ch.font.size = Pt(14)
    ch.font.color.rgb = rgb(INK)
    ca = ch.category_axis
    ca.reverse_order = True
    ca.major_tick_mark = XL_TICK_MARK.NONE
    ca.format.line.fill.background()
    ca.tick_labels.font.size = Pt(14)
    ca.tick_labels.font.color.rgb = rgb(INK)
    ca.tick_labels.font.name = FONT
    # category names wrap only at the explicit line breaks above
    ca._element.find(f"{{{NS_C}}}txPr").find(f"{{{NS_A}}}bodyPr").set("wrap", "none")
    # fixed inner plot area: about 51% of the width for the names (about 0.17 in clear of the panel edge), the rest (1.3 in) for the bars
    plotArea = ch._chartSpace.chart.plotArea
    for el in plotArea.findall(f"{{{NS_C}}}layout"):
        plotArea.remove(el)
    plotArea.insert(0, parse_xml(
        f'<c:layout xmlns:c="{NS_C}"><c:manualLayout><c:layoutTarget val="inner"/><c:xMode val="edge"/>'
        f'<c:yMode val="edge"/><c:x val="0.51"/><c:y val="0.01"/><c:w val="0.475"/><c:h val="0.98"/>'
        f'</c:manualLayout></c:layout>'))
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0.0, 1.0
    va.has_major_gridlines = False
    va.visible = False
    plot = ch.plots[0]
    plot.gap_width = 38
    plot.vary_by_categories = False
    ser = plot.series[0]
    # method bars: POST coral (detect/swegym is a POST board), set once on the series
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = rgb(POST)
    ser.format.line.color.rgb = rgb(POST_S)
    ser.format.line.width = Pt(1.0)
    # value labels on the series, inside the bar end
    dl = ser.data_labels
    dl.show_value = True
    dl.number_format = "0.000"
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.INSIDE_END
    dl.font.size = Pt(14)
    dl.font.bold = True
    dl.font.color.rgb = rgb(INK)
    # PowerPoint wraps a data label wider than about a fifth of the chart ("0.85" / "0"); keep it on one line
    dl._element.find(f"{{{NS_C}}}txPr").find(f"{{{NS_A}}}bodyPr").set("wrap", "none")
    # only the control rows override the series fill. The override is per point INDEX (a c:dPt), so after
    # reordering or adding rows in Edit Data, reapply the hatch to the control rows by hand.
    for i, (_, _, is_control) in enumerate(rows):
        if is_control:
            pt = ser.points[i]
            hatch(pt.format.fill)
            pt.format.line.color.rgb = rgb(GRAY_BAR)
            pt.format.line.width = Pt(1.0)
    textbox(S, tx, p["y"] + 4.64, p["w"] - 0.40, 0.26, "list the 6 boards:", "lb_hint", size=14,
            color=SEC, align=PP_ALIGN.LEFT)
    pill(S, p["x"] + 0.12, p["y"] + 4.92, 2.70, 0.30, "python -m auditable.bench", "lb_command", CODE_FILL, CODE_LINE,
         size=13, font=MONO, lw=1.0, r=0.06)
    return g


def build_footer(slide):
    g = group(slide, "Footer")
    S = g.shapes
    w, h = 9.90, 0.40
    f = rrect(S, (W - w) / 2, 5.67, w, h, "footer_pill", mix(MINT, 0.35), MINT_S, 1.75, r=h / 2)
    set_text(f, [[("auditable", dict(bold=True)), (" · ", {}), ("44 methods", dict(bold=True)), (" · ", {}),
                  ("4 families", dict(bold=True)), (" · one interface per family · ", {}),
                  ("6 leaderboards", dict(bold=True))]], size=18, wrap=False)
    return g


# --------------------------------------------------------------------------
# Data and assembly
# --------------------------------------------------------------------------
# Figure label -> row key in the golden board (detect/swegym, ROC-AUC). Drawing order = list order.
BOARD_ROWS = [("GAAN", "run.pygod_gaan"), ("spline size control", "run.lr_spline_flat"),
              ("G-Safeguard-Lite", "run.gsafeguard_lite"), ("GRADE full", "run.lr_full"),
              ("GRADE flatdep", "run.lr_flatdep"), ("random", "run.random")]
# benchmark_role == "control" in the method registry (the README marks both rows "(control)")
CONTROL_KEYS = {"run.lr_spline_flat", "run.random"}


def leaderboard_rows():
    """Chart values come from the golden board shipped with the package (no typed numbers)."""
    import json
    rows = json.loads(GOLDEN.read_text())["rows"]
    return [(label, float(rows[key]["roc_auc"]), key in CONTROL_KEYS) for label, key in BOARD_ROWS]


def build(path: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = E(W), E(H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(WHITE)

    build_plate(slide)
    live, bead0 = build_live(slide)
    pre = build_pre(slide)
    build_post(slide)
    build_leaderboards(slide, leaderboard_rows())
    build_footer(slide)
    build_banners(slide)

    # card -> plate arrow: glued from the run_shell chip to the first bead (top level, crosses regions)
    chip = next(s for s in pre.shapes if s.name == "chip_run_shell")
    connect(slide.shapes, chip, "right", bead0, "left", "arrow_config_to_trace", PRE_S, 3.0)
    prs.save(str(path))


def export_pdf(pptx: Path, pdf: Path) -> None:
    """Desktop PowerPoint on macOS exports the PDF (sandboxed, so work in a temp folder)."""
    with tempfile.TemporaryDirectory(prefix="hero_export_", dir="/private/tmp") as tmp_dir:
        tmp = Path(tmp_dir)
        src, dst = tmp / "hero.pptx", tmp / "hero.pdf"
        shutil.copy(pptx, src)
        dst.touch()  # the sandboxed app needs the target to exist
        script = [
            'tell application "Microsoft PowerPoint"',
            f'open POSIX file "{src}"',
            "delay 3",
            f'save active presentation in POSIX file "{dst}" as save as PDF',
            "close active presentation saving no",
            "end tell",
        ]
        cmd = ["osascript"] + [a for line_ in script for a in ("-e", line_)]
        subprocess.run(cmd, check=True)
        if dst.stat().st_size == 0:
            raise RuntimeError("PowerPoint wrote an empty PDF")
        shutil.copy(dst, pdf)


def rasterize(pdf: Path, png: Path, width_px: int = 2600, pad_px: int = PAD_PX) -> None:
    """PDF page -> PNG of width_px, with a white margin of pad_px on every side so the footer pill and the
    panels do not sit on the image edge (README viewers crop or invert at the border)."""
    import pymupdf
    from PIL import Image
    doc = pymupdf.open(str(pdf))
    page = doc[0]
    zoom = (width_px - 2 * pad_px) / page.rect.width
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    inner = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    out = Image.new("RGB", (pix.width + 2 * pad_px, pix.height + 2 * pad_px), "white")
    out.paste(inner, (pad_px, pad_px))
    png.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(png), optimize=True)
    print(f"wrote {png} ({out.width}x{out.height})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-export", action="store_true", help="write hero.pptx only")
    args = ap.parse_args()
    build(PPTX)
    print(f"wrote {PPTX}")
    if not args.no_export:
        export_pdf(PPTX, PDF)
        print(f"wrote {PDF}")
        rasterize(PDF, PNG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
