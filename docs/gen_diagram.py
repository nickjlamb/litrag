"""Generate docs/architecture-{light,dark}.svg for the README.

Hand-tuned layout; run after editing:  python docs/gen_diagram.py
"""

FONT = "-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,monospace"

THEMES = {
    "light": dict(
        text="#1f2328", muted="#59636e", border="#d0d7de", panel="#f6f8fa",
        node="#ffffff", accent="#8250df", accent_soft="#fbf0ff",
        red="#cf222e", red_fill="#ffebe9", red_border="#ffc1bc",
        green="#1a7f37", green_fill="#dafbe1", green_border="#aceebb",
        edge="#8c959f",
    ),
    "dark": dict(
        text="#e6edf3", muted="#9198a1", border="#3d444d", panel="#151b23",
        node="#212830", accent="#ab7df8", accent_soft="#2a2139",
        red="#f85149", red_fill="#3c1618", red_border="#6e2a2c",
        green="#3fb950", green_fill="#122117", green_border="#2b5233",
        edge="#767d86",
    ),
}

W, H = 960, 500


def build(c: dict) -> str:
    s = []
    s.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="{FONT}" role="img" '
        'aria-label="LitRAG architecture: a local ingest and retrieval stage feeds an '
        'LLM generation step whose every claim passes a two-stage citation-faithfulness '
        'eval — a deterministic quote locator, then an LLM-as-judge.">'
    )
    s.append(
        '<defs>'
        f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c["edge"]}"/></marker>'
        f'<marker id="arr-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c["red"]}"/></marker>'
        f'<marker id="arr-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c["green"]}"/></marker>'
        '</defs>'
    )

    def panel(x, y, w, h, title):
        s.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
            f'fill="{c["panel"]}" stroke="{c["border"]}"/>'
        )
        s.append(
            f'<text x="{x + 18}" y="{y + 26}" font-size="11" font-weight="600" '
            f'letter-spacing="1.5" fill="{c["muted"]}">{title}</text>'
        )

    def node(cx, y, w, h, title, sub=None, fill=None, stroke=None, tcol=None, mono=False):
        fill = fill or c["node"]
        stroke = stroke or c["border"]
        tcol = tcol or c["text"]
        x = cx - w / 2
        s.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
            f'fill="{fill}" stroke="{stroke}"/>'
        )
        if sub:
            s.append(
                f'<text x="{cx}" y="{y + 22}" font-size="13" font-weight="600" '
                f'text-anchor="middle" fill="{tcol}">{title}</text>'
            )
            fam = MONO if mono else FONT
            fs = 10.5 if mono else 11
            s.append(
                f'<text x="{cx}" y="{y + 40}" font-size="{fs}" font-family="{fam}" '
                f'text-anchor="middle" fill="{c["muted"]}">{sub}</text>'
            )
        else:
            s.append(
                f'<text x="{cx}" y="{y + h / 2 + 4.5}" font-size="13" font-weight="600" '
                f'text-anchor="middle" fill="{tcol}">{title}</text>'
            )

    def vline(x, y1, y2, label=None, marker="arr", lx_off=8):
        s.append(
            f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{c["edge"]}" '
            f'stroke-width="1.5" marker-end="url(#{marker})"/>'
        )
        if label:
            s.append(
                f'<text x="{x + lx_off}" y="{(y1 + y2) / 2 + 4}" font-size="11" '
                f'fill="{c["muted"]}">{label}</text>'
            )

    # ---------------- left panel: local ----------------
    panel(16, 52, 384, 432, "LOCAL &#183; NO API KEY REQUIRED")
    lcx = 208
    node(lcx, 92, 264, 52, "PubMed abstracts", "data/ &#183; 15 real trial abstracts", mono=False)
    vline(lcx, 144, 176, "chunk")
    node(lcx, 178, 264, 52, "Passages + provenance", "{pmid, title, source}", mono=True)
    vline(lcx, 230, 262, "embed &#183; MiniLM, local")
    # FAISS cylinder
    fy, fh, frx, fry = 266, 52, 90, 9
    s.append(
        f'<path d="M {lcx - frx} {fy + fry} v {fh - 2 * fry} a {frx} {fry} 0 0 0 {2 * frx} 0 '
        f'v {-(fh - 2 * fry)}" fill="{c["node"]}" stroke="{c["border"]}"/>'
    )
    s.append(
        f'<ellipse cx="{lcx}" cy="{fy + fry}" rx="{frx}" ry="{fry}" '
        f'fill="{c["node"]}" stroke="{c["border"]}"/>'
    )
    s.append(
        f'<text x="{lcx}" y="{fy + 36}" font-size="13" font-weight="600" '
        f'text-anchor="middle" fill="{c["text"]}">FAISS index</text>'
    )
    # question + retrieve row
    qcx, rcx, rowy = 106, 300, 392
    node(qcx, rowy, 132, 40, "Question")
    node(rcx, rowy, 160, 40, "Retrieve top-k")
    # faiss -> retrieve (elbow)
    s.append(
        f'<polyline points="{lcx},{fy + fh} {lcx},{rowy - 22} {rcx},{rowy - 22} {rcx},{rowy - 3}" '
        f'fill="none" stroke="{c["edge"]}" stroke-width="1.5" marker-end="url(#arr)"/>'
    )
    s.append(
        f'<text x="{lcx + 8}" y="{rowy - 28}" font-size="11" fill="{c["muted"]}">similarity search</text>'
    )
    # question -> retrieve
    s.append(
        f'<line x1="{qcx + 66}" y1="{rowy + 20}" x2="{rcx - 83}" y2="{rowy + 20}" '
        f'stroke="{c["edge"]}" stroke-width="1.5" marker-end="url(#arr)"/>'
    )

    # ---------------- right panel: llm api ----------------
    panel(416, 52, 528, 432, "LLM API")
    gcx = 690
    node(gcx, 92, 400, 52, "Generate structured answer",
         "{answer, claims:[{text, cited_quote, source}]}", mono=True,
         fill=c["accent_soft"], stroke=c["accent"], tcol=c["accent"])
    # retrieve -> generate (elbow across panels)
    s.append(
        f'<polyline points="{rcx + 80},{rowy + 20} {450},{rowy + 20} {450},{118} {gcx - 203},{118}" '
        f'fill="none" stroke="{c["edge"]}" stroke-width="1.5" marker-end="url(#arr)"/>'
    )
    s.append(
        f'<text x="442" y="265" font-size="11" fill="{c["muted"]}" text-anchor="middle" '
        f'transform="rotate(-90 442 265)">top-k passages + provenance</text>'
    )
    vline(gcx, 144, 194, "every claim, with its verbatim quote", lx_off=10)
    node(gcx, 196, 400, 52, "Stage 1 &#183; Locate the quote",
         "exact + fuzzy match in the retrieved source &#8212; deterministic")
    # branches
    redcx, s2cx, by = 560, 810, 306
    s.append(
        f'<polyline points="{gcx - 90},248 {gcx - 90},270 {redcx},270 {redcx},{by - 3}" '
        f'fill="none" stroke="{c["red"]}" stroke-width="1.5" marker-end="url(#arr-red)"/>'
    )
    s.append(
        f'<text x="{redcx - 66}" y="292" font-size="11" font-weight="600" fill="{c["red"]}">not found</text>'
    )
    s.append(
        f'<polyline points="{gcx + 90},248 {gcx + 90},270 {s2cx},270 {s2cx},{by - 3}" '
        f'fill="none" stroke="{c["edge"]}" stroke-width="1.5" marker-end="url(#arr)"/>'
    )
    s.append(
        f'<text x="{s2cx + 12}" y="292" font-size="11" fill="{c["muted"]}">located</text>'
    )
    node(redcx, by, 216, 52, "hallucinated_quote",
         "flagged &#8212; no judge call spent",
         fill=c["red_fill"], stroke=c["red_border"], tcol=c["red"])
    node(s2cx, by, 220, 52, "Stage 2 &#183; LLM-as-judge",
         "grades from the passage only")
    vline(s2cx, by + 52, 396, marker="arr-green")
    node(s2cx, 398, 244, 52, "Verdict + grounded flag",
         "supports &#183; partial &#183; contradicts &#183; not_found",
         fill=c["green_fill"], stroke=c["green_border"], tcol=c["green"])

    s.append("</svg>")
    return "\n".join(s)


for name, palette in THEMES.items():
    path = f"docs/architecture-{name}.svg"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build(palette))
    print("wrote", path)
