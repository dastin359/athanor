"""Render the CCARC3 handoff markdown as a standalone HTML page.

    .venv/bin/python tools/render_handoff.py out.html

The handoff is published as an artifact so the operator can read and copy from
it, and the artifact's source file lived in the scratchpad -- which reverts to an
image snapshot when the container is replaced. It did, and the source went with
it, so the next refresh had to rebuild the renderer from nothing. Same lesson as
`tools/mutation_check.py`: a tool kept only in the scratchpad is a tool that
expires. The markdown in `docs/` stays the single source of truth; this only
dresses it.

Self-contained by necessity -- the artifact CSP blocks every external host, so
no CDN stylesheet and no webfont link. Colours are defined as tokens on bare
`:root` and *redefined* (never first defined) under `prefers-color-scheme` and
`[data-theme]`, because a colour whose only definition sits inside a themed
block never applies to a viewer on the default "system" setting, which stamps
neither attribute.
"""
import html as _html
import pathlib
import re
import sys

SRC = pathlib.Path("docs/ccarc3_handoff_0809b.md")
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "handoff.html")


def inline(text: str) -> str:
    out = _html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", out)
    out = re.sub(r"~~([^~]+)~~", r"<del>\1</del>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    return out


def render(md: str) -> str:
    lines = md.split("\n")
    html, i = [], 0
    list_stack = []          # 'ul' | 'ol'

    def close_lists(to: int = 0) -> None:
        while len(list_stack) > to:
            html.append(f"</{list_stack.pop()}>")

    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            close_lists()
            lang = line[3:].strip()
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            label = f'<span class="code-lang">{_html.escape(lang)}</span>' if lang else ""
            html.append(f'<div class="code">{label}<pre><code>'
                        f'{_html.escape(chr(10).join(body))}</code></pre></div>')
            continue

        if re.match(r"^\|.*\|\s*$", line):
            close_lists()
            rows = []
            while i < len(lines) and re.match(r"^\|.*\|\s*$", lines[i]):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], rows[1:]
            if body and all(re.fullmatch(r":?-{2,}:?", c) for c in body[0]):
                body = body[1:]
            cells = "".join(f"<th>{inline(c)}</th>" for c in head)
            trs = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body
            )
            html.append(f'<div class="scroller"><table><thead><tr>{cells}</tr></thead>'
                        f"<tbody>{trs}</tbody></table></div>")
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            close_lists()
            level, text = len(heading.group(1)), heading.group(2)
            num = re.match(r"^(\d+)\.\s+(.*)$", text)
            if level == 2 and num:
                anchor = "s" + num.group(1)
                html.append(
                    f'<h2 id="{anchor}"><span class="sec" aria-hidden="true">'
                    f'§{num.group(1)}</span>{inline(num.group(2))}</h2>'
                )
            else:
                html.append(f"<h{level}>{inline(text)}</h{level}>")
            i += 1
            continue

        if re.match(r"^---+\s*$", line):
            close_lists()
            html.append("<hr />")
            i += 1
            continue

        item = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if item:
            indent, marker, text = len(item.group(1)), item.group(2), item.group(3)
            depth = indent // 3 + 1
            kind = "ul" if marker in "-*" else "ol"
            while len(list_stack) > depth:
                html.append(f"</{list_stack.pop()}>")
            if len(list_stack) < depth:
                html.append(f"<{kind}>")
                list_stack.append(kind)
            elif list_stack and list_stack[-1] != kind:
                html.append(f"</{list_stack.pop()}>")
                html.append(f"<{kind}>")
                list_stack.append(kind)
            body = [text]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(
                r"^(\s*)([-*]|\d+\.)\s+|^#{1,4}\s|^```|^\|", lines[i]
            ):
                body.append(lines[i].strip())
                i += 1
            html.append(f"<li>{inline(' '.join(body))}</li>")
            continue

        if not line.strip():
            i += 1
            continue

        close_lists()
        para = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^#{1,4}\s|^```|^\||^---+\s*$|^(\s*)([-*]|\d+\.)\s+", lines[i]
        ):
            para.append(lines[i].strip())
            i += 1
        html.append(f"<p>{inline(' '.join(para))}</p>")

    close_lists()
    return "\n".join(html)


md = SRC.read_text(encoding="utf-8")
body = render(md)

# Contents strip, built from the section headings the document actually numbers.
toc = "".join(
    f'<a href="#s{n}">§{n} <span>{_html.escape(t)}</span></a>'
    for n, t in re.findall(r"^## (\d+)\.\s+(.*)$", md, re.M)
)

CSS = """
:root{
  --ground:#FAFBFB; --raised:#F1F5F5; --ink:#101718; --muted:#57696C;
  --accent:#0E6A68; --alarm:#8C3A1E; --rule:#DCE3E3; --code-bg:#EDF2F2;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --serif:ui-serif,Georgia,Cambria,"Times New Roman",serif;
  --measure:68ch;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0D1315; --raised:#141D20; --ink:#E2E9E9; --muted:#8FA2A5;
    --accent:#54B6B0; --alarm:#D97757; --rule:#222E31; --code-bg:#111A1C;
  }
}
:root[data-theme="dark"]{
  --ground:#0D1315; --raised:#141D20; --ink:#E2E9E9; --muted:#8FA2A5;
  --accent:#54B6B0; --alarm:#D97757; --rule:#222E31; --code-bg:#111A1C;
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink);
  font-family:var(--serif); font-size:17px; line-height:1.65;
  margin:0; padding:0 1.5rem 6rem;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:var(--measure); margin:0 auto}
.masthead{
  max-width:var(--measure); margin:0 auto; padding:3.5rem 0 1.25rem;
  border-bottom:2px solid var(--ink);
}
.eyebrow{
  font-family:var(--mono); font-size:.7rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--accent); margin:0 0 .9rem;
}
.masthead h1{
  font-family:var(--mono); font-size:clamp(1.6rem,4.2vw,2.3rem); font-weight:600;
  letter-spacing:-.02em; line-height:1.15; margin:0; text-wrap:balance;
}
.standfirst{color:var(--muted); margin:.9rem 0 0; font-size:1.02rem}
nav.toc{
  max-width:var(--measure); margin:0 auto 3rem; padding:1rem 0;
  border-bottom:1px solid var(--rule);
  display:flex; flex-wrap:wrap; gap:.4rem 1.1rem;
}
nav.toc a{
  font-family:var(--mono); font-size:.74rem; letter-spacing:.02em;
  color:var(--muted); text-decoration:none; white-space:nowrap;
}
nav.toc a span{color:var(--ink)}
nav.toc a:hover span,nav.toc a:focus-visible span{color:var(--accent)}
h2{
  font-family:var(--mono); font-size:1.32rem; font-weight:600; letter-spacing:-.01em;
  line-height:1.25; margin:3.4rem 0 1rem; padding-top:1.1rem;
  border-top:1px solid var(--rule); position:relative; text-wrap:balance;
}
h2 .sec{
  display:block; font-size:.7rem; letter-spacing:.14em; color:var(--accent);
  margin-bottom:.45rem; font-weight:500;
}
h3{
  font-family:var(--mono); font-size:1rem; font-weight:600; letter-spacing:.01em;
  margin:2.2rem 0 .7rem; color:var(--ink); text-wrap:balance;
}
h4{font-family:var(--mono); font-size:.88rem; margin:1.6rem 0 .5rem}
p{margin:0 0 1.05rem}
a{color:var(--accent); text-decoration-thickness:1px; text-underline-offset:2px}
strong{font-weight:700}
del{color:var(--muted); text-decoration-color:var(--alarm)}
code{
  font-family:var(--mono); font-size:.845em; background:var(--code-bg);
  padding:.1em .34em; border-radius:2px; word-break:break-word;
}
ul,ol{margin:0 0 1.05rem; padding-left:1.35rem}
li{margin:0 0 .5rem}
li::marker{color:var(--accent); font-family:var(--mono); font-size:.85em}
hr{border:0; border-top:1px solid var(--rule); margin:2.4rem 0}
.scroller{overflow-x:auto; margin:0 0 1.4rem; border:1px solid var(--rule)}
table{border-collapse:collapse; width:100%; font-family:var(--mono); font-size:.78rem}
th,td{
  text-align:left; padding:.5rem .7rem; border-bottom:1px solid var(--rule);
  vertical-align:top; font-variant-numeric:tabular-nums;
}
th{
  background:var(--raised); font-weight:600; letter-spacing:.04em;
  text-transform:uppercase; font-size:.68rem; color:var(--muted);
  position:sticky; top:0;
}
tbody tr:last-child td{border-bottom:0}
td code,th code{background:transparent; padding:0}
.code{position:relative; margin:0 0 1.4rem; background:var(--code-bg); border:1px solid var(--rule)}
.code pre{margin:0; overflow-x:auto; padding:.95rem 1rem}
.code code{background:transparent; padding:0; font-size:.79rem; line-height:1.6}
.code-lang{
  position:absolute; top:0; right:0; font-family:var(--mono); font-size:.62rem;
  letter-spacing:.12em; text-transform:uppercase; color:var(--muted);
  padding:.3rem .55rem;
}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
@media (max-width:640px){
  body{font-size:16px; padding:0 1.1rem 4rem}
  nav.toc{gap:.35rem .8rem}
}
"""

page = f"""<title>CCARC3 handoff — 2026-08-09</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>{CSS}</style>
<header class="masthead">
  <p class="eyebrow">Claude Code as harness · ARC-AGI-3</p>
  <h1>CCARC3 handoff</h1>
  <p class="standfirst">Everything the next session needs and cannot re-derive.
  Written to be read in full before acting.</p>
</header>
<nav class="toc" aria-label="Contents">{toc}</nav>
<main class="wrap">
{body}
</main>
"""
OUT.write_text(page, encoding="utf-8")
print(f"wrote {OUT} ({len(page):,} bytes), {len(toc.split('</a>')) - 1} sections")
