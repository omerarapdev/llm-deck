#!/usr/bin/env python3
"""Inlines every stylesheet, script, the model data and the content files into
dist/one-query.html, a single file that opens straight from disk or any static
host. Only the Google Fonts stylesheet stays external (the page falls back to
system fonts without it)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / "index.html").read_text()


def inline_css(m):
    return "<style>\n" + (ROOT / m.group(1)).read_text() + "\n</style>"


def inline_js(m):
    code = (ROOT / m.group(1)).read_text().replace("</script", "<\\/script")
    return f"<script>/* {m.group(1)} */\n" + code + "\n</script>"


page = re.sub(r'<link rel="stylesheet" href="(css/[^"]+)">', inline_css, src)
page = re.sub(r'<script src="((?:js|data|content)/[^"]+)"></script>', inline_js, page)
leftover = re.findall(r'(?:href|src)="(?!https://)[^"#]+\.(?:css|js)"', page)
assert not leftover, leftover

out = ROOT / "dist" / "one-query.html"
out.parent.mkdir(exist_ok=True)
out.write_text(page)
print(f"{out.relative_to(ROOT)}: {out.stat().st_size / 1024:.0f} KB")
