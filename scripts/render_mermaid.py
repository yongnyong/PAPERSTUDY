import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "papers"
ASSETS_DIR = ROOT / "assets" / "diagrams"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

MERMAID_BLOCK = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)

def slug_for(path: Path) -> str:
    return path.stem

def render_svg(mmd_text: str, out_svg: Path):
    tmp_mmd = out_svg.with_suffix(".mmd")
    tmp_mmd.write_text(mmd_text, encoding="utf-8")
    cmd = [
        "npx", "-y", "@mermaid-js/mermaid-cli",
        "-i", str(tmp_mmd),
        "-o", str(out_svg),
        "-b", "white",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, shell=(sys.platform == "win32"))
    if result.returncode != 0:
        print(f"FAILED: {out_svg}\n{result.stdout}\n{result.stderr}")
        return False
    tmp_mmd.unlink(missing_ok=True)
    return True

def process_file(md_path: Path):
    text = md_path.read_text(encoding="utf-8")
    slug = slug_for(md_path)
    matches = list(MERMAID_BLOCK.finditer(text))
    if not matches:
        return 0

    count = 0
    new_text_parts = []
    last_end = 0
    for i, m in enumerate(matches, start=1):
        count += 1
        svg_name = f"{slug}-{i}.svg"
        svg_path = ASSETS_DIR / svg_name
        ok = render_svg(m.group(1), svg_path)
        new_text_parts.append(text[last_end:m.start()])
        if ok:
            rel_path = f"../assets/diagrams/{svg_name}"
            new_text_parts.append(f'<img src="{rel_path}" alt="diagram" width="720">\n')
        else:
            new_text_parts.append(m.group(0))
        last_end = m.end()
    new_text_parts.append(text[last_end:])
    new_text = "".join(new_text_parts)
    md_path.write_text(new_text, encoding="utf-8")
    return count

def main():
    total = 0
    for md_path in sorted(PAPERS_DIR.glob("*.md")):
        n = process_file(md_path)
        if n:
            print(f"{md_path.name}: {n} diagram(s) rendered")
            total += n
    print(f"Total diagrams rendered: {total}")

if __name__ == "__main__":
    main()
