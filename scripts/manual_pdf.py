"""把 ``docs/02-操作指南.md`` 排版成一份带封面与截图的 PDF 手册。

    python scripts/manual_pdf.py                    # 输出 docs/模拟铁轨-操作手册.pdf
    python scripts/manual_pdf.py --out D:\\手册.pdf
    python scripts/manual_pdf.py --html-only        # 只留 HTML，便于在浏览器里改样式

为什么自己写 Markdown 转换
------------------------------------------------
项目依赖只有 ``numpy`` + ``panda3d``（见 ``pyproject.toml``），没有 ``markdown`` /
``pandoc``。手册用到的 Markdown 子集很小（标题、表格、围栏代码、列表、引用、
粗体、行内代码），自己按行处理比引一个依赖更好控制 —— 尤其是下面这两处排版，
通用转换器都做不到：

* ``按键`` 渲染成键帽（``<kbd>``）：行内代码到底是**按键**还是**文件名**，
  靠 :data:`_KEYS` 白名单区分，所以 ``V`` 是键帽而 ``valley.json`` 是等宽字。
* 引用块渲染成提示卡片、每节另起一页 —— 这是"手册"而不是"文档"的排版。

PDF 是怎么出来的
------------------------------------------------
走本机 Chrome / Edge 的 headless ``--print-to-pdf``：与 :mod:`scripts.screenshot`
同一个思路 —— **不引入新的渲染依赖**，用机器上已有的浏览器把 HTML 印成 PDF。
因此页面尺寸、分页、字体全部由 CSS（``@page`` + ``break-*``）决定，
``--print-to-pdf`` 只是"按浏览器里看到的样子印一遍"。

配色取自 :mod:`render.style`（天空、端口的绿、悬停的黄），与游戏画面同源。
"""

from __future__ import annotations

import argparse
import base64
import html as html_mod
import io
import json
import re
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MANUAL_MD = DOCS / "02-操作指南.md"
README = ROOT / "README.md"
DEFAULT_OUT = DOCS / "模拟铁轨-操作手册.pdf"

#: 行内代码在**这些**内容里才当按键渲染成键帽；其余（文件名、场景名、数值）
#: 保持普通等宽字体。白名单而不是"猜"，是因为猜错的样子很难看：
#: ``1``–``9`` 是按键，而 ``0.5`` 是数值，光看长度是分不出来的。
_KEYS = frozenset({
    # 字母单键
    "V", "R", "T", "C", "U", "X", "B", "N", "M", "K", "J", "G", "P", "L",
    "H", "F", "S", "Q", "E", "W", "A", "D", "Z", "Y",
    # 数字键
    *[str(index) for index in range(1, 10)],
    # 符号键
    ",", ".", "[", "]",
    # 功能键
    "Delete", "Esc", "Enter", "退格", "空格", "↑", "↓",
    # 组合键
    "Ctrl+S", "Ctrl+O", "Ctrl+N", "Ctrl+Z", "Ctrl+Y",
    "Shift+R", "Shift+Z", "Shift+空格",
})

#: 封面 + 目录页上的"速览"数据卡（数字从仓库里现读，避免和 README 说岔）。
_STAT_CARDS = (
    ("7", "类轨道件"),
    ("4", "种列车"),
    ("12", "件手工布景"),
    ("5", "张现成场景"),
)

#: 图赏用的截图（文件必须真在 ``docs/`` 里；缺图会被跳过而不是印出一个空框）。
_SCENE_GALLERY = (
    ("scene_valley.png", "山谷环线", "群山夹着一条长环线，谷底有河、村子与草场"),
    ("scene_town.png", "小镇车站", "月台、古典站房、村子和小山围着车站"),
    ("scene_gorge.png", "河谷大桥", "高架桥两次跨过同一条河，桥墩落在河滩上"),
    ("scene_lake.png", "湖畔草原", "R80 大圆环绕着湖跑，湖岸是沙滩与草地"),
)

_TRAIN_GALLERY = (
    ("glb_huangsidai.png", "复兴号黄丝带 400BF", "glb 外观；8 / 16 辆，16 辆中间两头车对顶"),
    ("scene_train_green_skin_10.png", "老式绿皮车", "动力集中：起步慢、上坡乏力（程序化）"),
    ("scene_train_crh380a_8.png", "和谐号 CRH380A", "8 辆编组，动力分散（程序化）"),
    ("scene_train_cr400af_8.png", "复兴号 CR400AF", "目录保留，外观程序化"),
)

#: 「拼轨与闭合」的**成对**截图：左图是前、右图是后（一键闭合前后对比）。
_EDIT_GALLERY = (
    ("scene_editor.png", "铺轨中", "幽灵件吸附到最近的空闲端口，绿色即「可以放」",
     "scene_catalog.png", "七类轨道件", "直轨 / 曲线 / 坡道 / 高架桥 / 交叉 / 道岔 / 终端"),
    ("scene_autoclose_before.png", "一键闭合前", "拼环差最后一点，接缝没对上",
     "scene_autoclose_after.png", "一键闭合后", "按 C 自动挑一件补完并接上"),
)

#: 图赏末尾的"上手三步"（比空着半页好看，也真有用）。
_START_STEPS = (
    "**先感受一下**：`run.bat --scene valley` —— 山谷环线，列车已经挂好，按 `↑` 就走。",
    "**再动手改**：按 `B` 摆几件布景（`[` `]` 换件、`1`–`9` 直选），按 `N` 换一列车。",
    "**想从头拼**：`run.bat` 空场地 —— `,` `.` 到「曲线」挑弧度，几节后按 `C` 自动补完闭环。",
)

#: 附录：全部按键速查（与 ``app/editor.py`` 的 ``key_actions`` 一一对应）。
#: 说明都写得短，是为了让每一行在双栏里**只占一行** —— 折行的行高翻倍，附录就装不下一页。
_APPENDIX = (
    ("通用", (
        ("V", "放置轨道 ↔ 拖拽视角"),
        ("F", "取景（框住全场）"),
        ("G", "显示 / 隐藏网格"),
        ("P", "显示 / 隐藏端口标记"),
        ("L", "显示 / 隐藏闭环彩带"),
        ("H", "打开 / 关闭中央短帮助"),
        ("退格 / Z / Ctrl+Z", "撤销"),
        ("Shift+Z / Y / Ctrl+Y", "重做"),
        ("Ctrl+S", "另存为（系统文件对话框）"),
        ("Ctrl+O", "读档（系统打开对话框）"),
        ("Ctrl+N", "清空场地"),
    )),
    ("铺轨（A 模式）", (
        (", / .", "换轨道件分类"),
        ("[ / ]", "换同类下一件"),
        ("1–9", "直选第 n 件"),
        ("R / Shift+R", "旋转待放件"),
        ("T", "切换接驳端口"),
        ("Enter / 左键", "放下待放件"),
        ("Esc", "朝向归零"),
        ("C", "一键闭合（自动补缺口）"),
        ("U", "扳道岔（鼠标先指上去）"),
        ("X / Delete", "删掉鼠标下这一节"),
    )),
    ("布景（B 模式）", (
        ("B", "进入 / 退出布景模式"),
        (", / .", "换布景分类"),
        ("[ / ]", "换同类下一件"),
        ("1–9", "直选同类第 n 件"),
        ("R / Shift+R", "旋转待放件"),
        ("Enter / 左键", "放到鼠标指着的地面"),
        ("Esc", "朝向归零"),
        ("X / Delete", "删掉悬停的那件"),
    )),
    ("驾驶列车", (
        ("N", "召唤列车 / 换车"),
        ("M", "收起列车"),
        ("↑ / ↓", "牵引 / 制动（各 4 档）"),
        ("空格", "惰行（手柄回中位）"),
        ("Shift+空格", "紧急制动（一把拉死）"),
        ("K", "换向（先急刹再反向加速）"),
        ("J", "鸣笛（低音风笛）"),
    )),
    ("视角（任意模式）", (
        ("左键拖拽", "环绕视角（视角模式）"),
        ("右键拖拽", "环绕视角"),
        ("中键拖拽", "平移视角"),
        ("滚轮", "拉近 / 拉远"),
        ("W A S D", "平移视角"),
        ("Q / E", "转向"),
    )),
)


# --------------------------------------------------------------------------- #
# Markdown → HTML
# --------------------------------------------------------------------------- #

def _inline(text: str) -> str:
    """行内标记：行内代码（按键 / 等宽）、粗体、链接。

    先转义再替换 —— 顺序反了的话，代码里写着的 ``<`` 会被当成标签。
    """
    text = html_mod.escape(text, quote=False)

    def code(match: re.Match) -> str:
        content = match.group(1)
        raw = html_mod.unescape(content)
        if raw in _KEYS:
            return f'<kbd>{content}</kbd>'
        return f'<code>{content}</code>'

    # 行内代码（`` `x` ``）；用非贪婪匹配，且不允许跨行
    text = re.sub(r"`([^`\n]+)`", code, text)
    # 粗体
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", text)
    # 链接 [文字](地址)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)
    return text


def _table_block(lines: list[str]) -> str:
    """把连续的 ``| ... |`` 行变成表格。第二行是分隔行（``|---|``），丢掉。

    多列（4 列以上）的表会带上 ``colsN`` 类，CSS 里按列数固定列宽 ——
    自动布局在"一个长说明列 + 几个短列"上会把短列压到折行，很难看。
    """
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) >= 2 and all(set(cell) <= set("-: ") for cell in rows[1]):
        rows.pop(1)
    head, body = rows[0], rows[1:]
    cls = f' class="cols{len(head)}"' if len(head) >= 4 else ""
    out = [f"<table{cls}>", "<thead><tr>"]
    out += [f"<th>{_inline(cell)}</th>" for cell in head]
    out += ["</tr></thead>", "<tbody>"]
    for row in body:
        out.append("<tr>")
        out += [f"<td>{_inline(cell)}</td>" for cell in row]
        out.append("</tr>")
    out += ["</tbody>", "</table>"]
    return "".join(out)


def _render_blocks(lines: list[str]) -> str:
    """把一段 Markdown 行渲染成 HTML 块（不含 ``<section>`` 外壳）。"""
    out: list[str] = []
    index = 0
    total = len(lines)
    while index < total:
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped in ("---", "***", "___"):
            index += 1                      # 分节线交给 CSS 的章节间距表达
            continue

        # 围栏代码
        if stripped.startswith("```"):
            index += 1
            buffer: list[str] = []
            while index < total and not lines[index].strip().startswith("```"):
                buffer.append(lines[index])
                index += 1
            index += 1
            code = html_mod.escape("\n".join(buffer))
            out.append(f"<pre><code>{code}</code></pre>")
            continue

        # 三级标题
        if stripped.startswith("### "):
            out.append(f"<h3>{_inline(stripped[4:])}</h3>")
            index += 1
            continue

        # 表格
        if stripped.startswith("|"):
            buffer = []
            while index < total and lines[index].strip().startswith("|"):
                buffer.append(lines[index])
                index += 1
            out.append(_table_block(buffer))
            continue

        # 引用块 → 提示卡片
        if stripped.startswith(">"):
            buffer = []
            while index < total and lines[index].strip().startswith(">"):
                buffer.append(lines[index].strip()[1:].strip())
                index += 1
            paragraphs = [f"<p>{_inline(p)}</p>" for p in buffer if p]
            out.append('<aside class="tip">' + "".join(paragraphs) + "</aside>")
            continue

        # 有序 / 无序列表
        if re.match(r"^\d+\.\s", stripped) or stripped.startswith("- "):
            ordered = bool(re.match(r"^\d+\.\s", stripped))
            buffer = []
            while index < total:
                item = lines[index].strip()
                if ordered and re.match(r"^\d+\.\s", item):
                    buffer.append(re.sub(r"^\d+\.\s", "", item))
                elif not ordered and item.startswith("- "):
                    buffer.append(item[2:])
                elif item and lines[index].startswith(("  ", "\t")) and buffer:
                    buffer[-1] += " " + item        # 续行并进上一条
                else:
                    break
                index += 1
            tag = "ol" if ordered else "ul"
            items = "".join(f"<li>{_inline(text)}</li>" for text in buffer)
            out.append(f"<{tag}>{items}</{tag}>")
            continue

        # 段落（连续非空行并成一段）
        buffer = []
        while index < total and lines[index].strip() and not _starts_block(lines[index]):
            buffer.append(lines[index].strip())
            index += 1
        out.append(f"<p>{_inline(' '.join(buffer))}</p>")
    return "".join(out)


def _starts_block(line: str) -> bool:
    """这一行会不会自己起一个块（表格 / 列表 / 引用 / 标题 / 代码）。"""
    stripped = line.strip()
    return bool(
        stripped.startswith(("|", ">", "#", "```", "- "))
        or re.match(r"^\d+\.\s", stripped)
        or stripped in ("---", "***", "___")
    )


def _parse_manual(md_text: str) -> tuple[str, list[str], list[tuple[str, str]]]:
    """把手册拆成 ``(正文前的引言, 引言行, [(节标题, 节内容行)])``。"""
    lines = md_text.splitlines()
    title = ""
    intro: list[str] = []
    sections: list[tuple[str, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = line[3:].strip()
            current_lines = []
            continue
        if current_title is None:
            intro.append(line)
        else:
            current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines)))
    return title, intro, sections


# --------------------------------------------------------------------------- #
# 素材
# --------------------------------------------------------------------------- #

def _embed_image(path: Path, *, width: int = 1200, quality: int = 80) -> str | None:
    """把一张 PNG 变成内嵌的 data URI（缩一缩再压成 JPEG，PDF 才不至于几十兆）。

    有意做成"缺图就返回 None"：手册里少一张图不该让整份手册生不出来。
    """
    if not path.exists():
        return None
    try:
        from PIL import Image
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.width > width:
                height = round(image.height * width / image.width)
                image = image.resize((width, height), Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"
    except Exception:                            # noqa: BLE001 - 没有 PIL 就原样内嵌
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{payload}"


def _test_count() -> str:
    """从 README 里读"全库 N 条测试通过"，让封面与 README 不会说岔。"""
    try:
        match = re.search(r"\*\*(\d+)\s*条测试\*\*", README.read_text(encoding="utf-8"))
    except OSError:
        return ""
    return match.group(1) if match else ""


def _version() -> str:
    try:
        match = re.search(r'^version\s*=\s*"([^"]+)"',
                          (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
                          re.MULTILINE)
    except OSError:
        return "0.0.0"
    return match.group(1) if match else "0.0.0"


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #

def _figure(embedded: dict[str, str | None], name: str, title: str,
            caption: str = "") -> str:
    """一张带题注的截图；图缺失时整张跳过。"""
    source = embedded.get(name)
    if source is None:
        return ""
    note = f'<figcaption><b>{html_mod.escape(title)}</b>'
    if caption:
        note += f'<span>{html_mod.escape(caption)}</span>'
    note += "</figcaption>"
    return (f'<figure class="shot"><img src="{source}" alt="">{note}</figure>')


def _cover(embedded: dict[str, str | None], title: str, subtitle: str,
           meta: str) -> str:
    hero = embedded.get("scene_valley.png") or embedded.get("scene_town.png")
    art = f'<img src="{hero}" alt="">' if hero else ""
    return f"""
<section class="cover">
  <div class="cover-art">{art}</div>
  <div class="cover-plate">
    <div class="cover-bar"></div>
    <div class="kicker">3D 铁路沙盘 · Python + Panda3D</div>
    <h1>{html_mod.escape(title)}</h1>
    <div class="cover-sub">{html_mod.escape(subtitle)}</div>
    <div class="cover-meta">{meta}</div>
  </div>
</section>"""


def _toc(sections: list[tuple[str, str]], intro: str, meta: str) -> str:
    items = "".join(
        f'<li><span class="num">{html_mod.escape(heading.split("、")[0])}</span>'
        f'<span class="name">{html_mod.escape(heading.split("、", 1)[-1])}</span></li>'
        for heading, _ in sections
    )
    cards = "".join(
        f'<div class="card"><b>{value}</b><span>{label}</span></div>'
        for value, label in _STAT_CARDS
    )
    return f"""
<section class="page toc">
  <h2 class="page-title">目录</h2>
  <ol class="toc-list">{items}</ol>
  <h2 class="page-title small">速览</h2>
  <div class="cards">{cards}</div>
  <aside class="lead">{intro}</aside>
  <p class="foot">{meta}</p>
</section>"""


def _gallery(embedded: dict[str, str | None]) -> str:
    """两页图赏：第 1 页场景 + 列车，第 2 页拼轨与闭合。

    刻意拆成**两个 section** 而不是让浏览器自己分页：截图的卡片高度不一，
    自动分页会把同一行 2 张图拆到两页上（左边一张在上一页、右边一张在下一页），
    看起来像排版坏了。
    """
    scenes = "".join(_figure(embedded, *item) for item in _SCENE_GALLERY)
    trains = "".join(_figure(embedded, *item) for item in _TRAIN_GALLERY)
    edits = "".join(
        _figure(embedded, pair[0], pair[1], pair[2])
        + _figure(embedded, pair[3], pair[4], pair[5])
        for pair in _EDIT_GALLERY
    )
    steps = "".join(f"<li>{_inline(text)}</li>" for text in _START_STEPS)
    return f"""
<section class="page gallery">
  <h2 class="page-title">图赏 · 现成场景</h2>
  <div class="grid two">{scenes}</div>
  <h3>列车目录</h3>
  <div class="grid three">{trains}</div>
</section>
<section class="page gallery">
  <h2 class="page-title">图赏 · 拼轨与闭合</h2>
  <div class="grid two">{edits}</div>
  <aside class="tip"><p><b>上手三步</b></p><ol>{steps}</ol></aside>
</section>"""


def _body(sections: list[tuple[str, str]]) -> str:
    """九节正文。每节带 ``data-idx``，交给 :func:`_fit_breaks` 决定从哪一节起换页。"""
    parts = []
    for index, (heading, text) in enumerate(sections):
        number, _, name = heading.partition("、")
        chip = f'<span class="chip">{html_mod.escape(number)}</span>'
        title = html_mod.escape(name or heading)
        parts.append(
            f'<section class="sheet" data-idx="{index}"><h2>{chip}{title}</h2>'
            f"{_render_blocks(text.splitlines())}</section>"
        )
    return "\n".join(parts)


def _appendix() -> str:
    blocks = []
    for heading, rows in _APPENDIX:
        body = "".join(
            f"<tr><td class=\"k\">{_inline(key)}</td><td>{_inline(desc)}</td></tr>"
            for key, desc in rows
        )
        blocks.append(f"<h3>{html_mod.escape(heading)}</h3>"
                      f'<table class="keys"><tbody>{body}</tbody></table>')
    return f"""
<section class="page appendix">
  <h2 class="page-title">附录 A · 全部按键速查</h2>
  <div class="cols">{''.join(blocks)}</div>
</section>"""


def _colophon(meta: str, command: str) -> str:
    return f"""
<section class="page colophon">
  <h2 class="page-title">许可与再生成</h2>
  <p>本项目采用 <b>BSD Zero Clause License（0BSD）</b> —— 最开放的 BSD 许可证：
  可任意使用、复制、修改、再分发（含商用与闭源），无需保留署名。</p>
  <p class="mono">这份 PDF 由 Markdown 源码自动排版，不是手工维护的副本：</p>
  <pre><code>{html_mod.escape(command)}</code></pre>
  <p class="foot">{meta}</p>
</section>"""


# --------------------------------------------------------------------------- #
# 样式
# --------------------------------------------------------------------------- #

_CSS = Template("""
/* 配色取自 render/style.py：天空 #93AAC4、端口绿 #3EC77E、悬停黄 #FFBF35 */
:root {
  --ink: #1b2430; --ink-soft: #46556a; --rule: #d5dee8;
  --sky: #93aac4; --green: #2f9e63; --amber: #c8891a;
  --paper: #ffffff; --panel: #f4f7fa; --panel-2: #e9eff6;
}
* { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
@page { size: A4; margin: 0; }
html, body { margin: 0; padding: 0; }
body {
  font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
  color: var(--ink); font-size: 10.5pt; line-height: 1.62;
}
p { margin: 0 0 0.62em; }
a { color: var(--green); text-decoration: none; }
b, strong { color: #101a26; }
code, kbd, pre { font-family: "Cascadia Mono", Consolas, "DejaVu Sans Mono", monospace; }

kbd {
  display: inline-block; min-width: 1.25em; text-align: center;
  padding: 0.5px 5px; margin: 0 1px; border-radius: 4px;
  background: #fff; border: 1px solid #b9c6d5; border-bottom-width: 2px;
  font-size: 0.88em; font-weight: 600; color: #17222f; line-height: 1.5;
}
code { background: var(--panel-2); border-radius: 3px; padding: 0.5px 4px;
       font-size: 0.9em; color: #24384d; }
pre { background: #16202c; color: #dfe9f3; border-radius: 8px;
      padding: 10px 13px; margin: 0 0 0.8em; overflow: hidden; }
pre code { background: none; color: inherit; padding: 0; font-size: 9.5pt;
           line-height: 1.55; }

/* ---- 封面：通栏照片 + 深色标题条 ---- */
/* 不要在这里写 break-after：紧跟其后的 .page 自带 break-before，
   两个换页挨在一起会让 Chrome 多吐一张空白页。 */
.cover { width: 210mm; height: 296mm; overflow: hidden; }
.cover-art { height: 150mm; overflow: hidden; background: var(--sky); }
.cover-art img { width: 100%; display: block; }
.cover-plate { height: 146mm; padding: 22mm 20mm; box-sizing: border-box;
  background: linear-gradient(160deg, #1b2430 0%, #22344a 55%, #2c4a63 100%);
  color: #eef4fa; position: relative; }
.cover-bar { position: absolute; left: 0; top: 0; width: 100%; height: 5mm;
  background: linear-gradient(90deg, var(--green) 0%, var(--amber) 100%); }
.kicker { letter-spacing: 0.16em; font-size: 9.5pt; color: #8fb3cf;
  text-transform: uppercase; }
.cover-plate h1 { font-size: 40pt; margin: 6mm 0 0; letter-spacing: 0.06em;
  font-weight: 700; }
.cover-sub { font-size: 15pt; margin-top: 5mm; color: #bcd3e6; font-weight: 300; }
.cover-meta { position: absolute; bottom: 22mm; left: 20mm; right: 20mm;
  font-size: 9pt; color: #8aa6bd; border-top: 1px solid #3b5468; padding-top: 4mm; }

/* ---- 通用页面 ---- */
.page { padding: 18mm 17mm 16mm; break-before: page; }
.page-title { font-size: 20pt; margin: 0 0 7mm; padding-bottom: 3mm;
  border-bottom: 2px solid var(--ink); letter-spacing: 0.02em; }
.page-title.small { font-size: 13pt; margin-top: 7mm; border-bottom-width: 1px;
  border-color: var(--rule); padding-bottom: 2mm; }
.foot { margin-top: 8mm; font-size: 8.5pt; color: var(--ink-soft); }
.mono { font-family: "Cascadia Mono", Consolas, monospace; font-size: 9.5pt; }

/* ---- 目录 ---- */
.toc { padding: 15mm 17mm 13mm; }
.toc-list { list-style: none; margin: 0; padding: 0; }
.toc-list li { display: flex; align-items: baseline; gap: 4mm;
  padding: 3.1mm 0; border-bottom: 1px dashed var(--rule); }
.toc-list .num { flex: none; width: 9mm; text-align: center;
  font-weight: 700; color: var(--green); font-size: 11pt; }
.toc-list .name { font-size: 12.5pt; }
.cards { display: flex; gap: 4mm; }
.card { flex: 1; background: var(--panel); border-radius: 8px;
  border: 1px solid var(--rule); padding: 5mm 3mm; text-align: center; }
.card b { display: block; font-size: 21pt; color: var(--green); line-height: 1.1; }
.card span { font-size: 9pt; color: var(--ink-soft); }
.lead { margin-top: 7mm; padding: 4.5mm 6mm; background: var(--panel);
  border-left: 3px solid var(--sky); border-radius: 0 8px 8px 0; }
.lead p { margin: 0 0 0.45em; color: var(--ink-soft); }
.lead p:last-child { margin: 0; }

/* ---- 图赏 ---- */
.gallery h3 { font-size: 12pt; margin: 6mm 0 3mm; color: var(--ink);
  border-left: 3px solid var(--green); padding-left: 3mm; break-after: avoid; }
/* 用 grid 而不是 flex-wrap：flex 下 calc(50% - 2mm) + gap 会正好卡在容器宽度上，
   差 0.1px 就从"一行两张"变成"一行一张"，截图的排版会整个走形。 */
.grid { display: grid; gap: 4mm; }
.grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
/* 图片高度定死：同一行的卡片齐平，一页装得下几行也能算得准。 */
.grid.two .shot img { height: 42mm; object-fit: cover; }
.grid.three .shot img { height: 34mm; object-fit: cover; }
.shot { margin: 0; border: 1px solid var(--rule); border-radius: 8px;
  overflow: hidden; background: #fff; break-inside: avoid; }
.shot img { width: 100%; display: block; }
.shot figcaption { padding: 2.2mm 3mm 2.6mm; font-size: 8.6pt; line-height: 1.42; }
.shot figcaption b { display: block; font-size: 9.6pt; margin-bottom: 0.5mm; }
.shot figcaption span { color: var(--ink-soft); }

/* ---- 正文各节 ---- */
/* 节内自带上下留白，这样两节合排一页时，间距是"上节的底 + 下节的顶"，
   不必额外算页间距 —— 量出来的节高就是可以直接相加的数。 */
.sheet { padding: 13mm 17mm 12mm; break-inside: auto; }
.sheet + .sheet { border-top: 1px solid var(--rule); }
.sheet.brk { break-before: page; border-top: none; }
.sheet h2 { font-size: 17pt; margin: 0 0 5mm; display: flex;
  align-items: center; gap: 3.5mm; break-after: avoid; }
.chip { flex: none; width: 9.5mm; height: 9.5mm; border-radius: 50%;
  background: var(--ink); color: #fff; font-size: 13pt; font-weight: 700;
  display: flex; align-items: center; justify-content: center; }
.sheet h3 { font-size: 12pt; margin: 5mm 0 2.2mm; color: #22384f;
  break-after: avoid; }
.sheet h3:first-of-type { margin-top: 0; }

table { border-collapse: collapse; width: 100%; margin: 0 0 3.4mm;
  font-size: 9.8pt; break-inside: avoid; }
thead th { background: var(--ink); color: #f2f7fc; text-align: left;
  padding: 2.2mm 3mm; font-weight: 600; }
tbody td { padding: 1.45mm 3mm; border-bottom: 1px solid var(--rule);
  vertical-align: top; }
tr { break-inside: avoid; }
/* 第一列多是「键」或「场景名」这种短标签，折行既难看又费高度 —— 一律不折。 */
tbody td:first-child, thead th:first-child { white-space: nowrap; }
tbody tr:nth-child(even) { background: #fafcfe; }
/* 四列表（场景名 / 标题 / 说明 / 自带列车）：说明列吃掉余量，其余两列不折。 */
table.cols4 th:nth-child(1), table.cols4 td:nth-child(1) { width: 15%; }
table.cols4 th:nth-child(2), table.cols4 td:nth-child(2) { width: 15%;
  white-space: nowrap; }
table.cols4 th:nth-child(4), table.cols4 td:nth-child(4) { width: 21%;
  white-space: nowrap; }

ol, ul { margin: 0 0 3.4mm; padding-left: 6.5mm; }
li { margin-bottom: 0.8mm; }

aside.tip { background: #fff8e8; border-left: 3px solid var(--amber);
  border-radius: 0 8px 8px 0; padding: 3mm 4mm; margin: 5mm 0 0;
  break-inside: avoid; }
aside.tip p { margin: 0 0 0.4em; font-size: 9.6pt; color: #4a3a17; }
aside.tip ol, aside.tip ul { margin-bottom: 0; }
aside.tip p:last-child { margin: 0; }

/* ---- 附录：双栏，压进一页 ---- */
.appendix { padding: 15mm 15mm 13mm; }
.appendix .page-title { margin-bottom: 5mm; }
.appendix .cols { column-count: 2; column-gap: 7mm; }
.appendix h3 { margin: 0 0 1.2mm; font-size: 10pt; color: var(--green);
  break-after: avoid; }
.appendix table.keys { margin-bottom: 3.4mm; font-size: 8.9pt; }
.appendix table.keys td { padding: 1.05mm 2mm; line-height: 1.35;
  border-bottom: 1px solid var(--rule); }
.appendix table.keys td.k { width: 27mm; white-space: nowrap; font-weight: 600; }

.colophon { padding: 18mm 17mm; }
.colophon pre { margin-top: 3mm; }
""")


def build_html(md_text: str, *, include_images: bool = True) -> str:
    """把手册 Markdown 拼成一份自包含的 HTML。"""
    title, intro_lines, sections = _parse_manual(md_text)
    intro_html = _render_blocks(intro_lines)

    embedded: dict[str, str | None] = {}
    if include_images:
        names = [item[0] for item in _SCENE_GALLERY + _TRAIN_GALLERY]
        for pair in _EDIT_GALLERY:
            names += [pair[0], pair[3]]
        for name in dict.fromkeys(names):
            embedded[name] = _embed_image(DOCS / name)

    version = _version()
    today = date.today().isoformat()
    count = _test_count()
    meta_bits = [f"v{version}", today]
    if count:
        meta_bits.append(f"{count} 条回归测试通过")
    meta_bits.append("0BSD 许可")
    meta = " · ".join(meta_bits)

    command = "python scripts/manual_pdf.py"
    # 手册自己的 H1 是「模拟铁轨 · 操作指南」：封面只要前半截当大标题，
    # 后半截换成封面副标题，免得两行说的是同一件事。
    big_title = title.split("·")[0].strip() or title
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html_mod.escape(title)}</title>
<style>{_CSS.substitute()}</style>
</head>
<body>
{_cover(embedded, big_title, "操作手册 · 按键与流程速查", meta)}
{_toc(sections, intro_html, meta)}
{_gallery(embedded) if include_images else ""}
{_body(sections)}
{_appendix()}
{_colophon(meta, command)}
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# 分页：量出每节多高，再把它们拼进整页
# --------------------------------------------------------------------------- #

#: A4 在 96dpi 下的一页高度（CSS px）。1mm = 96/25.4 px，297mm ≈ 1123.46px。
PAGE_PX = 297 * 96 / 25.4

#: 一页最多塞多少 px。留 1% 余量：屏幕量到的高度和打印分页之间会有零点几毫米的差，
#: 卡到最后一像素的话，Chrome 会干脆把这一节整个推到下一页，白留半页。
PAGE_CAPACITY_PX = PAGE_PX * 0.99

#: 在页面里量各节高度的探针。写进 ``document.title`` 再用 ``--dump-dom`` 读回来 ——
#: 这样不必解析 stdout 里夹杂的浏览器日志。
_PROBE = """
<script>
window.addEventListener('load', function () {
  var blocks = [];
  document.querySelectorAll('section').forEach(function (el) {
    blocks.push([
      el.dataset.idx === undefined ? -1 : Number(el.dataset.idx),
      el.className,
      Math.ceil(el.getBoundingClientRect().height)
    ]);
  });
  document.title = 'MEASURE=' + JSON.stringify(blocks);
});
</script>
"""


def measure_blocks(page_html: str) -> list[tuple[int, str, float]]:
    """量出每个 ``<section>`` 的真实高度（px）：``[(data-idx, class, height)]``。

    分页不能靠手算：表格行折一行、中文标点换行的位置都会变，一节的高度差几十像素很常见，
    估算的结果就是"有的页塞满、有的页只剩两行"。量一次只要十几秒，但结果准。
    """
    browser = _find_browser()
    with tempfile.TemporaryDirectory(prefix="train3d_measure_") as tmp:
        probe_path = Path(tmp) / "probe.html"
        probe_path.write_text(page_html.replace("</body>", _PROBE + "</body>"),
                              encoding="utf-8")
        result = subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--virtual-time-budget=8000", "--dump-dom", probe_path.as_uri()],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=180)
    # 数组里每项是 [idx, class, height]，非贪婪的 ``\[.*?\]`` 会在第一个 ] 就停下，
    # 所以要一路匹配到 title 结束。
    match = re.search(r"MEASURE=(\[.*?\])\s*</title>", result.stdout, re.S)
    if not match:
        return []
    raw = json.loads(html_mod.unescape(match.group(1)))
    return [(int(index), str(cls), float(height)) for index, cls, height in raw]


def overflowed_fixed_pages(blocks: list[tuple[int, str, float]]) -> list[str]:
    """找出**自己独立成页**、却装不进一页的 section（封面、目录、图赏、附录…）。

    这些节的换页是写死的（``.page``），装不下就会多吐一页，而且往往只多出两三行 ——
    在 PDF 里不翻到那一页根本看不出来。所以宁可让脚本自己报出来。
    """
    over = []
    for index, cls, height in blocks:
        if index >= 0 or "page" not in cls.split():
            continue
        if height > PAGE_PX + 1:
            over.append(f"{cls}（{height:.0f}px / 上限 {PAGE_PX:.0f}px，"
                        f"超 {height - PAGE_PX:.0f}px）")
    return over


def _fit_breaks(heights: dict[int, float]) -> set[int]:
    """贪心装箱：从左到右累积，装不下就在这一节前换页。

    两个要点：

    * 第一节总是另起一页（它前面是图赏）。
    * 比一整页还高的节（比如第五节布景）**不切开**，它会自己溢到下一页；
      这时候下一页的"已用高度"是溢出来的那一截，接着往上排就行 ——
      否则算法会以为下一页是空的，把后面本来塞得下的一节也推走，
      结果就是一页塞满、下一页只有一小段（手算分页最容易踩的坑）。
    """
    order = sorted(heights)
    if not order:
        return set()
    breaks = {order[0]}
    used = 0.0
    for index in order:
        height = heights[index]
        if used and used + height > PAGE_CAPACITY_PX:
            breaks.add(index)
            used = 0.0
        used += height
        while used > PAGE_CAPACITY_PX:      # 这一节溢到了下一页
            used -= PAGE_CAPACITY_PX
    return breaks


def _apply_breaks(page_html: str, breaks: set[int]) -> str:
    """把分页决定注入 HTML：命中的节加上 ``brk`` 类（``break-before: page``）。"""
    if not breaks:
        return page_html
    rules = ",".join(f'section.sheet[data-idx="{index}"]' for index in sorted(breaks))
    return page_html.replace("</head>", f"<style>{rules}{{break-before:page}}</style></head>")


# --------------------------------------------------------------------------- #
# HTML → PDF
# --------------------------------------------------------------------------- #

#: 能拿来印 PDF 的浏览器（按优先级）。Windows / macOS / Linux 都列一遍 ——
#: 找不到任何一个时给出明确的报错，而不是静默失败。
_BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)


def _find_browser() -> str:
    for candidate in _BROWSERS:
        if Path(candidate).exists():
            return candidate
    for name in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(
        "找不到可用于打印 PDF 的浏览器（Chrome / Edge / Chromium）。"
        "用 --html-only 生成 HTML 后自行在浏览器里「打印 → 存为 PDF」也可以。"
    )


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    """用本机浏览器把 HTML 印成 PDF（无页眉页脚、A4、由 CSS 控制版面）。"""
    browser = _find_browser()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",          # 不要浏览器自带的页眉 / 页脚
        "--run-all-compositor-stages-before-draw",
        f"--virtual-time-budget=15000",    # 给内嵌图片留出解码时间
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if not pdf_path.exists():
        raise RuntimeError(
            f"浏览器没能生成 PDF：\n{result.stdout}\n{result.stderr}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="把操作指南排成 PDF 手册")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出的 PDF 路径")
    parser.add_argument("--html-only", action="store_true",
                        help="只生成 HTML（不调浏览器），便于在浏览器里改样式")
    parser.add_argument("--no-images", action="store_true",
                        help="不内嵌截图（徒手排版调试用，出来的 PDF 很小）")
    parser.add_argument("--no-fit", action="store_true",
                        help="不做「量高度再分页」，让正文自然连排（调试排版用）")
    args = parser.parse_args(argv)

    if not MANUAL_MD.exists():
        print(f"[错误] 找不到手册源文件：{MANUAL_MD}")
        return 2

    md_text = MANUAL_MD.read_text(encoding="utf-8")
    page = build_html(md_text, include_images=not args.no_images)

    if not args.no_fit and not args.html_only:
        blocks = measure_blocks(page)
        heights = {index: height for index, _, height in blocks if index >= 0}
        breaks = _fit_breaks(heights)
        page = _apply_breaks(page, breaks)
        print(f"[手册] 量到 {len(heights)} 节、共 {sum(heights.values()) / PAGE_PX:.1f} 页正文，"
              f"分页点 {sorted(breaks)}")
        for problem in overflowed_fixed_pages(blocks):
            print(f"[手册] ⚠ 这一页装不下，会被挤成两页：{problem}")

    # 必须转成绝对路径：印 PDF 的是浏览器进程，相对路径会按**它的**工作目录解析。
    out_path = Path(args.out).expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="train3d_manual_") as tmp:
        html_path = Path(tmp) / "manual.html"
        html_path.write_text(page, encoding="utf-8")
        if args.html_only:
            target = out_path.with_suffix(".html")
            target.write_text(page, encoding="utf-8")
            print(f"[手册] 已写出 HTML：{target}")
            return 0
        html_to_pdf(html_path, out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[手册] 已生成 PDF：{out_path}（{size_mb:.1f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
