"""操作手册的排版脚本：Markdown 转换、分页装箱、以及"手册有没有漏掉按键"。

``scripts/manual_pdf.py`` 干两件事 —— 把 ``docs/02-操作指南.md`` 转成 HTML，
再用本机浏览器印成 PDF。这里只测**不用浏览器**的那一半（纯函数）：

1. 行内标记：**按键渲染成键帽**、文件名保持等宽、转义不串味；
2. 块级：表格丢掉分隔行、引用变提示卡、围栏代码转义、``---`` 被吃掉；
3. :func:`_fit_breaks` 的装箱 —— 尤其是"比一整页还高的节会溢到下一页"这条，
   算错的话就会一页塞满、下一页只剩两行；
4. **附录是否覆盖了编辑器绑过的每一个键**（``TrackEditor.key_actions`` /
   ``HELD_KEYS``）—— 加了新键却忘了写手册，是这份文档最容易烂掉的地方。

印 PDF 那一步依赖本机装没装 Chrome / Edge，不进测试。
"""

from __future__ import annotations

import re

from app.editor import HELD_KEYS, TrackEditor
from scripts import manual_pdf as manual


class _Stub:
    """只要能"取到任意属性"就行。

    ``key_actions()`` 里既有 ``self.toogle_... `` 这样的取值（会立刻求值），
    也有 ``lambda: self.xxx()``（只在按下时才求值）。给个什么属性都有的假对象，
    就能不启动 Panda3D 问到"编辑器绑了哪些键"。
    """

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None


def _bound_events() -> list[str]:
    return list(TrackEditor.key_actions(_Stub())) + list(HELD_KEYS)


# --------------------------------------------------------------------------- #
# 行内标记
# --------------------------------------------------------------------------- #

def test_keys_become_keycaps_and_others_stay_monospace():
    html = manual._inline("按 `V` 切模式，`Ctrl+O` 读档，`Shift+空格` 急停")
    assert "<kbd>V</kbd>" in html
    assert "<kbd>Ctrl+O</kbd>" in html
    assert "<kbd>Shift+空格</kbd>" in html

    # 文件名 / 场景 id / 命令行不是按键，不能渲染成键帽
    other = manual._inline(r"`run.bat --scene valley` 与 `saves\layout.json`")
    assert "<kbd>" not in other
    assert "<code>run.bat --scene valley</code>" in other


def test_inline_escapes_angle_brackets_and_keeps_bold_and_links():
    assert "&lt;b&gt;" in manual._inline("`<b>` 不是标签")
    assert manual._inline("**急刹**") == "<strong>急刹</strong>"
    assert manual._inline("[手册](docs/02-操作指南.md)") == \
        '<a href="docs/02-操作指南.md">手册</a>'


def test_digits_inside_code_are_keys_but_scores_are_not():
    """``1`` 是按键，``0.5`` 是数值 —— 光看长度分不出来，靠白名单。"""
    assert "<kbd>1</kbd>" in manual._inline("按 `1` 直选")
    assert "<kbd>" not in manual._inline("限速 `0.5`")


# --------------------------------------------------------------------------- #
# 块级
# --------------------------------------------------------------------------- #

def test_table_drops_separator_row_and_tags_wide_tables():
    html = manual._render_blocks([
        "| 场景名 | 标题 | 说明 | 列车 |",
        "|---|---|---|---|",
        "| `valley` | 山谷环线 | 长环线 | 和谐号 |",
    ])
    assert 'class="cols4"' in html
    assert "---" not in html
    assert html.count("<tr>") == 2, "表头 + 一行数据，分隔行该被吃掉"
    assert "<td><code>valley</code></td>" in html


def test_narrow_table_has_no_column_class():
    html = manual._render_blocks(["| 键 | 效果 |", "|---|---|", "| `C` | 闭合 |"])
    assert "cols4" not in html
    assert "<kbd>C</kbd>" in html


def test_blockquote_becomes_tip_card_and_rule_is_dropped():
    html = manual._render_blocks(["> 提示一行", "> 还是这条提示", "", "---", "", "正文"])
    assert html.count('<aside class="tip">') == 1
    assert html.count("<p>") == 3, "两行引用合成一段 + 正文一段"
    assert "<hr" not in html


def test_fenced_code_is_escaped_and_lists_are_tagged():
    html = manual._render_blocks(["```", "run.bat --scene valley", "```", "",
                                  "1. 第一步", "2. 第二步", "", "- 摘要"])
    assert "<pre><code>run.bat --scene valley</code></pre>" in html
    assert "<ol><li>第一步</li><li>第二步</li></ol>" in html
    assert "<ul><li>摘要</li></ul>" in html


def test_parse_manual_splits_title_intro_and_sections():
    md = (
        "# 模拟铁轨 · 操作指南\n"
        "\n"
        "引言第一段\n"
        "\n"
        "引言第二段\n"
        "\n"
        "---\n"
        "\n"
        "## 一、启动\n"
        "\n"
        "正文 A\n"
        "\n"
        "## 二、视角\n"
        "\n"
        "正文 B\n"
    )
    title, intro, sections = manual._parse_manual(md)
    assert title == "模拟铁轨 · 操作指南"
    assert [name for name, _ in sections] == ["一、启动", "二、视角"]
    assert "引言第一段" in "\n".join(intro)
    assert "正文 A" in sections[0][1]
    assert "正文 B" in sections[1][1]


# --------------------------------------------------------------------------- #
# 分页装箱
# --------------------------------------------------------------------------- #

def test_fit_breaks_packs_whole_sections_and_starts_on_a_new_page():
    capacity = manual.PAGE_CAPACITY_PX
    heights = {0: capacity * 0.6, 1: capacity * 0.6, 2: capacity * 0.2}
    # 0 + 1 装不下 -> 1 换页；1 + 2 装得下 -> 2 跟着 1 走
    assert manual._fit_breaks(heights) == {0, 1}


def test_fit_breaks_accounts_for_a_section_spilling_over_one_page():
    capacity = manual.PAGE_CAPACITY_PX
    heights = {0: capacity * 1.5, 1: capacity * 0.3}
    # 第一节溢到第二页、还剩半页；这半页装得下第二节，就不该白换一页。
    assert manual._fit_breaks(heights) == {0}


def test_fit_breaks_breaks_when_the_spill_has_no_room():
    capacity = manual.PAGE_CAPACITY_PX
    heights = {0: capacity * 1.8, 1: capacity * 0.5}
    assert manual._fit_breaks(heights) == {0, 1}


def test_fit_breaks_on_empty_input_is_empty():
    assert manual._fit_breaks({}) == set()


def test_apply_breaks_injects_one_rule_per_breaking_section():
    html = manual._apply_breaks("<html><head></head><body></body></html>", {0, 3})
    assert ('section.sheet[data-idx="0"],section.sheet[data-idx="3"]'
            "{break-before:page}") in html
    assert manual._apply_breaks("<head></head>", set()) == "<head></head>"


# --------------------------------------------------------------------------- #
# 附录与正文内容
# --------------------------------------------------------------------------- #

def _appendix_tokens() -> set[str]:
    """附录左栏出现过的所有"键"（按 ``/``、``–``、空格拆开）。"""
    tokens: set[str] = set()
    for _, rows in manual._APPENDIX:
        for key, _ in rows:
            tokens.update(part.strip() for part in re.split(r"[/–\s]+", key))
    tokens.discard("")
    return tokens


#: ``app/editor.py`` 绑定的**事件名** → 手册里的**键名**。
_EVENT_TO_KEY = {
    "arrow_up": "↑", "arrow_down": "↓",
    "space": "空格", "shift-space": "Shift+空格",
    "backspace": "退格", "escape": "Esc", "enter": "Enter",
    "delete": "Delete",
}


def _display_name(event: str) -> str:
    """事件名 → 手册里的写法（``control-s`` → ``Ctrl+S``）。"""
    if event in _EVENT_TO_KEY:
        return _EVENT_TO_KEY[event]
    if event.startswith("control-"):
        return f"Ctrl+{event[len('control-'):].upper()}"
    if event.startswith("shift-"):
        return f"Shift+{event[len('shift-'):].upper()}"
    return event.upper()


def test_appendix_documents_every_key_the_editor_binds():
    """手册漏掉一个键，这条就红 —— 文档漂移比代码漂移更难发现。"""
    tokens = _appendix_tokens()
    # 数字键在附录里写成区间 "1–9"，先展开
    for token in list(tokens):
        if re.fullmatch(r"\d–\d", token):
            low, high = (int(part) for part in token.split("–"))
            tokens.update(str(number) for number in range(low, high + 1))

    missing = sorted(event for event in _bound_events()
                     if _display_name(event) not in tokens)
    assert not missing, f"这些键绑在编辑器上、却没写进手册附录：{missing}"


def test_appendix_has_no_key_that_the_editor_does_not_bind():
    """反过来也查：附录里写了、编辑器却不认的键，就是过期的文档。

    "视角"那一栏写的是鼠标（左键拖拽 / 右键 / 中键 / 滚轮），它们绑在
    ``mouse_actions()`` 上而不是键盘表上 —— 这里顺手把这一点也核对掉。
    """
    bound = {_display_name(event) for event in _bound_events()}
    bound.update(str(number) for number in range(1, 10))

    mouse_events = set(TrackEditor.mouse_actions(_Stub()))
    assert {"mouse1", "mouse2", "mouse3", "wheel_up", "wheel_down"} <= mouse_events
    mouse_tokens = {"左键", "左键拖拽", "右键拖拽", "中键拖拽", "滚轮"}

    stale = sorted(token for token in _appendix_tokens()
                   if token not in bound and token not in mouse_tokens
                   and not re.fullmatch(r"\d–\d|\d", token))
    assert not stale, f"附录里这些键编辑器并不认（文档过期）：{stale}"


def test_manual_lists_every_track_category_in_the_real_order():
    """铺轨那一节的分类清单得跟 ``data/pieces.json`` 对得上。

    "共 N 类轨道件"这种说法最容易过期 —— 目录里加了「终端」，正文那句却还写着
    六个名字（这次就是这样漏的），照着按 `,` 换一圈就会发现少了一类。
    """
    from core.track.catalog import Catalog

    order = [TrackEditor.category_label(category)
             for category in Catalog.builtin().categories]
    md = manual.MANUAL_MD.read_text(encoding="utf-8")
    line = next(line for line in md.splitlines() if "在分类间切换" in line)
    assert " → ".join(order) in line, f"分类清单过期，实际顺序是：{' → '.join(order)}"


def test_manual_source_is_the_single_source_of_truth():
    """手册正本必须真在 ``docs/02-操作指南.md`` 里 —— PDF 是它的排版产物。"""
    assert manual.MANUAL_MD.exists()
    md = manual.MANUAL_MD.read_text(encoding="utf-8")
    title, _, sections = manual._parse_manual(md)
    assert title == "模拟铁轨 · 操作指南"
    assert len(sections) == 9, "操作指南该是九节"
    assert "## 五、布景" in md and "## 七、驾驶列车" in md


def test_build_html_contains_cover_toc_body_and_appendix():
    md = manual.MANUAL_MD.read_text(encoding="utf-8")
    page = manual.build_html(md, include_images=False)
    assert "<h1>模拟铁轨</h1>" in page, "封面大标题该是书名，不是整行 H1"
    assert "操作手册 · 按键与流程速查" in page
    assert "目录" in page and "附录 A · 全部按键速查" in page
    assert page.count('class="sheet"') == 9, "正文九节"
    assert "data:image" not in page, "关掉配图后不该内嵌任何图片"


def test_manual_has_no_leftover_markdown_markers():
    """转换漏掉的 ``**`` / 反引号会原样印在 PDF 上，是最难发现的一类错。"""
    md = manual.MANUAL_MD.read_text(encoding="utf-8")
    body = manual.build_html(md, include_images=False).split("<body>", 1)[1]
    for marker in ("**", "](http", "```", "<kbd>`"):
        assert marker not in body, f"正文里还留着 Markdown 标记 {marker!r}"


def test_every_gallery_image_really_exists():
    """图赏里写错文件名的话，那一格会静默消失 —— 手册上少一张图不容易察觉。"""
    names = [item[0] for item in manual._SCENE_GALLERY + manual._TRAIN_GALLERY]
    names += [pair[index] for pair in manual._EDIT_GALLERY for index in (0, 3)]
    missing = [name for name in names if not (manual.DOCS / name).exists()]
    assert not missing, f"这些配图在 docs/ 里找不到：{missing}"
