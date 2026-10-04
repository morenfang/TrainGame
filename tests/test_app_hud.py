"""HUD 的版面契约：锚点、贴边、底衬贴身、面板互不压住，以及**渲染位置**。

为什么值得为几块文字面板写测试
------------------------------------------------
HUD 的坐标在 ``aspect2d`` 里是"高度归一化、宽度乘宽高比"的，而且本项目开了
``coordinate-system yup`` —— 屏幕的"上"是 **y**，**z 是进深**。把纵向写进 z
不会报错，只会让所有面板一起塌到屏幕正中间：同一列的两块文字直接叠在一起，
看着像"渲染坏了"。

这不是假设出来的风险，是真出过一次的故障。当时已有的几何断言**全都通过**了，
因为它们和实现犯的是同一个错：拿 ``getPos()[2]``（z）当纵向去比对。
所以这里除了几何断言，还补了一条**像素断言**：离屏渲染一遍，
数出"互不相连的块"个数 —— 面板塌成一坨时这个数会变小，几何断言看不出来。
"""
from __future__ import annotations

import pytest
from panda3d.core import PNMImage

from app import editor as editor_mod
from app.hud import _MARGIN, Hud, build_default_hud

#: 浮点比较容差（aspect2d 单位，屏幕高度 = 2.0）。
EPS = 1e-5

#: 最坏情况文本：HUD 上真正会出现的最长内容。用最坏值测版面才叫测版面。
#:
#: ``help`` 直接引用编辑器那边的常量，而不是抄一遍：抄一遍的话，往帮助里加一行
#: 只会让测试继续用旧文本通过，而真实窗口里的字已经挤出面板压到一块了。
_WORST_CASE = {
    "status": ("轨道编辑器 · 放置轨道\n"
               "当前件  左转 R40 22.5° 弯轨   [曲线 8/8]\n"
               "分类  [直轨] 曲线 坡道 高架桥 交叉 道岔 终端\n"
               "接驳端口  a（吸附中）\n"
               "轨道 16 节   总长 502.7 m   三角形 29,504\n"
               "\n"
               "鼠标下：#15 右开道岔 40m  道岔档位 route #3"),
    "loop": ("未闭环  已铺 13 段 / 408.41 m\n"
             "缺口 3.4210 m   航向差 12.500°\n"
             "原因：末端端口没有接上\n"
             "再补 8 节「左弯轨 R40 / 22.5°」→ 按 C 自动补"),
    "help": editor_mod.HELP_TEXT,
    "train": ("列车  老式绿皮车 · 机车 + 16 辆\n"
              "17 节 / 176.2 m   三角形 48,118\n"
              "速度  100.0 km/h   （27.78 m/s）　倒行\n"
              "手柄  牵引 100%\n"
              "里程  12345.6 m   运行 1234.5 s"),
    "toast": "已合拢 0/a → 7/b（误差 0.000000 mm）",
}

#: 逼真的窗口宽高比：4:3、5:4、16:10、16:9、21:9，外加一个极窄的竖屏。
_ASPECTS = (0.625, 1.25, 1.333, 1.6, 1.778, 2.37)


def panel_box(hud: Hud, name: str) -> tuple[float, float, float, float]:
    """面板实测的屏幕矩形 ``(x0, x1, y0, y1)``，单位是 aspect2d（纵向 = y）。"""
    box = hud.panel_box(name)
    assert box is not None, f"{name} 没有排版结果（是不是空文本？）"
    return box


def overlaps(a, b) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return min(ax1, bx1) - max(ax0, bx0) > EPS and \
        min(ay1, by1) - max(ay0, by0) > EPS


# --------------------------------------------------------------------------- #
# 基本行为
# --------------------------------------------------------------------------- #

def test_unknown_anchor_is_rejected(app):
    hud = Hud(app)
    with pytest.raises(ValueError, match="锚点"):
        hud.add_panel("nope", "middle")


def test_panels_are_created_at_scale(app):
    hud = Hud(app)
    hud.add_panel("a", "tl", scale=0.07)
    hud.set_text("a", "测试")
    assert hud._panels["a"].node.getScale()[0] == pytest.approx(0.07)


def test_set_visible_toggles_the_root(app):
    hud = Hud(app)
    hud.add_panel("a")
    assert not hud.root.isHidden()
    hud.set_visible(False)
    assert hud.root.isHidden() and hud.visible is False
    hud.set_visible(True)
    assert not hud.root.isHidden() and hud.visible is True


def test_remove_panel_frees_the_name(app):
    hud = Hud(app)
    hud.add_panel("a")
    hud.remove_panel("a")
    with pytest.raises(KeyError):
        hud.set_text("a", "再来")


def test_identical_text_is_not_repositioned(app):
    """文本没变时应当整体早退，而不是重排一遍。

    验证手法：把挂点挪到一个哨兵位置，再喂同一段文本 —— 如果实现真的早退了，
    位置就还停在哨兵处；如果它老老实实重排，位置会被改回去。
    """
    hud = Hud(app)
    hud.add_panel("a", "tl")
    hud.set_text("a", "同一段文字")
    holder = hud._panels["a"].holder
    holder.setPos(0.123, -0.456, 0.0)
    hud.set_text("a", "同一段文字")
    assert holder.getPos()[0] == pytest.approx(0.123)
    assert holder.getPos()[1] == pytest.approx(-0.456)


# --------------------------------------------------------------------------- #
# 版面几何
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("anchor", ["tl", "tr", "bl", "br", "bc"])
def test_each_anchor_keeps_the_panel_on_screen(app, anchor):
    hud = Hud(app)
    hud.add_panel("a", anchor)
    hud.set_text("a", _WORST_CASE["help"])

    aspect = hud.aspect
    x0, x1, y0, y1 = panel_box(hud, "a")
    assert x0 >= -aspect + _MARGIN - EPS
    assert x1 <= aspect - _MARGIN + EPS
    assert y0 >= -1.0 + _MARGIN - EPS
    assert y1 <= 1.0 - _MARGIN + EPS

    # 锚点方向也要真的贴住：贴顶的必须顶到边距处，贴底的必须落到边距处
    if anchor in ("tl", "tr"):
        assert y1 == pytest.approx(1.0 - _MARGIN, abs=EPS)
    else:
        assert y0 == pytest.approx(-1.0 + _MARGIN, abs=EPS)
    if anchor in ("tl", "bl"):
        assert x0 == pytest.approx(-aspect + _MARGIN, abs=EPS)
    if anchor in ("tr", "br"):
        assert x1 == pytest.approx(aspect - _MARGIN, abs=EPS)
    if anchor == "bc":
        assert x0 == pytest.approx(-x1, abs=EPS)     # 居中


def test_panel_box_matches_the_rendered_card(app):
    """HUD 报出来的矩形必须就是底衬真正占的那块几何。

    只测"贴边"是不够的：底衬自己算错尺寸时，贴边依然成立。这条把
    :meth:`Hud.panel_box` 和节点的实际包围盒钉在一起，框和字才不会错开。
    """
    hud = Hud(app)
    hud.add_panel("a", "tl", scale=0.05)
    hud.set_text("a", "两行\n文字")

    x0, x1, y0, y1 = panel_box(hud, "a")
    panel = hud._panels["a"]
    holder_pos = panel.holder.getPos()
    p0, p1 = panel.node.getTightBounds()
    assert x0 == pytest.approx(holder_pos[0] + p0[0], abs=EPS)
    assert x1 == pytest.approx(holder_pos[0] + p1[0], abs=EPS)
    assert y0 == pytest.approx(holder_pos[1] + p0[1], abs=EPS)
    assert y1 == pytest.approx(holder_pos[1] + p1[1], abs=EPS)


def test_card_wraps_the_text_and_scales_with_the_font(app):
    """底衬要包住文字（四边都留白），且留白按**文本单位**给 —— 随字号一起缩放。

    不断言留白的具体数值：那是 Panda3D 自己卡片框的口径（``setCardAsMargin``
    还会带上字体行高），照抄过来只是把实现细节再写一遍。要守住的性质是
    "四边都有留白"和"留白与字号成正比"—— 后者意味着换字号时观感一致。
    """
    def margins(scale: float):
        hud = Hud(app)
        hud.add_panel("a", "tl", scale=scale)
        hud.set_text("a", "两行\n文字")
        node = hud._panels["a"].node
        effective = node.getScale()[0]
        with_card = node.getTightBounds()
        node.node().clearCard()
        without = node.getTightBounds()
        hud.remove_panel("a")
        return effective, (without[0][0] - with_card[0][0],
                           with_card[1][0] - without[1][0],
                           without[0][1] - with_card[0][1],
                           with_card[1][1] - without[1][1])

    scale, big = margins(0.05)
    for value in big:
        assert value > 0.0, f"底衬有一边没包住文字：{big}"

    small_scale, small = margins(0.03)
    for big_value, small_value in zip(big, small):
        assert big_value / scale == pytest.approx(small_value / small_scale,
                                                  rel=1e-3)


def test_empty_text_hides_the_panel_entirely(app):
    """空文本不能留下"什么都没有"的黑块挡场景 —— 这是真出现过的故障。"""
    hud = Hud(app)
    hud.add_panel("a", "bc")
    hud.set_text("a", "先有内容")
    assert hud.panel_box("a") is not None

    hud.set_text("a", "")
    assert hud.panel_box("a") is None
    assert hud._panels["a"].holder.isHidden()


def test_same_anchor_panels_stack_without_overlapping(app):
    """同一锚点多块面板要沿纵向堆叠，而不是叠在同一个位置。"""
    hud = Hud(app)
    hud.add_panel("first", "tl")
    hud.add_panel("second", "tl")
    hud.set_text("first", "第一块\n第二行")
    hud.set_text("second", "第二块")

    first = panel_box(hud, "first")
    second = panel_box(hud, "second")
    assert not overlaps(first, second)
    top = 1.0 - _MARGIN
    assert first[3] == pytest.approx(top, abs=EPS)          # 第一块贴顶
    assert second[3] < first[2]                             # 第二块在它下面


@pytest.mark.parametrize("aspect", _ASPECTS)
def test_default_hud_panels_never_overlap(app, aspect):
    """默认四块面板在"最坏文本 + 各种窗口比例"下都不能互相压住。

    以前这条只在测试窗口那一个比例上跑过，窄窗口下的挤压就漏掉了
    （所以 :class:`Hud` 才把锚点归成三条互不相交的纵列）。
    """
    hud = build_default_hud(app)
    hud.aspect_override = aspect
    for name, text in _WORST_CASE.items():
        hud.set_text(name, text)

    rects = {name: panel_box(hud, name) for name in _WORST_CASE}
    names = sorted(rects)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            assert not overlaps(rects[first], rects[second]), (
                f"aspect={aspect}: {first} 与 {second} 重叠 "
                f"{rects[first]} vs {rects[second]}"
            )
    for name, rect in rects.items():
        assert rect[0] >= -aspect + _MARGIN - EPS, name
        assert rect[1] <= aspect - _MARGIN + EPS, name
        assert rect[2] >= -1.0 + _MARGIN - EPS, name
        assert rect[3] <= 1.0 - _MARGIN + EPS, name


class _PromptStub:
    """借真身的方法量最坏提示条：只备齐 ``_prompt_text`` 用到的那几样。

    为什么要借而不是抄一份文本：提示条的措辞、名单预算、回显截断都在
    ``app/editor.py`` 里，抄一份的话改了实现这条测试就测不到真东西了。
    """

    _prompt = editor_mod.PROMPT_LOAD
    #: 输入框的上限就是 40 个字符（``_SAVE_NAME_MAX``）。
    _prompt_buffer = "abcdefghijklmnopqrstuvwxyz0123456789abcd"
    _prompt_error = "　（abcdefghijklmnopqrstuvwxyz0123456789abcd.json 不存在）"
    _prompt_target = editor_mod.TrackEditor._prompt_target
    _prompt_target_label = editor_mod.TrackEditor._prompt_target_label
    _prompt_files = editor_mod.TrackEditor._prompt_files
    _prompt_text = editor_mod.TrackEditor._prompt_text

    def __init__(self, save_dir):
        self._save_dir = save_dir
        self.save_path = save_dir / "layout.json"


@pytest.mark.parametrize("aspect", _ASPECTS)
def test_load_prompt_line_never_leaves_the_screen(app, aspect, tmp_path):
    """写到最满的读档提示条也要落在屏幕内，且不压住还亮着的面板。

    "提示条是一行居中文字"这个设计最怕变长：40 字的文件名 + 一长串存档名单
    加起来能把面板顶出屏幕。所以名单有字符预算、回显有截断，这条测试就是给
    这两个上限兜底的。至于底部那两块 —— 填文件名时编辑器会把它们收起来
    （见 ``test_prompt_hides_the_bottom_panels``），这里照抄那个状态。
    """
    for index in range(20):
        (tmp_path / f"abcdefghijklmnopqrstuvwxyz0123456789abcd{index}.json"
         ).write_text("{}", encoding="utf-8")
    long_name = "abcdefghijklmnopqrstuvwxyz0123456789abcd"
    stub = _PromptStub(tmp_path)
    stub.save_path = tmp_path / f"{long_name}.json"
    text = stub._prompt_text()
    assert text, "最坏提示条不该是空的（否则面板会整块隐藏，测了个寂寞）"

    hud = build_default_hud(app)
    hud.aspect_override = aspect
    hud.set_text("status", _WORST_CASE["status"])
    hud.set_text("loop", _WORST_CASE["loop"])
    hud.set_text("help", "")            # 填文件名时这两块被收起来
    hud.set_text("train", "")
    hud.set_text("toast", text)

    toast = panel_box(hud, "toast")
    assert toast[0] >= -aspect + _MARGIN - EPS
    assert toast[1] <= aspect - _MARGIN + EPS
    assert toast[2] >= -1.0 + _MARGIN - EPS
    assert toast[3] <= 1.0 - _MARGIN + EPS
    for name in ("status", "loop"):
        assert not overlaps(toast, panel_box(hud, name)), (
            f"aspect={aspect}: 读档提示条压住了 {name}：{toast}"
        )


# --------------------------------------------------------------------------- #
# 渲染位置（几何断言兜不住的那一类）
# --------------------------------------------------------------------------- #

def _screenshot(base, *, with_hud: bool, hud: Hud) -> PNMImage:
    hud.set_visible(with_hud)
    for _ in range(2):
        base.graphicsEngine.renderFrame()
    image = PNMImage()
    base.win.getScreenshot(image)
    return image


def _hud_blocks(app, hud: Hud, *, step: int = 3) -> list[tuple[int, int, int, int]]:
    """渲染"有 HUD / 无 HUD"两帧相减，返回互不相连的 HUD 块。

    相减之后只有 HUD 会留下痕迹，背景是什么颜色都不影响判断。
    """
    on = _screenshot(app, with_hud=True, hud=hud)
    off = _screenshot(app, with_hud=False, hud=hud)
    width, height = on.getXSize(), on.getYSize()

    mask = [[False] * width for _ in range(height)]
    for y in range(0, height, step):
        for x in range(0, width, step):
            a, b = on.getXel(x, y), off.getXel(x, y)
            if (abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])) > 0.05:
                mask[y][x] = True

    seen = [[False] * width for _ in range(height)]
    blocks = []
    for y0 in range(0, height, step):
        for x0 in range(0, width, step):
            if not mask[y0][x0] or seen[y0][x0]:
                continue
            stack = [(x0, y0)]
            seen[y0][x0] = True
            xs, ys = [], []
            while stack:
                x, y = stack.pop()
                xs.append(x)
                ys.append(y)
                for dy in (-step, 0, step):
                    for dx in (-step, 0, step):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < width and 0 <= ny < height
                                and mask[ny][nx] and not seen[ny][nx]):
                            seen[ny][nx] = True
                            stack.append((nx, ny))
            if len(xs) >= 20:
                blocks.append((min(xs), max(xs), min(ys), max(ys)))
    return blocks


def test_panel_text_reads_back_what_was_set(app):
    """``panel_text`` 是给外部（脚本 / 调试）的只读查询，必须与 ``set_text`` 一致。"""
    hud = Hud(app)
    hud.add_panel("a", "tl")
    assert hud.panel_text("a") == ""
    hud.set_text("a", "速度 100.1 km/h")
    assert hud.panel_text("a") == "速度 100.1 km/h"


def test_rendered_hud_has_one_block_per_panel(app):
    """屏幕上要能看到"几块分开的 HUD"，而不是塌成一块或一条横贯的带子。

    这条是冲着真实故障去的：纵向轴写错时，四块底衬会全变成半屏高的黑带、
    同一列的两块叠在一起 —— 数出来的块数会从 4 掉到 3，且那块会占掉近两成屏幕。
    几何断言对这种情况无能为力（它自己也用了错的轴），只有数像素才看得见。
    """
    hud = build_default_hud(app)
    for name, text in _WORST_CASE.items():
        hud.set_text(name, text)

    blocks = _hud_blocks(app, hud)
    assert len(blocks) == len(_WORST_CASE), (
        f"应当看到 {len(_WORST_CASE)} 块分开的面板，实际 {len(blocks)} 块：{blocks}"
    )

    width = app.win.getXSize()
    height = app.win.getYSize()
    for (x0, x1, y0, y1) in blocks:
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        assert area < 0.12 * width * height, (
            f"有一块 HUD 占了 {(x1 - x0 + 1)}x{(y1 - y0 + 1)} 像素，"
            f"大得不像一块文字面板：x[{x0},{x1}] y[{y0},{y1}]"
        )


def test_rendered_top_panel_is_above_the_bottom_panel(app):
    """左上那块渲染出来必须在左下那块**上方**。

    这里不靠"图像第 0 行是顶部"这类约定：直接比较两块像素在图像行方向上的
    先后，并且要求方向与 :meth:`Hud.panel_box` 给出的 y 一致 —— 也就是说
    "声明的位置"和"画出来的位置"必须同向。轴写反时这条必挂。
    """
    hud = Hud(app)
    hud.add_panel("top", "tl")
    hud.add_panel("bottom", "bl")
    hud.set_text("top", "上面")
    hud.set_text("bottom", "下面")

    top_box = panel_box(hud, "top")
    bottom_box = panel_box(hud, "bottom")
    assert top_box[2] > bottom_box[3]        # 声明上：上面的在下面那块之上

    texts = {"top": "上面", "bottom": "下面"}

    def rows_of(name: str) -> tuple[int, int]:
        """只让一块面板有文字，量它的像素行范围。"""
        for other, text in texts.items():
            hud.set_text(other, text if other == name else "")
        blocks = _hud_blocks(app, hud)
        assert len(blocks) == 1, f"{name} 应当只有一块像素，实际 {blocks}"
        return blocks[0][2], blocks[0][3]

    top_rows = rows_of("top")
    bottom_rows = rows_of("bottom")
    # 图像行与 y 必须反向：y 大的在上面，所以图像行号要小
    assert top_rows[0] < bottom_rows[0], (
        f"左上那块落在图像行 {top_rows}，左下那块在 {bottom_rows} —— "
        "渲染方向和 panel_box 声明方向不一致"
    )
