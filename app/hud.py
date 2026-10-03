"""屏幕叠加文字面板（HUD）。

为什么自己写而不是用 ``DirectGui``
------------------------------------------------
HUD 上全是**每帧都可能变的中文**：件名、闭环缺口、撤销栈深度。``DirectGui``
的 ``OnscreenText`` 每次改文本都要重建 Geom，而且它默认用 Panda 自带字体
（不含汉字，会显示成方框）。这里直接用 ``TextNode`` + 一个共享的中文字体，
只在文本真的变了才重排，开销可以忽略。

二维坐标：**纵向是 y，不是 z**
------------------------------------------------
本项目全局 ``coordinate-system yup``，``aspect2d`` 下屏幕的"右"是 x、
"上"是 y，而 **z 是进深**（这一点 ``DirectGui`` 也遵守：它内部用
``Vec3.rfu(sx, 1, sy)`` / ``Point3.rfu(px, 0, py)``，在 yup 下 ``rfu(r,f,u)``
展开正好是 ``(r, u, -f)``，也就是把纵向写进 y）。

把纵向写进 z 是**不会报错**的：所有面板会一起塌到屏幕正中间，
同一列的两块文字直接叠在一起，看着像"渲染坏了"。这类故障只能靠
"实测矩形 + 渲染像素"两类断言兜住，光看代码看不出问题。

底衬为什么不自己造卡片
------------------------------------------------
原来是自己用 ``CardMaker`` 贴一块四边形，于是要手工算它的中心与缩放 —— 而
"卡片的屏幕高度该写进哪个 scale 分量"恰好是同一个坑：卡片几何在**局部 XY
平面**，屏幕纵向既然是 y，高度就必须写进 **y 的 scale**；写进 z 只改变进深，
屏幕上就变成一块**高达半屏**的黑带（单位卡片 y 方向正好 1.0，即屏幕高度的一半）。

改用 ``TextNode`` 自带的卡片（``setCardColor`` + ``setCardAsMargin``）之后，
底衬和字形同面、随文字自动贴身，尺寸不用再手工算一遍，也就没机会算错。

版面：分列 + 分层，**结构上不可能重叠**
------------------------------------------------
锚点先归到一条**纵列**（left / center / right）和一侧纵向带（top / bottom）：

    tl, bl → left      tr, br → right      bc → center

同一锚点的多块面板沿纵向**堆叠**（贴顶的向下排，贴底的向上排），
列与列之间按屏幕宽度的固定比例切分，互不相交。文字若宽于本列，
就按比例缩小该面板（缩到请求值的 35% 为止），因此**换窗口大小也不会挤在一起**。

空文本的面板**整块隐藏** —— 否则会留下一块"什么都没有"的黑块挡着场景。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from panda3d.core import NodePath, TransparencyAttrib

from render.text import cjk_font, make_label

#: 面板距屏幕边缘的留白（aspect2d 单位：屏幕高度 = 2.0、宽度 = 2 × 宽高比）。
_MARGIN = 0.022
#: 底衬相对文字外扩的留白，**单位是文本单位**（scale = 1），随面板缩放一起放大。
#: 注意 Panda3D 的卡片框是套在字体行框上的，所以屏幕上的实际留白会比这个数略大。
_PAD_X = 0.42
_PAD_Y = 0.22
#: 底衬颜色（半透明深色，保证任何背景上文字都可读）。
_PANEL_COLOR = (0.055, 0.067, 0.086, 0.52)
#: 同锚点堆叠时两块之间的空隙。
_STACK_GAP = 0.012
#: 文字窄于本列时按比例缩小的下限（相对请求值），避免在窄窗口里缩成看不清。
_MIN_SCALE_RATIO = 0.35

#: 锚点 → （纵列, 贴哪一侧）。纵列决定横向范围，一侧决定堆叠方向。
_ANCHORS: dict[str, tuple[str, str]] = {
    "tl": ("left", "top"),
    "tr": ("right", "top"),
    "bl": ("left", "bottom"),
    "br": ("right", "bottom"),
    "bc": ("center", "bottom"),
}
#: 各纵列占屏幕宽度的比例。左右两列 + 中间列必须明显小于 1，
#: 否则窄窗口下会碰在一起（0.33 + 0.33 + 0.30 = 0.96，留 4% 余量）。
_COLUMN_WIDTH = 0.33
_CENTER_WIDTH = 0.30
#: 贴顶/贴底的面板最多能各用掉屏幕高度的多少（两侧加起来必须 < 1）。
_BAND_HEIGHT = 0.44


@dataclass
class _Panel:
    #: 挂点：排版只动它的位置，不动文字节点自己的变换。
    holder: NodePath
    node: NodePath
    anchor: str
    #: 请求的字号；实际字号可能因为放不下而被缩小，存在 :attr:`scale` 里。
    scale: float
    color: tuple[float, float, float, float]
    text: str = ""
    #: 上一轮排版的实测屏幕矩形 ``(x0, x1, y0, y1)``，空文本时是 ``None``。
    box: tuple[float, float, float, float] | None = field(default=None)


class Hud:
    """一组按锚点定位的中文文字面板。"""

    def __init__(self, base, font=None, *, aspect_override: float | None = None):
        self.base = base
        self.font = font
        self.root = base.aspect2d.attachNewNode("hud")
        self._panels: dict[str, _Panel] = {}
        self.visible = True
        #: 强制使用某个窗口宽高比（给测试与离屏预览用；``None`` = 问窗口）。
        self.aspect_override = aspect_override
        self._aspect = self.aspect

    @property
    def aspect(self) -> float:
        if self.aspect_override is not None:
            return self.aspect_override
        return self.base.getAspectRatio()

    # ---------------------------------------------------------------- 构建

    def add_panel(self, name: str, anchor: str = "tl", *, scale: float = 0.040,
                  color=(1.0, 1.0, 1.0, 1.0)) -> None:
        if anchor not in _ANCHORS:
            raise ValueError(f"未知锚点 {anchor!r}，应为 {tuple(_ANCHORS)} 之一")

        holder = self.root.attachNewNode(f"panel_{name}")
        node = make_label("", font=self.font, color=color)
        node.reparentTo(holder)
        node.setScale(scale)
        node.setDepthTest(False)
        node.setDepthWrite(False)
        node.setTransparency(TransparencyAttrib.MAlpha)

        self._panels[name] = _Panel(holder=holder, node=node, anchor=anchor,
                                    scale=scale, color=tuple(color))
        self._hide(holder, node)

    def remove_panel(self, name: str) -> None:
        panel = self._panels.pop(name, None)
        if panel is not None:
            panel.holder.removeNode()

    def set_visible(self, flag: bool) -> None:
        self.visible = flag
        self.root.show() if flag else self.root.hide()

    # ---------------------------------------------------------------- 更新

    def set_text(self, name: str, text: str) -> None:
        """更新某个面板的文本；文本和窗口宽高比都没变就什么都不做。"""
        panel = self._panels[name]
        resized = self.aspect != self._aspect
        if panel.text == text and not resized:
            return
        panel.text = text
        self._relayout()

    def relayout(self) -> None:
        """按当前窗口宽高比重新排版（窗口被拖动后调用）。"""
        self._relayout()

    # ---------------------------------------------------------------- 查询

    def panel_box(self, name: str) -> tuple[float, float, float, float] | None:
        """面板实测的屏幕矩形 ``(x0, x1, y0, y1)``；空文本返回 ``None``。"""
        return self._panels[name].box

    def panel_text(self, name: str) -> str:
        """面板当前显示的文本。

        和 :meth:`panel_box` 一样是给**外部**用的只读查询：离线脚本要断言"这块
        面板确实在报速度"，而"文本是什么"本来就该有个正经的读法，不该让调用方
        去摸 ``_panels[name].text``。
        """
        return self._panels[name].text

    # ---------------------------------------------------------------- 排版

    def _hide(self, holder: NodePath, node: NodePath) -> None:
        node.node().clearCard()
        holder.hide()

    def _relayout(self) -> None:
        self._aspect = self.aspect
        aspect = self._aspect
        left, right = -aspect + _MARGIN, aspect - _MARGIN
        top, bottom = 1.0 - _MARGIN, -1.0 + _MARGIN
        screen_w = 2.0 * aspect

        columns = {
            "left": (left, left + _COLUMN_WIDTH * screen_w),
            "right": (right - _COLUMN_WIDTH * screen_w, right),
            "center": (-_CENTER_WIDTH * screen_w * 0.5,
                       _CENTER_WIDTH * screen_w * 0.5),
        }
        band = _BAND_HEIGHT * (top - bottom)
        #: 同锚点依次往下/往上排的游标
        cursor = {anchor: (top if side == "top" else bottom)
                  for anchor, (_col, side) in _ANCHORS.items()}

        for panel in self._panels.values():
            column, side = _ANCHORS[panel.anchor]
            x0_limit, x1_limit = columns[column]
            unit_w, unit_h, off_x, off_y = self._measure(panel)

            if unit_w <= 0.0 or unit_h <= 0.0:
                # 空文本（或只有空白字符）：整块收起来，不留空黑块挡场景。
                self._hide(panel.holder, panel.node)
                panel.box = None
                continue

            scale = min(panel.scale,
                        (x1_limit - x0_limit) / unit_w,
                        band / unit_h)
            scale = max(scale, panel.scale * _MIN_SCALE_RATIO)
            panel.node.setScale(scale)

            width, height = unit_w * scale, unit_h * scale
            if column == "center":
                box_x0 = -width * 0.5
            elif column == "left":
                box_x0 = x0_limit
            else:
                box_x0 = x1_limit - width

            if side == "top":
                box_y1 = cursor[panel.anchor]
                box_y0 = box_y1 - height
                cursor[panel.anchor] = box_y0 - _STACK_GAP
            else:
                box_y0 = cursor[panel.anchor]
                box_y1 = box_y0 + height
                cursor[panel.anchor] = box_y1 + _STACK_GAP

            # 文字节点的原点未必在文本框角上（TextNode 的原点在**首行基线**），
            # 所以按"文本框左下角要落在哪"反推挂点的位置。
            panel.holder.setPos(box_x0 - off_x * scale,
                                box_y0 - off_y * scale, 0.0)
            panel.box = (box_x0, box_x0 + width, box_y0, box_y0 + height)

    def _show_card(self, panel: _Panel) -> None:
        node = panel.node.node()
        r, g, b, a = _PANEL_COLOR
        node.setCardColor(r, g, b, a)
        node.setCardAsMargin(_PAD_X, _PAD_X, _PAD_Y, _PAD_Y)
        panel.holder.show()

    def _measure(self, panel: _Panel) -> tuple[float, float, float, float]:
        """量出文本框：宽、高、以及左下角相对**挂点**的偏移（**scale = 1 的单位**）。

        用 ``getTightBounds`` 而不是 :func:`render.text.label_size`：前者量的是
        **真实几何**（含 TextNode 自带底衬），因此不受"TextNode 原点在首行基线"
        这一约定影响，多行/单行/有无底衬都一样准。

        文字节点的缩放写在它自己身上，而包围盒是相对挂点算的、会带上这个缩放，
        所以这里先把缩放按 1.0 量，再由调用方乘以最终字号 —— 顺带让
        "字太大就缩小"这一步不需要重新测量。
        """
        node = panel.node.node()
        node.setText(panel.text)
        if not panel.text:
            return (0.0, 0.0, 0.0, 0.0)
        self._show_card(panel)
        panel.node.setScale(1.0)
        bounds = panel.node.getTightBounds()
        if not bounds:
            return (0.0, 0.0, 0.0, 0.0)
        p0, p1 = bounds
        width, height = p1[0] - p0[0], p1[1] - p0[1]
        if not (width > 0.0 and height > 0.0):
            return (0.0, 0.0, 0.0, 0.0)
        return (width, height, p0[0], p0[1])


# --------------------------------------------------------------------------- #
# 默认布局
# --------------------------------------------------------------------------- #

def build_default_hud(base) -> Hud:
    """编辑器用的五块面板。

    * ``status`` 左上、``loop`` 右上、``train`` 右下：分居左右两列，其中右列的
      两块一贴顶一贴底（``_BAND_HEIGHT`` 0.44 + 0.44 < 1，中间必然留着空隙），
      所以既不会互相压住，也不会压住中间的场景；
    * ``help`` 左下、``toast`` 底部居中。

    空文本的面板会自动整块隐藏 —— 所以"还没召唤列车"时右下方是干净的，
    不会留下一块挡场景的黑块。
    """
    hud = Hud(base, font=cjk_font(base))
    hud.add_panel("status", "tl", scale=0.040)
    hud.add_panel("loop", "tr", scale=0.040, color=(0.78, 1.0, 0.86, 1.0))
    hud.add_panel("train", "br", scale=0.040, color=(1.0, 0.88, 0.68, 1.0))
    hud.add_panel("help", "bl", scale=0.034, color=(0.86, 0.90, 0.96, 1.0))
    hud.add_panel("toast", "bc", scale=0.040, color=(1.0, 0.90, 0.62, 1.0))
    return hud
