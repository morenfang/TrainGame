"""屏幕叠加文字面板（HUD）—— 左侧可伸缩信息栏 + 右下常驻运行信息。

版面
------------------------------------------------
状态 / 闭环 / 帮助三块收进**左侧**可伸缩栏，避免铺满角落挡住场景。
车辆**运行信息**单独放在**右下角**，不随左侧栏收起，有车就常驻。

左侧留一枚**固定按钮**（始终可见）：

* 收起时点一下 → 展开
* 展开时点「固定」→ 钉住，不再自动收
* 未钉住时鼠标离开栏一段时间 → 自动收起
* 展开时再点按钮本体 → 收起

底部 ``toast`` 仍居中，只作短暂提示，不占常驻版面。
加载列车时用居中 ``progress`` 条，避免界面假死。

坐标仍是 aspect2d（yup：右 = x，上 = y，z = 进深）。
底衬用 ``TextNode`` 自带卡片，避免手算缩放踩坑。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from panda3d.core import (
    CardMaker,
    NodePath,
    TextNode,
    TransparencyAttrib,
    Vec4,
)

from render.text import cjk_font, make_label

#: 面板距屏幕边缘的留白（aspect2d：高度 = 2.0，宽度 = 2 × 宽高比）。
_MARGIN = 0.018
_PAD_X = 0.38
_PAD_Y = 0.18
_PANEL_COLOR = (0.07, 0.09, 0.12, 0.78)
_TAB_COLOR = (0.12, 0.16, 0.22, 0.92)
_STACK_GAP = 0.010
_MIN_SCALE_RATIO = 0.32
#: 左侧信息栏占屏宽比例（收起时不占，展开时这一列）。
_DOCK_WIDTH = 0.30
#: 未钉住时，鼠标离开后多久自动收起（秒）。
_AUTO_COLLAPSE_SEC = 2.2
#: 左侧固定按钮的宽、高（aspect2d）。
_TAB_W = 0.072
_TAB_H = 0.118

_ANCHORS: dict[str, tuple[str, str]] = {
    "tl": ("left", "top"),
    "tr": ("right", "top"),
    "bl": ("left", "bottom"),
    "br": ("right", "bottom"),
    "bc": ("center", "bottom"),
}
_COLUMN_WIDTH = 0.30
_CENTER_WIDTH = 0.34
_BAND_HEIGHT = 0.92
#: 右下常驻面板从底边再抬一截，给居中 toast 留空，避免窄屏上压住提示条。
_BR_TOAST_CLEARANCE = 0.055


@dataclass
class _Panel:
    holder: NodePath
    node: NodePath
    anchor: str
    scale: float
    color: tuple[float, float, float, float]
    text: str = ""
    box: tuple[float, float, float, float] | None = field(default=None)
    #: 收进左侧栏的内容面板；折叠时整块隐藏。
    docked: bool = False


class Hud:
    """左侧可伸缩信息栏 + 右下常驻运行信息 + 底部提示 + 加载进度。"""

    def __init__(self, base, font=None, *, aspect_override: float | None = None):
        self.base = base
        self.font = font
        self.root = base.aspect2d.attachNewNode("hud")
        self._panels: dict[str, _Panel] = {}
        self.visible = True
        self.aspect_override = aspect_override
        self._aspect = self.aspect

        self.pinned = False
        self.expanded = False
        self._idle = 0.0
        self._tab_box: tuple[float, float, float, float] | None = None
        self._pin_box: tuple[float, float, float, float] | None = None
        self._lights_box: tuple[float, float, float, float] | None = None
        self._dock_box: tuple[float, float, float, float] | None = None
        #: 夜景总开关（默认开）；点左侧「夜景」按钮切换。
        self.street_lights_on = True
        #: 切换夜景时回调 ``(enabled: bool) -> None``，由编辑器显隐光晕 / 窗灯。
        self.on_toggle_street_lights = None

        self._tab_root = self.root.attachNewNode("hud_tab")
        self._tab_bg = self._make_card("tab_bg", _TAB_COLOR)
        self._tab_bg.reparentTo(self._tab_root)
        self._tab_label = make_label(
            "信息", font=self.font, color=(0.93, 0.95, 0.98, 1.0),
            align=TextNode.ACenter,
        )
        self._tab_label.setScale(0.034)
        self._tab_label.setDepthTest(False)
        self._tab_label.setDepthWrite(False)
        self._tab_label.reparentTo(self._tab_root)

        self._pin_root = self.root.attachNewNode("hud_pin")
        self._pin_bg = self._make_card("pin_bg", (0.16, 0.22, 0.30, 0.92))
        self._pin_bg.reparentTo(self._pin_root)
        self._pin_label = make_label(
            "固定", font=self.font, color=(0.95, 0.90, 0.70, 1.0),
            align=TextNode.ACenter,
        )
        self._pin_label.setScale(0.030)
        self._pin_label.setDepthTest(False)
        self._pin_label.setDepthWrite(False)
        self._pin_label.reparentTo(self._pin_root)
        self._pin_root.hide()

        self._lights_root = self.root.attachNewNode("hud_lights")
        self._lights_bg = self._make_card("lights_bg", (0.18, 0.20, 0.14, 0.92))
        self._lights_bg.reparentTo(self._lights_root)
        self._lights_label = make_label(
            "夜景", font=self.font, color=(1.0, 0.92, 0.55, 1.0),
            align=TextNode.ACenter,
        )
        self._lights_label.setScale(0.030)
        self._lights_label.setDepthTest(False)
        self._lights_label.setDepthWrite(False)
        self._lights_label.reparentTo(self._lights_root)

        self._progress_root = self.root.attachNewNode("hud_progress")
        self._progress_root.hide()
        self._progress_card = self._make_card(
            "progress_card", (0.05, 0.06, 0.08, 0.88))
        self._progress_card.reparentTo(self._progress_root)
        self._progress_fill = self._make_card(
            "progress_fill", (0.35, 0.72, 0.95, 0.95))
        self._progress_fill.reparentTo(self._progress_root)
        self._progress_label = make_label(
            "", font=self.font, color=(0.95, 0.97, 1.0, 1.0),
            align=TextNode.ACenter,
        )
        self._progress_label.setScale(0.045)
        self._progress_label.reparentTo(self._progress_root)
        self._progress_text = ""
        self._progress_frac = 0.0

        self._place_chrome()

    # ---------------------------------------------------------------- 基础

    @staticmethod
    def _make_card(name: str, rgba) -> NodePath:
        maker = CardMaker(name)
        maker.setFrame(-0.5, 0.5, -0.5, 0.5)
        node = NodePath(maker.generate())
        node.setColor(Vec4(*rgba))
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setDepthTest(False)
        node.setDepthWrite(False)
        return node

    @property
    def aspect(self) -> float:
        if self.aspect_override is not None:
            return self.aspect_override
        return self.base.getAspectRatio()

    def add_panel(self, name: str, anchor: str = "tl", *, scale: float = 0.038,
                  color=(1.0, 1.0, 1.0, 1.0), docked: bool = False) -> None:
        if anchor not in _ANCHORS:
            raise ValueError(f"未知锚点 {anchor!r}，应为 {tuple(_ANCHORS)} 之一")
        holder = self.root.attachNewNode(f"panel_{name}")
        node = make_label("", font=self.font, color=color)
        node.reparentTo(holder)
        node.setScale(scale)
        node.setDepthTest(False)
        node.setDepthWrite(False)
        node.setTransparency(TransparencyAttrib.MAlpha)
        self._panels[name] = _Panel(
            holder=holder, node=node, anchor=anchor,
            scale=scale, color=tuple(color), docked=docked,
        )
        self._hide(holder, node)

    def remove_panel(self, name: str) -> None:
        panel = self._panels.pop(name, None)
        if panel is not None:
            panel.holder.removeNode()

    def set_visible(self, flag: bool) -> None:
        self.visible = flag
        if flag:
            self.root.show()
        else:
            self.root.hide()

    def set_text(self, name: str, text: str) -> None:
        panel = self._panels[name]
        resized = self.aspect != self._aspect
        if panel.text == text and not resized:
            return
        panel.text = text
        self._relayout()

    def relayout(self) -> None:
        self._relayout()

    def panel_box(self, name: str) -> tuple[float, float, float, float] | None:
        return self._panels[name].box

    def panel_text(self, name: str) -> str:
        return self._panels[name].text

    # ---------------------------------------------------------------- 伸缩 / 钉住

    def expand(self) -> None:
        if self.expanded:
            return
        self.expanded = True
        self._idle = 0.0
        self._relayout()

    def collapse(self) -> None:
        if not self.expanded and not self.pinned:
            self._place_chrome()
            return
        self.expanded = False
        self.pinned = False
        self._idle = 0.0
        self._relayout()

    def set_pinned(self, flag: bool) -> None:
        self.pinned = bool(flag)
        if self.pinned:
            self.expanded = True
            self._idle = 0.0
        self._relayout()

    def toggle_pin(self) -> None:
        self.set_pinned(not self.pinned)

    def toggle_expanded(self) -> None:
        if self.expanded:
            self.collapse()
        else:
            self.expand()

    def tick(self, dt: float) -> None:
        """由编辑器每帧调用：未钉住时离开栏外自动收起。"""
        if not self.expanded or self.pinned:
            self._idle = 0.0
            return
        if self.mouse_over_dock():
            self._idle = 0.0
            return
        self._idle += max(dt, 0.0)
        if self._idle >= _AUTO_COLLAPSE_SEC:
            self.collapse()

    def mouse_aspect(self) -> tuple[float, float] | None:
        watcher = getattr(self.base, "mouseWatcherNode", None)
        if watcher is None or not watcher.hasMouse():
            return None
        return (watcher.getMouseX() * self.aspect, watcher.getMouseY())

    def mouse_over_dock(self) -> bool:
        mouse = self.mouse_aspect()
        if mouse is None:
            return False
        mx, my = mouse
        for box in (self._tab_box, self._pin_box, self._lights_box,
                    self._dock_box):
            if box is not None and box[0] <= mx <= box[1] and box[2] <= my <= box[3]:
                return True
        return False

    def toggle_street_lights(self) -> bool:
        """翻转路灯总开关，触发回调；返回新状态。"""
        self.street_lights_on = not self.street_lights_on
        self._place_chrome()
        callback = self.on_toggle_street_lights
        if callback is not None:
            callback(self.street_lights_on)
        return self.street_lights_on

    def handle_click(self) -> bool:
        """若点中左侧按钮 / 固定 / 路灯则处理并返回 True（吞掉点击）。"""
        mouse = self.mouse_aspect()
        if mouse is None:
            return False
        mx, my = mouse
        if self._lights_box is not None:
            x0, x1, y0, y1 = self._lights_box
            if x0 <= mx <= x1 and y0 <= my <= y1:
                self.toggle_street_lights()
                return True
        if self._pin_box is not None and self.expanded:
            x0, x1, y0, y1 = self._pin_box
            if x0 <= mx <= x1 and y0 <= my <= y1:
                self.toggle_pin()
                return True
        if self._tab_box is not None:
            x0, x1, y0, y1 = self._tab_box
            if x0 <= mx <= x1 and y0 <= my <= y1:
                self.toggle_expanded()
                return True
        return False

    # ---------------------------------------------------------------- 进度条

    def set_progress(self, message: str, fraction: float) -> None:
        """居中进度条；``fraction`` 0..1。空消息则隐藏。"""
        if not message:
            self._progress_text = ""
            self._progress_root.hide()
            return
        self._progress_text = message
        self._progress_frac = max(0.0, min(1.0, fraction))
        self._progress_label.node().setText(message)
        self._layout_progress()
        self._progress_root.show()

    def clear_progress(self) -> None:
        self.set_progress("", 0.0)

    def _layout_progress(self) -> None:
        aspect = self.aspect
        width = min(1.15, aspect * 0.55)
        height = 0.11
        cy = 0.05
        self._progress_card.setPos(0.0, cy, 0.0)
        self._progress_card.setScale(width, height, 1.0)
        fill_w = max(0.02, width * self._progress_frac)
        self._progress_fill.setPos((-width + fill_w) * 0.5, cy - 0.018, 0.0)
        self._progress_fill.setScale(fill_w, 0.028, 1.0)
        self._progress_label.setPos(0.0, cy + 0.018, 0.0)

    # ---------------------------------------------------------------- 排版

    def _hide(self, holder: NodePath, node: NodePath) -> None:
        node.node().clearCard()
        holder.hide()

    def _place_chrome(self) -> None:
        aspect = self.aspect
        left = -aspect + _MARGIN
        tab_x = left + _TAB_W * 0.5
        tab_y = 0.62
        self._tab_bg.setPos(tab_x, tab_y, 0.0)
        self._tab_bg.setScale(_TAB_W, _TAB_H, 1.0)
        self._tab_label.node().setText("信息" if not self.expanded else "收起")
        self._tab_label.setPos(tab_x, tab_y, 0.0)
        self._tab_box = (left, left + _TAB_W, tab_y - _TAB_H * 0.5,
                         tab_y + _TAB_H * 0.5)

        if self.expanded:
            pin_y = tab_y - _TAB_H * 0.5 - 0.055
            pin_h = 0.070
            self._pin_bg.setPos(tab_x, pin_y, 0.0)
            self._pin_bg.setScale(_TAB_W, pin_h, 1.0)
            self._pin_label.node().setText("已固定" if self.pinned else "固定")
            color = ((0.95, 0.82, 0.40, 1.0) if self.pinned
                     else (0.90, 0.92, 0.95, 1.0))
            self._pin_label.node().setTextColor(Vec4(*color))
            self._pin_label.setPos(tab_x, pin_y, 0.0)
            self._pin_box = (left, left + _TAB_W, pin_y - pin_h * 0.5,
                             pin_y + pin_h * 0.5)
            self._pin_root.show()
            lights_y = pin_y - pin_h * 0.5 - 0.055
        else:
            self._pin_box = None
            self._pin_root.hide()
            lights_y = tab_y - _TAB_H * 0.5 - 0.055

        lights_h = 0.070
        self._lights_bg.setPos(tab_x, lights_y, 0.0)
        self._lights_bg.setScale(_TAB_W, lights_h, 1.0)
        if self.street_lights_on:
            self._lights_label.node().setText("夜景开")
            self._lights_label.node().setTextColor(Vec4(1.0, 0.92, 0.55, 1.0))
            self._lights_bg.setColor(Vec4(0.22, 0.24, 0.12, 0.92))
        else:
            self._lights_label.node().setText("夜景关")
            self._lights_label.node().setTextColor(Vec4(0.70, 0.72, 0.76, 1.0))
            self._lights_bg.setColor(Vec4(0.12, 0.13, 0.16, 0.92))
        self._lights_label.setPos(tab_x, lights_y, 0.0)
        self._lights_box = (left, left + _TAB_W, lights_y - lights_h * 0.5,
                            lights_y + lights_h * 0.5)

    def _relayout(self) -> None:
        self._aspect = self.aspect
        aspect = self._aspect
        left, right = -aspect + _MARGIN, aspect - _MARGIN
        top, bottom = 1.0 - _MARGIN, -1.0 + _MARGIN
        screen_w = 2.0 * aspect
        self._place_chrome()

        dock_left = left + _TAB_W + 0.012
        left_free = (left, left + _COLUMN_WIDTH * screen_w)
        left_dock = (dock_left, dock_left + _DOCK_WIDTH * screen_w)
        columns = {
            "right": (right - _COLUMN_WIDTH * screen_w, right),
            "center": (-_CENTER_WIDTH * screen_w * 0.5,
                       _CENTER_WIDTH * screen_w * 0.5),
        }
        band = _BAND_HEIGHT * (top - bottom)
        cursor = {anchor: (top if side == "top" else bottom)
                  for anchor, (_col, side) in _ANCHORS.items()}

        dock_x0 = dock_x1 = dock_left
        dock_y0, dock_y1 = top, bottom

        def place_one(panel: _Panel) -> None:
            nonlocal dock_x0, dock_x1, dock_y0, dock_y1
            column, side = _ANCHORS[panel.anchor]
            if column == "left":
                x0_limit, x1_limit = left_dock if panel.docked else left_free
            else:
                x0_limit, x1_limit = columns[column]
            hide = (panel.docked and not self.expanded) or not panel.text
            if hide:
                self._hide(panel.holder, panel.node)
                panel.box = None
                return

            unit_w, unit_h, off_x, off_y = self._measure(panel)
            if unit_w <= 0.0 or unit_h <= 0.0:
                self._hide(panel.holder, panel.node)
                panel.box = None
                return

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

            panel.holder.setPos(box_x0 - off_x * scale,
                                box_y0 - off_y * scale, 0.0)
            panel.box = (box_x0, box_x0 + width, box_y0, box_y0 + height)
            if panel.docked:
                dock_x0 = min(dock_x0, box_x0)
                dock_x1 = max(dock_x1, box_x0 + width)
                dock_y0 = min(dock_y0, box_y0)
                dock_y1 = max(dock_y1, box_y0 + height)

        # 先排其它面板（含 toast），再排右下常驻块：抬到 toast 顶上，避免窄屏重叠
        deferred = []
        for panel in self._panels.values():
            if panel.anchor == "br":
                deferred.append(panel)
            else:
                place_one(panel)
        toast = self._panels.get("toast")
        br_floor = bottom + _BR_TOAST_CLEARANCE
        if toast is not None and toast.box is not None:
            br_floor = max(br_floor, toast.box[3] + _STACK_GAP)
        cursor["br"] = br_floor
        for panel in deferred:
            place_one(panel)

        if self.expanded:
            self._dock_box = (dock_x0, dock_x1, dock_y0, dock_y1)
        else:
            self._dock_box = None

        if self._progress_text:
            self._layout_progress()

    def _show_card(self, panel: _Panel) -> None:
        node = panel.node.node()
        r, g, b, a = _PANEL_COLOR
        node.setCardColor(r, g, b, a)
        node.setCardAsMargin(_PAD_X, _PAD_X, _PAD_Y, _PAD_Y)
        panel.holder.show()

    def _measure(self, panel: _Panel) -> tuple[float, float, float, float]:
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
    """左侧：状态 / 闭环 / 帮助；右下常驻：运行信息；底部 toast；加载进度条。"""
    hud = Hud(base, font=cjk_font(base))
    # 字号略收、颜色分区；编辑信息靠左，运行信息钉在右下
    hud.add_panel("status", "tl", scale=0.036, docked=True,
                  color=(0.96, 0.97, 0.99, 1.0))
    hud.add_panel("loop", "tl", scale=0.034, docked=True,
                  color=(0.72, 0.95, 0.82, 1.0))
    hud.add_panel("help", "tl", scale=0.028, docked=True,
                  color=(0.78, 0.84, 0.92, 1.0))
    hud.add_panel("train", "br", scale=0.036, docked=False,
                  color=(1.0, 0.90, 0.72, 1.0))
    hud.add_panel("toast", "bc", scale=0.038,
                  color=(1.0, 0.92, 0.68, 1.0))
    return hud
