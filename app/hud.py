"""v2 HUD：左栏五模式 + 可固定分类列 + 底栏 Tomix 控制台。

视觉：中性灰度；字号可读；分类/件用文字区分；档位条收窄。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from panda3d.core import (
    CardMaker,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LineSegs,
    NodePath,
    TextNode,
    TransparencyAttrib,
    Vec4,
)

from render.text import cjk_font, make_label

_MARGIN = 0.012
_CONSOLE_H = 0.34

# 可读字号（相对先前整体加大）
_FONT = 0.032
_FONT_SM = 0.026
_FONT_MD = 0.030
_FONT_LG = 0.070
_FONT_ICON = 0.042

# 灰度配色（参考 ColorsWall SaaS dashboard：#141414 / #282828 / #505050 / #646464 / #f0f0f0）
def _g(hex6: str, a: float = 1.0) -> tuple[float, float, float, float]:
    h = hex6.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, a)


_BG_RAIL = _g("141414", 0.96)
_BG_PANEL = _g("1e1e1e", 0.94)
_BG_BTN = _g("282828", 0.96)
_BG_BTN_ON = _g("505050", 0.98)       # 选中：中灰提亮，不用彩色
_BG_CHIP = _g("282828", 0.95)
_BG_CONSOLE = _g("141414", 0.94)
_BG_DIM = _g("282828", 0.50)
_BG_DANGER = _g("5a5a5a", 0.95)       # 紧急也走灰阶，靠位置/文案区分
_BG_DANGER_DIM = _g("3c3c3c", 0.75)
_FG = _g("f0f0f0")
_FG_ON = _g("ffffff")
_FG_MUTED = _g("646464")
_ACCENT = _g("c8c8c8")                # 点缀：浅灰，非彩色
_ACCENT_DIM = _g("505050")
_TRACK = _g("3c3c3c")
_EDGE = _g("3c3c3c", 0.90)

# 时速表热度（雅致、低饱和）：低绿 → 中金黄 → 高砖红（刻度用，指针不用）
_SPEED_GREEN = (0.42, 0.68, 0.50)
_SPEED_YELLOW = (0.86, 0.74, 0.38)
_SPEED_RED = (0.78, 0.38, 0.36)
_SPEED_MAX_KMH = 320.0
# 标准车速表扫角：约 240°（210° → -30°），圆心几何，等半径
_SPEED_A0 = math.radians(210)
_SPEED_A1 = math.radians(-30)

# 模式：矢量图标 + 文字标签
HUD_MODES = (
    ("track", "轨道"),
    ("scenery", "布景"),
    ("train", "列车"),
    ("view", "视角"),
    ("file", "存档"),
)

TRACK_CATEGORY_ORDER = (
    ("straight", "直轨"),
    ("curve", "曲线"),
    ("grade", "坡道"),
    ("viaduct", "高架"),
    ("crossing", "交叉"),
    ("switch", "道岔"),
    ("terminal", "终端"),
)

TRACK_CATEGORY_FULL = TRACK_CATEGORY_ORDER

NOTCH_VALUES = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
NOTCH_LABELS = ("B4", "B3", "B2", "B1", "N", "P1", "P2", "P3", "P4")


def handle_to_notch(handle: float) -> int:
    return min(range(len(NOTCH_VALUES)),
               key=lambda i: abs(NOTCH_VALUES[i] - handle))


def notch_to_handle(notch: int) -> float:
    return NOTCH_VALUES[max(0, min(len(NOTCH_VALUES) - 1, int(notch)))]


def _lerp3(a, b, t: float) -> tuple[float, float, float]:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def speed_heat(frac: float) -> tuple[float, float, float]:
    """0→1 映射到绿→黄→红（分段线性，中间偏柔）。"""
    t = max(0.0, min(1.0, float(frac)))
    if t <= 0.5:
        return _lerp3(_SPEED_GREEN, _SPEED_YELLOW, t / 0.5)
    return _lerp3(_SPEED_YELLOW, _SPEED_RED, (t - 0.5) / 0.5)


def _speed_angle(frac: float) -> float:
    t = max(0.0, min(1.0, float(frac)))
    return _SPEED_A0 + (_SPEED_A1 - _SPEED_A0) * t


def _circle_xy(ang: float, radius: float = 1.0) -> tuple[float, float]:
    """单位圆上的点（aspect2d 上等半径 → 屏幕上为正圆）。"""
    return (radius * math.cos(ang), radius * math.sin(ang))


def _finish_lines(segs: LineSegs) -> NodePath:
    np = NodePath(segs.create())
    np.setDepthTest(False)
    np.setDepthWrite(False)
    np.setTransparency(TransparencyAttrib.MAlpha)
    return np


def _make_speed_scale(*, arc_segments: int = 64, major_ticks: int = 8,
                      minor_per_major: int = 4) -> NodePath:
    """车速表刻度：细渐变弧 + 径向刻度（低绿→中黄→顶红）；非粗进度条。"""
    root = NodePath("speed_scale")

    # 细弧底轨（全量程渐变，等半径圆弧焊线）
    arc = LineSegs("speed_scale_arc")
    arc.setThickness(2.2)
    r_arc = 0.94
    for i in range(arc_segments + 1):
        t = i / arc_segments
        ang = _speed_angle(t)
        cr, cg, cb = speed_heat(t)
        arc.setColor(cr, cg, cb, 0.88)
        x, y = _circle_xy(ang, r_arc)
        if i == 0:
            arc.moveTo(x, y, 0.0)
        else:
            arc.drawTo(x, y, 0.0)
    _finish_lines(arc).reparentTo(root)

    # 主/副刻度：按角度上色，径向线段落在同一圆周上
    majors = LineSegs("speed_ticks_major")
    majors.setThickness(2.0)
    minors = LineSegs("speed_ticks_minor")
    minors.setThickness(1.2)
    n = major_ticks * minor_per_major
    for i in range(n + 1):
        t = i / n
        ang = _speed_angle(t)
        cr, cg, cb = speed_heat(t)
        c, s = math.cos(ang), math.sin(ang)
        is_major = (i % minor_per_major == 0)
        r_out, r_in = (1.0, 0.78) if is_major else (1.0, 0.88)
        segs = majors if is_major else minors
        segs.setColor(cr, cg, cb, 0.95)
        segs.moveTo(c * r_in, s * r_in, 0.0)
        segs.drawTo(c * r_out, s * r_out, 0.0)
    _finish_lines(majors).reparentTo(root)
    _finish_lines(minors).reparentTo(root)
    return root


@dataclass
class _Hit:
    box: tuple[float, float, float, float]
    action: str
    arg: object = None


@dataclass
class HudState:
    hud_mode: str = "track"
    placing: bool = True
    scenery: bool = False
    category: str = "straight"
    category_label: str = "直轨"
    piece_name: str = ""
    piece_index: int = 0
    piece_count: int = 0
    catalog_items: tuple[tuple[str, str], ...] = ()
    scenery_category: str = ""
    scenery_rail: tuple[tuple[str, str], ...] = ()
    scenery_items: tuple[tuple[str, str], ...] = ()
    scenery_index: int = 0
    loop_line: str = ""
    train_name: str = ""
    train_line: str = ""
    handle_label: str = "惰行"
    handle: float = 0.0
    train_online: bool = False
    speed_kmh: float = 0.0
    distance_m: float = 0.0
    direction: str = "正向"
    train_list: tuple[tuple[str, str], ...] = ()
    train_index: int = 0
    toast: str = ""
    help_visible: bool = False
    street_lights_on: bool = True
    track_categories: tuple[str, ...] = ()
    mode: str = "place"


def _make_disc(name: str, segments: int = 48) -> NodePath:
    fmt = GeomVertexFormat.getV3()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    writer = GeomVertexWriter(vdata, "vertex")
    writer.addData3(0.0, 0.0, 0.0)
    for i in range(segments + 1):
        ang = 2.0 * math.pi * i / segments
        writer.addData3(math.cos(ang), math.sin(ang), 0.0)
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(segments):
        tris.addVertices(0, i + 1, i + 2)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    np = NodePath(node)
    np.setTransparency(TransparencyAttrib.MAlpha)
    np.setDepthTest(False)
    np.setDepthWrite(False)
    return np


def _make_arc(name: str, radius: float, a0: float, a1: float,
              thickness: float = 1.6, segments: int = 64) -> NodePath:
    """等半径圆弧（aspect2d 上为正圆）。thickness 为 LineSegs 像素线宽。"""
    segs = LineSegs(name)
    segs.setThickness(thickness)
    for i in range(segments + 1):
        t = i / segments
        ang = a0 + (a1 - a0) * t
        x, y = _circle_xy(ang, radius)
        if i == 0:
            segs.moveTo(x, y, 0.0)
        else:
            segs.drawTo(x, y, 0.0)
    return _finish_lines(segs)


def _make_circle(name: str, radius: float = 1.0, *,
                 thickness: float = 1.6, segments: int = 72) -> NodePath:
    """标准正圆描边（表盘外圈）。"""
    return _make_arc(name, radius, 0.0, 2.0 * math.pi, thickness, segments)


def _line_icon(name: str, thickness: float = 2.2) -> LineSegs:
    segs = LineSegs(name)
    segs.setThickness(thickness)
    segs.setColor(Vec4(1, 1, 1, 1))
    return segs


def _finish_icon(segs: LineSegs) -> NodePath:
    np = NodePath(segs.create())
    np.setDepthTest(False)
    np.setDepthWrite(False)
    np.setTransparency(TransparencyAttrib.MAlpha)
    return np


def icon_curve_track() -> NodePath:
    """弯轨：两根弧轨 + 几根轨枕。"""
    segs = _line_icon("ico_track", 2.4)
    a0, a1 = math.radians(-15), math.radians(100)
    for r in (0.38, 0.26):
        for i in range(14):
            t = i / 13
            ang = a0 + (a1 - a0) * t
            x, y = r * math.cos(ang) - 0.08, r * math.sin(ang) - 0.12
            if i == 0:
                segs.moveTo(x, y, 0.0)
            else:
                segs.drawTo(x, y, 0.0)
    # 轨枕
    for t in (0.15, 0.4, 0.65, 0.9):
        ang = a0 + (a1 - a0) * t
        c, s = math.cos(ang), math.sin(ang)
        x0, y0 = 0.24 * c - 0.08, 0.24 * s - 0.12
        x1, y1 = 0.40 * c - 0.08, 0.40 * s - 0.12
        segs.moveTo(x0, y0, 0.0)
        segs.drawTo(x1, y1, 0.0)
    return _finish_icon(segs)


def icon_tree() -> NodePath:
    """一棵树：树干 + 三角冠。"""
    segs = _line_icon("ico_tree", 2.3)
    # 冠（三角形）
    segs.moveTo(0.0, 0.42, 0.0)
    segs.drawTo(-0.36, -0.02, 0.0)
    segs.drawTo(0.36, -0.02, 0.0)
    segs.drawTo(0.0, 0.42, 0.0)
    # 第二层冠
    segs.moveTo(0.0, 0.18, 0.0)
    segs.drawTo(-0.30, -0.22, 0.0)
    segs.drawTo(0.30, -0.22, 0.0)
    segs.drawTo(0.0, 0.18, 0.0)
    # 树干
    segs.moveTo(-0.07, -0.22, 0.0)
    segs.drawTo(-0.07, -0.42, 0.0)
    segs.drawTo(0.07, -0.42, 0.0)
    segs.drawTo(0.07, -0.22, 0.0)
    return _finish_icon(segs)


def icon_hsr_nose() -> NodePath:
    """高铁车头（正面简笔）：车体、风挡、灯、鼻锥。"""
    segs = _line_icon("ico_train", 2.2)
    # 车体外轮廓
    segs.moveTo(-0.34, 0.28, 0.0)
    segs.drawTo(-0.30, 0.40, 0.0)
    segs.drawTo(0.30, 0.40, 0.0)
    segs.drawTo(0.34, 0.28, 0.0)
    segs.drawTo(0.34, -0.05, 0.0)
    segs.drawTo(0.18, -0.32, 0.0)
    segs.drawTo(-0.18, -0.32, 0.0)
    segs.drawTo(-0.34, -0.05, 0.0)
    segs.drawTo(-0.34, 0.28, 0.0)
    # 风挡
    segs.moveTo(-0.22, 0.22, 0.0)
    segs.drawTo(-0.18, 0.32, 0.0)
    segs.drawTo(0.18, 0.32, 0.0)
    segs.drawTo(0.22, 0.22, 0.0)
    segs.drawTo(-0.22, 0.22, 0.0)
    # 灯
    segs.moveTo(-0.22, 0.05, 0.0)
    segs.drawTo(-0.12, 0.05, 0.0)
    segs.drawTo(-0.12, 0.14, 0.0)
    segs.drawTo(-0.22, 0.14, 0.0)
    segs.drawTo(-0.22, 0.05, 0.0)
    segs.moveTo(0.12, 0.05, 0.0)
    segs.drawTo(0.22, 0.05, 0.0)
    segs.drawTo(0.22, 0.14, 0.0)
    segs.drawTo(0.12, 0.14, 0.0)
    segs.drawTo(0.12, 0.05, 0.0)
    # 鼻锥中线
    segs.moveTo(0.0, -0.05, 0.0)
    segs.drawTo(0.0, -0.32, 0.0)
    return _finish_icon(segs)


def icon_camera() -> NodePath:
    """视角：相机机身 + 镜头。"""
    segs = _line_icon("ico_view", 2.2)
    segs.moveTo(-0.36, -0.18, 0.0)
    segs.drawTo(-0.36, 0.18, 0.0)
    segs.drawTo(0.22, 0.18, 0.0)
    segs.drawTo(0.22, -0.18, 0.0)
    segs.drawTo(-0.36, -0.18, 0.0)
    # 取景凸起
    segs.moveTo(-0.18, 0.18, 0.0)
    segs.drawTo(-0.18, 0.30, 0.0)
    segs.drawTo(0.05, 0.30, 0.0)
    segs.drawTo(0.05, 0.18, 0.0)
    # 镜头圆
    for i in range(20):
        a0 = 2 * math.pi * i / 20
        a1 = 2 * math.pi * (i + 1) / 20
        if i == 0:
            segs.moveTo(0.08 + 0.16 * math.cos(a0), 0.16 * math.sin(a0), 0.0)
        segs.drawTo(0.08 + 0.16 * math.cos(a1), 0.16 * math.sin(a1), 0.0)
    return _finish_icon(segs)


def icon_floppy() -> NodePath:
    """经典保存：软盘外框 + 金属片 + 标签区。"""
    segs = _line_icon("ico_file", 2.2)
    # 外框（右上切角）
    segs.moveTo(-0.34, -0.38, 0.0)
    segs.drawTo(-0.34, 0.38, 0.0)
    segs.drawTo(0.18, 0.38, 0.0)
    segs.drawTo(0.34, 0.22, 0.0)
    segs.drawTo(0.34, -0.38, 0.0)
    segs.drawTo(-0.34, -0.38, 0.0)
    # 顶部滑动片
    segs.moveTo(-0.22, 0.38, 0.0)
    segs.drawTo(-0.22, 0.12, 0.0)
    segs.drawTo(0.12, 0.12, 0.0)
    segs.drawTo(0.12, 0.38, 0.0)
    # 标签区
    segs.moveTo(-0.24, -0.05, 0.0)
    segs.drawTo(-0.24, -0.32, 0.0)
    segs.drawTo(0.24, -0.32, 0.0)
    segs.drawTo(0.24, -0.05, 0.0)
    segs.drawTo(-0.24, -0.05, 0.0)
    # 标签横线
    segs.moveTo(-0.16, -0.14, 0.0)
    segs.drawTo(0.16, -0.14, 0.0)
    segs.moveTo(-0.16, -0.22, 0.0)
    segs.drawTo(0.16, -0.22, 0.0)
    return _finish_icon(segs)


_MODE_ICON_BUILDERS = {
    "track": icon_curve_track,
    "scenery": icon_tree,
    "train": icon_hsr_nose,
    "view": icon_camera,
    "file": icon_floppy,
}


class Hud:
    def __init__(self, base, font=None, *, aspect_override: float | None = None):
        self.base = base
        self.font = font
        self.root = base.aspect2d.attachNewNode("hud")
        self.visible = True
        self.aspect_override = aspect_override

        self.state = HudState()
        self.texts: dict[str, str] = {}
        self.expanded = False
        self.pinned = False
        self.street_lights_on = True
        self.on_toggle_street_lights = None
        self.actions: dict[str, object] = {}

        self._hits: list[_Hit] = []
        self._nodes: dict[str, NodePath] = {}
        self._labels: dict[str, NodePath] = {}
        self._sublabels: dict[str, NodePath] = {}
        self._edges: dict[str, NodePath] = {}
        self._mode_icons: dict[str, NodePath] = {}
        self._catalog_scroll = 0
        self._dragging_notch = False
        self._notch_x0 = 0.0
        self._notch_x1 = 1.0

        self._rail = self.root.attachNewNode("rail")
        self._catalog = self.root.attachNewNode("catalog")
        self._console = self.root.attachNewNode("console")
        self._help = self.root.attachNewNode("help")
        self._help.hide()

        self._rail_bg = self._card("rail_bg", _BG_RAIL)
        self._rail_bg.reparentTo(self._rail)
        self._rail_edge = self._card("rail_edge", _EDGE)
        self._rail_edge.reparentTo(self._rail)
        self._catalog_bg = self._card("catalog_bg", _BG_PANEL)
        self._catalog_bg.reparentTo(self._catalog)
        self._console_bg = self._card("console_bg", _BG_CONSOLE)
        self._console_bg.reparentTo(self._console)
        self._console_top_line = self._card("console_line", _ACCENT)
        self._console_top_line.reparentTo(self._console)

        self._dial_face = _make_disc("dial_face")
        self._dial_face.reparentTo(self._console)
        # 标准正圆外圈（非花瓣/粗弧瓣）
        self._dial_ring = _make_circle("dial_ring", 1.0, thickness=1.8, segments=72)
        self._dial_ring.reparentTo(self._console)
        self._dial_ring.setColor(Vec4(*_ACCENT_DIM))
        self._speed_arc_root = _make_speed_scale()
        self._speed_arc_root.reparentTo(self._console)
        self._needle_root = self._console.attachNewNode("needle_root")
        self._needle_hub = _make_disc("needle_hub", 16)
        self._needle_hub.reparentTo(self._console)
        self._needle_hub.setColor(Vec4(*_ACCENT))

        self._progress_root = self.root.attachNewNode("hud_progress")
        self._progress_root.hide()
        self._progress_card = self._card("progress_card", _g("141414", 0.92))
        self._progress_card.reparentTo(self._progress_root)
        self._progress_fill = self._card("progress_fill", _g("c8c8c8", 0.95))
        self._progress_fill.reparentTo(self._progress_root)
        self._progress_label = make_label(
            "", font=self.font, color=_FG, align=TextNode.ACenter,
        )
        self._progress_label.setScale(0.045)
        self._progress_label.reparentTo(self._progress_root)
        self._progress_text = ""
        self._progress_frac = 0.0

        self._help_card = self._card("help_card", _g("1e1e1e", 0.96))
        self._help_card.reparentTo(self._help)
        self._help_label = make_label(
            "", font=self.font, color=_FG, align=TextNode.ALeft,
        )
        self._help_label.setScale(_FONT_SM)
        self._help_label.reparentTo(self._help)

        self._build_static()
        self._paint()

    @staticmethod
    def _card(name: str, rgba) -> NodePath:
        maker = CardMaker(name)
        maker.setFrame(-0.5, 0.5, -0.5, 0.5)
        node = NodePath(maker.generate())
        node.setColor(Vec4(*rgba))
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setDepthTest(False)
        node.setDepthWrite(False)
        return node

    def _ensure(self, parent: NodePath, key: str, *, dual: bool = False) -> None:
        if key not in self._nodes:
            edge = self._card(f"{key}_edge", _EDGE)
            edge.reparentTo(parent)
            self._edges[key] = edge
            bg = self._card(f"{key}_bg", _BG_BTN)
            bg.reparentTo(parent)
            self._nodes[key] = bg
            lab = make_label("", font=self.font, color=_FG, align=TextNode.ACenter)
            lab.setScale(_FONT_SM)
            lab.setDepthTest(False)
            lab.setDepthWrite(False)
            lab.reparentTo(parent)
            self._labels[key] = lab
        if dual and key not in self._sublabels:
            sub = make_label("", font=self.font, color=_FG_MUTED, align=TextNode.ACenter)
            sub.setScale(_FONT_SM * 0.85)
            sub.setDepthTest(False)
            sub.setDepthWrite(False)
            sub.reparentTo(parent)
            self._sublabels[key] = sub

    def _drop_prefix(self, prefix: str) -> None:
        for store in (self._nodes, self._labels, self._sublabels, self._edges):
            for key in list(store):
                if key.startswith(prefix):
                    store.pop(key).removeNode()

    def _build_static(self) -> None:
        for mode_id, _ in HUD_MODES:
            self._ensure(self._rail, f"mode_{mode_id}")
            builder = _MODE_ICON_BUILDERS[mode_id]
            ico = builder()
            ico.reparentTo(self._rail)
            self._mode_icons[mode_id] = ico
        for key in ("night", "help", "pin"):
            self._ensure(self._rail, key)
        for key in (
            "dial_num", "dial_unit",
            "chip_train", "chip_handle", "chip_dist", "chip_dir",
            "notch_caption", "notch_hint", "notch_track", "notch_knob",
            "btn_emergency", "btn_reverse", "btn_horn",
            "toast", "page_up", "page_down", "cat_header",
        ):
            parent = self._catalog if key.startswith(("page_", "cat_")) else self._console
            self._ensure(parent, key)
        self._labels["dial_num"].setScale(_FONT_LG)
        self._labels["btn_emergency"].node().setText("紧急")
        self._labels["btn_reverse"].node().setText("换向")
        self._labels["btn_horn"].node().setText("鸣笛")

    @property
    def aspect(self) -> float:
        if self.aspect_override is not None:
            return self.aspect_override
        return self.base.getAspectRatio()

    def set_visible(self, flag: bool) -> None:
        self.visible = flag
        (self.root.show if flag else self.root.hide)()

    def set_actions(self, actions: dict) -> None:
        self.actions = dict(actions)

    def add_panel(self, name: str, anchor: str = "tl", **_kwargs) -> None:
        self.texts.setdefault(name, "")

    def remove_panel(self, name: str) -> None:
        self.texts.pop(name, None)

    def set_text(self, name: str, text: str) -> None:
        self.texts[name] = text
        if name == "toast":
            self.state.toast = text
            self._paint()
        elif name == "help" and text:
            self.state.help_visible = True
            self.expanded = True
            self._paint()

    def panel_text(self, name: str) -> str:
        return self.texts.get(name, "")

    def panel_box(self, name: str):
        return None

    def relayout(self) -> None:
        self._paint()

    def apply_state(self, state: HudState) -> None:
        prev = self.state.hud_mode
        self.state = state
        self.street_lights_on = state.street_lights_on
        self.expanded = state.help_visible
        if state.hud_mode != prev:
            self._catalog_scroll = 0
        self.texts["status"] = (
            f"{state.category_label} · {state.piece_name}"
            f"  [{state.piece_index + 1}/{max(state.piece_count, 1)}]"
        )
        self.texts["loop"] = state.loop_line
        self.texts["train"] = (
            f"运行信息  {state.train_name}\n"
            f"{state.speed_kmh:.1f} km/h · {state.direction}\n"
            f"{state.handle_label}"
            if state.train_name else ""
        )
        self.texts["toast"] = state.toast
        self.texts["help"] = HELP_OVERLAY if state.help_visible else ""
        self._paint()

    def expand(self) -> None:
        self.expanded = True
        self.state.help_visible = True
        self._paint()

    def collapse(self) -> None:
        self.expanded = False
        self.state.help_visible = False
        self._paint()

    def toggle_expanded(self) -> None:
        (self.collapse if self.expanded else self.expand)()

    def set_pinned(self, flag: bool) -> None:
        self.pinned = bool(flag)
        self._paint()

    def toggle_pin(self) -> None:
        self.set_pinned(not self.pinned)

    def catalog_open(self) -> bool:
        if self.pinned:
            return True
        return self.state.hud_mode in ("track", "scenery", "train", "file")

    def mouse_aspect(self) -> tuple[float, float] | None:
        watcher = getattr(self.base, "mouseWatcherNode", None)
        if watcher is None or not watcher.hasMouse():
            return None
        return (watcher.getMouseX() * self.aspect, watcher.getMouseY())

    def toggle_street_lights(self) -> bool:
        self.street_lights_on = not self.street_lights_on
        self.state.street_lights_on = self.street_lights_on
        if self.on_toggle_street_lights is not None:
            self.on_toggle_street_lights(self.street_lights_on)
        self._paint()
        return self.street_lights_on

    def tick(self, dt: float) -> None:
        if not self._dragging_notch:
            return
        mouse = self.mouse_aspect()
        if mouse is None:
            return
        notch = self._notch_at_x(mouse[0])
        if notch is not None:
            self._dispatch("handle_notch", notch)

    def handle_click(self) -> bool:
        mouse = self.mouse_aspect()
        if mouse is None:
            return False
        mx, my = mouse
        for hit in self._hits:
            x0, x1, y0, y1 = hit.box
            if x0 <= mx <= x1 and y0 <= my <= y1:
                if hit.action == "handle_notch":
                    self._dragging_notch = True
                self._dispatch(hit.action, hit.arg)
                return True
        return False

    def handle_release(self) -> None:
        self._dragging_notch = False

    def _dispatch(self, action: str, arg) -> None:
        if action == "night":
            self.toggle_street_lights()
            return
        if action == "help":
            self.toggle_expanded()
            return
        if action == "pin":
            self.toggle_pin()
            return
        if action == "page":
            self._catalog_scroll = max(0, self._catalog_scroll + int(arg))
            self._paint()
            return
        fn = self.actions.get(action)
        if fn is None:
            return
        fn() if arg is None else fn(arg)

    def set_progress(self, message: str, fraction: float) -> None:
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
        width = min(1.15, self.aspect * 0.55)
        cy = 0.08
        self._progress_card.setPos(0.0, cy, 0.0)
        self._progress_card.setScale(width, 0.11, 1.0)
        fill_w = max(0.02, width * self._progress_frac)
        self._progress_fill.setPos((-width + fill_w) * 0.5, cy - 0.018, 0.0)
        self._progress_fill.setScale(fill_w, 0.028, 1.0)
        self._progress_label.setPos(0.0, cy + 0.018, 0.0)

    def _place(self, key: str, x0, x1, y0, y1, *,
               action=None, arg=None, on=False, text=None, sub=None,
               dim=False, danger=False, scale=None, fill=None, fg=None,
               edge=True) -> None:
        parent = self._rail
        if key.startswith(("cat_", "item_", "file_", "trainitem_", "page_")):
            parent = self._catalog
        elif key.startswith(("chip_", "dial_", "notch_", "btn_", "toast")):
            parent = self._console
        self._ensure(parent, key, dual=sub is not None)
        bg = self._nodes[key]
        lab = self._labels[key]
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        w, h = max(x1 - x0, 0.008), max(y1 - y0, 0.008)

        if edge and key in self._edges:
            e = self._edges[key]
            e.setColor(Vec4(*(_ACCENT if on else _EDGE)))
            e.setPos(cx, cy, 0.0)
            e.setScale(w + 0.006, h + 0.006, 1.0)
            e.show()
        elif key in self._edges:
            self._edges[key].hide()

        if fill is not None:
            color = fill
        elif dim:
            color = _BG_DIM
        elif danger:
            color = _BG_DANGER if not dim else _BG_DANGER_DIM
        elif on:
            color = _BG_BTN_ON
        else:
            color = _BG_BTN
        bg.setColor(Vec4(*color))
        bg.setPos(cx, cy, 0.0)
        bg.setScale(w, h, 1.0)
        bg.show()

        if text is not None:
            lab.node().setText(text)
        if scale is not None:
            lab.setScale(scale)
        if sub is not None and key in self._sublabels:
            lab.setPos(cx, cy + h * 0.18, 0.0)
            lab.setScale(scale or _FONT_ICON)
            sub_lab = self._sublabels[key]
            sub_lab.node().setText(sub)
            sub_lab.node().setTextColor(Vec4(*(
                _FG_ON if on else (_FG_MUTED if dim else _FG)
            )))
            sub_lab.setScale(_FONT_SM)
            sub_lab.setPos(cx, cy - h * 0.28, 0.0)
            sub_lab.show()
        else:
            if key in self._sublabels:
                self._sublabels[key].hide()
            lab.setPos(cx, cy, 0.0)

        lab.node().setTextColor(Vec4(*(fg or (
            _FG_ON if on else (_FG_MUTED if dim else _FG)
        ))))
        lab.show()
        if action is not None:
            self._hits.append(_Hit((x0, x1, y0, y1), action, arg))

    def _hide(self, key: str) -> None:
        for store in (self._nodes, self._labels, self._sublabels, self._edges):
            if key in store:
                store[key].hide()

    def _short(self, text: str, n: int = 8) -> str:
        return text if len(text) <= n else text[: n - 1] + "…"

    def _paint(self) -> None:
        aspect = self.aspect
        self._hits.clear()
        left = -aspect + _MARGIN
        right = aspect - _MARGIN
        top = 1.0 - _MARGIN
        bottom = -1.0 + _MARGIN
        st = self.state
        console_top = bottom + _CONSOLE_H

        # 左轨略宽，分类列加宽以便写清文字
        rail_w = 0.125
        cat_w = 0.155
        rail_x0, rail_x1 = left, left + rail_w
        show_cat = self.catalog_open()

        self._rail_bg.setPos(0.5 * (rail_x0 + rail_x1),
                             0.5 * (console_top + top), 0.0)
        self._rail_bg.setScale(rail_w, top - console_top, 1.0)
        self._rail_bg.show()
        # 右边一条分隔线
        self._rail_edge.setPos(rail_x1 - 0.003, 0.5 * (console_top + top), 0.0)
        self._rail_edge.setScale(0.006, top - console_top, 1.0)
        self._rail_edge.show()

        mode_h = 0.125
        mode_gap = 0.010
        y = top - 0.016
        for mode_id, label in HUD_MODES:
            y1, y0 = y, y - mode_h
            pad = 0.008
            on = st.hud_mode == mode_id
            # 按钮底 + 下方文字；上方放矢量图标
            self._place(
                f"mode_{mode_id}",
                rail_x0 + pad, rail_x1 - pad, y0, y1,
                action="hud_mode", arg=mode_id,
                on=on, text=label,
                scale=_FONT_SM,
                fill=_BG_BTN_ON if on else _BG_BTN,
                fg=_FG_ON if on else _FG,
            )
            # 文字靠下
            key = f"mode_{mode_id}"
            cx = 0.5 * (rail_x0 + rail_x1)
            self._labels[key].setPos(cx, y0 + (y1 - y0) * 0.20, 0.0)
            # 图标居上
            ico = self._mode_icons[mode_id]
            ico_scale = min(rail_w - 2 * pad, mode_h) * 0.45
            ico.setPos(cx, y0 + (y1 - y0) * 0.60, 0.0)
            ico.setScale(ico_scale)
            # LineSegs 顶点色是白的，用 ColorScale 上灰度
            ico.setColorScale(Vec4(*(_FG_ON if on else _ACCENT)))
            ico.show()
            y = y0 - mode_gap

        tools = [
            ("night", "夜景" if st.street_lights_on else "昼间", st.street_lights_on),
            ("help", "帮助", st.help_visible),
            ("pin", "固定" if self.pinned else "未钉", self.pinned),
        ]
        ty = console_top + 0.012
        for key, text, on in reversed(tools):
            y0, y1 = ty, ty + 0.072
            self._place(
                key, rail_x0 + 0.008, rail_x1 - 0.008, y0, y1,
                action=key, on=on, text=text, scale=_FONT_SM,
                fill=_BG_BTN_ON if on else _BG_BTN,
                fg=_FG_ON if on else _FG,
            )
            ty = y1 + 0.008

        # ---- 分类列（文字为主）
        self._drop_prefix("cat_")
        self._drop_prefix("item_")
        self._drop_prefix("file_")
        self._drop_prefix("trainitem_")
        cat_x0 = rail_x1
        cat_x1 = cat_x0 + cat_w
        if show_cat:
            self._catalog_bg.setPos(0.5 * (cat_x0 + cat_x1),
                                    0.5 * (console_top + top), 0.0)
            self._catalog_bg.setScale(cat_w, top - console_top, 1.0)
            self._catalog_bg.show()

            cells = self._catalog_cells(st)
            titles = {
                "track": "轨道件",
                "scenery": "布景",
                "train": "编组",
                "file": "存档",
                "view": "件库",
            }
            self._place(
                "cat_header", cat_x0 + 0.006, cat_x1 - 0.006,
                top - 0.052, top - 0.010,
                text=titles.get(st.hud_mode, "件库"),
                scale=_FONT_MD, fill=_BG_BTN, fg=_ACCENT, edge=True,
            )

            cell = 0.072
            gap = 0.008
            usable_top = top - 0.062
            usable_bot = console_top + 0.09
            capacity = max(1, int((usable_top - usable_bot) / (cell + gap)))
            if self._catalog_scroll > max(0, len(cells) - capacity):
                self._catalog_scroll = max(0, len(cells) - capacity)
            window = cells[self._catalog_scroll: self._catalog_scroll + capacity]
            iy = usable_top
            for kind, cid, label, selected, action, arg in window:
                key = f"{kind}_{cid}"
                y1, y0 = iy, iy - cell
                pad = 0.006
                # 分类用全称，件名截短但可读
                shown = label if kind == "cat" else self._short(label, 6)
                self._place(
                    key, cat_x0 + pad, cat_x1 - pad, y0, y1,
                    action=action, arg=arg, on=selected,
                    text=shown, scale=_FONT_SM,
                    fill=_BG_BTN_ON if selected else _BG_BTN,
                    fg=_FG_ON if selected else _FG,
                )
                iy = y0 - gap

            if len(cells) > capacity:
                self._place(
                    "page_up", cat_x0 + 0.006, cat_x1 - 0.006,
                    usable_bot + cell + gap, usable_bot + 2 * cell + gap,
                    action="page", arg=-capacity, text="上一页",
                    scale=_FONT_SM, dim=self._catalog_scroll <= 0,
                )
                self._place(
                    "page_down", cat_x0 + 0.006, cat_x1 - 0.006,
                    usable_bot, usable_bot + cell,
                    action="page", arg=capacity, text="下一页",
                    scale=_FONT_SM,
                    dim=self._catalog_scroll + capacity >= len(cells),
                )
            else:
                self._hide("page_up")
                self._hide("page_down")
        else:
            self._catalog_bg.hide()
            self._hide("page_up")
            self._hide("page_down")
            self._hide("cat_header")

        # ---- 底栏
        self._console_bg.setPos(0.0, 0.5 * (bottom + console_top), 0.0)
        self._console_bg.setScale(right - left, _CONSOLE_H, 1.0)
        self._console_top_line.setPos(0.0, console_top - 0.004, 0.0)
        self._console_top_line.setScale(right - left, 0.006, 1.0)
        self._console_top_line.setColor(Vec4(*_ACCENT))
        online = st.train_online
        dim = not online

        dial_r = 0.105
        dial_cx = left + 0.14
        dial_cy = 0.5 * (bottom + console_top) + 0.005
        # aspect2d 上等比例缩放 → 正圆（勿再按宽高比拉伸，否则变椭圆/瓣状）
        self._dial_face.setPos(dial_cx, dial_cy, 0.0)
        self._dial_face.setScale(dial_r)
        self._dial_face.setColor(Vec4(*_g("1a1a1a", 0.95 if online else 0.55)))
        self._dial_face.show()
        self._dial_ring.setPos(dial_cx, dial_cy, 0.0)
        self._dial_ring.setScale(dial_r)
        self._dial_ring.setColor(Vec4(*_g("3a3a3a", 0.55 if dim else 0.90)))
        self._dial_ring.show()

        speed = st.speed_kmh if online else 0.0
        frac = max(0.0, min(1.0, speed / _SPEED_MAX_KMH))
        needle_ang = _speed_angle(frac)
        needle_fg = _FG_MUTED if dim else _ACCENT

        # 刻度盘：绿→黄→红渐变（全量程固定，不上色指针）
        self._speed_arc_root.setPos(dial_cx, dial_cy, 0.0)
        self._speed_arc_root.setScale(dial_r * 0.92)
        if dim:
            self._speed_arc_root.setColorScale(0.45, 0.45, 0.45, 0.45)
        else:
            self._speed_arc_root.setColorScale(1.0, 1.0, 1.0, 1.0)
        self._speed_arc_root.show()

        self._needle_root.removeNode()
        segs = LineSegs("needle")
        segs.setThickness(2.6)
        segs.setColor(Vec4(*needle_fg))
        segs.moveTo(0.0, 0.0, 0.0)
        nx, ny = _circle_xy(needle_ang, 0.72)
        segs.drawTo(nx, ny, 0.0)
        self._needle_root = self._console.attachNewNode(segs.create())
        self._needle_root.setPos(dial_cx, dial_cy, 0.0)
        self._needle_root.setScale(dial_r)
        self._needle_root.setDepthTest(False)
        self._needle_hub.setPos(dial_cx, dial_cy, 0.0)
        self._needle_hub.setScale(0.012)
        self._needle_hub.setColor(Vec4(*needle_fg))
        self._needle_hub.show()

        speed_txt = f"{speed:.0f}" if online else "—"
        self._place(
            "dial_num", dial_cx - 0.07, dial_cx + 0.07,
            dial_cy - 0.015, dial_cy + 0.045,
            text=speed_txt, scale=_FONT_LG,
            fill=(0, 0, 0, 0),
            fg=_FG_MUTED if dim else _FG,
            edge=False,
        )
        self._nodes["dial_num"].hide()
        self._edges["dial_num"].hide()
        self._labels["dial_num"].setPos(dial_cx, dial_cy + 0.005, 0.0)
        self._place(
            "dial_unit", dial_cx - 0.05, dial_cx + 0.05,
            dial_cy - 0.055, dial_cy - 0.025,
            text="km/h", scale=_FONT_SM,
            fill=(0, 0, 0, 0), fg=_FG_MUTED, edge=False,
        )
        self._nodes["dial_unit"].hide()
        self._edges["dial_unit"].hide()

        # 芯片
        chip_y1 = console_top - 0.022
        chip_y0 = chip_y1 - 0.058
        notch_label = NOTCH_LABELS[handle_to_notch(st.handle)]
        if online:
            if st.handle > 0:
                handle_chip = f"{notch_label} 牵引"
            elif st.handle < 0:
                handle_chip = f"{notch_label} 制动"
            else:
                handle_chip = "N 惰行"
        else:
            handle_chip = st.handle_label or "—"
        chips = [
            ("chip_train", self._short(st.train_name or "未选车", 10), 0.22),
            ("chip_handle", handle_chip, 0.16),
            ("chip_dist",
             f"{st.distance_m / 1000.0:.1f} km" if online else "— km", 0.14),
            ("chip_dir", st.direction if online else "—", 0.10),
        ]
        cx = dial_cx + dial_r + 0.04
        for key, text, width in chips:
            self._place(
                key, cx, cx + width, chip_y0, chip_y1,
                text=text, dim=dim, scale=_FONT_SM,
                fill=_BG_CHIP, fg=_FG_MUTED if dim else _FG,
            )
            cx += width + 0.010

        # 档位条：收窄，放在表盘右侧中段（不再拉满到右钮）
        track_w = 0.52
        track_x0 = dial_cx + dial_r + 0.04
        track_x1 = track_x0 + track_w
        track_y = bottom + 0.125
        self._notch_x0, self._notch_x1 = track_x0, track_x1

        self._place(
            "notch_caption", track_x0, track_x1,
            track_y + 0.048, track_y + 0.078,
            text="制动 ←  惰行  → 牵引",
            scale=_FONT_SM, fill=(0, 0, 0, 0), fg=_FG_MUTED, edge=False,
        )
        self._nodes["notch_caption"].hide()
        self._edges["notch_caption"].hide()

        self._place(
            "notch_track", track_x0, track_x1,
            track_y - 0.012, track_y + 0.012,
            dim=dim, fill=_TRACK if not dim else _BG_DIM,
        )
        notch = handle_to_notch(st.handle)
        for i, lab in enumerate(NOTCH_LABELS):
            t = i / (len(NOTCH_LABELS) - 1)
            nx = track_x0 + t * (track_x1 - track_x0)
            tick_key = f"notch_{i}"
            self._ensure(self._console, tick_key)
            self._place(
                tick_key, nx - 0.005, nx + 0.005,
                track_y - 0.030, track_y + 0.030,
                action="handle_notch" if online else None,
                arg=i, text="",
                fill=_ACCENT if i == 4 else _TRACK,
                dim=dim, edge=False,
            )
            lab_key = f"notchlab_{i}"
            self._ensure(self._console, lab_key)
            show_lab = lab if i in (0, 3, 4, 5, 8) else ""
            self._place(
                lab_key, nx - 0.028, nx + 0.028,
                track_y - 0.068, track_y - 0.038,
                text=show_lab, scale=_FONT_SM * 0.9,
                fill=(0, 0, 0, 0),
                fg=_ACCENT if (i == notch and online) else _FG_MUTED,
                edge=False,
            )
            self._nodes[lab_key].hide()
            self._edges[lab_key].hide()

        t = notch / (len(NOTCH_LABELS) - 1)
        kx = track_x0 + t * (track_x1 - track_x0)
        self._place(
            "notch_knob", kx - 0.016, kx + 0.016,
            track_y - 0.026, track_y + 0.026,
            action="handle_notch" if online else None,
            arg=notch, on=online, text="",
            fill=_ACCENT if online else _BG_DIM,
        )
        self._place(
            "notch_hint", track_x0, track_x1,
            bottom + 0.016, bottom + 0.048,
            text=("拖动手柄或点档位" if online
                  else "先选列车 · 按 N 或点左栏「列车」上线"),
            scale=_FONT_SM, fill=(0, 0, 0, 0), fg=_FG_MUTED, edge=False,
        )
        self._nodes["notch_hint"].hide()
        self._edges["notch_hint"].hide()

        # 右三钮
        bw, bh = 0.100, 0.090
        bx = right - 0.34
        by0 = bottom + 0.075
        by1 = by0 + bh
        self._place(
            "btn_emergency", bx, bx + bw, by0, by1,
            action="emergency" if online else None,
            text="紧急", danger=True, dim=dim, scale=_FONT_MD,
            fill=_BG_DANGER if online else _BG_DANGER_DIM,
            fg=_FG_ON,
        )
        self._place(
            "btn_reverse", bx + bw + 0.014, bx + 2 * bw + 0.014, by0, by1,
            action="reverse" if online else None,
            text="换向", dim=dim, scale=_FONT_MD,
            fill=_BG_CHIP, fg=_FG_MUTED if dim else _FG,
        )
        self._place(
            "btn_horn", bx + 2 * (bw + 0.014), bx + 3 * bw + 0.028, by0, by1,
            action="horn" if online else None,
            text="鸣笛", dim=dim, scale=_FONT_MD,
            fill=_BG_CHIP, fg=_FG_MUTED if dim else _FG,
        )

        toast = st.toast or ""
        if len(toast) > 24:
            toast = toast[:23] + "…"
        toast_x0 = min(cx + 0.02, right - 0.38)
        self._place(
            "toast", toast_x0, right - 0.36,
            chip_y0, chip_y1,
            text=toast, scale=_FONT_SM,
            fill=_BG_BTN if toast else (0, 0, 0, 0),
            fg=_ACCENT if toast else _FG_MUTED,
            edge=bool(toast),
        )
        if not toast:
            self._hide("toast")

        if st.help_visible:
            self._help.show()
            hw = min(1.2, aspect * 0.75)
            self._help_card.setPos(0.0, 0.12, 0.0)
            self._help_card.setScale(hw, 0.42, 1.0)
            self._help_label.node().setText(HELP_OVERLAY)
            self._help_label.setScale(_FONT_SM)
            self._help_label.setPos(-hw * 0.46, 0.26, 0.0)
        else:
            self._help.hide()

        if self._progress_text:
            self._layout_progress()

    def _catalog_cells(self, st: HudState):
        cells = []
        mode = st.hud_mode
        if mode == "file":
            cells.append(("file", "save", "保存", False, "file_save", None))
            cells.append(("file", "open", "读取", False, "file_open", None))
            return cells
        if mode == "train":
            for i, (tid, name) in enumerate(st.train_list):
                cells.append((
                    "trainitem", tid, name, i == st.train_index,
                    "select_train", i,
                ))
            if st.train_list:
                cells.append(
                    ("trainitem", "go", "上线", False, "spawn_train", None)
                )
            return cells
        show_scenery = mode == "scenery" or (self.pinned and st.scenery)
        if show_scenery:
            for cat_id, label in st.scenery_rail:
                cells.append((
                    "cat", cat_id, label,
                    st.scenery_category == cat_id,
                    "scenery_category", cat_id,
                ))
            for i, (iid, name) in enumerate(st.scenery_items):
                cells.append((
                    "item", iid, name, i == st.scenery_index,
                    "select_item", i,
                ))
            return cells
        for cat_id, label in TRACK_CATEGORY_ORDER:
            if st.track_categories and cat_id not in st.track_categories:
                continue
            cells.append((
                "cat", cat_id, label, st.category == cat_id,
                "category", cat_id,
            ))
        for i, (iid, name) in enumerate(st.catalog_items):
            cells.append((
                "item", iid, name, i == st.piece_index,
                "select_item", i,
            ))
        return cells

    def _notch_at_x(self, mx: float) -> int | None:
        if self._notch_x1 <= self._notch_x0:
            return None
        t = (mx - self._notch_x0) / (self._notch_x1 - self._notch_x0)
        t = max(0.0, min(1.0, t))
        return int(round(t * (len(NOTCH_LABELS) - 1)))


HELP_OVERLAY = (
    "V 放置/视角　B 布景　H 帮助　左键 操作　右键 环绕　滚轮 缩放\n"
    "R 旋转　C 闭合　X 删除　, . 分类　[ ] 换件　1-9 直选　退格 撤销\n"
    "N 上列车　↑↓ 手柄　空格 惰行　K 换向　J 鸣笛　Ctrl+S / O 存读档\n"
    "左栏：轨/景/车/视/档　·　固定钉住件库　·　底栏 Tomix 九档"
)


def build_default_hud(base) -> Hud:
    return Hud(base, font=cjk_font(base))
