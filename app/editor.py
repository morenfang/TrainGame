"""轨道编辑器：交互状态机。

一句话说清它做什么
------------------------------------------------
把"鼠标在哪、按了什么键"翻译成对 :class:`core.track.layout.Layout` 的修改，
然后让 :class:`render.scene.LayoutView` 把结果同步到画面。**编辑器自己不做任何
几何计算** —— 吸附位姿由 ``Layout.attach`` 的公式给出，闭环判定由
``core.track.path.detect_closure`` 给出。编辑器只负责选目标、给预览、记撤销。

为什么端口吸附用"屏幕像素距离"而不是三维射线求交
------------------------------------------------
拉远看整个沙盘时，几十个端口在屏幕上挤在几十个像素内，而它们在三维空间里
可能相距十几米。射线求交会选中"离射线最近"的那个 —— 但那常常不是用户**看着**
想点的那个。屏幕距离恰好等价于"用户瞄的是哪个"，而且天然自带一个以像素为单位
的吸附半径，手感可以按屏幕直觉去调。

一个当前的核心限制（不是编辑器能绕过的）
------------------------------------------------
``Layout`` 只支持**一张连通的装配图**（恰有一个根件，``rebuild_poses`` 会校验）。
所以自由放置只在空布局时可用；已有轨道后，新件必须吸附到某个空闲端口上。
非空布局下把鼠标移到没有端口的地方，幽灵会变红并给出提示 —— 这样限制是
可见的、可理解的，而不是"点了没反应"。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from panda3d.core import ClockObject, CollisionRay, Point3

from core.geometry import Pose
from core.track.catalog import Catalog, PieceDef
from core.track.layout import Layout, LayoutError, PortKey
from core.train.consist import TrainCatalog, TrainSpec
from core.train.dynamics import Train
from render import scenery as scenery_mod
from render import style
from render.audio import TrainAudio
from render.camera import OrbitCamera
from render.overlay import build_footprint_marker, tint
from render.scene import LayoutView
from render.train_view import TrainView
from render.transform import apply_pose

#: 鼠标到端口的吸附半径（像素）。
PORT_PICK_RADIUS_PX = 64.0
#: 鼠标到轨道件的悬停判定半径（像素）。
PIECE_PICK_RADIUS_PX = 110.0
#: 自由放置时每次旋转的角度（度）。取 22.5° 与件库的转角对齐，方便手工对缝。
ROTATE_STEP_DEG = 22.5
#: 撤销栈上限（每步存一份完整拓扑快照，几十份的开销可以忽略）。
UNDO_LIMIT = 80
#: 判定"这是一次点击"的最大鼠标位移（像素）。超过就当成拖拽，不放置。
CLICK_SLOP_PX = 5.0
#: 按住不放的相机移动键（WASD 平移视角，QE 转向）。
#: 单独列出来是因为它们要同时挂 "按下" 和 "抬起" 两个事件。
HELD_KEYS = ("w", "a", "s", "d", "q", "e")

#: 弹出文件名输入框（Ctrl+S / Ctrl+O）后可输入的字符（小写字母 + 数字）。
#: 按键事件名就是字符本身（见 ``key_actions`` 里关于符号键的说明）。
_SAVE_NAME_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
#: 文件名长度上限（字符数）。
_SAVE_NAME_MAX = 40
#: 读档提示条里"已有存档"名单的总字符预算（超出就截断成 "…"）。
#: 提示条是一行居中文字，名单太长会把面板挤出屏幕。
_PROMPT_LIST_CHARS = 36
#: 提示条里回显的目标文件名最多显示几个字符（超出掐尾加 "…"）。
#: 掐了也不影响读档 —— 输入框里那一串才是用户真正写的名字。
_PROMPT_ECHO_MAX = 22

#: 鼠标两种模式：**放置轨道**（左键放件、选中）与**视角**（左键拖拽转视角）。
#:
#: 为什么要分：拼轨道时左键是"放下这一节"，开车看车时左键却是"转一下视角"，
#: 同一只手不可能既放轨道又转视角。加载列车时自动切到视角模式，用户再用 V 切回来。
MODE_BUILD = "build"
MODE_VIEW = "view"
#: 第三种鼠标模式：**布景**。左键在空地上放下房屋 / 车站 / 大山 / 树，
#: 按 B 进出。与「放置轨道」「视角」并列 —— 布景不参与走线，所以它不能复用
#: 轨道那套「吸附到端口」的逻辑，而是直接把物件摆到鼠标指着的地面上。
MODE_SCENERY = "scenery"

#: 文件名输入框的两种用途：Ctrl+S 另存、Ctrl+O 读档。
#: 两者共用同一个输入框，只是回车之后的动作不同（见 :meth:`TrackEditor._confirm_file_prompt`）。
PROMPT_SAVE = "save"
PROMPT_LOAD = "load"

#: 布景模式里的分类（按 ``,`` / ``.`` 切换）。
SCENERY_CATEGORIES = ("building", "station", "mountain", "tree")

#: 每个布景分类的显示名（HUD 的"分类菜单"共用）。
SCENERY_CATEGORY_LABELS = {
    "building": "建筑",
    "station": "车站",
    "mountain": "山",
    "tree": "树木",
}

#: 每类里可摆放的物件：``id → 显示名``（按 ``[`` / ``]`` 循环、``1``–``9`` 直选）。
SCENERY_ITEMS = {
    "building": (
        ("cottage", "普通民房"),
        ("tower", "高楼"),
        ("mall", "商场"),
        ("shop", "便利店"),
    ),
    "station": (
        ("classical", "老式车站（古典）"),
        ("modern", "新式车站（现代）"),
    ),
    "mountain": (
        ("mountain", "大山"),
        ("hill", "山丘"),
    ),
    "tree": (
        ("pine", "针叶松"),
        ("oak", "阔叶树"),
        ("poplar", "白杨"),
        ("palm", "棕榈"),
    ),
}

#: 布景悬停拾取的半径（米，世界坐标）：鼠标落到某件布景占地轮廓这么近就算选中。
SCENERY_PICK_RADIUS = 14.0

#: 布景物件 id → 它在 ``Scenery`` 里存进哪个字段。
_SCENERY_KIND_FIELD = {
    "cottage": "houses", "tower": "houses", "mall": "houses", "shop": "houses",
    "classical": "stations", "modern": "stations",
    "mountain": "peaks", "hill": "peaks",
    "pine": "trees", "oak": "trees", "poplar": "trees", "palm": "trees",
}
#: 屏幕投影时"点在镜头前方"的最小深度（米）。比它更近就当在相机背后 / 贴着镜头。
_MIN_CAMERA_DEPTH = 1e-4
#: 合拢接缝的判定阈值（米 / 弧度）。
#: ``core.track.path`` 里解释过：真闭环的接缝误差在 1e-13 量级，接不上的在米级，
#: 相差十几个数量级，所以这里可以卡得极死而绝不会误判。
SEAM_POSITION_TOL = 1e-6
SEAM_HEADING_TOL = 1e-9

#: 手柄每次推拉的档位步长。0.25 = 4 下推到底（原来的 0.1 要按 10 下，
#: 而方向键按住时的系统重复率不一，用户很容易觉得"推了没反应"）。
HANDLE_STEP = 0.25

#: **沙盘手感倍率**：游戏里牵引力与制动力一起放大的倍数（1.0 = 物理标定值）。
#:
#: 为什么不直接改 `data/trains.json`：那套参数是照**真实车型**标定的 —— 绿皮车
#: 0-100 km/h 要两分多钟、复兴号 0-200 要一分半，`tests/test_dynamics.py` 里有
#: 断言把这几个量级钉死了，它同时也是"绿皮车肉、复兴号猛"这个设计意图的兑现处。
#: 而"在沙盘上开起来跟不跟手"是**另一个问题**：环线只有两百多米，按真实加速
#: 曲线开，用户会觉得车"没劲"。
#:
#: 所以把它做成一个独立的旋钮：数据仍然说真话，游戏的手感由这里定。
#: 只有牵引与制动被放大，阻力与坡度**不**参与（否则上坡限速那些结论会被一起改掉）。
#: 觉得还不够跟手就调大它 —— 这是唯一需要动的地方。
DRIVE_BOOST = 1.8

#: 左下角的操作说明。
#:
#: 单独提成常量，是为了让"面板放不放得下这段文字"这件事**可以测**：
#: ``tests/test_app_hud.py`` 的 ``_WORST_CASE`` 直接引用它，于是往这里加一行、
#: 加了之后超出面板能容纳的高度，测试会当场发现（而不是等用户看到字压在一起）。
HELP_TEXT = (
    "V 放置/视角切换  左键 放轨或拖视角  右键 环绕  中键 平移  滚轮 缩放\n"
    "R 旋转  T 换接驳端口  C 一键闭合  U 扳道岔  X 只删鼠标下这一节\n"
    ", . 换分类（直轨/曲线/坡道/高架桥/交叉/道岔）  [ ] 换同类里下一件\n"
    "1-9 直选第 n 件  退格 撤销  Ctrl+S 另存  Ctrl+O 读档（都可填名字）\n"
    "W A S D 平移视角  Q E 转向  F 取景\n"
    "B 布景模式（, . 换分类  [ ] 换件  1-9 直选  左键放  X 删  R 转）\n"
    "N 召唤列车（自动切到视角）  M 收起  ↑ 牵引  ↓ 制动  空格 惰行\n"
    "K 换向（先急刹再反向加速）  J 鸣笛  Shift+空格 急停\n"
    "拼环：按 . 到「曲线」挑弧度，几节后按 C 补完"
)


# --------------------------------------------------------------------------- #
# 待确认的放置
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Placement:
    """一次待确认的放置。

    ``target`` 为 ``None`` 表示自由放置（只在空布局里允许）；
    ``allowed`` 为假时幽灵会显示成红色，按左键会被拒绝并给出 ``note``。
    """

    def_id: str
    pose: Pose
    my_port: str | None = None
    target: PortKey | None = None
    allowed: bool = True
    note: str = ""

    @property
    def snapped(self) -> bool:
        return self.target is not None


# --------------------------------------------------------------------------- #
# 编辑器
# --------------------------------------------------------------------------- #

class TrackEditor:
    """鼠标驱动的轨道编辑器。"""

    def __init__(self, base, catalog: Catalog, *, layout: Layout | None = None,
                 hud=None, save_path: str | Path | None = None,
                 trains: TrainCatalog | None = None,
                 save_extra: Mapping[str, object] | None = None,
                 scenery: scenery_mod.Scenery | None = None,
                 user_scenery: scenery_mod.Scenery | None = None):
        self.base = base
        self.catalog = catalog
        self.layout = layout if layout is not None else Layout(catalog=catalog)
        self.hud = hud
        self.save_path = Path(save_path) if save_path else \
            Path(__file__).resolve().parents[1] / "saves" / "layout.json"
        #: 存档里额外记的一行（场景的布景预设等）。Ctrl+S 时并进 JSON —— 存档只写
        #: 轨道拓扑的话，从场景里存一次档就把"这局配的是哪份布景"丢了。
        self.save_extra: dict[str, object] = dict(save_extra or {})

        self.view = LayoutView(self.layout, base.render)
        self.camera = OrbitCamera(base)
        self._ghost_root = base.render.attachNewNode("ghost")

        #: 列车目录与"当前上线的那一列"。目录是纯数据，载入一次即可。
        self.trains = trains if trains is not None else TrainCatalog.builtin()
        self._train_index = 0
        self.train_view: TrainView | None = None
        #: 手柄位：``> 0`` 牵引、``< 0`` 制动、``0`` 惰行。维护在编辑器这一层，
        #: 于是"换一列车"不会把用户推好的档位弄丢。
        self.train_handle = 0.0

        self.categories = list(catalog.categories)
        self._category_index = 0
        self._piece_index = 0
        self.attach_slot = 0
        self.free_heading = 0.0

        self.placement: Placement | None = None
        self.hover_piece: int | None = None
        self.hover_port: PortKey | None = None

        self._ghost = None
        self._ghost_key: tuple[str, float] | None = None
        self._anchors: dict[str, tuple[float, float]] = {}
        #: 件 id → 局部抓手点（端口 + route 中点），供 pick_piece 用
        self._handles: dict[str, tuple[tuple[float, float, float], ...]] = {}

        # 合拢接缝的缓存：HUD 每帧都要问，而枚举是 O(空闲端口²)
        self._seam_key: tuple | None = None
        self._seam_cache: tuple | None = None

        self._undo: list[dict] = []
        self._redo: list[dict] = []

        self._drag_mode: str | None = None
        self._last_mouse: tuple[float, float] | None = None
        self._press_mouse: tuple[float, float] | None = None
        self._press_moved = 0.0
        self._held: set[str] = set()

        self._toast = ""
        self._toast_frames = 0

        #: 存档所在的目录（Ctrl+S 填文件名时，保存到 ``<该目录>/<名字>.json``）。
        self._save_dir = self.save_path.parent
        #: 文件名输入框当前开着没；开着时值是用途（:data:`PROMPT_SAVE` /
        #: :data:`PROMPT_LOAD`），关着时是 ``None``。键盘也只喂给这个输入框。
        self._prompt: str | None = None
        #: 文件名输入框当前的内容。
        self._prompt_buffer = ""
        #: 文件名校验失败时的一句话（显示在提示里）。
        self._prompt_error = ""

        #: 左键现在是"放轨道"还是"拖视角"（见 :data:`MODE_BUILD` / :data:`MODE_VIEW`）。
        self.mode = MODE_BUILD
        #: 当前这次左键按住是否在拖视角（视角模式下）。右键 / 中键的拖拽各有事件，
        #: 不能被左键的抬起顺手取消掉。
        self._left_orbiting = False

        # ---- 布景（见 MODE_SCENERY）：程序化底座 + 用户手工摆的几件
        #: 底座布景（场景预设 / 存档里记的预设重算出来的那部分），不可编辑。
        self.base_scenery: scenery_mod.Scenery = (
            scenery if scenery is not None else scenery_mod.Scenery())
        #: 用户手工摆放的布景（房 / 车站 / 大山 / 树），可增删、随存档一起保存。
        self.user_scenery: scenery_mod.Scenery = (
            user_scenery if user_scenery is not None else scenery_mod.Scenery())
        #: 两者并成一份、烘成一个几何节点后的结果（一次绘制调用）。
        self.scenery_node = None
        self._scenery_category_index = 0
        self._scenery_kind_index = 0
        self._scenery_heading = 0.0
        #: 鼠标正悬停在哪件**手工布景**上（``(类别字段, 该类别内下标)``）。
        self.hover_scenery: tuple[str, int] | None = None
        #: 幽灵预览（布景模式）：临时一个节点，``_scenery_ghost_key`` 变了才重建。
        self._scenery_ghost_root = base.render.attachNewNode("scenery_ghost")
        self._scenery_ghost = None
        self._scenery_ghost_key: tuple | None = None
        #: 悬停高亮：一件半透明的占地轮廓（房 / 车站是矩形，山 / 树是圆）。
        self._scenery_highlight = base.render.attachNewNode("scenery_highlight")
        self._rebuild_scenery()

        #: 列车音效（行驶声 / 轮轨声随车速变速 + 鸣笛）。null 音频下自动退化为空壳。
        self.audio = TrainAudio(base)

    # ==================================================================== #
    # 当前选中的件
    # ==================================================================== #

    @property
    def category(self) -> str:
        return self.categories[self._category_index]

    @property
    def pieces_in_category(self) -> tuple[PieceDef, ...]:
        return self.catalog.by_category(self.category)

    @property
    def current_piece(self) -> PieceDef:
        pieces = self.pieces_in_category
        return pieces[self._piece_index % len(pieces)]

    def cycle_piece(self, step: int) -> None:
        pieces = self.pieces_in_category
        self._piece_index = (self._piece_index + step) % len(pieces)
        self.attach_slot = 0
        self.notify(f"当前件：{self.current_piece.name}")

    def cycle_category(self, step: int) -> None:
        self._category_index = (self._category_index + step) % len(self.categories)
        self._piece_index = 0
        self.attach_slot = 0
        self.notify(f"类别：{self._category_label()}")

    def select_piece(self, index: int) -> None:
        pieces = self.pieces_in_category
        if 0 <= index < len(pieces):
            self._piece_index = index
            self.attach_slot = 0
            self.notify(f"当前件：{self.current_piece.name}")

    def select_piece_id(self, piece_id: str) -> bool:
        """按 id 选中件，并自动切到它所在的类别。

        比 ``select_piece(index)`` 好用得多 —— 件在目录里的先后顺序是数据细节，
        调用者（脚本、未来的选件面板、快捷栏）想说的是"我要那个 R40 的 45° 弯轨"。
        """
        piece = self.catalog[piece_id]
        categories = list(self.categories)
        if piece.category not in categories:
            return False
        self._category_index = categories.index(piece.category)
        ids = [p.id for p in self.pieces_in_category]
        self._piece_index = ids.index(piece_id)
        self.attach_slot = 0
        return True

    _CATEGORY_LABELS = {
        "straight": "直轨", "curve": "曲线", "grade": "坡道",
        "viaduct": "高架桥", "crossing": "交叉", "switch": "道岔",
        "terminal": "终端",
    }

    @classmethod
    def category_label(cls, category: str) -> str:
        """类别的中文名（未知类别原样返回）。"""
        return cls._CATEGORY_LABELS.get(category, category)

    def _category_label(self) -> str:
        return self.category_label(self.category)

    def category_menu(self) -> str:
        """一行列出**所有**类别，当前那个用方括号框起来。

        存在的理由：类别是靠 ``,`` / ``.`` 循环切换的，而循环切换的菜单如果不显示
        出来就等于不存在 —— 用户的第一反应是"只有直道？那怎么闭合？"，
        因为他看不到"曲线"这一类。一个可见的菜单把"猜"变成"看"。
        """
        return " ".join(
            f"[{self.category_label(cat)}]" if cat == self.category
            else self.category_label(cat)
            for cat in self.categories
        )

    def cycle_attach_port(self) -> None:
        ids = self.current_piece.port_ids
        if len(ids) < 2:
            self.notify("这一节只有一个端口")
            return
        self.attach_slot = (self.attach_slot + 1) % len(ids)
        self.notify(f"接驳端口：{ids[self.attach_slot]}")

    def rotate(self, step: int) -> None:
        self.free_heading += math.radians(ROTATE_STEP_DEG * step)
        self.notify(f"朝向 {math.degrees(self.free_heading) % 360:.1f}°"
                    "（仅在空场上自由放置时生效）")

    def reset_heading(self) -> None:
        self.free_heading = 0.0
        self.notify("朝向已归零")

    # ==================================================================== #
    # 鼠标模式：放置轨道 / 视角
    # ==================================================================== #

    @property
    def building(self) -> bool:
        """左键现在是"放轨道"吗？（HUD、放置、键位都问它。）"""
        return self.mode == MODE_BUILD

    @property
    def placing_scenery(self) -> bool:
        """左键现在是"摆布景"吗？"""
        return self.mode == MODE_SCENERY

    def _mode_label(self) -> str:
        if self.placing_scenery:
            return "布景"
        return "放置轨道" if self.building else "视角"

    def toggle_mode(self) -> None:
        """在「放置轨道」与「视角」之间切换左键的含义。

        视角模式下左键改拖视角（与右键一致），幽灵预览整块收起 —— 加载列车之后
        进入的就是这一档，免得鼠标一划就把轨道甩到场景里。
        """
        self.mode = MODE_VIEW if self.building else MODE_BUILD
        if not self.building:
            self._hide_ghost()
            self._hide_scenery_ghost()
        self.notify(f"鼠标模式：{self._mode_label()}")

    def toggle_scenery_mode(self) -> None:
        """进出布景模式：B 开、B 关（关的时候回到放置轨道）。"""
        if self.placing_scenery:
            self.mode = MODE_BUILD
            self._hide_scenery_ghost()
            self._clear_scenery_highlight()
        else:
            self.mode = MODE_SCENERY
            self._hide_ghost()
        self.notify(f"鼠标模式：{self._mode_label()}")

    # ==================================================================== #
    # 布景模式：选件 / 放置 / 删除 / 重建
    # ==================================================================== #

    @property
    def scenery_category(self) -> str:
        return SCENERY_CATEGORIES[self._scenery_category_index
                                  % len(SCENERY_CATEGORIES)]

    @property
    def scenery_items(self) -> tuple[tuple[str, str], ...]:
        return SCENERY_ITEMS[self.scenery_category]

    def current_scenery_kind(self) -> str:
        """布景模式下当前选中的物件 id。"""
        return self.scenery_items[self._scenery_kind_index
                                  % len(self.scenery_items)][0]

    def current_scenery_label(self) -> str:
        """当前物件的显示名。"""
        return self.scenery_items[self._scenery_kind_index
                                  % len(self.scenery_items)][1]

    def cycle_scenery_category(self, step: int) -> None:
        """换下一个布景分类（``,`` / ``.``）。"""
        self._scenery_category_index = (
            self._scenery_category_index + step) % len(SCENERY_CATEGORIES)
        self._scenery_kind_index = 0
        self._scenery_ghost_key = None
        self.notify(f"布景分类：{SCENERY_CATEGORY_LABELS[self.scenery_category]}")

    def cycle_scenery_kind(self, step: int) -> None:
        """换下一种可摆放的布景（``[`` / ``]``）。"""
        self._scenery_kind_index = (
            self._scenery_kind_index + step) % len(self.scenery_items)
        self._scenery_ghost_key = None      # 换件，幽灵得重烘
        self.notify(f"布景：{self.current_scenery_label()}")

    def select_scenery_kind(self, index: int) -> None:
        """直选当前分类里的第 ``index`` 件（``1``–``9``）。"""
        if 0 <= index < len(self.scenery_items):
            self._scenery_kind_index = index
            self._scenery_ghost_key = None
            self.notify(f"布景：{self.current_scenery_label()}")

    def scenery_menu(self) -> str:
        """一行列出**所有**布景分类，当前那个用方括号框起来。"""
        return " ".join(
            f"[{SCENERY_CATEGORY_LABELS[cat]}]"
            if cat == self.scenery_category else SCENERY_CATEGORY_LABELS[cat]
            for cat in SCENERY_CATEGORIES
        )

    def scenery_items_menu(self) -> str:
        """一行列出当前分类里的物件，选中那个用方括号框起来。"""
        parts = []
        for i, (_item_id, label) in enumerate(self.scenery_items):
            parts.append(f"[{label}]" if i == self._scenery_kind_index else label)
        return " ".join(parts)

    def rotate_scenery(self, step: int) -> None:
        """旋转待放的布景朝向（``R`` / ``Shift+R``）。"""
        self._scenery_heading += math.radians(ROTATE_STEP_DEG * step)
        self.notify(f"布景朝向 {math.degrees(self._scenery_heading) % 360:.0f}°")

    def reset_scenery_heading(self) -> None:
        self._scenery_heading = 0.0
        self.notify("布景朝向已归零")

    # 以下四个是「键位派发」：同一个键在布景模式与轨道模式做不同的事。
    def _rotate_action(self, step: int) -> None:
        if self.placing_scenery:
            self.rotate_scenery(step)
        else:
            self.rotate(step)

    def _cycle_item_action(self, step: int) -> None:
        if self.placing_scenery:
            self.cycle_scenery_kind(step)
        else:
            self.cycle_piece(step)

    def _cycle_category_action(self, step: int) -> None:
        if self.placing_scenery:
            self.cycle_scenery_category(step)
        else:
            self.cycle_category(step)

    def _select_item_action(self, index: int) -> None:
        if self.placing_scenery:
            self.select_scenery_kind(index)
        else:
            self.select_piece(index)

    def _delete_action(self) -> None:
        if self.placing_scenery:
            self.delete_scenery_under_cursor()
        else:
            self.delete_hovered()

    def _escape_action(self) -> None:
        if self.placing_scenery:
            self.reset_scenery_heading()
        else:
            self.reset_heading()

    def _place_action(self) -> None:
        if self.placing_scenery:
            self.place_scenery()
        else:
            self.place()

    def _make_scenery_item(self, kind: str, x: float, z: float, heading: float):
        """按物件类型 + 落点造一件布景数据（尺寸取沙盘里合眼的默认值）。"""
        seed = random.randrange(1 << 30)
        if kind == "cottage":
            return scenery_mod.House(x=x, z=z, width=9.0, depth=7.0, height=4.6,
                                     heading=heading, seed=seed)
        if kind == "tower":
            return scenery_mod.House(
                x=x, z=z, kind=scenery_mod.HOUSE_TOWER,
                width=13.0, depth=13.0, height=26.0, heading=heading, seed=seed)
        if kind == "mall":
            return scenery_mod.House(
                x=x, z=z, kind=scenery_mod.HOUSE_MALL,
                width=30.0, depth=18.0, height=8.5, heading=heading, seed=seed)
        if kind == "shop":
            return scenery_mod.House(
                x=x, z=z, kind=scenery_mod.HOUSE_SHOP,
                width=7.0, depth=6.0, height=3.4, heading=heading, seed=seed)
        if kind == "classical":
            return scenery_mod.Station(
                x=x, z=z, style=scenery_mod.STATION_CLASSICAL,
                width=22.0, depth=11.0, height=6.8, heading=heading, seed=seed,
                with_platform=True)
        if kind == "modern":
            return scenery_mod.Station(
                x=x, z=z, style=scenery_mod.STATION_MODERN,
                width=26.0, depth=12.0, height=6.0, heading=heading, seed=seed,
                with_platform=True)
        if kind == "mountain":
            return scenery_mod.Peak(x=x, z=z, radius=40.0, height=30.0, seed=seed)
        if kind == "hill":
            return scenery_mod.Peak.hill(x=x, z=z, radius=26.0, height=10.0,
                                         seed=seed)
        if kind == "pine":
            return scenery_mod.Tree(x=x, z=z, kind=scenery_mod.TREE_PINE,
                                    height=11.0, seed=seed)
        if kind == "oak":
            return scenery_mod.Tree(x=x, z=z, kind=scenery_mod.TREE_OAK,
                                    height=9.0, seed=seed)
        if kind == "poplar":
            return scenery_mod.Tree(x=x, z=z, kind=scenery_mod.TREE_POPLAR,
                                    height=15.0, seed=seed)
        if kind == "palm":
            return scenery_mod.Tree(x=x, z=z, kind=scenery_mod.TREE_PALM,
                                    height=8.0, seed=seed)
        raise ValueError(f"未知的布景物件 {kind!r}")

    def _rebuild_scenery(self) -> None:
        """把「底座 + 手工」并成一份，烘成一个节点挂到场景上。"""
        if self.scenery_node is not None:
            self.scenery_node.removeNode()
        combined = self.base_scenery.merged(self.user_scenery)
        self.scenery_node = scenery_mod.build_scenery(combined, name="scenery")
        self.scenery_node.reparentTo(self.base.render)

    def compute_scenery_placement(self):
        """布景模式下，待放物件落在鼠标指着的哪个地面点。"""
        if not self.placing_scenery:
            return None
        ground = self.mouse_ground()
        if ground is None:
            return None
        return (self.current_scenery_kind(), ground[0], ground[2],
                self._scenery_heading)

    def place_scenery(self) -> tuple[str, int] | None:
        """在鼠标指着的地面放下一件布景；返回 ``(类别字段, 新下标)``。"""
        placement = self.compute_scenery_placement()
        if placement is None:
            self.notify("把鼠标移到空地上，左键放下布景")
            return None
        kind, x, z, heading = placement
        item = self._make_scenery_item(kind, x, z, heading)
        field = _SCENERY_KIND_FIELD[kind]
        self.push_undo()
        items = list(getattr(self.user_scenery, field))
        items.append(item)
        kwargs = {f: getattr(self.user_scenery, f)
                  for f in scenery_mod.ALL_SCENERY_FIELDS}
        kwargs[field] = tuple(items)
        self.user_scenery = scenery_mod.Scenery(**kwargs)
        self._rebuild_scenery()
        self.notify(f"放下 {self.current_scenery_label()}")
        return (field, len(items) - 1)

    def delete_scenery_under_cursor(self) -> None:
        """删掉鼠标正悬停的那件手工布景。"""
        if self.hover_scenery is None:
            self.notify("把鼠标移到要删除的布景上")
            return
        field, index = self.hover_scenery
        self.push_undo()
        self.user_scenery = scenery_mod.without_item(
            self.user_scenery, field, index)
        self.hover_scenery = None
        self._clear_scenery_highlight()
        self._rebuild_scenery()
        self.notify("已删除那件布景")

    def _pick_scenery(self) -> tuple[str, int] | None:
        """鼠标指着的那件手工布景（按占地轮廓的最近距离）。"""
        ground = self.mouse_ground()
        if ground is None:
            return None
        mx, mz = ground[0], ground[2]
        best, best_distance = None, SCENERY_PICK_RADIUS
        for field, index, item in scenery_mod.placeable_items(self.user_scenery):
            distance = item.footprint().distance_to(mx, mz)
            if distance < best_distance:
                best, best_distance = (field, index), distance
        return best

    def _refresh_scenery_ghost(self) -> None:
        """布景模式的幽灵预览：在落点烘一件半透明的物件。"""
        placement = self.compute_scenery_placement()
        if placement is None:
            self._hide_scenery_ghost()
            return
        kind, x, z, heading = placement
        # 位置按 0.25 m / 1° 取整做缓存键，鼠标动一点点时不必整件重烘
        key = (kind, round(x * 4.0), round(z * 4.0), round(math.degrees(heading)))
        if key == self._scenery_ghost_key and self._scenery_ghost is not None:
            return
        self._hide_scenery_ghost()
        item = self._make_scenery_item(kind, x, z, heading)
        node = scenery_mod.build_scenery(
            scenery_mod.Scenery(**{_SCENERY_KIND_FIELD[kind]: (item,)}),
            name="scenery_ghost",
        )
        tint(node, style.GHOST_OK_TINT)
        node.reparentTo(self._scenery_ghost_root)
        self._scenery_ghost = node
        self._scenery_ghost_key = key

    def _hide_scenery_ghost(self) -> None:
        if self._scenery_ghost is not None:
            self._scenery_ghost.removeNode()
            self._scenery_ghost = None
        self._scenery_ghost_key = None

    def _refresh_scenery_highlight(self) -> None:
        """给鼠标悬停的那件布景画一圈半透明的占地轮廓。"""
        self._clear_scenery_highlight()
        if self.hover_scenery is None or not self.placing_scenery:
            return
        field, index = self.hover_scenery
        items = getattr(self.user_scenery, field)
        if not (0 <= index < len(items)):
            return
        marker = build_footprint_marker(items[index].footprint())
        marker.reparentTo(self._scenery_highlight)

    def _clear_scenery_highlight(self) -> None:
        for child in self._scenery_highlight.getChildren():
            child.removeNode()

    # ==================================================================== #
    # 鼠标 → 世界
    # ==================================================================== #

    def _has_mouse(self) -> bool:
        return self.base.mouseWatcherNode is not None \
            and self.base.mouseWatcherNode.hasMouse()

    def _mouse(self) -> tuple[float, float]:
        pointer = self.base.mouseWatcherNode.getMouse()
        return (pointer.x, pointer.y)

    def _mouse_pixels(self) -> tuple[float, float]:
        x, y = self._mouse()
        return (x * self.base.win.getXSize() * 0.5,
                y * self.base.win.getYSize() * 0.5)

    def _project(self, world) -> tuple[float, float] | None:
        """世界点 → 屏幕像素；点在相机后方时返回 ``None``。

        Panda3D 的投影有两处容易踩的坑，都在这里一次性处理掉：

        1. ``camLens.project`` 吃的是**镜头节点坐标系**下的点，不是 render 坐标。
           直接喂 render 坐标会得到"所有点都在镜头背后"的假象（实测：连相机
           正对着的目标点都会被判为不可见）。所以先 ``getRelativePoint``。
        2. ``project`` 的返回值会把"**画面左右之外**"也一并算作假 —— 实测偏出
           视锥 200 m 的点返回 ``False``，可它填出来的 xy 却是完全正确的投影值。
           这里不信它，改用"点在镜头前方多远"自己判：否则贴近窗口边缘的空闲
           端口会忽然变得点不到，而这种"偶尔点不上"最难查。
        """
        point = Point3(world[0], world[1], world[2])
        camera_position = self.base.cam.getPos(self.base.render)
        forward = self.base.cam.getQuat(self.base.render).getForward()
        if (point - camera_position).dot(forward) <= _MIN_CAMERA_DEPTH:
            return None

        screen = Point3()
        self.base.camLens.project(
            self.base.cam.getRelativePoint(self.base.render, point), screen)
        return (screen.x * self.base.win.getXSize() * 0.5,
                screen.y * self.base.win.getYSize() * 0.5)

    def mouse_ground(self) -> tuple[float, float, float] | None:
        """鼠标射线与地面（``y = GROUND_Y``）的交点。

        用 ``CollisionRay.setFromLens`` 从相机与归一化鼠标坐标构造射线，再与
        水平面解析求交 —— 不需要碰撞系统、不需要给地面加 collider。
        """
        origin, direction = self.mouse_ray()
        if origin is None or abs(direction[1]) < 1e-9:
            return None
        t = (style.GROUND_Y - origin[1]) / direction[1]
        if t < 0.0:
            return None
        return (origin[0] + direction[0] * t, style.GROUND_Y,
                origin[2] + direction[2] * t)

    def mouse_ray(self, x: float | None = None, y: float | None = None):
        """返回 ``(origin, direction)``（都在 render 坐标系里，direction 已归一化）。"""
        if not self._has_mouse() and (x is None or y is None):
            return (None, None)
        if x is None or y is None:
            x, y = self._mouse()

        ray = CollisionRay()
        try:
            ray.setFromLens(self.base.camNode, x, y)
        except (AttributeError, TypeError):
            return (None, None)

        matrix = self.base.camera.getMat(self.base.render)
        origin = matrix.xformPoint(ray.getOrigin())
        direction = matrix.xformVec(ray.getDirection())
        length = direction.length()
        if length < 1e-9:
            return (None, None)
        direction /= length
        return ((origin.x, origin.y, origin.z),
                (direction.x, direction.y, direction.z))

    def pick_port(self) -> PortKey | None:
        """屏幕上离鼠标最近的空闲端口（在吸附半径内）。"""
        if not self._has_mouse():
            return None
        mx, my = self._mouse_pixels()
        best, best_distance = None, PORT_PICK_RADIUS_PX
        for key in self.layout.free_ports():
            world = self.layout.world_port(*key)
            projected = self._project(world.position)
            if projected is None:
                continue
            distance = math.hypot(projected[0] - mx, projected[1] - my)
            if distance < best_distance:
                best, best_distance = key, distance
        return best

    def pick_piece(self) -> int | None:
        """屏幕上离鼠标最近的轨道件。

        判据取该件**所有抓手点**（端口 + 每条 route 的中点）与鼠标的最小屏幕
        距离，而不是取件中心 —— 40 m 长的直轨中心可能离鼠标很远，但用户明明把
        鼠标放在它的端部；反过来 20 m 直轨的中段若没有 route 中点当抓手，就会变成
        指上去没反应的死区。细节见 :meth:`PieceDef.route_midpoint_local`。
        """
        if not self._has_mouse():
            return None
        mx, my = self._mouse_pixels()
        best, best_distance = None, PIECE_PICK_RADIUS_PX
        for index, placed in self.layout.pieces.items():
            for local in self._local_handles(self.layout.definition(index)):
                world = placed.pose.compose(
                    Pose(local[0], local[1], local[2], 0.0)
                ).position
                projected = self._project(world)
                if projected is None:
                    continue
                distance = math.hypot(projected[0] - mx, projected[1] - my)
                if distance < best_distance:
                    best, best_distance = index, distance
        return best

    # ==================================================================== #
    # 放置计算
    # ==================================================================== #

    def _local_anchor(self, piece: PieceDef) -> tuple[float, float]:
        """件局部坐标系里的"视觉中心"：所有端口位置的平均值。

        自由放置时用它当鼠标的抓手，这样鼠标所在处就是件的中心，而不是
        某个碰巧叫 ``a`` 的端口。交叉件的这个点正好是两条轨的交点。
        """
        cached = self._anchors.get(piece.id)
        if cached is not None:
            return cached
        poses = [piece.port(pid).pose for pid in piece.port_ids]
        anchor = (
            sum(p.x for p in poses) / len(poses),
            sum(p.z for p in poses) / len(poses),
        )
        self._anchors[piece.id] = anchor
        return anchor

    def _local_handles(self, piece: PieceDef) -> tuple[tuple[float, float, float], ...]:
        """"鼠标可以抓住这个件的哪些点"（件局部坐标）：端口 + route 中点。

        按件 id 缓存 —— 这是纯目录数据，和件摆在哪、摆了几节都无关。
        """
        cached = self._handles.get(piece.id)
        if cached is None:
            points = [
                (piece.port(pid).pose.x, piece.port(pid).pose.y, piece.port(pid).pose.z)
                for pid in piece.port_ids
            ]
            points.extend(
                piece.route_midpoint_local(route.index) for route in piece.routes
            )
            cached = tuple(points)
            self._handles[piece.id] = cached
        return cached

    def _attach_port(self, piece: PieceDef) -> str:
        ids = piece.port_ids
        return ids[self.attach_slot % len(ids)]

    def compute_placement(self) -> Placement | None:
        """由当前鼠标位置算出"将要放到哪"。视角模式下没有待放置的件。"""
        if not self.building:
            return None
        piece = self.current_piece

        if self.hover_port is not None:
            my_port = self._attach_port(piece)
            world_target = self.layout.world_port(*self.hover_port)
            local_q = piece.port(my_port).pose
            # 与 Layout.attach 用的公式逐字一致 —— 预览与实际放置不可能不一致
            pose = world_target.flipped().compose(local_q.inverse())
            return Placement(piece.id, pose, my_port, self.hover_port)

        if self.layout.is_empty:
            ground = self.mouse_ground()
            if ground is None:
                return None
            anchor = self._local_anchor(piece)
            base_pose = Pose(ground[0], 0.0, ground[2], self.free_heading)
            pose = base_pose.compose(Pose(-anchor[0], 0.0, -anchor[1], 0.0))
            return Placement(piece.id, pose)

        # 非空布局 + 没吸到端口：Layout 只支持单张连通图，所以这里不能放
        ground = self.mouse_ground()
        if ground is None:
            return None
        anchor = self._local_anchor(piece)
        base_pose = Pose(ground[0], 0.0, ground[2], self.free_heading)
        pose = base_pose.compose(Pose(-anchor[0], 0.0, -anchor[1], 0.0))
        return Placement(piece.id, pose, None, None, allowed=False,
                         note="新件必须接到绿色端口上（把鼠标移近某个端口）")

    # ==================================================================== #
    # 修改布局
    # ==================================================================== #

    def place(self) -> int | None:
        placement = self.placement
        if placement is None:
            return None
        if not placement.allowed:
            self.notify(placement.note)
            return None

        self.push_undo()
        try:
            if placement.target is None:
                index = self.layout.add_root(placement.def_id, placement.pose)
                self.notify(f"放下第一节：{self.catalog[placement.def_id].name}")
            else:
                index = self.layout.attach(placement.def_id, placement.my_port,
                                           placement.target)
                gap = self._placement_error(index, placement)
                self.notify(f"接上 {self.catalog[placement.def_id].name}"
                            f"（接缝误差 {gap * 1000:.4f} mm）")
        except LayoutError as exc:
            self._undo.pop()
            self.notify(f"放置失败：{exc}")
            return None

        self.view.sync()
        return index

    def _placement_error(self, index: int, placement: Placement) -> float:
        """预览位姿与实际放置位姿之差（应该恒为 0，用来早期发现公式漂移）。"""
        actual = self.layout.piece(index).pose
        return actual.distance_to(placement.pose)

    def delete_hovered(self) -> None:
        index = self.hover_piece
        if index is None:
            self.notify("先把鼠标移到要删除的轨道上")
            return
        self.push_undo()
        try:
            removed = self.layout.remove_piece(index)
        except LayoutError as exc:
            self._undo.pop()
            self.notify(f"删除失败：{exc}")
            return
        # 要删的这一节上正压着列车的话，先把列车摘掉 —— 否则下一帧重走线可能
        # 摆不下列车（闭环断成开链、车被锚到更短的路上），直接崩掉进程。
        train_removed = False
        if self.train_view is not None and self.train_view.occupies_piece(index):
            self.train_view.destroy()
            self.train_view = None
            train_removed = True
        self.hover_piece = None
        self.view.sync()
        note = "；列车已下轨" if train_removed else ""
        self.notify(f"删除了 {len(removed)} 节轨道{note}"
                    f"（其余原地不动；放一节或按 C 闭合缺口）")

    def best_seam(self):
        """缺口最小的那一对空闲端口，返回 ``((a, b), (位置差, 航向差))``。

        HUD 每帧都要问"现在能不能合拢"，而枚举是 O(空闲端口数²)。所以按
        **布局版本**缓存：布局只会因增删件、增删连接而变，用这两个计数当版本键
        就足够，且撤销 / 读档（换了 layout 对象）也会自然失效（比了 ``id``）。

        端口对的筛选规则在 :meth:`core.track.layout.Layout.closest_free_pair`
        —— 那里是纯几何，能 headless 单测。
        """
        key = (id(self.layout), len(self.layout.pieces),
               len(self.layout.connections))
        if self._seam_key == key:
            return self._seam_cache

        self._seam_key = key
        self._seam_cache = self.layout.closest_free_pair()
        return self._seam_cache

    def seam_is_aligned(self) -> bool:
        """当前是否有两个空闲端口已经对齐到可以合拢。"""
        seam = self.best_seam()
        if seam is None:
            return False
        _, (distance, heading_gap) = seam
        return distance <= SEAM_POSITION_TOL and heading_gap <= SEAM_HEADING_TOL

    def auto_close_hint(self) -> str:
        """一句话说明"离闭环还差什么"（HUD 用）；没什么可说时返回空串。

        这里刻意做**心算**而不是真的去试接一遍：每帧跑一次
        ``grow_to_close``（最多 32 次 attach + 快照）太重，而这个问题有一个精确解
        —— 还差多少转角除以手上这一节的转角就是要补的节数。接缝的 ``heading_gap``
        正好就是"还差的转角"（7 节 45° 剩 45°、8 节 22.5° 剩 180°，都对得上）。

        改半径就另说了（转角对得上、位置对不上），那种情况交给 `C` 去验，这里
        只报"大概还差几节"。
        """
        if not self.layout.pieces:
            return ""
        seam = self.best_seam()
        if seam is None or self.seam_is_aligned():
            return ""
        _, (distance, heading_gap) = seam
        routes = self.current_piece.routes
        if not routes or abs(routes[0].dtheta) < 1e-12:
            return "当前件是直轨，接不出环 —— 按 . 切到「曲线」"
        piece_turn = abs(math.degrees(routes[0].dtheta))
        need = round(math.degrees(heading_gap) / piece_turn)
        if need <= 0:
            return f"方向对不上（缺口 {distance:.2f} m）—— 半径选错了？"
        return f"再补 {need} 节「{self.current_piece.name}」→ 按 C 自动补"

    def close_loop(self) -> bool:
        """合拢闭环。已经对齐就直接接上；没对齐就用**当前件自动补齐**再接上。

        为什么需要它：``core.track.path.trace`` 只认**拓扑**上能绕回来的环（它靠
        ``Layout.peer_of`` 走）。最后一节搭上之后，接缝两端在几何上已经严丝合缝，
        但连接表里还没有那一条边，所以走线到缝口就停了。按 C 就是补上这条边。

        而"还差几节"是用户没法心算的：22.5° 的弯轨要 16 节才绕满一圈，接 8 节
        正好停在直径的另一头（差 80 m）。所以没对齐时不是报个数字让用户自己数，
        而是直接按当前件补下去 —— 补不上就原样回滚（``grow_to_close`` 是原子的），
        再把缺口数字报出来，让"差多少"永远是可见的量。
        """
        free = self.layout.free_ports()
        if len(free) < 2:
            self.notify("空闲端口不足两个，没有可合拢的接缝")
            return False

        seam = self.best_seam()
        if seam is not None:
            (a, b), (distance, heading_gap) = seam
            if distance <= SEAM_POSITION_TOL and heading_gap <= SEAM_HEADING_TOL:
                self.push_undo()
                self.layout.connect(a, b)
                self.view.sync()
                self.notify(f"已合拢 {a[0]}/{a[1]} → {b[0]}/{b[1]}"
                            f"（误差 {distance * 1000:.6f} mm）"
                            f"{self.spare_ports_note()}")
                return True

        # 没对齐 —— 补线补到首尾能合上。
        #
        # 顺序是"先用手上这一件，合不上再替你换"：``grow_to_close`` 先用第一件
        # （= 用户当前选中的那一件）单独试一次，只有它接不通时才在整串候选里挑。
        # 候选 = 手上这一件 + 件库里所有"单通路的直轨 / 弯轨"
        # （见 Catalog.closure_helpers）。为什么要留这个后手：缺口要的是**几何**，
        # 而用户手上恰好拿着不合适的那一节是常态 —— 差一节 22.5° 的弯、手里却是
        # 40 m 直轨时，旧行为只会回一句"接了 32 节还是合不上"，用户看到的就是
        # "没有合适长度的铁轨对接"。这时该由机器换一件。
        piece = self.current_piece
        candidates = self.layout.catalog.closure_helpers(
            piece.id if piece is not None else None
        )
        before = set(self.layout.pieces)
        snapshot = self._undo_snapshot()
        try:
            added = self.layout.grow_to_close(candidates)
        except LayoutError as exc:
            self.notify(f"自动合拢失败：{exc}")
            return False

        # 只有真的补成了才记这笔撤销，失败时不留痕（rollback 由 grow_to_close 负责）
        self.push_undo(snapshot)
        self.view.sync()
        used = [self.layout.pieces[i].def_id
                for i in sorted(set(self.layout.pieces) - before)]
        self.notify(self._closed_message(self._used_piece_label(used), added))
        return True

    def _used_piece_label(self, used: list[str]) -> str:
        """自动补线到底补了些什么 —— 提示里那句「自动补上 N 节「…」」的括号内容。

        候选是一串时补出来的可能不止一种件。只有一种时就把名字写出来（读起来与
        从前一样）；混了多种时**只报种类数**，不把名字一个个列出来 —— 提示是屏幕
        正下方**居中**的那一条，宽度必须有个上限（``tests/test_app_hud.py`` 是按
        最坏情况量版面的，撑宽了会压到左下/右下的面板上）。
        """
        names = [
            self.layout.catalog[def_id].name for def_id in dict.fromkeys(used)
        ]
        if not names:
            return "轨道件"
        if len(names) == 1:
            return names[0]
        return f"{len(names)} 种轨道件"

    def spare_ports_note(self) -> str:
        """合拢之后还有几个空闲端口留在**环外**（支线），没有就返回空串。

        这件事必须说出来：列车只跑那个环，支线上的岔尖、尽头线它上不去。用户按了
        C 看到"闭环成立"，很容易以为整张图都通电了 —— 而一条成环的干线旁边挂着
        一条没接上的支线，恰恰是站场里最常见的样子。**不崩**是对的下限，**认得对**
        才是这里要交的东西。
        """
        spare = len(self.layout.free_ports())
        return f"；另有 {spare} 个空闲端口没进环（支线）" if spare else ""

    def _closed_message(self, piece_name: str, added: int) -> str:
        """合拢之后那句提示：补了几节、环有多大、有没有东西留在环外。"""
        closure = self.view.closure
        spare = self.spare_ports_note()
        if not added:
            return "已合拢闭环" + spare
        if closure is not None and closure.closed:
            return (f"自动补上 {added} 节「{piece_name}」，闭环成立"
                    f"（{closure.visit_count} 段 / {closure.total_length:.2f} m）"
                    f"{spare}")
        if closure is not None:
            return (f"自动补上 {added} 节「{piece_name}」，但还没闭环"
                    f"（缺口 {closure.gap_distance:.3f} m）")
        return f"自动补上 {added} 节「{piece_name}」"

    def toggle_switch_under_cursor(self) -> None:
        index = self.hover_piece
        if index is None:
            self.notify("先把鼠标移到道岔上")
            return
        if not self.layout.definition(index).is_switch:
            self.notify("这一节不是道岔")
            return
        self.push_undo()
        route_index = self.layout.toggle_switch(index)
        self.view.sync()
        self.notify(f"道岔 #{index} 已扳到 route #{route_index}")

    def clear(self) -> None:
        if self.layout.is_empty and not self.user_scenery:
            return
        self.push_undo()
        self.layout.clear()
        self.user_scenery = scenery_mod.Scenery()
        self.hover_piece = None
        self.hover_port = None
        self.hover_scenery = None
        self.view.sync()
        self._rebuild_scenery()
        self._clear_scenery_highlight()
        self.notify("已清空")

    # ==================================================================== #
    # 撤销 / 存档
    # ==================================================================== #

    def push_undo(self, payload: dict | None = None) -> None:
        """把"改动前"的布局压进撤销栈。

        ``payload`` 让调用方传入一个更早抓的快照 —— 用于**先试后记**的操作
        （例如自动合拢：先在副本上推演，成功了才记这笔撤销）。不传就现抓一份。
        """
        # _undo_snapshot 每次构造全新的 dict，因此这是一份真正的快照（无别名共享）
        self._undo.append(self._undo_snapshot() if payload is None else payload)
        if len(self._undo) > UNDO_LIMIT:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self) -> None:
        if not self._undo:
            self.notify("没有可撤销的操作")
            return
        self._redo.append(self._undo_snapshot())
        self._restore(self._undo.pop())
        self.notify("已撤销")

    def redo(self) -> None:
        if not self._redo:
            self.notify("没有可重做的操作")
            return
        self._undo.append(self._undo_snapshot())
        self._restore(self._redo.pop())
        self.notify("已重做")

    def _undo_snapshot(self) -> dict:
        """一份撤销快照：轨道拓扑 + 用户手工摆的布景（一起撤、一起重做）。"""
        payload = self.layout.to_dict()
        payload["_user_scenery"] = scenery_mod.scenery_to_dict(self.user_scenery)
        return payload

    def _restore(self, payload: dict) -> None:
        self.layout = Layout.from_dict(payload, self.catalog)
        self.user_scenery = scenery_mod.scenery_from_dict(
            payload.get("_user_scenery"))
        self.view.layout = self.layout
        self.hover_piece = None
        self.hover_port = None
        self.hover_scenery = None
        self.view.sync()
        self._rebuild_scenery()
        self._clear_scenery_highlight()
        self._resync_train()

    def save(self) -> bool:
        payload = self.layout.to_dict()
        payload["saved_from"] = "track_editor"
        payload.update(self.save_extra)
        if self.user_scenery:
            payload["user_scenery"] = scenery_mod.scenery_to_dict(
                self.user_scenery)
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with self.save_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            self.notify(f"保存失败：{exc}")
            return False
        self.notify(f"已保存 {len(self.layout)} 节到 {self.save_path}"
                    "（下次 run.bat --open 这个文件继续编辑）")
        return True

    # ---------------------------------------------------------------- 存档命名

    def begin_save_as(self) -> None:
        """Ctrl+S：进入"填写文件名"模式，把布局存到 ``saves/<名字>.json``。

        之后键盘的字母 / 数字键喂进一个输入框（编辑器自己的快捷键暂时让位），
        回车 = 保存，退格删字，Esc 取消。不输入名字直接回车，就存回当前的
        ``save_path``（和原来一样"固定文件名"的快速保存）。
        """
        self._begin_file_prompt(PROMPT_SAVE)

    def begin_open_as(self) -> None:
        """Ctrl+O：进入"填写文件名"模式，从 ``saves/<名字>.json`` 读档。

        和 Ctrl+S 共用同一个输入框，提示里会把 ``saves`` 目录下**已有哪些存档**
        列出来。空名字直接回车 = 读回当前 ``save_path``（快速读档）；写了名字但
        文件不存在则留在输入框里报错，不会去动当前布局。
        """
        self._begin_file_prompt(PROMPT_LOAD)

    def _begin_file_prompt(self, purpose: str) -> None:
        """打开文件名输入框。``purpose`` 决定回车之后是保存还是读档。"""
        if self._prompt is not None:
            return
        self._prompt = purpose
        self._prompt_buffer = ""
        self._prompt_error = ""
        self._held.clear()
        # 暂时让编辑器的快捷键全部闭嘴，只留文件名输入
        for event in self.bound_key_names():
            self.base.ignore(event)
        for char in _SAVE_NAME_CHARS:
            self.base.accept(char, self._name_append, [char])
        self.base.accept("backspace", self._name_backspace)
        self.base.accept("enter", self._confirm_file_prompt)
        self.base.accept("escape", self._cancel_file_prompt)

    def _exit_file_prompt(self) -> None:
        """退出命名模式，恢复全套编辑器快捷键。"""
        self._prompt = None
        for char in _SAVE_NAME_CHARS:
            self.base.ignore(char)
        self.base.ignore("backspace")
        self.base.ignore("enter")
        self.base.ignore("escape")
        self.bind()

    def _name_append(self, char: str) -> None:
        if len(self._prompt_buffer) < _SAVE_NAME_MAX:
            self._prompt_buffer += char
            self._prompt_error = ""

    def _name_backspace(self) -> None:
        self._prompt_buffer = self._prompt_buffer[:-1]
        self._prompt_error = ""

    def _cancel_file_prompt(self) -> None:
        self._exit_file_prompt()
        self.notify("已取消")

    def _prompt_target(self) -> Path:
        """输入框里的名字解析成的目标路径；空名字回落到当前 ``save_path``。"""
        name = self._prompt_buffer.strip()
        if not name:
            return self.save_path
        if not name.lower().endswith(".json"):
            name += ".json"
        return self._save_dir / name

    def _prompt_target_label(self) -> str:
        """提示条里回显的目标：目录内只写文件名，超长就掐尾加 "…"。

        必须**有界**：提示条是一行居中文字，把一长串路径照抄上去会顶出屏幕
        （输入框本身已限 40 字符，回显再带上目录就更长了）。
        """
        target = self._prompt_target()
        label = target.name if target.parent == self._save_dir else str(target)
        if len(label) > _PROMPT_ECHO_MAX:
            label = label[: _PROMPT_ECHO_MAX - 1] + "…"
        return label

    def _prompt_files(self) -> str:
        """``saves`` 目录里已有存档的名字，拼成一行（总量受字符预算限制）。

        名字按字典序，超出 :data:`_PROMPT_LIST_CHARS` 就截断并以 ``…`` 收尾 ——
        提示条一行字太长，面板会被挤出屏幕。
        """
        try:
            names = sorted(path.stem for path in self._save_dir.glob("*.json"))
        except OSError:
            return ""
        shown: list[str] = []
        used = 0
        for name in names:
            if used + len(name) > _PROMPT_LIST_CHARS:
                shown.append("…")
                break
            shown.append(name)
            used += len(name) + 1
        return " ".join(shown)

    def _confirm_file_prompt(self) -> None:
        """回车：按当前用途保存 / 读档。名字非法或文件不存在就留在输入框报错。"""
        name = self._prompt_buffer.strip()
        if name:
            if name in (".", "..") or any(ch in name for ch in '/\\:*?"<>|'):
                self._prompt_error = "　（文件名含非法字符）"
                return
            target = self._prompt_target()
            if self._prompt == PROMPT_LOAD and not target.exists():
                self._prompt_error = "　（这个文件不存在）"
                return
            self.save_path = target
        purpose = self._prompt
        self._exit_file_prompt()
        if purpose == PROMPT_LOAD:
            self.load()
        else:
            self.save()

    def _prompt_text(self) -> str:
        """底部提示条：输入框 + 目标；读档时另起一行列出已有存档。

        刻意写得**短**：提示条是"底部居中"的一行字，宽度受中间那一列限制，
        太长就会从左右两块面板底下穿过去。所以目标路径只回显截断后的文件名
        （见 :meth:`_prompt_target_label`），名单另起一行并另有字符预算。
        """
        buffer = self._prompt_buffer
        label = self._prompt_target_label()
        if self._prompt == PROMPT_LOAD:
            head = (f"读档  {buffer}▏　回车 读 {label}"
                    f"　Esc 取消{self._prompt_error}")
            listing = self._prompt_files()
            tail = f"已有：{listing}" if listing else "（这个目录里还没有存档）"
            return f"{head}\n{tail}"
        return (f"保存  {buffer}▏　回车 存到 {label}"
                f"　Esc 取消{self._prompt_error}")

    def load(self) -> bool:
        try:
            with self.save_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            loaded = Layout.from_dict(payload, self.catalog)
            user_scenery = scenery_mod.scenery_from_dict(
                payload.get("user_scenery"))
        except (OSError, ValueError, KeyError) as exc:
            self.notify(f"读档失败：{exc}")
            return False
        self.push_undo()
        self.layout = loaded
        self.user_scenery = user_scenery
        self.view.layout = self.layout
        self.hover_piece = None
        self.hover_scenery = None
        self.view.sync()
        self._rebuild_scenery()
        self._clear_scenery_highlight()
        self.camera.frame(self.view.bounds())
        self.notify(f"已载入 {len(self.layout)} 节")
        return True

    # ==================================================================== #
    # 显示开关与取景
    # ==================================================================== #

    def toggle_grid(self) -> None:
        node = self._grid_node()
        if node is None:
            self.notify("场景里没有网格节点")
            return
        if node.isHidden():
            node.show()
            self.notify("网格：显示")
        else:
            node.hide()
            self.notify("网格：隐藏")

    def _grid_node(self):
        node = self.base.render.find("**/ground/grid")
        return None if node.isEmpty() else node

    def toggle_ports(self) -> None:
        self.view.set_show_ports(not self.view.show_ports)
        self.notify(f"端口标记：{'显示' if self.view.show_ports else '隐藏'}")

    def toggle_loop(self) -> None:
        self.view.set_show_loop(not self.view.show_loop)
        self.notify(f"闭环显示：{'开' if self.view.show_loop else '关'}")

    def toggle_help(self) -> None:
        if self.hud is None:
            return
        self.hud.set_visible(not self.hud.visible)

    def frame_layout(self) -> None:
        self.camera.frame(self.view.bounds())

    # ==================================================================== #
    # 列车：上轨、换编组、手柄调速
    # ==================================================================== #

    @property
    def train_spec(self) -> TrainSpec | None:
        """当前选中的编组（还没召唤时也有 —— 召唤用的是它）。"""
        if not len(self.trains):
            return None
        return self.trains[self.trains.ids[self._train_index % len(self.trains)]]

    def spawn_train(self, step: int = 1, train_id: str | None = None
                    ) -> TrainView | None:
        """召唤一列列车上轨；再按一次就换下一列编组。

        "换一列"是**就地**换的（先把旧车摘掉），所以连续按不会在场景里堆车。
        新车上线时保留当前手柄档位 —— 换车不该把司机推好的档位清零。

        ``train_id`` 用来直接点名一列编组（启动场景时按预设上那列车）：点名不影响
        "再按 N 换下一列"的顺序，只是把计数器拨到那一列上。

        门槛是**连通**而不是**闭合**：只要轨道是一整条连通、能走行的线，就允许上车
        —— 闭环绕圈、开链跑到尽头自动停（"线还没铺完先开一段"、以及立交场景扳
        道岔改走疏解线，都靠它）。只有当轨道被拆散成互不相连的几段时才拒绝。
        """
        if not len(self.trains):
            self.notify("列车目录是空的")
            return None
        components = self.layout.component_count()
        if components == 0:
            self.notify("场地还是空的 —— 先在网格上放下一节轨道")
            return None
        if components > 1:
            self.notify("轨道断成了几段 —— 先把它们接起来再上车")
            return None
        if train_id is not None and train_id in self.trains:
            self._train_index = self.trains.ids.index(train_id)
        else:
            self._train_index = (self._train_index + step) % len(self.trains)
        spec = self.train_spec

        if self.train_view is not None:
            self.train_view.destroy()
        view = TrainView(spec, self.base.render,
                         train=Train(spec, drive_boost=DRIVE_BOOST))
        view.set_handle(self.train_handle)
        self.train_view = view
        view.sync(self.view.path, self.layout)
        self.audio.start()

        # 加载列车 = 要开车看车了，左键自动从"放轨道"切到"拖视角"，
        # 免得鼠标一动就把轨道甩进场景（用户按 V 可随时切回来）。
        self.mode = MODE_VIEW
        self._hide_ghost()

        if view.placement_error is not None:
            self.notify(f"{spec.name} —— {view.placement_error}")
        else:
            self.notify(f"已上线：{spec.name}"
                        f"（{spec.car_count} 节 / {view.consist_length:.1f} m）"
                        "　↑↓ 推手柄　V 切回放置")
        return view

    def dismiss_train(self) -> None:
        """把列车从场景里摘掉。"""
        if self.train_view is None:
            self.notify("现在没有列车在线上")
            return
        self.train_view.destroy()
        self.train_view = None
        self.audio.stop()
        self.notify("列车已下轨")

    def push_train_handle(self, step: float = HANDLE_STEP) -> float | None:
        """推一档手柄（正 = 牵引，负 = 制动），返回新档位。"""
        if self.train_view is None:
            self.notify("先按 N 召唤一列列车")
            return None
        self.train_handle = self.train_view.nudge_handle(step)
        self.notify(f"手柄　{self.handle_label()}")
        return self.train_handle

    def release_train_handle(self) -> None:
        """手柄回中位（惰行）。"""
        if self.train_view is None:
            return
        self.train_view.set_handle(0.0)
        self.train_handle = 0.0
        self.notify("手柄回中位 —— 惰行")

    def emergency_stop_train(self) -> None:
        """紧急制动：把手柄拉到"紧急"那一档（制动力比常用制动大得多）。

        标注要短：HUD 上这一行与"牵引 100%"共用同一个位置，``tests/test_app_hud.py``
        是按最坏情况量版面的，写长了会把那一栏撑宽。
        """
        if self.train_view is None:
            return
        self.train_view.train.emergency_stop()
        self.train_handle = -1.0
        self.notify("紧急制动 · 已拉死（回中位或给牵引解除）")

    def reverse_train_direction(self) -> None:
        """换向：先把列车快速刹停，再反向满牵引加速。

        刹停用的是紧急制动力级（所以停得很快），反向起步与正常牵引是**同一个**
        加速度 —— 只是方向翻了个。换向过程中用户可以随时按 ↑↓ / 空格接管。
        """
        if self.train_view is None:
            self.notify("先按 N 召唤一列列车")
            return
        self.train_view.train.request_reverse()
        # 换向完成后会自动给满牵引，手柄位跟着设满，UI 才不会"看着惰行其实在猛跑"。
        self.train_handle = 1.0
        self.notify("换向：快速刹停 → 反向加速")

    def horn_train(self) -> None:
        """鸣笛：只有列车在线时才响。"""
        if self.train_view is None:
            self.notify("先按 N 召唤一列列车")
            return
        self.audio.horn()

    def handle_label(self) -> str:
        """手柄的一句话说明（HUD 与提示共用）。"""
        if self.train_view is not None and self.train_view.train.is_emergency:
            return "紧急制动"
        if self.train_handle > 0.0:
            return f"牵引 {self.train_handle * 100:.0f}%"
        if self.train_handle < 0.0:
            return f"制动 {-self.train_handle * 100:.0f}%"
        return "惰行"

    def _advance_train(self, dt: float) -> None:
        """推进一帧的列车运行。

        路径来自 :attr:`view` 上一次 ``sync`` 的结果 —— 布局一变（放件 / 删件 /
        撤销 / 读档），``LayoutView`` 就重跑闭环检测。

        列车跑**一条连通的线**：闭环绕圈、开链跑到尽头自动停。扳道岔改线（立交
        场景把主线截成疏解线）仍是一整条连通的线，车会重新锚定到新路径、继续开；
        只有当轨道被真正拆散成互不相连的几段时才把车请下轨。
        """
        if self.train_view is None:
            return
        components = self.layout.component_count()
        if components != 1:
            self.train_view.destroy()
            self.train_view = None
            self.audio.stop()
            if components == 0:
                self.notify("轨道清空了，列车下轨")
            else:
                self.notify("轨道断成了几段，列车下轨 —— 接起来后再按 N")
            return
        self.train_view.advance(dt, self.view.path, self.layout)
        self.audio.update(self.train_view.speed_kmh(), self.train_handle)

    def _resync_train(self) -> None:
        """布局被整体换掉（撤销 / 读档）后让列车重新上线。"""
        if self.train_view is not None:
            self.train_view.sync(self.view.path, self.layout)

    # ==================================================================== #
    # 提示信息
    # ==================================================================== #

    def notify(self, message: str, frames: int = 190) -> None:
        """在屏幕底部显示一条会自行消失的提示。"""
        self._toast = message
        self._toast_frames = frames

    # ==================================================================== #
    # 每帧更新
    # ==================================================================== #

    def update(self, task):
        self.tick()
        return task.cont

    def tick(self) -> None:
        """推进一帧的编辑器逻辑。

        和 :meth:`update` 分开，是为了让脚本 / 测试能在没有 Panda3D 任务系统的
        情况下驱动编辑器 —— ``update`` 只是"tick 一次然后把 task.cont 还回去"。
        """
        dt = min(ClockObject.getGlobalClock().getDt(), 0.1)
        self._apply_drag()
        self._apply_held_keys(dt)
        self._refresh_hover()
        self._refresh_ghost()
        self._advance_train(dt)
        self._refresh_hud()
        if self._toast_frames > 0:
            self._toast_frames -= 1
            if self._toast_frames == 0:
                self._toast = ""

    # ---------------------------------------------------------------- 悬停

    def _refresh_hover(self) -> None:
        if not self._has_mouse():
            self.hover_piece = None
            self.hover_port = None
            self.hover_scenery = None
        elif self.placing_scenery:
            self.hover_piece = None
            self.hover_port = None
            self.hover_scenery = self._pick_scenery()
        else:
            self.hover_scenery = None
            self.hover_port = self.pick_port()
            self.hover_piece = self.pick_piece()
        self.view.highlight(self.hover_piece)
        self._refresh_scenery_highlight()

    # ---------------------------------------------------------------- 幽灵

    def _hide_ghost(self) -> None:
        """把预览件收起来（切到视角模式时用）。"""
        self.placement = None
        if self._ghost is not None:
            self._ghost.hide()

    def _refresh_ghost(self) -> None:
        # 布景模式走它自己的幽灵（摆在鼠标指的地面上，不吸附端口）。
        if self.placing_scenery:
            self.placement = None
            self._hide_ghost()
            self._refresh_scenery_ghost()
            self.view.set_hover_port(None)
            return
        self._hide_scenery_ghost()

        self.placement = self.compute_placement()
        placement = self.placement

        if placement is None:
            self._hide_ghost()
            self.view.set_hover_port(self.hover_port)
            return

        ground_key = round(style.GROUND_Y - placement.pose.y, 3)
        key = (placement.def_id, ground_key)
        if self._ghost_key != key:
            if self._ghost is not None:
                self._ghost.removeNode()
            self._ghost = self.view.piece_instance(placement.def_id, ground_key,
                                                   parent=self._ghost_root)
            self._ghost.setName("ghost")
            self._ghost_key = key

        if self._ghost is not None:
            apply_pose(self._ghost, placement.pose)
            tint(self._ghost,
                 style.GHOST_OK_TINT if placement.allowed
                 else style.GHOST_BLOCKED_TINT)
            self._ghost.show()
        self.view.set_hover_port(self.hover_port)

    # ---------------------------------------------------------------- 输入

    def _apply_drag(self) -> None:
        if not self._has_mouse():
            return
        current = self._mouse()
        if self._last_mouse is not None:
            dx = current[0] - self._last_mouse[0]
            dy = current[1] - self._last_mouse[1]
            if self._drag_mode == "orbit" and (dx or dy):
                # 像素位移换算：widget 的坐标是归一化的 -1..1，乘窗口尺寸得到像素
                self.camera.orbit(dx * self.base.win.getXSize() * 0.5,
                                  dy * self.base.win.getYSize() * 0.5)
            elif self._drag_mode == "pan" and (dx or dy):
                self.camera.pan(dx * self.base.win.getXSize() * 0.5,
                                dy * self.base.win.getYSize() * 0.5)
            if self._press_mouse is not None:
                self._press_moved = max(
                    self._press_moved,
                    math.hypot((current[0] - self._press_mouse[0])
                               * self.base.win.getXSize() * 0.5,
                               (current[1] - self._press_mouse[1])
                               * self.base.win.getYSize() * 0.5),
                )
        self._last_mouse = current

    def _apply_held_keys(self, dt: float) -> None:
        if self._prompt is not None:
            return                      # 命名模式下相机不该被 WASDQE 拖动
        if not self._held:
            return
        distance = (1 if "w" in self._held else 0) - (1 if "s" in self._held else 0)
        strafe = (1 if "d" in self._held else 0) - (1 if "a" in self._held else 0)
        if distance or strafe:
            speed = self.camera.distance * 0.6 * dt
            self.camera.move_ground(distance * speed, strafe * speed)
        spin = (1 if "e" in self._held else 0) - (1 if "q" in self._held else 0)
        if spin:
            self.camera.spin(spin * 1.0 * dt)

    # ---------------------------------------------------------------- HUD

    def _refresh_hud(self) -> None:
        if self.hud is None:
            return

        # 填文件名时是**模态**的：底部那两块（帮助 / 列车）让位，好让提示条
        # 这一行长得下 —— 提示条是"底部居中"的一行字，宽度受中间那一列限制。
        prompting = self._prompt is not None

        if self.placing_scenery:
            placed = sum(1 for _ in scenery_mod.placeable_items(self.user_scenery))
            lines = [
                f"轨道编辑器 · {self._mode_label()}",
                f"当前布景  {self.current_scenery_label()}   "
                f"[{self._scenery_kind_index + 1}/{len(self.scenery_items)}]",
                f"分类  {self.scenery_menu()}",
                f"物件  {self.scenery_items_menu()}",
                f"朝向  {math.degrees(self._scenery_heading) % 360:.0f}°   已摆放  "
                f"{placed} 件",
                f"左键 放下   X 删除   [ ] 换件   , . 换分类   1-9 直选   R 旋转",
            ]
        else:
            pieces = self.pieces_in_category
            piece = self.current_piece
            attach = self._attach_port(piece)
            lines = [
                f"轨道编辑器 · {self._mode_label()}",
                f"当前件  {piece.name}   [{self._category_label()} "
                f"{self._piece_index + 1}/{len(pieces)}]",
                f"分类  {self.category_menu()}",
                f"接驳端口  {attach}"
                + ("（吸附中）" if self.placement and self.placement.snapped else ""),
                f"轨道 {len(self.layout)} 节   总长 {self.layout.total_length():.1f} m"
                f"   三角形 {self.view.triangle_count():,}",
            ]

        closure = self.view.closure
        if closure is None or self.layout.is_empty:
            loop = "空场地：在网格上点左键放下第一节"
        elif closure.closed:
            loop = (f"闭环成立  {closure.visit_count} 段 / "
                    f"{closure.total_length:.2f} m   接缝误差 "
                    f"{closure.gap_distance * 1000:.6f} mm")
        else:
            hint = ""
            if self.seam_is_aligned():
                hint = "\n接缝已对齐 —— 按 C 合拢"
            else:
                tip = self.auto_close_hint()
                if tip:
                    hint = f"\n{tip}"
            loop = (f"未闭环  已铺 {closure.visit_count} 段 / "
                    f"{closure.total_length:.2f} m\n"
                    f"缺口 {closure.gap_distance:.4f} m   "
                    f"航向差 {math.degrees(closure.gap_heading):.3f}°\n"
                    f"原因：{closure.reason}{hint}")
        self.hud.set_text("loop", loop)

        cursor = "—"
        if self.placing_scenery and self.hover_scenery is not None:
            field, index = self.hover_scenery
            items = getattr(self.user_scenery, field)
            if 0 <= index < len(items):
                cursor = items[index].footprint().label
        elif self.hover_piece is not None:
            definition = self.layout.definition(self.hover_piece)
            extra = ""
            if definition.is_switch:
                extra = f"  道岔档位 route #{self.layout.switch_of(self.hover_piece)}"
            cursor = f"#{self.hover_piece} {definition.name}{extra}"
        lines.append("")
        lines.append(f"鼠标下：{cursor}")
        self.hud.set_text("status", "\n".join(lines))

        self.hud.set_text("help", "" if prompting else HELP_TEXT)
        self.hud.set_text("train", "" if prompting else self._train_hud_text())
        self.hud.set_text("toast", self._prompt_text() if prompting else self._toast)

    def _train_hud_text(self) -> str:
        """右下角那块"列车"面板的文本。空字符串 = 整块隐藏。"""
        view = self.train_view
        if view is None:
            return ""
        spec = view.spec
        lines = [
            f"列车  {spec.name}",
            f"{view.car_count} 节 / {view.consist_length:.1f} m"
            f"   三角形 {view.triangle_count():,}",
        ]
        reason = view.placement_error
        if reason is not None:
            lines.append(f"未上线：{reason}")
            lines.append("（铺一段更长的轨道再按 N）")
            return "\n".join(lines)

        state = view.state
        mode = "　倒行" if view.train.state.direction < 0 else ""
        lines.append(f"速度  {state.speed_kmh:5.1f} km/h"
                     f"   （{state.v:.2f} m/s）{mode}")
        lines.append(f"手柄  {self.handle_label()}")
        lines.append(f"里程  {state.distance:8.1f} m"
                     f"   运行 {state.elapsed:6.1f} s")
        return "\n".join(lines)

    # ==================================================================== #
    # 键鼠绑定
    # ==================================================================== #

    def mouse_actions(self) -> dict:
        """鼠标 / 滚轮事件名 → 处理方式。"""
        return {
            "mouse1": self._on_press,
            "mouse1-up": self._on_release,
            "mouse2": lambda: self._on_drag_start("orbit"),
            "mouse3": lambda: self._on_drag_start("pan"),
            "mouse2-up": self._on_drag_end,
            "mouse3-up": self._on_drag_end,
            "wheel_up": lambda: self._on_zoom(1.0),
            "wheel_down": lambda: self._on_zoom(-1.0),
        }

    def key_actions(self) -> dict:
        """键盘事件名 → 动作。

        事件名必须是 Panda3D **真的会发出的名字**，否则那个键就是一个哑键：按下去
        毫无反应，而且不会报任何错。有两个反直觉的地方记在这里：

        * ``[`` ``]`` ``,`` ``.`` 这类符号键的事件名**就是符号本身**，没有
          ``bracket_left`` / ``comma`` / ``period`` 这种别名（它们曾经是哑键）；
        * 方向键叫 ``arrow_up`` / ``arrow_down`` / ``arrow_left`` / ``arrow_right``，
          不叫 ``up`` / ``down`` / ``left`` / ``right``。

        ``tests/test_app_editor.py`` 会把这张表里的每个名字跟 ``KeyboardButton``
        逐条对照，再写错就会被测试拦下来。
        """
        return {
            "r": lambda: self._rotate_action(1),
            "shift-r": lambda: self._rotate_action(-1),
            "t": self.cycle_attach_port,
            "f": self.frame_layout,
            "u": self.toggle_switch_under_cursor,
            "x": self._delete_action,
            "delete": self._delete_action,
            "c": self.close_loop,
            "backspace": self.undo,
            "z": self.undo,
            "shift-z": self.redo,
            "y": self.redo,
            "g": self.toggle_grid,
            "p": self.toggle_ports,
            "l": self.toggle_loop,
            "h": self.toggle_help,
            "b": self.toggle_scenery_mode,
            "n": lambda: self.spawn_train(1),
            "m": self.dismiss_train,
            "k": self.reverse_train_direction,
            "j": self.horn_train,
            "arrow_up": lambda: self.push_train_handle(HANDLE_STEP),
            "arrow_down": lambda: self.push_train_handle(-HANDLE_STEP),
            "space": self.release_train_handle,
            "shift-space": self.emergency_stop_train,
            "enter": self._place_action,
            "escape": self._escape_action,
            "[": lambda: self._cycle_item_action(-1),
            "]": lambda: self._cycle_item_action(1),
            ",": lambda: self._cycle_category_action(-1),
            ".": lambda: self._cycle_category_action(1),
            "v": self.toggle_mode,
            "control-s": self.begin_save_as,
            "control-o": self.begin_open_as,
            "control-n": self.clear,
            "control-z": self.undo,
            "control-y": self.redo,
        }

    def bound_key_names(self) -> list[str]:
        """本编辑器注册过的全部键盘事件名（供测试核对名字是否真实存在）。"""
        names = list(self.key_actions())
        names += [str(index) for index in range(1, 10)]
        names += list(HELD_KEYS) + [f"{key}-up" for key in HELD_KEYS]
        names += list(_SAVE_NAME_CHARS)      # 文件名输入的字符键（Ctrl+S / Ctrl+O 之后）
        return names

    def bind(self) -> None:
        """把键鼠事件挂到 Panda3D 的 messenger 上。"""
        base = self.base
        accept = base.accept

        for event, action in self.mouse_actions().items():
            accept(event, action)
        for event, action in self.key_actions().items():
            accept(event, action)
        for index in range(1, 10):
            accept(str(index), lambda i=index - 1: self._select_item_action(i))

        # 按住不放的键（相机移动）用按下 / 抬起维护一个集合
        for key in HELD_KEYS:
            accept(key, self._held.add, [key])
            accept(f"{key}-up", self._held.discard, [key])

    def _on_press(self) -> None:
        if self._prompt is not None:
            return
        self._press_mouse = self._mouse()
        self._press_moved = 0.0
        # 视角模式下左键当右键用：按住就转视角。布景模式是放置，不转视角。
        if self.mode == MODE_VIEW:
            self._on_drag_start("orbit")
            self._left_orbiting = True

    def _on_release(self) -> None:
        if self._prompt is not None:
            return
        # 拖动过就不算点击，避免"环绕视角时顺手放下一节轨道"。
        # 视角模式下左键根本不放置，只有放置 / 布景模式才认这一次点击。
        if self._press_mouse is not None and self._press_moved <= CLICK_SLOP_PX:
            if self.building:
                self.place()
            elif self.placing_scenery:
                self.place_scenery()
        self._press_mouse = None
        self._press_moved = 0.0
        if self._left_orbiting:
            self._left_orbiting = False
            self._on_drag_end()

    def _on_drag_start(self, mode: str) -> None:
        self._drag_mode = mode
        self._last_mouse = self._mouse()

    def _on_drag_end(self) -> None:
        self._drag_mode = None
        self._last_mouse = None

    def _on_zoom(self, notches: float) -> None:
        self.camera.zoom(notches)
