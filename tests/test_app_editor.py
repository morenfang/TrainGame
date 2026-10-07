"""编辑器的交互契约：投影、拾取、吸附预览、撤销、存档。

端到端已经有了（``scripts/editor_demo.py`` 用鼠标把一个圆拼出来），那为什么还要
单测？因为端到端只走**顺利**的那一条路，而真正会坏的恰恰是边界：

* 鼠标点在相机背后、点在画面外；
* 鼠标在吸附半径的内侧 / 外侧；
* 非空布局下点到空地（编辑器必须在界面上说清"这里不能放"，而不是没反应）；
* 撤销到一半再放置、连续撤销超过栈上限。

这些在端到端里几乎不会出现，坏了也不会被发现，但用户随手一划就会撞上。

    10|一个反复出现的主题：屏幕像素是编辑器唯一的度量
------------------------------------------------
端口吸附、件拾取、闭环合拢，全都在屏幕像素或世界坐标里判"够不够近"。同一个
概念如果两处用了不同的度量，就会出现"看着明明指上了却选不中"这种最难受的
故障。所以这里的断言大多是"在两个度量之间来回换算一次"——把世界点投到屏幕、
再按屏幕输入构造射线，看它是否回到原处。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from panda3d.core import Vec3

from app import editor as editor_mod
from app.editor import (DRIVE_BOOST, HANDLE_STEP, PIECE_PICK_RADIUS_PX,
                        PORT_PICK_RADIUS_PX, SEAM_POSITION_TOL, ROTATE_STEP_DEG,
                        UNDO_LIMIT, TrackEditor)
from core.geometry import Pose
from core.track.catalog import Catalog
from core.track.layout import Layout
from render import scenery as scenery_mod
from render import style

#: 引脚点/端口的屏幕坐标往返容差（像素）。Panda3D 的矩阵是 float32，
#: 几十米开外的点在屏幕上会有几百分之一的像素误差。
PIXEL_TOL = 0.5


class FakeHud:
    """只记下被写进去的文本，不碰任何 Panda3D 文本节点。"""

    def __init__(self):
        self.texts: dict[str, str] = {}
        self.visible = True
        self.expanded = False
        self.pinned = False
        self.progress: list[tuple[str, float]] = []
        self.street_lights_on = True
        self.on_toggle_street_lights = None

    def set_text(self, name: str, text: str) -> None:
        self.texts[name] = text

    def set_visible(self, flag: bool) -> None:
        self.visible = flag

    def tick(self, dt: float) -> None:
        return None

    def handle_click(self) -> bool:
        return False

    def toggle_expanded(self) -> None:
        self.expanded = not self.expanded

    def set_progress(self, message: str, fraction: float) -> None:
        self.progress.append((message, fraction))

    def toggle_street_lights(self) -> bool:
        self.street_lights_on = not self.street_lights_on
        if self.on_toggle_street_lights is not None:
            self.on_toggle_street_lights(self.street_lights_on)
        return self.street_lights_on


# --------------------------------------------------------------------------- #
# 脚手架
# --------------------------------------------------------------------------- #

@pytest.fixture()
def editor(app, tmp_path):
    """一个"鼠标由脚本控制"的编辑器。

    只覆盖 ``_has_mouse`` / ``_mouse`` 两个实例属性 —— 这正是真实鼠标与编辑器
    逻辑之间唯一的接口，所以脚本驱动和真人驾驶走的是同一条代码路径。
    """
    catalog = Catalog.builtin()
    ed = TrackEditor(app, catalog, hud=FakeHud(),
                     save_path=tmp_path / "layout.json")
    ed.fake_mouse = (0.0, 0.0)
    ed._has_mouse = lambda: True
    ed._mouse = lambda: ed.fake_mouse
    # 固定取景，让"多少像素"在测试里可复现
    ed.camera.frame(((-40.0, -40.0), (40.0, 40.0)))
    return ed


def straight_piece(catalog: Catalog, length: float = 20.0):
    for piece in catalog:
        if piece.category == "straight" and len(piece.routes) == 1 \
                and abs(piece.routes[0].length - length) < 1e-9:
            return piece
    raise AssertionError(f"件库里没有 {length} m 的直轨")


def curve_piece(catalog: Catalog):
    for piece in catalog:
        if piece.category == "curve" and len(piece.routes) == 1 \
                and abs(piece.routes[0].length - 40.0 * math.radians(45.0)) < 1e-9:
            return piece
    raise AssertionError("件库里没有 R40 的 45° 弯轨")


def to_widget(app, pixel: tuple[float, float]) -> tuple[float, float]:
    """屏幕像素 → Panda3D 鼠标的归一化坐标。"""
    return (pixel[0] / (app.win.getXSize() * 0.5),
            pixel[1] / (app.win.getYSize() * 0.5))


def aim_pixel(editor, app, pixel) -> None:
    editor.fake_mouse = to_widget(app, pixel)
    editor._refresh_hover()


def aim_world(editor, app, world) -> None:
    pixel = editor._project(world)
    assert pixel is not None, f"世界点 {world} 投影失败"
    aim_pixel(editor, app, pixel)


def pixel_of(app, editor, world) -> tuple[float, float]:
    pixel = editor._project(world)
    assert pixel is not None, f"世界点 {world} 投影失败"
    return pixel


def unit_across(app, editor, a_pixel, b_pixel) -> tuple[float, float]:
    """从 ``a`` 指向 ``b`` 的屏幕单位方向，再往外延伸时用。"""
    dx, dy = b_pixel[0] - a_pixel[0], b_pixel[1] - a_pixel[1]
    length = math.hypot(dx, dy)
    assert length > 1.0, "两个端点在屏幕上几乎重合，测试前提不成立"
    return (dx / length, dy / length)


def build_chain(editor, def_id: str, count: int, my_port: str = "a",
                target_port: str = "b") -> list[int]:
    """直接（不经鼠标）搭一条链，供与拾取无关的用例使用。"""
    first = editor.layout.add_root(def_id)
    indices = [first]
    for _ in range(count - 1):
        indices.append(editor.layout.attach(def_id, my_port,
                                            (indices[-1], target_port)))
    editor.view.sync()
    return indices


# --------------------------------------------------------------------------- #
# 世界 ↔ 屏幕
# --------------------------------------------------------------------------- #

def test_project_and_mouse_ray_are_inverses(app, editor):
    """``_project`` 与 ``mouse_ray`` 必须互为逆运算。

    这是编辑器里最容易悄悄写错、又最难用眼睛发现的一环：Panda3D 的
    ``Lens.project`` 要的是**镜头节点坐标系**下的点，喂 render 坐标会得到
    "连正对的目标都判为不可见"的假象。而它一旦错了，表现只是"端口偶尔选不中"。
    """
    for world in [(0.0, 0.0, 0.0), (30.0, -12.0, 20.0), (-25.0, 5.0, -30.0),
                  (5.0, -0.64, 28.0)]:
        pixel = pixel_of(app, editor, world)
        origin, direction = editor.mouse_ray(*to_widget(app, pixel))
        assert origin is not None and direction is not None

        to_point = Vec3(*(world[i] - origin[i] for i in range(3)))
        ray = Vec3(*direction)
        along = to_point.dot(ray)
        assert along > 0.0, "射线方向反了"
        perpendicular = (to_point - ray * along).length()
        # 距离几十米的点在屏幕上差半个像素，换算回世界就是几厘米
        assert perpendicular < 0.05


def test_project_rejects_points_behind_the_camera(app, editor):
    target = editor.camera.target
    cam = editor.camera.camera_position()
    behind = tuple(target[i] + 3.0 * (cam[i] - target[i]) for i in range(3))
    assert editor._project(behind) is None


def test_project_keeps_points_in_front_but_off_screen(app, editor):
    """画面**左右之外**的点仍然要能投影出正确的像素值。

    这条不是学术细节：``camLens.project`` 对视野外的点返回 ``False``（实测偏出
    视锥 200 m 也返回 False），但填出来的 xy 是完全正确的。如果照它的返回值判
    "看不见"，那么贴近窗口边缘的空闲端口就会忽然点不中 —— 用户只会觉得"这游戏
    有时候点不上"。
    """
    quat = app.cam.getQuat(app.render)
    forward = Vec3(quat.getForward())
    right = Vec3(quat.getRight())
    cam = Vec3(*editor.camera.camera_position())
    far_side = cam + forward * 50.0 + right * 5000.0

    pixel = editor._project((far_side[0], far_side[1], far_side[2]))
    assert pixel is not None
    assert abs(pixel[0]) > app.win.getXSize(), "这个点本该远在画面之外"


# --------------------------------------------------------------------------- #
# 屏幕拾取
# --------------------------------------------------------------------------- #

def test_pick_port_honours_the_pixel_radius(app, editor):
    piece = straight_piece(editor.catalog)
    build_chain(editor, piece.id, 1)

    a_pixel = pixel_of(app, editor, editor.layout.world_port(0, "a").position)
    b_pixel = pixel_of(app, editor, editor.layout.world_port(0, "b").position)
    across = unit_across(app, editor, a_pixel, b_pixel)

    aim_pixel(editor, app, b_pixel)
    assert editor.pick_port() == (0, "b")

    # 半径以内：仍然吸到同一个端口
    inside = (b_pixel[0] + across[0] * 30.0, b_pixel[1] + across[1] * 30.0)
    aim_pixel(editor, app, inside)
    assert 30.0 < PORT_PICK_RADIUS_PX
    assert editor.pick_port() == (0, "b")

    # 半径以外：什么都不该吸住（另一个端口离得更远）
    outside = (b_pixel[0] + across[0] * 200.0, b_pixel[1] + across[1] * 200.0)
    assert 200.0 > PORT_PICK_RADIUS_PX
    aim_pixel(editor, app, outside)
    assert editor.pick_port() is None


def test_piece_middle_is_pickable(app, editor):
    """件**正中间**必须能选中 —— 这正是"用 route 中点当抓手"存在的理由。

    取两个端口的像素中点作鼠标位置：它到两个端口的屏幕距离都超过 110 像素的
    拾取半径（20 m 直轨的两个端口在屏幕上相隔两百多像素）。所以如果只把端口
    当抓手，这里会返回 ``None``，用户就会觉得"轨道中间指不上去"。
    """
    piece = straight_piece(editor.catalog)
    build_chain(editor, piece.id, 1)

    a_pixel = pixel_of(app, editor, editor.layout.world_port(0, "a").position)
    b_pixel = pixel_of(app, editor, editor.layout.world_port(0, "b").position)
    middle = ((a_pixel[0] + b_pixel[0]) * 0.5, (a_pixel[1] + b_pixel[1]) * 0.5)
    assert math.dist(middle, a_pixel) > PIECE_PICK_RADIUS_PX
    assert math.dist(middle, b_pixel) > PIECE_PICK_RADIUS_PX

    aim_pixel(editor, app, middle)
    assert editor.pick_piece() == 0


def test_pick_piece_gives_up_when_nothing_is_near(app, editor):
    piece = straight_piece(editor.catalog)
    build_chain(editor, piece.id, 1)

    a_pixel = pixel_of(app, editor, editor.layout.world_port(0, "a").position)
    b_pixel = pixel_of(app, editor, editor.layout.world_port(0, "b").position)
    across = unit_across(app, editor, a_pixel, b_pixel)
    perpendicular = (-across[1], across[0])
    middle = ((a_pixel[0] + b_pixel[0]) * 0.5, (a_pixel[1] + b_pixel[1]) * 0.5)

    far = (middle[0] + perpendicular[0] * 400.0,
           middle[1] + perpendicular[1] * 400.0)
    aim_pixel(editor, app, far)
    assert editor.pick_piece() is None


# --------------------------------------------------------------------------- #
# 放置
# --------------------------------------------------------------------------- #

def test_free_placement_lands_under_the_cursor(app, editor):
    """空场地上的第一节要真的落在鼠标指着的地面点上 —— 用**渲染出来的几何**验。

    刻意不去重算放置公式（那就是循环论证），而是取件的实际渲染包围盒：鼠标点在
    件的包围盒里、且道砟底面正好贴在地面高度上。
    """
    piece = curve_piece(editor.catalog)
    editor.select_piece_id(piece.id)
    assert editor.current_piece.id == piece.id

    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))
    editor._refresh_ghost()
    assert editor.placement is not None and not editor.placement.snapped
    assert editor.placement.allowed

    index = editor.place()
    assert index == 0

    node = editor.view.node_for(index)
    assert node is not None
    low, high = node.getTightBounds(app.render)
    assert low[0] - 1e-3 <= 0.0 <= high[0] + 1e-3
    assert low[2] - 1e-3 <= 0.0 <= high[2] + 1e-3
    assert low[1] <= style.GROUND_Y + 1e-3, "道砟不该悬在地面之上"
    assert high[1] >= style.GROUND_Y - 1e-3, "件不该整块沉到地面之下"


def test_preview_and_actual_pose_agree_exactly(app, editor):
    """幽灵预览的位姿必须与实际放置的位姿**逐位相同**。

    两者分别由 ``compute_placement`` 与 ``Layout.attach`` 算出来；一旦有人只改了
    其中一处，用户看到的就是"预览在这里、放下去跑到那里"。所以这里不断言"误差
    很小"，而是要求恰好为 0 —— 因为两处本来就应该走同一条公式。
    """
    piece = straight_piece(editor.catalog)
    editor.select_piece_id(piece.id)

    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))
    editor._refresh_ghost()
    preview = editor.placement
    assert preview is not None
    index = editor.place()
    assert editor._placement_error(index, preview) == 0.0

    for _ in range(3):
        target = (index, "b")
        aim_world(editor, app, editor.layout.world_port(*target).position)
        editor._refresh_ghost()
        assert editor.hover_port == target, "屏幕拾取没选到瞄准的那个端口"
        preview = editor.placement
        assert preview is not None and preview.snapped and preview.allowed
        index = editor.place()
        assert editor._placement_error(index, preview) == 0.0


def test_free_placement_is_refused_once_the_layout_is_not_empty(app, editor):
    """非空布局上点到空地：必须在界面上说清"这里不能放"，而不是悄悄没反应。

    ``Layout`` 只支持一张连通装配图，所以这是它的硬限制；编辑器的责任是把这个
    限制变成**可见的**反馈（幽灵变红 + 一句提示）。
    """
    piece = straight_piece(editor.catalog)
    editor.select_piece_id(piece.id)
    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))
    editor._refresh_ghost()
    assert editor.place() == 0

    # 挪到一个离所有端口都很远的空地
    a_pixel = pixel_of(app, editor, editor.layout.world_port(0, "a").position)
    b_pixel = pixel_of(app, editor, editor.layout.world_port(0, "b").position)
    across = unit_across(app, editor, a_pixel, b_pixel)
    perpendicular = (-across[1], across[0])
    middle = ((a_pixel[0] + b_pixel[0]) * 0.5, (a_pixel[1] + b_pixel[1]) * 0.5)
    aim_pixel(editor, app, (middle[0] + perpendicular[0] * 500.0,
                            middle[1] + perpendicular[1] * 500.0))
    editor._refresh_ghost()

    placement = editor.placement
    assert placement is not None
    assert placement.snapped is False
    assert placement.allowed is False
    assert placement.note, "拒绝放置时必须给出可读的理由"

    assert editor.place() is None
    assert len(editor.layout) == 1
    assert editor.view.closure is not None and not editor.view.closure.closed


# --------------------------------------------------------------------------- #
# 删除 / 撤销 / 存档
# --------------------------------------------------------------------------- #

def test_delete_hovered_removes_only_the_hovered_piece(app, editor):
    """**「删当前轨道会把后面的一起删掉」那条抱怨的回归测试。**

    指着一条三节直轨的中间那节按 X：只有它消失，第三节**原地冻结**不动（留出真实
    缺口），不再被"接骨"拽回去。
    """
    piece = straight_piece(editor.catalog)
    indices = build_chain(editor, piece.id, 3)
    middle = indices[1]
    tail_before = editor.layout.world_port(indices[2], "b").position

    # 瞄准中间那一节的**中段**。不能直接瞄它的 pose.position —— 对直轨来说那正好
    # 是两节之间的接缝，而接缝处前面的那一节会先被选中（遍历顺序如此）。
    a_pixel = pixel_of(app, editor, editor.layout.world_port(middle, "a").position)
    b_pixel = pixel_of(app, editor, editor.layout.world_port(middle, "b").position)
    aim_pixel(editor, app, ((a_pixel[0] + b_pixel[0]) * 0.5,
                            (a_pixel[1] + b_pixel[1]) * 0.5))
    assert editor.hover_piece == middle

    editor.delete_hovered()
    assert sorted(editor.layout.pieces) == sorted([indices[0], indices[2]])
    assert len(editor.layout) == 2
    assert math.isclose(editor.layout.total_length(), 2 * piece.routes[0].length)

    # 断成两段：首节与末节各是一段的根，中间留了真实的缺口
    assert editor.layout.root_indices() == sorted([indices[0], indices[2]])
    assert not editor.layout.is_port_connected(indices[0], "b")
    assert editor.layout.peer_of(indices[0], "b") is None

    # 后半截原地不动：末节的末端还在老地方
    assert editor.layout.world_port(indices[2], "b").position == tail_before


def test_delete_hovered_removes_a_tip_without_touching_the_rest(app, editor):
    """末节（没有下游）按 X：只少它一节，前面原样不动。"""
    piece = straight_piece(editor.catalog)
    indices = build_chain(editor, piece.id, 2)
    tip = indices[1]

    a_pixel = pixel_of(app, editor, editor.layout.world_port(tip, "a").position)
    b_pixel = pixel_of(app, editor, editor.layout.world_port(tip, "b").position)
    aim_pixel(editor, app, ((a_pixel[0] + b_pixel[0]) * 0.5,
                            (a_pixel[1] + b_pixel[1]) * 0.5))
    assert editor.hover_piece == tip

    editor.delete_hovered()
    assert sorted(editor.layout.pieces) == [indices[0]]
    assert not editor.layout.is_port_connected(indices[0], "b")


def test_delete_hovered_refuses_to_guess_at_a_branch(app, editor):
    """岔口上挂着支线时不能"只删它" —— 接骨会分不清主次，要拒绝并说清原因。"""
    layout = editor.layout
    turnout = layout.add_root("turnout_l_40")
    left = layout.attach("straight_20", "a", (turnout, "b"))
    right = layout.attach("straight_20", "a", (turnout, "c"))
    editor.view.sync()

    editor.hover_piece = turnout
    editor.delete_hovered()
    assert sorted(editor.layout.pieces) == sorted([turnout, left, right])
    assert "删除失败" in editor._toast


def test_delete_without_hover_explains_itself(app, editor):
    editor.hover_piece = None
    editor.delete_hovered()
    assert "鼠标" in editor._toast


# --------------------------------------------------------------------------- #
# 鼠标模式：放置轨道 / 视角
# --------------------------------------------------------------------------- #

def test_toggle_mode_flips_what_the_left_button_does(app, editor):
    """V 在「放置轨道」与「视角」间切换；视角模式下不再有待放置的幽灵。"""
    piece = straight_piece(editor.catalog)
    editor.select_piece_id(piece.id)
    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))
    editor._refresh_ghost()
    assert editor.building and editor.placement is not None

    editor.toggle_mode()
    assert not editor.building
    assert "视角" in editor._toast
    editor._refresh_ghost()
    assert editor.placement is None
    assert editor._ghost is not None and editor._ghost.isHidden()

    editor.toggle_mode()
    assert editor.building
    assert "放置轨道" in editor._toast


def test_view_mode_left_drag_orbits_instead_of_placing(app, editor):
    """视角模式下左键是"拖视角"：拖了之后既没放轨道，相机也确实转了。"""
    piece = straight_piece(editor.catalog)
    editor.select_piece_id(piece.id)
    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))
    editor._refresh_ghost()
    assert editor.place() == 0

    editor.toggle_mode()
    before = editor.camera.camera_position()

    editor._on_press()
    editor.fake_mouse = (editor.fake_mouse[0] + 0.35, editor.fake_mouse[1])
    editor.tick()
    editor._on_release()

    assert len(editor.layout) == 1, "视角模式下左键不该再放下轨道"
    assert editor.camera.camera_position() != before, "左键拖拽没有转动相机"


def _camera_axis_drift(app, action):
    """执行一次鼠标动作，返回相机位移在"屏幕右 / 屏幕上"上的投影。"""
    quat = app.cam.getQuat(app.render)
    right, up = quat.getRight(), quat.getUp()
    before = Vec3(app.cam.getPos(app.render))
    action()
    delta = Vec3(app.cam.getPos(app.render)) - before
    return delta.dot(right), delta.dot(up)


def test_dragging_right_keeps_the_world_under_the_cursor(app, editor):
    """整条输入链路一起验：鼠标往右拖，画面里的东西就得往右走。

    相机那两个正负号已经在 ``tests/test_camera.py`` 里量过了，这里补的是它**上游**
    那一段 —— 鼠标归一化坐标（Panda3D 的 y 是向上为正）→ 乘窗口尺寸 → 喂给相机。
    链条上任何一环弄错，表现出来都一模一样（"方向反了"），所以必须从"人拖鼠标"
    这一头量一遍。
    """
    editor.toggle_mode()                       # 视角模式：左键 = 环绕

    def drag_right():
        editor._on_press()
        editor.fake_mouse = (editor.fake_mouse[0] + 0.4, editor.fake_mouse[1])
        editor._apply_drag()
        editor._on_release()

    along_right, _ = _camera_axis_drift(app, drag_right)
    assert along_right < -1.0, (
        "鼠标往右拖，相机也跟着往右走 —— 画面会往左跑，方向反了")


def test_dragging_up_keeps_the_world_under_the_cursor(app, editor):
    """纵向同理：鼠标往上拖（归一化坐标 y 变大），画面得往上走。"""
    editor.toggle_mode()

    def drag_up():
        editor._on_press()
        editor.fake_mouse = (editor.fake_mouse[0], editor.fake_mouse[1] + 0.4)
        editor._apply_drag()
        editor._on_release()

    _, along_up = _camera_axis_drift(app, drag_up)
    assert along_up < -1.0, (
        "鼠标往上拖，相机也跟着往上走 —— 画面的移动方向是反的")


def test_spawning_a_train_switches_to_view_mode(app, editor):
    """**加载列车时鼠标不该还是放置状态** —— N 之后自动切到视角。"""
    close_a_circle(editor)
    assert editor.building

    editor.spawn_train(1)
    assert not editor.building
    assert editor.compute_placement() is None

    editor.toggle_mode()                        # V 切回放置，继续编辑
    assert editor.building


def test_v_key_is_wired_to_the_mode_toggle(app, editor):
    editor.bind()
    assert editor.building
    app.messenger.send("v")
    assert not editor.building
    app.messenger.send("v")
    assert editor.building


def test_undo_redo_restore_the_exact_topology(app, editor):
    """撤销 / 重做必须逐位还原拓扑，而不只是"看起来差不多"。"""
    piece = straight_piece(editor.catalog)
    editor.select_piece_id(piece.id)

    states = [editor.layout.to_dict()]
    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))
    for _ in range(4):
        editor._refresh_ghost()
        index = editor.place()
        assert index is not None
        states.append(editor.layout.to_dict())
        aim_world(editor, app, editor.layout.world_port(index, "b").position)

    assert len(editor.layout) == 4

    for step in range(len(states) - 1, 0, -1):
        editor.undo()
        assert editor.layout.to_dict() == states[step - 1], f"第 {step} 步撤销不对"

    for step in range(1, len(states)):
        editor.redo()
        assert editor.layout.to_dict() == states[step], f"第 {step} 步重做不对"


def test_undo_stack_is_bounded(app, editor):
    """撤销栈超过上限时丢最老的，而不是无限长。"""
    piece = straight_piece(editor.catalog)
    editor.select_piece_id(piece.id)
    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))
    editor._refresh_ghost()
    assert editor.place() == 0

    for _ in range(UNDO_LIMIT + 10):
        editor.push_undo()
    assert len(editor._undo) == UNDO_LIMIT


def test_undo_on_an_empty_stack_says_so(app, editor):
    editor.undo()
    assert "撤销" in editor._toast
    editor.redo()
    assert "重做" in editor._toast


def test_save_and_load_round_trip_keeps_the_loop_closed(app, editor, tmp_path):
    """存档只存拓扑、读档重建位姿 —— 所以闭环必须原样存活。

    这条同时守着 ``Layout.to_dict`` 的一个关键设计：位姿**不入档**，读档后由
    父子链重新推导。如果哪天有人顺手把位姿也存进去、读档时又直接信任它，
    浮点误差会慢慢累积，闭环就会从 1e-13 变成 1e-9 再变成肉眼可见的缝。
    """
    piece = curve_piece(editor.catalog)
    indices = build_chain(editor, piece.id, 8)
    editor.layout.connect((indices[-1], "b"), (indices[0], "a"))
    editor.view.sync()
    assert editor.view.closure is not None and editor.view.closure.closed

    payload = editor.layout.to_dict()
    assert editor.save()
    editor.clear()
    assert len(editor.layout) == 0

    assert editor.load()
    assert editor.layout.to_dict() == payload
    closure = editor.view.closure
    assert closure is not None and closure.closed
    assert closure.visit_count == 8
    assert closure.gap_distance < SEAM_POSITION_TOL


def test_save_reports_failure_instead_of_raising(app, editor, tmp_path):
    """存档路径不可写时要给提示，不能让异常穿过界面。"""
    editor.save_path = tmp_path / "no_such_dir" / "layout.json"
    (tmp_path / "no_such_dir").write_text("我是个文件，不是目录", encoding="utf-8")
    assert editor.save() is False
    assert "保存失败" in editor._toast


def test_load_without_a_save_file_says_so(app, editor, tmp_path):
    editor.save_path = tmp_path / "missing.json"
    assert editor.load() is False
    assert "读档失败" in editor._toast


def test_ctrl_s_asks_for_a_filename_then_saves_to_it(app, editor, tmp_path):
    """Ctrl+S 进入填名模式，输入名字回车 → 存到 ``saves/<名字>.json``。"""
    piece = straight_piece(editor.catalog)
    build_chain(editor, piece.id, 2)
    editor.bind()
    send = app.messenger.send

    send("control-s")
    assert editor._prompt == editor_mod.PROMPT_SAVE

    for key in ("m", "y", "2"):
        send(key)
    assert editor._prompt_buffer == "my2"

    send("enter")
    assert editor._prompt is None
    assert editor.save_path == tmp_path / "my2.json"
    assert (tmp_path / "my2.json").exists()
    assert "my2.json" in editor._toast


def test_save_as_escape_cancels_without_writing(app, editor, tmp_path):
    """Esc 取消命名，什么都不写。"""
    piece = straight_piece(editor.catalog)
    build_chain(editor, piece.id, 2)
    editor.bind()
    send = app.messenger.send

    send("control-s")
    send("a")
    send("escape")
    assert editor._prompt is None
    assert not (tmp_path / "a.json").exists()
    assert "取消" in editor._toast


def test_save_as_backspace_edits_the_buffer(app, editor):
    editor.bind()
    send = app.messenger.send

    send("control-s")
    send("a")
    send("b")
    send("c")
    send("backspace")
    send("backspace")
    assert editor._prompt_buffer == "a"


def test_save_as_empty_name_falls_back_to_the_current_path(app, editor, tmp_path):
    """不输名字直接回车 = 存回当前 ``save_path``（快速保存）。"""
    piece = straight_piece(editor.catalog)
    build_chain(editor, piece.id, 2)
    editor.bind()

    app.messenger.send("control-s")
    app.messenger.send("enter")
    assert editor._prompt is None
    assert editor.save_path == tmp_path / "layout.json"
    assert (tmp_path / "layout.json").exists()


def test_save_as_rejects_names_with_illegal_characters(app, editor):
    editor.bind()
    editor.begin_save_as()
    editor._prompt_buffer = "bad/name"
    editor._confirm_file_prompt()
    assert editor._prompt == editor_mod.PROMPT_SAVE, "非法文件名不应退出命名模式"
    assert "非法" in editor._prompt_error
    editor._exit_file_prompt()


# --------------------------------------------------------------------------- #
# 读档填名字（Ctrl+O）
# --------------------------------------------------------------------------- #

def test_ctrl_o_asks_for_a_filename_then_loads_it(app, editor, tmp_path):
    """Ctrl+O 也弹输入框：写下某个存档的名字，回车就读那一个。"""
    piece = straight_piece(editor.catalog)
    build_chain(editor, piece.id, 3)
    editor.bind()
    send = app.messenger.send

    # 先造出一份命名存档
    send("control-s")
    for key in ("r", "o", "u", "t", "e", "1"):
        send(key)
    send("enter")
    assert (tmp_path / "route1.json").exists()
    saved = editor.layout.to_dict()

    # 换一份布局（清空），再从输入框读回 route1
    send("control-n")
    assert len(editor.layout) == 0

    send("control-o")
    assert editor._prompt == editor_mod.PROMPT_LOAD
    for key in ("r", "o", "u", "t", "e", "1"):
        send(key)
    send("enter")
    assert editor._prompt is None
    assert editor.save_path == tmp_path / "route1.json"
    assert editor.layout.to_dict() == saved
    assert "已载入" in editor._toast


def test_open_prompt_lists_existing_saves(app, editor, tmp_path):
    """读档提示条要把已有存档列出来，省得用户瞎猜名字。"""
    (tmp_path / "alpha.json").write_text("{}", encoding="utf-8")
    (tmp_path / "beta.json").write_text("{}", encoding="utf-8")
    editor.bind()
    editor.begin_open_as()
    text = editor._prompt_text()
    assert "alpha" in text and "beta" in text
    editor._exit_file_prompt()


def test_open_prompt_caps_a_long_file_listing(app, editor, tmp_path):
    """存档很多时名单要截断 —— 那一行字太长会把提示面板挤出屏幕。"""
    for index in range(20):
        (tmp_path / f"save{index:02d}.json").write_text("{}", encoding="utf-8")
    editor.bind()
    editor.begin_open_as()
    listing = editor._prompt_files()
    assert listing.endswith("…")
    assert len(listing) <= 36 + 1
    assert editor._prompt_files().count(" ") < 20
    editor._exit_file_prompt()


def test_save_prompt_is_a_single_line(app, editor, tmp_path):
    """保存提示条只有一行；读档提示条是两行（名字 / 名单各一行）。"""
    editor.bind()
    editor.begin_save_as()
    editor._name_append("a")
    assert "\n" not in editor._prompt_text()
    editor._exit_file_prompt()

    editor.begin_open_as()
    assert editor._prompt_text().count("\n") == 1
    editor._exit_file_prompt()


def test_open_prompt_rejects_a_missing_file_without_touching_the_layout(app, editor, tmp_path):
    """写了个不存在的名字：留在输入框里报错，当前布局一动不动。"""
    piece = straight_piece(editor.catalog)
    build_chain(editor, piece.id, 2)
    editor.bind()
    before = editor.layout.to_dict()

    editor.begin_open_as()
    assert editor._prompt == editor_mod.PROMPT_LOAD_CONFIRM
    editor._load_confirm_discard()
    assert editor._prompt == editor_mod.PROMPT_LOAD
    for key in ("n", "o", "p", "e"):
        editor._name_append(key)
    editor._confirm_file_prompt()
    assert editor._prompt == editor_mod.PROMPT_LOAD, "文件不存在不该退出输入框"
    assert "不存在" in editor._prompt_error
    assert editor.layout.to_dict() == before

    editor._exit_file_prompt()
    assert editor._prompt is None


def test_open_prompt_empty_name_loads_the_current_path(app, editor, tmp_path):
    """不输名字直接回车 = 读回当前 ``save_path``（快速读档）。"""
    piece = straight_piece(editor.catalog)
    build_chain(editor, piece.id, 2)
    editor.bind()

    app.messenger.send("control-s")
    app.messenger.send("enter")
    saved = editor.layout.to_dict()

    asset = editor.save_path
    editor.clear()
    assert len(editor.layout) == 0

    app.messenger.send("control-o")
    app.messenger.send("enter")
    assert editor.save_path == asset
    assert editor.layout.to_dict() == saved


def test_open_prompt_blocks_editor_shortcuts_while_typing(app, editor, tmp_path):
    """填名模式里按 ``n`` 是打字，不该顺手召唤列车。"""
    piece = straight_piece(editor.catalog)
    build_chain(editor, piece.id, 2)
    editor.bind()

    app.messenger.send("control-o")
    assert editor._prompt == editor_mod.PROMPT_LOAD_CONFIRM
    app.messenger.send("n")                      # 确认：不保存
    assert editor._prompt == editor_mod.PROMPT_LOAD
    app.messenger.send("n")                      # 填名：字母 n
    assert editor._prompt_buffer == "n"
    assert editor.train_view is None
    editor._exit_file_prompt()


def test_ctrl_o_asks_to_save_before_replacing_the_scene(app, editor, tmp_path):
    """有未保存改动时 Ctrl+O 先问保存；N 后读档会换掉当前场景。"""
    piece = straight_piece(editor.catalog)
    build_chain(editor, piece.id, 3)
    editor.bind()
    send = app.messenger.send

    send("control-s")
    for key in ("o", "t", "h", "e", "r"):
        send(key)
    send("enter")
    other = editor.layout.to_dict()

    editor.clear()
    build_chain(editor, piece.id, 1)
    assert editor.has_unsaved_work()

    send("control-o")
    assert editor._prompt == editor_mod.PROMPT_LOAD_CONFIRM
    assert "未保存" in editor._prompt_text()

    send("escape")
    assert editor._prompt is None
    assert len(editor.layout) == 1

    send("control-o")
    send("n")
    assert editor._prompt == editor_mod.PROMPT_LOAD
    for key in ("o", "t", "h", "e", "r"):
        send(key)
    send("enter")
    assert editor._prompt is None
    assert editor.layout.to_dict() == other
    assert not editor.has_unsaved_work()


def test_prompt_hides_the_bottom_panels(app, editor, tmp_path):
    """填文件名时左侧「帮助」让位；右下运行信息有车时仍常驻。"""
    editor.bind()
    editor.begin_open_as()
    editor.tick()
    assert editor.hud.texts["help"] == ""
    assert "读档" in editor.hud.texts["toast"]
    # 没上列车时运行信息本就是空的
    assert editor.hud.texts.get("train", "") == ""

    editor._exit_file_prompt()
    editor.tick()
    assert editor.hud.texts["help"] == editor_mod.HELP_TEXT


def test_train_running_info_stays_on_during_file_prompt(app, editor, tmp_path):
    """有车时，读档提示条不把右下运行信息收掉。"""
    close_a_circle(editor)
    editor.spawn_train(1)
    editor.tick()
    assert "运行信息" in editor.hud.texts["train"]

    editor.begin_open_as()
    editor.tick()
    assert editor.hud.texts["help"] == ""
    assert "运行信息" in editor.hud.texts["train"]
    assert "读档" in editor.hud.texts["toast"]


# --------------------------------------------------------------------------- #
# 闭环合拢
# --------------------------------------------------------------------------- #

def test_close_loop_grows_a_single_curve_into_a_whole_circle(app, editor):
    """单独一节**弯轨**按 C：两个端口同属一节，接不起来 —— 于是按当前件补下去。

    补什么、补几节，由 ``grow_to_close`` 算（90° 的弯轨 4 节一圈）。这里量的是
    编辑器的接线真的接上了，而且补的是**当前选中的那一件**。
    """
    editor.select_piece_id("curve_r40_l90")
    build_chain(editor, "curve_r40_l90", 1)
    assert sorted(editor.layout.free_ports()) == [(0, "a"), (0, "b")]
    assert editor.best_seam() is None

    assert editor.close_loop() is True
    assert len(editor.layout) == 4
    closure = editor.view.closure
    assert closure is not None and closure.closed
    assert closure.visit_count == 4
    assert closure.total_length == pytest.approx(2 * math.pi * 40.0, rel=1e-9)
    assert "自动补上 3 节" in editor._toast
    assert "闭环成立" in editor._toast


def test_close_loop_refuses_when_the_piece_can_never_close(app, editor):
    """直轨补不出环：必须原样回滚，并把"为什么"说清楚。

    关键是**不留痕** —— 用户按一下，要么真的合上，要么什么都没发生。留下一串
    接出去又合不上的直轨，比直接拒绝难收拾得多。
    """
    editor.select_piece_id("straight_20")
    build_chain(editor, "straight_20", 2)
    before = editor.layout.to_dict()
    undo_before = len(editor._undo)

    assert editor.close_loop() is False
    assert "自动合拢失败" in editor._toast
    assert editor.layout.to_dict() == before, "失败时不该留下任何多接的轨道"
    assert len(editor._undo) == undo_before, "失败的尝试不该占一格撤销"


def test_close_loop_completes_a_half_built_circle(app, editor):
    """**用户实际撞上的那个坑**：22.5° 的弯轨要 16 节，接 8 节差 80 m。

    以前按 C 只会得到"最接近的端口还差 80.0000 m，合不拢" —— 用户知道差 80 m
    也毫无办法，因为他没法心算"还差几节"。现在按 C 直接补齐并接上。
    """
    editor.select_piece_id("curve_r40_l22_5")
    build_chain(editor, "curve_r40_l22_5", 8)

    seam = editor.best_seam()
    assert seam is not None
    assert seam[1][0] == pytest.approx(80.0, rel=1e-6)
    assert editor.seam_is_aligned() is False

    assert editor.close_loop() is True
    assert len(editor.layout) == 16
    closure = editor.view.closure
    assert closure is not None and closure.closed
    assert closure.visit_count == 16
    assert closure.total_length == pytest.approx(2 * math.pi * 40.0, rel=1e-9)
    assert "自动补上 8 节" in editor._toast


def test_auto_closed_loop_can_be_undone_in_one_step(app, editor):
    """自动补齐是**一笔**撤销 —— 不能逼用户按 8 次退格才能回到补之前。"""
    editor.select_piece_id("curve_r40_l22_5")
    build_chain(editor, "curve_r40_l22_5", 8)
    before = editor.layout.to_dict()

    assert editor.close_loop() is True
    assert len(editor.layout) == 16

    editor.undo()
    assert editor.layout.to_dict() == before
    assert len(editor.layout) == 8
    assert editor.view.closure is not None and not editor.view.closure.closed


def test_auto_closed_loop_is_drivable_by_a_train(app, editor):
    """补出来的环必须是**能跑车**的环 —— 这才是这个功能的最终目的。"""
    editor.select_piece_id("curve_r40_l22_5")
    build_chain(editor, "curve_r40_l22_5", 8)
    assert editor.close_loop() is True

    view = editor.spawn_train(1)
    assert view is not None and view.visible, view.placement_error
    editor.push_train_handle(1.0)
    for _ in range(600):
        editor._advance_train(1.0 / 60.0)
    assert view.state.v > 1.0
    assert view.state.distance > 1.0


def test_auto_close_hint_counts_the_missing_pieces(app, editor):
    """HUD 上那句"再补 N 节"必须数对 —— 它是用户在按 C 之前唯一的线索。

    这里刻意用**心算**公式（还差的转角 ÷ 当前件的转角），因为每帧跑一遍
    ``grow_to_close`` 太重。所以这条测试同时也在钉住"心算 == 真补"。
    """
    editor.select_piece_id("curve_r40_l22_5")
    build_chain(editor, "curve_r40_l22_5", 8)
    hint = editor.auto_close_hint()
    assert "再补 8 节" in hint, hint
    assert "左弯轨 R40 / 22.5°" in hint
    assert "按 C" in hint

    # 心算与真补必须一致
    assert editor.close_loop() is True
    assert len(editor.layout) == 8 + 8

    # 闭环之后没有可说的了
    assert editor.auto_close_hint() == ""


def test_auto_close_hint_says_straights_cannot_close(app, editor):
    editor.select_piece_id("straight_20")
    build_chain(editor, "straight_20", 2)
    hint = editor.auto_close_hint()
    assert "直轨" in hint and "曲线" in hint, hint


# --------------------------------------------------------------------------- #
# 「环 + 支线」：按 C 不能把游戏带崩
# --------------------------------------------------------------------------- #

def build_loop_with_a_spare_switch_branch(editor):
    """**成环的道岔，侧股空着** —— 用户按一下 C 就崩的那个布局。

    道岔的直股是环的一部分，侧股（岔尖开向的另一条股道）上什么都没铺。从道岔的
    直股接一条「半圆 + 40 m 直线 + 半圆」的链，末端恰好落回道岔的咽喉端口，
    是一个**完全对齐**的接缝。

    返回 ``(道岔下标, 链末端下标)``。
    """
    layout = editor.layout
    turnout = layout.add_root("turnout_l_40")
    index = turnout
    for def_id in (["curve_r40_l45"] * 4 + ["straight_20"] * 2
                   + ["curve_r40_l45"] * 4):
        index = layout.attach(def_id, "a", (index, "b"))
    editor.view.sync()
    return turnout, index


def test_close_loop_survives_a_spare_switch_branch(app, editor):
    """**这条就是那个崩溃的回归测试。**

    接缝严丝合缝、``connect`` 放行，但环上还留着道岔侧股那个空闲端口。老版本
    ``detect_closure`` 从它出发走进环里绕着转，走到十万段抛 ``PathError``，异常
    从 ``LayoutView.sync()`` 冒出来 —— 按一下 C，整个游戏进程退出。

    现在要同时满足两件事：**不崩**，以及**认得对**（把环认出来，并告诉用户侧股
    没进环）。
    """
    turnout, tail = build_loop_with_a_spare_switch_branch(editor)
    assert sorted(editor.layout.free_ports()) == \
        [(turnout, "a"), (turnout, "c"), (tail, "b")]
    assert editor.best_seam()[0] == ((turnout, "a"), (tail, "b"))

    assert editor.close_loop() is True

    closure = editor.view.closure
    assert closure is not None and closure.closed, closure
    assert closure.visit_count == len(editor.layout)
    # 侧股那个空闲端口还在，而且必须被报出来
    assert editor.layout.free_ports() == [(turnout, "c")]
    assert "已合拢" in editor._toast
    assert "支线" in editor._toast, "留在环外的侧股必须报出来"
    assert "1 个空闲端口" in editor._toast


def test_a_train_can_run_on_a_loop_that_has_a_spare_branch(app, editor):
    """环上挂着支线时，列车仍然应该能在**环**上跑起来。"""
    build_loop_with_a_spare_switch_branch(editor)
    assert editor.close_loop() is True

    view = editor.spawn_train(1)
    assert view is not None and view.visible, view.placement_error

    editor.push_train_handle(1.0)
    for _ in range(600):
        editor._advance_train(1.0 / 60.0)
    assert view.state.v > 1.0
    assert view.state.distance > 1.0


def test_the_reported_layout_json_stall_is_gone_in_the_real_editor(app, editor):
    """**实盘回归（编辑器全栈）**：读 saves/layout.json → 召唤列车 → 加手柄。

    这份存档 38 件、首尾端口相距 1.2e-13 m（严丝合缝）却从没焊过缝。老版本里
    ``TrainView`` 把路径当开链，``head_range`` 只给 ``[编组全长, 总长]`` 这一小段，
    列车跑满 1748.32 m 后就被钉死在 #40→#39 的缝上（HUD 还报"缺口 0.0000 m"，
    看上去轨道是连续的）。这里走的是用户那条路：读档 → 上车 → 推手柄。
    """
    save = Path(__file__).resolve().parent.parent / "saves" / "layout.json"
    if not save.exists():
        pytest.skip(f"示例存档不在：{save}")

    editor.save_path = save
    assert editor.load()
    assert editor.view.closure is not None and editor.view.closure.closed
    total = editor.view.path.total_length

    view = editor.spawn_train(1)
    assert view is not None and view.visible, view.placement_error
    editor.push_train_handle(1.0)

    # 直接把列车搬到**缝口前 5 m**并给一个初速 —— 老版本就是在这里被钉死的：
    # 开链路径只给 [编组全长, 总长] 这一小段，列车冲到 s == 总长 就 v = 0 不动了。
    view.state.s = total - 5.0
    view.state.v = 20.0
    editor._advance_train(1.0 / 60.0)
    for _ in range(60):                        # 1 秒 × 20 m/s，够跨过那 5 m
        editor._advance_train(1.0 / 60.0)

    assert view.visible, view.placement_error
    assert view.state.v > 1.0, "列车在 #40→#39 的缝上停下了"
    assert view.state.s < total - 5.0, "没跨过那道缝（还是被开链限位钉住了）"


def test_close_loop_keeps_working_after_a_spare_branch_is_attached(app, editor):
    """侧股上接了东西之后，环还在那儿，而且环**不包含**那条支线。"""
    turnout, tail = build_loop_with_a_spare_switch_branch(editor)
    spare = editor.layout.attach("straight_20", "a", (turnout, "c"))
    editor.view.sync()
    before = len(editor.layout)

    assert editor.close_loop() is True
    closure = editor.view.closure
    assert closure is not None and closure.closed
    assert closure.visit_count == before - 1, "环不该把侧股那条支线也算进去"
    assert editor.layout.free_ports() == [(spare, "b")]
    assert editor.layout.closing_seam_ports(), "按 C 应当真的焊上一道缝"

    editor.undo()
    assert len(editor.layout) == before
    # 撤销掉的是**那道缝**（连接表里的边），不是几何：首尾依旧严丝合缝，所以它
    # 仍然是一条能跑车的环 —— 见 core.track.path 模块文档结论 2。所以要验的是
    # "缝没了"，而不是"不闭环了"（后者是旧判据的副作用）。
    assert editor.layout.closing_seam_ports() == []
    assert editor.view.closure is not None and editor.view.closure.closed
    assert editor.view.closure.gap_distance < 1e-9


def test_the_hint_reaches_the_hud(app, editor):
    """线索要真的显示出来 —— 只算出来不画，等于没做。"""
    editor.select_piece_id("curve_r40_l22_5")
    build_chain(editor, "curve_r40_l22_5", 8)
    editor._refresh_hud()
    loop = editor.hud.texts["loop"]
    assert "再补 8 节" in loop, loop
    assert "按 C" in loop


def test_close_loop_grows_the_missing_arcs_onto_a_partial_stadium(app, editor):
    """直线和曲线混着铺，最后缺一段弧 —— 按 C 补齐再接上。

    这里刻意用**真能闭上**的形状：两条 40 m 直轨 + 两端各一个 R40 半圆
    （stadium）。铺到最后一个半圆之前，接缝差 80 m（正好是直径），按 C 应当自动
    补上那 2 节 90° 弯轨。

    顺带钉住一件事：**纯直轨链永远合不上**。接缝的航向差恒为 0 而位置差恒为直轨
    总长，所以直轨怎么接都只是把那个数字撑大 —— 这就是为什么 auto-close 必须能
    在补不通时干脆回滚，而不是硬接。
    """
    chain = [editor.layout.add_root("straight_20")]
    for def_id in ("straight_20", "curve_r40_l90", "curve_r40_l90",
                   "straight_20", "straight_20"):
        chain.append(editor.layout.attach(def_id, "a", (chain[-1], "b")))
    editor.view.sync()

    seam = editor.best_seam()
    assert seam is not None
    (a, b), (distance, heading_gap) = seam
    assert a == (0, "a") and b == (chain[-1], "b")
    assert distance == pytest.approx(80.0, rel=1e-6), "差一个直径"
    assert editor.seam_is_aligned() is False

    assert editor.select_piece_id("curve_r40_l90")
    assert editor.close_loop() is True
    assert len(editor.layout) == 8
    closure = editor.view.closure
    assert closure is not None and closure.closed
    assert "自动补上 2 节" in editor._toast
    assert "闭环成立" in editor._toast


def test_close_loop_rolls_back_when_a_straight_chain_cannot_close(app, editor):
    """直轨链按 C：补不通就**原样回滚**，并且提示里带着具体缺口数字。"""
    piece = straight_piece(editor.catalog)
    indices = build_chain(editor, piece.id, 2)

    seam = editor.best_seam()
    assert seam is not None
    (a, b), (distance, heading_gap) = seam
    assert (a, b) == ((0, "a"), (indices[1], "b"))
    assert distance == pytest.approx(2 * piece.routes[0].length, abs=1e-9)
    assert heading_gap == pytest.approx(0.0, abs=1e-12), "两端的朝向其实是对的"

    before = editor.layout.to_dict()
    assert editor.close_loop() is False
    assert "自动合拢失败" in editor._toast
    assert "合不上" in editor._toast, "提示里必须有具体缺口，不能只说『失败了』"
    assert editor.layout.to_dict() == before, "失败时不该改动布局"


def test_close_loop_refuses_once_every_port_is_taken(app, editor):
    piece = curve_piece(editor.catalog)
    indices = build_chain(editor, piece.id, 8)
    assert editor.close_loop() is True

    assert editor.layout.free_ports() == []
    assert editor.best_seam() is None
    assert editor.close_loop() is False
    assert "空闲端口" in editor._toast
    assert len(indices) == 8


def test_close_loop_accepts_an_aligned_seam(app, editor):
    piece = curve_piece(editor.catalog)
    indices = build_chain(editor, piece.id, 8)

    assert editor.seam_is_aligned() is True
    assert editor.close_loop() is True
    closure = editor.view.closure
    assert closure is not None and closure.closed
    assert closure.visit_count == 8
    assert closure.gap_distance < SEAM_POSITION_TOL
    assert abs(closure.total_length - 8 * piece.routes[0].length) < 1e-9
    assert len(indices) == 8


def test_best_seam_is_cached_but_follows_edits(app, editor):
    """``best_seam`` 每帧都会被 HUD 问到，所以按布局版本缓存；布局一变必须失效。"""
    piece = curve_piece(editor.catalog)
    build_chain(editor, piece.id, 8)

    first = editor.best_seam()
    assert editor.best_seam() is first, "同一布局版本内应当复用缓存"

    editor.layout.attach(piece.id, "a", (7, "b"))
    editor.view.sync()
    second = editor.best_seam()
    assert second is not first, "布局改了，缓存必须失效"
    assert second is not None and second[0] != first[0]


# --------------------------------------------------------------------------- #
# 选件与按键
# --------------------------------------------------------------------------- #

def test_select_piece_id_switches_category(app, editor):
    target = curve_piece(editor.catalog)
    assert editor.category != target.category
    assert editor.select_piece_id(target.id) is True
    assert editor.category == target.category
    assert editor.current_piece.id == target.id
    assert editor.attach_slot == 0


def test_cycling_wraps_around(app, editor):
    count = len(editor.pieces_in_category)
    editor.cycle_piece(-1)
    assert editor._piece_index == count - 1
    editor.cycle_piece(1)
    assert editor._piece_index == 0

    categories = len(editor.categories)
    editor.cycle_category(-1)
    assert editor._category_index == categories - 1
    editor.cycle_category(1)
    assert editor._category_index == 0

    piece = editor.current_piece
    assert len(piece.port_ids) >= 2
    seen = set()
    for _ in range(len(piece.port_ids)):
        seen.add(editor._attach_port(piece))
        editor.cycle_attach_port()
    assert seen == set(piece.port_ids)


def test_rotate_only_touches_the_free_heading(app, editor):
    before = editor.layout.to_dict()
    editor.rotate(1)
    assert editor.free_heading == pytest.approx(math.radians(ROTATE_STEP_DEG))
    editor.rotate(-2)
    assert editor.free_heading == pytest.approx(-math.radians(ROTATE_STEP_DEG))
    editor.reset_heading()
    assert editor.free_heading == 0.0
    assert editor.layout.to_dict() == before


def test_key_bindings_do_what_the_help_says(app, editor):
    """把帮助里承诺的按键逐个按一遍。

    这条测的是"接线"：按键名写错（``bracket_right`` 还是 ``]``）、两个功能绑到
    同一个键、或者漏绑，都只会在真人操作时才发现。所以这里直接往 messenger 上
    发事件，看效果对不对。（这条测试确实抓到过 ``bracket_right`` / ``period``
    两个哑键 —— 事件名写错时它不会有任何报错。）
    """
    grid = app.render.attachNewNode("ground").attachNewNode("grid")
    editor.bind()
    send = app.messenger.send

    # 旋转 / 归零
    send("r")
    assert editor.free_heading == pytest.approx(math.radians(ROTATE_STEP_DEG))
    send("escape")
    assert editor.free_heading == 0.0

    # 换接驳端口 / 换件 / 换类别
    slot = editor.attach_slot
    send("t")
    assert editor.attach_slot != slot
    piece_index = editor._piece_index
    send("]")
    assert editor._piece_index != piece_index
    category = editor._category_index
    send(".")
    assert editor._category_index != category

    # 显示开关
    send("g")
    assert grid.isHidden(), "G 应当收起网格"
    send("g")
    assert not grid.isHidden()

    assert editor.view.show_ports is True
    send("p")
    assert editor.view.show_ports is False
    send("l")
    assert editor.view.show_loop is False

    assert editor.hud.expanded is False
    send("h")
    assert editor.hud.expanded is True
    send("h")
    assert editor.hud.expanded is False

    # 撤销
    piece = straight_piece(editor.catalog)
    editor.select_piece_id(piece.id)
    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))
    editor._refresh_ghost()
    assert editor.place() == 0
    send("backspace")
    assert len(editor.layout) == 0
    send("control-y")
    assert len(editor.layout) == 1

    # 没有悬停目标时扳道岔 / 删除都应当只给提示
    editor.hover_piece = None
    send("u")
    assert "道岔" in editor._toast
    send("x")
    assert len(editor.layout) == 1


# --------------------------------------------------------------------------- #
# 每帧更新
# --------------------------------------------------------------------------- #

def test_tick_updates_the_hud(app, editor):
    piece = curve_piece(editor.catalog)
    editor.select_piece_id(piece.id)
    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))
    editor.tick()

    status = editor.hud.texts["status"]
    assert piece.name in status
    assert "鼠标下" in status
    assert "空场地" in editor.hud.texts["loop"]
    assert "闭合" in editor.hud.texts["help"]

    editor._refresh_ghost()
    assert editor.place() == 0
    editor.tick()
    assert "轨道 1 节" in editor.hud.texts["status"]


def test_tick_survives_a_missing_grid_node(app, editor):
    """场景里没有网格节点时（比如换了个场景），G 键只能给提示，不能炸。"""
    editor.toggle_grid()
    assert "网格" in editor._toast


def test_local_handles_cover_ports_and_true_arc_midpoints(app, editor):
    """抓手点 = 端口 + 每条 route 的**真实弧**中点（不是弦中点）。

    直轨上两者重合，看不出区别；弯轨上差别是实打实的（R40/45° 的弧中点在
    (15.307, 3.061)，弦中点在 (14.142, 5.858)）。必须用弧中点，是因为渲染出来的
    轨道就在弧上 —— 抓手落在弦上，就等于"用户指着的那个像素其实没有轨道"。
    """
    straight = straight_piece(editor.catalog)
    handles = editor._local_handles(straight)
    assert len(handles) == len(straight.port_ids) + len(straight.routes)

    a, b = straight.port("a").pose, straight.port("b").pose
    assert handles[-1][0] == pytest.approx((a.x + b.x) * 0.5)
    assert handles[-1][2] == pytest.approx((a.z + b.z) * 0.5)

    curve = curve_piece(editor.catalog)
    radius = curve.routes[0].radius
    start = curve.port("a").pose
    arc_mid = editor._local_handles(curve)[-1]

    # 左转 45° 的圆弧：圆心在起点的 +Z 侧一个半径处（起点处行进方向为 +X）。
    # 弧中点到圆心的距离必须**正好**是半径；弦中点到圆心则明显更近。
    center = (start.x, start.z + radius)
    assert math.dist((arc_mid[0], arc_mid[2]), center) == pytest.approx(
        radius, abs=1e-9)
    chord_mid = ((a.x + curve.port("b").pose.x) * 0.5,
                 (a.z + curve.port("b").pose.z) * 0.5)
    assert math.dist(chord_mid, center) < radius - 0.5, "弦中点该在圆内"


def test_hover_is_cleared_without_a_mouse(app, editor):
    editor._has_mouse = lambda: False
    editor.tick()
    assert editor.hover_piece is None and editor.hover_port is None


# --------------------------------------------------------------------------- #
# 键鼠绑定
# --------------------------------------------------------------------------- #

def _base_button_name(event: str) -> str:
    """事件名 → 按钮本身的名字。

    ``shift-r`` → ``r``，``control-shift-z`` → ``z``，``w-up`` → ``w``。

    Panda3D 的事件名 = 修饰键前缀 + 按钮名（可再带一个 ``-up``）。抬起的键
    ``w-up`` 里的 ``-up`` 是后缀而不是修饰键，所以要从头、尾各剥一次。
    """
    name = event
    while True:
        head, separator, tail = name.partition("-")
        if separator and head in ("shift", "control", "alt", "meta"):
            name = tail
            continue
        break
    if name.endswith("-up"):
        name = name[:-3]
    return name


def _real_keyboard_names() -> set[str]:
    """Panda3D 键盘真的会发出的事件名。

    ``KeyboardButton`` 上作为属性的只有"有名字的键"（``backspace`` / ``arrow_up``
    / ``f5`` …），字母数字和符号键要用 ``ascii_key`` 按字符取 —— 这也正是符号键
    的事件名就是符号本身的原因。
    """
    from panda3d.core import ButtonHandle, KeyboardButton

    names: set[str] = set()
    for attribute in dir(KeyboardButton):
        if attribute.startswith("_") or attribute in ("this", "this_const",
                                                     "this_metatype"):
            continue
        try:
            handle = getattr(KeyboardButton, attribute)()
        except Exception:                      # noqa: BLE001 —— 只认能取到名字的
            continue
        if isinstance(handle, ButtonHandle):
            names.add(handle.get_name())
    for code in list(range(32, 127)) + [127]:   # 可打印 ASCII + DEL（Delete 键）
        handle = KeyboardButton.ascii_key(chr(code))
        if handle is not None:
            names.add(handle.get_name())
    return names


def test_every_key_bound_is_a_real_panda3d_event_name(editor):
    """绑定的每个键盘事件名都必须是 Panda3D 真会发出的名字。

    写错一个键名不会有任何报错，只会表现为"那个键按了没反应" —— 用户要试很久
    才能确认是程序坏了而不是自己按错了。``[`` ``]`` ``,`` ``.`` 四个键就曾经因为
    写成了 ``bracket_left`` / ``comma`` / ``period`` 而全部是哑的。
    """
    real = _real_keyboard_names()
    unknown = sorted({event for event in editor.bound_key_names()
                      if _base_button_name(event) not in real})
    assert not unknown, (
        f"这些事件名 Panda3D 永远不会发出，等于哑键：{unknown}；"
        f"符号键的事件名就是符号本身（[ ] , .），方向键叫 arrow_up 之类")


def test_every_mouse_binding_uses_a_real_button_name(editor):
    """鼠标 / 滚轮同理：``mouse1`` 而不是 ``left``，``-up`` 是后缀。"""
    from panda3d.core import MouseButton

    known = {MouseButton.one().get_name(), MouseButton.two().get_name(),
             MouseButton.three().get_name(),
             "wheel_up", "wheel_down"}     # 滚轮的 ButtonHandle 没导出到 Python
    for event in editor.mouse_actions():
        assert _base_button_name(event) in known, \
            f"{event!r} 不是 Panda3D 会发出的鼠标事件名"


def test_help_text_documents_every_placement_key(editor):
    """帮助面板必须写清"怎么换轨道件"，否则用户只会一直放同一节轨道。"""
    editor._refresh_hud()
    help_text = editor.hud.texts["help"]
    for token in ("左键", "R 旋转", "[ ]", "1-9", "退格"):
        assert token in help_text, f"帮助里没提 {token}"


def test_punctuation_keys_actually_cycle_the_piece(app, editor):
    """符号键真的接到了换件 / 换分类上（而不是哑键）。

    这里替换掉两个动作再 bind，是为了精确验证"事件名 → 动作"这一步 —— 直接按
    真实实现去断言"件换了"会依赖当前分类里有几件，是测试在依赖数据细节。
    """
    calls: list[tuple[str, int]] = []
    editor.cycle_piece = lambda step: calls.append(("piece", step))
    editor.cycle_category = lambda step: calls.append(("category", step))
    editor.bind()

    for event in ("]", "[", ".", ","):
        app.messenger.send(event)
    assert calls == [("piece", 1), ("piece", -1),
                     ("category", 1), ("category", -1)]


def test_shifted_key_does_not_also_fire_the_plain_key(app, editor):
    """按 Shift+R 只能触发反向旋转，不能再触发一次正向。

    Panda3D 的事件名是"按键 + 当时的修饰键"拼出来的，一次物理按键只发一个事件。
    万一它连基础键一起发，Shift+R 就会 +1 又 -1、净效果是没转 —— 在真人手里
    表现为"旋转偶发失灵"，极难定位，所以在这里钉死。
    """
    calls: list[int] = []
    editor.rotate = lambda step: calls.append(step)
    editor.bind()

    app.messenger.send("shift-r")
    assert calls == [-1]


# --------------------------------------------------------------------------- #
# 列车：召唤、换编组、手柄调速
# --------------------------------------------------------------------------- #

def close_a_circle(editor) -> None:
    """铺一个 R40 × 45° × 8 的整圆并合拢，好让列车有路可跑。"""
    first = editor.layout.add_root("curve_r40_l45")
    index = first
    for _ in range(7):
        index = editor.layout.attach("curve_r40_l45", "a", (index, "b"))
    editor.layout.connect((index, "b"), (first, "a"))
    editor.view.sync()


def lay_an_open_line(editor, pieces: int = 40) -> None:
    """铺一条 800 m 的**开链**（没合拢）。"""
    first = editor.layout.add_root("straight_20")
    index = first
    for _ in range(pieces - 1):
        index = editor.layout.attach("straight_20", "a", (index, "b"))
    editor.view.sync()


def test_a_layout_passed_to_the_constructor_is_drawn(app, tmp_path):
    """``--open`` 的整条链路：存档 → 编辑器 → **画面上真的看得到**。

    ``main.py`` 把载入的布局交给 ``TrackEditor(layout=...)``；如果编辑器建好
    视图却忘了同步一次，数据、闭环检测、HUD 全都正确，唯独场景是空的 ——
    用户看到的就是"载入 circle.json 里面什么也没有"。所以这条断言落在
    **场景节点数**上，而不是布局数据上。
    """
    from app.editor import TrackEditor
    from core.track.layout import Layout

    catalog = Catalog.builtin()
    saved = Layout(catalog=catalog)
    first = saved.add_root("curve_r40_l45")
    index = first
    for _ in range(7):
        index = saved.attach("curve_r40_l45", "a", (index, "b"))
    saved.connect((index, "b"), (first, "a"))
    path = tmp_path / "circle.json"
    saved.save_json(path)

    reloaded = Layout.load_json(path, catalog)
    assert len(reloaded) == 8

    # 完全按 main.py 的方式构造：布局交给编辑器之后不该再做任何事
    ed = TrackEditor(app, catalog, layout=reloaded, hud=FakeHud())
    assert len(ed.view._piece_nodes) == 8, "构造完就该看得见，不该等外部再 sync 一次"
    assert ed.view.triangle_count() > 0
    assert ed.view.closure is not None and ed.view.closure.closed
    assert ed.view.path is not None

    # 取景也要能用（main.py 紧接着就调它）
    ed.frame_layout()
    assert ed.camera.distance > 0.0


def test_the_category_menu_shows_every_category_and_marks_the_current_one(app,
                                                                        editor):
    """类别菜单必须把**所有**类别都列出来。

    这也是补的：类别靠 ``,`` / ``.`` 循环切换，而循环切换的菜单如果不显示出来就
    等于不存在 —— 用户看不到"曲线"这一类，就会以为"只有直道，那没法闭合"。
    """
    labels = [editor.category_label(cat) for cat in editor.categories]
    assert labels == ["直轨", "曲线", "坡道", "高架桥", "交叉", "道岔", "终端"]

    menu = editor.category_menu()
    for label in labels:
        assert label in menu, f"菜单里少了「{label}」：{menu}"
    assert menu.split()[0] == "[直轨]", menu

    editor.cycle_category(1)
    assert editor.category == "curve"
    assert "[曲线]" in editor.category_menu()
    assert "[直轨]" not in editor.category_menu()

    # 状态面板里也要真的出现（菜单是给用户看的，不是只给测试看的）
    editor._refresh_hud()
    assert "分类" in editor.hud.texts["status"]
    assert "[曲线]" in editor.hud.texts["status"]


def test_curve_pieces_are_reachable_by_cycling_the_category(app, editor):
    """从默认的「直轨」按几下就能摸到弯轨 —— 这是"怎么拼出闭环"的入口。"""
    assert editor.category == "straight"
    editor.cycle_category(1)
    assert editor.category == "curve"
    pieces = editor.pieces_in_category
    assert len(pieces) >= 4
    # 每个方向、每种弧度都要能选到
    for step in range(len(pieces)):
        editor.select_piece(step)
        assert editor.current_piece.category == "curve"
    assert any("l22_5" in p.id for p in pieces)
    assert any("r45" in p.id for p in pieces)
    assert any("l90" in p.id for p in pieces)

    # 按 id 直选也要能跳到弯轨（脚本 / 未来的快捷栏用这条）
    assert editor.select_piece_id("curve_r40_l45")
    assert editor.category == "curve"
    assert editor.current_piece.id == "curve_r40_l45"


def test_a_train_can_spawn_on_an_open_but_connected_line(app, editor):
    """**「扳道岔列车消失」那条抱怨的根源修正。**

    门槛是**连通**而不是**闭合**：一条没合拢但**连成一整条**的开链应当能上车
    （跑到尽头停下），只有轨道被拆散成互不相连的几段时才拒绝。这里铺 800 m
    开链、没有环，按 N 仍应上车。
    """
    lay_an_open_line(editor, 40)                 # 800 m 开链，连成一整条
    assert editor.view.closure is not None and not editor.view.closure.closed
    assert editor.layout.component_count() == 1

    view = editor.spawn_train(1)
    assert view is not None and view.visible, view.placement_error
    assert editor.train_view is view
    view.destroy()


def test_a_train_cannot_spawn_when_the_track_is_split_into_segments(app, editor):
    """**删断成互不相连的几段后，仍然不能上车** —— 保留"删断无车"的保护。"""
    lay_an_open_line(editor, 40)
    middle = 20
    editor.push_undo()
    editor.layout.remove_piece(middle)            # 断成两段
    editor.view.sync()
    assert editor.layout.component_count() == 2

    view = editor.spawn_train(1)
    assert view is None
    assert editor.train_view is None
    assert "断成了几段" in editor._toast


def test_toggling_a_switch_reroutes_the_train_without_dismissing_it(app, tmp_path):
    """**立交场景的回归测试：按 U 扳道岔，列车改线、不消失。**

    主线被道岔截断成走疏解线的开链，但轨道仍是**一整条连通的线** —— 列车应当
    重新锚定到新路径继续开，而不是被请下轨；扳回直股又回到环线继续跑。
    """
    from scenes import presets

    catalog = Catalog.builtin()
    scene = presets.build("overpass", catalog)
    ed = TrackEditor(app, catalog, layout=scene.layout, hud=FakeHud(),
                     save_path=tmp_path / "layout.json")
    turnouts = [i for i in range(len(ed.layout))
                if ed.layout.definition(i).is_switch]
    assert len(turnouts) == 2

    view = ed.spawn_train(1)
    assert view is not None and view.visible, view.placement_error
    assert ed.view.closure.closed

    # 扳到岔股：主线截断，但仍是连通的一条线 —— 列车不下轨
    ed.layout.toggle_switch(turnouts[0])
    ed.view.sync()
    assert not ed.view.closure.closed
    assert ed.layout.component_count() == 1

    ed._advance_train(1.0 / 60.0)
    assert ed.train_view is not None, "扳道岔后列车不该被请下轨"
    assert ed.train_view.visible

    # 扳回直股：回到环线，列车仍在
    ed.layout.toggle_switch(turnouts[0])
    ed.view.sync()
    assert ed.view.closure.closed
    ed._advance_train(1.0 / 60.0)
    assert ed.train_view is not None
    ed.train_view.destroy()


def test_calling_a_train_twice_swaps_it_instead_of_stacking(app, editor):
    """连续按 N 只能有一列车在场景里 —— 否则每按一次就多一列，很快就卡死。"""
    close_a_circle(editor)
    first = editor.spawn_train(1)
    assert first is not None and first.visible
    assert editor.train_view is first

    second = editor.spawn_train(1)
    assert second is not first
    assert editor.train_view is second
    roots = [node for node in app.render.getChildren()
             if node.getName().startswith("train_")]
    assert len(roots) == 1, f"场景里有 {len(roots)} 列列车"
    second.destroy()


def test_cycling_through_the_catalogue_and_back(app, editor):
    """N 会在整个列车目录里循环 —— 每个车型都得能上轨，不能有"上不去"的。"""
    close_a_circle(editor)
    seen = []
    for _ in range(len(editor.trains)):
        view = editor.spawn_train(1)
        assert view is not None
        assert view.visible, f"{view.spec.id}: {view.placement_error}"
        seen.append(view.spec.id)
    assert len(set(seen)) == len(editor.trains), seen
    editor.dismiss_train()
    assert editor.train_view is None
    assert app.render.find("**/train*").isEmpty()


def test_spawning_on_an_empty_field_says_why_instead_of_crashing(app, editor):
    """空场地上召唤列车：不抛异常，只拒绝并把原因写进提示。"""
    view = editor.spawn_train(1)
    assert view is None
    assert editor.train_view is None
    assert "空的" in editor._toast


def test_handle_keys_push_and_release_the_single_handle(app, editor):
    """↑ 牵引、↓ 制动、空格 惰行 —— 三条都必须真的接到手柄上。"""
    close_a_circle(editor)
    editor.spawn_train(1)
    editor.bind()

    app.messenger.send("arrow_up")
    assert editor.train_handle == pytest.approx(HANDLE_STEP)
    assert editor.train_view.state.throttle == pytest.approx(HANDLE_STEP)
    assert editor.train_view.state.brake == 0.0

    app.messenger.send("arrow_down")
    app.messenger.send("arrow_down")
    app.messenger.send("arrow_down")
    # 推上去一档、再往下三档：从 +1 档落回 -2 档。
    assert editor.train_handle == pytest.approx(-2 * HANDLE_STEP)
    assert editor.train_view.state.brake == pytest.approx(2 * HANDLE_STEP)
    assert editor.train_view.state.throttle == 0.0

    app.messenger.send("space")
    assert editor.train_handle == 0.0
    assert editor.train_view.state.brake == 0.0
    assert editor.train_view.state.throttle == 0.0
    editor.dismiss_train()


def test_four_handle_taps_reach_the_full_notch(app, editor):
    """**「加速减速太慢」那条抱怨的手感部分。**

    档位是照操纵台设的：手柄行程分 4 档拉开（``HANDLE_STEP = 0.25``），所以
    ↑↑↑↑ 就到了满级牵引，而不是要连按十几次才推满。方向键接的就是这个步长，
    这条测试盯的是**手感契约**本身，不是某一次按下的数值。
    """
    close_a_circle(editor)
    editor.spawn_train(1)
    editor.bind()

    for _ in range(4):
        app.messenger.send("arrow_up")
    assert editor.train_handle == pytest.approx(1.0)
    assert editor.train_view.state.throttle == pytest.approx(1.0)

    app.messenger.send("space")               # 先回中位，再从零往制动侧推
    assert editor.train_handle == 0.0

    for _ in range(4):
        app.messenger.send("arrow_down")
    assert editor.train_handle == pytest.approx(-1.0)
    assert editor.train_view.state.brake == pytest.approx(1.0)
    editor.dismiss_train()


def test_close_loop_swaps_in_a_piece_that_fits_when_the_held_one_cannot(app, editor):
    """**「没有合适长度的铁轨对接」那条抱怨的解法（端到端）。**

    环只差一节 22.5° 的弯，而用户手上推着的是 40 m 直轨：

    * 旧行为：拿直轨一直接下去，接满 32 节回一句"还是合不上" —— 用户看到的就是
      "差一点，但没有一节轨道能对接"；
    * 现在：先拿手上这一件试一次（合不上），再在候选里自己换成弯轨，**一节**合上。

    关键是提示要说清"换成了哪一件"：用户手上推着直轨，屏幕上却接出一节弯轨，
    这件事必须说出来，否则就成了"这编辑器怎么自作主张"。
    """
    editor.select_piece_id("curve_r40_l22_5")
    build_chain(editor, "curve_r40_l22_5", 15)
    editor.select_piece_id("straight_40")      # 手上换成接不通的那一件

    assert editor.best_seam() is not None
    assert editor.seam_is_aligned() is False

    assert editor.close_loop() is True
    assert len(editor.layout) == 16
    closure = editor.view.closure
    assert closure is not None and closure.closed
    assert closure.total_length == pytest.approx(2 * math.pi * 40.0, rel=1e-9)
    assert "自动补上 1 节" in editor._toast
    assert "左弯轨 R40 / 22.5°" in editor._toast, "提示里要写清实际补的是哪一件"


def test_close_loop_still_prefers_the_piece_the_user_is_holding(app, editor):
    """手上那一件能合上时**就用它** —— 自动换件只是后手，不该抢在前面。

    这条守住的是行为兼容：从前的"补当前件"是用户已经熟悉的东西，加了候选表之后
    结果必须一模一样（补的节数、补的是哪一件都不能变）。
    """
    editor.select_piece_id("curve_r40_l90")
    build_chain(editor, "curve_r40_l90", 1)

    assert editor.close_loop() is True
    assert len(editor.layout) == 4
    assert "自动补上 3 节" in editor._toast
    assert "左弯轨 R40 / 90°" in editor._toast


def test_summoned_train_runs_with_the_sandbox_drive_boost(app, editor):
    """游戏里的列车带着**沙盘手感倍率**上场。

    数据（`data/trains.json`）仍是照真实车型标定的值，倍率只活在游戏这一侧 ——
    见 `app/editor.py` 里 ``DRIVE_BOOST`` 的说明。
    """
    assert DRIVE_BOOST > 1.0
    close_a_circle(editor)
    editor.spawn_train(1)
    assert editor.train_view.train.drive_boost == pytest.approx(DRIVE_BOOST)
    editor.dismiss_train()


def test_emergency_stop_key_pulls_the_handle_all_the_way_back(app, editor):
    close_a_circle(editor)
    editor.spawn_train(1)
    editor.bind()
    app.messenger.send("shift-space")
    assert editor.train_view.state.brake == pytest.approx(1.0)
    assert editor.train_view.state.throttle == 0.0
    assert editor.train_handle == pytest.approx(-1.0)
    editor.dismiss_train()


def test_emergency_braking_shows_on_the_handle_and_lasts_until_released(app, editor):
    """紧急制动在 HUD 上看得见，而且不会因为再点一下 ↓ 就被悄悄降级。"""
    close_a_circle(editor)
    editor.spawn_train(1)
    editor.bind()
    app.messenger.send("shift-space")
    assert editor.train_view.train.is_emergency
    assert editor.handle_label() == "紧急制动"

    # 手柄本来就在 -1.0，再点一下 ↓ 只是重复拉到底 —— 不该把紧急制动撤掉
    editor.push_train_handle(-0.25)
    assert editor.train_view.train.is_emergency, "补一点制动力把紧急制动弄丢了"

    app.messenger.send("space")               # 回中位（惰行）才解除
    assert not editor.train_view.train.is_emergency
    assert editor.handle_label() == "惰行"
    editor.dismiss_train()


def test_pushing_the_handle_without_a_train_explains_instead_of_crashing(app, editor):
    assert editor.push_train_handle(0.1) is None
    assert "N" in editor._toast, "没车时推手柄应当提示怎么召唤列车"
    assert editor.train_view is None


def test_reverse_key_fast_stops_then_reverses_direction(app, editor):
    """K 键换向：先紧急制动力级刹停，方向翻个后再反向满牵引加速。"""
    close_a_circle(editor)
    view = editor.spawn_train(1)
    editor.bind()

    # 给列车一个前进的速度，再触发换向
    view.state.v = 20.0
    view.state.direction = 1.0
    app.messenger.send("k")
    assert view.train.is_emergency, "换向应当先用紧急制动力快速刹停"
    assert editor.train_handle == pytest.approx(1.0), "换向完成后手柄应为满牵引"

    # 一路刹停后，方向翻个、反向起步
    for _ in range(20000):
        editor._advance_train(0.05)
        if view.train.direction < 0.0:
            break
    assert view.train.direction == -1.0
    assert not view.train.is_emergency
    assert view.train.state.throttle == pytest.approx(1.0)
    view.destroy()


def test_reverse_key_without_a_train_explains_itself(app, editor):
    assert editor.train_view is None
    editor.reverse_train_direction()
    assert "N" in editor._toast, "没车时按 K 应当提示怎么召唤列车"


def test_swapping_the_train_keeps_the_handle_notch(app, editor):
    """换一列编组不该把司机推好的档位清零。"""
    close_a_circle(editor)
    first = editor.spawn_train(1)
    editor.push_train_handle(0.4)
    assert editor.train_handle == pytest.approx(0.4)

    second = editor.spawn_train(1)
    assert second is not first
    assert second.state.throttle == pytest.approx(0.4)
    assert second.state.brake == 0.0
    second.destroy()


def test_running_a_train_actually_moves_it_along_the_loop(app, editor):
    close_a_circle(editor)
    view = editor.spawn_train(1)
    editor.push_train_handle(1.0)
    for _ in range(80):
        editor._advance_train(0.1)
    assert view.state.v > 1.0, "推了手柄列车却没动"
    assert view.visible
    for index in range(view.car_count):
        assert not view.node_for(index).isHidden()
    view.destroy()


def test_train_hud_reports_speed_and_handle(app, editor):
    close_a_circle(editor)
    view = editor.spawn_train(1)
    view.state.v = 20.0
    editor.push_train_handle(0.5)
    editor._refresh_hud()
    text = editor.hud.texts["train"]
    assert "72.0 km/h" in text, text
    assert "牵引 50%" in text, text
    view.destroy()


def test_train_hud_is_hidden_until_a_train_is_called(app, editor):
    """没车时右下角必须是空的 —— 否则会留下一块什么都没有的黑块挡住场景。"""
    editor._refresh_hud()
    assert editor.hud.texts["train"] == ""


def test_deleting_the_track_under_a_train_dismisses_the_train(app, editor):
    """把列车脚下的轨道拆掉：闭环断开，车被请下轨（回到无车状态），不抛异常。"""
    close_a_circle(editor)
    editor.spawn_train(1)
    editor.push_train_handle(1.0)
    editor._advance_train(0.1)

    editor.push_undo()
    editor.layout.clear()
    editor.view.sync()
    editor._advance_train(0.1)

    assert editor.train_view is None, "闭环断开后列车应当下轨"
    assert "下轨" in editor._toast


def test_deleting_a_single_piece_under_a_train_dismisses_the_train(app, editor):
    """**按 X 只删列车脚下那一节的回归测试。**

    删掉一节轨道后闭环断成开链，但仍是一个连通分量，``_advance_train`` 不会据此
    摘车；旧代码在下一帧重走线时把车锚到一条比编组还短的路上，``assert head is
    not None`` 直接把进程崩掉。现在删到车压着的那一节时先把列车摘下。
    """
    close_a_circle(editor)
    view = editor.spawn_train(1)
    editor.push_train_handle(1.0)
    for _ in range(30):
        editor._advance_train(0.1)
    assert editor.train_view is not None

    # 车头正压着的那一节
    head_seg = view._driving.segment_at(view.state.s)[0]
    assert view.occupies_piece(head_seg.piece_index)
    editor.hover_piece = head_seg.piece_index

    editor.delete_hovered()
    assert editor.train_view is None, "删到列车脚下，列车应当先下轨"

    # 再推几帧确认不崩（旧代码在这里 assert 崩掉）
    for _ in range(30):
        editor._advance_train(0.1)
    assert "列车已下轨" in editor._toast


def test_deleting_a_piece_off_the_train_keeps_it_running(app, editor):
    """删的不是列车压着的那一节，列车应当继续跑，不误伤。"""
    close_a_circle(editor)
    view = editor.spawn_train(1)
    editor.push_train_handle(1.0)
    for _ in range(30):
        editor._advance_train(0.1)

    # 找一节**不在车身之下**的轨道
    off_piece = next(
        seg.piece_index for seg in view._driving.segments
        if not view.occupies_piece(seg.piece_index)
    )
    editor.hover_piece = off_piece
    editor.delete_hovered()
    assert editor.train_view is not None, "删到车身之外，列车不该下轨"

    for _ in range(30):
        editor._advance_train(0.1)
    assert editor.train_view is not None and editor.train_view.visible


def test_help_text_documents_the_train_keys(app, editor):
    editor._refresh_hud()
    help_text = editor.hud.texts["help"]
    for token in ("N 上列车", "↑↓ 手柄", "空格", "K 换向"):
        assert token in help_text, f"帮助里没提 {token}"


# --------------------------------------------------------------------------- #
# 布景模式：手工摆放房屋 / 车站 / 大山 / 树
# --------------------------------------------------------------------------- #

def test_b_key_enters_scenery_mode(app, editor):
    editor.bind()
    assert editor.building
    app.messenger.send("b")
    assert editor.placing_scenery
    assert not editor.building
    assert "布景" in editor._toast
    app.messenger.send("b")
    assert editor.building and not editor.placing_scenery


def test_scenery_mode_places_a_house_on_the_ground(app, editor):
    editor.toggle_scenery_mode()
    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))
    editor._refresh_ghost()
    assert editor._scenery_ghost is not None, "布景模式下该有幽灵预览"

    result = editor.place_scenery()
    assert result == ("houses", 0)
    assert len(editor.user_scenery.houses) == 1
    house = editor.user_scenery.houses[0]
    # 世界↔屏幕往返有 float32 误差，位置只需落在毫米级以内
    assert house.x == pytest.approx(0.0, abs=1e-3)
    assert house.z == pytest.approx(0.0, abs=1e-3)


def test_scenery_mode_cycles_and_places_a_classical_station(app, editor):
    editor.toggle_scenery_mode()
    assert editor.scenery_category == "building"
    editor.cycle_scenery_category(1)                   # 建筑 → 车站
    assert editor.scenery_category == "station"
    assert editor.current_scenery_kind() == "classical"
    aim_world(editor, app, (10.0, style.GROUND_Y, 20.0))
    field, index = editor.place_scenery()
    assert field == "stations"
    assert editor.user_scenery.stations[index].style == scenery_mod.STATION_CLASSICAL
    # 老式车站自带站台
    assert editor.user_scenery.stations[index].with_platform


def test_scenery_number_key_selects_item_in_category(app, editor):
    """布景模式下数字键选的是布景物件，而不是轨道件。"""
    editor.bind()
    editor.toggle_scenery_mode()
    app.messenger.send("3")                            # 建筑分类 → 公寓
    assert editor.current_scenery_kind() == "block"
    app.messenger.send("6")                            # → 商场
    assert editor.current_scenery_kind() == "mall"
    editor.cycle_scenery_category(1)                   # → 车站
    app.messenger.send("2")                            # 车站分类 → 新式车站
    assert editor.current_scenery_kind() == "modern"


def test_scenery_hover_picks_and_delete_removes(app, editor):
    editor.toggle_scenery_mode()
    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))
    editor.place_scenery()
    assert len(editor.user_scenery.houses) == 1

    editor._refresh_hover()
    assert editor.hover_scenery == ("user", "houses", 0), "落下的房屋应当能被拾取到"

    editor.delete_scenery_under_cursor()
    assert len(editor.user_scenery.houses) == 0


def test_base_scenery_house_can_be_deleted_and_undone(app, editor):
    """场景自带的房子也能在布景模式里删掉，撤销后回来。"""
    from render.scenery import House, Scenery

    editor.base_scenery = Scenery(houses=(
        House(x=5.0, z=5.0, width=8.0, depth=6.0, height=4.0, seed=1),
    ))
    editor._rebuild_scenery()
    editor.toggle_scenery_mode()
    aim_world(editor, app, (5.0, style.GROUND_Y, 5.0))
    editor._refresh_hover()
    assert editor.hover_scenery == ("base", "houses", 0)

    editor.delete_scenery_under_cursor()
    assert len(editor.base_scenery.houses) == 0
    assert editor._base_scenery_edited

    editor.undo()
    assert len(editor.base_scenery.houses) == 1


def test_scenery_placement_is_undoable_and_redoable(app, editor):
    editor.toggle_scenery_mode()
    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))
    editor.place_scenery()
    assert len(editor.user_scenery.houses) == 1

    editor.undo()
    assert len(editor.user_scenery.houses) == 0
    editor.redo()
    assert len(editor.user_scenery.houses) == 1


def test_user_scenery_survives_a_save_and_load(app, editor):
    editor.toggle_scenery_mode()
    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))
    editor.place_scenery()
    assert editor.save()

    editor.clear()
    assert not editor.user_scenery
    assert editor.load()
    assert len(editor.user_scenery.houses) == 1


def test_scenery_catalog_has_mountains_trees_lakes_and_meadows(app, editor):
    """布景模式须含山 / 树 / 湖 / 绿地，且每档都能放下。"""
    from app.editor import SCENERY_CATEGORIES, SCENERY_ITEMS

    assert "mountain" in SCENERY_CATEGORIES
    assert "tree" in SCENERY_CATEGORIES
    assert "lake" in SCENERY_CATEGORIES
    assert "meadow" in SCENERY_CATEGORIES
    assert len(SCENERY_ITEMS["mountain"]) >= 5
    assert len(SCENERY_ITEMS["tree"]) >= 6
    assert len(SCENERY_ITEMS["lake"]) == 5
    assert len(SCENERY_ITEMS["meadow"]) == 5

    editor.toggle_scenery_mode()
    aim_world(editor, app, (0.0, style.GROUND_Y, 0.0))

    # 山 → 大山
    while editor.scenery_category != "mountain":
        editor.cycle_scenery_category(1)
    field, _ = editor.place_scenery()
    assert field == "peaks"
    assert editor.user_scenery.peaks[0].kind == scenery_mod.PEAK_MOUNTAIN

    # 树 → 云杉（第 5 件）
    while editor.scenery_category != "tree":
        editor.cycle_scenery_category(1)
    editor.select_scenery_kind(4)
    field, _ = editor.place_scenery()
    assert field == "trees"
    assert editor.user_scenery.trees[-1].kind == scenery_mod.TREE_SPRUCE

    # 湖 → 中湖（第 3 件）
    while editor.scenery_category != "lake":
        editor.cycle_scenery_category(1)
    editor.select_scenery_kind(2)
    field, _ = editor.place_scenery()
    assert field == "lakes"
    assert editor.user_scenery.lakes[-1].size == "lake_m"

    # 绿地 → 大草原（第 5 件）
    while editor.scenery_category != "meadow":
        editor.cycle_scenery_category(1)
    editor.select_scenery_kind(4)
    field, _ = editor.place_scenery()
    assert field == "meadows"
    assert editor.user_scenery.meadows[-1].size == "meadow_xl"



