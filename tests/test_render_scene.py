"""``render.scene.LayoutView`` 的测试：布局 → 场景图的增量同步，以及"能不能开车"。

为什么这一层以前漏测、现在必须补
--------------------------------------------------------------------
``LayoutView`` 是 core 与画面之间唯一的一层，它同时决定了三件事：

1. **画什么**：每件轨道的节点是否存在、位姿对不对；
2. **增量对不对**：删掉一件、挪动一件之后，旧节点有没有真的被摘掉；
3. **能不能开车**：``view.path`` 是编辑器交给 :class:`render.train_view.TrainView`
   的唯一输入 —— 它为 ``None`` 时列车只会说一句"还没有可行驶的轨道"。

第三条正是踩过的坑：``recompute_closure`` 一开始直接拿
``detect_closure`` 的返回值当路径，而那个函数在**没闭合成环**时按定义返回
``None``。于是"线还没铺完就先跑一段"这个 TrainView 明确支持、还专门写了
开链限位代码的玩法，在真游戏里根本走不到。这里用一个开链布局把它钉住。
"""

from __future__ import annotations

import math

import pytest

from core.geometry import Pose
from core.track.catalog import Catalog
from core.track.layout import Layout
from render.scene import LayoutView


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return Catalog.builtin()


# --------------------------------------------------------------------------- #
# 场景构造
# --------------------------------------------------------------------------- #

def build_circle(catalog: Catalog, pieces: int = 8) -> tuple[Layout, int]:
    """N 节 R40/45° 左弯轨；``pieces == 8`` 时合拢成整圆。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root("curve_r40_l45")
    index = first
    for _ in range(pieces - 1):
        index = layout.attach("curve_r40_l45", "a", (index, "b"))
    if pieces == 8:
        layout.connect((index, "b"), (first, "a"))
    return layout, first


def build_straight_chain(catalog: Catalog, count: int) -> tuple[Layout, int]:
    """``count`` 节 20 m 直轨串成的开链。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    index = first
    for _ in range(count - 1):
        index = layout.attach("straight_20", "a", (index, "b"))
    return layout, first


# --------------------------------------------------------------------------- #
# 能不能开车
# --------------------------------------------------------------------------- #

def test_closed_layout_offers_the_loop_as_the_drivable_path(app, catalog):
    layout, _ = build_circle(catalog)
    view = LayoutView(layout, app.render)
    view.sync()

    assert view.closure is not None and view.closure.closed
    assert view.path is not None and view.path.closed
    assert view.path.total_length == pytest.approx(2 * math.pi * 40.0, rel=1e-9)


def test_open_layout_still_offers_a_drivable_path(app, catalog):
    """开链也要交出路径 —— 否则"线没铺完先跑一段"在游戏里根本做不到。

    同时 ``closure.closed`` 必须仍然是 ``False``：可行驶 ≠ 能绕一圈，
    这两个结论不能因为共用一次检测就被搅在一起。
    """
    layout, _ = build_straight_chain(catalog, 5)          # 100 m 直轨
    view = LayoutView(layout, app.render)
    view.sync()

    assert view.closure is not None and not view.closure.closed
    assert view.path is not None
    assert not view.path.closed
    assert view.path.total_length == pytest.approx(100.0, rel=1e-12)


def test_open_layout_path_is_long_enough_to_hold_a_train(app, catalog, hexie):
    """开链路径的长度必须是真的弧长 —— TrainView 拿它和编组全长比大小。

    这条是上一条的**下游后果**：路径长度错一点，列车上线时就会莫名其妙地
    报"编组比线路还长"。
    """
    from render.train_view import TrainView

    layout, _ = build_straight_chain(catalog, 20)         # 400 m
    view = LayoutView(layout, app.render)
    view.sync()

    train = TrainView(hexie, app.render)
    assert view.path.total_length > train.consist_length
    train.state.s = 200.0
    assert train.sync(view.path), train.placement_error
    train.destroy()


# --------------------------------------------------------------------------- #
# 构造即渲染
# --------------------------------------------------------------------------- #

def test_constructing_a_view_renders_the_layout_immediately(app, catalog):
    """``LayoutView(layout, parent)`` 之后布局就该是**可见**的。

    这一条是补的，因为漏掉它的代价特别隐蔽：``TrackEditor`` 建好 view 之后忘了
    调 ``sync()``，于是 ``main.py --open saves/circle.json`` 里 8 节轨道、闭环
    251.327 m、HUD 一切正常，**画面上却什么都没有**。数据对、画面空，
    比直接报错难查得多。

    把"构造即渲染"做成结构保证，就不用再靠"记得调 sync"这种约定。
    """
    layout, _ = build_circle(catalog)
    view = LayoutView(layout, app.render)

    assert len(view._piece_nodes) == 8, "构造之后场景里应当已经有 8 节轨道"
    assert view.triangle_count() > 0
    assert view.closure is not None and view.closure.closed
    assert view.path is not None and view.path.closed


def test_constructing_a_view_of_an_empty_layout_is_harmless(app, catalog):
    view = LayoutView(Layout(catalog=catalog), app.render)
    assert view._piece_nodes == {}
    assert view.path is None


def test_empty_layout_has_no_path(app, catalog):
    view = LayoutView(Layout(catalog=catalog), app.render)
    assert view.path is None
    assert view.closure is not None and not view.closure.closed


# --------------------------------------------------------------------------- #
# 增量同步
# --------------------------------------------------------------------------- #

def test_one_node_per_piece_and_none_after_removal(app, catalog):
    layout, first = build_straight_chain(catalog, 4)
    view = LayoutView(layout, app.render)
    view.sync()

    assert len(view._piece_nodes) == 4
    for index in range(4):
        assert view.node_for(index) is not None
    assert view.node_for(99) is None

    # detach 会连子树一起摘掉，所以从最后一节往回删
    removed = layout.detach(3)
    assert removed == [3]
    view.sync()
    assert len(view._piece_nodes) == 3
    assert view.node_for(3) is None


def test_node_pose_follows_the_piece(app, catalog):
    """件的位姿必须原样写进节点（``sync`` 每次都重写，不是只在建的时候写一次）。"""
    layout = Layout(catalog=catalog)
    index = layout.add_root("straight_20",
                            Pose(123.0, 7.0, -45.0, math.radians(30.0)))
    view = LayoutView(layout, app.render)
    view.sync()
    node = view.node_for(index)

    pos = node.getPos(app.render)
    assert (pos.x, pos.y, pos.z) == pytest.approx((123.0, 7.0, -45.0), abs=1e-3)
    # apply_pose 写的是 H = -heading（yup 下与 core 的航向约定相反，见 render.transform）
    assert node.getHpr(app.render)[0] == pytest.approx(-30.0, abs=1e-3)

    # 再 sync 一次不该重建节点（重建就说明缓存没命中，白算一遍网格）
    view.sync()
    assert view.node_for(index) is node


def test_sync_is_idempotent(app, catalog):
    """不增不减地再同步一次，节点数、三角形数都不该变。"""
    layout, _ = build_straight_chain(catalog, 5)
    view = LayoutView(layout, app.render)
    view.sync()
    nodes = view.triangle_count()
    view.sync()
    assert view.triangle_count() == nodes
    assert len(view._piece_nodes) == 5


def test_identical_pieces_share_one_geometry(app, catalog):
    """同型件之间必须共享 Geom（``copyTo`` 实例化），否则每节轨道都存一份顶点。"""
    layout, _ = build_straight_chain(catalog, 6)
    view = LayoutView(layout, app.render)
    view.sync()

    geoms = {view.node_for(i).node().getGeom(0) for i in range(6)}
    assert len(geoms) == 1


# --------------------------------------------------------------------------- #
# 叠加显示：彩带只在真闭环时出现
# --------------------------------------------------------------------------- #

def test_loop_ribbon_only_appears_for_a_closed_loop(app, catalog):
    """彩带是"能绕一圈"的视觉承诺 —— 开链上出现就是骗人。"""
    open_layout, _ = build_straight_chain(catalog, 3)
    view = LayoutView(open_layout, app.render)
    view.sync()
    assert view._loop_node is None

    closed_layout, _ = build_circle(catalog)
    view.layout = closed_layout
    view.sync()
    assert view._loop_node is not None

    view.layout = open_layout
    view.sync()
    assert view._loop_node is None


def test_show_loop_can_be_turned_off(app, catalog):
    layout, _ = build_circle(catalog)
    view = LayoutView(layout, app.render)
    view.sync()
    view.set_show_loop(False)
    assert view._loop_node is None
    view.set_show_loop(True)
    assert view._loop_node is not None


def test_port_markers_follow_free_ports(app, catalog):
    layout, _ = build_straight_chain(catalog, 3)
    view = LayoutView(layout, app.render)
    view.sync()
    # 3 节开链：两端各一个空闲端口
    assert view._port_markers.getNumChildren() == 2

    markers = {c.getName() for c in view._port_markers.getChildren()}
    assert markers == {"port_0_a", "port_2_b"}

    view.set_show_ports(False)
    assert view._port_markers.getNumChildren() == 0
    view.set_show_ports(True)
    assert view._port_markers.getNumChildren() == 2


def test_hover_marker_survives_the_first_sync(app, catalog):
    """悬停高亮必须活过 ``sync`` —— 它挂在与端口标记**分开**的一支上。

    以前端口标记的重建是"清空 ``_ports`` 下所有子节点，跳过悬停标记"，而那个
    跳过靠 ``child is self._hover_marker``：Panda3D 的 ``getChildren()`` 每次返回
    新的 Python 包装对象，``is`` 永远为假，于是第一次 ``sync()`` 就把悬停标记
    删了。现场表现只是"吸附目标的高亮不出现"，不抛异常、不打日志。
    """
    layout, _ = build_straight_chain(catalog, 3)
    view = LayoutView(layout, app.render)
    view.sync()

    view.set_hover_port((0, "a"))
    assert not view._hover_marker.isHidden()
    assert not view._hover_marker.getParent().isEmpty(), "悬停标记被 sync 删掉了"
    # 而且它真的挪到了那个端口上
    got = tuple(view._hover_marker.getPos(app.render))
    want = tuple(layout.world_port(0, "a").position)
    assert math.dist(got, want) < 1e-3

    view.set_hover_port(None)
    assert view._hover_marker.isHidden()


def test_hover_marker_hides_when_the_target_is_gone(app, catalog):
    """悬停目标失效时必须**隐藏**，而不是留在上一个位置骗人。"""
    layout, _ = build_straight_chain(catalog, 2)
    view = LayoutView(layout, app.render)
    view.sync()
    view.set_hover_port((0, "a"))
    assert not view._hover_marker.isHidden()

    view.set_hover_port((99, "a"))         # 根本没有这一件
    assert view._hover_marker.isHidden()
