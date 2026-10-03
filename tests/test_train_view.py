"""``render.train_view`` 的测试：编组真的被摆到轨道上，而且摆得对。

为什么这一层值得单独测
--------------------------------------------------------------------
``core`` 已经把「转向架挂在哪、车体中心在哪、航向多少」全算好了，而且有自己的
测试。这一层的职责只有一件：**把那些数原样写进场景图**。凡是这类"搬运"代码，
出错方式都很隐蔽 —— 少传一个 pitch、镜像多做一次、朝后那节忘了掉头，画面看
起来都还是"一列火车"，只有几何量能戳穿它。

所以下面的断言全部落在**几何**上，而且判据都拿 core 自己的值当基准：

* 每节车的两个转向架，必须与世界坐标里 core 算出的转向架位置**重合**；
* 车头朝向必须与 core 的 heading 一致；
* 坡道上车体必须**顺着坡抬起来**（而不是横着漂）；
* 放不下时必须**说不清就不摆**（藏起来 + 给出原因），而不是抛异常。

最后一条特别重要：空场地上还没闭合的轨道是常态，"编组比线路还长"是用户很容易
造出来的状态，它必须是一句人话，而不是一个 traceback。
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from panda3d.core import Vec3

from core.geometry import Pose, normalize_angle
from core.track.catalog import Catalog
from core.track.layout import Layout
from core.track.path import PathError, detect_closure, place_consist, trace
from core.train.consist import TrainCatalog
from render.train_view import (
    TrainView,
    _best_anchored_path,
    _retrace_from_state,
    _reversal_preserves_facing,
)

#: float32 的容差：Panda3D 的 Mat4 是单精度（见 test_render_transform.py 的说明）。
TOL = 1e-3


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return Catalog.builtin()


@pytest.fixture(scope="module")
def trains() -> TrainCatalog:
    return TrainCatalog.builtin()


# --------------------------------------------------------------------------- #
# 场景构造
# --------------------------------------------------------------------------- #

def build_circle(catalog: Catalog, pieces: int = 8) -> tuple[Layout, int]:
    """N 节 R40/45° 左弯轨拼一个整圆并合拢。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root("curve_r40_l45")
    index = first
    for _ in range(pieces - 1):
        index = layout.attach("curve_r40_l45", "a", (index, "b"))
    layout.connect((index, "b"), (first, "a"))
    return layout, first


def build_ramp(catalog: Catalog) -> tuple[Layout, int]:
    """上坡引桥 + 两节高架直线（开链，坡道 3%）：全长 120 m，够摆下 8 辆编组。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root("ramp_up_40")
    index = layout.attach("straight_40", "a", (first, "b"))
    layout.attach("straight_40", "a", (index, "b"))
    return layout, first


def build_straight_chain(catalog: Catalog, count: int) -> tuple[Layout, int]:
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    index = first
    for _ in range(count - 1):
        index = layout.attach("straight_20", "a", (index, "b"))
    return layout, first


def _overpass_layout(catalog: Catalog) -> Layout:
    """复刻 overpass 场景的**轨道拓扑**（不含布景）：椭圆环 + 一条立交疏解线。

    与 ``tests/test_path.py::build_overpass_track`` 一致，道岔落在 #20、#22 上，
    疏解线挂在两处岔股之间。默认两处都扳直股。
    """
    up, down = "bridge_up_40", "bridge_down_40"
    side_a = (("straight_40",) * 2 + (up,) * 5 + ("bridge_80",) + (down,) * 5)
    side_b = (("straight_40",) * 3 + ("turnout_l_40",) + ("straight_40",)
              + ("turnout_l_40",) + ("straight_40",) * 8)

    layout = Layout(catalog=catalog)
    indices: list[int] = []
    tip = layout.add_root("curve_r40_l90")
    indices.append(tip)
    tip = layout.attach("curve_r40_l90", "a", (tip, "b"))
    indices.append(tip)
    root = indices[0]
    for def_id in side_a:
        tip = layout.attach(def_id, "a", (tip, "b"))
        indices.append(tip)
    for _ in range(2):
        tip = layout.attach("curve_r40_l90", "a", (tip, "b"))
        indices.append(tip)
    for def_id in side_b:
        tip = layout.attach(def_id, "a", (tip, "b"))
        indices.append(tip)
    layout.connect((tip, "b"), (root, "a"))

    turnouts = [i for i in indices if layout.definition(i).is_switch]
    assert turnouts == [20, 22], "overpass 轨道要两处道岔，且在 #20、#22 上"

    run = (
        ("curve_r40_r22_5",)
        + ("curve_r40_l45", "curve_r40_l45")
        + ("curve_r40_r45", "curve_r40_r45")
        + ("curve_r40_r45",) * 4
        + ("curve_r40_l22_5",)
    )
    tip = layout.attach(run[0], "a", (turnouts[0], "c"))
    for def_id in run[1:]:
        tip = layout.attach(def_id, "a", (tip, "b"))
    layout.connect((tip, "b"), (turnouts[1], "c"))
    for turnout in turnouts:
        layout.set_switch(turnout, 0)
    return layout


def loop_of(catalog: Catalog, start, layout: Layout):
    report, path = detect_closure(layout, start)
    assert report.closed, report.describe()
    return path


def local_to_world(node, point):
    """把车体局部坐标点搬到世界（走的是节点**真实的**变换矩阵）。"""
    return node.getMat().xformPoint(Vec3(*point))


def distance_to_centreline(path, point) -> float:
    """世界点 ``point`` 到轨道中心线的最短距离（米）。

    中心线是分段等曲率圆弧（直线是它的退化情形），所以这个距离有闭式解 ——
    在每一段的入口帧里，一段圆弧就是「圆心在 ``(0, R)``、半径 ``R``」的圆，
    半径 ``R = 段长 / 段转角``。逐段算、取最小即可，不需要采样：采样法的量化
    误差会直接盖住要测的毫米级量（0.01 m 步长下，最近采样点可以离真实最近点
    5 mm 远 —— 那正好和要测的数同量级）。
    """
    best = float("inf")
    for segment in path.segments:
        local = segment.entry_frame.inverse().compose(
            Pose(point[0], point[1], point[2], 0.0)
        )
        x, z = local.x, local.z
        if abs(segment.dtheta) < 1e-12:
            u = min(max(x, 0.0), segment.length)
            best = min(best, math.hypot(x - u, z))
            continue
        radius = segment.length / segment.dtheta          # 与转角同号
        # 弧上参数角 du 满足 (R sin du, R (1 - cos du))，故 du = atan2(x, R - z)
        ang = math.atan2(x, radius - z)
        ang = min(max(ang, min(0.0, segment.dtheta)), max(0.0, segment.dtheta))
        cx = radius * math.sin(ang)
        cz = radius * (1.0 - math.cos(ang))
        best = min(best, math.hypot(x - cx, z - cz))
    return best


# --------------------------------------------------------------------------- #
# 节点结构
# --------------------------------------------------------------------------- #

def test_one_node_per_car(app, catalog, trains):
    layout, first = build_circle(catalog)
    path = loop_of(catalog, (first, "a"), layout)
    spec = trains["crh380a_8"]

    view = TrainView(spec, app.render)
    assert view.car_count == spec.car_count
    assert view.root.getNumChildren() == spec.car_count
    for index in range(spec.car_count):
        assert view.node_for(index) is not None
    assert view.node_for(-1) is None
    assert view.node_for(spec.car_count) is None
    view.destroy()
    assert app.render.find(f"**/train_{spec.id}").isEmpty()


def test_same_car_type_shares_one_geometry(app, catalog, trains):
    """同型车厢必须共享 Geom（``copyTo`` 实例化），否则 8 节车会各存一份顶点。"""
    layout, first = build_circle(catalog)
    spec = trains["crh380a_8"]
    view = TrainView(spec, app.render)

    middle = [i for i, car in enumerate(spec.cars) if car.id == "crh380a_mid"]
    assert len(middle) >= 2
    geoms = {view.node_for(i).node().getGeom(0) for i in middle}
    assert len(geoms) == 1, "同型车厢没有共享 Geom"

    # 车头车尾同型（都是 crh380a_end）但朝向相反，必须各有一份几何
    ends = [i for i, car in enumerate(spec.cars) if car.id == "crh380a_end"]
    assert len(ends) == 2
    assert (view.node_for(ends[0]).node().getGeom(0)
            is not view.node_for(ends[1]).node().getGeom(0)), \
        "掉头的和没掉头的头车共用了同一份几何 —— 镜像没生效"
    view.destroy()


# --------------------------------------------------------------------------- #
# 摆位：必须与 core 算出的转向架位置重合
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("train_id", ["crh380a_8", "cr400af_8", "green_skin_10"])
def test_car_origin_and_axis_match_core_exactly(app, catalog, trains, train_id):
    """车体节点的原点必须正好落在 ``state.center``，局部 +X 必须指向 ``front``。

    这两条合起来就是"刚性车体正确地骑在两个转向架之间"的完整定义，而且**精确**
    成立（float32 量级），与轨道是不是弯的无关。位置、航向、俯仰三者中错任何一个，
    这两条都会挂。
    """
    layout, first = build_circle(catalog)
    path = loop_of(catalog, (first, "a"), layout)
    spec = trains[train_id]

    view = TrainView(spec, app.render)
    view.state.s = 40.0
    assert view.sync(path)

    for index, state in enumerate(view.states):
        node = view.node_for(index)
        origin = local_to_world(node, (0.0, 0.0, 0.0))
        assert origin.x == pytest.approx(state.center[0], abs=TOL), (train_id, index)
        assert origin.y == pytest.approx(state.center[1], abs=TOL), (train_id, index)
        assert origin.z == pytest.approx(state.center[2], abs=TOL), (train_id, index)

        drawn = local_to_world(node, (1.0, 0.0, 0.0)) - origin
        front = Vec3(*state.front.position)
        rear = Vec3(*state.rear.position)
        want = front - rear
        drawn, want = drawn / drawn.length(), want / want.length()
        assert drawn.x == pytest.approx(want.x, abs=1e-5), (train_id, index)
        assert drawn.y == pytest.approx(want.y, abs=1e-5), (train_id, index)
        assert drawn.z == pytest.approx(want.z, abs=1e-5), (train_id, index)
    view.destroy()


@pytest.mark.parametrize("train_id", ["crh380a_8", "cr400af_8", "green_skin_10"])
def test_wheels_stay_on_the_rail_centreline_around_curves(app, catalog, trains,
                                                          train_id):
    """画出来的转向架必须压在**轨道中心线**上（弯道上也要）。

    弯道上刚性车体走的是**弦**，而轨道是弧，两者长度不同 —— 弦比弧短
    ``b - R sin(b/R)``（R=40、b=3.6 时约 4.9 mm）。这一小段差额落在**纵向**上：
    转向架仍然贴着弧线，只是沿轨道方向比 core 报的弧长位置前后差几毫米。
    所以要量的是**到中心线的垂距**，它必须是亚毫米级 —— 钢轨顶面有 110 mm 宽，
    0.5 mm 的横向偏差肉眼与物理上都看不出来。
    """
    layout, first = build_circle(catalog)
    path = loop_of(catalog, (first, "a"), layout)
    spec = trains[train_id]

    view = TrainView(spec, app.render)
    view.state.s = 63.0
    assert view.sync(path)

    worst = 0.0
    for index, car in enumerate(spec.cars):
        node = view.node_for(index)
        for local_x in (car.bogie_half_spacing, -car.bogie_half_spacing):
            drawn = local_to_world(node, (local_x, 0.0, 0.0))
            worst = max(worst, distance_to_centreline(path, (drawn.x, drawn.y,
                                                             drawn.z)))
    assert worst < 1.5e-3, f"{train_id}: 转向架离轨道中心线 {worst * 1000:.3f} mm"
    view.destroy()


def test_straight_track_has_no_chord_shortfall_at_all(app, catalog):
    """直线段上没有弦长亏损，画出来的转向架必须与 core 的**逐点重合**。"""
    layout, first = build_straight_chain(catalog, 8)
    path = trace(layout, (first, "a"))
    spec = TrainCatalog.builtin()["crh380a_8"]
    view = TrainView(spec, app.render)
    view.state.s = 60.0
    assert view.sync(path)

    for index, (state, car) in enumerate(zip(view.states, spec.cars)):
        node = view.node_for(index)
        half = car.bogie_half_spacing
        for local_x, want in ((half, state.front.position),
                              (-half, state.rear.position)):
            got = local_to_world(node, (local_x, 0.0, 0.0))
            assert got.x == pytest.approx(want[0], abs=TOL)
            assert got.y == pytest.approx(want[1], abs=TOL)
            assert got.z == pytest.approx(want[2], abs=TOL)
    view.destroy()


def test_wheels_sit_on_the_railhead(app, catalog, trains):
    """转向架中心的 ``y`` 必须正好是**轨面**高度 —— 车不能浮起来也不能陷下去。"""
    layout, first = build_circle(catalog)
    path = loop_of(catalog, (first, "a"), layout)
    view = TrainView(trains["green_skin_10"], app.render)
    view.state.s = 25.0
    view.sync(path)

    for state in view.states:
        assert state.front.position[1] == pytest.approx(0.0, abs=1e-6)
        assert state.rear.position[1] == pytest.approx(0.0, abs=1e-6)
    view.destroy()


def test_car_heading_matches_core(app, catalog, trains):
    """车体局部 +X 必须就是 core 说的车头方向。"""
    layout, first = build_circle(catalog)
    path = loop_of(catalog, (first, "a"), layout)
    view = TrainView(trains["crh380a_8"], app.render)
    view.state.s = 17.5
    view.sync(path)

    for index, state in enumerate(view.states):
        node = view.node_for(index)
        forward = local_to_world(node, (1.0, 0.0, 0.0))
        origin = local_to_world(node, (0.0, 0.0, 0.0))
        got = forward - origin
        want = (math.cos(state.heading), 0.0, math.sin(state.heading))
        got = got / got.length()
        for axis in range(3):
            assert got[axis] == pytest.approx(want[axis], abs=TOL), (index, axis)
    view.destroy()


def test_consist_is_laid_out_head_first_and_does_not_overlap(app, catalog, trains):
    """编组必须首尾相接、不叠在一起：相邻两车的车钩面间距 == coupling_gap。

    在**直线**上量，因为那里没有弦长亏损，间距是精确的；弯道上量到的会带上
    弦弧差，那是另一条测试（``test_wheels_stay_on_the_rail_centreline_...``）
    在管的事。
    """
    layout, first = build_straight_chain(catalog, 8)      # 160 m
    path = trace(layout, (first, "a"))
    spec = trains["green_skin_10"]                        # 113.2 m
    view = TrainView(spec, app.render)
    view.state.s = 140.0
    assert view.sync(path)

    for index in range(spec.car_count - 1):
        head = view.node_for(index)
        tail = view.node_for(index + 1)
        # 前车的尾钩面 与 后车的首钩面
        gap_face_front = local_to_world(head, (-0.5 * spec.lengths[index], 0.0, 0.0))
        gap_face_rear = local_to_world(tail, (0.5 * spec.lengths[index + 1], 0.0, 0.0))
        gap = (gap_face_front - gap_face_rear).length()
        assert gap == pytest.approx(spec.coupling_gap, abs=TOL), index

        # 而且不能叠在一起：两车中心距必须 >= 两车半长之和
        centres = (local_to_world(head, (0.0, 0.0, 0.0))
                   - local_to_world(tail, (0.0, 0.0, 0.0))).length()
        assert centres >= 0.5 * (spec.lengths[index] + spec.lengths[index + 1])
    view.destroy()


# --------------------------------------------------------------------------- #
# 坡道：俯仰
# --------------------------------------------------------------------------- #

def single_car(spec, index: int = 0):
    """把一份编组裁成只有一节车。

    坡道件只有 40 m，而最短的编组也有 81.75 m —— 整列车根本站不上去。要看俯仰，
    只需要**一节**车，所以正当的做法是造一个一节的编组，而不是把断言放宽到
    "只要有一节车在坡上就行"（那样测的就不再是"整列车都顺着坡抬起来"了）。
    """
    return replace(spec, id=f"{spec.id}_single", cars=(spec.cars[index],),
                   coupling_gap=0.0)


def test_car_pitches_nose_up_climbing_a_ramp(app, catalog, trains):
    """上坡时车体局部 +X 必须**抬起来**（y 分量为正），坡度与轨道一致。

    这里不能用"车头节点比车尾高"去判 —— 那是位置差，不涉及姿态。要判的是
    车体的**朝向**：只有俯仰真的写对了，局部 +X 才会带上正的 y 分量。
    """
    layout, first = build_ramp(catalog)
    path = trace(layout, (first, "a"))
    assert not path.closed
    assert path.segments[0].grade == pytest.approx(0.03)
    assert path.segments[0].length == pytest.approx(40.0)

    spec = single_car(trains["crh380a_8"])
    view = TrainView(spec, app.render)
    view.state.s = 20.0                       # 车体落在 11.5 ~ 21.5 m，全在坡上
    assert view.sync(path)
    assert view.state.s == pytest.approx(20.0)      # 没有被限位挪动

    for index, state in enumerate(view.states):
        node = view.node_for(index)
        origin = local_to_world(node, (0.0, 0.0, 0.0))
        forward = local_to_world(node, (1.0, 0.0, 0.0)) - origin
        forward = forward / forward.length()
        assert forward.y > 0.0, f"第 {index} 节车下坡了（应该上坡）"
        # 坡度 3%：sin θ ≈ tan θ，小角度下两者一致到 5 位
        assert forward.y == pytest.approx(0.03, abs=2e-3), index
        assert state.pitch > 0.0
    view.destroy()


def test_car_pitches_nose_down_going_the_other_way(app, catalog, trains):
    """同一条坡道反着走，车头必须低下去 —— 俯仰的**符号**也要对。

    这条才是真正在测符号：只测"上坡抬头"的话，把俯仰写成 ``|pitch|`` 也照样能过。
    """
    layout = Layout(catalog=catalog)
    foot = layout.add_root("straight_40")
    top = layout.attach("ramp_up_40", "a", (foot, "b"))
    path = trace(layout, (top, "b"))
    assert path.segments[0].grade == pytest.approx(-0.03)
    assert path.segments[0].reversed

    spec = single_car(trains["crh380a_8"])
    view = TrainView(spec, app.render)
    view.state.s = 20.0
    assert view.sync(path)
    for index, state in enumerate(view.states):
        node = view.node_for(index)
        origin = local_to_world(node, (0.0, 0.0, 0.0))
        forward = local_to_world(node, (1.0, 0.0, 0.0)) - origin
        forward = forward / forward.length()
        assert forward.y < 0.0, f"第 {index} 节车的俯仰符号反了"
        assert state.pitch < 0.0
    view.destroy()


def test_pitch_is_the_angle_between_the_two_bogies(app, catalog, trains):
    """坡顶（一段坡 + 一段平）上，俯仰必须正好是两转向架连线的倾角。

    这一条把"俯仰从哪来"钉死：它**不是**单独查一次坡度，而是两个转向架各自
    踩在轨道上之后连线的结果 —— 所以车头已经上了平路、车尾还在坡上时，
    车体应当是半抬的。
    """
    layout = Layout(catalog=catalog)
    foot = layout.add_root("ramp_up_40")
    layout.attach("straight_40", "a", (foot, "b"))
    path = trace(layout, (foot, "a"))

    spec = single_car(trains["crh380a_8"])
    view = TrainView(spec, app.render)
    view.state.s = 40.0            # 前转向架在 s=40（坡顶），后转向架在 s=33（坡上）
    assert view.sync(path)

    state = view.states[0]
    assert state.front.pitch == pytest.approx(0.0, abs=1e-12), "前转向架在平路上"
    assert state.rear.pitch == pytest.approx(math.atan(0.03), abs=1e-12)
    assert 0.0 < state.pitch < math.atan(0.03), \
        "跨在坡顶的车体应当半抬，而不是取任一端的坡度"
    view.destroy()


def test_flat_track_keeps_cars_level(app, catalog, trains):
    layout, first = build_circle(catalog)
    path = loop_of(catalog, (first, "a"), layout)
    view = TrainView(trains["cr400af_8"], app.render)
    view.state.s = 33.0
    view.sync(path)
    for index in range(view.car_count):
        origin = local_to_world(view.node_for(index), (0.0, 0.0, 0.0))
        forward = local_to_world(view.node_for(index), (1.0, 0.0, 0.0)) - origin
        assert abs(forward.y) < 1e-4, index
    view.destroy()


# --------------------------------------------------------------------------- #
# 摆不下的时候
# --------------------------------------------------------------------------- #

def test_consist_longer_than_the_loop_is_refused_with_a_reason(app, catalog, trains):
    """编组比线路还长：藏起来 + 说清楚，而且**不能抛异常**。

    "绿皮车 16 辆"有 176 m，而 6 节 R40/45° 只有 188 m —— 够呛；这里用更短的
    4 节（126 m）来造一个确定放不下的场景。
    """
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    index = first
    for _ in range(3):
        index = layout.attach("straight_20", "a", (index, "b"))
    path = trace(layout, (first, "a"))
    assert path.total_length == pytest.approx(80.0)

    spec = trains["green_skin_16"]
    view = TrainView(spec, app.render)
    assert view.consist_length > path.total_length

    assert view.sync(path) is False
    assert not view.visible
    reason = view.placement_error
    assert reason is not None and "比这条" in reason
    assert f"{view.consist_length:.1f}" in reason
    view.destroy()


def test_empty_layout_hides_the_train_without_crashing(app, catalog, trains):
    layout = Layout(catalog=catalog)
    path = trace(layout, (0, "a")) if False else None
    view = TrainView(trains["crh380a_8"], app.render)
    assert view.sync(path) is False
    assert not view.visible
    assert view.placement_error == "还没有可行驶的轨道"
    view.destroy()


def test_open_path_keeps_the_whole_train_on_the_track(app, catalog, trains):
    """开链上首车不能太靠前，否则车尾会伸到起点之前 —— 那时 ``pose_at`` 会抛错。

    这条同时是"限位真的生效"的证明：故意把 ``s`` 设成一个会越界的值。
    """
    layout, first = build_straight_chain(catalog, 8)     # 160 m
    path = trace(layout, (first, "a"))
    assert not path.closed
    spec = trains["crh380a_8"]                           # 81.75 m
    view = TrainView(spec, app.render)

    view.state.s = 0.0                       # 车尾会伸到 s < 0
    assert view.sync(path)
    assert view.state.s == pytest.approx(view.consist_length, abs=1e-9)
    assert len(view.states) == spec.car_count
    for state in view.states:
        assert 0.0 <= state.rear.s, state
        assert state.front.s <= path.total_length + 1e-9, state

    # 反向越界：推到线路尽头之外，必须被夹回来
    view.state.s = 10_000.0
    assert view.sync(path)
    assert view.state.s == pytest.approx(path.total_length, abs=1e-9)
    for state in view.states:
        assert 0.0 <= state.rear.s, state

    # 真正会抛错的调用（用来确认上面确实是在防一个真问题，而不是防空气）
    with pytest.raises(PathError):
        place_consist(path, head_s=0.0, lengths=spec.lengths,
                      bogie_half_spacing=spec.bogie_half_spacings,
                      coupling_gap=spec.coupling_gap)
    view.destroy()


def test_closed_loop_wraps_the_head_arc_length(app, catalog, trains):
    """闭环上 ``s`` 越界必须绕回，而不是被夹住。"""
    layout, first = build_circle(catalog)
    path = loop_of(catalog, (first, "a"), layout)
    total = path.total_length
    view = TrainView(trains["crh380a_8"], app.render)

    view.state.s = total + 12.5
    assert view.sync(path)
    assert view.state.s == pytest.approx(12.5, abs=1e-9)

    view.state.s = -7.5
    assert view.sync(path)
    assert view.state.s == pytest.approx(total - 7.5, abs=1e-9)
    view.destroy()


# --------------------------------------------------------------------------- #
# 换路径时不让列车掉头（扳道岔 / 接轨 / 删轨都会换路径）
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("flip", [False, True])
def test_re_anchoring_never_turns_the_train_around(catalog, flip):
    """**「U 键切道岔，车会突然变向」那条 bug 的核心回归测试。**

    闭环检测很可能在布局变动后交出一条起点不同、甚至**行进方向相反**的路径。
    单纯把 ``s`` 留着会让列车瞬移/倒开，所以换路径时按"上一次的世界位姿"重新
    锚定。这里喂进去的 ``source`` 故意可以是反着的路 —— 结果都必须还是**原来那个
    朝向**（位置贴住、航向不掉头）。
    """
    layout, first = build_circle(catalog)
    path = loop_of(catalog, (first, "a"), layout)
    source = path.reversed() if flip else path

    s = path.total_length * 0.37
    before = path.pose_at(s)

    chosen, chosen_s = _best_anchored_path(source, before.position, before.heading)
    after = chosen.pose_at(chosen_s)

    assert math.dist(after.position, before.position) < 0.05, "锚点位置跳了"
    assert abs(normalize_angle(after.heading - before.heading)) < math.radians(2.0), \
        "列车被折回去了（这正是用户看到的『突然变向』）"


def test_sync_onto_a_reversed_path_keeps_the_train_facing_forward(app, catalog, trains):
    """从列车角度看：同一条环线、交上来的路却是反的，它也必须接着原方向开。"""
    layout, first = build_circle(catalog)
    path = loop_of(catalog, (first, "a"), layout)
    view = TrainView(trains["green_skin_10"], app.render)

    view.state.s = path.total_length * 0.6
    assert view.sync(path)
    heading = view.states[0].heading
    position = view.states[0].center

    # 布局重建后闭环检测交上来的是反着的路（同一批物理轨道）
    assert view.sync(path.reversed())
    assert abs(normalize_angle(view.states[0].heading - heading)) < math.radians(1.0)
    assert math.dist(view.states[0].center, position) < 0.05
    view.destroy()


def test_reversal_rejected_when_it_would_face_into_a_wrong_switch_leg(catalog):
    """反向路径必须仍满足「顺向看档位」——这是 #22 岔股、#20 直股那条 bug 的根。

    立交疏解线两端是道岔 #20、#22。把 #22 扳到岔股后，列车正向走线是先直过 #20、
    再从 #22 岔股钻疏解线、最后**挤**回 #20 岔股并入主线（逆向）。这条正向路径本身
    合法；但把它反向，就会翻出「#20 从岔尖硬走进岔股」「#22 从岔尖硬走直股」——
    那两道顺向档位都不对，列车开上去就会闪、而且看着像从 #20 钻进了岔道。
    """
    layout = _overpass_layout(catalog)
    turnouts = [i for i in range(len(layout)) if layout.definition(i).is_switch]
    assert turnouts == [20, 22]

    layout.set_switch(22, 1)
    _, path = detect_closure(layout, allow_open=True)
    reversed_path = path.reversed()

    assert not _reversal_preserves_facing(path, reversed_path), \
        "反向会翻出顺向档位不对的道岔段，不该被采纳"

    # 重锚定也绝不能把这条反向路径挑出来：挑中的仍得是「#22 顺向进岔股、#20 不
    # 顺向进岔股」的那条正向路。先在直股状态取一个正向位姿当锚点。
    layout.set_switch(22, 0)
    _, forward_loop = detect_closure(layout, allow_open=True)
    anchor = forward_loop.pose_at(forward_loop.total_length * 0.25)

    layout.set_switch(22, 1)
    _, winding = detect_closure(layout, allow_open=True)
    driving, _s = _best_anchored_path(winding, anchor.position, anchor.heading)

    facing = {(seg.piece_index, seg.route_index)
              for seg in driving if not seg.reversed}
    assert (22, 1) in facing, "列车应顺向走进 #22 的岔股"
    assert (20, 1) not in facing, "列车不该顺向走进 #20 的岔股（#20 扳在直股）"


def test_retrace_keeps_the_train_heading_when_the_loop_breaks(catalog):
    """扳 #22 到岔股把主环截成「环 + 支线」时，重走线不能把列车折回头，也不能
    顺向走错腿。

    立交场景 #20=0、#22=1 时，``detect_closure`` 返回的"最长可行进路径"绕着支线
    把主环靠后那一截**反着**走。列车若恰好停在那一段（车尾落在 #22 上或其后），
    从车尾重走会在 #22 处就岔开、把车头甩到反向段（航向差 180°）；此时
    :func:`_retrace_from_state` 应改从车头所在段重走，方向必然一致，且顺向该
    #22 进岔股（route 1）、#20 只走直股（route 0）。
    """
    layout = _overpass_layout(catalog)
    _, forward = detect_closure(layout, allow_open=True)
    layout.set_switch(22, 1)

    tail_length = 81.75  # cr400af_8 / crh380a_8 的编组长
    for frac in (0.1, 0.5, 0.8, 0.85, 0.9, 0.95):
        s = forward.total_length * frac
        pose = forward.pose_at(s)
        driving, new_s = _retrace_from_state(
            layout, forward, s, pose.position, pose.heading,
            tail_length=tail_length)

        assert driving is not None
        landed = driving.pose_at(new_s)
        assert math.dist(landed.position, pose.position) < 0.05, \
            f"frac={frac}：列车位置跳了"
        assert abs(normalize_angle(landed.heading - pose.heading)) < math.radians(2.0), \
            f"frac={frac}：列车被折回头了"

        # 顺向道岔段必须尊重档位：#22 顺向进岔股（route 1），#20 只能顺向走直股
        facing = {(seg.piece_index, seg.route_index)
                  for seg in driving if not seg.reversed}
        assert (22, 1) in facing, f"frac={frac}：#22 顺向必须进岔股"
        assert (20, 1) not in facing, \
            f"frac={frac}：列车居然从 #20 顺向走进了岔股"


def test_train_does_not_flip_when_a_switch_breaks_the_loop(app, catalog, trains):
    """整条链走一遍：扳 #22 之后，列车既不闪现也不折回，更不能从 #20 挤进岔股。

    这一条是用户现场那句「#22 扳到岔路，车还会闪现、还从 #20 走进岔道」的端到端
    回归：列车上轨 → 停在主环靠后那一截（旧方案会折 180° 的地方）→ 扳 #22 →
    重新 ``sync``，列车的位姿与航向都必须原样保持。
    """
    layout = _overpass_layout(catalog)
    _, loop = detect_closure(layout, allow_open=True)
    view = TrainView(trains["crh380a_8"], app.render)

    view.state.s = loop.total_length * 0.85
    assert view.sync(loop, layout)
    heading = view.states[0].heading
    center = view.states[0].center

    layout.set_switch(22, 1)
    _, new_path = detect_closure(layout, allow_open=True)
    assert view.sync(new_path, layout)

    assert abs(normalize_angle(view.states[0].heading - heading)) \
        < math.radians(2.0), "扳道岔后列车被折回头了（正是用户看到的『闪现/变向』）"
    assert math.dist(view.states[0].center, center) < 0.05, "扳道岔后列车瞬移了"
    view.destroy()


def test_train_takes_the_bypass_when_22_is_set_to_diverging(app, catalog, trains):
    """扳 #22 到岔股后，列车必须顺向从 #22 走进岔股（route 1），而不是直着冲过去。

    这是用户「#22 扳到 route 1，列车没进 route 1」的直接回归：把车摆在主环靠后
    那一截（旧实现会折 180°、反向从 #22 逆向挤回主线），扳 #22 之后一路开，列车
    必须顺向先过 #20 直股、再顺向从 #22 进岔股。
    """
    layout = _overpass_layout(catalog)
    _, loop = detect_closure(layout, allow_open=True)
    view = TrainView(trains["crh380a_8"], app.render)
    view.set_handle(1.0)

    view.state.s = loop.total_length * 0.8
    assert view.sync(loop, layout)

    layout.set_switch(22, 1)
    _, new_path = detect_closure(layout, allow_open=True)
    assert view.sync(new_path, layout)

    facing_order: list[tuple[int, int]] = []
    for _ in range(1500):
        view.advance(0.1, new_path, layout)
        segment = view._driving.segment_at(view.state.s)[0]
        if segment.is_switch and not segment.reversed:
            key = (segment.piece_index, segment.route_index)
            if not facing_order or facing_order[-1] != key:
                facing_order.append(key)
        if (22, 1) in facing_order:
            break

    assert (22, 1) in facing_order, "列车始终没有顺向从 #22 走进岔股"
    assert (20, 1) not in facing_order, "列车不该顺向从 #20 挤进岔股"
    view.destroy()


# --------------------------------------------------------------------------- #
# 手柄与运行
# --------------------------------------------------------------------------- #

def test_handle_splits_into_throttle_and_brake(app, trains):
    view = TrainView(trains["crh380a_8"], app.render)
    view.set_handle(0.6)
    assert view.state.throttle == pytest.approx(0.6)
    assert view.state.brake == 0.0
    assert view.handle == pytest.approx(0.6)

    view.set_handle(-0.4)
    assert view.state.brake == pytest.approx(0.4)
    assert view.state.throttle == 0.0
    assert view.handle == pytest.approx(-0.4)

    view.set_handle(0.0)
    assert view.handle == 0.0
    view.destroy()


def test_nudge_handle_clamps_to_the_ends(app, trains):
    view = TrainView(trains["crh380a_8"], app.render)
    for _ in range(50):
        view.nudge_handle(0.1)
    assert view.handle == pytest.approx(1.0)
    for _ in range(50):
        view.nudge_handle(-0.1)
    assert view.handle == pytest.approx(-1.0)
    view.destroy()


def test_advance_moves_the_train_along_the_loop(app, catalog, trains):
    layout, first = build_circle(catalog)
    path = loop_of(catalog, (first, "a"), layout)
    view = TrainView(trains["green_skin_10"], app.render)
    view.state.s = 10.0
    view.sync(path)

    before = view.states[0].front.position
    view.set_handle(1.0)
    for _ in range(60):
        assert view.advance(0.1, path)
    assert view.state.v > 0.0
    assert view.state.s != pytest.approx(10.0)
    after = view.states[0].front.position
    assert (Vec3(*after) - Vec3(*before)).length() > 1.0, "列车没有真的动起来"

    # 摆位仍然精确：跑起来之后转向架依旧压在轨道上
    spec = trains["green_skin_10"]
    for index, car in enumerate(spec.cars):
        node = view.node_for(index)
        for local_x in (car.bogie_half_spacing, -car.bogie_half_spacing):
            drawn = local_to_world(node, (local_x, 0.0, 0.0))
            assert distance_to_centreline(path, (drawn.x, drawn.y, drawn.z)) < 1.5e-3
    view.destroy()


def test_braking_stops_the_train(app, catalog, trains):
    layout, first = build_circle(catalog)
    path = loop_of(catalog, (first, "a"), layout)
    view = TrainView(trains["cr400af_8"], app.render)
    view.set_handle(1.0)
    for _ in range(100):
        view.advance(0.1, path)
    assert view.state.v > 1.0

    view.set_handle(-1.0)
    for _ in range(500):
        view.advance(0.1, path)
    assert view.state.v == pytest.approx(0.0, abs=1e-9)
    view.destroy()


def test_triangle_count_covers_every_car(app, trains):
    """三角形统计要按节数算 —— 屏幕上画的就是这么多。"""
    spec = trains["green_skin_10"]
    view = TrainView(spec, app.render)
    single = TrainView(trains["green_skin_10"], app.render)
    assert single.triangle_count() > 0
    assert view.triangle_count() == single.triangle_count()
    view.destroy()
    single.destroy()


def test_speed_readout_is_kmh(app, trains):
    view = TrainView(trains["crh380a_8"], app.render)
    view.state.v = 10.0
    assert view.speed_kmh() == pytest.approx(36.0)
    view.destroy()
