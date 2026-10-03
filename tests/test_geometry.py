"""核心几何（``core/geometry.py``）的回归测试。

这里验证的是全项目的地基，重点有两类：

1. **闭式解本身对不对** —— 用独立的数值积分去对撞它，而不是拿它自己验自己。
2. **闭环与 8 字的精度** —— 这是概要设计里 G1 验收线的量化依据。
"""

from __future__ import annotations

import math

import pytest

from core.geometry import (
    InconsistentRouteError,
    Pose,
    arc_end_pose,
    arc_point,
    heading_between,
    normalize_angle,
    pitch_between,
    route_geometry,
)

DEG = math.radians


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #

def walk(start: Pose, segments) -> tuple[Pose, list[Pose]]:
    """从 ``start`` 连续走若干 ``(弧长, 总转角)`` 弧段，返回 (终点, 各段终点列表)。"""
    pose = start
    ends = []
    for length, dtheta in segments:
        dx, dz = arc_end_pose(length, dtheta)
        pose = pose.compose(Pose(dx, 0.0, dz, dtheta))
        ends.append(pose)
    return pose, ends


def circumcenter_2d(points) -> tuple[float, float]:
    """三点外接圆心（(x, z) 平面）。用于独立求弧的圆心。"""
    (ax, az), (bx, bz), (cx, cz) = points
    d = 2.0 * (ax * (bz - cz) + bx * (cz - az) + cx * (az - bz))
    assert abs(d) > 1e-12, "three points are collinear"
    sa = ax * ax + az * az
    sb = bx * bx + bz * bz
    sc = cx * cx + cz * cz
    ux = (sa * (bz - cz) + sb * (cz - az) + sc * (az - bz)) / d
    uz = (sa * (cx - bx) + sb * (ax - cx) + sc * (bx - ax)) / d
    return ux, uz


# --------------------------------------------------------------------------- #
# 角度归一化
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw, expected",
    [
        (0.0, 0.0),
        (math.pi, math.pi),
        (-math.pi, math.pi),
        (math.pi / 2, math.pi / 2),
        (3 * math.pi / 2, -math.pi / 2),
        (2 * math.pi, 0.0),
        (math.pi + 0.1, -math.pi + 0.1),
        (10 * math.pi, 0.0),
    ],
)
def test_normalize_angle(raw, expected):
    assert normalize_angle(raw) == pytest.approx(expected, abs=1e-12)


def test_normalize_angle_wraps_just_past_the_boundary():
    """略大于 +pi 的值应折到略大于 -pi（这是 route_geometry 反解转角时的常见情形）。"""
    raw = math.pi + 1e-12
    got = normalize_angle(raw)
    assert got == pytest.approx(-math.pi + 1e-12, abs=1e-11)
    assert got > -math.pi


def test_normalize_angle_is_idempotent():
    for raw in [-7.3, -0.1, 0.0, 0.5, 3.0, 6.29, 100.0]:
        once = normalize_angle(raw)
        assert normalize_angle(once) == pytest.approx(once, abs=1e-15)


# --------------------------------------------------------------------------- #
# 闭式解的正确性：与独立数值积分对撞
# --------------------------------------------------------------------------- #

def integrate_numerically(length: float, dtheta: float, steps: int = 200_000):
    """用中点法独立积分 dp/ds = (cos θ, sin θ)，θ(s) 沿弧线性变化。

    这是对闭式解的**独立**检验（不走同一套公式）。
    """
    ds = length / steps
    x = z = 0.0
    for i in range(steps):
        s_mid = (i + 0.5) * ds
        theta = dtheta * (s_mid / length)
        x += ds * math.cos(theta)
        z += ds * math.sin(theta)
    return x, z


@pytest.mark.parametrize(
    "length, dtheta_deg",
    [
        (20.0, 0.0),
        (20.0, 1.0),
        (20.0, 22.5),
        (20.0, 45.0),
        (20.0, 90.0),
        (20.0, -45.0),
        (20.0, -90.0),
        (20.0, 179.0),
    ],
)
def test_arc_end_pose_matches_numerical_integration(length, dtheta_deg):
    dtheta = DEG(dtheta_deg)
    dx, dz = arc_end_pose(length, dtheta)
    nx, nz = integrate_numerically(length, dtheta)
    # 中点法误差 O(ds^2)，20 m / 2e5 步时远小于 1e-9
    assert dx == pytest.approx(nx, abs=1e-9)
    assert dz == pytest.approx(nz, abs=1e-9)


def test_arc_end_pose_degenerates_to_straight_line():
    for dtheta in [0.0, 1e-12, -1e-12]:
        dx, dz = arc_end_pose(20.0, dtheta)
        assert dx == pytest.approx(20.0, abs=1e-11)
        assert dz == pytest.approx(0.0, abs=1e-11)


def test_arc_end_pose_small_angle_has_no_cancellation_error():
    """1e-5 rad 落在级数分支内，dz 必须仍然准确（这是 1-cos 抵消的经典坑）。"""
    dtheta = 1e-5
    dx, dz = arc_end_pose(20.0, dtheta)
    assert dx == pytest.approx(20.0, abs=1e-9)
    assert dz == pytest.approx(20.0 * dtheta / 2, rel=1e-9)


def test_arc_point_endpoint_agrees_with_arc_end_pose():
    length, dtheta = 20.0, DEG(45.0)
    ex, ez = arc_end_pose(length, dtheta)
    px, pz, ph = arc_point(length, dtheta, length)
    assert (px, pz) == pytest.approx((ex, ez), abs=1e-12)
    assert ph == pytest.approx(dtheta, abs=1e-12)


def test_arc_point_is_linear_in_turn_angle():
    """弧长比例 == 转角比例（等曲率的直接推论）。"""
    length, dtheta = 20.0, DEG(90.0)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _, _, turn = arc_point(length, dtheta, length * frac)
        assert turn == pytest.approx(dtheta * frac, abs=1e-12)


# --------------------------------------------------------------------------- #
# 位姿代数
# --------------------------------------------------------------------------- #

def test_compose_then_inverse_is_identity():
    poses = [
        Pose.origin(),
        Pose(3.0, 1.0, -2.0, DEG(37.0)),
        Pose(-5.0, 0.0, 8.0, DEG(-120.0)),
        Pose(0.0, 2.0, 0.0, math.pi),
    ]
    for p in poses:
        assert p.compose(p.inverse()).approx_equal(Pose.origin(), 1e-12, 1e-12)
        assert p.inverse().compose(p).approx_equal(Pose.origin(), 1e-12, 1e-12)


def test_compose_rotates_local_offset_into_world():
    """heading=90° 的父帧，其局部 +x 应当指向世界 +z。"""
    parent = Pose(10.0, 0.0, 5.0, DEG(90.0))
    child = Pose(2.0, 0.0, 0.0, 0.0)
    world = parent.compose(child)
    assert (world.x, world.z) == pytest.approx((10.0, 7.0), abs=1e-12)
    assert world.heading == pytest.approx(DEG(90.0), abs=1e-12)


def test_compose_preserves_local_y():
    """y 只做平移（本设计的坡度近似依赖这一点）。"""
    parent = Pose(1.0, 3.0, 2.0, DEG(45.0))
    world = parent.compose(Pose(4.0, 1.5, 0.0, 0.0))
    assert world.y == pytest.approx(4.5, abs=1e-12)


def test_compose_associativity():
    a = Pose(1.0, 0.0, 2.0, DEG(20.0))
    b = Pose(3.0, 1.0, -1.0, DEG(-35.0))
    c = Pose(-2.0, 0.5, 4.0, DEG(80.0))
    left = a.compose(b).compose(c)
    right = a.compose(b.compose(c))
    assert left.approx_equal(right, 1e-12, 1e-12)


def test_flipped_is_180_degree_turn_in_place():
    p = Pose(4.0, 1.0, -3.0, DEG(30.0))
    f = p.flipped()
    assert f.position == p.position
    assert abs(f.heading_error_to(p)) == pytest.approx(math.pi, abs=1e-12)


def test_translated_local_moves_along_heading():
    p = Pose(5.0, 0.0, 5.0, DEG(90.0))
    assert (p.translated_local(dx=7.0).position) == pytest.approx(
        (5.0, 0.0, 12.0), abs=1e-12
    )


def test_heading_and_pitch_between():
    assert heading_between((0, 0, 0), (1, 0, 1)) == pytest.approx(DEG(45.0))
    assert pitch_between((0, 0, 0), (1, 1, 0)) == pytest.approx(DEG(45.0))
    # 俯仰按水平距离算（不随航向变化）
    assert pitch_between((0, 0, 0), (0, 3, 0)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 闭式解 -> 反解（route_geometry），目录数据自校验的基石
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "length, dtheta_deg",
    [(20.0, 0.0), (20.0, 22.5), (20.0, 45.0), (20.0, 90.0), (24.0, -22.5), (15.0, 170.0)],
)
def test_route_geometry_round_trips_direct_spec(length, dtheta_deg):
    """正向构造端口 -> 反解必须还原回同一组 (弧长, 总转角)。"""
    dtheta = DEG(dtheta_deg)
    dx, dz = arc_end_pose(length, dtheta)
    entry = Pose.origin()
    exit_pose = Pose(dx, 0.0, dz, dtheta)
    got_length, got_dtheta = route_geometry(entry, exit_pose)
    assert got_length == pytest.approx(length, rel=1e-12)
    assert got_dtheta == pytest.approx(dtheta, abs=1e-12)


def test_route_geometry_works_relative_to_a_non_trivial_entry_frame():
    """入口帧本身有位置和航向时同样成立（真实装配中就是这样）。"""
    dtheta = DEG(45.0)
    length = 20.0
    entry = Pose(13.0, 2.0, -7.0, DEG(120.0))
    dx, dz = arc_end_pose(length, dtheta)
    exit_pose = entry.compose(Pose(dx, 0.0, dz, dtheta))
    assert route_geometry(entry, exit_pose) == pytest.approx((length, dtheta), rel=1e-12)


def test_route_geometry_rejects_off_axis_straight():
    """声明为直线、但出口偏离轴线 —— 必须报错而不是悄悄接受。"""
    with pytest.raises(InconsistentRouteError, match="off-axis"):
        route_geometry(Pose.origin(), Pose(20.0, 0.0, 3.0, 0.0))


def test_route_geometry_rejects_non_circular_port_pair():
    """任取两点 + 一个转角，若不落在同一圆弧上必须报错。"""
    with pytest.raises(InconsistentRouteError, match="single circular arc"):
        route_geometry(Pose.origin(), Pose(20.0, 0.0, 3.0, DEG(45.0)))


def test_route_geometry_rejects_backwards_route():
    with pytest.raises(InconsistentRouteError, match="forward"):
        route_geometry(Pose.origin(), Pose(-20.0, 0.0, 0.0, 0.0))


def test_route_geometry_rejects_degenerate_full_turn():
    with pytest.raises(InconsistentRouteError, match="180"):
        route_geometry(Pose.origin(), Pose(0.0, 0.0, 0.0, math.pi))


# --------------------------------------------------------------------------- #
# 闭环：G1 验收线的量化依据
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("step_deg, expected", [(22.5, 16), (45.0, 8), (90.0, 4)])
def test_circle_closes_exactly(step_deg, expected):
    """n 个同向弯轨拼成整圆 —— 必须精确闭合（G1）。"""
    radius = 40.0
    per_piece_length = radius * DEG(step_deg)
    end, _ = walk(Pose.origin(), [(per_piece_length, DEG(step_deg))] * expected)

    # 位置精确回到原点
    assert end.planar_distance_to(Pose.origin()) < 1e-12
    # 航向累计恰好 360°
    assert normalize_angle(end.heading) == pytest.approx(0.0, abs=1e-12)
    assert end.heading == pytest.approx(math.tau, abs=1e-12)


def test_circle_has_the_expected_center_and_radius():
    """用三点外接圆独立求出圆心与半径，确认它真的是半径 R 的圆。"""
    radius = 40.0
    piece_length = radius * DEG(45.0)
    segments = [(piece_length, DEG(45.0))] * 8
    _, ends = walk(Pose.origin(), segments)
    samples = [(p.x, p.z) for p in ends]
    cx, cz = circumcenter_2d([samples[0], samples[2], samples[4]])
    assert math.hypot(cx, cz - radius) == pytest.approx(0.0, abs=1e-9)


def test_figure_eight_closes_exactly():
    """8 字形 = 8 节同向 + 8 节反向。两环各自精确闭合。"""
    radius = 40.0
    piece_length = radius * DEG(45.0)
    left = [(piece_length, DEG(45.0))] * 8
    right = [(piece_length, DEG(-45.0))] * 8

    loop1_end, _ = walk(Pose.origin(), left)
    assert loop1_end.planar_distance_to(Pose.origin()) < 1e-12

    # 第二个环从「第一个环的终点」出发（即接缝处），同样必须闭合
    seam = loop1_end
    loop2_end, _ = walk(seam, right)
    assert loop2_end.planar_distance_to(seam) < 1e-12


def test_figure_eight_is_two_externally_tangent_circles():
    """两环是精确外切的圆：圆心距 == 2R，切点 == 接缝。

    这条是「8 字不需要 45° 交叉件」这一结论的数学依据。
    """
    radius = 40.0
    piece_length = radius * DEG(45.0)

    _, left_ends = walk(Pose.origin(), [(piece_length, DEG(45.0))] * 8)
    left_pts = [(p.x, p.z) for p in left_ends]
    left_center = circumcenter_2d([left_pts[0], left_pts[2], left_pts[4]])

    seam = left_ends[-1]
    _, right_ends = walk(seam, [(piece_length, DEG(-45.0))] * 8)
    right_pts = [(p.x, p.z) for p in right_ends]
    right_center = circumcenter_2d([right_pts[0], right_pts[2], right_pts[4]])

    center_distance = math.dist(left_center, right_center)
    assert center_distance == pytest.approx(2.0 * radius, abs=1e-9)

    # 两环半径相同
    for center, pts in ((left_center, left_pts), (right_center, right_pts)):
        r = math.dist(center, pts[0])
        assert r == pytest.approx(radius, abs=1e-9)

    # 接缝（切点）同时落在两环上
    assert math.dist(left_center, (seam.x, seam.z)) == pytest.approx(radius, abs=1e-9)
    assert math.dist(right_center, (seam.x, seam.z)) == pytest.approx(radius, abs=1e-9)


def test_mismatched_radius_does_not_close():
    """半径不一致（混用两规格弯轨）时必须留下显著缺口 —— 说明判据不是宽容差。"""
    a = 40.0 * DEG(45.0)
    b = 25.0 * DEG(45.0)
    end, _ = walk(Pose.origin(), [(a, DEG(45.0))] * 4 + [(b, DEG(45.0))] * 4)
    gap = end.planar_distance_to(Pose.origin())
    assert gap > 1.0  # 米级缺口，与真正闭环的 1e-14 相差 14 个数量级


def test_gap_and_closed_are_separated_by_many_orders_of_magnitude():
    """闭环判据之所以能卡到 1e-6 m：真实闭环与未闭环之间没有中间地带。"""
    radius = 40.0
    piece_length = radius * DEG(45.0)
    closed_gap = walk(Pose.origin(), [(piece_length, DEG(45.0))] * 8)[0]
    closed_gap = closed_gap.planar_distance_to(Pose.origin())

    open_gap = walk(Pose.origin(), [(piece_length, DEG(45.0))] * 7)[0]
    open_gap = open_gap.planar_distance_to(Pose.origin())

    assert closed_gap < 1e-12
    assert open_gap > 1.0
    assert open_gap / max(closed_gap, 1e-300) > 1e12
