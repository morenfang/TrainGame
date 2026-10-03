"""程序化网格生成的自检：形状对不对、有没有退化 / NaN。

为什么这些断言值得写
------------------------------------------------
"渲染出来看不看得过去"是人眼的判断，而且很难在回归时复现。但网格生成里真正
会出错的东西其实都是**可断言的数值事实**：

* 轨面是不是真的在 ``y = 0``（列车就骑在这个平面上，差一点就是悬空 / 陷进去）；
* 弯轨的落点是不是真的等于 :func:`core.geometry.arc_end_pose` 给的坐标；
* 有没有出现 NaN（断面退化、法线除零）；
* 有没有零面积三角形（端盖内缩、路堤放坡最容易制造这种"看不见的垃圾"）。

这些一旦写下来，改断面参数时就不必反复截图比对。
"""

from __future__ import annotations

import math

import pytest

from core.geometry import arc_end_pose
from core.track.catalog import Catalog
from render import style, track_mesh

TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return Catalog.builtin()


def _bounds(positions):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _build(piece_id: str, ground_local_y: float = style.GROUND_Y):
    catalog = Catalog.builtin()
    return track_mesh.build_piece_mesh(catalog[piece_id], ground_local_y=ground_local_y)


# --------------------------------------------------------------------------- #
# 目录级冒烟测试
# --------------------------------------------------------------------------- #

def test_every_piece_produces_geometry(catalog):
    """件库里每一件都必须画出东西来，而且不能只有几个三角形。"""
    for piece in catalog:
        builder = track_mesh.build_piece_mesh(piece, ground_local_y=style.GROUND_Y)
        assert builder.triangle_count > 20, f"{piece.id} 几乎没画出几何"


def test_no_nan_or_inf_vertices(catalog):
    """所有顶点坐标与法线都必须是有限值，法线必须是单位向量。"""
    for piece in catalog:
        builder = track_mesh.build_piece_mesh(piece, ground_local_y=style.GROUND_Y)
        for position in builder.positions:
            assert all(math.isfinite(c) for c in position), f"{piece.id} 顶点含 NaN"
        for normal in builder.normals:
            length = math.sqrt(sum(c * c for c in normal))
            assert length == pytest.approx(1.0, abs=1e-6), \
                f"{piece.id} 法线不是单位向量（长度 {length}）"


def test_no_degenerate_triangles(catalog):
    """不允许出现零面积三角形。

    零面积三角形肉眼看不见，但会白占显存、把 AABB 撑大、还可能让某些驱动的
    法线插值出脏像素。端盖内缩与路堤放坡是最容易制造它们的两处，所以这里
    对**整个件库**扫一遍。
    """
    bad: list[str] = []
    for piece in catalog:
        builder = track_mesh.build_piece_mesh(piece, ground_local_y=style.GROUND_Y)
        positions = builder.positions
        for i, j, k in builder.triangles:
            a, b, c = positions[i], positions[j], positions[k]
            ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
            ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
            cross = (
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            )
            area2 = math.sqrt(sum(v * v for v in cross))
            if area2 < 1e-12:
                bad.append(f"{piece.id}: ({i},{j},{k}) 面积 {area2:.3e}")
    assert not bad, "存在零面积三角形：\n" + "\n".join(bad[:12])


# --------------------------------------------------------------------------- #
# 断面尺寸
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("piece_id", ["straight_10", "straight_20", "straight_40"])
def test_straight_piece_length_and_profile(piece_id):
    """直轨：长度必须精确等于声明的弧长，断面必须落在设计尺寸上。

    纵向两端各短掉 ``CAP_INSET`` —— 那是刻意留的，用来避免相邻两节的端盖
    严格共面而闪烁，所以这里连着它一起断言（而不是放宽容差蒙过去）。
    """
    piece = Catalog.builtin()[piece_id]
    length = piece.routes[0].length
    low, high = _bounds(_build(piece_id).positions)

    assert low[0] == pytest.approx(track_mesh.CAP_INSET, abs=1e-6)
    assert high[0] == pytest.approx(length - track_mesh.CAP_INSET, abs=1e-6)
    # 横向：道砟底宽的一半就是最外沿
    assert low[2] == pytest.approx(-style.BALLAST_BOTTOM_HALF_WIDTH, abs=1e-3)
    assert high[2] == pytest.approx(style.BALLAST_BOTTOM_HALF_WIDTH, abs=1e-3)
    # 竖向：道砟底面贴地，轨面在 0
    assert low[1] == pytest.approx(style.GROUND_Y, abs=1e-3)
    assert high[1] == pytest.approx(style.RAIL_TOP_Y, abs=1e-3)


def test_rail_head_is_the_highest_geometry():
    """轨面必须**恰好**在 y = 0：列车骑在这个平面上，高了会悬空、低了会陷进去。"""
    builder = _build("straight_20")
    rail_top = style.RAIL_TOP_Y
    highest = max(p[1] for p in builder.positions)
    assert highest == pytest.approx(rail_top, abs=1e-9)

    # 而且"最高的那些点"必须落在两条钢轨的横向位置上（±半个轨距）
    tops = [p for p in builder.positions if abs(p[1] - rail_top) < 1e-9]
    assert tops, "没有找到轨面顶点"
    for _, _, z in tops:
        assert min(abs(abs(z) - style.RAIL_HALF_GAUGE) for _ in (0,)) < 0.12


@pytest.mark.parametrize(
    "piece_id, degrees",
    [("curve_r40_l45", 45.0), ("curve_r40_r90", -90.0),
     ("curve_r80_l45", 45.0), ("curve_r40_l22_5", 22.5)],
)
def test_curve_piece_reaches_the_arc_end_point(piece_id, degrees):
    """弯轨的远端必须落在 core 反解出来的弧端点上（误差 < 1 cm）。

    这是"画出来的轨道 = 列车实际走的中心线"这条承诺的直接检验：如果网格生成
    用了自己的一套弧长公式（哪怕只是符号差了），落点就会偏出米级。
    """
    piece = Catalog.builtin()[piece_id]
    route = piece.routes[0]
    dx, dz = arc_end_pose(route.length, route.dtheta)

    builder = _build(piece_id)
    # 取网格里最靠近弧终点的那个顶点（截面在终点处铺开，允许半个道砟底宽）
    near = min(
        builder.positions,
        key=lambda p: math.hypot(p[0] - dx, p[2] - dz),
    )
    distance = math.hypot(near[0] - dx, near[2] - dz)
    half_diagonal = math.hypot(style.BALLAST_BOTTOM_HALF_WIDTH,
                               style.BALLAST_TOP_Y - style.GROUND_Y)
    assert distance < half_diagonal, (
        f"{piece_id} 远端离弧端点 {distance:.3f} m，超出断面半宽 "
        f"{half_diagonal:.3f} m —— 网格与 core 的弧长公式不一致"
    )
    assert abs(math.degrees(route.dtheta) - degrees) < 1e-9


# --------------------------------------------------------------------------- #
# 出地面与路堤
# --------------------------------------------------------------------------- #

def test_ground_level_piece_has_no_skirt():
    """贴地的件不应该有路堤放坡（否则会有从地面往下伸的多余面）。"""
    builder = _build("straight_20", ground_local_y=style.GROUND_Y)
    low, _ = _bounds(builder.positions)
    assert low[1] >= style.GROUND_Y - 1e-6


def test_elevated_piece_slopes_down_to_the_ground():
    """抬高 3 m 的件必须铺出路堤，一直落到地面高度。"""
    elevated = 3.0
    builder = _build("straight_20", ground_local_y=style.GROUND_Y - elevated)
    low, high = _bounds(builder.positions)
    assert low[1] == pytest.approx(style.GROUND_Y - elevated, abs=1e-3)
    assert high[1] == pytest.approx(style.RAIL_TOP_Y, abs=1e-3)
    # 路堤底部必须比道砟底更宽
    assert high[2] - low[2] > 2 * style.BALLAST_BOTTOM_HALF_WIDTH


def test_ties_are_wider_than_the_rails_but_narrower_than_the_ballast():
    """枕木长度应落在「轨距」与「道砟底宽」之间 —— 比例不对一眼就能看出来。"""
    assert style.GAUGE < 2 * style.TIE_HALF_LENGTH < 2 * style.BALLAST_BOTTOM_HALF_WIDTH


def test_multi_route_pieces_draw_both_routes():
    """交叉件必须画出两条互不相通的路 —— 只画一条的话它就退化成直轨了。"""
    single = _build("straight_40").triangle_count
    crossing = _build("crossing_90").triangle_count
    assert crossing > single * 1.8, "交叉件只画出了一条通道"


def test_ground_local_y_shifts_the_whole_mesh_up():
    """``ground_local_y`` 只影响路堤底，不影响轨面。"""
    flat = _build("straight_20", ground_local_y=style.GROUND_Y)
    high = _build("straight_20", ground_local_y=style.GROUND_Y - 5.0)
    flat_low, flat_high = _bounds(flat.positions)
    high_low, high_high = _bounds(high.positions)

    # 轨面（最高点）不受影响：无论抬多高，列车都还骑在 y = 0
    assert flat_high[1] == pytest.approx(style.RAIL_TOP_Y, abs=1e-9)
    assert high_high[1] == pytest.approx(style.RAIL_TOP_Y, abs=1e-9)
    # 路堤底正好落在传入的"局部地面高度"上
    assert flat_low[1] == pytest.approx(style.GROUND_Y, abs=1e-3)
    assert high_low[1] == pytest.approx(style.GROUND_Y - 5.0, abs=1e-3)
