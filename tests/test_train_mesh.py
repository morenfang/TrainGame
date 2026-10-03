"""参数化车体网格的自检：尺寸、涂装分区、走行部接触、三种车型可辨。

这一层能断言的东西比轨道那层多得多，因为车体外观是**数据**（``data/trains.json``
的 ``shape`` 段）而网格是它的**推论**。于是"色带画在哪儿""鼻锥有多长""车轮是不是
压在钢轨上"全是可算的，不必靠截图比对。

三个最值得盯住的点：

* **车轮踏面必须正好落在 ``y = 0``**（轨面）。高了列车悬空、低了陷进钢轨，
  而这个高度差在 40 m 外根本看不出来；
* **每一块四边形必须整个落在一个涂装分区里**。这是"车身只放样一次、颜色逐块
  决定"能成立的前提：边界若切在四边形中间，色带与窗口的实际长度就会随网格
  疏密漂移，而它们是数据里写死的；
* **绕序算出来的几何法线必须和顶点法线同向**。双面渲染让它暂时看不出来，
  但绕序反了意味着网格里外颠倒。
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from core.train.consist import CarSpec, Nose, TrainCatalog
from render import style, train_mesh

#: 每节车的三角形预算（概要设计 §5.8：8 辆编组要在普通显卡上跑满帧）。
TRIANGLE_BUDGET = 3000


@pytest.fixture(scope="module")
def catalog() -> TrainCatalog:
    return TrainCatalog.builtin()


@pytest.fixture(scope="module")
def meshes(catalog) -> dict[str, "train_mesh.MeshBuilder"]:
    """每个车型的完整网格（含走行部与车顶设备）。"""
    return {cid: train_mesh.build_car_mesh(car)
            for cid, car in catalog.car_types.items()}


@pytest.fixture(scope="module")
def bodies(catalog) -> dict[str, "train_mesh.MeshBuilder"]:
    """只有车身本体的网格 —— 量尺寸时用它，免得走行部把包围盒撑歪。"""
    return {cid: train_mesh.build_car_mesh(car, details=False)
            for cid, car in catalog.car_types.items()}


def _bounds(positions):
    return (
        (min(p[0] for p in positions), min(p[1] for p in positions),
         min(p[2] for p in positions)),
        (max(p[0] for p in positions), max(p[1] for p in positions),
         max(p[2] for p in positions)),
    )


def _width_profile(builder) -> list[tuple[float, float]]:
    """侧影包线：``(x, max|z|)`` 按 x 排序。"""
    profile: dict[float, float] = {}
    for x, _, z in builder.positions:
        profile[x] = max(profile.get(x, 0.0), abs(z))
    return sorted(profile.items())


def _nose_taper(car: CarSpec, builder) -> float:
    """从 +x 端面往车内走，到"宽度回到 98% 全宽"为止，占车长的比例。

    这就是"车头尖不尖"的量化定义 —— 三种车型的辨识度主要来自它。
    """
    half = 0.5 * car.length
    full = car.shape.half_width
    for x, width in reversed(_width_profile(builder)):
        if width >= 0.98 * full:
            return (half - x) / car.length
    return 1.0


def _straight(car: CarSpec) -> CarSpec:
    """把鼻锥抹掉（整节车等截面）的一份副本。

    有鼻锥时断面的高度会被缩放，顶点坐标就不再等于断面的高度，没法从几何反推
    "这一点属于哪个分区"。抹掉之后车体是一个棱柱，逐块核对才是精确的。
    """
    shape = car.shape
    return CarSpec(
        id=car.id, name=car.name, length=car.length,
        bogie_half_spacing=car.bogie_half_spacing, mass_kg=car.mass_kg,
        powered=car.powered, livery=car.livery, roof_gear=car.roof_gear,
        shape=replace(shape, nose=Nose(0.0, 1.0, 1.0, shape.nose.pivot_height)),
    )


# --------------------------------------------------------------------------- #
# 目录级冒烟
# --------------------------------------------------------------------------- #

def test_every_car_type_has_shape_data(catalog):
    """外观几何是必填项：缺了它这节车根本画不出来，只会在运行时炸。"""
    missing = [w for w in catalog.validate() if "shape" in w]
    assert not missing, missing


def test_every_car_type_produces_geometry(meshes):
    for cid, builder in meshes.items():
        assert builder.triangle_count > 100, f"{cid} 几乎没画出几何"


def test_triangle_budget(meshes):
    """每节车不超过预算 —— 8 辆编组的三角形上限就直接由它决定。"""
    for cid, builder in meshes.items():
        assert builder.triangle_count <= TRIANGLE_BUDGET, (
            f"{cid} 用了 {builder.triangle_count} 个三角形，超过预算 {TRIANGLE_BUDGET}"
        )


def test_no_nan_and_unit_normals(meshes):
    for cid, builder in meshes.items():
        for position in builder.positions:
            assert all(math.isfinite(c) for c in position), f"{cid} 顶点含 NaN"
        for normal in builder.normals:
            length = math.sqrt(sum(c * c for c in normal))
            assert length == pytest.approx(1.0, abs=1e-6), \
                f"{cid} 法线不是单位向量（长度 {length}）"


def test_no_degenerate_triangles(meshes):
    """不允许零面积三角形：它们肉眼看不见，却白占显存、把包围盒撑大。"""
    bad: list[str] = []
    for cid, builder in meshes.items():
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
            if math.sqrt(sum(v * v for v in cross)) < 1e-12:
                bad.append(f"{cid}: ({i},{j},{k})")
    assert not bad, "存在零面积三角形：\n" + "\n".join(bad[:12])


def test_triangle_winding_agrees_with_vertex_normals(meshes):
    """绕序算出的几何法线必须与顶点法线同向（否则网格里外颠倒）。"""
    bad: list[str] = []
    for cid, builder in meshes.items():
        positions = builder.positions
        normals = builder.normals
        for i, j, k in builder.triangles:
            ax, ay, az = positions[i]
            bx, by, bz = positions[j]
            cx, cy, cz = positions[k]
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            ni, nj, nk = normals[i], normals[j], normals[k]
            total = (
                nx * (ni[0] + nj[0] + nk[0])
                + ny * (ni[1] + nj[1] + nk[1])
                + nz * (ni[2] + nj[2] + nk[2])
            )
            if total <= 0.0:
                bad.append(f"{cid}: ({i},{j},{k}) 点积 {total:.3e}")
    assert not bad, "绕序与法线不一致：\n" + "\n".join(bad[:12])


# --------------------------------------------------------------------------- #
# 尺寸契约
# --------------------------------------------------------------------------- #

def test_body_spans_exactly_the_declared_length(catalog, bodies):
    """车身本体必须正好跨满 :attr:`CarSpec.length`（车钩面到车钩面）。"""
    for cid, car in catalog.car_types.items():
        low, high = _bounds(bodies[cid].positions)
        half = 0.5 * car.length
        assert low[0] == pytest.approx(-half, abs=1e-6), cid
        assert high[0] == pytest.approx(half, abs=1e-6), cid


def test_body_matches_the_cross_section(catalog, bodies):
    """车身的高度与半宽必须精确等于断面的推论，一个毫米都不能多。"""
    for cid, car in catalog.car_types.items():
        shape = car.shape
        low, high = _bounds(bodies[cid].positions)
        assert low[1] == pytest.approx(shape.floor_height, abs=1e-6), cid
        assert high[1] == pytest.approx(shape.roof_height, abs=1e-6), cid
        assert high[2] == pytest.approx(shape.half_width, abs=1e-6), cid
        assert low[2] == pytest.approx(-shape.half_width, abs=1e-6), cid


def test_loading_gauge_fits_the_track(catalog, bodies):
    """车体不能比轨道还窄 —— 那会让整节车"悬在两条钢轨之间"。"""
    for cid, car in catalog.car_types.items():
        half_width = car.shape.half_width
        assert half_width > style.RAIL_HALF_GAUGE, (
            f"{cid} 半宽 {half_width:.3f} m 不到半个轨距 "
            f"{style.RAIL_HALF_GAUGE:.3f} m，车会掉进轨道里"
        )
        # 车底平面也必须比轨距宽，否则车下设备会骑在钢轨上
        assert car.shape.bottom_half_width > style.RAIL_HALF_GAUGE, cid


def test_details_only_add_geometry_below_the_floor(catalog, bodies, meshes):
    """``details=False`` 只给车身；带上走行部之后会多出地板以下的东西。"""
    for cid, car in catalog.car_types.items():
        floor = car.shape.floor_height
        assert min(p[1] for p in bodies[cid].positions) == \
            pytest.approx(floor, abs=1e-6), cid
        assert min(p[1] for p in meshes[cid].positions) < floor - 0.3, cid


# --------------------------------------------------------------------------- #
# 走行部与轨面
# --------------------------------------------------------------------------- #

def test_wheel_tread_sits_exactly_on_the_railhead(catalog, meshes):
    """整车最低点必须是 ``y = 0``（轨面）—— 列车就骑在这个平面上。"""
    for cid, builder in meshes.items():
        lowest = min(p[1] for p in builder.positions)
        assert lowest == pytest.approx(style.RAIL_TOP_Y, abs=1e-9), (
            f"{cid} 最低点在 y = {lowest:.6f}，不是轨面 {style.RAIL_TOP_Y}"
        )


def test_nothing_sinks_below_the_railhead(catalog, meshes):
    for cid, builder in meshes.items():
        lowest = min(p[1] for p in builder.positions)
        assert lowest >= style.RAIL_TOP_Y - 1e-9, cid


def test_only_the_wheels_touch_rail_level(catalog, meshes):
    """轨面高度上只应该有车轮 —— 别的任何东西碰到它都是"陷进钢轨里了"。"""
    for cid, builder in meshes.items():
        near_rail = [p for p in builder.positions if p[1] < 0.02]
        assert near_rail, cid
        inner = style.RAIL_HALF_GAUGE - style.TRAIN_WHEEL_THICKNESS
        for x, _, z in near_rail:
            assert abs(z) >= inner - 1e-6, (
                f"{cid} 在 (x={x:.2f}, z={z:.3f}) 处有几何碰到轨面高度，"
                "那里不是车轮踏面"
            )


def test_wheel_count(catalog, meshes):
    """两个转向架、每个两根轴、每根轴两个轮 —— 一共 8 个车轮着地点。"""
    gauge_half = style.RAIL_HALF_GAUGE
    inner = gauge_half - style.TRAIN_WHEEL_THICKNESS
    for cid, builder in meshes.items():
        car = catalog.car_types[cid]
        contacts = {
            (round(x, 3), 1 if z > 0 else -1)
            for x, y, z in builder.positions
            if abs(y - style.RAIL_TOP_Y) < 1e-9
        }
        assert len(contacts) == 8, f"{cid} 有 {len(contacts)} 个车轮着地点"
        for sign in (-1.0, 1.0):
            centre = sign * car.bogie_half_spacing
            for axle in (centre - style.TRAIN_AXLE_HALF_SPACING,
                         centre + style.TRAIN_AXLE_HALF_SPACING):
                assert any(abs(x - axle) < 1e-3 for x, _ in contacts), (cid, axle)


def test_wheel_tread_width_is_realistic(meshes):
    """车轮外侧面必须正好骑在钢轨中心线上（``±轨距/2``）。"""
    for cid, builder in meshes.items():
        low = style.RAIL_TOP_Y
        zs = [abs(z) for _, y, z in builder.positions if abs(y - low) < 1e-9]
        assert zs, cid
        assert max(zs) == pytest.approx(style.RAIL_HALF_GAUGE, abs=1e-6), cid
        assert min(zs) == pytest.approx(
            style.RAIL_HALF_GAUGE - style.TRAIN_WHEEL_THICKNESS, abs=1e-6
        ), cid


def test_bogie_frame_stays_clear_of_the_wheels():
    """构架的横向半宽必须小于车轮内侧面，否则两者会**实体相交**。

    这不是审美问题：相交的两个实体在画面上会出现穿插的碎面，而"差 2 cm"这种
    错误看截图是发现不了的。所以它是一条常量之间的关系，直接被断言。
    """
    assert (style.TRAIN_BOGIE_FRAME_HALF_WIDTH + style.TRAIN_WHEEL_THICKNESS
            <= style.RAIL_HALF_GAUGE)


# --------------------------------------------------------------------------- #
# 涂装分区
# --------------------------------------------------------------------------- #

def test_band_edges_are_exact_section_points(catalog):
    """色带 / 车窗 / 车门的上下沿必须真的出现在断面点里。

    这是"没有一块四边形跨在涂装边界上"的**断面侧**那一半。
    """
    for cid, car in catalog.car_types.items():
        shape = car.shape
        heights = [h for _, h in shape.split_section()]
        for edge in list(shape.band_edges()) + [shape.shoulder_height]:
            assert any(abs(h - edge) < 1e-9 for h in heights), (
                f"{cid} 的高度 {edge} 不是断面点，涂装边界会落在四边形中间"
            )


def test_feature_bands_are_exact_mesh_lines(catalog):
    """鼻锥 / 车门 / 车窗的纵向边界必须真的成为网格线（**纵向侧**那一半）。"""
    for cid, car in catalog.car_types.items():
        stations = {round(x, 9) for x in train_mesh._stations(car)}
        half = 0.5 * car.length
        for start, stop in car.feature_bands():
            for edge in (start, stop):
                assert round(edge - half, 9) in stations, (
                    f"{cid} 的纵向边界 {edge} 不是网格线，"
                    "色带 / 窗口的实际长度会随网格疏密漂移"
                )


@pytest.mark.parametrize("car_id", ["df4b_loco", "coach_25b", "crh380a_mid",
                                    "crh380a_end", "cr400af_mid"])
def test_paint_role_is_constant_across_every_quad(catalog, car_id):
    """逐块扫一遍：没有任何一块四边形**内部**横跨两个分区。

    把每块四边形再细分 7 段，每一小段用 :func:`quad_role` 判一次 —— 全部相同才
    说明"用中点决定整块的颜色"是精确的。这正是涂装能跟着一次放样一起画出来的
    前提。
    """
    car = catalog.car_types[car_id]
    section = car.shape.split_section()
    heights = [h for _, h in section]
    count = len(section)
    stations = train_mesh._stations(car)
    half = 0.5 * car.length
    slices = 7
    for k in range(len(stations) - 1):
        u = 0.5 * (stations[k] + stations[k + 1]) + half
        for i in range(count):
            low = heights[i]
            high = heights[(i + 1) % count]
            for step in range(slices):
                a = low + (high - low) * step / slices
                b = low + (high - low) * (step + 1) / slices
                roles = {train_mesh.quad_role(car, u, a, b)}
                assert roles == {train_mesh.quad_role(car, u, low, high)}, (
                    f"{car_id} 在 u={u:.3f} 的四边形 [{low}, {high}] 里，"
                    f"子区间 [{a:.4f}, {b:.4f}] 的分区是 {roles}，"
                    "和整块的不一致 —— 分区边界切在四边形中间了"
                )


def test_underframe_role_is_geometry_not_height(catalog):
    """车底那片水平面必须被认出来：它的高度和侧板下沿完全一样。"""
    for cid, car in catalog.car_types.items():
        floor = car.shape.floor_height
        assert train_mesh.quad_role(car, 0.5 * car.length, floor, floor) == \
            "underframe", cid
        # 同样是"下沿在车底高度"，但上沿高一点 —— 那就是侧板，不是车底
        assert train_mesh.quad_role(car, 0.5 * car.length, floor, floor + 0.2) != \
            "underframe", cid


def test_window_bands_read_as_glass(catalog):
    """每一扇窗的正中必须是玻璃；窗带之外紧邻的位置必须不是。"""
    for cid, car in catalog.car_types.items():
        windows = car.shape.windows
        if windows is None or windows.count == 0:
            continue
        mid = 0.5 * (windows.y.y0 + windows.y.y1)
        for start, stop in car.window_bands():
            centre = 0.5 * (start + stop)
            assert train_mesh.section_role(car, centre, mid) == "glass", (cid, centre)
            assert stop - start == pytest.approx(windows.width, abs=1e-9), cid


def test_door_glass_lines_up_with_the_window_band(catalog):
    """车门上半截是玻璃（与窗带同高），下半截是门板。"""
    for cid, car in catalog.car_types.items():
        doors, windows = car.shape.doors, car.shape.windows
        if doors is None or doors.per_end == 0 or windows is None:
            continue
        for start, stop in car.door_bands():
            centre = 0.5 * (start + stop)
            assert train_mesh.section_role(car, centre, windows.y.y1 - 0.01) == "glass"
            assert train_mesh.section_role(car, centre, doors.y.y0 + 0.01) == "door"


def test_paint_per_face_on_a_straight_body(catalog):
    """逐块核对颜色：把鼻锥抹掉之后，每一个四边形的颜色必须正好是它所属分区的色。

    这是唯一一处依赖 :meth:`MeshBuilder.add_tube` **顶点写出顺序**的断言
    （每个四边形连写 4 个顶点、先侧面后端盖），换来的是"整节车的涂装一次全查完"
    —— 比抽查几个点强得多。
    """
    for car_id in ("df4b_loco", "coach_25b", "crh380a_mid", "cr400af_mid"):
        car = _straight(catalog.car_types[car_id])
        builder = train_mesh.build_car_mesh(car, details=False)
        colors = train_mesh.train_colors(car)
        palette = {role: colors.for_role(role) for role in train_mesh.SECTION_ROLES}
        section = car.shape.split_section()
        heights = [h for _, h in section]
        count = len(section)
        stations = train_mesh._stations(car)
        half = 0.5 * car.length

        quads = (len(stations) - 1) * count
        # 侧面永远是每块 2 个三角形；端盖由扇形三角化给出，其中退化的那些不会
        # 被写进网格（断面底边中线那个点是共线的），所以这里只约束下界。
        assert builder.triangle_count >= 2 * quads

        for quad in range(quads):
            k, i = divmod(quad, count)
            u = 0.5 * (stations[k] + stations[k + 1]) + half
            low = heights[i]
            high = heights[(i + 1) % count]
            role = train_mesh.quad_role(car, u, low, high)
            expected = style.shade(builder.normals[4 * quad],
                                   palette[role])
            for vertex in range(4 * quad, 4 * quad + 4):
                got = builder.colors[vertex]
                assert all(abs(got[c] - expected[c]) < 1e-9 for c in range(4)), (
                    f"{car_id} 第 {quad} 块（u={u:.3f}, y=[{low}, {high}]）的颜色"
                    f"不对：期望 {expected}，得到 {got}"
                )


def test_every_face_carries_a_known_livery_colour(catalog, meshes):
    """放大镜：任取一批顶点，它的颜色必须等于「某个用过的颜色 × 该处光照」。"""
    for car_id in ("coach_25b", "crh380a_end"):
        car = catalog.car_types[car_id]
        builder = meshes[car_id]
        colors = train_mesh.train_colors(car)
        palette = colors.palette()
        for vertex in range(0, len(builder.colors), 37):
            normal = builder.normals[vertex]
            got = builder.colors[vertex]
            assert any(
                all(abs(got[c] - style.shade(normal, candidate)[c]) < 1e-9
                    for c in range(4))
                for candidate in palette
            ), f"{car_id} 第 {vertex} 个顶点用了调色板以外的颜色 {got}"


def test_livery_without_body_colour_raises(catalog):
    car = catalog.car_types["coach_25b"]
    bare = CarSpec(id="probe", name="probe", length=car.length,
                   bogie_half_spacing=car.bogie_half_spacing,
                   mass_kg=car.mass_kg, powered=False,
                   livery={"band": "#FFFFFF"}, shape=car.shape)
    with pytest.raises(ValueError, match="body"):
        train_mesh.train_colors(bare)


# --------------------------------------------------------------------------- #
# 车顶设备
# --------------------------------------------------------------------------- #

def test_roof_gear_is_declared_for_every_car_type(catalog):
    """``data/trains.json`` 里 7 个车型都显式写了车顶设备，没有靠默认值。"""
    for cid, car in catalog.car_types.items():
        assert car.roof_gear != "none", f"{cid} 没写 roof_gear"


def test_roof_gear_stands_above_the_roof(catalog, meshes, bodies):
    for cid, car in catalog.car_types.items():
        roof = car.shape.roof_height
        assert max(p[1] for p in bodies[cid].positions) == \
            pytest.approx(roof, abs=1e-6), cid
        assert max(p[1] for p in meshes[cid].positions) > roof + 0.02, cid


def test_pantograph_lives_on_emu_trailer_cars(catalog):
    """受电弓挂在中间车（真实编组就是这样，头车车顶是平整的）。"""
    assert catalog.car_types["crh380a_mid"].roof_gear == "pantograph"
    assert catalog.car_types["cr400af_mid"].roof_gear == "pantograph"
    assert catalog.car_types["df4b_loco"].roof_gear == "fans"


# --------------------------------------------------------------------------- #
# 三种车型必须一眼可辨
# --------------------------------------------------------------------------- #

def test_three_families_have_distinguishable_noses(catalog, bodies):
    """车头收敛段的长度必须把三款车**分档**拉开。

    「远看一眼就知道是哪款车」这个要求没法直接断言，但它的主要来源是侧影，
    而侧影的主要来源是车头收敛段的长度占比 —— 这个可以量。
    """
    blunt = _nose_taper(catalog.car_types["df4b_loco"], bodies["df4b_loco"])
    coach = _nose_taper(catalog.car_types["coach_25b"], bodies["coach_25b"])
    fuxing = _nose_taper(catalog.car_types["cr400af_end"], bodies["cr400af_end"])
    harmony = _nose_taper(catalog.car_types["crh380a_end"], bodies["crh380a_end"])

    assert blunt < 0.06, f"东风4B 的车头倒角占 {blunt:.1%}，太长了"
    assert coach < 0.06, f"25B 的端部倒角占 {coach:.1%}，太长了"
    assert 0.08 < fuxing < 0.22, f"复兴号的剑形车头占 {fuxing:.1%}，不在预期档位"
    assert 0.16 < harmony < 0.32, f"和谐号的尖长车头占 {harmony:.1%}，不在预期档位"
    assert fuxing < harmony, "复兴号的车头应当明显短于和谐号"


def test_one_ended_nose_leaves_the_other_end_flat(catalog, bodies):
    """``ends = "forward"`` 的头车：另一端必须是完整的贯通断面。"""
    for car_id in ("crh380a_end", "cr400af_end"):
        car = catalog.car_types[car_id]
        shape = car.shape
        assert shape.nose.has_nose_at("end")
        assert not shape.nose.has_nose_at("start")
        widths = dict(_width_profile(bodies[car_id]))
        half = 0.5 * car.length
        assert widths[round(-half, 6)] == pytest.approx(shape.half_width, abs=1e-6), \
            f"{car_id} 的平端不是完整断面"
        assert widths[round(half, 6)] == pytest.approx(
            shape.half_width * shape.nose.end_scale, abs=1e-6
        ), f"{car_id} 的鼻尖宽度不等于 end_scale × 半宽"


def test_mid_cars_are_slimmer_than_they_are_stubby(catalog, bodies):
    """车厢比"短粗"要更瘦一点 —— 这是"看着像火车"的粗略门槛。

    概要设计 §5.8 里已经承认了缩尺取舍（弯道半径决定了车长不能太长），
    但至少长细比要明显大于 1，不能缩成一个方块。
    """
    for cid, car in catalog.car_types.items():
        shape = car.shape
        ratio = car.length / (2.0 * shape.half_width)
        assert ratio > 2.5, f"{cid} 的长细比只有 {ratio:.2f}，看起来是个方块"


def test_body_half_width_is_the_widest_section_point(catalog, bodies):
    """半宽必须来自断面本身（最宽的那一点），而不是又一个手写的标量。"""
    for cid, car in catalog.car_types.items():
        shape = car.shape
        widest = max(lateral for lateral, _ in shape.split_section())
        assert widest == pytest.approx(shape.half_width, abs=1e-9), cid


# --------------------------------------------------------------------------- #
# 编组朝向
# --------------------------------------------------------------------------- #

def test_consist_facings_point_the_noses_outward(catalog):
    """一列动车组的两端各有一个头型；中间车两端都平。"""
    for train_id in ("crh380a_8", "cr400af_8", "cr400af_16"):
        spec = catalog[train_id]
        facings = spec.car_facings()
        assert facings[0] is False, f"{train_id} 的头车不该掉头"
        assert facings[-1] is True, f"{train_id} 的尾车必须掉头"
        assert sum(facings) == 1, f"{train_id} 掉头的车不止一节"

    # 两头都是头型的车（东风4B / 客车）不需要掉头
    green = catalog["green_skin_10"].car_facings()
    assert not any(green)


def test_flipped_cars_actually_point_backwards(catalog):
    """镜像过的那节头车，鼻尖必须真的跑到了 -x 一侧（尾车朝后）。"""
    for train_id in ("crh380a_8", "cr400af_8", "cr400af_16"):
        spec = catalog[train_id]
        items = train_mesh.build_train_mesh(spec, details=False)
        assert len(items) == spec.car_count
        for index in (0, spec.car_count - 1):
            item = items[index]
            nose = item.car.shape.nose
            assert nose.has_nose_at("end") and not nose.has_nose_at("start")
            samples = _width_profile(item.builder)
            tail, head = samples[0][1], samples[-1][1]
            assert head != pytest.approx(tail), \
                f"{train_id} 第 {index} 节两端一样宽，测不出朝向"
            tip_at_front = head < tail
            assert tip_at_front == (index == 0), (
                f"{train_id} 第 {index} 节（flipped={item.flipped}）的鼻尖在 "
                f"{'+x' if tip_at_front else '-x'} 一侧，应当朝车外"
            )


def test_mirrored_mesh_is_a_true_reflection(catalog):
    """镜像的三种自洽性：位置取反、法线取反、绕序反转。"""
    builder = train_mesh.build_car_mesh(catalog.car_types["coach_25b"])
    mirror = builder.mirrored_x()
    assert mirror.triangle_count == builder.triangle_count
    assert mirror.colors == builder.colors
    for original, flipped in zip(builder.positions, mirror.positions):
        assert flipped == pytest.approx((-original[0], original[1], original[2]))
    for original, flipped in zip(builder.normals, mirror.normals):
        assert flipped == pytest.approx((-original[0], original[1], original[2]))
    for (i, j, k), (a, b, c) in zip(builder.triangles, mirror.triangles):
        assert (a, b, c) == (i, k, j), "镜像后绕序必须反转"
