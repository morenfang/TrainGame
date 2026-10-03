"""布景的几何契约：轮廓、局部坐标、水面、净空、取景。

这里查的全是"**摆位**"的数学，不是"好看不好看"。摆位能查，是因为
:mod:`render.scenery` 把数据与网格分开了：这些用例不需要建任何一个几何节点，
纯算坐标就能跑 —— 而摆位错（山压在轨道上、房子立在湖心）恰恰是最难用眼睛发现、
后果最难看的一类错。

网格那一侧只查一条：``build_scenery`` 出来的东西**是完整可渲染的**（有顶点、
有三角形、名字对得上）。它不查"像不像一座山"—— 那件事只能靠出图。
"""

from __future__ import annotations

import math

import pytest

from render import scenery as sc

#: 浮点比较的松紧。坐标是"乘以三角函数再加起来"，量级几十米时末位必然有误差。
EPS = 1e-9


def approx(value, tol=1e-9):
    return pytest.approx(value, abs=tol)


# --------------------------------------------------------------------------- #
# 局部坐标：+x 沿 heading、+z 朝右手边
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("heading", [0.0, 0.7, math.pi / 2, math.pi, -2.4])
def test_local_plus_x_points_along_heading(heading):
    """局部 ``+x`` 必须落到 heading 方向上 —— 房子的门朝哪边全看这一条。"""
    x, y, z = sc._local_to_world(3.0, -7.0, heading, 1.0, 0.5, 0.0)
    assert (x, y, z) == pytest.approx((
        3.0 + math.cos(heading), 0.5, -7.0 + math.sin(heading)))


@pytest.mark.parametrize("heading", [0.0, 0.7, math.pi / 2, math.pi, -2.4])
def test_local_plus_z_points_to_the_right_hand_side(heading):
    """局部 ``+z`` 是右手边，与 ``render.transform`` 的约定一致。

    轨道、列车、布景三处都靠这个"右手边"对齐；哪一处反了，站台就会摆到轨道里
    侧去（而内侧看上去也"没错"，只是月台在环线里面）。
    """
    x, y, z = sc._local_to_world(0.0, 0.0, heading, 0.0, 0.0, 1.0)
    assert (x, z) == pytest.approx((-math.sin(heading), math.cos(heading)))


def test_local_direction_keeps_the_y_component():
    """方向变换不能把 ``y`` 压平。

    屋顶斜面的法线靠 ``y`` 才"抬起来"；压成 0 之后屋顶就变成一堵竖墙，
    朝向也会跟着翻 —— 表现为"屋顶从屋里被打亮"。
    """
    _, y, _ = sc._local_dir(1.2, 0.0, 0.7, 0.0)
    assert y == approx(0.7)


# --------------------------------------------------------------------------- #
# 占地轮廓
# --------------------------------------------------------------------------- #

def test_round_footprint_distance_is_zero_inside():
    fp = sc.Footprint(x=0.0, z=0.0, half_x=5.0, half_z=5.0, round_=True)
    assert fp.distance_to(0.0, 0.0) == 0.0
    assert fp.distance_to(3.0, 4.0) == 0.0            # 正好在轮廓上
    assert fp.distance_to(0.0, 8.0) == approx(3.0)    # 外面 3 m


def test_rect_footprint_measures_from_the_edge_not_the_centre():
    """长条物的净距要按**边**算：站台中心离轨道 20 m，边缘可不一定。"""
    fp = sc.Footprint(x=0.0, z=0.0, half_x=20.0, half_z=3.0)
    assert fp.distance_to(0.0, 25.0) == approx(22.0)  # 不在 x 范围内的边角
    assert fp.distance_to(0.0, 1.0) == 0.0
    assert fp.distance_to(24.0, 0.0) == approx(4.0)


def test_rotated_footprint_swaps_its_axes():
    """转了 90° 之后，长边就跑到 z 上去了。"""
    fp = sc.Footprint(x=0.0, z=0.0, half_x=20.0, half_z=3.0,
                      heading=math.pi / 2)
    assert fp.distance_to(0.0, 25.0) == approx(5.0)    # 撞上长边（20）
    assert fp.distance_to(24.0, 0.0) == approx(21.0)   # 撞上短边（3）


def test_trees_and_peaks_report_a_round_footprint():
    tree = sc.Tree(x=1.0, z=2.0, height=10.0, seed=1)
    assert tree.footprint().round_
    assert tree.footprint().half_x == approx(10.0 * 0.22)
    peak = sc.Peak(x=1.0, z=2.0, radius=30.0, height=40.0)
    assert peak.footprint().round_
    assert peak.footprint().half_x == approx(30.0)


# --------------------------------------------------------------------------- #
# 水面
# --------------------------------------------------------------------------- #

def test_river_contains_its_own_centre_line():
    river = sc.River(points=((-50.0, 0.0), (50.0, 0.0)), width=12.0, seed=1)
    assert river.contains(0.0, 0.0)
    assert not river.contains(0.0, 20.0)
    assert river.distance_to(0.0, 100.0) > 0.0


def test_lake_inner_ring_stays_inside_the_shore():
    """深水内圈必须整个待在湖里。

    这条查的是一个真实的坑：抖动数列只算一次再按 ``scale`` 生成两圈，若改成
    "按 scale 各摇一次"，内圈就会从岸边戳出去 —— 看上去像湖面上浮着一片草。
    """
    lake = sc.Lake(x=10.0, z=-4.0, radius=40.0, beach=5.0, seed=7)
    for x, z in lake.outline(0.72):
        assert lake.contains(x, z), f"内圈点 ({x:.2f}, {z:.2f}) 跑到岸外了"


def test_lake_contains_returns_false_outside_and_distance_is_zero_inside():
    lake = sc.Lake(x=0.0, z=0.0, radius=40.0, beach=5.0, seed=3)
    assert lake.distance_to(0.0, 0.0) == 0.0
    assert not lake.contains(0.0, 200.0)
    assert lake.distance_to(0.0, 80.0) > 0.0


# --------------------------------------------------------------------------- #
# 一份布景：净空、包围盒、合并
# --------------------------------------------------------------------------- #

def _sample_scenery() -> sc.Scenery:
    return sc.Scenery(
        peaks=(sc.Peak(x=0.0, z=120.0, radius=40.0, height=60.0, seed=1),),
        houses=(sc.House(x=-30.0, z=0.0, width=10.0, depth=8.0, seed=2),),
        trees=(sc.Tree(x=30.0, z=0.0, height=10.0, seed=3),),
        meadows=(sc.Meadow(x=0.0, z=-50.0, radius=20.0, seed=4),),
    )


def test_empty_scenery_is_falsy_and_has_no_clearance():
    empty = sc.Scenery()
    assert not empty
    assert empty.item_count == 0
    assert empty.clearance([(0.0, 0.0, 0.0)])[0] == math.inf
    assert empty.bounds() == (0.0, 0.0, 0.0, 0.0)


def test_clearance_finds_the_tightest_item():
    """净距取的是**最小**的那个，并且要能说清是谁 —— 报错时才知道去挪谁。"""
    scenery = _sample_scenery()
    distance, label = scenery.clearance([(0.0, 0.0, 0.0)])
    # 树在心离原点 30 m、树冠半径 2.2 m；房子轮廓的边在 x = -25
    assert distance == approx(25.0)
    assert "民房" in label


def test_clearance_only_counts_solid_things():
    """草地不算数：它平贴地面，列车压过去既不挡路也看不出来。"""
    meadow = sc.Scenery(meadows=(sc.Meadow(x=0.0, z=0.0, radius=30.0, seed=1),))
    assert meadow.clearance([(0.0, 0.0, 0.0)])[0] == math.inf
    assert not list(meadow.footprints())


def test_bounds_covers_every_thing_and_gets_merged():
    scenery = _sample_scenery()
    min_x, min_z, max_x, max_z = scenery.bounds()
    for fp in scenery.footprints():
        reach = max(fp.half_x, fp.half_z)
        assert min_x <= fp.x - reach + EPS and max_x >= fp.x + reach - EPS
        assert min_z <= fp.z - reach + EPS and max_z >= fp.z + reach - EPS
    # 草地也要算进包围盒（取景时它铺得最开）
    assert min_z <= -70.0 + EPS

    merged = sc.Scenery().merged(scenery, sc.Scenery())
    assert merged.item_count == scenery.item_count


def test_river_pushes_its_bounds_out_by_the_bank():
    """河岸也是"看得见的部分"，取景框得把它算进去。"""
    river = sc.River(points=((-100.0, 0.0), (100.0, 0.0)), width=10.0, bank=4.0,
                     seed=1)
    min_x, min_z, max_x, max_z = sc.Scenery(rivers=(river,)).bounds()
    assert (min_x, max_x) == (approx(-104.0), approx(104.0))


# --------------------------------------------------------------------------- #
# 网格：烘出来的东西必须真的能画
# --------------------------------------------------------------------------- #

def test_build_scenery_bakes_everything_into_one_node(app):
    """一份布景烘成**一个**节点（一次绘制调用）。"""
    scenery = _sample_scenery().merged(sc.Scenery(
        rivers=(sc.River(points=((-60.0, 60.0), (60.0, 60.0)), width=10.0, seed=9),),
        lakes=(sc.Lake(x=0.0, z=-90.0, radius=30.0, seed=8),),
        platforms=(sc.Platform(x=0.0, z=40.0, length=40.0, width=6.0, heading=0.0,
                               seed=5),),
    ))
    node = sc.build_scenery(scenery)
    node.reparentTo(app.render)
    assert node.getName() == "scenery"
    # 烘出来的节点**本身就是** GeomNode（不是套一层空父节点）：没有多余的一层
    assert node.getNumChildren() == 0
    geom_node = node.node()
    assert geom_node.getNumGeoms() == 1
    # 顶点数走 GeomPrimitive 这一级：``GeomVertexData`` 得从 ``Geom`` 上取，
    # 而 ``Geom.getVertexData()`` 的签名要的是 thread，不是下标。
    geom = geom_node.getGeom(0)
    assert geom.getNumPrimitives() == 1
    assert geom.getPrimitive(0).getNumPrimitives() > 0


# --------------------------------------------------------------------------- #
# 车站：两种样式 + 手工布景的序列化
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("style", [sc.STATION_CLASSICAL, sc.STATION_MODERN])
def test_station_mesh_is_renderable(app, style):
    """两套车站都能烘出真的几何：有三角形、有顶点、名字对得上。"""
    scenery = sc.Scenery(stations=(sc.Station(x=10.0, z=-5.0, style=style,
                                              heading=0.7, seed=11),))
    node = sc.build_scenery(scenery)
    node.reparentTo(app.render)
    assert node.getName() == "scenery"
    geom = node.node().getGeom(0)
    assert geom.getPrimitive(0).getNumPrimitives() > 0


def test_station_footprint_label_names_the_style():
    classic = sc.Station(x=0.0, z=0.0, style=sc.STATION_CLASSICAL).footprint()
    modern = sc.Station(x=0.0, z=0.0, style=sc.STATION_MODERN).footprint()
    assert "老式" in classic.label
    assert "新式" in modern.label
    # 占地轮廓是矩形，尺寸取自 width / depth
    assert classic.half_x == approx(11.0)
    assert classic.half_z == approx(5.0)


@pytest.mark.parametrize("kind,label", [
    (sc.HOUSE_COTTAGE, "民房"),
    (sc.HOUSE_TOWER, "高楼"),
    (sc.HOUSE_MALL, "商场"),
    (sc.HOUSE_SHOP, "便利店"),
])
def test_house_footprint_label_names_the_kind(kind, label):
    house = sc.House(x=0.0, z=0.0, kind=kind)
    assert label in house.footprint().label


@pytest.mark.parametrize("kind", [
    sc.HOUSE_COTTAGE, sc.HOUSE_TOWER, sc.HOUSE_MALL, sc.HOUSE_SHOP,
])
def test_every_house_kind_bakes_renderable_geometry(app, kind):
    node = sc.build_scenery(sc.Scenery(
        houses=(sc.House(x=0.0, z=0.0, kind=kind, seed=3),)))
    node.reparentTo(app.render)
    geom = node.node().getGeom(0)
    assert geom.getPrimitive(0).getNumPrimitives() > 0


@pytest.mark.parametrize("kind,label", [
    (sc.TREE_PINE, "针叶松"),
    (sc.TREE_OAK, "阔叶树"),
    (sc.TREE_POPLAR, "白杨"),
    (sc.TREE_PALM, "棕榈"),
])
def test_tree_footprint_label_names_the_kind(kind, label):
    tree = sc.Tree(x=0.0, z=0.0, kind=kind)
    assert label in tree.footprint().label


@pytest.mark.parametrize("kind", [
    sc.TREE_PINE, sc.TREE_OAK, sc.TREE_POPLAR, sc.TREE_PALM,
])
def test_every_tree_kind_bakes_renderable_geometry(app, kind):
    node = sc.build_scenery(sc.Scenery(
        trees=(sc.Tree(x=0.0, z=0.0, kind=kind, seed=5),)))
    node.reparentTo(app.render)
    geom = node.node().getGeom(0)
    assert geom.getPrimitive(0).getNumPrimitives() > 0


def test_station_with_platform_extends_its_footprint_forward():
    bare = sc.Station(x=0.0, z=0.0, width=22.0, depth=11.0)
    with_platform = sc.Station(x=0.0, z=0.0, width=22.0, depth=11.0,
                               with_platform=True)
    assert bare.footprint().half_z == approx(5.5)
    # 站台从站房正面（-z）探出去，占地要在纵深方向多出一段
    assert with_platform.footprint().half_z > bare.footprint().half_z
    assert with_platform.platform_offset() == approx(11.0 * 0.5 + 2.5 + 5.0 * 0.5)


def test_scenery_serialization_round_trips_placeables():
    """手工布景（房 / 车站 / 山 / 树）存了再读必须逐字段一致。"""
    scenery = sc.Scenery(
        houses=(sc.House(x=1.0, z=2.0, heading=0.3, seed=7),),
        stations=(sc.Station(x=4.0, z=5.0, style=sc.STATION_MODERN, seed=8),),
        peaks=(sc.Peak(x=6.0, z=7.0, radius=10.0, height=20.0, seed=9),),
        trees=(sc.Tree(x=8.0, z=9.0, height=11.0, seed=10),),
    )
    restored = sc.scenery_from_dict(sc.scenery_to_dict(scenery))
    assert restored == scenery


def test_scenery_from_dict_tolerates_missing_and_unknown_keys():
    restored = sc.scenery_from_dict({"houses": [{"x": 1.0, "z": 2.0}], "nope": [1]})
    assert len(restored.houses) == 1
    assert restored.houses[0].x == 1.0
    assert not restored.stations and not restored.peaks and not restored.trees
    assert sc.scenery_from_dict(None).item_count == 0


def test_placeable_items_and_without_item():
    scenery = sc.Scenery(
        houses=(sc.House(0.0, 0.0), sc.House(1.0, 1.0)),
        stations=(sc.Station(2.0, 2.0),),
        peaks=(sc.Peak(3.0, 3.0, 10.0, 10.0),),
        trees=(sc.Tree(4.0, 4.0),),
    )
    keys = [(key, index) for key, index, _ in sc.placeable_items(scenery)]
    assert keys == [("houses", 0), ("houses", 1), ("stations", 0),
                    ("peaks", 0), ("trees", 0)]

    without = sc.without_item(scenery, "houses", 0)
    assert [h.x for h in without.houses] == [1.0]
    assert len(without.stations) == 1 and len(without.peaks) == 1
