"""轨道件目录（``core/track/catalog.py`` + ``data/pieces.json``）的测试。

重点验证「目录数据不可能自相矛盾」这条承诺：
端口位姿由 route 推导，再用 ``route_geometry`` 反解回去，两者必须严格相等。
"""

from __future__ import annotations

import math

import pytest

from core.geometry import Pose, arc_end_pose, normalize_angle
from core.track.catalog import (
    Catalog,
    CatalogError,
    build_piece,
)

DEG = math.radians


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return Catalog.builtin()


# --------------------------------------------------------------------------- #
# 目录整体
# --------------------------------------------------------------------------- #

def test_builtin_catalog_loads(catalog):
    assert len(catalog) >= 15
    assert "straight_20" in catalog
    assert "curve_r40_l45" in catalog
    assert "crossing_45" in catalog
    assert "turnout_l_40" in catalog
    assert "buffer_stop" in catalog


def test_builtin_catalog_has_no_validation_warnings(catalog):
    """核心自检：每条 route 的几何都必须能由端口位姿精确反解回来。"""
    warnings = catalog.validate()
    assert warnings == [], "\n".join(warnings)


def test_catalog_reports_unknown_piece_id_clearly(catalog):
    with pytest.raises(CatalogError, match="unknown piece"):
        catalog["no_such_piece"]


def test_catalog_categories_are_populated(catalog):
    for category in ("straight", "curve", "crossing", "switch", "terminal",
                     "grade", "viaduct"):
        assert catalog.by_category(category), f"category {category!r} is empty"


@pytest.mark.parametrize("piece", list(Catalog.builtin()))
def test_every_route_round_trips_through_geometry(piece):
    """对目录里**每一个**件逐条 route 做往返校验。"""
    for route in piece.routes:
        length, dtheta = piece.derive_traversal(route.from_port, route.to_port)
        assert length == pytest.approx(route.length, rel=1e-9), (
            f"{piece.id} route #{route.index} length"
        )
        assert dtheta == pytest.approx(route.dtheta, abs=1e-9), (
            f"{piece.id} route #{route.index} turn"
        )


@pytest.mark.parametrize("piece", list(Catalog.builtin()))
def test_reverse_traversal_negates_turn(piece):
    """反走同一段弧：弧长不变、总转角取负。"""
    for route in piece.routes:
        fwd_len, fwd_turn = piece.derive_traversal(route.from_port, route.to_port)
        rev_len, rev_turn = piece.derive_traversal(route.to_port, route.from_port)
        assert rev_len == pytest.approx(fwd_len, rel=1e-9)
        if abs(fwd_turn) < 1e-12:
            assert abs(rev_turn) < 1e-9
        else:
            assert rev_turn == pytest.approx(-fwd_turn, abs=1e-9)


# --------------------------------------------------------------------------- #
# 端口约定
# --------------------------------------------------------------------------- #

def test_straight_ports_point_outward(catalog):
    piece = catalog["straight_20"]
    a = piece.port("a")
    b = piece.port("b")
    assert (a.pose.x, a.pose.z) == pytest.approx((0.0, 0.0))
    assert (b.pose.x, b.pose.z) == pytest.approx((20.0, 0.0))
    # 入口外指向 = 掉头 180°；出口外指向 = 前进方向
    assert abs(a.pose.heading - math.pi) < 1e-12
    assert abs(b.pose.heading) < 1e-12


def test_port_inbound_is_flipped_outward(catalog):
    piece = catalog["straight_20"]
    # Pose.heading 不做归一化（累加式），比较时用 normalize_angle
    assert normalize_angle(piece.port("a").inbound.heading) == pytest.approx(0.0, abs=1e-12)
    assert normalize_angle(piece.port("b").inbound.heading) == pytest.approx(math.pi, abs=1e-12)
    assert abs(piece.port("a").inbound.heading_error_to(piece.port("a").pose)) == pytest.approx(
        math.pi, abs=1e-12
    )


# --------------------------------------------------------------------------- #
# 弯轨：R 与弧长、左右镜像
# --------------------------------------------------------------------------- #

def test_curve_arc_length_matches_radius(catalog):
    piece = catalog["curve_r40_l45"]
    route = piece.routes[0]
    assert route.radius == pytest.approx(40.0, rel=1e-12)
    assert route.length == pytest.approx(40.0 * DEG(45.0), rel=1e-12)


def test_curve_end_port_lies_on_the_arc(catalog):
    piece = catalog["curve_r40_l45"]
    route = piece.routes[0]
    dx, dz = arc_end_pose(route.length, route.dtheta)
    end = piece.port("b").pose
    assert (end.x, end.z) == pytest.approx((dx, dz), abs=1e-12)
    assert end.heading == pytest.approx(DEG(45.0), abs=1e-12)


def test_left_and_right_curves_are_mirror_images(catalog):
    left = catalog["curve_r40_l45"].port("b").pose
    right = catalog["curve_r40_r45"].port("b").pose
    assert right.x == pytest.approx(left.x, abs=1e-12)
    assert right.z == pytest.approx(-left.z, abs=1e-12)
    assert right.heading == pytest.approx(-left.heading, abs=1e-12)


def test_all_curve_variants_share_radius(catalog):
    for piece in catalog.by_category("curve"):
        route = piece.routes[0]
        expected_radius = 80.0 if "_r80_" in piece.id else 40.0
        assert route.radius == pytest.approx(expected_radius, rel=1e-12), piece.id


# --------------------------------------------------------------------------- #
# 交叉：两条互不相通的通路
# --------------------------------------------------------------------------- #

def test_crossing_has_four_ports_and_two_independent_routes(catalog):
    piece = catalog["crossing_45"]
    assert set(piece.port_ids) == {"a", "b", "c", "d"}
    assert len(piece.routes) == 2
    assert not piece.is_switch, "交叉不是道岔：它没有共享入口"


def test_crossing_ports_cross_at_the_origin(catalog):
    piece = catalog["crossing_45"]
    a, c = piece.port("a").pose, piece.port("c").pose
    b, d = piece.port("b").pose, piece.port("d").pose

    # 通道一沿 +X
    assert (a.x, a.z) == pytest.approx((-20.0, 0.0), abs=1e-9)
    assert (c.x, c.z) == pytest.approx((20.0, 0.0), abs=1e-9)
    # 通道二沿 45°，两线均以原点为中点
    half = 20.0 / math.sqrt(2.0)
    assert (b.x, b.z) == pytest.approx((-half, -half), abs=1e-9)
    assert (d.x, d.z) == pytest.approx((half, half), abs=1e-9)

    assert abs(b.heading - DEG(225.0)) < 1e-9
    assert abs(d.heading - DEG(45.0)) < 1e-9


def test_crossing_only_connects_straight_through(catalog):
    """交叉的走线规则：a 只能到 c，b 只能到 d；a 到 d 必须无路可走。"""
    piece = catalog["crossing_45"]
    assert piece.traversals("a") == (("c", 0, False),)
    assert piece.traversals("b") == (("d", 1, False),)
    assert piece.traversals("c") == (("a", 0, True),)
    assert piece.traversals("d") == (("b", 1, True),)
    # a -> d 没有 route，因此 traversals 不会给出 d
    assert all(exit_port != "d" for exit_port, _, _ in piece.traversals("a"))


def test_crossing_90_arms_are_perpendicular(catalog):
    piece = catalog["crossing_90"]
    b = piece.port("b").pose
    d = piece.port("d").pose
    assert (b.x, b.z) == pytest.approx((0.0, -20.0), abs=1e-9)
    assert (d.x, d.z) == pytest.approx((0.0, 20.0), abs=1e-9)


# --------------------------------------------------------------------------- #
# 道岔
# --------------------------------------------------------------------------- #

def test_turnout_is_a_switch_with_two_positions(catalog):
    piece = catalog["turnout_l_40"]
    assert piece.is_switch
    assert piece.fog_port == "a"
    assert piece.switch_route_indices == (0, 1)


def test_turnout_fog_port_offers_both_exits(catalog):
    piece = catalog["turnout_l_40"]
    exits = piece.traversals("a")
    assert {e[0] for e in exits} == {"b", "c"}


def test_turnout_switch_index_selects_the_route(catalog):
    piece = catalog["turnout_l_40"]
    assert piece.traversals("a", switch_index=0) == (("b", 0, False),)
    assert piece.traversals("a", switch_index=1) == (("c", 1, False),)


def test_turnout_rejects_an_index_that_does_not_leave_that_port(catalog):
    piece = catalog["turnout_l_40"]
    with pytest.raises(CatalogError, match="not a route leaving"):
        piece.traversals("a", switch_index=99)


def test_turnout_merges_from_either_leg(catalog):
    """从两条腿任一进来都只能回到岔尖。"""
    piece = catalog["turnout_l_40"]
    assert piece.traversals("b") == (("a", 0, True),)
    assert piece.traversals("c") == (("a", 1, True),)


def test_turnout_backward_merge_is_always_trailable(catalog):
    """逆向（可挤岔）永远放行：从直股或岔股回到岔尖，都不看档位。

    这是「可挤岔」道岔（trailable turnout）的约定：列车从后面顶着走时，尖轨会被
    车体顺势挤开，所以「从岔道回到主线」不该被道岔档位卡成死路 —— 只有顺向（从
    岔尖 ``a`` 进）才看档位。
    """
    piece = catalog["turnout_l_40"]
    # 无论扳直股还是岔股，两条腿都能挤回岔尖。
    assert piece.traversals("b", switch_index=0) == (("a", 0, True),)
    assert piece.traversals("c", switch_index=0) == (("a", 1, True),)
    assert piece.traversals("b", switch_index=1) == (("a", 0, True),)
    assert piece.traversals("c", switch_index=1) == (("a", 1, True),)


def test_turnout_straight_leg_is_straight_and_leg_diverges(catalog):
    piece = catalog["turnout_l_40"]
    straight, diverge = piece.routes
    assert abs(straight.dtheta) < 1e-12
    assert diverge.dtheta == pytest.approx(DEG(22.5), abs=1e-12)
    # 岔股是一条**标准 R40 弧** —— 与件库里那节 22.5° 弯轨同一个半径。
    # （老数据把两条腿都写成 40 m，半径因此是 40/(pi/8) ≈ 101.86 m。）
    assert diverge.radius == pytest.approx(40.0, rel=1e-12)


def test_turnout_diverging_port_lands_where_a_standard_curve_would(catalog):
    """**「加了道岔就合不上环」的几何根源，就卡在这一条上。**

    岔股的出口端口必须落在**标准件能走到的地方**，否则拿件库里任何东西去接都合
    不上。最省事、也最不留余地的判据：它与 ``curve_r40_l22_5`` 的远端**逐位重合**
    —— 于是「在岔股后面接弯轨」和「在弯轨后面接弯轨」在几何上就是同一件事，
    件库的能力自动覆盖岔股。

    两条腿等长（都是 40 m）会把岔股半径逼成 101.86 m，而件库里没有这个半径的
    弯轨，端口 c 于是落到了标准件永远走不到的格点之外。
    """
    turnout = catalog["turnout_l_40"]
    curve = catalog["curve_r40_l22_5"]
    gap = turnout.port("c").pose.distance_to(curve.port("b").pose)
    heading_gap = turnout.port("c").pose.heading_error_to(curve.port("b").pose)
    assert gap < 1e-12, f"岔股出口离标准弯轨的远端差了 {gap:.6f} m"
    assert abs(heading_gap) < 1e-12


def test_right_turnout_diverging_port_matches_the_right_curve(catalog):
    turnout = catalog["turnout_r_40"]
    curve = catalog["curve_r40_r22_5"]
    assert turnout.port("c").pose.distance_to(curve.port("b").pose) < 1e-12
    assert abs(turnout.port("c").pose.heading_error_to(curve.port("b").pose)) < 1e-12


def test_turnout_straight_leg_is_a_standard_straight(catalog):
    """直股也要落在模块格上：长度取件库里**真有的**那几种直轨长度之一。"""
    turnout = catalog["turnout_l_40"]
    standard = {p.routes[0].length for p in catalog.by_category("straight")}
    assert turnout.routes[0].length in standard
    assert turnout.routes[0].length in {10.0, 20.0, 40.0}


def test_left_and_right_turnouts_mirror(catalog):
    left = catalog["turnout_l_40"].routes[1]
    right = catalog["turnout_r_40"].routes[1]
    assert right.dtheta == pytest.approx(-left.dtheta, abs=1e-12)


# --------------------------------------------------------------------------- #
# 自动补线的候选表（closure_helpers）
# --------------------------------------------------------------------------- #

def test_closure_helpers_are_single_route_straights_and_curves(catalog):
    """自动补线的候选表只收**单通路的直轨与弯轨**。

    道岔 / 交叉件会把线接出岔路（"补"出一堆分叉不是用户想要的）；缓冲端没有通路，
    接上去只是把线封死；坡道会让闭环在竖向对不上（闭环判据是平面的）。
    """
    helpers = catalog.closure_helpers()
    assert set(helpers) == {
        "straight_10", "straight_20", "straight_40",
        "curve_r40_l22_5", "curve_r40_r22_5",
        "curve_r40_l45", "curve_r40_r45",
        "curve_r40_l90", "curve_r40_r90",
        "curve_r80_l45", "curve_r80_r45",
    }
    for excluded in ("turnout_l_40", "crossing_45", "ramp_up_40", "buffer_stop"):
        assert excluded not in helpers


def test_closure_helpers_put_the_preferred_piece_first(catalog):
    """用户手上那一件排最前 —— 用它就能合上时绝不该被擅自换成别的。"""
    assert catalog.closure_helpers() != catalog.closure_helpers("curve_r40_l90")
    assert catalog.closure_helpers("curve_r40_l90")[0] == "curve_r40_l90"
    helpers = catalog.closure_helpers("straight_20")
    assert helpers[0] == "straight_20"
    assert len(set(helpers)) == len(helpers), "候选表里不该有重复项"


def test_closure_helpers_ignore_a_preference_that_is_not_usable(catalog):
    """手上是道岔 / 缓冲端时它进不了候选表，于是照默认顺序来。"""
    assert catalog.closure_helpers("turnout_l_40") == catalog.closure_helpers()
    assert catalog.closure_helpers("buffer_stop") == catalog.closure_helpers()
    assert catalog.closure_helpers("no_such_piece") == catalog.closure_helpers()


# --------------------------------------------------------------------------- #
# 缓冲端与坡道
# --------------------------------------------------------------------------- #

def test_buffer_is_a_zero_length_dead_end(catalog):
    """缓冲端只是一个立在接缝处的车挡：一个端口、没有 route。"""
    piece = catalog["buffer_stop"]
    assert piece.routes == ()
    assert piece.port_ids == ("a",)
    assert piece.traversals("a") == ()


def test_ramp_raises_and_lowers_by_the_declared_grade(catalog):
    up = catalog["ramp_up_40"]
    down = catalog["ramp_down_40"]
    assert up.port("b").pose.y == pytest.approx(40.0 * 0.03, abs=1e-12)
    assert down.port("b").pose.y == pytest.approx(-40.0 * 0.03, abs=1e-12)
    # 平面几何不受坡度影响
    assert up.port("b").pose.x == pytest.approx(40.0, abs=1e-12)
    assert up.port("b").pose.z == pytest.approx(0.0, abs=1e-12)


def test_ramp_route_length_is_plan_length(catalog):
    """route.length 定义为**平面**弧长，因此与坡度无关。"""
    up = catalog["ramp_up_40"]
    length, dtheta = up.derive_traversal("a", "b")
    assert length == pytest.approx(40.0, rel=1e-12)
    assert dtheta == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# 构建期的错误必须被拦住
# --------------------------------------------------------------------------- #

def test_build_rejects_route_with_neither_length_nor_radius():
    with pytest.raises(CatalogError, match="needs 'length' or 'radius'"):
        build_piece({"id": "x", "routes": [{"from": "a", "to": "b"}]})


def test_build_rejects_route_with_both_length_and_radius():
    with pytest.raises(CatalogError, match="not both"):
        build_piece(
            {
                "id": "x",
                "routes": [
                    {"from": "a", "to": "b", "length": 10.0, "radius": 10.0}
                ],
            }
        )


def test_build_rejects_self_looping_route():
    with pytest.raises(CatalogError, match="must differ"):
        build_piece(
            {"id": "x", "routes": [{"from": "a", "to": "a", "length": 10.0}]}
        )


def test_build_rejects_piece_without_ports():
    with pytest.raises(CatalogError, match="no ports"):
        build_piece({"id": "x"})


def test_build_rejects_inconsistent_shared_port():
    """两条 route 对同一个端口给出不同位姿 —— 必须报错，不能悄悄取其一。"""
    with pytest.raises(CatalogError, match="inconsistently"):
        build_piece(
            {
                "id": "broken",
                "routes": [
                    {"from": "a", "to": "b", "length": 20.0, "dtheta_deg": 0.0},
                    # 同样从 a 出发，但整体转了 30° -> 推出端口 a 的位姿不同
                    {
                        "from": "a",
                        "to": "c",
                        "length": 20.0,
                        "dtheta_deg": 0.0,
                        "offset_deg": 30.0,
                    },
                ],
            }
        )


def test_build_rejects_explicit_port_that_no_route_uses():
    """件上有 route，却声明了一个谁也用不到的端口 —— 多半是端口名写错了。"""
    with pytest.raises(CatalogError, match="no route"):
        build_piece(
            {
                "id": "x",
                "ports": [{"id": "typo", "x": 0.0, "z": 0.0, "heading_deg": 180.0}],
                "routes": [{"from": "a", "to": "b", "length": 10.0}],
            }
        )


def test_route_ports_are_derived_so_they_always_exist():
    """route 的端点由几何派生，因此「引用不存在的端口」这个状态无法表达。"""
    piece = build_piece(
        {"id": "x", "routes": [{"from": "left", "to": "right", "length": 10.0}]}
    )
    assert set(piece.port_ids) == {"left", "right"}
