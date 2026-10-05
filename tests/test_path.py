"""路径求解（``core/track/path.py``）的测试。

这是 G1（闭环）与 G2（列车沿闭环连续运行不脱节）验收线的直接依据：

* 闭环：由装配图求出环路，并验证「图上的环 == 几何上的精确闭环」；
* 不脱节：验证**每一个转向架都严格落在轨道中心线上**（半径误差 = 0）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from core.geometry import Pose, normalize_angle
from core.track.catalog import Catalog
from core.track.layout import Layout
from core.track.path import (
    PathError,
    bogie_at,
    consist_length,
    detect_closure,
    find_loop,
    place_consist,
    trace,
)

DEG = math.radians
RADIUS = 40.0
CIRCLE_CENTER = (0.0, 40.0)  # 左转整圆从原点出发：圆心在 (0, R)


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return Catalog.builtin()


# --------------------------------------------------------------------------- #
# 场景构造
# --------------------------------------------------------------------------- #

def build_circle(catalog: Catalog, pieces: int = 8) -> tuple[Layout, int]:
    """用 N 节 R40/45° 左弯轨拼一个整圆并合拢。返回 ``(装配图, 首件)``。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root("curve_r40_l45")
    index = first
    for _ in range(pieces - 1):
        index = layout.attach("curve_r40_l45", "a", (index, "b"))
    if pieces == 8:
        layout.connect((index, "b"), (first, "a"))
    return layout, first


def build_figure_eight(catalog: Catalog) -> tuple[Layout, int]:
    """8 字形：8 节左弯 + 8 节右弯，在切点处合拢。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root("curve_r40_l45")
    index = first
    for _ in range(7):
        index = layout.attach("curve_r40_l45", "a", (index, "b"))
    index = layout.attach("curve_r40_r45", "a", (index, "b"))
    for _ in range(7):
        index = layout.attach("curve_r40_r45", "a", (index, "b"))
    layout.connect((index, "b"), (first, "a"))
    return layout, first


def build_straight_chain(catalog: Catalog, count: int) -> tuple[Layout, int]:
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    index = first
    for _ in range(count - 1):
        index = layout.attach("straight_20", "a", (index, "b"))
    return layout, first


# --------------------------------------------------------------------------- #
# 闭环检测（G1）
# --------------------------------------------------------------------------- #

def test_circle_traces_back_to_its_start(catalog):
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    assert path.closed
    assert path.termination == "closed"
    assert len(path) == 8
    assert path.total_length == pytest.approx(2 * math.pi * RADIUS, rel=1e-12)


def test_closed_loop_can_be_started_from_any_port(catalog):
    """闭环上从任意端口出发都应当绕回自己。"""
    layout, first = build_circle(catalog)
    for port in ("a", "b"):
        assert trace(layout, (first, port)).closed


def test_detect_closure_reports_success(catalog):
    layout, first = build_circle(catalog)
    report, path = detect_closure(layout, (first, "a"))
    assert report.closed
    assert path is not None
    assert report.visit_count == 8
    assert report.total_length == pytest.approx(2 * math.pi * RADIUS, rel=1e-12)
    # 真正闭环的接缝误差在 float64 噪声级别
    assert report.gap_distance < 1e-9
    assert abs(report.gap_heading) < 1e-9


def test_detect_closure_finds_the_loop_without_a_hint(catalog):
    layout, _first = build_circle(catalog)
    report, path = detect_closure(layout)
    assert report.closed
    assert path is not None and len(path) == 8


# --------------------------------------------------------------------------- #
# 「看不见的缝」：首尾严丝合缝、但没 connect()
#
# 回归自实盘存档 saves/layout.json（38 件，首尾端口 39/b 与 40/b 相距 1.2e-13 m，
# seams 却是空的）。老版本的判据是"必须回到同一个端口"，于是这条链被当成开链，
# 列车开到头就停在缝上：HUD 报"缺口 0.0000 m"、轨道看上去完全连续，用户看到的
# 就是"车卡在 #40 和 #39 之间不动"。
# --------------------------------------------------------------------------- #

def build_unwelded_circle(catalog: Catalog, pieces: int = 8):
    """拼一个整圆，但**不焊**最后那道缝 —— 首尾端口对齐、连接表里却没有边。

    8 节 45° 正好 360°，末端落回起点，误差在 float64 噪声量级（1e-13 m），
    与"真闭环"同一量级 —— 所以只能靠容差判据认出来。
    """
    layout = Layout(catalog=catalog)
    first = layout.add_root("curve_r40_l45")
    index = first
    for _ in range(pieces - 1):
        index = layout.attach("curve_r40_l45", "a", (index, "b"))
    return layout, first


def test_unwelded_ring_is_still_traced_as_a_loop(catalog):
    """首尾严丝合缝的链必须当环走 —— 否则列车会停在看不见的缝上。"""
    layout, first = build_unwelded_circle(catalog)
    # 前提：确实没焊，而且确实严丝合缝（两道缝都在，只是没连）
    assert len(layout) == 8
    assert layout.free_ports() == [(first, "a"), (first + 7, "b")]
    pair, (distance, heading_gap) = layout.closest_free_pair()
    assert distance < 1e-12 and heading_gap < 1e-12

    path = trace(layout, (first, "a"))
    assert path.closed, "首尾严丝合缝的链没被认成环 —— 列车会停在缝上"
    assert path.termination == "closed"
    assert len(path) == 8
    assert path.total_length == pytest.approx(2 * math.pi * RADIUS, rel=1e-12)
    # 闭环的判据是几何，不是端口表：走线不能顺手改拓扑
    assert layout.free_ports() == [(first, "a"), (first + 7, "b")]


def test_unwelded_ring_is_drivable_by_a_train(catalog):
    """闭环判据的最终目的：列车能一直跑下去，而不是停在缝上。"""
    from core.train.consist import TrainCatalog
    from core.train.dynamics import TrainState, step

    layout, _first = build_unwelded_circle(catalog)
    report, path = detect_closure(layout, allow_open=True)
    assert report.closed and path is not None and path.closed

    spec = TrainCatalog.builtin()["green_skin_10"]
    state = TrainState(s=spec.total_length, v=0.0, throttle=1.0, brake=0.0,
                       direction=1.0)
    for _ in range(3000):                      # 300 s，足够绕好几圈
        step(state, spec, 0.1, total_length=path.total_length,
             closed=path.closed, direction=state.direction)
    assert state.v > 1.0, "列车在缝上停下了"
    assert state.distance > path.total_length, "列车没绕过一圈"


def test_an_open_chain_whose_gap_is_real_is_still_not_a_loop(catalog):
    """容差不能把"没铺完的线"也认成环 —— 少一节的圆环缺口是米级的。

    这是上面那条判据的安全边界：真闭环误差 1e-13 m，而 45° 的圆环少一节差
    **30.615 m**（22.5° 的圆环接一半差 80 m = 直径），中间隔着十几个数量级，
    所以 1e-6 m 的判据不可能误判。
    """
    layout, first = build_unwelded_circle(catalog, pieces=7)
    path = trace(layout, (first, "a"))
    assert not path.closed
    assert path.termination == "open_end"
    report, _ = detect_closure(layout, allow_open=True)
    assert not report.closed
    # 缺口正好是缺那 45° 弧对应的弦长 —— 与 1e-6 m 的判据差着七个数量级
    assert report.gap_distance == pytest.approx(
        2 * RADIUS * math.sin(DEG(45.0) / 2), rel=1e-9)
    assert abs(report.gap_heading) == pytest.approx(DEG(45.0), rel=1e-9)


def test_the_reported_stall_in_layout_json_is_gone(catalog):
    """**用户报的那个 #40 → #39 卡死**：直接拿实盘存档回归。

    ``saves/layout.json`` 是 38 件的一条链，首尾端口 ``39/b`` 与 ``40/b`` 相距
    1.2e-13 m（严丝合缝），可是 ``seams`` 是空的 —— 用户没按 C。老版本走线只认
    端口表，于是把它当成开链：列车跑满 1748.32 m 后停在缝上，HUD 还报"缺口
    0.0000 m"，看起来就是轨道明明接上了车却卡住。

    这里直接吃那份存档（不是合成布局），因为它就是出问题的那份数据。
    """
    save = Path(__file__).resolve().parent.parent / "saves" / "layout.json"
    if not save.exists():
        pytest.skip(f"示例存档不在：{save}")

    layout = Layout.from_dict(
        json.loads(save.read_text(encoding="utf-8")), catalog)
    assert len(layout) == 38
    assert layout.to_dict()["seams"] == [], "这份存档本来就没焊过缝"
    assert layout.free_ports() == [(39, "b"), (40, "b")]

    report, path = detect_closure(layout, allow_open=True)
    assert report.closed, f"#40→#39 那道缝没被认出来：{report.describe()}"
    assert path is not None and path.closed

    # 列车必须绕得下去，而不是跑满一圈就停在缝上
    from core.train.consist import TrainCatalog
    from core.train.dynamics import TrainState, step

    spec = TrainCatalog.builtin()["green_skin_10"]
    state = TrainState(s=spec.total_length, v=0.0, throttle=1.0, brake=0.0,
                       direction=1.0)
    for _ in range(6000):                      # 600 s ≈ 好几圈
        step(state, spec, 0.1, total_length=path.total_length,
             closed=path.closed, direction=state.direction)
    assert state.v > 1.0, "列车还是停在了 #40→#39 的缝上"
    assert state.distance > 2.0 * path.total_length


def test_unfinished_loop_reports_a_metre_scale_gap(catalog):
    """少一节的链必须报出米级缺口 —— 与真闭环的 1e-14 相差 14 个数量级。"""
    layout, first = build_circle(catalog, pieces=7)
    report, path = detect_closure(layout, (first, "a"))
    assert not report.closed
    assert path is None
    assert report.gap_distance > 1.0
    assert report.visit_count == 7


def test_open_chain_terminates_at_its_free_end(catalog):
    layout, first = build_straight_chain(catalog, 3)
    path = trace(layout, (first, "a"))
    assert not path.closed
    assert path.termination == "open_end"
    assert len(path) == 3
    assert path.total_length == pytest.approx(60.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# allow_open：把"最长的那条可行进路径"交出来
# --------------------------------------------------------------------------- #

def test_allow_open_hands_back_the_chain_that_is_not_a_loop(catalog):
    """没闭合成环时，``allow_open`` 要给出那条**能开车**的路径。

    这是编辑器要的语义：线还没铺完就召唤一列车跑一段，是正常的玩法；
    "能不能绕一圈"（``report.closed``）与"能不能开车"（``path``）是两个问题。
    """
    layout, _ = build_straight_chain(catalog, 3)
    report, path = detect_closure(layout, allow_open=True)
    assert not report.closed
    assert path is not None
    assert not path.closed
    assert len(path) == 3
    assert path.total_length == pytest.approx(60.0, rel=1e-12)


def test_allow_open_still_prefers_a_closed_loop(catalog):
    """环线上 ``allow_open`` 不能把闭环让给开链 —— 环路永远优先。"""
    layout, _ = build_circle(catalog)
    report, path = detect_closure(layout, allow_open=True)
    assert report.closed and path is not None and path.closed
    assert len(path) == 8


def test_allow_open_picks_the_longest_run_of_an_unfinished_loop(catalog):
    """缺一节的圆环：最优开链就是那 7 节，长度正好是 7/8 个圆。"""
    layout, _ = build_circle(catalog, pieces=7)
    report, path = detect_closure(layout, allow_open=True)
    assert not report.closed
    assert path is not None
    assert report.visit_count == 7
    assert path.total_length == pytest.approx(7.0 / 8.0 * 2 * math.pi * RADIUS,
                                              rel=1e-9)


def test_allow_open_still_says_no_on_an_empty_layout(catalog):
    """空场地没有任何路径可给 —— 这条路必须还是 ``None``，不能是零长路径。"""
    report, path = detect_closure(Layout(catalog=catalog), allow_open=True)
    assert not report.closed
    assert path is None
    assert report.visit_count == 0


def test_buffer_piece_terminates_the_traversal_as_a_dead_end(catalog):
    """缓冲端没有 route，因此列车停在它前面的那一节上（这就是"死端"的实现）。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    layout.attach("buffer_stop", "a", (first, "b"))
    path = trace(layout, (first, "a"))
    assert not path.closed
    assert path.termination == "dead_end"
    # 缓冲端不贡献可走行长度，所以只有直轨那一段
    assert len(path) == 1
    assert path.total_length == pytest.approx(20.0, rel=1e-12)


def test_find_loop_raises_a_descriptive_error_when_there_is_no_loop(catalog):
    layout, _first = build_circle(catalog, pieces=7)
    with pytest.raises(PathError, match="no closed loop"):
        find_loop(layout)


def test_find_loop_returns_the_loop(catalog):
    layout, first = build_circle(catalog)
    path = find_loop(layout, (first, "a"))
    assert path.closed


def test_trace_rejects_an_unknown_start_piece(catalog):
    layout, _first = build_circle(catalog)
    with pytest.raises(PathError, match="does not exist"):
        trace(layout, (42, "a"))


# --------------------------------------------------------------------------- #
# 8 字形
# --------------------------------------------------------------------------- #

def test_figure_eight_traces_as_one_closed_loop(catalog):
    """16 节件、在切点处共用一个接缝，走线应当一次走完并绕回来。"""
    layout, first = build_figure_eight(catalog)
    path = trace(layout, (first, "a"))
    assert path.closed
    assert len(path) == 16
    left = [s for s in path if s.dtheta > 0]
    right = [s for s in path if s.dtheta < 0]
    assert len(left) == 8 and len(right) == 8


def test_figure_eight_visits_the_tangent_seam_twice(catalog):
    """切点是两环共用的接缝，因此走线会两次经过同一个位置。"""
    layout, first = build_figure_eight(catalog)
    path = trace(layout, (first, "a"))
    seam = layout.world_port(first, "a")
    hits = [
        seg
        for seg in path
        if seg.entry_frame.planar_distance_to(seam) < 1e-9
    ]
    assert len(hits) == 2


def test_figure_eight_total_length_is_two_circles(catalog):
    layout, first = build_figure_eight(catalog)
    path = trace(layout, (first, "a"))
    assert path.total_length == pytest.approx(4 * math.pi * RADIUS, rel=1e-12)


# --------------------------------------------------------------------------- #
# 「环 + 支线」：走线必须停下来
# --------------------------------------------------------------------------- #

def build_loop_with_a_spare_branch(catalog: Catalog) -> tuple[Layout, int, int]:
    """**成环的道岔，侧股空着** —— 也就是"环 + 支线"。

    道岔 ``turnout_l_40`` 的直股 a→b 是环的一部分，侧股 c 上什么都没接（现实中
    就是"岔尖开向别处、那条股道还没铺"的常态）。从 b 接一条「半圆 + 40 m 直线
    + 半圆」的链，末端**恰好**落回 a（误差 4e-14 m —— 完全对齐的接缝）。

    这个布局是用户在编辑器里按 C 一按就崩的那一个：接缝对齐、``connect`` 成功，
    但环上还留着一个空闲端口 ``c``，而 ``_closure_candidates`` 只拿空闲端口当
    起点，于是从 ``c`` 出发走进环里，绕着环一圈圈转，永远回不到起点 ``c``。
    """
    layout = Layout(catalog=catalog)
    turnout = layout.add_root("turnout_l_40")
    index = turnout
    for def_id in (
        ["curve_r40_l45"] * 4 + ["straight_20"] * 2 + ["curve_r40_l45"] * 4
    ):
        index = layout.attach(def_id, "a", (index, "b"))
    return layout, turnout, index


def test_loop_with_a_spare_branch_seals_at_a_perfect_aligned_seam(catalog):
    """先把前提钉住：这条缝是**严丝合缝**的，所以 ``connect`` 一定会放行。"""
    layout, turnout, tail = build_loop_with_a_spare_branch(catalog)
    assert sorted(layout.free_ports()) == [(turnout, "a"), (turnout, "c"), (tail, "b")]

    pair, (distance, heading_gap) = layout.closest_free_pair()
    assert pair == ((turnout, "a"), (tail, "b"))
    assert distance < 1e-9 and heading_gap < 1e-9, "这条缝必须是对齐的"

    layout.connect(*pair)
    # 环接通了，但岔尖那条侧股还空着 —— 这就是"支线"
    assert layout.free_ports() == [(turnout, "c")]


def test_trace_from_a_spare_branch_stops_instead_of_spinning(catalog):
    """**这一条就是那个崩溃**：从支线末端走线必须停下来，而且要快。

    道岔可挤岔：从岔股（``c``）逆向进来能挤回岔尖（``a``）、再并进环里 —— 所以
    它不再是「一步都走不了的死端」，而是「并进环之后绕一圈、撞上访问集就停」。
    无论哪种语义，``trace`` 都必须当场认清、立即停下，绝不能一路走到
    ``MAX_SEGMENTS``（十万段）才抛 ``PathError``（那个异常从 ``LayoutView.sync()``
    里冒出来，按一下 C 就把整个游戏带崩）。
    """
    layout, turnout, tail = build_loop_with_a_spare_branch(catalog)
    layout.connect((turnout, "a"), (tail, "b"))

    path = trace(layout, (turnout, "c"))
    # 逆向挤岔进环之后，靠访问集认出循环并停下 —— 不再是一步都走不了的死端。
    assert path.termination == "entered_loop", "逆向挤岔进环后，应靠访问集停下"
    assert not path.closed
    # 先挤回岔尖，再并入环绕一圈；段数必须小（远小于 MAX_SEGMENTS），
    # 否则就是没停下、又开始转了。
    assert 0 < len(path) < 100
    assert path.segments[0].piece_index == turnout
    assert path.segments[0].reversed


def test_detect_closure_finds_the_loop_even_with_a_spare_branch(catalog):
    """环本身必须被认出来 —— "不崩"不够，还得"认得对"。

    空闲端口全在支线上，从它们出发永远走不出闭环，所以候选起点必须补上**合拢缝
    的端点**（任何环都经过至少一道缝）。
    """
    layout, turnout, tail = build_loop_with_a_spare_branch(catalog)
    layout.connect((turnout, "a"), (tail, "b"))

    report, path = detect_closure(layout, allow_open=True)
    assert report.closed, report.describe()
    assert path is not None and path.closed
    # 环 = 道岔直股（40 m）+ 8 节 45° 弯（整圆 2πR）+ 2 节 20 m 直线
    assert report.visit_count == len(layout)
    assert report.total_length == pytest.approx(
        40.0 + 8 * RADIUS * DEG(45.0) + 40.0, rel=1e-12
    )
    assert report.gap_distance < 1e-9


def test_find_loop_works_on_a_loop_with_a_spare_branch(catalog):
    """``find_loop`` 是列车真正用来拿路径的那个入口，它也必须过得去。"""
    layout, turnout, tail = build_loop_with_a_spare_branch(catalog)
    layout.connect((turnout, "a"), (tail, "b"))

    loop = find_loop(layout)
    assert loop.closed
    assert loop.total_length == pytest.approx(
        40.0 + 8 * RADIUS * DEG(45.0) + 40.0, rel=1e-12
    )


def test_a_loop_closes_through_the_turnout_diverging_leg(catalog):
    """**用户「道岔加进去就合不上环」的根治处。**

    岔股既然是一节标准的 22.5° R40 弯轨，那「岔股 + 15 节同样的弯轨」正好是
    16 × 22.5° = 360°，首尾精确重合，周长 = 2πR = 251.327 m。

    老数据里岔股半径是 40/(pi/8) ≈ 101.86 m（"两条腿等长"的副作用），这一圈就
    永远差一截，而且差多少**都不落在件库的长度格上** —— 用户在屏幕上看到的就是
    「差一点，但没有任何长度的铁轨能对接」。
    """
    layout = Layout(catalog=catalog)
    turnout = layout.add_root("turnout_l_40")
    index = layout.attach("curve_r40_l22_5", "a", (turnout, "c"))
    for _ in range(14):
        index = layout.attach("curve_r40_l22_5", "a", (index, "b"))

    # 先确认这条测试测得到东西：岔股那一圈在**几何上**必须正好合拢，也就是
    # 还没 connect() 时首尾就已经严丝合缝，只差连接表里的那一条边。
    # （老数据里岔股半径 ≈ 101.86 m，这里就会差一大截 —— 这正是"没有合适长度的
    #   铁轨能对接"那个抱怨的根。）
    assert layout.free_ports() == [(turnout, "a"), (turnout, "b"), (index, "b")]
    # 岔股那一圈的缺口：只剩 (turnout, "a") ↔ (index, "b") 这一对，且严丝合缝
    pair, (distance, heading_gap) = layout.closest_free_pair()
    assert pair == ((turnout, "a"), (index, "b"))
    assert distance < 1e-9 and heading_gap < 1e-9, (
        f"岔股 + 15 节 22.5° 应当正好绕满一圈，实测缺口 {distance:.3f} m"
    )

    # 接上：岔股的出口正好落在起点上，所以这一步必须**几何上成立**（老数据会抛错）
    layout.connect((index, "b"), (turnout, "a"))
    # 这一圈走的是岔股，得先把道岔扳到岔股，岔股才在两个方向上都通
    layout.set_switch(turnout, 1)

    report, path = detect_closure(layout)
    assert report.closed, report.describe()
    assert path is not None and path.closed
    assert len(path) == 16, "岔股那一节也算进环里"
    assert report.total_length == pytest.approx(2 * math.pi * RADIUS, rel=1e-12)
    assert report.gap_distance < 1e-9


def test_the_closed_ring_through_a_turnout_can_drive_a_train(catalog):
    """环上有道岔时，列车照样要能沿环跑（道岔档位选岔股那一档）。"""
    layout = Layout(catalog=catalog)
    turnout = layout.add_root("turnout_l_40")
    index = layout.attach("curve_r40_l22_5", "a", (turnout, "c"))
    for _ in range(14):
        index = layout.attach("curve_r40_l22_5", "a", (index, "b"))
    layout.connect((index, "b"), (turnout, "a"))
    layout.set_switch(turnout, 1)          # 扳到岔股

    loop = find_loop(layout)
    assert loop.closed
    # 从岔股那头走：a --岔股--> c --> 15 节弯轨 --> 回到 a
    path = trace(layout, (turnout, "a"))
    assert path.closed
    assert path.segments[0].piece_index == turnout
    assert path.segments[0].route_index == 1


def test_a_layout_of_only_branches_off_a_loop_still_reports_the_loop(catalog):
    """两条支线挂在一个环上：照样要认出环来（候选起点不是"有空闲端口就够"）。"""
    layout = Layout(catalog=catalog)
    turnout = layout.add_root("turnout_l_40")
    index = turnout
    for def_id in (
        ["curve_r40_l45"] * 4 + ["straight_20"] * 2 + ["curve_r40_l45"] * 4
    ):
        index = layout.attach(def_id, "a", (index, "b"))
    # 侧股再挂一小段，让它离环更远一点
    spare = layout.attach("straight_20", "a", (turnout, "c"))
    layout.connect((turnout, "a"), (index, "b"))

    assert sorted(layout.free_ports()) == [(spare, "b")]
    report, path = detect_closure(layout, allow_open=True)
    assert report.closed, report.describe()
    assert path is not None and path.closed
    # 环本身没有被那条支线污染
    assert report.visit_count == len(layout) - 1


# --------------------------------------------------------------------------- #
# 弧长参数化
# --------------------------------------------------------------------------- #

def test_every_sample_lies_exactly_on_the_circle(catalog):
    """核心验证：路径上任意弧长处的中心线点，到圆心距离都恰好是 R。"""
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))

    cx, cz = CIRCLE_CENTER
    for i in range(721):
        s = path.total_length * i / 720
        x, y, z = path.point_at(s)
        radius = math.hypot(x - cx, z - cz)
        assert radius == pytest.approx(RADIUS, abs=1e-9), f"off-track at s={s}"
        assert y == pytest.approx(0.0, abs=1e-12)


def angle_difference(a: float, b: float) -> float:
    """两个角度之差，已折到 (-pi, pi]。比较角度时用它，避免 ±pi 处的分支歧义。"""
    return normalize_angle(a - b)


def test_heading_is_tangent_to_the_circle(catalog):
    """航向必须与半径方向垂直 —— 否则列车会斜着走。"""
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    cx, cz = CIRCLE_CENTER

    for i in range(72):
        s = path.total_length * i / 72
        pose = path.pose_at(s)
        radial = math.atan2(pose.z - cz, pose.x - cx)
        # 左转：航向 = 半径方向 + 90°
        delta = angle_difference(pose.heading, radial + math.pi / 2)
        assert abs(delta) < 1e-9, f"not tangent at s={s}"


def test_segments_are_contiguous(catalog):
    """每一段的终点位姿必须等于下一段的起点位姿（无缝隙）。"""
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    for previous, following in zip(path.segments, path.segments[1:]):
        assert previous.exit_frame.approx_equal(following.entry_frame, 1e-9, 1e-9)


def test_arc_length_offsets_accumulate(catalog):
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    assert path.segments[0].s0 == pytest.approx(0.0)
    for previous, following in zip(path.segments, path.segments[1:]):
        assert following.s0 == pytest.approx(previous.s1, abs=1e-12)


def test_pose_at_wraps_around_a_closed_loop(catalog):
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    total = path.total_length
    for s in (0.0, 10.0, 77.7, total - 1.0):
        assert path.pose_at(s).approx_equal(path.pose_at(s + total), 1e-9, 1e-9)
        assert path.pose_at(s).approx_equal(path.pose_at(s - 3 * total), 1e-9, 1e-9)


def test_pose_at_rejects_out_of_range_on_an_open_path(catalog):
    layout, first = build_straight_chain(catalog, 2)
    path = trace(layout, (first, "a"))
    with pytest.raises(PathError, match="outside"):
        path.pose_at(-1.0)
    with pytest.raises(PathError, match="outside"):
        path.pose_at(1000.0)


def test_sample_returns_points_along_the_path(catalog):
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    poses = path.sample(step=10.0)
    assert len(poses) >= 20
    assert all(isinstance(p, Pose) for p in poses)


def test_direction_and_slope_are_available(catalog):
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    assert path.slope_at(0.0) == pytest.approx(0.0, abs=1e-12)
    assert normalize_angle(path.direction_at(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert path.pose_at(0.0).normalized_heading == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# 道岔决定走哪条路
# --------------------------------------------------------------------------- #

def build_turnout_scene(catalog: Catalog) -> tuple[Layout, int]:
    """直轨 + 道岔 + 两条支线（分别接到道岔的直股与岔股）。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    turnout = layout.attach("turnout_l_40", "a", (first, "b"))
    layout.attach("straight_40", "a", (turnout, "b"))
    layout.attach("straight_40", "a", (turnout, "c"))
    return layout, first


def test_turnout_default_position_sends_the_train_down_the_straight_leg(catalog):
    layout, first = build_turnout_scene(catalog)
    path = trace(layout, (first, "a"))
    assert len(path) == 3
    assert path.segments[1].route_index == 0  # 道岔的直股


def test_turnout_switched_position_sends_the_train_down_the_diverging_leg(catalog):
    layout, first = build_turnout_scene(catalog)
    turnout = path_turnout_index(layout)
    layout.set_switch(turnout, 1)
    path = trace(layout, (first, "a"))
    assert len(path) == 3
    assert path.segments[1].route_index == 1  # 岔股
    # 真的拐了 22.5°
    assert path.segments[1].dtheta == pytest.approx(DEG(22.5), abs=1e-12)


def test_trailing_entry_from_a_branch_merges_back_into_the_main(catalog):
    """用户原话：「从岔道走进主线，不应该是不通的。」

    道岔扳在直股时，从岔股（``c``）逆向进来，列车要能挤回岔尖（``a``）、并入主线
    继续走，而不是被判成死端。顺向（从岔尖进）才看档位。
    """
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    turnout = layout.attach("turnout_l_40", "a", (first, "b"))
    layout.attach("straight_40", "a", (turnout, "b"))          # 主线（直股）
    branch = layout.attach("straight_40", "a", (turnout, "c"))  # 岔道

    assert layout.switch_of(turnout) == 0  # 仍扳在直股

    path = trace(layout, (branch, "b"))
    # 走到主线那一头的开链末端（而不是"走进死端"）。
    assert path.termination == "open_end"
    # 先逆行走完岔道那节 40 m，再挤过道岔（c→a），最后并入主线。
    assert path.segments[0].piece_index == branch and path.segments[0].reversed
    assert path.segments[1].piece_index == turnout
    assert path.segments[1].route_index == 1          # c→a 是岔股那条 route 的反向
    assert path.segments[1].reversed
    assert path.segments[2].piece_index == first


def path_turnout_index(layout: Layout) -> int:
    for index in layout.pieces:
        if layout.definition(index).is_switch:
            return index
    raise AssertionError("no switch in layout")


def test_switch_state_can_be_passed_explicitly(catalog):
    layout, first = build_turnout_scene(catalog)
    turnout = path_turnout_index(layout)
    path = trace(layout, (first, "a"), switch_states={turnout: 1})
    assert path.segments[1].route_index == 1
    # 布局里的档位没被改动
    assert layout.switch_of(turnout) == 0


def test_crossing_lets_the_train_pass_straight_through(catalog):
    """菱形交叉：列车直穿，不能转弯。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_40")
    crossing = layout.attach("crossing_45", "a", (first, "b"))
    tail = layout.attach("straight_40", "a", (crossing, "c"))

    path = trace(layout, (first, "a"))
    assert [s.piece_index for s in path] == [first, crossing, tail]
    assert all(s.dtheta == pytest.approx(0.0, abs=1e-12) for s in path)
    # crossing 的 b/d 那条通路完全没被踩到
    assert {(crossing, "b"), (crossing, "d")} <= set(layout.free_ports())


def test_crossing_route_through_the_diagonal_is_also_straight(catalog):
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_40")
    crossing = layout.attach("crossing_45", "b", (first, "b"))
    path = trace(layout, (first, "a"))
    assert path.segments[1].piece_index == crossing
    assert path.segments[1].dtheta == pytest.approx(0.0, abs=1e-12)
    assert path.segments[1].entry_port == "b"
    assert path.segments[1].exit_port == "d"


# --------------------------------------------------------------------------- #
# 坡度
# --------------------------------------------------------------------------- #

def test_slope_is_derived_from_the_port_heights(catalog):
    layout = Layout(catalog=catalog)
    first = layout.add_root("ramp_up_40")
    ramp_index = first
    layout.attach("straight_40", "a", (ramp_index, "b"))

    path = trace(layout, (ramp_index, "a"))
    assert path.segments[0].grade == pytest.approx(0.03, rel=1e-12)
    assert path.segments[1].grade == pytest.approx(0.0, abs=1e-12)
    # 上到坡顶后高度保持
    assert path.point_at(40.0)[1] == pytest.approx(1.2, abs=1e-12)
    assert path.point_at(60.0)[1] == pytest.approx(1.2, abs=1e-12)


def test_reverse_traversal_down_a_ramp_has_negative_slope(catalog):
    layout = Layout(catalog=catalog)
    ramp_index = layout.add_root("straight_40")
    top = layout.attach("ramp_up_40", "a", (ramp_index, "b"))
    # 反着走：从坡顶的 b 端往 a 端下坡
    path = trace(layout, (top, "b"))
    assert path.segments[0].grade == pytest.approx(-0.03, rel=1e-12)
    assert path.segments[0].reversed


# --------------------------------------------------------------------------- #
# 转向架与编组（G2）
# --------------------------------------------------------------------------- #

CAR_LENGTH = 10.2      # 缩尺后的一节客车（真实 25.5 m * 0.4）
BOGIE_HALF = 3.6       # 转向架中心到车体中心
COUPLING_GAP = 0.3


def consist_arguments(count: int):
    return [CAR_LENGTH] * count, [BOGIE_HALF] * count


def test_bogie_is_always_exactly_on_the_rail(catalog):
    """G2 的核心：每个转向架到圆心的距离都严格等于 R，一毫米都不差。"""
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    lengths, halves = consist_arguments(8)
    cars = place_consist(path, head_s=20.0, lengths=lengths,
                         bogie_half_spacing=halves, coupling_gap=COUPLING_GAP)

    cx, cz = CIRCLE_CENTER
    checked = 0
    for car in cars:
        for bogie in (car.front, car.rear):
            radius = math.hypot(bogie.position[0] - cx, bogie.position[2] - cz)
            assert radius == pytest.approx(RADIUS, abs=1e-9)
            checked += 1
    assert checked == 16


def test_bogie_survives_a_full_run_around_the_loop(catalog):
    """绕环一整圈、逐步推进，转向架始终在轨（不脱节）。"""
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    lengths, halves = consist_arguments(8)
    cx, cz = CIRCLE_CENTER

    s = 0.0
    while s < path.total_length:
        cars = place_consist(path, head_s=s, lengths=lengths,
                             bogie_half_spacing=halves, coupling_gap=COUPLING_GAP)
        for car in cars:
            for bogie in (car.front, car.rear):
                radius = math.hypot(bogie.position[0] - cx, bogie.position[2] - cz)
                assert radius == pytest.approx(RADIUS, abs=1e-9)
        s += 7.5


def test_car_body_cuts_the_chord_on_a_curve(catalog):
    """车体中心落在弦上（比转向架更靠圆心）—— 这就是真实的「切内弦」。

    两个转向架在圆上相隔弧长 2b，其中点到圆心的距离应恰好为 R·cos(b/R)。
    """
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    lengths, halves = consist_arguments(4)
    cars = place_consist(path, head_s=30.0, lengths=lengths,
                         bogie_half_spacing=halves, coupling_gap=COUPLING_GAP)

    cx, cz = CIRCLE_CENTER
    expected = RADIUS * math.cos(BOGIE_HALF / RADIUS)
    for car in cars:
        center_radius = math.hypot(car.center[0] - cx, car.center[2] - cz)
        assert center_radius == pytest.approx(expected, abs=1e-9)
        assert center_radius < RADIUS


def test_car_heading_is_a_chord_not_a_tangent(catalog):
    """车体航向略偏离切线（落后于转向架航向），这正是车体「切内弦」的表现。"""
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    lengths, halves = consist_arguments(2)
    cars = place_consist(path, head_s=30.0, lengths=lengths,
                         bogie_half_spacing=halves, coupling_gap=COUPLING_GAP)

    car = cars[0]
    front_heading = car.front.heading
    offset = abs((car.heading - front_heading + math.pi) % (2 * math.pi) - math.pi)
    assert offset == pytest.approx(BOGIE_HALF / RADIUS, rel=1e-6)


def test_flat_track_gives_zero_pitch(catalog):
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    lengths, halves = consist_arguments(2)
    cars = place_consist(path, head_s=15.0, lengths=lengths,
                         bogie_half_spacing=halves, coupling_gap=COUPLING_GAP)
    for car in cars:
        assert car.pitch == pytest.approx(0.0, abs=1e-12)
        assert car.front.pitch == pytest.approx(0.0, abs=1e-12)


def test_consist_wraps_across_the_seam_without_breaking(catalog):
    """编组尾部跨过接缝（s 变成负数）时必须正确绕回，不能跳到场地另一头。"""
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    lengths, halves = consist_arguments(4)
    cars = place_consist(path, head_s=2.0, lengths=lengths,
                         bogie_half_spacing=halves, coupling_gap=COUPLING_GAP)

    cx, cz = CIRCLE_CENTER
    for car in cars:
        for bogie in (car.front, car.rear):
            radius = math.hypot(bogie.position[0] - cx, bogie.position[2] - cz)
            assert radius == pytest.approx(RADIUS, abs=1e-9)


def test_consist_spacing_follows_car_lengths(catalog):
    """相邻两车车体中心间距 = 两半车长 + 车钩间隙。"""
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    lengths, halves = consist_arguments(5)
    cars = place_consist(path, head_s=40.0, lengths=lengths,
                         bogie_half_spacing=halves, coupling_gap=COUPLING_GAP)

    for previous, following in zip(cars, cars[1:]):
        ds = previous.front.s - following.front.s
        assert ds == pytest.approx(CAR_LENGTH + COUPLING_GAP, abs=1e-9)


def test_place_consist_rejects_mismatched_arguments(catalog):
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    with pytest.raises(PathError, match="same length"):
        place_consist(path, 0.0, lengths=[10.0], bogie_half_spacing=[1.0, 2.0])


def test_place_consist_rejects_bogies_longer_than_the_body(catalog):
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    with pytest.raises(PathError, match="exceeds car length"):
        place_consist(path, 0.0, lengths=[5.0], bogie_half_spacing=[4.0])


def test_consist_is_about_a_third_of_the_loop(catalog):
    """设计意图：8 节编组约占环线 1/3（见概要设计 §5.8 的尺度表）。"""
    lengths, _ = consist_arguments(8)
    total = consist_length(lengths, COUPLING_GAP)
    circumference = 2 * math.pi * RADIUS
    assert 0.3 < total / circumference < 0.4


def test_consist_length_counts_couplings(catalog):
    assert consist_length([10.0, 10.0, 10.0], 0.5) == pytest.approx(31.0)


def test_bogie_at_is_a_thin_wrapper_over_pose_at(catalog):
    layout, first = build_circle(catalog)
    path = trace(layout, (first, "a"))
    bogie = bogie_at(path, 33.0)
    pose = path.pose_at(33.0)
    assert bogie.position == pose.position
    assert bogie.heading == pose.heading


# --------------------------------------------------------------------------- #
# 一个完整的端到端场景
# --------------------------------------------------------------------------- #

def test_circle_plus_figure_eight_shaped_layout_from_scratch(catalog):
    """端到端：从零拼出圆环，检查每个可量的量与设计文档一致。"""
    layout, first = build_circle(catalog)
    report, path = detect_closure(layout)

    assert report.closed
    assert path is not None
    # 圆周长 == 2πR
    assert path.total_length == pytest.approx(2 * math.pi * RADIUS, rel=1e-12)
    # 8 节件、16 个端口、8 条连接、无空闲端口
    assert len(layout) == 8
    assert len(layout.all_ports()) == 16
    assert len(layout.connections) == 16
    assert layout.free_ports() == []
    # 全部走完一圈后回到起点
    start = path.pose_at(0.0)
    end = path.pose_at(path.total_length)
    assert start.approx_equal(end, 1e-9, 1e-9)


# --------------------------------------------------------------------------- #
# 立交疏解线：完整链（用户报的"钻进死端出不来"根治处）
# --------------------------------------------------------------------------- #

def build_overpass_track(catalog: Catalog) -> tuple[Layout, list[int]]:
    """复刻 overpass 场景的**轨道拓扑**（不含布景）：椭圆环 + 一条立交疏解线。

    只用 ``core`` 拼出来，让这条测试能在没有 Panda3D 的环境里跑 —— 轨道部分与
    ``scenes.presets.build_overpass`` 完全一致：side_a 是高架直股（爬坡 + 大跨 +
    下坡），side_b 是地面直股（第 4、6 节换成左开道岔，相隔 80 m），疏解线挂
    在两处岔股之间。默认两处道岔都扳在直股。
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
    assert len(turnouts) == 2, "overpass 轨道要两处道岔"

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
    return layout, turnouts


def test_overpass_track_defaults_to_a_closed_loop(catalog):
    """前提：两处道岔都直股时，环线闭合（列车绕高架环线跑）。"""
    layout, _ = build_overpass_track(catalog)
    report, path = detect_closure(layout, allow_open=True)
    assert report.closed, report.describe()
    assert path is not None and path.closed


def test_overpass_both_diverging_keeps_the_whole_network_drivable(catalog):
    """两处道岔都扳到岔股：整个路网都可走（列车能钻疏解线、再挤回主线）。

    用户报的 bug 就出在这里：列车钻疏解线到头（第二处道岔岔股），把第二处也扳通
    之后应当能继续走回主线。可挤岔（逆向永远放行）之后，这不再是「死端到死端的
    一条开链」，而是能钻桥下、再挤回主线继续走的连通路网 —— 每一件都上得了路。
    """
    layout, turnouts = build_overpass_track(catalog)
    first, second = turnouts
    layout.set_switch(first, 1)
    layout.set_switch(second, 1)

    report, path = detect_closure(layout, allow_open=True)
    assert path is not None

    # 整个路网都可走：路径覆盖**每一个**轨道件，没有哪一节被切掉、变成死端。
    pieces = {seg.piece_index for seg in path}
    assert pieces == set(layout.pieces), (
        f"整个路网都应可走，覆盖全部 {len(layout)} 件，实际 {len(pieces)} 件"
    )

    # 列车确实钻了疏解线：第一处道岔**顺向**（岔尖进）走岔股进疏解线，
    # 第二处道岔**逆向**从岔股挤回岔尖（回主线）—— 这正是"从岔道走进主线不设卡"。
    assert any(seg.piece_index == first and seg.route_index == 1
               and not seg.reversed for seg in path), (
        "第一处道岔必须顺向走岔股，列车才真的进了疏解线"
    )
    assert any(seg.piece_index == second and seg.route_index == 1
               and seg.reversed for seg in path), (
        "第二处道岔必须逆向从岔股挤回岔尖，列车才回得到主线"
    )


def test_open_chain_extension_does_not_touch_a_plain_straight_run(catalog):
    """普通开链（没有对侧可延伸的支路）保持原样，长度不变。"""
    layout, _first = build_straight_chain(catalog, 3)
    report, path = detect_closure(layout, allow_open=True)
    assert path is not None and not path.closed
    assert len(path) == 3
    assert path.total_length == pytest.approx(60.0, rel=1e-12)
