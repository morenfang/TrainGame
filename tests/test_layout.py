"""装配图（``core/track/layout.py``）的测试。

重点是「用轨道件真的拼出一个闭环」这件事，以及存档的确定性。
"""

from __future__ import annotations

import math

import pytest

from core.geometry import Pose, normalize_angle
from core.track.catalog import Catalog
from core.track.layout import Layout, LayoutError

DEG = math.radians


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return Catalog.builtin()


def chain(catalog: Catalog, def_id: str, count: int,
          start_port: str = "a", exit_port: str = "b") -> tuple[Layout, int, int]:
    """从根件起连续接 ``count`` 节同规格件，返回 ``(装配图, 首件, 末件)``。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root(def_id)
    index = first
    for _ in range(count - 1):
        index = layout.attach(def_id, start_port, (index, exit_port))
    return layout, first, index


# --------------------------------------------------------------------------- #
# 放置与位姿推导
# --------------------------------------------------------------------------- #

def test_add_root_places_the_first_piece(catalog):
    layout = Layout(catalog=catalog)
    index = layout.add_root("straight_20")
    assert len(layout) == 1
    assert layout.piece(index).pose == Pose.origin()
    assert layout.root_index == index


def test_add_root_twice_is_rejected(catalog):
    layout = Layout(catalog=catalog)
    layout.add_root("straight_20")
    with pytest.raises(LayoutError, match="add_root"):
        layout.add_root("straight_20")


def test_attaching_straight_to_straight_continues_the_line(catalog):
    layout, first, second = chain(catalog, "straight_20", 2)
    assert layout.piece(second).pose == Pose(20.0, 0.0, 0.0, 0.0)
    # 接缝处两个端口外指向相反
    assert layout.world_port(first, "b").heading == pytest.approx(0.0, abs=1e-12)
    assert normalize_angle(layout.world_port(second, "a").heading) == pytest.approx(
        math.pi, abs=1e-12
    )


def test_attachment_records_parent_link(catalog):
    layout, first, second = chain(catalog, "straight_20", 2)
    assert layout.piece(second).parent == (first, "b", "a")
    assert layout.peer_of(second, "a") == (first, "b")
    assert layout.peer_of(first, "b") == (second, "a")


def test_attach_rejects_an_occupied_port(catalog):
    layout, first, _second = chain(catalog, "straight_20", 2)
    with pytest.raises(LayoutError, match="already connected"):
        layout.attach("straight_20", "a", (first, "b"))


def test_attach_rejects_unknown_piece_definition(catalog):
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    with pytest.raises(LayoutError, match="unknown piece"):
        layout.attach("does_not_exist", "a", (first, "b"))


def test_attach_rejects_unknown_port_names(catalog):
    layout, first, _ = chain(catalog, "straight_20", 1)
    with pytest.raises(LayoutError, match="has no port"):
        layout.attach("straight_20", "zzz", (first, "b"))
    with pytest.raises(LayoutError, match="has no port"):
        layout.attach("straight_20", "a", (first, "zzz"))


def test_attach_rejects_missing_target_piece(catalog):
    layout = Layout(catalog=catalog)
    layout.add_root("straight_20")
    with pytest.raises(LayoutError, match="does not exist"):
        layout.attach("straight_20", "a", (99, "b"))


def test_new_piece_is_a_leaf_so_both_ends_stay_free(catalog):
    """接上来的件自己一端被占用、另一端空着。"""
    layout, first, second = chain(catalog, "straight_20", 2)
    free = set(layout.free_ports())
    assert free == {(first, "a"), (second, "b")}


def test_ramp_chain_accumulates_elevation(catalog):
    """坡道把高度带上去了；后续平轨从新的高度继续。"""
    layout = Layout(catalog=catalog)
    root = layout.add_root("ramp_up_40")
    second = layout.attach("straight_20", "a", (root, "b"))
    assert layout.piece(second).pose.y == pytest.approx(40.0 * 0.03, abs=1e-12)
    assert layout.world_port(second, "b").y == pytest.approx(1.2, abs=1e-12)


# --------------------------------------------------------------------------- #
# 拼出闭环
# --------------------------------------------------------------------------- #

def test_eight_curves_return_exactly_to_the_start(catalog):
    """8 节同向 45° 弯轨：末端端口必须与起点端口精确重合。"""
    layout, first, last = chain(catalog, "curve_r40_l45", 8)

    start = layout.world_port(first, "a")
    end = layout.world_port(last, "b")
    assert start.planar_distance_to(end) < 1e-12, "the loop does not close"
    # 外指向相反（否则轨道会在接缝处折回）
    assert abs(normalize_angle(start.heading - end.heading - math.pi)) < 1e-12


def test_join_gap_is_zero_for_a_true_loop(catalog):
    layout, first, last = chain(catalog, "curve_r40_l45", 8)
    distance, heading_gap = layout.join_gap((last, "b"), (first, "a"))
    assert distance < 1e-12
    assert heading_gap < 1e-12


def test_join_gap_reports_a_measurable_gap_for_an_unfinished_loop(catalog):
    """少一节：缺口必须是米级，并且能被编辑器读出来画红标。"""
    layout, first, last = chain(catalog, "curve_r40_l45", 7)
    distance, _ = layout.join_gap((last, "b"), (first, "a"))
    assert distance > 1.0


def test_connect_verifies_the_seam_and_closes_the_loop(catalog):
    layout, first, last = chain(catalog, "curve_r40_l45", 8)
    layout.connect((last, "b"), (first, "a"))
    assert layout.is_port_connected(first, "a")
    assert layout.is_port_connected(last, "b")
    assert layout.free_ports() == []


def test_connect_refuses_a_seam_that_does_not_line_up(catalog):
    """安全网：用户以为接上了、其实差几米 —— 必须拒绝。"""
    layout, first, last = chain(catalog, "curve_r40_l45", 7)
    with pytest.raises(LayoutError, match="do not line up"):
        layout.connect((last, "b"), (first, "a"))


def test_connect_can_be_forced_without_verification(catalog):
    """读档等场景允许跳过校验（verify=False）。"""
    layout, first, last = chain(catalog, "curve_r40_l45", 7)
    layout.connect((last, "b"), (first, "a"), verify=False)
    assert layout.is_port_connected(first, "a")


def test_connect_rejects_a_port_to_itself(catalog):
    layout, first, _ = chain(catalog, "straight_20", 1)
    with pytest.raises(LayoutError, match="itself"):
        layout.connect((first, "a"), (first, "a"))


def test_connect_rejects_already_connected_ports(catalog):
    layout, first, second = chain(catalog, "straight_20", 2)
    with pytest.raises(LayoutError, match="already connected"):
        layout.connect((first, "b"), (second, "a"))


# --------------------------------------------------------------------------- #
# 删除
# --------------------------------------------------------------------------- #

def test_detach_removes_the_subtree_below_a_piece(catalog):
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    middle = layout.attach("straight_20", "a", (first, "b"))
    last = layout.attach("straight_20", "a", (middle, "b"))

    removed = layout.detach(middle)
    assert sorted(removed) == sorted([middle, last])
    assert set(layout.pieces) == {first}
    # 父件的端口被释放了
    assert not layout.is_port_connected(first, "b")
    assert set(layout.free_ports()) == {(first, "a"), (first, "b")}


def test_detach_rejects_unknown_piece(catalog):
    layout, _first, _ = chain(catalog, "straight_20", 1)
    with pytest.raises(LayoutError, match="no placed piece"):
        layout.detach(12345)


# --------------------------------------------------------------------------- #
# 只删一节（remove_piece）——「删当前轨道会连后面一起删」的解法
# --------------------------------------------------------------------------- #

def test_remove_piece_freezes_the_downstream_half_in_place(catalog):
    """拆中间一节：只少它，后半截**原地冻结**（留真实缺口），不被拽回来。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    second = layout.attach("straight_20", "a", (first, "b"))
    third = layout.attach("straight_20", "a", (second, "b"))
    fourth = layout.attach("straight_20", "a", (third, "b"))
    before = layout.to_dict()
    fourth_a_before = layout.world_port(fourth, "a").position
    fourth_b_before = layout.world_port(fourth, "b").position

    removed = layout.remove_piece(third)
    assert removed == [third]
    assert sorted(layout.pieces) == sorted([first, second, fourth])
    assert layout.total_length() == pytest.approx(3 * 20.0)

    # 断成两段：首节、末节各是一段的根，中间留了真实缺口（没有接骨）
    assert layout.root_indices() == sorted([first, fourth])
    assert layout.peer_of(second, "b") is None
    assert not layout.is_port_connected(fourth, "a")

    # 后半截原地不动：第四节的两个端口都还在老地方
    assert layout.world_port(fourth, "a").position == fourth_a_before
    assert layout.world_port(fourth, "b").position == fourth_b_before

    # 删之前的那份快照还在，可以整图回退
    restored = Layout.from_dict(before, catalog)
    assert len(restored) == 4
    assert restored.world_port(fourth, "b").position == fourth_b_before


def test_remove_piece_round_trips_a_disconnected_layout(catalog):
    """断成两段之后仍能存档读档（多根 + 冻结位姿都得进存档）。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    second = layout.attach("straight_20", "a", (first, "b"))
    third = layout.attach("straight_20", "a", (second, "b"))
    fourth = layout.attach("straight_20", "a", (third, "b"))
    fourth_b_before = layout.world_port(fourth, "b").position
    layout.remove_piece(third)

    restored = Layout.from_dict(layout.to_dict(), catalog)
    assert sorted(restored.pieces) == sorted([first, second, fourth])
    assert restored.root_indices() == sorted([first, fourth])
    assert restored.world_port(fourth, "b").position == fourth_b_before


def test_remove_piece_leaves_a_tip_alone(catalog):
    """删末节：把它的端口释放掉，前面一节原样不动。"""
    layout, first, last = chain(catalog, "straight_20", 2)
    removed = layout.remove_piece(last)
    assert removed == [last]
    assert set(layout.pieces) == {first}
    assert not layout.is_port_connected(first, "b")
    assert layout.total_length() == pytest.approx(20.0)


def test_remove_piece_promotes_the_only_child_when_removing_the_root(catalog):
    """删根件（只有一个子件）：把子件扶正成新根，它挂着的部分一寸都不动。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    second = layout.attach("straight_20", "a", (first, "b"))
    third = layout.attach("straight_20", "a", (second, "b"))
    tail_position = layout.world_port(third, "b").position

    removed = layout.remove_piece(first)
    assert removed == [first]
    assert sorted(layout.pieces) == sorted([second, third])
    assert layout.root_index == second
    assert layout.pieces[second].parent is None
    # 新根位姿不变，所以末端的绝对位置没被挪动
    assert layout.world_port(third, "b").position == pytest.approx(tail_position)


def test_remove_piece_refuses_to_guess_at_a_branch(catalog):
    """岔口上挂着两条支线：拒绝，而不是替用户决定接哪一条。"""
    layout = Layout(catalog=catalog)
    turnout = layout.add_root("turnout_l_40")
    layout.attach("straight_20", "a", (turnout, "b"))
    layout.attach("straight_20", "a", (turnout, "c"))
    with pytest.raises(LayoutError, match="branches"):
        layout.remove_piece(turnout)


def test_remove_piece_rejects_unknown_piece(catalog):
    layout, _first, _last = chain(catalog, "straight_20", 1)
    with pytest.raises(LayoutError, match="no placed piece"):
        layout.remove_piece(999)


def test_clear_empties_the_layout(catalog):
    layout, _first, _last = chain(catalog, "curve_r40_l45", 8)
    layout.clear()
    assert layout.is_empty
    assert layout.free_ports() == []
    assert layout.bounds() is None


# --------------------------------------------------------------------------- #
# 道岔
# --------------------------------------------------------------------------- #

def test_turnout_defaults_to_its_first_position(catalog):
    layout = Layout(catalog=catalog)
    index = layout.add_root("turnout_l_40")
    assert layout.switch_of(index) == 0


def test_turnout_switch_can_be_changed_and_toggled(catalog):
    layout = Layout(catalog=catalog)
    index = layout.add_root("turnout_l_40")
    layout.set_switch(index, 1)
    assert layout.switch_of(index) == 1
    assert layout.toggle_switch(index) == 0
    assert layout.switch_of(index) == 0


def test_switch_operations_are_rejected_on_a_plain_piece(catalog):
    layout = Layout(catalog=catalog)
    index = layout.add_root("straight_20")
    assert layout.switch_of(index) is None
    with pytest.raises(LayoutError, match="not a switch"):
        layout.set_switch(index, 0)


def test_switch_rejects_an_invalid_position(catalog):
    layout = Layout(catalog=catalog)
    index = layout.add_root("turnout_l_40")
    with pytest.raises(LayoutError, match="not a valid switch position"):
        layout.set_switch(index, 7)


# --------------------------------------------------------------------------- #
# 存档：只存父子引用，位姿读档重建
# --------------------------------------------------------------------------- #

def _build_figure_eight(catalog: Catalog) -> Layout:
    """8 字形：8 节左弯 + 8 节右弯，共用切点接缝。"""
    layout = Layout(catalog=catalog)
    first = layout.add_root("curve_r40_l45")
    index = first
    for _ in range(7):
        index = layout.attach("curve_r40_l45", "a", (index, "b"))
    # 第二个环从接缝出发，反向弯
    index = layout.attach("curve_r40_r45", "a", (index, "b"))
    for _ in range(7):
        index = layout.attach("curve_r40_r45", "a", (index, "b"))
    layout.connect((index, "b"), (first, "a"))
    return layout


def test_figure_eight_layout_is_geometrically_closed(catalog):
    layout = _build_figure_eight(catalog)
    assert len(layout) == 16
    assert layout.free_ports() == []


def test_save_round_trip_preserves_every_pose(catalog):
    layout = _build_figure_eight(catalog)
    payload = layout.to_dict()
    restored = Layout.from_dict(payload, catalog)

    assert set(restored.pieces) == set(layout.pieces)
    for index, placed in layout.pieces.items():
        other = restored.pieces[index]
        assert other.def_id == placed.def_id
        assert other.pose.approx_equal(placed.pose, 1e-12, 1e-12), (
            f"piece {index} moved on reload"
        )


def test_save_round_trip_preserves_connectivity(catalog):
    layout = _build_figure_eight(catalog)
    restored = Layout.from_dict(layout.to_dict(), catalog)
    assert restored.connections == layout.connections


def test_save_does_not_store_poses_only_parent_links(catalog):
    """存档条款（概要设计 §6.4）：位姿由父子链重建，不落盘。"""
    layout = _build_figure_eight(catalog)
    payload = layout.to_dict()
    for record in payload["pieces"]:
        assert "pose" not in record
        assert {"parent", "parent_port", "my_port"} <= set(record)
    # 只有根件记录自己的位姿（多段断开时每段一个根）
    assert payload["roots"]
    for root in payload["roots"]:
        assert "pose" in root


def test_rebuild_poses_is_idempotent(catalog):
    """反复重建位姿不应有任何漂移（确定性）。"""
    layout = _build_figure_eight(catalog)
    snapshot = {i: p.pose for i, p in layout.pieces.items()}
    for _ in range(5):
        layout.rebuild_poses()
    for index, pose in snapshot.items():
        assert layout.pieces[index].pose.approx_equal(pose, 1e-12, 1e-12), (
            f"piece {index} drifted after rebuild"
        )


def test_from_dict_rejects_orphan_pieces(catalog):
    payload = {
        "schema": 1,
        "root": {"index": 0, "def": "straight_20",
                 "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "heading_rad": 0.0}},
        "pieces": [
            {"index": 1, "def": "straight_20", "parent": 0,
             "parent_port": "b", "my_port": "a"},
            # 父件 42 根本不存在 -> 无法挂上去
            {"index": 2, "def": "straight_20", "parent": 42,
             "parent_port": "b", "my_port": "a"},
        ],
        "switches": {},
    }
    with pytest.raises(LayoutError, match="orphan"):
        Layout.from_dict(payload, catalog)


def test_empty_layout_round_trips(catalog):
    layout = Layout(catalog=catalog)
    restored = Layout.from_dict(layout.to_dict(), catalog)
    assert restored.is_empty


def test_switch_positions_survive_a_save_round_trip(catalog):
    layout = Layout(catalog=catalog)
    index = layout.add_root("turnout_l_40")
    layout.set_switch(index, 1)
    restored = Layout.from_dict(layout.to_dict(), catalog)
    assert restored.switch_of(index) == 1


# --------------------------------------------------------------------------- #
# 存档落到**文件**上：真实场景的往返
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scene", ["circle", "figure8", "yard", "ramp"])
def test_save_and_load_round_trip_the_builtin_scenes(catalog, tmp_path, scene):
    """把内置场景存成文件再读回来，几何必须逐字段一致。

    ``to_dict`` / ``from_dict`` 已经被上面几条覆盖了，但那一层只是"在内存里递一个
    dict"。真正会丢东西的是文件这一层：路径、编码、``indent`` 之后浮点数的往返
    精度。这里刻意用件数最多、且带道岔档位的真实场景（figure8 有 16 件、
    yard 有三个分叉口），比手工搭的样例更容易撞出格式问题。

    存档的契约是"只存拓扑，位姿读档由父子链重建"，所以位姿和连接表都在
    比较范围内 —— 那正是重建逻辑的产物。
    """
    from scripts import screenshot

    layout = Layout(catalog=catalog)
    getattr(screenshot, f"scene_{scene}")(layout)

    path = layout.save_json(tmp_path / f"{scene}.json")
    restored = Layout.load_json(path, catalog)

    assert len(restored) == len(layout)
    assert restored.total_length() == pytest.approx(layout.total_length(), rel=1e-15)
    assert restored.free_ports() == layout.free_ports()
    for index in layout.pieces:
        assert restored.definition(index).id == layout.definition(index).id
        assert restored.piece(index).parent == layout.piece(index).parent
        assert restored.piece(index).pose == layout.piece(index).pose
        assert restored.switch_of(index) == layout.switch_of(index)
        for port_id in layout.definition(index).port_ids:
            assert restored.world_port(index, port_id) == \
                layout.world_port(index, port_id)


def test_a_closed_loop_stays_closed_after_being_written_and_read(catalog, tmp_path):
    """闭环读档之后必须还是闭环。

    闭环的成立靠的是接缝误差在 1e-13 量级 —— 存档只记拓扑、位姿靠重建，所以
    "读回来的环还闭着"正好证明了重建路径与首次拼装路径给出同一组位姿（没有
    因为浮点格式或重建顺序引入累计误差）。
    """
    from core.track.path import detect_closure
    from scripts import screenshot

    layout = Layout(catalog=catalog)
    screenshot.scene_circle(layout)
    before, before_path = detect_closure(layout)
    assert before.closed and before_path is not None

    restored = Layout.load_json(layout.save_json(tmp_path / "circle.json"), catalog)
    after, after_path = detect_closure(restored)
    assert after.closed and after_path is not None
    assert after.visit_count == before.visit_count
    assert after.gap_distance < 1e-9
    assert after.total_length == pytest.approx(before.total_length, rel=1e-15)

    # 读档后走一遍环，逐段进入帧都要和原图完全一致（"重建没引入误差"的实证）
    assert [(s.piece_index, s.route_index, s.reversed, s.entry_frame)
            for s in after_path.segments] == \
        [(s.piece_index, s.route_index, s.reversed, s.entry_frame)
         for s in before_path.segments]


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #

def test_total_length_sums_route_lengths(catalog):
    layout, _first, _last = chain(catalog, "curve_r40_l45", 8)
    expected = 8 * 40.0 * DEG(45.0)
    assert layout.total_length() == pytest.approx(expected, rel=1e-12)


def test_total_length_of_a_circle_equals_its_circumference(catalog):
    """8 节 R40/45° 的总长必须等于 2*pi*R —— 顺带验证了半径定义。"""
    layout, _first, _last = chain(catalog, "curve_r40_l45", 8)
    assert layout.total_length() == pytest.approx(2 * math.pi * 40.0, rel=1e-12)


def test_bounds_cover_the_whole_loop(catalog):
    layout, _first, _last = chain(catalog, "curve_r40_l45", 8)
    bounds = layout.bounds()
    assert bounds is not None
    (min_x, min_z), (max_x, max_z) = bounds
    # 左转整圆从原点出发、圆心在 (0, R)，故 x 落在 [-R, R]，z 落在 [0, 2R]
    assert min_x == pytest.approx(-40.0, abs=1e-6)
    assert max_x == pytest.approx(40.0, abs=1e-6)
    assert min_z == pytest.approx(0.0, abs=1e-6)
    assert max_z == pytest.approx(80.0, abs=1e-6)


def test_repr_is_informative(catalog):
    layout, _first, _last = chain(catalog, "curve_r40_l45", 8)
    text = repr(layout)
    assert "pieces=8" in text
    assert "free_ports=2" in text


def test_component_count_counts_connected_segments(catalog):
    """连通分量数：空场地 0、一整条线 1、拆成几段就几段。"""
    empty = Layout(catalog=catalog)
    assert empty.component_count() == 0

    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    second = layout.attach("straight_20", "a", (first, "b"))
    third = layout.attach("straight_20", "a", (second, "b"))
    fourth = layout.attach("straight_20", "a", (third, "b"))
    assert layout.component_count() == 1

    layout.remove_piece(third)                 # 断成两段
    assert layout.component_count() == 2


def test_component_count_stays_one_when_a_loop_is_opened_but_not_split(catalog):
    """删掉闭环里的一节：下游冻结成新根，却仍靠合拢缝与主线连通 —— 还是 1 个分量。

    这是「扳道岔 / 开口成链」与「真正拆散」的分界线：前者仍能开链跑车，后者才
    该进无车状态。删闭环里的一节只把环开成一条链，分量数不该变成 2。
    """
    layout = Layout(catalog=catalog)
    first = layout.add_root("curve_r40_l45")
    index = first
    for _ in range(7):
        index = layout.attach("curve_r40_l45", "a", (index, "b"))
    layout.connect((index, "b"), (first, "a"))   # 8 节弯轨闭成一个整圆
    assert layout.component_count() == 1

    layout.remove_piece(4)                       # 删掉环上一节 → 开口成链
    assert len(layout.root_indices()) == 2       # 确实冻结出了第二个根……
    assert layout.component_count() == 1         # ……但仍是连通的一条线


# --------------------------------------------------------------------------- #
# 自动闭合（grow_to_close）与它的两个前置查询
# --------------------------------------------------------------------------- #

def test_closest_free_pair_never_pairs_a_piece_with_itself(catalog):
    """场地里只有一节时，它的两个端口都空着 —— 但把它们接起来是自环。

    那种"接上"只会让列车到端头就折返，而且几何上必然对不上。所以候选端口对必须
    排除同一节，宁可返回 ``None``（"没有可用的接缝"）也不要给出一个假的候选。
    """
    layout = Layout(catalog=catalog)
    layout.add_root("straight_20")
    assert len(layout.free_ports()) == 2
    assert layout.closest_free_pair() is None

    other = layout.attach("straight_20", "a", (0, "b"))
    pair, (distance, heading_gap) = layout.closest_free_pair()
    assert set(pair) == {(0, "a"), (other, "b")}
    assert distance == pytest.approx(40.0, rel=1e-12)   # 两节 20 m 直轨，两端相距 40 m
    assert heading_gap == pytest.approx(0.0, abs=1e-12)


def test_growing_tip_is_the_far_end_of_the_chain(catalog):
    """该从"最后接上的那一节"继续长，而不是回头从根件长。"""
    layout, first, last = chain(catalog, "straight_20", 4)
    assert layout.growing_tip() == (last, "b")
    assert layout.growing_tip()[0] != first


def test_growing_tip_of_a_lone_piece_is_its_route_exit(catalog):
    """只有一节时 a / b 并列，取 route 的**出端** —— 顺着正方向长，不往回长。"""
    layout = Layout(catalog=catalog)
    layout.add_root("curve_r40_l45")
    piece = catalog["curve_r40_l45"]
    assert piece.routes[0].from_port == "a"
    assert piece.routes[0].to_port == "b"
    assert layout.growing_tip() == (0, "b")


def test_grow_to_close_finishes_a_half_built_circle(catalog):
    """**这一条就是用户撞上的那个坑**：22.5° 的弯轨要 16 节才绕满一圈。

    接 8 节正好停在直径的另一头（差 80 m），而"还差 8 节"这件事用户没法心算 ——
    所以让 ``grow_to_close`` 去数。补完之后必须是一个**真闭环**：周长等于 2πR。
    """
    layout, _first, _last = chain(catalog, "curve_r40_l22_5", 8)
    pair, (distance, heading_gap) = layout.closest_free_pair()
    assert pair is not None and pair[0][0] != pair[1][0]
    assert distance == pytest.approx(80.0, rel=1e-6)          # 半径的两倍 = 直径
    assert math.degrees(heading_gap) == pytest.approx(180.0, abs=1e-6)

    added = layout.grow_to_close("curve_r40_l22_5")
    assert added == 8
    assert len(layout) == 16
    assert layout.free_ports() == []                          # 端口全接满了
    assert layout.total_length() == pytest.approx(2 * math.pi * 40.0, rel=1e-9)


def test_grow_to_close_adds_just_one_piece_when_that_is_all_that_is_missing(catalog):
    layout, _first, _last = chain(catalog, "curve_r40_l45", 7)
    added = layout.grow_to_close("curve_r40_l45")
    assert added == 1
    assert len(layout) == 8
    assert layout.total_length() == pytest.approx(2 * math.pi * 40.0, rel=1e-9)


def test_grow_to_close_turns_a_lone_piece_into_a_whole_circle(catalog):
    """放下一节 90° 弯轨就能得到一整圈：1 → 4 节。"""
    layout = Layout(catalog=catalog)
    layout.add_root("curve_r40_l90")
    assert layout.grow_to_close("curve_r40_l90") == 3
    assert len(layout) == 4
    assert layout.total_length() == pytest.approx(2 * math.pi * 40.0, rel=1e-9)


def test_grow_to_close_on_an_aligned_seam_adds_nothing(catalog):
    """已经严丝合缝时只补上那条拓扑边，一节都不多接。"""
    layout, _first, _last = chain(catalog, "curve_r40_l45", 8)
    assert layout.closest_free_pair()[1][0] < 1e-9
    assert layout.grow_to_close("curve_r40_l45") == 0
    assert len(layout) == 8


@pytest.mark.parametrize("piece_id, seed_piece, seed_count", [
    ("straight_20", "straight_20", 3),        # 直轨永远接不出环
    ("curve_r80_l45", "curve_r40_l22_5", 8),  # 半径不对，位置永远对不上
    ("curve_r40_r45", "curve_r40_l45", 4),    # 左右反了，越接越远
])
def test_grow_to_close_is_atomic_when_it_cannot_close(catalog, piece_id,
                                                      seed_piece, seed_count):
    """接不通时**原样回滚** —— 这是"一键"的前提。

    按一下要么真的合上，要么什么都没发生。绝不能留下半圈多余的废轨道：用户看到
    的应该是"没成功 + 为什么"，而不是一坨需要自己慢慢删的东西。
    """
    layout, _first, _last = chain(catalog, seed_piece, seed_count)
    before = layout.to_dict()
    free_before = sorted(layout.free_ports())

    with pytest.raises(LayoutError, match="合不上"):
        layout.grow_to_close(piece_id)

    assert layout.to_dict() == before, "失败之后布局必须与调用前逐字段一致"
    assert sorted(layout.free_ports()) == free_before
    assert len(layout) == seed_count


def test_grow_to_close_can_pick_among_candidates(catalog):
    """**「没有合适长度的铁轨对接」那条抱怨的解法。**

    场景：环只差一节 22.5° 的弯，而用户手上推着的是 40 m 直轨。

    * 只给直轨（= 从前的行为）：接 32 节也合不上，于是回滚 + 报缺口；
    * 给一串候选：机器自己换成弯轨，**一节**就合上。

    这就是"一键闭合"该有的样子：缺口要的是几何，而在候选里换一件是机器该做的事，
    不该把"你自己去找一节对的轨道"丢回给用户。
    """
    layout, _first, _last = chain(catalog, "curve_r40_l22_5", 15)

    before = layout.to_dict()
    with pytest.raises(LayoutError, match="合不上"):
        layout.grow_to_close("straight_40")
    assert layout.to_dict() == before, "失败的尝试必须原样回滚"

    added = layout.grow_to_close(catalog.closure_helpers("straight_40"))
    assert added == 1, "补一节弯轨就该合上"
    assert len(layout) == 16
    assert layout.free_ports() == []
    assert layout.total_length() == pytest.approx(2 * math.pi * 40.0, rel=1e-9)


def test_pick_closing_piece_prefers_the_one_that_actually_fits(catalog):
    """挑件用的是**前瞻**：先算出"接上之后新端口落在哪"，再挑离合拢最近的那个。

    这里直接问那个选择器，不跑整条补线 —— 于是这条测试钉住的是"挑得对"，
    上面那条钉住的是"挑完之后真的合上了"。
    """
    layout, _first, _last = chain(catalog, "curve_r40_l22_5", 15)
    tip = layout.growing_tip()
    assert tip is not None

    candidates = [catalog[def_id] for def_id in
                  ("straight_40", "straight_20", "straight_10",
                   "curve_r40_l22_5", "curve_r80_l45")]
    chosen = layout.pick_closing_piece(tip, candidates)
    assert chosen.id == "curve_r40_l22_5", (
        "缺口正好差一节 22.5° 的弯，挑件器却选了别的"
    )


def test_grow_to_close_refuses_on_an_empty_field(catalog):
    layout = Layout(catalog=catalog)
    with pytest.raises(LayoutError, match="空"):
        layout.grow_to_close("curve_r40_l45")


def test_grow_to_close_refuses_an_unknown_piece(catalog):
    layout, _first, _last = chain(catalog, "curve_r40_l45", 2)
    with pytest.raises(LayoutError, match="unknown piece"):
        layout.grow_to_close("no_such_piece")


def test_grow_to_close_is_undone_by_a_fresh_layout_from_the_snapshot(catalog):
    """补完之后存档再读回来，还是同一个闭环（新接的节点也进得了存档）。"""
    from core.track.path import detect_closure

    layout, _first, _last = chain(catalog, "curve_r40_l22_5", 5)
    layout.grow_to_close("curve_r40_l22_5")

    restored = Layout.from_dict(layout.to_dict(), catalog)
    report, path = detect_closure(restored)
    assert report.closed and path is not None
    assert len(restored) == 16
    assert path.total_length == pytest.approx(2 * math.pi * 40.0, rel=1e-9)
