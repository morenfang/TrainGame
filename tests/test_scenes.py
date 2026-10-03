"""四个现成场景的"能不能开跑"契约。

场景是**唯一**一种"用户不会去检查"的东西：他按 ``--scene valley`` 就是想要一局能
开的，不会去读坐标。所以这里把"能开"拆成几条能独立量的性质，逐条查：

1. **闭环误差是浮点噪声级** —— 不是"差不多合上"。571 m 的环上差 1 cm，列车开到
   接缝处就会跳一下。
2. **布景不压轨道**：用一个**比生成时更细**的步长把路径重采样一遍，独立地量一遍
   净距（不依赖预设自己的过滤），并且把主环之外的岔线也算进去。
3. **布景站在场地里**：山脚伸到地面之外就成了悬空的一片，比缺一块还难看。
4. **桥真的跨在河上**：河谷那两处跨中的位置应当在**水面多边形之内**。这条如果
   只查"桥面比地面高"，河挪走了测试也不会发现。
5. **水不盖轨道**（河谷除外 —— 它那两处就该跨过去）。
6. **房子不叠在一起**：同一块地里两栋房子重合，看上去像渲染出了错。
7. **推荐编组跑得完一圈**：真的把它放上线、真的推进一整圈，不报错、不脱轨。
8. **存了再读回来还是这一局**：存档要带上"布景用哪个预设"。

第 2 条的"独立"是有意的：预设内部用 2 m 的走廊过滤，这里用 1 m；如果哪天有人把
过滤的余量调没了，这条会立刻红 —— 而它红了很容易看懂（是净距，不是"看着不对"）。
"""

from __future__ import annotations

import math

import pytest

import scenes
from core.geometry import Pose, arc_point
from core.track.catalog import Catalog
from core.track.path import detect_closure
from core.train.consist import TrainCatalog
from render import scenery as sc
from render import style
from render.train_view import TrainView
from scenes import presets
from scenes.tracks import SceneError

#: 净空检查用的路径重采样步长（米）。比预设内部的（2 m）更细。
_PROBE_STEP = 1.0

#: 仿真步长与总时长（秒）：推满手柄跑够一整圈的余量。
_SIM_DT = 0.1
_SIM_STEPS = 1500


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return Catalog.builtin()


@pytest.fixture(scope="module")
def trains() -> TrainCatalog:
    return TrainCatalog.builtin()


@pytest.fixture(scope="module")
def built(catalog: Catalog) -> dict[str, presets.Scene]:
    """四个场景各拼一次（模块级：拼一遍要零点几秒，别每个用例都拼）。"""
    return {key: presets.build(key, catalog) for key in scenes.keys()}


def _path(scene: presets.Scene):
    report, path = detect_closure(scene.layout, allow_open=True)
    assert path is not None
    return report, path


def _offloop_points(scene: presets.Scene, path) -> list[tuple[float, float, float]]:
    """主环之外那些件（尽头线、立交疏解线、车挡）**中心线**上的采样点。

    不写死"第几节是岔线"：路径没走到哪几件，那几件就是环外的轨道。点按**件下标
    顺序**给出，所以"疏解线从哪一头走到哪一头"是有序的。

    为什么按 route 取点、而不是把端口连成直线：尽头线确实是几节直轨，但立交疏解线
    是弯的、还带坡 —— 拿端口拉直线会量出一条根本不存在的线，于是"有东西压在疏解线
    上"反而量不出来。这里直接用 route 的几何（与列车走线是同一个公式），弯的坡的
    都对。
    """
    on_loop = {segment.piece_index for segment in path}
    points: list[tuple[float, float, float]] = []
    for index in range(len(scene.layout)):
        if index in on_loop:
            continue
        definition = scene.layout.definition(index)
        for route in definition.routes:
            entry = scene.layout.world_inbound(index, route.from_port)
            steps = max(2, int(route.length // _PROBE_STEP))
            for k in range(steps + 1):
                u = route.length * k / steps
                dx, dz, du = arc_point(route.length, route.dtheta, u)
                points.append(
                    entry.compose(Pose(dx, u * route.grade, dz, du)).position
                )
    return points


def _in_water(x: float, z: float, scenery: sc.Scenery) -> bool:
    """这个点是不是落在水面上（独立实现，不调预设内部那一个）。"""
    return (any(river.contains(x, z) for river in scenery.rivers)
            or any(lake.contains(x, z) for lake in scenery.lakes))


def _probes(scene: presets.Scene, path) -> list[tuple[float, float, float]]:
    return [pose.position for pose in path.sample(_PROBE_STEP)] + \
        _offloop_points(scene, path)


# --------------------------------------------------------------------------- #
# 1. 拼得出来、闭得严实
# --------------------------------------------------------------------------- #

def test_every_scene_builds(built):
    assert set(built) == set(scenes.keys())
    for key, scene in built.items():
        assert scene.key == key
        assert scene.title and scene.blurb
        assert not scene.layout.is_empty
        assert scene.plot_size > 0.0
        assert scene.train_id, f"{key} 没说先上哪列车"
        assert scene.clearance_points, "没有量尺就没法查净空"


def test_loop_closes_to_floating_point(built):
    """闭环误差必须是浮点噪声级 —— 场景是"直接就能跑"，不是"差不多能跑"。"""
    for key, scene in built.items():
        report, path = _path(scene)
        assert report.closed, f"{key}：{report.describe()}"
        assert report.gap_distance < 1e-6, f"{key} 缺口 {report.gap_distance}"
        assert report.gap_heading < 1e-9, f"{key} 航向差 {report.gap_heading}"
        assert path.total_length == pytest.approx(report.total_length)


def test_off_loop_track_is_a_connected_branch(built):
    """每节件都要么在环上、要么是**一条连着**的支线：不能有孤立的轨道飘着。

    只数件数是不够的 —— 立交疏解线一个人就有十节。要查的是：环外那几件彼此接得
    上（不是一截截断头轨），而且整条支线与主线**只在一处或两处**接头：尽头线是
    一处（另一头撞车挡），立交疏解线是两处（另一头接回主线）。
    """
    for key, scene in built.items():
        _, path = _path(scene)
        on_loop = {segment.piece_index for segment in path}
        loose = set(range(len(scene.layout))) - on_loop
        assert len(loose) <= 16, f"{key} 有 {len(loose)} 件不在环上"
        joints = 0
        for index in sorted(loose):
            definition = scene.layout.definition(index)
            mates = [scene.layout.peer_of(index, port_id)
                     for port_id in definition.port_ids]
            mates = [peer for peer in mates if peer is not None]
            assert mates, (
                f"{key}：环外的第 {index} 件（{definition.id}）没接上任何东西，"
                "那是一截断头轨"
            )
            inside = sum(1 for peer in mates if peer[0] in loose)
            assert inside <= 2, f"{key}：环外的支线在第 {index} 件处岔开了"
            joints += sum(1 for peer in mates if peer[0] in on_loop)
        if loose:
            assert joints in (1, 2), (
                f"{key}：环外的支线与主线有 {joints} 处接头"
                "（尽头线 1 处、立交疏解线 2 处）"
            )


def test_the_named_train_exists_and_fits(built, trains):
    for key, scene in built.items():
        assert scene.train_id in trains, f"{key} 点名的编组不在目录里"
        spec = trains[scene.train_id]
        _, path = _path(scene)
        # 留出余量：编组必须比环线短得多，否则首尾会撞上自己
        assert spec.total_length * 2.5 < path.total_length, (
            f"{key}：{spec.name} 长 {spec.total_length:.1f} m，"
            f"环线只有 {path.total_length:.1f} m"
        )


# --------------------------------------------------------------------------- #
# 2. 布景不压轨道
# --------------------------------------------------------------------------- #

def test_scenery_keeps_clear_of_the_track(built):
    """独立地重量一遍净距 —— 不依赖预设自己那套过滤。"""
    for key, scene in built.items():
        _, path = _path(scene)
        distance, tightest = scene.scenery.clearance(_probes(scene, path))
        assert distance >= style.TRACK_CLEARANCE, (
            f"{key}：{tightest} 离轨道只有 {distance:.2f} m"
            f"（要求 ≥ {style.TRACK_CLEARANCE} m）"
        )


def test_clearance_matches_what_the_scene_claims(built):
    """场景自己报的净距，应当与独立量出来的对得上（误差只来自采样步长）。"""
    for key, scene in built.items():
        claimed, _ = scene.clearance()
        _, path = _path(scene)
        measured, _ = scene.scenery.clearance(_probes(scene, path))
        assert measured == pytest.approx(claimed, abs=0.5), f"{key} 自报 {claimed}"


# --------------------------------------------------------------------------- #
# 3. 布景站在场地里
# --------------------------------------------------------------------------- #

def test_scenery_stands_inside_the_plot(built):
    """山脚、屋角、湖岸都不能伸到地面之外。"""
    for key, scene in built.items():
        half = scene.plot_size * 0.5
        min_x, min_z, max_x, max_z = scene.scenery.bounds()
        assert min_x >= -half and max_x <= half, f"{key} 布景横向出界：{min_x:.1f}..{max_x:.1f}"
        assert min_z >= -half and max_z <= half, f"{key} 布景纵向出界：{min_z:.1f}..{max_z:.1f}"
        # 轨道自己也要在场地里
        box = scene.layout.bounds()
        assert box is not None
        (tx0, tz0), (tx1, tz1) = box
        assert min(tx0, tz0) >= -half and max(tx1, tz1) <= half, f"{key} 轨道出界"


def test_camera_bounds_cover_both_track_and_scenery(built):
    """取景框要包住轨道**和**布景，否则山会在画面外。"""
    for key, scene in built.items():
        (x0, z0), (x1, z1) = scene.camera_bounds()
        min_x, min_z, max_x, max_z = scene.scenery.bounds()
        assert x0 <= min_x + 1e-6 and x1 >= max_x - 1e-6, key
        assert z0 <= min_z + 1e-6 and z1 >= max_z - 1e-6, key


# --------------------------------------------------------------------------- #
# 4/5. 水和桥的关系
# --------------------------------------------------------------------------- #

def test_bridges_really_span_the_river(built):
    """"桥跨在河上"要量：两处跨中都得落在**水面多边形之内**。

    只查"桥面比地面高"是不够的 —— 河挪走了、桥还悬在那儿，那种错误照样通过。
    """
    scene = built["gorge"]
    _, path = _path(scene)
    crossings = presets.high_points(path)
    assert len(crossings) == 2, f"应当正好有两处桥面，数出来 {len(crossings)}"
    assert len(scene.scenery.rivers) == 1
    river = scene.scenery.rivers[0]
    for x, y, z in crossings:
        assert y > 0.9, "桥面的高度不足以让河从底下过去"
        assert river.contains(x, z), f"跨中 ({x:.1f}, {z:.1f}) 不在水面上"
    # 两处跨中要**错开**，否则河是轨道底下一根笔直的带子
    assert abs(crossings[0][0] - crossings[1][0]) > 20.0


def test_water_never_covers_the_rails_except_where_the_bridge_is(built):
    for key, scene in built.items():
        if key == "gorge":
            continue                    # 河谷那两处本来就是跨过去的
        _, path = _path(scene)
        water = scene.scenery
        for pose in path.sample(_PROBE_STEP):
            x, z = pose.position[0], pose.position[2]
            assert not _in_water(x, z, water), (
                f"{key}：轨道在 ({x:.1f}, {z:.1f}) 落到了水里"
            )


# --------------------------------------------------------------------------- #
# 6. 布景自己别打架
# --------------------------------------------------------------------------- #

def test_houses_do_not_overlap_each_other(built):
    for key, scene in built.items():
        houses = scene.scenery.houses
        for i, a in enumerate(houses):
            for b in houses[i + 1:]:
                assert a.footprint().distance_to(b.x, b.z) > 0.5, (
                    f"{key}：两栋房子叠在一起 ({a.x:.1f}, {a.z:.1f}) / ({b.x:.1f}, {b.z:.1f})"
                )


def test_town_platform_sits_outside_the_loop(built):
    """月台必须在环线**外侧**。

    这条查的是一个真实的反转：往哪一侧挪，是由"背离环心"算出来的，不写死正负号。
    算反了月台就落进环线里 —— 画面里看着也"有个月台"，只是方向错了。
    """
    scene = built["town"]
    _, path = _path(scene)
    walk = [pose.position for pose in path.sample(2.0)]
    cx, cz = presets.path_centroid(walk)
    platform = scene.scenery.platforms[0]
    here = math.hypot(platform.x - cx, platform.z - cz)
    nearest = min(math.hypot(p[0] - cx, p[2] - cz) for p in walk)
    assert here > nearest + 3.0, "月台比轨道还靠近环心，摆到里面去了"
    assert platform.track_side != 0.0


def test_town_siding_ends_in_a_buffer_stop(built, catalog):
    """岔线是尽头线：末端必须是车挡，而且主线上的道岔要扳在直股。"""
    scene = built["town"]
    _, path = _path(scene)
    on_loop = {segment.piece_index for segment in path}
    loose = [i for i in range(len(scene.layout)) if i not in on_loop]
    assert loose, "小镇车站应当有一条岔线"
    assert scene.layout.definition(loose[-1]).id == "buffer_stop"
    switches = [i for i in range(len(scene.layout))
                if scene.layout.definition(i).is_switch]
    assert len(switches) == 1
    turnout = switches[0]
    definition = scene.layout.definition(turnout)
    assert scene.layout.switch_of(turnout) == definition.switch_route_indices[0], \
        "道岔没有扳在直股，主线就不是那条环线了"


def test_overpass_is_a_grade_separated_crossing(built):
    """立交：疏解线从**高架底下**钻过去两次，而且两次都留够了净空。

    这条是"立体交叉"这个场景的全部意思，而它也是唯一一条**只查几何查不出来**的
    契约：疏解线完全可以贴着地面从高架**旁边**绕回去，那样环线照样闭合、列车照样
    能跑，场景却没有立交。所以这里逐点问一句：疏解线只要在平面上贴着主线（横向
    4 m 以内）走，主线就得在它头顶 5 m 以上。

    疏解线两端各是一个**汇入**主线的道岔（那是合流，不是交叉），先把两端各 60 m
    摘掉，免得把合流处当成"没净空的交叉"。
    """
    scene = built["overpass"]
    _, path = _path(scene)
    on_loop = {segment.piece_index for segment in path}
    loose = [i for i in range(len(scene.layout)) if i not in on_loop]
    assert loose, "立交场景应当有一条疏解线（不在主环上的那几件）"

    walk = [pose.position for pose in path.sample(_PROBE_STEP)]
    bypass = _offloop_points(scene, path)
    trim = int(60.0 / _PROBE_STEP)
    core = bypass[trim:-trim]
    assert len(core) > 100, "疏解线太短了，掐掉两端就没剩下什么"

    under: list[int] = []                       # 疏解线上"贴着主线"的那几点的序号
    for order, (x, y, z) in enumerate(core):
        near = min(walk, key=lambda p: (p[0] - x) ** 2 + (p[2] - z) ** 2)
        if math.hypot(near[0] - x, near[2] - z) > 4.0:
            continue
        assert near[1] - y >= 5.0, (
            f"疏解线在 ({x:.1f}, {z:.1f}) 处贴着主线，头顶只有 {near[1] - y:.2f} m"
            " —— 这不是立交，是相撞"
        )
        under.append(order)

    assert under, "疏解线没有一处走到主线底下，那就不是立交"
    # "贴着主线"的点必须是**两段**（钻进去、钻出来），中间隔着几十米
    runs = 1 + sum(1 for a, b in zip(under, under[1:]) if b - a > 5)
    assert runs == 2, (
        f"疏解线与主线的交叉分成 {runs} 段（应当 2 段：钻进去、钻出来）"
    )
    tallest = max(core[order][1] for order in under)
    assert tallest < 1.0, "疏解线应当始终在地面高度，不该自己爬上去"


def test_overpass_switch_decides_which_line_the_train_takes(built):
    """扳道岔 = 换一条线：直股是那条**高架**环线，岔股把列车引到桥下的疏解线。

    对用户来说"立交"就是这件事：道岔默认扳在直股（列车绕高架环线跑），把第一处
    道岔扳到岔股，主线就从那一处被截断 —— 列车没法再从直股穿过那处道岔，只能
    改走疏解线、从桥跨底下钻过去。道岔**可挤岔**：疏解线接在第二处道岔的岔股上，
    列车钻过桥下后从岔股逆向挤回主线、并入主环继续走（不再卡成死端）。这正是
    "扳道岔决定走哪条线"要验证的：档位一变，能走通的路就变。
    """
    scene = built["overpass"]
    turnouts = [index for index in range(len(scene.layout))
                if scene.layout.definition(index).is_switch]
    assert len(turnouts) == 2, "立交场景要两处道岔"

    report, loop = detect_closure(scene.layout, allow_open=True)
    assert report.closed and loop is not None, report.describe()
    on_loop = {segment.piece_index for segment in loop}
    bypass = [index for index in range(len(scene.layout)) if index not in on_loop]
    assert len(bypass) == 10, (
        f"默认（直股）时疏解线应当在环外，数出来 {len(bypass)} 件不在环上"
    )

    scene.layout.toggle_switch(turnouts[0])          # 第一处扳到岔股
    try:
        report, loop = detect_closure(scene.layout, allow_open=True)
        assert loop is not None
        now_on_loop = {segment.piece_index for segment in loop}
        assert all(index in now_on_loop for index in bypass), (
            "扳到岔股之后，列车应当改走疏解线（从桥底下穿过去）"
        )
    finally:
        scene.layout.toggle_switch(turnouts[0])      # 扳回直股，别把共享的场景弄脏
    report, loop = detect_closure(scene.layout, allow_open=True)
    assert report.closed and loop is not None, report.describe()



def test_recommended_train_runs_a_full_lap(app, built, trains):
    """把推荐编组放上线、推满手柄跑完一整圈。

    只查几何是不够的：闭环误差、净距全都合格，列车仍然可能在某处"卡住"（比如
    坡道与弯道叠在一起时转向架的弦长亏损把车拽出轨道）。所以这一条真的是推着它
    跑，跑够一圈的距离，并且跑完之后它还压在轨道上。
    """
    for key, scene in built.items():
        _, path = _path(scene)
        spec = trains[scene.train_id]
        view = TrainView(spec, app.render)
        try:
            view.state.s = path.total_length * 0.25
            assert view.sync(path), f"{key}：{view.placement_error}"
            assert view.placement_error is None

            view.set_handle(1.0)
            travelled = 0.0
            for _ in range(_SIM_STEPS):
                assert view.advance(_SIM_DT, path)
                travelled += view.state.v * _SIM_DT

            assert travelled >= path.total_length, (
                f"{key}：{spec.name} 只跑了 {travelled:.1f} m，"
                f"环线 {path.total_length:.1f} m"
            )
            assert all(math.isfinite(c) for state in view.states
                       for c in state.center), f"{key} 跑出了 NaN"
            assert view.sync(path), f"{key}：跑完一圈之后对不上轨道了"
        finally:
            view.destroy()


def test_the_whole_loop_is_drivable_for_the_named_train(app, built, trains):
    """把推荐编组沿环线一寸寸摆过去：任何一处都不能"摆不下"。

    闭环合格、净距合格，仍然可能在**某一段**摆不下 —— 比如坡道与弯道叠在一起
    时，转向架的弦长亏损让车体探到轨道外面去。逐点摆一遍是唯一能提前发现它的
    办法，而它的代价只是几十次摆位计算。
    """
    for key, scene in built.items():
        _, path = _path(scene)
        view = TrainView(trains[scene.train_id], app.render)
        try:
            s = 0.0
            while s < path.total_length:
                view.state.s = s
                assert view.sync(path), (
                    f"{key}：在 s = {s:.1f} m 处摆不下（{view.placement_error}）"
                )
                for state in view.states:
                    # 闭环上"车尾落在 s = 0 之前"是常态：``place_consist`` 记在
                    # ``BogieState.s`` 上的是**没取模**的弧长，而 ``pose_at`` 自己
                    # 会绕回来。所以这里不能拿 s ≥ 0 当"摆对了"的判据（车头在
                    # s = 0 处时车尾的 s 就是负的，那是环的另一头，不是脱轨），
                    # 要查的是"这个 s 说它在哪，它就在哪"。
                    for bogie in (state.front, state.rear):
                        pose = path.pose_at(bogie.s)
                        assert math.dist(bogie.position, pose.position) < 1e-9, (key, s)
                    assert math.isfinite(state.heading)
                s += 10.0
        finally:
            view.destroy()


# --------------------------------------------------------------------------- #
# 8. 存 / 读
# --------------------------------------------------------------------------- #

def test_scene_survives_a_save_and_reload(built, catalog, tmp_path):
    for key, scene in built.items():
        target = tmp_path / f"{key}.json"
        scenes.save(scene, target)
        assert target.exists()

        loaded = scenes.load(target, catalog)
        assert len(loaded.layout) == len(scene.layout)
        assert loaded.layout.total_length() == pytest.approx(
            scene.layout.total_length())
        assert loaded.scenery_key == key
        assert loaded.plot_size == scene.plot_size
        assert loaded.scenery.item_count == scene.scenery.item_count


def test_unknown_scene_is_a_clear_error():
    with pytest.raises(SceneError) as info:
        presets.build("没有这个场景")
    assert "没有这个场景" in str(info.value)
    for key in scenes.keys():
        assert key in str(info.value)


def test_scenes_are_deterministic(catalog):
    """同一个场景拼两遍必须逐点一致 —— 不然截图对不上，回归也没意义。"""
    first = presets.build("valley", catalog)
    second = presets.build("valley", catalog)
    assert first.layout.to_dict() == second.layout.to_dict()
    assert first.scenery == second.scenery
