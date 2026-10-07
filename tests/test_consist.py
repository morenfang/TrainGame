"""列车编组目录（``core/train/consist.py`` + ``data/trains.json``）的测试。

重点是两个设计承诺：
1. **换车 = 换一行数据**（改 JSON 就能加车型与编组，不动代码）；
2. **动力集中 vs 动力分散的差别由黏着质量自动体现**，不需要为车型写特例。
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from core.train.consist import (
    FEATURE_MARGIN,
    G,
    Band,
    CarShape,
    CarSpec,
    Doors,
    Nose,
    TrainCatalog,
    TrainCatalogError,
    Windows,
    _build_shape,
    build_train,
)


@pytest.fixture(scope="module")
def catalog() -> TrainCatalog:
    return TrainCatalog.builtin()


# --------------------------------------------------------------------------- #
# 目录整体
# --------------------------------------------------------------------------- #

def test_builtin_catalog_has_the_three_requested_trains(catalog):
    """用户明确要求的三类列车都要在。"""
    assert "green_skin_10" in catalog  # 老式绿皮车
    assert "crh380a_8" in catalog      # 和谐号动车组
    assert "cr400af_8" in catalog      # 复兴号动车组


def test_huangsidai_points_at_the_glb(catalog):
    """目前 models/ 里只留黄丝带一份 glb；8/16 节共用，其它编组走程序化车体。"""
    for train_id in ("cr400bf_huangsidai_8", "cr400bf_huangsidai_16"):
        assert catalog[train_id].mesh == "CR400BF_HuangSiDai_6car.glb"
    assert catalog["cr400bf_huangsidai_8"].car_count == 8
    assert catalog["cr400bf_huangsidai_16"].car_count == 16
    assert not catalog["cr400af_8"].mesh
    assert not catalog["crh380a_8"].mesh
    assert not catalog["green_skin_10"].mesh
    assert not catalog["steam_qj"].mesh


def test_huangsidai_16_has_cabs_facing_in_the_middle(catalog):
    """16 节重联：第 8、9 节都是头车，鼻锥对顶。"""
    spec = catalog["cr400bf_huangsidai_16"]
    assert [c.id for c in spec.cars[6:10]] == [
        "cr400bf_hsd_mid",
        "cr400bf_hsd_end",
        "cr400bf_hsd_end",
        "cr400bf_hsd_mid",
    ]
    facings = spec.car_facings()
    assert facings[0] is False
    assert facings[7] is True
    assert facings[8] is False
    assert facings[15] is True
    assert sum(facings) == 2


def test_catalog_reports_unknown_train_clearly(catalog):
    with pytest.raises(TrainCatalogError, match="unknown train"):
        catalog["no_such_train"]


def test_catalog_has_cardata_for_every_referenced_type(catalog):
    referenced = {car.id for train in catalog for car in train.cars}
    assert referenced <= set(catalog.car_types)


@pytest.mark.parametrize("train", list(TrainCatalog.builtin()))
def test_every_train_is_self_consistent(train):
    assert train.car_count >= 1
    assert train.total_mass > 0.0
    assert train.adhesive_mass > 0.0
    assert train.total_length > 0.0
    assert train.max_speed_ms > 0.0
    assert all(c > 0.0 for c in train.davis)
    # 每节车的转向架都在车体内部
    for car in train.cars:
        assert 2 * car.bogie_half_spacing <= car.length


# --------------------------------------------------------------------------- #
# 派生量
# --------------------------------------------------------------------------- #

def test_total_length_accounts_for_coupling_gaps(catalog):
    train = catalog["cr400af_8"]
    expected = sum(train.lengths) + train.coupling_gap * (train.car_count - 1)
    assert train.total_length == pytest.approx(expected, rel=1e-12)
    assert train.car_count == 8


def test_total_mass_is_the_sum_of_the_cars(catalog):
    train = catalog["green_skin_10"]
    assert train.total_mass == pytest.approx(
        sum(car.mass_kg for car in train.cars), rel=1e-12
    )


def test_adhesive_mass_only_counts_powered_cars(catalog):
    """绿皮车只有 1 台机车提供黏着；动车组几乎全列提供黏着。"""
    green = catalog["green_skin_10"]
    emu = catalog["cr400af_8"]

    locomotive = green.cars[0]
    assert locomotive.powered
    assert green.adhesive_mass == pytest.approx(locomotive.mass_kg, rel=1e-12)
    # 黏着质量占比：绿皮车极低，动车组极高
    assert green.adhesive_mass / green.total_mass < 0.25
    assert emu.adhesive_mass / emu.total_mass > 0.95


def test_adhesion_limit_uses_the_adhesive_mass(catalog):
    """牵引力上限 = adhesion * 黏着质量 * g —— 动力集中的瓶颈就在这里。"""
    green = catalog["green_skin_10"]
    expected = green.adhesion * green.adhesive_mass * G
    assert green.max_traction_from_adhesion == pytest.approx(expected, rel=1e-12)

    emu = catalog["cr400af_8"]
    # 动车组黏着极限远高于绿皮车（约 2.7 倍：全列动力 vs 单机）
    assert emu.max_traction_from_adhesion > 2.5 * green.max_traction_from_adhesion


def test_effective_traction_is_the_lower_of_the_two_limits(catalog):
    for train in catalog:
        assert train.effective_max_tractive_force == pytest.approx(
            min(train.max_tractive_force_n, train.max_traction_from_adhesion),
            rel=1e-12,
        )


def test_power_limited_crossover_speed(catalog):
    """P/v 与最大牵引力相等的那个速度 —— 牵引特性由恒牵引力转恒功率。"""
    emu = catalog["cr400af_8"]
    f = emu.effective_max_tractive_force
    assert emu.power_limited_speed() == pytest.approx(emu.power_w / f, rel=1e-12)
    # 动车组的拐点速度在 100~200 km/h 之间（真实特征）
    assert 100.0 / 3.6 < emu.power_limited_speed() < 200.0 / 3.6


def test_power_to_weight_orders_the_trains_as_designed(catalog):
    """概要设计 §5.7 的定性结论：复兴号 > 和谐号 >> 绿皮车。"""
    green = catalog["green_skin_10"].power_to_weight()
    hexie = catalog["crh380a_8"].power_to_weight()
    fuxing = catalog["cr400af_8"].power_to_weight()

    assert green < 5.0
    assert hexie > 20.0
    assert fuxing > hexie
    assert fuxing > 8 * green


def test_speed_limits(catalog):
    assert catalog["green_skin_10"].max_speed_ms == pytest.approx(100.0 / 3.6)
    assert catalog["crh380a_8"].max_speed_ms == pytest.approx(350.0 / 3.6)
    assert catalog["cr400af_8"].max_speed_ms == pytest.approx(350.0 / 3.6)


def test_more_coaches_makes_the_green_train_heavier_and_longer(catalog):
    short = catalog["green_skin_10"]
    long = catalog["green_skin_16"]
    assert long.total_mass > short.total_mass
    assert long.total_length > short.total_length
    # 功率一样，所以长编组更肉
    assert long.power_to_weight() < short.power_to_weight()


# --------------------------------------------------------------------------- #
# 涂装（render 层会用）
# --------------------------------------------------------------------------- #

def test_every_car_has_a_livery(catalog):
    for train in catalog:
        for car in train.cars:
            assert car.livery, f"{car.id} has no livery"
            assert car.livery.get("body", "").startswith("#")


def test_the_three_train_families_are_visually_distinguishable(catalog):
    """三种车的主色 / 色带必须不同 —— 这是「外观可辨」验收线的数据依据。"""
    green = {c.livery["body"] for c in catalog["green_skin_10"].cars}
    hexie = {c.livery["band"] for c in catalog["crh380a_8"].cars}
    fuxing = {c.livery["band"] for c in catalog["cr400af_8"].cars}
    assert not (green & hexie)
    assert not (hexie & fuxing)
    assert not (green & fuxing)


def test_green_skin_train_uses_the_classic_green_with_yellow_band(catalog):
    car = catalog.car_types["coach_25b"]
    assert car.livery["body"].lower() == "#2e6b3e"
    assert car.livery["band"].lower() == "#e8c84a"


def test_fuxing_uses_red_and_hexie_uses_blue(catalog):
    assert catalog.car_types["cr400af_mid"].livery["band"].lower() == "#c0392b"
    assert catalog.car_types["crh380a_mid"].livery["band"].lower() == "#1f5fa8"


# --------------------------------------------------------------------------- #
# 车体形状：断面 / 头型 / 门窗 / 色带
# --------------------------------------------------------------------------- #
# 这一节全部是"数据对不对"的检查。它们有一个共同点：**不需要开窗口**。
# 车体长什么样是纯几何，量得出来就能断言 —— 把"外观可辨"这条验收线从"看图
# 觉得像"降级成"数字对得上"，正是这一步存在的理由。


def _self_intersects(polygon) -> bool:
    """闭合折线是否有**非相邻**的两条边真正穿过彼此。"""
    n = len(polygon)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    for i in range(n):
        a1, a2 = polygon[i], polygon[(i + 1) % n]
        for j in range(i + 1, n):
            b1, b2 = polygon[j], polygon[(j + 1) % n]
            if b1 in (a1, a2) or b2 in (a1, a2):
                continue                      # 相邻（共用顶点）的边不算相交
            d1 = cross(b1, b2, a1)
            d2 = cross(b1, b2, a2)
            d3 = cross(a1, a2, b1)
            d4 = cross(a1, a2, b2)
            if (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0):
                return True
    return False


def test_every_car_type_has_a_shape(catalog):
    """``data/trains.json`` 里每个车型都必须有外观几何，否则这节车画不出来。"""
    assert catalog.validate() == []
    for car_id, car in catalog.car_types.items():
        assert car.shape is not None, f"{car_id} 缺 shape"


def test_shape_dimensions_are_derived_from_the_profile(catalog):
    """半宽 / 地板高 / 车高**不是数据**，是断面的推论 —— 改断面它们必须跟着变。"""
    shape = catalog.car_types["coach_25b"].shape
    assert shape.half_width == pytest.approx(max(p[0] for p in shape.profile))
    assert shape.floor_height == pytest.approx(min(p[1] for p in shape.profile))
    assert shape.roof_height == pytest.approx(max(p[1] for p in shape.profile))

    # 把断面整体加宽 50%：半宽必须跟着变，其余尺寸（地板高、车高）不动
    moved = replace(shape, profile=tuple((p[0] * 1.5, p[1]) for p in shape.profile))
    assert moved.half_width == pytest.approx(shape.half_width * 1.5)
    assert moved.floor_height == pytest.approx(shape.floor_height)
    assert moved.roof_height == pytest.approx(shape.roof_height)


def test_full_section_is_a_closed_symmetric_outline(catalog):
    """半断面镜像出来的全断面：点数正确、首尾在中线、左右严格对称。"""
    for car_id, car in catalog.car_types.items():
        shape = car.shape
        half = shape.profile
        full = shape.full_section()
        assert len(full) == 2 * len(half) - 2, car_id
        # 底中线与顶中线各只有一个点，不会被镜像出第二个
        assert full[0][0] == 0.0 and full[len(half) - 1][0] == 0.0, car_id
        # 绕序应当是逆时针（有向面积为正），否则放样出来的面片会朝里
        area = sum(
            full[i][0] * full[(i + 1) % len(full)][1]
            - full[(i + 1) % len(full)][0] * full[i][1]
            for i in range(len(full))
        )
        assert area > 0.0, f"{car_id}: 全断面绕序应为逆时针"
        for i in range(1, len(half) - 1):
            mirror = full[len(full) - i]
            assert full[i][0] == pytest.approx(-mirror[0]), car_id
            assert full[i][1] == pytest.approx(mirror[1]), car_id


def test_full_section_never_self_intersects(catalog):
    """镜像后的轮廓不自交 —— 这正是"高度单调不降"那条约束要保证的事。"""
    for car_id, car in catalog.car_types.items():
        full = car.shape.full_section()
        assert not _self_intersects(full), f"{car_id}: 断面轮廓自交"


def test_section_sizes_are_in_the_right_ballpark(catalog):
    """横断面取真实值约 0.9 倍：全宽 2.6~3.0 m、车高 3.2~3.8 m。

    下限是硬约束：全宽必须**明显大于轨距 1.435 m**，否则车会窄到"悬在钢轨里"。
    """
    from render.style import GAUGE

    for car_id, car in catalog.car_types.items():
        shape = car.shape
        width = 2.0 * shape.half_width
        assert width > GAUGE * 1.5, f"{car_id}: 全宽 {width:.2f} m 相对轨距太窄"
        assert 2.60 < width < 3.00, f"{car_id}: 全宽 {width:.2f} m 不合理"
        assert 3.20 < shape.roof_height < 3.80, f"{car_id}: 车高不合理"
        assert shape.floor_height > 0.8, f"{car_id}: 地板低于轨面"


def test_profile_heights_are_monotone_for_every_car(catalog):
    """高度单调不降：镜像后每条水平线最多与轮廓交两次，这是不自交的充分条件。"""
    for car_id, car in catalog.car_types.items():
        heights = [p[1] for p in car.shape.profile]
        assert heights == sorted(heights), car_id


# --------------------------------------------------------------------------- #
# 鼻锥（头型）
# --------------------------------------------------------------------------- #

def test_nose_scale_hits_both_ends_and_is_monotone(catalog):
    for car_id, car in catalog.car_types.items():
        nose = car.shape.nose
        assert nose.scale_at(0.0) == pytest.approx(nose.end_scale), car_id
        assert nose.scale_at(nose.length) == pytest.approx(1.0), car_id
        # 越界要钳制，而不是外推成 >1 或 <end_scale
        assert nose.scale_at(-5.0) == pytest.approx(nose.end_scale), car_id
        assert nose.scale_at(nose.length * 5.0) == pytest.approx(1.0), car_id
        samples = [nose.scale_at(nose.length * i / 40) for i in range(41)]
        assert samples == sorted(samples), f"{car_id}: 鼻锥缩放曲线不单调"


def test_nose_power_controls_how_pointed_the_nose_is(catalog):
    """指数越大越尖：跑到收敛段一半时，断面还剩下的比例越小。"""
    hexie = catalog.car_types["crh380a_end"].shape.nose
    fuxing = catalog.car_types["cr400af_end"].shape.nose
    assert hexie.power > fuxing.power
    at_half = lambda n: n.scale_at(0.5 * n.length)          # noqa: E731
    assert at_half(hexie) < at_half(fuxing)


def test_forward_nose_is_only_on_one_end(catalog):
    """头车只有一端是头型，另一端是平的贯通端 —— 编组时最后一节要镜像。"""
    for car_id in ("crh380a_end", "cr400af_end"):
        nose = catalog.car_types[car_id].shape.nose
        assert nose.ends == "forward", car_id
        assert nose.has_nose_at("end") and not nose.has_nose_at("start"), car_id
        assert len(catalog.car_types[car_id].nose_bands()) == 1, car_id

    for car_id in ("crh380a_mid", "cr400af_mid", "coach_25b", "df4b_loco"):
        nose = catalog.car_types[car_id].shape.nose
        assert nose.ends == "both", car_id
        assert len(catalog.car_types[car_id].nose_bands()) == 2, car_id


def test_the_three_families_have_distinct_silhouettes(catalog):
    """三款车头的量化差别 —— 这是「三种车外观可辨」（G3）的**数据依据**。

    只用涂装区分是不够的：远看一列车的辨识度主要来自侧影。所以头车收敛段占
    车长的比例必须拉开，且顺序与真实车型一致（绿皮车平头 < 复兴号 < 和谐号）。
    """
    def tapered(car_id: str) -> float:
        car = catalog.car_types[car_id]
        return car.shape.nose.length / car.length

    green = tapered("df4b_loco")          # 内燃机车：棱角分明的方头
    fuxing = tapered("cr400af_end")       # 复兴号：较短的"剑形"车头
    hexie = tapered("crh380a_end")        # 和谐号：细长尖鼻

    assert green < 0.05
    assert green < fuxing < hexie
    assert fuxing > 1.5 * green and hexie > 1.3 * fuxing

    # 鼻尖处的断面高度必须远小于整车 —— 否则"头型"只是个说辞
    for car_id in ("crh380a_end", "cr400af_end"):
        shape = catalog.car_types[car_id].shape
        tip = shape.nose.scale_at(0.0) * (shape.roof_height - shape.floor_height)
        full = shape.roof_height - shape.floor_height
        assert tip < 0.35 * full, car_id


# --------------------------------------------------------------------------- #
# 门窗与色带：位置是算出来的
# --------------------------------------------------------------------------- #

def test_window_bands_sit_inside_the_free_gap(catalog):
    """窗带必须完全落在"鼻锥与车门之间剩下的那段连续空档"里，且居中。"""
    for car_id, car in catalog.car_types.items():
        bands = car.window_bands()
        if not bands:
            continue
        room = car.window_room()
        assert room is not None, car_id
        assert room[0] <= bands[0][0], car_id
        assert bands[-1][1] <= room[1], car_id
        # 居中：两端留白相等
        assert (bands[0][0] - room[0]) == pytest.approx(room[1] - bands[-1][1]), car_id


def test_window_bands_never_touch_doors_or_the_nose(catalog):
    """把鼻锥、车门、车窗三组区间放到一起，两两之间都必须留出净空。"""
    for car_id, car in catalog.car_types.items():
        windows = car.window_bands()
        if not windows:
            continue
        blockers = car.nose_bands() + car.door_bands()
        for w0, w1 in windows:
            for b0, b1 in blockers:
                gap = max(b0, w0) - min(b1, w1)     # >0 即净空，<=0 即重叠
                assert gap >= FEATURE_MARGIN - 1e-9, (
                    f"{car_id}: 车窗 [{w0:.2f}, {w1:.2f}] 与 "
                    f"[{b0:.2f}, {b1:.2f}] 只隔 {gap:.3f} m"
                )


def test_windows_are_evenly_pitched_and_do_not_merge(catalog):
    for car_id, car in catalog.car_types.items():
        bands = car.window_bands()
        for (a0, a1), (b0, b1) in zip(bands, bands[1:]):
            assert b0 - a1 == pytest.approx(
                car.shape.windows.pitch - car.shape.windows.width
            ), car_id
            assert a1 < b0, f"{car_id}: 相邻车窗连成一片"
        assert all(0.0 <= a and b <= car.length for a, b in bands), car_id


def test_every_band_stays_on_the_car(catalog):
    for car in catalog.car_types.values():
        for band in car.nose_bands() + car.door_bands() + car.window_bands():
            assert 0.0 <= band[0] < band[1] <= car.length + 1e-9, car.id


def test_doors_make_way_for_the_nose(catalog):
    """有鼻锥的那一端，车门不能压在收敛段上。"""
    car = catalog.car_types["crh380a_end"]
    inset = car.shape.doors.inset
    width = car.shape.doors.width
    nose_length = car.shape.nose.length
    (front_door, front_end), (rear_door, rear_end) = car.door_bands()

    # -x 端没有鼻锥：门就贴在端部倒角之后
    assert front_door == pytest.approx(inset)
    assert front_end - front_door == pytest.approx(width)
    # +x 端有 2.30 m 鼻锥：门整体往车内让开鼻锥
    assert rear_end == pytest.approx(car.length - inset - nose_length)
    assert rear_end - rear_door == pytest.approx(width)
    # 所以两端门到各自车端面的距离并不相同
    assert front_door < car.length - rear_end


def test_cars_without_doors_still_produce_a_window_room(catalog):
    """内燃机车没有车门，窗带仍然要有地方放。"""
    loco = catalog.car_types["df4b_loco"]
    assert loco.shape.doors is None
    assert loco.door_bands() == ()
    assert loco.window_room() == (loco.shape.nose.length,
                                  loco.length - loco.shape.nose.length)
    assert loco.window_bands()


def test_stripe_and_window_band_do_not_overlap(catalog):
    for car_id, car in catalog.car_types.items():
        shape = car.shape
        if shape.stripe is None or shape.windows is None:
            continue
        assert not shape.stripe.overlaps(shape.windows.y), car_id


def test_green_skin_waist_stripe_sits_below_the_windows(catalog):
    """绿皮车的黄腰线在窗下沿以下 —— 这是它一望即知的画法。"""
    shape = catalog.car_types["coach_25b"].shape
    assert shape.stripe.y1 <= shape.windows.y.y0
    assert shape.stripe.height > 0.3


def test_emu_stripe_sits_below_a_higher_window_band(catalog):
    """动车组的窗带比绿皮车更靠上、也更高（观光车窗），飘带仍在窗带以下。"""
    coach = catalog.car_types["coach_25b"].shape
    for car_id in ("crh380a_mid", "cr400af_mid"):
        shape = catalog.car_types[car_id].shape
        assert shape.stripe.y1 <= shape.windows.y.y0, car_id
        assert shape.windows.y.y0 > coach.windows.y.y0, car_id
        assert shape.windows.y.height > coach.windows.y.height, car_id


# --------------------------------------------------------------------------- #
# 形状数据的错误必须在构建期被拦住
# --------------------------------------------------------------------------- #

_COACH_PROFILE = ((0.0, 1.05), (1.10, 1.05), (1.38, 1.28), (1.40, 1.70),
                  (1.40, 3.08), (1.30, 3.38), (0.96, 3.56), (0.0, 3.62))


def _win(count: int, width: float, pitch: float) -> Windows:
    return Windows(Band(1.98, 2.62), count, width, pitch)


def _coach_shape(profile=_COACH_PROFILE, nose=None, windows=None,
                 doors=None, stripe=None) -> CarShape:
    return CarShape(
        profile=tuple(profile),
        nose=nose if nose is not None else Nose(0.34, 0.955, 1.0, 2.35),
        windows=windows if windows is not None else _win(7, 0.75, 1.02),
        doors=doors if doors is not None else Doors(Band(1.05, 2.62), 1, 0.62, 0.55),
        stripe=stripe if stripe is not None else Band(1.30, 1.68),
    )


def _coach(shape: CarShape, length: float = 10.2) -> CarSpec:
    """把形状挂到一节车上；转向架间距随车长缩，免得撞在别的校验上。"""
    return CarSpec("probe", "probe", length, min(3.6, 0.4 * length), 55000.0, False,
                   livery={"body": "#2E6B3E"}, shape=shape)


def test_shape_parses_a_minimal_spec():
    """``windows`` / ``doors`` / ``stripe`` 都可以省 —— 省掉就是"没有"。"""
    shape = _build_shape("probe", {
        "profile": [[0.0, 1.0], [1.2, 1.0], [1.3, 2.0], [0.0, 3.0]],
        "nose": {"length": 0.2, "end_scale": 0.95, "power": 1.0,
                 "pivot_height": 2.0},
    })
    assert shape.windows is None and shape.doors is None and shape.stripe is None
    assert shape.nose.ends == "both"
    assert _coach(shape, 10.0).window_bands() == ()


@pytest.mark.parametrize("spec, match", [
    ({}, "缺少 'profile'"),
    ({"profile": [[0.0, 1.0], [1.2, 1.0], [1.3, 2.0], [0.0, 3.0]]}, "缺少 'nose'"),
    ({"profile": [[0.0, 1.0]], "nose": {"length": 0.2, "pivot_height": 2.0}},
     "至少要有 4 个点"),
    ({"profile": [[0.0, 1.0], [1.2, 1.0], [1.3, 2.0], [0.1, 3.0]],
      "nose": {"length": 0.2, "pivot_height": 2.0}}, "顶面中线"),
    ({"profile": [[0.0, 1.0], [-1.2, 1.0], [1.3, 2.0], [0.0, 3.0]],
      "nose": {"length": 0.2, "pivot_height": 2.0}}, "横向坐标必须为正"),
    ({"profile": [[0.0, 1.0], [1.2, 1.0], [1.3, 2.0], [0.0, 3.0]],
      "nose": {"pivot_height": 2.0}}, "缺少 'nose.length'"),
    ({"profile": [[0.0, 1.0], [1.2, 1.0], [1.3, 2.0], [0.0, 3.0]],
      "nose": {"length": 0.2}}, "缺少 'nose.pivot_height'"),
    ({"profile": [[0.0, 1.0], [1.2, 1.0], [1.3, 2.0], [0.0, 3.0]],
      "nose": {"length": 0.2, "pivot_height": 2.0},
      "windows": {"count": 3}}, "缺少 'windows.y'"),
    ({"profile": [[0.0, 1.0], [1.2, 1.0], [1.3, 2.0], [0.0, 3.0]],
      "nose": {"length": 0.2, "pivot_height": 2.0},
      "windows": "yes"}, "windows 必须是一个对象"),
])
def test_shape_parser_points_at_the_offending_key(spec, match):
    with pytest.raises(TrainCatalogError, match=match):
        _build_shape("probe", spec)


def test_missing_shape_is_flagged_by_validate():
    """车型没有形状时必须被 ``validate()`` 报出来，而不是等到出图才发现。"""
    catalog = TrainCatalog(
        trains={}, car_types={"bare": CarSpec("bare", "bare", 10.0, 3.0, 1000.0, True)}
    )
    assert any("shape" in w for w in catalog.validate())


#: 一份完全合法的 shape 规格（就是 ``coach_25b`` 那一份），下面的坏数据都是它的微改。
_GOOD_SHAPE_SPEC = {
    "profile": [[0.0, 1.05], [1.10, 1.05], [1.38, 1.28], [1.40, 1.70],
                [1.40, 3.08], [1.30, 3.38], [0.96, 3.56], [0.0, 3.62]],
    "nose": {"length": 0.34, "end_scale": 0.955, "power": 1.0, "pivot_height": 2.35},
    "windows": {"y": [1.98, 2.62], "count": 7, "width": 0.75, "pitch": 1.02},
    "doors": {"y": [1.05, 2.62], "per_end": 1, "width": 0.62, "inset": 0.55},
    "stripe": [1.30, 1.68],
}


@pytest.mark.parametrize("section, override, match", [
    ("profile", [[0.0, 1.05], [1.10, 1.05], [1.38, 3.28], [1.40, 1.70],
                 [1.40, 3.08], [1.30, 3.38], [0.96, 3.56], [0.0, 3.62]],
     "单调不降"),
    ("profile", [[0.2, 1.05], [1.10, 1.05], [1.38, 1.28], [1.40, 1.70],
                 [1.40, 3.08], [1.30, 3.38], [0.96, 3.56], [0.0, 3.62]],
     "底面中线"),
    ("profile", [[0.0, 1.05], [1.10, 1.05], [0.0, 1.28], [1.40, 1.70],
                 [1.40, 3.08], [1.30, 3.38], [0.96, 3.56], [0.0, 3.62]],
     "横向坐标必须为正"),
    ("profile", [[0.0, 1.05], [1.10, 1.05], [1.38, 1.28]], "至少要有 4 个点"),
    # 车顶只有 2.2 m，却把鼻锥支点写在 2.35 m —— 鼻尖会被顶到车顶外面
    ("profile", [[0.0, 1.05], [1.20, 1.05], [1.40, 2.00], [0.0, 2.20]],
     "鼻锥支点高度"),
    ("nose", {"end_scale": 1.4}, "端面缩放"),
    ("nose", {"length": 0.0, "end_scale": 0.5}, "零长度的收敛段"),
    ("nose", {"power": 0.0}, "收敛指数必须为正"),
    ("nose", {"pivot_height": 99.0}, "鼻锥支点高度"),
    ("nose", {"ends": "sideways"}, "ends 必须是"),
    ("stripe", [2.0, 2.6], "重叠"),
    ("stripe", [2.6, 2.0], "上沿必须高于下沿"),
    ("windows", {"count": 7, "width": 0.8, "pitch": 0.8}, "中心距"),
    ("windows", {"count": -1, "width": 0.0, "pitch": 0.0}, "不能为负"),
    ("doors", {"per_end": 2}, "只支持 0 或 1"),
])
def test_shape_rejects_bad_geometry(section, override, match):
    """坏数据要走**完整的 JSON 解析 → 构建**那条路，和真实载入一模一样。"""
    spec = dict(_GOOD_SHAPE_SPEC)
    spec[section] = ({**spec[section], **override}
                     if isinstance(override, dict) else override)
    with pytest.raises(TrainCatalogError, match=match):
        _coach(_build_shape("probe", spec))


@pytest.mark.parametrize("break_it, length, match", [
    # 6 m 的车里放不下默认的 7 扇窗
    (dict(), 6.0, "只剩下"),
    # 两端鼻锥加起来比车还长
    (dict(nose=Nose(2.0, 0.5, 1.0, 2.35)), 4.0, "会碰在一起"),
    # 单端鼻锥比车还长
    (dict(nose=Nose(3.95, 0.5, 1.0, 2.35, ends="forward")), 4.0, "的车还长"),
])
def test_shape_rejects_geometry_that_does_not_fit_the_car(break_it, length, match):
    """形状本身合法、但放到这个车长上放不下 —— 这类错只有车长知道。"""
    with pytest.raises(TrainCatalogError, match=match):
        _coach(_coach_shape(**break_it), length)


def test_a_closed_window_room_is_reported():
    """门横在车体中央、车窗没地方放 —— 报错要说清楚是"没有连续空档"。"""
    shape = _coach_shape(doors=Doors(Band(1.05, 2.62), 1, 6.0, 3.3))
    with pytest.raises(TrainCatalogError, match="横跨车体中心"):
        _coach(shape)


def test_band_overlap_is_strict_and_touching_is_allowed():
    """相接不算重叠 —— 否则数据作者只好去凑一个说不出理由的小缝隙。"""
    assert not Band(1.0, 2.0).overlaps(Band(2.0, 3.0))
    assert not Band(1.0, 2.0).overlaps(Band(0.0, 1.0))
    assert Band(1.0, 2.0).overlaps(Band(1.5, 3.0))
    assert Band(1.0, 3.0).overlaps(Band(1.5, 2.0))
    assert Band(1.0, 2.0).contains(1.5) and Band(1.0, 2.0).contains(1.0)
    assert not Band(1.0, 2.0).contains(2.5)
    assert Band(1.0, 2.0).height == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 构建期的错误必须被拦住
# --------------------------------------------------------------------------- #

def test_build_rejects_empty_formation():
    with pytest.raises(TrainCatalogError, match="empty 'formation'"):
        build_train(
            {"id": "x", "formation": [], "max_tractive_force_n": 1.0,
             "max_brake_force_n": 1.0},
            {},
        )


def test_build_rejects_unknown_car_type():
    with pytest.raises(TrainCatalogError, match="unknown car type"):
        build_train(
            {"id": "x", "formation": [["ghost", 1]],
             "max_tractive_force_n": 1.0, "max_brake_force_n": 1.0},
            {},
        )


def test_build_rejects_malformed_formation_entry():
    car_types = {"c": CarSpec("c", "c", 10.0, 3.0, 1000.0, True)}
    with pytest.raises(TrainCatalogError, match=r"\[car_type, count\]"):
        build_train(
            {"id": "x", "formation": [["c", 1, 2]],
             "max_tractive_force_n": 1.0, "max_brake_force_n": 1.0},
            car_types,
        )


def test_build_rejects_non_positive_count():
    car_types = {"c": CarSpec("c", "c", 10.0, 3.0, 1000.0, True)}
    with pytest.raises(TrainCatalogError, match="must be positive"):
        build_train(
            {"id": "x", "formation": [["c", 0]],
             "max_tractive_force_n": 1.0, "max_brake_force_n": 1.0},
            car_types,
        )


def test_build_rejects_a_train_with_no_powered_car():
    """没有动力车 —— 永远开不动，必须在校验期就拦住。"""
    car_types = {"c": CarSpec("c", "c", 10.0, 3.0, 1000.0, False)}
    with pytest.raises(TrainCatalogError, match="no powered car"):
        build_train(
            {"id": "x", "formation": [["c", 1]],
             "max_tractive_force_n": 1.0, "max_brake_force_n": 1.0},
            car_types,
        )


def test_build_rejects_a_bad_davis_vector():
    car_types = {"c": CarSpec("c", "c", 10.0, 3.0, 1000.0, True)}
    with pytest.raises(TrainCatalogError, match="'davis' must have 3"):
        build_train(
            {"id": "x", "formation": [["c", 1]], "davis": [1.0, 2.0],
             "max_tractive_force_n": 1.0, "max_brake_force_n": 1.0},
            car_types,
        )


def test_car_type_rejects_bogies_outside_the_body():
    with pytest.raises(TrainCatalogError, match="exceeds car length"):
        CarSpec("c", "c", 5.0, 4.0, 1000.0, True)


def test_car_type_rejects_non_positive_mass():
    with pytest.raises(TrainCatalogError, match="mass must be positive"):
        CarSpec("c", "c", 10.0, 3.0, 0.0, True)


# --------------------------------------------------------------------------- #
# 与轨道几何的配合
# --------------------------------------------------------------------------- #

def test_consist_fits_on_the_standard_loop(catalog):
    """默认环线周长 2*pi*40 ≈ 251 m；三种车都要放得下。"""
    circumference = 2 * math.pi * 40.0
    for train in catalog:
        assert train.total_length < circumference, train.id


def test_eight_car_emu_is_about_a_third_of_the_loop(catalog):
    """概要设计 §5.8 的尺度意图：8 节编组约占环线 1/3。"""
    circumference = 2 * math.pi * 40.0
    ratio = catalog["cr400af_8"].total_length / circumference
    assert 0.28 < ratio < 0.42


def test_long_green_train_is_a_large_fraction_of_the_loop(catalog):
    """16 辆绿皮车很长，这正是「长编组在沙盘上很壮观」的来源。"""
    circumference = 2 * math.pi * 40.0
    ratio = catalog["green_skin_16"].total_length / circumference
    assert ratio > 0.5
