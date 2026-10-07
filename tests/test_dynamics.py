"""列车纵向动力学（``core/train/dynamics.py``）的测试。

参数标定的参考数字见 ``scripts/calibrate_trains.py``。这里的断言分两类：

* **物理正确性**：受力表达式、恒牵引力/恒功率分段、黏着极限、积分行为；
* **设计意图**：绿皮车肉、复兴号最猛 —— 这是用户要「感觉得到」的差别。
"""

from __future__ import annotations

import math

import pytest

from core.train.consist import CarSpec, TrainCatalog, TrainSpec, build_train
from core.train.dynamics import (
    Train,
    TrainState,
    acceleration,
    brake_force,
    resistance_force,
    steady_state_speed,
    step,
    time_to_reach,
    tractive_force,
)
from core.track.catalog import Catalog
from core.track.layout import Layout
from core.track.path import trace

KMH = 1.0 / 3.6


@pytest.fixture(scope="module")
def catalog() -> TrainCatalog:
    return TrainCatalog.builtin()


@pytest.fixture(scope="module")
def green(catalog) -> TrainSpec:
    """目录里已不挂绿皮编组；测试用车型库现拼一份，参数与旧 green_skin_10 一致。"""
    return build_train(
        {
            "id": "green_skin_10",
            "coupling_gap": 0.3,
            "power_w": 1985000,
            "max_tractive_force_n": 300000,
            "max_brake_force_n": 584800,
            "max_emergency_brake_force_n": 1651200,
            "adhesion": 0.25,
            "davis": [0.015, 1.5e-4, 2.0e-5],
            "max_speed_kmh": 100,
            "formation": [["df4b_loco", 1], ["coach_25b", 10]],
        },
        catalog.car_types,
    )


@pytest.fixture(scope="module")
def hexie(catalog) -> TrainSpec:
    return build_train(
        {
            "id": "crh380a_8",
            "coupling_gap": 0.25,
            "power_w": 9600000,
            "max_tractive_force_n": 250000,
            "max_brake_force_n": 383000,
            "max_emergency_brake_force_n": 919200,
            "adhesion": 0.25,
            "davis": [0.010, 1.1e-4, 1.4e-5],
            "max_speed_kmh": 350,
            "formation": [
                ["crh380a_end", 1],
                ["crh380a_mid", 6],
                ["crh380a_end", 1],
            ],
        },
        catalog.car_types,
    )


@pytest.fixture(scope="module")
def fuxing(catalog) -> TrainSpec:
    return build_train(
        {
            "id": "cr400af_8",
            "coupling_gap": 0.25,
            "power_w": 11000000,
            "max_tractive_force_n": 260000,
            "max_brake_force_n": 367000,
            "max_emergency_brake_force_n": 880800,
            "adhesion": 0.25,
            "davis": [0.009, 1.0e-4, 1.25e-5],
            "max_speed_kmh": 350,
            "formation": [
                ["cr400af_end", 1],
                ["cr400af_mid", 6],
                ["cr400af_end", 1],
            ],
        },
        catalog.car_types,
    )

def make_spec(*, mass: float = 400_000.0, power: float = 5_000_000.0,
              max_te: float = 200_000.0, brake: float = 200_000.0,
              adhesion: float = 1.0, powered: bool = True,
              vmax: float = 300.0) -> TrainSpec:
    """构造一个参数可控的合成编组，便于单点验证受力公式。"""
    car = CarSpec("test_car", "test", 20.0, 7.0, mass, powered)
    return build_train(
        {
            "id": "test_train",
            "formation": [["test_car", 1]],
            "coupling_gap": 0.0,
            "power_w": power,
            "max_tractive_force_n": max_te,
            "max_brake_force_n": brake,
            "adhesion": adhesion,
            "davis": [0.0, 0.0, 0.0],  # 关掉阻力，单独看牵引
            "max_speed_kmh": vmax,
        },
        {"test_car": car},
    )


# --------------------------------------------------------------------------- #
# 牵引特性：恒牵引力段 / 恒功率段 / 黏着极限
# --------------------------------------------------------------------------- #

def test_full_throttle_at_rest_gives_the_max_tractive_force(fuxing):
    assert tractive_force(fuxing, 0.0, 1.0) == pytest.approx(
        fuxing.effective_max_tractive_force, rel=1e-12
    )


def test_high_speed_falls_into_the_constant_power_region(fuxing):
    """超过拐点速度后 F = P/v —— 这是高速列车加速越来越慢的根本原因。"""
    v = 90.0  # 324 km/h，远超拐点
    assert tractive_force(fuxing, v, 1.0) == pytest.approx(fuxing.power_w / v, rel=1e-12)


def test_crossover_speed_is_where_both_branches_meet(fuxing):
    v_cross = fuxing.power_limited_speed()
    assert tractive_force(fuxing, v_cross, 1.0) == pytest.approx(
        fuxing.effective_max_tractive_force, rel=1e-9
    )
    assert tractive_force(fuxing, v_cross * 2, 1.0) == pytest.approx(
        fuxing.effective_max_tractive_force / 2, rel=1e-9
    )


def test_traction_is_monotonically_decreasing_with_speed(fuxing):
    forces = [tractive_force(fuxing, v, 1.0) for v in range(0, 100, 5)]
    assert all(a >= b for a, b in zip(forces, forces[1:]))


def test_throttle_scales_traction_linearly(fuxing):
    full = tractive_force(fuxing, 10.0, 1.0)
    assert tractive_force(fuxing, 10.0, 0.5) == pytest.approx(full / 2, rel=1e-12)
    assert tractive_force(fuxing, 10.0, 0.0) == pytest.approx(0.0)


def test_throttle_is_clamped_to_the_unit_interval(fuxing):
    assert tractive_force(fuxing, 10.0, 5.0) == pytest.approx(
        tractive_force(fuxing, 10.0, 1.0)
    )
    assert tractive_force(fuxing, 10.0, -1.0) == pytest.approx(0.0)


def test_adhesion_limit_caps_traction():
    """特意把黏着系数调低：牵引力上限必须掉到黏着极限，而不是标称牵引力。"""
    weak = make_spec(mass=100_000.0, max_te=500_000.0, adhesion=0.10)
    expected_adhesion = 0.10 * 100_000.0 * 9.80665
    assert weak.max_traction_from_adhesion == pytest.approx(expected_adhesion, rel=1e-12)
    assert weak.effective_max_tractive_force == pytest.approx(expected_adhesion)
    assert tractive_force(weak, 0.0, 1.0) == pytest.approx(expected_adhesion, rel=1e-12)


def test_green_skin_traction_is_limited_by_the_locomotive_only(green):
    """绿皮车「动力集中」：黏着质量只有 1 台机车，这是它起步无力的原因。"""
    assert green.max_traction_from_adhesion == pytest.approx(
        green.adhesion * green.cars[0].mass_kg * 9.80665, rel=1e-12
    )
    assert green.adhesive_mass < green.total_mass / 4.0


# --------------------------------------------------------------------------- #
# 阻力与坡道
# --------------------------------------------------------------------------- #

def test_resistance_is_never_negative_on_the_flat(fuxing):
    for v in range(0, 120, 10):
        assert resistance_force(fuxing, v) > 0.0


def test_resistance_grows_with_speed(fuxing):
    forces = [resistance_force(fuxing, float(v)) for v in range(0, 120, 10)]
    assert all(a < b for a, b in zip(forces, forces[1:]))


def test_quadratic_term_dominates_at_high_speed(fuxing):
    """高速时 c*v^2 项应当明显超过常数项 —— 这是真实阻力的特征。"""
    low = resistance_force(fuxing, 1.0) / fuxing.total_mass
    high = resistance_force(fuxing, 100.0) / fuxing.total_mass
    assert high > 10 * low


def test_uphill_grade_adds_resistance_and_downhill_subtracts_it(fuxing):
    flat = resistance_force(fuxing, 10.0, grade=0.0)
    uphill = resistance_force(fuxing, 10.0, grade=0.03)
    downhill = resistance_force(fuxing, 10.0, grade=-0.03)
    assert uphill > flat > downhill


def test_steep_downhill_accelerates_with_no_throttle(fuxing):
    """空挡下坡会被重力拽着走 —— 坡道分量真的进了方程。"""
    assert acceleration(fuxing, 10.0, throttle=0.0, grade=-0.03) > 0.0


def test_grade_uses_the_sine_so_steep_slopes_stay_sane(fuxing):
    """坡度以正切给出，内部换算成正弦，因此不会在陡坡上算出无穷大的分力。"""
    flat = resistance_force(fuxing, 0.0, grade=0.0)
    steep = resistance_force(fuxing, 0.0, grade=1.0)  # 45°
    expected_slope = fuxing.total_mass * 9.80665 * math.sin(math.pi / 4)
    assert steep - flat == pytest.approx(expected_slope, rel=1e-9)


# --------------------------------------------------------------------------- #
# 加速度：设计意图（绿皮车肉、复兴号猛）
# --------------------------------------------------------------------------- #

def test_starting_acceleration_is_set_by_traction_not_power(green, fuxing):
    """起步阶段两者都在「恒牵引力段」，差距**不大** —— 这是真实的。

    机车起步牵引力是按车钩强度设计的（300 kN 对 138 t 的机车而言很大），所以
    绿皮车并不是"起步没劲"，而是**维持不住**：它很快就进入恒功率段。
    """
    a_green = acceleration(green, 0.0, 1.0)
    a_fuxing = acceleration(fuxing, 0.0, 1.0)
    assert 0.3 < a_green < 0.5
    assert 0.6 < a_fuxing < 0.8
    assert a_fuxing > a_green


def test_green_train_enters_the_power_limited_region_almost_at_once(green, fuxing):
    """真正拉开差距的地方：拐点速度相差 6 倍。

    绿皮车 24 km/h 就转恒功率（之后就越来越没劲），复兴号到 152 km/h 还在恒牵引力段。
    """
    assert green.power_limited_speed() * 3.6 < 30.0
    assert fuxing.power_limited_speed() * 3.6 > 120.0
    assert fuxing.power_limited_speed() > 6 * green.power_limited_speed()


def test_at_speed_the_gap_becomes_dramatic(green, fuxing):
    """80 km/h 时：绿皮车几乎没余力，复兴号还在猛推（差 6 倍以上）。"""
    v = 80.0 * KMH
    a_green = acceleration(green, v, throttle=1.0)
    a_fuxing = acceleration(fuxing, v, throttle=1.0)
    assert a_green < 0.15
    assert a_fuxing > 5 * a_green


def test_time_to_accelerate_matches_real_world_ballparks(green, hexie, fuxing):
    """校准断言：加速时间必须与真实车型同量级（不是精确值，是量级）。"""
    # 复兴号 0-200 km/h 大约 1.5~3 分钟
    assert 60.0 < time_to_reach(fuxing, 200.0) < 180.0
    # 和谐号比复兴号慢一点
    assert time_to_reach(hexie, 200.0) > time_to_reach(fuxing, 200.0)
    # 绿皮车 0-100 km/h 要好几分钟
    assert 120.0 < time_to_reach(green, 100.0) < 400.0


def test_green_train_cannot_reach_200_kmh(green):
    """速度上限就是上限：全手柄也到不了。"""
    assert time_to_reach(green, 200.0) is None


def test_green_train_struggles_on_a_grade_where_the_emu_does_not(green, fuxing):
    """绿皮车在小坡上就掉速，复兴号照样跑 —— 沙盘上最容易看出的差别。"""
    assert steady_state_speed(green, grade=0.01) * 3.6 < 90.0
    assert steady_state_speed(fuxing, grade=0.01) * 3.6 == pytest.approx(350.0, rel=0.02)


def test_green_train_cannot_even_start_on_a_5_percent_grade(green):
    """5% 坡上绿皮车起步就超黏着/牵引能力 —— 这正是真实铁路限坡的由来。"""
    assert acceleration(green, 0.0, throttle=1.0, grade=0.05) < 0.0
    assert time_to_reach(green, 10.0, grade=0.05) is None


def test_time_to_reach_needs_a_finite_target(fuxing):
    assert time_to_reach(fuxing, 0.0) is not None
    assert time_to_reach(fuxing, 100.0) is not None


# --------------------------------------------------------------------------- #
# 制动
# --------------------------------------------------------------------------- #

def test_brake_force_scales_with_the_brake_notch(fuxing):
    assert brake_force(fuxing, 0.0) == pytest.approx(0.0)
    assert brake_force(fuxing, 0.5) == pytest.approx(fuxing.max_brake_force_n / 2)
    assert brake_force(fuxing, 1.5) == pytest.approx(fuxing.max_brake_force_n)


def test_emergency_braking_is_much_stronger_than_service_braking(green, fuxing):
    """**紧急制动必须"明显"强于把制动位推到底。**

    否则 Shift+空格 与 ↓ 拉到底就是同一件事，用户按下去只会觉得"没反应" ——
    这正是"紧急刹车应该快速刹停"那条抱怨的由来（拆分前两档一样大）。
    """
    for spec in (green, fuxing):
        service = spec.brake_deceleration()
        emergency = spec.brake_deceleration(emergency=True)
        assert 0.8 <= service <= 1.1, f"{spec.id} 常用制动减速度 {service:.3f} 不像真的"
        assert emergency > 2.0 * service, f"{spec.id} 紧急制动不够狠"
        assert brake_force(spec, 1.0, emergency=True) > brake_force(spec, 1.0)


def test_emergency_braking_stops_a_train_fast(fuxing):
    """100 km/h 一把拉死：距离与时间都要短到"刹得住"这个量级。

    常用制动同样不该再是"滑出去一公里"（拆分前绿皮车只有 0.38 m/s²，
    100 km/h 要 1 km 才停）。
    """
    v0 = 100.0 * KMH

    def stopping(emergency: bool) -> tuple[float, float]:
        state = TrainState(v=v0, brake=1.0, emergency=emergency)
        distance = 0.0
        time = 0.0
        while state.v > 0.0 and distance < 5000.0:
            before = state.v
            step(state, fuxing, 0.01)
            distance += 0.5 * (before + state.v) * 0.01
            time += 0.01
        return distance, time

    emergency_distance, emergency_time = stopping(emergency=True)
    service_distance, _ = stopping(emergency=False)

    assert emergency_distance < 180.0, f"紧急制动滑了 {emergency_distance:.0f} m"
    assert emergency_time < 13.0
    assert emergency_distance < 0.6 * service_distance


def test_braking_uses_the_emergency_peak_only_when_asked(fuxing):
    """没标 emergency 时一切照旧 —— 于是老数据/合成编组的结论一字不变。"""
    assert brake_force(fuxing, 1.0, emergency=False) == pytest.approx(
        fuxing.max_brake_force_n
    )
    assert acceleration(fuxing, 20.0, brake=1.0, emergency=True) < \
        acceleration(fuxing, 20.0, brake=1.0)


def test_drive_boost_scales_traction_and_braking_but_not_resistance(fuxing):
    """沙盘手感倍率只放大牵引与制动。

    阻力与坡度**不**参与放大：否则"绿皮车在小坡上就掉速"这类结论会被一起改掉
    （那正是 `data/trains.json` 标定值要守住的东西）。
    """
    resistance = resistance_force(fuxing, 10.0) / fuxing.total_mass
    plain = acceleration(fuxing, 10.0, throttle=1.0)
    boosted = acceleration(fuxing, 10.0, throttle=1.0, boost=2.0)
    assert boosted == pytest.approx(2.0 * (plain + resistance) - resistance, rel=1e-12)

    # 惰行（既不牵引也不制动）不受倍率影响
    assert acceleration(fuxing, 10.0, boost=2.0) == pytest.approx(
        acceleration(fuxing, 10.0), rel=1e-12
    )
    # 坡道分量也不受影响
    assert acceleration(fuxing, 10.0, grade=-0.02, boost=3.0) == pytest.approx(
        acceleration(fuxing, 10.0, grade=-0.02), rel=1e-12
    )


def test_a_stationary_train_will_not_move_on_a_flat_without_a_handle(green):
    """守住惰行的下限：不给手柄、平道，车必须原地不动（不会自己溜）。

    手感倍率把牵引力也放大了，所以这条要连着倍率一起验：倍率乘的是"手柄给出多少
    牵引"，手柄在 0 位时它必须仍然是 0。
    """
    state = TrainState(v=0.0)
    for _ in range(200):
        step(state, green, 0.05, boost=5.0)
    assert state.v == pytest.approx(0.0, abs=1e-12)


def test_braking_decelerates(fuxing):
    coasting = acceleration(fuxing, 20.0, throttle=0.0, brake=0.0)
    braking = acceleration(fuxing, 20.0, throttle=0.0, brake=1.0)
    assert braking < coasting < 0.0


# --------------------------------------------------------------------------- #
# 积分：速度永不越界、永不倒退
# --------------------------------------------------------------------------- #

def test_train_never_exceeds_its_speed_limit(green):
    state = TrainState(throttle=1.0)
    for _ in range(20000):
        step(state, green, 0.1)
    assert state.v == pytest.approx(green.max_speed_ms, rel=1e-12)
    assert state.speed_kmh == pytest.approx(100.0, rel=1e-9)


def test_braking_from_speed_stops_without_reversing(green):
    state = TrainState(v=20.0, brake=1.0)
    for _ in range(20000):
        step(state, green, 0.05)
    assert state.v == pytest.approx(0.0, abs=1e-12)
    assert state.s >= 0.0


def test_coasting_eventually_stops(fuxing):
    """松手柄滑行：阻力会把它慢慢拖停，停在原地不倒退。"""
    state = TrainState(v=40.0)
    for _ in range(200000):
        step(state, fuxing, 0.1)
    assert state.v == pytest.approx(0.0, abs=1e-9)


def test_speed_to_kmh_conversion(green):
    state = TrainState(v=25.0)
    assert state.speed_kmh == pytest.approx(90.0)
    assert green.max_speed_ms == pytest.approx(100.0 * KMH)


def test_state_reports_stopped(green):
    assert TrainState().is_stopped
    assert not TrainState(v=1.0).is_stopped


def test_step_accumulates_time_and_distance(fuxing):
    state = TrainState(throttle=1.0)
    for _ in range(100):
        step(state, fuxing, 0.1)
    assert state.elapsed == pytest.approx(10.0, rel=1e-12)
    assert state.distance > 0.0
    assert state.distance == pytest.approx(state.s, rel=1e-12)


def test_step_ignores_non_positive_dt(fuxing):
    state = TrainState(throttle=1.0)
    step(state, fuxing, 0.0)
    step(state, fuxing, -1.0)
    assert state.elapsed == 0.0


def test_integration_is_close_to_analytic_constant_acceleration():
    """恒牵引力段上，位移应当接近 1/2*a*t^2（梯形积分的误差很小）。"""
    spec = make_spec(mass=100_000.0, power=1e9, max_te=100_000.0, adhesion=1.0)
    a = acceleration(spec, 0.0, 1.0)
    state = TrainState(throttle=1.0)
    dt = 0.001
    for _ in range(1000):
        step(state, spec, dt)
    t = 1000 * dt
    assert state.v == pytest.approx(a * t, rel=1e-3)
    assert state.distance == pytest.approx(0.5 * a * t * t, rel=1e-3)


def test_result_is_dt_independent(fuxing):
    """步长不影响结论（模拟与帧率无关）。"""
    coarse = TrainState(throttle=1.0)
    fine = TrainState(throttle=1.0)
    for _ in range(100):
        step(coarse, fuxing, 0.1)
    for _ in range(1000):
        step(fine, fuxing, 0.01)
    assert coarse.v == pytest.approx(fine.v, rel=0.01)
    assert coarse.s == pytest.approx(fine.s, rel=0.01)


def test_simulation_is_deterministic(fuxing):
    first = TrainState(throttle=1.0)
    second = TrainState(throttle=1.0)
    for _ in range(500):
        step(first, fuxing, 0.05)
        step(second, fuxing, 0.05)
    assert first.v == second.v
    assert first.s == second.s


# --------------------------------------------------------------------------- #
# 与轨道闭环的配合
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def loop():
    catalog = Catalog.builtin()
    layout = Layout(catalog=catalog)
    first = layout.add_root("curve_r40_l45")
    index = first
    for _ in range(7):
        index = layout.attach("curve_r40_l45", "a", (index, "b"))
    layout.connect((index, "b"), (first, "a"))
    return trace(layout, (first, "a"))


def test_position_wraps_around_a_closed_loop(fuxing, loop):
    state = TrainState(s=loop.total_length - 1.0, throttle=1.0)
    step(state, fuxing, 1.0, total_length=loop.total_length, closed=True)
    assert 0.0 <= state.s < loop.total_length


def test_open_path_stops_at_the_end(fuxing):
    state = TrainState(s=0.0, throttle=1.0)
    for _ in range(20000):
        step(state, fuxing, 0.1, total_length=100.0, closed=False)
    assert state.s == pytest.approx(100.0)
    assert state.v == pytest.approx(0.0)


def test_train_advances_along_the_loop_and_stays_on_the_rail(fuxing, loop):
    """端到端：跑起来之后，整列车的每个转向架都还在轨道中心线上。"""
    train = Train(fuxing)
    train.set_throttle(1.0)

    cx, cz = 0.0, 40.0
    for _ in range(600):  # 60 秒
        train.advance(0.1, loop)
        cars = train.place(loop)
        for car in cars:
            for bogie in (car.front, car.rear):
                radius = math.hypot(bogie.position[0] - cx, bogie.position[2] - cz)
                assert radius == pytest.approx(40.0, abs=1e-6)


def test_train_place_returns_cars_front_to_back(fuxing, loop):
    train = Train(fuxing, TrainState(s=100.0))
    cars = train.place(loop)
    assert len(cars) == fuxing.car_count
    # 0 号车在最前面（弧长最大）
    for previous, following in zip(cars, cars[1:]):
        assert previous.front.s > following.front.s


def test_train_spec_length_must_fit_the_loop(catalog, loop):
    """16 辆绿皮车在 251 m 环线上要放得下（否则会自己追尾）。"""
    longest = max(catalog, key=lambda t: t.total_length)
    assert longest.total_length < loop.total_length


def test_train_run_is_deterministic_end_to_end(fuxing, loop):
    def run() -> float:
        train = Train(fuxing, TrainState(s=0.0, throttle=1.0))
        train.simulate(30.0, dt=0.05, path=loop)
        return train.state.v

    assert run() == run()


# --------------------------------------------------------------------------- #
# 控制接口
# --------------------------------------------------------------------------- #

def test_throttle_and_brake_are_mutually_exclusive(fuxing):
    train = Train(fuxing)
    train.set_throttle(1.0)
    assert train.state.brake == 0.0
    train.set_brake(0.7)
    assert train.state.throttle == 0.0
    assert train.state.brake == pytest.approx(0.7)


def test_emergency_stop_and_release(fuxing):
    train = Train(fuxing, TrainState(v=30.0, throttle=1.0))
    train.emergency_stop()
    assert train.state.throttle == 0.0 and train.state.brake == 1.0
    assert train.is_emergency
    train.release()
    assert train.state.throttle == 0.0 and train.state.brake == 0.0
    assert not train.is_emergency


def test_emergency_braking_stays_latched_until_the_driver_takes_over(fuxing):
    """紧急制动是**锁存**，这是它区别于"把制动位推到底"的地方。

    往下再补一点制动力**不该**把它降级成常用制动 —— 用户按 Shift+空格 之后再点
    一下 ↓（手柄本来就在 -1.0），如果制动力因此变回常用值，那就是个隐蔽的踩坑：
    屏幕上"紧急制动"还亮着，车却突然没那么能停了。

    解除只有两条路：回中位（惰行）、或者重新给牵引 —— 都是"司机接管了"的动作。
    """
    train = Train(fuxing, TrainState(v=30.0, throttle=1.0))
    train.emergency_stop()

    train.set_handle(-1.0)
    assert train.is_emergency, "补制动力把紧急制动弄丢了"
    assert brake_force(fuxing, train.state.brake, train.state.emergency) > \
        brake_force(fuxing, train.state.brake)

    train.set_handle(0.0)
    assert not train.is_emergency, "回中位应当解除紧急制动"

    train.emergency_stop()
    train.set_handle(0.4)
    assert not train.is_emergency and train.state.throttle == pytest.approx(0.4)


def test_single_handle_covers_throttle_notches_and_brake_notches(fuxing):
    """单手柄：正数是牵引、负数是制动、零位两边都松开。

    第三条是这里真正要守的：``set_throttle(0)`` **不会**动制动（那是对的，
    "松牵引"不该顺手撤掉制动力）。所以手柄回 0 位时必须显式走 ``release``，
    否则从制动位推回 0 会留下一个"没人要"的制动力 —— 而 UI 只看得见手柄，
    这个残留就变成了"手柄在 0 位车却还在刹车"的鬼故事。
    """
    train = Train(fuxing)

    train.set_handle(0.6)
    assert train.state.throttle == pytest.approx(0.6)
    assert train.state.brake == 0.0
    assert train.handle == pytest.approx(0.6)

    train.set_handle(-0.35)
    assert train.state.brake == pytest.approx(0.35)
    assert train.state.throttle == 0.0
    assert train.handle == pytest.approx(-0.35)

    train.set_handle(0.0)
    assert train.state.throttle == 0.0
    assert train.state.brake == 0.0, "手柄回 0 位之后还残留着制动力"
    assert train.handle == 0.0


def test_single_handle_is_clamped(fuxing):
    train = Train(fuxing)
    train.set_handle(9.0)
    assert train.handle == pytest.approx(1.0)
    train.set_handle(-9.0)
    assert train.handle == pytest.approx(-1.0)


def test_coasting_applies_resistance_only_not_a_brake(fuxing):
    """惰行（0 位）就是"既不牵引也不制动"，减速只来自阻力。"""
    train = Train(fuxing, TrainState(v=20.0))
    train.set_handle(0.0)
    coast = acceleration(fuxing, 20.0, train.state.throttle, train.state.brake)
    assert coast == pytest.approx(-resistance_force(fuxing, 20.0) / fuxing.total_mass)
    assert coast > acceleration(fuxing, 20.0, 0.0, 1.0)


def test_simulate_uses_a_fixed_step(fuxing):
    train = Train(fuxing, TrainState(throttle=1.0))
    train.simulate(10.0, dt=0.05)
    assert train.state.elapsed == pytest.approx(10.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# 换向：快速刹停 → 反向满牵引加速
# --------------------------------------------------------------------------- #

def test_reverse_brakes_with_emergency_force_then_flips(fuxing):
    """换向先是一段**紧急制动力级**的快速刹停，方向不会在刹停前翻个。"""
    train = Train(fuxing, TrainState(v=30.0))
    train.request_reverse()

    assert train.is_emergency
    assert train.state.brake == pytest.approx(1.0)
    assert train.state.throttle == 0.0
    assert train.direction == 1.0, "还没刹停，方向不该提前翻"

    v0 = train.state.v
    for _ in range(100):  # 5 秒，速度必须明显下降
        train.advance(0.05)
    assert train.state.v < v0

    # 一路刹停后：方向翻个、解除紧急制动、自动满牵引反向起步
    for _ in range(10000):
        train.advance(0.05)
        if train.direction < 0.0:
            break
    assert train.direction == -1.0
    assert not train.is_emergency
    assert train.state.brake == 0.0
    assert train.state.throttle == pytest.approx(1.0)


def test_reverse_moves_s_backwards(fuxing):
    """方向翻个之后，``s`` 沿反方向减小 —— 这才是"运行方向"真的切换了。"""
    train = Train(fuxing, TrainState(s=100.0))
    train.request_reverse()
    train.advance(0.0)          # v 已经是 0，立即完成翻向
    assert train.direction == -1.0
    s0 = train.state.s
    for _ in range(100):
        train.advance(0.05)
    assert train.state.s < s0


def test_reverse_acceleration_matches_forward(fuxing):
    """反向起步与正常牵引是**同一个**加速度（用户要的那条）。"""
    fwd = Train(fuxing, TrainState(throttle=1.0))
    rev = Train(fuxing)
    rev.request_reverse()
    rev.advance(0.0)            # 立即完成翻向（v=0）
    assert rev.direction == -1.0

    for _ in range(100):
        fwd.advance(0.05)
        rev.advance(0.05)

    assert rev.state.v == pytest.approx(fwd.state.v, rel=1e-6)
    assert rev.state.s == pytest.approx(-fwd.state.s, rel=1e-6)


def test_reverse_on_an_open_path_stops_at_the_consist_end(fuxing):
    """开链上倒行，车头该停在编组全长处（车尾贴住起点），而不是滑过 0 悬空。"""
    class OpenPath:
        total_length = 300.0
        closed = False

        def slope_at(self, s: float) -> float:
            return 0.0

    train = Train(fuxing, TrainState(s=300.0))
    train.request_reverse()
    path = OpenPath()
    for _ in range(10000):
        train.advance(0.05, path)
        if train.state.v == 0.0:
            break

    assert train.direction == -1.0
    assert train.state.s == pytest.approx(fuxing.total_length)
    assert train.state.v == pytest.approx(0.0)


def test_driver_taking_over_cancels_a_pending_reverse(fuxing):
    """换向刹停途中司机一接管手柄（↑↓/空格），自动换向就作废。"""
    train = Train(fuxing, TrainState(v=30.0))
    train.request_reverse()
    train.set_handle(0.5)       # 司机给牵引 = 接管
    train.advance(0.05)
    assert train.direction == 1.0, "司机接管后不该自动翻向"
    assert train.state.throttle == pytest.approx(0.5)
