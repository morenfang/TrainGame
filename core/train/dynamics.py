"""列车纵向动力学：沿轨道方向的一维模型。

对应概要设计 §5.7。只做**一个自由度**（沿轨道的位置与速度），因此既真实又
几乎不耗性能::

    m·dv/dt = F牵引(v, 手柄位) − F制动 − 阻力 − m·g·sinθ
    约束   : F牵引 <= adhesion · 黏着质量 · g      （黏着极限）

车型差异全部落在参数上（见 ``data/trains.json``），所以「换车」是换一行数据，
不是改代码。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.train.consist import G, TrainSpec

#: 计算恒功率段牵引力时的速度下限，避免 v→0 时除零。
_MIN_SPEED = 0.5


# --------------------------------------------------------------------------- #
# 受力
# --------------------------------------------------------------------------- #

def tractive_force(spec: TrainSpec, v: float, throttle: float) -> float:
    """当前牵引力（N）。

    牵引特性分两段：

    * **恒牵引力段**（低速）：受最大牵引力约束；
    * **恒功率段**（高速）：``F = P / v``。

    两者取小，再乘手柄位，最后受黏着极限约束。
    """
    throttle = max(0.0, min(1.0, throttle))
    if throttle <= 0.0 or spec.power_w <= 0.0:
        return 0.0
    power_limited = spec.power_w / max(abs(v), _MIN_SPEED)
    force = min(spec.max_tractive_force_n, power_limited)
    force *= throttle
    return min(force, spec.max_traction_from_adhesion)


def resistance_force(spec: TrainSpec, v: float, grade: float = 0.0) -> float:
    """阻力（N）：戴维斯公式 + 坡道分量。

    ``grade`` 是坡度（``dy / 平面弧长``，即正切值）。内部换算成正弦，因此陡坡
    也正确。
    """
    a, b, c = spec.davis
    speed = abs(v)
    rolling = spec.total_mass * (a + b * speed + c * speed * speed)
    sin_theta = grade / math.sqrt(1.0 + grade * grade) if grade else 0.0
    slope = spec.total_mass * G * sin_theta
    return rolling + slope


def brake_force(spec: TrainSpec, brake: float, emergency: bool = False) -> float:
    """制动力（N），恒定制动力模型（简化，不做到黏着-速度曲线）。

    ``emergency=True`` 时用**紧急制动**的峰值（``spec.max_emergency_brake_force_n``，
    数据里没给就等同常用制动）。这是"拉死"与"正常情况下把制动位推到底"的区别：
    前者是保命的，制动力明显更大（见 `data/trains.json` 的 notes）。
    """
    brake = max(0.0, min(1.0, brake))
    peak = (spec.effective_emergency_brake_force if emergency
            else spec.max_brake_force_n)
    return brake * peak


def acceleration(spec: TrainSpec, v: float, throttle: float = 0.0,
                 brake: float = 0.0, grade: float = 0.0,
                 emergency: bool = False, boost: float = 1.0) -> float:
    """当前纵向加速度（m/s²）。正的表示在加速。

    ``boost`` 是**沙盘手感倍率**：把牵引力与制动力一起放大（1.0 = 物理标定值）。
    它刻意**不**进 `data/trains.json` —— 那套参数是照真实车型标定的，有测试守着
    （见 ``tests/test_dynamics.py`` 的"校准断言"）；而"沙盘上开起来够不够跟手"
    是另一件事，由 `app/editor.py` 的 ``DRIVE_BOOST`` 决定。阻力与坡度不参与放大，
    否则上坡限速那些结论会被一起改掉。
    """
    net = (
        tractive_force(spec, v, throttle) * boost
        - brake_force(spec, brake, emergency) * boost
        - resistance_force(spec, v, grade)
    )
    return net / spec.total_mass


# --------------------------------------------------------------------------- #
# 状态与积分
# --------------------------------------------------------------------------- #

@dataclass
class TrainState:
    """列车状态：车头所在弧长 + 速度 + 手柄 / 制动。"""

    #: **车头**（首车前端转向架中心）所在的弧长
    s: float = 0.0
    #: 速度（m/s）。``v`` 恒非负 —— 它是**速率**，行进方向由 :attr:`direction`
    #: 单独表达；倒行只是 ``direction`` 翻个，速率依旧走同一条牵引/制动曲线。
    v: float = 0.0
    #: 牵引手柄位 0..1
    throttle: float = 0.0
    #: 制动位 0..1
    brake: float = 0.0
    #: 是否处于**紧急制动**（手柄拉到"紧急"那一档）。它是一枚**锁存**：拉了之后
    #: 只有回到惰行（:meth:`Train.release`）或重新给牵引才会解除 —— 否则用户按住
    #: ↓ 补一点制动力，就会把"拉死"悄悄降级成常用制动。
    emergency: bool = False
    #: 已运行时间（s）
    elapsed: float = 0.0
    #: 累计走行距离（m）
    distance: float = 0.0
    #: 行进方向：``+1`` 沿路径 ``+s`` 前进，``-1`` 沿 ``-s`` 倒行。
    direction: float = 1.0

    @property
    def speed_kmh(self) -> float:
        return self.v * 3.6

    @property
    def is_stopped(self) -> bool:
        return self.v <= 1e-9

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"TrainState(s={self.s:.2f}m, v={self.v:.3f}m/s "
            f"({self.speed_kmh:.1f}km/h), throttle={self.throttle:.2f}, "
            f"brake={self.brake:.2f})"
        )


def step(state: TrainState, spec: TrainSpec, dt: float,
         total_length: float | None = None, closed: bool = False,
         grade: float = 0.0, boost: float = 1.0,
         direction: float = 1.0, min_s: float = 0.0) -> TrainState:
    """推进 ``dt`` 秒（就地修改并返回 ``state``）。

    * 速度 ``v`` 恒非负（速率）；``direction`` 决定弧长往哪边走，倒行时 ``s`` 减小。
      阻力和制动只能把车停住，不会把速率推成负的。
    * 速度不超过 ``spec.max_speed_ms``。
    * ``closed=True`` 时 ``s`` 对 ``total_length`` 取模；否则被夹在
      ``[min_s, total_length]`` 内，到端点自动停车（开到尽头了）。
    * ``state.emergency`` 为真时按紧急制动的峰值算制动力；``boost`` 见
      :func:`acceleration`。
    * ``grade`` 是**沿 ``+s`` 方向**定义的坡度；倒行时内部乘 ``direction`` 翻个，
      于是"倒着爬坡"和"正着下坡"的受力是一致的。
    """
    if dt <= 0.0:
        return state

    acc = acceleration(spec, state.v, state.throttle, state.brake,
                       grade * direction, emergency=state.emergency, boost=boost)
    v_next = state.v + acc * dt

    # 停住之后不会被阻力和制动推着倒退
    if v_next < 0.0:
        v_next = 0.0
    limit = spec.max_speed_ms
    if v_next > limit:
        v_next = limit

    # 速率积分（恒非负），沿路径的位移再乘方向
    rate = 0.5 * (state.v + v_next) * dt
    ds = direction * rate

    state.v = v_next
    state.elapsed += dt
    state.distance += rate

    if total_length is not None and total_length > 0.0:
        if closed:
            state.s = (state.s + ds) % total_length
        else:
            new_s = state.s + ds
            if new_s >= total_length:
                state.s = total_length
                state.v = 0.0  # 到尽头就停
            elif new_s <= min_s:
                state.s = min_s
                state.v = 0.0
            else:
                state.s = new_s
    else:
        state.s += ds

    return state


# --------------------------------------------------------------------------- #
# 面向使用的封装
# --------------------------------------------------------------------------- #

@dataclass
class Train:
    """一列在轨道上运行的列车。"""

    spec: TrainSpec
    state: TrainState = None  # type: ignore[assignment]
    #: **沙盘手感倍率**：牵引力与制动力一起放大（1.0 = `data/trains.json` 的标定值）。
    #: 见 :func:`acceleration` 的说明 —— 数据是照真实车型标定的，游戏里跟不跟手是
    #: 另一个旋钮，`app/editor.py` 的 ``DRIVE_BOOST`` 在这里落地。
    drive_boost: float = 1.0

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = TrainState()
        #: 是否在"换向"过程中（先快速刹停，再反向满牵引加速）。
        self._reversing = False

    # ---------------------------------------------------------------- 控制

    @property
    def direction(self) -> float:
        """当前行进方向：``+1`` 前进、``-1`` 倒行。"""
        return self.state.direction

    def request_reverse(self) -> None:
        """请求换向：先**快速刹停**（紧急制动力级），再反向满牵引加速。

        换向是一段**过程**而不是一次跳变 —— 正是"车辆不能突然变向"那类抱怨的
        反面：先把手柄拉到紧急制动、速率一路降到 0，然后方向翻个、自动给满牵引，
        用与正常牵引**一样**的加速度反向起步。过程中用户随时可用 ↑↓ / 空格接管
        （接管即取消换向）。
        """
        self._reversing = True
        self.state.throttle = 0.0
        self.state.brake = 1.0
        self.state.emergency = True

    def set_throttle(self, throttle: float) -> None:
        self.state.throttle = max(0.0, min(1.0, throttle))
        if self.state.throttle > 0.0:
            self.state.brake = 0.0
            # 重新给牵引 = 司机接管了，紧急制动解除。**只有这一条路（以及 release）
            # 能解除它** —— 见 TrainState.emergency 上为什么必须是锁存。
            self.state.emergency = False

    def set_brake(self, brake: float) -> None:
        self.state.brake = max(0.0, min(1.0, brake))
        if self.state.brake > 0.0:
            self.state.throttle = 0.0

    def set_handle(self, handle: float) -> None:
        """**单手柄**：``> 0`` 牵引、``< 0`` 制动、``0`` 惰行。

        真实的司机手柄就是一个把手在一条槽里推拉：往前是牵引位、往后是制动位、
        中间是惰行。这里照搬这个手感，于是 UI 只需要维护**一个**数，
        而"牵引与制动不可能同时给"这条不变量由 :meth:`set_throttle` /
        :meth:`set_brake` 各自保证（它们会互相清零），不必在 UI 里再写一遍。

        回到 0 位要**同时**松开牵引与制动 —— ``set_throttle(0)`` 只把牵引清零，
        它不会去动制动（否则"松牵引"会顺手把制动力也撤掉，那是个危险的意外）。
        """
        handle = max(-1.0, min(1.0, handle))
        # 司机手动接管手柄 = 取消进行中的换向（换向会自动给满牵引，别跟手动抢）
        self._reversing = False
        if handle > 0.0:
            self.set_throttle(handle)
        elif handle < 0.0:
            self.set_brake(-handle)
        else:
            self.release()

    @property
    def handle(self) -> float:
        """当前手柄位：牵引为正、制动为负、惰行为 0。"""
        if self.state.throttle > 0.0:
            return self.state.throttle
        if self.state.brake > 0.0:
            return -self.state.brake
        return 0.0

    @property
    def is_emergency(self) -> bool:
        """手柄是否压在"紧急"那一档。"""
        return self.state.emergency

    def emergency_stop(self) -> None:
        """紧急制动：牵引清零、制动位拉满，并切到**紧急制动力**。"""
        self.state.throttle = 0.0
        self.state.brake = 1.0
        self.state.emergency = True

    def release(self) -> None:
        self.state.throttle = 0.0
        self.state.brake = 0.0
        self.state.emergency = False

    # ---------------------------------------------------------------- 运行

    def advance(self, dt: float, path=None) -> TrainState:
        """沿 ``path`` 推进 ``dt`` 秒。"""
        total = None
        closed = False
        grade = 0.0
        min_s = 0.0
        if path is not None:
            total = path.total_length
            closed = path.closed
            grade = path.slope_at(self.state.s % total if total else 0.0)
            if not closed:
                # 开链上，车头弧长不能小于编组全长 —— 否则车尾会悬出轨道起点。
                # 倒行时同样适用：车尾照样落在 ``head_s - 编组全长`` 那一端。
                min_s = self.spec.total_length

        if self._reversing:
            if self.state.v <= 1e-6:
                # 刹停了：翻方向、解除紧急制动、自动满牵引反向起步。
                self._reversing = False
                self.state.direction = -self.state.direction
                self.state.emergency = False
                self.state.brake = 0.0
                self.state.throttle = 1.0

        return step(self.state, self.spec, dt, total_length=total,
                    closed=closed, grade=grade, boost=self.drive_boost,
                    direction=self.state.direction, min_s=min_s)

    def simulate(self, seconds: float, dt: float = 0.1, path=None) -> TrainState:
        """跑 ``seconds`` 秒（固定步长），返回末态。测试与离线调试用。"""
        steps = int(round(seconds / dt))
        for _ in range(steps):
            self.advance(dt, path)
        return self.state

    # ---------------------------------------------------------------- 摆放

    def place(self, path):
        """把编组摆到路径上，返回每节车的状态（转向架模型）。"""
        from core.track.path import place_consist

        return place_consist(
            path,
            head_s=self.state.s,
            lengths=self.spec.lengths,
            bogie_half_spacing=self.spec.bogie_half_spacings,
            coupling_gap=self.spec.coupling_gap,
        )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Train({self.spec.id!r}, {self.state!r})"


# --------------------------------------------------------------------------- #
# 离线测算工具（用于把参数调得合理）
# --------------------------------------------------------------------------- #

def time_to_reach(spec: TrainSpec, target_kmh: float, dt: float = 0.05,
                  grade: float = 0.0) -> float | None:
    """全手柄加速到 ``target_kmh`` 所需时间（s）；达不到则返回 ``None``。

    这是校准 ``data/trains.json`` 参数的主要工具 —— 拿它算出来的加速时间应当
    与真实车型大致同量级（见 tests/test_dynamics.py）。
    """
    target = target_kmh / 3.6
    state = TrainState(throttle=1.0)
    elapsed = 0.0
    while elapsed < 3600.0:
        step(state, spec, dt, grade=grade)
        elapsed += dt
        if state.v >= target:
            return elapsed
        if state.v < 1e-6 and elapsed > 1.0:
            return None  # 起不来（坡太陡或黏着不足）
    return None


def steady_state_speed(spec: TrainSpec, grade: float = 0.0) -> float:
    """全手柄下的平衡速度（m/s）。达不到（例如爬不上去）时返回 0。"""
    v = 0.0
    for _ in range(20000):
        acc = acceleration(spec, v, throttle=1.0, grade=grade)
        if acc <= 0.0:
            return v
        v += acc * 0.05
        if v > spec.max_speed_ms:
            return spec.max_speed_ms
    return v
