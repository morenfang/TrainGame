"""2D 位姿与等曲率弧段几何。

这是整个项目的地基：轨道装配、闭环判定、列车跟随全部建立在它上面。
纯 Python（只用 math / dataclasses），**不依赖任何引擎**，因此可以 headless 单测。

坐标与角度约定（全项目统一，改动会波及所有模块）
--------------------------------------------------
* 右手系，**+Y 向上**，地面为 **XZ 平面**。
* 地面上的位置记为 (x, z)；``heading`` 是绕 +Y 轴的转角，**从 +X 轴转向 +Z 轴为正**。
* 因此 (x, z) 平面内的旋转矩阵为::

      R(h) = [[cos h, -sin h],
              [sin h,  cos h]]      作用于列向量 (x, z)

  校验：局部方向 (1, 0) 旋转 h 后得到 (cos h, sin h) —— 正是 heading 对应的
  「前进方向」。

为什么不直接手写 sin/cos 拼世界坐标
------------------------------------
见 ``route_geometry``：只要两个端口的位姿落在同一条等曲率圆弧上，就能**反解**
出 (弧长, 总转角)，且反解失败即说明该件几何自相矛盾。于是轨道件的目录数据
不需要把几何写两遍（写一遍端口、再写一遍弧长），**不可能出现两者不一致**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

TAU = 2.0 * math.pi

#：小于此角度时改用级数展开，避免 ``1 - cos`` 的灾难性抵消。
_TINY_ANGLE = 1e-4

#：单个 route 允许的最大总转角（超过 180° 的圆弧两端无法唯一反解）。
MAX_ROUTE_TURN = math.pi


def normalize_angle(a: float) -> float:
    """把角度折到闭区间 ``(-pi, pi]``。"""
    a = math.fmod(a + math.pi, TAU)
    if a <= 0.0:
        a += TAU
    return a - math.pi


def arc_end_pose(length: float, dtheta: float) -> Tuple[float, float]:
    """从局部原点、heading=0 出发的弧段终点。

    沿弧长 :math:`L`、总转角 :math:`\\Delta\\theta` 行进后的落点 ``(dx, dz)``。
    闭式解来自复平面上的积分 :math:`dp/ds = e^{i\\theta(s)}`；等价地设半径
    :math:`R = L / \\Delta\\theta`：:

        dx = R * sin(dtheta)          （实部）
        dz = R * (1 - cos(dtheta))    （虚部）

    ``dtheta`` 很小时退化为直线 (``dx -> length``, ``dz -> 0``)，此时走级数展开，
    精度不会因 ``1 - cos`` 的抵消而损失。
    """
    if length == 0.0:
        return 0.0, 0.0
    if abs(dtheta) < _TINY_ANGLE:
        d2 = dtheta * dtheta
        return length * (1.0 - d2 / 6.0), length * dtheta * 0.5 * (1.0 - d2 / 12.0)
    r = length / dtheta
    return r * math.sin(dtheta), r * (1.0 - math.cos(dtheta))


def arc_point(length: float, dtheta: float, u: float) -> Tuple[float, float, float]:
    """弧段上「距起点弧长 ``u``」处的 ``(dx, dz, 该处累计转角)``。

    等曲率弧段的转角与弧长成正比，所以 ``u`` 处已转过的角度是
    ``dtheta * u / length``。
    """
    if length == 0.0:
        return 0.0, 0.0, 0.0
    du = dtheta * (u / length)
    dx, dz = arc_end_pose(u, du)
    return dx, dz, du


@dataclass(frozen=True, slots=True)
class Pose:
    """刚体位姿：位置 + 绕 +Y 的航向。

    ``heading`` 保持为**未归一化**的原始值（累加即可），需要比较角度差时用
    :func:`normalize_angle`。
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    heading: float = 0.0

    # ---------------------------------------------------------------- 构造

    @staticmethod
    def origin() -> "Pose":
        return Pose(0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_degrees(x: float = 0.0, y: float = 0.0, z: float = 0.0,
                     heading_deg: float = 0.0) -> "Pose":
        """便捷构造：以**度**给出航向（用于读 JSON 目录）。"""
        return Pose(x, y, z, math.radians(heading_deg))

    # ---------------------------------------------------------------- 基本量

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def planar_position(self) -> Tuple[float, float]:
        """地面投影 ``(x, z)``，闭环误差与距离都在这个平面上算。"""
        return (self.x, self.z)

    def forward(self) -> Tuple[float, float, float]:
        """单位前进方向向量（水平分量），俯仰由 y 另行表达。"""
        return (math.cos(self.heading), 0.0, math.sin(self.heading))

    def heading_deg(self) -> float:
        return math.degrees(self.heading)

    @property
    def normalized_heading(self) -> float:
        """折到 ``(-pi, pi]`` 的航向。

        ``heading`` 本身**不做归一化**（这样累加式推导才有代数上的透明性），
        因此显示、比较、断言时用这个属性，或直接用 :func:`normalize_angle`。
        """
        return normalize_angle(self.heading)

    # ---------------------------------------------------------------- 变换

    def compose(self, local: "Pose") -> "Pose":
        """把 ``local`` 从本帧的局部坐标系变换到世界（``self ∘ local``）。

        旋转在 (x, z) 平面内进行，y 只做平移 —— 这也是本设计把「俯仰 / 坡度」
        简化成「y 沿弧长线性变化」的原因：它让平面几何保持精确的闭式解。
        """
        c = math.cos(self.heading)
        s = math.sin(self.heading)
        return Pose(
            self.x + c * local.x - s * local.z,
            self.y + local.y,
            self.z + s * local.x + c * local.z,
            self.heading + local.heading,
        )

    def inverse(self) -> "Pose":
        """逆变换：满足 ``self.compose(self.inverse())`` 为单位位姿。"""
        c = math.cos(self.heading)
        s = math.sin(self.heading)
        return Pose(
            -(c * self.x + s * self.z),
            -self.y,
            s * self.x - c * self.z,
            -self.heading,
        )

    def translated_local(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Pose":
        """在本帧局部坐标下平移。"""
        return self.compose(Pose(dx, dy, dz, 0.0))

    def flipped(self) -> "Pose":
        """位置不变、航向掉头 180°。用于「端口外指向 → 进入该件的行进方向」。"""
        return Pose(self.x, self.y, self.z, self.heading + math.pi)

    def with_heading(self, heading: float) -> "Pose":
        return Pose(self.x, self.y, self.z, heading)

    def with_position(self, x: float, y: float, z: float) -> "Pose":
        return Pose(x, y, z, self.heading)

    def rotate_about(self, pivot: "Pose", angle: float) -> "Pose":
        """绕 ``pivot`` 旋转 ``angle``（pivot 自身也参与旋转）。"""
        return pivot.compose(Pose.origin().with_heading(angle)).compose(
            pivot.inverse().compose(self)
        )

    def apply_rotation_only(self, angle: float) -> "Pose":
        """只旋转局部方向（位置当作向量一起转），用于由「路由几何」派生端口。"""
        c = math.cos(angle)
        s = math.sin(angle)
        return Pose(
            c * self.x - s * self.z,
            self.y,
            s * self.x + c * self.z,
            self.heading + angle,
        )

    # ---------------------------------------------------------------- 度量

    def planar_distance_to(self, other: "Pose") -> float:
        return math.hypot(other.x - self.x, other.z - self.z)

    def distance_to(self, other: "Pose") -> float:
        return math.sqrt(
            (other.x - self.x) ** 2 + (other.y - self.y) ** 2 + (other.z - self.z) ** 2
        )

    def heading_error_to(self, other: "Pose") -> float:
        """航向差（已归一化到 ``(-pi, pi]``）。"""
        return normalize_angle(other.heading - self.heading)

    def approx_equal(self, other: "Pose", pos_tol: float = 1e-9,
                     ang_tol: float = 1e-9) -> bool:
        return (
            self.distance_to(other) <= pos_tol
            and abs(self.heading_error_to(other)) <= ang_tol
        )

    def lerp_local(self, other: "Pose", t: float) -> "Pose":
        """局部线性插值（逐分量），仅用于预览与调试显示。"""
        return Pose(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
            self.heading + normalize_angle(other.heading - self.heading) * t,
        )


def heading_between(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """由两点（前 → 后）求水平航向。"""
    dx = b[0] - a[0]
    dz = b[2] - a[2]
    if abs(dx) < 1e-12 and abs(dz) < 1e-12:
        return 0.0
    return math.atan2(dz, dx)


def pitch_between(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """由两点（前 → 后）求俯仰角，向上为正。"""
    horizontal = math.hypot(b[0] - a[0], b[2] - a[2])
    dy = b[1] - a[1]
    if horizontal < 1e-12:
        return 0.0
    return math.atan2(dy, horizontal)


class InconsistentRouteError(ValueError):
    """两个端口的位姿不落在同一条等曲率圆弧上（件定义有误）。"""


def route_geometry(entry: Pose, exit_pose: Pose,
                   pos_tol: float = 1e-6) -> Tuple[float, float]:
    """由入口帧与出口帧**反解**弧段参数，返回 ``(弧长, 总转角)``。

    这是全项目最有用的一条规则：**任意一对端口，当且仅当它们的位姿落在同一条
    等曲率圆弧上时，才定义出一条合法的路由**。判据来自闭式解推导出的切线条件

        dz / dx == tan(dtheta / 2)

    （其退化情形 ``dtheta -> 180°`` 时改用 ``dz`` 反解半径）。

    好处：目录数据里只写端口位姿（或只写 ``length/dtheta``），另一份由本函数
    推导，**两者不可能不一致**；而一旦有人手改错了端口，这里会直接抛错。
    """
    local = entry.inverse().compose(exit_pose)
    dtheta = normalize_angle(local.heading)
    dx, dz = local.x, local.z

    if dx < 0.0:
        raise InconsistentRouteError(
            f"route must go forward: local exit is behind the entry (dx={dx:.6f})"
        )

    if abs(dtheta) < 1e-9:
        if abs(dz) > pos_tol:
            raise InconsistentRouteError(
                f"declared as straight but exit is off-axis (dz={dz:.9f})"
            )
        if dx <= 0.0:
            raise InconsistentRouteError("route length must be positive")
        return dx, 0.0

    if abs(dtheta) >= MAX_ROUTE_TURN:
        raise InconsistentRouteError(
            f"single route may not turn >= 180 deg (got {math.degrees(dtheta):.3f} deg)"
        )

    s = math.sin(dtheta)
    c = math.cos(dtheta)
    # 优先用 sin 分量反解（dtheta 接近 0 或 180 时它才不可靠）
    if abs(s) > 1e-6:
        radius = dx / s
    else:
        radius = dz / (1.0 - c)

    length = radius * dtheta
    if length <= 0.0:
        raise InconsistentRouteError(
            f"route length must be positive (got {length:.9f})"
        )

    ex, ez = arc_end_pose(length, dtheta)
    if abs(ex - dx) > pos_tol or abs(ez - dz) > pos_tol:
        raise InconsistentRouteError(
            "ports do not lie on a single circular arc: "
            f"expected end ({ex:.9f}, {ez:.9f}), got ({dx:.9f}, {dz:.9f}); "
            f"turn = {math.degrees(dtheta):.6f} deg"
        )
    return length, dtheta


class RouteUnreachableError(ValueError):
    """列车无法从给定入口走到目标出口（装配图/路径求解失败）。"""


@dataclass(frozen=True)
class ClosureReport:
    """闭环检测结果。

    ``closed`` 为真时 ``gap_distance`` / ``gap_heading`` 应当小到浮点噪声级别
    （实测 1e-14 m 量级）；为真以外的情形给出缺口大小，供 UI 画红色标记。
    """

    closed: bool
    visit_count: int
    total_length: float
    gap_distance: float
    gap_heading: float
    open_end: "Pose | None" = None
    reason: str = ""

    def describe(self) -> str:
        if self.closed:
            return f"闭环成立：{self.visit_count} 个路由段，总长 {self.total_length:.3f} m"
        return (
            f"未闭环（{self.reason}）：已走 {self.visit_count} 段 / "
            f"{self.total_length:.3f} m，缺口 {self.gap_distance:.6f} m，"
            f"航向差 {math.degrees(self.gap_heading):.6f}°"
        )
