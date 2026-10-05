"""路径求解：由装配图求出列车能走的闭环，并做弧长参数化。

对应概要设计 §5.4（闭环检测）与 §6.2（列车运行）。

走线规则只有一条
------------------------------------------------
从某端口进入某件，沿该件的 route 走到出口，再跨过连接走到下一件，如此递归。
「能不能走通」完全由 ``PieceDef.traversals`` 决定，这里不含任何针对道岔 / 交叉 /
缓冲端的特例判断 —— 那三种件的行为差异全部落在目录数据的 ``routes`` 里。

两条重要结论
------------------------------------------------
1. **回到同一个端口 ⇒ 几何上精确闭环。** 因为端口位姿由吸附链精确推导，回到
   同一个端口意味着累计变换恰好是单位变换，不需要再判"误差够不够小"。
   注意这只是**单向**的：几何上闭环**不**意味着能回到同一个端口 —— 见下一条。
2. **首尾严丝合缝但没 ``connect()`` 的链，必须当成环来开。** 反过来的那一半：
   链自己围成了一个环，可最后那道缝没写进连接表（``connect()`` 只由按 C /
   自动补齐建立）。这时走线到末端就"没接东西"了，可缝两端在几何上**严丝合缝**
   —— 实测误差 1e-13 m 量级，与真闭环同一量级。要是只认端口，列车会停在一道
   **看不见的缝**上：缺口显示 0.0000 m、轨道看上去是连续的，用户眼里就是
   "轨道明明接上了，车却卡住不动"。   所以 :func:`trace` 在末端无 peer 时补一次
   几何判据（位置 + 航向，见 ``CLOSURE_POSITION_TOL``）。
   判据仍卡在 1e-6 m：真闭环的误差在 1e-14 m 量级，而"少一节"的开链缺口是
   **米级**（实测：45° 的圆环少一节差 **30.6 m**，22.5° 的圆环接一半差
   **80 m** = 直径），中间隔着七八个数量级，所以不会把没铺完的线误判成环。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from core.geometry import (
    ClosureReport,
    Pose,
    arc_point,
    heading_between,
    pitch_between,
    route_geometry,
)
from core.track.catalog import PieceDef
from core.track.layout import Layout, PortKey

#: 判定闭环的位置 / 航向阈值。见模块文档：真实闭环远小于它，未闭环远大于它。
CLOSURE_POSITION_TOL = 1e-6
CLOSURE_HEADING_TOL = 1e-9

#: ``trace`` 的兜底步数上限。
#:
#: 真正让走线停下来的是**访问集去重**（同一个 ``(件, 入口端口)`` 走第二次就是绕圈，
#: 见 :func:`trace`），步数天然被状态数卡住。这个上限只是"万一将来有人改坏了那个
#: 不变量"的兜底 —— 走到它说明**代码**有问题，不是布局有问题。
MAX_SEGMENTS = 100_000


class PathError(ValueError):
    """无法求出可走路径。"""


@dataclass(frozen=True, slots=True)
class PathSegment:
    """路径上的一段：一列车通过某一节轨道件的一条 route。"""

    piece_index: int
    route_index: int
    reversed: bool
    entry_port: str
    exit_port: str
    #: 世界坐标下的行进帧（起点，朝向列车前进方向）
    entry_frame: Pose
    length: float
    dtheta: float
    #: 每单位平面弧长的高程变化（由两端世界高度差推出，反走自动取负）
    grade: float
    #: 该段起点在整条路径弧长上的偏移
    s0: float
    #: 这一节是否是道岔。反向走线时顺向/逆向会翻个，需要它来判断反向路径里
    #: 的「顺向（从岔尖进）」是否还满足档位约束（见 render.train_view 的重锚定）。
    is_switch: bool = False

    @property
    def s1(self) -> float:
        return self.s0 + self.length

    @property
    def exit_frame(self) -> Pose:
        dx, dz, dtheta = arc_point(self.length, self.dtheta, self.length)
        return self.entry_frame.compose(
            Pose(dx, self.length * self.grade, dz, dtheta)
        )

    def pose_at(self, u: float) -> Pose:
        """段内 ``u``（0..length）处的世界位姿。"""
        dx, dz, du = arc_point(self.length, self.dtheta, u)
        return self.entry_frame.compose(Pose(dx, u * self.grade, dz, du))

    def point_at(self, u: float) -> tuple[float, float, float]:
        return self.pose_at(u).position

    def contains(self, s: float) -> bool:
        return self.s0 <= s <= self.s1

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        direction = "rev" if self.reversed else "fwd"
        return (
            f"PathSegment(piece={self.piece_index}, route={self.route_index}, "
            f"{self.entry_port}->{self.exit_port} {direction}, "
            f"L={self.length:.3f}, dθ={math.degrees(self.dtheta):.2f}°)"
        )


@dataclass
class RoutePath:
    """一条可行驶的路径（闭环或开放）。"""

    segments: tuple[PathSegment, ...]
    closed: bool
    start: PortKey | None = None
    #: 路径起点处的世界位姿（= 第一段的 entry_frame）
    origin: Pose = field(default_factory=Pose.origin)
    #: 终止原因：``closed`` / ``open_end``（末端未连接）/ ``dead_end``（走进死端）
    termination: str = "open_end"

    # ---------------------------------------------------------------- 基本量

    @property
    def total_length(self) -> float:
        if not self.segments:
            return 0.0
        last = self.segments[-1]
        return last.s0 + last.length

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self) -> Iterator[PathSegment]:
        return iter(self.segments)

    # ---------------------------------------------------------------- 采样

    def segment_at(self, s: float) -> tuple[PathSegment, float]:
        """定位弧长 ``s`` 落在哪一段，返回 ``(段, 段内偏移 u)``。"""
        if not self.segments:
            raise PathError("path is empty")
        total = self.total_length
        if self.closed:
            if total <= 0.0:
                raise PathError("closed path has zero length")
            s = s % total
        elif s < 0.0 or s > total:
            raise PathError(f"arc length {s} is outside [0, {total}] of an open path")

        # 先线性扫描再二分其实没差别（段数很少），但保持实现对段数不敏感。
        lo, hi = 0, len(self.segments) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.segments[mid].s0 <= s:
                lo = mid
            else:
                hi = mid - 1
        segment = self.segments[lo]
        return segment, s - segment.s0

    def pose_at(self, s: float) -> Pose:
        """弧长 ``s`` 处的**轨道中心线**世界位姿（列车转向架挂在这个位姿上）。"""
        segment, u = self.segment_at(s)
        return segment.pose_at(u)

    def point_at(self, s: float) -> tuple[float, float, float]:
        return self.pose_at(s).position

    def direction_at(self, s: float) -> float:
        """弧长 ``s`` 处的水平航向。"""
        return self.pose_at(s).heading

    def slope_at(self, s: float) -> float:
        """弧长 ``s`` 处的坡度（dy / 平面弧长）。"""
        segment, _ = self.segment_at(s)
        return segment.grade

    def sample(self, step: float = 5.0) -> list[Pose]:
        """按固定弧长间隔采样，供渲染中心线 / 调试绘制。"""
        total = self.total_length
        if total <= 0.0:
            return []
        count = max(2, int(total / step) + 1)
        return [self.pose_at(total * i / (count - 1)) for i in range(count)]

    # ---------------------------------------------------------------- 反向

    def reversed(self) -> "RoutePath":
        """**同一条路、反着走**的路径：每一段都掉头，段的先后也倒过来。

        为什么需要它：布局一变（扳道岔、接/删轨、读档），闭环检测可能交出一条
        行进方向**相反**的路径 —— 而 ``state.s`` 只是个弧长数，直接留着就会让
        列车瞬移到别处、甚至倒着开。有了这条反向路径，调用方就能挑出与列车当前
        朝向一致的那一条，让它接着往前开（见
        :mod:`render.train_view` 的重锚定）。

        几何上是**同一批弧**：每一段的入口帧取原段的出口帧掉头（列车从终点倒退
        进来），出口帧取原段入口帧掉头，弧长与转角由
        :func:`core.geometry.route_geometry` 反解 —— 于是"反向路径长什么样"这件事
        仍然只有一处几何实现，不会与正向路径各写一份。
        """
        if not self.segments:
            return RoutePath(segments=(), closed=self.closed,
                             start=None, origin=Pose.origin(),
                             termination=self.termination)

        segments: list[PathSegment] = []
        cum = 0.0
        for segment in reversed(self.segments):
            entry = segment.exit_frame.flipped()
            exit_pose = segment.entry_frame.flipped()
            length, dtheta = route_geometry(entry, exit_pose)
            grade = (exit_pose.y - entry.y) / length if length else 0.0
            segments.append(PathSegment(
                piece_index=segment.piece_index,
                route_index=segment.route_index,
                reversed=not segment.reversed,
                entry_port=segment.exit_port,
                exit_port=segment.entry_port,
                entry_frame=entry,
                length=length,
                dtheta=dtheta,
                grade=grade,
                s0=cum,
                is_switch=segment.is_switch,
            ))
            cum += length

        return RoutePath(
            segments=tuple(segments),
            closed=self.closed,
            start=None,
            origin=segments[0].entry_frame,
            termination=self.termination,
        )


# --------------------------------------------------------------------------- #
# 走线
# --------------------------------------------------------------------------- #

def trace(layout: Layout, start: PortKey,
          switch_states: dict[int, int] | None = None) -> RoutePath:
    """从 ``start`` 端口进入其所属件开始走，**一定停下来**。

    三种收场（``RoutePath.termination``）：

    * ``"closed"`` —— 回到起点，**或者**首尾在几何上严丝合缝（含没 ``connect()``
      的"看不见的缝"，见模块文档结论 2）。两者都按环交给列车。
    * ``"open_end"`` / ``"dead_end"`` —— 走到没接东西的端口，或没路由的端口。
    * ``"entered_loop"`` —— 一头扎进了一个**不包含起点**的环。

    第三种是为什么这个函数必须记「走过的状态」：图是「一棵树 + 几道合拢缝」，
    所以完全可能"环 + 挂在环上的支线"（典型例子：道岔直股成环，侧股那一段空着）。
    从支线末端出发，走进环之后会**绕着环一圈圈转，永远回不到起点** —— 光靠
    "回到起点"这个判据是判不出来的，于是老版本要转到 ``MAX_SEGMENTS`` 才抛异常，
    而那个异常会从 ``LayoutView.sync()`` 里冒出来，把整个游戏带崩（按一下 C 就崩）。

    所以这里记 ``(件, 入口端口)`` 的访问集：图的走线是确定性的，同一个状态走两次
    就等于进了周期，此后永不再有新信息 —— 认清它、停下，比跑到十万段再抛异常
    既快得多（O(件数)）也诚实得多。
    """
    start_piece, start_port = start
    if start_piece not in layout.pieces:
        raise PathError(f"piece {start_piece} does not exist")

    segments: list[PathSegment] = []
    cum = 0.0
    piece_index, entry_port = start_piece, start_port
    origin: Pose | None = None
    termination = "open_end"
    visited: set[tuple[int, str]] = set()

    for _ in range(MAX_SEGMENTS):          # 兜底；真正终止走线的是下面的 visited
        state = (piece_index, entry_port)
        if state in visited:
            termination = "entered_loop"
            break
        visited.add(state)

        piece_def = layout.definition(piece_index)
        switch_index = (
            switch_states.get(piece_index)
            if switch_states is not None
            else layout.switch_of(piece_index)
        )
        exits = piece_def.traversals(entry_port, switch_index)
        if not exits:
            termination = "dead_end"  # 缓冲端，或该端口没有 route
            break

        exit_port, route_index, reversed_ = exits[0]
        segment = _make_segment(
            layout, piece_index, route_index, entry_port, exit_port, reversed_, cum
        )
        if origin is None:
            origin = segment.entry_frame
        segments.append(segment)
        cum += segment.length

        peer = layout.peer_of(piece_index, exit_port)
        if peer is None:
            # 末端没接东西。这里有两种情形，必须分清：
            #
            # * **真的开链末端** —— 走线到此为止（termination="open_end"）；
            # * **末端与起点严丝合缝** —— 链自己围成了一个环，只是最后那道缝
            #   还没写进连接表（见模块文档结论 2）。这时必须把路径当**环**交给
            #   列车：缝既然严丝合缝，跨过去在几何上就是连续的，否则列车会停在
            #   一道看不见的缝上（HUD 显示"缺口 0.0000 m"），也就是用户报的
            #   "轨道明明接上了、车却卡在这一节和下一节之间"。
            #
            #   比位置还不够，还要比航向：`approx_equal` 同时卡位置与航向，于是
            #   "出去又原路折回来"的链（航向差 π）不会被误判成环 —— 那种链列车
            #   得掉头才走得通，本来就跑不了一圈。
            if origin is not None and segment.exit_frame.approx_equal(
                    origin, CLOSURE_POSITION_TOL, CLOSURE_HEADING_TOL):
                return RoutePath(
                    segments=tuple(segments),
                    closed=True,
                    start=start,
                    origin=origin,
                    termination="closed",
                )
            termination = "open_end"  # 开链的末端
            break

        next_piece, next_port = peer
        if (next_piece, next_port) == start:
            # 回到出发点：图上的环 = 几何上的精确闭环
            return RoutePath(
                segments=tuple(segments),
                closed=True,
                start=start,
                origin=origin or Pose.origin(),
                termination="closed",
            )
        piece_index, entry_port = next_piece, next_port
    else:
        raise PathError(
            f"trace did not terminate within {MAX_SEGMENTS} segments; the "
            "visited-state check in trace() must have been broken (this is an "
            "internal invariant violation, not a property of the layout)"
        )

    return RoutePath(
        segments=tuple(segments),
        closed=False,
        start=start,
        origin=origin or Pose.origin(),
        termination=termination,
    )


def _make_segment(layout: Layout, piece_index: int, route_index: int,
                  entry_port: str, exit_port: str, reversed_: bool,
                  s0: float) -> PathSegment:
    """由装配图求出某一段的世界几何。"""
    entry_frame = layout.world_inbound(piece_index, entry_port)
    exit_frame = layout.world_port(piece_index, exit_port)
    length, dtheta = route_geometry(entry_frame, exit_frame)
    grade = (exit_frame.y - entry_frame.y) / length if length else 0.0
    return PathSegment(
        piece_index=piece_index,
        route_index=route_index,
        reversed=reversed_,
        entry_port=entry_port,
        exit_port=exit_port,
        entry_frame=entry_frame,
        length=length,
        dtheta=dtheta,
        grade=grade,
        s0=s0,
        is_switch=layout.definition(piece_index).is_switch,
    )


# --------------------------------------------------------------------------- #
# 闭环检测
# --------------------------------------------------------------------------- #

def detect_closure(layout: Layout, start: PortKey | None = None,
                   switch_states: dict[int, int] | None = None,
                   allow_open: bool = False
                   ) -> tuple[ClosureReport, RoutePath | None]:
    """检测闭环。返回 ``(报告, 路径或 None)``。

    搜索策略：优先用调用者给的 ``start``；否则贪心地从各个空闲端口试起
    （空闲端口必然是开链的末端，是闭环的天然起止点）。

    ``allow_open`` 决定"没闭合成环"时那条**最长的可行进路径**要不要交出来：

    * ``False``（默认）—— 返回 ``None``。这是"闭环检测"的语义：问的是
      "能不能绕一圈"，答案是"不能"的时候，路径是干扰项。
    * ``True`` —— 把走过最多件的那条开链路径返回。给的是**能开车**的语义
      （"这段路最长能跑到哪儿"）。编辑器用这个：线还没铺完就召唤一列车来跑跑
      看，是很正常的需求，而 :class:`render.train_view.TrainView` 本来就支持
      开链限位（见它的 ``head_range``）。

    两种语义都保留，是因为它们真的是两个问题 —— 让一个函数同时回答两个问题、
    再靠调用者猜，正是这类接口容易出错的地方。
    """
    if layout.is_empty:
        return ClosureReport(False, 0, 0.0, 0.0, 0.0, reason="布局为空"), None

    candidates = [start] if start is not None else _closure_candidates(layout)
    results: list[tuple[ClosureReport, RoutePath]] = []
    for candidate in candidates:
        path = trace(layout, candidate, switch_states)
        results.append((_report_for(layout, path, candidate), path))

    if allow_open:
        # 给「能开车」的语义：优先**正向**（第一段不是 reverse）的走线 ——
        # 可挤岔道岔让同一圈环在反向上也永远走得通，若不偏向正向，会把
        # 列车甩到反向主环上，从而无视道岔、直着开过去。次优才是闭环、走件数。
        def _prefer_forward(item: tuple[ClosureReport, RoutePath]) -> tuple:
            report, path = item
            forward = 1 if (path.segments and not path.segments[0].reversed) else 0
            return (forward, 1 if report.closed else 0, report.visit_count)

        best_report, best_path = max(results, key=_prefer_forward)
        return best_report, best_path

    # 非开链语义问的是「能不能绕一圈」：任一方向的闭环都算。
    for report, path in results:
        if report.closed:
            return report, path
    best_report, _ = max(results, key=lambda item: item[0].visit_count)
    return best_report, None


def _closure_candidates(layout: Layout) -> list[PortKey]:
    """候选起点：空闲端口 + 合拢缝的端点，都不行才退到任意端口。

    空闲端口优先（开链末端是环最自然的起止点），但**光有空闲端口不够**：
    图是「一棵树 + 几道合拢缝」，而"环 + 挂在环上的支线"这种布局里，环上的端口
    可能全都接满了、一个空闲端口都不剩 —— 空闲端口全在支线上，从支线出发只会
    一头扎进环（``termination="entered_loop"``），永远判不出闭环。

    补上**缝的端点**就够了，因为**任何环都必须经过至少一道缝**（缝之外全是树边），
    所以环上一定有缝端点在。这也让候选集保持在 O(缝数) 而不是 O(端口数)。
    """
    free = layout.free_ports()
    seams = layout.closing_seam_ports()
    if free or seams:
        return free + seams
    return list(layout.all_ports())


def _report_for(layout: Layout, path: RoutePath, start: PortKey) -> ClosureReport:
    if path.closed:
        # 回到起点 -> 精确闭环。实测误差在 1e-14 m 量级。
        end_pose = path.segments[-1].exit_frame
        start_pose = layout.world_inbound(*start)
        return ClosureReport(
            closed=True,
            visit_count=len(path),
            total_length=path.total_length,
            gap_distance=end_pose.planar_distance_to(start_pose),
            gap_heading=end_pose.heading_error_to(start_pose),
            open_end=end_pose,
        )

    if not path.segments:
        return ClosureReport(
            False, 0, 0.0, 0.0, 0.0,
            open_end=layout.world_inbound(*start),
            reason="起点端口没有可走的路由",
        )

    last = path.segments[-1]
    end_pose = last.exit_frame
    reason = {
        "dead_end": "走进死端",
        "open_end": "末端端口没有接上",
        "entered_loop": "这头走进了另一处已经接通的环",
    }.get(path.termination, path.termination)
    return ClosureReport(
        closed=False,
        visit_count=len(path),
        total_length=path.total_length,
        gap_distance=end_pose.planar_distance_to(path.origin),
        gap_heading=end_pose.heading_error_to(path.origin),
        open_end=end_pose,
        reason=reason,
    )


def find_loop(layout: Layout, start: PortKey | None = None,
              switch_states: dict[int, int] | None = None) -> RoutePath:
    """求出一条闭环路径；求不出则抛 :class:`PathError`。

    道岔档位会影响结果，因此 ``switch_states`` 是"布局的一部分"：
    换一档道岔可能需要重新求解路径。
    """
    report, path = detect_closure(layout, start, switch_states)
    if path is None or not report.closed:
        raise PathError(f"layout has no closed loop: {report.describe()}")
    return path


# --------------------------------------------------------------------------- #
# 列车编组在路径上的位置（转向架模型，概要设计 §5.6）
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class BogieState:
    """一个转向架的状态。"""

    #: 转向架中心的弧长
    s: float
    position: tuple[float, float, float]
    heading: float
    pitch: float

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"BogieState(s={self.s:.3f}, "
            f"pos=({self.position[0]:.3f},{self.position[1]:.3f},{self.position[2]:.3f}), "
            f"h={math.degrees(self.heading):.2f}°, "
            f"p={math.degrees(self.pitch):.2f}°)"
        )


@dataclass(frozen=True, slots=True)
class CarState:
    """一节车厢的状态（由两个转向架推出车体位姿）。"""

    car_index: int
    front: BogieState
    rear: BogieState
    #: 车体中心（两转向架中点）
    center: tuple[float, float, float]
    heading: float
    pitch: float

    @property
    def body_pose(self) -> Pose:
        return Pose(self.center[0], self.center[1], self.center[2], self.heading)


def bogie_at(path: RoutePath, s: float) -> BogieState:
    """求弧长 ``s`` 处转向架的状态（挂在中心线上，故天然在轨）。"""
    pose = path.pose_at(s)
    return BogieState(
        s=s,
        position=pose.position,
        heading=pose.heading,
        pitch=math.atan(path.slope_at(s)),
    )


def car_from_bogies(car_index: int, front: BogieState, rear: BogieState) -> CarState:
    """由两个转向架推出车体：中心取中点，航向与俯仰由两转向架连线决定。

    这样过弯时车体会自然「切内弦」（真实列车就是这样），
    且**不需要任何物理引擎** —— 纯运动学，绝对稳定、与帧率无关。
    """
    fx, fy, fz = front.position
    rx, ry, rz = rear.position
    center = ((fx + rx) / 2.0, (fy + ry) / 2.0, (fz + rz) / 2.0)
    return CarState(
        car_index=car_index,
        front=front,
        rear=rear,
        center=center,
        # 由**后转向架指向前转向架**才是车头方向（反过来车体会倒着走）
        heading=heading_between(rear.position, front.position),
        pitch=pitch_between(rear.position, front.position),
    )


def place_consist(path: RoutePath, head_s: float,
                  lengths: Sequence[float],
                  bogie_half_spacing: Sequence[float],
                  coupling_gap: float = 0.0) -> list[CarState]:
    """把一列编组摆到路径上。

    参数均为**每节车**一个值（下标即车厢序号，0 号在前）：

    * ``lengths`` —— 车长（车钩面到车钩面）；
    * ``bogie_half_spacing`` —— 转向架中心到车体中心的距离 ``b``
      （即两个转向架的中心距为 ``2b``）；
    * ``coupling_gap`` —— 相邻车厢车钩面之间的间隙。

    ``head_s`` 是**首车前端转向架中心**所在的弧长。

    递推关系（概要设计 §5.6）::

        车体中心   center_0 = head_s - b_0
        首/尾转向架 = center_i ± b_i
        center_{i+1} = center_i - lengths_i/2 - coupling_gap - lengths_{i+1}/2

    后一个式子是"相邻两车的车钩面之间留出 coupling_gap"，比直接递推转向架间距
    更贴近真实列车的连挂关系。
    """
    if not lengths or len(lengths) != len(bogie_half_spacing):
        raise PathError(
            "lengths and bogie_half_spacing must be non-empty and the same length"
        )
    for i, (length, half) in enumerate(zip(lengths, bogie_half_spacing)):
        if 2.0 * half > length:
            raise PathError(
                f"car {i}: bogie spacing ({2 * half:.3f} m) exceeds car length "
                f"({length:.3f} m); the bogies would stick out of the body"
            )

    cars: list[CarState] = []
    center_s = head_s - bogie_half_spacing[0]
    for i, half in enumerate(bogie_half_spacing):
        front = bogie_at(path, center_s + half)
        rear = bogie_at(path, center_s - half)
        cars.append(car_from_bogies(i, front, rear))
        if i + 1 < len(lengths):
            center_s -= (
                lengths[i] / 2.0
                + coupling_gap
                + lengths[i + 1] / 2.0
            )
    return cars


def consist_length(lengths: Sequence[float], coupling_gap: float = 0.0) -> float:
    """编组首尾占用的总弧长（用于检查是否比闭环还长）。"""
    if not lengths:
        return 0.0
    return sum(lengths) + coupling_gap * (len(lengths) - 1)
