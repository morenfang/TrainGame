"""一列编组在轨道上的可见表示：把 core 解算好的位姿搬到场景图。

职责边界
--------------------------------------------------------------------
只做两件事，而且两件的**内容**都已经在 core 里算完了：

1. **摆位**：``core.track.path.place_consist`` 由「首车前端转向架所在的弧长」
   解出每节车的车体中心 / 航向 / 俯仰；这里把它们写进场景图节点。
2. **推进**：``core.train.dynamics.Train`` 管手柄、速度、牵引与阻力；这里每帧
   调一次 ``advance`` 再 ``sync`` 一次。

所以本模块里**没有任何物理与几何公式** —— 车过弯为什么"切内弦"、牵引力怎么
随速度衰减、哪些车要掉头，全在 core 那一侧，而且都能 headless 单测。这里一旦
出现了公式，就说明有一份几何被写了两遍，两份迟早会对不上。

坐标系：车体局部 +X 就是车头
--------------------------------------------------------------------
``render.train_mesh`` 生成的网格原点在车体几何中心、+X 指向车头、``y = 0`` 是
轨面；``CarState.body_pose`` 给的正是「两转向架中点 + 由后转向架指向前转向架」
的位姿。两者是同一套约定，所以把位姿直接写进节点就对位了，中间不需要任何
修正量 —— 修正量就是"两份几何对不上"的第一个征兆。

俯仰单独传
--------------------------------------------------------------------
``CarState`` 有 ``pitch``，但 ``Pose`` 装不下（见 ``render.transform``：俯仰在
Panda3D 的 **R** 分量上）。于是这里显式地把 ``state.pitch`` 交给
:func:`render.transform.apply_pose`，而不是先把 ``body_pose`` 拼出来再补。

编组放不下的时候
--------------------------------------------------------------------
空场地上还没闭合的轨道是很常见的状态，所以 :meth:`TrainView.sync` **不抛异常**：
放不下就把整列车收起来，并用 :meth:`TrainView.placement_error` 把原因说清楚
（"编组 176 m，比这条 120 m 的线路还长"）。用户看到的是"车没出来 + 为什么"，
而不是一个 traceback。
"""

from __future__ import annotations

import math

from panda3d.core import NodePath

from core.geometry import Pose, normalize_angle
from core.track.path import CarState, PathError, RoutePath, place_consist, trace
from core.train.consist import TrainSpec
from core.train.dynamics import Train, TrainState
from render import gltf_train, train_mesh
from render.transform import apply_pose

#: 开链上首车与线路末端的最小间隙（米）。贴着端头摆没有意义，也容易让人以为
#: 车"卡住了"。闭环路径不需要它 —— 那里任何位置都能摆下。
_OPEN_PATH_MARGIN = 0.0

#: 路径换向时重新锚定首车的粗扫步长（米），细扫会自行收到亚毫米级。
_ANCHOR_STEP = 0.5
#: 重锚定打分里"航向差"折算成"差多少米"的权重。
#:
#: 只看位置最近是不够的：立体交叉处上下两条轨道在平面上挨得很近，位置分不出
#: 谁是谁；加上航向项才能选中"顺着列车走"的那条。取 8 m/rad —— 90° 的航向差
#: 相当于 12.6 m，比"上下两层轨道"的平面间距大、又比一个曲线半径小。
_ANCHOR_HEADING_WEIGHT = 8.0

#: 重走线落点航向与列车航向差超过这个角（rad）就判定为「被折回头」。道岔一扳，
#: 从车尾重走可能在道岔处就岔开、把车头甩到反向段（航向差 180°）；此时改从车头
#: 所在段重走。取 90° 足够区分"顺着走"与"整个掉头"。
_RETRACE_FLIP_TOLERANCE = math.radians(90.0)


class TrainView:
    """一列编组的可见表示 + 每帧跟随。

    节点结构::

        root
          car_0_df4b_loco      ← 每节车一个挂点，位姿每帧被 sync 覆写
          car_1_coach_25b
          ...

    每节车的**网格模板按 ``(车型 id, 是否掉头)`` 缓存**，实例之间用 ``copyTo``
    共享 Geom —— 8 节同型车厢在显存里只占一份，这是 ``LayoutView`` 已经用过的
    同一套做法。
    """

    def __init__(self, spec: TrainSpec, parent: NodePath, *,
                 details: bool = True, train: Train | None = None,
                 name: str | None = None):
        self.spec = spec
        self.details = details
        self.train = train if train is not None else Train(spec)
        self.root = parent.attachNewNode(name or f"train_{spec.id}")

        self._templates: dict[tuple[str, bool], NodePath] = {}
        self._template_triangles: dict[tuple[str, bool], int] = {}
        self._cars: list[NodePath] = []
        self._states: tuple[CarState, ...] = ()
        self._reason: str | None = "还没有可行驶的轨道"

        #: 上一次拿到的**源**路径（布局交出来的那条）与真正在跑的路径。两者不总
        #: 相同：源路径方向若与列车朝向相反，就换成它的反向路径来跑（见 :meth:`_align`）。
        self._source: RoutePath | None = None
        self._driving: RoutePath | None = None
        #: 首车前端转向架上一次的世界位姿。路径一换就拿它当锚，把 ``state.s``
        #: 重新定到新路径上 —— 列车的世界位置与朝向因此不被"换了一条路"打扰。
        self._anchor: tuple[float, float, float] | None = None
        self._anchor_heading = 0.0

        self._facings = spec.car_facings()
        self._light_holder = None
        self._build_cars()

    # ==================================================================== #
    # 构建
    # ==================================================================== #

    def _template(self, car, flipped: bool) -> NodePath:
        key = (car.id, flipped)
        template = self._templates.get(key)
        if template is None:
            builder = train_mesh.build_car_mesh_oriented(car, flipped,
                                                         details=self.details)
            self._template_triangles[key] = builder.triangle_count
            template = builder.build()
            template.setName(f"tpl_{car.id}{'_rev' if flipped else ''}")
            self._templates[key] = template
        return template

    def _build_cars(self) -> None:
        gltf_cars = gltf_train.cars_for(self.spec)
        if gltf_cars is not None:
            self._light_holder = gltf_train.light_train(self.root)
            for index, (car, template) in enumerate(zip(self.spec.cars, gltf_cars)):
                node = template.copyTo(self.root)
                node.setName(f"car_{index}_{car.id}")
                self._cars.append(node)
                key = (car.id, self._facings[index])
                if key not in self._template_triangles:
                    self._template_triangles[key] = gltf_train.triangle_count(template)
            return
        for index, (car, flipped) in enumerate(zip(self.spec.cars, self._facings)):
            node = self._template(car, flipped).copyTo(self.root)
            node.setName(f"car_{index}_{car.id}")
            self._cars.append(node)

    def destroy(self) -> None:
        """从场景里摘掉整列车（模板与车节点一起）。"""
        if self._light_holder is not None:
            self._light_holder.removeNode()
            self._light_holder = None
        self.root.removeNode()
        self._cars.clear()
        self._templates.clear()
        self._template_triangles.clear()
        self._states = ()
        self._source = None
        self._driving = None
        self._anchor = None

    # ==================================================================== #
    # 查询
    # ==================================================================== #

    @property
    def state(self) -> TrainState:
        return self.train.state

    @property
    def states(self) -> tuple[CarState, ...]:
        """上一帧每节车的状态（``sync`` 之后才有）。"""
        return self._states

    @property
    def car_count(self) -> int:
        return len(self._cars)

    @property
    def consist_length(self) -> float:
        """编组首尾占用的总弧长（含车钩间隙），单位米。"""
        if not self.spec.cars:
            return 0.0
        return (sum(self.spec.lengths)
                + self.spec.coupling_gap * (self.spec.car_count - 1))

    @property
    def placement_error(self) -> str | None:
        """上一帧"摆不上去"的原因；能摆下时是 ``None``。"""
        return self._reason

    @property
    def visible(self) -> bool:
        return not self.root.isHidden()

    def speed_kmh(self) -> float:
        return self.state.speed_kmh

    def node_for(self, index: int) -> NodePath | None:
        if 0 <= index < len(self._cars):
            return self._cars[index]
        return None

    def occupies_piece(self, piece_index: int) -> bool:
        """列车**车身**当前是否压在这一节轨道上。

        删轨前用它判断"要删的这一节上有没有列车"：编组占的是路径上一段连续的
        弧长 ``[head - 编组全长, head]``，只要这一节轨道的弧长区间与它相交，
        就认为车正压在它上面。闭环时弧长会绕回，所以要多平移一个整圈去对齐。
        """
        path = self._driving if self._driving is not None else self._source
        if path is None or not path.segments:
            return False
        head = self.state.s
        tail = head - self.consist_length
        total = path.total_length
        for segment in path.segments:
            if segment.piece_index != piece_index:
                continue
            if path.closed:
                for shift in (-total, 0.0, total):
                    if segment.s0 + shift <= head and segment.s1 + shift >= tail:
                        return True
            elif segment.s1 >= tail and segment.s0 <= head:
                return True
        return False

    def triangle_count(self) -> int:
        """整个编组的三角形数（模板各算一次 —— 实例共享 Geom）。

        按**节数**累加而不是按模板去重：屏幕上确实画了这么多三角形，
        这正是显存里的一份 Geom 被画了多次的意思。
        """
        total = 0
        for car, flipped in zip(self.spec.cars, self._facings):
            total += self._template_triangles.get((car.id, flipped), 0)
        return total

    # ==================================================================== #
    # 摆放
    # ==================================================================== #

    def head_range(self, path: RoutePath | None) -> tuple[float, float] | None:
        """首车前端转向架**可以**待在的弧长区间；放不下时返回 ``None``。

        闭环上任何位置都行（弧长对总长取模），所以区间就是 ``[0, 总长)``。
        开链上整列车必须落在轨道内，于是首车不能比 ``编组全长`` 更靠前 ——
        车尾会在起点之前"悬空"，那时候 ``pose_at`` 会直接抛错。
        """
        if path is None or not path.segments:
            return None
        total = path.total_length
        length = self.consist_length
        if length > total:
            return None
        if path.closed:
            return (0.0, total)
        return (min(length + _OPEN_PATH_MARGIN, total), total)

    def clamp_head(self, path: RoutePath | None) -> float | None:
        """把首车弧长夹进合法区间，返回夹住之后的值（放不下时 ``None``）。"""
        span = self.head_range(path)
        if span is None:
            return None
        low, high = span
        head = self.state.s
        if path.closed:
            total = path.total_length
            return head % total if total > 0.0 else 0.0
        return max(low, min(high, head))

    def sync(self, path: RoutePath | None, layout=None) -> bool:
        """按当前 ``state.s`` 把编组摆到 ``path`` 上。摆得下返回 ``True``。

        摆不下时把整列车藏起来并记下原因 —— 见模块文档"编组放不下的时候"。
        """
        self._reason = self._placement_error_for(path)
        if self._reason is not None:
            self.root.hide()
            self._states = ()
            return False

        self._align(path, layout)
        return self._place()

    def advance(self, dt: float, path: RoutePath | None, layout=None) -> bool:
        """推进 ``dt`` 秒并重新摆位。返回"这一帧列车在场景里"与否。"""
        self._reason = self._placement_error_for(path)
        if self._reason is not None:
            return self.sync(path, layout)
        self._align(path, layout)
        head = self.clamp_head(self._driving)
        if head is None:
            # _align 可能把列车锚到一条更短的路（删轨 / 改线后重走线），编组摆不下。
            # 这时收起列车并记下原因，而不是让 assert 把进程带崩。
            self.root.hide()
            self._states = ()
            self._reason = self._placement_error_for(self._driving)
            return False
        self.state.s = head
        self.train.advance(dt, self._driving)
        return self._place()

    # ---------------------------------------------------------------- 锚定

    def _align(self, path: RoutePath | None, layout=None) -> None:
        """路径换了就把列车的 ``state.s`` 重新定到新路径上，**不让它原地掉头**。

        什么时候会换路径：扳道岔、接轨 / 删轨、撤销、读档 —— 只要布局变了，
        :class:`render.scene.LayoutView` 就会重跑闭环检测。新路径可能换了起点、
        换了总长，甚至**方向相反**（同一圈环从另一头起步走），而 ``state.s`` 只是
        一个弧长数，直接留着会让列车瞬移甚至倒着开。

        首选方案：只要给了 ``layout``，就从列车**车尾所在那一段的入口端口**重新
        走线（:func:`_retrace_from_state`）。入口端口就是列车此刻的来向，新路径
        首段方向必然与列车一致，车头不会被折回；而且 ``trace`` 全程按当前道岔档位
        走线，顺向该走哪条腿就哪条腿 —— 这正是立交场景扳道岔时"车不从岔尖硬挤
        进岔股"的关键。

        退一步（没有 ``layout``、或重走失败）：按首车**上一次的世界位姿**，在源
        路径与它的反向路径里各找最贴合的那一点（位置最近 + 朝向最顺），把 ``s``
        定上去。方向一致的那条会被选中，列车于是接着原方向往前开。
        """
        if path is self._source and self._driving is not None:
            return                              # 还是同一条路，什么都不用做
        self._source = path
        if path is None or not path.segments:
            self._driving = path
            return
        if self._anchor is None or not self._states:
            # 第一次上轨（或上一帧还被藏着）：没有可依据的位姿，按原 s 放。
            self._driving = path
            return

        previous = self._driving
        if layout is not None and previous is not None and previous.segments:
            retraced = _retrace_from_state(
                layout, previous, self.state.s, self._anchor, self._anchor_heading,
                self.consist_length)
            if retraced is not None:
                self._driving, self.state.s = retraced
                return

        self._driving, self.state.s = _best_anchored_path(
            path, self._anchor, self._anchor_heading)

    def _place(self) -> bool:
        """按当前 ``state.s`` 把编组摆到 :attr:`_driving` 上（并记下新锚点）。"""
        driving = self._driving
        head = self.clamp_head(driving)
        if head is None:
            self.root.hide()
            self._states = ()
            return False
        self.state.s = head
        self._states = tuple(self.train.place(driving))
        for node, state in zip(self._cars, self._states):
            apply_pose(node, state.body_pose, state.pitch)
        anchor = driving.pose_at(head)
        self._anchor = anchor.position
        self._anchor_heading = anchor.heading
        self.root.show()
        return True

    def set_handle(self, handle: float) -> None:
        """驾驶手柄：``> 0`` 牵引、``< 0`` 制动、``0`` 惰行。"""
        self.train.set_handle(handle)

    @property
    def handle(self) -> float:
        """当前手柄位（牵引为正、制动为负、惰行为 0）。"""
        return self.train.handle

    def nudge_handle(self, step: float) -> float:
        """手柄推一档（并夹在 ``[-1, 1]`` 内），返回新档位。"""
        self.set_handle(self.handle + step)
        return self.handle

    def _placement_error_for(self, path: RoutePath | None) -> str | None:
        if path is None or not path.segments:
            return "还没有可行驶的轨道"
        total = path.total_length
        if total <= 0.0:
            return "轨道总长为零，摆不下"
        if self.consist_length > total:
            return (f"编组全长 {self.consist_length:.1f} m，"
                    f"比这条 {total:.1f} m 的线路还长")
        return None


# --------------------------------------------------------------------------- #
# 换路径时的重锚定（纯几何，可 headless 单测）
# --------------------------------------------------------------------------- #

def _reversal_preserves_facing(path: RoutePath, reversed_path: RoutePath) -> bool:
    """反向之后，道岔的**顺向**约束是不是还成立。

    可挤岔道岔的规则只有一条：顺向（从岔尖 ``a`` 进）看档位，逆向（从直股 ``b``
    或岔股 ``c`` 回岔尖）永远放行。把一条路径反个方向，会把这两种方向逐一翻个：

    * 原来的**顺向**段 → 反向里的**逆向**段：仍然放行，没问题；
    * 原来的**逆向**段 → 反向里的**顺向**段：现在要看档位了。

    所以反向是否合法，取决于反向路径里每个**顺向**道岔段用的档位，是不是等于这条
    道岔在正向前进时被允许走的那一档（正向前进时的顺向档位就是道岔当前档位）。
    只有相等才允许反向 —— 否则列车会「挤」出一条顺向档位根本不对的路（例如
    ``#20`` 扳直股，反向却从岔尖硬走进岔股）。
    """
    facing_routes: dict[int, int] = {}
    for segment in path:
        if segment.is_switch and not segment.reversed:
            facing_routes[segment.piece_index] = segment.route_index

    for segment in reversed_path:
        if segment.is_switch and not segment.reversed:
            authorized = facing_routes.get(segment.piece_index)
            if authorized is None or segment.route_index != authorized:
                return False
    return True


def _best_anchored_path(path: RoutePath, anchor: tuple[float, float, float],
                        heading: float) -> tuple[RoutePath, float]:
    """在 ``path`` 与它的反向里挑与 ``(anchor, heading)`` 最贴合的一条。

    返回 ``(要跑的路径, 弧长)``。位置最近 **且** 朝向最顺的那条胜出；因此当新路径
    与原路径方向相反时，选中的会是反向路径，列车不会被"折"回去。

    但反向**不一定合法**：可挤岔道岔只对「顺向」设卡，反向会把顺向/逆向翻个，
    可能翻出一条「顺向却走错档位」的路。那种反向路径不能拿给列车开（见
    :func:`_reversal_preserves_facing`），所以这里把它从候选里剔除。
    """
    candidates = [path]
    try:
        reversed_path = path.reversed()
    except Exception:                           # noqa: BLE001 - 反解失败就只留正向
        reversed_path = None
    if (reversed_path is not None and reversed_path.segments
            and _reversal_preserves_facing(path, reversed_path)):
        candidates.append(reversed_path)

    best: tuple[float, RoutePath, float] | None = None
    for candidate in candidates:
        s, score = _nearest_anchor(candidate, anchor, heading)
        if best is None or score < best[0]:
            best = (score, candidate, s)
    assert best is not None
    return best[1], best[2]


def _retrace_from_state(layout, previous: RoutePath, s: float,
                        anchor: tuple[float, float, float],
                        heading: float,
                        tail_length: float) -> tuple[RoutePath, float] | None:
    """从列车车尾所在处沿行进方向重新走线，返回 ``(新路径, 弧长)`` 或 ``None``。

    :func:`core.track.path.detect_closure` 交上来的路径只有**一个固定起点**。当
    布局变成「环 + 挂在环上的支线」（立交场景把 #22 扳到岔股就是这样），它返回的
    "最长可行进路径"会绕着支线把主环的一部分**反着**走 —— 列车正处在那一段时，
    按位姿锚定只能把车头折 180°，看起来就是"闪现"。

    于是退一步：从**车尾所在那一段**的入口端口出发，让 :func:`trace` 在当前布局
    下重新探一条路。入口端口就是列车此刻的来向，新路径首段方向必然与列车一致；
    ``trace`` 全程按当前道岔档位走线，顺向该走哪条腿就哪条腿，绝不会从岔尖硬挤
    进岔股。从车尾（而非车头）出发，是为了让车头落点落在 ``tail_length`` 之后，
    避免被开链的 ``head_range`` 限位把车头"夹"到前面去。

    车尾若是恰好落在**被扳动的道岔上**（或其前），从车尾重走会在这处道岔就岔开，
    车头于是被甩到路径的反向段（航向差 180°）。这时改从**车头所在段**重走：方向
    必然一致，列车接着往前走、绕一圈再进那处道岔的岔股；代价是开链限位会把车头
    往前"夹"一个编组长 —— 一次小幅前移，远好过整个掉头、甚至反向钻进错腿。
    """
    if not previous.segments:
        return None

    # 先按车尾重走。
    tail_s = s - tail_length
    if not previous.closed and tail_s < 0.0:
        tail_s = 0.0
    try:
        tail_seg, _ = previous.segment_at(tail_s)
    except PathError:
        return None
    path = _trace_from_segment(layout, tail_seg)
    if path is None:
        return None
    new_s, _ = _nearest_anchor(path, anchor, heading)
    if not _anchored_flipped(path, new_s, heading):
        return path, new_s

    # 车尾重走把车头甩反向了 —— 改从车头所在段重走。
    try:
        head_seg, _ = previous.segment_at(s)
    except PathError:
        return None
    path = _trace_from_segment(layout, head_seg)
    if path is None:
        return None
    new_s, _ = _nearest_anchor(path, anchor, heading)
    return path, new_s


def _trace_from_segment(layout, segment) -> RoutePath | None:
    """从 ``segment`` 的入口端口在当前布局下重走一条路；失败返回 ``None``。"""
    try:
        path = trace(layout, start=(segment.piece_index, segment.entry_port))
    except PathError:
        return None
    if not path.segments:
        return None
    if not path.segments[0].entry_frame.approx_equal(
            segment.entry_frame, pos_tol=1e-3, ang_tol=1e-3):
        return None
    return path


def _anchored_flipped(path: RoutePath, s: float, heading: float) -> bool:
    """重走线落点 ``s`` 处的航向与列车航向是否差过 :data:`_RETRACE_FLIP_TOLERANCE`。"""
    return abs(normalize_angle(path.pose_at(s).heading - heading)) \
        > _RETRACE_FLIP_TOLERANCE



def _nearest_anchor(path: RoutePath, anchor: tuple[float, float, float],
                    heading: float) -> tuple[float, float]:
    """路径上离 ``anchor`` 最近、朝向又最顺的弧长 ``s`` 及打分量。

    三段细化：先按 :data:`_ANCHOR_STEP` 粗扫，再在每个赢家两侧的邻域里两次放大
    搜寻 —— 收尾精度在毫米级，远小于车长，肉眼看不到跳。
    """
    total = path.total_length
    if total <= 0.0:
        return 0.0, float("inf")

    lo, hi = 0.0, total
    step = max(_ANCHOR_STEP, total / 4000.0)
    best_s, best_score = 0.0, float("inf")
    for _ in range(3):
        span = hi - lo
        count = max(1, int(math.ceil(span / step)))
        for index in range(count + 1):
            s = lo + span * index / count
            if path.closed:
                s %= total
            else:
                s = min(max(s, 0.0), total)
            pose = path.pose_at(s)
            score = _anchor_score(pose, anchor, heading)
            if score < best_score:
                best_score, best_s = score, s
        half = span / count
        lo, hi = best_s - half, best_s + half
        step = max(1e-4, (2.0 * half) / 20.0)
    return best_s, best_score


def _anchor_score(pose: Pose, anchor: tuple[float, float, float],
                  heading: float) -> float:
    """位姿与锚点的贴合度：三维距离平方 + 航向差平方（乘权重）—— 越小越贴合。"""
    dx = pose.x - anchor[0]
    dy = pose.y - anchor[1]
    dz = pose.z - anchor[2]
    mismatch = abs(normalize_angle(pose.heading - heading))
    return (dx * dx + dy * dy + dz * dz
            + (_ANCHOR_HEADING_WEIGHT * mismatch) ** 2)
