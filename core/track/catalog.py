"""轨道件目录：件的定义、端口与路由。

核心设计（对应概要设计 §5.3）
------------------------------------------------
轨道件的目录数据里**只写一份几何**，端口位姿由路由反推出来，因此不存在
"端口写错、弧长对不上"的可能。每个件由若干条 *route* 组成，一条 route 描述
"从入口端口走到出口端口"的一段等曲率圆弧：

    {"from": "a", "to": "b", "length": 20.0, "dtheta_deg": 0.0}

派生规则
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. 在 route 自身的局部坐标里，入口行进帧是原点、heading=0，出口行进帧是
   ``arc_end_pose(length, dtheta)``。
2. route 可选 ``translate``（先平移）与 ``offset_deg``（后旋转），把整条 route
   摆到件坐标系里。次序是"**先平移、后旋转**"，注意:

       T = R(offset) ∘ Translate(t)

   这样 ``offset_deg=45`` 的等长 route 才是绕件原点旋转 45°，而不是绕自己转。
3. 端口位姿由行进帧推出：入口端口的**外指向**是行进方向掉头 180°，出口端口的
   外指向就是行进方向。「外指向」= 列车离开该件时朝的方向 —— 见
   ``core.geometry`` 里对端口约定的说明。

于是「件坐标里某端口朝哪」这件事被推导了两次（至少两条 route 共享端口时），
``_build_ports`` 会校验它们**互相一致**，不一致直接报错。这是目录数据自检的
第一道闸。

列车能不能走通
------------------------------------------------
"能不能从某端口走到某端口"完全由 route 决定，不需要任何特殊逻辑：
直轨/弯轨是单通路，交叉是两条**互不相通**的通路（列车直穿、不能转弯），
道岔是两条**共享入口**的通路（必须按道岔状态选路）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from core.geometry import Pose, arc_end_pose, arc_point, normalize_angle

#: 数据目录（项目根 / data）。
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

#: 端口位姿一致性判定的容差（米 / 弧度）。
PORT_TOL = 1e-9


class CatalogError(ValueError):
    """目录数据有误（件定义自相矛盾）。"""


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class Port:
    """轨道件的一个接驳端口。

    ``pose`` 是**件局部坐标系**下的位姿，其 heading 指向**件外**
    （列车从该端口离开本件时的方向）。
    """

    id: str
    pose: Pose

    @property
    def outward(self) -> Pose:
        return self.pose

    @property
    def inbound(self) -> Pose:
        """列车**进入**本件时的行进帧（外指向掉头 180°）。"""
        return self.pose.flipped()


@dataclass(frozen=True, slots=True)
class Route:
    """一条通路：从 ``from_port`` 走到 ``to_port`` 的一段等曲率圆弧。

    ``length`` 是**地面投影**弧长，``dtheta`` 是总转角（正 = 向左/朝 +Z 转）。
    高度变化由 ``grade``（每单位平面弧长的抬升）表达。
    """

    index: int
    from_port: str
    to_port: str
    length: float
    dtheta: float
    grade: float = 0.0

    @property
    def rise(self) -> float:
        return self.length * self.grade

    @property
    def radius(self) -> float:
        """曲线半径（**恒为正**，转向由 ``dtheta`` 的符号表达）；直线时为 ``inf``。"""
        if abs(self.dtheta) < 1e-12:
            return math.inf
        return abs(self.length / self.dtheta)

    @property
    def signed_radius(self) -> float:
        """带符号半径（正 = 左转）。仅在内部几何推导里用。"""
        if abs(self.dtheta) < 1e-12:
            return math.inf
        return self.length / self.dtheta

    def endpoints(self) -> tuple[str, str]:
        return (self.from_port, self.to_port)


@dataclass(frozen=True, slots=True)
class PieceDef:
    """一种轨道件的完整定义。"""

    id: str
    name: str
    category: str
    mesh: str
    ports: Mapping[str, Port]
    routes: tuple[Route, ...]
    spec: Mapping = field(default_factory=dict, compare=False, repr=False)

    # ------------------------------------------------------------ 端口查询

    def port(self, port_id: str) -> Port:
        try:
            return self.ports[port_id]
        except KeyError:
            raise CatalogError(
                f"piece {self.id!r} has no port {port_id!r}; known: {sorted(self.ports)}"
            ) from None

    @property
    def port_ids(self) -> tuple[str, ...]:
        return tuple(self.ports)

    # ------------------------------------------------------------ 路由查询

    def routes_from(self, port_id: str) -> tuple[Route, ...]:
        """以 ``port_id`` 为入口的 route（正走）。"""
        return tuple(r for r in self.routes if r.from_port == port_id)

    def routes_to(self, port_id: str) -> tuple[Route, ...]:
        """以 ``port_id`` 为出口的 route（反走的那些）。"""
        return tuple(r for r in self.routes if r.to_port == port_id)

    def route(self, index: int) -> Route:
        return self.routes[index]

    @property
    def is_switch(self) -> bool:
        """是否为道岔：存在某个端口发出多于一条 route。"""
        return any(len(self.routes_from(p)) > 1 for p in self.ports)

    @property
    def fog_port(self) -> str | None:
        """道岔的岔尖端口（多条 route 共同的入口）。"""
        for pid in self.ports:
            if len(self.routes_from(pid)) > 1:
                return pid
        return None

    @property
    def switch_route_indices(self) -> tuple[int, ...]:
        """道岔可选档位（route 下标）；非道岔时为空。"""
        fog = self.fog_port
        if fog is None:
            return ()
        return tuple(r.index for r in self.routes_from(fog))

    # ------------------------------------------------------------------ 走线

    def route_midpoint_local(self, route_index: int) -> tuple[float, float, float]:
        """某条 route 中点在**件局部**坐标系里的位置 ``(x, y, z)``。

        给编辑器当"鼠标抓手"用。只用端口和件原点是不够的：一节 20 m 直轨的两个
        端口在屏幕上相隔两百多像素，而拾取半径只有 110 像素，正中间就会变成
        "指上去没反应"的死区。补上每条 route 的中点后，件上任何一点到最近抓手的
        距离都小于半个件长，死区自然消失。

        推导与 :func:`_ports_from_route` 共用同一个 ``_route_placement``，所以
        "route 摆在件坐标的哪里"这件事全项目仍然只有一处实现。
        """
        route = self.route(route_index)
        spec = self.spec["routes"][route_index]
        placement = _route_placement(spec)
        dx, dz, _ = arc_point(route.length, route.dtheta, route.length * 0.5)
        rise = route.rise * 0.5

        # 先平移后旋转，与 _ports_from_route 一致
        cos_h = math.cos(placement.heading)
        sin_h = math.sin(placement.heading)
        return (
            placement.x + cos_h * dx - sin_h * dz,
            placement.y + rise,
            placement.z + sin_h * dx + cos_h * dz,
        )

    def traversals(self, entry_port: str, switch_index: int | None = None):
        """从 ``entry_port`` 进入本件后，可能的出口。

        返回 ``[(exit_port_id, route_index, reversed), ...]``。
        单通路件固定一个结果；道岔件在岔尖处有多个结果，由 ``switch_index``
        选出（为 ``None`` 时全部返回，供 UI 预览或路径搜索）。

        反向走也支持：若 route 的 ``to_port`` 是入口，则沿同一几何反向通过。

        道岔只对**顺向**（从岔尖 ``a`` 进）施加 ``switch_index`` 约束：扳到哪条
        route，岔尖就只放哪条走。**逆向**（从直股 ``b`` 或岔股 ``c`` 回到岔尖
        ``a``）一律放行 —— 这是「可挤岔」道岔（trailable turnout）的约定：列车
        从后面顶着走时尖轨会被车体顺势挤开，所以「从岔道回到主线」不该被
        道岔状态卡成死路。
        """
        forward = self.routes_from(entry_port)
        backward = self.routes_to(entry_port)

        if forward and backward:
            raise CatalogError(
                f"piece {self.id!r} port {entry_port!r} is both an entry and an exit "
                "(ambiguous traversal)"
            )

        if forward:
            if switch_index is not None:
                chosen = [r for r in forward if r.index == switch_index]
                if not chosen:
                    raise CatalogError(
                        f"switch index {switch_index} is not a route leaving "
                        f"{entry_port!r} of piece {self.id!r}"
                    )
                forward = tuple(chosen)
            return tuple((r.to_port, r.index, False) for r in forward)

        # 逆向不拦：可挤岔，从直股 / 岔股回到岔尖总是走得通。
        return tuple((r.from_port, r.index, True) for r in backward)

    def derive_traversal(self, entry_port: str, exit_port: str,
                         pos_tol: float = 1e-9) -> tuple[float, float]:
        """给出实际要走的方向（由入口/出口端口位姿反解），返回 ``(弧长, 总转角)``。

        对正向与反向走一视同仁 —— 反走就是同一段圆弧走两遍，总转角取负。
        这里复用 ``route_geometry`` 而不是自己判方向，因此走线几何只有一处实现。
        """
        from core.geometry import route_geometry

        entry = self.port(entry_port).inbound
        exit_pose = self.port(exit_port).pose
        return route_geometry(entry, exit_pose, pos_tol=pos_tol)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"PieceDef({self.id!r}, ports={list(self.ports)!r}, "
            f"routes={len(self.routes)}, switch={self.is_switch})"
        )


# --------------------------------------------------------------------------- #
# 由 spec 构建件
# --------------------------------------------------------------------------- #

def _route_placement(spec: Mapping) -> Pose:
    """把 (translate, offset_deg) 合成 route 在件坐标里的摆放变换。

    次序为「先平移、后旋转」：``T = R(offset) ∘ Translate(t)``。
    """
    tx, tz = spec.get("translate", (0.0, 0.0))
    offset = math.radians(float(spec.get("offset_deg", 0.0)))
    c = math.cos(offset)
    s = math.sin(offset)
    return Pose(c * tx - s * tz, 0.0, s * tx + c * tz, offset)


def _route_geometry_from_spec(spec: Mapping, piece_id: str) -> tuple[float, float]:
    """从 spec 取出 ``(length, dtheta)``；允许用 ``radius`` 代替 ``length``。"""
    dtheta = math.radians(float(spec.get("dtheta_deg", 0.0)))
    has_length = "length" in spec
    has_radius = "radius" in spec
    if has_length and has_radius:
        raise CatalogError(
            f"piece {piece_id!r} route {spec.get('from')!r}->{spec.get('to')!r}: "
            "specify either 'length' or 'radius', not both"
        )
    if has_length:
        length = float(spec["length"])
    elif has_radius:
        length = abs(float(spec["radius"]) * dtheta)
    else:
        raise CatalogError(
            f"piece {piece_id!r} route {spec.get('from')!r}->{spec.get('to')!r}: "
            "needs 'length' or 'radius'"
        )
    if length <= 0.0:
        raise CatalogError(
            f"piece {piece_id!r}: route length must be positive (got {length})"
        )
    return length, dtheta


def _build_route(spec: Mapping, index: int, piece_id: str) -> Route:
    from_port = spec.get("from")
    to_port = spec.get("to")
    if not from_port or not to_port:
        raise CatalogError(
            f"piece {piece_id!r} route #{index}: 'from' and 'to' are required"
        )
    if from_port == to_port:
        raise CatalogError(
            f"piece {piece_id!r} route #{index}: 'from' and 'to' must differ"
        )
    length, dtheta = _route_geometry_from_spec(spec, piece_id)
    grade = float(spec.get("grade", 0.0))
    return Route(
        index=index,
        from_port=str(from_port),
        to_port=str(to_port),
        length=length,
        dtheta=dtheta,
        grade=grade,
    )


def _ports_from_route(route: Route, spec: Mapping) -> dict[str, Pose]:
    """由一条 route 推出它两个端口的**件局部**位姿。"""
    placement = _route_placement(spec)
    dx, dz = arc_end_pose(route.length, route.dtheta)
    rise = route.rise

    entry_travel = placement  # 入口行进帧
    exit_travel = placement.compose(Pose(dx, rise, dz, route.dtheta))

    return {
        route.from_port: entry_travel.flipped(),   # 入口外指向 = 行进方向掉头
        route.to_port: exit_travel,                # 出口外指向 = 行进方向
    }


def _merge_port(ports: dict[str, Pose], port_id: str, pose: Pose, piece_id: str) -> None:
    """把新推出的端口位姿并入；若与已推出的不一致则报错。"""
    existing = ports.get(port_id)
    if existing is None:
        ports[port_id] = pose
        return
    if (
        existing.distance_to(pose) > PORT_TOL
        or abs(existing.heading_error_to(pose)) > PORT_TOL
    ):
        raise CatalogError(
            f"piece {piece_id!r}: port {port_id!r} derived inconsistently "
            f"(routes disagree). first={existing} second={pose}"
        )


def build_piece(spec: Mapping) -> PieceDef:
    """由一个 spec（dict / JSON 对象）构建 :class:`PieceDef` 并自检。"""
    piece_id = spec.get("id")
    if not piece_id:
        raise CatalogError("piece spec needs an 'id'")

    route_specs: Sequence[Mapping] = spec.get("routes", ())
    routes = tuple(_build_route(rs, i, piece_id) for i, rs in enumerate(route_specs))

    ports: dict[str, Pose] = {}
    for route, route_spec in zip(routes, route_specs):
        for port_id, pose in _ports_from_route(route, route_spec).items():
            _merge_port(ports, port_id, pose, piece_id)

    # 无 route 的件（如缓冲端）允许显式声明端口。
    derived = set(ports)
    for explicit in spec.get("ports", ()):
        pid = explicit["id"]
        pose = Pose.from_degrees(
            explicit.get("x", 0.0),
            explicit.get("y", 0.0),
            explicit.get("z", 0.0),
            explicit.get("heading_deg", 0.0),
        )
        _merge_port(ports, pid, pose, piece_id)

    if not ports:
        raise CatalogError(f"piece {piece_id!r} declared no ports")

    # 每个端口都必须被至少一条 route 用到；否则多半是写错了端口名。
    # （route 的端点在派生时必然存在，所以这里能抓的是「声明了却没接上」的端口。）
    if routes:
        referenced = {pid for route in routes for pid in route.endpoints()}
        unreferenced = sorted(set(ports) - referenced - derived)
        if unreferenced:
            raise CatalogError(
                f"piece {piece_id!r} declares port(s) {unreferenced} that no route "
                "uses; a port on a piece with routes must be reachable"
            )

    return PieceDef(
        id=str(piece_id),
        name=str(spec.get("name", piece_id)),
        category=str(spec.get("category", "misc")),
        mesh=str(spec.get("mesh", "ballast")),
        ports={pid: Port(id=pid, pose=pose) for pid, pose in ports.items()},
        routes=routes,
        spec=dict(spec),
    )


# --------------------------------------------------------------------------- #
# 目录
# --------------------------------------------------------------------------- #

@dataclass
class Catalog:
    """轨道件目录。按 id 索引，可自检。"""

    pieces: dict[str, PieceDef]
    source: Path | None = None

    # ---------------------------------------------------------------- 构建

    @classmethod
    def from_specs(cls, specs: Iterable[Mapping],
                   source: Path | None = None) -> "Catalog":
        table: dict[str, PieceDef] = {}
        for spec in specs:
            piece = build_piece(spec)
            if piece.id in table:
                raise CatalogError(f"duplicate piece id {piece.id!r}")
            table[piece.id] = piece
        return cls(pieces=table, source=source)

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls.from_specs(payload.get("pieces", ()), source=path)

    @classmethod
    def builtin(cls) -> "Catalog":
        """加载仓库自带的 ``data/pieces.json``。"""
        return cls.load(DATA_DIR / "pieces.json")

    # ---------------------------------------------------------------- 查询

    def __getitem__(self, piece_id: str) -> PieceDef:
        try:
            return self.pieces[piece_id]
        except KeyError:
            raise CatalogError(
                f"unknown piece {piece_id!r}; known: {sorted(self.pieces)}"
            ) from None

    def __contains__(self, piece_id: object) -> bool:
        return piece_id in self.pieces

    def __iter__(self) -> Iterator[PieceDef]:
        return iter(self.pieces.values())

    def __len__(self) -> int:
        return len(self.pieces)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self.pieces)

    def by_category(self, category: str) -> tuple[PieceDef, ...]:
        return tuple(p for p in self.pieces.values() if p.category == category)

    @property
    def categories(self) -> tuple[str, ...]:
        seen: list[str] = []
        for piece in self.pieces.values():
            if piece.category not in seen:
                seen.append(piece.category)
        return tuple(seen)

    def switches(self) -> tuple[PieceDef, ...]:
        return tuple(p for p in self.pieces.values() if p.is_switch)

    def with_port_count(self, count: int) -> tuple[PieceDef, ...]:
        return tuple(p for p in self.pieces.values() if len(p.ports) == count)

    def closure_helpers(self, preferred: str | None = None) -> tuple[str, ...]:
        """可以拿去把开链补成闭环的「普通件」id；``preferred`` 排在最前。

        只有**单通路**的件能进这张表：

        * 道岔、交叉件会把线接出岔路 —— 自动补线"补"出一堆分叉不是用户想要的；
        * 缓冲端没有通路，接上去只会把线封死；
        * 坡道会让闭环在**竖向**对不上（闭环判据是平面的，见
          :func:`core.track.path.detect_closure`），所以也不进表。

        顺序是确定的（直轨由短到长，再弯轨按弧长），于是
        :meth:`core.track.layout.Layout.pick_closing_piece` 在打平时结果可复现。
        ``preferred`` 是用户当前手上那一件：它排最前，所以**用它就能合上时绝不会
        被擅自换成别的**。
        """
        usable = [
            piece for piece in self.pieces.values()
            if len(piece.routes) == 1
            and piece.category in ("straight", "curve")
        ]
        usable.sort(key=lambda p: (
            p.category != "straight", p.routes[0].length, p.routes[0].dtheta,
        ))
        ids = [piece.id for piece in usable]
        if preferred in ids:
            ids.remove(preferred)
            ids.insert(0, preferred)
        return tuple(ids)

    # -------------------------------------------------------------- 自检

    def validate(self) -> list[str]:
        """对每个件、每条 route 做一次几何往返校验，返回告警文本。

        校验内容：由端口位姿反解出的 ``(弧长, 总转角)`` 是否等于 route 声明的值。
        这是「目录数据不可能自相矛盾」这条承诺的兑现处 —— 一旦有人手改错误，
        这里立刻报出来。
        """
        warnings: list[str] = []
        for piece in self.pieces.values():
            for route in piece.routes:
                try:
                    got_length, got_dtheta = piece.derive_traversal(
                        route.from_port, route.to_port
                    )
                except Exception as exc:  # noqa: BLE001 - 汇总为告警
                    warnings.append(
                        f"{piece.id} route #{route.index} "
                        f"({route.from_port}->{route.to_port}): 反解失败: {exc}"
                    )
                    continue

                if abs(got_length - route.length) > 1e-6:
                    warnings.append(
                        f"{piece.id} route #{route.index}: 弧长不符 "
                        f"声明 {route.length:.9f} 反解 {got_length:.9f}"
                    )
                if abs(normalize_angle(got_dtheta - route.dtheta)) > 1e-9:
                    warnings.append(
                        f"{piece.id} route #{route.index}: 转角不符 "
                        f"声明 {math.degrees(route.dtheta):.9f}° "
                        f"反解 {math.degrees(got_dtheta):.9f}°"
                    )

                if abs(route.dtheta) >= math.pi:
                    warnings.append(
                        f"{piece.id} route #{route.index}: 单条 route 转角必须 < 180°"
                    )
        return warnings
