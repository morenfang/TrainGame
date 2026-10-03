"""轨道装配图：把轨道件一节一节接起来，并推导出每节的位姿。

对应概要设计 §5.2（端口吸附）与 §6.4（存档只存父子引用）。

为什么不用「位置吸附到格点」
------------------------------------------------
45° 圆弧的弦长是 ``2R·sin22.5°``，是无理数，**永远吸不上格点**。网格吸附会留下
一堆微小缝隙（"看起来接上了但差 3 cm"），列车经过时会跳。改成端口驱动后，件的
位姿由吸附链推导，闭环在数学上严丝合缝。

吸附公式
------------------------------------------------
把新件的端口 ``Q`` 接到已占用的世界端口 ``world_P`` 上，要求
① 两个接点重合；② 两者**外指向相反**（否则轨道会在接缝处折回）。于是

    new_pose ∘ Q.pose = world_P.flipped()
    =>  new_pose     = world_P.flipped() ∘ Q.pose.inverse()

`attach` 只有这一行几何，其余都是记账。
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from core.geometry import Pose, normalize_angle
from core.track.catalog import Catalog, PieceDef

#: 端口键：``(件下标, 端口名)``
PortKey = tuple[int, str]

#: ``grow_to_close`` 最多补几节。够绕满 22.5° 那档弯轨的一整圈（16 节），还留了一倍
#: 余量给"混着不同弧度接"的情况；再往上基本都是弧度/半径选错了，早点报错比
#: 默默接出一圈乱轨道好。
AUTO_CLOSE_LIMIT = 32

#: 把"航向差"折算成"差多少米"时用的权重（米/弧度）。
#:
#: 自动补线每一步要在候选件里挑最接近合拢的那一节，而缺口是**两个量**
#: （位置差、航向差）。这里合成一个标量来比较：22.5°（0.393 rad）的航向差算作
#: 约 7.9 m —— 半节 22.5° 弯轨的弦长，量级上正是"再补一节弯轨就能消掉"的意思。
_HEADING_AS_METERS = 20.0


def _pose_gap(a: Pose, b: Pose) -> tuple[float, float]:
    """:meth:`Layout.join_gap` 的位姿版：两个**端口外指向**之间的 ``(位置差, 航向差)``。"""
    return (
        a.planar_distance_to(b),
        abs(normalize_angle(a.heading - b.heading - math.pi)),
    )



class LayoutError(ValueError):
    """装配操作非法（端口已被占用、件不存在等）。"""


@dataclass(frozen=True, slots=True)
class PlacedPiece:
    """放在场地里的一节轨道件。"""

    index: int
    def_id: str
    pose: Pose
    #: 挂到哪个端口上来的：(父件下标, 父件端口, 本件端口)；根件为 None
    parent: tuple[int, str, str] | None = None

    @property
    def def_(self) -> str:  # 便于阅读的别名
        return self.def_id


@dataclass
class Layout:
    """轨道装配图。"""

    catalog: Catalog
    pieces: dict[int, PlacedPiece] = field(default_factory=dict)
    #: 双向连接表：``(件, 端口) -> (件, 端口)``
    connections: dict[PortKey, PortKey] = field(default_factory=dict)
    #: 道岔状态：件下标 -> route 下标（缺省用该件的第 0 档）
    switches: dict[int, int] = field(default_factory=dict)
    _next_index: int = 0

    # ---------------------------------------------------------------- 基本访问

    def __len__(self) -> int:
        return len(self.pieces)

    def __iter__(self) -> Iterator[PlacedPiece]:
        return iter(self.pieces.values())

    def piece(self, index: int) -> PlacedPiece:
        try:
            return self.pieces[index]
        except KeyError:
            raise LayoutError(f"no placed piece with index {index}") from None

    def definition(self, index: int) -> PieceDef:
        return self.catalog[self.piece(index).def_id]

    def root_indices(self) -> list[int]:
        """所有根件（``parent is None``）的下标 —— 每段独立轨道的起点。

        正常情况下只有一节（一张连通的装配图）。删除中间一节时，被删节下游的那
        段会原地冻结成**新的根件**，于是这里会返回多个根：每根各是一段独立轨道。
        """
        return [index for index, placed in self.pieces.items()
                if placed.parent is None]

    def component_count(self) -> int:
        """连通分量数（按 ``connections`` 图，忽略道岔走线）。空场地返回 0。

        与 :meth:`root_indices` 的不同：根数是**父子森林**里的树棵数，而删掉一个
        闭环里的一节时，下游虽冻结成新根，却仍然靠着合拢缝（``connections`` 里
        的另一类边）与主线连通 —— 那种情况根数 > 1，但这里应当仍是 1 个分量。
        编辑器用它区分两种「没闭合」：

        * 1 个分量 —— 扳道岔改线 / 半条开链：**一条连通**的线，可以开链跑车；
        * ≥ 2 个分量 —— 轨道被真正拆散成互不相连的几段，该进无车状态。
        """
        if not self.pieces:
            return 0
        seen: set[int] = set()
        count = 0
        for start in self.pieces:
            if start in seen:
                continue
            count += 1
            stack = [start]
            seen.add(start)
            while stack:
                current = stack.pop()
                for port_id in self.definition(current).port_ids:
                    peer = self.connections.get((current, port_id))
                    if peer is not None and peer[0] not in seen:
                        seen.add(peer[0])
                        stack.append(peer[0])
        return count

    @property
    def root_index(self) -> int:
        """第一个根件（没有父件的那一节）—— 单根布局下的"起点"。

        多段断开时只返回**其中一段**的根；要遍历全部根请用 :meth:`root_indices`。
        """
        for index, placed in self.pieces.items():
            if placed.parent is None:
                return index
        raise LayoutError("layout is empty")

    @property
    def is_empty(self) -> bool:
        return not self.pieces

    # ---------------------------------------------------------------- 端口查询

    def world_port(self, index: int, port_id: str) -> Pose:
        """某件某端口的**世界**位姿（外指向）。"""
        placed = self.piece(index)
        local = self.definition(index).port(port_id).pose
        return placed.pose.compose(local)

    def world_inbound(self, index: int, port_id: str) -> Pose:
        """列车从该端口**进入**该件时的世界行进帧。"""
        return self.world_port(index, port_id).flipped()

    def is_port_connected(self, index: int, port_id: str) -> bool:
        return (index, port_id) in self.connections

    def peer_of(self, index: int, port_id: str) -> PortKey | None:
        """与某端口相连的对面端口；未连接返回 ``None``。"""
        return self.connections.get((index, port_id))

    def free_ports(self) -> list[PortKey]:
        """所有还没有接上东西的端口。放置时的候选吸附点。"""
        out: list[PortKey] = []
        for index in self.pieces:
            for port_id in self.definition(index).port_ids:
                if not self.is_port_connected(index, port_id):
                    out.append((index, port_id))
        return out

    def all_ports(self) -> list[PortKey]:
        return [
            (index, port_id)
            for index in self.pieces
            for port_id in self.definition(index).port_ids
        ]

    # ---------------------------------------------------------------- 放置

    def add_root(self, def_id: str, pose: Pose | None = None) -> int:
        """放入第一节（整张图的起点）。"""
        if self.pieces:
            raise LayoutError(
                "layout already has pieces; use add_root only for the first one"
            )
        if def_id not in self.catalog:
            raise LayoutError(f"unknown piece {def_id!r}")
        index = self._take_index()
        self.pieces[index] = PlacedPiece(
            index=index, def_id=def_id, pose=pose or Pose.origin(), parent=None
        )
        return index

    def attach(self, def_id: str, my_port: str, target: PortKey) -> int:
        """把 ``def_id`` 的 ``my_port`` 接到 ``target`` 端口上，返回新件下标。"""
        target_index, target_port = target
        if target_index not in self.pieces:
            raise LayoutError(f"target piece {target_index} does not exist")
        if target_port not in self.definition(target_index).port_ids:
            raise LayoutError(
                f"piece {target_index} ({self.piece(target_index).def_id}) "
                f"has no port {target_port!r}"
            )
        if self.is_port_connected(target_index, target_port):
            raise LayoutError(
                f"port {target_port!r} of piece {target_index} is already connected"
            )
        if def_id not in self.catalog:
            raise LayoutError(f"unknown piece {def_id!r}")

        new_def = self.catalog[def_id]
        if my_port not in new_def.ports:
            raise LayoutError(f"piece {def_id!r} has no port {my_port!r}")

        world_p = self.world_port(target_index, target_port)
        local_q = new_def.port(my_port).pose
        pose = world_p.flipped().compose(local_q.inverse())

        index = self._take_index()
        self.pieces[index] = PlacedPiece(
            index=index,
            def_id=def_id,
            pose=pose,
            parent=(target_index, target_port, my_port),
        )
        self.connections[(target_index, target_port)] = (index, my_port)
        self.connections[(index, my_port)] = (target_index, target_port)
        return index

    def join_gap(self, a: PortKey, b: PortKey) -> tuple[float, float]:
        """把 ``a``、``b`` 接起来时的**缺口**，返回 ``(位置差, 航向差)``。

        两个端口能接上的条件是：接点重合、且**外指向相反**（否则轨道会在接缝处
        折回）。理想值为 ``(0.0, 0.0)``。

        这是编辑器「合拢闭环」时给用户看的那个数字 —— 真正能闭合的接缝误差在
        1e-14 m 量级，接不上的则是米级，两者相差 14 个数量级，因此
        ``connect(verify=True)`` 用的阈值可以卡得很死而不会误判。
        """
        pose_a = self.world_port(*a)
        pose_b = self.world_port(*b)
        return _pose_gap(pose_a, pose_b)

    def connect(self, a: PortKey, b: PortKey, verify: bool = True,
                pos_tol: float = 1e-6, ang_tol: float = 1e-9) -> None:
        """声明两个已有端口相连。

        ``verify=True``（默认）时先检查两端口 **位姿是否真的对得上**，对不上就抛
        :class:`LayoutError` 并在消息里给出缺口大小 —— 这是编辑器「合拢闭环」的
        安全网，避免用户以为接上了其实差 3 cm。

        注意：本方法只建立拓扑连接、**不改变任何件的位姿**（位姿一律由父子链推导）。
        若要靠吸附来求位姿，用 :meth:`attach`。
        """
        for key in (a, b):
            if key[0] not in self.pieces:
                raise LayoutError(f"piece {key[0]} does not exist")
            if key[1] not in self.definition(key[0]).port_ids:
                raise LayoutError(f"piece {key[0]} has no port {key[1]!r}")
        if a == b:
            raise LayoutError("cannot connect a port to itself")
        for key in (a, b):
            if self.is_port_connected(*key):
                raise LayoutError(
                    f"port {key[1]!r} of piece {key[0]} is already connected"
                )

        if verify:
            distance, heading_gap = self.join_gap(a, b)
            if distance > pos_tol or heading_gap > ang_tol:
                raise LayoutError(
                    f"ports {a} and {b} do not line up: gap {distance:.6f} m, "
                    f"heading off by {math.degrees(heading_gap):.6f} deg. "
                    "(this is the gap the editor should draw in red)"
                )

        self.connections[a] = b
        self.connections[b] = a

    # ---------------------------------------------------------------- 自动合拢

    def free_port_pairs(self) -> Iterator[tuple[PortKey, PortKey]]:
        """所有能把开链合拢的端口对：两个都还空着，且**不在同一节**上。

        同一节自己的两个端口不算 —— 把它们接起来只会让列车到端头就折返，
        而且几何上必然对不上（``connect`` 会拒绝），报出来反而误导。
        """
        free = self.free_ports()
        for position, a in enumerate(free):
            for b in free[position + 1:]:
                if a[0] != b[0]:
                    yield a, b

    def closest_free_pair(
        self,
    ) -> tuple[tuple[PortKey, PortKey], tuple[float, float]] | None:
        """缺口最小的那一对可合拢端口，返回 ``((a, b), (位置差, 航向差))``。

        一对都没有（空闲端口不足两个，或只剩同一节的两个端口）时返回 ``None``。
        """
        best: tuple[PortKey, PortKey] | None = None
        best_gap: tuple[float, float] | None = None
        for a, b in self.free_port_pairs():
            gap = self.join_gap(a, b)
            if best_gap is None or gap < best_gap:
                best, best_gap = (a, b), gap
        return None if best is None else (best, best_gap)

    def _branch_depth(self) -> dict[int, int]:
        """每个件到根件的距离（以件数计）。

        父子链按构造是一棵树（``attach`` 不许闭环，``rebuild_poses`` 见到重复
        子件就抛），所以"走到重复件"本来不可能发生；这里还是显式跳过已记过深度
        的子件 —— 花代价很低的是这一行，代价很高的是万一哪天不变量被弄丢时
        无限排队。
        """
        if not self.pieces:
            return {}
        children = self._child_map()
        depth: dict[int, int] = {}
        queue: deque[int] = deque()
        for root in self.root_indices():
            depth[root] = 0
            queue.append(root)
        while queue:
            current = queue.popleft()
            for child in children.get(current, ()):
                if child in depth:
                    continue
                depth[child] = depth[current] + 1
                queue.append(child)
        return depth

    def growing_tip(self) -> PortKey | None:
        """该从哪个端口继续往下接（"还没长完的那一端"）；没得接则 ``None``。

        规则：取**离根件最远**的那个空闲端口 —— 一条链上就是最后接上的那一节。
        场地里只有一节时它的 ``a`` / ``b`` 并列，这时用 route 的**出端**打破平局：
        应该顺着 route 的正方向长，而不是往回长。
        """
        free = self.free_ports()
        if not free:
            return None
        depth = self._branch_depth()
        exits = {
            (placed.index, route.to_port)
            for placed in self.pieces.values()
            for route in self.definition(placed.index).routes
        }
        return max(free, key=lambda key: (depth.get(key[0], 0), key in exits))

    def grow_to_close(self, piece_ids: str | Sequence[str], *,
                      limit: int = AUTO_CLOSE_LIMIT,
                      pos_tol: float = 1e-6,
                      ang_tol: float = 1e-9) -> int:
        """从开链末端一节节接下去，直到首尾能合拢；返回补了几节。

        **要么成功、要么原样不动** —— 这是"一键"的前提：按一下要么真的合上，要么
        什么都没发生并把原因说清楚，绝不能留下半圈多余的废轨道。实现方式是在一份
        **副本**上推演，成功了才把状态搬回来，于是回滚是免费的。

        为什么需要它：``attach`` 是逐节点吸附的，接缝能不能对齐完全取决于用户手上
        那一节对不对，而"还差几节"这件事用户根本没法心算 —— 22.5° 的弯轨要
        **16 节**才绕满一圈，接 8 节正好停在直径的另一头（差 80 m）。所以让机器去数。

        ``piece_ids`` 可以是**一串候选**件。规则是"**先用手上这一件，合不上再替你换**"：

        1. 先用第一件（= 用户当前选中的那一件）单独试一次。它合得上就用它 ——
           从前就有的、"补当前件"这个用户已经熟悉的行为因此一字不变；
        2. 只有它合不上时，才在整串候选里一节一节地挑（见
           :meth:`pick_closing_piece`）。

        第 2 步是"没有合适长度的铁轨对接"那个抱怨的解：缺口要的是**几何**，而用户
        手上恰好拿着不合适的那一节是常态（差一节 22.5° 的弯、手里却是 40 m 直轨）。
        这时该由机器换一件，而不是把"你自己去找一节对的轨道"丢回给用户。
        """
        if not self.pieces:
            raise LayoutError("场地还是空的 —— 先放下一节轨道")

        candidates = [piece_ids] if isinstance(piece_ids, str) else list(piece_ids)
        pieces: list[PieceDef] = []
        for candidate in candidates:
            if candidate not in self.catalog:
                raise LayoutError(f"unknown piece {candidate!r}")
            piece = self.catalog[candidate]
            if not piece.routes:
                raise LayoutError(f"{piece.name} 没有通路，接不出环")
            pieces.append(piece)
        if not pieces:
            raise LayoutError("没有可以用来补线的轨道件")

        try:
            return self._grow_to_close_with(pieces[:1], limit=limit,
                                            pos_tol=pos_tol, ang_tol=ang_tol)
        except LayoutError:
            if len(pieces) == 1:
                raise
            # 手上这一件接不通 —— 换候选里的别的试。再失败就把这一次的错误报出去。
            return self._grow_to_close_with(pieces, limit=limit,
                                            pos_tol=pos_tol, ang_tol=ang_tol)

    def _grow_to_close_with(self, pieces: Sequence[PieceDef], *,
                            limit: int, pos_tol: float,
                            ang_tol: float) -> int:
        """在副本上推演，直到首尾合拢；成功才把状态搬回来（失败时原样不动）。"""
        work = Layout.from_dict(self.to_dict(), self.catalog)
        added = 0
        while True:
            pair = work.closest_free_pair()
            if pair is not None:
                (a, b), (distance, heading_gap) = pair
                if distance <= pos_tol and heading_gap <= ang_tol:
                    work.connect(a, b, pos_tol=pos_tol, ang_tol=ang_tol)
                    self._adopt(work)
                    return added
            tip = work.growing_tip()
            if tip is None:
                raise LayoutError("没有空闲端口可以继续接了")
            if added >= limit:
                gap = "？" if pair is None else f"{pair[1][0]:.3f} m"
                raise LayoutError(
                    f"接了 {added} 节还是合不上（缺口 {gap}）—— "
                    "件库里没有能接上这一段弧度的组合"
                )
            piece = work.pick_closing_piece(tip, pieces)
            work.attach(piece.id, piece.routes[0].from_port, tip)
            added += 1

    def pick_closing_piece(self, tip: PortKey,
                           pieces: Sequence[PieceDef]) -> PieceDef:
        """在候选件里挑**接上去之后最接近合拢**的那一节（一步前瞻）。

        只看"接上之后新露出来的那个端口"离其余空闲端口有多近，所以代价是
        ``候选数 × 空闲端口数``；把每个候选真的接上去试一遍要贵一个量级
        （``候选数 × 空闲端口数²``），而这是按一下键就要跑完的路径。

        位姿是**算**出来的、不是试出来的：:meth:`_far_port_pose` 与 :meth:`attach`
        共用同一条摆放公式，所以它给出的位置与真接上去之后逐位相同。

        打分把位置差与航向差合成一个标量（权重见 ``_HEADING_AS_METERS``）。它是
        一步前瞻，只保证"差一两节就能合上"这类常见情形挑得对 —— 真正长距离的
        补线仍然靠前面"先用手上这一件"那一步（用户选对了弧度时它一击即中）。
        """
        tip_pose = self.world_port(*tip)
        others = [self.world_port(*key) for key in self.free_ports() if key != tip]
        best = pieces[0]
        best_score: float | None = None
        for piece in pieces:
            far = self._far_port_pose(tip_pose, piece)
            nearest: float | None = None
            for other in others:
                distance, heading_gap = _pose_gap(far, other)
                score = distance + _HEADING_AS_METERS * heading_gap
                if nearest is None or score < nearest:
                    nearest = score
            if nearest is not None and (best_score is None or nearest < best_score):
                best, best_score = piece, nearest
        return best

    def _far_port_pose(self, tip_pose: Pose, piece: PieceDef) -> Pose:
        """假想在 ``tip_pose`` 上接一节 ``piece``，新露出来的那个端口的世界位姿。

        摆放公式与 :meth:`attach` 逐字相同（``world.flipped() ∘ local⁻¹``），
        区别只是这里没有真的往布局里放东西 —— 所以可以拿它当"如果接上会怎样"来
        打分，而不必为每个候选复制一遍布局。
        """
        route = piece.routes[0]
        placement = tip_pose.flipped().compose(
            piece.port(route.from_port).pose.inverse()
        )
        return placement.compose(piece.port(route.to_port).pose)

    def _adopt(self, other: "Layout") -> None:
        """把另一个布局的状态搬进自己（``grow_to_close`` 推演成功后用）。"""
        self.pieces = other.pieces
        self.connections = other.connections
        self.switches = other.switches
        self._next_index = other._next_index

    def parent_ports(self) -> set[PortKey]:
        """父子链占用的端口（挂接关系，不是合拢缝）。"""
        out: set[PortKey] = set()
        for placed in self.pieces.values():
            if placed.parent is None:
                continue
            parent_index, parent_port, my_port = placed.parent
            out.add((parent_index, parent_port))
            out.add((placed.index, my_port))
        return out

    def closing_seam_ports(self) -> list[PortKey]:
        """``connect()`` 建立的那几道**合拢缝**的端点（按端口序，去掉父子链）。

        为什么单独要这个：``connections`` 里混着两种边 —— 父子链（挂接）和合拢缝
        （把一个开链合拢成环）。**任何环都必须经过至少一道缝**（缝之外全是树边，
        树没有环），所以"环在哪儿"这个问题只要在**缝的端点**里找就够，候选集是
        O(缝数) 而不是 O(端口数)。

        :meth:`core.track.path.detect_closure` 用它来兜住"环 + 支线"那种布局：
        环上的端口可能全接满了，空闲端口只剩支线末端，从支线出发永远走不出闭环。
        """
        parents = self.parent_ports()
        out: set[PortKey] = set()
        for key in self.connections:
            if key not in parents:
                out.add(key)
                out.add(self.connections[key])
        return sorted(out)

    def _take_index(self) -> int:
        index = self._next_index
        self._next_index += 1
        return index

    # ---------------------------------------------------------------- 删除

    def subtree_of(self, index: int) -> list[int]:
        """返回以 ``index`` 为根的子树（含自己）。"""
        if index not in self.pieces:
            raise LayoutError(f"no placed piece with index {index}")
        children = self._child_map()
        order: list[int] = []
        stack = [index]
        while stack:
            current = stack.pop()
            order.append(current)
            stack.extend(children.get(current, ()))
        return order

    def _child_map(self) -> dict[int, list[int]]:
        children: dict[int, list[int]] = {}
        for placed in self.pieces.values():
            if placed.parent is not None:
                children.setdefault(placed.parent[0], []).append(placed.index)
        return children

    def detach(self, index: int) -> list[int]:
        """摘掉 ``index`` 及其下游子树，返回被移除的件下标。"""
        removed = self.subtree_of(index)
        for piece_index in removed:
            for port_id in self.definition(piece_index).port_ids:
                key = (piece_index, port_id)
                peer = self.connections.pop(key, None)
                if peer is not None:
                    self.connections.pop(peer, None)
            self.switches.pop(piece_index, None)
        for piece_index in removed:
            self.pieces.pop(piece_index, None)
        return removed

    def remove_piece(self, index: int) -> list[int]:
        """只摘掉 ``index`` **这一节**；其余轨道原样留下、**原地不动**。

        和 :meth:`detach`（连下游子树一起删）的区别正是用户按 X 时想要的：把鼠标
        指着的这一节拿掉，后面铺的轨道**既不能跟着消失，也不能被拽得挪动**。

        做法是**冻结**而不是接骨：删掉这一节后，它下游的那一段失去了父件，于是把
        这段最前面的那一节（本节唯一的子件，如果有）改成**新的根件** —— 它的世界
        位姿保持不动（``pose`` 字段本来就存着它当前的世界位姿，清掉 ``parent``
        即可，什么都不用重算）。场上于是出现一段真实的缺口，后半截原地不动，用户
        可以再放一节补上、或按 C 一键闭合。

        四种情形：

        * **末端**（没有子件）：直接拿掉，父件的那个端口重新空出来；
        * **中段**（恰好一个子件）：子件原地冻结成新根，前后形成真实缺口；
        * **根件**（恰好一个子件）：子件扶正成新根，整段原地下沉；
        * **岔口**（挂着两条以上支线）：冻结分不清主次，**拒绝** ——
          让用户先把支线删掉，而不是替他把哪条支线留下。

        返回只含 ``index`` 一个元素的列表（沿用 :meth:`detach` 的返回值形状，
        调用方的提示文案因此不用改分支）。
        """
        if index not in self.pieces:
            raise LayoutError(f"no placed piece with index {index}")

        children = self._child_map().get(index, [])
        if len(children) > 1:
            raise LayoutError(
                f"piece {index} has {len(children)} branches attached; "
                "remove the branches first"
            )
        child_index = children[0] if children else None

        self._unlink(index)
        self.pieces.pop(index)
        if child_index is not None:
            child = self.pieces[child_index]
            # 原地冻结：清掉父指针，位姿保持当前世界位姿不动。
            self.pieces[child_index] = PlacedPiece(
                index=child_index, def_id=child.def_id, pose=child.pose,
                parent=None,
            )
        self.rebuild_poses()
        return [index]

    def _unlink(self, index: int) -> None:
        """清掉某件身上所有的连接与道岔状态（不动别的件）。"""
        for port_id in self.definition(index).port_ids:
            key = (index, port_id)
            peer = self.connections.pop(key, None)
            if peer is not None:
                self.connections.pop(peer, None)
        self.switches.pop(index, None)

    def clear(self) -> None:
        self.pieces.clear()
        self.connections.clear()
        self.switches.clear()
        self._next_index = 0

    # ---------------------------------------------------------------- 道岔

    def set_switch(self, index: int, route_index: int) -> None:
        piece_def = self.definition(index)
        if not piece_def.is_switch:
            raise LayoutError(f"piece {index} ({piece_def.id}) is not a switch")
        if route_index not in piece_def.switch_route_indices:
            raise LayoutError(
                f"route {route_index} is not a valid switch position for "
                f"{piece_def.id} (expected one of {piece_def.switch_route_indices})"
            )
        self.switches[index] = route_index

    def switch_of(self, index: int) -> int | None:
        piece_def = self.definition(index)
        if not piece_def.is_switch:
            return None
        if index in self.switches:
            return self.switches[index]
        return piece_def.switch_route_indices[0]

    def toggle_switch(self, index: int) -> int:
        indices = self.definition(index).switch_route_indices
        current = self.switch_of(index)
        position = indices.index(current) if current in indices else 0
        new_route = indices[(position + 1) % len(indices)]
        self.set_switch(index, new_route)
        return new_route

    # ---------------------------------------------------------- 位姿重建 / 存档

    def rebuild_poses(self) -> None:
        """按父子链从**每个根件**重新推导所有位姿（读档后的确定性保证）。

        支持多段独立轨道：每一段从它自己的根件出发（根件存着绝对位姿，子件位姿
        由父件推导），互不干扰 —— 这正是"删除中间一节时，后半截原地冻结"的底层
        支撑：被删节的下游根件位姿不变，它下面挂的一整段也就原地不动。
        """
        if not self.pieces:
            return
        children = self._child_map()
        poses: dict[int, Pose] = {}
        for root in self.root_indices():
            poses[root] = self.pieces[root].pose
            queue = [root]
            while queue:
                current = queue.pop()
                for child_index in children.get(current, ()):
                    placed = self.pieces[child_index]
                    assert placed.parent is not None
                    parent_index, parent_port, my_port = placed.parent
                    if child_index in poses:
                        raise LayoutError(f"piece {child_index} is reachable twice (cycle)")
                    world_p = poses[parent_index].compose(
                        self.definition(parent_index).port(parent_port).pose
                    )
                    local_q = self.definition(child_index).port(my_port).pose
                    poses[child_index] = world_p.flipped().compose(local_q.inverse())
                    queue.append(child_index)

        if len(poses) != len(self.pieces):
            missing = sorted(set(self.pieces) - set(poses))
            raise LayoutError(f"pieces not connected to any root: {missing}")

        for index, pose in poses.items():
            self.pieces[index] = PlacedPiece(
                index=index,
                def_id=self.pieces[index].def_id,
                pose=pose,
                parent=self.pieces[index].parent,
            )

    def to_dict(self) -> dict:
        """序列化为只含**拓扑**的字典（位姿不入档，读档时由父子链重建）。

        存档里有两类连接：

        * **父子链**（``pieces``）—— 构成一片森林（每段独立轨道一棵树），非根件
          的位姿全部由它推导；**根件**（``roots``）各自存着绝对位姿 —— 删除中间
          一节时后半截原地冻结，靠的就是"每个根件都记下自己的绝对位姿"。
        * **合拢接缝**（``seams``）—— ``connect()`` 建立的额外连接，用来把一个
          开链合拢成闭环（例如 8 字形在切点处的那道缝）。它们不影响位姿，只影响
          拓扑（走线能不能绕回来），因此必须单独存。
        """
        if not self.pieces:
            return {"schema": 2, "roots": [], "pieces": [], "seams": [], "switches": {}}

        records = []
        parent_pairs: set[frozenset] = set()
        for index in sorted(self.pieces):
            placed = self.pieces[index]
            if placed.parent is None:
                continue
            parent_index, parent_port, my_port = placed.parent
            records.append(
                {
                    "index": index,
                    "def": placed.def_id,
                    "parent": parent_index,
                    "parent_port": parent_port,
                    "my_port": my_port,
                }
            )
            parent_pairs.add(
                frozenset({(index, my_port), (parent_index, parent_port)})
            )

        seen: set[frozenset] = set()
        seams: list[list] = []
        for key, peer in self.connections.items():
            pair = frozenset({key, peer})
            if pair in parent_pairs or pair in seen:
                continue
            seen.add(pair)
            (a_index, a_port), (b_index, b_port) = sorted(pair)
            seams.append([a_index, a_port, b_index, b_port])

        roots = []
        for index in sorted(self.pieces):
            placed = self.pieces[index]
            if placed.parent is None:
                roots.append({
                    "index": index,
                    "def": placed.def_id,
                    "pose": {
                        "x": placed.pose.x,
                        "y": placed.pose.y,
                        "z": placed.pose.z,
                        "heading_rad": placed.pose.heading,
                    },
                })

        return {
            "schema": 2,
            "roots": roots,
            "pieces": records,
            "seams": seams,
            "switches": {str(k): v for k, v in sorted(self.switches.items())},
            "next_index": self._next_index,
        }

    @classmethod
    def from_dict(cls, payload: Mapping, catalog: Catalog) -> "Layout":
        layout = cls(catalog=catalog)
        roots = payload.get("roots")
        if roots is None:
            # schema 1 旧存档：单个 ``root``，向上兼容。
            root = payload.get("root")
            roots = [root] if root is not None else []
        for root in roots:
            layout.pieces[root["index"]] = PlacedPiece(
                index=root["index"],
                def_id=root["def"],
                pose=Pose(
                    root["pose"]["x"],
                    root["pose"]["y"],
                    root["pose"]["z"],
                    root["pose"]["heading_rad"],
                ),
                parent=None,
            )
        # 父件下标一定小于子件下标才有意义，但为了稳妥这里按拓扑序反复扫。
        pending = list(payload.get("pieces", []))
        while pending:
            progressed = False
            remaining = []
            for record in pending:
                if record["parent"] not in layout.pieces:
                    remaining.append(record)
                    continue
                layout.pieces[record["index"]] = PlacedPiece(
                    index=record["index"],
                    def_id=record["def"],
                    pose=Pose.origin(),  # 稍后由 rebuild_poses 推导
                    parent=(record["parent"], record["parent_port"], record["my_port"]),
                )
                progressed = True
            if not progressed:
                orphans = [r["index"] for r in remaining]
                raise LayoutError(f"orphan pieces in save data: {orphans}")
            pending = remaining

        for key, value in payload.get("switches", {}).items():
            layout.switches[int(key)] = int(value)
        layout._next_index = int(
            payload.get("next_index", max(layout.pieces, default=-1) + 1)
        )
        layout.rebuild_poses()
        layout._reconnect()

        # 合拢接缝：connect() 建立的额外连接，不影响位姿，只补拓扑。
        for a_index, a_port, b_index, b_port in payload.get("seams", ()):
            layout.connections[(a_index, a_port)] = (b_index, b_port)
            layout.connections[(b_index, b_port)] = (a_index, a_port)
        return layout

    def _reconnect(self) -> None:
        """由父子关系补出双向连接表（读档后调用）。"""
        self.connections.clear()
        for placed in self.pieces.values():
            if placed.parent is None:
                continue
            parent_index, parent_port, my_port = placed.parent
            self.connections[(parent_index, parent_port)] = (placed.index, my_port)
            self.connections[(placed.index, my_port)] = (parent_index, parent_port)

    # ------------------------------------------------------------ 文件读写

    def save_json(self, path: str | Path) -> Path:
        """写存档（自动建目录）。返回真正写入的路径。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load_json(cls, path: str | Path, catalog: Catalog) -> "Layout":
        """读存档。存档里只有拓扑，位姿由 :meth:`rebuild_poses` 重新推导。"""
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload, catalog)

    # ---------------------------------------------------------------- 统计

    def total_length(self) -> float:
        """轨道中心线总长（只算 route，不计重复连接）。"""
        total = 0.0
        for index in self.pieces:
            for route in self.definition(index).routes:
                total += route.length
        return total

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """世界坐标包围盒 ``((min_x, min_z), (max_x, max_z))``，供相机取景。"""
        if not self.pieces:
            return None
        xs: list[float] = []
        zs: list[float] = []
        for index in self.pieces:
            for port_id in self.definition(index).port_ids:
                pose = self.world_port(index, port_id)
                xs.append(pose.x)
                zs.append(pose.z)
            pose = self.piece(index).pose
            xs.append(pose.x)
            zs.append(pose.z)
        return ((min(xs), min(zs)), (max(xs), max(zs)))

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"Layout(pieces={len(self.pieces)}, "
            f"connections={len(self.connections) // 2}, "
            f"free_ports={len(self.free_ports())})"
        )
