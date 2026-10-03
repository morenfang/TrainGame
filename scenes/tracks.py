"""可直接开车的闭环轨道：场景的"路"。

这一层只做一件事 —— 拼出**精确闭合**、列车马上能跑的环。布景在
:mod:`scenes.presets` 里另说，两者互不依赖。

椭圆为什么这样摆
------------------------------------------------
绕一个凸环走一圈，总转角是 360°，而且**两个端头转的是同一个方向**（都是左转
180°）。真正决定闭不闭合的其实是两条直股 —— 它们必须**等长**：

    起点 (0, 0) 朝 +X
      左 180° ──────────────→ (0, 2·2R) 朝 -X     端头 1
      直股 L（沿 -X）────────→ (-L, 2·2R) 朝 -X
      左 180° ──────────────→ (-L, 0) 朝 +X        端头 2
      直股 L（沿 +X）────────→ (0, 0) 朝 +X        回到起点 ✔

两条直股等长时残差是 1e-14 m 量级（``tests/test_scenes.py`` 直接量闭环误差）。
**等长不是约定，是条件**：所以本模块把它做成构造时的断言 —— 直股长度或升降不
配平就当场报错，而不是等到 ``connect()`` 抛一句"ports do not line up"，再去猜
是哪一节拼歪了。

环上带道岔 / 尽头线
------------------------------------------------
道岔的直股就是一根 40 m 直轨（与 ``straight_40`` 长度一致），所以**拿它替掉直股
里的任意一节，椭圆的几何分毫不变**。岔股再往外接几节 + 一个车挡，就是一条站场
尽头线。``add_stub`` 负责这一段，返回新接的件，供场景继续挂月台、房子。

环上带**立交疏解线**（两条左右开弓的岔股之间接一条跨线）
------------------------------------------------
尽头线是"接出去就完事"，疏解线要**接回来**，于是两端的朝向与位置都得正好对上。
配平的办法见 :func:`bypass_run`：那组转角（右 22.5 / 左 90 / 右 90 / 右 180 /
左 22.5）恰好把航向拧回 180°、横向挪出 80 m、纵向挪出 0 m，因此在**另一条**直股
上、与起点相隔偶数节的位置上必有它的落点。两条直股之间就有了"一条从主线底下
钻过去再回来"的立体交叉 —— 不需要菱形交叉件（那种交叉是同一平面上的十字，列车
只能直穿），也不会有任何一处平交。
"""

from __future__ import annotations

import math

from core.geometry import Pose
from core.track.catalog import Catalog
from core.track.layout import Layout

#: 160°/180° 端头的两种拼法：R40 的 90° 两节，或 R80 的 45° 四节。
CAP_R40 = ("curve_r40_l90", 2)
CAP_R80 = ("curve_r80_l45", 4)


class SceneError(ValueError):
    """场景拼不出来（几何配不平、件不存在……）。

    与 :class:`core.track.layout.LayoutError` 区分开：那个说的是"这两节接不上"，
    这个说的是"这套场景的定义本身就不成立"，报错时要指向场景定义。
    """


# --------------------------------------------------------------------------- #
# 基本拼装
# --------------------------------------------------------------------------- #

def _grow(layout: Layout, def_id: str, count: int, tip: int) -> int:
    """从 ``tip`` 的 ``b`` 端往外接 ``count`` 节同型件，返回新的末端件下标。"""
    for _ in range(count):
        tip = layout.attach(def_id, "a", (tip, "b"))
    return tip


def _run_profile(catalog: Catalog, def_ids) -> tuple[float, float]:
    """一串单 route 件的 ``(平面总长, 总升降)``。"""
    planar = 0.0
    rise = 0.0
    for def_id in def_ids:
        route = catalog[def_id].routes[0]
        planar += route.length
        rise += route.rise
    return planar, rise


def _placed(assemble, root_pose: Pose | None, center):
    """摆好之后再整体挪一下，让**包围盒中心**落在 ``center``。

    为什么不手推"根位姿该填多少"：椭圆 / 组的包围盒中心相对根位姿偏多少，取决于
    端头**往哪边拐**，而拐向写在件库的 ``dtheta`` 符号里。推错一个符号就是整体偏
    心几十米 —— 摆在场地里一眼看得见，在代码里却看不出来。所以这里摆两遍：第一遍
    量，第二遍挪。位移是刚体平移，量一次就准。

    ``assemble(pose) -> (布局, 件下标表)``；``center=None`` 表示原地不动。
    """
    start = root_pose if root_pose is not None else Pose.origin()
    layout, indices = assemble(start)
    if center is None:
        return layout, indices
    box = layout.bounds()
    if box is None:
        raise SceneError("空布局没有包围盒，挪不了位置")
    (x0, z0), (x1, z1) = box
    moved = Pose(start.x + center[0] - (x0 + x1) * 0.5, start.y,
                 start.z + center[1] - (z0 + z1) * 0.5)
    return assemble(moved)


def oval(catalog: Catalog, *, cap_piece: str = "curve_r40_l90", cap_pieces: int = 2,
         side_a=("straight_40",) * 4, side_b=None,
         root_pose: Pose | None = None, center=(0.0, 0.0)
         ) -> tuple[Layout, list[int]]:
    """一条椭圆跑道：``端头 + 直股 A + 端头 + 直股 B``。

    返回 ``(布局, 件下标表)``。件下标表按拼装顺序给出**每一节**的下标，所以调用方
    不需要反向去猜"道岔是第几节"（用 :func:`switch_index` 找就行）。

    ``center`` 是包围盒中心要落的地方（缺省场地原点）；给 ``None`` 就按 ``root_pose``
    原样摆着不动。
    """
    side_a = tuple(side_a)
    side_b = tuple(side_a if side_b is None else side_b)
    route = catalog[cap_piece].routes[0]
    if route.length <= 0.0:
        raise SceneError(f"端头件 {cap_piece!r} 没有可走行的 route")
    if abs(cap_pieces * route.dtheta) < math.pi - 1e-9:
        raise SceneError(
            f"端头拼不满 180°：{cap_pieces} × {cap_piece} = "
            f"{math.degrees(cap_pieces * route.dtheta):.3f}°"
        )

    planar_a, rise_a = _run_profile(catalog, side_a)
    planar_b, rise_b = _run_profile(catalog, side_b)
    if abs(planar_a - planar_b) > 1e-9:
        raise SceneError(
            f"两条直股必须等长，否则椭圆合不拢：{planar_a:.3f} m vs {planar_b:.3f} m"
        )
    for name, rise in (("A", rise_a), ("B", rise_b)):
        if abs(rise) > 1e-9:
            raise SceneError(
                f"直股 {name} 的升降没有配平（{rise:+.3f} m）："
                "上坡引桥必须与下坡引桥成对出现，合拢缝的两端才会等高"
            )

    def assemble(pose: Pose) -> tuple[Layout, list[int]]:
        layout = Layout(catalog=catalog)
        indices: list[int] = []
        tip = layout.add_root(cap_piece, pose)
        indices.append(tip)
        for _ in range(cap_pieces - 1):
            tip = _grow(layout, cap_piece, 1, tip)
            indices.append(tip)
        for def_id in side_a:
            tip = _grow(layout, def_id, 1, tip)
            indices.append(tip)
        root = indices[0]
        for _ in range(cap_pieces):
            tip = _grow(layout, cap_piece, 1, tip)
            indices.append(tip)
        for def_id in side_b:
            tip = _grow(layout, def_id, 1, tip)
            indices.append(tip)
        layout.connect((tip, "b"), (root, "a"))
        return layout, indices

    return _placed(assemble, root_pose, center)


def circle(catalog: Catalog, *, piece: str = "curve_r40_l45", count: int = 8,
           root_pose: Pose | None = None, center=(0.0, 0.0)
           ) -> tuple[Layout, list[int]]:
    """一个整圆：``count`` 节同向弯轨，转角加起来正好一圈。

    ``center`` 的语义与 :func:`oval` 相同：圆心落在哪里。
    """
    dtheta = catalog[piece].routes[0].dtheta
    if abs(abs(dtheta) * count - math.tau) > 1e-9:
        raise SceneError(
            f"拼不出一圈：{count} × {piece} = "
            f"{math.degrees(abs(dtheta) * count):.3f}°，应当是 360°"
        )

    def assemble(pose: Pose) -> tuple[Layout, list[int]]:
        layout = Layout(catalog=catalog)
        root = layout.add_root(piece, pose)
        indices = [root]
        tip = root
        for _ in range(count - 1):
            tip = _grow(layout, piece, 1, tip)
            indices.append(tip)
        layout.connect((tip, "b"), (root, "a"))
        return layout, indices

    return _placed(assemble, root_pose, center)


def switch_index(layout: Layout, indices) -> int:
    """件下标表里的第一处道岔；没有就抛 :class:`SceneError`。"""
    for index in indices:
        if layout.definition(index).is_switch:
            return index
    raise SceneError("这套轨道里没有道岔")


def add_stub(layout: Layout, turnout: int, run=("straight_40", "straight_20"),
             cap: str = "buffer_stop") -> list[int]:
    """从道岔的岔股接一条**尽头线**（车挡收尾），返回新接上的件下标表。

    尽头线不需要闭合 —— 这正是它好用之处：站场里"多出来的一股道"不必去凑
    那套严苛的等长条件，接几节、撞上车挡就完事。道岔的直股仍旧是椭圆的直股，
    所以主线照样能跑（道岔默认档位就是直股）。
    """
    run = tuple(run)
    if not run:
        raise SceneError("尽头线至少要有一节轨道")
    tip = layout.attach(run[0], "a", (turnout, "c"))
    indices = [tip]
    for def_id in run[1:]:
        tip = layout.attach(def_id, "a", (tip, "b"))
        indices.append(tip)
    indices.append(layout.attach(cap, "a", (tip, "b")))
    # 显式扳到直股：主线可跑这件事不该依赖"默认档位恰好是第一条 route"
    layout.set_switch(turnout, layout.definition(turnout).switch_route_indices[0])
    return indices


def switch_indices(layout: Layout, indices) -> list[int]:
    """件下标表里**所有**道岔，按拼装顺序。"""
    return [index for index in indices if layout.definition(index).is_switch]


#: 立交疏解线（"飞线"）的件序列。见 :func:`bypass_run` 里为什么只能是这一组。
def bypass_run(straights: int = 0) -> tuple[str, ...]:
    """跨线疏解线的件序列；``straights`` = 中间拉长几节 40 m。

    调用方只给一个整数，转弯那几节是**配平好**的、不该由调用方挑 —— 改一个转角
    就合不拢。配平的理由（也是 :func:`add_bypass` 报错时要说的那句话）：

    ``turnout_l_40`` 的岔股是一节 R40 的 22.5° 弯轨，所以疏解线两端各扛着一个
    22.5° 的"斜接"，横向各偏出 ``R·(1 - cos22.5°) = 3.045 m``。要接上另一处岔股，
    中间的转角之和必须是 180°（否则航向回不到与岔股对齐），而且横向位移必须正好
    抵掉/补上两处道岔的间距。唯一能做到这件的组合是::

        右 22.5°（把岔股的斜接拧回正） → [直轨 × straights] →
        左 90° → 右 90° → 右 180° → 左 22.5°（拧回岔股的斜接）

    转角合计 ``-22.5 + 90 - 90 - 180 + 22.5 = -180°`` ✔；横向位移合计
    ``40 + 40 = 80 m``（头尾两个 22.5° 的斜接互相抵消）✔；纵向位移合计 0 ✔。
    于是它 **只**在两处道岔相隔 ``40 × (straights + 2)`` 米时合得上 —— 差 40 m
    就是米级的缺口，``connect()`` 会当场拒收。
    """
    if straights < 0:
        raise SceneError("疏解线不能是负长度")
    return (
        ("curve_r40_r22_5",)                       # 岔股的 22.5° 拧回正
        + ("straight_40",) * straights             # 中间拉长（每节 40 m）
        + ("curve_r40_l45", "curve_r40_l45")       # 左 90°：拐向主线
        + ("curve_r40_r45", "curve_r40_r45")       # 右 90°：转向、越过主线
        + ("curve_r40_r45",) * 4                   # 右 180°：兜回来
        + ("curve_r40_l22_5",)                     # 左 22.5°：对准另一处岔股
    )


def add_bypass(layout: Layout, start: int, join: int,
               straights: int = 0) -> list[int]:
    """接一条**立交疏解线**：``start`` 的岔股 → 跨过/钻过主线 → ``join`` 的岔股。

    与 :func:`add_stub` 的分工：尽头线接几节就完事（末端撞上车挡），疏解线**要
    回到主线**，于是两端的位置与朝向都得正好对上 —— 这正是立交比尽头线难的地方。
    能对上的条件不在这里现算，而是被 :func:`bypass_run` 那组转角锁死；这里只负责
    "接上去、量缺口、量不过就报清楚"。两处道岔都扳回直股，主线（那条环线）可跑
    这件事不依赖"默认档位恰好是第一条 route"。

    返回新接上的件下标表（与 :func:`add_stub` 同形，供场景继续挂布景）。
    """
    run = bypass_run(straights)
    tip = layout.attach(run[0], "a", (start, "c"))
    indices = [tip]
    for def_id in run[1:]:
        tip = layout.attach(def_id, "a", (tip, "b"))
        indices.append(tip)

    distance, heading = layout.join_gap((tip, "b"), (join, "c"))
    if distance > 1e-6 or heading > 1e-9:
        raise SceneError(
            f"立交疏解线接不上另一处道岔：缺口 {distance:.3f} m、航向差 "
            f"{math.degrees(heading):.3f}°。疏解线只配得平一种间距 —— 两处道岔"
            f"必须相隔 {40 * (straights + 2)} m（= 40 m × 拉长 {straights} 节 + 2）"
        )
    layout.connect((tip, "b"), (join, "c"))
    for turnout in (start, join):
        route = layout.definition(turnout).switch_route_indices[0]
        layout.set_switch(turnout, route)
    return indices


# --------------------------------------------------------------------------- #
# 场景用到的几条线路
# --------------------------------------------------------------------------- #

#: 一条 160 m 直股（都是 4 × 40 m，方便按需换成引桥 / 桥面）。
STRAIGHT_160 = ("straight_40",) * 4
