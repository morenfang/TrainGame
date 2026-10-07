"""五个开箱就能开跑的场景：山谷环线 / 河谷大桥 / 小镇车站 / 湖畔草原 / 立体交叉。

场景 = **一条闭环轨道** + **一份布景** + 一点建议（场地多大、先上哪列车）。

为什么布景的摆位要"绕着轨道算"
------------------------------------------------
"山压住了环线"、"月台盖在轨道上"这类错误，光看代码是看不出来的 —— 一堆坐标对
不对，只有摆出来才知道。所以这里**不手写坐标之间的相对关系**，而是：

1. 先把轨道拼出来，取它的可行驶路径，按弧长采样成一条"轨道走廊"；
2. 布景只在走廊**以外** ``style.TRACK_CLEARANCE`` 米处生成候选，靠得太近的候选
   直接丢掉（:func:`_only_clear`）；
3. 需要精确位置的（月台、站房）用 :func:`_aside` 从轨道中心线**沿法线**挪出去，
   而"往哪一侧挪"交给"背离环心"这个判据 —— 不写死正负号。

第 3 条是有教训的：路径的**行进方向**由闭环检测决定，不由场景决定，所以"往右侧
挪 5.6 m"这句话在路径反向时会把月台摆到环线里面去。用"背离环心"定向，方向反了
也还是对的。

``tests/test_scenes.py`` 会独立地再查一遍净空（不依赖这里的过滤），并逐条检查
"能否开跑"、"桥是否真的跨在河上"。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from core.geometry import Pose, arc_point
from core.track.catalog import Catalog
from core.track.layout import Layout
from core.track.path import RoutePath, detect_closure
from render import scenery as scenery_mod
from render import style
from scenes import tracks
from scenes.tracks import SceneError

#: 采样轨道中心线的步长（米）。净空检查的量尺。
#:
#: 为什么细到 2 m：走廊是**采样点**，而"离采样点最近"总比"离轨道最近"大一点 ——
#: 采样点之间隔着 2 m 弧长时，凹的一侧的误差上限是 ``(_CORRIDOR_STEP / 2)² / (2h)``，
#: 净距 h = 2 m 时约 0.25 m。5 m 一步的话这个误差会到 1.5 m，也就是"看起来净距
#: 2 m、实际贴着轨道"，而贴着轨道是测不出来的 —— 摆位的错就该在摆位时挡住。
_CORRIDOR_STEP = 2.0

#: 过滤时额外留的余量（米）。把上面那个采样误差吃进去，过滤才是**保守**的：
#: 测试用更细的采样再量一遍时，不会量出比过滤时更小的值。
_CLEARANCE_SLACK = 0.3

#: 默认的场地边长（米）。存档里没记场景预设时用它。
DEFAULT_PLOT = 520.0

#: 每个场景要多大的场地（米）。四张都**量过**：轨道、街景、山脚，最外那一圈都得
#: 落在场地里 —— 山站到场地外面去就成了悬空的一片，比缺一块还难看。
#:
#: 单独列一张表（而不是各场景里写一个字面量）是因为读档时也要知道它，而那会儿
#: 不该把整个场景拼一遍。
PLOT_SIZES = {
    "valley": 800.0,     # 远山铺在 290 × 230 的椭圆上，山脚最远到 ±378
    "gorge": 880.0,      # 峡谷壁沿河铺到 |z| ≈ 407，河自己伸到 340
    "town": 560.0,       # 外围山丘在 145 × 115 的椭圆上（山脚最远 ±205）
    "lake": 560.0,       # 远处的草丘在半径 175 上（湖在环线里面）
    "overpass": 920.0,   # 环线自己就有 640 × 80，远山在 370 × 200 的椭圆上
}

#: 读档时只凭 scenery key 就能知道该上哪列车，不必把整份场景再拼一遍。
#: 必须与下面各 ``build_*`` 里写的 ``train_id`` 一致（``tests/test_scenes.py`` 盯着）。
TRAIN_IDS = {
    "valley": "cr400bf_huangsidai_8",
    "gorge": "cr400bf_huangsidai_8",
    "town": "cr400bf_huangsidai_8",
    "lake": "cr400bf_huangsidai_8",
    "overpass": "cr400bf_huangsidai_16",
}

#: 站台近边离轨道中心线的距离（米）。比 ``TRACK_CLEARANCE`` 多留 0.6 m：
#: 真实站台边缘离轨道中心约 1.75 m，沙盘尺度下取 2.6 m 既贴得住又不侵限。
_PLATFORM_GAP = 2.6


# --------------------------------------------------------------------------- #
# 轨道 → 走廊
# --------------------------------------------------------------------------- #

def closed_path(layout: Layout) -> RoutePath:
    """取布局的可行驶闭环路径；没闭合就报错（场景定义写错了要当场说清楚）。"""
    report, path = detect_closure(layout, allow_open=True)
    if path is None or not report.closed:
        raise SceneError(
            f"场景的轨道没有闭合成环，不能直接开跑：{report.describe()}"
        )
    return path


def corridor(path: RoutePath, step: float = _CORRIDOR_STEP
             ) -> list[tuple[float, float, float]]:
    """轨道走廊：中心线的采样点。布景不许摆进这条走廊。"""
    return [pose.position for pose in path.sample(step)]


def path_centroid(points) -> tuple[float, float]:
    """采样点的形心（环线内部的一个点，用来判断"哪边是外侧"）。"""
    count = float(len(points)) or 1.0
    return (sum(p[0] for p in points) / count, sum(p[2] for p in points) / count)


def _along(path: RoutePath, s: float, lateral: float) -> tuple[float, float, float]:
    """中心线上弧长 ``s`` 处、横向偏移 ``lateral`` 米（正 = 局部 ``+z`` 一侧）。

    横向取 ``forward × up``，与轨道断面 / 列车网格用的是同一个"右手边"。
    """
    pose = path.pose_at(s)
    sin_h, cos_h = math.sin(pose.heading), math.cos(pose.heading)
    return (pose.x - sin_h * lateral, pose.z + cos_h * lateral, pose.heading)


def _aside(path: RoutePath, s: float, distance: float, outward: tuple[float, float]
           ) -> tuple[float, float, float, float]:
    """把点从中心线沿法线挪 ``distance`` 米到**朝着 ``outward`` 的那一侧**。

    返回 ``(x, z, heading, sign)``；``sign`` 是该侧的横向符号（``+1`` / ``-1``），
    调用方拿它推站台的 ``track_side``（站台的轨道永远在挪出去的反面）。
    """
    plus = _along(path, s, distance)
    minus = _along(path, s, -distance)
    toward = ((plus[0] - minus[0]) * outward[0] + (plus[1] - minus[1]) * outward[1])
    if toward >= 0.0:
        return (plus[0], plus[1], plus[2], 1.0)
    return (minus[0], minus[1], minus[2], -1.0)


def _outward_axis(points, x: float, z: float) -> tuple[float, float]:
    """从环心指向 ``(x, z)`` 的单位向量 —— "外侧"的定义。"""
    cx, cz = path_centroid(points)
    dx, dz = x - cx, z - cz
    length = math.hypot(dx, dz)
    if length < 1e-9:
        return (0.0, 1.0)
    return (dx / length, dz / length)


# --------------------------------------------------------------------------- #
# 摆位工具
# --------------------------------------------------------------------------- #

def _axis(start: float, stop: float, step: float) -> list[float]:
    """``start`` 到 ``stop``（含）之间按 ``step`` 均分。"""
    count = max(1, int(round(abs(stop - start) / step)))
    return [start + (stop - start) * i / count for i in range(count + 1)]


def _grid(rng: random.Random, x0: float, x1: float, z0: float, z1: float,
          step: float, jitter: float = 0.45) -> list[tuple[float, float]]:
    """一块矩形里的**抖动网格**。

    比"纯随机撒点"好：纯随机会这儿挤一堆、那儿空一片，而场景要的是"一片草地里
    长着几棵树"这种均匀感。逐行再错开半格，才不会一眼看出是网格。
    """
    points: list[tuple[float, float]] = []
    for row, z in enumerate(_axis(z0, z1, step)):
        stagger = step * 0.5 if row % 2 else 0.0
        for x in _axis(x0, x1, step):
            points.append((
                x + stagger + jitter * (rng.random() * 2.0 - 1.0) * step,
                z + jitter * (rng.random() * 2.0 - 1.0) * step,
            ))
    return points


def _ring(rng: random.Random, cx: float, cz: float, radius: float, count: int,
          spread: float = 0.0) -> list[tuple[float, float]]:
    """一圈均布的点（每个再随机偏一点）。"""
    return [
        (cx + (radius + spread * (rng.random() * 2.0 - 1.0)) * math.cos(math.tau * i / count),
         cz + (radius + spread * (rng.random() * 2.0 - 1.0)) * math.sin(math.tau * i / count))
        for i in range(count)
    ]


def _ellipse_ring(rng: random.Random, cx: float, cz: float, ax: float, az: float,
                  count: int, spread: float = 0.0) -> list[tuple[float, float]]:
    """**椭圆**周长上均布的点（每个再随机偏一点）。

    为什么不是圆：环线是细长的（山谷那条 240 m × 80 m）。围着它画圆，两头会贴着
    轨道、两侧却空出上百米 —— 山就挤在轨道端头、谷底反而空荡荡。按环线的长宽比
    拉成椭圆，才是一圈**等宽**的余地。
    """
    return [
        (cx + (ax + spread * (rng.random() * 2.0 - 1.0)) * math.cos(math.tau * i / count),
         cz + (az + spread * (rng.random() * 2.0 - 1.0)) * math.sin(math.tau * i / count))
        for i in range(count)
    ]


def _plan(point) -> tuple[float, float]:
    """取一个走廊点的平面坐标。

    净空量尺上的点是 ``(x, z, y)``（``walk``），但铺山、铺草时用的是投影过的
    ``(x, z)``（见各 ``build_*`` 里的 ``plane``）。两种都得认，否则一取 ``p[2]``
    就在两元组上炸 —— 而炸的还是场景构建的第一行。
    """
    return (point[0], point[2]) if len(point) >= 3 else (point[0], point[1])


def _only_clear(items, points, margin: float, key=None):
    """只留下占地轮廓离轨道走廊不少于 ``margin`` 米的那些。

    ``key`` 取一个物件的轮廓；缺省假定物件自带 ``footprint()``。

    实际卡的是 ``margin + _CLEARANCE_SLACK``：走廊是采样出来的，采样点之间的凹处
    量不到（见 ``_CORRIDOR_STEP``）。少留这一点余量，就会出现"过滤说合格、更细地
    量一遍却不合格"的布景 —— 那种布景看着没问题，等到出问题已经说不清是谁的错。
    """
    limit = margin + _CLEARANCE_SLACK
    key = key if key is not None else (lambda item: item.footprint())
    kept = []
    for item in items:
        footprint = key(item)
        if all(footprint.distance_to(*_plan(p)) >= limit for p in points):
            kept.append(item)
    return tuple(kept)


def _cull_overlapping_houses(houses, gap: float = 1.5):
    """大房子先占坑，后面的若踩到已有轮廓就丢掉（高矮长宽随机后网格间距不够用）。"""
    ordered = sorted(houses, key=lambda h: -(h.width * h.depth * h.height))
    kept: list = []
    for house in ordered:
        if any(
            house.footprint().distance_to(other.x, other.z) < gap
            or other.footprint().distance_to(house.x, house.z) < gap
            for other in kept
        ):
            continue
        kept.append(house)
    return tuple(kept)


def _far_enough(x: float, z: float, points, margin: float) -> bool:
    """一个点离轨道走廊是否够远（草地这类没有轮廓的东西用它）。"""
    return all(math.hypot(x - px, z - pz) >= margin
               for px, pz in (_plan(p) for p in points))


def _in_water(x: float, z: float, scenery: scenery_mod.Scenery) -> bool:
    """这个点是不是落在水面（河 / 湖）上。房子、树、草皮都得躲开它。"""
    for river in scenery.rivers:
        if river.contains(x, z):
            return True
    for lake in scenery.lakes:
        if lake.contains(x, z):
            return True
    return False


def _near_water(x: float, z: float, scenery: scenery_mod.Scenery,
                margin: float = 2.0) -> bool:
    """这个点是不是离水面太近（含岸边 ``margin`` 米）。"""
    for river in scenery.rivers:
        if river.distance_to(x, z) < river.width * 0.5 + river.bank + margin:
            return True
    for lake in scenery.lakes:
        if lake.distance_to(x, z) < margin:
            return True
    return False


def _dry(items, scenery: scenery_mod.Scenery, margin: float = 3.0):
    """挑掉**压在水面上**的东西（山脚伸进河里、房子立在湖心）。

    ``_only_clear`` 量的是"离轨道多远"，量不了"离水面多远"，所以这一条得单独有。
    判据取物件外接圆上的一圈探针点，逐个问"这儿是不是水面" —— 比"圆心离水多远"
    靠谱：圆心在岸上、半个身子探进水里的房子照样得挑掉。
    """
    kept = []
    for item in items:
        footprint = item.footprint()
        reach = max(footprint.half_x, footprint.half_z) + margin
        probes = [(footprint.x, footprint.z)] + [
            (footprint.x + reach * math.cos(i * math.tau / 8.0),
             footprint.z + reach * math.sin(i * math.tau / 8.0))
            for i in range(8)
        ]
        if not any(_in_water(x, z, scenery) for x, z in probes):
            kept.append(item)
    return tuple(kept)



def _port_polyline(layout: Layout, indices, step: float = _CORRIDOR_STEP
                   ) -> list[tuple[float, float, float]]:
    """一串件的端口连成折线，再按 ``step`` 加密 —— 主环之外那几节轨道的净空量尺。

    主环之外还有轨道的地方（站场的尽头线、岔线）**也必须**参与净空检查，否则
    "房子盖在岔线上"照样查不出来。

    为什么要在端口之间插点：尽头线是 40 m + 20 m 两节直轨，端口之间隔着 40 m。
    房子正好卡在一节中间时，离两端端口各有 20 m，量出来的距离大得离谱 —— 而它
    其实就压在轨道上。插点之后这条量尺才是连续的。
    """
    chain = [
        layout.world_port(index, port_id).position
        for index in indices
        for port_id in layout.definition(index).port_ids
    ]
    dense: list[tuple[float, float, float]] = []
    for index, point in enumerate(chain):
        dense.append(point)
        if index + 1 >= len(chain):
            break
        nxt = chain[index + 1]
        gap = math.hypot(nxt[0] - point[0], nxt[2] - point[2])
        if gap <= step:
            continue
        count = int(gap // step)
        dense.extend(
            (point[0] + (nxt[0] - point[0]) * (k * step / gap),
             point[1] + (nxt[1] - point[1]) * (k * step / gap),
             point[2] + (nxt[2] - point[2]) * (k * step / gap))
            for k in range(1, count + 1)
        )
    return dense


def _route_polyline(layout: Layout, indices, step: float = _CORRIDOR_STEP
                    ) -> list[tuple[float, float, float]]:
    """一串件的**中心线**（沿各自的 route 走，不是端口之间的直线）。

    与 :func:`_port_polyline` 的分工：那个把端口连起来再线性加密，够用于**直件**
    （尽头线就是几节直轨）；这个按 route 的几何逐点取，弯件、坡道件都不会跑偏。

    立交疏解线是弯的，所以必须用这一个 —— 把弯件的端口拉成直线会量出一条穿过
    环线内侧的假线，于是"真有东西压在疏解线上"反而量不出来（这正是净空检查最
    怕的漏网方式：检查器自己错了，比没检查更糟）。
    """
    points: list[tuple[float, float, float]] = []
    for index in indices:
        definition = layout.definition(index)
        for route in definition.routes:
            entry = layout.world_inbound(index, route.from_port)
            count = max(2, int(route.length // step))
            for k in range(count + 1):
                u = route.length * k / count
                dx, dz, du = arc_point(route.length, route.dtheta, u)
                points.append(
                    entry.compose(Pose(dx, u * route.grade, dz, du)).position
                )
    return points


def planar_crossings(points_a, points_b, *, radius: float = 8.0,
                     merge: float | None = None
                     ) -> list[tuple[float, float, float, float]]:
    """两条中心线在**平面上**的交叉点，返回 ``[(x, z, y_a, y_b), ...]``。

    输入点是 ``(x, y, z)`` 顺序（``+Y`` 向上、地面为 XZ 平面，与 ``Pose.position``
    一致）。采样点挨得比 ``radius`` 近就算交叉，挨在一起的一串并成一个点（取最近
    的那一对，合并半径缺省 ``2.5 × radius``）。竖向完全不参与判断 —— 这样"立体
    交叉"才有可量之处：

    * 平面上有交叉 + 竖向差得开 = **立交**（一上一下）；
    * 平面上有交叉 + 竖向差不多 = **相撞**（同平面十字，或者桥没架够高）。

    也就是说：这个函数量的是"哪里需要小心"，够不够高由调用方对着返回值量
    （见 :func:`build_overpass` 与 ``tests/test_scenes.py``）。
    """
    if merge is None:
        merge = radius * 2.5
    pairs = sorted(
        (math.hypot(ax - bx, az - bz), (ax + bx) * 0.5, (az + bz) * 0.5, ay, by)
        for bx, by, bz in points_b
        for ax, ay, az in points_a
        if math.hypot(ax - bx, az - bz) <= radius
    )
    hits: list[tuple[float, float, float, float]] = []
    for _, x, z, ay, by in pairs:
        if all(math.hypot(x - hx, z - hz) > merge for hx, hz, _, _ in hits):
            hits.append((x, z, ay, by))
    return hits


# --------------------------------------------------------------------------- #
# 场景描述
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Scene:
    """一个可直接运行的场景。"""

    key: str
    title: str
    blurb: str
    layout: Layout
    scenery: scenery_mod.Scenery
    plot_size: float = 520.0
    #: 建议先上线的编组（``data/trains.json`` 里的 id）。目录里没有就随便来一列。
    train_id: str | None = None
    #: 参与净空检查的点：主环中心线 + 尽头线端口。测试与自检都用它。
    clearance_points: tuple[tuple[float, float, float], ...] = ()

    def clearance(self) -> tuple[float, str]:
        """整份布景到轨道的最小净距，以及最紧的那一件。"""
        return self.scenery.clearance(list(self.clearance_points))

    def camera_bounds(self):
        """取景框：轨道 + 布景合起来的包围盒（``OrbitCamera.frame`` 直接吃）。

        为什么要并上布景：只看轨道的话相机贴得很近，山全在画面外面 —— 而"山谷
        环线"这个场景一半的意思就在那圈山上。
        """
        boxes = []
        if not self.layout.is_empty:
            box = self.layout.bounds()
            if box is not None:
                boxes.append(box)
        if self.scenery:
            x0, z0, x1, z1 = self.scenery.bounds()
            boxes.append(((x0, z0), (x1, z1)))
        if not boxes:
            return None
        return ((min(b[0][0] for b in boxes), min(b[0][1] for b in boxes)),
                (max(b[1][0] for b in boxes), max(b[1][1] for b in boxes)))



# --------------------------------------------------------------------------- #
# 一：山谷环线
# --------------------------------------------------------------------------- #

def build_valley(catalog: Catalog) -> Scene:
    """群山夹着的一条 R40 椭圆环线，谷底有河、有村子、有草场。

    几何：两个 180° 端头（各两节 R40/90）+ 两条 160 m 直股 = 571.3 m 环长，
    摆得下任何一种编组（最长的一列车约 176 m）。环线的包围盒是 240 m × 80 m
    （直股 160 × 两端各探出 R = 40），正中落在场地原点。
    """
    layout, _ = tracks.oval(
        catalog,
        cap_piece="curve_r40_l90", cap_pieces=2,
        side_a=tracks.STRAIGHT_160,
    )
    path = closed_path(layout)
    walk = corridor(path)
    plane = [(p[0], p[2]) for p in walk]
    rng = random.Random(20240501)

    # ---- 河：横着流过谷底，离最近的一条直股 55 m —— 不跨轨，所以不需要桥
    river = scenery_mod.River(
        points=tuple(
            (x + 12.0 * math.sin(x / 88.0), -95.0 + 10.0 * math.sin(x / 150.0))
            for x in _axis(-300.0, 300.0, 55.0)
        ),
        width=16.0, bank=2.8, seed=7,
    )
    water = scenery_mod.Scenery(rivers=(river,))

    # ---- 山：谷壁两层，铺在**椭圆**上而不是圆上：环线细长，用圆会让山贴住
    # 轨道两头、谷底两侧却空出几百米
    peaks = [
        scenery_mod.Peak(x=x, z=z, radius=radius,
                         height=radius * rng.uniform(0.9, 1.3),
                         seed=rng.randrange(1 << 30),
                         sides=rng.choice((12, 14, 16)))
        for radius, (x, z) in ((rng.uniform(36.0, 58.0), spot)
                               for spot in _ellipse_ring(rng, 0.0, 0.0, 250.0, 170.0,
                                                         18, spread=20.0))
    ]
    peaks += [
        scenery_mod.Peak(x=x, z=z, radius=radius,
                         height=radius * rng.uniform(1.1, 1.6),
                         seed=rng.randrange(1 << 30), sides=12)
        for radius, (x, z) in ((rng.uniform(44.0, 66.0), spot)
                               for spot in _ellipse_ring(rng, 0.0, 0.0, 290.0, 230.0,
                                                         14, spread=22.0))
    ]
    peaks = list(_only_clear(peaks, plane, style.TRACK_CLEARANCE))
    # 河从谷壁里流出来，所以撞上河的那几座山头得让开 —— 让出来的缺口就是河谷口
    peaks = list(_dry(peaks, water, margin=6.0))

    # ---- 草场：谷底铺满；轨道 6 m 以内、水面附近、山脚上的都丢掉
    meadows = tuple(
        scenery_mod.Meadow(x=x, z=z, radius=rng.uniform(20.0, 38.0),
                           seed=rng.randrange(1 << 30), tone=rng.randrange(4),
                           tufts=rng.randrange(6, 12))
        for x, z in _grid(rng, -300.0, 300.0, -150.0, 150.0, 46.0, jitter=0.42)
        if _far_enough(x, z, plane, 6.0) and not _near_water(x, z, water, 3.0)
    )

    # ---- 村子：环线**里**的一簇房子（环内那块 240 × 80 的空地正好当宅基地）
    houses = _cull_overlapping_houses(_only_clear([
        scenery_mod.random_village_house(x, z, rng)
        for x, z in _grid(rng, -58.0, 58.0, -24.0, 24.0, 28.0)
    ], plane, style.TRACK_CLEARANCE + 1.0))

    # ---- 树：山坡脚一条林带 + 谷底散树
    trees = [scenery_mod.Tree(x=x, z=z, height=rng.uniform(7.0, 13.0),
                              seed=rng.randrange(1 << 30),
                              conifer=rng.random() < 0.6)
             for x, z in _ellipse_ring(rng, 0.0, 0.0, 170.0, 115.0, 46, spread=16.0)]
    trees += [scenery_mod.Tree(x=x, z=z, height=rng.uniform(6.0, 11.0),
                               seed=rng.randrange(1 << 30))
              for x, z in _grid(rng, -140.0, 140.0, -80.0, 80.0, 34.0)
              if rng.random() < 0.35 and not _near_water(x, z, water, 2.0)]
    trees = _only_clear(trees, plane, style.TRACK_CLEARANCE)
    trees = _dry(trees, water, margin=2.0)

    return Scene(
        key="valley", title="山谷环线",
        blurb="群山夹着一条长环线，谷底有河、有村子、有草场。",
        layout=layout,
        scenery=scenery_mod.Scenery(peaks=tuple(peaks), rivers=(river,),
                                    meadows=meadows, trees=trees, houses=houses),
        plot_size=PLOT_SIZES["valley"], train_id="cr400bf_huangsidai_8",
        clearance_points=tuple(walk),
    )



# --------------------------------------------------------------------------- #
# 二：河谷大桥
# --------------------------------------------------------------------------- #

def high_points(path: RoutePath, minimum: float = 0.9
                ) -> list[tuple[float, float, float]]:
    """**每一段高架**的最高点（= 桥面正中），一段给一个点。

    桥面在哪一节、跨中在弧长多少，全都不写死 —— 直接问路径"哪儿高"。这样以后把
    引桥挪个位置、把一跨换长一点，河也跟着挪，不需要同步改两组数字。

    "一段"是按**连续**算的，不是按节：一段高架是"引桥 + 桥面 + 引桥"，五节轨道
    都高过地面，但它们只是**一座**桥。按节给点会把一座桥拆成五个跨中点，河就会
    沿着轨道来回折 —— 这是最要命的一种错：河面盖在轨道上，看上去像是"轨道修在
    河里"，而且不报错。

    取的是"最高那一段的**中点**"：桥面是水平的，等高的点连成一片，取它们的弧长
    中点才是跨中；取"第一个最高点"会落到引桥刚爬上来的那一端。
    """
    hits: list[tuple[float, float, float]] = []
    top: list[tuple[float, float]] = []       # 当前这段高架的 (全局弧长, 高度)

    def flush() -> None:
        if not top:
            return
        highest = max(y for _, y in top)
        # 桥面是平的，等高的点取弧长中点；顶着尖的那种（没有平段）就退化成一点
        flat = [s for s, y in top if y >= highest - 0.01]
        hits.append(path.pose_at((min(flat) + max(flat)) * 0.5).position)
        top.clear()

    for segment in path:
        steps = max(2, int(segment.length // 2.0))
        best_u, best_y = 0.0, -math.inf
        for k in range(steps + 1):
            u = segment.length * k / steps
            y = segment.pose_at(u).y
            if y > best_y:
                best_y, best_u = y, u
        if best_y > minimum:
            top.append((segment.s0 + best_u, best_y))
        else:
            flush()
    flush()
    return hits



def _extend_to_edge(points: list[tuple[float, float]], reach: float
                    ) -> list[tuple[float, float]]:
    """把折线两端沿首 / 末段的方向延长，直到 ``|z|`` 超出 ``reach``。"""
    if len(points) < 2:
        return points

    def outward(base, neighbour):
        dx, dz = base[0] - neighbour[0], base[1] - neighbour[1]
        length = math.hypot(dx, dz)
        if length < 1e-9:
            return base
        ux, uz = dx / length, dz / length
        if abs(uz) < 1e-6:
            return (base[0] + ux * reach, base[1])
        t = (math.copysign(reach, uz) - base[1]) / uz
        if t <= 0.0:
            return base
        return (base[0] + ux * t, base[1] + uz * t)

    return [outward(points[0], points[1])] + list(points) + \
           [outward(points[-1], points[-2])]


def build_gorge(catalog: Catalog) -> Scene:
    """一条高架跨过同一条河**两次**的环线，两侧是峡谷壁。

    桥面件（``bridge_40``）架在两节引桥之间，所以桥下真的空着、河真的能过去；
    跨中位置由路径上的最高点反推（:func:`high_points`），河的中心线再由这两个跨
    中点接出去并伸出场地 —— "桥跨在河上"是**算出来的**，不是摆出来的。

    两条直股各 8 节 40 m：每条都是 3 节平轨 + 上坡 + 上坡 + 桥面 + 下坡 + 下坡。
    桥面的位置**故意错开**（一在第 4 节、一在第 6 节），于是两处跨中差着 40 m，
    河是斜着过去的；不错开的话两个跨中点会落在同一条竖线上，河就成了轨道底下
    一根笔直的带子，一眼假。
    """
    up, down = "ramp_up_40", "ramp_down_40"
    layout, _ = tracks.oval(
        catalog,
        cap_piece="curve_r40_l90", cap_pieces=2,
        side_a=("straight_40", up, up, "bridge_40", down, down,
                "straight_40", "straight_40"),
        side_b=("straight_40", "straight_40", "straight_40", up, up, "bridge_40",
                down, down),
    )
    path = closed_path(layout)
    walk = corridor(path)
    plane = [(p[0], p[2]) for p in walk]
    rng = random.Random(20240502)

    bridges = high_points(path)
    if len(bridges) != 2:
        raise SceneError(
            f"河谷场景要有两处桥面，数出来 {len(bridges)} 处 —— "
            "引桥与桥面的排法动了（一段连续高架只该算一处）"
        )
    crossings = sorted(((p[0], p[2]) for p in bridges), key=lambda p: p[1])
    river = scenery_mod.River(
        points=tuple(_extend_to_edge(list(crossings), reach=340.0)),
        width=26.0, bank=3.2, seed=11,
    )
    water = scenery_mod.Scenery(rivers=(river,))

    # ---- 峡谷壁：沿河两岸往场地两头铺开，离轨道很远
    peaks: list[scenery_mod.Peak] = []
    for base_x, sign in ((crossings[0][0], -1.0), (crossings[-1][0], 1.0)):
        for offset in (50.0, 105.0, 160.0):
            for side in (-1.0, 1.0):
                radius = rng.uniform(38.0, 58.0)
                peaks.append(scenery_mod.Peak(
                    x=base_x + side * (offset * 0.55 + rng.uniform(-14.0, 14.0)),
                    z=sign * (175.0 + offset) + rng.uniform(-14.0, 14.0),
                    radius=radius, height=radius * rng.uniform(1.1, 1.7),
                    seed=rng.randrange(1 << 30), sides=12,
                ))
    peaks += [scenery_mod.Peak(x=x, z=z, radius=rng.uniform(40.0, 64.0),
                              height=rng.uniform(70.0, 115.0),
                              seed=rng.randrange(1 << 30), sides=12)
              for x, z in _ellipse_ring(rng, 0.0, 0.0, 280.0, 210.0, 10, spread=20.0)]
    peaks = list(_only_clear(peaks, plane, style.TRACK_CLEARANCE))
    # 河从峡谷里穿出去，撞到河面上的那几座得让开（让出来的就是河谷口）
    peaks = list(_dry(peaks, water, margin=6.0))

    # ---- 河滩上的碎石丘与草
    hills = [scenery_mod.Peak.hill(x=x, z=z, radius=rng.uniform(12.0, 22.0),
                                   height=rng.uniform(2.0, 4.5),
                                   seed=rng.randrange(1 << 30))
             for x, z in _grid(rng, -240.0, 240.0, -150.0, 150.0, 48.0)]
    hills = list(_only_clear(hills, plane, style.TRACK_CLEARANCE))
    hills = list(_dry(hills, water, margin=3.0))

    meadows = tuple(
        scenery_mod.Meadow(x=x, z=z, radius=rng.uniform(16.0, 30.0),
                           seed=rng.randrange(1 << 30), tone=rng.randrange(4),
                           tufts=rng.randrange(4, 9))
        for x, z in _grid(rng, -260.0, 260.0, -180.0, 180.0, 44.0, jitter=0.4)
        if _far_enough(x, z, plane, 7.0) and not _near_water(x, z, water, 3.0)
    )

    houses = _cull_overlapping_houses(_only_clear([
        scenery_mod.random_village_house(x, z, rng)
        for x, z in _grid(rng, -120.0, 120.0, -120.0, 120.0, 34.0)
        if rng.random() < 0.55 and not _near_water(x, z, water, 6.0)
    ], plane, style.TRACK_CLEARANCE + 1.0))

    trees = [scenery_mod.Tree(x=x, z=z, height=rng.uniform(6.0, 12.0),
                              seed=rng.randrange(1 << 30),
                              conifer=rng.random() < 0.7)
             for x, z in _grid(rng, -240.0, 240.0, -170.0, 170.0, 30.0)
             if rng.random() < 0.45 and not _near_water(x, z, water, 4.0)]
    trees = _only_clear(trees, plane, style.TRACK_CLEARANCE)
    trees = _dry(trees, water, margin=2.0)

    return Scene(
        key="gorge", title="河谷大桥",
        blurb="高架桥两次跨过同一条河，两岸是峡谷壁，桥墩落在河滩上。",
        layout=layout,
        scenery=scenery_mod.Scenery(peaks=tuple(peaks) + tuple(hills),
                                    rivers=(river,), meadows=meadows,
                                    trees=trees, houses=houses),
        plot_size=PLOT_SIZES["gorge"], train_id="cr400bf_huangsidai_8",
        clearance_points=tuple(walk),
    )



# --------------------------------------------------------------------------- #
# 三：小镇车站
# --------------------------------------------------------------------------- #

def _station_straight(path: RoutePath, near: tuple[float, float]):
    """离 ``near`` 最近的一段直轨（月台只摆得下在直段上）。

    "最近"是关键：直段有好几节，随便挑一节会让车站和它那条岔线隔开半条环线。
    """
    best, best_distance = None, math.inf
    for segment in path:
        if abs(segment.dtheta) > 1e-9:
            continue
        middle = segment.pose_at(segment.length * 0.5)
        distance = math.hypot(middle.x - near[0], middle.z - near[1])
        if distance < best_distance:
            best, best_distance = segment, distance
    if best is None:
        raise SceneError("这条环线上没有直段，摆不了站台")
    return best


def build_town(catalog: Catalog) -> Scene:
    """一条 R40 环线 + 一条尽头式站场岔线；月台、站房、村子、小山围着车站。"""
    layout, indices = tracks.oval(
        catalog,
        cap_piece="curve_r40_l90", cap_pieces=2,
        # 中间那节直轨换成道岔：直股仍是 40 m，环线几何分毫不动
        side_a=("straight_40", "turnout_l_40", "straight_40"),
        # 对面那条直股必须**显式**给成三节 40 m。``oval`` 的缺省是"B 照抄 A"，
        # 照抄的结果是环线对面又长出一个道岔 —— 站场里凭空多一条岔线，而且它没接
        # 尽头线，末端就是一段悬空的轨道头。
        side_b=("straight_40",) * 3,
    )
    turnout = tracks.switch_index(layout, indices)
    stub = tracks.add_stub(layout, turnout, run=("straight_40", "straight_20"))

    path = closed_path(layout)
    walk = corridor(path)
    plane = [(p[0], p[2]) for p in walk]
    # 尽头线也要参与净空检查（房子盖到岔线上，是这套检查唯一看不见的死角）
    plane += [(p[0], p[2]) for p in _port_polyline(layout, stub)]
    rng = random.Random(20240503)

    # ---- 月台：摆在道岔那一节直轨上、环线的**外侧**
    #
    # 摆在哪一节不写死"第几节"：直接取"离道岔最近的那一段直轨"。原因是直段有好
    # 几节，随手挑一节会让车站和它那条岔线隔开半条环线。
    turnout_pose = layout.piece(turnout).pose
    straight = _station_straight(path, (turnout_pose.x, turnout_pose.z))
    s_mid = straight.s0 + straight.length * 0.5
    centre = _along(path, s_mid, 0.0)
    outward = _outward_axis(walk, centre[0], centre[1])
    width = 6.0
    px, pz, heading, sign = _aside(path, s_mid, _PLATFORM_GAP + width * 0.5, outward)
    platform = scenery_mod.Platform(
        x=px, z=pz, length=straight.length, width=width,
        heading=heading, seed=3, track_side=-sign, lamps=4,
    )
    # 站房摆在月台背后（沿同一条法线再往外挪）：老式欧式古典车站
    behind = width * 0.5 + 13.0
    station = scenery_mod.Station(
        x=px - math.sin(heading) * sign * behind,
        z=pz + math.cos(heading) * sign * behind,
        style=scenery_mod.STATION_CLASSICAL,
        width=20.0, depth=11.0, height=6.8, heading=heading, seed=1,
    )

    # ---- 村子：环线**里**那片空地
    houses = _cull_overlapping_houses(_only_clear([
        scenery_mod.random_village_house(x, z, rng)
        for x, z in _grid(rng, -68.0, 68.0, -25.0, 25.0, 28.0)
    ], plane, style.TRACK_CLEARANCE + 1.0))

    meadows = tuple(
        scenery_mod.Meadow(x=x, z=z, radius=rng.uniform(16.0, 30.0),
                           seed=rng.randrange(1 << 30), tone=rng.randrange(4),
                           tufts=rng.randrange(6, 11))
        for x, z in _grid(rng, -190.0, 190.0, -140.0, 140.0, 40.0, jitter=0.4)
        if _far_enough(x, z, plane, 6.0)
    )

    trees = [scenery_mod.Tree(x=x, z=z, height=rng.uniform(6.0, 11.0),
                              seed=rng.randrange(1 << 30))
             for x, z in _grid(rng, -170.0, 170.0, -130.0, 130.0, 30.0)
             if rng.random() < 0.42]
    trees = _only_clear(trees, plane, style.TRACK_CLEARANCE)

    hills = [scenery_mod.Peak.hill(x=x, z=z, radius=rng.uniform(26.0, 42.0),
                                   height=rng.uniform(6.0, 14.0),
                                   seed=rng.randrange(1 << 30))
             for x, z in _ellipse_ring(rng, 0.0, 0.0, 145.0, 115.0, 12, spread=18.0)]
    hills = list(_only_clear(hills, plane, style.TRACK_CLEARANCE))

    return Scene(
        key="town", title="小镇车站",
        blurb="一条环线 + 一条尽头式岔线，月台、站房、村子和小山围着车站。",
        layout=layout,
        scenery=scenery_mod.Scenery(peaks=tuple(hills), meadows=meadows,
                                    trees=trees, houses=tuple(houses),
                                    stations=(station,),
                                    platforms=(platform,)),
        plot_size=PLOT_SIZES["town"], train_id="cr400bf_huangsidai_8",
        clearance_points=tuple(walk) + tuple(_port_polyline(layout, stub)),
    )


# --------------------------------------------------------------------------- #
# 四：湖畔草原
# --------------------------------------------------------------------------- #

def build_lake(catalog: Catalog) -> Scene:
    """一条 R80 的大圆环，环里一个湖，湖畔散着几座小屋。"""
    layout, _ = tracks.circle(
        catalog, piece="curve_r80_l45", count=8,     # 8 × 45° = 一整圈，R = 80 m
    )                                                # 圆心落在场地原点（tracks 里量过）
    path = closed_path(layout)
    walk = corridor(path)
    plane = [(p[0], p[2]) for p in walk]
    cx, cz = path_centroid(walk)                     # R80 整圆的圆心
    rng = random.Random(20240504)

    lake = scenery_mod.Lake(x=cx, z=cz, radius=52.0, beach=5.5, seed=5)
    water = scenery_mod.Scenery(lakes=(lake,))
    shore = lake.radius + lake.beach

    # ---- 湖畔小屋：湖岸与轨道之间那条环形带里，窗子朝着湖
    houses = _cull_overlapping_houses(_only_clear([
        scenery_mod.random_village_house(
            x, z, rng,
            heading=math.atan2(cz - z, cx - x),
        )
        for x, z in _ring(rng, cx, cz, shore + 12.0, 11, spread=3.5)
    ], plane, style.TRACK_CLEARANCE + 1.0))

    meadows = tuple(
        scenery_mod.Meadow(x=x, z=z, radius=rng.uniform(18.0, 36.0),
                           seed=rng.randrange(1 << 30), tone=rng.randrange(4),
                           tufts=rng.randrange(6, 12))
        for x, z in _grid(rng, cx - 210.0, cx + 210.0, cz - 210.0, cz + 210.0, 44.0,
                          jitter=0.42)
        if _far_enough(x, z, plane, 7.0) and not _near_water(x, z, water, 3.0)
    )

    trees = [scenery_mod.Tree(x=x, z=z, height=rng.uniform(5.5, 10.5),
                              seed=rng.randrange(1 << 30),
                              conifer=rng.random() < 0.35)
             for x, z in _grid(rng, cx - 190.0, cx + 190.0, cz - 190.0, cz + 190.0, 32.0)
             if rng.random() < 0.4 and not _near_water(x, z, water, 2.5)]
    trees += [scenery_mod.Tree(x=x, z=z, height=rng.uniform(4.5, 8.0),
                               seed=rng.randrange(1 << 30), conifer=False)
              for x, z in _ring(rng, cx, cz, shore + 3.5, 14, spread=1.5)
              if lake.distance_to(x, z) > 2.5]
    trees = _only_clear(trees, plane, style.TRACK_CLEARANCE)

    hills = [scenery_mod.Peak.hill(x=x, z=z, radius=rng.uniform(30.0, 48.0),
                                   height=rng.uniform(9.0, 20.0),
                                   seed=rng.randrange(1 << 30))
             for x, z in _ring(rng, cx, cz, 175.0, 14, spread=26.0)]
    hills = list(_only_clear(hills, plane, style.TRACK_CLEARANCE))

    return Scene(
        key="lake", title="湖畔草原",
        blurb="R80 大圆环绕着湖跑，湖岸是沙滩、草地、几座小屋和远处的草丘。",
        layout=layout,
        scenery=scenery_mod.Scenery(peaks=tuple(hills), lakes=(lake,),
                                    meadows=meadows, trees=trees, houses=houses),
        plot_size=PLOT_SIZES["lake"], train_id="cr400bf_huangsidai_8",
        clearance_points=tuple(walk),
    )


# --------------------------------------------------------------------------- #
# 五：立体交叉（跨线高架 + 疏解线）
# --------------------------------------------------------------------------- #

#: 桥下净空不得低于这个值（米）。最高的编组约 4.0 m（车顶 3.6 m + 空调/受电弓），
#: 再留一米余量 —— 桥是给列车（和玩家）看的，净空不够就是"能不能过去"的问题。
BRIDGE_CLEARANCE_MIN = 5.0

#: 高架那一段的爬升：5 节 40 m × 3% = 6.0 m。桥面高度 = 6.0，桥底 = 6.0 - 0.64。
_RAMP_PIECES = 5

#: 跨线桥面用 80 m 的那一件（见 pieces.json）：它的桥墩在距两端 20 m 处，正中间
#: 40 m 无墩 —— 疏解线的两个交叉点就落在这段里。
_OVERSPAN = "bridge_80"

#: 两处道岔之间隔几节 40 m（``straights``）。0 = 只隔必须的那两节（80 m）。
_BYPASS_STRAIGHTS = 0


def build_overpass(catalog: Catalog) -> Scene:
    """一条**跨线高架**＋一条从桥底下钻过去的**立交疏解线**。

    这是全件库里唯一一处"两条轨道在**不同高度**上相交"的场景 —— 地面上的线从
    高架的桥跨底下穿过。它不需要菱形交叉件（那是同一平面上的十字，列车只能直穿、
    不能转弯），全程也没有任何一处平交。

    几何是**算出来**的，不是摆出来的：

    * 环线是 R40 的椭圆，两条直股各 560 m（14 节 40 m）。
    * 高架那条直股：2 节平轨 → 5 节上坡引桥（5 × 40 m × 3% = +6.0 m）→ 1 节
      ``bridge_80`` 大跨桥面 → 5 节下坡引桥 → 回到地面。升降配平，闭环照样精确
      闭合；桥下净空 = 6.0 − 0.64 = **5.36 m**。
    * 另一条直股保持地面高度，第 3、5 节换成道岔；疏解线挂在这两处岔股之间
      （相隔 2 × 40 m），先拐进环线内侧、再折回来钻到桥跨底下，最后回到地面线。
    * 疏解线的两个交叉点落在桥面**正中间那 40 m 无墩段**里：桥墩在距桥面两端
      20 m 处，两个交叉点离最近的墩子各有 8 m 以上 —— 所以列车真的是从桥下过，
      而不是撞在墩子上。这一条由 :func:`planar_crossings` 当场量，量不过就不出图。

    道岔默认扳在直股，所以一开局列车跑的是那条**高架**环线；在编辑器里把鼠标移到
    第一处道岔上按 ``U``，主线就被那处道岔截断，列车改走疏解线、从桥底下穿过去。
    道岔是**可挤岔**的：疏解线接在第二处道岔的岔股上，列车从岔股逆向挤回岔尖、
    并入主线继续走 —— 所以"从岔道回到主线"不会卡成死端；想回到高架环线就把道岔
    扳回直股。
    """
    up, down = "bridge_up_40", "bridge_down_40"
    side_a = (
        ("straight_40",) * 2          # 平段：让引桥从 x = 200 处起步
        + (up,) * _RAMP_PIECES        # 引桥：200 m 爬 6.0 m
        + (_OVERSPAN,)                # 大跨桥面：正中间 40 m 无墩
        + (down,) * _RAMP_PIECES      # 引桥：原样落回地面
    )
    side_b = (
        ("straight_40",) * 3
        + ("turnout_l_40",)           # 疏解线的起点
        + ("straight_40",)
        + ("turnout_l_40",)           # 疏解线的终点（与上一处相隔 2 节 = 80 m）
        + ("straight_40",) * 8
    )
    layout, indices = tracks.oval(catalog, side_a=side_a, side_b=side_b)
    turnouts = tracks.switch_indices(layout, indices)
    if len(turnouts) != 2:
        raise SceneError(f"立交场景要两处道岔，数出来 {len(turnouts)} 处")
    bypass = tracks.add_bypass(layout, turnouts[0], turnouts[1],
                               _BYPASS_STRAIGHTS)

    path = closed_path(layout)
    walk = corridor(path)
    plane = [(p[0], p[2]) for p in walk]
    # 疏解线不在主环上，但布景同样不许压到它 —— 净空量尺必须把它算进去
    bypass_line = _route_polyline(layout, bypass)
    plane += [(p[0], p[2]) for p in bypass_line]

    # ---- 当场量一次"桥下够不够高"：立交场景的全部意思都在这一条上。
    # 疏解线在平面上贴着主线的点有两类：① 两处道岔的**合流**（同一高度，不算交叉）；
    # ② 钻到高架底下的**立体交叉**。这里用 4 m 的贴合半径（小于桥面顶点距高架线的
    # 6 m，所以顶点不算一次交叉），再按"主线在疏解线头顶留够净空"筛掉合流点，
    # 剩下的应当是**钻进去、钻出来**两处。
    crossings = planar_crossings(
        [pose.position for pose in path.sample(_CORRIDOR_STEP)], bypass_line,
        radius=4.0)
    grade_sep = [
        (x, z, y_main, y_bypass)
        for (x, z, y_main, y_bypass) in crossings
        if (y_main - style.BALLAST_BOTTOM_Y) - y_bypass >= BRIDGE_CLEARANCE_MIN
    ]
    if len(grade_sep) != 2:
        raise SceneError(
            f"疏解线应当与高架交叉 **2** 次（钻进去、钻出来），数出来 "
            f"{len(grade_sep)} 次 —— 疏解线的件序列或道岔位置动了"
        )
    for x, z, y_main, y_bypass in grade_sep:
        headroom = (y_main - style.BALLAST_BOTTOM_Y) - y_bypass
        if headroom < BRIDGE_CLEARANCE_MIN:
            raise SceneError(
                f"立交在 ({x:.1f}, {z:.1f}) 的净空只有 {headroom:.2f} m"
                f"（要求 ≥ {BRIDGE_CLEARANCE_MIN} m）：疏解线钻不过去"
            )

    rng = random.Random(20240505)

    # ---- 山：围一圈远山。环线又长又窄（640 × 80），所以铺在按长宽比拉开的椭圆上
    peaks = [
        scenery_mod.Peak(x=x, z=z, radius=radius,
                         height=radius * rng.uniform(1.1, 1.7),
                         seed=rng.randrange(1 << 30),
                         sides=rng.choice((12, 14, 16)))
        for radius, (x, z) in ((rng.uniform(34.0, 56.0), spot)
                               for spot in _ellipse_ring(rng, 0.0, 0.0, 370.0, 200.0,
                                                         22, spread=18.0))
    ]
    peaks = list(_only_clear(peaks, plane, style.TRACK_CLEARANCE))

    # ---- 草场：环线内外都铺，绕着轨道和水面之外的每一处
    meadows = tuple(
        scenery_mod.Meadow(x=x, z=z, radius=rng.uniform(18.0, 34.0),
                           seed=rng.randrange(1 << 30), tone=rng.randrange(4),
                           tufts=rng.randrange(6, 12))
        for x, z in _grid(rng, -350.0, 350.0, -210.0, 210.0, 50.0, jitter=0.42)
        if _far_enough(x, z, plane, 7.0)
    )

    # ---- 村子：环线**里**东边那片空地（西边留给疏解线）
    houses = _cull_overlapping_houses(_only_clear([
        scenery_mod.random_village_house(x, z, rng)
        for x, z in _grid(rng, 20.0, 240.0, -28.0, 28.0, 32.0)
    ], plane, style.TRACK_CLEARANCE + 1.0))

    # ---- 树：山坡脚的林带 + 谷地散树
    trees = [scenery_mod.Tree(x=x, z=z, height=rng.uniform(7.0, 13.0),
                              seed=rng.randrange(1 << 30),
                              conifer=rng.random() < 0.6)
             for x, z in _ellipse_ring(rng, 0.0, 0.0, 300.0, 165.0, 40, spread=16.0)]
    trees += [scenery_mod.Tree(x=x, z=z, height=rng.uniform(6.0, 11.0),
                               seed=rng.randrange(1 << 30))
              for x, z in _grid(rng, -320.0, 320.0, -180.0, 180.0, 34.0)
              if rng.random() < 0.34]
    trees = _only_clear(trees, plane, style.TRACK_CLEARANCE)

    return Scene(
        key="overpass", title="立体交叉",
        blurb="高架跨过地面线：把道岔扳到岔股，列车就从桥底下钻过去。",
        layout=layout,
        scenery=scenery_mod.Scenery(peaks=tuple(peaks), meadows=meadows,
                                    trees=trees, houses=houses),
        plot_size=PLOT_SIZES["overpass"], train_id="cr400bf_huangsidai_16",
        clearance_points=tuple(walk) + tuple(bypass_line),
    )


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #

#: 场景 key → 构造函数。``app/main.py --list-scenes`` 与截图脚本都读这张表。
BUILDERS = {
    "valley": build_valley,
    "gorge": build_gorge,
    "town": build_town,
    "lake": build_lake,
    "overpass": build_overpass,
}


def keys() -> tuple[str, ...]:
    return tuple(BUILDERS)


def build(key: str, catalog: Catalog | None = None) -> Scene:
    """按 key 拼一个场景出来（每次调用都是全新的一份，互不干扰）。"""
    try:
        builder = BUILDERS[key]
    except KeyError:
        raise SceneError(
            f"没有这个场景：{key!r}（可选：{', '.join(keys())}）"
        ) from None
    return builder(catalog if catalog is not None else Catalog.builtin())
