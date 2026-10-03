"""程序化布景：山体 / 河流 / 湖泊 / 房屋 / 树木 / 草地 / 站台。

布景与轨道为什么是两层
------------------------------------------------
轨道那一层（``render.track_mesh``）由 ``data/pieces.json`` 的几何决定，是**可走行**
的东西：走线、动力学、闭环检测全都长在它上面。布景这一层纯粹是外观，不参与任何
计算 —— 分成两层之后，"把山挪一挪"这种事在结构上就不可能让列车跑偏。

**数据与网格也分开**：每个布景物先是下面这些 :mod:`dataclasses` 里的纯数据
（位置、尺寸、随机种子），再由 ``build_*`` 函数变成网格。这样两件事各自都能独立
成立：

* **摆位**（"村子放在环线里面"）可以在不建任何 Panda3D 对象的情况下算出来、量
  距离、写测试 —— ``tests/test_scenery.py`` 的"布景不许压到轨道上"就是这么查的；
* **网格**只需要管"一座山长什么样"，不必知道它为什么摆在那儿。

这一点是有代价换来的：早先的写法是把两件事糅在一起（摆一个就画一个），于是
"某个山脚正好盖住环线"只能靠截图发现，而且发现时已经分不清是摆位错了还是
网格画歪了。

确定性
------------------------------------------------
所有随机起伏都走 ``random.Random(seed)``，种子由摆位那一层给定。同一个场景每次
打开都**逐顶点一致**，所以离屏截图与窗口渲染可以逐像素比对，回归也才有意义
（这一点与 ``render/style.py`` 把光照烘焙进顶点色是同一个理由）。

局部坐标约定
------------------------------------------------
房屋、站台这类有朝向的东西在自己的局部坐标里描述：``+x`` 沿 ``heading``、
``+z`` 朝右手边、``y`` 用**世界高度**（地面是平的，布景都立在同一层上，
所以不另外引入"局部基准高度"）。换算只有 :func:`_local_to_world` 一处实现，
``tests/test_scenery.py`` 按"局部 +x 必须落到 heading 方向"逐点比对。
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, fields
from typing import Iterator, Sequence

from panda3d.core import NodePath

from render import style
from render.mesh import MeshBuilder

#: 无 alpha 的颜色（``shade`` 会原样透传 alpha，布景全是不透明的）。
Color = tuple[float, float, float, float]
Point2 = tuple[float, float]


# --------------------------------------------------------------------------- #
# 局部坐标 ↔ 世界坐标
# --------------------------------------------------------------------------- #

def _local_to_world(origin_x: float, origin_z: float, heading: float,
                    lx: float, ly: float, lz: float) -> tuple[float, float, float]:
    """把局部点 ``(lx, ly, lz)`` 摆到世界坐标。

    与 :func:`render.transform.apply_pose` 是同一个约定：局部 ``+x`` 沿 heading
    （``(cos h, 0, sin h)``）、局部 ``+z`` 朝右手边（``(-sin h, 0, cos h)``）。
    这里不调 apply_pose，是因为布景要的是**点**的变换而不是节点的变换 ——
    一个网格里有几十个点，逐个建节点既慢又没法在无 Panda3D 的环境里测。
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return (
        origin_x + lx * cos_h - lz * sin_h,
        ly,
        origin_z + lx * sin_h + lz * cos_h,
    )


def _local_dir(heading: float, lx: float, ly: float, lz: float
               ) -> tuple[float, float, float]:
    """把局部**方向**（法线）转到世界方向；与点变换用同一个旋转。

    ``y`` 分量原样保留 —— 屋顶斜面的法线是靠它才"抬起来"的，压成 0 就变成了
    一堵竖直的墙，屋里那面会被打亮。
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return (lx * cos_h - lz * sin_h, ly, lx * sin_h + lz * cos_h)


def _local_box(builder: MeshBuilder, origin_x: float, origin_z: float,
               heading: float, x0: float, x1: float, y0: float, y1: float,
               z0: float, z1: float, color: Color,
               top_color: Color | None = None) -> None:
    """在局部坐标里放一个长方体（``x`` 向前、``z`` 横向、``y`` 世界高度）。"""
    origin = _local_to_world(origin_x, origin_z, heading, x0, y0, z0)
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    ex = ((x1 - x0) * cos_h, 0.0, (x1 - x0) * sin_h)
    ey = (0.0, y1 - y0, 0.0)
    ez = (-(z1 - z0) * sin_h, 0.0, (z1 - z0) * cos_h)
    builder.add_oriented_box(origin, ex, ey, ez, color, top_color)


def _local_quad(builder: MeshBuilder, origin_x: float, origin_z: float,
                heading: float, local_points: Sequence[tuple[float, float, float]],
                color: Color, normal_local: tuple[float, float, float] | None = None
                ) -> None:
    """在局部坐标里放一个平面多边形（屋顶斜面走它）。

    ``normal_local`` 给出**局部**法线。它必须给：顶点色里烘焙的是光照，而
    ``MeshBuilder`` 在没给法线时会拿绕序叉乘去猜 —— 屋顶斜面猜反了就会"从下面
    被打亮"，看起来像屋里点了灯。这里显式给法线，绕序由 ``add_polygon`` 负责
    跟它对齐。
    """
    points = [_local_to_world(origin_x, origin_z, heading, *p) for p in local_points]
    normal = None if normal_local is None else _local_dir(
        heading, normal_local[0], normal_local[1], normal_local[2])
    builder.add_polygon(points, color, normal=normal)


# --------------------------------------------------------------------------- #
# 基本体
# --------------------------------------------------------------------------- #

def _add_cone(builder: MeshBuilder, center: tuple[float, float, float],
              radius: float, height: float, sides: int, color: Color,
              cap: bool = True) -> None:
    """低模圆锥：``center`` 是**底面圆心**，锥尖在正上方 ``height`` 处。"""
    if sides < 3 or radius <= 0.0 or height <= 0.0:
        return
    cx, cy, cz = center
    apex = (cx, cy + height, cz)
    ring = [
        (cx + radius * math.cos(math.tau * i / sides), cy,
         cz + radius * math.sin(math.tau * i / sides))
        for i in range(sides)
    ]
    # 绕序：锥尖 → 逆着角度方向才得到**朝外**的法线（顶点色里烘焙的光照按它算，
    # 绕反了整棵树都会"背光"）。下同。
    for i in range(sides):
        builder.add_polygon((apex, ring[(i + 1) % sides], ring[i]), color)
    if cap:
        builder.add_polygon(tuple(reversed(ring)), color, normal=(0.0, -1.0, 0.0))


def _add_ball(builder: MeshBuilder, center: tuple[float, float, float],
              radius: float, sides: int, rings: int, color: Color,
              squash: float = 0.85) -> None:
    """低模球（阔叶树冠）。``squash`` < 1 把它压扁一点，看着更沉。"""
    if sides < 3 or rings < 2 or radius <= 0.0:
        return
    cx, cy, cz = center

    def ring_at(k: int) -> list[tuple[float, float, float]]:
        phi = math.pi * k / rings
        y = cy + math.cos(phi) * radius * squash
        r = math.sin(phi) * radius
        return [
            (cx + r * math.cos(math.tau * i / sides), y,
             cz + r * math.sin(math.tau * i / sides))
            for i in range(sides)
        ]

    top = (cx, cy + radius * squash, cz)
    bottom = (cx, cy - radius * squash, cz)
    first = ring_at(1)
    for i in range(sides):
        builder.add_polygon((top, first[(i + 1) % sides], first[i]), color)
    for k in range(1, rings - 1):
        low, high = ring_at(k + 1), ring_at(k)
        for i in range(sides):
            j = (i + 1) % sides
            builder.add_polygon((high[i], high[j], low[j], low[i]), color)
    last = ring_at(rings - 1)
    for i in range(sides):
        builder.add_polygon((bottom, last[i], last[(i + 1) % sides]), color)


def _tinted(rng: random.Random, color: Color, amount: float = 0.06) -> Color:
    """整体明暗抖一点点：一片树 / 一座山才不会像复制粘贴出来的。"""
    factor = 1.0 + amount * (rng.random() * 2.0 - 1.0)
    return (color[0] * factor, color[1] * factor, color[2] * factor, color[3])


def _polyline_laterals(points: Sequence[Point2]) -> list[Point2]:
    """折线每个点处的横向单位向量（取前后两段的平均，拐弯处才不会打折）。"""
    laterals: list[Point2] = []
    count = len(points)
    for i in range(count):
        before = points[max(i - 1, 0)]
        after = points[min(i + 1, count - 1)]
        tx, tz = after[0] - before[0], after[1] - before[1]
        length = math.hypot(tx, tz)
        if length < 1e-9:
            laterals.append((0.0, 0.0))
        else:
            # 右手法线：切线 (tx, tz) → 横向 (-tz, tx)
            laterals.append((-tz / length, tx / length))
    return laterals


def _point_segment_distance(px: float, pz: float,
                            ax: float, az: float,
                            bx: float, bz: float) -> float:
    dx, dz = bx - ax, bz - az
    length_sq = dx * dx + dz * dz
    if length_sq < 1e-12:
        return math.hypot(px - ax, pz - az)
    t = ((px - ax) * dx + (pz - az) * dz) / length_sq
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(px - (ax + dx * t), pz - (az + dz * t))


# --------------------------------------------------------------------------- #
# 占地轮廓
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Footprint:
    """一个布景物在 XZ 平面上的占地轮廓：矩形，或者圆（``round_``）。

    净空检查（"别把房子盖到轨道上"）量的是**轮廓到轨道中心线的距离**，不是中心到
    中心的距离。这个区别对长条形的东西是决定性的：60 m × 6 m 的站台中心离轨道
    20 m，按中心算绰绰有余，按轮廓算它其实只离 17 m —— 而真正要知道的是后者。
    """

    x: float
    z: float
    half_x: float = 0.0
    half_z: float = 0.0
    heading: float = 0.0
    label: str = ""
    round_: bool = False

    def distance_to(self, px: float, pz: float) -> float:
        """点到轮廓的距离；点在轮廓内部时为 0。"""
        dx, dz = px - self.x, pz - self.z
        if self.round_:
            return max(0.0, math.hypot(dx, dz) - self.half_x)
        cos_h, sin_h = math.cos(self.heading), math.sin(self.heading)
        local_x = dx * cos_h + dz * sin_h
        local_z = -dx * sin_h + dz * cos_h
        outside_x = max(abs(local_x) - self.half_x, 0.0)
        outside_z = max(abs(local_z) - self.half_z, 0.0)
        return math.hypot(outside_x, outside_z)


# --------------------------------------------------------------------------- #
# 布景物：纯数据
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Peak:
    """一座山头（低模圆锥，可以带岩石与雪线）。

    ``rough`` 是轮廓的随机起伏幅度：0 就是一个规规矩矩的圆锥，0.2 左右才像山。
    """

    x: float
    z: float
    radius: float
    height: float
    seed: int = 0
    sides: int = 14
    rings: int = 5
    rough: float = 0.20
    rocky: bool = True
    snow: bool = True

    @classmethod
    def hill(cls, x: float, z: float, radius: float, height: float, *,
             seed: int = 0, sides: int = 12) -> "Peak":
        """一个草丘：没有岩石、没有雪，起伏也小。"""
        return cls(x=x, z=z, radius=radius, height=height, seed=seed, sides=sides,
                   rings=3, rough=0.14, rocky=False, snow=False)

    def footprint(self) -> Footprint:
        return Footprint(x=self.x, z=self.z, half_x=self.radius, half_z=self.radius,
                         label=f"山体({self.height:.0f}m)", round_=True)


@dataclass(frozen=True)
class River:
    """一条河：``points`` 是中心线（XZ），``width`` 是水面宽度。

    中心线通常**伸出场地之外**（场景那边会把两端各延长一截），这样河是"从画面
    外面流进来、又流出去"，而不是在沙盘中间凭空断掉。
    """

    points: tuple[Point2, ...]
    width: float
    bank: float = 1.8
    seed: int = 0

    def distance_to(self, px: float, pz: float) -> float:
        """点到中心线的距离。"""
        best = math.inf
        for i in range(len(self.points) - 1):
            best = min(best, _point_segment_distance(
                px, pz, *self.points[i], *self.points[i + 1]))
        return best

    def contains(self, px: float, pz: float) -> bool:
        """点是否落在水面里。"""
        return self.distance_to(px, pz) <= self.width * 0.5


@dataclass(frozen=True)
class Lake:
    """一个湖：``radius`` 是平均半径，轮廓按种子抖一圈（不是正圆）。"""

    x: float
    z: float
    radius: float
    beach: float = 5.0
    seed: int = 0
    sides: int = 28

    def outline(self, scale: float = 1.0) -> tuple[Point2, ...]:
        """湖岸轮廓；``scale`` 可以再把轮廓整体缩放一次（做深水的内圈）。"""
        rng = random.Random(self.seed)
        # 先把抖动一次算完，再按 scale 生成 —— 直接按 scale 循环会得到两圈**不同的**
        # 抖动，深水内圈就会从岸边戳出去。
        wobble = [1.0 + 0.16 * (rng.random() * 2.0 - 1.0) for _ in range(self.sides)]
        return tuple(
            (self.x + self.radius * scale * wobble[i] * math.cos(math.tau * i / self.sides),
             self.z + self.radius * scale * wobble[i] * math.sin(math.tau * i / self.sides))
            for i in range(self.sides)
        )

    def contains(self, px: float, pz: float) -> bool:
        outline = self.outline()
        return _inside_polygon(px, pz, outline)

    def distance_to(self, px: float, pz: float) -> float:
        """点到湖岸的距离；落在湖里时为 0。

        摆位要用它来"别把草皮铺到水面上"：需要的正是"在水里 = 0"这个语义，
        而不是"离岸边有多深"。
        """
        outline = self.outline()
        if _inside_polygon(px, pz, outline):
            return 0.0
        count = len(outline)
        return min(
            _point_segment_distance(px, pz, *outline[i], *outline[(i + 1) % count])
            for i in range(count)
        )


#: 建筑（``House.kind``）的几种样式：民房 / 高楼 / 商场 / 便利店。
HOUSE_COTTAGE = "cottage"
HOUSE_TOWER = "tower"
HOUSE_MALL = "mall"
HOUSE_SHOP = "shop"

#: 建筑的显示名（HUD / 占地标签共用）。
HOUSE_KIND_LABELS = {
    HOUSE_COTTAGE: "民房",
    HOUSE_TOWER: "高楼",
    HOUSE_MALL: "商场",
    HOUSE_SHOP: "便利店",
}


@dataclass(frozen=True)
class House:
    """一座建筑：按 ``kind`` 在民房 / 高楼 / 商场 / 便利店间切换。

    民房带双坡屋顶，其余是平顶（高楼整面玻璃、商场整面橱窗、便利店带遮阳篷）。
    尺寸与朝向沿用同一套局部坐标：``+x`` 沿 ``heading``、门面开在 ``-z``。
    """

    x: float
    z: float
    width: float = 8.0
    depth: float = 6.5
    height: float = 4.2
    heading: float = 0.0
    seed: int = 0
    kind: str = HOUSE_COTTAGE

    def footprint(self) -> Footprint:
        label = HOUSE_KIND_LABELS.get(self.kind, "房屋")
        return Footprint(x=self.x, z=self.z, half_x=self.width * 0.5,
                         half_z=self.depth * 0.5, heading=self.heading,
                         label=f"{label}({self.width:.0f}×{self.depth:.0f}m)")


#: 树的几种样式（``Tree.kind``）：针叶松 / 阔叶树 / 白杨 / 棕榈。空串 = 按种子选。
TREE_PINE = "pine"
TREE_OAK = "oak"
TREE_POPLAR = "poplar"
TREE_PALM = "palm"

#: 树的显示名。
TREE_KIND_LABELS = {
    TREE_PINE: "针叶松",
    TREE_OAK: "阔叶树",
    TREE_POPLAR: "白杨",
    TREE_PALM: "棕榈",
}


@dataclass(frozen=True)
class Tree:
    """一棵树；``kind`` 决定树形（针叶 / 阔叶 / 白杨 / 棕榈），空串时按种子选。"""

    x: float
    z: float
    height: float = 9.0
    seed: int = 0
    conifer: bool | None = None
    kind: str = ""

    def footprint(self) -> Footprint:
        # 树冠会被风吹得比树干宽，但也不该夸张：按高度的 0.22 折成半径。
        radius = self.height * 0.22
        label = TREE_KIND_LABELS.get(self.kind, "树木")
        return Footprint(x=self.x, z=self.z, half_x=radius, half_z=radius,
                         label=f"{label}({self.height:.0f}m)", round_=True)


@dataclass(frozen=True)
class Meadow:
    """一片草地（贴地的不规则多边形 + 几丛草）。

    草地**不参与净空检查**：它平贴地面，列车压过去既不挡路也看不出来。真要检查
    它，反而会逼着场景作者去躲一块纯装饰的色块。
    """

    x: float
    z: float
    radius: float
    seed: int = 0
    tone: int = 0
    tufts: int = 8
    sides: int = 11


@dataclass(frozen=True)
class Platform:
    """站台：台体 + 靠轨道一侧的黄色安全线 + 几根灯柱。

    ``track_side`` 说明轨道在局部 ``z`` 的哪一侧（``-1`` 表示在 ``-z`` 一侧）。
    安全线和灯柱都按它摆放 —— 站台反了，黄线就会画到背面去。
    """

    x: float
    z: float
    length: float = 60.0
    width: float = 6.0
    height: float = 0.55
    heading: float = 0.0
    seed: int = 0
    track_side: float = -1.0
    lamps: int = 4

    def footprint(self) -> Footprint:
        return Footprint(x=self.x, z=self.z, half_x=self.length * 0.5,
                         half_z=self.width * 0.5, heading=self.heading,
                         label=f"站台({self.length:.0f}m)")


#: 车站的两种样式（``Station.style`` 取值）。
STATION_CLASSICAL = "classical"
STATION_MODERN = "modern"

#: 车站自带站台的尺寸（``Station.with_platform=True`` 时在正前方搭一条）。
_STATION_PLATFORM_WIDTH = 5.0
_STATION_PLATFORM_GAP = 2.5      # 站房墙到站台近边的人行通道宽
_STATION_PLATFORM_EXTRA = 6.0    # 站台比正立面两端各长出一截


@dataclass(frozen=True)
class Station:
    """一座车站建筑：**老式欧式古典**或**新式现代化**两种样式。

    局部坐标与 :class:`House` 一致：``+x`` 沿 ``heading``（正面朝向月台 / 轨道），
    ``width`` 是沿轨道的正立面宽，``depth`` 是纵深。样式只影响网格，不影响占地
    轮廓 —— 轮廓按矩形算，净空检查照常工作。

    ``with_platform=True`` 时，站房正面（局部 ``-z``）会自带一条站台，随网格一起
    烘出来：这是给「手工摆车站」用的，预设场景则照旧单放站台、独立算净空。
    """

    x: float
    z: float
    style: str = STATION_CLASSICAL
    width: float = 22.0
    depth: float = 10.0
    height: float = 7.0
    heading: float = 0.0
    seed: int = 0
    with_platform: bool = False

    def platform_length(self) -> float:
        return self.width + _STATION_PLATFORM_EXTRA

    def platform_offset(self) -> float:
        """自带站台中心到站房中心在局部 ``-z`` 方向的距离（米）。"""
        return self.depth * 0.5 + _STATION_PLATFORM_GAP \
            + _STATION_PLATFORM_WIDTH * 0.5

    def footprint(self) -> Footprint:
        label = ("老式车站" if self.style != STATION_MODERN else "新式车站")
        half_x = self.width * 0.5
        half_z = self.depth * 0.5
        if self.with_platform:
            half_z = self.platform_offset() + _STATION_PLATFORM_WIDTH * 0.5
            half_x = max(half_x, self.platform_length() * 0.5)
        return Footprint(x=self.x, z=self.z, half_x=half_x,
                         half_z=half_z, heading=self.heading,
                         label=f"{label}({self.width:.0f}×{self.depth:.0f}m)")


def _inside_polygon(px: float, pz: float, polygon: Sequence[Point2]) -> bool:
    """射线法判断点在多边形内（湖面/草地这类简单多边形够用）。"""
    inside = False
    count = len(polygon)
    for i in range(count):
        ax, az = polygon[i]
        bx, bz = polygon[(i + 1) % count]
        if (az > pz) != (bz > pz):
            t = (pz - az) / (bz - az)
            if px < ax + (bx - ax) * t:
                inside = not inside
    return inside


# --------------------------------------------------------------------------- #
# 布景集合
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Scenery:
    """一个场景的全部布景（纯数据）。"""

    peaks: tuple[Peak, ...] = ()
    rivers: tuple[River, ...] = ()
    lakes: tuple[Lake, ...] = ()
    meadows: tuple[Meadow, ...] = ()
    trees: tuple[Tree, ...] = ()
    houses: tuple[House, ...] = ()
    stations: tuple[Station, ...] = ()
    platforms: tuple[Platform, ...] = ()

    # ---------------------------------------------------------------- 组合

    def __bool__(self) -> bool:
        return self.item_count > 0

    @property
    def item_count(self) -> int:
        return sum(len(part) for part in (
            self.peaks, self.rivers, self.lakes, self.meadows, self.trees,
            self.houses, self.stations, self.platforms,
        ))

    def merged(self, *others: "Scenery") -> "Scenery":
        """把几份布景并成一份（每个场景的各区域各写一个函数，最后并起来）。"""
        parts = (self,) + others
        return Scenery(
            peaks=tuple(p for part in parts for p in part.peaks),
            rivers=tuple(r for part in parts for r in part.rivers),
            lakes=tuple(l for part in parts for l in part.lakes),
            meadows=tuple(m for part in parts for m in part.meadows),
            trees=tuple(t for part in parts for t in part.trees),
            houses=tuple(h for part in parts for h in part.houses),
            stations=tuple(s for part in parts for s in part.stations),
            platforms=tuple(p for part in parts for p in part.platforms),
        )

    # ---------------------------------------------------------------- 查询

    def footprints(self) -> Iterator[Footprint]:
        """所有**实心**布景的占地轮廓（草地与水面不算）。"""
        for peak in self.peaks:
            yield peak.footprint()
        for tree in self.trees:
            yield tree.footprint()
        for house in self.houses:
            yield house.footprint()
        for station in self.stations:
            yield station.footprint()
        for platform in self.platforms:
            yield platform.footprint()

    def clearance(self, points: Sequence[tuple[float, float, float]]
                  ) -> tuple[float, str]:
        """整份布景到给定点列的**最小净距**，以及最紧的那一件是谁。

        ``points`` 一般取轨道中心线的采样点（``RoutePath.sample()``）。返回
        ``(净距, 名称)``；没有实心布景时返回 ``(inf, "")``。

        这里**必须**把每个轮廓的所有点都量完。曾经有一版是"中途一旦比当前最紧的
        还紧就跳出"—— 省下的那点乘加（几十个轮廓 × 几百个点）换来的是：报出来的
        净距随采样点列的相位浮动，而且偏**大**（提前跳出 ⇒ 没量到真正最近的那个
        点）。"自报 18.8 m、独立量出来 19.7 m"这种账，就是它记出来的。
        """
        worst = (math.inf, "")
        for footprint in self.footprints():
            nearest = math.inf
            for point in points:
                nearest = min(nearest, footprint.distance_to(point[0], point[2]))
            if nearest < worst[0]:
                worst = (nearest, footprint.label or "布景")
        return worst

    def bounds(self) -> tuple[float, float, float, float]:
        """布景的 XZ 包围盒 ``(min_x, min_z, max_x, max_z)``（取景与"别出场地"检查用）。"""
        xs: list[float] = []
        zs: list[float] = []
        for peak in self.peaks:
            xs += [peak.x - peak.radius, peak.x + peak.radius]
            zs += [peak.z - peak.radius, peak.z + peak.radius]
        for river in self.rivers:
            xs += [min(x for x, _ in river.points) - river.bank,
                   max(x for x, _ in river.points) + river.bank]
            zs += [min(z for _, z in river.points) - river.bank,
                   max(z for _, z in river.points) + river.bank]
        for lake in self.lakes:
            reach = lake.radius + lake.beach
            xs += [lake.x - reach, lake.x + reach]
            zs += [lake.z - reach, lake.z + reach]
        for meadow in self.meadows:
            xs += [meadow.x - meadow.radius, meadow.x + meadow.radius]
            zs += [meadow.z - meadow.radius, meadow.z + meadow.radius]
        for tree in self.trees:
            xs.append(tree.x)
            zs.append(tree.z)
        for house in self.houses:
            reach = max(house.width, house.depth) * 0.5
            xs += [house.x - reach, house.x + reach]
            zs += [house.z - reach, house.z + reach]
        for station in self.stations:
            reach = max(station.width, station.depth) * 0.5
            xs += [station.x - reach, station.x + reach]
            zs += [station.z - reach, station.z + reach]
        for platform in self.platforms:
            reach = max(platform.length, platform.width) * 0.5
            xs += [platform.x - reach, platform.x + reach]
            zs += [platform.z - reach, platform.z + reach]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(zs), max(xs), max(zs))


# --------------------------------------------------------------------------- #
# 网格生成
# --------------------------------------------------------------------------- #

def _add_peak(builder: MeshBuilder, peak: Peak) -> None:
    """一座山：一圈一圈往上收的环缝成锥，按高度分层上色。

    每个环都是**等高的多边形**，所以山是"一层层台阶"上去的 —— 低模山该有的样子。
    顶点高度不逐点随机（那样会变成一圈尖刺），只让半径逐方位抖动、让每层的整体
    高度微微起伏。
    """
    sides = max(6, peak.sides)
    rings = max(2, peak.rings)
    rng = random.Random(peak.seed)
    base_y = style.GROUND_Y + style.SCENERY_LIFT

    wobble = [1.0 + peak.rough * (rng.random() * 2.0 - 1.0) for _ in range(sides)]
    layer = [1.0 + peak.rough * 0.30 * (rng.random() * 2.0 - 1.0)
             for _ in range(rings + 1)]
    layer[0] = 1.0                     # 山顶不高不低，就是一个尖

    def ring(k: int) -> list[tuple[float, float, float]]:
        t = k / rings
        radius = peak.radius * (t ** 0.88)
        height = peak.height * ((1.0 - t) ** 1.30) * layer[k]
        return [
            (peak.x + radius * wobble[i] * math.cos(math.tau * i / sides),
             base_y + height,
             peak.z + radius * wobble[i] * math.sin(math.tau * i / sides))
            for i in range(sides)
        ]

    def color_at(frac: float) -> Color:
        if peak.snow and frac > 0.80:
            return style.MOUNTAIN_SNOW_COLOR
        if peak.rocky and frac > 0.52:
            return style.MOUNTAIN_ROCK_HIGH_COLOR
        if peak.rocky and frac > 0.30:
            return style.MOUNTAIN_ROCK_COLOR
        if peak.rocky:
            return style.MOUNTAIN_FOOT_COLOR
        return style.HILL_COLOR

    apex = (peak.x, base_y + peak.height * layer[0], peak.z)
    lower = ring(1)
    upper_color = color_at(1.0 - 0.5 / rings)
    for i in range(sides):
        builder.add_polygon(
            (apex, lower[(i + 1) % sides], lower[i]),
            _tinted(rng, upper_color),
        )

    for k in range(1, rings):
        low, high = ring(k + 1), ring(k)
        # 这一圈台阶的"平均高度比例"，用来决定它是林地、岩石还是雪
        frac = 1.0 - (k + 0.5) / rings
        color = color_at(frac)
        for i in range(sides):
            j = (i + 1) % sides
            builder.add_polygon((high[i], high[j], low[j], low[i]),
                                _tinted(rng, color))


def _add_river(builder: MeshBuilder, river: River) -> None:
    """一条河：沙洲带 + 水面 + 河心深水，三层依次抬高一点点，互不闪烁。"""
    points = river.points
    if len(points) < 2:
        return
    laterals = _polyline_laterals(points)
    y_shore = style.GROUND_Y + style.SCENERY_LIFT
    y_water = y_shore + style.WATER_LIFT
    y_deep = y_water + style.WATER_LIFT

    bands = (
        (river.width * 0.5 + river.bank, y_shore, style.SHORE_COLOR),
        (river.width * 0.5, y_water, style.WATER_COLOR),
        (river.width * 0.24, y_deep, style.WATER_DEEP_COLOR),
    )
    for half, y, color in bands:
        for i in range(len(points) - 1):
            (ax, az), (bx, bz) = points[i], points[i + 1]
            (lax, laz), (lbx, lbz) = laterals[i], laterals[i + 1]
            builder.add_quad_flat(
                (ax + lax * half, y, az + laz * half),
                (ax - lax * half, y, az - laz * half),
                (bx - lbx * half, y, bz - lbz * half),
                (bx + lbx * half, y, bz + lbz * half),
                color, (0.0, 1.0, 0.0),
            )


def _add_lake(builder: MeshBuilder, lake: Lake) -> None:
    """一个湖：沙滩环 + 水面 + 湖心深水。

    沙滩环是把湖岸轮廓整体外扩 ``beach`` 米得到的**同一圈抖动**上的点，因此环带
    宽度处处相等 —— 若各算各的抖动，环带会一侧宽一侧窄，看着像岸被啃掉一块。
    """
    outline = lake.outline()
    outer = lake.outline(scale=(lake.radius + lake.beach) / max(lake.radius, 1e-6))
    y_shore = style.GROUND_Y + style.SCENERY_LIFT
    y_water = y_shore + style.WATER_LIFT
    y_deep = y_water + style.WATER_LIFT

    count = len(outline)
    for i in range(count):
        j = (i + 1) % count
        builder.add_quad_flat(
            (outline[i][0], y_shore, outline[i][1]),
            (outer[i][0], y_shore, outer[i][1]),
            (outer[j][0], y_shore, outer[j][1]),
            (outline[j][0], y_shore, outline[j][1]),
            style.SHORE_COLOR, (0.0, 1.0, 0.0),
        )

    for scale, y, color in ((1.0, y_water, style.WATER_COLOR),
                            (0.62, y_deep, style.WATER_DEEP_COLOR)):
        ring = lake.outline(scale=scale)
        center = (lake.x, y, lake.z)
        for i in range(count):
            j = (i + 1) % count
            builder.add_polygon(
                (center, (ring[i][0], y, ring[i][1]), (ring[j][0], y, ring[j][1])),
                color, normal=(0.0, 1.0, 0.0),
            )


def _add_meadow(builder: MeshBuilder, meadow: Meadow) -> None:
    """一片草地：不规则贴地多边形 + 几丛草。

    每片草皮按种子取一个"层号"再抬高 4 mm —— 草地是会互相压着的（一片挨着一片
    才连成草原），同高就会共面闪烁。4 mm 在沙盘尺度下看不出来，却足够分层。
    """
    rng = random.Random(meadow.seed)
    y = style.GROUND_Y + style.SCENERY_LIFT + 0.004 * (meadow.seed % 4)
    sides = max(5, meadow.sides)
    wobble = [1.0 + 0.30 * (rng.random() * 2.0 - 1.0) for _ in range(sides)]
    outline = tuple(
        (meadow.x + meadow.radius * wobble[i] * math.cos(math.tau * i / sides),
         meadow.z + meadow.radius * wobble[i] * math.sin(math.tau * i / sides))
        for i in range(sides)
    )
    tone = style.MEADOW_COLORS[meadow.tone % len(style.MEADOW_COLORS)]
    builder.add_polygon([(x, y, z) for x, z in outline], tone,
                        normal=(0.0, 1.0, 0.0))

    for _ in range(max(0, meadow.tufts)):
        angle = math.tau * rng.random()
        reach = meadow.radius * 0.86 * math.sqrt(rng.random())
        _add_cone(
            builder,
            (meadow.x + reach * math.cos(angle), y, meadow.z + reach * math.sin(angle)),
            radius=0.30, height=0.62, sides=3,
            color=_tinted(rng, style.GRASS_TUFT_COLOR, 0.10),
        )


def _add_tree(builder: MeshBuilder, tree: Tree) -> None:
    """一棵树：按 ``kind`` 选树形（针叶三层锥 / 阔叶球冠 / 白杨细高 / 棕榈）。"""
    rng = random.Random(tree.seed)
    height = max(1.5, tree.height)
    base_y = style.GROUND_Y + style.SCENERY_LIFT

    if tree.kind == TREE_POPLAR:
        _add_tree_poplar(builder, tree, rng, height, base_y)
        return
    if tree.kind == TREE_PALM:
        _add_tree_palm(builder, tree, rng, height, base_y)
        return

    if tree.kind:
        conifer = tree.kind == TREE_PINE
    else:
        conifer = rng.random() < 0.5 if tree.conifer is None else tree.conifer
    leaf = _tinted(rng, style.TREE_LEAF_COLORS[rng.randrange(
        len(style.TREE_LEAF_COLORS))], 0.10)
    trunk_height = height * (0.32 if conifer else 0.44)
    trunk_radius = max(0.07, height * 0.028)
    builder.add_cylinder((tree.x, base_y, tree.z),
                         (tree.x, base_y + trunk_height, tree.z),
                         trunk_radius, 6, style.TREE_TRUNK_COLOR)

    if conifer:
        # 三层锥，从下往上收：越上面越小，也就越高
        tiers = ((height * 0.40, height * 0.42),
                 (height * 0.31, height * 0.36),
                 (height * 0.20, height * 0.30))
        for index, (radius, span) in enumerate(tiers):
            _add_cone(builder,
                      (tree.x, base_y + trunk_height * 0.7 + index * height * 0.20,
                       tree.z),
                      radius=radius, height=span, sides=7,
                      color=_tinted(rng, leaf, 0.06))
    else:
        _add_ball(builder, (tree.x, base_y + trunk_height + height * 0.26, tree.z),
                  radius=height * 0.34, sides=7, rings=3, color=leaf, squash=0.86)


def _add_tree_poplar(builder: MeshBuilder, tree: Tree, rng: random.Random,
                     height: float, base_y: float) -> None:
    """白杨：细高的树干 + 一个瘦长的锥冠，整体像一根收窄的蜡烛。"""
    leaf = _tinted(rng, style.TREE_LEAF_COLORS[rng.randrange(
        len(style.TREE_LEAF_COLORS))], 0.10)
    trunk_height = height * 0.5
    trunk_radius = max(0.06, height * 0.022)
    builder.add_cylinder((tree.x, base_y, tree.z),
                         (tree.x, base_y + trunk_height, tree.z),
                         trunk_radius, 6, style.TREE_TRUNK_COLOR)
    _add_cone(builder, (tree.x, base_y + trunk_height * 0.8, tree.z),
              radius=height * 0.16, height=height * 0.58, sides=8,
              color=leaf)


def _add_tree_palm(builder: MeshBuilder, tree: Tree, rng: random.Random,
                   height: float, base_y: float) -> None:
    """棕榈：一段略弯的树干 + 顶部一圈展开的叶片。"""
    trunk_height = height * 0.62
    trunk_radius = max(0.07, height * 0.05)
    x, z = tree.x, tree.z
    builder.add_cylinder((x, base_y, z), (x, base_y + trunk_height, z),
                         trunk_radius, 6, style.TREE_PALM_TRUNK_COLOR)
    crown = (x, base_y + trunk_height, z)
    frond = style.TREE_PALM_FROND_COLOR
    count = 7
    for i in range(count):
        angle = math.tau * i / count + rng.random() * 0.15
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        reach = height * 0.38
        tip = (x + cos_a * reach, base_y + trunk_height - height * 0.10,
               z + sin_a * reach)
        half = height * 0.08
        lx, lz = -sin_a * half, cos_a * half
        builder.add_polygon(
            ((crown[0] + lx, crown[1], crown[2] + lz),
             (crown[0] - lx, crown[1], crown[2] - lz),
             tip),
            frond, normal=(0.0, 1.0, 0.0),
        )


def _add_house(builder: MeshBuilder, house: House) -> None:
    """一座建筑：按 ``house.kind`` 分派到民房 / 高楼 / 商场 / 便利店。"""
    if house.kind == HOUSE_TOWER:
        _add_house_tower(builder, house)
    elif house.kind == HOUSE_MALL:
        _add_house_mall(builder, house)
    elif house.kind == HOUSE_SHOP:
        _add_house_shop(builder, house)
    else:
        _add_house_cottage(builder, house)


def _add_house_cottage(builder: MeshBuilder, house: House) -> None:
    """普通民房：主体 + 双坡屋顶 + 门 + 窗（+ 一半概率的烟囱）。"""
    rng = random.Random(house.seed)
    wall = style.HOUSE_WALL_COLORS[rng.randrange(len(style.HOUSE_WALL_COLORS))]
    roof = style.HOUSE_ROOF_COLORS[rng.randrange(len(style.HOUSE_ROOF_COLORS))]
    base_y = style.GROUND_Y + style.SCENERY_LIFT
    half_w, half_d = house.width * 0.5, house.depth * 0.5
    top = base_y + house.height
    x, z, heading = house.x, house.z, house.heading

    _local_box(builder, x, z, heading, -half_w, half_w, base_y, top,
               -half_d, half_d, wall, top_color=wall)

    # 屋顶：两片斜面 + 两端山墙。挑檐 0.35 m，不然屋顶像块贴在墙上的板。
    overhang = 0.35
    roof_half_w, roof_half_d = half_w + overhang, half_d + overhang
    ridge_h = min(2.2, house.depth * 0.45)
    slope = math.hypot(roof_half_d, ridge_h)
    normal_z = ridge_h / slope
    normal_y = roof_half_d / slope
    _local_quad(builder, x, z, heading,
                ((-roof_half_w, top, -roof_half_d), (roof_half_w, top, -roof_half_d),
                 (roof_half_w, top + ridge_h, 0.0), (-roof_half_w, top + ridge_h, 0.0)),
                roof, normal_local=(0.0, normal_y, -normal_z))
    _local_quad(builder, x, z, heading,
                ((roof_half_w, top, roof_half_d), (-roof_half_w, top, roof_half_d),
                 (-roof_half_w, top + ridge_h, 0.0), (roof_half_w, top + ridge_h, 0.0)),
                roof, normal_local=(0.0, normal_y, normal_z))
    for side in (-1.0, 1.0):
        _local_polygon_wall_gable(builder, x, z, heading, side * half_w,
                                  roof_half_d, top, ridge_h, wall)

    # 门：开在 -z 那一面（临街的一面）
    door_w, door_h = 1.10, 2.10
    door_x = -half_w * 0.35
    _local_box(builder, x, z, heading, door_x - door_w * 0.5, door_x + door_w * 0.5,
               base_y, base_y + door_h, -half_d - 0.03, -half_d + 0.10,
               style.HOUSE_DOOR_COLOR, top_color=style.HOUSE_DOOR_COLOR)

    # 窗：南面一排，东西各一扇 —— 位置按尺寸摊开，房子大了窗也跟着多
    window_rows = 2 if house.height > 5.0 else 1
    for row in range(window_rows):
        win_y0 = base_y + 2.60 + row * 3.00
        win_y1 = win_y0 + 1.10
        if win_y1 > top - 0.20:
            break
        for slot in (-1, 1):
            slot_x = slot * half_w * 0.48
            _local_box(builder, x, z, heading,
                       slot_x - 0.70, slot_x + 0.70, win_y0, win_y1,
                       -half_d - 0.02, -half_d + 0.08,
                       style.HOUSE_WINDOW_COLOR, top_color=style.HOUSE_WINDOW_COLOR)

    if rng.random() < 0.5:
        chimney = 0.45
        _local_box(builder, x, z, heading, -chimney, chimney,
                   top, top + ridge_h + 0.9, -chimney, chimney,
                   style.HOUSE_ROOF_COLORS[1], top_color=style.HOUSE_ROOF_COLORS[1])


def _add_house_tower(builder: MeshBuilder, house: House) -> None:
    """高楼：细高的板楼 + 成排玻璃幕墙 + 楼顶一圈女儿墙。"""
    wall = style.HOUSE_TOWER_WALL_COLOR
    glass = style.HOUSE_TOWER_GLASS_COLOR
    roof = style.HOUSE_TOWER_ROOF_COLOR
    base_y = style.GROUND_Y + style.SCENERY_LIFT
    half_w, half_d = house.width * 0.5, house.depth * 0.5
    top = base_y + house.height
    x, z, heading = house.x, house.z, house.heading

    _local_box(builder, x, z, heading, -half_w, half_w, base_y, top,
               -half_d, half_d, wall, top_color=wall)

    # 楼顶女儿墙 + 一个设备房
    parapet = 0.6
    _local_box(builder, x, z, heading, -half_w - 0.15, half_w + 0.15, top,
               top + parapet, -half_d - 0.15, half_d + 0.15, roof, top_color=roof)
    _local_box(builder, x, z, heading, -half_w * 0.3, half_w * 0.3, top,
               top + 2.2, -half_d * 0.3, half_d * 0.3, roof, top_color=roof)

    # 玻璃幕墙：沿高度分层、沿立面按开间铺；正面背面各一排，两端山墙各一排
    floors = max(2, int(house.height // 3.2))
    bays = max(1, int(house.width // 3.4))
    bay_step = house.width / max(1, bays)
    for f in range(floors):
        y0 = base_y + 0.9 + f * 3.2
        y1 = y0 + 1.7
        if y1 > top - 0.4:
            break
        for k in range(bays):
            wx = -half_w + bay_step * (k + 0.5)
            _local_box(builder, x, z, heading, wx - 0.9, wx + 0.9, y0, y1,
                       -half_d - 0.02, -half_d + 0.07, glass, top_color=glass)
            _local_box(builder, x, z, heading, wx - 0.9, wx + 0.9, y0, y1,
                       half_d - 0.07, half_d + 0.02, glass, top_color=glass)
        for k in range(max(1, int(house.depth // 3.4))):
            dz_step = house.depth / max(1, int(house.depth // 3.4))
            wz = -half_d + dz_step * (k + 0.5)
            _local_box(builder, x, z, heading, half_w - 0.07, half_w + 0.02,
                       y0, y1, wz - 0.9, wz + 0.9, glass, top_color=glass)
            _local_box(builder, x, z, heading, -half_w - 0.02, -half_w + 0.07,
                       y0, y1, wz - 0.9, wz + 0.9, glass, top_color=glass)


def _add_house_mall(builder: MeshBuilder, house: House) -> None:
    """商场：宽扁的体量 + 正面整面橱窗 + 门头招牌 + 出挑平檐。"""
    wall = style.HOUSE_MALL_WALL_COLOR
    glass = style.HOUSE_MALL_GLASS_COLOR
    sign = style.HOUSE_MALL_SIGN_COLOR
    base_y = style.GROUND_Y + style.SCENERY_LIFT
    half_w, half_d = house.width * 0.5, house.depth * 0.5
    top = base_y + house.height
    x, z, heading = house.x, house.z, house.heading

    _local_box(builder, x, z, heading, -half_w, half_w, base_y, top,
               -half_d, half_d, wall, top_color=wall)

    # 正面整面橱窗（-z），离地到近顶
    _local_box(builder, x, z, heading, -half_w + 0.6, half_w - 0.6,
               base_y + 0.5, top - 1.3, -half_d - 0.02, -half_d + 0.08,
               glass, top_color=glass)
    # 门头招牌带
    _local_box(builder, x, z, heading, -half_w + 1.0, half_w - 1.0,
               top - 1.3, top - 0.35, -half_d - 0.06, -half_d + 0.02,
               sign, top_color=sign)
    # 出挑平檐
    overhang = 1.2
    slab = 0.4
    _local_box(builder, x, z, heading,
               -half_w - overhang, half_w + overhang, top, top + slab,
               -half_d - overhang, half_d + overhang, wall, top_color=wall)


def _add_house_shop(builder: MeshBuilder, house: House) -> None:
    """便利店：小门脸 + 前檐遮阳篷 + 门头招牌。"""
    wall = style.HOUSE_SHOP_WALL_COLOR
    awning = style.HOUSE_SHOP_AWNING_COLOR
    sign = style.HOUSE_SHOP_SIGN_COLOR
    base_y = style.GROUND_Y + style.SCENERY_LIFT
    half_w, half_d = house.width * 0.5, house.depth * 0.5
    top = base_y + house.height
    x, z, heading = house.x, house.z, house.heading

    _local_box(builder, x, z, heading, -half_w, half_w, base_y, top,
               -half_d, half_d, wall, top_color=wall)

    # 门头招牌（-z 正面）
    _local_box(builder, x, z, heading, -half_w * 0.72, half_w * 0.72,
               base_y + 2.35, base_y + 2.85, -half_d - 0.04, -half_d + 0.03,
               sign, top_color=sign)
    # 遮阳篷：一块挑出正面的扁板
    _local_box(builder, x, z, heading, -half_w - 0.5, half_w + 0.5,
               base_y + 2.15, base_y + 2.30, -half_d - 1.2, -half_d + 0.1,
               awning, top_color=awning)
    # 门
    _local_box(builder, x, z, heading, -0.6, 0.6, base_y, base_y + 2.1,
               -half_d - 0.03, -half_d + 0.10,
               style.HOUSE_DOOR_COLOR, top_color=style.HOUSE_DOOR_COLOR)
    # 两侧橱窗
    for slot in (-1, 1):
        _local_box(builder, x, z, heading,
                   slot * half_w * 0.62 - 0.7, slot * half_w * 0.62 + 0.7,
                   base_y + 0.5, base_y + 1.9, -half_d - 0.02, -half_d + 0.06,
                   style.HOUSE_WINDOW_COLOR, top_color=style.HOUSE_WINDOW_COLOR)


def _local_polygon_wall_gable(builder: MeshBuilder, x: float, z: float,
                              heading: float, plane_x: float, half_d: float,
                              base_y: float, ridge_h: float, color: Color) -> None:
    """山墙（屋顶两端那块三角形）。"""
    direction_x = 1.0 if plane_x >= 0 else -1.0
    _local_quad(builder, x, z, heading,
                ((plane_x, base_y, half_d), (plane_x, base_y, -half_d),
                 (plane_x, base_y + ridge_h, 0.0)),
                color, normal_local=(direction_x, 0.0, 0.0))


def _add_pyramid(builder: MeshBuilder, x: float, z: float, heading: float,
                 cx: float, base_y: float, cz: float, half: float,
                 height: float, color: Color) -> None:
    """一个底面与局部轴对齐的四棱锥（钟楼尖顶用）。``(cx, base_y, cz)`` 是底心。"""
    apex = _local_to_world(x, z, heading, cx, base_y + height, cz)
    ring = [
        _local_to_world(x, z, heading, cx + dx, base_y, cz + dz)
        for dx, dz in ((half, half), (-half, half), (-half, -half), (half, -half))
    ]
    for i in range(4):
        builder.add_polygon((apex, ring[i], ring[(i + 1) % 4]), color)


def _add_station_classical(builder: MeshBuilder, station: Station) -> None:
    """老式欧式古典车站：长形站房 + 双坡顶 + 中央钟楼 + 尖顶 + 拱窗与大门。

    正面（局部 ``-z``，朝月台那一侧）开着门与窗；钟楼立在正立面的中央。
    """
    rng = random.Random(station.seed)
    wall = style.STATION_CLASSICAL_WALL_COLOR
    roof = style.STATION_CLASSICAL_ROOF_COLOR
    trim = style.STATION_CLASSICAL_TRIM_COLOR
    glass = style.STATION_CLASSICAL_GLASS_COLOR
    base_y = style.GROUND_Y + style.SCENERY_LIFT
    half_w, half_d = station.width * 0.5, station.depth * 0.5
    top = base_y + station.height
    x, z, heading = station.x, station.z, station.heading

    # 主体
    _local_box(builder, x, z, heading, -half_w, half_w, base_y, top,
               -half_d, half_d, wall, top_color=wall)

    # 双坡屋顶（屋脊沿 x，即沿轨道方向），挑檐 0.5 m
    overhang = 0.5
    roof_half_w, roof_half_d = half_w + overhang, half_d + overhang
    ridge_h = min(3.0, station.depth * 0.5)
    slope = math.hypot(roof_half_d, ridge_h)
    normal_z = ridge_h / slope
    normal_y = roof_half_d / slope
    _local_quad(builder, x, z, heading,
                ((-roof_half_w, top, -roof_half_d), (roof_half_w, top, -roof_half_d),
                 (roof_half_w, top + ridge_h, 0.0), (-roof_half_w, top + ridge_h, 0.0)),
                roof, normal_local=(0.0, normal_y, -normal_z))
    _local_quad(builder, x, z, heading,
                ((roof_half_w, top, roof_half_d), (-roof_half_w, top, roof_half_d),
                 (-roof_half_w, top + ridge_h, 0.0), (roof_half_w, top + ridge_h, 0.0)),
                roof, normal_local=(0.0, normal_y, normal_z))
    for side in (-1.0, 1.0):
        _local_polygon_wall_gable(builder, x, z, heading, side * half_w,
                                  roof_half_d, top, ridge_h, trim)

    # 中央钟楼（跨在屋脊上）：塔身 + 尖顶 + 表盘
    tower_half = min(1.6, half_w * 0.20)
    tower_top = top + ridge_h
    tower_height = min(5.5, station.height * 0.7)
    _local_box(builder, x, z, heading, -tower_half, tower_half, base_y,
               tower_top + tower_height, -tower_half, tower_half, wall,
               top_color=wall)
    _add_pyramid(builder, x, z, heading, 0.0, tower_top + tower_height, 0.0,
                 tower_half + 0.25, min(3.0, tower_height * 0.5), roof)

    # 表盘：朝正面（-z）贴一块圆盘 —— 低模下用一块浅色方板 + 深色刻度代替
    face_half = min(0.85, tower_half * 0.62)
    _local_quad(builder, x, z, heading,
                ((-face_half, tower_top + tower_height * 0.62, -tower_half - 0.02),
                 (face_half, tower_top + tower_height * 0.62, -tower_half - 0.02),
                 (face_half, tower_top + tower_height * 0.62 + 2.0 * face_half,
                  -tower_half - 0.02),
                 (-face_half, tower_top + tower_height * 0.62 + 2.0 * face_half,
                  -tower_half - 0.02)),
                style.STATION_CLOCK_FACE_COLOR, normal_local=(0.0, 0.0, -1.0))
    _local_box(builder, x, z, heading, -face_half * 0.12, face_half * 0.12,
               tower_top + tower_height * 0.62 + 0.1 * face_half,
               tower_top + tower_height * 0.62 + 2.0 * face_half - 0.1 * face_half,
               -tower_half - 0.04, -tower_half + 0.02,
               style.STATION_CLOCK_HAND_COLOR, top_color=style.STATION_CLOCK_HAND_COLOR)

    # 正面：一扇大门（居中）+ 两侧成排拱窗
    door_w, door_h = 2.4, 3.4
    _local_box(builder, x, z, heading, -door_w * 0.5, door_w * 0.5,
               base_y, base_y + door_h, -half_d - 0.03, -half_d + 0.12,
               trim, top_color=trim)
    window_y0 = base_y + 2.2
    window_y1 = min(window_y0 + 1.5, top - 0.4)
    for slot in (-1, 1):
        for k in range(1, 4):
            wx = slot * (half_w * 0.35 + k * half_w * 0.22)
            if abs(wx) > half_w - 1.2:
                continue
            _local_box(builder, x, z, heading, wx - 0.7, wx + 0.7,
                       window_y0, window_y1, -half_d - 0.02, -half_d + 0.08,
                       glass, top_color=glass)
    # 山墙两端各一扇圆窗
    for side in (-1.0, 1.0):
        _local_box(builder, x, z, heading, side * half_w - 0.06, side * half_w + 0.06,
                   base_y + station.height * 0.55, base_y + station.height * 0.55 + 1.2,
                   -1.0, 1.0, glass, top_color=glass)


def _add_station_modern(builder: MeshBuilder, station: Station) -> None:
    """新式现代化车站：低矮平直体量 + 出挑平檐 + 整面玻璃幕墙 + 钢柱。

    正面（局部 ``-z``）几乎全是玻璃，屋顶是一片出挑的平檐，几根钢柱撑在正面。
    """
    wall = style.STATION_MODERN_WALL_COLOR
    roof = style.STATION_MODERN_ROOF_COLOR
    glass = style.STATION_MODERN_GLASS_COLOR
    steel = style.STATION_MODERN_STEEL_COLOR
    base_y = style.GROUND_Y + style.SCENERY_LIFT
    half_w, half_d = station.width * 0.5, station.depth * 0.5
    top = base_y + station.height
    x, z, heading = station.x, station.z, station.heading

    # 主体（玻璃幕墙朝正面，其余三面是浅灰实墙）
    _local_box(builder, x, z, heading, -half_w, half_w, base_y, top,
               -half_d, half_d, wall, top_color=wall)

    # 整面玻璃幕墙：正面从离地 0.6 m 到近顶
    glass_y0 = base_y + 0.6
    glass_y1 = top - 0.4
    if glass_y1 > glass_y0:
        _local_box(builder, x, z, heading, -half_w + 0.3, half_w - 0.3,
                   glass_y0, glass_y1, -half_d - 0.02, -half_d + 0.10,
                   glass, top_color=glass)

    # 出挑平檐：一片比主体宽一圈的扁板
    overhang = 1.6
    slab = 0.35
    _local_box(builder, x, z, heading,
               -half_w - overhang, half_w + overhang, top, top + slab,
               -half_d - overhang, half_d + overhang, roof, top_color=roof)

    # 正面几根钢柱撑住平檐（从地面到檐底）
    column_half = 0.14
    for k in range(max(1, int(station.width // 5.0))):
        cx = -half_w + station.width * (k + 0.5) / max(1, int(station.width // 5.0))
        builder.add_cylinder(
            _local_to_world(x, z, heading, cx, base_y, -half_d - overhang * 0.5),
            _local_to_world(x, z, heading, cx, top, -half_d - overhang * 0.5),
            column_half, 6, steel)
    # 一道横梁
    _local_box(builder, x, z, heading, -half_w - overhang, half_w + overhang,
               top - 0.12, top, -half_d - overhang * 0.5 - 0.2,
               -half_d - overhang * 0.5 + 0.2, steel, top_color=steel)


def _add_station_platform(builder: MeshBuilder, station: Station) -> None:
    """车站自带站台：在正前方（局部 ``-z``）搭一条与正立面等长的站台。"""
    off = station.platform_offset()
    length = station.platform_length()
    lamps = max(1, int(length // 12.0))
    platform = Platform(
        x=station.x + math.sin(station.heading) * off,
        z=station.z - math.cos(station.heading) * off,
        length=length, width=_STATION_PLATFORM_WIDTH, height=0.55,
        heading=station.heading, seed=station.seed,
        track_side=-1.0, lamps=lamps,
    )
    _add_platform(builder, platform)


def _add_platform(builder: MeshBuilder, platform: Platform) -> None:
    """站台：台体 + 靠轨道一侧的黄色安全线 + 灯柱。"""
    base_y = style.GROUND_Y + style.SCENERY_LIFT
    half_l, half_w = platform.length * 0.5, platform.width * 0.5
    top = base_y + platform.height
    x, z, heading = platform.x, platform.z, platform.heading
    side = -1.0 if platform.track_side < 0 else 1.0

    _local_box(builder, x, z, heading, -half_l, half_l, base_y, top,
               -half_w, half_w, style.PLATFORM_COLOR,
               top_color=style.PLATFORM_TOP_COLOR)

    # 安全线：贴着靠轨道那一条边，抬高 5 mm 免得与台面共面闪烁
    edge = side * half_w
    strip = 0.34
    _local_quad(builder, x, z, heading,
                ((-half_l, top + 0.005, edge - side * strip),
                 (half_l, top + 0.005, edge - side * strip),
                 (half_l, top + 0.005, edge),
                 (-half_l, top + 0.005, edge)),
                style.PLATFORM_EDGE_COLOR, normal_local=(0.0, 1.0, 0.0))

    # 灯柱立在**外侧**（背向轨道那一侧），所以永远不会侵限
    lamps = max(0, platform.lamps)
    for index in range(lamps):
        lamp_x = -half_l + platform.length * (index + 0.5) / lamps
        lamp_z = -side * (half_w - 0.55)
        post_h = 4.4
        builder.add_cylinder(
            _local_to_world(x, z, heading, lamp_x, base_y + platform.height - 0.1,
                            lamp_z),
            _local_to_world(x, z, heading, lamp_x, base_y + platform.height + post_h,
                            lamp_z),
            0.10, 6, style.LAMP_POST_COLOR)
        _local_box(builder, x, z, heading,
                   lamp_x - 0.28, lamp_x + 0.28,
                   base_y + platform.height + post_h - 0.28,
                   base_y + platform.height + post_h,
                   lamp_z - 0.22, lamp_z + 0.22,
                   style.LAMP_HEAD_COLOR, top_color=style.LAMP_HEAD_COLOR)


def build_scenery(scenery: Scenery, *, name: str = "scenery") -> NodePath:
    """把一份布景烘成**一个** :class:`NodePath`（一个 GeomNode = 一次绘制调用）。

    布景全是静态几何，逐件建节点只会白白多出几百次绘制调用；合成一个之后，四个
    场景的布景加起来也只是一个节点。
    """
    builder = MeshBuilder(name)
    for meadow in scenery.meadows:
        _add_meadow(builder, meadow)
    for river in scenery.rivers:
        _add_river(builder, river)
    for lake in scenery.lakes:
        _add_lake(builder, lake)
    for peak in scenery.peaks:
        _add_peak(builder, peak)
    for tree in scenery.trees:
        _add_tree(builder, tree)
    for house in scenery.houses:
        _add_house(builder, house)
    for station in scenery.stations:
        if station.style == STATION_MODERN:
            _add_station_modern(builder, station)
        else:
            _add_station_classical(builder, station)
        if station.with_platform:
            _add_station_platform(builder, station)
    for platform in scenery.platforms:
        _add_platform(builder, platform)
    node = builder.build()
    node.setName(name)
    return node


# --------------------------------------------------------------------------- #
# 手工摆放布景的序列化（存档里记「用户自己摆的那几件」）
# --------------------------------------------------------------------------- #
# 程序化场景的布景按预设 key 重算（见 scenes/__init__.py），用户手工加的几件则
# 需要逐件记下来。这里只序列化**可手工摆放**的那几类（房 / 车站 / 山 / 树），
# 水面、草地、站台这类纯程序化产物不在其列。

#: 可手工摆放的布景类别 → 对应 ``Scenery`` 字段名。
_PLACEABLE_FIELDS = {
    "houses": House,
    "stations": Station,
    "peaks": Peak,
    "trees": Tree,
}

#: ``Scenery`` 的全部字段名（整份拷贝 / 删一件时都要照它列一遍）。
ALL_SCENERY_FIELDS = (
    "peaks", "rivers", "lakes", "meadows",
    "trees", "houses", "stations", "platforms",
)


def placeable_items(scenery: Scenery) -> Iterator[tuple[str, int, object]]:
    """枚举一份布景里**可手工摆放**的物件：``(类别字段, 该类别内下标, 物件)``。"""
    for key in _PLACEABLE_FIELDS:
        for index, item in enumerate(getattr(scenery, key)):
            yield (key, index, item)


def without_item(scenery: Scenery, key: str, index: int) -> Scenery:
    """去掉某一类里第 ``index`` 件，返回一份新布景（其余原样不动）。"""
    data = {field: tuple(getattr(scenery, field)) for field in ALL_SCENERY_FIELDS}
    items = list(data[key])
    items.pop(index)
    data[key] = tuple(items)
    return Scenery(**data)


def scenery_to_dict(scenery: Scenery) -> dict:
    """把可手工摆放的布景序列化成 JSON 友好的字典（键 = 类别，值 = 逐件字段表）。"""
    return {
        key: [asdict(item) for item in getattr(scenery, key)]
        for key in _PLACEABLE_FIELDS
    }


def scenery_from_dict(data: dict | None) -> Scenery:
    """反向重建：``scenery_to_dict`` 的逆（未知键 / 缺失键都当空处理）。"""
    data = data or {}
    result: dict[str, tuple] = {}
    for key, cls in _PLACEABLE_FIELDS.items():
        records = data.get(key, ())
        result[key] = tuple(cls(**record) for record in records)
    return Scenery(**result)


def field_names(cls) -> tuple[str, ...]:
    """某类布景物的字段名（编辑器摆布景时按它造一个实例）。"""
    return tuple(f.name for f in fields(cls))
