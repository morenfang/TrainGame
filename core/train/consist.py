"""列车编组定义：车型、编组、以及整列车的派生量。

对应概要设计 §5.7 / §5.8。两个刻意的设计选择：

**1. 几何按沙盘缩尺，质量与功率保持真实。**
   动力学是沿轨道的一维问题，只与 v / 牵引 / 阻力 / 质量 / 坡度有关，与视觉
   尺度无关。保留真实质量与功率，才能让「绿皮车起步慢、复兴号起步快」真正
   体现出来；若把质量也按体积缩到 0.064 倍，加速时间就毫无参考意义了。

**2. 黏着质量 = 所有动力车的质量之和。**
   可达牵引力上限 = ``adhesion * 黏着质量 * g``。这一条自动区分了两种牵引方式：

   * 绿皮车「动力集中」—— 只有 1 台机车 138 t 提供黏着；
   * 和谐号 / 复兴号「动力分散」—— 8 节车几乎全部提供黏着。

   两者在起步时的牵引力上限相差数倍，这正是设计文档里说的差别所在。

**3. 车体外观是数据，而且能被 headless 单测量出来。**
   三种车型的差别不能只靠颜色：远看一列车的辨识度主要来自侧影。所以每个车型都
   带一份 :class:`CarShape`（半断面轮廓 + 头部收敛曲线 + 车窗/车门/色带），
   数据里**只写不可能推导出来的东西** —— 半宽、车高、门窗位置全是推论。见
   :class:`CarShape` 那一节。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from core import paths

#: 重力加速度（m/s²）
G = 9.80665

#: 数据目录（项目根 / data）。打包成 exe 后 ``data/`` 跟着程序走（见 ``core.paths``）。
DATA_DIR = paths.data_dir()

#: 门窗之间、门窗与鼻锥之间必须留出的最小净空（米）。
#: 净空为零并不会让几何退化，只会让两个开口贴在一起 —— 那一定是数据写错了。
FEATURE_MARGIN = 0.10

#: 鼻锥的 ``ends`` 取值：两端都有 / 只在 +x 端（编组时要镜像另一端的那节车）。
_NOSE_ENDS = ("both", "forward")

#: 车顶设备的取值。**写进数据而不是靠车型 id 猜** —— "哪节车顶上有受电弓"是
#: 事实，不是能从断面推出来的东西；用 id 前缀去猜迟早会猜错。
_ROOF_GEAR = ("none", "ac", "pantograph", "fans")


class TrainCatalogError(ValueError):
    """列车目录数据有误。"""


# --------------------------------------------------------------------------- #
# 车体形状：断面 / 头型 / 门窗 / 色带
# --------------------------------------------------------------------------- #
# 这一节全是**纯几何数据**，不含任何颜色与渲染参数（颜色在 livery 里，渲染常量在
# render/style.py 里）。于是"一节车长什么样"可以被 headless 单测直接量出来，
# 不需要开窗口、更不需要显卡。
#
# 三个刻意的取舍：
#
# **1. 断面只写一半。** 真实车体左右对称，写一半再镜像，就没有"左边比右边宽 3 mm"
#    这类数据错误的可能，数据量也减半。半断面从**底面中线**起、到**顶面中线**止；
#    横向坐标为正、高度单调不降 —— 这两条一起保证了镜像出来的闭合轮廓不自交。
#
# **2. 没有"半宽 / 车高 / 地板高"这类标量。** 它们全都是断面的推论（见
#    :meth:`CarShape.half_width` 等）。同一个尺寸写两遍，就一定会有一天对不上。
#
# **3. 门窗沿车长的位置是「算」出来的，不是「写」出来的。** 数据只给"几扇、多宽、
#    多大间距"，具体摆在哪由 :meth:`CarSpec.window_bands` 在"鼻锥与车门之间剩下的
#    连续空档"里居中排布。于是"窗切进了门里"这种错误在构建期就抛错，
#    而不是等到出图才发现。


@dataclass(frozen=True, slots=True)
class Nose:
    """头部收敛段（"鼻锥"）：横断面沿纵向收缩到 :attr:`end_scale` 的那一段。

    缩放曲线是::

        scale(u) = end_scale + (1 - end_scale) * t ** power,   t = u / length

    ``u`` 是**距车端面**的距离，所以 ``u = 0`` 在端面上、``u >= length`` 已经
    是完整断面。这条曲线同时定下了三款车头的性格：

    * ``power < 1`` —— 断面一上来就迅速张开，端部显得**钝圆**（客车端部倒角）；
    * ``power > 1`` —— 断面长时间保持很细、最后才张开，端部显得**尖长**（CRH380A）；
    * ``end_scale`` —— 端面本身还剩多少。真实车头不是数学意义上的一个点，留一点
      端面反而更像铸造出来的鼻锥；写成 0 会得到一个刀片状的退化端面。

    ``pivot_height`` 是缩放的**支点高度**：断面是"关于这条水平线"收缩的，所以鼻尖
    自然落在支点附近，而不是被拉到车顶或车底。这正是一列动车组"低伏前伸"的来源
    —— 车顶线与车底线同时向腰线收拢。
    """

    length: float
    end_scale: float
    power: float
    pivot_height: float
    #: ``"both"`` = 两端同型；``"forward"`` = 只有 +x 端是头型，另一端是平的贯通端。
    ends: str = "both"

    def __post_init__(self) -> None:
        if not math.isfinite(self.length) or self.length < 0.0:
            raise TrainCatalogError(
                f"车体形状：鼻锥长度必须是非负有限数，得到 {self.length!r}"
            )
        if not math.isfinite(self.end_scale) or not (0.0 < self.end_scale <= 1.0):
            raise TrainCatalogError(
                f"车体形状：鼻锥端面缩放必须在 (0, 1] 内，得到 {self.end_scale!r}"
            )
        if self.length == 0.0 and self.end_scale != 1.0:
            raise TrainCatalogError(
                f"车体形状：鼻锥长度为零却写着端面缩放 {self.end_scale} —— "
                "零长度的收敛段不可能有收缩比"
            )
        if not math.isfinite(self.power) or self.power <= 0.0:
            raise TrainCatalogError(
                f"车体形状：鼻锥收敛指数必须为正，得到 {self.power!r}"
            )
        if not math.isfinite(self.pivot_height):
            raise TrainCatalogError(
                f"车体形状：鼻锥支点高度必须是有限数，得到 {self.pivot_height!r}"
            )
        if self.ends not in _NOSE_ENDS:
            raise TrainCatalogError(
                f"车体形状：鼻锥 ends 必须是 {_NOSE_ENDS} 之一，得到 {self.ends!r}"
            )

    def has_nose_at(self, which: str) -> bool:
        """``which`` ∈ {``"start"``, ``"end"``}，对应车体局部坐标的 -x / +x 两端。"""
        if which == "start":
            return self.length > 0.0 and self.ends == "both"
        if which == "end":
            return self.length > 0.0 and self.ends in ("both", "forward")
        raise ValueError(f"which 必须是 'start' 或 'end'，得到 {which!r}")

    def scale_at(self, distance_from_end: float) -> float:
        """距车端面 ``distance_from_end`` 米处的断面缩放（1.0 = 完整断面）。"""
        if self.length <= 0.0:
            return 1.0
        t = min(1.0, max(0.0, distance_from_end / self.length))
        return self.end_scale + (1.0 - self.end_scale) * t ** self.power


@dataclass(frozen=True, slots=True)
class Band:
    """沿车体高度方向的一个区间 ``[y0, y1]``（高度以**轨面**为 0）。"""

    y0: float
    y1: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.y0) and math.isfinite(self.y1)):
            raise TrainCatalogError(f"车体形状：区间上下沿必须是有限数，得到 {self!r}")
        if self.y1 <= self.y0:
            raise TrainCatalogError(
                f"车体形状：区间上沿必须高于下沿，得到 [{self.y0}, {self.y1}]"
            )

    def contains(self, y: float) -> bool:
        """高度 ``y`` 是否落在这个区间内（含端点）。"""
        return self.y0 <= y <= self.y1

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def overlaps(self, other: "Band") -> bool:
        """两个区间是否有**正长度**的交集。

        相接不算重叠 —— 色带紧贴着车窗下沿是完全正常的画法；反过来把它算成
        重叠，数据作者就只好去凑一个谁也说不出理由的小缝隙。
        """
        return min(self.y1, other.y1) - max(self.y0, other.y0) > 0.0


@dataclass(frozen=True, slots=True)
class Windows:
    """车窗带：高度范围固定，沿车长均匀排布。"""

    y: Band
    #: 每一侧的车窗数量（0 = 没有车窗）。
    count: int
    #: 单扇窗的纵向宽度（米）。
    width: float
    #: 相邻两扇窗**中心距**（米）。必须大于 :attr:`width`，否则两扇窗会连成一片。
    pitch: float

    def __post_init__(self) -> None:
        if self.count < 0:
            raise TrainCatalogError(f"车体形状：车窗数量不能为负，得到 {self.count}")
        if self.count == 0:
            return
        if not (math.isfinite(self.width) and self.width > 0.0):
            raise TrainCatalogError(f"车体形状：车窗宽度必须为正，得到 {self.width!r}")
        if not math.isfinite(self.pitch) or self.pitch <= self.width:
            raise TrainCatalogError(
                f"车体形状：车窗中心距 {self.pitch} 必须大于窗宽 {self.width}，"
                "否则相邻两扇窗会连成一片"
            )

    @property
    def span(self) -> float:
        """整条窗带占用的纵向长度（米）。"""
        if self.count <= 0:
            return 0.0
        return (self.count - 1) * self.pitch + self.width


@dataclass(frozen=True, slots=True)
class Doors:
    """车门：每端各 ``per_end`` 扇，从车端往车内退 :attr:`inset`。"""

    y: Band
    #: 每一端（也即每一侧）的车门数量，目前只允许 0 或 1。
    per_end: int
    width: float
    #: 车门近端边缘距车端面的距离（米）——**不含**鼻锥占掉的那一段。
    inset: float

    def __post_init__(self) -> None:
        if self.per_end not in (0, 1):
            raise TrainCatalogError(
                f"车体形状：每端车门数只支持 0 或 1，得到 {self.per_end}"
            )
        if self.per_end == 0:
            return
        if not (math.isfinite(self.width) and self.width > 0.0):
            raise TrainCatalogError(f"车体形状：车门宽度必须为正，得到 {self.width!r}")
        if not math.isfinite(self.inset) or self.inset < 0.0:
            raise TrainCatalogError(f"车体形状：车门内缩量不能为负，得到 {self.inset!r}")


@dataclass(frozen=True, slots=True)
class CarShape:
    """一节车的**全部外观几何**：断面 + 头型 + 门窗 + 色带。"""

    #: 半断面：从底面中线 ``(0, 地板高)`` 到顶面中线 ``(0, 车顶高)`` 的折线。
    #: 中段的横向坐标必须为正、高度必须单调不降。
    profile: tuple[tuple[float, float], ...]
    nose: Nose
    windows: Windows | None = None
    doors: Doors | None = None
    #: 车身色带（绿皮车的黄腰线、复兴号的红飘带）占据的高度区间。
    stripe: Band | None = None

    # ------------------------------------------------------------ 自己就能查

    def __post_init__(self) -> None:
        _check_profile(self.profile)
        low, high = self.floor_height, self.roof_height
        if not low < self.nose.pivot_height < high:
            raise TrainCatalogError(
                f"车体形状：鼻锥支点高度 {self.nose.pivot_height} 必须落在断面范围内 "
                f"({low}, {high})，否则鼻尖会被压到车顶或车底之外"
            )
        for name, band in (("车窗", self.windows.y if self.windows else None),
                           ("车门", self.doors.y if self.doors else None),
                           ("色带", self.stripe)):
            if band is None:
                continue
            if band.y0 < low or band.y1 > high:
                raise TrainCatalogError(
                    f"车体形状：{name}高度区间 [{band.y0}, {band.y1}] 超出了断面范围 "
                    f"[{low}, {high}]"
                )

    # ------------------------------------------------------------ 断面推论

    @property
    def half_width(self) -> float:
        """半宽（米）—— 断面里最宽的那一点，不另写数据。"""
        return max(lateral for lateral, _ in self.profile)

    @property
    def floor_height(self) -> float:
        """车体底面（地板下缘）距轨面的高度。"""
        return min(height for _, height in self.profile)

    @property
    def roof_height(self) -> float:
        """车顶最高点距轨面的高度。"""
        return max(height for _, height in self.profile)

    @property
    def bottom_half_width(self) -> float:
        """车底平面的半宽（米）—— 也就是「地板那一个高度上的横向范围」。

        真实车体从来不是个矩形：底面比腰线窄、两侧再向外鼓出去。下面是车下
        设备箱能占多宽的唯一依据，所以它得是推论而不是又一个手写标量。
        """
        floor = self.floor_height
        return max(
            lateral for lateral, height in self.profile
            if abs(height - floor) < 1e-9
        )

    @property
    def shoulder_height(self) -> float:
        """腰线上沿（米）—— 断面上**最后一次**达到最大半宽的高度。

        这一条把「侧墙」与「车顶弧」分开了：断面高度单调不降、横向先增后减，
        所以最大半宽之上的那些点必然在收顶。车顶该刷成灰白色还是车身色，
        靠的就是这条线 —— 而不是再写一个「车顶从多高开始」的数据。
        """
        widest = self.half_width
        return max(
            height for lateral, height in self.profile
            if abs(lateral - widest) < 1e-9
        )

    def full_section(self) -> tuple[tuple[float, float], ...]:
        """把半断面镜像成**闭合全断面**（逆时针），可直接交给
        :func:`render.mesh.ring_from_section` 放样。

        所有断面环的点数都由它统一决定，因此车身段与鼻锥段能一一对应地缝合。
        """
        return _mirror(self.profile)

    # ------------------------------------------------------ 涂装分区（几何侧）

    def band_edges(self) -> tuple[float, ...]:
        """所有「涂装分区边界」的高度：车窗 / 车门 / 色带的上下沿。

        这些高度**必须**出现在断面点里，车身才能被切成「每块四边形只属于一个
        分区」。否则一块跨在边界上的四边形只能整块取一个颜色，色带的边界就会
        随着断面点的疏密随机漂移 —— 而这些边界是数据里明确写出来的，不该漂。
        """
        edges: list[float] = []
        for band in self._bands():
            edges.extend((band.y0, band.y1))
        return tuple(sorted(set(edges)))

    def split_profile(self) -> tuple[tuple[float, float], ...]:
        """半断面，但在 :meth:`band_edges` 的每个高度上都插了插值点。

        插入点由它所在的那一段断面**线性插值**得到，也就是"色带上沿落在侧壁上
        的那一点"。于是色带的高度永远精确等于数据里写的那个数。
        """
        return _insert_heights(self.profile, self.band_edges())

    def split_section(self) -> tuple[tuple[float, float], ...]:
        """:meth:`split_profile` 的镜像闭合版 —— 放样真正用的那一条断面。"""
        return _mirror(self.split_profile())

    def _bands(self) -> Iterator[Band]:
        if self.windows is not None:
            yield self.windows.y
        if self.doors is not None:
            yield self.doors.y
        if self.stripe is not None:
            yield self.stripe

    # ------------------------------------------------------------ 放到车长上查

    def check_against(self, car_id: str, length: float) -> None:
        """把形状放到**一个具体车长**上校验。

        这些检查必须在 :meth:`__post_init__` 之外做，因为形状本身看不出
        "10 米的车里放不下 8 扇 1 米宽的窗" —— 那要等有了车长才知道。
        """
        def fail(message: str) -> None:
            raise TrainCatalogError(f"car type {car_id!r}: {message}")

        nose = self.nose
        if nose.ends == "both":
            if 2.0 * nose.length + FEATURE_MARGIN > length:
                fail(
                    f"两端的鼻锥各长 {nose.length} m，在一节 {length} m 的车里"
                    "会碰在一起（中间还要留 "
                    f"{FEATURE_MARGIN} m 的等截面段）"
                )
        elif nose.length + FEATURE_MARGIN > length:
            fail(f"鼻锥长 {nose.length} m，比 {length} m 的车还长")

        room = _window_room(car_id, length, nose, self.doors)
        if self.windows is not None and self.windows.count > 0:
            span = self.windows.span
            available = room[1] - room[0]
            if span + 2.0 * FEATURE_MARGIN > available:
                fail(
                    f"{self.windows.count} 扇窗（间距 {self.windows.pitch} m、"
                    f"宽 {self.windows.width} m）共占 {span:.2f} m，"
                    f"但鼻锥与车门之间只剩下 {available:.2f} m 的连续空档 "
                    f"[{room[0]:.2f}, {room[1]:.2f}]"
                )

        if self.stripe is not None and self.windows is not None:
            band = self.windows.y
            if self.stripe.overlaps(band):
                fail(
                    f"色带 [{self.stripe.y0}, {self.stripe.y1}] 与车窗 "
                    f"[{band.y0}, {band.y1}] 有 "
                    f"{min(self.stripe.y1, band.y1) - max(self.stripe.y0, band.y0):.3f} m "
                    "的重叠；两者都是车身表面的区域，重叠会让涂装的边界变成二义的"
                )


# --------------------------------------------------------------------------- #
# 形状的小工具（放这里而不是 CarShape 里，因为它们只依赖入参）
# --------------------------------------------------------------------------- #

def _check_profile(profile: Sequence[tuple[float, float]]) -> None:
    """半断面的三条不变量：起止在中线、中段横向为正、高度单调不降。"""
    if len(profile) < 4:
        raise TrainCatalogError(
            f"车体形状：半断面至少要有 4 个点（底中线、两个侧点、顶中线），"
            f"得到 {len(profile)} 个"
        )
    for point in profile:
        if len(point) != 2 or not all(math.isfinite(c) for c in point):
            raise TrainCatalogError(f"车体形状：半断面点必须是两个有限数，得到 {point!r}")
    laterals = [p[0] for p in profile]
    heights = [p[1] for p in profile]
    if laterals[0] != 0.0 or laterals[-1] != 0.0:
        raise TrainCatalogError(
            f"车体形状：半断面必须从底面中线 (0, ...) 起、到顶面中线 (0, ...) 止，"
            f"得到起点横向 {laterals[0]}、终点横向 {laterals[-1]}"
        )
    if any(lateral <= 0.0 for lateral in laterals[1:-1]):
        raise TrainCatalogError(
            "车体形状：半断面中段的横向坐标必须为正（中线上的点才允许是 0），"
            "否则镜像之后轮廓会自交"
        )
    for index in range(len(heights) - 1):
        if heights[index + 1] < heights[index] - 1e-9:
            raise TrainCatalogError(
                f"车体形状：半断面高度必须单调不降，但第 {index} 个点 "
                f"(y={heights[index]}) 高于第 {index + 1} 个点 (y={heights[index + 1]})"
                " —— 单调不降正是「镜像后轮廓不自交」的充分条件"
            )


def _mirror(profile: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """把半断面镜像成闭合全断面（逆时针，从底面中线出发、回到顶面中线）。

    镜像时**去掉**中线上的两个端点再反向 —— 它们自己就是自己的镜像，留着会在
    闭合环里出现两次，放样时就是一个零长度边。
    """
    left = [(-lateral, height) for lateral, height in reversed(profile[1:-1])]
    return tuple(profile) + tuple(left)


def _insert_heights(profile: Sequence[tuple[float, float]],
                    heights: Iterable[float]) -> tuple[tuple[float, float], ...]:
    """在断面折线的指定高度处插入插值点（横向由相邻两点线性插值）。

    每个高度只可能落在**一段**折线上：断面高度单调不降，所以各段的高度区间互不
    重叠（最多在端点相接，而那种情况已经被"高度已存在"这一支处理掉了）。
    """
    points = list(profile)
    for target in sorted(set(heights)):
        if any(abs(height - target) < 1e-9 for _, height in points):
            continue
        for index in range(len(points) - 1):
            low = points[index][1]
            high = points[index + 1][1]
            if not low < target < high:
                continue
            t = (target - low) / (high - low)
            lateral = points[index][0] + (points[index + 1][0] - points[index][0]) * t
            points.insert(index + 1, (lateral, target))
            break
    return tuple(points)


def _window_room(car_id: str, length: float, nose: Nose,
                 doors: Doors | None) -> tuple[float, float]:
    """鼻锥与车门之间**唯一的那段连续空档**（车窗只能摆在这里）。

    做法是收集所有"被占掉"的区间，再从车体中心向两边走：完全落在中心左侧的
    把左界往右推、完全落在右侧的把右界往左推。若某个障碍横跨车体中心，
    说明中间没有连续空档 —— 这本身就是数据错误。
    """
    blocked: list[tuple[float, float]] = []
    if nose.has_nose_at("start"):
        blocked.append((0.0, nose.length))
    if nose.has_nose_at("end"):
        blocked.append((length - nose.length, length))
    if doors is not None and doors.per_end > 0:
        for at_start in (True, False):
            has_nose = nose.has_nose_at("start" if at_start else "end")
            offset = doors.inset + (nose.length if has_nose else 0.0)
            if at_start:
                blocked.append((offset, offset + doors.width))
            else:
                blocked.append((length - offset - doors.width, length - offset))

    centre = 0.5 * length
    lo, hi = 0.0, length
    for start, stop in sorted(blocked):
        if stop <= centre:
            lo = max(lo, stop)
        elif start >= centre:
            hi = min(hi, start)
        else:
            raise TrainCatalogError(
                f"car type {car_id!r}: 有障碍区间 [{start:.2f}, {stop:.2f}] 横跨车体"
                f"中心 {centre:.2f}，中间没有任何连续空档可以排车窗"
            )
    if hi - lo <= 0.0:
        raise TrainCatalogError(
            f"car type {car_id!r}: 鼻锥与车门把车体占满了，没有空档可以排车窗"
        )
    return lo, hi


@dataclass(frozen=True, slots=True)
class CarSpec:
    """一种车型（一节车）的物理与外观参数。"""

    id: str
    name: str
    #: 车长（车钩面到车钩面），沙盘缩尺后的值
    length: float
    #: 转向架中心到车体中心的距离（两转向架中心距 = 2 * 该值）
    bogie_half_spacing: float
    mass_kg: float
    #: 是否自带动力（决定黏着质量与功率分布）
    powered: bool
    #: 涂装：主色 / 色带 / 车顶（render 层用）
    livery: Mapping[str, str] = field(default_factory=dict)
    #: 车顶设备：``"none"`` / ``"ac"``（空调机组）/ ``"pantograph"``（受电弓）/
    #: ``"fans"``（内燃机车的冷却风扇）。
    roof_gear: str = "none"
    #: 车体几何：断面 / 头型 / 门窗 / 色带（见 :class:`CarShape`）。
    #: 允许为空 —— 程序里临时造一节用于算力学的车不需要外观；但
    #: ``data/trains.json`` 里每个车型都必须给，:meth:`TrainCatalog.validate`
    #: 会盯着这一条。
    shape: CarShape | None = None

    def __post_init__(self) -> None:
        if 2.0 * self.bogie_half_spacing > self.length:
            raise TrainCatalogError(
                f"car type {self.id!r}: bogie spacing "
                f"({2 * self.bogie_half_spacing:.3f} m) exceeds car length "
                f"({self.length:.3f} m)"
            )
        if self.mass_kg <= 0.0:
            raise TrainCatalogError(f"car type {self.id!r}: mass must be positive")
        if self.roof_gear not in _ROOF_GEAR:
            raise TrainCatalogError(
                f"car type {self.id!r}: roof_gear 必须是 {_ROOF_GEAR} 之一，"
                f"得到 {self.roof_gear!r}"
            )
        if self.shape is not None:
            self.shape.check_against(self.id, self.length)

    # ------------------------------------------------------------ 外观几何

    @property
    def nose(self) -> Nose | None:
        return self.shape.nose if self.shape is not None else None

    def nose_bands(self) -> tuple[tuple[float, float], ...]:
        """两端收敛段占用的纵向区间 ``(起, 止)``（车体局部坐标，0 = 一端面）。"""
        shape = self.shape
        if shape is None or shape.nose.length <= 0.0:
            return ()
        bands: list[tuple[float, float]] = []
        if shape.nose.has_nose_at("start"):
            bands.append((0.0, shape.nose.length))
        if shape.nose.has_nose_at("end"):
            bands.append((self.length - shape.nose.length, self.length))
        return tuple(bands)

    def door_bands(self) -> tuple[tuple[float, float], ...]:
        """车门的纵向区间；有鼻锥的那一端，车门整体向内让开鼻锥。"""
        shape = self.shape
        if shape is None or shape.doors is None or shape.doors.per_end <= 0:
            return ()
        doors, nose = shape.doors, shape.nose
        bands: list[tuple[float, float]] = []
        for at_start in (True, False):
            offset = doors.inset + (
                nose.length if nose.has_nose_at("start" if at_start else "end") else 0.0
            )
            if at_start:
                bands.append((offset, offset + doors.width))
            else:
                bands.append((self.length - offset - doors.width,
                              self.length - offset))
        return tuple(bands)

    def window_room(self) -> tuple[float, float] | None:
        """车窗唯一可以落进去的那段连续空档；没有外观数据时返回 ``None``。"""
        if self.shape is None:
            return None
        return _window_room(self.id, self.length, self.shape.nose, self.shape.doors)

    def window_bands(self) -> tuple[tuple[float, float], ...]:
        """车窗的纵向区间：在 :meth:`window_room` 里**居中**排布。

        居中而不是"从某端开始数"：车门在两端是对称的，居中排出来的窗带自然
        也是对称的，不用额外写一个起点。
        """
        shape = self.shape
        if shape is None or shape.windows is None or shape.windows.count <= 0:
            return ()
        windows = shape.windows
        room = self.window_room()
        assert room is not None                 # check_against 已经保证存在
        start = room[0] + 0.5 * ((room[1] - room[0]) - windows.span)
        return tuple(
            (start + i * windows.pitch, start + i * windows.pitch + windows.width)
            for i in range(windows.count)
        )

    def feature_bands(self) -> tuple[tuple[float, float], ...]:
        """车身上**所有纵向分界线**的并集（鼻锥 / 车门 / 车窗），已排序去重。

        这是"放样网格至少要在这些位置切一刀"的依据。少切一刀，某个四边形就会
        跨在车窗的边沿上，那一块的玻璃颜色只能整块取一种 —— 于是窗口长度
        随着网格密度漂移，而窗口长度是数据里写死的。
        """
        bands = list(self.nose_bands())
        bands.extend(self.door_bands())
        bands.extend(self.window_bands())
        return tuple(sorted(set(bands)))


@dataclass(frozen=True, slots=True)
class TrainSpec:
    """一列完整编组（可直接投入运行）。"""

    id: str
    name: str
    cars: tuple[CarSpec, ...]
    coupling_gap: float
    power_w: float
    max_tractive_force_n: float
    max_brake_force_n: float
    adhesion: float
    #: 单位质量阻力系数 [a, b, c]：r(v) = a + b*v + c*v^2（m/s²，v 单位 m/s）
    davis: tuple[float, float, float]
    max_speed_kmh: float
    #: 速度上限的额外缩放，用来把「模型感」调慢（1.0 = 真实速度）。见概要设计 §5.8。
    speed_limit_scale: float = 1.0
    #: 紧急制动力的上限（N）。**0（缺省）= 退回常用制动力** —— 于是测试里那些合成
    #: 编组的行为一字不变，只有真在 `data/trains.json` 里写了这一项的车型才有
    #: "一把拉死"的额外制动力。见 `data/trains.json` 的 notes 与
    #: `core.train.dynamics.brake_force`。
    max_emergency_brake_force_n: float = 0.0
    spec: Mapping = field(default_factory=dict, compare=False, repr=False)

    # ---------------------------------------------------------------- 派生量

    @property
    def car_count(self) -> int:
        return len(self.cars)

    @property
    def lengths(self) -> tuple[float, ...]:
        return tuple(car.length for car in self.cars)

    @property
    def bogie_half_spacings(self) -> tuple[float, ...]:
        return tuple(car.bogie_half_spacing for car in self.cars)

    @property
    def total_mass(self) -> float:
        """整列车质量（kg）。"""
        return sum(car.mass_kg for car in self.cars)

    @property
    def adhesive_mass(self) -> float:
        """黏着质量（所有动力车的质量之和，kg）。"""
        return sum(car.mass_kg for car in self.cars if car.powered)

    @property
    def total_length(self) -> float:
        """编组首尾占用的总弧长（含车钩间隙）。"""
        if not self.cars:
            return 0.0
        return sum(self.lengths) + self.coupling_gap * (self.car_count - 1)

    def car_facings(self) -> tuple[bool, ...]:
        """每节车是否要**掉头**摆放（``True`` = 它的局部 +x 要指向列车后方）。

        「头型只朝一端」的车（CRH380A / CR400AF 的头车）排在编组**后半段**时
        必须掉头，头型才会永远朝外。规则本身只有一句话，但它不是审美问题：
        不掉头的话，整列车会一头平、一头尖，中间还冒出一个尖头。
        """
        count = self.car_count
        facings: list[bool] = []
        for index, car in enumerate(self.cars):
            nose = car.shape.nose if car.shape is not None else None
            one_ended = (nose is not None and nose.has_nose_at("end")
                         and not nose.has_nose_at("start"))
            facings.append(index * 2 >= count and one_ended)
        return tuple(facings)

    @property
    def max_speed_ms(self) -> float:
        """运行速度上限（m/s）。"""
        return self.max_speed_kmh / 3.6 * self.speed_limit_scale

    @property
    def max_traction_from_adhesion(self) -> float:
        """黏着极限能提供的最大牵引力（N）。

        ``adhesion * 黏着质量 * g`` —— 起步时牵引力的真正瓶颈。
        """
        return self.adhesion * self.adhesive_mass * G

    @property
    def effective_max_tractive_force(self) -> float:
        """实际可用的最大牵引力（N）：取「牵引能力」与「黏着极限」的小者。"""
        return min(self.max_tractive_force_n, self.max_traction_from_adhesion)

    @property
    def effective_emergency_brake_force(self) -> float:
        """紧急制动可用制动力（N）：数据里没单独给就退回常用制动力。"""
        if self.max_emergency_brake_force_n > 0.0:
            return self.max_emergency_brake_force_n
        return self.max_brake_force_n

    def brake_deceleration(self, emergency: bool = False) -> float:
        """制动减速度（m/s²）。**这是调 `data/trains.json` 时最该看的一列** ——
        制动力要和整列车质量放在一起才有意义（同样的 kN 拉 8 辆和拉 16 辆差一倍）。"""
        force = (self.effective_emergency_brake_force if emergency
                 else self.max_brake_force_n)
        return force / self.total_mass if self.total_mass else 0.0

    def power_limited_speed(self) -> float:
        """牵引力由「恒牵引力」转为「恒功率」的拐点速度（m/s）。"""
        force = self.effective_max_tractive_force
        return self.power_w / force if force > 0.0 else float("inf")

    def power_to_weight(self) -> float:
        """单位质量功率（W/kg）。跨车型比较加速性能最直接的指标。"""
        return self.power_w / self.total_mass if self.total_mass else 0.0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"TrainSpec({self.id!r}, cars={self.car_count}, "
            f"mass={self.total_mass / 1000:.0f}t, "
            f"P={self.power_w / 1000:.0f}kW, "
            f"vmax={self.max_speed_kmh:.0f}km/h)"
        )


# --------------------------------------------------------------------------- #
# 构建与目录
# --------------------------------------------------------------------------- #

def _need(mapping: Mapping, car_id: str, key: str, where: str | None = None):
    """取一个**必填**键；``where`` 只用于把报错定位到 ``nose.length`` 这种路径上。"""
    if key not in mapping:
        raise TrainCatalogError(
            f"car type {car_id!r}: 车体形状缺少 {(where or key)!r}"
            f"（该层的现有键：{sorted(mapping)}）"
        )
    return mapping[key]


def _pair(value, car_id: str, what: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TrainCatalogError(
            f"car type {car_id!r}: {what}必须是 [a, b] 两个数，得到 {value!r}"
        )
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        raise TrainCatalogError(
            f"car type {car_id!r}: {what}里有两个非数值项：{value!r}"
        ) from None


def _build_shape(car_id: str, spec: Mapping) -> CarShape:
    """把 JSON 里的 ``shape`` 解析成 :class:`CarShape`。

    分布：``profile`` / ``nose`` 必填，``windows`` / ``doors`` / ``stripe``
    可省（省掉就是"没有"）。每个错误都点名是哪个车型的哪个键，因为这类数据
    错误的唯一排查线索就是这条消息。
    """
    raw_profile = _need(spec, car_id, "profile")
    if not isinstance(raw_profile, (list, tuple)):
        raise TrainCatalogError(
            f"car type {car_id!r}: profile 必须是 [[横向, 高度], ...] 的列表，"
            f"得到 {type(raw_profile).__name__}"
        )
    profile = tuple(_pair(p, car_id, "profile 里的断面点") for p in raw_profile)

    raw_nose = _need(spec, car_id, "nose")
    if not isinstance(raw_nose, Mapping):
        raise TrainCatalogError(
            f"car type {car_id!r}: nose 必须是一个对象，得到 {type(raw_nose).__name__}"
        )
    nose = Nose(
        length=float(_need(raw_nose, car_id, "length", "nose.length")),
        end_scale=float(raw_nose.get("end_scale", 1.0)),
        power=float(raw_nose.get("power", 1.0)),
        pivot_height=float(_need(raw_nose, car_id, "pivot_height",
                                 "nose.pivot_height")),
        ends=str(raw_nose.get("ends", "both")),
    )

    windows = None
    if "windows" in spec:
        raw = spec["windows"]
        if not isinstance(raw, Mapping):
            raise TrainCatalogError(
                f"car type {car_id!r}: windows 必须是一个对象，"
                f"得到 {type(raw).__name__}"
            )
        count = int(raw.get("count", 0))
        windows = Windows(
            y=Band(*_pair(_need(raw, car_id, "y", "windows.y"), car_id, "windows.y")),
            count=count,
            width=float(raw.get("width", 0.0)),
            pitch=float(raw.get("pitch", 0.0)),
        )

    doors = None
    if "doors" in spec:
        raw = spec["doors"]
        if not isinstance(raw, Mapping):
            raise TrainCatalogError(
                f"car type {car_id!r}: doors 必须是一个对象，"
                f"得到 {type(raw).__name__}"
            )
        doors = Doors(
            y=Band(*_pair(_need(raw, car_id, "y", "doors.y"), car_id, "doors.y")),
            per_end=int(raw.get("per_end", 0)),
            width=float(raw.get("width", 0.0)),
            inset=float(raw.get("inset", 0.0)),
        )

    stripe = None
    if "stripe" in spec:
        stripe = Band(*(float(v) for v in _pair(spec["stripe"], car_id, "stripe")))

    return CarShape(profile=profile, nose=nose, windows=windows, doors=doors,
                    stripe=stripe)


def _build_car(car_id: str, spec: Mapping) -> CarSpec:
    required = ("length", "bogie_half_spacing", "mass_kg")
    missing = [key for key in required if key not in spec]
    if missing:
        raise TrainCatalogError(f"car type {car_id!r} is missing {missing}")
    return CarSpec(
        id=car_id,
        name=str(spec.get("name", car_id)),
        length=float(spec["length"]),
        bogie_half_spacing=float(spec["bogie_half_spacing"]),
        mass_kg=float(spec["mass_kg"]),
        powered=bool(spec.get("powered", False)),
        livery=dict(spec.get("livery", {})),
        roof_gear=str(spec.get("roof_gear", "none")),
        shape=_build_shape(car_id, _need(spec, car_id, "shape")),
    )


def build_train(spec: Mapping, car_types: Mapping[str, CarSpec]) -> TrainSpec:
    """由一个 train spec 构建 :class:`TrainSpec`。"""
    train_id = spec.get("id")
    if not train_id:
        raise TrainCatalogError("train spec needs an 'id'")

    formation = spec.get("formation")
    if not formation:
        raise TrainCatalogError(f"train {train_id!r} has an empty 'formation'")

    cars: list[CarSpec] = []
    for entry in formation:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise TrainCatalogError(
                f"train {train_id!r}: each formation entry must be [car_type, count]"
            )
        car_type_id, count = entry
        if car_type_id not in car_types:
            raise TrainCatalogError(
                f"train {train_id!r} references unknown car type {car_type_id!r}"
            )
        count = int(count)
        if count <= 0:
            raise TrainCatalogError(
                f"train {train_id!r}: count for {car_type_id!r} must be positive"
            )
        cars.extend([car_types[car_type_id]] * count)

    davis = spec.get("davis", [0.01, 1.0e-4, 1.2e-5])
    if len(davis) != 3:
        raise TrainCatalogError(f"train {train_id!r}: 'davis' must have 3 entries")

    result = TrainSpec(
        id=str(train_id),
        name=str(spec.get("name", train_id)),
        cars=tuple(cars),
        coupling_gap=float(spec.get("coupling_gap", 0.0)),
        power_w=float(spec.get("power_w", 0.0)),
        max_tractive_force_n=float(spec["max_tractive_force_n"]),
        max_brake_force_n=float(spec["max_brake_force_n"]),
        max_emergency_brake_force_n=float(spec.get("max_emergency_brake_force_n", 0.0)),
        adhesion=float(spec.get("adhesion", 0.3)),
        davis=(float(davis[0]), float(davis[1]), float(davis[2])),
        max_speed_kmh=float(spec.get("max_speed_kmh", 120.0)),
        speed_limit_scale=float(spec.get("speed_limit_scale", 1.0)),
        spec=dict(spec),
    )

    if result.cars and not result.adhesive_mass:
        raise TrainCatalogError(
            f"train {train_id!r} has no powered car; it could never move"
        )
    return result


@dataclass
class TrainCatalog:
    """列车目录。"""

    trains: dict[str, TrainSpec]
    car_types: dict[str, CarSpec]
    source: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "TrainCatalog":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        car_types = {
            car_id: _build_car(car_id, spec)
            for car_id, spec in payload.get("car_types", {}).items()
        }
        trains = {}
        for spec in payload.get("trains", ()):
            train = build_train(spec, car_types)
            if train.id in trains:
                raise TrainCatalogError(f"duplicate train id {train.id!r}")
            trains[train.id] = train
        return cls(trains=trains, car_types=car_types, source=path)

    @classmethod
    def builtin(cls) -> "TrainCatalog":
        return cls.load(DATA_DIR / "trains.json")

    def __getitem__(self, train_id: str) -> TrainSpec:
        try:
            return self.trains[train_id]
        except KeyError:
            raise TrainCatalogError(
                f"unknown train {train_id!r}; known: {sorted(self.trains)}"
            ) from None

    def __contains__(self, train_id: object) -> bool:
        return train_id in self.trains

    def __iter__(self) -> Iterator[TrainSpec]:
        return iter(self.trains.values())

    def __len__(self) -> int:
        return len(self.trains)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self.trains)

    def validate(self) -> list[str]:
        """自检：编组长度为正、有动力车、参数在合理范围、每个车型都有外观几何。"""
        warnings: list[str] = []
        for car_id, car in self.car_types.items():
            if car.shape is None:
                warnings.append(f"{car_id}: 缺 shape 数据，这节车画不出来")
        for train in self.trains.values():
            if train.total_length <= 0.0:
                warnings.append(f"{train.id}: 编组总长非正")
            if not train.adhesive_mass:
                warnings.append(f"{train.id}: 没有动力车")
            if train.power_w <= 0.0:
                warnings.append(f"{train.id}: 牵引功率非正")
            if train.max_tractive_force_n <= 0.0:
                warnings.append(f"{train.id}: 最大牵引力非正")
            if train.max_brake_force_n <= 0.0:
                warnings.append(f"{train.id}: 最大制动力非正")
            if 0.0 < train.max_emergency_brake_force_n < train.max_brake_force_n:
                warnings.append(
                    f"{train.id}: 紧急制动力({train.max_emergency_brake_force_n:.0f} N)"
                    f" 小于常用制动({train.max_brake_force_n:.0f} N)"
                    " —— 紧急制动必须比常用制动更强，否则那个键按下去等于没按"
                )
            if any(c <= 0.0 for c in train.davis):
                warnings.append(f"{train.id}: davis 系数应全为正")
        return warnings
