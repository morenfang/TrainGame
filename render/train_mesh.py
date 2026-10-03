"""放样生成参数化车体：断面 + 鼻锥 + 窗带 + 走行部。

车体局部坐标系（与 ``core.track.path.CarState.body_pose`` 严格对齐）
--------------------------------------------------------------------
* **+X = 车头方向**（列车前进方向），**原点在车体几何中心**；
* **+Y = 上**，``y = 0`` 在**轨面**（列车走行面，见 ``render/style.RAIL_TOP_Y``）；
* **+Z = 右侧**（``forward × up``，与 :func:`render.mesh.ring_from_section` 一致）。

于是把网格挂到 ``CarState.body_pose`` 上就直接对位：车体中心对两转向架中点、
轨面对轨面。**编组里朝后那节头车需要镜像**（``setScale(-1, 1, 1)``）；顶点色里
已经烘焙好光照，所以镜像不会把明暗翻过来。

一次放样画出整套涂装
--------------------------------------------------------------------
车身**只放样一次**，颜色逐块决定。做法是先让几何把涂装分区「切」出来：

* :meth:`core.train.consist.CarShape.split_section` 在色带 / 车窗 / 车门的高度
  边界上插了插值点 —— 于是**没有任何一块四边形跨在分区边界上**；
* :meth:`core.train.consist.CarSpec.feature_bands` 给出所有纵向分界线，网格在
  这些位置必切一刀，理由同上。

两条都成立之后，"这块面是什么颜色"就只是一次区间查询，不需要把车窗做成单独的
薄片贴上去。副作用是色带与窗口的长度**精确等于数据里写的数**，不会随网格疏密
漂移 —— 这一点在 :mod:`tests.test_train_mesh` 里是被断言过的。

鼻锥的缩放是**沿断面参数**做的，所以涂装跟着表面一起收拢：CRH380A 的蓝带在
车头上自然收成一条细线并汇聚到鼻尖，而不是被拦腰切断。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.train.consist import CarShape, CarSpec, TrainSpec
from render import style
from render.mesh import MeshBuilder

#: 鼻锥段的纵向采样步长（米）。收敛曲线在这里拐得最快，所以要密。
NOSE_STEP = 0.20
#: 等截面段的纵向采样步长（米）。这一段断面完全一样，只是为了让长条面有中间
#: 顶点（否则一块 10 m 长的四边形在光照下是一整块死色）。
BODY_STEP = 1.20
#: 两个纵向分割点靠得比这还近就并成一个（米）。避免放样出针尖一样的碎面。
STATION_EPS = 1e-4

#: 鼻锥端面缩放到比这还小的时候，端面已经盖不住车钩了，就不再画车钩。
COUPLER_MIN_NOSE_SCALE = 0.55
#: 车钩从车端面往车内伸出的长度（米）。
COUPLER_LENGTH = 0.30

#: 车下设备箱：比车底板窄这么多（米），免得从侧面看出去比车体还宽。
UNDERFRAME_INSET = 0.05
#: 车下设备箱的高度（米）。
UNDERFRAME_DEPTH = 0.34

#: 车顶设备的离顶高度与尺寸（米）。
ROOF_GEAR_SINK = 0.10
ROOF_AC_HEIGHT = 0.26
ROOF_AC_HALF_LENGTH = 0.95
ROOF_AC_HALF_WIDTH = 0.55
ROOF_FAN_RADIUS = 0.32
ROOF_FAN_HEIGHT = 0.16
ROOF_FAN_POSITIONS = (-1.35, 1.35)


# --------------------------------------------------------------------------- #
# 涂装
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class TrainColors:
    """一节车用到的全部颜色（顶点色，已解析成 0..1 的 sRGB）。

    字段名与 :func:`section_role` 返回的分区名**一一对应**（外加走行部那几项），
    所以「这块面是什么颜色」就是一次 ``for_role`` —— 中间不再有机会插进一张
    手写的对照表。
    """

    #: 与 section_role 的分区名同名，可直接 for_role 取。
    body: tuple
    band: tuple
    roof: tuple
    glass: tuple
    door: tuple
    underframe: tuple
    #: 以下是车身以外的件
    end: tuple
    bogie: tuple
    wheel: tuple
    light: tuple

    def for_role(self, role: str) -> tuple:
        """取某个车身分区对应的颜色。"""
        return getattr(self, role)

    def palette(self) -> tuple[tuple, ...]:
        """整车用到的**全部**颜色。

        车体生成器只用这里的具名颜色，绝不临时调一个出来 —— 于是"网格里每个顶点
        的颜色都等于某个具名颜色乘上该处光照"成了一条可以逐顶点断言的性质
        （见 ``tests/test_train_mesh.py``），比抽查几个点强得多。
        """
        return (self.body, self.band, self.roof, self.glass, self.door,
                self.underframe, self.end, self.bogie, self.wheel, self.light)


#: :func:`section_role` 可能返回的全部分区名。
SECTION_ROLES = ("underframe", "glass", "door", "roof", "band", "body")


def train_colors(car: CarSpec) -> TrainColors:
    """从 ``livery`` 里解出配色；没写的项全部有兜底。

    ``body`` 是唯一的必填项 —— 其余（色带、车顶）都是"没有就更简单"的东西，
    而车身色缺了就只能瞎猜，不如早点报错。
    """
    if "body" not in car.livery:
        raise ValueError(
            f"car type {car.id!r}: livery 里没有 'body'，车身主色是唯一无法兜底的项"
        )
    body = style.hex_to_color(car.livery["body"])
    band = (style.hex_to_color(car.livery["band"]) if "band" in car.livery
            else body)
    roof = (style.hex_to_color(car.livery["roof"]) if "roof" in car.livery
            else style.TRAIN_ROOF_FALLBACK)
    return TrainColors(
        body=body,
        band=band,
        roof=roof,
        glass=style.TRAIN_GLASS_COLOR,
        door=style.darken(body, style.TRAIN_DOOR_TINT),
        underframe=style.TRAIN_UNDERFRAME_COLOR,
        end=style.darken(body, style.TRAIN_END_TINT),
        bogie=style.TRAIN_BOGIE_COLOR,
        wheel=style.TRAIN_WHEEL_COLOR,
        light=style.TRAIN_HEADLIGHT_COLOR,
    )


def section_role(car: CarSpec, u: float, height: float) -> str:
    """车身**侧面**上「纵向 ``u``、高度 ``height``」这一点属于哪个涂装分区。

    ``u`` 与 :meth:`CarSpec.window_bands` 是同一套坐标（``0`` = -x 端面）。返回
    :data:`SECTION_ROLES` 之一 —— 但**不会**返回 ``"underframe"``，那片水平车底
    由 :func:`quad_role` 按几何单独认出来（它的高度和侧板下沿完全一样，光看高度
    分不开）。

    这里能这么简单，是因为几何已经把分区边界切成了网格线（断面里的
    :meth:`CarShape.split_profile`、纵向的 :meth:`CarSpec.feature_bands`）——
    于是"这一点属于哪个分区"没有边界上的二义性，也就不需要任何"取最近的分区"
    之类的模糊逻辑。

    判断顺序是刻意的：

    1. **色带最先** —— 它是横贯全车的腰带，按理就是要刷过车门的（真实绿皮车的
       黄腰带正是如此）。要是把车门排在前面，腰带上就会缺两个门洞；
    2. 车窗（真的开口）；
    3. 车门（上半截与窗带同高，那一段是门玻璃，下半截才是门板）；
    4. 车顶（腰线上沿以上）；
    5. 剩下都是车身。
    """
    shape = car.shape
    assert shape is not None
    if shape.stripe is not None and shape.stripe.contains(height):
        return "band"
    windows, doors = shape.windows, shape.doors
    if (windows is not None and windows.y.contains(height)
            and _in_any(car.window_bands(), u)):
        return "glass"
    if (doors is not None and doors.y.contains(height)
            and _in_any(car.door_bands(), u)):
        if windows is not None and height >= windows.y.y0:
            return "glass"
        return "door"
    if height >= shape.shoulder_height:
        return "roof"
    return "body"


def quad_role(car: CarSpec, u: float, low: float, high: float) -> str:
    """一块四边形属于哪个涂装分区 —— 拿它的**上下沿一起**判断。

    为什么要两个高度而不是一个中点：车底那一片**水平面**与侧板的下沿高度完全
    相同（都是 ``floor_height``），只有"上下沿都落在车底高度上"才能把它俩分开 ——
    否则车厢底面会被刷成车身色，从低角度看过去就是"车没有底"。
    """
    shape = car.shape
    assert shape is not None
    floor = shape.floor_height
    if max(low, high) <= floor + 1e-9:
        return "underframe"
    return section_role(car, u, 0.5 * (low + high))




# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

def _box(builder: MeshBuilder, x0: float, x1: float, y0: float, y1: float,
         z0: float, z1: float, color, top_color=None) -> None:
    """车体局部坐标下的轴对齐长方体。尺寸顺序无所谓，内部会摆正。"""
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    z0, z1 = min(z0, z1), max(z0, z1)
    builder.add_oriented_box((x0, y0, z0), (x1 - x0, 0.0, 0.0),
                             (0.0, y1 - y0, 0.0), (0.0, 0.0, z1 - z0),
                             color, top_color)


def _in_any(bands, u: float) -> bool:
    """``u`` 是否落在任意一个纵向区间里（含端点）。"""
    return any(start - 1e-9 <= u <= stop + 1e-9 for start, stop in bands)


# --------------------------------------------------------------------------- #
# 纵向分割
# --------------------------------------------------------------------------- #

def _stations(car: CarSpec) -> list[float]:
    """车体沿纵向的分割位置（车局部坐标，0 = 车体中心）。

    两类点：**必须**有的（车端面、鼻锥端、每一扇门窗的两条边）与**为了光滑**
    补的（按步长均分）。鼻锥段的步长更小 —— 收敛曲线在那里拐得最快。
    """
    length = car.length
    marks = {0.0, length}
    for start, stop in car.feature_bands():
        marks.add(start)
        marks.add(stop)

    ordered: list[float] = []
    for value in sorted(marks):
        if not ordered or value - ordered[-1] >= STATION_EPS:
            ordered.append(value)

    nose_bands = car.nose_bands()
    stations = [ordered[0]]
    for start, stop in zip(ordered, ordered[1:]):
        inside_nose = any(
            start >= lo - 1e-9 and stop <= hi + 1e-9 for lo, hi in nose_bands
        )
        step = NOSE_STEP if inside_nose else BODY_STEP
        pieces = max(1, int(math.ceil((stop - start) / step)))
        for k in range(1, pieces + 1):
            stations.append(start + (stop - start) * k / pieces)

    half = 0.5 * length
    return [value - half for value in stations]


def _nose_scale(shape: CarShape, length: float, u: float) -> float:
    """距「起点端面」弧长 ``u`` 处的断面缩放。

    ``u`` 是 :meth:`CarShape.check_against` / ``CarSpec.feature_bands`` 用的那套
    纵向坐标（``u = 0`` 是 **-x 端面**）。两端各有鼻锥时取更小的那个 ——
    理论上两者不会同时 < 1（车够长），但取 min 比断言"不会发生"更省事。
    """
    nose = shape.nose
    scale = 1.0
    if nose.has_nose_at("start"):
        scale = min(scale, nose.scale_at(u))
    if nose.has_nose_at("end"):
        scale = min(scale, nose.scale_at(length - u))
    return scale


# --------------------------------------------------------------------------- #
# 车身
# --------------------------------------------------------------------------- #

def _build_body(builder: MeshBuilder, car: CarSpec, colors: TrainColors) -> None:
    """放样车身，并逐块上涂装色。"""
    shape = car.shape
    assert shape is not None
    length = car.length
    half = 0.5 * length
    pivot = shape.nose.pivot_height

    stations = _stations(car)
    section = shape.split_section()
    heights = [height for _, height in section]

    rings = []
    for x in stations:
        scale = _nose_scale(shape, length, x + half)
        if scale == 1.0:
            rings.append([(x, height, lateral) for lateral, height in section])
        else:
            rings.append([
                (x, pivot + (height - pivot) * scale, lateral * scale)
                for lateral, height in section
            ])

    def classify(k: int, i: int, normal) -> tuple:
        """这块四边形属于哪个分区。k = 纵向环序，i = 断面点序。

        因为所有分区边界都已经被切成了网格线，**没有四边形跨在边界上**，
        所以拿它的中点（和上下沿）去判是精确的、不是近似。
        """
        u = 0.5 * (stations[k] + stations[k + 1]) + half
        low = heights[i]
        high = heights[(i + 1) % len(heights)]
        return colors.for_role(quad_role(car, u, low, high))

    # 端盖用「压暗一档的车身色」，两节车之间才有条分界；侧面全由 classify 接管。
    builder.add_tube(rings, colors.end, cap_start=True, cap_end=True,
                     face_color=classify)


def _add_underframe(builder: MeshBuilder, car: CarSpec,
                    colors: TrainColors) -> None:
    """车下设备箱：从车底板往下挂一条，把车身与转向架之间的空档填上。"""
    shape = car.shape
    assert shape is not None
    floor = shape.floor_height
    half_width = shape.bottom_half_width - UNDERFRAME_INSET
    if half_width <= 0.05:
        return
    reach = 0.5 * car.length - COUPLER_LENGTH
    if reach <= 0.1:
        return
    _box(builder, -reach, reach, floor - UNDERFRAME_DEPTH, floor + 0.02,
         -half_width, half_width, colors.underframe)


def _add_couplers(builder: MeshBuilder, car: CarSpec,
                  colors: TrainColors) -> None:
    """两端车钩。鼻锥尖到盖不住它的那一端就不画 —— 画了会浮在半空中。"""
    shape = car.shape
    assert shape is not None
    nose = shape.nose
    floor = shape.floor_height
    half = 0.5 * car.length
    for which, sign in (("start", -1.0), ("end", 1.0)):
        if nose.has_nose_at(which) and nose.scale_at(0.0) < COUPLER_MIN_NOSE_SCALE:
            continue
        x_out = sign * (half - 0.01)
        x_in = sign * (half - 0.01 - COUPLER_LENGTH)
        _box(builder, x_in, x_out, floor - 0.20, floor + 0.05, -0.14, 0.14,
             colors.bogie)


def _add_lights(builder: MeshBuilder, car: CarSpec, colors: TrainColors) -> None:
    """鼻锥端面上的两盏前照灯。

    位置由**鼻尖断面**推出来（缩放后的实际半宽 / 高度范围），所以钝头的东风4B
    灯装在两角、尖头的 CRH380A 灯挤在鼻尖上 —— 同一段代码两种效果。
    """
    shape = car.shape
    assert shape is not None
    nose = shape.nose
    half = 0.5 * car.length
    pivot = nose.pivot_height
    for which, sign in (("start", -1.0), ("end", 1.0)):
        if not nose.has_nose_at(which):
            continue
        scale = nose.scale_at(0.0)
        tip_half = shape.half_width * scale
        if tip_half < 0.06:
            continue
        low = pivot + (shape.floor_height - pivot) * scale
        high = pivot + (shape.roof_height - pivot) * scale
        y = low + 0.30 * (high - low)
        width = min(0.14, 0.55 * tip_half)
        for z_sign in (-1.0, 1.0):
            z = z_sign * 0.40 * tip_half
            _box(builder,
                 sign * (half - 0.05), sign * (half + 0.03),
                 y - 0.5 * width, y + 0.5 * width,
                 z - 0.5 * width, z + 0.5 * width,
                 colors.light)


def _add_roof_gear(builder: MeshBuilder, car: CarSpec,
                   colors: TrainColors) -> None:
    """车顶设备。做的是"看得出来是什么"，不是还原细节。"""
    shape = car.shape
    assert shape is not None
    gear = car.roof_gear
    if gear == "none":
        return
    top = shape.roof_height
    base = top - ROOF_GEAR_SINK
    half = 0.5 * car.length

    if gear == "ac":
        length = min(ROOF_AC_HALF_LENGTH, half - 0.9)
        if length <= 0.1:
            return
        _box(builder, -length, length, base, base + ROOF_AC_HEIGHT,
             -ROOF_AC_HALF_WIDTH, ROOF_AC_HALF_WIDTH, colors.roof)
    elif gear == "fans":
        for cx in ROOF_FAN_POSITIONS:
            if abs(cx) > half - ROOF_FAN_RADIUS - 0.2:
                continue
            builder.add_cylinder(
                (cx, base, 0.0), (cx, base + ROOF_FAN_HEIGHT, 0.0),
                ROOF_FAN_RADIUS, style.TRAIN_WHEEL_SIDES, colors.roof,
            )
    elif gear == "pantograph":
        cx = -car.bogie_half_spacing
        if abs(cx) > half - 0.9:
            cx = 0.0
        dark = colors.bogie
        # 底座
        _box(builder, cx - 0.50, cx + 0.50, base, base + 0.08, -0.36, 0.36, dark)
        # 折起来的上下臂：两条贴着车顶前伸的细杆
        _box(builder, cx - 0.95, cx + 0.95, base + 0.08, base + 0.15, -0.10, 0.10,
             dark)
        _box(builder, cx - 0.30, cx + 0.75, base + 0.15, base + 0.20, -0.07, 0.07,
             colors.roof)
        # 弓头（受流的那根横杆）
        _box(builder, cx + 0.70, cx + 0.85, base + 0.14, base + 0.22, -0.42, 0.42,
             dark)


# --------------------------------------------------------------------------- #
# 走行部
# --------------------------------------------------------------------------- #

def _add_bogies(builder: MeshBuilder, car: CarSpec, colors: TrainColors) -> None:
    """两个转向架：构架 + 车轮 + 车轴。

    车轮中心高于轨面一个半径（踏面就压在轨面上），轮对骑在 ``±GAUGE/2``。构架
    的横向半宽**必须小于车轮内侧面**，否则构架会穿进轮子 —— 这不是审美问题，
    是"两个实体相交"的硬错误，所以它是一条常量而不是一个手调的数。
    """
    gauge_half = style.GAUGE / 2.0
    wheel_r = style.TRAIN_WHEEL_RADIUS
    if wheel_r >= car.shape.floor_height - 0.05:
        wheel_r = max(0.05, car.shape.floor_height - 0.25)
    frame_inner = min(style.TRAIN_BOGIE_FRAME_HALF_WIDTH,
                      gauge_half - style.TRAIN_WHEEL_THICKNESS)
    axle_half = min(style.TRAIN_AXLE_HALF_SPACING, 0.45 * car.length)

    for sign in (-1.0, 1.0):
        centre = sign * car.bogie_half_spacing
        # 两侧构架（沿纵向跨过两根车轴）
        for z_sign in (-1.0, 1.0):
            z_outer = z_sign * frame_inner
            z_inner = z_sign * (frame_inner - 0.16)
            _box(builder, centre - axle_half - 0.22, centre + axle_half + 0.22,
                 style.TRAIN_BOGIE_FRAME_BOTTOM_Y, style.TRAIN_BOGIE_FRAME_TOP_Y,
                 z_inner, z_outer, colors.bogie)
        # 枕梁：横跨两根构架，位于两车轴中间
        _box(builder, centre - 0.24, centre + 0.24,
             style.TRAIN_BOGIE_FRAME_BOTTOM_Y, style.TRAIN_BOGIE_FRAME_TOP_Y - 0.06,
             -frame_inner, frame_inner, colors.bogie)
        for axle_sign in (-1.0, 1.0):
            axle_x = centre + axle_sign * axle_half
            # 车轴
            builder.add_cylinder((axle_x, wheel_r, -gauge_half),
                                 (axle_x, wheel_r, gauge_half),
                                 0.055, 6, colors.wheel)
            # 车轮
            for z_sign in (-1.0, 1.0):
                outer = z_sign * gauge_half
                inner = z_sign * (gauge_half - style.TRAIN_WHEEL_THICKNESS)
                builder.add_cylinder((axle_x, wheel_r, inner),
                                     (axle_x, wheel_r, outer),
                                     wheel_r, style.TRAIN_WHEEL_SIDES,
                                     colors.wheel)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class CarMesh:
    """编组里的一节车：车型 + 是否掉过头 + 已经摆在正确朝向的网格。"""

    car: CarSpec
    #: ``True`` 表示这节车的头型朝列车后方（网格已经镜像好了）。
    flipped: bool
    builder: MeshBuilder


def build_car_mesh(car: CarSpec, details: bool = True) -> MeshBuilder:
    """生成一节车的网格（**车体局部坐标**，见模块开头）。

    ``details=False`` 只放样车身本体 —— 测试量「侧影」时用它，免得转向架与
    车顶设备把包围盒撑歪。
    """
    if car.shape is None:
        raise ValueError(
            f"car type {car.id!r} 没有 shape 数据，画不出车体；"
            "检查 data/trains.json 里该车型的 shape 段"
        )
    builder = MeshBuilder(f"car_{car.id}")
    colors = train_colors(car)
    _build_body(builder, car, colors)
    if details:
        _add_underframe(builder, car, colors)
        _add_couplers(builder, car, colors)
        _add_lights(builder, car, colors)
        _add_roof_gear(builder, car, colors)
        _add_bogies(builder, car, colors)
    return builder


def build_car_mesh_oriented(car: CarSpec, flipped: bool,
                            details: bool = True) -> MeshBuilder:
    """一节车，并按 ``flipped`` 决定是否掉头。

    "掉头"就是关于 ``x = 0`` 镜像（:meth:`render.mesh.MeshBuilder.mirrored_x`），
    而不是在场景图里挂 ``setScale(-1, 1, 1)`` —— 后者会让法线与绕序一起反掉，
    顶点色里烘焙的光照就跟着错了。

    编组（:func:`build_train_mesh`）与运行时视图（``render.train_view``）都走这里，
    于是"哪些车要掉头、掉头怎么做"只有一处定义。
    """
    builder = build_car_mesh(car, details=details)
    return builder.mirrored_x() if flipped else builder


def build_train_mesh(spec: TrainSpec, details: bool = True) -> list["CarMesh"]:
    """整列编组的网格。``car_facings()`` 说要掉头的那几节已经镜像好了。"""
    return [
        CarMesh(car=car, flipped=flipped,
                builder=build_car_mesh_oriented(car, flipped, details=details))
        for car, flipped in zip(spec.cars, spec.car_facings())
    ]
