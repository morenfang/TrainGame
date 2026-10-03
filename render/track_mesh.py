"""由轨道件定义生成可见网格（道砟 / 路堤 / 枕木 / 钢轨 / 桥墙 / 车挡）。

几何从哪来
------------------------------------------------
**不手写任何坐标**。每条 route 在件局部坐标下就是一段等曲率圆弧，入口行进帧
是件局部坐标系里的 ``port(from).pose.flipped()``，弧上任意点由
:func:`core.geometry.arc_point` 给出。于是：

    frame(u) = entry.compose(Pose(dx, u * grade, dz, du))

这与 ``core.track.path`` 算世界走线用的是**同一个公式**，所以画出来的轨道和
列车实际走的中心线必然重合 —— 不存在"看着接上了、车却在飘"的可能。

截面与"右"方向
------------------------------------------------
横截面在 ``(横向, 高度)`` 平面里描述，横向取 ``forward × up``。高度直接用世界
+Y，因此**坡道上的断面依然竖直**，和真实轨道一致（轨枕不会跟着倾斜）。

多 route 件的重叠
------------------------------------------------
交叉件与道岔件有两条 route，它们的道砟顶面在交叉区会共面重叠，直接画会
z-fighting。因此每条 route 抬升 ``ROUTE_Y_BIAS * index``（3 mm，肉眼不可见），
把共面变成微小的阶梯，闪烁随之消失。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.geometry import Pose, arc_point
from core.track.catalog import PieceDef, Route
from render import style
from render.mesh import MeshBuilder, ring_from_section

#: 同一件内多条 route 之间的抬升，用来消除共面 z-fighting（米）。
ROUTE_Y_BIAS = 0.003

#: 断面端盖向内缩进，避免相邻件的端盖严格共面而闪烁（米）。
CAP_INSET = 0.002

#: 桥面离地低于这个高度就不立桥墩（米）。贴地的桥面立墩子等于埋一根柱子。
_PIER_MIN_HEIGHT = 0.55

#: 桥面件的标准跨径（米）：一件桥面按弧长的 1/4、3/4 落两根墩，也就是**两跨**。
#: 于是 40 m 的桥面墩距 20 m、20 m 的墩距 10 m。一件比 ``2 ×`` 本值更长时按同样
#: 的间距多分几档 —— 长桥面不该只有两根墩子撑着中间那一大段。
_PIER_MAX_SPAN = 40.0

#: 曲线 route 的采样步长（米）。直线只需要 2 个断面。
CURVE_SAMPLE_STEP = 1.5

#: 断面：道砟 / 路堤（闭合折线，横向向右为正）
_SECTION_BALLAST_FLAT = (
    (-style.BALLAST_TOP_HALF_WIDTH, style.BALLAST_TOP_Y),
    (style.BALLAST_TOP_HALF_WIDTH, style.BALLAST_TOP_Y),
    (style.BALLAST_BOTTOM_HALF_WIDTH, style.BALLAST_BOTTOM_Y),
    (-style.BALLAST_BOTTOM_HALF_WIDTH, style.BALLAST_BOTTOM_Y),
)


# --------------------------------------------------------------------------- #
# route 采样
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RouteSampler:
    """把一条 route 变成件局部坐标系里的连续位姿函数。"""

    piece: PieceDef
    route: Route

    @property
    def entry(self) -> Pose:
        """route 的入口**行进帧**（朝列车前进方向）。"""
        return self.piece.port(self.route.from_port).pose.flipped()

    def pose_at(self, u: float) -> Pose:
        """弧长 ``u`` 处的行进帧（u ∈ [0, route.length]）。"""
        dx, dz, du = arc_point(self.route.length, self.route.dtheta, u)
        return self.entry.compose(Pose(dx, u * self.route.grade, dz, du))

    @property
    def exit(self) -> Pose:
        return self.pose_at(self.route.length)

    def frames(self, step: float = CURVE_SAMPLE_STEP) -> list[Pose]:
        """给渲染用的断面序列。直线两帧就够（弧是线性的），曲线按步长采样。"""
        length = self.route.length
        if abs(self.route.dtheta) < 1e-9:
            count = 1
        else:
            count = max(2, int(math.ceil(length / step)))
        return [self.pose_at(length * i / count) for i in range(count + 1)]

    @property
    def tie_positions(self) -> list[float]:
        """枕木中心距起点的弧长；两端各留半个间距，免得断面边缘悬空。"""
        length = self.route.length
        count = max(1, int(round(length / style.TIE_SPACING)))
        pitch = length / count
        return [pitch * (i + 0.5) for i in range(count)]


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

def _basis(frame: Pose):
    """返回件局部的 ``(forward, right, up)`` 单位向量。"""
    sin_h = math.sin(frame.heading)
    cos_h = math.cos(frame.heading)
    return (cos_h, 0.0, sin_h), (-sin_h, 0.0, cos_h), (0.0, 1.0, 0.0)


def _local_box(builder: MeshBuilder, frame: Pose,
               x0: float, x1: float, y0: float, y1: float, z0: float, z1: float,
               color, top_color=None) -> None:
    """在 ``frame`` 的局部坐标里放一个长方体（x 向前、y 向上、z 横向）。"""
    forward, right, up = _basis(frame)
    origin = (
        frame.x + forward[0] * x0 + right[0] * z0 + up[0] * y0,
        frame.y + forward[1] * x0 + right[1] * z0 + up[1] * y0,
        frame.z + forward[2] * x0 + right[2] * z0 + up[2] * y0,
    )
    ex = tuple(forward[i] * (x1 - x0) for i in range(3))
    ey = tuple(up[i] * (y1 - y0) for i in range(3))
    ez = tuple(right[i] * (z1 - z0) for i in range(3))
    builder.add_oriented_box(origin, ex, ey, ez, color, top_color)


def _shrink_ends(rings: list[list[tuple[float, float, float]]],
                 amount: float) -> None:
    """把首尾断面沿轴向各向内挪 ``amount``，避免相邻件端盖共面闪烁。

    只做**整环刚性平移**（沿"从端面指向邻环"的方向），不做逐点位移 ——
    逐点位移会让端面发生剪切，反倒把一排断面推歪。
    """
    if len(rings) < 2:
        return
    for index, neighbour_index in ((0, 1), (-1, -2)):
        head = rings[index]
        neighbour = rings[neighbour_index]
        count = len(head)
        dx = sum(b[0] - a[0] for a, b in zip(head, neighbour)) / count
        dy = sum(b[1] - a[1] for a, b in zip(head, neighbour)) / count
        dz = sum(b[2] - a[2] for a, b in zip(head, neighbour)) / count
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-12:
            continue
        scale = amount / length
        dx, dy, dz = dx * scale, dy * scale, dz * scale
        rings[index] = [(p[0] + dx, p[1] + dy, p[2] + dz) for p in head]


# --------------------------------------------------------------------------- #
# 各部件
# --------------------------------------------------------------------------- #

def _ballast_section(frame: Pose, ground_local_y: float,
                     with_skirt: bool) -> tuple[tuple[float, float], ...]:
    """道砟断面；件高出地面时把两侧放坡到地面，形成路堤 / 引桥外观。"""
    top = style.BALLAST_TOP_Y
    bottom = style.BALLAST_BOTTOM_Y
    if not with_skirt:
        return _SECTION_BALLAST_FLAT

    skirt_y = min(frame.y + bottom, ground_local_y)
    height = frame.y + bottom - skirt_y
    if height < 1e-4:
        # 贴地时给一点点厚度，避免退化成零面积三角形
        height = 1e-3
        skirt_y = frame.y + bottom - height
    half = style.BALLAST_BOTTOM_HALF_WIDTH + height * style.EMBANKMENT_SLOPE
    return (
        (-style.BALLAST_TOP_HALF_WIDTH, top),
        (style.BALLAST_TOP_HALF_WIDTH, top),
        (style.BALLAST_BOTTOM_HALF_WIDTH, bottom),
        (half, skirt_y - frame.y),
        (-half, skirt_y - frame.y),
        (-style.BALLAST_BOTTOM_HALF_WIDTH, bottom),
    )


def _add_ballast(builder: MeshBuilder, sampler: RouteSampler, frames: list[Pose],
                 ground_local_y: float, color, skirt: bool | None = None) -> None:
    if skirt is None:
        needs_skirt = any(
            frame.y + style.BALLAST_BOTTOM_Y - ground_local_y > 1e-3 for frame in frames
        )
    else:
        needs_skirt = skirt
    rings = [
        ring_from_section(frame, _ballast_section(frame, ground_local_y, needs_skirt))
        for frame in frames
    ]
    _shrink_ends(rings, CAP_INSET)
    builder.add_tube(rings, color)


def _add_ties(builder: MeshBuilder, sampler: RouteSampler) -> None:
    for index, u in enumerate(sampler.tie_positions):
        frame = sampler.pose_at(u)
        color = style.TIE_COLOR if index % 2 == 0 else style.TIE_COLOR_ALT
        _local_box(
            builder, frame,
            0.0, 2.0 * style.TIE_HALF_THICKNESS,
            style.TIE_BOTTOM_Y, style.TIE_TOP_Y,
            -style.TIE_HALF_LENGTH, style.TIE_HALF_LENGTH,
            color,
        )


def _add_rails(builder: MeshBuilder, sampler: RouteSampler, frames: list[Pose]) -> None:
    left: list[list[tuple[float, float, float]]] = []
    right: list[list[tuple[float, float, float]]] = []
    for frame in frames:
        left.append(ring_from_section(
            frame, [(s[0] - style.RAIL_HALF_GAUGE, s[1]) for s in style.RAIL_SECTION]
        ))
        right.append(ring_from_section(
            frame, [(s[0] + style.RAIL_HALF_GAUGE, s[1]) for s in style.RAIL_SECTION]
        ))
    for rings in (left, right):
        _shrink_ends(rings, CAP_INSET)
        builder.add_tube(rings, style.RAIL_COLOR, head_color=style.RAIL_HEAD_COLOR)


def _add_bridge_parapets(builder: MeshBuilder, frames: list[Pose]) -> None:
    half = style.BRIDGE_PARAPET_THICKNESS / 2.0
    low = style.BALLAST_TOP_Y
    high = style.BALLAST_TOP_Y + style.BRIDGE_PARAPET_HEIGHT
    section = ((-half, low), (half, low), (half, high), (-half, high))
    for lateral in (-style.BALLAST_TOP_HALF_WIDTH, style.BALLAST_TOP_HALF_WIDTH):
        rings = []
        for frame in frames:
            # right = (-sin h, 0, cos h)，把断面沿横向挪到桥面两侧
            moved = frame.with_position(
                frame.x - math.sin(frame.heading) * lateral,
                frame.y,
                frame.z + math.cos(frame.heading) * lateral,
            )
            rings.append(ring_from_section(moved, section))
        _shrink_ends(rings, CAP_INSET)
        builder.add_tube(rings, style.BRIDGE_PARAPET_COLOR)


def _pier_fractions(length: float) -> tuple[float, ...]:
    """一件桥面上桥墩所在的比例位置（弧长的 1/4 与 3/4，长件多分几档）。

    墩距只由**件长**决定，这件事是有意的：想让桥下空出一段（跨线、跨路、跨另一
    股道），就得换一件更长的桥面，而不是"把两件短的接起来" —— 两件 40 m 接起来
    虽然也是 80 m，墩子却会落在 10/30/50/70 m 处，正中间照样有墩（见 pieces.json
    里 ``bridge_80`` 的说明）。
    """
    if length <= _PIER_MAX_SPAN * 2.0:
        return (0.25, 0.75)
    count = max(2, int(round(length / _PIER_MAX_SPAN)))
    return tuple((i + 0.5) / count for i in range(count))


def _add_deck_piers(builder: MeshBuilder, sampler: RouteSampler,
                    ground_local_y: float) -> None:
    """架空桥面下面立桥墩，一直落到地面。

    没有它，架空的那一跨看起来像**浮在半空**：桥面与地面之间是一片虚空，从侧面
    一眼就能看出"这不是桥"。只在真的离地（> ``PIER_MIN_HEIGHT``）时才立 ——
    贴地的桥面不需要墩子，立了就成了一截埋在地里的柱子。

    高度是**逐墩**判的，不是整件一起判：引桥（``bridge_up_40`` 这类带坡度的桥面）
    从地面起步，首墩底下只有几十厘米、末墩底下有几米，整件一起判会一律不立墩，
    于是引桥看起来像一条悬空的斜板。
    """
    length = sampler.route.length
    for fraction in _pier_fractions(length):
        frame = sampler.pose_at(length * fraction)
        if frame.y + style.BALLAST_BOTTOM_Y - ground_local_y < _PIER_MIN_HEIGHT:
            continue
        _local_box(
            builder, frame,
            -style.BRIDGE_PIER_HALF_LENGTH, style.BRIDGE_PIER_HALF_LENGTH,
            ground_local_y - frame.y, style.BALLAST_BOTTOM_Y,
            -style.BRIDGE_PIER_HALF_WIDTH, style.BRIDGE_PIER_HALF_WIDTH,
            style.BRIDGE_COLOR,
        )


def _add_buffer_stop(builder: MeshBuilder, piece: PieceDef,
                     ground_local_y: float) -> None:
    """车挡：件只有一个端口、没有 route，所以直接按端口朝向摆方块。"""
    port = piece.port(piece.port_ids[0])
    frame = port.pose.flipped()          # 列车驶来的方向
    # 地基小台
    _local_box(builder, frame, -1.2, 1.6, ground_local_y - frame.y,
               style.BALLAST_TOP_Y, -2.2, 2.2, style.BALLAST_COLOR)
    # 主体混凝土块
    _local_box(builder, frame, 0.30, 1.05, style.BALLAST_TOP_Y, 0.18,
               -1.05, 1.05, style.BUFFER_COLOR)
    # 红色横梁
    _local_box(builder, frame, 0.20, 0.46, 0.55, 0.95,
               -1.15, 1.15, style.BUFFER_BEAM_COLOR)
    # 两根立柱
    for z in (-0.95, 0.80):
        _local_box(builder, frame, 0.30, 0.46, style.BALLAST_TOP_Y, 1.05,
                   z, z + 0.15, style.BUFFER_COLOR)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def build_piece_mesh(piece: PieceDef, ground_local_y: float) -> MeshBuilder:
    """生成一个轨道件的网格（件局部坐标）。

    ``ground_local_y`` 是**地面在件局部坐标下的高度**（= ``GROUND_Y - 件世界 y``），
    路堤放坡要用它才能一直落到地面。
    """
    builder = MeshBuilder(f"piece_{piece.id}")

    if not piece.routes or piece.mesh == "buffer":
        _add_buffer_stop(builder, piece, ground_local_y)
        return builder

    is_bridge = piece.mesh == "bridge"
    is_deck = piece.mesh == "deck"
    for route in piece.routes:
        sampler = RouteSampler(piece, route)
        bias = ROUTE_Y_BIAS * route.index
        frames = sampler.frames()
        if bias:
            frames = [
                f.with_position(f.x, f.y + bias, f.z) for f in frames
            ]
        # 桥面（deck）**不放坡到地面**：桥就该是架空的，两侧填土成了路堤就不像桥了。
        _add_ballast(
            builder, sampler, frames, ground_local_y,
            style.BRIDGE_COLOR if (is_bridge or is_deck) else style.BALLAST_COLOR,
            skirt=False if is_deck else None,
        )
        _add_ties(builder, sampler)
        _add_rails(builder, sampler, frames)
        if is_bridge or is_deck:
            _add_bridge_parapets(builder, frames)
        if is_deck:
            _add_deck_piers(builder, sampler, ground_local_y)

    return builder
