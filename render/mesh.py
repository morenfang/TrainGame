"""程序化网格构建：把三角形塞进 :class:`GeomNode`。

项目不使用任何外部 3D 素材（决策 A2：procedural_first），因此道砟、枕木、
钢轨、地面、标记全部在这里用代码生成。

设计要点
------------------------------------------------
* **顶点色烘焙**：写入时就把 :func:`render.style.shade` 的结果算好，不挂 Light、
  不开 shader。离屏截图与窗口渲染因此逐像素一致（这一点对"截图自检"很关键）。
* **法线由几何算，不手写**：`add_tube` 会拿每个四边形的几何法线与「离开轴心」
  方向比对，必要时**同时翻转法线和绕序**，所以不可能出现"面朝里"导致背光变黑。
* 所有几何都是**双面**渲染（``setTwoSided``）作为兜底，代价可忽略，
  好处是彻底消灭一整类"某块面看不见"的 bug。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData,
                          GeomVertexFormat, GeomVertexWriter, NodePath)

from render.style import shade

Color = tuple[float, float, float, float]
Point = tuple[float, float, float]

_VERTEX_FORMAT = GeomVertexFormat.get_v3n3c4()

#: 法线不可用（退化三角形）时的兜底值：朝上。
_FALLBACK_NORMAL: Point = (0.0, 1.0, 0.0)

#: 叉积模长小于这个值的三角形一律不写进网格（= 面积 5e-11 m²，远小于任何
#: 真几何）。只用来挡"三点共线"这类真退化，不会误伤正常的小面。
_MIN_CROSS = 1e-10


# --------------------------------------------------------------------------- #
# 向量小工具（保持纯 tuple，避免在生成阶段产生 Vec3 分配开销）
# --------------------------------------------------------------------------- #

def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Point, b: Point) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(v: Point) -> Point:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length < 1e-12:
        return _FALLBACK_NORMAL
    return (v[0] / length, v[1] / length, v[2] / length)


def _centroid(points: Sequence[Point]) -> Point:
    n = float(len(points))
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


# --------------------------------------------------------------------------- #
# 构建器
# --------------------------------------------------------------------------- #

class MeshBuilder:
    """累积三角形，最后一次性吐出一个 :class:`NodePath`。

    用法是「先倒几何、再 `build()`」：所有坐标都是**件局部坐标系**下的
    core 约定坐标（+Y 向上），摆放由 :func:`render.transform.apply_pose` 负责。
    """

    __slots__ = ("name", "_pos", "_nrm", "_col", "_tri")

    def __init__(self, name: str = "mesh"):
        self.name = name
        self._pos: list[Point] = []
        self._nrm: list[Point] = []
        self._col: list[Color] = []
        self._tri: list[tuple[int, int, int]] = []

    # ---------------------------------------------------------------- 基本量

    def __len__(self) -> int:
        return len(self._tri)

    def __bool__(self) -> bool:
        return bool(self._tri)

    @property
    def triangle_count(self) -> int:
        return len(self._tri)

    # 只读访问器：给测试与统计用（外部不应该改这些数组）
    @property
    def positions(self) -> tuple[Point, ...]:
        return tuple(self._pos)

    @property
    def normals(self) -> tuple[Point, ...]:
        return tuple(self._nrm)

    @property
    def colors(self) -> tuple[Color, ...]:
        return tuple(self._col)

    @property
    def triangles(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(self._tri)

    # ---------------------------------------------------------------- 写入

    def vertex(self, position: Point, normal: Point, color: Color) -> int:
        index = len(self._pos)
        self._pos.append((float(position[0]), float(position[1]), float(position[2])))
        self._nrm.append(_unit((float(normal[0]), float(normal[1]), float(normal[2]))))
        self._col.append(
            (float(color[0]), float(color[1]), float(color[2]),
             float(color[3]) if len(color) > 3 else 1.0)
        )
        return index

    def triangle(self, i: int, j: int, k: int) -> None:
        self._tri.append((i, j, k))

    def face(self, indices: Sequence[int]) -> None:
        """扇形三角化（凸多边形足够）。

        **零面积的三角形会被丢掉。** 扇形三角化碰上"三个点共线"就会切出零面积
        三角形 —— 而断面里偏偏到处是共线：车体的平底、道砟顶面、钢轨顶面都是。
        它们肉眼看不见，却白占显存、把包围盒撑大、还会让"有没有退化三角形"
        这类自检失灵。与其让每个调用方自己躲，不如在这里一次挡掉。
        """
        for k in range(1, len(indices) - 1):
            self._add_triangle(indices[0], indices[k], indices[k + 1])

    def _add_triangle(self, i: int, j: int, k: int) -> None:
        """写一个三角形；面积小到没有意义的直接丢掉。"""
        a, b, c = self._pos[i], self._pos[j], self._pos[k]
        cross = _cross(_sub(b, a), _sub(c, a))
        if math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) < _MIN_CROSS:
            return
        self._tri.append((i, j, k))

    # ---------------------------------------------------------------- 图元

    def add_polygon(self, points: Sequence[Point], color: Color,
                    normal: Point | None = None) -> None:
        """一个平面多边形，绕序逆时针（法线由前三点推定）。

        给了 ``normal`` 而绕序与它相反时，**把点序翻过来**再三角化。这样"这块面
        朝哪边"在网格里始终自洽 —— 顶点法线、三角形绕序、以及任何外部程序按绕序
        算出来的几何法线三者一致。放在双面渲染下看不出差别，但一旦将来要做背面
        剔除、或者拿网格去做碰撞 / 体素化，绕序反了就是"里外颠倒"。
        """
        if len(points) < 3:
            return
        geometric = _cross(_sub(points[1], points[0]), _sub(points[2], points[0]))
        if normal is None:
            normal = geometric
        elif _dot(geometric, normal) < 0.0:
            points = tuple(reversed(points))
        normal = _unit(normal)          # shade 要求单位法线（见 add_quad_flat）
        indices = [self.vertex(p, normal, shade(normal, color)) for p in points]
        self.face(indices)

    def add_quad(self, p0: Point, p1: Point, p2: Point, p3: Point,
                 color: Color, normal: Point | None = None) -> None:
        self.add_polygon((p0, p1, p2, p3), color, normal)

    def add_quad_flat(self, p0: Point, p1: Point, p2: Point, p3: Point,
                      color: Color, normal: Point) -> None:
        """已知法线的四边形（比 `add_quad` 少一次叉乘）。

        法线在这里**统一归一化**再烘焙光照。曾经这里直接拿调用方给的叉积去
        ``shade`` —— 而叉积的模长是面积的两倍，Lambert 项被压成一点点，烘出来的
        颜色整体偏暗、只剩环境光；同时 :meth:`vertex` 存进去的又是归一化后的
        法线，于是"法线是单位向量"那条自检照样通过，谁也看不出问题。
        """
        normal = _unit(normal)
        shaded = shade(normal, color)
        indices = [self.vertex(p, normal, shaded) for p in (p0, p1, p2, p3)]
        self.triangle(indices[0], indices[1], indices[2])
        self.triangle(indices[0], indices[2], indices[3])

    def add_oriented_box(self, origin: Point, ex: Point, ey: Point, ez: Point,
                         color: Color, top_color: Color | None = None) -> None:
        """由「一个角 + 三条完整棱向量」确定的平行六面体。

        ``ex``/``ey``/``ez`` 是**从 origin 出发的完整棱向量**（长度即棱长）。
        顶面（+ey 侧）可以单独给色，用来做钢轨那种"被车轮磨亮的走行面"。
        """
        c000 = origin
        c100 = (origin[0] + ex[0], origin[1] + ex[1], origin[2] + ex[2])
        c010 = (origin[0] + ey[0], origin[1] + ey[1], origin[2] + ey[2])
        c110 = (c100[0] + ey[0], c100[1] + ey[1], c100[2] + ey[2])
        c001 = (origin[0] + ez[0], origin[1] + ez[1], origin[2] + ez[2])
        c101 = (c100[0] + ez[0], c100[1] + ez[1], c100[2] + ez[2])
        c011 = (c010[0] + ez[0], c010[1] + ez[1], c010[2] + ez[2])
        c111 = (c110[0] + ez[0], c110[1] + ez[1], c110[2] + ez[2])

        centre = _centroid((c000, c100, c010, c110, c001, c101, c011, c111))
        faces = (
            # (四个角，顶面?)
            ((c000, c100, c110, c010), True),    # +ey
            ((c001, c101, c111, c011), False),   # -ey
            ((c000, c100, c101, c001), False),   # -ez
            ((c010, c110, c111, c011), False),   # +ez
            ((c000, c010, c011, c001), False),   # -ex
            ((c100, c110, c111, c101), False),   # +ex
        )
        for corners, is_top in faces:
            self._quad_outward(corners, centre,
                               top_color if (is_top and top_color) else color)

    def _quad_outward(self, corners: Sequence[Point], inside: Point,
                      color: Color) -> None:
        """给一个四边形，选出「背离 inside」的法线，并让绕序与之一致。"""
        normal = _cross(_sub(corners[1], corners[0]), _sub(corners[2], corners[0]))
        outward = _sub(_centroid(corners), inside)
        if _dot(normal, outward) < 0.0:
            corners = (corners[3], corners[2], corners[1], corners[0])
            normal = (-normal[0], -normal[1], -normal[2])
        self.add_quad_flat(*corners, color=color, normal=normal)

    # ---------------------------------------------------------------- 扫掠

    def add_tube(self, rings: Sequence[Sequence[Point]], color: Color,
                 cap_start: bool = True, cap_end: bool = True,
                 head_color: Color | None = None,
                 face_color=None) -> None:
        """把一串**等点数闭合截面环**缝成管状体。

        ``rings[k]`` 是第 k 个断面的世界坐标点列表（顺序一致）。法线一律由几何
        算出并朝外修正，因此即便截面点序反了也不会出现背面变黑。

        ``head_color`` 用于把法线与 ``+Y`` 接近的面单独上色（钢轨走行面）。

        ``face_color`` 是 ``(k, i, normal) -> Color | None`` 的回调，用来**逐块**
        上色（车身涂装就靠它：车窗、车门、色带、车顶各是各的颜色，却共用同一次
        放样）。返回 ``None`` 就退回 ``color`` / ``head_color`` 的规则。
        回调拿到的是**环序 k 与断面点序 i**，所以调用方不必去猜顶点编号。
        """
        if len(rings) < 2:
            return
        count = len(rings[0])
        axis = _unit(_sub(_centroid(rings[-1]), _centroid(rings[0])))

        for k in range(len(rings) - 1):
            ring_a = rings[k]
            ring_b = rings[k + 1]
            centre = _centroid(tuple(ring_a) + tuple(ring_b))
            for i in range(count):
                j = (i + 1) % count
                corners = (ring_a[i], ring_a[j], ring_b[j], ring_b[i])
                normal = _cross(_sub(corners[1], corners[0]),
                                _sub(corners[2], corners[0]))
                outward = _sub(_centroid(corners), centre)
                if _dot(normal, outward) < 0.0:
                    corners = (corners[3], corners[2], corners[1], corners[0])
                    normal = (-normal[0], -normal[1], -normal[2])
                normal = _unit(normal)
                face_color_value = color
                if head_color is not None and normal[1] > 0.92:
                    face_color_value = head_color
                if face_color is not None:
                    override = face_color(k, i, normal)
                    if override is not None:
                        face_color_value = override
                self.add_quad_flat(*corners, color=face_color_value, normal=normal)

        centre_of = _centroid(rings[0])
        if cap_start:
            self.add_polygon(tuple(rings[0]), color, normal=(-axis[0], -axis[1], -axis[2]))
        if cap_end:
            self.add_polygon(tuple(rings[-1]), color, normal=axis)

    def add_cylinder(self, start: Point, end: Point, radius: float, sides: int,
                     color: Color, cap: bool = True) -> None:
        """一根圆柱：轴从 ``start`` 到 ``end``，半径 ``radius``。

        轴**不必**是坐标轴 —— 垂直的轮子、横着的车轴、斜着的撑杆都走这一条。
        端盖交给 :meth:`add_tube`，因此法线朝向也是它统一修的。
        """
        if sides < 3 or radius <= 0.0:
            return
        axis = _unit(_sub(end, start))
        # 取一个与轴不平行的参考方向，叉乘出两个正交基
        reference = (0.0, 1.0, 0.0) if abs(axis[1]) < 0.9 else (1.0, 0.0, 0.0)
        side_a = _unit(_cross(axis, reference))
        side_b = _unit(_cross(axis, side_a))
        rings: list[list[Point]] = []
        for centre in (start, end):
            ring = []
            for k in range(sides):
                angle = math.tau * k / sides
                cos_a, sin_a = math.cos(angle) * radius, math.sin(angle) * radius
                ring.append((
                    centre[0] + side_a[0] * cos_a + side_b[0] * sin_a,
                    centre[1] + side_a[1] * cos_a + side_b[1] * sin_a,
                    centre[2] + side_a[2] * cos_a + side_b[2] * sin_a,
                ))
            rings.append(ring)
        self.add_tube(rings, color, cap_start=cap, cap_end=cap)

    def add_ribbon(self, centers: Sequence[Point], half_width: float, color: Color,
                   up: Point = (0.0, 1.0, 0.0)) -> None:
        """沿一串中心点铺一条水平窄带（用于闭环高亮）。

        每个采样点的横向 = 前进方向 × up，因此拐弯处带宽不会被拉扁。
        """
        if len(centers) < 2:
            return
        left: list[Point] = []
        right: list[Point] = []
        last = len(centers) - 1
        for i, point in enumerate(centers):
            forward = _unit(_sub(centers[min(i + 1, last)],
                                 centers[max(i - 1, 0)]))
            side = _unit(_cross(forward, up))
            if side == _FALLBACK_NORMAL:
                side = (1.0, 0.0, 0.0)
            left.append((point[0] + side[0] * half_width,
                         point[1] + side[1] * half_width,
                         point[2] + side[2] * half_width))
            right.append((point[0] - side[0] * half_width,
                          point[1] - side[1] * half_width,
                          point[2] - side[2] * half_width))
        for i in range(len(centers) - 1):
            self.add_quad(left[i], right[i], right[i + 1], left[i + 1], color,
                          normal=(0.0, 1.0, 0.0))

    # ---------------------------------------------------------------- 输出

    def mirrored_x(self) -> "MeshBuilder":
        """关于 ``x = 0`` 镜像的一份新网格。

        编组里朝后的那节头车要掉头，靠的就是它。三件事必须一起做：

        * 位置与法线的 x 分量取反 —— 顶点色里烘焙的是 ``shade(normal, color)``，
          法线跟着翻，明暗才会跟着一起翻；
        * **三角形绕序反转** —— 镜像会把左手系变回右手系，不反绕序所有面都成了
          背面。

        比起在场景图里挂 ``setScale(-1, 1, 1)``，这样做的好处是网格本身自洽：
        包围盒、法线、三角形朝向全都是对的，后面要拿它做碰撞或量尺寸都不会被骗。
        """
        result = MeshBuilder(f"{self.name}_mirror")
        result._pos = [(-x, y, z) for x, y, z in self._pos]
        result._nrm = [(-nx, ny, nz) for nx, ny, nz in self._nrm]
        result._col = list(self._col)
        result._tri = [(i, k, j) for i, j, k in self._tri]
        return result

    def build(self) -> NodePath:
        """生成 :class:`NodePath`。空构建器也能安全产出一个空节点。"""
        return make_node_path(self.name, self._pos, self._nrm, self._col, self._tri)


def make_node_path(name: str, positions: Sequence[Point], normals: Sequence[Point],
                   colors: Sequence[Color], triangles: Sequence[tuple[int, int, int]]
                   ) -> NodePath:
    """把裸数组组装成 GeomNode（双面渲染）。"""
    vdata = GeomVertexData(name, _VERTEX_FORMAT, Geom.UHStatic)
    vdata.setNumRows(len(positions))
    writer_pos = GeomVertexWriter(vdata, "vertex")
    writer_nrm = GeomVertexWriter(vdata, "normal")
    writer_col = GeomVertexWriter(vdata, "color")
    for position, normal, color in zip(positions, normals, colors):
        writer_pos.addData3f(*position)
        writer_nrm.addData3f(*normal)
        writer_col.addData4f(*color)

    primitive = GeomTriangles(Geom.UHStatic)
    for i, j, k in triangles:
        primitive.addVertices(i, j, k)
    primitive.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(primitive)
    node = GeomNode(name)
    node.addGeom(geom)
    node_path = NodePath(node)
    # 双面兜底：即使某处绕序反了也不会"缺一块面"。
    node_path.setTwoSided(True)
    return node_path


# --------------------------------------------------------------------------- #
# 断面工具
# --------------------------------------------------------------------------- #

def ring_from_section(frame, section: Iterable[tuple[float, float]]) -> list[Point]:
    """把一条 ``(横向, 高度)`` 断面按位姿摆成一个截面环。

    横向取 ``forward × up``（右手系下的"右"），高度直接用世界 +Y —— 所以坡度上
    的断面依然**竖直**，与真实轨道一致。
    """
    sin_h = math.sin(frame.heading)
    cos_h = math.cos(frame.heading)
    # right = forward × up，forward = (cos h, 0, sin h)
    right_x, right_z = -sin_h, cos_h
    origin = (frame.x, frame.y, frame.z)
    return [
        (origin[0] + right_x * lateral, origin[1] + height, origin[2] + right_z * lateral)
        for lateral, height in section
    ]
