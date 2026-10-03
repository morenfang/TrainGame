"""场地底面与参考网格。

沙盘是一块固定大小的方形场地（概要设计 D4：400 m × 400 m 起步），轨道摆在上面。
底面与网格都烘焙了光照（水平面法线恒为 +Y），因此不挂 Light 也能显示。

为什么细网格只铺中间一块
------------------------------------------------
5 m 一条细线铺满 400 m 就是各方向 80 条。相机拉到最远时线间距不足一个像素，
会出现明显的摩尔纹。因此细网格只铺中心 ±120 m，粗网格（25 m）铺满全场 ——
拉远时看到干净的大格，拉近时看到细格，观感与性能都更好。
"""

from __future__ import annotations

import math

from panda3d.core import NodePath

from render.mesh import MeshBuilder
from render.style import (AXIS_X_COLOR, AXIS_Z_COLOR, GRID_LIFT,
                          GRID_MAJOR_COLOR, GRID_MAJOR_STEP, GRID_MINOR_COLOR,
                          GRID_MINOR_STEP, GROUND_COLOR, GROUND_Y, PLOT_SIZE,
                          shade)

#: 底面细分格数（只影响渐变过渡的平滑度）。
_SURFACE_CELLS = 24

#: 细网格的铺设半径（米）。
_MINOR_EXTENT = 120.0

#: 网格线半宽（米）。取值是按"一眼能看出格子"来的，不是真实线宽。
_MINOR_HALF_WIDTH = 0.035
_MAJOR_HALF_WIDTH = 0.075
_AXIS_HALF_WIDTH = 0.16

_UP = (0.0, 1.0, 0.0)


def _surface_color(x: float, z: float) -> tuple[float, float, float, float]:
    """底面颜色：从场地中心向外缓缓压暗，形成一点纵深。"""
    base = shade(_UP, GROUND_COLOR)
    distance = math.hypot(x, z) / (PLOT_SIZE * 0.5)
    factor = 1.0 - 0.38 * min(1.0, distance)
    return (base[0] * factor, base[1] * factor, base[2] * factor, 1.0)


def _add_line(builder: MeshBuilder, *, along_x: bool, offset: float,
              start: float, end: float, half_width: float, color) -> None:
    """铺一条水平线段（用窄四边形，避免依赖各显卡实现不一的线宽）。"""
    y = GROUND_Y + GRID_LIFT
    shaded = shade(_UP, color)
    a, b = offset - half_width, offset + half_width
    if along_x:
        corners = ((start, y, a), (start, y, b), (end, y, b), (end, y, a))
    else:
        corners = ((a, y, start), (b, y, start), (b, y, end), (a, y, end))
    indices = [builder.vertex(corner, _UP, shaded) for corner in corners]
    builder.triangle(indices[0], indices[1], indices[2])
    builder.triangle(indices[0], indices[2], indices[3])


def _build_surface(plot_size: float) -> NodePath:
    builder = MeshBuilder("ground_surface")
    half = plot_size * 0.5
    step = plot_size / _SURFACE_CELLS
    for i in range(_SURFACE_CELLS):
        x0 = -half + i * step
        x1 = x0 + step
        for j in range(_SURFACE_CELLS):
            z0 = -half + j * step
            z1 = z0 + step
            corners = ((x0, GROUND_Y, z0), (x1, GROUND_Y, z0),
                       (x1, GROUND_Y, z1), (x0, GROUND_Y, z1))
            indices = [
                builder.vertex(corner, _UP, _surface_color(corner[0], corner[2]))
                for corner in corners
            ]
            builder.triangle(indices[0], indices[1], indices[2])
            builder.triangle(indices[0], indices[2], indices[3])
    return builder.build()


def _build_grid(plot_size: float) -> NodePath:
    builder = MeshBuilder("ground_grid")
    half = plot_size * 0.5

    # 细网格：只铺中心一块
    minor_count = int(_MINOR_EXTENT / GRID_MINOR_STEP)
    for k in range(-minor_count, minor_count + 1):
        offset = k * GRID_MINOR_STEP
        if abs(offset) < 1e-9:
            continue
        _add_line(builder, along_x=True, offset=offset,
                  start=-_MINOR_EXTENT, end=_MINOR_EXTENT,
                  half_width=_MINOR_HALF_WIDTH, color=GRID_MINOR_COLOR)
        _add_line(builder, along_x=False, offset=offset,
                  start=-_MINOR_EXTENT, end=_MINOR_EXTENT,
                  half_width=_MINOR_HALF_WIDTH, color=GRID_MINOR_COLOR)

    # 粗网格：铺满全场
    major_count = int(half / GRID_MAJOR_STEP)
    for k in range(-major_count, major_count + 1):
        offset = k * GRID_MAJOR_STEP
        if abs(offset) < 1e-9:
            continue
        _add_line(builder, along_x=True, offset=offset, start=-half, end=half,
                  half_width=_MAJOR_HALF_WIDTH, color=GRID_MAJOR_COLOR)
        _add_line(builder, along_x=False, offset=offset, start=-half, end=half,
                  half_width=_MAJOR_HALF_WIDTH, color=GRID_MAJOR_COLOR)

    # 原点两根坐标轴：+X 偏红、+Z 偏蓝，方便一眼判断朝向
    _add_line(builder, along_x=True, offset=0.0, start=-half, end=half,
              half_width=_AXIS_HALF_WIDTH, color=AXIS_X_COLOR)
    _add_line(builder, along_x=False, offset=0.0, start=-half, end=half,
              half_width=_AXIS_HALF_WIDTH, color=AXIS_Z_COLOR)

    return builder.build()


def build_ground(plot_size: float = PLOT_SIZE) -> NodePath:
    """返回 ``ground`` 节点，子节点 ``surface`` 与 ``grid`` 可分别开关。"""
    root = NodePath("ground")
    surface = _build_surface(plot_size)
    surface.setName("surface")
    surface.reparentTo(root)
    grid = _build_grid(plot_size)
    grid.setName("grid")
    grid.reparentTo(root)
    return root
