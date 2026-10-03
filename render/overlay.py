"""编辑器用的叠加显示：端口标记、吸附目标、幽灵预览、闭环彩带、缺口告警。

这些都是"UI 几何"，同样走 :class:`render.mesh.MeshBuilder`（顶点色里烘焙好光照），
因此不需要挂 Light，也不会与轨道网格抢渲染状态。
"""

from __future__ import annotations

import math

from panda3d.core import NodePath, TransparencyAttrib

from render import style
from render.mesh import MeshBuilder
from render.transform import apply_pose

#: 端口标记：箭头贴地 + 一根竖起的小柱（拉远了也能看见）。
_ARROW_LENGTH = 1.9
_ARROW_HALF_WIDTH = 0.62
_ARROW_TAIL = 0.85
_ARROW_TAIL_HALF_WIDTH = 0.20
_POST_HEIGHT = 0.85
_POST_HALF = 0.055
_ARROW_LIFT = 0.03

_UP = (0.0, 1.0, 0.0)
_FORWARD = (1.0, 0.0, 0.0)


def build_port_marker(color, *, scale: float = 1.0) -> NodePath:
    """指向端口**外指向**的箭头标记（局部 +X 就是外指向）。"""
    builder = MeshBuilder("port_marker")
    lift = _ARROW_LIFT

    def head(x0: float, x1: float, half_at_x0: float, apex: bool) -> None:
        if apex:
            builder.add_polygon(
                ((x1, lift, 0.0), (x0, lift, half_at_x0), (x0, lift, -half_at_x0)),
                color, normal=_UP,
            )
        else:
            builder.add_polygon(
                ((x0, lift, -half_at_x0), (x1, lift, -half_at_x0),
                 (x1, lift, half_at_x0), (x0, lift, half_at_x0)),
                color, normal=_UP,
            )

    tail = _ARROW_TAIL * scale
    tip = _ARROW_LENGTH * scale
    head(0.0, tail, _ARROW_TAIL_HALF_WIDTH * scale, apex=False)
    head(tail, tip, _ARROW_HALF_WIDTH * scale, apex=True)

    # 立柱：给远处一个可靠的"这里有个口"的视觉锚点
    half = _POST_HALF * scale
    builder.add_polygon(
        ((0.0, 0.0, -half), (0.0, _POST_HEIGHT * scale, -half),
         (0.0, _POST_HEIGHT * scale, half), (0.0, 0.0, half)),
        color, normal=(-1.0, 0.0, 0.0),
    )
    builder.add_polygon(
        ((0.0, 0.0, half), (0.0, _POST_HEIGHT * scale, half),
         (0.0, _POST_HEIGHT * scale, -half), (0.0, 0.0, -half)),
        color, normal=(1.0, 0.0, 0.0),
    )

    node = builder.build()
    node.setName("port_marker")
    return node


def build_gap_marker(color=style.GAP_WARN_COLOR, *, scale: float = 1.0) -> NodePath:
    """未闭环时立在缺口处的告警柱（一根亮色方柱 + 顶端小方块）。"""
    builder = MeshBuilder("gap_marker")
    half = 0.09 * scale
    height = 1.6 * scale
    for normal, corners in (
        ((-1.0, 0.0, 0.0), ((0.0, 0.0, -half), (0.0, height, -half),
                            (0.0, height, half), (0.0, 0.0, half))),
        ((1.0, 0.0, 0.0), ((0.0, 0.0, half), (0.0, height, half),
                           (0.0, height, -half), (0.0, 0.0, -half))),
    ):
        builder.add_polygon(corners, color, normal=normal)
    cap = 0.26 * scale
    builder.add_oriented_box(
        (-cap, height, -cap), (2 * cap, 0.0, 0.0), (0.0, 0.22 * scale, 0.0),
        (0.0, 0.0, 2 * cap), color,
    )
    node = builder.build()
    node.setName("gap_marker")
    return node


def build_loop_ribbon(path, color=style.LOOP_RIBBON_COLOR,
                      step: float = 2.0) -> NodePath:
    """沿一条可行驶闭环铺一条亮带，直观确认"车能绕一圈回来"。"""
    builder = MeshBuilder("loop_ribbon")
    total = path.total_length
    if total <= 0.0:
        return builder.build()

    count = max(8, int(total / step))
    centers = []
    for i in range(count + 1):
        pose = path.pose_at(total * i / count)
        centers.append((pose.x, pose.y + style.LOOP_RIBBON_LIFT, pose.z))
    builder.add_ribbon(centers, style.LOOP_RIBBON_HALF_WIDTH, color)
    node = builder.build()
    node.setName("loop_ribbon")
    tint(node, None, transparent=True)
    return node


def tint(node_path: NodePath, rgba, *, transparent: bool = True) -> NodePath:
    """给一棵子树整体染色 / 调透明。

    用 ``ColorScaleAttrib`` 而不是 ``ColorAttrib``：前者与顶点色**相乘**，
    于是幽灵预览仍然看得出钢轨、枕木的结构，而不会变成一块纯色色块。
    """
    if transparent:
        node_path.setTransparency(TransparencyAttrib.MAlpha)
    if rgba is None:
        node_path.clearColorScale()
    else:
        node_path.setColorScale(rgba[0], rgba[1], rgba[2], rgba[3])
    return node_path


def place(node_path: NodePath, pose) -> NodePath:
    """把一个叠加节点摆到 core 位姿上（唯一的坐标转换入口）。"""
    return apply_pose(node_path, pose)


def build_footprint_marker(footprint, color=style.SCENERY_HOVER_COLOR) -> NodePath:
    """给一件布景的**占地轮廓**画一层半透明高亮（矩形或圆，贴地）。

    布景被烘成一个整体几何节点，没法逐件染色，所以悬停/删除的视觉反馈要靠这一层
    独立的标记：轮廓来自 :class:`render.scenery.Footprint`（房/车站是矩形，山/树
    是圆），摆到地面略高处，不遮物件本身。
    """
    builder = MeshBuilder("scenery_highlight")
    lift = style.GROUND_Y + style.SCENERY_LIFT + 0.10
    cos_h, sin_h = math.cos(footprint.heading), math.sin(footprint.heading)
    hx, hz = footprint.half_x, footprint.half_z

    def to_world(lx: float, lz: float) -> tuple[float, float, float]:
        return (footprint.x + lx * cos_h - lz * sin_h, lift,
                footprint.z + lx * sin_h + lz * cos_h)

    if footprint.round_:
        sides = 24
        outer = [to_world(hx * math.cos(math.tau * i / sides),
                          hx * math.sin(math.tau * i / sides))
                 for i in range(sides)]
        inner = [to_world(hx * 0.86 * math.cos(math.tau * i / sides),
                          hx * 0.86 * math.sin(math.tau * i / sides))
                 for i in range(sides)]
        for i in range(sides):
            j = (i + 1) % sides
            builder.add_quad(outer[i], outer[j], inner[j], inner[i],
                             color, normal=(0.0, 1.0, 0.0))
    else:
        corners = [to_world(sx * hx, sz * hz) for sx in (-1.0, 1.0) for sz in (-1.0, 1.0)]
        builder.add_quad(corners[0], corners[1], corners[3], corners[2],
                         color, normal=(0.0, 1.0, 0.0))

    node = builder.build()
    node.setName("scenery_highlight")
    tint(node, color, transparent=True)
    return node
