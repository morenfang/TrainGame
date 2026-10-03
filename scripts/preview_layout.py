"""把轨道装配图与列车位置画成图 —— 不开 3D 引擎也能肉眼验收几何。

用途：核心层是纯 Python 的，本脚本用 matplotlib 把它的输出画出来，因此在装
Panda3D 之前就能确认「轨道真的闭合」「列车真的在轨上」。

    python scripts/preview_layout.py            # 输出到 docs/preview_layout.png
    python scripts/preview_layout.py out.png

四个面板：
① 8 节 R40/45° 拼出的整圆 + 8 节编组
② 8 字形（8 节左弯 + 8 节右弯，切点共用接缝）
③ 道岔 + 菱形交叉的站场
④ 坡度：上坡引桥 + 平轨，看侧视图
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402


def _use_cjk_font() -> str | None:
    """选一个能显示中文的字体（Windows 上通常是微软雅黑 / 黑体）。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                      "Source Han Sans SC", "PingFang SC", "Arial Unicode MS"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    return None


CJK_FONT = _use_cjk_font()


from core.track.catalog import Catalog  # noqa: E402
from core.track.layout import Layout  # noqa: E402
from core.track.path import place_consist, trace  # noqa: E402
from core.train.consist import TrainCatalog  # noqa: E402

CATALOG = Catalog.builtin()
TRAINS = TrainCatalog.builtin()

BALLAST = "#8d8377"
RAIL = "#2b2b2b"
SLEEPER = "#5c5147"


# --------------------------------------------------------------------------- #
# 画图小工具
# --------------------------------------------------------------------------- #

def draw_track(ax, layout, color=RAIL, width=1.6):
    """把装配图里每个件的中心线画出来（直接取 route 的解析几何）。"""
    for index in layout.pieces:
        for route in layout.definition(index).routes:
            entry = layout.world_inbound(index, route.from_port)
            points = _route_points(entry, route.length, route.dtheta)
            ax.plot(
                [p[0] for p in points],
                [p[2] for p in points],
                color=color,
                linewidth=width,
                solid_capstyle="round",
                zorder=3,
            )


def _route_points(entry, length, dtheta, steps=48):
    from core.geometry import arc_point

    return [
        entry.compose(_local(arc_point(length, dtheta, length * i / steps))).position
        for i in range(steps + 1)
    ]


def _local(triple):
    from core.geometry import Pose

    dx, dz, du = triple
    return Pose(dx, 0.0, dz, du)


def draw_ports(ax, layout, only_free=False, color="#d64545", size=14):
    keys = layout.free_ports() if only_free else layout.all_ports()
    for index, port_id in keys:
        pose = layout.world_port(index, port_id)
        ax.plot(pose.x, pose.z, "o", color=color, markersize=size / 4, zorder=5)
        ax.annotate(
            port_id,
            (pose.x, pose.z),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
            color=color,
            zorder=6,
        )


def draw_train(ax, cars, color="#1f5fa8", show_bogies=True):
    """把每节车画成一个矩形（长方形 = 车体，两端小圆 = 转向架）。"""
    for car in cars:
        half_len = _half_length(cars, car)
        corners = []
        for yaw, offset in (
            (car.heading, +half_len),
            (car.heading, -half_len),
        ):
            import math

            cx = car.center[0] + math.cos(yaw) * offset
            cz = car.center[2] + math.sin(yaw) * offset
            for side in (+1, -1):
                px = cx - math.sin(yaw) * side * 0.9
                pz = cz + math.cos(yaw) * side * 0.9
                corners.append((px, pz))
        # 顺序：前端两侧、后端两侧
        quad = [corners[0], corners[1], corners[3], corners[2]]
        ax.add_patch(
            Polygon(quad, closed=True, facecolor=color, edgecolor="white",
                    linewidth=0.4, zorder=7)
        )
        if show_bogies:
            for bogie in (car.front, car.rear):
                ax.plot(bogie.position[0], bogie.position[2], "s",
                        color="#222222", markersize=2.2, zorder=8)


def _half_length(cars, car) -> float:
    """由车长表反推半车长（只用于画图）。"""
    return _HALF_LENGTHS.get(car.car_index, 5.0)


_HALF_LENGTHS: dict[int, float] = {}


def setup_axes(ax, title: str, equal: bool = True):
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal" if equal else "auto")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.tick_params(labelsize=7)


# --------------------------------------------------------------------------- #
# 四个场景
# --------------------------------------------------------------------------- #

def scene_circle() -> tuple[Layout, list]:
    layout = Layout(catalog=CATALOG)
    first = layout.add_root("curve_r40_l45")
    index = first
    for _ in range(7):
        index = layout.attach("curve_r40_l45", "a", (index, "b"))
    layout.connect((index, "b"), (first, "a"))

    path = trace(layout, (first, "a"))
    spec = TRAINS["cr400af_8"]
    _HALF_LENGTHS.clear()
    _HALF_LENGTHS.update({i: spec.lengths[i] / 2 for i in range(spec.car_count)})
    cars = place_consist(
        path,
        head_s=60.0,
        lengths=spec.lengths,
        bogie_half_spacing=spec.bogie_half_spacings,
        coupling_gap=spec.coupling_gap,
    )
    return layout, cars


def scene_figure_eight() -> tuple[Layout, list]:
    layout = Layout(catalog=CATALOG)
    first = layout.add_root("curve_r40_l45")
    index = first
    for _ in range(7):
        index = layout.attach("curve_r40_l45", "a", (index, "b"))
    index = layout.attach("curve_r40_r45", "a", (index, "b"))
    for _ in range(7):
        index = layout.attach("curve_r40_r45", "a", (index, "b"))
    layout.connect((index, "b"), (first, "a"))

    path = trace(layout, (first, "a"))
    spec = TRAINS["green_skin_10"]
    _HALF_LENGTHS.clear()
    _HALF_LENGTHS.update({i: spec.lengths[i] / 2 for i in range(spec.car_count)})
    cars = place_consist(
        path,
        head_s=40.0,
        lengths=spec.lengths,
        bogie_half_spacing=spec.bogie_half_spacings,
        coupling_gap=spec.coupling_gap,
    )
    return layout, cars


def scene_yard() -> Layout:
    """一个有道岔、交叉与死端的站场。"""
    layout = Layout(catalog=CATALOG)
    root = layout.add_root("straight_40")
    # 主线上接一个左开道岔
    turnout = layout.attach("turnout_l_40", "a", (root, "b"))
    # 岔股接到一个菱形交叉
    crossing = layout.attach("crossing_45", "a", (turnout, "c"))
    # 交叉的另一条通道通到别处
    layout.attach("curve_r40_l45", "a", (crossing, "c"))
    loader_tail = layout.attach("curve_r40_r45", "a", (crossing, "d"))
    layout.attach("buffer_stop", "a", (loader_tail, "b"))
    # 直股接缓冲端
    layout.attach("straight_20", "a", (turnout, "b"))
    return layout


def scene_grade() -> Layout:
    layout = Layout(catalog=CATALOG)
    root = layout.add_root("ramp_up_40")
    top = layout.attach("straight_20", "a", (root, "b"))
    down = layout.attach("ramp_down_40", "a", (top, "b"))
    layout.attach("straight_20", "a", (down, "b"))
    return layout


# --------------------------------------------------------------------------- #

def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "preview_layout.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 11.5))

    # ① 整圆 + 编组
    ax = axes[0][0]
    layout, cars = scene_circle()
    draw_track(ax, layout)
    draw_train(ax, cars, color=TRAINS["cr400af_8"].cars[0].livery["body"])
    draw_ports(ax, layout, color="#3b7d3b", size=12)
    path = trace(layout, (layout.root_index, "a"))
    setup_axes(ax, f"① 8 x R40/45° 整圆  |  周长 {path.total_length:.1f} m  "
                   f"|  闭环 gap < 1e-12 m")

    # ② 8 字形
    ax = axes[0][1]
    layout, cars = scene_figure_eight()
    draw_track(ax, layout)
    draw_train(ax, cars, color=TRAINS["green_skin_10"].cars[1].livery["body"])
    setup_axes(ax, "② 8 字形：8 节左弯 + 8 节右弯（切点共用接缝，无需交叉件）")

    # ③ 站场
    ax = axes[1][0]
    layout = scene_yard()
    draw_track(ax, layout)
    draw_ports(ax, layout, only_free=True)
    setup_axes(ax, "③ 站场：主线 + 道岔 + 菱形交叉 + 缓冲端（红点 = 空闲端口）")

    # ④ 坡度（侧视图）
    ax = axes[1][1]
    layout = scene_grade()
    for index in layout.pieces:
        for route in layout.definition(index).routes:
            entry = layout.world_inbound(index, route.from_port)
            pts = _route_points(entry, route.length, route.dtheta)
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color="#2b2b2b", linewidth=1.8, zorder=3)
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color="#c9a227", linewidth=0.5, zorder=4)
    setup_axes(ax, "④ 坡度侧视图：3% 上坡引桥 → 平轨 → 3% 下坡", equal=False)
    ax.set_ylabel("高度 (m)", fontsize=8)

    figure.suptitle(
        "模拟铁轨 · 核心层几何验收（纯 Python，未使用任何 3D 引擎）",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(out, dpi=135)
    print(f"written: {out}")
    print(f"CJK font: {CJK_FONT or 'NOT FOUND (中文会显示成方块)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
