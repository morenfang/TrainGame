"""离屏把三种编组放上环线并出图 —— 列车运行层的端到端自检。

    python scripts/train_demo.py                    # 出图到 docs/
    python scripts/train_demo.py -o docs/foo.png    # 指定对比图的路径

它验证的是什么
--------------------------------------------------------------------
``tests/test_train_view.py`` 已经证明"车摆得对"。这个脚本要证明的是另外两件事，
它们都只有在**真的画一帧**时才看得见：

1. **编组真的能上线、能跑起来。** 铺一个整圆，把三款车依次召上线、推满手柄、
   跑够时间，再断言它确实动起来了、所有车厢都还挂在场景里、每个转向架都还压在
   轨道中心线上（这一条用的是与单测同一套闭式解，见
   :func:`distance_to_centreline`）。
2. **三款车肉眼可辨。** 这一条没法用断言代替 —— 所以脚本把三款车各出一张图，
   外加一张三款头车并排的**对比图**。涂装与侧影的差别，最后还是要人看一眼；
   脚本能做的只是把客观侧影尺寸同时打出来（``长 × 宽 × 高``）。

与游戏里走同一条代码路径
--------------------------------------------------------------------
``TrackEditor`` / ``TrainView`` / ``LayoutView`` 都是真的实例，只是把"每帧的 dt"
换成固定值、相机换成脚本摆的位置。所以这里出问题，游戏里一定也出问题。

一处刻意的"摆拍"
--------------------------------------------------------------------
对比图里三节头车是**一节一节各自单独**摆在同一段直轨上的（每节都是一个只有一
节的编组）。真实编组当然不会这样。它只是为了把三款头车塞进同一个画面，
所以单独出一张图，且**不参与**任何"能跑"的断言。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panda3d.core import Filename, PNMImage, Vec3, loadPrcFileData  # noqa: E402

from core.track.catalog import Catalog  # noqa: E402
from core.train.consist import TrainCatalog  # noqa: E402

#: 整圆用几节 R40 的 45° 弯轨。
ARC_PIECES = 8
#: 仿真步长（秒）。固定步长 ⇒ 每次跑出来的位置一样，出图可复现。
DT = 1.0 / 60.0
#: 出图前让列车跑多久（秒）。够绿皮车加速到明显速度，又不至于跑掉一整圈。
RUN_SECONDS = 40.0
#: 对比图里相邻两节头车的中心距（米）。要比车长（约 26 m）大，否则会叠在一起。
COMPARE_SPACING = 31.0
#: 三节头车各自的起始弧长（米），让相机正好在正中间那节上。
COMPARE_HEAD = 70.0

#: 三个车系各挑一列代表（绿皮车 / 和谐号 / 复兴号）。
FAMILIES = ("green_skin_10", "crh380a_8", "cr400af_8")
#: 对比图用哪三种头车（代表三个车系的侧影与涂装）。
COMPARE_CARS = ("df4b_loco", "crh380a_end", "cr400af_end")


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


# --------------------------------------------------------------------------- #
# 场景构造
# --------------------------------------------------------------------------- #

def build_circle(catalog: Catalog):
    """8 节 R40/45° 拼一个整圆并合拢，返回 ``(layout, 首件)``。"""
    from core.track.layout import Layout

    layout = Layout(catalog=catalog)
    first = layout.add_root("curve_r40_l45")
    index = first
    for _ in range(ARC_PIECES - 1):
        index = layout.attach("curve_r40_l45", "a", (index, "b"))
    layout.connect((index, "b"), (first, "a"))
    return layout, first


def build_straight(catalog: Catalog, count: int = 8):
    """``count`` 节 20 m 直轨串成一条开链（默认 160 m）。"""
    from core.track.layout import Layout

    layout = Layout(catalog=catalog)
    first = layout.add_root("straight_20")
    index = first
    for _ in range(count - 1):
        index = layout.attach("straight_20", "a", (index, "b"))
    return layout, first


def adopt(editor, layout) -> None:
    """把脚本拼好的布局整个交给编辑器，并让它重新检测闭环、重算路径。

    要写两处 ``layout``：``LayoutView`` 在构造时就把布局**引用**抓住了
    （它是增量同步的基准），所以换布局必须两边一起换。
    """
    editor.layout = layout
    editor.view.layout = layout
    editor.view.sync()


def distance_to_centreline(path, point) -> float:
    """世界点到轨道中心线的最短距离（米）。

    与 ``tests/test_train_view.py`` 里的同名辅助函数是同一套闭式解：中心线是分段
    等曲率圆弧（直线是退化情形），在每段的入口帧里就是"圆心在 ``(0, R)``、
    半径 ``R = 段长 / 段转角``"的圆，逐段算取最小。**不用采样** —— 采样法的
    量化误差（0.01 m 步长下最多偏 5 mm）比要测的毫米级量还大。
    """
    from core.geometry import Pose

    best = float("inf")
    for segment in path.segments:
        local = segment.entry_frame.inverse().compose(
            Pose(point[0], point[1], point[2], 0.0)
        )
        x, z = local.x, local.z
        if abs(segment.dtheta) < 1e-12:
            u = min(max(x, 0.0), segment.length)
            best = min(best, math.hypot(x - u, z))
            continue
        radius = segment.length / segment.dtheta          # 与转角同号
        # 弧上参数角 du 满足 (R sin du, R (1 - cos du))，故 du = atan2(x, R - z)
        ang = math.atan2(x, radius - z)
        ang = min(max(ang, min(0.0, segment.dtheta)), max(0.0, segment.dtheta))
        best = min(best, math.hypot(x - radius * math.sin(ang),
                                    z - radius * (1.0 - math.cos(ang))))
    return best


# --------------------------------------------------------------------------- #
# 出图
# --------------------------------------------------------------------------- #

def frame_nodes(camera, nodes, *, margin: float = 1.25, height: float = 4.0) -> None:
    """把一组节点的世界包围盒放进画面（俯视一点的角度）。"""
    low = [float("inf")] * 3
    high = [float("-inf")] * 3
    for node in nodes:
        lo, hi = node.getTightBounds()
        for axis in range(3):
            low[axis] = min(low[axis], lo[axis])
            high[axis] = max(high[axis], hi[axis])
    if not all(math.isfinite(v) for v in low + high):
        raise AssertionError("包围盒是空的 —— 节点没进场景？")
    camera.frame(((low[0], low[2]), (high[0], high[2])), margin=margin)
    camera.target[1] = height
    camera.apply()


def shoot(base, output: Path) -> tuple[int, int]:
    base.graphicsEngine.renderFrame()
    base.graphicsEngine.renderFrame()
    image = PNMImage()
    if not base.win.getScreenshot(image):
        raise AssertionError("截图失败")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.write(Filename.fromOsSpecific(str(output)))
    return (image.getXSize(), image.getYSize())


def dims(node) -> tuple[float, float, float]:
    """节点的世界包围盒尺寸，按 yup 的轴序返回 ``(长, 高, 宽)``。

    ``coordinate-system yup`` 下 X 是纵向、Y 是高度、Z 是横向，所以
    ``getTightBounds`` 的第二个分量是**车高**、第三个是**车宽** —— 这两个很容易
    顺手写反（第一版就写反了，打出来的"宽 3.7 m / 高 2.8 m"其实是高 3.7 / 宽 2.8）。
    """
    lo, hi = node.getTightBounds()
    return (abs(hi[0] - lo[0]), abs(hi[1] - lo[1]), abs(hi[2] - lo[2]))


# --------------------------------------------------------------------------- #
# 三列车系上环线
# --------------------------------------------------------------------------- #

def spawn_family(editor, train_id: str):
    """把目录里指定的那一列召上线（``spawn_train`` 是"换下一列"，所以转着找）。"""
    view = None
    for _ in range(len(editor.trains)):
        view = editor.spawn_train(1)
        if view is not None and view.spec.id == train_id:
            return view
    raise AssertionError(f"转过整本目录也没找到 {train_id}"
                         f"（最后一个是 {view and view.spec.id}）")


def run_family(editor, train_id: str, output: Path, seconds: float) -> None:
    """把一个车系召上线、推满手柄、跑一段时间、出图。"""
    view = spawn_family(editor, train_id)
    spec = view.spec
    print(f"\n--- {spec.name}")

    if view.placement_error is not None:
        raise AssertionError(f"上不了环线：{view.placement_error}")
    print(f"  {view.car_count} 节 / 编组全长 {view.consist_length:.1f} m "
          f"/ 环线 {editor.view.path.total_length:.1f} m")

    editor.push_train_handle(1.0)
    for _ in range(int(seconds / DT)):
        editor._advance_train(DT)

    if view.state.v <= 1.0:
        raise AssertionError(f"推满手柄 {seconds:.0f} s 后速度仍只有 "
                             f"{view.state.v:.3f} m/s")
    if view.state.distance <= 1.0:
        raise AssertionError(f"几乎没有走（{view.state.distance:.3f} m）")

    # 每节车都还得在场，而且两个转向架都还压在轨道中心线上
    worst = 0.0
    for index, car in enumerate(spec.cars):
        node = view.node_for(index)
        if node is None or node.isHidden():
            raise AssertionError(f"第 {index} 节车不见了")
        for local_x in (car.bogie_half_spacing, -car.bogie_half_spacing):
            drawn = node.getMat().xformPoint(Vec3(local_x, 0.0, 0.0))
            worst = max(worst, distance_to_centreline(
                editor.view.path, (drawn.x, drawn.y, drawn.z)))
    if worst > 1.5e-3:
        raise AssertionError(f"转向架离轨道中心线 {worst * 1000:.3f} mm")

    frame_nodes(editor.camera, [view.root])
    size = shoot(editor.base, output)
    print(f"  {view.state.speed_kmh:6.1f} km/h，已走 {view.state.distance:7.1f} m"
          f"，转向架最大离轨 {worst * 1000:.3f} mm")
    print(f"  三角形 {view.triangle_count():,}；图 {output.name} "
          f"({size[0]}×{size[1]})")


# --------------------------------------------------------------------------- #
# 三款头车并排（摆拍）
# --------------------------------------------------------------------------- #

def pick_car(trains: TrainCatalog, car_id: str):
    """在目录里找到含有该车型的那一列，返回它的一节**单车编组**。"""
    for spec in trains:
        for car in spec.cars:
            if car.id == car_id:
                return replace(spec, id=f"{car_id}_solo", cars=(car,),
                               coupling_gap=0.0)
    raise AssertionError(f"目录里没有 {car_id} 这个车型")


def build_compare_rig(editor, trains: TrainCatalog):
    """三节头车各自单独摆在同一条直轨上（摆拍，见模块文档）。"""
    from render.train_view import TrainView

    views = []
    heads = [COMPARE_HEAD + i * COMPARE_SPACING for i in range(len(COMPARE_CARS))]
    for car_id, head in zip(COMPARE_CARS, heads):
        view = TrainView(pick_car(trains, car_id), editor.base.render)
        view.set_handle(0.0)
        view.state.s = head
        if not view.sync(editor.view.path):
            raise AssertionError(f"{car_id} 摆不上对比轨道：{view.placement_error}")
        views.append(view)
    return views


def shoot_game_view(editor, output: Path) -> tuple[int, int]:
    """出图：**游戏里的样子** —— 带 HUD 的列车运行画面。

    前面的图都是把相机取景到列车包围盒上、不加 HUD 的"证件照"；这一张反过来，
    用跟游戏一致的斜俯视角 + v2 HUD（左栏 + 底栏控制台），用来确认
    时速表 / Tomix 手柄 / 状态芯片真的出现，并且没跟别的面板压在一起。
    """
    view = editor.train_view
    if view is None:
        raise AssertionError("要出游戏视角的图，得先有一列车在线上")
    head = view.node_for(0).getPos(editor.base.render)
    editor.camera.target = [head[0], 1.6, head[2]]
    editor.camera.distance = max(90.0, 1.5 * view.consist_length)
    editor.camera.elevation = math.radians(24.0)
    editor.camera.azimuth = math.radians(-46.0)
    editor.camera.apply()

    editor._refresh_hud()                    # 让 HUD 反映刚才那一段运行
    text = editor.hud.panel_text("train")
    for want in ("速度", "手柄", "里程"):
        if want not in text:
            raise AssertionError(f"列车面板里没有「{want}」：{text!r}")
    if "km/h" not in text:
        raise AssertionError(f"列车面板没报速度：{text!r}")
    print(f"  列车面板：{' / '.join(line.strip() for line in text.splitlines())}")
    return shoot(editor.base, output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="离屏跑列车并出图")
    parser.add_argument("-o", "--output", default=None,
                        help="对比图的输出路径（其余图固定写到 docs/）")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--seconds", type=float, default=RUN_SECONDS,
                        help="每列车出图前跑多久（秒）")
    parser.add_argument("--docs", default=None, help="出图目录，默认 docs/")
    args = parser.parse_args(argv)

    loadPrcFileData("", "coordinate-system yup")
    loadPrcFileData("", "window-type offscreen")
    loadPrcFileData("", f"win-size {args.width} {args.height}")
    loadPrcFileData("", "audio-library-name null")
    loadPrcFileData("", "sync-video false")
    loadPrcFileData("", "notify-level-display error")

    from direct.showbase.ShowBase import ShowBase

    from app import config
    from app.editor import TrackEditor
    from app.hud import build_default_hud
    from render import ground as ground_mod

    base = ShowBase()
    base.disableMouse()
    # 与真正的窗口渲染走同一份场景配置（天空色 / 雾 / 裁剪面）
    config.configure_scene(base)
    ground_mod.build_ground().reparentTo(base.render)

    docs = Path(args.docs) if args.docs else ROOT / "docs"
    catalog = Catalog.builtin()
    trains = TrainCatalog.builtin()
    for problem in catalog.validate() + trains.validate():
        print(f"[数据告警] {problem}")
    print(f"件库 {len(catalog)} 件；列车目录 {len(trains)} 列："
          f"{'、'.join(trains.ids)}")

    editor = TrackEditor(base, catalog, hud=build_default_hud(base),
                         trains=trains)

    banner("环线")
    layout, _ = build_circle(catalog)
    adopt(editor, layout)
    closure = editor.view.closure
    if closure is None or not closure.closed:
        raise AssertionError(f"没拼出闭环：{closure and closure.describe()}")
    print(f"  {closure.describe()}")

    for train_id in FAMILIES:
        run_family(editor, train_id, docs / f"scene_train_{train_id}.png",
                   args.seconds)

    banner("游戏视角（带 HUD 的运行画面）")
    game_size = shoot_game_view(editor, docs / "scene_train_game.png")
    print(f"  图 scene_train_game.png ({game_size[0]}×{game_size[1]})")

    banner("三款头车并排（摆拍对比图）")
    editor.dismiss_train()
    straight, _ = build_straight(catalog, 8)              # 160 m
    adopt(editor, straight)
    views = build_compare_rig(editor, trains)
    frame_nodes(editor.camera, [v.root for v in views], margin=1.12, height=3.5)
    editor.camera.azimuth = math.radians(-92.0)
    editor.camera.elevation = math.radians(15.0)
    editor.camera.apply()
    compare_path = Path(args.output) if args.output else \
        docs / "scene_train_compare.png"
    size = shoot(base, compare_path)

    for car_id, view in zip(COMPARE_CARS, views):
        car = view.spec.cars[0]
        length, height, width = dims(view.node_for(0))
        print(f"  {car_id:<14} {car.name:<12} 侧影 长 {length:5.2f} × "
              f"高 {height:4.2f} × 宽 {width:4.2f} m")
    print(f"  图 {compare_path.name} ({size[0]}×{size[1]})")

    banner("结果")
    print(f"  三列编组都跑起来了，出图在 {docs}")
    print("  请肉眼确认：绿皮车的黄腰带、CRH380A 的蓝带、CR400AF 的红飘带，")
    print("  以及三种头型（钝圆 / 尖长 / 低伏前伸）在图上分得开。")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
