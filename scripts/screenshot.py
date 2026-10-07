"""离屏渲染一个场景为 PNG：用于自检与文档配图。

用法::

    python scripts/screenshot.py                      # 默认渲染 catalog 全景
    python scripts/screenshot.py circle -o out.png     # 指定场景与输出
    python scripts/screenshot.py valley                # 现成场景之一（**带布景**）
    python scripts/screenshot.py --list                # 看有哪些场景

为什么要离屏渲染
------------------------------------------------
它把"改完代码到底长什么样"变成一个**可复现的、不依赖人眼实时操作**的检查：
CI 里能跑、diff 里能看、讨论时有共同的参照物。参数化了相机与分辨率，
所以同一场景可以固定角度出图的对比。

现成场景（``valley`` / ``gorge`` / ``town`` / ``lake``）走 :mod:`scenes.presets`，
与 ``app/main.py --scene`` 是**同一份**构造代码 —— 所以这里出的图就是游戏里的样子
（唯一差别是没有交互与 HUD）。它们自带布景，出图时顺手把"布景离轨道最近的净距"
打印出来：这是最值得盯的一个数，山压住轨道一眼未必看得出，数字不会骗人。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panda3d.core import Filename, PNMImage, Vec3, loadPrcFileData  # noqa: E402

import scenes  # noqa: E402
from core.geometry import Pose  # noqa: E402
from core.track.catalog import Catalog  # noqa: E402
from core.track.layout import Layout, LayoutError  # noqa: E402
from render import style  # noqa: E402
from scenes.tracks import SceneError  # noqa: E402

#: 每个场景一个构造函数：``(layout) -> None``，往布局里摆轨道。
SCENES: dict[str, str] = {
    "circle": "8 节 R40/45° 左弯 + 合拢接缝 → 整圆",
    "figure8": "8 节左弯 + 8 节右弯 → 两个外切的整圆（8 字形）",
    "yard": "道岔 + 交叉 + 直线 → 站场雏形",
    "ramp": "上坡引桥 + 高架直线 + 下坡引桥 → 立交跨线",
    "rails": "一节直轨 + 一节弯轨 → 近景看道砟/枕木/钢轨断面",
    "catalog": "件库全景（每 60 m 摆一件，不受闭环约束）",
}

#: 场景名字的来源：本文件里的轨道小样 + ``scenes.presets`` 里的现成场景。
PRESETS = scenes.keys()


# --------------------------------------------------------------------------- #
# 场景构造
# --------------------------------------------------------------------------- #

def scene_circle(layout: Layout) -> None:
    first = layout.add_root("curve_r40_l45")
    last = first
    for _ in range(7):
        last = layout.attach("curve_r40_l45", "a", (last, "b"))
    layout.connect((last, "b"), (first, "a"))


def scene_figure8(layout: Layout) -> None:
    """8 字形：同向 8 节拼一个整圆，再反向 8 节拼另一个，两圆外切于原点。

    关键几何：`curve_r40_l45` 从原点、heading=0 出发时圆心在 (0, +40)；
    换成 `curve_r40_r45` 圆心就在 (0, -40)。两心距 80 m 正好等于 2R，
    因此两圆在**原点**外切 —— 不需要任何交叉件，接缝也落在原点。
    """
    first = layout.add_root("curve_r40_l45")
    last = first
    for _ in range(7):
        last = layout.attach("curve_r40_l45", "a", (last, "b"))

    # 暂不接缝：先把反向一圈挂到 last.b（它正好回到原点、heading=0）
    entry = layout.attach("curve_r40_r45", "a", (last, "b"))
    tail = entry
    for _ in range(7):
        tail = layout.attach("curve_r40_r45", "a", (tail, "b"))

    # 合拢：tail.b（原点，外指向 0°）接 first.a（原点，外指向 180°）
    layout.connect((tail, "b"), (first, "a"))


def scene_yard(layout: Layout) -> None:
    """站场雏形：一条主线带左右开道岔与 45° 菱形交叉。"""
    root = layout.add_root("straight_40")
    main = layout.attach("turnout_l_40", "a", (root, "b"))
    layout.attach("straight_20", "a", (main, "b"))
    branch = layout.attach("curve_r40_l45", "a", (main, "c"))
    layout.attach("straight_10", "a", (branch, "b"))

    # 从 root 的另一端也接一组，形成分叉的观感
    back = layout.attach("turnout_r_40", "a", (root, "a"))
    layout.attach("straight_20", "a", (back, "b"))
    cross = layout.attach("crossing_45", "a", (back, "c"))
    layout.attach("straight_20", "a", (cross, "c"))


def scene_ramp(layout: Layout) -> None:
    """剖面：地面 → 3% 上坡（路堤放坡到地面）→ 高架直线 → 3% 下坡 → 地面。"""
    root = layout.add_root("ramp_up_40")
    top = layout.attach("straight_40", "a", (root, "b"))
    down = layout.attach("ramp_down_40", "a", (top, "b"))
    layout.attach("straight_40", "a", (down, "b"))


def scene_rails(layout: Layout) -> None:
    """一节直轨接一节弯轨：近景检查断面（道砟坡、枕木、钢轨工字形）。"""
    root = layout.add_root("straight_20")
    layout.attach("curve_r40_l22_5", "a", (root, "b"))


def _catalog_grid(parent, catalog: Catalog) -> tuple[float, float]:
    """把件库每一件摆成一个网格，返回 ``(跨度x, 跨度z)``。"""
    from render import style, track_mesh
    from render.transform import apply_pose

    columns = 4
    spacing_x, spacing_z = 62.0, 46.0
    for i, piece in enumerate(catalog):
        # 弯轨 / 道岔会伸出自己的格子，按类别给一点额外留白
        col, row = i % columns, i // columns
        builder = track_mesh.build_piece_mesh(piece, ground_local_y=style.GROUND_Y)
        node = builder.build()
        node.setName(f"catalog_{piece.id}")
        apply_pose(node, Pose(col * spacing_x, 0.0, row * spacing_z, 0.0))
        node.reparentTo(parent)
    rows = (len(catalog) + columns - 1) // columns
    return (columns - 1) * spacing_x, (rows - 1) * spacing_z


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="离屏渲染一个场景为 PNG")
    parser.add_argument("scene", nargs="?", default="catalog",
                        choices=sorted(set(SCENES) | set(PRESETS)))
    parser.add_argument("-o", "--output", default=None, help="输出 PNG 路径")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--distance", type=float, default=None,
                        help="相机距离（默认按场景自动取景）")
    parser.add_argument("--azimuth", type=float, default=-55.0)
    parser.add_argument("--elevation", type=float, default=34.0)
    parser.add_argument("--no-ports", action="store_true", help="不画空闲端口标记")
    parser.add_argument("--no-loop", action="store_true", help="不画闭环彩带")
    parser.add_argument("--save", default=None, metavar="PATH",
                        help="把该场景的布局存成存档 JSON"
                             "（之后可以直接 app/main.py --open 打开它接着拼）")
    parser.add_argument("--open", dest="open_path", default=None, metavar="PATH",
                        help="渲染一份**存档**（而不是内置场景）；"
                             "用来在不开窗口的情况下确认存档里到底有什么")
    parser.add_argument("--hud", action="store_true",
                        help="带上 HUD 渲染（与游戏里看到的一致）")
    parser.add_argument("--train", default=None, metavar="ID",
                        help="召唤一列编组上轨（例如 cr400af_8）；会走编辑器那条路径")
    parser.add_argument("--list", action="store_true", help="列出可用场景")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list:
        for name in sorted(SCENES):
            print(f"  {name:10s} {SCENES[name]}")
        print("  现成场景（自带布景，与 app/main.py --scene 同一份代码）：")
        catalog = Catalog.builtin()
        for key in PRESETS:
            scene = scenes.build(key, catalog)
            print(f"  {key:10s} {scene.title}　{scene.blurb}")
        return 0
    if args.save and args.scene == "catalog":
        print("[错误] catalog 是件库全景，不构成一张装配图，无法存档")
        return 2

    loadPrcFileData("", "coordinate-system yup")
    loadPrcFileData("", "window-type offscreen")
    loadPrcFileData("", f"win-size {args.width} {args.height}")
    loadPrcFileData("", "audio-library-name null")
    loadPrcFileData("", "sync-video false")
    loadPrcFileData("", "notify-level-display error")

    from direct.showbase.ShowBase import ShowBase

    from app import config
    from render import camera as camera_mod
    from render import ground as ground_mod
    from render import scenery as scenery_mod
    from render import scene as scene_mod

    base = ShowBase()
    # 与真正的窗口渲染走**同一份**场景配置（天空色 / 雾 / 裁剪面）。
    # 这里以前是自己抄了一份设置，结果 configure_scene 成了没人调用的死代码，
    # 里面的 LColor(*SKY_COLOR) 少一个 alpha 分量一直没被发现。
    config.configure_scene(base)

    span = None
    view = None
    layout = None
    editor = None
    preset = None
    #: 场地边长。现成场景自带（山要站得下），其余用默认值。
    plot = style.PLOT_SIZE
    catalog = Catalog.builtin()

    if args.open_path:
        # 渲染一份存档 —— 用来在**不开窗口**的情况下确认"存档里到底有什么"。
        # 这条路径曾经是坏的（编辑器建了视图却忘了同步一次，于是载入 8 节
        # circle.json 画面全空），所以这里刻意也支持 --hud：把游戏里的样子
        # 原样离屏画出来，比看数据可靠。
        try:
            loaded = scenes.load(args.open_path, catalog)
        except (OSError, LayoutError, KeyError, SceneError) as exc:
            print(f"[错误] 载入存档 {args.open_path} 失败：{exc}")
            return 2
        layout = loaded.layout
        if loaded.scenery_key:
            # 存档只记"当时配的是哪个预设"，布景按预设重算 —— 与 main.py --open 一致
            preset = scenes.build(loaded.scenery_key, catalog)
        plot = loaded.plot_size
        source = Path(args.open_path).stem
    elif args.scene in PRESETS:
        preset = scenes.build(args.scene, catalog)
        layout = preset.layout
        plot = preset.plot_size
        source = args.scene
    elif args.scene == "catalog":
        span = (0.0, 0.0)                       # 由 _catalog_grid 稍后改写
        source = "catalog"
    else:
        layout = Layout(catalog=catalog)
        try:
            globals()[f"scene_{args.scene}"](layout)
        except LayoutError as exc:
            print(f"[错误] 场景 {args.scene} 构造失败：{exc}")
            return 2
        source = args.scene

    ground_mod.build_ground(plot).reparentTo(base.render)

    if preset is not None and preset.scenery:
        # 布景烘成一个几何节点（一次绘制调用），与游戏里走同一条路径
        scenery_node = scenery_mod.build_scenery(preset.scenery)
        scenery_node.reparentTo(base.render)
        scenery_mod.attach_street_lights(base.render, preset.scenery)
        clearance, tightest = preset.clearance()
        print(f"[布景] {preset.scenery.item_count} 件，"
              f"离轨道最近的净距 {clearance:.2f} m（{tightest}）")

    if args.scene == "catalog":
        parent = base.render.attachNewNode("catalog")
        span = _catalog_grid(parent, catalog)

    if layout is not None:
        if args.hud or args.train:
            # 与游戏走**同一条**构造路径（main.py 也是这样建的）
            from app.editor import TrackEditor
            from app.hud import build_default_hud

            editor = TrackEditor(base, catalog, layout=layout,
                                 hud=build_default_hud(base) if args.hud else None)
            if args.hud:
                editor._refresh_hud()
            view = editor.view
        else:
            view = scene_mod.LayoutView(layout, base.render)
            view.set_show_ports(not args.no_ports)
            view.set_show_loop(not args.no_loop)

        print(f"{source}：件数 {len(layout)}，"
              f"总长 {layout.total_length():.3f} m，"
              f"三角形 {view.triangle_count()}")
        if view.closure is not None:
            print("闭环检测：", view.closure.describe())
        if args.save and not args.open_path:
            target = Path(args.save)
            target.parent.mkdir(parents=True, exist_ok=True)
            if preset is not None:
                # 场景走 scenes.save：存档里会多记一行"布景用哪个预设"
                scenes.save(preset, target)
            else:
                layout.save_json(target)
            print(f"已存档 {target}（{target.stat().st_size} 字节）")
        if args.save and args.open_path:
            print("[提示] --open 是只读渲染，--save 已忽略（不想覆盖你正在看的存档）")

        if args.train:
            spawned = editor.spawn_train(step=0, train_id=args.train)
            if spawned is None:
                print(f"[错误] 无法上线列车 {args.train}")
                return 2
            print(f"列车 {spawned.spec.id}：{spawned.spec.name}，"
                  f"{spawned.car_count} 节 / {spawned.consist_length:.1f} m")
            if spawned.placement_error:
                print("摆位：", spawned.placement_error)

    camera = editor.camera if editor is not None else camera_mod.OrbitCamera(
        base, azimuth_deg=args.azimuth, elevation_deg=args.elevation,
    )
    camera.azimuth = math.radians(args.azimuth)
    camera.elevation = math.radians(args.elevation)
    if args.distance is not None:
        camera.distance = args.distance
        camera.apply()
    elif editor is not None:
        editor.frame_layout()
    elif preset is not None:
        # 取景要把布景一起收进来（只看轨道的话，山全在画面外）
        camera.frame(preset.camera_bounds(), margin=1.15)
    elif view is not None:
        camera.frame(view.bounds())
    elif span is not None:
        camera.target = [span[0] * 0.5, 0.0, span[1] * 0.5]
        camera.distance = max(span) * 1.15
        camera.apply()

    if args.train and editor is not None and editor.train_view is not None:
        # 默认取景是整条线路，车只是画面里一条细线。有 --train 时围着头车鼻锥。
        from panda3d.core import Vec3 as _V3
        head = editor.train_view.node_for(0)
        if head is not None:
            nose = head.getMat().xformPoint(_V3(4.5, 1.5, 0.0))
            camera.target = [nose.x, nose.y, nose.z]
            if args.distance is None:
                camera.distance = 22.0
            camera.apply()

    # 默认文件名要能区分"内置场景"与"你的一份存档" —— 否则 ``--open saves/circle.json``
    # 会正好盖掉内置场景的 ``scene_circle.png``，让人以为内置场景出了问题。
    if args.output:
        output = Path(args.output)
    elif args.open_path:
        output = ROOT / "docs" / f"open_{source}.png"
    else:
        output = ROOT / "docs" / f"scene_{source}.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    base.graphicsEngine.renderFrame()
    base.graphicsEngine.renderFrame()
    image = PNMImage()
    if not base.win.getScreenshot(image):
        print("[错误] 截图失败")
        return 3
    image.write(Filename.fromOsSpecific(str(output)))
    print(f"已写出 {output}  ({image.getXSize()}×{image.getYSize()})")
    print(f"相机 target={camera.target} distance={camera.distance:.1f} "
          f"azimuth={math.degrees(camera.azimuth):.1f}° "
          f"elevation={math.degrees(camera.elevation):.1f}°")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
