"""交互式入口：打开窗口，进入轨道编辑器沙盘。

    python app/main.py                       # 空场地开工
    python app/main.py --list-scenes         # 看看有哪些现成场景
    python app/main.py --scene valley        # 直接开一局：山谷环线，列车已经上线
    python app/main.py --open saves/layout.json
    python app/main.py --width 1600 --height 900

打开后就是"在一片草垫子上拼轨道"：鼠标移到网格上点左键放下第一节，
之后每一节都吸到淡绿色的空闲端口上。屏幕四角常驻 HUD，右上角实时告诉你
"现在这套轨道闭合成环了没有、差多少毫米"。

``--scene`` 与布景
------------------------------------------------
场景 = **闭环轨道 + 程序化布景**（山、河、湖、房子、树、草、站台）。轨道在
:mod:`scenes.tracks` 里拼、布景在 :mod:`scenes.presets` 里摆，两者都在启动时现
算 —— 没有网格文件要读，也不依赖任何下载资源。进来按 N 就把列车放上线。

首帧之前会跑一遍 ``Catalog.validate()``，把件库里的几何问题打印到控制台 ——
拼不出闭环最可能的原因就是件库里有件被改坏了，让它在启动时就暴露出来。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scenes                              # noqa: E402
from app import config                     # noqa: E402
from app.editor import TrackEditor         # noqa: E402
from app.hud import build_default_hud      # noqa: E402
from core.track.catalog import Catalog     # noqa: E402
from render import ground as ground_mod    # noqa: E402
from render import style                   # noqa: E402
from scenes.tracks import SceneError       # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D 火车沙盘 —— 轨道编辑器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--width", type=int, default=1440, help="窗口宽度（像素）")
    parser.add_argument("--height", type=int, default=900, help="窗口高度（像素）")
    parser.add_argument("--scene", default=None,
                        help="启动时拼一个现成场景（见 --list-scenes）后直接开跑，"
                             "例如 --scene valley")
    parser.add_argument("--list-scenes", action="store_true",
                        help="列出所有现成场景后退出")
    parser.add_argument("--open", dest="open_path", default=None,
                        help="启动时载入的存档（JSON）")
    parser.add_argument("--save", dest="save_path", default=None,
                        help="Ctrl+S 的保存位置")
    parser.add_argument("--no-vsync", dest="vsync", action="store_false",
                        help="关闭垂直同步（画面可能撕裂，但延迟更低）")
    parser.add_argument("--offscreen", action="store_true",
                        help="不开窗口（自检用；配合 --frames 可以当冒烟测试跑）")
    parser.add_argument("--frames", type=int, default=0,
                        help="渲染这么多帧后自动退出；0 = 一直跑")
    return parser.parse_args(argv)



def _print_scenes(catalog: Catalog) -> None:
    """``--list-scenes``：把几张牌摊开 —— 不建窗口，纯打印。"""
    from core.train.consist import TrainCatalog

    trains = TrainCatalog.builtin()
    print("现成场景（用 ``run.bat --scene <名字>`` 直接开跑）：\n")
    for key in scenes.keys():
        try:
            scene = scenes.build(key, catalog)
        except SceneError as exc:
            print(f"  {key:<8} ✗ 拼不出来：{exc}")
            continue
        spec = trains[scene.train_id] if scene.train_id in trains else None
        print(f"  {key:<8} {scene.title}　环长 {scene.layout.total_length():.1f} m，"
              f"{len(scene.layout)} 节；建议编组 {spec.name if spec else '（目录里没有）'}")
        print(f"           {scene.blurb}")
        print(f"           布景 {scene.scenery.item_count} 件，场地 "
              f"{scene.plot_size:.0f} m × {scene.plot_size:.0f} m")
    print("\n进去之后：N 换车（自动上线），↑↓ 推手柄，空格 惰行，"
          "Shift+空格 急停，H 收起 HUD，Ctrl+S 保存。")


def _load_startup_layout(path: str, catalog: Catalog):
    """读一个存档：轨道照旧，布景看存档里记的预设（没有就不摆）。"""
    loaded = scenes.load(path, catalog)
    print(f"[启动] 已载入 {len(loaded.layout)} 节轨道：{path}")
    if loaded.scenery_key:
        print(f"[启动] 存档配的布景是 {loaded.scenery_key!r}（按预设重算）")
    else:
        print("[启动] 这份存档没记布景预设，只摆轨道")
    return loaded

def main(argv=None) -> int:
    args = parse_args(argv)

    catalog = Catalog.builtin()
    if args.list_scenes:
        _print_scenes(catalog)
        return 0

    config.configure_engine(width=args.width, height=args.height,
                            offscreen=args.offscreen, vsync=args.vsync)

    # 必须等引擎配置好之后再 import ShowBase
    from direct.showbase.ShowBase import ShowBase

    base = ShowBase()
    config.configure_scene(base)
    base.disableMouse()          # 相机完全交给 OrbitCamera

    problems = catalog.validate()
    for problem in problems:
        print(f"[件库告警] {problem}")
    if problems:
        print(f"[件库告警] 共 {len(problems)} 条；这些件拼装后可能对不齐端口")

    scene = None
    loaded = None
    if args.scene:
        try:
            scene = scenes.build(args.scene, catalog)
        except SceneError as exc:
            # 命令行给错名字是很常见的事，不该甩一屏 traceback
            print(f"[错误] {exc}")
            return 2
        print(f"[场景] {scene.title}：{scene.blurb}")
        print(f"[场景] 环长 {scene.layout.total_length():.1f} m，{len(scene.layout)} 节；"
              f"布景 {scene.scenery.item_count} 件，场地 {scene.plot_size:.0f} m 见方")
    elif args.open_path:
        loaded = _load_startup_layout(args.open_path, catalog)

    # ---- 场地：场景自己知道要多大的地，别用默认的 400 m 把山切在场地外
    plot = scene.plot_size if scene is not None else (
        loaded.plot_size if loaded is not None else style.PLOT_SIZE)
    ground_mod.build_ground(plot).reparentTo(base.render)

    # ---- 布景：程序化底座（场景预设 / 存档记的预设重算）+ 用户手工摆的那几件。
    # 这两层都交给编辑器去烘成一个节点 —— 这样布景模式能就地增删、随时重建。
    base_scenery = None
    user_scenery = None
    if scene is not None:
        base_scenery = scene.scenery
    elif loaded is not None:
        base_scenery = loaded.scenery
        user_scenery = loaded.user_scenery

    layout = scene.layout if scene is not None else (
        loaded.layout if loaded is not None else None)

    # 存档里记一行"这局配的是哪份布景"，Ctrl+S 存回去时不会丢
    save_extra: dict[str, object] = {}
    if scene is not None:
        save_extra = {scenes.SCENERY_KEY: scene.key, "title": scene.title}
    elif loaded is not None and loaded.scenery_key:
        save_extra = {scenes.SCENERY_KEY: loaded.scenery_key}

    editor = TrackEditor(base, catalog, layout=layout,
                         hud=build_default_hud(base),
                         save_path=args.save_path or args.open_path,
                         save_extra=save_extra,
                         scenery=base_scenery, user_scenery=user_scenery)
    editor.bind()

    if layout is not None and not layout.is_empty:
        if scene is not None:
            # 场景模式下取景要把布景也收进去 —— 只看轨道的话，山就全在画面外
            editor.camera.frame(scene.camera_bounds(), margin=1.15)
        else:
            editor.frame_layout()

    if scene is not None:
        # 场景是"直接就能跑"的，所以列车自己上线；用户按 ↑ 就走
        editor.spawn_train(step=0, train_id=scene.train_id)
    else:
        editor.notify("鼠标移到网格上，左键放下第一节轨道")

    base.taskMgr.add(editor.update, "track_editor_update")
    if args.frames > 0:
        base.taskMgr.add(_frame_limiter(base, args.frames), "frame_limiter")

    print(f"[启动] {len(catalog)} 个轨道件，{len(catalog.categories)} 个类别。"
          f" 按 H 收起 HUD，Ctrl+S 保存。")
    base.run()
    return 0


def _frame_limiter(base, frames: int):
    """渲染够 ``frames`` 帧就自己退出（自检 / 冒烟测试用）。"""
    remaining = [frames]

    def task(_task):
        remaining[0] -= 1
        if remaining[0] <= 0:
            base.userExit()
            return _task.done
        return _task.cont

    return task


if __name__ == "__main__":
    raise SystemExit(main())
