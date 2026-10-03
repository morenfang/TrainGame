"""离屏跑一遍编辑器的完整交互流程 —— 编辑器的端到端自检脚本。

    python scripts/editor_demo.py            # 拼一个整圆并出图
    python scripts/editor_demo.py -o out.png

它验证的是什么
------------------------------------------------
单元测试可以证明"吸附公式对"，但证明不了"**用鼠标**拼得出来"。这个脚本把
真实操作链完整走一遍，而且每一步都做出**可证伪的断言**：

1. 把鼠标移到某个空闲端口的屏幕位置 —— 这个位置是用 :meth:`TrackEditor._project`
   算出来的，和用户眼睛看到的是同一个投影；
2. 让编辑器自己 ``_refresh_hover() -> _refresh_ghost()``；
3. 断言 **hover_port 恰好就是要接的那个端口**（屏幕拾取没选错）；
4. 断言 **幽灵预览的位姿 == 实放后的位姿**（预览与实际不可能不一致）；
5. 摆完最后一节后断言接缝缺口 ≈ 0，按 C 合拢，再断言 ``closure.closed``。

全程不做任何几何计算 —— 一旦吸附公式、屏幕投影、端口拾取、闭环判定任何一环
出错，脚本会立刻失败并指出是哪一步。这就是"编辑器能用"这句话的证明。
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

from panda3d.core import Filename, PNMImage, loadPrcFileData  # noqa: E402

from core.track.catalog import Catalog  # noqa: E402
from render import style  # noqa: E402

#: 拼一个整圆用几节：R40 的 45° 弯轨刚好 8 节一圈。
ARC_PIECES = 8
#: 一键闭合那一段：22.5° 弯轨接多少节。「8 节 = 180°」正是用户最容易停下的位置
#: （接满一圈要 16 节），此时接缝差一个直径 80 m。
AUTO_CLOSE_SEED = 8
#: 真闭环的接缝缺口上限（米）。见 core.track.path：真闭环在 1e-13 量级。
CLOSED_TOL = 1e-9


class ScriptedEditor:
    """把"鼠标在哪"换成脚本给定的归一化坐标的编辑器。

    只覆盖两个取鼠标的方法 —— 这正是"鼠标输入"与"编辑器逻辑"之间唯一的接口，
    所以脚本驱动和真人驾驶走的是**同一条**代码路径。
    """

    @staticmethod
    def mixin(editor_class):
        class _Scripted(editor_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fake_mouse = (0.0, 0.0)
                self.fake_inside = True

            def _has_mouse(self) -> bool:
                return self.fake_inside

            def _mouse(self):
                return self.fake_mouse

        return _Scripted


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def aim_at_world(editor, world) -> tuple[float, float]:
    """把脚本鼠标移到某个世界点在屏幕上的位置（归一化坐标）。"""
    pixel = editor._project(world)
    if pixel is None:
        raise AssertionError(f"世界点 {world} 投影到了屏幕外/相机后方")
    width = editor.base.win.getXSize()
    height = editor.base.win.getYSize()
    editor.fake_mouse = (pixel[0] / (width * 0.5), pixel[1] / (height * 0.5))
    return pixel


def settle(editor) -> None:
    """让编辑器按当前鼠标位置重算悬停与幽灵（等价于"下一帧"）。"""
    editor._refresh_hover()
    editor._refresh_ghost()


def check_aim_picked(editor, expected, label: str) -> None:
    """断言屏幕拾取选中的正是期望的端口，并在失败时报出实际选中的是谁。"""
    if editor.hover_port != expected:
        raise AssertionError(
            f"{label}：期望吸附到 {expected}，实际选中 {editor.hover_port}"
        )


def run_arc(editor, catalog: Catalog, piece_id: str = "curve_r40_l45",
            count: int = ARC_PIECES) -> list[int]:
    """用鼠标在屏幕上把 ``count`` 节 ``piece_id`` 接成一段弧，返回件下标序列。"""
    if piece_id not in catalog:
        raise AssertionError(f"件库里没有 {piece_id}，无法拼弧")
    # 编辑器放的是"当前选中的件"，所以必须先明确选中它 —— 这一步同时也是在
    # 验证"按 id 选件"这条路径。
    if not editor.select_piece_id(piece_id):
        raise AssertionError(f"无法选中 {piece_id}")
    if editor.current_piece.id != piece_id:
        raise AssertionError(
            f"选中后当前件是 {editor.current_piece.id}，不是 {piece_id}"
        )
    if editor._attach_port(editor.current_piece) != "a":
        raise AssertionError("接弧需要按 'a' 端口接驳")

    placed: list[int] = []

    # 第一节：把鼠标挪到原点，自由放置
    banner(f"第 1 节：鼠标指向地面原点，自由放置（{catalog[piece_id].name}）")
    aim_at_world(editor, (0.0, style.GROUND_Y, 0.0))
    settle(editor)

    placement = editor.placement
    if placement is None:
        raise AssertionError("鼠标在地面上，编辑器却没给出放置预览")
    if placement.snapped:
        raise AssertionError("空场地上的第一节不该是吸附放置")
    if not placement.allowed:
        raise AssertionError(f"空场地上自由放置被拒绝：{placement.note}")

    index = editor.place()
    if index is None:
        raise AssertionError("放置第一节失败")
    placed.append(index)
    drift = editor._placement_error(index, placement)
    if drift > CLOSED_TOL:
        raise AssertionError(f"预览与实放位姿差了 {drift} m")
    print(f"  放下 #{index}，预览/实放位姿一致（差 {drift:.3e} m）")

    # 后续：每次都瞄准"上一节的 b 端口"
    for step in range(2, count + 1):
        target = (placed[-1], "b")
        world = editor.layout.world_port(*target)
        aim_at_world(editor, world.position)
        settle(editor)

        check_aim_picked(editor, target, f"第 {step} 节瞄准 {target}")

        placement = editor.placement
        if placement is None:
            raise AssertionError(f"瞄准 {target} 时编辑器没有给出预览")
        if not placement.snapped:
            raise AssertionError(f"瞄准 {target} 时预览没有进入吸附状态")

        index = editor.place()
        if index is None:
            raise AssertionError(f"第 {step} 节放置失败")
        placed.append(index)

        drift = editor._placement_error(index, placement)
        if drift > CLOSED_TOL:
            raise AssertionError(f"第 {step} 节预览与实放位姿差了 {drift} m")
        print(f"  接上 #{index}（接到 {target}），误差 {drift:.3e} m")

    return placed


def grab(base, output: Path) -> tuple[int, int]:
    """渲染两帧后截图存盘，返回像素尺寸。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    base.graphicsEngine.renderFrame()
    base.graphicsEngine.renderFrame()
    image = PNMImage()
    if not base.win.getScreenshot(image):
        raise AssertionError("截图失败")
    image.write(Filename.fromOsSpecific(str(output)))
    return image.getXSize(), image.getYSize()


def run_auto_close(editor, catalog: Catalog, base, docs: Path) -> None:
    """**一键闭合**：用鼠标拼半圈 22.5° 弯轨，然后让 C 自己补完另半圈。

    这是用户实际撞上的坑：曲线分类里第一个件就是 22.5° 弯轨，而它要 **16 节**
    才绕满一圈。用鼠标接 8 节正好停在直径的另一头 —— 接缝差 80 m，按老版本的 C
    只会得到一句"合不拢"，而"还差 8 节"这件事用户根本没法心算。

    这里量的是"用户看到的那条链路"：鼠标真接出来的半圈 → C → 真闭环 → 周长等于
    2πR。同时也量了提示：按之前 HUD 上必须写着还差几节。
    """
    banner("场景二：一键闭合（半圈 22.5° 弯轨 → 按 C）")
    editor.clear()
    piece_id = "curve_r40_l22_5"
    placed = run_arc(editor, catalog, piece_id, AUTO_CLOSE_SEED)
    if len(placed) != AUTO_CLOSE_SEED:
        raise AssertionError("半圈没铺完")

    seam = editor.best_seam()
    if seam is None:
        raise AssertionError("半圈之后找不到接缝候选")
    _pair, (distance, _heading_gap) = seam
    print(f"  铺了 {len(placed)} 节，接缝还差 {distance:.3f} m")
    if abs(distance - 80.0) > 1e-6:
        raise AssertionError(f"半圈的接缝应当是直径 80 m，实测 {distance}")
    if editor.seam_is_aligned():
        raise AssertionError("半圈居然被判成已经对齐")

    # 按 C 之前，HUD 上必须先说清"还差几节" —— 否则用户没有任何线索
    hint = editor.auto_close_hint()
    print(f"  HUD 提示：{hint}")
    if f"再补 {AUTO_CLOSE_SEED} 节" not in hint:
        raise AssertionError(f"提示没数对节数：{hint}")

    editor._refresh_hud()
    if "按 C" not in editor.hud.panel_text("loop"):
        raise AssertionError("提示没画到 HUD 上")
    before_png = docs / "scene_autoclose_before.png"
    print(f"  截图 {before_png.name}（半圈 + 提示）  "
          f"{grab(base, before_png)}")

    banner("按 C 一键闭合")
    if not editor.close_loop():
        raise AssertionError(f"一键闭合失败：{editor._toast}")
    print(f"  {editor._toast}")
    if len(editor.layout) != 2 * AUTO_CLOSE_SEED:
        raise AssertionError(f"补完应当是 16 节，实际 {len(editor.layout)}")

    closure = editor.view.closure
    if closure is None or not closure.closed:
        raise AssertionError(f"补完之后仍未闭环：{closure}")
    if closure.visit_count != 2 * AUTO_CLOSE_SEED:
        raise AssertionError(f"闭环段数 {closure.visit_count} 不对")
    if abs(closure.total_length - 2 * math.pi * 40.0) > 1e-6:
        raise AssertionError(
            f"闭环周长 {closure.total_length} 不等于 2πR"
        )
    print(f"  {closure.describe()}")

    # 一笔撤销：不能逼用户按 8 次退格
    editor.undo()
    if len(editor.layout) != AUTO_CLOSE_SEED:
        raise AssertionError("自动补齐不是一笔撤销")
    editor.redo()
    if len(editor.layout) != 2 * AUTO_CLOSE_SEED:
        raise AssertionError("重做之后闭环丢了")
    print("  撤销/重做都是**一笔**（不是 8 笔）")

    banner("一键闭合后召唤列车跑起来")
    view = editor.spawn_train(1)
    if view is None or not view.visible:
        raise AssertionError(f"闭环上放不下列车：{view and view.placement_error}")
    editor.push_train_handle(1.0)
    for _ in range(600):                      # 10 s
        editor._advance_train(1.0 / 60.0)
    if view.state.distance <= 1.0:
        raise AssertionError("列车在自动补出来的环上没有跑起来")
    print(f"  列车跑了 {view.state.distance:.1f} m，速度 "
          f"{view.state.v * 3.6:.1f} km/h")
    for _ in range(3):
        editor.tick()

    after_png = docs / "scene_autoclose_after.png"
    print(f"  截图 {after_png.name}（整圈 + 列车）  {grab(base, after_png)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="离屏驱动编辑器拼一个整圆并出图")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
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
    base.disableMouse()          # 相机完全交给 OrbitCamera / 脚本
    # 与真正的窗口渲染走同一份场景配置（否则截图里的天空和雾跟游戏里对不上）
    config.configure_scene(base)
    ground_mod.build_ground().reparentTo(base.render)

    catalog = Catalog.builtin()
    problems = catalog.validate()
    if problems:
        print(f"[件库告警] {len(problems)} 条")
        for problem in problems:
            print("   ", problem)

    scripted = ScriptedEditor.mixin(TrackEditor)
    editor = scripted(base, catalog, hud=build_default_hud(base))

    # 先取景到"整圆会占的地方"，这样中途不需要再动相机
    editor.camera.frame(((-50.0, -15.0), (50.0, 90.0)))

    placed = run_arc(editor, catalog)

    banner("接缝检查")
    free = sorted(editor.layout.free_ports())
    seam = editor.best_seam()
    if seam is None:
        raise AssertionError("拼完一圈后居然找不到接缝候选")
    (a, b), (distance, heading_gap) = seam
    print(f"  空闲端口 {free}")
    print(f"  接缝 {a} → {b}：位置差 {distance:.3e} m，航向差 "
          f"{math.degrees(heading_gap):.3e}°")
    if distance > CLOSED_TOL or heading_gap > CLOSED_TOL:
        raise AssertionError(f"接缝没对齐：差 {distance} m / {heading_gap} rad")
    if not editor.seam_is_aligned():
        raise AssertionError("seam_is_aligned() 与实测缺口不一致")

    if editor.view.closure is not None and editor.view.closure.closed:
        raise AssertionError("还没合拢，闭环检测却已经报闭环")

    banner("按 C 合拢")
    if not editor.close_loop():
        raise AssertionError("close_loop() 报告失败，但接缝是齐的")
    closure = editor.view.closure
    if closure is None or not closure.closed:
        raise AssertionError(f"合拢后仍未闭环：{closure}")
    print(f"  {closure.describe()}")

    expected_length = ARC_PIECES * 40.0 * math.radians(45.0)
    if abs(closure.total_length - expected_length) > 1e-6:
        raise AssertionError(
            f"闭环周长 {closure.total_length} 与理论值 {expected_length} 不符"
        )
    if closure.visit_count != ARC_PIECES:
        raise AssertionError(f"闭环段数 {closure.visit_count} != {ARC_PIECES}")

    banner("撤销 / 重做 / 存档往返")
    before = len(editor.layout)
    editor.undo()
    if editor.view.closure is None or editor.view.closure.closed:
        raise AssertionError("撤销合拢之后不该还是闭环")
    editor.redo()
    if editor.view.closure is None or not editor.view.closure.closed:
        raise AssertionError("重做合拢之后应当重新闭环")
    if len(editor.layout) != before:
        raise AssertionError("撤销+重做之后件数变了")

    scratch = ROOT / "saves" / "editor_demo_roundtrip.json"
    editor.save_path = scratch
    if not editor.save():
        raise AssertionError("保存失败")
    editor.clear()
    if not editor.load():
        raise AssertionError("读档失败")
    reloaded = editor.view.closure
    if reloaded is None or not reloaded.closed:
        raise AssertionError(f"读档后闭环丢失：{reloaded}")
    print(f"  存读往返后仍是闭环：{reloaded.describe()}")
    scratch.unlink(missing_ok=True)

    banner("鼠标悬停在轨道上（用于截图里的高亮与 HUD）")
    aim_at_world(editor, editor.layout.piece(0).pose.position)
    settle(editor)
    for _ in range(3):           # 跑几帧真实 update，顺带压一遍 HUD
        editor.tick()

    output = Path(args.output) if args.output else ROOT / "docs" / "scene_editor.png"
    size = grab(base, output)

    run_auto_close(editor, catalog, base, output.parent)

    banner("结果")
    print(f"  闭环 {closure.visit_count} 段 / {closure.total_length:.6f} m，"
          f"共放置 {len(placed)} 节")
    print(f"  画面三角形 {editor.view.triangle_count():,}")
    print(f"  已写出 {output}  ({size[0]}×{size[1]})")
    print("\n编辑器的鼠标操作链全部通过。")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
