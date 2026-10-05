"""一条命令把「模拟铁轨」打成 Windows exe。

    python scripts/build_exe.py                # 单文件 dist/TrainGame.exe（默认）
    python scripts/build_exe.py --onedir       # 目录版，启动快、好排查
    python scripts/build_exe.py --skip-smoke   # 只打包，不跑冒烟测试
    python scripts/build_exe.py --no-clean     # 复用上次的 build/（打包更快）

它会依次做七件事，**任何一步失败就当场停下**（打出来的 exe 起不来比没打更糟）：

1. **检查保留名单**（``packaging/check_dlls.py``）：panda3d 那堆 DLL 里哪些删不得
   是量出来的，不是猜的 —— 这一步专门防"照名字删掉 ``cg.dll``，exe 开窗口就挂"。
2. **画图标**（``packaging/make_icon.py``）：颜色直接引 ``render.style`` 与车体涂装。
3. **清缓存**：``build/`` 里的旧分析结果会把删掉的模块带回来，打包脚本自己清最省心。
4. **跑 PyInstaller**（``packaging/TrainGame.spec``）。
5. **复核整包依赖**（``check_dlls.py --bundle``）：读 PyInstaller 的清单，逐个二进制
   核它的依赖在不在包里。这一步抓的是"少了一个 DLL 也能打出来、只在运行时炸"那类
   —— 本项目的 ``ffi.dll`` 就是这么被抓出来的。
6. **铺附件**：示例存档、操作手册 PDF、README 复制到 exe 旁边 —— ``--open
   saves\\valley.json`` 这种命令在文档里到处都是，得让它在 exe 上也成立。
7. **冒烟测试**：拿真 exe 跑 ``--offscreen --frames 5``，并把日志尾部打出来。

最后给一句"能发出去的东西在哪"。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUILD = ROOT / "build"
DIST = ROOT / "dist"
SPEC = ROOT / "packaging" / "TrainGame.spec"
#: 冒烟测试的帧数。5 帧足够走到"开窗口 / 建场景 / 渲染 / 退出"这一整条路。
SMOKE_FRAMES = 5
SMOKE_TIMEOUT = 300


def _run(cmd: list[str], *, what: str, timeout: int | None = None) -> int:
    print(f"\n=== {what} ===")
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, timeout=timeout)
    if proc.returncode != 0:
        raise SystemExit(f"[失败] {what}：退出码 {proc.returncode}")
    return proc.returncode


def ensure_pyinstaller() -> None:
    """没有 PyInstaller 就装一个 —— "一条命令"不该从手动装包开始。"""
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[准备] 没装 PyInstaller，先装一个（pip install pyinstaller）")
        _run([sys.executable, "-m", "pip", "install", "pyinstaller"],
             what="安装 PyInstaller")
        return
    import PyInstaller

    print(f"[准备] PyInstaller {PyInstaller.__version__}")


def clean(onefile: bool) -> None:
    for path in (BUILD, DIST):
        if path.exists():
            shutil.rmtree(path)
            print(f"[清理] 删掉 {path.relative_to(ROOT)}")
    for leftover in ROOT.glob("*.spec"):
        # 直接在仓库根跑 pyinstaller 会生成一个临时 spec，别让它留在仓库里
        if leftover.resolve() != SPEC.resolve():
            leftover.unlink()
            print(f"[清理] 删掉根目录的临时 spec {leftover.name}")


def pyinstaller(onefile: bool) -> Path:
    env = dict(os.environ)
    env["TRAIN3D_ONEFILE"] = "1" if onefile else "0"
    print("\n=== 打包 ===")
    print(f"$ pyinstaller {SPEC.relative_to(ROOT)}   (TRAIN3D_ONEFILE="
          f"{env['TRAIN3D_ONEFILE']})", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", str(SPEC)],
        cwd=str(ROOT), env=dict(env, PYTHONIOENCODING="utf-8"),
    )
    if proc.returncode != 0:
        raise SystemExit(f"[失败] PyInstaller 退出码 {proc.returncode}")
    exe = DIST / "TrainGame.exe"
    if not exe.exists():
        raise SystemExit(f"[失败] 没找到 {exe}")
    return exe


def stage_assets() -> list[str]:
    """把示例存档与文档放到 exe 旁边。返回放过去的东西（好打印）。"""
    staged: list[str] = []
    saves = DIST / "saves"
    saves.mkdir(parents=True, exist_ok=True)
    for src in sorted((ROOT / "saves").glob("*.json")):
        shutil.copy(src, saves / src.name)
        staged.append(f"saves/{src.name}")
    for src, name in ((ROOT / "README.md", "README.md"),
                      (ROOT / "docs" / "模拟铁轨-操作手册.pdf", "模拟铁轨-操作手册.pdf")):
        if src.exists():
            shutil.copy(src, DIST / name)
            staged.append(name)
    return staged


def smoke(exe: Path) -> bool:
    """拿真 exe 跑一遍离屏渲染 —— 这是"打出来能不能用"最便宜的答案。"""
    print("\n=== 冒烟测试（离屏渲染 5 帧） ===")
    log = DIST / "train3d_log.txt"
    if log.exists():
        log.unlink()
    started = time.time()
    try:
        proc = subprocess.run(
            [str(exe), "--offscreen", "--frames", str(SMOKE_FRAMES)],
            cwd=str(DIST), timeout=SMOKE_TIMEOUT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        code = proc.returncode
    except subprocess.TimeoutExpired:
        print(f"[冒烟] ✗ {SMOKE_TIMEOUT} 秒没退出来 —— 卡住了")
        return False
    elapsed = time.time() - started

    print(f"[冒烟] 退出码 {code}，用时 {elapsed:.1f} 秒")
    if log.exists():
        tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        print("[冒烟] 日志尾部：")
        for line in tail[-12:]:
            print("        " + line)
        lowered = "\n".join(tail).lower()
        for bad in ("couldn't load", "traceback", "无法加载"):
            if bad in lowered:
                print(f"[冒烟] ✗ 日志里出现 {bad!r}")
                return False
    else:
        print("[冒烟] （没有日志文件）")

    if code != 0:
        print("[冒烟] ✗ 退出码不是 0")
        return False
    print("[冒烟] ✓ exe 能起场景、能渲染、能自己退出")
    return True


def human(size: int) -> str:
    return f"{size / 1048576:.1f} MB"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="把模拟铁轨打成 Windows exe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--onedir", action="store_true",
                        help="打成目录版（启动快、方便排查），默认是单文件")
    parser.add_argument("--skip-smoke", action="store_true", help="跳过冒烟测试")
    parser.add_argument("--no-clean", action="store_true",
                        help="不清 build/ 与 dist/，打包更快")
    args = parser.parse_args(argv)
    onefile = not args.onedir

    ensure_pyinstaller()
    _run([sys.executable, str(ROOT / "packaging" / "check_dlls.py")],
         what="检查 panda3d 依赖闭包")
    _run([sys.executable, str(ROOT / "packaging" / "make_icon.py")],
         what="生成图标")
    if not args.no_clean:
        clean(onefile)

    exe = pyinstaller(onefile)
    # 包里到底有没有缺 DLL —— 读 PyInstaller 的清单逐个二进制核一遍。
    # 必须**打包之后**跑：这一步看的是真打进包里的东西，而不是我们以为自己打了什么。
    _run([sys.executable, str(ROOT / "packaging" / "check_dlls.py"), "--bundle"],
         what="检查整包依赖闭包")
    staged = stage_assets()
    print(f"\n=== 铺附件（{len(staged)} 项）===")
    for name in staged:
        print("    " + name)

    ok = True
    if not args.skip_smoke:
        ok = smoke(exe)

    print("\n" + "=" * 62)
    if onefile:
        print(f"单文件：{exe}（{human(exe.stat().st_size)}）")
        print(f"发出去就是这一个文件；存档会写在它旁边的 saves\\ 里。")
    else:
        total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
        print(f"目录版：{DIST}（共 {human(total)}）")
        print(f"入口：{exe}")
    if staged:
        print("附件（连同 exe 一起发）：" + "、".join(staged))
    print("=" * 62)
    if not ok:
        print("\n[注意] 冒烟测试没过 —— 这个 exe 先别发出去。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
