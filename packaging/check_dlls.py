"""检查"打出来的包里，每个 DLL 依赖是不是都在"。

两种跑法
------------------------------------------------
::

    python packaging/check_dlls.py            # 打包前：检查 spec 里那份保留名单
    python packaging/check_dlls.py --bundle   # 打包后：检查真的打进包里的一切

``--bundle`` 读 PyInstaller 写在 ``build/TrainGame/Analysis-00.toc`` 里的清单，
把**每一个**二进制（不只是我们点名的那些）的静态依赖过一遍，报出"需要、但包里
没有、又不是系统 DLL"的那些。这一条是真正抓 bug 的：``DLLs\\_ctypes.pyd`` 依赖
``ffi.dll`` 这种事，只有把整个清单摊开看才看得见 —— 而它一旦漏掉，症状是
"游戏一切正常，只有出错该弹的窗没了"。

打包前那条检查另外一种东西：spec 里那份 panda3d 保留名单**没多也没少**。
"没少"是指名单里每个二进制的依赖都在名单内（靠 :mod:`dllclosure` 算）；
"没多"是指名单里每个 glob 都真的匹配到了文件（写错名字要立刻发现）。

保留名单不从本文件抄，而是 ``ast`` 解析 spec 里的 ``_PANDA_BINARIES`` —— 抄一份
迟早会不同步，而不同步的表现是"工具说没问题、exe 起不来"。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "TrainGame.spec"
TOC = ROOT / "build" / "TrainGame" / "Analysis-00.toc"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dllclosure import (  # noqa: E402
    imports_of, is_runtime, is_system,
)

#: PyInstaller 清单里代表"二进制"的条目类型。
_BINARY_CODES = ("BINARY", "EXTENSION")


def _parse_spec_binaries(spec: Path = SPEC) -> tuple[str, ...]:
    """从 spec 里读 ``_PANDA_BINARIES``。

    spec 是**脚本**不是模块（里面引用了 PyInstaller 注入的 ``Analysis`` 之类的
    名字），不能 import，但可以解析 —— 这样保留名单只有一处定义。
    """
    tree = ast.parse(spec.read_text(encoding="utf-8"), filename=str(spec))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "_PANDA_BINARIES" in targets:
                return tuple(ast.literal_eval(node.value))
    raise SystemExit(f"{spec} 里找不到 _PANDA_BINARIES —— 保留名单改了？")


def _panda_dir() -> Path:
    import panda3d

    return Path(panda3d.__file__).resolve().parent


def _expand(patterns: tuple[str, ...], folder: Path) -> tuple[list[Path], list[str]]:
    found: list[Path] = []
    problems: list[str] = []
    for pattern in patterns:
        hits = sorted(folder.glob(pattern))
        if not hits:
            problems.append(f"保留名单里的 {pattern!r} 在 {folder} 下一个文件都没匹配到")
        found.extend(hits)
    return found, problems


def check_keep_list(*, show_tree: bool = False) -> int:
    """打包前：spec 的保留名单是否自洽（依赖闭合、名字都对）。"""
    folder = _panda_dir()
    patterns = _parse_spec_binaries()
    keep, problems = _expand(patterns, folder)
    keep_names = {p.name.lower() for p in keep}

    print(f"panda3d: {folder}")
    print(f"保留名单: {len(patterns)} 条 glob → 命中 {len(keep)} 个文件，"
          f"共 {sum(p.stat().st_size for p in keep) / 1048576:.1f} MB\n")

    for binary in keep:
        deps = imports_of(binary)
        missing = sorted({d for d in deps
                          if d not in keep_names and not is_system(d)
                          and not is_runtime(d)})
        if show_tree:
            print(f"{binary.name}（{len(deps)} 个直接依赖）")
            for dep in deps:
                where = "打包" if dep in keep_names else "系统/运行库"
                print(f"    {where}  {dep}")
        if missing:
            problems.append(f"{binary.name} 依赖 {', '.join(missing)}"
                            f" —— 既不在保留名单里，也不是系统 DLL")

    if problems:
        print("✗ 保留名单有问题：")
        for line in problems:
            print("   -", line)
        return 1
    print("✓ 保留名单自洽：名单里每个二进制需要的非系统 DLL 都在名单内、"
          "每条 glob 都命中。")
    print("  （运行时 LoadLibrary 起来的图形/音频管道本检查看不见 —— 它们是")
    print("    libpandagl.dll 与 libp3openal_audio.dll，已在 spec 里显式点名。）")
    return 0


def _read_toc() -> list[tuple[str, str, str]]:
    """把 PyInstaller 的 ``Analysis-00.toc`` 读成 ``(目标名, 源路径, 类型)`` 列表。

    那个文件是个 Python 字面量（不是 pickle），所以 ``ast.literal_eval`` 就够。
    不写死结构：递归找出所有"三个字符串、第三个是类型码"的元组 —— PyInstaller
    改过几次清单布局，写死层级会跟着坏。
    """
    if not TOC.exists():
        raise SystemExit(f"没找到 {TOC}。先跑一次打包（scripts/build_exe.py）。")
    parsed = ast.literal_eval(TOC.read_text(encoding="utf-8"))
    entries: list[tuple[str, str, str]] = []

    def walk(node):
        if isinstance(node, (list, tuple)):
            if (len(node) == 3 and all(isinstance(x, str) for x in node)
                    and node[2] in ("BINARY", "EXTENSION", "DATA", "OPTION")):
                entries.append(node)            # type: ignore[arg-type]
                return
            for item in node:
                walk(item)

    walk(parsed)
    return entries


def check_bundle() -> int:
    """打包后：整个包里的二进制依赖是否闭合。"""
    entries = _read_toc()
    binaries = [(dest, src) for dest, src, code in entries if code in _BINARY_CODES]
    if not binaries:
        raise SystemExit(f"{TOC} 里一条 BINARY 都没读到 —— 清单格式变了？")

    # 用**文件名**判断"包里有":PyInstaller 会把 DLL 按原名放到包根或它的子目录，
    # 两种位置都在搜索路径里（子目录那条由 packaging/rthook_dlls.py 负责）。
    present = {Path(dest).name.lower() for dest, _src in binaries}
    present |= {Path(dest).name.lower() for dest, _src, code in entries
                if code == "DATA"}

    problems: list[str] = []
    for dest, src in binaries:
        try:
            deps = imports_of(src)
        except (OSError, ValueError) as exc:
            problems.append(f"{Path(dest).name}：读不了导入表（{exc}）")
            continue
        missing = sorted({d for d in deps
                          if d not in present and not is_system(d)})
        if missing:
            problems.append(
                f"{Path(dest).name} 需要 {', '.join(missing)} —— 包里有、但它没有。"
                f"（源：{src}）")

    print(f"包里二进制 {len(binaries)} 个（清单 {TOC.relative_to(ROOT)}）\n")
    if problems:
        print("✗ 依赖没闭合，这个包发出去会在运行时炸：")
        for line in problems:
            print("   -", line)
        print("\n  修法：在 packaging/TrainGame.spec 里把它加进 binaries"
              "（目标目录写 \".\"），或让 dllclosure.complete_binaries 补上。")
        return 1
    print("✓ 闭包完整：包里每个二进制的非系统依赖都能在包里找到。")
    return 0


def main(argv: list[str]) -> int:
    if "--bundle" in argv:
        return check_bundle()
    return check_keep_list(show_tree="--tree" in argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
