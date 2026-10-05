"""PE 导入表工具：读一个二进制静态依赖了哪些 DLL，并把漏掉的补回来。

为什么需要它
------------------------------------------------
打包时有两类"少了一个 DLL"的坑，两类都**不会在构建时报错**，只在运行时炸：

1. **插件不是 import 进来的。** Panda3D 的图形/音频管道（``libpandagl.dll`` /
   ``libp3openal_audio.dll``）是运行时按名字 ``LoadLibrary`` 的，打包器的静态分析
   看不见它们，必须人工点名。
2. **依赖在搜索路径之外。** ``DLLs\\_ctypes.pyd`` 静态依赖 ``ffi.dll``，而 Anaconda
   把 ``ffi.dll`` 放在 ``Library\\bin\\`` —— 既不在 ``_ctypes.pyd`` 旁边，也不在
   ``sys.path`` 上，于是 PyInstaller 的依赖分析找不到它、也不会报错。结果是打包后
   ``import ctypes`` 抛 ``ImportError: DLL load failed while importing _ctypes``。
   这个坑特别阴：游戏本体不用 ctypes，所以**游戏看着一切正常**，只有"出错时该弹的
   那个窗"没了 —— 恰好在最需要它的时候静默失效（本项目真踩过）。

所以这里只做一件事：**读 PE 导入表**（不看名字猜），然后据此把一个二进制集合的
依赖补成闭包。判断系统 DLL 也靠名单，不靠"名字像不像"。
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

#: Windows 自带、目标机器上一定有的 DLL，不需要跟着 exe 走。
#: ``api-ms-win-*`` / ``ext-ms-*`` 是 API set，同样是系统提供的。
SYSTEM_PREFIXES = ("api-ms-win-", "ext-ms-")
SYSTEM_NAMES = {
    "kernel32.dll", "user32.dll", "gdi32.dll", "advapi32.dll", "shell32.dll",
    "ole32.dll", "oleaut32.dll", "comdlg32.dll", "comctl32.dll", "shlwapi.dll",
    "ws2_32.dll", "wsock32.dll", "winmm.dll", "imm32.dll", "msimg32.dll",
    "opengl32.dll", "glu32.dll", "gdiplus.dll", "dwmapi.dll", "uxtheme.dll",
    "ntdll.dll", "rpcrt4.dll", "secur32.dll", "crypt32.dll", "bcrypt.dll",
    "version.dll", "setupapi.dll", "dbghelp.dll", "psapi.dll", "iphlpapi.dll",
    "userenv.dll", "avrt.dll", "powrprof.dll", "d3d9.dll", "d3d11.dll",
    "d3d12.dll", "dxgi.dll", "dinput8.dll", "xinput1_4.dll", "wininet.dll",
    "msvcrt.dll", "ucrtbase.dll",
    # 下面这些也是系统自带，只是不常出现在扩展模块的导入表里，所以第一次跑
    # 整包检查时被当成了"漏掉的"。``propsys`` 是 Windows 属性系统（``_wmi`` 用它）。
    "propsys.dll", "wintrust.dll", "cryptbase.dll", "ncrypt.dll", "dnsapi.dll",
    "mswsock.dll", "normaliz.dll", "winhttp.dll", "mpr.dll", "netapi32.dll",
    "winspool.drv", "odbc32.dll", "wtsapi32.dll",
    # MSVC 运行库：pyinstaller 会自己带上，见 RUNTIME_NAMES 里的那份说明
    "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll",
    "concrt140.dll", "vccorlib140.dll",
}

#: **PyInstaller 自己会**收进来的 DLL（Python 解释器与它的运行库），
#: 所以它们在"我的显式名单"里缺席不算漏 —— 但它们在"整包检查"里必须真的在。
RUNTIME_NAMES = {"python3.dll", "python312.dll", "python311.dll"}


def is_system(name: str) -> bool:
    name = name.lower()
    return name in SYSTEM_NAMES or name.startswith(SYSTEM_PREFIXES)


def is_runtime(name: str) -> bool:
    name = name.lower()
    return name in RUNTIME_NAMES or name.startswith("python3")


def _rva_to_offset(rva: int, sections: list[tuple[int, int, int]]) -> int | None:
    for vaddr, vsize, raw in sections:
        if vaddr <= rva < vaddr + vsize:
            return raw + (rva - vaddr)
    return None


def imports_of(path: str | Path) -> list[str]:
    """读 PE 导入表，返回**静态依赖**的 DLL 名字（小写，含 ``.dll``）。

    运行时 ``LoadLibrary`` 起来的插件本函数看不到 —— 那些只能人工点名，见
    :data:`check_dlls` 里的保留名单。

    实现是手写的最小 PE 解析：节表 → 数据目录第 1 项（导入表）→ 描述符数组 →
    名字表。不引第三方库（``pefile`` 只在装了 PyInstaller 的机器上有，而这个工具
    要能在随便哪个 Python 上跑）。
    """
    data = Path(path).read_bytes()
    if data[:2] != b"MZ":
        raise ValueError(f"{Path(path).name} 不是 PE 文件")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError(f"{Path(path).name} 的 PE 头不对")
    n_sections = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)      # PE32+ 与 PE32 差 16 字节
    import_rva = struct.unpack_from("<I", data, dd + 8)[0]

    sections: list[tuple[int, int, int]] = []
    sec = opt + opt_size
    for i in range(n_sections):
        base = sec + i * 40
        vsize = struct.unpack_from("<I", data, base + 8)[0]
        vaddr = struct.unpack_from("<I", data, base + 12)[0]
        raw_size = struct.unpack_from("<I", data, base + 16)[0]
        raw = struct.unpack_from("<I", data, base + 20)[0]
        sections.append((vaddr, max(vsize, raw_size), raw))

    names: list[str] = []
    if import_rva == 0:
        return names
    off = _rva_to_offset(import_rva, sections)
    if off is None:
        return names
    while True:                                     # 每个描述符 20 字节，全零收尾
        desc = data[off:off + 20]
        if len(desc) < 20 or desc == b"\0" * 20:
            break
        name_rva = struct.unpack_from("<I", desc, 12)[0]
        if name_rva:
            name_off = _rva_to_offset(name_rva, sections)
            if name_off is not None:
                end = data.index(b"\0", name_off)
                names.append(data[name_off:end].decode("ascii", "replace").lower())
        off += 20
    return names


def default_search_dirs() -> list[Path]:
    """PyInstaller 的依赖分析**不会**去的地方 —— 补依赖时要额外翻一遍。

    Anaconda 把一堆运行库放在 ``Library\\bin``（``ffi.dll`` 就在这儿），既不是哪个
    扩展模块的旁边，也不在 ``sys.path`` 上，正是"找不到也不报错"的那一类。
    """
    roots = {Path(sys.prefix), Path(sys.base_prefix)}
    dirs: list[Path] = []
    for root in roots:
        for sub in ("", "DLLs", "Library/bin", "Library/mingw-w64/bin", "Scripts"):
            candidate = root / sub if sub else root
            if candidate.is_dir() and candidate not in dirs:
                dirs.append(candidate)
    return dirs


def find_dll(name: str, search_dirs: list[Path] | None = None) -> Path | None:
    """按名字找一个 DLL：先看 PATH/环境目录，再翻 :func:`default_search_dirs`。"""
    for folder in search_dirs or []:
        hit = Path(folder) / name
        if hit.is_file():
            return hit
    for folder in default_search_dirs():
        hit = folder / name
        if hit.is_file():
            return hit
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        if not folder:
            continue
        hit = Path(folder) / name
        if hit.is_file():
            return hit
    return None


def complete_binaries(entries, *, search_dirs=None, log=print):
    """把 ``entries`` 里每个二进制的静态依赖补成闭包。

    ``entries`` 是 PyInstaller 的 ``binaries`` 列表：``(源路径, 目标目录)``。
    返回 ``(补齐后的列表, 说明文字列表)``；新增的 DLL 一律放到包的**根目录**
    （与 ``_ctypes.pyd`` 这类顶层扩展模块同处一级，那是 Windows 一定会搜的地方）。

    递归：补进来的 DLL 自己也可能有没被收的依赖（``ffi.dll`` 就有）。
    """
    search_dirs = list(search_dirs or [])
    known: dict[str, str] = {}                      # 小写文件名 → 目标目录
    for src, dest in entries:
        known[Path(src).name.lower()] = dest
        parent = str(Path(src).parent)
        if parent not in search_dirs:
            search_dirs.append(parent)

    result = list(entries)
    notes: list[str] = []
    queue = list(entries)
    while queue:
        src, _dest = queue.pop(0)
        try:
            deps = imports_of(src)
        except (OSError, ValueError) as exc:
            notes.append(f"读不了 {Path(src).name} 的导入表：{exc}")
            continue
        for dep in deps:
            if dep in known or is_system(dep) or is_runtime(dep):
                continue
            found = find_dll(dep, search_dirs)
            if found is None:
                notes.append(f"{Path(src).name} 依赖 {dep}，但环境里找不到它 —— "
                             f"打包后会在 import 时报 'DLL load failed'")
                continue
            known[dep] = "."
            result.append((str(found), "."))
            queue.append((str(found), "."))
            notes.append(f"补上 {dep}（{Path(src).name} 依赖它，"
                         f"来自 {found.parent}）")
    return result, notes
