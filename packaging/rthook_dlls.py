"""运行时钩子：让 Windows 能找到 panda3d 的 DLL 与"插件"。

为什么需要这个
------------------------------------------------
``panda3d/core.pyd`` 依赖同目录的 ``libpanda.dll``，而真正干活的**图形与音频管道**
（``libpandagl.dll`` / ``libp3openal_audio.dll``）不是被 import 进来的，是 Panda3D
在运行时**按名字**（配置里的 ``load-display pandagl`` / ``audio-library-name
p3openal_audio``）去 ``LoadLibrary`` 的。打包器静态分析看不见这种加载，Windows 的
默认搜索路径里也没有它们所在的子目录。

漏掉这一步的症状很具有迷惑性：``import panda3d`` 正常、几何算得飞快、``--list-scenes``
一切正常，**一开窗口**就报 "Couldn't load libpandagl.dll" 然后自己退出。所以这里
在**任何 panda3d 模块被导入之前**把目录塞进搜索路径。

三件事都要做，因为 Windows 的三种加载方式各认各的：

* ``os.add_dll_directory``   —— 认它的只有带 ``LOAD_LIBRARY_SEARCH_USER_DIRS`` 的加载；
* ``SetDllDirectoryW``       —— 裸名 ``LoadLibrary("libpandagl.dll")`` 走这条；
* ``PATH``                   —— 兜住"孙辈"依赖（例如 ``cg.dll`` 再去加载 ``cgGL.dll``）。

钩子在 ``--onefile`` 与 ``--onedir`` 下都跑：前者 DLL 在临时解包目录，后者在 exe
旁边，靠 :func:`_base_dir` 分辨。
"""

import os
import sys
from pathlib import Path


def _base_dir() -> Path:
    """``--onefile`` 用临时解包目录，``--onedir`` 用 exe 旁边的 ``_internal``。"""
    unpack = getattr(sys, "_MEIPASS", None)
    if unpack:
        return Path(unpack)
    return Path(sys.executable).resolve().parent


def _register(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    try:
        os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
    except Exception:                                    # noqa: BLE001
        pass
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(folder))
        except OSError:
            pass
    try:                                                 # 仅 Windows
        import ctypes

        ctypes.windll.kernel32.SetDllDirectoryW(str(folder))
    except Exception:                                    # noqa: BLE001
        pass
    return True


_base = _base_dir()
#: 顺序要紧：``panda3d/`` 放最后注册 —— ``SetDllDirectory`` 是"设置"而不是"追加"，
#: 后注册的会顶掉先注册的，而插件都在 ``panda3d/`` 里。
for _folder in (_base, _base / "panda3d"):
    _register(_folder)
