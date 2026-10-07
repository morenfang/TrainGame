"""运行时路径：只读资源在哪、可写的东西放哪。

为什么需要这一层
------------------------------------------------
源码态下"仓库根目录"这一个概念就够了：``data/`` 在它下面、``saves/`` 也在它下面，
所以代码里到处是 ``Path(__file__).resolve().parents[N]``。打包成 exe 之后这两个
概念**必须拆开**，否则会踩到两个都很隐蔽的坑：

* **只读资源**（``data/pieces.json``、``data/trains.json``）会被塞进一个临时解包
  目录（PyInstaller 的 ``sys._MEIPASS``）。那个目录每次启动都不一样，用
  ``parents[N]`` 拼出来的路径在 exe 里指的就是它 —— 能读到（我们把 data 一起打
  进去了），但路径本身不可预测、也不该被写。
* **可写的东西**（存档、日志）**绝不能**落在那个临时目录里：onefile 的 exe 退出时
  会把整个目录删掉。用户存了一晚上的线路，关掉游戏就没了，而且**没有任何报错**：
  Ctrl+S 一切正常、提示条还回了一句"已保存"，只是文件随进程一起蒸发。这是本项目
  对 exe 做的最重要的一处适配。

所以这里只暴露两个目录，全项目其余地方一律向它们要路径：

====================  ==========================  ============================
函数                   源码态                       打包后
====================  ==========================  ============================
:func:`bundle_dir`     仓库根目录                    临时解包目录（``_MEIPASS``）
:func:`user_dir`       仓库根目录                    exe 旁边（不可写则退到用户目录）
====================  ==========================  ============================

``data/`` 走 :func:`data_dir`，存档走 :func:`saves_dir`，日志走 :func:`log_path`。

绿色版还是安装版
------------------------------------------------
:func:`user_dir` 优先用 **exe 旁边**那个目录 —— 于是拷到 U 盘就能玩、存档也跟着
走，这是沙盘游戏最自然的用法。如果 exe 被装到 ``C:\\Program Files`` 这种地方
（普通用户没有写权限），才退回 ``%LOCALAPPDATA%\\TrainGame``。判断方式是**真的去写
一个探针文件**，而不是看路径字符串像不像系统目录 —— 权限这东西猜不准。

想强制指定时用环境变量 ``TRAIN3D_USER_DIR``（测试与打包脚本都靠它隔离）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: 退到用户目录时用的文件夹名。
APP_NAME = "TrainGame"

#: 指定可写目录的环境变量名（测试 / 便携启动器用）。
USER_DIR_ENV = "TRAIN3D_USER_DIR"

#: :func:`user_dir` 的探测结果缓存。每个进程只探一次 —— 它每次都要真的写一个
#: 探针文件，而 HUD、存档、日志都会反复问。
_user_dir_cache: Path | None = None


def is_frozen() -> bool:
    """当前是不是跑在 PyInstaller 打出来的 exe 里。"""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """只读资源所在目录（``data/`` 就在它下面）。

    onefile 的 exe 解包到 ``sys._MEIPASS``；onedir 的 exe 没有 ``_MEIPASS``，
    资源就在 exe 旁边；源码态就是仓库根目录。
    """
    if is_frozen():
        base = getattr(sys, "_MEIPASS", None)
        if base:
            return Path(base)
        return _executable_dir()
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """``data/`` 目录（轨道件与列车目录 JSON）。"""
    return bundle_dir() / "data"


def models_dir() -> Path:
    """外部 glTF 模型目录。

    优先 ``models/``（你放下的编组 glb），没有再退到 ``assets/models/``。
    打包后跟着 ``bundle_dir()`` 走。
    """
    primary = bundle_dir() / "models"
    if primary.exists():
        return primary
    return bundle_dir() / "assets" / "models"


def resolve_model_path(name: str | Path) -> Path | None:
    """按文件名找一个 ``.glb`` / ``.gltf``。没有就返回 ``None``，让调用方退回程序化网格。

    查找顺序：``models/`` → ``assets/models/`` → 仓库（或解包）根目录。
    """
    raw = Path(name)
    if raw.is_absolute():
        return raw if raw.exists() else None
    for base in (bundle_dir() / "models", bundle_dir() / "assets" / "models",
                 bundle_dir()):
        probe = base / raw
        if probe.exists():
            return probe
    return None


def _executable_dir() -> Path:
    return Path(sys.executable).resolve().parent


def _writable(folder: Path) -> bool:
    """真的写一个探针文件来判断能不能写。

    只看路径像不像 ``Program Files`` 是不准的：用户可能恰好有权限、也可能反过来
    因为组策略对某个看起来无害的目录没权限。能写就是能写。
    """
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".train3d_write_probe"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def user_dir() -> Path:
    """可写目录：exe 旁边优先，不可写就退到用户目录。结果会被记住。"""
    global _user_dir_cache
    if _user_dir_cache is not None:
        return _user_dir_cache

    override = os.environ.get(USER_DIR_ENV)
    if override:
        folder = Path(override).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        _user_dir_cache = folder
        return folder

    beside = _executable_dir() if is_frozen() else Path(__file__).resolve().parents[1]
    if _writable(beside):
        _user_dir_cache = beside
        return beside

    local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    fallback = base / APP_NAME
    fallback.mkdir(parents=True, exist_ok=True)
    _user_dir_cache = fallback
    return fallback


def saves_dir() -> Path:
    """存档目录（不存在就建出来 —— 读档框要列它的内容）。"""
    folder = user_dir() / "saves"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def default_save_path() -> Path:
    """没指定 ``--save`` 时，Ctrl+S 默认存到这里。"""
    return saves_dir() / "layout.json"


def log_path() -> Path:
    """打包后的运行日志（无控制台时，``print`` 与报错都落在这里）。"""
    return user_dir() / "train3d_log.txt"


def resolve_user_path(path: str | Path) -> Path:
    """把用户给的相对路径解释成一个**实际存在**的文件路径。

    ``--open saves\\valley.json`` 这种相对路径，源码态下相对仓库根目录、打包后相对
    "exe 旁边"，两者恰好都等于 :func:`user_dir`。所以查找顺序是
    **游戏目录 → 存档目录 → 当前工作目录**：

    * 游戏目录放第一：双击 exe 和用快捷方式启动时 CWD 可能不一样，而玩家心里的
      锚点是"游戏文件夹"，``saves\\valley.json`` 就该在游戏文件夹里找；
    * 当前工作目录放最后是兜底：命令行那套"相对当前目录"的直觉仍然成立，
      只要那一边确实有文件（用绝对路径当然更没问题）。

    都不存在时原样返回，让调用方去报"文件不存在" —— 错误信息里给用户看的还是他
    自己输入的那个路径，好对照。
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    for base in (user_dir(), saves_dir(), Path.cwd()):
        probe = base / candidate
        if probe.exists():
            return probe
    return candidate


def reset_cache() -> None:
    """清掉 :func:`user_dir` 的缓存。**只给测试用** —— 它要模拟不同的探测结果。"""
    global _user_dir_cache
    _user_dir_cache = None
