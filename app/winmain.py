"""打包成 exe 之后的入口：无控制台、出错也说得清。

为什么要单独一个入口
------------------------------------------------
``app/main.py`` 是给命令行用的 —— 它靠 ``print`` 报状态、出错就把 traceback 甩到
stderr。这两件事在**双击 exe** 的场景里同时失效：

* 用 ``--windowed`` 打包时没有控制台，``sys.stdout`` 是 ``None``。CPython 的
  ``print`` 遇到 ``None`` 会静默丢掉（不报错），于是"场景名拼错了""存档不存在"
  这类提示一个字都看不到 —— 用户只看到游戏没开起来；
* traceback 同理，写进虚空。结果是唯一可报告的线索也没了。

所以这里补三件事：

1. **把 ``stdout`` / ``stderr`` 接到 exe 旁边的日志文件**（``core.paths.log_path``）。
   ``print`` 与 Panda3D 的告警都会落进去，出问题时有一份可发过来的东西。
2. **兜住所有异常**：写进同一份日志，并弹一个中文 ``MessageBox`` 说明白发生了什么、
   日志在哪。宁可弹窗，也不要"窗口闪一下就没了"。
3. **把工作目录切到游戏目录**：``--open saves\\valley.json`` 这类相对路径与文档里
   写的保持一致（双击时本来就是它，用快捷方式启动时不一定）。

纯文本的那几个参数（``--list-scenes`` / ``--help``）怎么办
------------------------------------------------
窗口版 exe 的"控制台"是假的 —— 输出没有任何地方可去，所以这两个参数要另想办法。
这里**临时开一个属于自己的控制台窗口**（``AllocConsole``），把输出接过去，读完
再"按回车关闭"。为什么不 ``AttachConsole`` 到父进程的控制台：图形子系统的 exe 被
命令行启动时，命令行**不会等它**（提示符已经回来了），此时去借用那个控制台，输出会
和提示符糊在一起，末尾的 ``input()`` 更会把用户的下一条命令吃掉。

开不出控制台时（极少数环境）才退回弹窗。两条路都失败也仍有日志可查。
"""

from __future__ import annotations

import io
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import paths                       # noqa: E402

#: 这些参数的结果是纯文本，无控制台时得另开一个窗口给用户看。
_TEXT_MODES = ("--list-scenes", "--help", "-h")

#: 控制台窗口的标题（``AllocConsole`` 建出来的那个窗口）。
_CONSOLE_TITLE = "模拟铁轨 —— 命令行模式"


class _Tee(io.TextIOBase):
    """把写入同时送到日志文件与原来的流。

    无控制台时"原来的流"是 ``None``，此时它就退化成只写文件。``write`` 必须永远
    不抛异常 —— 它会在解释器关停、Panda3D 写告警、甚至异常处理内部被调用，
    在那里再炸一次只会把真正的错误盖掉。
    """

    def __init__(self, handle, mirror=None):
        self._handle = handle
        self._mirror = mirror

    def write(self, text: str) -> int:          # type: ignore[override]
        if not isinstance(text, str):
            text = str(text)
        for stream in (self._handle, self._mirror):
            if stream is None:
                continue
            try:
                stream.write(text)
            except Exception:                   # noqa: BLE001 - 见 docstring
                pass
        return len(text)

    def flush(self) -> None:
        for stream in (self._handle, self._mirror):
            if stream is None:
                continue
            try:
                stream.flush()
            except Exception:                   # noqa: BLE001
                pass

    def isatty(self) -> bool:
        return False


def _open_log():
    """打开日志文件；打不开就返回 ``None``（宁可没日志也不能因此起不来）。"""
    try:
        return paths.log_path().open("w", encoding="utf-8", buffering=1)
    except OSError:
        return None


def _message_box(title: str, text: str, log=None) -> bool:
    """弹一个中文对话框，返回是否真的弹出来了。

    用 ``MessageBoxW``（宽字符）而不是 ``MessageBoxA``：后者按 ANSI 代码页解码，
    中文在非中文区域会变成问号。**失败时要留下痕迹** —— 一个负责报告错误的函数
    自己静默失败是最坑的那种失败，所以这里把原因写进日志。
    """
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, title, 0x00000040)
        return True
    except Exception:                           # noqa: BLE001 - 弹不出来也不能再抛
        if log is not None:
            try:
                log.write("[exe] 弹窗失败：\n" + traceback.format_exc())
                log.flush()
            except Exception:                   # noqa: BLE001
                pass
        return False


def _attach_console(log=None) -> bool:
    """给这个无控制台的 exe 临时开一个**自己的**控制台窗口，并接上标准流。

    代码页必须一起设成 UTF-8：中文 Windows 的控制台默认是 cp936，而 Python 按
    已设的编码往外写 —— 不改的话中文会变成乱码（这正是"文档里说能看，实际看到
    一堆方块"的常见来源）。
    """
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        if not kernel32.AllocConsole():
            return False
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleTitleW(_CONSOLE_TITLE)
        sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
        sys.stderr = open("CONOUT$", "w", encoding="utf-8", buffering=1)
        sys.stdin = open("CONIN$", "r", encoding="utf-8")
        return True
    except Exception:                           # noqa: BLE001
        if log is not None:
            try:
                log.write("[exe] 开控制台失败：\n" + traceback.format_exc())
                log.flush()
            except Exception:                   # noqa: BLE001
                pass
        return False


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # ---- 切到游戏目录：相对路径与文档里写的一致
    try:
        os.chdir(paths.user_dir())
    except OSError:
        pass

    log = _open_log()
    if log is not None:
        sys.stdout = _Tee(log, sys.__stdout__)
        sys.stderr = _Tee(log, sys.__stderr__)

    text_mode = any(flag in argv for flag in _TEXT_MODES)
    print(f"[exe] 日志文件：{paths.log_path()}")
    print(f"[exe] 游戏目录：{paths.user_dir()}")
    print(f"[exe] 资源目录：{paths.bundle_dir()}")
    print(f"[exe] 参数：{argv or '（无，直接开游戏）'}")

    # ---- 纯文本模式：优先自己开个控制台；开不出来再用弹窗兜住输出
    capture: io.StringIO | None = None
    console = False
    if text_mode:
        console = _attach_console(log)
        if console:
            # 控制台和日志两边都写：日志是"事后能发过来"的那份
            sys.stdout = _Tee(sys.stdout, log)
            sys.stderr = _Tee(sys.stderr, log)
        else:
            capture = io.StringIO()
            sys.stdout = _Tee(capture, log)
        print(f"[exe] 纯文本模式：控制台 = {console}"
              f"（{'另开了一个窗口' if console else '退回弹窗'}）")

    try:
        from app.main import main as game_main

        code = game_main(argv)
    except SystemExit as exc:                   # argparse 的 --help / 参数写错
        code = int(exc.code or 0)
    except BaseException:                       # noqa: BLE001 - 兜住一切并解释给人听
        detail = traceback.format_exc()
        print(detail)
        _message_box(
            "模拟铁轨 —— 启动失败",
            "游戏没能启动。\n\n"
            f"{detail.strip().splitlines()[-1]}\n\n"
            f"完整日志：\n{paths.log_path()}",
            log,
        )
        return 1

    if text_mode:
        if capture is not None and capture.getvalue():
            # 顺序是 (标题, 正文)：正文才是场景清单，别把一长串清单塞进标题栏
            _message_box("模拟铁轨 —— 场景列表", capture.getvalue(), log)
        elif console:
            try:
                input("\n按回车键关闭这个窗口…")
            except (EOFError, OSError, KeyboardInterrupt):
                pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
