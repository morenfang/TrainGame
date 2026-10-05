"""无控制台入口（``app/winmain.py``）的调度逻辑。

为什么这些值得测
------------------------------------------------
这个文件里真正会出错的不是"能不能弹窗"，而是**分流**：

* 纯文本参数（``--list-scenes`` / ``--help``）在窗口版 exe 里结果没有任何地方可去，
  必须走"开控制台 / 退弹窗"这条路；
* 开游戏时绝不能附带开一个控制台窗口；
* 启动炸了必须**有东西弹出来**并返回非零，而不是静默退出。

这些分支在源码态跑的时候全都有终端兜着，所以不测就等于没验 —— 出问题只会在
用户双击 exe 的时候暴露。``app.main.main`` 在这里被替换掉，因为本文件测的是
"怎么把结果送出去"，不是游戏本身。
"""

from __future__ import annotations

import builtins
import io
import sys

import pytest

from app import winmain
from core import paths


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """把可写目录指到临时目录：这样 ``chdir`` 与日志都落在沙盒里。"""
    monkeypatch.setenv(paths.USER_DIR_ENV, str(tmp_path))
    paths.reset_cache()
    yield
    paths.reset_cache()


@pytest.fixture()
def fake_game(monkeypatch):
    """替换掉真正的游戏入口，返回一个"记录被怎么调用"的替身。"""
    calls = []

    def fake_main(argv):
        calls.append(list(argv))
        print("现成场景：valley 山谷环线")
        return 0

    import app.main as real_main

    monkeypatch.setattr(real_main, "main", fake_main, raising=False)
    return calls


# --------------------------------------------------------------------------- #
# _Tee
# --------------------------------------------------------------------------- #

def test_tee_writes_to_every_target():
    first, second = io.StringIO(), io.StringIO()
    tee = winmain._Tee(first, second)
    tee.write("轨道")
    tee.write("闭环")
    assert first.getvalue() == "轨道闭环"
    assert second.getvalue() == "轨道闭环"


def test_tee_survives_a_broken_mirror():
    """无控制台时 mirror 就是 ``None``；流被关掉时也不能抛。"""

    class Broken:
        def write(self, _text):
            raise ValueError("这个流已经关了")

        def flush(self):
            raise ValueError("这个流已经关了")

    good = io.StringIO()
    tee = winmain._Tee(good, Broken())
    tee.write("还能写")          # 不许抛
    tee.flush()
    assert good.getvalue() == "还能写"

    assert winmain._Tee(good, None).write("没有镜像也行") == len("没有镜像也行")


def test_tee_converts_non_string():
    out = io.StringIO()
    winmain._Tee(out).write(42)                 # type: ignore[arg-type]
    assert out.getvalue() == "42"


# --------------------------------------------------------------------------- #
# 分流：开游戏 vs 纯文本
# --------------------------------------------------------------------------- #

def test_game_mode_never_opens_a_console(monkeypatch, fake_game):
    monkeypatch.setattr(winmain, "_attach_console",
                        lambda log=None: pytest.fail("开游戏时不该开控制台"))
    monkeypatch.setattr(winmain, "_message_box",
                        lambda *a, **k: pytest.fail("开游戏时不该弹窗"))

    assert winmain.main([]) == 0
    assert fake_game == [[]]


def test_text_mode_uses_the_console_when_it_can(monkeypatch, fake_game, tmp_path):
    """开出控制台时：不弹窗，等用户按回车。"""
    opened = []
    monkeypatch.setattr(winmain, "_attach_console",
                        lambda log=None: (opened.append(True), True)[1])
    monkeypatch.setattr(winmain, "_message_box",
                        lambda *a, **k: pytest.fail("有控制台就不该弹窗"))
    waited = []
    monkeypatch.setattr(builtins, "input", lambda prompt="": waited.append(prompt) or "")

    assert winmain.main(["--list-scenes"]) == 0
    assert opened == [True]
    assert len(waited) == 1, "应该停一下让用户看得到输出"
    assert fake_game == [["--list-scenes"]]


def test_text_mode_falls_back_to_a_dialog(monkeypatch, fake_game):
    """开不出控制台时，输出必须装进弹窗 —— 否则纯文本模式等于什么都没发生。"""
    monkeypatch.setattr(winmain, "_attach_console", lambda log=None: False)
    boxes = []
    monkeypatch.setattr(winmain, "_message_box",
                        lambda title, text, log=None: boxes.append((title, text)) or True)
    monkeypatch.setattr(builtins, "input",
                        lambda prompt="": pytest.fail("没有控制台，不该等回车"))

    assert winmain.main(["--list-scenes"]) == 0
    assert len(boxes) == 1
    _title, text = boxes[0]
    assert "valley" in text, f"弹窗里得有场景清单，实际是 {text!r}"


def test_help_is_treated_as_text_mode(monkeypatch):
    """``--help`` 也是纯文本 —— argparse 打完就 SystemExit(0)。"""
    import app.main as real_main

    monkeypatch.setattr(winmain, "_attach_console", lambda log=None: False)
    boxes = []
    monkeypatch.setattr(winmain, "_message_box",
                        lambda title, text, log=None: boxes.append(text) or True)

    def fake_help(argv):
        print("usage: TrainGame [-h] [--scene SCENE]")
        raise SystemExit(0)

    monkeypatch.setattr(real_main, "main", fake_help, raising=False)
    assert winmain.main(["--help"]) == 0
    assert boxes and "usage" in boxes[0]


# --------------------------------------------------------------------------- #
# 出错
# --------------------------------------------------------------------------- #

def test_crash_reports_and_returns_nonzero(monkeypatch):
    """启动炸了：弹窗要出来、返回非零、日志里留下 traceback。"""
    import app.main as real_main

    monkeypatch.setattr(winmain, "_attach_console", lambda log=None: False)
    boxes = []
    monkeypatch.setattr(winmain, "_message_box",
                        lambda title, text, log=None: boxes.append((title, text)) or True)

    def boom(argv):
        raise RuntimeError("显卡驱动炸了")

    monkeypatch.setattr(real_main, "main", boom, raising=False)

    code = winmain.main([])
    assert code == 1
    assert boxes, "崩了必须让人看见"
    title, text = boxes[0]
    assert "启动失败" in title
    assert "显卡驱动炸了" in text
    assert str(paths.log_path()) in text, "得告诉用户日志在哪"

    logged = paths.log_path().read_text(encoding="utf-8")
    assert "显卡驱动炸了" in logged
    assert "Traceback" in logged


def test_bad_arguments_exit_cleanly(monkeypatch):
    """参数写错时 argparse 会 SystemExit(2)；不该变成"启动失败"弹窗。"""
    import app.main as real_main

    monkeypatch.setattr(winmain, "_attach_console", lambda log=None: False)
    monkeypatch.setattr(winmain, "_message_box",
                        lambda *a, **k: pytest.fail("参数写错不算崩溃"))

    def bad(argv):
        raise SystemExit(2)

    monkeypatch.setattr(real_main, "main", bad, raising=False)
    assert winmain.main(["--nonsense"]) == 2


def test_log_records_the_three_directories(monkeypatch, fake_game):
    """日志开头必须有三条路径 —— 出问题时第一句要问的就是"它在哪跑的"。"""
    assert winmain.main([]) == 0
    logged = paths.log_path().read_text(encoding="utf-8")
    for label in ("日志文件：", "游戏目录：", "资源目录：", "参数："):
        assert label in logged, f"日志里缺 {label!r}"
    assert str(paths.bundle_dir()) in logged


def test_module_is_importable_without_panda3d():
    """``app.main`` 是**延迟**导入的：这个文件要能在没装 panda3d 的机器上被读。"""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(winmain.__file__).read_text(encoding="utf-8"))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level.add(node.module or "")
    assert not any(name.startswith("panda3d") for name in top_level), (
        f"顶层导入了 panda3d，winmain 就不再是无依赖入口了：{sorted(top_level)}")
    assert "sys" in sys.modules
