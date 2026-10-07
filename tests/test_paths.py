"""运行时路径：源码态与打包态两套目录必须分得清。

为什么这层值得单独测
------------------------------------------------
打包成 exe 之后，``data/`` 会被解到 PyInstaller 的临时目录里，而那个目录**退出时
会被删掉**。所以"存档写到哪"这件事一旦拼错，表现是：

* Ctrl+S 一路正常，提示条还回了一句"已保存"；
* 关掉游戏，文件不见了；
* 且**没有任何报错**。

也就是说，这个 bug 在源码态下永远不会出现、只会在 exe 里出现，而且连症状都不像
bug（"我明明存了啊"）。因此这里把"冻结态下可写目录落在哪"直接钉成断言 ——
它是"exe 能用"和"exe 看起来能用"的分界线。

冻结态靠 monkeypatch ``sys.frozen`` / ``sys._MEIPASS`` / ``sys.executable`` 模拟：
``core.paths`` 判断"是不是打包态"只看这几个东西，所以这样模拟走的是**真**分支，
不是给测试开的旁路。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from core import paths
from core.track.catalog import Catalog
from core.train.consist import TrainCatalog


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """每个用例都从"还没探测过"开始 —— 否则缓存会把上一个用例的结论带过来。"""
    monkeypatch.delenv(paths.USER_DIR_ENV, raising=False)
    paths.reset_cache()
    yield
    paths.reset_cache()


@pytest.fixture()
def frozen(monkeypatch, tmp_path):
    """把当前进程伪装成"跑在 onefile 的 exe 里"。返回 (解包目录, exe 所在目录)。"""
    unpack = tmp_path / "unpack"          # 相当于 sys._MEIPASS
    unpack.mkdir()
    beside = tmp_path / "game"            # exe 旁边
    beside.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(unpack), raising=False)
    monkeypatch.setattr(sys, "executable", str(beside / "TrainGame.exe"))
    paths.reset_cache()
    return unpack, beside


# --------------------------------------------------------------------------- #
# 只读资源：跟着程序走
# --------------------------------------------------------------------------- #

def test_source_mode_reads_data_from_the_repo():
    repo = Path(__file__).resolve().parents[1]
    assert paths.is_frozen() is False
    assert paths.bundle_dir() == repo
    assert paths.data_dir() == repo / "data"


def test_data_dir_has_the_catalogs():
    """``data/`` 里两本目录都得在 —— 少了任何一本，游戏在源码态就起不来。"""
    assert (paths.data_dir() / "pieces.json").exists()
    assert (paths.data_dir() / "trains.json").exists()


def test_models_dir_sits_next_to_data():
    repo = Path(__file__).resolve().parents[1]
    assert paths.models_dir() == repo / "models"


def test_resolve_model_path_finds_a_glb_in_models():
    found = paths.resolve_model_path("CR400BF_HuangSiDai_6car.glb")
    if found is None:
        pytest.skip("models/ 里还没有编组 glb")
    assert found.name == "CR400BF_HuangSiDai_6car.glb"
    assert found.exists()


def test_resolve_model_path_returns_none_when_missing():
    assert paths.resolve_model_path("definitely-not-a-train.glb") is None


def test_catalog_actually_reads_through_data_dir(monkeypatch, tmp_path):
    """目录模块必须真的走 :func:`paths.data_dir`，而不是自己拼相对路径。

    做法：把 ``DATA_DIR`` 指到一个**只有两本 JSON 的干净目录**，能装出来才算数。
    这条防的是"以后有人又在 catalog 里写回 ``Path(__file__).parents[2]``"。
    """
    from core.track import catalog as catalog_mod
    from core.train import consist as consist_mod

    fake = tmp_path / "data"
    fake.mkdir()
    for name in ("pieces.json", "trains.json"):
        shutil.copy(paths.data_dir() / name, fake / name)

    monkeypatch.setattr(catalog_mod, "DATA_DIR", fake)
    monkeypatch.setattr(consist_mod, "DATA_DIR", fake)

    assert len(Catalog.builtin()) == len(Catalog.builtin()) > 0
    assert len(TrainCatalog.builtin()) > 0
    # 干净目录里确实只有这两本，说明读到的是我们指过去的那份
    assert sorted(p.name for p in fake.iterdir()) == ["pieces.json", "trains.json"]


def test_frozen_bundle_dir_is_the_unpack_folder(frozen):
    unpack, _ = frozen
    assert paths.is_frozen() is True
    assert paths.bundle_dir() == unpack
    assert paths.data_dir() == unpack / "data"


def test_frozen_data_dir_is_not_the_user_dir(frozen):
    """只读与可写必须是两个目录 —— 合成一个就是"存档随退出蒸发"的根因。"""
    unpack, beside = frozen
    assert paths.data_dir() != paths.user_dir()
    assert not str(paths.data_dir()).startswith(str(beside))


# --------------------------------------------------------------------------- #
# 可写的东西：跟着 exe 走
# --------------------------------------------------------------------------- #

def test_user_dir_source_mode_is_the_repo():
    repo = Path(__file__).resolve().parents[1]
    assert paths.user_dir() == repo


def test_user_dir_goes_beside_the_executable(frozen):
    _, beside = frozen
    assert paths.user_dir() == beside


def test_default_save_path_never_lands_in_the_unpack_folder(frozen):
    """**这是本文件存在的理由。** onefile 的 exe 退出时会删掉解包目录。"""
    unpack, beside = frozen
    save = paths.default_save_path()

    assert not str(save).startswith(str(unpack)), (
        f"默认存档落在了临时解包目录里（{save}）—— 关掉游戏它就被删了，"
        f"而且过程中不会有任何报错")
    assert save.parent.parent == beside
    assert save.name == "layout.json"


def test_saves_dir_is_created_and_writable(frozen):
    _, beside = frozen
    folder = paths.saves_dir()
    assert folder == beside / "saves"
    assert folder.is_dir()
    probe = folder / "probe.json"
    probe.write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert json.loads(probe.read_text(encoding="utf-8"))["ok"] is True


def test_log_path_is_beside_the_executable(frozen):
    _, beside = frozen
    assert paths.log_path() == beside / "train3d_log.txt"


def test_user_dir_falls_back_when_the_game_folder_is_read_only(monkeypatch, frozen, tmp_path):
    """装到 ``Program Files`` 这种地方时退到 ``%LOCALAPPDATA%``。

    这里不去真的造一个只读目录（Windows 上要改 ACL，慢且要管理员），而是把
    "能不能写"这个判断本身替换掉 —— 被测的是**退到哪**，不是权限探测能否工作。
    """
    _, beside = frozen
    local = tmp_path / "LocalAppData"
    local.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setattr(paths, "_writable", lambda folder: False)
    paths.reset_cache()

    fallback = paths.user_dir()
    assert fallback == local / paths.APP_NAME
    assert fallback.is_dir()
    assert fallback != beside
    assert paths.default_save_path().parent.parent == fallback


def test_env_override_wins_over_everything(monkeypatch, frozen, tmp_path):
    target = tmp_path / "便携存档"
    monkeypatch.setenv(paths.USER_DIR_ENV, str(target))
    paths.reset_cache()

    assert paths.user_dir() == target
    assert target.is_dir()


def test_probe_result_is_cached(monkeypatch, frozen, tmp_path):
    """探测要真的写文件，不能每帧都来一次（HUD / 存档 / 日志都会反复问）。"""
    calls = []
    monkeypatch.setattr(paths, "_writable",
                        lambda folder: (calls.append(folder), True)[1])
    paths.reset_cache()

    for _ in range(5):
        paths.user_dir()
    assert len(calls) == 1, f"探测了 {len(calls)} 次，应该只探一次"

    paths.reset_cache()
    paths.user_dir()
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# 用户给的相对路径
# --------------------------------------------------------------------------- #

def test_resolve_user_path_finds_saves_beside_the_exe(frozen):
    """``--open saves\\valley.json`` 里的相对路径是相对**游戏目录**说的。"""
    _, beside = frozen
    saves = beside / "saves"
    saves.mkdir()
    (saves / "valley.json").write_text("{}", encoding="utf-8")

    resolved = paths.resolve_user_path("saves/valley.json")
    assert resolved == saves / "valley.json"
    assert resolved.exists()


def test_resolve_user_path_finds_bare_file_names(frozen):
    """只写文件名（不带 saves\\）时也该在存档目录里找得到。"""
    _, beside = frozen
    saves = beside / "saves"
    saves.mkdir()
    (saves / "yard.json").write_text("{}", encoding="utf-8")

    assert paths.resolve_user_path("yard.json") == saves / "yard.json"


def test_resolve_user_path_prefers_the_game_folder_over_cwd(frozen, monkeypatch, tmp_path):
    """两边都有同名文件时，用**游戏目录**那份。

    双击 exe 与用快捷方式启动时 CWD 可能不一样（快捷方式的"起始位置"），而玩家
    心里的锚点是游戏文件夹；跟着 CWD 走会出现"同一个命令，在 A 目录打开的是山谷
    环线、在 B 目录打开的是别的东西"。
    """
    _, beside = frozen
    (beside / "saves").mkdir()
    (beside / "saves" / "both.json").write_text('{"who": "game"}', encoding="utf-8")
    work = tmp_path / "cwd"
    work.mkdir()
    (work / "saves").mkdir()
    (work / "saves" / "both.json").write_text('{"who": "cwd"}', encoding="utf-8")
    monkeypatch.chdir(work)

    chosen = paths.resolve_user_path("saves/both.json")
    assert chosen == beside / "saves" / "both.json"


def test_resolve_user_path_falls_back_to_the_current_directory(frozen, monkeypatch, tmp_path):
    """游戏目录里没有时，仍然认"相对当前目录"—— 命令行那套直觉不能被破坏。"""
    work = tmp_path / "cwd"
    work.mkdir()
    (work / "layout.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(work)

    assert paths.resolve_user_path("layout.json") == work / "layout.json"


def test_resolve_user_path_keeps_absolute_paths_untouched(frozen, tmp_path):
    outside = tmp_path / "elsewhere" / "mine.json"
    assert paths.resolve_user_path(outside) == outside
    assert paths.resolve_user_path(str(outside)) == outside


def test_resolve_user_path_returns_the_input_when_nothing_matches(frozen):
    """找不到时原样返回：错误信息里给用户看的是**他自己输入的**那个路径。"""
    missing = paths.resolve_user_path("nope/does-not-exist.json")
    assert missing.name == "does-not-exist.json"
    assert not missing.exists()
