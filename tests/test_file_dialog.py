"""原生文件对话框：可注入路径，不弹真窗。"""

from __future__ import annotations

from pathlib import Path

from app.editor import TrackEditor
from core.track.catalog import Catalog
from tests.test_app_editor import FakeHud


def test_dialog_save_writes_picked_path(app, tmp_path):
    ed = TrackEditor(app, Catalog.builtin(), hud=FakeHud(),
                     save_path=tmp_path / "layout.json")
    ed.use_file_dialog = True
    target = tmp_path / "from_dialog.json"
    ed.file_dialog = lambda **kw: target
    piece = next(p for p in ed.catalog if p.category == "straight")
    ed.select_piece_id(piece.id)
    ed._dialog_save()
    assert target.exists()
    assert ed.save_path == target


def test_dialog_load_reads_picked_path(app, tmp_path):
    ed = TrackEditor(app, Catalog.builtin(), hud=FakeHud(),
                     save_path=tmp_path / "layout.json")
    ed.use_file_dialog = True
    piece = next(p for p in ed.catalog if p.category == "straight")
    ed.layout.add_root(piece.id)
    ed.view.sync()
    asset = tmp_path / "route.json"
    ed.save_path = asset
    assert ed.save()
    ed.layout.clear()
    ed.view.sync()
    assert len(ed.layout) == 0

    ed.file_dialog = lambda **kw: asset
    ed._dialog_load()
    assert len(ed.layout) > 0
    assert ed.save_path == asset


def test_dialog_cancel_notifies(app, tmp_path):
    ed = TrackEditor(app, Catalog.builtin(), hud=FakeHud(),
                     save_path=tmp_path / "layout.json")
    ed.use_file_dialog = True
    ed.file_dialog = lambda **kw: None
    ed._dialog_save()
    assert "取消" in ed._toast


def test_ask_save_appends_json_suffix(monkeypatch, tmp_path):
    from app import file_dialog as fd

    monkeypatch.setattr(
        fd, "_win32_file_dialog",
        lambda **kw: str(tmp_path / "no_ext"),
    )
    monkeypatch.setattr(fd.sys, "platform", "win32")
    path = fd.ask_save_json(initial_dir=tmp_path)
    assert path == tmp_path / "no_ext.json"
