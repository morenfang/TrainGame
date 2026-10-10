"""v2 HUD（Proposal）：五模式轨 + 可固定分类列 + Tomix 底栏。"""

from __future__ import annotations

from app.hud import (
    HELP_OVERLAY,
    HUD_MODES,
    Hud,
    HudState,
    NOTCH_LABELS,
    TRACK_CATEGORY_ORDER,
    _make_speed_scale,
    _speed_angle,
    build_default_hud,
    handle_to_notch,
    notch_to_handle,
    speed_heat,
)


def test_build_default_hud_has_rail_and_console(app):
    hud = build_default_hud(app)
    assert not hud.root.isHidden()
    assert hud._rail.getNumChildren() > 0
    assert hud._console.getNumChildren() > 0


def test_set_visible_toggles_the_root(app):
    hud = Hud(app)
    hud.set_visible(False)
    assert hud.root.isHidden() and hud.visible is False
    hud.set_visible(True)
    assert not hud.root.isHidden() and hud.visible is True


def test_five_mode_rail_hits(app):
    hud = Hud(app)
    hud.apply_state(HudState(hud_mode="track", catalog_items=(("a", "直"),)))
    modes = {h.arg for h in hud._hits if h.action == "hud_mode"}
    assert modes == {m for m, *_ in HUD_MODES}
    assert set(hud._mode_icons) == modes
    for ico in hud._mode_icons.values():
        assert not ico.isEmpty()


def test_catalog_open_only_for_build_modes_unless_pinned(app):
    hud = Hud(app)
    hud.apply_state(HudState(hud_mode="view"))
    assert hud.catalog_open() is False
    assert not any(h.action == "category" for h in hud._hits)

    hud.toggle_pin()
    hud.apply_state(HudState(
        hud_mode="view",
        category="straight",
        catalog_items=(("s10", "直轨 10m"),),
        track_categories=("straight",),
    ))
    assert hud.catalog_open() is True
    assert any(h.action == "category" for h in hud._hits)
    assert any(h.action == "select_item" for h in hud._hits)


def test_track_catalog_cells(app):
    hud = Hud(app)
    items = tuple((f"id{i}", f"件{i}") for i in range(4))
    hud.apply_state(HudState(
        hud_mode="track",
        catalog_items=items,
        piece_index=2,
        track_categories=("straight", "curve"),
    ))
    select_hits = [h for h in hud._hits if h.action == "select_item"]
    assert len(select_hits) == 4
    cat_hits = [h for h in hud._hits if h.action == "category"]
    assert {h.arg for h in cat_hits} == {"straight", "curve"}


def test_scenery_catalog(app):
    hud = Hud(app)
    hud.apply_state(HudState(
        hud_mode="scenery",
        scenery=True,
        scenery_category="tree",
        scenery_rail=(("building", "建筑"), ("tree", "树")),
        scenery_items=(("pine", "松树"), ("oak", "橡树")),
        scenery_index=1,
    ))
    assert {h.arg for h in hud._hits if h.action == "scenery_category"} == {
        "building", "tree",
    }
    assert not any(h.action == "category" for h in hud._hits)


def test_train_catalog_and_spawn(app):
    hud = Hud(app)
    hud.apply_state(HudState(
        hud_mode="train",
        train_list=(("a", "黄丝带"), ("b", "绿皮")),
        train_index=0,
    ))
    assert any(h.action == "select_train" for h in hud._hits)
    assert any(h.action == "spawn_train" for h in hud._hits)


def test_file_catalog(app):
    hud = Hud(app)
    hud.apply_state(HudState(hud_mode="file"))
    actions = {h.action for h in hud._hits}
    assert "file_save" in actions and "file_open" in actions


def test_offline_console_is_dimmed_without_notch_actions(app):
    hud = Hud(app)
    hud.apply_state(HudState(
        hud_mode="track",
        train_online=False,
        train_name="黄丝带",
    ))
    assert not any(h.action == "handle_notch" for h in hud._hits)
    assert not any(h.action == "emergency" for h in hud._hits)
    assert hud._labels["dial_num"].node().getText() == "—"


def test_online_console_exposes_notches_and_actions(app):
    hud = Hud(app)
    hud.apply_state(HudState(
        hud_mode="train",
        train_online=True,
        train_name="黄丝带",
        speed_kmh=186.0,
        handle=0.75,
        handle_label="牵引 75%",
        direction="正向",
        distance_m=12400.0,
    ))
    assert hud._labels["dial_num"].node().getText() == "186"
    notches = [h for h in hud._hits if h.action == "handle_notch"]
    assert len(notches) >= len(NOTCH_LABELS)
    assert any(h.action == "emergency" for h in hud._hits)
    assert any(h.action == "reverse" for h in hud._hits)
    assert any(h.action == "horn" for h in hud._hits)


def test_notch_mapping_roundtrip():
    assert notch_to_handle(4) == 0.0
    assert notch_to_handle(8) == 1.0
    assert notch_to_handle(0) == -1.0
    assert handle_to_notch(0.75) == 7
    assert handle_to_notch(0.0) == 4


def test_speed_heat_goes_green_yellow_red():
    g = speed_heat(0.0)
    y = speed_heat(0.5)
    r = speed_heat(1.0)
    assert g[1] > g[0] and g[1] > g[2]   # 绿通道最高
    assert y[0] > 0.7 and y[1] > 0.5      # 偏黄
    assert r[0] > r[1] and r[0] > r[2]   # 红通道最高


def test_speed_scale_is_circular_geometry():
    """刻度落在等半径圆上；扫角随 frac 线性。"""
    assert abs(_speed_angle(0.0) - _speed_angle(1.0)) > 1.0  # ~240°
    mid = 0.5 * (_speed_angle(0.0) + _speed_angle(1.0))
    assert abs(_speed_angle(0.5) - mid) < 1e-9
    scale = _make_speed_scale()
    assert not scale.isEmpty()
    assert scale.getNumChildren() >= 2


def test_speed_dial_keeps_neutral_needle(app):
    """热度上刻度，不上指针/数字。"""
    hud = Hud(app)
    hud.apply_state(HudState(train_online=True, speed_kmh=280.0))
    assert hud._labels["dial_num"].node().getText() == "280"
    assert not hud._speed_arc_root.isEmpty()
    assert not hud._needle_root.isEmpty()
    # 数字为中性前景灰，非热度红
    fg = hud._labels["dial_num"].node().getTextColor()
    assert fg[0] == fg[1] == fg[2]


def test_help_toggle_shows_overlay(app):
    hud = Hud(app)
    assert hud._help.isHidden()
    hud.toggle_expanded()
    hud.apply_state(HudState(help_visible=True))
    assert not hud._help.isHidden()
    assert HELP_OVERLAY.splitlines()[0][:3] == "V 放"
    hud.collapse()
    hud.apply_state(HudState(help_visible=False))
    assert hud._help.isHidden()


def test_pin_toggle_highlights(app):
    hud = Hud(app)
    assert hud.pinned is False
    hud.toggle_pin()
    assert hud.pinned is True
    hud.apply_state(HudState(hud_mode="view"))
    pin_hits = [h for h in hud._hits if h.action == "pin"]
    assert pin_hits


def test_night_toggle_callback(app):
    hud = Hud(app)
    seen = []
    hud.on_toggle_street_lights = seen.append
    assert hud.toggle_street_lights() is False
    assert seen == [False]


def test_handle_click_dispatches_hud_mode(app):
    hud = Hud(app)
    calls = []
    hud.set_actions({"hud_mode": lambda m: calls.append(m)})
    hud.apply_state(HudState(hud_mode="track"))
    hit = next(h for h in hud._hits if h.action == "hud_mode" and h.arg == "scenery")
    x0, x1, y0, y1 = hit.box
    mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    class _Watcher:
        def hasMouse(self):
            return True

        def getMouseX(self):
            return mx / hud.aspect

        def getMouseY(self):
            return my

    hud.base.mouseWatcherNode = _Watcher()
    assert hud.handle_click() is True
    assert calls == ["scenery"]


def test_progress_bar_shows_and_clears(app):
    hud = Hud(app)
    hud.set_progress("加载中", 0.4)
    assert not hud._progress_root.isHidden()
    hud.clear_progress()
    assert hud._progress_root.isHidden()


def test_track_category_order_covers_common_labels():
    labels = {label for _, label in TRACK_CATEGORY_ORDER}
    assert {"直轨", "曲线", "道岔", "终端"} <= labels


def test_compat_set_text_toast(app):
    hud = Hud(app)
    hud.set_text("toast", "提示一下")
    assert hud.state.toast == "提示一下"
    assert hud.texts["toast"] == "提示一下"


def test_apply_state_fills_compat_texts(app):
    hud = Hud(app)
    hud.apply_state(HudState(
        hud_mode="track",
        category_label="曲线",
        piece_name="左转 R40",
        loop_line="闭环 8 段",
        train_name="黄丝带",
        train_online=True,
        speed_kmh=80.0,
        direction="正向",
        handle_label="牵引 50%",
        toast="已保存",
    ))
    assert "曲线" in hud.texts["status"]
    assert "闭环" in hud.texts["loop"]
    assert "运行信息" in hud.texts["train"]
    assert hud.texts["toast"] == "已保存"
