"""环绕相机：拖拽方向必须"抓住画面拖"。

为什么方向要有专门的测试
------------------------------------------------
拖拽方向是**纯体感**的东西 —— 代码里两个正负号看着都成立，只有真的拖一下才知道
反没反。早期版本把环绕的两个轴都写成了"相机跟着鼠标走"，于是往右拖画面往左跑，
用户一眼就看出不对。这类错一旦回归，测试之外没有任何东西能发现它。

判据用"相机自己往哪动了"，而不是"某个点投影到哪了"：
相机朝屏幕左边走，画面就相对往右走。前者是刚体平移，方向和距离都干净；后者会被
透视和"绕目标旋转"的视差搅混（同一个拖拽，近处和远处的点可能往相反方向跑）。
"""

from __future__ import annotations

import pytest
from panda3d.core import Vec3

from render.camera import OrbitCamera

#: 一次拖拽的像素位移（和 ``Editor._apply_drag`` 传给相机的量纲一致）。
_DRAG = 40.0


def _screen_axes(base):
    """相机自身的"屏幕右"与"屏幕上"两个世界方向。"""
    quat = base.cam.getQuat(base.render)
    return quat.getRight(), quat.getUp()


def _drag_result(camera, base, action):
    """执行一次拖拽，返回相机位移在"屏幕右 / 屏幕上"上的投影。

    相机沿着 **+屏幕右** 走 → 画面内容往左跑；沿着 **-屏幕右** 走 → 画面往右跑。
    所以想验证"画面跟手"，就看这两个数是不是和鼠标同号的反面。
    """
    before = Vec3(base.cam.getPos(base.render))
    right, up = _screen_axes(base)
    action()
    delta = Vec3(base.cam.getPos(base.render)) - before
    return delta.dot(right), delta.dot(up)


@pytest.fixture()
def camera(app):
    return OrbitCamera(app, distance=330.0)


# --------------------------------------------------------------------------- #
# 环绕
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pixels, label", ((_DRAG, "右"), (-_DRAG, "左")))
def test_orbit_horizontally_follows_the_cursor(app, camera, pixels, label):
    """鼠标往右拖，画面里的东西往右走（相机就得往左绕）。"""
    along_right, _ = _drag_result(
        camera, app, lambda: camera.orbit(pixels, 0.0))
    assert abs(along_right) > 1.0, "相机根本没动，环绕没生效"
    assert along_right * pixels < 0, (
        f"鼠标往{label}拖，相机也跟着往{label}走 —— 画面会往反方向跑，方向反了")


@pytest.mark.parametrize("pixels, label", ((_DRAG, "上"), (-_DRAG, "下")))
def test_orbit_vertically_follows_the_cursor(app, camera, pixels, label):
    """鼠标往上拖，画面往上走（相机压低）。"""
    _, along_up = _drag_result(camera, app, lambda: camera.orbit(0.0, pixels))
    assert abs(along_up) > 1.0, "相机根本没动，环绕没生效"
    assert along_up * pixels < 0, (
        f"鼠标往{label}拖，相机也跟着往{label}走 —— 画面的移动方向是反的")


def test_orbit_diagonal_moves_both_axes_together(app, camera):
    """斜着拖：两个轴的跟手感要一致，不能一个跟手一个反着。"""
    along_right, along_up = _drag_result(
        camera, app, lambda: camera.orbit(_DRAG, _DRAG))
    assert along_right < 0 and along_up < 0


def test_orbit_keeps_the_target_fixed(app, camera):
    """环绕只转角度，目标点（画面中心）不能漂。"""
    before = list(camera.target)
    camera.orbit(_DRAG, _DRAG)
    assert camera.target == before


def test_orbit_clamps_elevation_after_the_sign_flip(app, camera):
    """方向翻转后俯仰限位照样有效 —— 别翻到地面以下。"""
    for _ in range(200):
        camera.orbit(0.0, -_DRAG * 4)          # 一路往下拖
    assert camera.elevation > 0.0
    for _ in range(400):
        camera.orbit(0.0, _DRAG * 4)           # 一路往上拖
    assert camera.elevation < 1.6             # < 90°


# --------------------------------------------------------------------------- #
# 平移
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pixels, label", ((_DRAG, "右"), (-_DRAG, "左")))
def test_pan_horizontally_follows_the_cursor(app, camera, pixels, label):
    along_right, _ = _drag_result(camera, app, lambda: camera.pan(pixels, 0.0))
    assert abs(along_right) > 0.5, "焦点没动，平移没生效"
    assert along_right * pixels < 0, f"鼠标往{label}拖，画面却往反方向跑"


@pytest.mark.parametrize("pixels, label", ((_DRAG, "上"), (-_DRAG, "下")))
def test_pan_vertically_follows_the_cursor(app, camera, pixels, label):
    _, along_up = _drag_result(camera, app, lambda: camera.pan(0.0, pixels))
    assert abs(along_up) > 0.5, "焦点没动，平移没生效"
    assert along_up * pixels < 0, (
        f"鼠标往{label}拖，焦点也跟着往{label}走 —— "
        f"平移和环绕的纵向手感会不一致")


def test_pan_moves_the_target_and_orbit_does_not(app, camera):
    """分工不能串：平移动目标点，环绕不动。"""
    before = list(camera.target)
    camera.pan(_DRAG, _DRAG)
    assert camera.target != before
    orbiting = list(camera.target)
    camera.orbit(_DRAG, _DRAG)
    assert camera.target == orbiting


# --------------------------------------------------------------------------- #
# 滚轮与取景
# --------------------------------------------------------------------------- #

def test_zoom_in_on_wheel_up(app, camera):
    before = camera.distance
    camera.zoom(1.0)
    assert camera.distance < before, "滚轮上滚该拉近"


def test_frame_centres_on_the_bounds(app, camera):
    camera.frame(((-60.0, -20.0), (60.0, 20.0)))
    assert camera.target[0] == pytest.approx(0.0)
    assert camera.target[2] == pytest.approx(0.0)
    assert camera.distance > 0.0


def test_frame_handles_an_empty_plot(app, camera):
    camera.frame(None)
    assert camera.target == [0.0, 0.0, 0.0]
