"""``app/config.py`` 那条"只在真正开窗口时才走到"的路径。

为什么单独给这几行写测试
------------------------------------------------
``configure_scene`` 曾经是**没人调用过**的死代码：离屏截图脚本自己抄了一份
同样的设置，于是 ``LColor(*SKY_COLOR)`` 里少了一个 alpha 分量这件事一直没人发现
—— 直到真的去开一次窗口才当场炸掉。教训很直白：只被"真实入口"调用到的代码，
必须有一条测试真的走一遍真实入口的那段逻辑，否则它就是没验证过的。

（同一份逻辑在 ``scripts/editor_demo.py`` 与 ``scripts/screenshot.py`` 里各抄了
一份，所以下面的断言同时也在守着"三者行为一致"这件事。）
"""

from __future__ import annotations

import pytest
from panda3d.core import LColor

from app import config
from render import style


def test_configure_scene_sets_sky_and_fog(app):
    fog = config.configure_scene(app)

    assert app.getBackgroundColor() == LColor(*style.SKY_COLOR)
    assert app.render.getFog() is not None
    assert fog.getExpDensity() == pytest.approx(style.FOG_DENSITY)
    # 雾色必须与天空同色，否则远景会淡出成另一种颜色，出现一条可见的边界
    assert fog.getColor() == LColor(*style.SKY_COLOR)


def test_configure_scene_sets_the_clipping_planes(app):
    config.configure_scene(app)
    assert app.camLens.getNear() == pytest.approx(style.CAMERA_NEAR)
    assert app.camLens.getFar() == pytest.approx(style.CAMERA_FAR)


def test_sky_colour_has_four_components():
    """``LColor(*colour)`` 只接受 1 / 2 / 4 个分量，三个会直接抛 TypeError。"""
    assert len(style.SKY_COLOR) == 4
    LColor(*style.SKY_COLOR)          # 能构造出来才算数
