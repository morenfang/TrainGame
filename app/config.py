"""引擎与窗口配置。**必须在创建 ``ShowBase`` 之前调用** :func:`configure_engine`。

为什么这些设置要集中在一处
------------------------------------------------
其中两条是"写错了不会报错、只会让画面悄悄不对"的类型：

* ``coordinate-system yup`` —— 不设的话世界是 Z 向上，整个场景会侧翻 90°，
  而且 :func:`render.transform.apply_pose` 的符号约定也随之失效。
* 背景色 —— Panda3D 的默认清屏色是一块中灰（实测 ``#696969``）。在有远景的
  场景里它看起来像"渲染坏了"，而不是"天空"。

集中在这里还有个好处：``scripts/screenshot.py`` 的离屏渲染与真正的窗口渲染
走的是同一份配置，因此**截图看到的就是运行看到的**，截图才能当自检用。
"""

from __future__ import annotations

from panda3d.core import Fog, LColor, loadPrcFileData

from render import style

#: 默认窗口尺寸。
DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 900


def configure_engine(*, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
                     offscreen: bool = False, vsync: bool = True,
                     audio: bool = True) -> None:
    """装载 PRC 设置。必须在 ``ShowBase()`` 之前调用，且只调用一次。"""
    # 世界坐标：+Y 向上、地面为 XZ 平面，与 core 完全一致。
    loadPrcFileData("", "coordinate-system yup")
    loadPrcFileData("", f"win-size {width} {height}")
    # 窗口模式保留默认音频库（OpenAL，列车音效用）；离屏 / 测试显式关成 null。
    # 不写 "openal" —— 那会被当成动态库名（libopenal.so）去找，反而加载失败。
    if not (audio and not offscreen):
        loadPrcFileData("", "audio-library-name null")
    loadPrcFileData("", "sync-video " + ("true" if vsync else "false"))
    # 关掉与画面无关的通知，免得盖住我们自己的日志；出错仍会显示。
    loadPrcFileData("", "notify-level-display error")
    loadPrcFileData("", "notify-level-glgsg error")
    if offscreen:
        loadPrcFileData("", "window-type offscreen")


def configure_scene(base) -> Fog:
    """给场景装上天空色与远景雾。返回雾对象，便于运行时调密度。"""
    sky = LColor(*style.SKY_COLOR)
    base.setBackgroundColor(sky)

    fog = Fog("scene_fog")
    fog.setColor(sky)
    fog.setExpDensity(style.FOG_DENSITY)
    base.render.setFog(fog)

    # 近裁剪面不能太小：400 m 处的深度分辨率与 near 成正比，太小会让网格线闪烁。
    base.camLens.setNearFar(style.CAMERA_NEAR, style.CAMERA_FAR)
    return fog
