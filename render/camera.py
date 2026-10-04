"""环绕式（orbit / pan / zoom）相机。

交互分工
------------------------------------------------
* **右键拖拽** = 环绕：改变方位角与俯仰角，目标点不动。
* **中键拖拽** = 平移：目标点沿屏幕平面移动（左键留给放轨道）。
* **滚轮** = 沿视线前后推拉。

拖拽方向：一律"抓住画面拖"
------------------------------------------------
鼠标往右拖，**画面里的东西就往右走**；往上拖就往上去 —— 和 Google Earth / 地图
类应用一致。它的实现方式是让**相机往鼠标的反方向动**（相机退，画面就进）。

这条约定是有来由的：早期版本把两个轴都写成了"相机跟着鼠标走"，于是拖右变成画面
往左跑，用户一眼就看出"方向反了"。方向这种东西在代码里看不出来、只能靠体感，所以
``tests/test_camera.py`` 把"画面跟手"钉成了断言 —— 它量的是相机自身的位移方向。

角度约定（与 core 一致，减少记忆负担）
------------------------------------------------
``azimuth`` 沿用 core 的 heading 语义：从目标点看相机，**0 = 相机在 +X 方向**，
正方向朝 +Z 转。``elevation`` 是相机高出水平面的角度。于是默认视角
``azimuth = -55°``（相机落在 +X/-Z 一侧）、``elevation = 34°``，正好是沙盘游戏
常见的那种斜俯视。
"""

from __future__ import annotations

import math

from panda3d.core import NodePath, Vec3

from render.style import CAMERA_MAX_DISTANCE, CAMERA_MIN_DISTANCE

#: 鼠标像素 → 角度的灵敏度。
_ORBIT_RADIANS_PER_PIXEL = 0.0072
#: 鼠标像素 → 平移距离的比例（再乘当前距离，做到"拉远了移动得快"）。
_PAN_PER_PIXEL = 0.0016
#: 滚轮一格 → 距离缩放比例。
_ZOOM_PER_NOTCH = 1.12
#: 俯仰角限制，避免翻到地面以下产生万向节错觉。
_MIN_ELEVATION = math.radians(4.0)
_MAX_ELEVATION = math.radians(88.0)


class OrbitCamera:
    """一个把 ``target / distance / azimuth / elevation`` 映射到相机 HPR 的控件。"""

    def __init__(self, base, *, target=(0.0, 0.0, 0.0), distance: float = 330.0,
                 azimuth_deg: float = -55.0, elevation_deg: float = 34.0):
        self.base = base
        self.camera = base.camera
        self._focus = base.render.attachNewNode("camera_focus")
        self.target = list(target)
        self.distance = distance
        self.azimuth = math.radians(azimuth_deg)
        self.elevation = math.radians(elevation_deg)
        self.apply()

    # ---------------------------------------------------------------- 状态

    def camera_position(self) -> tuple[float, float, float]:
        horizontal = self.distance * math.cos(self.elevation)
        return (
            self.target[0] + horizontal * math.cos(self.azimuth),
            self.target[1] + self.distance * math.sin(self.elevation),
            self.target[2] + horizontal * math.sin(self.azimuth),
        )

    def apply(self) -> None:
        """把当前参数写进相机节点。"""
        self.camera.setPos(*self.camera_position())
        self._focus.setPos(*self.target)
        self.camera.lookAt(self._focus)

    # ---------------------------------------------------------------- 交互

    def orbit(self, dx: float, dy: float) -> None:
        """环绕视角。

        符号按"抓住画面拖"约定：``dx > 0``（鼠标往右）时**增大**方位角，相机跟着
        往左绕，画面里的东西于是往右走。俯仰同理 —— 鼠标往上拖，相机压低，画面
        往上走（看到的东西往上翻）。
        """
        self.azimuth += dx * _ORBIT_RADIANS_PER_PIXEL
        self.elevation = _clamp(
            self.elevation - dy * _ORBIT_RADIANS_PER_PIXEL,
            _MIN_ELEVATION, _MAX_ELEVATION,
        )
        self.apply()

    def pan(self, dx: float, dy: float) -> None:
        """沿屏幕平面平移。

        平移的"跟手"是反着来的：想让画面往右走，得把**目标点往左挪**（相机跟着
        目标点走，画面就相对往右）。所以这里整体乘一个负号，和 :meth:`orbit`
        共用一个约定。
        """
        quat = self.camera.getQuat()
        right = quat.getRight()
        up = quat.getUp()
        scale = self.distance * _PAN_PER_PIXEL
        offset = (right * dx + up * dy) * -scale
        self.target[0] += offset[0]
        self.target[1] += offset[1]
        self.target[2] += offset[2]
        self.apply()

    def move_ground(self, forward: float, right: float) -> None:
        """沿**地面**平移焦点：``forward`` 朝视线前方，``right`` 朝右手边。

        和 :meth:`pan`（沿屏幕平面，会把焦点抬离地面）不同，这里刻意把两个方向
        都压平到 XZ 平面 —— 编辑轨道时视点应当始终贴着沙盘滑动，
        否则按几下 W，焦点就升到天上、鼠标射线再也打不到地面了。

        方向直接取相机自身的 ``getForward`` / ``getRight`` 再投影，
        因此永远不会和 :meth:`pan` 的"右手边"定义打架。
        """
        quat = self.camera.getQuat()
        look = quat.getForward()
        side = quat.getRight()

        fx, fz = look[0], look[2]
        rx, rz = side[0], side[2]
        # 相机几乎垂直向下看时水平投影会退化成零向量，退回到"不移动"而不是胡乱走
        length_f = math.hypot(fx, fz)
        length_r = math.hypot(rx, rz)

        if length_f > 1e-6:
            self.target[0] += fx / length_f * forward
            self.target[2] += fz / length_f * forward
        if length_r > 1e-6:
            self.target[0] += rx / length_r * right
            self.target[2] += rz / length_r * right
        self.apply()

    def spin(self, angle: float) -> None:
        """绕竖轴转动视角（键盘用）。正角度 = 相机向 +Z 一侧绕过去。"""
        self.azimuth += angle
        self.apply()

    def zoom(self, notches: float) -> None:
        self.distance = _clamp(
            self.distance * (_ZOOM_PER_NOTCH ** (-notches)),
            CAMERA_MIN_DISTANCE, CAMERA_MAX_DISTANCE,
        )
        self.apply()

    def dolly(self, factor: float) -> None:
        """按比例直接推拉（键盘 W/S 用）。"""
        self.distance = _clamp(self.distance * factor,
                               CAMERA_MIN_DISTANCE, CAMERA_MAX_DISTANCE)
        self.apply()

    # ---------------------------------------------------------------- 取景

    def frame(self, bounds, margin: float = 1.45) -> None:
        """把给定的世界包围盒放进画面。

        ``bounds`` 形如 ``((min_x, min_z), (max_x, max_z))``，即
        :meth:`core.track.layout.Layout.bounds` 的返回值；也可以是
        ``(min_x, min_y, min_z, max_x, max_y, max_z)`` 六元组。
        """
        if bounds is None:
            self.target = [0.0, 0.0, 0.0]
            self.distance = 330.0
            self.apply()
            return

        if len(bounds) == 2:
            (min_x, min_z), (max_x, max_z) = bounds
            min_y = max_y = 0.0
        else:
            min_x, min_y, min_z, max_x, max_y, max_z = bounds

        self.target = [(min_x + max_x) * 0.5, (min_y + max_y) * 0.5,
                       (min_z + max_z) * 0.5]
        span = max(max_x - min_x, max_z - min_z, 20.0)
        # 透视相机水平半视角约 30°，取跨度的一半除以 tan(30°) 再留点余量。
        self.distance = _clamp(span * margin, CAMERA_MIN_DISTANCE,
                               CAMERA_MAX_DISTANCE)
        self.apply()


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)
