"""应用层：窗口、输入、编辑器状态机。

这一层是唯一允许同时 import ``core`` 与 ``render`` 的地方 —— 它负责把用户的
鼠标键盘动作翻译成对 ``core.track.layout.Layout`` 的修改，再让
``render.scene.LayoutView`` 把结果同步到场景图。两个方向都不反向依赖。
"""

from __future__ import annotations
