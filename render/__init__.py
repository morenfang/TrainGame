"""渲染层：把 core 的位姿代数变成 Panda3D 的场景图。

分层约定
------------------------------------------------
* ``core/`` 不 import 本包的任何东西，本包只读 core。
* 全项目**唯一**的方向转换在 :mod:`render.transform`（``H = -degrees(heading)``）。
  其余模块一律直接使用 core 的 (x, y_up, z) 坐标。
* 所有几何都用 :class:`render.mesh.MeshBuilder` 程序化生成，顶点色里已经烘焙好
  光照，因此不需要挂 Light / 开 shader。
"""

from __future__ import annotations
