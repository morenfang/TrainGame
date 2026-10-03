"""core 位姿 ↔ Panda3D 场景图变换：**全项目唯一的一处方向转换**。

为什么需要转换
------------------------------------------------
core 的角度约定是「heading 绕 +Y，从 +X 转向 +Z 为正」，即 2D 复平面上
``(x, z)`` 的标准旋向。而 Panda3D 的 HPR 是右手系旋转 —— 实测（``yup`` 世界下）

    H(+90°) 把 +X 转到 -Z        H(-90°) 把 +X 转到 +Z

两者符号相反，因此只要令 ``H = -degrees(heading)``，Panda3D 就会把**局部网格
坐标原样搬到 core 所要求的世界位置**：

    core 局部 (1, 0, 0) --heading h--> world (cos h, 0, sin h)
    Panda 局部 (1, 0, 0) --H = -h----> world (cos h, 0, sin h)      ✔ 一致
    core 局部 (0, 0, 1) ------------> world (-sin h, 0, cos h)
    Panda 局部 (0, 0, 1) --H = -h---> world (-sin h, 0, cos h)     ✔ 一致

位置分量是**恒等映射**（不做 x/z 交换、不做取反），所以本模块只有下面这几个
函数、只有几行几何。``tests/test_render_transform.py`` 用随机位姿比对
``Pose.compose`` 与 Panda3D 矩阵乘法，把这个不变量钉死。

俯仰为什么写在 **R** 分量上
------------------------------------------------
``Pose`` 只装得下位置与航向（这是刻意的：它让平面几何有精确的闭式解，见
``core.geometry``），而坡道上的车体还需要一个**俯仰角**。俯仰不能塞进 ``Pose``，
就只能在渲染这一层补上 —— 补在哪儿是实测出来的，不是按直觉猜的：

    H = +90°  →  局部 +X 落到 -Z          绕 +Y 偏航（与 heading 反号）
    P = +20°  →  局部 +X 不动、+Z 下沉      绕**车体纵轴**翻滚，**不是**俯仰
    R = +20°  →  局部 +X 下沉              绕**车体横轴**俯仰，符号取反

也就是说 ``HPR`` 里的 ``P`` 在 ``yup`` 世界里是翻滚、``R`` 才是俯仰。而且 H 与 R
同时给出时绕序是「先偏航、再绕**已经转过去的**车体横轴俯仰」—— 实测
``H=90°, R=-20°`` 时局部 +X 变成 ``(0, +0.342, -0.940)``，正是"车头朝 -Z 又顺着
坡抬了起来"。坡道上这一点写错，车会**横着漂**而不是顺着坡走，而且看起来还挺"像
那么回事"，所以 :func:`apply_pose` 的俯仰参数是被测试逐点比对过的。
"""

from __future__ import annotations

import math

from core.geometry import Pose


def heading_to_hpr(heading_rad: float) -> float:
    """core 的 heading（弧度）→ Panda3D 的 H（度）。"""
    return -math.degrees(heading_rad)


def pitch_to_hpr(pitch_rad: float) -> float:
    """core 的俯仰角（弧度，抬头为正）→ Panda3D 的 **R**（度）。

    见模块文档：``yup`` 世界里俯仰在 R 分量上，且符号与直觉相反。
    """
    return -math.degrees(pitch_rad)


def apply_pose(node_path, pose: Pose, pitch_rad: float = 0.0):
    """把一个 core 位姿写进场景图节点（位置恒等、航向取负、可选俯仰）。

    ``pitch_rad`` 缺省为 0，于是平地上的调用者完全不必知道有这么个参数 ——
    轨道件、端口标记、幽灵预览走的都还是原来那条路径。
    """
    node_path.setPos(pose.x, pose.y, pose.z)
    node_path.setHpr(heading_to_hpr(pose.heading), 0.0, pitch_to_hpr(pitch_rad))
    return node_path


def local_point(pose: Pose, x: float, y: float, z: float):
    """把件局部坐标点变换到世界坐标（用 core 的位姿代数，保证与走线一致）。"""
    world = pose.compose(Pose(x, y, z, 0.0))
    return (world.x, world.y, world.z)
