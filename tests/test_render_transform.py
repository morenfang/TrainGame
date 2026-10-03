"""钉死渲染层的方向约定：``apply_pose`` 必须与 ``core`` 的位姿代数逐位一致。

为什么这个测试是整个渲染层最重要的一条
------------------------------------------------
core 的角度约定是「heading 绕 +Y、从 +X 转向 +Z 为正」，而 Panda3D 的 HPR 是
右手系旋转 —— 实测在 ``coordinate-system yup`` 下 ``H(+90°)`` 把 ``+X`` 转到
``-Z``，**符号正好相反**。于是渲染层必须做一次 ``H = -degrees(heading)``。

麻烦的是：这层转换一旦写错（漏了、或者别处又翻了一次），场景看起来仍然"像那么
回事"—— 轨道照样横平竖直、圆照样是圆，只有**左右手性**变了。人眼很难发现，
但所有弯轨的转向、道岔的开向、列车的前后关系全都会反。所以必须用机器检查：

    对随机位姿 P 与随机局部点 q：
        apply_pose(节点, P) 之后的 Panda3D 变换矩阵 · q   ≡   P.compose(q)

右边是 core 自己的代数（走线、列车定位都基于它），左边是实际交给显卡的矩阵。
两者一致 ⇔ 渲染出来的东西和 core 算出来的东西是同一套几何。

注意：这里刻意用了 :func:`panda3d.core.TransformState.makePosHpr` **而不是自己写
矩阵**，因为前者的解释依赖 ``coordinate-system`` 这个 prc 设置（zup 下 H 绕 +Z，
yup 下 H 绕 +Y），只有用它才是在检验"引擎真的这么理解"，而不是在检验我自己
推导的公式（那会变成循环论证）。
"""

from __future__ import annotations

import math
import random

import pytest

from panda3d.core import TransformState, Vec3, loadPrcFileData

# 必须在任何 Panda3D 对象生成之前设定，且与 app/config.py 保持一致。
loadPrcFileData("", "coordinate-system yup")

from core.geometry import Pose  # noqa: E402
from render.transform import apply_pose, heading_to_hpr  # noqa: E402

#: 容差必须按 **float32** 定，而不是按 float64。
#:
#: Panda3D 的 ``Mat4`` 是 ``PN_stdfloat`` 矩阵，标准发行版里就是 float32：连
#: ``cos(90°)`` 都会算成 ``-4.37e-08`` 而不是 0。所以拿 1e-9 去比是必错的，
#: 那不是 bug 而是引擎的数值精度上限。坐标最高到几百米，float32 的相对误差
#: ~6e-8，绝对误差因此可到 ~1e-5 m 量级；下面取 1e-4 m（0.1 毫米）留足余量。
#:
#: 这个宽松度**完全不影响测试的价值**：这层最典型的 bug 是"漏翻 / 多翻一次
#: 方向"，误差是**米级甚至整体镜像**，与 1e-4 m 相差 5 个数量级。
_TOLERANCE_ABS = 1e-4
_TOLERANCE_REL = 1e-5


def _assert_point_matches(got, want) -> None:
    assert got.x == pytest.approx(want.x, abs=_TOLERANCE_ABS, rel=_TOLERANCE_REL)
    assert got.y == pytest.approx(want.y, abs=_TOLERANCE_ABS, rel=_TOLERANCE_REL)
    assert got.z == pytest.approx(want.z, abs=_TOLERANCE_ABS, rel=_TOLERANCE_REL)


def _panda_matrix(pose: Pose):
    """按 :func:`apply_pose` 的方式构造 Panda3D 变换矩阵。"""
    return TransformState.makePosHpr(
        Vec3(pose.x, pose.y, pose.z),
        Vec3(heading_to_hpr(pose.heading), 0.0, 0.0),
    ).getMat()


def _random_pose(rng: random.Random) -> Pose:
    return Pose(
        rng.uniform(-120.0, 120.0),
        rng.uniform(-6.0, 6.0),
        rng.uniform(-120.0, 120.0),
        rng.uniform(-3.0 * math.pi, 3.0 * math.pi),
    )


_LOCAL_POINTS = (
    (1.0, 0.0, 0.0),      # 局部 +X：轨道的"前"
    (0.0, 1.0, 0.0),      # 局部 +Y：向上
    (0.0, 0.0, 1.0),      # 局部 +Z：横向
    (0.7175, -0.18, 0.0),  # 钢轨位置
    (-0.3, 0.7, -1.9),    # 一个不对称的点
)


@pytest.mark.parametrize("local_point", _LOCAL_POINTS)
def test_apply_pose_matches_core_compose(local_point):
    """随机位姿下，Panda3D 的矩阵乘法与 core 的 compose 结果一致。"""
    rng = random.Random(20240607)
    for _ in range(120):
        pose = _random_pose(rng)
        matrix = _panda_matrix(pose)
        got = matrix.xformPoint(Vec3(*local_point))
        want = pose.compose(Pose(*local_point))
        _assert_point_matches(got, want)


def test_heading_matches_core_forward_direction():
    """heading 必须与 core 的 forward 定义严格一致（这是"转向对不对"的判据）。"""
    for degrees in (-180.0, -90.0, -22.5, 0.0, 22.5, 45.0, 90.0, 137.0, 180.0):
        pose = Pose(0.0, 0.0, 0.0, math.radians(degrees))
        matrix = _panda_matrix(pose)
        got = matrix.xformPoint(Vec3(1.0, 0.0, 0.0))
        fx, _, fz = pose.forward()
        assert got.x == pytest.approx(fx, abs=1e-6)
        assert got.z == pytest.approx(fz, abs=1e-6)
        assert got.y == pytest.approx(0.0, abs=1e-6)


def test_apply_pose_actually_writes_the_node():
    """`apply_pose` 真把位姿写进了节点（而不是只算了一个正确的矩阵）。"""
    from panda3d.core import NodePath

    node = NodePath("probe")
    pose = Pose(12.5, 1.25, -30.0, math.radians(37.0))
    apply_pose(node, pose)
    matrix = node.getMat()
    for local_point in _LOCAL_POINTS:
        got = matrix.xformPoint(Vec3(*local_point))
        _assert_point_matches(got, pose.compose(Pose(*local_point)))
    node.removeNode()


def test_handedness_is_not_mirrored():
    """左右手性：绕 +Y 转 +90° 后，局部 +X 必须落到 +Z（而不是 -Z）。

    这条单独拎出来，是因为"镜像"是这层转换最典型的失败模式 ——
    它在视觉上几乎不可察觉，但会让所有弯轨的转向整体反掉。
    """
    pose = Pose(0.0, 0.0, 0.0, math.pi / 2.0)
    got = _panda_matrix(pose).xformPoint(Vec3(1.0, 0.0, 0.0))
    assert got.z == pytest.approx(1.0, abs=1e-6), "轨道整体被镜像了"
    assert got.x == pytest.approx(0.0, abs=1e-6)


def test_random_poses_are_all_rigid_body_transforms():
    """抽查：把一整套局部点一次性变换后，点间距必须保持不变（不能有缩放 / 剪切）。"""
    rng = random.Random(90210)
    points = [Vec3(*p) for p in _LOCAL_POINTS]
    for _ in range(40):
        pose = _random_pose(rng)
        matrix = _panda_matrix(pose)
        moved = [matrix.xformPoint(p) for p in points]
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                before = (points[i] - points[j]).length()
                after = (moved[i] - moved[j]).length()
                assert after == pytest.approx(before, abs=1e-4, rel=1e-5)


# --------------------------------------------------------------------------- #
# 俯仰（坡道上的车体）
# --------------------------------------------------------------------------- #

def _pitched_matrix(heading_rad: float, pitch_rad: float):
    from render.transform import pitch_to_hpr
    return TransformState.makePosHpr(
        Vec3(0.0, 0.0, 0.0),
        Vec3(heading_to_hpr(heading_rad), 0.0, pitch_to_hpr(pitch_rad)),
    ).getMat()


@pytest.mark.parametrize("pitch_deg", [-30.0, -7.5, 1.0, 12.0, 25.0])
def test_nose_pitch_raises_the_forward_axis_only(pitch_deg):
    """抬头必须把**车头抬起来**，而且只抬车头。

    三条一起判，缺一条就兜不住最容易犯的错：

    * ``+X``（车头）的 y 分量符号与 pitch 一致 —— 写成 ``|pitch|`` 会挂；
    * ``+Z``（车体横轴）**完全不动** —— 俯仰绕的就是它，动了就说明转错轴了
      （``yup`` 世界里 ``P`` 分量转的是纵轴，那会让车"横着翻"）；
    * ``+X`` 的水平投影仍与 heading 同向 —— 说明是"先偏航、再绕车体横轴俯仰"。
    """
    pitch = math.radians(pitch_deg)
    heading = math.radians(35.0)
    matrix = _pitched_matrix(heading, pitch)

    forward = matrix.xformPoint(Vec3(1.0, 0.0, 0.0))
    right = matrix.xformPoint(Vec3(0.0, 0.0, 1.0))

    assert forward.y == pytest.approx(math.sin(pitch), abs=1e-6)
    # 横轴必须**纹丝不动**地留在水平面里（yup 下 +Z 本来就是水平横轴）
    assert right.y == pytest.approx(0.0, abs=1e-6)
    # core 的约定：局部 +Z 经 heading 后是 (-sin h, 0, cos h)
    assert right.x == pytest.approx(-math.sin(heading), abs=1e-6)
    assert right.z == pytest.approx(math.cos(heading), abs=1e-6)
    # 车头的水平投影仍指向 heading
    horizontal = math.hypot(forward.x, forward.z)
    assert horizontal == pytest.approx(math.cos(pitch), abs=1e-6)
    assert forward.x / horizontal == pytest.approx(math.cos(heading), abs=1e-6)
    assert forward.z / horizontal == pytest.approx(math.sin(heading), abs=1e-6)


def test_pitch_does_not_disturb_the_heading_convention():
    """pitch=0 时，带俯仰的写法必须与老 ``apply_pose`` **逐位相同**。

    这条是"加了参数没改坏老路径"的守门员：轨道件、端口标记、幽灵预览全都还走
    原来那条路，它们不该因为列车需要俯仰而多出任何一点变化。
    """
    from panda3d.core import NodePath

    from render.transform import apply_pose
    rng = random.Random(4242)
    for _ in range(30):
        pose = _random_pose(rng)
        plain, pitched = NodePath("a"), NodePath("b")
        apply_pose(plain, pose)
        apply_pose(pitched, pose, 0.0)
        for local_point in _LOCAL_POINTS:
            a = plain.getMat().xformPoint(Vec3(*local_point))
            b = pitched.getMat().xformPoint(Vec3(*local_point))
            _assert_point_matches(a, b)
        plain.removeNode()
        pitched.removeNode()


def test_apply_pose_writes_the_pitch_into_the_node():
    from panda3d.core import NodePath

    from render.transform import apply_pose
    node = NodePath("ramp_car")
    pose = Pose(3.0, 1.2, -7.0, math.radians(10.0))
    apply_pose(node, pose, math.radians(3.0))
    # xformPoint 会把位置也算进去，所以先减掉节点原点，只留方向
    origin = node.getMat().xformPoint(Vec3(0.0, 0.0, 0.0))
    got = node.getMat().xformPoint(Vec3(1.0, 0.0, 0.0)) - origin
    assert got.y == pytest.approx(math.sin(math.radians(3.0)), abs=1e-6)
    node.removeNode()
