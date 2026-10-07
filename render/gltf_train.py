"""把外部 glTF 编组拆成按节走的车体模板。

设计边界
--------------------------------------------------------------------
参数化放样（:mod:`render.train_mesh`）仍然是缺省外观：没有文件、加载失败、
或编组没声明 ``mesh``，全都退回它。本模块只在「确实有一份能用的 .glb」时
介入，并且只做三件渲染层该做的事：

1. **丢掉模型自带的轨道。** 沙盘的钢轨已经由 :mod:`render.track_mesh` 画了；
   再留一份会双线、还会按真实 200 m 编组尺度把场地撑爆。
2. **把每节车的原点挪到两转向架中点。** 这与 ``CarState.body_pose`` 是同一套
   约定，于是 :mod:`render.train_view` 仍然只写位姿、不算几何。
3. **只缩长度。** 概要设计 §5.8：``SCALE_MODEL ≈ 0.4`` 作用于车长，横断面不
   跟着缩（轨距是真实 1.435 m）。均匀缩小会让轮对落到轨距里面。

模型文件里动车组（``Car01``…）约定头车鼻锥在局部 +X；尾车再绕 +Y 转 180°。
实际资产两端常不一致，加载时按尖端半宽自动把鼻锥扳到 +X，再按角色加尾车掉头。
机车牵引的编组（绿皮、蒸汽）只转第一节。旋转打进子节点，车体节点本身保持单位
变换 —— 否则 :func:`render.transform.apply_pose` 每帧 ``setHpr`` 会盖掉。
"""

from __future__ import annotations

from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    GeomNode,
    GeomVertexReader,
    Mat4,
    Material,
    MaterialAttrib,
    NodePath,
    Point3,
    ShadeModelAttrib,
    Vec4,
)

from core import paths
from core.train.consist import TrainSpec
from render import style

#: 与概要设计 ``SCALE_MODEL`` 同一数值：只乘在车长（局部 X）上。
LENGTH_SCALE = 0.4
#: 观感加高（Y-up 车高），约 15%。
HEIGHT_SCALE = 1.15

#: 网上下来的碎网格（无名字、乱轴向、单位不是米）归一化到这个车高。
_TARGET_HEIGHT_M = 3.5
#: 整体最长边短于这个数，就当「不是米制」走碎网格路径。
_METER_SIZE_HINT = 10.0

_YAW_DEG = 180.0
_TRACK_NAMES = frozenset({"Track", "Ballast"})
#: 连接器 / 车钩装饰等，不是车厢，fallback 发现时要跳过。
_SKIP_PREFIXES = ("Sleeper", "Rail", "GW", "GWFrame")

_cache: dict[str, tuple[NodePath, ...]] = {}


class GltfTrainError(RuntimeError):
    """glTF 编组加载失败。调用方应退回参数化网格，而不是让游戏起不来。"""


def mesh_path_for(spec: TrainSpec):
    """编组声明的模型文件；没声明或找不到时返回 ``None``。"""
    if not spec.mesh:
        return None
    return paths.resolve_model_path(spec.mesh)


def triangle_count(node: NodePath) -> int:
    """一个节点树里的三角形数（含实例化前的模板本身）。"""
    total = 0
    for found in node.findAllMatches("**/+GeomNode"):
        geom_node = found.node()
        if not isinstance(geom_node, GeomNode):
            continue
        for index in range(geom_node.getNumGeoms()):
            geom = geom_node.getGeom(index)
            for prim_i in range(geom.getNumPrimitives()):
                total += geom.getPrimitive(prim_i).getNumPrimitives()
    return total


def _load_root(path) -> NodePath:
    try:
        import gltf
    except ImportError as exc:
        raise GltfTrainError("未安装 panda3d-gltf，无法加载 .glb") from exc
    node = gltf.load_model(str(path))
    if node is None:
        raise GltfTrainError(f"loader 没能读出 {path}")
    root = NodePath(node)
    if root.isEmpty():
        raise GltfTrainError(f"loader 读出了空节点：{path}")
    # 模型自带的钢轨 / 道砟 / 枕木：沙盘里已经有一份，去掉。
    for extra in ("Track", "Ballast", "Rail_-0.7175", "Rail_0.7175"):
        found = root.find(f"**/{extra}")
        if not found.isEmpty():
            found.removeNode()
    for sleeper in root.findAllMatches("**/Sleeper_*"):
        sleeper.removeNode()
    for rail in root.findAllMatches("**/Rail_*"):
        rail.removeNode()
    return root


def _is_numbered_car(name: str) -> bool:
    return name.startswith("Car") and name[3:].isdigit()


def _aabb(node: NodePath) -> tuple[Point3, Point3] | None:
    low, high = Point3(), Point3()
    if not node.calcTightBounds(low, high):
        return None
    return low, high


def _extent(node: NodePath) -> tuple[float, float, float] | None:
    box = _aabb(node)
    if box is None:
        return None
    low, high = box
    return (high.x - low.x, high.y - low.y, high.z - low.z)


def _looks_unstructured(root: NodePath) -> bool:
    """网上下来的碎网格：没有 Car01…，整体尺度也远不是米。"""
    for found in root.findAllMatches("**"):
        if _is_numbered_car(found.getName()):
            return False
    size = _extent(root)
    if size is None:
        return False
    return max(size) < _METER_SIZE_HINT


def _is_flat_track(dx: float, dy: float, dz: float) -> bool:
    """又长又扁的碎片当铁轨丢掉（沙盘自己有轨）。"""
    length = max(dx, dy, dz)
    height = min(dx, dy, dz)
    # 中间那一维是宽度；铁轨断面高度远小于长度。
    mid = dx + dy + dz - length - height
    return length > 0.4 and height < 0.012 and mid < 0.08


def _lift_dark_materials(root: NodePath, floor: float = 0.35) -> None:
    """资源站 glb 常把车体做成接近纯黑的 PBR 底色，沙盘灯光下等于隐形。

    把过暗的漫反射抬到 ``floor``，保留相对明暗；有贴图的不动。
    """
    for found in root.findAllMatches("**/+GeomNode"):
        geom_node = found.node()
        if not isinstance(geom_node, GeomNode):
            continue
        for index in range(geom_node.getNumGeoms()):
            state = geom_node.getGeomState(index)
            mat_attr = state.getAttrib(MaterialAttrib)
            if mat_attr is None:
                continue
            material = mat_attr.getMaterial()
            if material is None:
                continue
            base = material.getBaseColor()
            if max(base[0], base[1], base[2]) >= floor:
                continue
            lifted = Material(material)
            peak = max(base[0], base[1], base[2], 1e-6)
            scale = floor / peak
            lifted.setBaseColor(Vec4(
                min(1.0, base[0] * scale),
                min(1.0, base[1] * scale),
                min(1.0, base[2] * scale),
                base[3],
            ))
            # 金属漆太暗也看不见，稍微降一点粗糙度让高光回来。
            if hasattr(lifted, "setRoughness") and lifted.getRoughness() > 0.6:
                lifted.setRoughness(0.45)
            geom_node.setGeomState(
                index, state.setAttrib(MaterialAttrib.make(lifted)))


def _long_axis(dx: float, dy: float, dz: float) -> str:
    """包围盒最长边是哪一根轴。"""
    if dx >= dy and dx >= dz:
        return "x"
    if dy >= dx and dy >= dz:
        return "y"
    return "z"


def _remap_length_to_x(axis: str) -> Mat4:
    """把模型长轴扳到 +X，车高扳到 +Y（轮子朝下）。

    游戏是 Y-up：不能靠 setH。该 glb 在 Y-up 下长轴在 Z；先把
    (长, 高, 宽) 摆正，再绕 Z +90°，侧视确认轮子在下、受电弓在上。
    """
    if axis == "x":
        return Mat4.identMat()
    if axis == "y":
        return Mat4(0, 0, 1, 0,
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 0, 1)
    # base: 长 Z→X、高 X→Y、宽 Y→−Z；再 Rz(+90): (x,y,z)→(−y,x,z)
    base = Mat4(0, 1, 0, 0,
                0, 0, -1, 0,
                1, 0, 0, 0,
                0, 0, 0, 1)
    rz90 = Mat4(0, -1, 0, 0,
                1, 0, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1)
    return rz90 * base


def _regroup_unstructured(root: NodePath) -> list[NodePath]:
    """把无名字碎网格收成 ``Car01``…：去轨、按车长聚类、转到米制 X 前进。

    典型来源：资源站导出的整景 glb，子节点全是 ``Editable_Poly-…``，
    长轴不在 X，单位也不是米。Y-up 下 glTF 导入后长轴常在 Z。
    """
    consist = root.getChild(0) if root.getNumChildren() == 1 else root
    overall = _extent(consist)
    if overall is None:
        raise GltfTrainError("碎网格包围盒为空")
    # 整列最长轴 = 前进方向；碎片各自的长轴不可靠（受电弓等竖条会骗过投票）。
    length_axis = _long_axis(*overall)

    children = [consist.getChild(i) for i in range(consist.getNumChildren())]
    parts: list[dict] = []
    for child in children:
        box = _aabb(child)
        if box is None or child.find("**/+GeomNode").isEmpty():
            child.removeNode()
            continue
        low, high = box
        dx, dy, dz = high.x - low.x, high.y - low.y, high.z - low.z
        if _is_flat_track(dx, dy, dz):
            child.removeNode()
            continue
        # 退化碎片（几乎零尺寸）中心不可靠，丢掉，免得错误聚类拉长整节车厢。
        if max(dx, dy, dz) < 1e-4:
            child.removeNode()
            continue
        if length_axis == "x":
            center, length = 0.5 * (low.x + high.x), dx
        elif length_axis == "y":
            center, length = 0.5 * (low.y + high.y), dy
        else:
            center, length = 0.5 * (low.z + high.z), dz
        parts.append({
            "node": child,
            "center": center,
            "length": length,
        })
    if not parts:
        raise GltfTrainError("去掉铁轨后没有列车几何")

    bodies = [p for p in parts if p["length"] > 0.18]
    if not bodies:
        bodies = sorted(parts, key=lambda p: -p["length"])[: max(1, len(parts) // 6)]
    bodies.sort(key=lambda p: p["center"])
    seeds: list[float] = []
    for body in bodies:
        if not seeds or body["center"] - seeds[-1] > 0.12:
            seeds.append(body["center"])
        else:
            seeds[-1] = 0.5 * (seeds[-1] + body["center"])
    if not seeds:
        raise GltfTrainError("无法从碎网格里分出车厢")

    buckets: list[list[dict]] = [[] for _ in seeds]
    for part in parts:
        nearest = min(range(len(seeds)),
                      key=lambda i: abs(part["center"] - seeds[i]))
        buckets[nearest].append(part)

    cars: list[NodePath] = []
    for index, bucket in enumerate(buckets):
        car = consist.attachNewNode(f"Car{index + 1:02d}")
        for part in bucket:
            part["node"].wrtReparentTo(car)
        cars.append(car)

    remap = _remap_length_to_x(length_axis)
    need_remap = length_axis != "x"

    # 先给第一节转正，按其高度定统一缩尺（不要 copyTo，会搞乱共享 Geom）。
    uniform = 1.0
    for index, car in enumerate(cars):
        align = car.attachNewNode("prealign")
        for child in list(car.getChildren()):
            if child != align:
                child.reparentTo(align)
        if need_remap:
            align.setMat(remap)
        if index == 0:
            spun = _extent(car)
            if spun is None or spun[1] <= 1e-6:
                raise GltfTrainError("碎网格转正后没有高度")
            uniform = _TARGET_HEIGHT_M / spun[1]
        # setScale / setPos 都会冲掉 setMat，转正·缩尺·居中一次性乘进矩阵。
        # X 先 ×2，_prepare_car 对 Unit 再按 0.8 缩，净效果约「站立后车长再翻倍」。
        scale = Mat4.scaleMat(uniform * 2.0, uniform, uniform)
        oriented = scale * remap if need_remap else scale
        align.setMat(oriented)
        box = _aabb(car)
        if box is None:
            continue
        low, high = box
        center = Point3(
            0.5 * (low.x + high.x),
            0.5 * (low.y + high.y),
            0.5 * (low.z + high.z),
        )
        align.setMat(Mat4.translateMat(-center.x, -center.y, -center.z) * oriented)
        car.setPos(consist, center)

    cars.sort(key=lambda node: node.getPos(consist).x)
    for index, car in enumerate(cars):
        # 不用 Car01 前缀：那会触发动车组「尾车掉头」，轴刚转正时朝向还不稳。
        car.setName(f"Unit{index + 1:02d}")
        _lift_dark_materials(car)
    return cars


def _discover_cars(root: NodePath) -> list[NodePath]:
    """找出编组里的每一节：优先 ``Car01``…，否则按 consist 子节点的 X 排序。"""
    numbered: dict[int, NodePath] = {}
    for found in root.findAllMatches("**"):
        name = found.getName()
        if _is_numbered_car(name):
            numbered.setdefault(int(name[3:]), found)
    if numbered:
        return [numbered[key] for key in sorted(numbered)]

    if _looks_unstructured(root):
        return _regroup_unstructured(root)

    consist = root.getChild(0) if root.getNumChildren() == 1 else root
    cars: list[NodePath] = []
    for i in range(consist.getNumChildren()):
        child = consist.getChild(i)
        name = child.getName()
        if name in _TRACK_NAMES or name.startswith(_SKIP_PREFIXES):
            continue
        # 没几何的空节点（纯标记）不算一节车
        if child.find("**/+GeomNode").isEmpty():
            continue
        cars.append(child)
    cars.sort(key=lambda node: node.getPos(root).x)
    if not cars:
        raise GltfTrainError("没有找到任何车厢节点")
    return cars


def _tip_half_widths(node: NodePath) -> tuple[float, float] | None:
    """两端尖端的横向半宽（Y-up 下取 |Z|）。用来判断鼻锥在哪一头。"""
    box = _aabb(node)
    if box is None:
        return None
    low, high = box
    span = high.x - low.x
    if span < 1e-3:
        return None
    band = max(0.8, span * 0.08)
    left: list[float] = []
    right: list[float] = []
    for found in node.findAllMatches("**/+GeomNode"):
        geom_node = found.node()
        if not isinstance(geom_node, GeomNode):
            continue
        mat = found.getMat(node)
        for index in range(geom_node.getNumGeoms()):
            reader = GeomVertexReader(geom_node.getGeom(index).getVertexData(), "vertex")
            while not reader.isAtEnd():
                world = mat.xformPoint(reader.getData3())
                lateral = abs(world.z)
                if world.x < low.x + band:
                    left.append(lateral)
                if world.x > high.x - band:
                    right.append(lateral)
    if not left or not right:
        return None
    return max(left), max(right)


def _nose_at_plus_x(node: NodePath) -> bool | None:
    """鼻锥在较窄一端。两端半宽差不到 5% → 中间车，返回 None。"""
    tips = _tip_half_widths(node)
    if tips is None:
        return None
    left, right = tips
    if left <= 1e-6 or right <= 1e-6:
        return None
    ratio = left / right
    if abs(ratio - 1.0) < 0.05:
        return None
    return right < left


def _role_yaw(cars: list[NodePath], index: int) -> float:
    """按编组角色给出额外偏航：动车组尾车 180°，机车牵引头车 180°。"""
    if not cars:
        return 0.0
    name = cars[0].getName()
    if name.startswith("Unit"):
        return 0.0
    if _is_numbered_car(name):
        return _YAW_DEG if index == len(cars) - 1 else 0.0
    return _YAW_DEG if index == 0 else 0.0


def _prepare_car(empty: NodePath, role_yaw_deg: float) -> NodePath:
    """把「编组里的一节车 empty」做成单位变换的模板。

    旋转与缩尺打在子节点 ``align`` 上，车体节点本身保持单位变换 ——
    :func:`render.transform.apply_pose` 每帧会 ``setHpr``，不能把朝向写在
    同一个节点上。
    """
    template = NodePath(f"tpl_{empty.getName()}")
    empty.wrtReparentTo(template)
    empty.setPos(0.0, 0.0, 0.0)
    empty.setHpr(0.0, 0.0, 0.0)
    empty.setScale(1.0)
    # 碎网格路径会在子节点上留 remapping 矩阵；先烤进顶点，
    # 否则后面的 setScale 会和置换矩阵拧成飞掉的原点。
    empty.flattenStrong()
    box = _aabb(empty)
    if box is not None:
        low, high = box
        mid = Point3(
            0.5 * (low.x + high.x),
            0.5 * (low.y + high.y),
            0.5 * (low.z + high.z),
        )
        for child in list(empty.getChildren()):
            child.setPos(child.getPos() - mid)
        empty.flattenStrong()
    # 先把鼻锥扳到 +X，再叠加角色偏航（尾车朝外）。
    nose = _nose_at_plus_x(empty)
    nose_fix = 0.0 if nose is not False else _YAW_DEG
    yaw_deg = (nose_fix + role_yaw_deg) % 360.0
    if yaw_deg > 180.0:
        yaw_deg -= 360.0
    inner = empty.attachNewNode("align")
    for child in list(empty.getChildren()):
        if child != inner:
            child.reparentTo(inner)
    inner.setH(yaw_deg)
    # 碎网格 Unit* 按高度对齐后仍偏短，长度方向放到约 2.5 倍沙盘缩尺。
    length_scale = LENGTH_SCALE * 2.5 if empty.getName().startswith("Unit") else LENGTH_SCALE
    inner.setScale(length_scale, HEIGHT_SCALE, 1.0)
    low = Point3()
    high = Point3()
    template.calcTightBounds(low, high)
    inner.setY(inner.getY() - low.y)
    return template


def load_consist_templates(path) -> tuple[NodePath, ...]:
    """加载一份编组 glb，返回按走行顺序排好的车体模板。"""
    key = str(path)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    root = _load_root(path)
    found = _discover_cars(root)
    templates = tuple(
        _prepare_car(car, _role_yaw(found, index))
        for index, car in enumerate(found)
    )
    _cache[key] = templates
    return templates


def _unit_cars(
    head: NodePath, mids: tuple[NodePath, ...], tail: NodePath, count: int,
) -> tuple[NodePath, ...]:
    """一列动力单元：头 + 中间 + 尾（尾车模板已掉头，鼻锥朝外）。"""
    if count <= 0:
        return ()
    if count == 1:
        return (head,)
    if count == 2:
        return (head, tail)
    need = count - 2
    if need <= len(mids):
        picked_mids = [
            mids[int(index * len(mids) / need)] for index in range(need)
        ]
    else:
        picked_mids = [mids[index % len(mids)] for index in range(need)]
    return (head, *picked_mids, tail)


def _pick_cars(templates: tuple[NodePath, ...], count: int) -> tuple[NodePath, ...]:
    n = len(templates)
    if count <= 0 or n == 0:
        return ()
    if count == n:
        return templates
    if n == 1:
        return tuple(templates[0] for _ in range(count))
    head, tail = templates[0], templates[-1]
    mids = templates[1:-1] or (head,)
    # 16 节重联：两列 8 节对顶，中间两头车鼻锥相对。
    if count == 16:
        unit = _unit_cars(head, mids, tail, 8)
        return unit + unit
    return _unit_cars(head, mids, tail, count)


def cars_for(spec: TrainSpec) -> tuple[NodePath, ...] | None:
    """按编组节数抽出对应的车体模板。没有模型时返回 ``None``。"""
    path = mesh_path_for(spec)
    if path is None:
        return None
    try:
        templates = load_consist_templates(path)
    except GltfTrainError as exc:
        print(f"[列车模型] {path.name} 加载失败，改用程序化车体：{exc}")
        return None
    picked = _pick_cars(templates, spec.car_count)
    return picked or None


def light_train(root: NodePath) -> NodePath:
    """只照亮这一列车。灯挂在列车根的**兄弟**节点上，不占车厢子节点。"""
    existing = root.getPythonTag("gltf_light_holder")
    if existing:
        return existing
    holder = root.getParent().attachNewNode(f"gltf_lights_{root.getName()}")
    ambient = AmbientLight("train_ambient")
    ambient.setColor((0.28, 0.30, 0.33, 1.0))
    ambient_np = holder.attachNewNode(ambient)
    root.setLight(ambient_np)

    sun = DirectionalLight("train_sun")
    sun.setColor((0.55, 0.52, 0.48, 1.0))
    sun_np = holder.attachNewNode(sun)
    sun_np.setPos(style.SUN_DIR[0] * 200.0,
                  style.SUN_DIR[1] * 200.0,
                  style.SUN_DIR[2] * 200.0)
    sun_np.lookAt(0.0, 0.0, 0.0)
    root.setLight(sun_np)
    smooth = getattr(ShadeModelAttrib, "MSmooth", None) or getattr(
        ShadeModelAttrib, "M_smooth", 1)
    root.setAttrib(ShadeModelAttrib.make(smooth))
    root.setShaderAuto()
    root.setPythonTag("gltf_light_holder", holder)
    return holder
