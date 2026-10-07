"""把源 glb 洗成「复兴号黄丝带400BF」编组。

Blender 是 Z-up；导出 ``export_yup=True`` 后游戏里变成：
车长 +X、车高 +Y、轮子贴 Y≈0。所以在 Blender 里把高放在 +Z。

用法::

    blender --background --python scripts/prep_glb_78566.py

源文件放在 ``models/`` 下，文件名见 ``SRC``；缺省仍认旧的 glbxz 文件名，
有新源时改 ``SRC`` 即可。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "glbxz-com-0000078566-1.glb"
OUT = ROOT / "models" / "CR400BF_HuangSiDai_6car.glb"

TARGET_HEIGHT = 3.5 * 1.15  # 观感再高约 15%
TRACK_LEN = 0.4
TRACK_H = 0.012
DEGENERATE = 1e-4
BODY_LEN = 0.18
SEED_GAP = 0.12


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                  bpy.data.armatures):
        for item in list(block):
            block.remove(item)


def _world_bounds(obj: bpy.types.Object):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    lo = Vector((min(xs), min(ys), min(zs)))
    hi = Vector((max(xs), max(ys), max(zs)))
    return lo, hi


def _extent(lo: Vector, hi: Vector) -> Vector:
    return hi - lo


def _is_track(dx: float, dy: float, dz: float) -> bool:
    length = max(dx, dy, dz)
    height = min(dx, dy, dz)
    mid = dx + dy + dz - length - height
    return length > TRACK_LEN and height < TRACK_H and mid < 0.08


def _long_axis(ext: Vector) -> str:
    if ext.x >= ext.y and ext.x >= ext.z:
        return "x"
    if ext.y >= ext.x and ext.y >= ext.z:
        return "y"
    return "z"


def _center_on_axis(lo: Vector, hi: Vector, axis: str) -> float:
    return 0.5 * (getattr(lo, axis) + getattr(hi, axis))


def _length_on_axis(ext: Vector, axis: str) -> float:
    return getattr(ext, axis)


def _mesh_objects() -> list[bpy.types.Object]:
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def _delete_objects(objs: list[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        if obj.name in bpy.context.scene.objects:
            obj.select_set(True)
    if bpy.context.selected_objects:
        bpy.ops.object.delete(use_global=False)


def _join_objects(objs: list[bpy.types.Object], name: str) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = name
    if joined.data:
        joined.data.name = name
    return joined


def _apply_transform(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _lift_dark_materials(floor: float = 0.35) -> None:
    for mat in bpy.data.materials:
        nt = getattr(mat, "node_tree", None)
        if nt is None:
            continue
        principled = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if principled is None:
            continue
        base = principled.inputs["Base Color"]
        if base.is_linked:
            continue
        color = list(base.default_value)
        peak = max(color[0], color[1], color[2], 1e-6)
        if peak >= floor:
            continue
        scale = floor / peak
        base.default_value = (
            min(1.0, color[0] * scale),
            min(1.0, color[1] * scale),
            min(1.0, color[2] * scale),
            color[3],
        )
        rough = principled.inputs.get("Roughness")
        if rough is not None and not rough.is_linked and rough.default_value > 0.6:
            rough.default_value = 0.45


def _blender_zup_remap(axis: str) -> Matrix:
    """Blender Z-up：结果为 X=车长、Z=车高、Y=车宽（导出 yup 后高→游戏 Y）。"""
    if axis == "x":
        return Matrix.Identity(4)
    if axis == "y":
        return Matrix((
            (0, 1, 0, 0),
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ))
    return Matrix((
        (0, 0, 1, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
    ))


def main() -> int:
    if not SRC.exists():
        print(f"[错误] 找不到源文件 {SRC}", file=sys.stderr)
        return 2

    print(f"[prep] 导入 {SRC}")
    _clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(SRC))

    meshes = _mesh_objects()
    print(f"[prep] 网格物体 {len(meshes)}")

    all_lo = Vector((1e9, 1e9, 1e9))
    all_hi = Vector((-1e9, -1e9, -1e9))
    trash: list[bpy.types.Object] = []
    candidates: list[bpy.types.Object] = []

    for obj in meshes:
        lo, hi = _world_bounds(obj)
        ext = _extent(lo, hi)
        if max(ext) < DEGENERATE or _is_track(ext.x, ext.y, ext.z):
            trash.append(obj)
            continue
        candidates.append(obj)
        all_lo = Vector((min(all_lo.x, lo.x), min(all_lo.y, lo.y), min(all_lo.z, lo.z)))
        all_hi = Vector((max(all_hi.x, hi.x), max(all_hi.y, hi.y), max(all_hi.z, hi.z)))

    print(f"[prep] 去掉铁轨/退化 {len(trash)}，剩余 {len(candidates)}")
    _delete_objects(trash)

    overall = _extent(all_lo, all_hi)
    axis = _long_axis(overall)
    print(f"[prep] 整体尺寸 {tuple(round(v, 4) for v in overall)} 长轴={axis}")

    parts: list[tuple[bpy.types.Object, float, float]] = []
    for obj in _mesh_objects():
        lo, hi = _world_bounds(obj)
        ext = _extent(lo, hi)
        parts.append((obj, _center_on_axis(lo, hi, axis), _length_on_axis(ext, axis)))

    bodies = [p for p in parts if p[2] > BODY_LEN]
    if not bodies:
        bodies = sorted(parts, key=lambda p: -p[2])[: max(1, len(parts) // 6)]
    bodies.sort(key=lambda p: p[1])
    seeds: list[float] = []
    for _, center, _ in bodies:
        if not seeds or center - seeds[-1] > SEED_GAP:
            seeds.append(center)
        else:
            seeds[-1] = 0.5 * (seeds[-1] + center)
    print(f"[prep] 车厢种子 {len(seeds)}: {[round(s, 3) for s in seeds]}")

    buckets: list[list[bpy.types.Object]] = [[] for _ in seeds]
    for obj, center, _ in parts:
        nearest = min(range(len(seeds)), key=lambda i: abs(center - seeds[i]))
        buckets[nearest].append(obj)

    cars: list[bpy.types.Object] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        cars.append(_join_objects(bucket, f"_raw_{index + 1:02d}"))

    remap = _blender_zup_remap(axis)
    for car in cars:
        car.matrix_world = remap @ car.matrix_world
        _apply_transform(car)

    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for car in cars:
        clo, chi = _world_bounds(car)
        lo = Vector((min(lo.x, clo.x), min(lo.y, clo.y), min(lo.z, clo.z)))
        hi = Vector((max(hi.x, chi.x), max(hi.y, chi.y), max(hi.z, chi.z)))
    height = hi.z - lo.z
    if height <= 1e-6:
        print("[错误] 转正后没有高度", file=sys.stderr)
        return 3
    uniform = TARGET_HEIGHT / height
    print(f"[prep] 高度 {height:.4f} → 缩尺 {uniform:.3f}")

    bpy.ops.object.select_all(action="DESELECT")
    for car in cars:
        car.select_set(True)
    bpy.context.view_layer.objects.active = cars[0]
    bpy.ops.transform.resize(value=(uniform, uniform, uniform), center_override=(0, 0, 0))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    lengths = []
    widths = []
    for car in cars:
        clo, chi = _world_bounds(car)
        lengths.append(chi.x - clo.x)
        widths.append(chi.y - clo.y)
    lengths.sort()
    widths.sort()
    median_len = lengths[len(lengths) // 2]
    median_w = widths[len(widths) // 2]
    sx = (25.0 / median_len) if median_len > 1e-3 else 1.0
    sy = (3.0 / median_w) if median_w > 1e-3 else 1.0
    if abs(sx - 1.0) > 0.05 or abs(sy - 1.0) > 0.05:
        print(f"[prep] 断面再调 X×{sx:.3f} Y×{sy:.3f}")
        bpy.ops.object.select_all(action="DESELECT")
        for car in cars:
            car.select_set(True)
        bpy.context.view_layer.objects.active = cars[0]
        bpy.ops.transform.resize(value=(sx, sy, 1.0), center_override=(0, 0, 0))
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for car in cars:
        clo, chi = _world_bounds(car)
        lo = Vector((min(lo.x, clo.x), min(lo.y, clo.y), min(lo.z, clo.z)))
        hi = Vector((max(hi.x, chi.x), max(hi.y, chi.y), max(hi.z, chi.z)))
    shift = Vector((-0.5 * (lo.x + hi.x), -0.5 * (lo.y + hi.y), -lo.z))
    for car in cars:
        car.location += shift
        _apply_transform(car)

    ranked: list[tuple[float, bpy.types.Object]] = []
    for car in cars:
        clo, chi = _world_bounds(car)
        ranked.append((0.5 * (clo.x + chi.x), car))
    ranked.sort(key=lambda t: t[0])

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    root = bpy.context.view_layer.objects.active
    root.name = "CR400BF_HuangSiDai"

    for index, (cx, car) in enumerate(ranked):
        clo, chi = _world_bounds(car)
        mid = 0.5 * (clo + chi)
        tip_band = 1.2
        if car.type == "MESH":
            xs = [(car.matrix_world @ v.co).x for v in car.data.vertices]
            lo_x, hi_x = min(xs), max(xs)
            ys_left, ys_right = [], []
            for vert in car.data.vertices:
                world = car.matrix_world @ vert.co
                if world.x < lo_x + tip_band:
                    ys_left.append(abs(world.y))
                if world.x > hi_x - tip_band:
                    ys_right.append(abs(world.y))
            left_w = max(ys_left) if ys_left else 1.0
            right_w = max(ys_right) if ys_right else 1.0
            if left_w < right_w * 0.72:
                pivot = Matrix.Translation(mid)
                rot = Matrix.Rotation(math.radians(180.0), 4, "Z")
                car.matrix_world = pivot @ rot @ pivot.inverted() @ car.matrix_world
                _apply_transform(car)
                clo, chi = _world_bounds(car)
                mid = 0.5 * (clo + chi)
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=mid)
        empty = bpy.context.view_layer.objects.active
        empty.name = f"Car{index + 1:02d}"
        car.parent = None
        car.parent = empty
        car.matrix_parent_inverse = empty.matrix_world.inverted()
        empty.parent = root
        print(
            f"[prep] {empty.name}: 长={chi.x - clo.x:.2f} "
            f"高={chi.z - clo.z:.2f} 宽={chi.y - clo.y:.2f} "
            f"中心X={mid.x:.2f}"
        )

    _lift_dark_materials()

    bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    for obj in bpy.context.scene.objects:
        if obj.name.startswith("Car") or obj.name.startswith("_raw_"):
            obj.select_set(True)
            for child in obj.children_recursive:
                child.select_set(True)
    bpy.context.view_layer.objects.active = root

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(OUT),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
    )
    print(f"[prep] 已写出 {OUT}  ({OUT.stat().st_size} 字节)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
