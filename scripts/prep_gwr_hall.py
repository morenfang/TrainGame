"""GWR 4900 Hall Class：机车 + 煤水车 + 18 节黑色货车。

用法（在已开的 Blender 里 exec，或）::

    blender --background --python scripts/prep_gwr_hall.py
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "gwr-4900-hall-class-express.glb"
OUT = ROOT / "models" / "GWR_HallClass_19car.glb"
TARGET_HEIGHT = 3.5 * 1.15
N_WAGONS = 18
WAGON_LEN = 12.0
WAGON_W = 2.75
WAGON_H = 3.15
GAP = 0.55


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for col in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.armatures):
        for item in list(col):
            col.remove(item)


def _world_bounds(obj: bpy.types.Object):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs, ys, zs = [c.x for c in corners], [c.y for c in corners], [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def _apply(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _join(objs: list[bpy.types.Object], name: str) -> bpy.types.Object:
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


def _black_mat() -> bpy.types.Material:
    mat = bpy.data.materials.get("FreightBlack")
    if mat is None:
        mat = bpy.data.materials.new("FreightBlack")
        mat.use_nodes = True
        principled = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        principled.inputs["Base Color"].default_value = (0.04, 0.04, 0.045, 1.0)
        principled.inputs["Roughness"].default_value = 0.72
        principled.inputs["Metallic"].default_value = 0.08
    return mat


def _make_wagon(x_center: float) -> bpy.types.Object:
    mat = _black_mat()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_center, 0.0, 1.05 + WAGON_H * 0.5))
    body = bpy.context.view_layer.objects.active
    body.scale = (WAGON_LEN * 0.92, WAGON_W, WAGON_H)
    _apply(body)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_center, 0.0, 0.78))
    frame = bpy.context.view_layer.objects.active
    frame.scale = (WAGON_LEN * 0.98, WAGON_W * 0.78, 0.28)
    _apply(frame)
    wheels = []
    for sx in (-WAGON_LEN * 0.32, WAGON_LEN * 0.32):
        for sy in (-0.78, 0.78):
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=10, radius=0.46, depth=0.22,
                location=(x_center + sx, sy, 0.46),
                rotation=(math.pi * 0.5, 0.0, 0.0),
            )
            wheels.append(bpy.context.view_layer.objects.active)
    parts = [body, frame, *wheels]
    for obj in parts:
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
    return _join(parts, f"_wagon_{x_center:.1f}")


def _parent_car(meshes: list[bpy.types.Object], name: str,
                root: bpy.types.Object) -> None:
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for mesh in meshes:
        clo, chi = _world_bounds(mesh)
        lo = Vector((min(lo.x, clo.x), min(lo.y, clo.y), min(lo.z, clo.z)))
        hi = Vector((max(hi.x, chi.x), max(hi.y, chi.y), max(hi.z, chi.z)))
    mid = 0.5 * (lo + hi)
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=mid)
    empty = bpy.context.view_layer.objects.active
    empty.name = name
    for mesh in meshes:
        mesh.parent = None
        mesh.parent = empty
        mesh.matrix_parent_inverse = empty.matrix_world.inverted()
    empty.parent = root
    print(f"[gwr] {name}: 长={hi.x - lo.x:.2f} 高={hi.z - lo.z:.2f} "
          f"宽={hi.y - lo.y:.2f} 中心X={mid.x:.2f} 网格={len(meshes)}")


def main() -> int:
    if not SRC.exists():
        print(f"[gwr] 找不到 {SRC}")
        return 2
    print(f"[gwr] 导入 {SRC}")
    _clear()
    bpy.ops.import_scene.gltf(filepath=str(SRC))

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    # 源是厘米：先缩到米
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.transform.resize(value=(0.01, 0.01, 0.01), center_override=(0, 0, 0))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # 长轴在 Y → X
    remap = Matrix((
        (0, 1, 0, 0),
        (-1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ))
    for obj in list(bpy.context.scene.objects):
        if obj.type == "MESH":
            obj.matrix_world = remap @ obj.matrix_world
            _apply(obj)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for obj in meshes:
        clo, chi = _world_bounds(obj)
        lo = Vector((min(lo.x, clo.x), min(lo.y, clo.y), min(lo.z, clo.z)))
        hi = Vector((max(hi.x, chi.x), max(hi.y, chi.y), max(hi.z, chi.z)))
    height = hi.z - lo.z
    uniform = TARGET_HEIGHT / height if height > 1e-6 else 1.0
    print(f"[gwr] 高度 {height:.3f} → 缩尺 {uniform:.3f}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.transform.resize(value=(uniform, uniform, uniform), center_override=(0, 0, 0))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for obj in meshes:
        clo, chi = _world_bounds(obj)
        lo = Vector((min(lo.x, clo.x), min(lo.y, clo.y), min(lo.z, clo.z)))
        hi = Vector((max(hi.x, chi.x), max(hi.y, chi.y), max(hi.z, chi.z)))
    shift = Vector((-lo.x, -0.5 * (lo.y + hi.y), -lo.z))
    for obj in meshes:
        obj.location += shift
        _apply(obj)

    print(f"[gwr] 机车+煤水车网格 {len(meshes)}（整机作 Car01，再挂 18 节货车）")

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    root = bpy.context.view_layer.objects.active
    root.name = "GWR_HallClass"

    _parent_car(meshes, "Car01", root)
    lo, hi = Vector((1e9, 1e9, 1e9)), Vector((-1e9, -1e9, -1e9))
    for obj in meshes:
        clo, chi = _world_bounds(obj)
        lo = Vector((min(lo.x, clo.x), min(lo.y, clo.y), min(lo.z, clo.z)))
        hi = Vector((max(hi.x, chi.x), max(hi.y, chi.y), max(hi.z, chi.z)))
    cursor = hi.x + GAP + WAGON_LEN * 0.5

    for n in range(N_WAGONS):
        wagon = _make_wagon(cursor)
        _parent_car([wagon], f"Car{n + 2:02d}", root)
        cursor += WAGON_LEN + GAP

    bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    for obj in bpy.context.scene.objects:
        if obj.name.startswith("Car") or obj.name.startswith("_"):
            obj.select_set(True)
            for child in obj.children_recursive:
                child.select_set(True)
    bpy.context.view_layer.objects.active = root
    OUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(OUT), export_format="GLB", use_selection=True,
        export_apply=True, export_yup=True,
    )
    print(f"[gwr] 已写出 {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
