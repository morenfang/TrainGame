"""把 GWR Hall 机车 Car01 绕 Z 转 180°，鼻锥朝 +X。"""

from __future__ import annotations

from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "models" / "GWR_HallClass_19car.glb"


def _world_bounds(obj: bpy.types.Object):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def _tree_bounds(root: bpy.types.Object):
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    found = False
    for obj in [root, *root.children_recursive]:
        if obj.type != "MESH":
            continue
        clo, chi = _world_bounds(obj)
        found = True
        lo = Vector((min(lo.x, clo.x), min(lo.y, clo.y), min(lo.z, clo.z)))
        hi = Vector((max(hi.x, chi.x), max(hi.y, chi.y), max(hi.z, chi.z)))
    return (lo, hi) if found else None


def main() -> int:
    if not PATH.exists():
        print(f"[flip] 缺少 {PATH}")
        return 2
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(PATH))
    car01 = bpy.data.objects.get("Car01")
    if car01 is None:
        print("[flip] 没有 Car01")
        return 1
    before = _tree_bounds(car01)
    meshes = [o for o in car01.children_recursive if o.type == "MESH"]
    if not meshes or before is None:
        print("[flip] Car01 空")
        return 1
    mid = 0.5 * (before[0] + before[1])
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.transform.rotate(
        value=3.141592653589793,
        orient_axis="Z",
        orient_type="GLOBAL",
        center_override=(mid.x, mid.y, mid.z),
    )
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    after = _tree_bounds(car01)
    print(f"[flip] Car01 已掉头  中心 {mid.x:.2f} → "
          f"{0.5 * (after[0].x + after[1].x):.2f}")

    bpy.ops.object.select_all(action="DESELECT")
    root = car01.parent
    if root is not None:
        root.select_set(True)
        bpy.context.view_layer.objects.active = root
    for obj in bpy.context.scene.objects:
        if obj.name.startswith("Car"):
            obj.select_set(True)
            for child in obj.children_recursive:
                child.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(PATH), export_format="GLB", use_selection=True,
        export_apply=True, export_yup=True,
        export_image_format="AUTO",
        export_materials="EXPORT",
        export_texcoords=True,
        export_normals=True,
    )
    print(f"[flip] 写回 {PATH.name} ({PATH.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
