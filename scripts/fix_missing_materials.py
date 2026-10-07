"""给列车 glb 里缺材质的网格补上默认材质，消掉 panda3d-gltf 的警告。

    blender --background --python scripts/fix_missing_materials.py -- \\
        models/CR400BF_LaJiTong_8car.glb
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DEFAULT = MODELS / "CR400BF_LaJiTong_8car.glb"
FALLBACK_COLOR = (0.91, 0.94, 0.90, 1.0)  # 垃圾桶浅绿白车身


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for col in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.armatures):
        for item in list(col):
            col.remove(item)


def _fallback_mat() -> bpy.types.Material:
    name = "LaJiTong_Fallback"
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = FALLBACK_COLOR
    bsdf.inputs["Roughness"].default_value = 0.45
    bsdf.inputs["Metallic"].default_value = 0.05
    return mat


def _has_usable_material(obj: bpy.types.Object) -> bool:
    if not obj.data or not getattr(obj.data, "materials", None):
        return False
    slots = list(obj.data.materials)
    if not slots:
        return False
    return any(m is not None for m in slots)


def _pick_donor(meshes: list[bpy.types.Object]) -> bpy.types.Material | None:
    """优先借用同场景里已有的车身材质，颜色更贴原模。"""
    for obj in meshes:
        if not obj.data or not obj.data.materials:
            continue
        for mat in obj.data.materials:
            if mat is None:
                continue
            # 跳过明显是玻璃 / 黑色底架的名字
            low = mat.name.lower()
            if any(k in low for k in ("glass", "window", "black", "track", "rail")):
                continue
            return mat
    return None


def fix(path: Path) -> int:
    if not path.exists():
        print(f"[mat] 缺少 {path}")
        return 2
    print(f"[mat] 检查 {path.name}")
    _clear()
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    donor = _pick_donor(meshes) or _fallback_mat()
    print(f"[mat] 缺省材质 → {donor.name}")

    fixed = 0
    for obj in meshes:
        if _has_usable_material(obj):
            # 槽位里有 None 的也补上
            dirty = False
            for i, mat in enumerate(list(obj.data.materials)):
                if mat is None:
                    obj.data.materials[i] = donor
                    dirty = True
            if dirty:
                fixed += 1
            continue
        if obj.data.materials:
            obj.data.materials[0] = donor
        else:
            obj.data.materials.append(donor)
        # 确保每个 polygon 的 material_index 落在有效槽
        for poly in obj.data.polygons:
            if poly.material_index >= len(obj.data.materials):
                poly.material_index = 0
        fixed += 1
        print(f"[mat]   + {obj.name}")

    if fixed == 0:
        print("[mat] 没有缺材质的网格")
        return 0

    bpy.ops.object.select_all(action="DESELECT")
    root = next(
        (o for o in bpy.data.objects
         if o.type == "EMPTY" and o.parent is None
         and any(c.name.startswith("Car") for c in o.children)),
        None,
    )
    if root is not None:
        root.select_set(True)
    for obj in bpy.context.scene.objects:
        if obj.name.startswith("Car") or (
            obj.type == "MESH" and (root is None or obj in root.children_recursive
                                    or (obj.parent and obj.parent.name.startswith("Car")))
        ):
            obj.select_set(True)
            for child in getattr(obj, "children_recursive", []):
                child.select_set(True)
    if root is None:
        bpy.ops.object.select_all(action="SELECT")
    else:
        bpy.context.view_layer.objects.active = root

    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB", use_selection=True,
        export_apply=True, export_yup=True,
    )
    print(f"[mat] 写回 {path.name}：补了 {fixed} 个网格，"
          f"{path.stat().st_size / 1e6:.1f} MB")
    return 0


def main() -> int:
    argv = list(sys.argv)
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    targets = [Path(a) for a in args] if args else [DEFAULT]
    code = 0
    for arg in targets:
        path = arg if arg.is_absolute() else (
            MODELS / arg.name if arg.parent == Path(".") else ROOT / arg
        )
        code = fix(path.resolve()) or code
    return code


if __name__ == "__main__":
    raise SystemExit(main())
