"""把垃圾桶受电弓材质（纯红 Standardmaterial.017）改成灰色。"""
from __future__ import annotations

from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "models" / "CR400BF_LaJiTong_8car.glb"
GRAY = (0.42, 0.43, 0.45, 1.0)


def main() -> int:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(PATH))
    hits = 0
    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        r, g, b, a = bsdf.inputs["Base Color"].default_value
        # 纯红 / 高饱和红 → 电弧
        if r > 0.85 and g < 0.15 and b < 0.15:
            bsdf.inputs["Base Color"].default_value = GRAY
            bsdf.inputs["Roughness"].default_value = 0.48
            bsdf.inputs["Metallic"].default_value = 0.55
            hits += 1
            print(f"[gray] {mat.name}: ({r:.2f},{g:.2f},{b:.2f}) → gray")
    if hits == 0:
        print("[gray] 没找到红色电弧材质")
        return 1
    bpy.ops.object.select_all(action="DESELECT")
    root = next(
        (o for o in bpy.data.objects
         if o.type == "EMPTY" and o.parent is None
         and any(c.name.startswith("Car") for c in o.children)),
        None,
    )
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
    )
    print(f"[gray] 写回 {PATH.name}，改了 {hits} 个材质")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
