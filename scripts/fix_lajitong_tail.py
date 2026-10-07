"""把垃圾桶 Car03 换成头车副本，8 节编组两端都是驾驶室。"""

from __future__ import annotations

from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "models" / "CR400BF_LaJiTong_8car.glb"


def _copy(obj: bpy.types.Object, parent: bpy.types.Object | None) -> bpy.types.Object:
    dup = obj.copy()
    if obj.data is not None:
        dup.data = obj.data.copy()
    bpy.context.collection.objects.link(dup)
    dup.parent = parent
    dup.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
    dup.location = obj.location.copy()
    dup.rotation_euler = obj.rotation_euler.copy()
    dup.scale = obj.scale.copy()
    return dup


def main() -> int:
    if not PATH.exists():
        print(f"[lajitong-tail] 缺少 {PATH}")
        return 2
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(PATH))
    car01 = bpy.data.objects.get("Car01")
    car03 = bpy.data.objects.get("Car03")
    if car01 is None:
        print("[lajitong-tail] 没有 Car01")
        return 1
    if car03 is None:
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
        car03 = bpy.context.view_layer.objects.active
        car03.name = "Car03"
        car03.parent = car01.parent
    for child in list(car03.children_recursive):
        bpy.data.objects.remove(child, do_unlink=True)
    stack = [(c, car03) for c in car01.children]
    while stack:
        src, parent = stack.pop()
        dup = _copy(src, parent)
        stack.extend((c, dup) for c in src.children)
    names = sorted(o.name for o in bpy.data.objects if o.name.startswith("Car"))
    print(f"[lajitong-tail] empties={names}  Car03 子网格="
          f"{sum(1 for o in car03.children_recursive if o.type == 'MESH')}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.name.startswith("Car") or obj == car01.parent:
            obj.select_set(True)
            for child in obj.children_recursive:
                child.select_set(True)
    if car01.parent:
        bpy.context.view_layer.objects.active = car01.parent
        car01.parent.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(PATH), export_format="GLB", use_selection=True,
        export_apply=True, export_yup=True,
    )
    print(f"[lajitong-tail] 写回 {PATH} ({PATH.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
