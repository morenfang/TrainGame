"""批量把现有 Car01… 编组 glb 洗成游戏约定（可重复跑，幂等）。

约定（与 ``render.gltf_train`` 一致）
--------------------------------------------------------------------
- 每节车局部：鼻锥在 +X；尾车由游戏再转 180°。
- 车体高度对齐 ``TARGET_BODY_HEIGHT``（约原 3.5 m × 1.15）。
- Blender Z-up，导出 ``export_yup=True``。

用法::

    blender --background --python scripts/prep_fleet_glb.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TARGET_BODY_HEIGHT = 3.5 * 1.15
TARGET_WIDTH = 3.05
DEFAULT_TARGETS = (
    "CR400BF_HuangSiDai_6car.glb",
)


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                  bpy.data.armatures):
        for item in list(block):
            block.remove(item)


def _car_empties() -> list[bpy.types.Object]:
    cars = [
        o for o in bpy.data.objects
        if o.name.startswith("Car") and len(o.name) == 5 and o.name[3:].isdigit()
    ]
    return sorted(cars, key=lambda o: o.name)


def _meshes_under(obj: bpy.types.Object, *, skip_panto: bool = True) -> list[bpy.types.Object]:
    out = []
    for child in obj.children_recursive:
        if child.type != "MESH":
            continue
        if skip_panto and ("Panto" in child.name or child.name.startswith("GW")):
            continue
        out.append(child)
    return out


def _local_verts(empty: bpy.types.Object) -> list[Vector]:
    inv = empty.matrix_world.inverted()
    verts: list[Vector] = []
    for mesh in _meshes_under(empty):
        for vert in mesh.data.vertices:
            verts.append(inv @ (mesh.matrix_world @ vert.co))
    return verts


def _tip_half_width(verts: list[Vector], side: int, band: float = 1.2) -> float:
    xs = [v.x for v in verts]
    lo, hi = min(xs), max(xs)
    if side > 0:
        sel = [v for v in verts if v.x > hi - band]
    else:
        sel = [v for v in verts if v.x < lo + band]
    if not sel:
        return 0.0
    return max(abs(v.y) for v in sel)


def _nose_on_plus_x(empty: bpy.types.Object) -> bool | None:
    verts = _local_verts(empty)
    if len(verts) < 32:
        return None
    left = _tip_half_width(verts, -1)
    right = _tip_half_width(verts, +1)
    if left <= 1e-6 or right <= 1e-6:
        return None
    ratio = left / right
    if ratio < 0.72:
        return False
    if ratio > 1.0 / 0.72:
        return True
    return None


def _world_bounds(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector] | None:
    xs, ys, zs = [], [], []
    for mesh in meshes:
        for corner in mesh.bound_box:
            world = mesh.matrix_world @ Vector(corner)
            xs.append(world.x)
            ys.append(world.y)
            zs.append(world.z)
    if not xs:
        return None
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def _rotate_car_180_z(empty: bpy.types.Object) -> None:
    """在 Empty 局部把子树绕 Z 转 180°（改 location / mesh 顶点）。"""
    rot = Matrix.Rotation(math.radians(180.0), 4, "Z")
    for obj in empty.children_recursive:
        obj.location = rot @ obj.location
        obj.rotation_euler.rotate(rot)
        if obj.type == "MESH" and obj.data is not None:
            obj.data.transform(rot)
            obj.data.update()
    bpy.context.view_layer.update()


def _lift_materials(floor: float = 0.42) -> None:
    for mat in bpy.data.materials:
        nt = getattr(mat, "node_tree", None)
        if nt is None:
            continue
        principled = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if principled is None:
            continue
        base = principled.inputs["Base Color"]
        if not base.is_linked:
            color = list(base.default_value)
            peak = max(color[0], color[1], color[2], 1e-6)
            if peak < floor:
                scale = floor / peak
                base.default_value = (
                    min(1.0, color[0] * scale),
                    min(1.0, color[1] * scale),
                    min(1.0, color[2] * scale),
                    color[3],
                )
        rough = principled.inputs.get("Roughness")
        if rough is not None and not rough.is_linked and rough.default_value > 0.65:
            rough.default_value = 0.48


def _body_extent(cars: list[bpy.types.Object]) -> tuple[float, float, float]:
    """用各节车体（不含受电弓）的最大高/宽、中位长，避免矮中间车带偏。"""
    lengths, widths, heights = [], [], []
    for car in cars:
        meshes = _meshes_under(car, skip_panto=True)
        bounds = _world_bounds(meshes)
        if bounds is None:
            continue
        lo, hi = bounds
        lengths.append(hi.x - lo.x)
        widths.append(hi.y - lo.y)
        heights.append(hi.z - lo.z)
    assert heights, "没有可测的车体网格"
    lengths.sort()
    widths.sort()
    return (
        lengths[len(lengths) // 2],
        max(widths),
        max(heights),
    )


def _scale_to_targets(cars: list[bpy.types.Object]) -> None:
    length, width, height = _body_extent(cars)
    sz = TARGET_BODY_HEIGHT / height if height > 1e-3 else 1.0
    sy = TARGET_WIDTH / width if width > 1e-3 else 1.0
    # 长向不动：LENGTH_SCALE 在游戏里负责
    print(
        f"[fleet] 车体 长={length:.2f} 宽={width:.2f} 高={height:.2f} "
        f"→ 宽×{sy:.3f} 高×{sz:.3f}"
    )
    if abs(sy - 1.0) < 0.02 and abs(sz - 1.0) < 0.02:
        print("[fleet] 比例已到位，跳过缩放")
    else:
        bpy.ops.object.select_all(action="DESELECT")
        for car in cars:
            car.select_set(True)
            for child in car.children_recursive:
                child.select_set(True)
        bpy.context.view_layer.objects.active = cars[0]
        bpy.ops.transform.resize(
            value=(1.0, sy, sz),
            center_override=(0.0, 0.0, 0.0),
        )
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # 全部网格贴 Z=0，Empty 挪回几何中心
    all_meshes: list[bpy.types.Object] = []
    for car in cars:
        all_meshes.extend(_meshes_under(car, skip_panto=False))
    bounds = _world_bounds(all_meshes)
    if bounds is None:
        return
    lo, _hi = bounds
    dz = -lo.z
    if abs(dz) > 1e-4:
        for mesh in all_meshes:
            mesh.location.z += dz
        bpy.ops.object.select_all(action="DESELECT")
        for mesh in all_meshes:
            mesh.select_set(True)
        bpy.context.view_layer.objects.active = all_meshes[0]
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    for car in cars:
        meshes = _meshes_under(car, skip_panto=False)
        b = _world_bounds(meshes)
        if b is None:
            continue
        clo, chi = b
        car.location = 0.5 * (clo + chi)


def _strip_track() -> None:
    trash = [
        o for o in bpy.data.objects
        if o.name in ("Track", "Ballast")
        or o.name.startswith(("Sleeper", "Rail", "GW", "GWFrame"))
    ]
    if not trash:
        return
    bpy.ops.object.select_all(action="DESELECT")
    for obj in trash:
        obj.select_set(True)
    bpy.ops.object.delete(use_global=False)


def _process(path: Path) -> int:
    print(f"[fleet] 处理 {path.name}")
    _clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(path))
    _strip_track()
    cars = _car_empties()
    if len(cars) < 2:
        print(f"[fleet] 跳过 {path.name}：找不到 Car01…")
        return 1

    flipped = 0
    for car in cars:
        nose = _nose_on_plus_x(car)
        if nose is False:
            _rotate_car_180_z(car)
            flipped += 1
            again = _nose_on_plus_x(car)
            print(f"[fleet]   {car.name}: −X → +X  (now={again})")
            if again is False:
                # 再转一次也修不好就报
                print(f"[fleet]   警告：{car.name} 翻转后仍不像 +X 鼻锥")
        elif nose is True:
            print(f"[fleet]   {car.name}: 鼻锥已在 +X")
        else:
            print(f"[fleet]   {car.name}: 中间车")

    _scale_to_targets(cars)
    _lift_materials()

    for car in (cars[0], cars[-1]):
        print(f"[fleet]   校验 {car.name} nose_plus_x={_nose_on_plus_x(car)}")

    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = cars[0]
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
    )
    print(f"[fleet] 已写回 {path.name}  （翻 {flipped} 节，{path.stat().st_size} 字节）")
    return 0


def main() -> int:
    argv = list(sys.argv)
    names = argv[argv.index("--") + 1 :] if "--" in argv else list(DEFAULT_TARGETS)
    failures = 0
    for name in names:
        path = MODELS / name
        if not path.exists():
            print(f"[fleet] 缺少 {path}")
            failures += 1
            continue
        try:
            failures += _process(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[fleet] 失败 {name}: {exc}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
