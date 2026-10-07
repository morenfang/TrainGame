"""把垃圾桶 / GWR Hall 车长加长一倍，并重做 Hall 黑色棚车造型。

源 glb 已清理，直接改现成编组文件::

    blender --background --python scripts/refine_new_trains.py
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LAJI = MODELS / "CR400BF_LaJiTong_8car.glb"
GWR = MODELS / "GWR_HallClass_19car.glb"

N_WAGONS = 18
WAGON_LEN = 24.0   # 原先 12 m 的两倍（真实米制；游戏再 ×0.4）
WAGON_W = 2.85
WAGON_H = 3.25
GAP = 0.70
STRETCH_X = 2.0


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for col in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.armatures):
        for item in list(col):
            col.remove(item)


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


def _apply(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _stretch_meshes_x(meshes: list[bpy.types.Object], factor: float) -> None:
    if not meshes:
        return
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for obj in meshes:
        clo, chi = _world_bounds(obj)
        lo = Vector((min(lo.x, clo.x), min(lo.y, clo.y), min(lo.z, clo.z)))
        hi = Vector((max(hi.x, chi.x), max(hi.y, chi.y), max(hi.z, chi.z)))
    cx = 0.5 * (lo.x + hi.x)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.transform.resize(
        value=(factor, 1.0, 1.0),
        center_override=(cx, 0.0, 0.0),
        orient_type="GLOBAL",
    )
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def _mat(name: str, color, roughness=0.7, metallic=0.05) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    return mat


def _assign(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def _cube(name, loc, scale, mat) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.scale = scale
    _apply(obj)
    _assign(obj, mat)
    return obj


def _cyl(name, loc, radius, depth, rot, mat, verts=12) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=verts, radius=radius, depth=depth, location=loc, rotation=rot,
    )
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    _apply(obj)
    _assign(obj, mat)
    return obj


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


def _make_boxcar(x_center: float) -> bpy.types.Object:
    """英式黑色棚车：厢体、弧顶、中门、侧筋、底架、转向架、缓冲器。"""
    body_m = _mat("FreightBody", (0.05, 0.05, 0.055), 0.78, 0.04)
    roof_m = _mat("FreightRoof", (0.09, 0.09, 0.10), 0.55, 0.12)
    frame_m = _mat("FreightFrame", (0.03, 0.03, 0.035), 0.65, 0.18)
    steel_m = _mat("FreightSteel", (0.12, 0.12, 0.13), 0.42, 0.55)
    door_m = _mat("FreightDoor", (0.07, 0.07, 0.075), 0.7, 0.08)

    L, W, H = WAGON_LEN, WAGON_W, WAGON_H
    floor_z = 1.05
    body_h = H - 0.55
    body_z = floor_z + body_h * 0.5
    parts: list[bpy.types.Object] = []

    # 厢体
    parts.append(_cube("body", (x_center, 0.0, body_z), (L * 0.92, W, body_h), body_m))

    # 弧顶：中脊 + 两侧坡板
    ridge_z = floor_z + body_h + 0.18
    parts.append(_cube("roof_ridge", (x_center, 0.0, ridge_z),
                       (L * 0.90, W * 0.22, 0.16), roof_m))
    for sy, ang_y in ((W * 0.28, 0.22), (-W * 0.28, -0.22)):
        bpy.ops.mesh.primitive_cube_add(
            size=1.0, location=(x_center, sy, ridge_z - 0.04),
        )
        panel = bpy.context.view_layer.objects.active
        panel.scale = (L * 0.90, W * 0.38, 0.10)
        panel.rotation_euler = (ang_y, 0.0, 0.0)
        _apply(panel)
        _assign(panel, roof_m)
        parts.append(panel)

    # 侧筋（竖向木条感）
    n_ribs = 9
    span = L * 0.84
    for i in range(n_ribs):
        t = i / (n_ribs - 1)
        x = x_center - span * 0.5 + t * span
        # 中门区域跳过
        if abs(x - x_center) < L * 0.08:
            continue
        for sy in (-W * 0.505, W * 0.505):
            parts.append(_cube(
                f"rib_{i}_{sy:+.1f}",
                (x, sy, body_z),
                (0.08, 0.06, body_h * 0.92),
                steel_m,
            ))

    # 中门（两侧各一扇滑动门）
    for sy in (-W * 0.52, W * 0.52):
        parts.append(_cube(
            f"door_{sy:+.1f}",
            (x_center, sy, body_z),
            (L * 0.16, 0.08, body_h * 0.88),
            door_m,
        ))
        # 门框
        for dx in (-L * 0.09, L * 0.09):
            parts.append(_cube(
                f"doorframe_{dx:+.2f}",
                (x_center + dx, sy, body_z),
                (0.07, 0.09, body_h * 0.92),
                steel_m,
            ))

    # 端墙交叉撑
    for sx in (-L * 0.455, L * 0.455):
        parts.append(_cube(f"end_{sx:+.1f}", (x_center + sx, 0.0, body_z),
                           (0.10, W * 0.96, body_h * 0.96), body_m))
        for sign in (-1, 1):
            bpy.ops.mesh.primitive_cube_add(
                size=1.0,
                location=(x_center + sx * 1.01, 0.0, body_z),
            )
            brace = bpy.context.view_layer.objects.active
            brace.scale = (0.05, W * 0.90, 0.10)
            brace.rotation_euler = (0.0, 0.0, sign * 0.55)
            _apply(brace)
            _assign(brace, steel_m)
            parts.append(brace)

    # 底架
    parts.append(_cube("solebar", (x_center, 0.0, floor_z - 0.12),
                       (L * 0.96, W * 0.72, 0.28), frame_m))
    for sy in (-W * 0.38, W * 0.38):
        parts.append(_cube(f"side_sill_{sy:+.1f}", (x_center, sy, floor_z - 0.05),
                           (L * 0.94, 0.10, 0.18), frame_m))

    # 两台转向架
    for bx in (-L * 0.30, L * 0.30):
        bogie_x = x_center + bx
        parts.append(_cube(f"bogie_{bx:+.1f}", (bogie_x, 0.0, 0.55),
                           (2.4, W * 0.70, 0.22), frame_m))
        for ax in (-0.72, 0.72):
            for sy in (-0.82, 0.82):
                parts.append(_cyl(
                    f"wheel_{bx:+.1f}_{ax:+.1f}_{sy:+.1f}",
                    (bogie_x + ax, sy, 0.42),
                    0.42, 0.20, (math.pi * 0.5, 0.0, 0.0), steel_m, verts=14,
                ))
            parts.append(_cyl(
                f"axle_{bx:+.1f}_{ax:+.1f}",
                (bogie_x + ax, 0.0, 0.42),
                0.08, W * 0.78, (math.pi * 0.5, 0.0, 0.0), steel_m, verts=8,
            ))

    # 缓冲器 + 车钩
    for sx in (-1.0, 1.0):
        tip = x_center + sx * L * 0.50
        for sy in (-0.55, 0.55):
            parts.append(_cyl(
                f"buffer_{sx:+.0f}_{sy:+.1f}",
                (tip, sy, floor_z + 0.15),
                0.12, 0.28, (0.0, math.pi * 0.5, 0.0), steel_m, verts=10,
            ))
        parts.append(_cube(f"coupler_{sx:+.0f}", (tip + sx * 0.18, 0.0, floor_z + 0.05),
                           (0.35, 0.18, 0.18), steel_m))

    # 脚蹬
    for sx in (-L * 0.38, L * 0.38):
        for sy in (-W * 0.55, W * 0.55):
            parts.append(_cube(f"step_{sx:+.1f}_{sy:+.1f}",
                               (x_center + sx, sy, floor_z - 0.35),
                               (0.35, 0.18, 0.08), steel_m))

    return _join(parts, f"_wagon_{x_center:.1f}")


def _parent_car(meshes: list[bpy.types.Object], name: str,
                root: bpy.types.Object) -> None:
    bb = None
    for mesh in meshes:
        clo, chi = _world_bounds(mesh)
        if bb is None:
            bb = (clo.copy(), chi.copy())
        else:
            lo, hi = bb
            bb = (
                Vector((min(lo.x, clo.x), min(lo.y, clo.y), min(lo.z, clo.z))),
                Vector((max(hi.x, chi.x), max(hi.y, chi.y), max(hi.z, chi.z))),
            )
    lo, hi = bb
    mid = 0.5 * (lo + hi)
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=mid)
    empty = bpy.context.view_layer.objects.active
    empty.name = name
    for mesh in meshes:
        mesh.parent = None
        mesh.parent = empty
        mesh.matrix_parent_inverse = empty.matrix_world.inverted()
    empty.parent = root
    print(f"[refine] {name}: 长={hi.x - lo.x:.2f} 高={hi.z - lo.z:.2f} "
          f"宽={hi.y - lo.y:.2f} 中心X={mid.x:.2f}")


def _export(path: Path, root: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    for obj in bpy.context.scene.objects:
        if obj.name.startswith("Car") or obj.name.startswith("_"):
            obj.select_set(True)
            for child in obj.children_recursive:
                child.select_set(True)
    bpy.context.view_layer.objects.active = root
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB", use_selection=True,
        export_apply=True, export_yup=True,
    )
    print(f"[refine] 写出 {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def refine_lajitong() -> int:
    if not LAJI.exists():
        print(f"[refine] 缺少 {LAJI}")
        return 2
    print(f"[refine] 垃圾桶车长 ×{STRETCH_X}")
    _clear()
    bpy.ops.import_scene.gltf(filepath=str(LAJI))
    cars = sorted(
        (o for o in bpy.data.objects if o.name.startswith("Car") and o.name[3:].isdigit()),
        key=lambda o: o.name,
    )
    if len(cars) < 2:
        print("[refine] 垃圾桶分不出车厢")
        return 3
    for car in cars:
        meshes = [o for o in car.children_recursive if o.type == "MESH"]
        before = _tree_bounds(car)
        _stretch_meshes_x(meshes, STRETCH_X)
        after = _tree_bounds(car)
        if before and after:
            print(f"[refine] {car.name}: {before[1].x - before[0].x:.2f} → "
                  f"{after[1].x - after[0].x:.2f} m")
    root = next((o for o in bpy.data.objects
                 if o.type == "EMPTY" and o.parent is None
                 and any(c.name.startswith("Car") for c in o.children)), None)
    if root is None:
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
        root = bpy.context.view_layer.objects.active
        root.name = "CR400BF_LaJiTong"
        for car in cars:
            car.parent = root
    _export(LAJI, root)
    return 0


def refine_gwr() -> int:
    if not GWR.exists():
        print(f"[refine] 缺少 {GWR}")
        return 2
    print(f"[refine] Hall 机车 ×{STRETCH_X}，重做 {N_WAGONS} 节棚车")
    _clear()
    bpy.ops.import_scene.gltf(filepath=str(GWR))

    car01 = bpy.data.objects.get("Car01")
    if car01 is None:
        print("[refine] 没有 Car01")
        return 3

    # 删掉旧方盒货车
    for obj in list(bpy.data.objects):
        if obj.name.startswith("Car") and obj.name != "Car01":
            for child in list(obj.children_recursive):
                bpy.data.objects.remove(child, do_unlink=True)
            bpy.data.objects.remove(obj, do_unlink=True)

    loco_meshes = [o for o in car01.children_recursive if o.type == "MESH"]
    before = _tree_bounds(car01)
    _stretch_meshes_x(loco_meshes, STRETCH_X)
    after = _tree_bounds(car01)
    if before and after:
        print(f"[refine] Car01: {before[1].x - before[0].x:.2f} → "
              f"{after[1].x - after[0].x:.2f} m")

    root = car01.parent
    if root is None:
        bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
        root = bpy.context.view_layer.objects.active
        root.name = "GWR_HallClass"
        car01.parent = root

    lo, hi = _tree_bounds(car01)
    cursor = hi.x + GAP + WAGON_LEN * 0.5
    for n in range(N_WAGONS):
        wagon = _make_boxcar(cursor)
        _parent_car([wagon], f"Car{n + 2:02d}", root)
        cursor += WAGON_LEN + GAP

    _export(GWR, root)
    return 0


def main() -> int:
    code = refine_lajitong()
    if code:
        return code
    return refine_gwr()


if __name__ == "__main__":
    raise SystemExit(main())
