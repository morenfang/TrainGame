"""GWR 4900 Hall Class：从原始红机车重做（保留贴图），再挂 18 节黑色棚车。

原始目录优先::

    C:\\baidunetdiskdownload\\gwr-4900-hall-class-express\\gwr-4900-hall-class-express.glb

用法::

    blender --background --python scripts/prep_gwr_hall.py
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
SRC_CANDIDATES = (
    Path(r"C:\baidunetdiskdownload\gwr-4900-hall-class-express\gwr-4900-hall-class-express.glb"),
    ROOT / "models" / "gwr-4900-hall-class-express.glb",
)
OUT = ROOT / "models" / "GWR_HallClass_19car.glb"

TARGET_HEIGHT = 3.5 * 1.15
# 沙盘车长 ≈ 17.5 m（× LENGTH_SCALE 0.4）→ 真实米制约 43.75
TARGET_LOCO_LEN = 43.75
N_WAGONS = 18
WAGON_LEN = 24.0
WAGON_W = 2.85
WAGON_H = 3.25
GAP = 0.70


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


def _make_boxcar(x_center: float) -> bpy.types.Object:
    """英式黑色棚车（与机车贴图无关，单独程序化）。"""
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

    parts.append(_cube("body", (x_center, 0.0, body_z), (L * 0.92, W, body_h), body_m))
    ridge_z = floor_z + body_h + 0.18
    parts.append(_cube("roof_ridge", (x_center, 0.0, ridge_z),
                       (L * 0.90, W * 0.22, 0.16), roof_m))
    for sy, ang in ((W * 0.28, 0.22), (-W * 0.28, -0.22)):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_center, sy, ridge_z - 0.04))
        panel = bpy.context.view_layer.objects.active
        panel.scale = (L * 0.90, W * 0.38, 0.10)
        panel.rotation_euler = (ang, 0.0, 0.0)
        _apply(panel)
        _assign(panel, roof_m)
        parts.append(panel)

    span = L * 0.84
    for i in range(9):
        x = x_center - span * 0.5 + (i / 8) * span
        if abs(x - x_center) < L * 0.08:
            continue
        for sy in (-W * 0.505, W * 0.505):
            parts.append(_cube(f"rib_{i}", (x, sy, body_z),
                               (0.08, 0.06, body_h * 0.92), steel_m))

    for sy in (-W * 0.52, W * 0.52):
        parts.append(_cube("door", (x_center, sy, body_z),
                           (L * 0.16, 0.08, body_h * 0.88), door_m))
        for dx in (-L * 0.09, L * 0.09):
            parts.append(_cube("doorframe", (x_center + dx, sy, body_z),
                               (0.07, 0.09, body_h * 0.92), steel_m))

    for sx in (-L * 0.455, L * 0.455):
        parts.append(_cube("end", (x_center + sx, 0.0, body_z),
                           (0.10, W * 0.96, body_h * 0.96), body_m))
        for sign in (-1, 1):
            bpy.ops.mesh.primitive_cube_add(
                size=1.0, location=(x_center + sx * 1.01, 0.0, body_z))
            brace = bpy.context.view_layer.objects.active
            brace.scale = (0.05, W * 0.90, 0.10)
            brace.rotation_euler = (0.0, 0.0, sign * 0.55)
            _apply(brace)
            _assign(brace, steel_m)
            parts.append(brace)

    parts.append(_cube("solebar", (x_center, 0.0, floor_z - 0.12),
                       (L * 0.96, W * 0.72, 0.28), frame_m))
    for sy in (-W * 0.38, W * 0.38):
        parts.append(_cube("sill", (x_center, sy, floor_z - 0.05),
                           (L * 0.94, 0.10, 0.18), frame_m))

    for bx in (-L * 0.30, L * 0.30):
        bogie_x = x_center + bx
        parts.append(_cube("bogie", (bogie_x, 0.0, 0.55),
                           (2.4, W * 0.70, 0.22), frame_m))
        for ax in (-0.72, 0.72):
            for sy in (-0.82, 0.82):
                parts.append(_cyl("wheel", (bogie_x + ax, sy, 0.42),
                                  0.42, 0.20, (math.pi * 0.5, 0.0, 0.0),
                                  steel_m, verts=14))
            parts.append(_cyl("axle", (bogie_x + ax, 0.0, 0.42),
                              0.08, W * 0.78, (math.pi * 0.5, 0.0, 0.0),
                              steel_m, verts=8))

    for sx in (-1.0, 1.0):
        tip = x_center + sx * L * 0.50
        for sy in (-0.55, 0.55):
            parts.append(_cyl("buffer", (tip, sy, floor_z + 0.15),
                              0.12, 0.28, (0.0, math.pi * 0.5, 0.0),
                              steel_m, verts=10))
        parts.append(_cube("coupler", (tip + sx * 0.18, 0.0, floor_z + 0.05),
                           (0.35, 0.18, 0.18), steel_m))

    for sx in (-L * 0.38, L * 0.38):
        for sy in (-W * 0.55, W * 0.55):
            parts.append(_cube("step", (x_center + sx, sy, floor_z - 0.35),
                               (0.35, 0.18, 0.08), steel_m))

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


def _select_meshes(meshes: list[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    if meshes:
        bpy.context.view_layer.objects.active = meshes[0]


def _scene_bounds(meshes: list[bpy.types.Object]):
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for obj in meshes:
        clo, chi = _world_bounds(obj)
        lo = Vector((min(lo.x, clo.x), min(lo.y, clo.y), min(lo.z, clo.z)))
        hi = Vector((max(hi.x, chi.x), max(hi.y, chi.y), max(hi.z, chi.z)))
    return lo, hi


def _report_materials() -> None:
    """只打印，绝不改材质——红色贴图必须原样留下。"""
    n_tex = 0
    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        linked = bool(bsdf.inputs["Base Color"].links)
        if linked:
            n_tex += 1
        rgba = tuple(round(c, 3) for c in bsdf.inputs["Base Color"].default_value)
        print(f"[gwr] mat {mat.name}: base={rgba} textured={linked}")
    print(f"[gwr] 贴图材质 {n_tex}/{len(bpy.data.materials)}")


def main() -> int:
    src = next((p for p in SRC_CANDIDATES if p.exists()), None)
    if src is None:
        print("[gwr] 找不到原始 glb")
        for p in SRC_CANDIDATES:
            print(f"  - {p}")
        return 2

    print(f"[gwr] 导入原始红机车 {src}")
    _clear()
    bpy.ops.import_scene.gltf(filepath=str(src))
    _report_materials()

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    # 源是厘米：缩到米（只动几何，不动材质）
    _select_meshes(meshes)
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
    lo, hi = _scene_bounds(meshes)
    height = hi.z - lo.z
    uniform = TARGET_HEIGHT / height if height > 1e-6 else 1.0
    print(f"[gwr] 高度 {height:.3f} → 缩尺 {uniform:.3f}")
    _select_meshes(meshes)
    bpy.ops.transform.resize(value=(uniform, uniform, uniform), center_override=(0, 0, 0))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    lo, hi = _scene_bounds(meshes)
    loco_len = hi.x - lo.x
    sx = TARGET_LOCO_LEN / loco_len if loco_len > 1e-6 else 1.0
    print(f"[gwr] 机车长 {loco_len:.2f} → ×{sx:.3f} = {TARGET_LOCO_LEN:.2f} m")
    cx = 0.5 * (lo.x + hi.x)
    _select_meshes(meshes)
    bpy.ops.transform.resize(
        value=(sx, 1.0, 1.0), center_override=(cx, 0.0, 0.0), orient_type="GLOBAL",
    )
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    lo, hi = _scene_bounds(meshes)
    shift = Vector((-lo.x, -0.5 * (lo.y + hi.y), -lo.z))
    for obj in meshes:
        obj.location += shift
        _apply(obj)

    # 原始模型烟箱朝 -X；绕中心掉头，让鼻锥朝 +X（列车前进方向）
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    lo, hi = _scene_bounds(meshes)
    mid = 0.5 * (lo + hi)
    _select_meshes(meshes)
    bpy.ops.transform.rotate(
        value=math.pi, orient_axis="Z", orient_type="GLOBAL",
        center_override=(mid.x, mid.y, mid.z),
    )
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    lo, hi = _scene_bounds(meshes)
    # 掉头后整体挪回 x≥0
    shift = Vector((-lo.x, -0.5 * (lo.y + hi.y), -lo.z))
    for obj in meshes:
        obj.location += shift
        _apply(obj)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    lo, hi = _scene_bounds(meshes)
    print(f"[gwr] 机车+煤水车就绪：长={hi.x - lo.x:.2f} 高={hi.z - lo.z:.2f} "
          f"宽={hi.y - lo.y:.2f}，网格 {len(meshes)}（材质未改，鼻锥朝 +X）")
    _report_materials()

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    root = bpy.context.view_layer.objects.active
    root.name = "GWR_HallClass"

    _parent_car(meshes, "Car01", root)
    lo, hi = _scene_bounds(meshes)
    cursor = hi.x + GAP + WAGON_LEN * 0.5
    for n in range(N_WAGONS):
        wagon = _make_boxcar(cursor)
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
        export_image_format="AUTO",
        export_materials="EXPORT",
        export_texcoords=True,
        export_normals=True,
    )
    print(f"[gwr] 已写出 {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
