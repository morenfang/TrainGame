"""垃圾桶：电弧改灰；Hall：机车再拉长 15% 并柔化涂装。

    blender --background --python scripts/tweak_laji_gwr.py
"""

from __future__ import annotations

from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LAJI = MODELS / "CR400BF_LaJiTong_8car.glb"
GWR = MODELS / "GWR_HallClass_19car.glb"
LOCO_STRETCH = 1.15
GRAY = (0.42, 0.43, 0.45, 1.0)


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


def _bsdf(mat: bpy.types.Material):
    if not mat.use_nodes or mat.node_tree is None:
        return None
    return next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def _base_color(mat: bpy.types.Material):
    bsdf = _bsdf(mat)
    if bsdf is None:
        return None
    return tuple(bsdf.inputs["Base Color"].default_value)


def _set_base(mat: bpy.types.Material, rgba, roughness=None, metallic=None) -> None:
    if not mat.use_nodes:
        mat.use_nodes = True
    bsdf = _bsdf(mat)
    if bsdf is None:
        return
    bsdf.inputs["Base Color"].default_value = rgba
    if roughness is not None:
        bsdf.inputs["Roughness"].default_value = roughness
    if metallic is not None:
        bsdf.inputs["Metallic"].default_value = metallic


def _export(path: Path, prefer_root: bool = True) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    root = None
    if prefer_root:
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
    else:
        bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB", use_selection=True,
        export_apply=True, export_yup=True,
    )
    print(f"[tweak] 写出 {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def _is_pantograph_mat(mat: bpy.types.Material) -> bool:
    """铜黄 / 橙红 / 高饱和暖色 → 电弧受电弓一类。"""
    name = mat.name.lower()
    if any(k in name for k in ("panto", "arc", "bow", "canten", "copper", "电弧", "受电")):
        return True
    rgba = _base_color(mat)
    if rgba is None:
        return False
    r, g, b, _ = rgba
    peak = max(r, g, b, 1e-6)
    # 明显偏暖且不接近白/灰
    warm = r > 0.35 and r > b * 1.25 and r >= g * 0.85
    sat = peak - min(r, g, b)
    not_white = peak < 0.92
    return warm and sat > 0.12 and not_white


def _is_pantograph_mesh(obj: bpy.types.Object, roof_z: float) -> bool:
    """车顶以上、又细又高的网格，当作受电弓。"""
    if obj.type != "MESH":
        return False
    lo, hi = _world_bounds(obj)
    dx, dy, dz = hi.x - lo.x, hi.y - lo.y, hi.z - lo.z
    if hi.z < roof_z - 0.05:
        return False
    # 细长立件：高度明显，平面投影不大
    footprint = max(dx, dy)
    return dz > 0.35 and footprint < 2.2 and hi.z > roof_z + 0.25


def tweak_lajitong() -> float | None:
    """电弧改灰；返回头车真实长度（米制，导出后未缩尺）。"""
    if not LAJI.exists():
        print(f"[tweak] 缺少 {LAJI}")
        return None
    _clear()
    bpy.ops.import_scene.gltf(filepath=str(LAJI))
    cars = sorted(
        (o for o in bpy.data.objects if o.name.startswith("Car") and o.name[3:].isdigit()),
        key=lambda o: o.name,
    )
    # 车顶高度取中间车
    mid = next((c for c in cars if c.name == "Car02"), cars[0] if cars else None)
    roof_z = 3.2
    if mid is not None:
        bb = _tree_bounds(mid)
        if bb:
            # 车体主体高度略低于最高点（受电弓）
            roof_z = bb[0][2] + (bb[1][2] - bb[0][2]) * 0.78
            print(f"[tweak] 估计车顶 z≈{roof_z:.2f}（整车高 {bb[1][2] - bb[0][2]:.2f}）")

    gray_mat = bpy.data.materials.get("PantographGray")
    if gray_mat is None:
        gray_mat = bpy.data.materials.new("PantographGray")
        gray_mat.use_nodes = True
        _set_base(gray_mat, GRAY, roughness=0.48, metallic=0.55)

    mat_hits = 0
    for mat in list(bpy.data.materials):
        if mat.name == "PantographGray":
            continue
        if _is_pantograph_mat(mat):
            before = _base_color(mat)
            _set_base(mat, GRAY, roughness=0.48, metallic=0.55)
            mat_hits += 1
            print(f"[tweak] 材质改灰 {mat.name}: {before} → {GRAY}")

    mesh_hits = 0
    for obj in bpy.context.scene.objects:
        if not _is_pantograph_mesh(obj, roof_z):
            continue
        if obj.data.materials:
            obj.data.materials[0] = gray_mat
        else:
            obj.data.materials.append(gray_mat)
        mesh_hits += 1
        print(f"[tweak] 网格改灰 {obj.name}")

    print(f"[tweak] 电弧：材质 {mat_hits} + 网格 {mesh_hits}")

    head = next((c for c in cars if c.name == "Car01"), None)
    head_len = None
    if head is not None:
        bb = _tree_bounds(head)
        if bb:
            head_len = bb[1].x - bb[0].x
            print(f"[tweak] 头车真实长 {head_len:.3f} m")
    _export(LAJI)
    return head_len


def tweak_gwr() -> float | None:
    if not GWR.exists():
        print(f"[tweak] 缺少 {GWR}")
        return None
    _clear()
    bpy.ops.import_scene.gltf(filepath=str(GWR))
    car01 = bpy.data.objects.get("Car01")
    if car01 is None:
        print("[tweak] 没有 Car01")
        return None

    meshes = [o for o in car01.children_recursive if o.type == "MESH"]
    before = _tree_bounds(car01)
    if not meshes or before is None:
        print("[tweak] Car01 无网格")
        return None
    cx = 0.5 * (before[0].x + before[1].x)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.transform.resize(
        value=(LOCO_STRETCH, 1.0, 1.0),
        center_override=(cx, 0.0, 0.0),
        orient_type="GLOBAL",
    )
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    after = _tree_bounds(car01)
    print(f"[tweak] 机车 {before[1].x - before[0].x:.2f} → "
          f"{after[1].x - after[0].x:.2f} m")

    # 注意：绝不改机车材质。红色贴图必须原样保留；要重做请跑 prep_gwr_hall.py。

    # 机车拉长后货车可能叠上——按新车尾重排货车
    lo, hi = after
    gap = 0.70
    wagon_len = 24.0
    wagons = sorted(
        (o for o in bpy.data.objects
         if o.name.startswith("Car") and o.name != "Car01" and o.name[3:].isdigit()),
        key=lambda o: o.name,
    )
    cursor = hi.x + gap + wagon_len * 0.5
    for empty in wagons:
        bb = _tree_bounds(empty)
        if bb is None:
            continue
        old_cx = 0.5 * (bb[0].x + bb[1].x)
        dx = cursor - old_cx
        empty.location.x += dx
        # 子网格跟 empty 走（已 parent）
        cursor += wagon_len + gap
        print(f"[tweak] {empty.name} → x={empty.location.x:.2f}")

    loco_len = after[1].x - after[0].x
    _export(GWR)
    return loco_len


def main() -> int:
    head = tweak_lajitong()
    loco = tweak_gwr()
    print(f"[tweak] DONE head_real={head} loco_real={loco}")
    # 把真实米制长度写成旁路文件，方便改 trains.json（沙盘 = 真实 × 0.4）
    out = ROOT / "scripts" / "_tweak_lengths.txt"
    with out.open("w", encoding="utf-8") as fh:
        if head is not None:
            fh.write(f"lajitong_end_real={head:.4f}\n")
            fh.write(f"lajitong_end_sandbox={head * 0.4:.4f}\n")
        if loco is not None:
            fh.write(f"gwr_loco_real={loco:.4f}\n")
            fh.write(f"gwr_loco_sandbox={loco * 0.4:.4f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
