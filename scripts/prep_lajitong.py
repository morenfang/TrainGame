"""复兴号「垃圾桶」：源约 3 节碎网格，整理成 Car01 头 / Car02 中间 / Car03 尾。

游戏按节数抽编组（8 节 = 头 + 6 中间 + 尾）。不 join，避免把 Blender 打崩。
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "glbxz-com-0000083178-1.glb"
OUT = ROOT / "models" / "CR400BF_LaJiTong_8car.glb"
TARGET_HEIGHT = 3.5 * 1.15
TARGET_LEN = 25.0
TARGET_W = 3.05


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for col in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.armatures):
        for item in list(col):
            col.remove(item)


def _mesh_bb(obj: bpy.types.Object):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs, ys, zs = [c.x for c in corners], [c.y for c in corners], [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def _tree_bb(root: bpy.types.Object):
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    found = False
    nodes = [root, *root.children_recursive]
    for obj in nodes:
        if obj.type != "MESH":
            continue
        clo, chi = _mesh_bb(obj)
        found = True
        lo = Vector((min(lo.x, clo.x), min(lo.y, clo.y), min(lo.z, clo.z)))
        hi = Vector((max(hi.x, chi.x), max(hi.y, chi.y), max(hi.z, chi.z)))
    if not found:
        return None
    return lo, hi


def _select_all_meshes() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for obj in meshes:
        obj.hide_set(False)
        obj.select_set(True)
    if meshes:
        bpy.context.view_layer.objects.active = meshes[0]


def _is_track(dx: float, dy: float, dz: float) -> bool:
    length = max(dx, dy, dz)
    height = min(dx, dy, dz)
    mid = dx + dy + dz - length - height
    return length > 4.0 and height < 0.05 and mid < 0.5


def main() -> int:
    if not SRC.exists():
        print(f"[lajitong] 找不到 {SRC}")
        return 2
    print(f"[lajitong] 导入 {SRC}")
    _clear()
    bpy.ops.import_scene.gltf(filepath=str(SRC))

    trash = []
    for obj in list(bpy.context.scene.objects):
        if obj.type != "MESH":
            continue
        lo, hi = _mesh_bb(obj)
        ext = hi - lo
        if max(ext) < 1e-4 or _is_track(ext.x, ext.y, ext.z):
            trash.append(obj)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in trash:
        obj.select_set(True)
    if bpy.context.selected_objects:
        bpy.ops.object.delete(use_global=False)
    print(f"[lajitong] 去掉铁轨/退化 {len(trash)}")

    tops = [
        o for o in bpy.data.objects
        if o.type == "EMPTY" and o.parent is None and _tree_bb(o)
    ]
    ranked = []
    for empty in tops:
        bb = _tree_bb(empty)
        if bb is None:
            continue
        lo, hi = bb
        ranked.append((0.5 * (lo.y + hi.y), empty))
    ranked.sort(key=lambda t: t[0])
    print(f"[lajitong] 车厢组 {len(ranked)}: {[e.name for _, e in ranked]}")
    if len(ranked) < 2:
        print("[lajitong] 分不出车厢")
        return 3

    _select_all_meshes()
    bpy.ops.transform.rotate(value=-math.pi / 2.0, orient_axis="Z",
                             center_override=(0.0, 0.0, 0.0),
                             orient_type="GLOBAL")
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    bb = None
    for _, empty in ranked:
        cur = _tree_bb(empty)
        if cur is None:
            continue
        clo, chi = cur
        if bb is None:
            bb = (clo.copy(), chi.copy())
        else:
            lo, hi = bb
            bb = (
                Vector((min(lo.x, clo.x), min(lo.y, clo.y), min(lo.z, clo.z))),
                Vector((max(hi.x, chi.x), max(hi.y, chi.y), max(hi.z, chi.z))),
            )
    lo, hi = bb
    height = hi.z - lo.z
    uniform = TARGET_HEIGHT / height if height > 1e-6 else 1.0
    print(f"[lajitong] 高度 {height:.4f} → {uniform:.3f}")
    _select_all_meshes()
    bpy.ops.transform.resize(value=(uniform, uniform, uniform),
                             center_override=(0.0, 0.0, 0.0))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    lengths, widths = [], []
    for _, empty in ranked:
        clo, chi = _tree_bb(empty)
        lengths.append(chi.x - clo.x)
        widths.append(chi.y - clo.y)
    lengths.sort()
    widths.sort()
    sx = TARGET_LEN / lengths[len(lengths) // 2]
    sy = TARGET_W / widths[len(widths) // 2]
    print(f"[lajitong] 断面 X×{sx:.3f} Y×{sy:.3f}")
    _select_all_meshes()
    bpy.ops.transform.resize(value=(sx, sy, 1.0), center_override=(0.0, 0.0, 0.0))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bb = None
    for _, empty in ranked:
        clo, chi = _tree_bb(empty)
        if bb is None:
            bb = (clo.copy(), chi.copy())
        else:
            lo, hi = bb
            bb = (
                Vector((min(lo.x, clo.x), min(lo.y, clo.y), min(lo.z, clo.z))),
                Vector((max(hi.x, chi.x), max(hi.y, chi.y), max(hi.z, chi.z))),
            )
    lo, hi = bb
    shift = Vector((-0.5 * (lo.x + hi.x), -0.5 * (lo.y + hi.y), -lo.z))
    _select_all_meshes()
    bpy.ops.transform.translate(value=tuple(shift))
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    ordered = []
    for _, empty in ranked:
        clo, chi = _tree_bb(empty)
        ordered.append((0.5 * (clo.x + chi.x), empty, clo, chi))
    ordered.sort(key=lambda t: t[0])

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    root = bpy.context.view_layer.objects.active
    root.name = "CR400BF_LaJiTong"

    # 3 节：头、中间、尾（头尾都用最外侧两节）
    picks = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    used = set()
    index = 1
    for cx, empty, clo, chi in picks:
        if empty.name in used:
            continue
        used.add(empty.name)
        empty.name = f"Car{index:02d}"
        empty.parent = root
        print(f"[lajitong] {empty.name}: 长={chi.x - clo.x:.2f} "
              f"高={chi.z - clo.z:.2f} 宽={chi.y - clo.y:.2f} 中心X={cx:.2f}")
        index += 1

    bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    for obj in bpy.context.scene.objects:
        if obj.name.startswith("Car"):
            obj.select_set(True)
            for child in obj.children_recursive:
                child.select_set(True)
    bpy.context.view_layer.objects.active = root
    OUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(OUT), export_format="GLB", use_selection=True,
        export_apply=True, export_yup=True,
    )
    print(f"[lajitong] 已写出 {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
