"""减面压缩列车 .glb：几何太密才是 55 MB 的原因（贴图只有几百 KB）。

Panda3D 的 panda3d-gltf **不读 Draco**，所以只能减三角面，不能靠网格压缩编码。

默认保留约 28% 面数：沙盘视距下外形几乎看不出差，文件大约掉到原来的 1/3～1/4。

用法::

    blender --background --python scripts/optimize_train_glb.py -- \\
        models/CR400BF_HuangSiDai_6car.glb

或只传文件名（相对 ``models/``）::

    blender --background --python scripts/optimize_train_glb.py -- \\
        CR400BF_HuangSiDai_6car.glb
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DEFAULT_RATIO = 0.28


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                  bpy.data.armatures):
        for item in list(block):
            block.remove(item)


def _mesh_objects() -> list[bpy.types.Object]:
    return [o for o in bpy.data.objects if o.type == "MESH" and o.data is not None]


def _face_count(obj: bpy.types.Object) -> int:
    return len(obj.data.polygons)


def _decimate(obj: bpy.types.Object, ratio: float) -> tuple[int, int]:
    before = _face_count(obj)
    if before < 500 or ratio >= 0.999:
        return before, before
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    # 先合并极近顶点，去掉导入时的碎缝
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0008)
    bpy.ops.object.mode_set(mode="OBJECT")
    mod = obj.modifiers.new(name="OptDecimate", type="DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=mod.name)
    after = _face_count(obj)
    return before, after


def _resolve_target(arg: str) -> Path:
    path = Path(arg)
    if not path.is_absolute():
        cand = MODELS / path.name if path.parent == Path(".") else ROOT / path
        path = cand
    return path.resolve()


def optimize(path: Path, *, ratio: float = DEFAULT_RATIO) -> int:
    if not path.exists():
        print(f"[opt] 缺少 {path}")
        return 1
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[opt] 备份 → {backup.name}  ({backup.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"[opt] 已有备份 {backup.name}，跳过覆盖")

    _clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = _mesh_objects()
    if not meshes:
        print("[opt] 导入后没有网格")
        return 1

    total_before = sum(_face_count(m) for m in meshes)
    print(f"[opt] 减面 ratio={ratio:.2f}，共 {len(meshes)} 个网格，"
          f"约 {total_before} 面")
    kept = 0
    for mesh in meshes:
        before, after = _decimate(mesh, ratio)
        kept += after
        print(f"[opt]   {mesh.name}: {before} → {after}")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
        export_animations=False,
        export_extras=False,
        export_cameras=False,
        export_lights=False,
        # panda3d-gltf 不支持 Draco，切勿打开
        export_draco_mesh_compression_enable=False,
    )
    size_mb = path.stat().st_size / 1e6
    print(f"[opt] 写回 {path.name}：{total_before} → {kept} 面，"
          f"{size_mb:.1f} MB")
    return 0


def main() -> int:
    argv = list(sys.argv)
    args = argv[argv.index("--") + 1 :] if "--" in argv else []
    ratio = DEFAULT_RATIO
    targets: list[str] = []
    i = 0
    while i < len(args):
        if args[i] in ("--ratio", "-r") and i + 1 < len(args):
            ratio = float(args[i + 1])
            i += 2
            continue
        targets.append(args[i])
        i += 1
    if not targets:
        targets = ["CR400BF_HuangSiDai_6car.glb"]
    code = 0
    for name in targets:
        code |= optimize(_resolve_target(name), ratio=ratio)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
