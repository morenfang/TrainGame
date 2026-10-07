"""可直接运行的场景：轨道 + 布景，开箱就能开跑。

对外只有四个入口：

    scenes.keys()                 有哪些场景
    scenes.build(key)             拼一个出来（闭环轨道 + 布景数据）
    scenes.save(scene, path)      存档（轨道拓扑 + 一行"布景用哪个预设"）
    scenes.load(path, catalog)    读档，连布景一起还原

为什么存档里只记"布景是哪个预设"，而不是记一堆坐标
------------------------------------------------
布景是**程序化**的（见 :mod:`render.scenery`）：同一份数据每次生成的网格逐顶点
一致。所以存档里存几百个坐标既冗余、又会在预设改过之后与轨道对不上；存一个 key、
读档时重算，反而永远自洽。

**手工摆的那几件例外**：用户在编辑器里手动放下的房 / 车站 / 大山 / 树，不能靠
预设重算出来，所以单独存进 ``user_scenery`` 一段（见 :func:`scenery.scenery_to_dict`）。
程序化底座按 key 重算、手工件逐件还原，两层各管各的。

若用户在布景模式里删过底座上的房 / 车站 / 山 / 树，存档会多一段 ``base_scenery``
（只含可摆放类），读档时盖回预设重算出来的对应字段。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.track.catalog import Catalog
from core.track.layout import Layout
from render import scenery as scenery_mod
from scenes import presets, tracks
from scenes.presets import DEFAULT_PLOT, Scene, build, keys
from scenes.tracks import SceneError

#: 存档里记录布景预设的键名。``core.track.layout`` 读档时只挑自己认识的键，
#: 多出来的这个不会影响它（见 ``Layout.from_dict``）。
SCENERY_KEY = "scenery"

__all__ = [
    "SCENERY_KEY", "Scene", "SceneError", "LoadedScene", "build", "keys",
    "load", "presets", "plot_size_of", "save", "scenery_for", "tracks",
    "train_id_for",
]


@dataclass(frozen=True)
class LoadedScene:
    """从存档里读出来的东西：轨道 + 该配哪份布景 + 场地多大 + 手工摆的布景。"""

    layout: Layout
    scenery_key: str | None = None
    plot_size: float = DEFAULT_PLOT
    #: 用户手工摆放的布景（房 / 车站 / 大山 / 树 / 湖 / 绿地），存档里 ``user_scenery``。
    user_scenery: scenery_mod.Scenery = field(default_factory=scenery_mod.Scenery)
    #: 底座上被删改过的可摆放件原始字典（``None`` = 仍用预设原样）。
    #: 保留原始 dict，旧存档缺 ``lakes`` / ``meadows`` 键时不会把预设湖草清掉。
    base_placeables: dict | None = None

    @property
    def scenery(self) -> scenery_mod.Scenery:
        """跟着这份存档的程序化布景（存档里没记预设时是一份空布景）。"""
        if self.scenery_key is None:
            base = scenery_mod.Scenery()
        else:
            base = scenery_for(self.scenery_key)
        if self.base_placeables is None:
            return base
        return scenery_mod.apply_placeables_dict(base, self.base_placeables)


def scenery_for(key: str) -> scenery_mod.Scenery:
    """按预设 key 重建布景数据。

    布景的坐标是**绝对**的，由预设自己算出来（绕着预设里那条轨道摆），与调用方
    手上的轨道无关。所以这里连轨道一起重建、再把轨道丢掉 —— 多花几十毫秒，
    换来的是"布景只有一个来源"。
    """
    return presets.build(key).scenery


def save(scene: Scene, path: str | Path) -> Path:
    """把场景存成存档：轨道拓扑 + 一行布景预设（+ 标题，方便人肉认档）。"""
    payload = scene.layout.to_dict()
    payload[SCENERY_KEY] = scene.key
    payload["title"] = scene.title
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target


def load(path: str | Path, catalog: Catalog | None = None) -> LoadedScene:
    """读一份存档，顺带把布景预设也读出来。"""
    catalog = catalog if catalog is not None else Catalog.builtin()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    key = payload.get(SCENERY_KEY)
    if key not in presets.BUILDERS:
        key = None                        # 存档没记 / 记的是已经不存在的预设
    base_raw = payload.get("base_scenery")
    return LoadedScene(
        layout=Layout.from_dict(payload, catalog),
        scenery_key=key,
        plot_size=plot_size_of(key),
        user_scenery=scenery_mod.scenery_from_dict(payload.get("user_scenery")),
        base_placeables=base_raw if isinstance(base_raw, dict) else None,
    )


def plot_size_of(key: str | None) -> float:
    """这个场景要多大的地。

    查表而不是"把场景拼一遍再看" —— 读档时**不能**因为预设拼不出来就失败：
    存档里的轨道是可以独立打开的，布景不过是锦上添花。
    """
    return presets.PLOT_SIZES.get(key, DEFAULT_PLOT)


def train_id_for(key: str | None) -> str | None:
    """这个场景默认上哪列车。存档只记布景 key，读档时用同一张表。"""
    if key is None:
        return None
    return presets.TRAIN_IDS.get(key)
