"""pytest 共享 fixture：一个离屏的 Panda3D 应用 + 干净的场景图。

为什么必须是 session 作用域
------------------------------------------------
``ShowBase`` 是进程单例 —— 同一个进程里 new 第二次会直接报错，所以整个测试会话
只能有一个。但它本身是无状态的容器（窗口 + render 根 + aspect2d 根），
真正的隔离交给下面的 :func:`app` 去做。

    10|为什么 :func:`app` 要手动清场景图
------------------------------------------------
``ShowBase`` 的所有场景内容都挂在 ``base.render`` / ``base.aspect2d`` 下面。
测试之间不清理的话，上一个用例留下的轨道节点会让下一个用例的 ``find``、
三角形统计、甚至 ``toggle_grid`` 全都看到别人的东西 —— 这种串味故障最难查。
所以记录初始子节点集合，测试结束把所有"新来的"摘掉。
"""

from __future__ import annotations

import pytest
from panda3d.core import loadPrcFileData

from core.train.consist import TrainCatalog, TrainSpec, build_train

#: 必须在任何 Panda3D 对象生成之前设定，且与 app/config.py 保持一致。
loadPrcFileData("", "coordinate-system yup")
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "win-size 1280 720")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "sync-video false")
loadPrcFileData("", "notify-level-display error")
loadPrcFileData("", "notify-level-glgsg error")


@pytest.fixture(scope="session")
def base():
    """整个测试会话共用的离屏 ShowBase。"""
    from direct.showbase.ShowBase import ShowBase

    app = ShowBase()
    app.disableMouse()          # 相机交给 OrbitCamera / 脚本，不要引擎自己转
    yield app


@pytest.fixture(scope="module")
def train_catalog() -> TrainCatalog:
    return TrainCatalog.builtin()


@pytest.fixture(scope="module")
def green(train_catalog: TrainCatalog) -> TrainSpec:
    """Synthetic green-skin consist (same params as former catalog entry)."""
    return build_train(
        {
            "id": "green_skin_10",
            "coupling_gap": 0.3,
            "power_w": 1985000,
            "max_tractive_force_n": 300000,
            "max_brake_force_n": 584800,
            "max_emergency_brake_force_n": 1651200,
            "adhesion": 0.25,
            "davis": [0.015, 1.5e-4, 2.0e-5],
            "max_speed_kmh": 100,
            "formation": [["df4b_loco", 1], ["coach_25b", 10]],
        },
        train_catalog.car_types,
    )


@pytest.fixture(scope="module")
def green_16(train_catalog: TrainCatalog) -> TrainSpec:
    return build_train(
        {
            "id": "green_skin_16",
            "coupling_gap": 0.3,
            "power_w": 1985000,
            "max_tractive_force_n": 300000,
            "max_brake_force_n": 584800,
            "max_emergency_brake_force_n": 1651200,
            "adhesion": 0.25,
            "davis": [0.015, 1.5e-4, 2.0e-5],
            "max_speed_kmh": 100,
            "formation": [["df4b_loco", 1], ["coach_25b", 16]],
        },
        train_catalog.car_types,
    )


@pytest.fixture(scope="module")
def hexie(train_catalog: TrainCatalog) -> TrainSpec:
    return build_train(
        {
            "id": "crh380a_8",
            "coupling_gap": 0.25,
            "power_w": 9600000,
            "max_tractive_force_n": 250000,
            "max_brake_force_n": 383000,
            "max_emergency_brake_force_n": 919200,
            "adhesion": 0.25,
            "davis": [0.010, 1.1e-4, 1.4e-5],
            "max_speed_kmh": 350,
            "formation": [
                ["crh380a_end", 1],
                ["crh380a_mid", 6],
                ["crh380a_end", 1],
            ],
        },
        train_catalog.car_types,
    )


@pytest.fixture(scope="module")
def fuxing_16(train_catalog: TrainCatalog) -> TrainSpec:
    return build_train(
        {
            "id": "cr400af_16",
            "coupling_gap": 0.25,
            "power_w": 22000000,
            "max_tractive_force_n": 520000,
            "max_brake_force_n": 734000,
            "max_emergency_brake_force_n": 1761600,
            "adhesion": 0.25,
            "davis": [0.009, 1.0e-4, 1.25e-5],
            "max_speed_kmh": 350,
            "formation": [
                ["cr400af_end", 1],
                ["cr400af_mid", 14],
                ["cr400af_end", 1],
            ],
        },
        train_catalog.car_types,
    )


@pytest.fixture(scope="module")
def fuxing(train_catalog: TrainCatalog) -> TrainSpec:
    return build_train(
        {
            "id": "cr400af_8",
            "coupling_gap": 0.25,
            "power_w": 11000000,
            "max_tractive_force_n": 260000,
            "max_brake_force_n": 367000,
            "max_emergency_brake_force_n": 880800,
            "adhesion": 0.25,
            "davis": [0.009, 1.0e-4, 1.25e-5],
            "max_speed_kmh": 350,
            "formation": [
                ["cr400af_end", 1],
                ["cr400af_mid", 6],
                ["cr400af_end", 1],
            ],
        },
        train_catalog.car_types,
    )


@pytest.fixture(params=["hexie", "fuxing", "green"])
def synthetic_train(request, hexie, fuxing, green) -> TrainSpec:
    return {"hexie": hexie, "fuxing": fuxing, "green": green}[request.param]


@pytest.fixture()
def app(base):
    """一个用完即净的场景图。"""
    roots = (base.render, base.aspect2d)
    keep = {root: list(root.getChildren()) for root in roots}
    yield base
    base.ignoreAll()             # 清掉测试 bind 过的事件，免得上一个用例的编辑器还活着
    for root, original in keep.items():
        for child in root.getChildren():
            if all(child != kept for kept in original):
                child.removeNode()
