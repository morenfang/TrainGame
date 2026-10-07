"""外部 glTF 编组：能从 models/ 拆出车厢并摆到轨道上。"""

from __future__ import annotations

import pytest
from panda3d.core import Point3, Vec3

from core import paths
from core.train.consist import TrainCatalog
from render.gltf_train import cars_for, load_consist_templates, triangle_count

MODELS = paths.models_dir()
GLBS = sorted(MODELS.glob("*.glb")) if MODELS.exists() else []
pytestmark = pytest.mark.skipif(not GLBS, reason="models/ 里还没有 .glb")

HUANGSIDAI = paths.resolve_model_path("CR400BF_HuangSiDai_6car.glb")


@pytest.fixture(scope="module")
def catalog():
    from core.track.catalog import Catalog
    return Catalog.builtin()


@pytest.mark.parametrize("path", GLBS, ids=lambda p: p.name)
def test_every_glb_splits_into_cars(app, path):
    templates = load_consist_templates(path)
    assert len(templates) >= 2
    for car in templates:
        assert triangle_count(car) > 200
        lo, hi = Point3(), Point3()
        car.calcTightBounds(lo, hi)
        assert hi.x - lo.x > 3.0
        assert hi.z - lo.z > 1.5
        assert lo.y == pytest.approx(0.0, abs=0.05)


def test_huangsidai_is_six_cars_at_sandbox_length(app):
    if HUANGSIDAI is None:
        pytest.skip("没有黄丝带 400BF")
    templates = load_consist_templates(HUANGSIDAI)
    assert len(templates) == 6
    mid = templates[1]
    lo, hi = Point3(), Point3()
    mid.calcTightBounds(lo, hi)
    assert 8.5 < hi.x - lo.x < 12.0
    assert 2.0 < hi.y - lo.y < 5.0


def test_catalog_huangsidai_resolves_glb(app):
    catalog = TrainCatalog.builtin()
    for train_id, count in (("cr400bf_huangsidai_8", 8), ("cr400bf_huangsidai_16", 16)):
        spec = catalog[train_id]
        assert spec.mesh == "CR400BF_HuangSiDai_6car.glb"
        picked = cars_for(spec)
        assert picked is not None
        assert len(picked) == spec.car_count == count


def test_train_view_places_huangsidai_on_a_circle(app, catalog):
    from core.track.layout import Layout
    from core.track.path import detect_closure
    from core.train.consist import TrainCatalog as TC
    from render.train_view import TrainView

    layout = Layout(catalog=catalog)
    first = layout.add_root("curve_r40_l45")
    index = first
    for _ in range(7):
        index = layout.attach("curve_r40_l45", "a", (index, "b"))
    layout.connect((index, "b"), (first, "a"))
    report, path = detect_closure(layout, (first, "a"))
    assert report.closed

    spec = TC.builtin()["cr400bf_huangsidai_8"]
    view = TrainView(spec, app.render)
    view.state.s = 40.0
    assert view.sync(path)
    assert view.car_count == 8
    origin = view.node_for(0).getMat().xformPoint(Vec3(0, 0, 0))
    assert origin.length() > 1.0
    view.destroy()
