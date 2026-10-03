"""轨道子系统：件目录、装配图、路径求解。

对外只暴露这一组名字；``render/`` 与 ``app/`` 应当只通过它们接触轨道逻辑。
"""

from core.track.catalog import Catalog, CatalogError, PieceDef, Port, Route, build_piece
from core.track.layout import Layout, LayoutError, PlacedPiece, PortKey
from core.track.path import (
    CLOSURE_HEADING_TOL,
    CLOSURE_POSITION_TOL,
    BogieState,
    CarState,
    PathError,
    PathSegment,
    RoutePath,
    bogie_at,
    car_from_bogies,
    consist_length,
    detect_closure,
    find_loop,
    place_consist,
    trace,
)

__all__ = [
    # catalog
    "Catalog",
    "CatalogError",
    "PieceDef",
    "Port",
    "Route",
    "build_piece",
    # layout
    "Layout",
    "LayoutError",
    "PlacedPiece",
    "PortKey",
    # path
    "RoutePath",
    "PathSegment",
    "PathError",
    "BogieState",
    "CarState",
    "trace",
    "detect_closure",
    "find_loop",
    "bogie_at",
    "car_from_bogies",
    "place_consist",
    "consist_length",
    "CLOSURE_POSITION_TOL",
    "CLOSURE_HEADING_TOL",
]
