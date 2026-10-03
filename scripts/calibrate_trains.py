"""列车参数标定工具：把 ``data/trains.json`` 的参数与真实车型对一下。

用途：调完 ``trains.json`` 里的功率 / 牵引力 / davis 系数后，跑一下这个脚本，
看看加速时间与平衡速度是否还在真实车型的量级上。

    python scripts/calibrate_trains.py

参考真值（大致量级，用于判断参数是否离谱）::

    CR400AF  0-200 km/h 约 2 min，350 km/h 巡航，8 节编组约 420~460 t
    CRH380A  加速略逊于 CR400AF，8 节编组
    绿皮车   DF4B 单机 1985 kW 牵引 10~16 辆，0-100 km/h 需要几分钟
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.train.consist import TrainCatalog  # noqa: E402
from core.train.dynamics import (  # noqa: E402
    steady_state_speed,
    time_to_reach,
)

HEADER = (
    f"{'列车':<18}{'质量t':>8}{'P/m':>7}{'Fmax':>7}{'黏着':>7}"
    f"{'拐点':>8}{'极速':>8}{'0-100':>8}{'0-200':>8}{'0-300':>8}"
)
UNITS = (
    f"{'':<18}{'':>8}{'W/kg':>7}{'kN':>7}{'kN':>7}"
    f"{'km/h':>8}{'km/h':>8}{'s':>8}{'s':>8}{'s':>8}"
)


def cell(seconds: float | None) -> str:
    return f"{seconds:.0f}" if seconds is not None else "--"


def main() -> int:
    catalog = TrainCatalog.builtin()
    print(HEADER)
    print(UNITS)
    print("-" * len(UNITS))

    for train in catalog:
        reached = [time_to_reach(train, target) for target in (100.0, 200.0, 300.0)]
        cruise = steady_state_speed(train) * 3.6
        print(
            f"{train.id:<18}"
            f"{train.total_mass / 1000:>8.1f}"
            f"{train.power_to_weight():>7.1f}"
            f"{train.max_tractive_force_n / 1e3:>7.0f}"
            f"{train.max_traction_from_adhesion / 1e3:>7.0f}"
            f"{train.power_limited_speed() * 3.6:>8.0f}"
            f"{cruise:>8.0f}"
            f"{cell(reached[0]):>8}{cell(reached[1]):>8}{cell(reached[2]):>8}"
        )

    print()
    print("坡道表现（全手柄，目标 100 km/h）")
    print("-" * len(UNITS))
    for train in catalog:
        row = [f"{train.id:<18}"]
        for grade in (0.01, 0.02, 0.03, 0.05):
            seconds = time_to_reach(train, 100.0, grade=grade)
            cruise = steady_state_speed(train, grade) * 3.6
            if seconds is not None:
                row.append(f"{grade * 100:.0f}%: {seconds:>4.0f}s")
            else:
                row.append(f"{grade * 100:.0f}%: 爬不上去({cruise:.0f})")
        print("  ".join(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
