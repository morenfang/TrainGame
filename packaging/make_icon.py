"""生成 exe 的图标：``packaging/TrainGame.ico``。

为什么用代码画而不是塞一个 png 进仓库
------------------------------------------------
图标要能反映**这个游戏**长什么样：天蓝、草地、道砟与钢轨、一列绿皮车 ——
这些颜色在 :mod:`render.style` 与 ``data/trains.json`` 里都已经定死了，所以这里
直接**引用它们**，而不是另抄一套十六进制。以后调了天空色或涂装，重新生成图标即可，
图标不会跟游戏里的样子跑偏。

画法用的是"4 倍超采样再缩下来"：Pillow 的 ``ImageDraw`` 没有抗锯齿，直接画 256
像素的圆角矩形会看到毛边。先在 4 倍画布上画，再用 ``LANCZOS`` 缩，边缘就干净了。

用法::

    python packaging/make_icon.py           # 生成 packaging/TrainGame.ico
    python packaging/make_icon.py --preview # 同时存一张 png 方便肉眼看
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from render import style                                  # noqa: E402

OUT = Path(__file__).resolve().parent / "TrainGame.ico"

#: 图标里出现的每一档尺寸。Windows 会按用途挑：任务栏要 32/48、桌面要 256。
SIZES = (256, 128, 64, 48, 32, 16)

#: 4 倍超采样。
_SS = 4

#: 绿皮车的涂装，抄自 ``data/trains.json`` 的 ``car_types`` —— 图标上就是它。
_BODY = "#2F5D3A"
_BAND = "#E8C84A"
_ROOF = "#7D8279"
_WINDOW = "#25333A"
_WHEEL = "#2B2F33"
_WHEEL_HUB = "#6E7478"


def _rgb(value) -> tuple[int, int, int]:
    """``render.style`` 里的颜色是 0..1 的浮点四元组，转成 0..255。"""
    return tuple(round(max(0.0, min(1.0, c)) * 255) for c in value[:3])


class _Canvas:
    """在 4 倍画布上按 0..1 的相对坐标画图，缩回目标尺寸时自动抗锯齿。"""

    def __init__(self, size: int):
        from PIL import Image, ImageDraw

        self.size = size
        self.px = size * _SS
        self.image = Image.new("RGBA", (self.px, self.px), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

    # -- 坐标换算 ---------------------------------------------------------- #

    def x(self, value: float) -> float:
        return value * self.px

    def y(self, value: float) -> float:
        return value * self.px

    def box(self, left, top, right, bottom):
        return (self.x(left), self.y(top), self.x(right), self.y(bottom))

    # -- 基本图元 ---------------------------------------------------------- #

    def rounded(self, box, radius: float, fill):
        self.draw.rounded_rectangle(self.box(*box), radius=self.x(radius), fill=fill)

    def rect(self, box, fill):
        self.draw.rectangle(self.box(*box), fill=fill)

    def ellipse(self, box, fill):
        self.draw.ellipse(self.box(*box), fill=fill)

    def polygon(self, points, fill):
        self.draw.polygon([(self.x(px), self.y(py)) for px, py in points], fill=fill)


def build(size: int = 256, *, preview: Path | None = None) -> Path:
    """画出 ``size`` 见方的图标并存成多尺寸 ``.ico``。"""
    from PIL import Image

    c = _Canvas(max(SIZES))
    sky = _rgb(style.SKY_COLOR)
    ground = _rgb(style.GROUND_COLOR)
    ballast = _rgb(style.BALLAST_COLOR)
    tie = _rgb(style.TIE_COLOR)
    rail = _rgb(style.RAIL_HEAD_COLOR)

    # ---- 背景：圆角方块 + 天空到地面的两段
    c.rounded((0.0, 0.0, 1.0, 1.0), 0.19, sky)
    c.rect((0.0, 0.695, 1.0, 1.0), ground)

    # ---- 道砟（上窄下宽的梯形）与枕木
    c.polygon(((0.055, 0.796), (0.945, 0.796), (0.985, 0.885), (0.015, 0.885)), ballast)
    for i in range(6):
        left = 0.085 + i * 0.148
        c.rect((left, 0.762, left + 0.072, 0.802), tie)

    # ---- 钢轨：亮的一条是轨头（车走在它的顶面，y = 0.742），下面那条是轨腰
    c.rect((0.0, 0.742, 1.0, 0.752), rail)
    c.rect((0.0, 0.752, 1.0, 0.762), _rgb(style.RAIL_COLOR))

    # ---- 车体（绿皮车）：圆角矩形 + 车顶带 + 黄色色带
    c.rounded((0.075, 0.335, 0.925, 0.690), 0.045, _BODY)
    c.rounded((0.070, 0.320, 0.930, 0.396), 0.038, _ROOF)
    c.rect((0.075, 0.545, 0.925, 0.596), _BAND)

    # ---- 车窗：5 扇，等距
    for i in range(5):
        left = 0.113 + i * 0.166
        c.rounded((left, 0.400, left + 0.132, 0.510), 0.018, _WINDOW)

    # ---- 车轮：**所有轮子的下缘都落在轨头顶面**（0.742）上，所以圆心是
    #      "轨面高度 - 半径" —— 转向架的大小轮共用一个切点，这才是车站在轨上的样子
    for cx, radius in ((0.215, 0.075), (0.785, 0.075), (0.395, 0.055), (0.605, 0.055)):
        cy = 0.742 - radius
        c.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), _WHEEL)
        c.ellipse((cx - radius * 0.42, cy - radius * 0.42,
                   cx + radius * 0.42, cy + radius * 0.42), _WHEEL_HUB)

    # ---- 缩放到各档尺寸（LANCZOS 就是这里的抗锯齿）
    frames = [c.image.resize((s, s), Image.LANCZOS) for s in SIZES]
    if preview is not None:
        c.image.resize((256, 256), Image.LANCZOS).save(preview)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, format="ICO",
                   sizes=[(s, s) for s in SIZES], append_images=frames[1:])
    return OUT


def main(argv: list[str]) -> int:
    preview = (Path(__file__).resolve().parent / "TrainGame-preview.png") \
        if "--preview" in argv else None
    path = build(preview=preview)
    print(f"[图标] 已生成 {path}（{path.stat().st_size} 字节，"
          f"{len(SIZES)} 档尺寸）")
    if preview is not None:
        print(f"[图标] 预览图 {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
