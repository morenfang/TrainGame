"""中文文本：字体加载与标签工厂。

Panda3D 的字体加载有两个坑，都在这里处理掉：

1. 必须传 **Panda 虚拟文件系统风格**的路径（``/c/Windows/Fonts/msyh.ttc``），
   直接给 ``C:\\Windows\\...`` 会报 "Unable to find font file"。
   用 ``Filename.fromOsSpecific(p).getFullpath()`` 转换即可。
2. ``TextNode`` 上才叫 ``setFont``，``NodePath`` 上没有 —— 常踩。
"""

from __future__ import annotations

import os

from panda3d.core import Filename, NodePath, TextNode, Vec4

#: 优先用微软雅黑（字形最好看），后面几个是各版本 Windows 的兜底。
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\Deng.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    # 非 Windows 兜底（方便以后在别的机器上跑）
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)

_font_cache: dict[str, object] = {}
_resolved: object | None = None
_resolved_once = False


def cjk_font(base):
    """加载一个能显示中文的字体；都找不到时返回 ``None``（Panda 会用自带字体）。"""
    global _resolved, _resolved_once
    if _resolved_once:
        return _resolved
    _resolved_once = True

    for path in _FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        vfs_path = Filename.fromOsSpecific(path).getFullpath()
        try:
            font = base.loader.loadFont(vfs_path)
        except Exception:  # noqa: BLE001 - 字体缺失不应让程序起不来
            continue
        if font is None:
            continue
        _font_cache[vfs_path] = font
        _resolved = font
        break
    return _resolved


def make_label(text: str, *, font=None, color=(1.0, 1.0, 1.0, 1.0),
               align=TextNode.ALeft, wordwrap: float | None = None) -> NodePath:
    """造一个 ``TextNode`` 的 NodePath（默认左对齐，锚点在文本块左下角）。"""
    node = TextNode("label")
    if font is not None:
        node.setFont(font)
    node.setText(text)
    node.setAlign(align)
    node.setTextColor(Vec4(*color))
    if wordwrap is not None:
        node.setWordwrap(wordwrap)
    return NodePath(node)


def set_label(node_path: NodePath, text: str) -> None:
    """改文本内容（保持节点对象不变，避免每帧重建）。"""
    node = node_path.node()
    if node.getText() != text:
        node.setText(text)


def label_size(node_path: NodePath) -> tuple[float, float]:
    """文本块的 ``(宽, 高)``，单位是"字号 1.0 下的文本单位"。

    更新内容后要重新调用才会拿到新值 —— 叠放多行文本时用它做排版。

    **空文本返回 ``(0, 0)``**：Panda3D 在这种情况不会生成几何，
    ``getWidth`` / ``getHeight`` 于是返回未初始化的内存（实测量到过 ``1e-36``），
    或者干脆停在上一段文字的旧值上 —— 两者都不是"零"，据此排版会留下一块
    尺寸莫名其妙的空底衬。要量真实几何（含 TextNode 自带底衬）用
    ``NodePath.getTightBounds``。
    """
    node = node_path.node()
    if not node.getText():
        return (0.0, 0.0)
    node.generate()
    width, height = node.getWidth(), node.getHeight()
    if width != width or height != height:      # NaN 兜底
        return (0.0, 0.0)
    return (max(width, 0.0), max(height, 0.0))
