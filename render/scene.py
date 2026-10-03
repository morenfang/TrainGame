"""把 :class:`core.track.layout.Layout` 增量同步成 Panda3D 场景图。

为什么是增量而不是每帧重建
------------------------------------------------
一次 ``sync()`` 只做三件必要的事：删掉已被移除的件的节点、给新件或"地面高度变了
的件"换网格、把位姿写进节点。件的**网格模板按 ``(件 id, 地面高度)`` 缓存**，
新增同类件时走 ``copyTo`` 共享 Geom（真正的实例化，显存只占一份），
所以摆几百节轨道也不会卡。

为什么网格模板要带"地面高度"
------------------------------------------------
路堤放坡要一直落到地面（``style.GROUND_Y``），而网格是在**件局部坐标**下生成的。
件的世界 ``y`` 一变，"地面在局部坐标里有多高"就跟着变，放坡的断面也就不同。
旋转不影响它（绕 +Y 转不改变高度），所以缓存键只需要 ``(件 id, round(局部地面高度))``。

闭环检测也在这里
------------------------------------------------
每次同步都重跑一次 :func:`core.track.path.detect_closure`。它一次回答两个问题，
而这里**两个都要**：

* "能不能绕一圈" —— 驱动"闭环彩带"与"缺口告警柱"两个叠加显示，用户摆完最后一节
  就能立刻看到是否真的闭合；
* "能不能开车" —— :attr:`LayoutView.path` 是交给
  :class:`render.train_view.TrainView` 的唯一输入。没闭合成环时也要把**最长的那条
  开链**交出去（``allow_open=True``），否则"线还没铺完先开一段"就永远做不到。
"""

from __future__ import annotations

from panda3d.core import NodePath

from core.track.path import detect_closure
from render import overlay, style, track_mesh
from render.transform import apply_pose

#: 网格缓存键里"地面高度"的取整位数（毫米级足够，且能让缓存命中）。
_GROUND_KEY_DIGITS = 3


class LayoutView:
    """一个 :class:`Layout` 的可见表示。"""

    def __init__(self, layout, parent: NodePath, *, ground_y: float = style.GROUND_Y):
        self.layout = layout
        self.ground_y = ground_y

        self.root = parent.attachNewNode("layout")
        self._pieces = self.root.attachNewNode("pieces")
        self._ports = self.root.attachNewNode("ports")
        #: 空闲端口标记挂这一支上（每次 sync 整支重建），悬停高亮挂 ``_ports`` 上
        #: —— 分开两支，重建就不会误伤悬停高亮，见 :meth:`_rebuild_port_markers`。
        self._port_markers = self._ports.attachNewNode("markers")
        self._loop_root = self.root.attachNewNode("loop")
        self._gap_root = self.root.attachNewNode("gap")
        self._staging = self.root.attachNewNode("staging")
        self._staging.hide()

        #: 件下标 → 场景节点
        self._piece_nodes: dict[int, NodePath] = {}
        #: 件下标 → 网格缓存键
        self._node_keys: dict[int, tuple[str, float]] = {}
        #: 网格缓存键 → 模板节点
        self._templates: dict[tuple[str, float], NodePath] = {}
        #: 网格缓存键 → 三角形数（统计用）
        self._template_triangles: dict[tuple[str, float], int] = {}

        self._show_ports = True
        self._show_loop = True
        self._hover_index: int | None = None
        self._highlight_scale = None

        self.closure = None
        self.path = None

        self._port_template = overlay.build_port_marker(style.PORT_FREE_COLOR)
        self._hover_marker = overlay.build_port_marker(style.PORT_HOVER_COLOR,
                                                       scale=1.7)
        self._hover_marker.hide()
        self._hover_marker.reparentTo(self._ports)

        self._gap_marker = overlay.build_gap_marker()
        self._gap_marker.hide()
        self._gap_marker.reparentTo(self._gap_root)

        self._loop_node: NodePath | None = None

        # 构造即渲染：传进来的布局**立刻**可见。
        #
        # 这一条是后来补的，因为漏掉它的代价很隐蔽：``TrackEditor`` 在建好 view
        # 之后忘了调 ``sync()``，于是 ``main.py --open saves/circle.json`` 载入了
        # 8 节轨道、闭环检测也算出了 251.327 m，**画面上却什么都没有** ——
        # 数据、HUD、控制台全都正常，只有场景是空的。
        #
        # 与其在文档里写"别忘了 sync"，不如让"忘了"这件事不可能发生：
        # ``sync()`` 本来就是增量的、可重复调用的，多调这一次不花任何代价。
        self.sync()

    # ---------------------------------------------------------------- 网格

    @staticmethod
    def _ground_key(pose, ground_y: float) -> float:
        return round(ground_y - pose.y, _GROUND_KEY_DIGITS)

    def _template(self, def_id: str, ground_key: float) -> NodePath:
        key = (def_id, ground_key)
        template = self._templates.get(key)
        if template is None:
            piece = self.layout.catalog[def_id]
            builder = track_mesh.build_piece_mesh(piece, ground_local_y=ground_key)
            self._template_triangles[key] = builder.triangle_count
            template = builder.build()
            template.setName(f"tpl_{def_id}")
            self._templates[key] = template
        return template

    def piece_instance(self, def_id: str, ground_key: float,
                       parent: NodePath | None = None) -> NodePath:
        """拿一份该件网格的实例（共享 Geom）。幽灵预览也走这里。"""
        template = self._template(def_id, ground_key)
        return template.copyTo(parent if parent is not None else self._staging)

    # ---------------------------------------------------------------- 同步

    def sync(self) -> None:
        """把场景图对齐到当前 ``layout``。改过布局就调一次。"""
        layout = self.layout

        for index in [i for i in self._piece_nodes if i not in layout.pieces]:
            self._piece_nodes.pop(index).removeNode()
            self._node_keys.pop(index, None)

        for index, placed in layout.pieces.items():
            key = (placed.def_id, self._ground_key(placed.pose, self.ground_y))
            if self._node_keys.get(index) != key:
                old = self._piece_nodes.pop(index, None)
                if old is not None:
                    old.removeNode()
                node = self.piece_instance(placed.def_id, key[1], parent=self._pieces)
                node.setName(f"piece_{index}_{placed.def_id}")
                self._piece_nodes[index] = node
                self._node_keys[index] = key
            apply_pose(self._piece_nodes[index], placed.pose)

        if self._hover_index not in layout.pieces:
            self._hover_index = None

        self._rebuild_port_markers()
        self.recompute_closure()
        self._apply_highlight()

    def recompute_closure(self) -> None:
        """重跑闭环检测并刷新彩带 / 缺口柱，同时更新可供列车行驶的路径。

        ``allow_open=True``：没闭合成环时也把**最长的那条开链**当路径收下 ——
        编辑器要允许"线还没铺完就先开一段"。彩带仍然只在真闭环时画，所以
        "能不能绕一圈"与"能不能开车"这两件事不会互相混淆（见
        :func:`core.track.path.detect_closure` 的 ``allow_open``）。
        """
        report, path = detect_closure(self.layout, allow_open=True)
        self.closure = report
        self.path = path
        self._rebuild_loop()

    # ---------------------------------------------------------------- 叠加

    def _rebuild_port_markers(self) -> None:
        """重建空闲端口标记。

        只清 ``_port_markers`` 这一支 —— 悬停高亮挂在它的**兄弟**节点上，所以每次
        重建都不会把它一起扫掉。

        这里以前是"清空 ``_ports`` 的所有子节点，跳过 ``_hover_marker``"，而那个
        跳过是靠 ``child is self._hover_marker`` 判断的：Panda3D 的 ``getChildren()``
        每次返回**新的 Python 包装对象**，``is`` 永远是假。于是第一次 ``sync()``
        就把悬停标记删掉了 —— 现场看是"吸附目标的高亮不出现"，但日志里什么都没有。
        结构上分开（而不是靠判断跳过）才能让这件事不再取决于包装对象的身份。
        """
        for child in self._port_markers.getChildren():
            child.removeNode()
        if not self._show_ports:
            return
        for key in self.layout.free_ports():
            marker = self._port_template.copyTo(self._port_markers)
            marker.setName(f"port_{key[0]}_{key[1]}")
            apply_pose(marker, self.layout.world_port(*key))

    def _rebuild_loop(self) -> None:
        if self._loop_node is not None:
            self._loop_node.removeNode()
            self._loop_node = None

        if self._show_loop and self.closure is not None and self.closure.closed \
                and self.path is not None:
            self._loop_node = overlay.build_loop_ribbon(self.path)
            self._loop_node.reparentTo(self._loop_root)
            self._gap_marker.hide()
            return

        end = self.closure.open_end if self.closure is not None else None
        if self._show_loop and end is not None:
            apply_pose(self._gap_marker, end)
            self._gap_marker.show()
        else:
            self._gap_marker.hide()

    def set_show_ports(self, flag: bool) -> None:
        self._show_ports = flag
        self._rebuild_port_markers()

    def set_show_loop(self, flag: bool) -> None:
        self._show_loop = flag
        self._rebuild_loop()

    @property
    def show_ports(self) -> bool:
        return self._show_ports

    @property
    def show_loop(self) -> bool:
        return self._show_loop

    def set_hover_port(self, key) -> None:
        """把吸附目标标记挪到某个空闲端口上（``None`` 表示隐藏）。"""
        if key is None:
            self._hover_marker.hide()
            return
        index, port_id = key
        placed = self.layout.pieces.get(index)
        if placed is None or port_id not in self.layout.definition(index).port_ids:
            self._hover_marker.hide()
            return
        apply_pose(self._hover_marker, self.layout.world_port(index, port_id))
        self._hover_marker.show()

    def highlight(self, index: int | None) -> None:
        """高亮鼠标下的那一节轨道。"""
        if self._hover_index == index:
            return
        self._hover_index = index
        self._apply_highlight()

    def _apply_highlight(self) -> None:
        for index, node in self._piece_nodes.items():
            if index == self._hover_index:
                node.setColorScale(*style.HOVER_TINT)
            else:
                node.clearColorScale()

    # ---------------------------------------------------------------- 查询

    def node_for(self, index: int) -> NodePath | None:
        return self._piece_nodes.get(index)

    def triangle_count(self) -> int:
        total = 0
        for key in self._node_keys.values():
            total += self._template_triangles.get(key, 0)
        return total

    def bounds(self):
        """当前布局的世界包围盒（相机取景用）。"""
        return self.layout.bounds()
