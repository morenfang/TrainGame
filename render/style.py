"""渲染层的外观常量：世界尺寸、轨道断面、配色与烘焙光照。

与 core 的单位约定完全一致
------------------------------------------------
* 世界单位 = **米**；**+Y 向上**，地面为 XZ 平面；``heading`` 绕 +Y，从 +X 转向
  +Z 为正。
* 轨道件几何**不再缩放**。概要设计 §5.1：件库本身已按沙盘尺度取值
  （直轨 20 m、弯轨 R = 40 m、轨距 1.435 m）；``SCALE_MODEL ≈ 0.4`` 只作用于
  **列车与建筑**生成器。渲染层若再套一层缩放，列车与轨道就套不上了。

唯一的方向转换
------------------------------------------------
core 的 heading 与 Panda3D 的 HPR **符号相反**。实测（``coordinate-system yup``）：

    H(+90°) 把 +X 转到 -Z      而 core 要求 heading=+90° 把 +X 转到 +Z

所以全项目只有 :func:`render.transform.apply_pose` 一处做 ``H = -degrees(heading)``。
**其余任何地方都不得再手工翻转坐标** —— 否则会出现「轨道镜像」这类极难定位的 bug。

为什么把光照烘焙进顶点色
------------------------------------------------
不挂 Light、不开 ``setShaderAuto()``，而是在生成网格时就把 Lambert + 半球环境
光算进顶点颜色。理由：结果**确定**（离屏截图与窗口渲染逐像素一致）、**不依赖
显卡的着色器管线**、也省掉一整套光照对象的生命周期管理。代价是静态光照；
太阳方向变了重建网格即可（几十毫秒）。
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------- #
# 场地与地面
# --------------------------------------------------------------------------- #

#: 沙盘场地边长（米）。概要设计 D4：固定 400 m × 400 m 起步。
PLOT_SIZE = 400.0

#: 地面高度。轨面定为 ``y = 0``，道砟底面正好落在地面这一层。
GROUND_Y = -0.64

#: 网格间距：每 5 m 一条细线，每 25 m 一条粗线。
GRID_MINOR_STEP = 5.0
GRID_MAJOR_STEP = 25.0
#: 网格线抬离地面一点点，避免与地面共面时 z-fighting。
#:
#: 12 mm 是不够的：相机 near=1.5 / far=3000 时，24 位深度缓冲在 400 m 处的
#: 分辨率约 6 mm，留 50 mm 才有约 8 倍余量。50 mm 在画面上完全看不出来。
GRID_LIFT = 0.05

#: 相机可拉近 / 拉远的范围（米）。
CAMERA_MIN_DISTANCE = 6.0
CAMERA_MAX_DISTANCE = 900.0

#: 地平线天空色。**必须显式设置** —— Panda3D 默认清屏色是一块中灰
#: （#696969），在有远景的场景里会非常突兀。
#:
#: 带上 alpha 是刻意的：``LColor(*sky)`` 只接受 1 / 2 / 4 个分量，
#: 写三个分量会在运行时炸掉（而这条路径只在真正开窗口时才走到，
#: 所以离屏截图脚本一直没暴露它）。四个分量的元组谁都吃得下。
SKY_COLOR = (0.576, 0.667, 0.769, 1.0)
#: 指数雾密度（1/米）。让场地边界柔和淡出到天空色，而不是一条硬边。
FOG_DENSITY = 0.00055

#: 相机的近 / 远裁剪面。near 不能太小，否则远处深度精度不够会让网格线闪烁。
CAMERA_NEAR = 1.5
CAMERA_FAR = 3000.0

# --------------------------------------------------------------------------- #
# 轨道断面（局部坐标：x 沿轨道横向，y 从轨面向上为正）
# --------------------------------------------------------------------------- #
# 真实值：轨距 1.435 m、钢轨高 0.16 m、枕木 2.6 m × 0.26 m × 0.14 m。
# 下面几项在真实值基础上做了**视觉放大**（钢轨略宽、枕木略厚），因为一次看到
# 40 m 长的轨道时，0.07 m 宽的轨头只有一两个像素，放大后才看得出是"钢轨"。
# 放大只影响外观，不影响任何 core 层计算。

#: 标准轨距（真实值，与列车模型共用）。
GAUGE = 1.435
RAIL_HALF_GAUGE = GAUGE / 2.0

#: 轨面（列车走行面）= 局部 y = 0。
RAIL_TOP_Y = 0.0
RAIL_BOTTOM_Y = -0.18

#: 枕木顶面接钢轨底、底面坐落在道砟上。
TIE_TOP_Y = RAIL_BOTTOM_Y + 0.02
TIE_BOTTOM_Y = -0.34
TIE_HALF_LENGTH = 1.30
TIE_HALF_THICKNESS = 0.13
TIE_SPACING = 0.60

#: 道砟断面：顶面比枕木略宽，两侧放坡到地面。
BALLAST_TOP_Y = TIE_BOTTOM_Y
BALLAST_BOTTOM_Y = GROUND_Y
BALLAST_TOP_HALF_WIDTH = 2.05
BALLAST_BOTTOM_HALF_WIDTH = 2.70

#: 高出地面时路堤两侧的放坡系数（横 : 竖 = 1.5 : 1）。
EMBANKMENT_SLOPE = 1.5

#: 桥梁件：桥面换成混凝土、两侧加矮墙。
BRIDGE_PARAPET_HEIGHT = 0.55
BRIDGE_PARAPET_THICKNESS = 0.16

#: 钢轨工字断面的半个截面（局部横向 x, 高度 y），闭合折线，逆时针。
RAIL_SECTION = (
    (-0.100, RAIL_BOTTOM_Y),
    (0.100, RAIL_BOTTOM_Y),
    (0.100, RAIL_BOTTOM_Y + 0.030),
    (0.030, RAIL_BOTTOM_Y + 0.045),
    (0.030, RAIL_TOP_Y - 0.045),
    (0.055, RAIL_TOP_Y - 0.035),
    (0.055, RAIL_TOP_Y),
    (-0.055, RAIL_TOP_Y),
    (-0.055, RAIL_TOP_Y - 0.035),
    (-0.030, RAIL_TOP_Y - 0.045),
    (-0.030, RAIL_BOTTOM_Y + 0.045),
    (-0.100, RAIL_BOTTOM_Y + 0.030),
)

# --------------------------------------------------------------------------- #
# 配色（线性 sRGB，0..1）
# --------------------------------------------------------------------------- #

GROUND_COLOR = (0.286, 0.325, 0.259, 1.0)
GRID_MINOR_COLOR = (0.338, 0.376, 0.310, 1.0)
GRID_MAJOR_COLOR = (0.424, 0.463, 0.384, 1.0)
AXIS_X_COLOR = (0.612, 0.353, 0.353, 1.0)
AXIS_Z_COLOR = (0.353, 0.435, 0.635, 1.0)

BALLAST_COLOR = (0.451, 0.443, 0.427, 1.0)
TIE_COLOR = (0.310, 0.243, 0.192, 1.0)
TIE_COLOR_ALT = (0.278, 0.216, 0.169, 1.0)
RAIL_COLOR = (0.545, 0.565, 0.596, 1.0)
RAIL_HEAD_COLOR = (0.780, 0.796, 0.827, 1.0)
BRIDGE_COLOR = (0.545, 0.541, 0.522, 1.0)
BRIDGE_PARAPET_COLOR = (0.612, 0.608, 0.588, 1.0)
BUFFER_COLOR = (0.478, 0.463, 0.443, 1.0)
BUFFER_BEAM_COLOR = (0.706, 0.243, 0.208, 1.0)

#: 空闲端口标记（可以吸附的地方）
PORT_FREE_COLOR = (0.243, 0.780, 0.494, 1.0)
PORT_CENTER_COLOR = (0.145, 0.408, 0.278, 1.0)
#: 鼠标悬停的吸附目标
PORT_HOVER_COLOR = (1.000, 0.749, 0.208, 1.0)

#: 幽灵预览：能放 / 放不了
GHOST_OK_TINT = (0.451, 1.000, 0.749, 0.62)
GHOST_BLOCKED_TINT = (1.000, 0.333, 0.318, 0.62)

#: 悬停高亮：把顶点色整体提亮（>1 的分量会在输出时被截断，不改变色相）
HOVER_TINT = (1.34, 1.42, 1.34, 1.0)

#: 布景悬停高亮：一件半透明的占地轮廓（房/车站是矩形、山/树是圆）。
SCENERY_HOVER_COLOR = (1.000, 0.620, 0.231, 0.55)

#: 闭环彩带（沿可行驶闭环铺一条亮线）
LOOP_RIBBON_COLOR = (0.310, 0.741, 1.000, 0.85)
LOOP_RIBBON_HALF_WIDTH = 0.16
LOOP_RIBBON_LIFT = 0.03

#: 未闭环时末端缺口的告警颜色
GAP_WARN_COLOR = (1.000, 0.353, 0.278, 1.0)

# --------------------------------------------------------------------------- #
# 布景（山体 / 河流 / 湖泊 / 房屋 / 树木 / 草地 / 站台）
# --------------------------------------------------------------------------- #
# 布景**不参与**任何走线、碰撞或动力学计算，纯粹是外观（见 render/scenery.py）。
# 这里的数值只影响观感，改它们不会让列车跑偏。

#: 布景统一抬离地面这么高（米）。地面是一整块烘了色的平面，布景若与它共面就会
#: z-fighting。2 cm 在这个尺度下肉眼看不出，却稳稳地在深度缓冲里分了层。
SCENERY_LIFT = 0.02

#: 水面再抬一点（米）。河岸沙滩与水面是两层共面的带子，靠这个差值分层。
WATER_LIFT = 0.03

#: 轨道中心线两侧的净空（米）：布景的**占地轮廓**不得侵入这条走廊。
#:
#: 列车最宽处约 3.4 m（半宽 1.7 m），再留一点余量取 2.0 m。这就是"别把房子盖到
#: 轨道上"这条规矩的数值形式，`tests/test_scenery.py` 按它逐件检查 —— 于是
#: "某一座山正好压住环线"这种事故不用等截图，测试当场就能拦下来。
TRACK_CLEARANCE = 2.0

#: 山体按高度分层上色：山脚林地 → 岩石 → 高处岩石 → 雪线。
MOUNTAIN_FOOT_COLOR = (0.290, 0.365, 0.259, 1.0)
MOUNTAIN_ROCK_COLOR = (0.404, 0.396, 0.373, 1.0)
MOUNTAIN_ROCK_HIGH_COLOR = (0.510, 0.502, 0.486, 1.0)
MOUNTAIN_SNOW_COLOR = (0.878, 0.902, 0.933, 1.0)
#: 缓坡草丘（没有岩石和雪）。
HILL_COLOR = (0.376, 0.463, 0.290, 1.0)
#: 火山口深色熔岩 / 台地顶面风化岩。
VOLCANO_CRATER_COLOR = (0.220, 0.140, 0.110, 1.0)
MESA_TOP_COLOR = (0.560, 0.480, 0.360, 1.0)

#: 水体：水面、深水（河心 / 湖心），以及岸边的沙洲。
WATER_COLOR = (0.231, 0.408, 0.518, 1.0)
WATER_DEEP_COLOR = (0.145, 0.294, 0.416, 1.0)
SHORE_COLOR = (0.639, 0.600, 0.478, 1.0)

#: 草地：几档深浅不同的绿，铺在一起才有"一片草原"的层次（单色会像一块塑料板）。
MEADOW_COLORS = (
    (0.372, 0.470, 0.286, 1.0),
    (0.336, 0.443, 0.259, 1.0),
    (0.412, 0.502, 0.302, 1.0),
    (0.306, 0.412, 0.247, 1.0),
)
GRASS_TUFT_COLOR = (0.451, 0.549, 0.318, 1.0)

#: 树：树干与几档叶色。
TREE_TRUNK_COLOR = (0.294, 0.220, 0.157, 1.0)
TREE_LEAF_COLORS = (
    (0.243, 0.400, 0.208, 1.0),
    (0.290, 0.451, 0.227, 1.0),
    (0.196, 0.341, 0.176, 1.0),
    (0.322, 0.463, 0.216, 1.0),
)

#: 房屋：墙、屋顶、门、窗。真实房子的墙大多是浅色，屋顶才是有色的一层。
HOUSE_WALL_COLORS = (
    (0.855, 0.816, 0.729, 1.0),
    (0.780, 0.706, 0.596, 1.0),
    (0.706, 0.596, 0.510, 1.0),
    (0.753, 0.741, 0.706, 1.0),
    (0.831, 0.776, 0.663, 1.0),
)
HOUSE_ROOF_COLORS = (
    (0.478, 0.278, 0.212, 1.0),
    (0.400, 0.400, 0.408, 1.0),
    (0.549, 0.310, 0.243, 1.0),
    (0.353, 0.376, 0.400, 1.0),
)
HOUSE_DOOR_COLOR = (0.294, 0.208, 0.161, 1.0)
HOUSE_WINDOW_COLOR = (0.169, 0.216, 0.259, 1.0)

#: 站台：台面、台体、靠轨道那一侧的黄色安全线、灯柱。
PLATFORM_TOP_COLOR = (0.671, 0.663, 0.643, 1.0)
PLATFORM_COLOR = (0.612, 0.604, 0.580, 1.0)
PLATFORM_EDGE_COLOR = (0.902, 0.796, 0.310, 1.0)
LAMP_POST_COLOR = (0.267, 0.267, 0.278, 1.0)
LAMP_HEAD_COLOR = (1.000, 0.980, 0.880, 1.0)
#: 路灯光晕（贴地的暖色圆盘，假装有照明）
LAMP_GLOW_COLOR = (1.000, 0.920, 0.720, 1.0)

#: 板楼 / 公寓 / 别墅墙色
HOUSE_SLAB_WALL_COLOR = (0.720, 0.700, 0.670, 1.0)
HOUSE_SLAB_GLASS_COLOR = (0.380, 0.480, 0.560, 1.0)
HOUSE_BLOCK_WALL_COLORS = (
    (0.780, 0.720, 0.660, 1.0),
    (0.700, 0.730, 0.750, 1.0),
    (0.760, 0.700, 0.680, 1.0),
)
HOUSE_VILLA_WALL_COLORS = (
    (0.910, 0.880, 0.820, 1.0),
    (0.860, 0.840, 0.800, 1.0),
    (0.880, 0.820, 0.760, 1.0),
)

#: 车站（老式欧式古典）：奶油色石墙、陶土红坡顶、深棕窗棂与钟楼。
STATION_CLASSICAL_WALL_COLOR = (0.878, 0.843, 0.765, 1.0)
STATION_CLASSICAL_ROOF_COLOR = (0.522, 0.290, 0.224, 1.0)
STATION_CLASSICAL_TRIM_COLOR = (0.345, 0.286, 0.227, 1.0)
STATION_CLASSICAL_GLASS_COLOR = (0.353, 0.404, 0.478, 1.0)
STATION_CLOCK_FACE_COLOR = (0.953, 0.941, 0.890, 1.0)
STATION_CLOCK_HAND_COLOR = (0.169, 0.169, 0.176, 1.0)
#: 车站（新式现代）：灰白幕墙、钢灰平檐、玻璃幕墙带。
STATION_MODERN_WALL_COLOR = (0.820, 0.835, 0.855, 1.0)
STATION_MODERN_ROOF_COLOR = (0.478, 0.506, 0.549, 1.0)
STATION_MODERN_GLASS_COLOR = (0.329, 0.471, 0.565, 1.0)
STATION_MODERN_STEEL_COLOR = (0.345, 0.376, 0.412, 1.0)

#: 高楼：混凝土墙面、成排玻璃幕墙、楼顶女儿墙。
HOUSE_TOWER_WALL_COLOR = (0.635, 0.655, 0.682, 1.0)
HOUSE_TOWER_GLASS_COLOR = (0.302, 0.424, 0.522, 1.0)
HOUSE_TOWER_ROOF_COLOR = (0.420, 0.439, 0.463, 1.0)
#: 商场：浅色外墙 + 整面玻璃橱窗 + 门头招牌。
HOUSE_MALL_WALL_COLOR = (0.792, 0.757, 0.682, 1.0)
HOUSE_MALL_GLASS_COLOR = (0.286, 0.451, 0.557, 1.0)
HOUSE_MALL_SIGN_COLOR = (0.820, 0.302, 0.227, 1.0)
#: 便利店：浅色小门脸 + 前檐遮阳篷 + 招牌。
HOUSE_SHOP_WALL_COLOR = (0.851, 0.827, 0.757, 1.0)
HOUSE_SHOP_AWNING_COLOR = (0.800, 0.345, 0.275, 1.0)
HOUSE_SHOP_SIGN_COLOR = (0.165, 0.322, 0.427, 1.0)

#: 棕榈树：棕褐树干 + 深绿叶。
TREE_PALM_TRUNK_COLOR = (0.424, 0.333, 0.239, 1.0)
TREE_PALM_FROND_COLOR = (0.247, 0.447, 0.204, 1.0)

#: 高架桥墩（``mesh: "deck"`` 的件架空时立在下面）。断面是按道砟顶面选的：
#: 必须**收在桥面轮廓以内**，桥墩才不会从桥面边缘探出来。
BRIDGE_PIER_HALF_LENGTH = 0.90
BRIDGE_PIER_HALF_WIDTH = 1.70

# --------------------------------------------------------------------------- #
# 列车
# --------------------------------------------------------------------------- #
#: 车厢玻璃。压得很深、偏冷，在明亮的车身上一眼就能看出"这里是窗"。
TRAIN_GLASS_COLOR = (0.086, 0.114, 0.145, 1.0)
#: 车窗下沿的浅色饰条（真实客车窗下都有一条，能把窗带和腰带分开）。
TRAIN_TRIM_COLOR = (0.847, 0.859, 0.878, 1.0)
#: 裙板 / 车下：车身色的压暗版由 :func:`darken` 现场算，这里只给内燃机车那种
#: 本来就偏深的转向架与车钩。
TRAIN_BOGIE_COLOR = (0.145, 0.145, 0.157, 1.0)
TRAIN_WHEEL_COLOR = (0.098, 0.098, 0.106, 1.0)
TRAIN_UNDERFRAME_COLOR = (0.196, 0.196, 0.208, 1.0)
#: 前照灯 / 尾灯
TRAIN_HEADLIGHT_COLOR = (1.000, 0.973, 0.831, 1.0)
TRAIN_TAILLIGHT_COLOR = (0.859, 0.176, 0.153, 1.0)

#: 车门扇：车身色压暗到多少。真实车门的颜色和车身是同一桶漆，差别只来自
#: 门板与门框的朝向（门略微内凹），所以这里只压一点点 —— 压多了像贴了块补丁。
TRAIN_DOOR_TINT = 0.86
#: 涂装里没写 ``roof`` 时的兜底车顶色。
TRAIN_ROOF_FALLBACK = (0.647, 0.667, 0.686, 1.0)
#: 端面色（车钩面 / 鼻锥端面）。比车身暗一档，好让两节车之间有一条分界。
TRAIN_END_TINT = 0.72

#: 走行部尺寸（米）。轴距等由 ``CarSpec.bogie_half_spacing`` 决定，这里只放
#: 车轮与转向架构架本身的尺寸。车轮半径由**剖面标注的车高**推出，所以换车型
#: 不需要改这些数。
TRAIN_WHEEL_RADIUS = 0.41
#: 车轮踏面宽度（沿车轴方向）。
TRAIN_WHEEL_THICKNESS = 0.10
#: 同一转向架内两根车轴到转向架中心的距离。
TRAIN_AXLE_HALF_SPACING = 0.80
#: 转向架构架：横向占据的半宽（必须**小于**车轮内侧面，否则构架会穿进轮子里）。
TRAIN_BOGIE_FRAME_HALF_WIDTH = 0.60
TRAIN_BOGIE_FRAME_TOP_Y = 0.62
TRAIN_BOGIE_FRAME_BOTTOM_Y = 0.26
#: 车轮圆周的分段数。12 段在 40 m 外看不出多边形，三角形也省。
TRAIN_WHEEL_SIDES = 12


def hex_to_color(value: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """``"#2F5D3A"`` → ``(0.184, 0.365, 0.227, alpha)``。

    涂装写在 ``data/trains.json`` 里，用十六进制比三个浮点数好写也好核对
    （拿取色器量真实照片的色值，直接填进来就行）。``#`` 可省，三位缩写
    （``#abc``）也认 —— 但**只在真的写错时要报错**，不能悄悄返回个黑色：
    一个打错的色值会表现成"这节车少了一块涂装"，很难回溯到数据里。
    """
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"{value!r} 不是 6 位（或 3 位缩写）的十六进制颜色")
    try:
        channels = [int(text[i:i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        raise ValueError(f"{value!r} 含有非十六进制字符") from None
    return (channels[0] / 255.0, channels[1] / 255.0, channels[2] / 255.0, alpha)


def darken(color, factor: float):
    """把颜色整体压暗（保持色相与 alpha）。用来从车身色推出裙板色。"""
    return (color[0] * factor, color[1] * factor, color[2] * factor, color[3])


def mix(a, b, t: float):
    """线性混色，``t = 0`` 取 ``a``、``t = 1`` 取 ``b``。"""
    t = max(0.0, min(1.0, t))
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))


# --------------------------------------------------------------------------- #
# 烘焙光照
# --------------------------------------------------------------------------- #

#: 指向太阳的单位向量（+Y 向上）。
_SUN_RAW = (-0.40, 0.86, 0.32)
_SUN_LEN = math.sqrt(sum(c * c for c in _SUN_RAW))
SUN_DIR = tuple(c / _SUN_LEN for c in _SUN_RAW)

#: 太阳光的颜色（略偏暖）。
SUN_COLOR = (1.000, 0.965, 0.902)
#: 半球环境光：朝上的面吃天光（偏冷），朝下的面吃地面反光（偏暖）。
AMBIENT_SKY = (0.520, 0.545, 0.590)
AMBIENT_GROUND = (0.380, 0.360, 0.330)
#: 太阳光的强度（环境光不额外乘系数）。
SUN_STRENGTH = 0.82


def shade(normal, color):
    """把一把法线 + 固有色烘焙成最终顶点色（Lambert + 半球环境光）。

    ``normal`` 必须是单位向量。返回 ``(r, g, b, a)``，alpha 原样透传 ——
    这样标记类半透明几何也能直接走同一条路径。
    """
    nx, ny, nz = normal
    lambert = nx * SUN_DIR[0] + ny * SUN_DIR[1] + nz * SUN_DIR[2]
    if lambert < 0.0:
        lambert = 0.0
    lambert *= SUN_STRENGTH

    # 半球插值：ny = +1 取天光，ny = -1 取地面反光
    up = 0.5 + 0.5 * ny
    amb_r = AMBIENT_GROUND[0] + (AMBIENT_SKY[0] - AMBIENT_GROUND[0]) * up
    amb_g = AMBIENT_GROUND[1] + (AMBIENT_SKY[1] - AMBIENT_GROUND[1]) * up
    amb_b = AMBIENT_GROUND[2] + (AMBIENT_SKY[2] - AMBIENT_GROUND[2]) * up

    return (
        color[0] * (amb_r + lambert * SUN_COLOR[0]),
        color[1] * (amb_g + lambert * SUN_COLOR[1]),
        color[2] * (amb_b + lambert * SUN_COLOR[2]),
        color[3],
    )
