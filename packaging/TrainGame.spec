# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：把「模拟铁轨」做成一个能双击的 exe。

怎么用
------------------------------------------------
推荐走 :file:`scripts/build_exe.py`（它会顺手清缓存、生成图标、复制示例存档、
打印体积）。想直接调也行::

    pyinstaller packaging/TrainGame.spec

做单文件还是单目录
------------------------------------------------
默认 **单文件**（``TrainGame.exe``，双击即玩，符合"给我一个 exe"）。想要启动更快、
更好排查的目录版，设环境变量::

    set TRAIN3D_ONEFILE=0

单文件的代价是每次启动都要把上百 MB 解到临时目录（实测启动慢几秒），换来的是
"就一个文件"。

这个 spec 里真正要紧的三件事
------------------------------------------------
1. **panda3d 的 DLL 要挑着打。** 全量是 98 MB，其中约 40 MB 是我们这条代码路径
   永远不会加载的插件。但**不能照名字猜**：``libpanda.dll`` 与 ``libpandagl.dll``
   都静态依赖 ``cg.dll``（NVIDIA 的 Cg 着色器运行时，11.6 MB）—— 我们的代码一行
   Cg 都没写，删掉它却是 OpenGL 管道加载失败、游戏直接起不来。判断依据只能是对
   二进制做依赖检查（``packaging/check_dlls.py`` 就是干这个的）。
2. **图形/音频管道不是 import 进来的**，是 Panda3D 按名字 ``LoadLibrary`` 的，
   所以 ``libpandagl.dll`` / ``libp3openal_audio.dll`` 必须显式列进来（打包器静态
   分析看不见），并且要靠 :file:`packaging/rthook_dlls.py` 把目录塞进搜索路径。
3. **``data/*.json`` 要用相对布局打进 ``data/``。** 代码里读它的路径是
   ``core.paths.data_dir()``（冻结时 = 解包目录 / data），布局对不上就会在启动
   时报"找不到件库"。**可写的存档则绝不能进这儿** —— 解包目录退出即删。
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# --------------------------------------------------------------------------- #
# 目录
# --------------------------------------------------------------------------- #

ROOT = Path(SPECPATH).resolve().parent          # 仓库根目录
ONEFILE = os.environ.get("TRAIN3D_ONEFILE", "1") != "0"

#: 打包期的小工具（PE 依赖分析）。spec 会被 PyInstaller 当脚本执行，
#: 所以这里手动把 packaging/ 加进搜索路径。
sys.path.insert(0, str(Path(SPECPATH)))
import dllclosure                                # noqa: E402

import panda3d                                   # noqa: E402  （要拿到它的安装位置）

PANDA_DIR = Path(panda3d.__file__).resolve().parent

# --------------------------------------------------------------------------- #
# 二进制：panda3d 的部分要自己点名
# --------------------------------------------------------------------------- #

#: 要打进去的 panda3d 二进制（相对 ``panda3d/`` 目录的 glob）。
#:
#: 每一条都有存在的理由，而**没列进来的**那些同样有理由（见下面的注释表）：
#: ``core.*.pyd`` / ``direct.*.pyd`` 是 Python 绑定本身；``libpanda*.dll`` 是引擎；
#: ``libpandagl`` 与 ``libp3openal_audio`` 是运行时按名字加载的图形/音频管道；
#: ``cg.dll`` 是被上面前两个静态依赖的；``MSVCP140`` 与 ``api-ms-win-crt-*`` 是
#: 运行库（目标机器不一定有，跟着走最省事）。
_PANDA_BINARIES = (
    "core.*.pyd",
    "direct.*.pyd",
    "libpanda.dll",
    "libpandaexpress.dll",
    "libp3dtool.dll",
    "libp3dtoolconfig.dll",
    "libp3interrogatedb.dll",
    "libp3direct.dll",
    "libp3windisplay.dll",
    "libpandagl.dll",
    "libp3openal_audio.dll",
    "cg.dll",
    "cgGL.dll",
    "MSVCP140.dll",
    "api-ms-win-crt-*.dll",
)

#: **故意不打**的 panda3d 二进制，以及各自为什么（别"顺手"加回来，也别照这个
#: 名单反着删 —— 每一份都是量过的）：
#:
#: ==========================  ======  ========================================
#: 文件                         体积    为什么用不上
#: ==========================  ======  ========================================
#: avcodec/avformat/avutil/    22 MB   视频与音频文件解码（``libp3ffmpeg``）。
#: swscale/swresample                 我们的音效是 NumPy 现算的 WAV，不读媒体文件
#: d3dx9_43 + libpandadx9      2.8 MB  DirectX9 管道；``Config.prc`` 里只有 pandagl
#: fmodex64 + libp3fmod_audio  1.5 MB  FMOD 音频后端；配置里只有 p3openal_audio
#: libp3assimp                 5.2 MB  Assimp 模型加载器（我们一个模型文件都不读）
#: libpandaegg + egg.pyd       2.3 MB  egg 模型加载器（同上；几何全是代码算的）
#: libpandabullet/ode/physics   5 MB   物理引擎。运动学在 ``core.train.dynamics`` 里
#: libpandaai                  0.3 MB  AI 寻路
#: libp3tinydisplay            9.7 MB  软件光栅化兜底管道。目标机没有 OpenGL 时它
#:                                     也跑不动这个场景，不如把话说清楚
#: libp3vision/vrpn/skel/fx     0.7 MB  视觉、VR、骨骼、特效
#: ==========================  ======  ========================================
#:
#: 另外 ``panda3d_tools``（51 MB，egg2bam / interrogate / deploy-stub 一堆构建
#: 工具）与遗留绑定 ``pandac``（8.6 MB）是**构建期**的东西，也在 excludes 里。

binaries = []
for pattern in _PANDA_BINARIES:
    for src in sorted(PANDA_DIR.glob(pattern)):
        binaries.append((str(src), "panda3d"))

#: panda3d 之外、但**必须**跟着走的 DLL。全都来自 Anaconda 的 ``Library\\bin\\``，
#: 而 PyInstaller 只在"二进制自己旁边"和 ``sys.path`` 上找依赖 —— 两处都没有，
#: 于是它**不报错、也不打**。漏掉哪个，就是哪个 stdlib 扩展模块在 import 时抛
#: "DLL load failed"。这几条是 :file:`check_dlls.py --bundle` 一条条报出来的，
#: 不是猜的；名字写的是**导入表里的名字**（``ffi.dll`` 而不是 ``libffi-8.dll``，
#: Windows 的 ``LoadLibrary`` 认这个）。
#:
#: ``libcrypto`` / ``libssl`` 是跟着 ``_hashlib`` / ``_ssl`` 来的，占了这堆里
#: 绝大部分体积（约 8 MB）。想省这点空间就得把 ``_ssl`` 排除掉（本游戏不开任何
#: 网络连接），但那属于"为了 1 MB 砍掉一个标准库模块"，眼下不值得。
_EXTRA_DLLS = (
    "ffi.dll",                # _ctypes —— exe 的报错对话框要用它
    "liblzma.dll",            # _lzma
    "libbz2.dll",             # _bz2
    "libexpat.dll",           # pyexpat
    "libcrypto-3-x64.dll",    # _hashlib / _ssl
    "libssl-3-x64.dll",       # _ssl
)

for _name in _EXTRA_DLLS:
    _src = dllclosure.find_dll(_name)
    if _src is None:
        raise SystemExit(
            f"找不到 {_name}（ctypes 的依赖）。没有它打包后 import ctypes 会失败，"
            f"而 exe 的报错对话框正要用它。")
    binaries.append((str(_src), "."))

# Python 自己那些扩展模块的依赖，PyInstaller 大多能找齐；找不到时不报错、只是默默
# 不打。这里再按**导入表**把上面这批二进制的依赖补成闭包，并打印补了什么，方便
# 复核（打包之后 scripts/build_exe.py 还会拿真清单再核一遍）。
binaries, _notes = dllclosure.complete_binaries(binaries)
for _note in _notes:
    print(f"[依赖] {_note}")

# --------------------------------------------------------------------------- #
# 数据：panda3d 的 prc 与 models + 我们的件库
# --------------------------------------------------------------------------- #

#: ``etc/*.prc`` 必须在 —— 它就是 ``load-display pandagl`` /
#: ``audio-library-name p3openal_audio`` / 模型搜索路径的来源，少了它 Panda3D 会
#: 用编译进去的默认值，表现成"画面能出但配置全不对"。
#: ``models/`` 是自带的默认字体（``cmss12.egg.pz``）：界面用系统中文雅黑，
#: 万一某个 Windows 上一个中文字体都没有，TextNode 会退回它。
_BINARY_SUFFIXES = (".dll", ".pyd", ".so", ".dylib")
datas = [
    (src, dest)
    for src, dest in collect_data_files("panda3d")
    # collect_data_files 会把包目录下的 DLL 也当"数据"收进来，而它们已经作为
    # binaries 打过了 —— 不去重的话每份 DLL 在 exe 里存两遍（约 50 MB 白给）。
    if not src.lower().endswith(_BINARY_SUFFIXES)
]

# 件库与列车目录：布局必须是 ``data/``，因为 core.paths.data_dir() 就是这么拼的
for src in sorted((ROOT / "data").glob("*.json")):
    datas.append((str(src), "data"))

# --------------------------------------------------------------------------- #
# 模块
# --------------------------------------------------------------------------- #

#: 游戏自己需要的顶层包（PyInstaller 从入口脚本顺着 import 也能找到，这里显式
#: 列出来是为了让"改错了立刻报错"，而不是等到运行时某个动态导入悄悄失败）。
hiddenimports = [
    "panda3d.core",
    "panda3d.direct",
    "panda3d.dtoolconfig",
    "direct.showbase.ShowBase",
    "direct.directbase.DirectStart",
    "direct.gui.OnscreenText",
    "direct.task",
]

#: ``direct`` 里**不**收子模块的包。它们要么会拖进别的 GUI 工具包（tk / wx），
#: 要么是 Panda3D 的编辑器与分布式网络（本游戏一个都不用）：
#:
#: * tk / wx 系 —— 拖进 tkinter、wxPython，光这一项就是几十 MB；
#: * ``distributed`` / ``cluster`` / ``directdServer`` —— Panda3D 的联网同步；
#: * ``leveleditor`` / ``directtools`` —— 引擎自带的场景编辑器；
#: * ``p3d`` / ``plugin_*`` —— 已废弃的 web 插件运行时。
_DIRECT_SKIP = (
    "direct.tkwidgets", "direct.tkpanels", "direct.wxwidgets",
    "direct.distributed", "direct.cluster", "direct.directdServer",
    "direct.directdevices", "direct.leveleditor", "direct.directtools",
    "direct.p3d", "direct.plugin", "direct.doc", "direct.dcparse",
)
hiddenimports += [
    name for name in collect_submodules("direct")
    if not name.startswith(_DIRECT_SKIP)
]

#: 不收的东西。清单里每一条都是"体积大、且这条代码路径用不到"：
#:
#: * ``PIL`` / ``matplotlib`` / ``fontTools`` / ``pygments`` —— 只被
#:   ``scripts/manual_pdf.py`` 与 ``scripts/preview_layout.py`` 这两个**构建期**
#:   脚本用到（生成操作手册 PDF 与布局预览图），游戏本体一行都不 import；
#: * ``panda3d_tools``（51 MB）与 ``pandac``（8.6 MB）—— 见上面的二进制注释；
#: * ``panda3d`` 的几个 C++ 子系统 —— 对应的 DLL 本来就没打，收了也只会在运行
#:   时报"找不到 DLL"，不如现在就断掉；
#: * 科学栈与 Jupyter —— Anaconda 环境里装着，但与本游戏无关。
excludes = [
    "PIL", "Pillow", "matplotlib", "fontTools", "pygments",
    "panda3d_tools", "pandac", "pytesseract",
    "panda3d.bullet", "panda3d.physics", "panda3d.ode", "panda3d.ai",
    "panda3d.egg", "panda3d.skel", "panda3d.vision", "panda3d.vrpn",
    "panda3d.fx", "panda3d._rplight", "panda3d.assimp",
    "scipy", "pandas", "IPython", "jupyter", "notebook",
    "tkinter", "wx",
]

a = Analysis(
    [str(ROOT / "app" / "winmain.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(Path(SPECPATH) / "rthook_dlls.py")],
    excludes=excludes,
    # 别把源码塞进 exe：一是省体积，二是 ``sys._MEIPASS`` 里的 .py 会让
    # "从 __file__ 往上翻几层找 data/" 这种写法看着像能用（实际指向临时目录）。
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

_icon = Path(SPECPATH) / "TrainGame.ico"
_common = dict(
    name="TrainGame",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,              # UPX 压出来的 DLL 偶发加载失败，得不偿失
    console=False,          # 双击不弹黑框；报错进日志 + MessageBox（app/winmain.py）
    disable_windowed_traceback=False,
    icon=str(_icon) if _icon.exists() else None,
)

if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        exclude_binaries=False,
        **_common,
    )
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **_common)
    COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="TrainGame")
