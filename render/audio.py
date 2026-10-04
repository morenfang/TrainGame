"""列车音效：在运行时**程序化合成**几段循环声并随车速变速播放。

为什么合成而不是带音频文件
------------------------------------------------
与布景、轨道件同一个理由（见 :mod:`render.scenery`）：项目里**没有任何要下载或
随仓库提交的二进制资源**。列车行驶声是几段确定性的波形 —— 用 ``numpy`` 现算、
写成临时 WAV、交给 Panda3D 的音频管理器循环播放，既不用管版权、也不受"音源在哪"
这类外部依赖拖累。同一个种子每次合成出来的波形逐样本一致，因此可测、可复现。

声音设计
------------------------------------------------
三种声音，各自独立控制：

* **电机/行驶声**（循环）：低频轰鸣 + 逆变器高频成分。播放速率随车速升高，音量
  随车速变大 —— "越快越响、音调越高"。
* **轮轨声**（循环）：一段"哐当"的轮轨撞击声。只在行驶时出声，速率随车速加快。
* **鸣笛**（单次）：双音汽笛，按 ``J`` 触发。

音频库
------------------------------------------------
窗口模式下启用 OpenAL（Panda3D 自带），离屏 / 测试环境仍用 null 音频库 —— 此时
``loadSfx`` 返回零长度声音，本模块据此把自己关掉，绝不会在 headless 里炸。
"""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
from panda3d.core import Filename

#: 采样率（Hz）。
_RATE = 44100

#: 合成出的 WAV 落在这个临时目录下（按内容幂等：文件在就不再重算）。
_SFX_DIR = Path(tempfile.gettempdir()) / "train3d_sfx"

#: 行驶声的"参考速度"（km/h）：到这一速度时播放速率拉到最大。
_REF_SPEED = 300.0


def _write_wav(path: Path, samples: np.ndarray) -> None:
    """把 ``[-1, 1]`` 的浮点样本写成 16-bit 单声道 WAV。"""
    data = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(_RATE)
        f.writeframes(data.tobytes())


# --------------------------------------------------------------------------- #
# 波形合成（确定性：所有频率分量取整数 Hz，保证 1 s / 0.5 s 循环无缝）
# --------------------------------------------------------------------------- #

def _motor_wave() -> np.ndarray:
    """行驶声：低频轰鸣 + 高频逆变成分，6 Hz 缓慢起伏。1 秒无缝循环。"""
    duration = 1.0
    t = np.arange(int(_RATE * duration)) / _RATE
    sig = np.zeros_like(t)
    for freq, amp in ((46, 1.0), (92, 0.50), (138, 0.33),
                      (184, 0.22), (230, 0.15), (276, 0.10)):
        sig += amp * np.sin(2.0 * np.pi * freq * t)
    sig += 0.30 * np.sin(2.0 * np.pi * 380.0 * t)
    sig += 0.12 * np.sin(2.0 * np.pi * 570.0 * t)
    wobble = 0.72 + 0.28 * np.sin(2.0 * np.pi * 6.0 * t)
    sig = sig * wobble
    return sig / np.max(np.abs(sig)) * 0.80


def _wheels_wave() -> np.ndarray:
    """轮轨声：两次"哐当"（低通噪声 + 闷响），0.5 秒无缝循环。"""
    duration = 0.5
    n = int(_RATE * duration)
    sig = np.zeros(n)
    rng = np.random.RandomState(3)
    for pos in (0.0, 0.25):
        start = int(pos * _RATE)
        length = min(int(0.055 * _RATE), n - start)
        tt = np.arange(length) / _RATE
        noise = rng.randn(length)
        alpha = 0.12
        low = noise.copy()
        for i in range(1, length):
            low[i] = (1.0 - alpha) * low[i] + alpha * low[i - 1]
        click = (low * np.exp(-tt * 90.0)
                 + 0.5 * np.sin(2.0 * np.pi * 210.0 * tt) * np.exp(-tt * 55.0))
        sig[start:start + length] += click
    return sig / np.max(np.abs(sig)) * 0.85


def _horn_wave() -> np.ndarray:
    """双音汽笛：420 + 520 Hz，带轻微颤音，1.6 秒（起音快、收尾渐弱）。"""
    duration = 1.6
    n = int(_RATE * duration)
    t = np.arange(n) / _RATE
    sig = 0.55 * np.sin(2.0 * np.pi * 420.0 * t) \
        + 0.45 * np.sin(2.0 * np.pi * 520.0 * t)
    sig *= 1.0 + 0.03 * np.sin(2.0 * np.pi * 5.5 * t)
    env = np.ones(n)
    attack = int(0.04 * _RATE)
    release = int(0.30 * _RATE)
    env[:attack] = np.linspace(0.0, 1.0, attack)
    env[-release:] = np.linspace(1.0, 0.0, release)
    return sig * env * 0.80


def _ensure_sfx() -> dict[str, Path]:
    """把三段声音写到临时目录（已存在就不重写），返回名字 → 路径。"""
    specs = {
        "motor": _motor_wave,
        "wheels": _wheels_wave,
        "horn": _horn_wave,
    }
    paths: dict[str, Path] = {}
    for name, generator in specs.items():
        target = _SFX_DIR / f"{name}.wav"
        if not target.exists():
            _write_wav(target, generator())
        paths[name] = target
    return paths


# --------------------------------------------------------------------------- #
# 运行时
# --------------------------------------------------------------------------- #

class TrainAudio:
    """一列列车的声音：行驶声 + 轮轨声随车速变速，鸣笛按需触发。

    在 null 音频环境（离屏 / 测试）里自动退化成一个空壳 —— 所有方法都安全。
    """

    def __init__(self, base, *, enabled: bool = True):
        self.base = base
        self.enabled = enabled
        self._motor = None
        self._wheels = None
        self._horn = None
        self._ready = False
        self._active = False

    # ---------------------------------------------------------------- 装载

    def _load(self) -> bool:
        """第一次真正要用声音时才装载（惰性）—— 加载失败就关掉自己。

        ``_ready`` 记"装过了没"，``_active`` 记"是不是真的能出声"：null 音频
        环境下装过一次就永远返回 ``False``，不会每帧都重试。
        """
        if self._ready:
            return self._active
        self._ready = True
        self._active = False
        if not self.enabled:
            return False
        try:
            paths = _ensure_sfx()
        except Exception:                       # noqa: BLE001 - 合成失败就静音
            return False
        manager = getattr(self.base, "musicManager", None)
        if manager is None:
            return False
        try:
            self._motor = self.base.loader.loadSfx(Filename.fromOsSpecific(str(paths["motor"])))
            self._wheels = self.base.loader.loadSfx(Filename.fromOsSpecific(str(paths["wheels"])))
            self._horn = self.base.loader.loadSfx(Filename.fromOsSpecific(str(paths["horn"])))
        except Exception:                       # noqa: BLE001
            return False
        # null 音频库下 loadSfx 会返回零长度的空声音 —— 据此判断真的能出声吗。
        if self._motor.length() <= 0.0:
            self.enabled = False
            return False
        self._motor.setLoop(True)
        self._wheels.setLoop(True)
        self._motor.setVolume(0.0)
        self._wheels.setVolume(0.0)
        self._active = True
        return True

    # ---------------------------------------------------------------- 控制

    def start(self) -> None:
        """列车上线：开始放行驶声与轮轨声（音量先按静止状态置零）。"""
        if not self._load():
            return
        if self._motor.status() != 2:            # 2 = PLAYING
            self._motor.play()
        if self._wheels.status() != 2:
            self._wheels.play()

    def stop(self) -> None:
        """列车下轨：停掉两种循环声。"""
        if self._motor is not None:
            self._motor.stop()
        if self._wheels is not None:
            self._wheels.stop()

    def update(self, speed_kmh: float, throttle: float) -> None:
        """每帧调一次：按车速改播放速率与音量。``throttle`` 用于轻微强调加速。"""
        if not self._load():
            return
        v = max(0.0, min(speed_kmh, _REF_SPEED))
        frac = v / _REF_SPEED

        motor_rate = 0.55 + 1.25 * frac
        motor_volume = 0.18 + 0.50 * min(v / 200.0, 1.0)
        motor_volume += 0.10 * max(0.0, throttle)
        self._motor.setPlayRate(motor_rate)
        self._motor.setVolume(min(motor_volume, 1.0))

        # 轮轨声：停着不响，起步后随车速加快、变响。
        if v < 2.0:
            self._wheels.setVolume(0.0)
        else:
            self._wheels.setPlayRate(0.40 + 1.80 * frac)
            self._wheels.setVolume(min(0.10 + 0.45 * min(v / 200.0, 1.0), 0.75))

    def horn(self) -> None:
        """鸣笛（单次，重复调用即重新吹响）。"""
        if not self._load():
            return
        self._horn.setVolume(0.9)
        self._horn.play()

    def destroy(self) -> None:
        """退出 / 清场时停掉声音并释放引用（音频管理器按文件名缓存，重载很快）。"""
        self.stop()
        self._ready = False
        self._active = False
        self._motor = self._wheels = self._horn = None
