"""列车音效：在运行时**程序化合成**并随车速变速播放。

为什么合成而不是带音频文件
------------------------------------------------
与布景、轨道件同一个理由（见 :mod:`render.scenery`）：项目里**没有任何要下载或
随仓库提交的二进制资源**。行驶声是一段确定性的波形 —— 用 ``numpy`` 现算、写成
临时 WAV、交给 Panda3D 的音频管理器循环播放，既不用管版权、也不受"音源在哪"
这类外部依赖拖累。同一个种子每次合成出来的波形逐样本一致，因此可测、可复现。

两种声音
------------------------------------------------
* **行驶声**（循环）：一段"引擎" —— 低沉的牵引谐波堆 + 宽频轰鸣。播放速率与音量
  随车速升高，于是"越快越响、音调越高"。
* **风笛**（单次）：按 ``J`` 触发。刻意做成又低又厚的气笛，不是汽车喇叭。

为什么**没有**"嘀嘀"声
------------------------------------------------
曾经有两处会听成电子提示音，现在都拿掉了：

* **轮轨撞击声**：每 0.5 s 两声"哐当"，一提速就变成密集的嗒嗒嗒 —— 正好是
  "嘀嘀嘀"的听感。它已经删掉，行驶声音轨只剩引擎。
* **行驶声里的快速颤音与孤立纯音**：旧版在 46 Hz 的谐波堆上叠了 380 / 570 Hz
  两条纯正弦，还加了 6 Hz、深度 28% 的振幅调制。孤立纯音本身就"像提示音"，
  而每秒六下的振幅调制听起来就是**蜂鸣**而不是引擎。新版改用"整数倍频的谐波
  堆 + 多频点低通轰鸣"，只有一段 1 Hz、深度 5% 的缓慢起伏（那是呼吸，不是
  打点）。:func:`tests.test_audio.test_motor_wave_is_not_a_buzzer` 会把这条
  听感固化成断言，免得以后又被加回来。

频谱包络还有一个副产品：谐波堆是"音调感"的来源，播放速率一变，整段频谱跟着
平移，所以调速时听起来像在加速换挡，而不是把录音快放。

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

#: 合成结果的**修订号**：WAV 直接落在临时目录里按文件名复用，改了波形却不改这个
#: 数字的话，用户机器上 ``%TEMP%`` 里的旧 WAV 会被继续用 —— 表现为"改了没生效"，
#: 而且看起来完全不像缓存问题。**动过波形就 +1。**
_SFX_REVISION = 2

#: 合成出的 WAV 落在这个临时目录下（按文件名幂等：文件在就不再重算）。
_SFX_DIR = Path(tempfile.gettempdir()) / f"train3d_sfx_r{_SFX_REVISION}"

#: 行驶声的"参考速度"（km/h）：到这一速度时播放速率拉到最大。
_REF_SPEED = 300.0

#: 牵引谐波堆的基频（Hz）。取整数，保证 1 s 循环无缝。
_ENGINE_FUNDAMENTAL = 40.0


def _write_wav(path: Path, samples: np.ndarray) -> None:
    """把 ``[-1, 1]`` 的浮点样本写成 16-bit 单声道 WAV。"""
    data = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(_RATE)
        f.writeframes(data.tobytes())


def _band_noise(t: np.ndarray, low: float, high: float, decay: float,
                seed: int, count: int = 96) -> np.ndarray:
    """一段"像噪声"的宽频信号：整数 Hz 正弦按 ``1/f^decay`` 加权、随机相位叠加。

    为什么不用真随机噪声：白噪声接一个低通滤波，循环接头处必然有个台阶（听感上
    是"咔"）。而整数 Hz 的分量在 1 s 长度上**天然首尾相接**，一个样本都不差；
    几十条随机相位的分量叠起来，在听感上已经和白噪声没有区别。

    ``low`` / ``high`` 给出频段，``decay`` 越大越偏低频（越"闷"）。
    """
    rng = np.random.RandomState(seed)
    freqs = np.arange(int(low), int(high) + 1, max(1, int((high - low) / count) or 1))
    rng.shuffle(freqs)                     # 只取 count 条，均匀撒在频段里
    freqs = np.sort(freqs[:count]).astype(float)
    weights = 1.0 / np.power(freqs, decay)
    phases = rng.uniform(0.0, 2.0 * np.pi, freqs.size)

    signal = np.zeros_like(t)
    for freq, weight, phase in zip(freqs, weights, phases):
        signal += weight * np.sin(2.0 * np.pi * freq * t + phase)
    peak = np.max(np.abs(signal))
    return signal / peak if peak > 0.0 else signal


# --------------------------------------------------------------------------- #
# 波形合成（确定性：所有频率分量取整数 Hz，保证 1 s 循环无缝）
# --------------------------------------------------------------------------- #

def _motor_wave() -> np.ndarray:
    """行驶声 / 引擎：牵引谐波堆 + 宽频轰鸣，1 秒无缝循环。

    三层，都是"引擎"该有的东西，没有一层是提示音：

    1. **牵引谐波堆**（基频 40 Hz，1–14 次谐波，能量按 ``1/n^1.4`` 衰减）——
       有机件音调感，播放速率一提就像在升挡提速。
    2. **宽频轰鸣**（20–220 Hz，偏低频）—— 轮轨滚动、风噪、车体共振的那一层
       "厚"，没有它，上面那条就成了纯电子音。
    3. **一点点高频气流**（260–620 Hz，音量只有谐波堆的十几分之一）—— 速度感
       来自它，但不能多：孤立的纯高频最容易被听成"嘀"。
    """
    duration = 1.0
    t = np.arange(int(_RATE * duration)) / _RATE

    harmonic = np.zeros_like(t)
    for index in range(1, 15):
        freq = _ENGINE_FUNDAMENTAL * index
        harmonic += np.sin(2.0 * np.pi * freq * t) / index ** 1.4

    rumble = _band_noise(t, 20.0, 220.0, decay=0.6, seed=11)
    airflow = _band_noise(t, 260.0, 620.0, decay=0.5, seed=23, count=40)

    sig = harmonic / np.max(np.abs(harmonic))
    sig = sig + 0.70 * rumble + 0.07 * airflow
    # 1 Hz、深度 5% 的缓慢起伏 —— 一次呼吸两秒，不是每秒六下的打点。
    sig = sig * (0.95 + 0.05 * np.sin(2.0 * np.pi * 1.0 * t))
    return sig / np.max(np.abs(sig)) * 0.80


def _horn_wave() -> np.ndarray:
    """列车风笛：三支低音哨 + 气流噪声，1.8 秒（起音偏慢、收尾气散）。

    音高定为 311 / 370 / 466 Hz 的小三度和弦 —— 火车风笛用的是**低、暗**的和音；
    旧版 420 + 520 Hz 那一对太亮，听起来就是汽车喇叭的"嘀嘀"。起音留 90 ms 的
    渐强（气笛要靠气流吹起来），也没有了旧版那个 5.5 Hz 的快速颤音。
    """
    duration = 1.8
    n = int(_RATE * duration)
    t = np.arange(n) / _RATE

    sig = (0.50 * np.sin(2.0 * np.pi * 311.0 * t)
           + 0.34 * np.sin(2.0 * np.pi * 370.0 * t)
           + 0.16 * np.sin(2.0 * np.pi * 466.0 * t))
    sig += 0.10 * _band_noise(t, 180.0, 900.0, decay=0.4, seed=31, count=60)

    envelope = np.ones(n)
    attack = int(0.09 * _RATE)
    release = int(0.42 * _RATE)
    envelope[:attack] = np.linspace(0.0, 1.0, attack) ** 1.6
    envelope[-release:] = np.linspace(1.0, 0.0, release) ** 1.3
    return sig * envelope / np.max(np.abs(sig)) * 0.80


def _ensure_sfx() -> dict[str, Path]:
    """把两段声音写到临时目录（已存在就不重写），返回名字 → 路径。"""
    specs = {
        "motor": _motor_wave,
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
    """一列列车的声音：只有引擎（随车速变速）与风笛（按需触发）。

    在 null 音频环境（离屏 / 测试）里自动退化成一个空壳 —— 所有方法都安全。
    """

    def __init__(self, base, *, enabled: bool = True):
        self.base = base
        self.enabled = enabled
        self._motor = None
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
            self._motor = self.base.loader.loadSfx(
                Filename.fromOsSpecific(str(paths["motor"])))
            self._horn = self.base.loader.loadSfx(
                Filename.fromOsSpecific(str(paths["horn"])))
        except Exception:                       # noqa: BLE001
            return False
        # null 音频库下 loadSfx 会返回零长度的空声音 —— 据此判断真的能出声吗。
        if self._motor.length() <= 0.0:
            self.enabled = False
            return False
        self._motor.setLoop(True)
        self._motor.setVolume(0.0)
        self._active = True
        return True

    # ---------------------------------------------------------------- 控制

    def start(self) -> None:
        """列车上线：开始放行驶声（音量先按静止状态置低）。"""
        if not self._load():
            return
        if self._motor.status() != 2:            # 2 = PLAYING
            self._motor.play()

    def stop(self) -> None:
        """列车下轨：停掉循环声。"""
        if self._motor is not None:
            self._motor.stop()

    def update(self, speed_kmh: float, throttle: float) -> None:
        """每帧调一次：按车速改播放速率与音量。``throttle`` 用于轻微强调加速。"""
        if not self._load():
            return
        v = max(0.0, min(speed_kmh, _REF_SPEED))
        frac = v / _REF_SPEED

        motor_rate = 0.55 + 1.25 * frac
        motor_volume = 0.16 + 0.56 * min(v / 200.0, 1.0)
        motor_volume += 0.10 * max(0.0, throttle)
        self._motor.setPlayRate(motor_rate)
        self._motor.setVolume(min(motor_volume, 0.92))

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
        self._motor = self._horn = None
