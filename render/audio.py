"""列车音效：在运行时**程序化合成**并随车速变速播放。

为什么合成而不是带音频文件
------------------------------------------------
与布景、轨道件同一个理由（见 :mod:`render.scenery`）：项目里**没有任何要下载或
随仓库提交的二进制资源**。行驶声是一段确定性的波形 —— 用 ``numpy`` 现算、写成
临时 WAV、交给 Panda3D 的音频管理器循环播放，既不用管版权、也不受"音源在哪"
这类外部依赖拖累。同一个种子每次合成出来的波形逐样本一致，因此可测、可复现。

两种行驶声 + 一支风笛
------------------------------------------------
* **引擎声**（循环）：一段低沉的牵引谐波堆 + 宽频轰鸣。播放速率与音量随车速升高，
  于是"越快越响、音调越高"。
* **轮轨滚动声**（循环）：**连续**的低频"沙沙"，只在动起来之后出声、随车速变响变亮。
* **风笛**（单次）：按 ``J`` 触发。刻意做成又低又厚的气笛，不是汽车喇叭。

"哐当"和"嘀嘀"是两回事，别再把前者加回来
------------------------------------------------
用户说过"不要嘀嘀声，只要引擎声"，之后又补了一句"轮轨声可以回来，但不能像打点"。
这两句话合起来要求的是**把瞬态拿掉、把连续性留住**，所以：

* **删掉的是"撞击"**：旧版轮轨声在 0.5 s 里放两声"哐当"，每次撞击都是一个瞬态
  尖峰；速度一快，撞击间隔缩短，听起来就是密集的嗒嗒嗒 —— 那正是"嘀嘀"的第二个
  来源。它**不该回来**，因为问题不在音量、在波形形状。
* **加回来的是"滚动"**：新的轮轨声是一段**没有瞬态的连续底噪**（低频滚动 + 一丝
  轮缘摩擦）。它没有可以加密的"点"，所以快放只会变亮变响，不会变出一串打点。

另外两处旧版会听成电子提示音的地方也一并处理了：引擎声里曾经叠了 380 / 570 Hz
两条孤立纯正弦，以及一个 6 Hz、深度 28% 的颤音（每秒六下的振幅调制 = 嗡鸣）。
新版改用"整数倍频的谐波堆 + 多频点低通轰鸣"，只留一段 1 Hz、深度 5% 的缓慢起伏
（那是呼吸，不是打点）。`tests/test_audio.py` 把这两条听感都钉成了断言，并且每条
断言都配一个**故意做坏的对照样本**（旧版颤音 / 撞击声）来校验判据本身有效 ——
控制组必须被判超标，被测波形必须远低于它。

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
#: r1 → r2：去掉轮轨"哐当"声与引擎里的纯音 / 颤音
#: r2 → r3：轮轨声以"连续滚动"的形式加回来
_SFX_REVISION = 3

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


def _roll_wave() -> np.ndarray:
    """轮轨滚动声：**连续**的低频"沙沙"，1 秒无缝循环。

    这是旧版"哐当"声的替代品，两者的区别只有一件事 —— **有没有瞬态**：

    * 旧版每次撞击是一个瞬态尖峰（低通噪声 × 指数衰减 + 一段 210 Hz 闷响），
      0.5 s 里放两下。速度一快撞击间隔缩短，就变成密集的嗒嗒嗒 / 嘀嘀嘀。
    * 这里没有"点"可以加密：整段就是一条平稳的带限噪声，只有频段随车速整体
      平移（快放 = 变亮变响）。所以它怎么放都不会变成打点。

    两层：低频滚动（70–420 Hz，衰减缓，是主要的那层"轰"）加上一丝轮缘摩擦的
    高频沙沙（700–1600 Hz，音量只有十分之一）。高频那层刻意压得很低 ——
    孤立的、突出的高频最容易被听成电子音（见引擎声里删掉 380 / 570 Hz 那条）。
    """
    duration = 1.0
    t = np.arange(int(_RATE * duration)) / _RATE

    rolling = _band_noise(t, 70.0, 420.0, decay=0.35, seed=41)
    friction = _band_noise(t, 700.0, 1600.0, decay=0.5, seed=47, count=48)

    sig = rolling + 0.10 * friction
    return sig / np.max(np.abs(sig)) * 0.55


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
    """把三段声音写到临时目录（已存在就不重写），返回名字 → 路径。"""
    specs = {
        "motor": _motor_wave,
        "roll": _roll_wave,
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
    """一列列车的声音：引擎 + 轮轨滚动（随车速变速）与风笛（按需触发）。

    在 null 音频环境（离屏 / 测试）里自动退化成一个空壳 —— 所有方法都安全。
    """

    def __init__(self, base, *, enabled: bool = True):
        self.base = base
        self.enabled = enabled
        self._motor = None
        self._roll = None
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
            self._roll = self.base.loader.loadSfx(
                Filename.fromOsSpecific(str(paths["roll"])))
            self._horn = self.base.loader.loadSfx(
                Filename.fromOsSpecific(str(paths["horn"])))
        except Exception:                       # noqa: BLE001
            return False
        # null 音频库下 loadSfx 会返回零长度的空声音 —— 据此判断真的能出声吗。
        if self._motor.length() <= 0.0:
            self.enabled = False
            return False
        self._motor.setLoop(True)
        self._roll.setLoop(True)
        self._motor.setVolume(0.0)
        self._roll.setVolume(0.0)
        self._active = True
        return True

    # ---------------------------------------------------------------- 控制

    def start(self) -> None:
        """列车上线：开始放行驶声（音量先按静止状态置低）。"""
        if not self._load():
            return
        for track in (self._motor, self._roll):
            if track.status() != 2:              # 2 = PLAYING
                track.play()

    def stop(self) -> None:
        """列车下轨：停掉两种循环声。"""
        for track in (self._motor, self._roll):
            if track is not None:
                track.stop()

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

        # 轮轨滚动声：停着不响（"停着只有很轻的怠速声"），起步后随车速变响变亮。
        # 音量上限刻意压得比引擎低一截 —— 它是底噪，不该抢到前面来。
        if v < 2.0:
            self._roll.setVolume(0.0)
        else:
            roll_volume = 0.06 + 0.20 * min(v / 200.0, 1.0)
            self._roll.setPlayRate(0.75 + 0.45 * frac)
            self._roll.setVolume(min(roll_volume, 0.26))

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
        self._motor = self._roll = self._horn = None
