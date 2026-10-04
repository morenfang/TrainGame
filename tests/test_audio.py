"""列车音效：波形合成与 null 音频下的安全退化。

音效在运行时程序化合成（见 ``render/audio.py``），这里只查两条：

1. 合成出来的波形**确定、非空、长度正确**（无缝循环靠整数 Hz 分量保证，
   所以长度必须正好是整段循环）；
2. ``TrainAudio`` 在 null 音频环境（测试的 ``app`` fixture 就是）里照常安全
   no-op —— 任何调用都不该抛异常，也不该每帧都去重试装载。
"""

from __future__ import annotations

import numpy as np

from render import audio as audio_mod


# --------------------------------------------------------------------------- #
# 波形合成
# --------------------------------------------------------------------------- #

def test_waveforms_are_deterministic_and_nonempty():
    for generator, seconds in ((audio_mod._motor_wave, 1.0),
                               (audio_mod._wheels_wave, 0.5),
                               (audio_mod._horn_wave, 1.6)):
        first = generator()
        second = generator()
        assert first.shape == second.shape
        assert np.array_equal(first, second), "合成必须是确定性的（可复现）"
        assert np.max(np.abs(first)) > 0.1, "波形不能是空信号"
        assert first.shape[0] == int(audio_mod._RATE * seconds)


def test_motor_wave_has_rumble_and_inverter_energy():
    """行驶声该同时有低频轰鸣（电机/齿轮）与高频逆变成分，才像"动车在跑"。"""
    wave = audio_mod._motor_wave()
    spectrum = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(len(wave), d=1.0 / audio_mod._RATE)
    low = spectrum[(freqs >= 30) & (freqs <= 300)].sum()
    high = spectrum[(freqs >= 350) & (freqs <= 600)].sum()
    assert low > 0.0, "缺低频轰鸣"
    assert high > 0.0, "缺高频逆变成分"


def test_ensure_sfx_writes_valid_wav_files():
    paths = audio_mod._ensure_sfx()
    for name in ("motor", "wheels", "horn"):
        assert name in paths
        assert paths[name].exists()
        assert paths[name].stat().st_size > 100, f"{name}.wav 该有内容"


# --------------------------------------------------------------------------- #
# 运行时
# --------------------------------------------------------------------------- #

def test_train_audio_noops_on_null_audio(app):
    """测试环境的 null 音频库下，TrainAudio 所有方法都得安全 no-op。"""
    ta = audio_mod.TrainAudio(app)
    assert ta._load() is False
    assert ta.enabled is False
    # 重复调用也不该抛异常（模拟每帧 update）
    for _ in range(3):
        ta.start()
        ta.update(120.0, 1.0)
        ta.update(0.0, 0.0)
        ta.horn()
        ta.stop()
    ta.destroy()


def test_train_audio_load_is_remembered(app):
    """装过一次之后，结果被记住（null 环境下永远是 False，不每帧重试）。"""
    ta = audio_mod.TrainAudio(app)
    assert ta._load() is False
    assert ta._load() is False
