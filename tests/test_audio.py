"""列车音效：波形合成与 null 音频下的安全退化。

音效在运行时程序化合成（见 ``render/audio.py``），这里查三件事：

1. 合成出来的波形**确定、非空、长度正确**（无缝循环靠整数 Hz 分量保证，
   所以长度必须正好是整段循环）；
2. 行驶声听起来得是**引擎**、不是**蜂鸣器** —— 这条是听感需求，用"包络的
   周期性调制有多深"和"频段能量分布"两个量把它固化下来（见
   :func:`test_motor_wave_is_not_a_buzzer`）。它防的是回归：旧版在谐波堆上
   叠 380 / 570 Hz 纯音 + 6 Hz 颤音，听起来就是"嘀嘀嘀"，被用户点名去掉过；
3. ``TrainAudio`` 在 null 音频环境（测试的 ``app`` fixture 就是）里照常安全
   no-op —— 任何调用都不该抛异常，也不该每帧都去重试装载。
"""

from __future__ import annotations

import numpy as np

from render import audio as audio_mod


def _envelope(wave: np.ndarray, window: int = 512) -> np.ndarray:
    """短窗 RMS 包络 —— "音量随时间怎么变"的那条曲线。"""
    count = len(wave) // window
    return np.sqrt((wave[: count * window].reshape(count, window) ** 2).mean(axis=1))


def _beep_depth(envelope: np.ndarray, frame_rate: float,
                low: float = 3.0, high: float = 15.0) -> float:
    """包络里**每秒 3~15 下**那一段的调制深度（相对平均音量）。

    为什么必须先带通：宽频轰鸣本身就意味着包络是随机起伏的，直接拿包络的
    ``std/mean`` 量出来二十几个百分点，但那是"噪声的长相"，不是"打点"。听感上
    像蜂鸣的是**周期性**调制，所以只看打点那一段频带。

    "每秒 3~15 下"这个区间的来由：比 3 Hz 慢是呼吸 / 换挡的自然起伏（本波形
    有意留了 1 Hz 的轻微呼吸），比 15 Hz 快就该听成音高或"沙沙"了，不再是节拍。
    旧版那条 6 Hz、深度 28% 的颤音正好落在区间正中。
    """
    centered = envelope - envelope.mean()
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / frame_rate)
    spectrum[(freqs < low) | (freqs > high)] = 0.0
    band = np.fft.irfft(spectrum, n=centered.size)
    return float(band.std() / envelope.mean())


def _band_share(wave: np.ndarray, low: float, high: float) -> float:
    """``[low, high)`` 这个频段占整段信号能量的比例。"""
    spectrum = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(len(wave), d=1.0 / audio_mod._RATE)
    return float(spectrum[(freqs >= low) & (freqs < high)].sum() / spectrum.sum())


# --------------------------------------------------------------------------- #
# 波形合成
# --------------------------------------------------------------------------- #

def test_waveforms_are_deterministic_and_nonempty():
    for generator, seconds in ((audio_mod._motor_wave, 1.0),
                               (audio_mod._horn_wave, 1.8)):
        first = generator()
        second = generator()
        assert first.shape == second.shape
        assert np.array_equal(first, second), "合成必须是确定性的（可复现）"
        assert np.max(np.abs(first)) > 0.1, "波形不能是空信号"
        assert first.shape[0] == int(audio_mod._RATE * seconds)


def test_motor_wave_is_not_a_buzzer():
    """行驶声必须是引擎，不能是"嘀嘀"声。

    两处判据，对应两种"嘀嘀"的成因：

    * **周期性振幅调制**。每秒好几下的调制听起来就是打点 / 蜂鸣。旧版是 6 Hz、
      深度 28%，被用户点名听得像"嘀嘀"。这里用 :func:`_beep_depth` 量"打点频带"
      里的调制深度，并在同一条测试里拿旧版那种颤音做**对照** —— 对照必须被判成
      超标，本波形必须远低于它，这条测试才不会随参数微调而失去意义。
    * **孤立纯高频**。单独的纯正弦（旧版 380 / 570 Hz）在变速播放时最像电子
      提示音。所以高频段只允许占一小撮能量，主体必须是中低频。
    """
    wave = audio_mod._motor_wave()
    envelope = _envelope(wave, window=1024)
    frame_rate = audio_mod._RATE / 1024

    # 对照：套上旧版那种 6 Hz / 深度 28% 的颤音，就是"嘀嘀"声
    t = np.arange(wave.size) / audio_mod._RATE
    beeping = wave * (0.72 + 0.28 * np.sin(2.0 * np.pi * 6.0 * t))
    beeping_depth = _beep_depth(_envelope(beeping, window=1024), frame_rate)
    depth = _beep_depth(envelope, frame_rate)

    assert beeping_depth > 0.20, "对照样本没有被判成打点，这条测试的判据失效了"
    assert depth < 0.14, (
        f"行驶声里每秒 3~15 下的调制深度是 {depth * 100:.1f}%"
        f"（对照的旧版颤音是 {beeping_depth * 100:.1f}%）—— 听起来会像'嘀嘀'")

    low = _band_share(wave, 20.0, 200.0)
    high = _band_share(wave, 1500.0, audio_mod._RATE / 2)
    assert low > 0.55, f"低频轰鸣只占 {low * 100:.0f}%，太单薄，不像引擎"
    assert high < 0.02, f"高频纯音占了 {high * 100:.1f}%，会像提示音"


def test_motor_wave_has_harmonic_pitch_structure():
    """引擎得有"音调感"：40 Hz 基频的谐波应当在谱上站得住。

    靠它，调速（改播放速率）才像在提速，而不是把录音快放。
    """
    wave = audio_mod._motor_wave()
    spectrum = np.abs(np.fft.rfft(wave))
    fundamental = audio_mod._ENGINE_FUNDAMENTAL

    def energy_at(freq: float, width: float = 3.0) -> float:
        freqs = np.fft.rfftfreq(len(wave), d=1.0 / audio_mod._RATE)
        return float(spectrum[(freqs >= freq - width) & (freqs <= freq + width)].max())

    # 与邻近的"非谐波"频率比：谐波位置必须有明显的峰
    for index in (1, 2, 3, 4):
        target = fundamental * index
        neighbour = target + fundamental * 0.5
        assert energy_at(target) > 1.5 * energy_at(neighbour), (
            f"{target:.0f} Hz 处没有谐波峰 —— 引擎会失去音调感")


def test_waves_loop_without_a_click():
    """所有分量都是整数 Hz，循环接缝处必须连续（否则每圈"咔"一声）。"""
    for generator, seconds in ((audio_mod._motor_wave, 1.0),
                               (audio_mod._horn_wave, 1.8)):
        wave = generator()
        peak = np.max(np.abs(wave))
        # 相邻样本的正常步长
        step = np.max(np.abs(np.diff(wave)))
        seam = abs(wave[0] - wave[-1])
        assert seam <= step * 1.2 + 1e-9, (
            f"循环接头有一个 {seam:.4f} 的台阶（相邻样本步长上限 {step:.4f}）")
        assert peak <= 1.0 + 1e-9


def test_horn_is_a_low_air_horn_not_a_car_horn():
    """风笛要低、要厚；旧版 420 / 520 Hz 那对听起来就是汽车喇叭。"""
    wave = audio_mod._horn_wave()
    # 主要能量应当落在风笛那三个低音上，而不是更高处
    low = _band_share(wave, 250.0, 520.0)
    higher = _band_share(wave, 900.0, audio_mod._RATE / 2)
    assert low > higher * 4.0, "风笛的能量太靠上，会像汽车喇叭"
    # 起音得是渐强的（气笛靠气流吹起来），不能一上来就满
    envelope = _envelope(wave)
    assert envelope[0] < 0.5 * envelope.max(), "起音太硬，不像气笛"


def test_ensure_sfx_writes_valid_wav_files():
    """落盘的 WAV 必须是**能读回来**的单声道 16-bit —— 合成对了但写坏了同样没声音。"""
    import wave as wave_mod

    paths = audio_mod._ensure_sfx()
    assert set(paths) == {"motor", "horn"}, "行驶声音轨只该有引擎与风笛"
    for name, path in paths.items():
        assert path.exists()
        assert name in path.name
        with wave_mod.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == audio_mod._RATE
            frames = handle.getnframes()
            assert frames > 0
            raw = np.frombuffer(handle.readframes(frames), dtype=np.int16)
        assert np.max(np.abs(raw)) > 1000, f"{name}.wav 几乎是静音"


def test_ensure_sfx_reuses_the_cached_file():
    """同一个文件不该被反复重写（临时目录按文件名幂等）。"""
    first = audio_mod._ensure_sfx()
    stamp = first["motor"].stat().st_mtime_ns
    second = audio_mod._ensure_sfx()
    assert second["motor"] == first["motor"]
    assert second["motor"].stat().st_mtime_ns == stamp


def test_sfx_cache_directory_is_revisioned():
    """改了波形却沿用旧缓存，会表现成"改了没生效" —— 修订号必须进目录名。"""
    assert f"r{audio_mod._SFX_REVISION}" in audio_mod._SFX_DIR.name


def test_no_wheel_click_track_remains():
    """轮轨"哐当"声已经按用户要求删掉（那是"嘀嘀"的另一个来源）。

    留着这个合成器、却没人调用它，只会在几个月后被人重新接回播放链路 ——
    所以这里直接断言"它不存在"，删就要删干净。
    """
    assert not hasattr(audio_mod, "_wheels_wave")


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
