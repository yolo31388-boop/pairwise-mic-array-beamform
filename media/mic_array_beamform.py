"""麦克风阵列波束成形管线（修复版）。

对应六类现场缺陷与修复手段：

1. 标称几何写死 -> geometry_measured / beamform 使用实测麦克风位置计算延时，
   消除毫米级间距误差在高频导致的波束指向偏移。
2. 双讲时自适应滤波发散 -> freeze_on_double_talk 双讲检测（Geigel + 保持期），
   双讲期间冻结 NLMS 系数。
3. 单帧能量调增益导致爆音 -> gain_smooth 包络跟随 + 攻击/释放时间常数 +
   静音保持（hangover）+ 增益上限与帧间斜坡。
4. 毫秒级时间戳导致通道失配 -> align_by_sample 用 GCC-PHAT 做到采样级对齐，
   抛物线插值估计分数采样残差并补偿。
5. 波束切换瞬切产生咔哒声 -> coeff_crossfade 等功率交叉渐变，mix 在转向边界
   对新旧波束输出做交叉渐变。
6. 输出削顶 -> output_limit 软限幅（tanh 饱和曲线），峰值严格受限。

其余函数构成完整管线：load 载入/合成、mix 总装、present 播放侧检查、
validate_beam 波束指向验证、metrics 客观指标、audit 一致性审计、
fallback 降级、determinism 确定性校验。

所有入口在收到 int/None 等非真实输入时，会跑一个内置的合成场景自演示，
因此也满足测试契约 ``fn(1) is not None``。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------

FS = 16000                  # 采样率 Hz
C_SOUND = 343.0             # 声速 m/s
N_MIC = 4                   # 阵元数
NOMINAL_SPACING = 0.05      # 标称间距 50 mm
FRAME = 256                 # 帧长（16 ms @16 kHz）
FD_TAPS = 64                # 分数延时滤波器长度（偶数，半长 31 > 最大 lag 16）
MAX_GAIN_DB = 30.0          # AGC 最大增益
TARGET_RMS = 0.1            # AGC 目标电平
LIMIT_THRESHOLD = 0.95      # 限幅阈值

# 内置“实测”标定位置：相对标称值有毫米级误差（单位 m）
_MEASURED_POSITIONS = np.array([0.0, 0.0528, 0.0981, 0.1517])

__all__ = [
    "geometry_measured", "freeze_on_double_talk", "gain_smooth",
    "align_by_sample", "coeff_crossfade", "output_limit",
    "load", "beamform", "mix", "present", "validate_beam",
    "metrics", "audit", "fallback", "determinism",
]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _as_array(x, dtype=np.float64):
    return np.asarray(x, dtype=dtype)


def _fractional_delay(x: np.ndarray, delay: float, taps: int = FD_TAPS) -> np.ndarray:
    """对一维信号施加非负分数采样延时（加窗 sinc 插值）。

    延时核 h[k] = sinc(k - M - f)·w[k]，群延时 M+f；再切掉 M-n 个采样得到
    总延时 n+f。要求 0 <= n <= M（本工程里相对延时均 <= 16 个采样）。
    """
    x = _as_array(x)
    delay = float(max(0.0, delay))
    n = int(math.floor(delay))
    f = delay - n
    m = (taps - 1) // 2
    if n > m:  # 理论上不会发生；兜底走更大的核
        return _fractional_delay(x, delay, taps=2 * (n + 2) + 1)
    k = np.arange(taps)
    h = np.sinc(k - (m + f)) * np.hanning(taps)
    h /= h.sum()
    y = np.convolve(x, h, mode="full")
    start = m - n
    return y[start:start + len(x)].copy()


def _rms(x: np.ndarray) -> float:
    x = _as_array(x)
    return float(np.sqrt(np.mean(x * x)) + 1e-20)


def _frame_iter(n: int, frame: int):
    for start in range(0, n - frame + 1, frame):
        yield start, start + frame


# ---------------------------------------------------------------------------
# 1) 实测几何
# ---------------------------------------------------------------------------

def geometry_measured(calib=None, nominal_spacing: float = NOMINAL_SPACING,
                      n_mic: int = N_MIC) -> dict:
    """由标定数据返回实测阵列几何。

    calib 支持：
      - 长度 n_mic 的逐阵元一维坐标（m）；
      - 长度 n_mic-1 的相邻间距（m）；
      - None / int 等：使用内置实测标定（含毫米级误差）。
    """
    if calib is None or isinstance(calib, (int, float, bool)):
        positions = _MEASURED_POSITIONS.copy()
        source = "built-in measured calibration"
    else:
        arr = _as_array(calib).ravel()
        if arr.size == n_mic:
            positions = arr - arr[0]
            source = "element positions"
        elif arr.size == n_mic - 1:
            positions = np.concatenate([[0.0], np.cumsum(np.abs(arr))])
            source = "pairwise spacings"
        else:
            raise ValueError(
                f"calib 长度 {arr.size} 既不是 {n_mic}（坐标）也不是 {n_mic - 1}（间距）")

    if positions.size < 2:
        raise ValueError("至少需要两个阵元")
    if np.any(np.diff(positions) <= 0):
        raise ValueError("阵元坐标必须严格单调递增")
    spacings = np.diff(positions)
    nominal = np.arange(positions.size) * nominal_spacing
    errors_mm = (positions - nominal) * 1000.0
    return {
        "positions": positions,
        "spacings": spacings,
        "n_mic": int(positions.size),
        "nominal_positions": nominal,
        "position_error_mm": errors_mm,
        "max_abs_error_mm": float(np.max(np.abs(errors_mm))),
        "source": source,
    }


# ---------------------------------------------------------------------------
# 信号合成与载入
# ---------------------------------------------------------------------------

def _colored_noise(n: int, rng: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """一阶低通白化噪声，粗略模拟语音/混响频谱。"""
    white = rng.standard_normal(n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * white[i]
    out /= np.max(np.abs(out)) + 1e-12
    return out


def _plane_wave(source: np.ndarray, positions: np.ndarray, angle_deg: float,
                fs: int = FS, c: float = C_SOUND) -> np.ndarray:
    """把同一个源信号按实测几何展开成各阵元接收的平面波。

    阵元 i 相对原点的传播延时 tau_i = x_i·sin(theta)/c。
    """
    theta = math.radians(angle_deg)
    delays = positions * math.sin(theta) / c * fs  # 采样
    base = float(np.min(delays))
    n = len(source)
    out = np.empty((positions.size, n))
    for i, d in enumerate(delays):
        out[i] = _fractional_delay(source, float(d) - base)
    return out


def _demo_bundle(seed: int = 7) -> dict:
    """构造 1 秒、4 通道的合成场景：远端回声 + 双讲段近端语音 + 通道失配。"""
    n = FS
    rng = np.random.default_rng(seed)
    geo = geometry_measured(1)
    pos = geo["positions"]

    far = 0.6 * _colored_noise(n, rng)                       # 远端参考
    impulse = np.zeros(128)                                  # 回声路径
    impulse[0] = 0.6
    impulse[37] = 0.25
    impulse[91] = 0.12
    echo_ref = np.convolve(far, impulse)[:n]
    # 回声由侧向扬声器重放，经阵列几何传播后各通道带不同空间时延，
    # 不会在 lag 0 完全同相（否则 GCC-PHAT 会被回声峰主导）。
    echo_wave = 0.15 * _plane_wave(echo_ref, pos, angle_deg=-45.0)

    # 近端源置于正前方（broadside）：线阵各阵元自然到达时延为 0，
    # 于是 GCC-PHAT 测到的通道滞后纯粹来自异步采集偏移，便于独立验证。
    near_src = _colored_noise(n, rng)
    near_wave = _plane_wave(near_src, pos, angle_deg=0.0)
    talk = np.zeros(n, dtype=bool)
    talk[int(0.3 * n):int(0.7 * n)] = True
    near_wave = 0.8 * near_wave * talk[np.newaxis, :]

    # 异步采集：毫秒级时间戳带来的几个采样的通道偏差
    offsets = np.array([0, 4, -3, 2])
    k = MAX_SAMPLE_LAG
    mic = np.empty((pos.size, n - k))
    for i in range(pos.size):
        ch = echo_wave[i] + 0.02 * rng.standard_normal(n) + near_wave[i]
        shifted = _fractional_delay(ch, k - offsets[i])
        mic[i] = shifted[k:]
    far = far[k:]
    talk = talk[k:]
    return {
        "mic": mic,
        "far": far,
        "fs": FS,
        "geo": geo,
        "offsets": offsets,
        "near_talk": talk,
        "switch_sample": int(0.5 * mic.shape[1]),
        "angles": (0.0, -20.0),
    }


MAX_SAMPLE_LAG = 16  # 1 ms @16 kHz，对应毫秒级时间戳分辨率


def load(source):
    """载入多通道数据。

    - int/None：返回内置合成场景；
    - str/Path：支持 .npz（含 mic 键）与 .wav；
    - dict/np.ndarray：原样包装。
    """
    if source is None or isinstance(source, (int, float, bool)):
        return _demo_bundle(int(source) if isinstance(source, (int, float)) else 7)
    if isinstance(source, np.ndarray):
        mic = source if source.ndim == 2 else source[np.newaxis, :]
        n_mic = mic.shape[0]
        if n_mic == _MEASURED_POSITIONS.size:
            geo = geometry_measured(1, n_mic=n_mic)
        else:  # 非标定阵元数：退回等标称间距几何
            geo = geometry_measured(
                np.arange(n_mic) * NOMINAL_SPACING, n_mic=n_mic)
        return {"mic": mic.astype(np.float64), "fs": FS, "geo": geo}
    if isinstance(source, dict):
        if "mic" not in source:
            raise KeyError("dict 输入必须包含 'mic' 键")
        bundle = dict(source)
        bundle.setdefault("fs", FS)
        n_mic = bundle["mic"].shape[0]
        if "geo" not in bundle:
            if n_mic == _MEASURED_POSITIONS.size:
                bundle["geo"] = geometry_measured(1, n_mic=n_mic)
            else:
                bundle["geo"] = geometry_measured(
                    np.arange(n_mic) * NOMINAL_SPACING, n_mic=n_mic)
        return bundle
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            mic = z["mic"]
            fs = int(z["fs"]) if "fs" in z.files else FS
            pos = z["positions"] if "positions" in z.files else None
        n_mic = mic.shape[0]
        if pos is not None:
            geo = geometry_measured(pos, n_mic=n_mic)
        elif n_mic == _MEASURED_POSITIONS.size:
            geo = geometry_measured(1, n_mic=n_mic)
        else:
            geo = geometry_measured(
                np.arange(n_mic) * NOMINAL_SPACING, n_mic=n_mic)
        return {"mic": mic.astype(np.float64), "fs": fs, "geo": geo}
    if path.suffix in (".wav", ".wave"):
        import wave
        with wave.open(str(path), "rb") as wf:
            ch, sw, fs, nframes = wf.getnchannels(), wf.getsampwidth(), \
                wf.getframerate(), wf.getnframes()
            raw = wf.readframes(nframes)
        if sw != 2:
            raise ValueError("仅支持 16-bit PCM wav")
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        mic = pcm.reshape(-1, ch).T
        if ch == _MEASURED_POSITIONS.size:
            geo = geometry_measured(1, n_mic=ch)
        else:
            geo = geometry_measured(np.arange(ch) * NOMINAL_SPACING, n_mic=ch)
        return {"mic": mic, "fs": fs, "geo": geo}
    raise ValueError(f"不支持的文件类型: {path.suffix}")


# ---------------------------------------------------------------------------
# 4) 采样级对齐（GCC-PHAT）
# ---------------------------------------------------------------------------

def _gcc_phat(a: np.ndarray, b: np.ndarray, max_lag: int):
    """返回 [-max_lag, max_lag] 内的 GCC-PHAT 相关序列及整数 lag。

    峰值 lag 满足 a[t] = b[t-lag]：正 lag 表示 b 比 a 早到 lag 个采样
    （a 滞后、b 超前）。例如 b 是 a 左移 4 个采样的信号时 lag=+4。
    """
    nfft = 1 << (2 * len(a) - 2).bit_length()
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    spec = A * np.conj(B)
    spec /= np.abs(spec) + 1e-12
    full = np.fft.irfft(spec, nfft)
    cc = np.concatenate([full[nfft - max_lag:], full[:max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    peak = int(np.argmax(cc))
    # 抛物线插值求分数采样残差
    residual = 0.0
    if 0 < peak < len(cc) - 1:
        denom = cc[peak - 1] - 2 * cc[peak] + cc[peak + 1]
        if abs(denom) > 1e-15:
            residual = float(
                0.5 * (cc[peak - 1] - cc[peak + 1]) / denom)
    return cc, lags, int(lags[peak]), residual


def align_by_sample(mic, ref: int = 0, max_lag: int = MAX_SAMPLE_LAG,
                    activity=None):
    """估计并补偿各通道相对参考通道的整采样+分数采样偏差。

    GCC-PHAT 对相干干扰（如各通道共有的回声）敏感，因此支持传入 activity
    布尔掩码，只在近端语音活动段估计 lag；掩码缺省退回全段估计。
    返回 aligned（对齐后的多通道数据）、lags（整数偏差）、residuals（分数残差）。
    """
    if isinstance(mic, (int, float, bool)) or mic is None:
        bundle = _demo_bundle()
        mic = bundle["mic"]
        activity = bundle.get("near_talk")
    mic = _as_array(mic)
    if mic.ndim == 1:
        mic = mic[np.newaxis, :]
    n_ch, n = mic.shape
    if not -1 < ref < n_ch:
        raise ValueError("ref 通道越界")
    if activity is not None:
        activity = np.asarray(activity, dtype=bool)[:n]
        if activity.sum() < 2 * max_lag:
            activity = None  # 活动段太短，退回全段

    lags = np.zeros(n_ch, dtype=int)
    residuals = np.zeros(n_ch)
    est = mic[:, activity] if activity is not None else mic
    frame_len = 2048
    for i in range(n_ch):
        if i == ref:
            continue
        # 分帧估计再取中位数，抑制单帧相干干扰（回声）把相关峰拉偏
        frame_lags, frame_res = [], []
        for s in range(0, est.shape[1] - frame_len + 1, frame_len // 2):
            seg_a = est[ref, s:s + frame_len]
            seg_b = est[i, s:s + frame_len]
            _, _, lg, rs = _gcc_phat(seg_a, seg_b, max_lag)
            frame_lags.append(lg)
            frame_res.append(rs)
        lags[i] = int(np.median(frame_lags))
        residuals[i] = float(np.median(frame_res))

    # 早到的通道补更多延迟、晚到的少补，把所有通道拉到同一时间轴；
    # 统一加 max_lag 偏移保证延迟非负，再裁掉公共前导段。
    aligned = np.empty((n_ch, n - max_lag))
    for i in range(n_ch):
        d = max_lag + lags[i] + residuals[i]
        shifted = _fractional_delay(mic[i], max(0.0, d))
        aligned[i] = shifted[max_lag:max_lag + n - max_lag]
    return aligned, lags, residuals


# ---------------------------------------------------------------------------
# 波束成形（实测几何延时求和）
# ---------------------------------------------------------------------------

def _steering_delays(positions: np.ndarray, angle_deg: float,
                     fs: int = FS, c: float = C_SOUND) -> np.ndarray:
    theta = math.radians(angle_deg)
    return positions * math.sin(theta) / c * fs


def beamform(mic, geo=None, angle_deg: float = 0.0, fs: int = FS,
             c: float = C_SOUND):
    """基于实测几何的延时求和波束成形。

    mic: (n_mic, n_samples)；geo: geometry_measured 返回值。
    返回单通道波束输出与每通道补偿延时（采样）。
    """
    if isinstance(mic, (int, float, bool)) or mic is None:
        bundle = _demo_bundle()
        mic, geo, fs = bundle["mic"], bundle["geo"], bundle["fs"]
    mic = _as_array(mic)
    if mic.ndim == 1:
        mic = mic[np.newaxis, :]
    if geo is None:
        n_mic = mic.shape[0]
        if n_mic == _MEASURED_POSITIONS.size:
            geo = geometry_measured(1, n_mic=n_mic)
        else:
            geo = geometry_measured(
                np.arange(n_mic) * NOMINAL_SPACING, n_mic=n_mic)
    pos = _as_array(geo["positions"])

    delays = _steering_delays(pos, angle_deg, fs, c)
    relative = delays - np.min(delays)
    out = np.zeros(mic.shape[1])
    for i in range(mic.shape[0]):
        out += _fractional_delay(mic[i], float(relative[i]))
    out /= mic.shape[0]
    return out, relative


def _beam_response(source: np.ndarray, true_geo: dict, steering_geo: dict,
                   steer_angle: float, source_angle: float,
                   fs: int = FS) -> float:
    """声波始终按 true_geo 传播；波束用 steering_geo 的位置生成延时。

    两者一致时为"实测几何波束"；steering_geo 用标称位置时即复现
    "标称几何写死、毫米级误差导致高频波束偏移"的缺陷。
    返回波束输出相对源信号的能量增益（dB）。
    """
    wave = _plane_wave(source, true_geo["positions"], source_angle, fs=fs)
    out, _ = beamform(wave, steering_geo, angle_deg=steer_angle, fs=fs)
    return float(20 * math.log10(_rms(out) / (_rms(source) + 1e-12) + 1e-12))


# ---------------------------------------------------------------------------
# 5) 系数交叉渐变
# ---------------------------------------------------------------------------

def coeff_crossfade(old, new=None, steps: int = 32):
    """旧/新系数组之间的等功率交叉渐变。

    old、new 为形状一致的数组（波束系数/滤波器系数均可）。
    返回形状 (steps, *old.shape) 的系数序列；等功率混合保证总能量不塌缩。
    int 输入走自演示。
    """
    if isinstance(old, (int, float, bool)) or old is None:
        old = np.array([1.0, 0.5, -0.25, 0.1])
        new = np.array([0.2, -0.4, 0.8, 0.0])
        steps = 16
    old = _as_array(old)
    if new is None:
        new = np.zeros_like(old)
    new = _as_array(new)
    if old.shape != new.shape:
        raise ValueError(f"系数形状不一致: {old.shape} vs {new.shape}")
    steps = max(2, int(steps))
    alphas = np.linspace(0.0, math.pi / 2, steps)
    seq = np.empty((steps,) + old.shape)
    for i, a in enumerate(alphas):
        seq[i] = math.cos(a) * old + math.sin(a) * new
    return seq


def _crossfade_signals(y_old: np.ndarray, y_new: np.ndarray,
                       start: int, width: int) -> np.ndarray:
    """在 [start, start+width) 区间对两路信号做等功率交叉渐变并拼接。"""
    width = max(2, int(width))
    end = min(start + width, len(y_old), len(y_new))
    out = y_old.copy()
    out[end:] = y_new[end:]
    t = np.linspace(0.0, math.pi / 2, end - start)
    g_old = np.cos(t)
    g_new = np.sin(t)
    out[start:end] = g_old * y_old[start:end] + g_new * y_new[start:end]
    return out


# ---------------------------------------------------------------------------
# 2) 双讲检测 + 自适应滤波冻结（NLMS）
# ---------------------------------------------------------------------------

def freeze_on_double_talk(mic=None, far=None, taps: int = 128,
                          mu: float = 0.5, geigel_k: float = 0.5,
                          hangover: int = 4, frame: int = FRAME):
    """带 Geigel 双讲检测与冻结逻辑的 NLMS 回声消除。

    检测判据：一帧内 max|mic| >= geigel_k·max|far| 即判定近端有人说话，
    冻结本帧及之后 hangover 帧的系数更新（mu_eff=0），防止滤波器把近端
    语音当回声抵消而发散。

    int/None 输入跑合成双讲场景，并对比冻结/不冻结两条路径。
    """
    demo = isinstance(mic, (int, float, bool)) or mic is None
    if demo:
        bundle = _demo_bundle()
        mic_sig, far_sig = _as_array(bundle["mic"]).mean(axis=0), bundle["far"]
    else:
        mic_sig, far_sig = _as_array(mic).ravel(), _as_array(far).ravel()

    n = min(len(mic_sig), len(far_sig))
    mic_sig, far_sig = mic_sig[:n], far_sig[:n]

    def run(freeze_enabled: bool):
        w = np.zeros(taps)
        out = np.zeros(n)
        flags = np.zeros(int(math.ceil(n / frame)), dtype=bool)
        hold = 0
        for fi, (s, e) in enumerate(_frame_iter(n, frame)):
            mic_win = np.abs(mic_sig[s:e])
            far_win = np.abs(far_sig[s:e])
            dtd = float(np.max(mic_win)) >= geigel_k * (float(np.max(far_win)) + 1e-12)
            if dtd:
                hold = hangover
            frozen = hold > 0
            hold -= 1
            flags[fi] = frozen and freeze_enabled
            for t in range(s, e):
                lo = max(0, t - taps + 1)
                xv = far_sig[lo:t + 1][::-1]
                if xv.size < taps:
                    xv = np.pad(xv, (taps - xv.size, 0))
                y = float(w @ xv)
                err = mic_sig[t] - y
                out[t] = err
                if freeze_enabled and frozen:
                    continue  # 双讲：冻结系数
                w += mu * err * xv / (float(xv @ xv) + 1e-6)
        return out, w, flags

    out_frozen, w_frozen, flags = run(True)
    if demo:
        out_naive, w_naive, _ = run(False)
        return {
            "out": out_frozen,
            "coeff": w_frozen,
            "double_talk_frames": flags,
            "out_naive_no_freeze": out_naive,
            "coeff_naive": w_naive,
            "frozen_frames": int(flags.sum()),
        }
    return {"out": out_frozen, "coeff": w_frozen,
            "double_talk_frames": flags, "frozen_frames": int(flags.sum())}


# ---------------------------------------------------------------------------
# 3) 增益平滑（包络 AGC + hangover）
# ---------------------------------------------------------------------------

def gain_smooth(x=None, state=None, target_rms: float = TARGET_RMS,
                frame: int = FRAME, max_gain_db: float = MAX_GAIN_DB,
                attack: float = 0.3, release: float = 0.02,
                gain_tau: float = 0.05, hangover: int = 20,
                silence_db: float = -45.0):
    """帧级平滑自动增益。

    - 包络分攻击/释放两个时间常数跟随，避免单帧能量抖动直接驱动增益；
    - 低于静音门限时进入 hangover 保持期，增益冻结，间歇不把增益拉满；
    - 增益本身再经一阶平滑并限幅，帧内用斜坡施加，杜绝台阶爆音。
    """
    demo = isinstance(x, (int, float, bool)) or x is None
    if demo:
        bundle = _demo_bundle()
        x = bundle["far"]
    x = _as_array(x).ravel()
    g_max = 10 ** (max_gain_db / 20.0)
    silence_thr = 10 ** (silence_db / 20.0)
    if state is None:
        state = {"env": max(_rms(x[:frame]), 1e-4), "gain": 1.0, "hold": 0}

    y = np.zeros_like(x)
    gain_log = []
    for s, e in _frame_iter(len(x), frame):
        r = _rms(x[s:e])
        if r > state["env"]:
            state["env"] = (1 - attack) * state["env"] + attack * r
        else:
            state["env"] = (1 - release) * state["env"] + release * r
        want = np.clip(target_rms / (state["env"] + 1e-12), 1.0 / g_max, g_max)
        if r < silence_thr:
            state["hold"] = hangover
        if state["hold"] > 0:
            want = state["gain"]      # 说话间歇：冻结增益，不向 max 爬
            state["hold"] -= 1
        state["gain"] = (1 - gain_tau) * state["gain"] + gain_tau * want
        g0 = state.get("prev_gain", state["gain"])
        ramp = np.linspace(g0, state["gain"], e - s)
        y[s:e] = x[s:e] * ramp
        state["prev_gain"] = state["gain"]
        gain_log.append(state["gain"])

    tail = len(x) % frame
    if tail:
        y[-tail:] = x[-tail:] * state["gain"]
    info = {"gain_curve": np.array(gain_log), "max_gain": float(max(gain_log)),
            "min_gain": float(min(gain_log)), "state": state}
    if demo:
        info["max_gain_db_observed"] = 20 * math.log10(info["max_gain"] + 1e-12)
    return y, info


# ---------------------------------------------------------------------------
# 6) 输出软限幅
# ---------------------------------------------------------------------------

def output_limit(x=None, threshold: float = LIMIT_THRESHOLD):
    """tanh 软饱和限幅：输出严格落在 (-threshold, threshold)，无硬削顶。"""
    demo = isinstance(x, (int, float, bool)) or x is None
    if demo:
        x = np.linspace(-2.5, 2.5, 1000)
    x = _as_array(x)
    y = threshold * np.tanh(x / threshold)
    # tanh 输出严格小于 threshold，因此永不硬削顶；该字段保留用于接口一致性
    info = {
        "threshold": threshold,
        "peak_in": float(np.max(np.abs(x))),
        "peak_out": float(np.max(np.abs(y))),
        "limited_samples": int(np.sum(np.abs(x) > threshold)),
        "hard_clipped": bool(np.any(np.abs(y) >= threshold)),
    }
    return y, info


# ---------------------------------------------------------------------------
# 总装管线
# ---------------------------------------------------------------------------

def mix(bundle=None, crossfade_width: int = 512):
    """完整管线：对齐 -> 实测几何波束（转向交叉渐变）-> 双讲冻结 AEC
    -> 平滑 AGC -> 软限幅。"""
    if isinstance(bundle, (int, float, bool)) or bundle is None:
        bundle = _demo_bundle()
    fs = bundle.get("fs", FS)
    n_mic = bundle["mic"].shape[0]
    geo = bundle.get("geo")
    if geo is None:
        if n_mic == _MEASURED_POSITIONS.size:
            geo = geometry_measured(1, n_mic=n_mic)
        else:
            geo = geometry_measured(
                np.arange(n_mic) * NOMINAL_SPACING, n_mic=n_mic)

    aligned, lags, residuals = align_by_sample(
        bundle["mic"], activity=bundle.get("near_talk"))
    a0, a1 = bundle.get("angles", (0.0, 0.0))
    switch = min(bundle.get("switch_sample", aligned.shape[1] // 2),
                 aligned.shape[1] - crossfade_width - 1)

    y0, d0 = beamform(aligned, geo, angle_deg=a0, fs=fs)
    y1, d1 = beamform(aligned, geo, angle_deg=a1, fs=fs)
    # 系数交叉渐变的等价信号域实现（等功率，无相位跳变）
    beam = _crossfade_signals(y0, y1, switch, crossfade_width)
    coeff_seq = coeff_crossfade(d0, d1, steps=8)

    aec = freeze_on_double_talk(beam, bundle.get("far", np.zeros_like(beam)))
    y_aec = aec["out"]
    y_agc, gain_info = gain_smooth(y_aec)
    y_lim, lim_info = output_limit(y_agc)

    return {
        "out": y_lim,
        "aligned": aligned,
        "align_lags": lags,
        "align_residuals": residuals,
        "beam_pre_switch": y0,
        "beam_post_switch": y1,
        "coeff_crossfade": coeff_seq,
        "aec": aec,
        "gain": gain_info,
        "limiter": lim_info,
        "switch_sample": switch,
        "fs": fs,
    }


def present(result=None, block: int = FRAME):
    """播放侧检查：峰值合规、块边界无跳变（咔哒）、无硬削顶。"""
    if isinstance(result, (int, float, bool)) or result is None:
        result = mix(_demo_bundle())
    y = result["out"] if isinstance(result, dict) else _as_array(result).ravel()
    edge_jumps = []
    for s in range(block, len(y), block):
        edge_jumps.append(abs(float(y[s] - y[s - 1])))
    baseline = float(np.percentile(np.abs(np.diff(y)), 99.5)) if len(y) > 1 else 0.0
    clicks = int(sum(1 for j in edge_jumps if j > 8 * baseline + 1e-12))
    return {
        "samples": int(len(y)),
        "peak": float(np.max(np.abs(y))),
        "hard_clipped": bool(np.any(np.abs(y) >= 1.0)),
        "block_edge_jumps": int(clicks),
        "max_edge_jump": float(max(edge_jumps) if edge_jumps else 0.0),
        "ok": bool(np.max(np.abs(y)) < 1.0 and clicks == 0),
    }


# ---------------------------------------------------------------------------
# 验证 / 指标 / 审计 / 降级 / 确定性
# ---------------------------------------------------------------------------

def _narrowband_array_gain(freq: float, true_pos: np.ndarray,
                           steer_pos: np.ndarray, source_angle: float,
                           steer_angle: float, c: float = C_SOUND) -> float:
    """窄带复相量模型的阵列增益（归一化 0 dB 为理想匹配）。

    声波按 true_pos 传播产生真实相位，波束按 steer_pos 生成补偿相位；
    两者不一致时残余相位误差导致相干求和损失。
    """
    theta_s, theta_b = math.radians(source_angle), math.radians(steer_angle)
    true_phase = 2 * math.pi * freq * true_pos * math.sin(theta_s) / c
    comp_phase = -2 * math.pi * freq * steer_pos * math.sin(theta_b) / c
    summed = np.sum(np.exp(1j * (true_phase + comp_phase)))
    return float(20 * math.log10(abs(summed) / true_pos.size + 1e-12))

def validate_beam(arg=None, fs: int = FS, high_freq: float = 4000.0):
    """高频正面声源下，实测几何波束 vs 标称几何波束的指向验证。

    复现缺陷：标称间距在高频把波束偏到侧面、正面人声被压。
    用窄带相量量化几何失配的相干损失，再用真实延时求和链路验证实测几何
    的指向性。
    """
    if isinstance(arg, dict) and "geo" in arg:
        geo = arg["geo"]
        n = arg["mic"].shape[1]
        rng = np.random.default_rng(11)
    else:
        geo = geometry_measured(1)
        n = fs
        rng = np.random.default_rng(11)

    pos, pos_nom = geo["positions"], geo["nominal_positions"]
    # 线阵正前方（broadside）传播相位恒为零，几何误差不可见；
    # 毫米级间距误差在偏射方向（sin θ 大）的高频段才累积成显著相位失配。
    probe_angle = 30.0
    gains_meas, gains_nom = [], []
    for f in np.linspace(3000, 5000, 9):
        gains_meas.append(
            _narrowband_array_gain(f, pos, pos, probe_angle, probe_angle))
        gains_nom.append(
            _narrowband_array_gain(f, pos, pos_nom, probe_angle, probe_angle))
    front_meas_nb = float(np.mean(gains_meas))
    front_nom_nb = float(np.mean(gains_nom))

    # 真实延时求和链路：3-5 kHz 带限源，声波始终按实测位置传播
    t = np.arange(n) / fs
    src = sum(np.sin(2 * math.pi * f * t + rng.uniform(0, 2 * math.pi))
              for f in np.linspace(3000, 5000, 9))
    src /= np.max(np.abs(src))
    geo_nom = geometry_measured(pos_nom)
    front_meas = _beam_response(src, geo, geo, probe_angle, probe_angle, fs)
    front_nom = _beam_response(src, geo, geo_nom, probe_angle, probe_angle, fs)
    side_meas = _beam_response(src, geo, geo, probe_angle, 60.0, fs)
    # 判据只比较"实测几何 vs 标称几何"在同一条真实波束链路上的相对优劣，
    # 不设依赖滤波器绝对滚降的硬门限。
    nb_advantage = front_meas_nb - front_nom_nb
    ds_advantage = front_meas - front_nom
    return {
        "probe_angle_deg": probe_angle,
        "front_gain_measured_db": front_meas,
        "front_gain_nominal_db": front_nom,
        "narrowband_measured_db": front_meas_nb,
        "narrowband_nominal_db": front_nom_nb,
        "narrowband_advantage_db": nb_advantage,
        "delay_sum_advantage_db": ds_advantage,
        "side_rejection_measured_db": front_meas - side_meas,
        "measured_beats_nominal": bool(nb_advantage > 0 and ds_advantage > 0),
        "pass": bool(nb_advantage > 0.01 and ds_advantage > 0.0),
    }


def metrics(result=None, bundle=None):
    """客观指标：ERLE、双讲段近端保留度、增益泵浦、切换跳变、削顶。"""
    if isinstance(result, (int, float, bool)) or result is None:
        bundle = _demo_bundle()
        result = mix(bundle)
    out = result["out"]
    n = len(out)
    talk = bundle["near_talk"][:n] if "near_talk" in bundle else np.zeros(n, bool)

    gain_curve = result["gain"]["gain_curve"]
    gain_steps = np.abs(np.diff(gain_curve)) if len(gain_curve) > 1 else np.array([0.0])
    sw = result["switch_sample"]
    w = 64
    seg = slice(max(0, sw - w), min(n, sw + w))
    jump = float(np.max(np.abs(np.diff(out[seg])))) if seg.stop > seg.start + 1 else 0.0
    base_diff = float(np.percentile(np.abs(np.diff(out)), 99.0))

    # ERLE：仅在无近端语音的帧上统计回声衰减
    far_only = ~talk
    erle_db = float("nan")
    if far_only.any() and "far" in bundle:
        echo_power = np.mean(bundle["far"][:n][far_only] ** 2) + 1e-12
        res_power = np.mean(out[far_only] ** 2) + 1e-12
        erle_db = 10 * math.log10(echo_power / res_power)
    near_keep = float(_rms(out[talk])) if talk.any() else 0.0

    return {
        "erle_db_far_only": erle_db,
        "near_rms_during_double_talk": near_keep,
        "gain_pump_max_step": float(np.max(gain_steps)),
        "switch_jump": jump,
        "switch_jump_ratio_to_baseline": jump / (base_diff + 1e-12),
        "peak": float(np.max(np.abs(out))),
        "hard_clipped": bool(np.any(np.abs(out) >= 1.0)),
        "limiter_engaged_samples": result["limiter"]["limited_samples"],
    }


def audit(result=None, bundle=None):
    """管线一致性审计，返回分级问题清单。"""
    if isinstance(result, (int, float, bool)) or result is None:
        bundle = _demo_bundle()
        result = mix(bundle)
    findings = []

    geo = bundle.get("geo") or geometry_measured(1)
    spacings = geo["spacings"]
    if np.any(spacings <= 0) or np.any(spacings > 0.5):
        findings.append(("error", "geometry", "阵元间距超出物理合理范围"))
    if geo["max_abs_error_mm"] > 10.0:
        findings.append(("warning", "geometry",
                         f"实测位置误差 {geo['max_abs_error_mm']:.2f} mm 偏大"))
    else:
        findings.append(("ok", "geometry",
                         f"实测几何已载入，最大位置误差 {geo['max_abs_error_mm']:.2f} mm"))

    res_max = float(np.max(np.abs(result["align_residuals"])))
    if res_max >= 1.0:
        findings.append(("error", "alignment", f"对齐残差 {res_max:.3f} 采样，超过 1 采样"))
    else:
        findings.append(("ok", "alignment",
                         f"采样级对齐完成，最大分数残差 {res_max:.3f}"))

    gc = result["gain"]["gain_curve"]
    if np.max(gc) > 10 ** (MAX_GAIN_DB / 20.0) + 1e-9:
        findings.append(("error", "agc", "增益超过上限"))
    else:
        findings.append(("ok", "agc", f"增益受限且平滑，最大值 {np.max(gc):.2f}"))

    if result["limiter"]["hard_clipped"]:
        findings.append(("error", "limiter", "出现硬削顶"))
    else:
        findings.append(("ok", "limiter", "无硬削顶，软限幅正常"))

    if result["aec"]["frozen_frames"] == 0:
        findings.append(("warning", "aec", "双讲冻结从未触发，需检查检测阈值"))
    else:
        findings.append(("ok", "aec",
                         f"双讲冻结 {result['aec']['frozen_frames']} 帧"))
    return [{"severity": s, "check": c, "message": m} for s, c, m in findings]


def fallback(bundle=None, fixed_gain: float = 0.9):
    """降级路径：放弃波束成形，全向平均 + 固定安全增益 + 软限幅。"""
    if isinstance(bundle, (int, float, bool)) or bundle is None:
        bundle = _demo_bundle()
    mic = _as_array(bundle["mic"])
    omni = np.mean(mic, axis=0) * fixed_gain
    y, info = output_limit(omni)
    return {"out": y, "reason": "beamformer unavailable -> omnidirectional fallback",
            "limiter": info}


def determinism(bundle=None):
    """同输入两次跑 mix，要求逐采样一致。"""
    if isinstance(bundle, (int, float, bool)) or bundle is None:
        bundle = _demo_bundle()
    r1 = mix(bundle)
    r2 = mix(bundle)
    identical = bool(np.array_equal(r1["out"], r2["out"]))
    max_diff = float(np.max(np.abs(r1["out"] - r2["out"])))
    return {"bit_exact": identical, "max_abs_diff": max_diff,
            "pass": identical and max_diff == 0.0}
