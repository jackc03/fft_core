"""
------------------------
CLAUDE GENERATED FILE
------------------------ 


audio_in.py -- turn an MP3 into bit-accurate stimulus for the FFT golden model.

Chain modeled here, in the order the real hardware does it:

    MP3 file
      -> ffmpeg decode / downmix / resample     (float, [-1, 1])
      -> analog front end: gain + DC bias       (MAX9814 output, volts)
      -> ADC: clip, quantize, unsigned codes    (XADC, 12-bit)
      -> remove bias, scale to datapath width   (signed, W_DATA)
      -> DC blocker

That's mp3_to_fxp(): fixed-point samples ready to feed the FFT pipeline in
fft_sim.py. mp3_to_numpy_spectrogram() will chain mp3_to_fxp() into fft_sim's
windowing + FFT stages -- left as a stub until those land there.
"""

from __future__ import annotations
import queue
import shutil
import subprocess
import numpy as np

import fft_sim


# Arty S7 target parameters
TARGET_FS = 48000   # Hz -- decode/resample rate fed to the ADC model
ADC_BITS = 12        # XADC resolution
W_DATA = 16          # datapath width after the ADC

# shared by both spectrogram paths (mp3_to_numpy_spectrogram, mic_to_spectrogram)
# so their frequency-bin resolution always matches -- mismatched n_fft/hop between
# the two makes otherwise-identical audio group into visibly different bars
SPEC_N_FFT = 512
SPEC_HOP = 256


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _decode_audio(path: str, target_fs: int) -> np.ndarray:
    """Decode any ffmpeg-readable file (mp3, m4a, flac, wav) to mono float32
    in [-1, 1] at target_fs. ffmpeg does decode, downmix and resample in one
    pass, which avoids three separate libraries disagreeing about resampling.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found; apt install ffmpeg")
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "f32le",              # raw little-endian float32 on stdout
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(target_fs),
        "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4")


# ---------------------------------------------------------------------------
# Analog front end + ADC
# ---------------------------------------------------------------------------

def _adc_sample(x: np.ndarray,
                adc_bits: int = ADC_BITS,
                vref: float = 1.0,
                bias_v: float = 0.5,
                swing_v: float = 0.45,
                noise_lsb: float = 0.5,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Model the MAX9814 output and the XADC's unipolar input.

    The MAX9814 puts out a DC-biased signal; the XADC digitises 0..vref into
    unsigned codes. swing_v is the peak excursion for a full-scale input --
    set it so bias_v +/- swing_v stays inside [0, vref] or you WILL clip, which
    is exactly the condition you want your testbench to be able to reproduce.

    Returns unsigned integer codes, 0 .. 2^adc_bits - 1.
    """
    rng = rng or np.random.default_rng(0)
    volts = bias_v + x * swing_v
    volts = np.clip(volts, 0.0, vref)

    full = (1 << adc_bits) - 1
    codes = volts / vref * full
    if noise_lsb > 0:
        codes = codes + rng.normal(0.0, noise_lsb, size=codes.shape)
    codes = np.rint(codes).astype(np.int64)
    return np.clip(codes, 0, full)


def _sat(v: int, bits: int) -> int:
    """Saturate a signed integer to `bits` two's-complement range."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return max(lo, min(hi, v))


def _codes_to_signed(codes: np.ndarray, adc_bits: int, w_data: int,
                     bias_code: int | None = None) -> list[int]:
    """Strip the DC bias and left-align into the datapath width.

    NOTE: the XADC DRP register returns the 12-bit result in the UPPER bits of
    a 16-bit word (bits [15:4]). If your RTL reads the raw DRP word, model that
    here instead of shifting -- pick one convention and make both sides match.
    """
    if bias_code is None:
        bias_code = 1 << (adc_bits - 1)
    centred = codes.astype(np.int64) - bias_code
    shift = w_data - adc_bits            # left-align: 12-bit -> 16-bit
    if shift > 0:
        centred = centred << shift
    elif shift < 0:
        centred = centred >> (-shift)
    return [_sat(int(v), w_data) for v in centred]


def _dc_block(samples: list[int], w_data: int, shift: int = 10) -> list[int]:
    """Single-pole DC blocker, the fixed-point version:
        acc += (x - (acc >>> S));  y = x - (acc >>> S)
    Worth modeling because a residual DC offset dumps a big spike into bin 0
    and dominates your waterfall's autoscale."""
    acc = 0
    out = []
    for x in samples:
        avg = acc >> shift
        y = x - avg
        acc += x - avg
        out.append(_sat(y, w_data))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mp3_to_fxp(path: str) -> np.ndarray:
    """Decode `path` and emulate the Arty S7 analog front end + ADC.

    Returns a 1-D int16 array in Q1.15 (1 sign bit, 15 fractional bits;
    code -32768..32767 maps to value -1.0..0.99997) at TARGET_FS -- the
    fixed-point stimulus fed to the FFT pipeline / RTL testbench.
    """
    if W_DATA != 16:
        raise ValueError("Q1.15 output requires W_DATA == 16")
    x = _decode_audio(path, TARGET_FS)
    codes = _adc_sample(x)
    samples = _codes_to_signed(codes, ADC_BITS, W_DATA)
    samples = _dc_block(samples, W_DATA)
    return np.array(samples, dtype=np.int16)


def mp3_to_numpy_spectrogram(path: str, n_fft: int = SPEC_N_FFT, hop: int = SPEC_HOP):
    """Decode `path` and compute a magnitude (dB) spectrogram for visualization.

    Unlike mp3_to_fxp(), this stays in float and skips the ADC/fixed-point
    emulation -- it's for looking at/listening to the track, not for RTL
    stimulus.

    Returns (spec_db, audio, fs, hop):
      spec_db -- (n_fft//2 + 1, n_frames) magnitude in dB, low freq first
      audio   -- 1-D float32 samples in [-1, 1] at TARGET_FS, for playback
      fs      -- TARGET_FS
      hop     -- hop length in samples between frames
    """
    audio = _decode_audio(path, TARGET_FS)
    window = np.hanning(n_fft)
    n_frames = max(1 + (len(audio) - n_fft) // hop, 0)
    spec = np.empty((n_fft // 2 + 1, n_frames), dtype=np.float64)
    temp = np.empty((n_fft // 2 + 1, n_frames), dtype=np.int16)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + n_fft] * window
        # np.fft.fft is unnormalized -- a full-scale tone peaks around n_fft/4
        # after a Hann window, not 1.0 -- so divide by n_fft/2 to get magnitude
        # back in roughly "fraction of full scale" terms before taking dB
        spec[:, i] = np.abs(fft_sim.numpy_fft(frame)[:n_fft // 2 + 1]) / (n_fft / 2)
        # Test arrary reordering
        # if i == 25:
        #     p = bitrev_table(512) # Compute permutation array for 512 samples
        #     temp = fft_sim.custom_fft(frame)
        #     print(temp)
    spec_db = 20 * np.log10(np.maximum(spec, 1e-6))
    return spec_db, audio, TARGET_FS, hop


def mic_to_spectrogram(n_fft: int = SPEC_N_FFT, hop: int = SPEC_HOP, device: int | None = None):
    """Open the default Windows microphone and yield live magnitude (dB)
    spectrogram frames for real-time visualization.

    Unlike mp3_to_numpy_spectrogram(), there's no file to decode up front --
    this is a generator that blocks on live mic input and yields one frame
    per hop of audio, forever. Each frame is computed from a rolling
    n_fft-sample window (Hann-windowed, like the file-based path) so it
    updates every hop but still has n_fft worth of frequency resolution.

    Yields:
      spec_db -- (n_fft//2 + 1,) magnitude in dB, low freq first

    Stop by breaking out of the consuming loop or closing the generator
    (e.g. `gen.close()`) -- that tears down the input stream cleanly.
    """
    import sounddevice as sd

    window = np.hanning(n_fft)
    buf = np.zeros(n_fft, dtype=np.float32)
    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=TARGET_FS, channels=1, blocksize=hop,
                        dtype="float32", device=device, callback=callback):
        while True:
            block = q.get()
            buf = np.concatenate((buf[len(block):], block))
            # see mp3_to_numpy_spectrogram -- same n_fft/2 magnitude normalization
            spec = np.abs(fft_sim.numpy_fft(buf * window)[:n_fft // 2 + 1]) / (n_fft / 2)
            yield 20 * np.log10(np.maximum(spec, 1e-6))


"""
--------
My Code
--------
"""
def mp3_to_sim_spectrogram(path: str, n_fft: int = 2048, hop: int = 512):
    pass