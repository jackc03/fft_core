from audio_in import (mp3_to_fxp, mp3_to_numpy_spectrogram, mic_to_spectrogram,
                      TARGET_FS, SPEC_N_FFT, SPEC_HOP)


def visualize_video(path: str, n_bars: int = 48):
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import sounddevice as sd

    spec_db, audio, fs, hop = mp3_to_numpy_spectrogram(path)
    n_bins, n_frames = spec_db.shape
    duration = len(audio) / fs
    frame_dt = hop / fs

    # collapse the linear-spaced FFT bins into log-spaced bars (skip DC),
    # since raw FFT bins are far too fine-grained/linear to read as a bar chart
    freqs = np.linspace(0, fs / 2, n_bins)
    edges = np.logspace(np.log10(max(freqs[1], 1.0)), np.log10(fs / 2), n_bars + 1)
    bin_idx = np.searchsorted(freqs, edges)
    bars_db = np.empty((n_bars, n_frames))
    for i in range(n_bars):
        lo, hi = bin_idx[i], max(bin_idx[i + 1], bin_idx[i] + 1)
        bars_db[i] = spec_db[lo:hi].max(axis=0)

    floor, ceil = np.percentile(bars_db, 5), bars_db.max()
    span = max(ceil - floor, 1e-6)

    theta = np.linspace(0, 2 * np.pi, n_bars, endpoint=False)
    r0 = 0.5  # inner radius -- bars spiral outward from this hole

    fig = plt.figure(figsize=(8, 8), facecolor="black")
    ax = fig.add_subplot(111, projection="polar", facecolor="black")
    bars = ax.bar(theta, np.clip((bars_db[:, 0] - floor) / span, 0, 1),
                  width=2 * np.pi / n_bars * 0.9, bottom=r0, color="white")
    ax.set_ylim(0, r0 + 1)
    ax.axis("off")

    sd.play(audio, fs)
    start = time.perf_counter()

    def update(_frame):
        elapsed = time.perf_counter() - start
        col = min(int(elapsed / frame_dt), n_frames - 1)
        heights = np.clip((bars_db[:, col] - floor) / span, 0, 1)
        for rect, h in zip(bars, heights):
            rect.set_height(h)
        if elapsed >= duration:
            ani.event_source.stop()
        return bars.patches

    ani = animation.FuncAnimation(fig, update, interval=30, blit=True,
                                  cache_frame_data=False)
    plt.show()
    sd.stop()


def visualize_image(path: str):
    import matplotlib.pyplot as plt

    spec_db, audio, fs, _hop = mp3_to_numpy_spectrogram(path)
    duration = len(audio) / fs

    _fig, ax = plt.subplots(figsize=(10, 5), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(spec_db, origin="lower", aspect="auto", cmap="gray",
             extent=(0, duration, 0, fs / 2))
    ax.axis("off")
    plt.show()


def visualize_mic(n_bars: int = 48, n_fft: int = SPEC_N_FFT, hop: int = SPEC_HOP):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # same log-spaced bin grouping as visualize_video, just computed once
    # up front since the mic's fs/n_fft are fixed rather than read from a file
    freqs = np.linspace(0, TARGET_FS / 2, n_fft // 2 + 1)
    edges = np.logspace(np.log10(max(freqs[1], 1.0)), np.log10(TARGET_FS / 2), n_bars + 1)
    bin_idx = np.searchsorted(freqs, edges)

    def to_bars(spec_db):
        out = np.empty(n_bars)
        for i in range(n_bars):
            lo, hi = bin_idx[i], max(bin_idx[i + 1], bin_idx[i] + 1)
            out[i] = spec_db[lo:hi].max()
        return out

    theta = np.linspace(0, 2 * np.pi, n_bars, endpoint=False)
    r0 = 0.5  # inner radius -- bars spiral outward from this hole

    fig = plt.figure(figsize=(8, 8), facecolor="black")
    ax = fig.add_subplot(111, projection="polar", facecolor="black")
    bars = ax.bar(theta, np.zeros(n_bars), width=2 * np.pi / n_bars * 0.9,
                  bottom=r0, color="white")
    ax.set_ylim(0, r0 + 1)
    ax.axis("off")

    stream = mic_to_spectrogram(n_fft=n_fft, hop=hop)
    # adaptive ceiling: peak-hold the loudest thing seen recently, decaying
    # back down by a fixed dB/frame when things go quiet -- self-calibrates
    # to whatever this mic's actual range is instead of a guessed constant.
    AGC_DECAY_DB = 0.03  # per frame -- slow drain, ~5.6 dB/s at hop=256 (48000/256 fps)
    AGC_FLOOR_DB = -80.0  # ceiling never decays below this
    AGC_SPAN_DB = 40.0    # dB range below the ceiling mapped to full bar height
    agc = {"ceil": AGC_FLOOR_DB}

    def update(_frame):
        levels = to_bars(next(stream))
        agc["ceil"] = max(agc["ceil"] - AGC_DECAY_DB, levels.max(), AGC_FLOOR_DB)
        floor = agc["ceil"] - AGC_SPAN_DB
        heights = np.clip((levels - floor) / AGC_SPAN_DB, 0, 1)
        for rect, h in zip(bars, heights):
            rect.set_height(h)
        return bars.patches

    ani = animation.FuncAnimation(fig, update, interval=1, blit=True,
                                  cache_frame_data=False)
    plt.show()
    stream.close()


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    do_video = "-visualize_video" in args
    do_image = "-visualize_image" in args
    do_mic = "-visualize_mic" in args
    for flag in ("-visualize_video", "-visualize_image", "-visualize_mic"):
        if flag in args:
            args.remove(flag)
    src = args[0] if args else "test.mp3"

    if do_mic:
        visualize_mic()
    elif do_video:
        visualize_video(src)
    elif do_image:
        visualize_image(src)
    else:
        samples = mp3_to_fxp(src)
        print(f"samples : {samples.shape[0]}")
        print(f"fs      : {TARGET_FS} Hz")
        print(f"range   : [{samples.min()}, {samples.max()}]")
