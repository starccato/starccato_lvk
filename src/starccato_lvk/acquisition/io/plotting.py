import matplotlib.pyplot as plt
import numpy as np
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec


def plot(data: TimeSeries, psd: FrequencySeries, event_time: float, fname: str):
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(3, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = [fig.add_subplot(gs[2, i]) for i in range(4)]
    plot_psd_and_analysis_data(psd, data, ax1)
    plot_analysis_timeseries(data, event_time, ax2)
    plot_qtransform(data, event_time, ax3)
    plt.suptitle(f"Event Time: {event_time:.3f} GPS", fontsize=16)
    plt.tight_layout()
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def plot_qtransform(ts: TimeSeries, event_time: float, axes=None):
    nyquist = ts.sample_rate.value / 2
    f_high = min(1024, 0.63 * nyquist)
    if len(ts.value) < 4096:
        for ax in axes:
            ax.axis("off")
        axes[0].set_title("Q-transform unavailable (segment too short)")
        return

    q_scan = ts.q_transform(
        qrange=[4, 64], frange=[10, f_high], tres=0.002, fres=0.5, whiten=True
    )
    # Get time and frequency arrays
    times = q_scan.times.value - event_time  # time relative to event
    freqs = q_scan.frequencies.value
    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    offsets = [0.06, 0.1, 0.25, 0.5]
    for ax, offset in zip(axes, offsets):
        im = ax.imshow(
            q_scan.value.T, aspect='auto', extent=extent,
            origin='lower', cmap='viridis'
        )
        ax.set_xlim(-offset, offset)
        ax.set_yscale('log', base=2)
        ax.set_ylabel('Frequency [Hz]', fontsize=14)
        ax.set_xlabel('Time [s]', fontsize=14)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(f't0 ± {offset} s', fontsize=14)
        ax.grid(False)


def plot_psd_and_analysis_data(psd: FrequencySeries, analysis_data: TimeSeries, ax):
    if len(analysis_data.value) < 4096:
        freqs = np.fft.rfftfreq(len(analysis_data.value), d=1 / analysis_data.sample_rate.value)
        demeaned = analysis_data.value - np.mean(analysis_data.value)
        psd_values = (np.abs(np.fft.rfft(demeaned)) ** 2) * (2.0 / len(analysis_data.value))
        ax.plot(freqs, psd_values, label="Rapid PSD", color="tab:orange")
    else:
        analysis_psd = analysis_data.psd()
        ax.plot(psd, label="LIGO-Livingston", color="gwpy:ligo-livingston")
        ax.plot(analysis_psd, label="Analysis Data", color="tab:orange", alpha=0.3)
        freqs = np.fft.rfftfreq(len(analysis_data.value), d=1 / analysis_data.sample_rate.value)
        psd_interp = np.interp(freqs, psd.frequencies.value, psd.value, left=np.inf, right=np.inf)
        ax.plot(freqs, psd_interp, label="PSD Analysis", color="k", alpha=0.5)
    ax.set_xlim(100, 1600)
    ax.set_ylim(1e-24**2, 1e-21**2)
    ax.set_yscale('log', base=10)
    ax.set_xscale('log', base=10)
    ax.set_ylabel(r"Strain PSD [1/Hz]")
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(frameon=False, bbox_to_anchor=(1., 1.), loc="lower right", ncol=2)


def plot_analysis_timeseries(analysis_data: TimeSeries, event_time: float, ax):
    try:
        d = analysis_data.whiten()
        values = d.value
        times = d.times.value
    except Exception:
        values = analysis_data.value
        values = values - np.mean(values)
        std = np.std(values) or 1.0
        values = values / std
        times = analysis_data.times.value

    fs = analysis_data.sample_rate.value
    t0 = analysis_data.times.value[-1] - 1  # hardcoded trigger time to be 1 second before the last sample
    half_window = 256 / fs  # 512 samples total, centered on t0
    crop_start = max(t0 - half_window, times[0])
    crop_end = min(t0 + half_window, times[-1])

    mask = (times >= crop_start) & (times <= crop_end)
    t = times[mask] - t0
    vals = values[mask]

    ax.plot(t, vals, label="Analysis Data", color="tab:orange")
    ax.axvline(0, color='red',  label='Event Time', alpha=0.3,lw=3)
    ax.set_xlabel(f'Time [s] from event')
    ax.set_ylabel('Whitened Strain')
    ax.legend(frameon=False)
