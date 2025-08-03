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
    plt.savefig(fname, bbox_inches='tight')


def plot_qtransform(ts: TimeSeries, event_time: float, axes=None):
    q_scan = ts.q_transform(
        qrange=[4, 64], frange=[10, 2048], tres=0.002, fres=0.5, whiten=True
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
    analysis_psd = analysis_data.psd()
    ax.plot(psd, label="LIGO-Livingston", color="gwpy:ligo-livingston")
    ax.plot(analysis_psd, label="Analysis Data", color="tab:orange", alpha=0.3)
    N_time = 512
    freqs = np.fft.rfftfreq(N_time, d=1/analysis_data.sample_rate.value)
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
    d = analysis_data.whiten()
    fs = analysis_data.sample_rate.value
    t0 = analysis_data.times.value[-1] - 1  # hardcoded trigger time to be 1 second before the last sample
    n_samps = 512
    t_offset = n_samps / 2 / fs  # 256 samples before and after t0
    d = d.crop(t0 - t_offset, t0 + t_offset)  # 512 samples
    t = d.times.value - t0
    ax.plot(t, d.value, label="Analysis Data", color="tab:orange")
    ax.axvline(0, color='red',  label='Event Time', alpha=0.3,lw=3)
    ax.set_xlabel(f'Time [s] from event')
    ax.set_ylabel('Whitened Strain')
    ax.legend(frameon=False)
