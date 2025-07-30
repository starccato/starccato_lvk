import matplotlib.pyplot as plt
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec


def plot(data: TimeSeries, psd: FrequencySeries, event_time: float, fname: str, injection=None):
    """
    Plot the spectrogram and analysis data around an event time.

    Parameters:
        data (TimeSeries): Input time series data.
        psd (FrequencySeries): Power spectral density data.
        event_time (float): Time of the event.
        outdir (str): Output directory for saving plots.
    """
    # make  gridSpec layout (row 1, 1plot, row 2, 1 plot, row 3, 4 plots)
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = [fig.add_subplot(gs[2, i]) for i in range(4)]


    # Plot the spectrogram
    plot_psd_and_analysis_data(psd, data, ax1)
    plot_analysis_timeseries(data, event_time, ax2, injection)
    plot_qtransform(data, event_time, ax3)

    plt.suptitle(f"Event Time: {event_time:.3f} GPS", fontsize=16)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')


def plot_qtransform(ts: TimeSeries, event_time: float, axes=None):
    """
    Perform Q-transform and plot spectrogram in Gravity Spy style.

    Parameters:
        data (array-like): Input time series data.
        srate (float): Sample rate of the data.
        tlim (tuple): Time limits for x-axis (default: (1.5, 2.5)).
    """
    q_scan = ts.q_transform(
        qrange=[4, 64], frange=[10, 2048], tres=0.002, fres=0.5, whiten=True
    )
    # Get time and frequency arrays
    times = q_scan.times.value - event_time  # time relative to event
    freqs = q_scan.frequencies.value

    # Define extent for imshow
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
    """
    Plot PSD and analysis data around an event time.

    Parameters:
        psd (FrequencySeries): Power spectral density data.
        analysis_data (TimeSeries): Analysis data segment.
        event_time (float): Time of the event.
        axes (list): List of axes to plot on.
    """
    signal_psd = analysis_data.psd()

    ax.plot(psd, label="LIGO-Livingston", color="gwpy:ligo-livingston")
    ax.plot(signal_psd, label="Analysis Data", color="tab:orange", alpha=0.3)
    ax.set_xlim(16, 1600)
    ax.set_ylim(1e-24**2, 1e-21**2)
    ax.set_yscale('log', base=10)
    ax.set_xscale('log', base=10)
    ax.set_ylabel(r"Strain PSD [1/Hz]")
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(frameon=False, bbox_to_anchor=(1., 1.), loc="lower right", ncol=2)


def plot_analysis_timeseries(analysis_data: TimeSeries, event_time: float, ax, injection=None):
    """
    Plot analysis data segment around an event time.

    Parameters:
        analysis_data (TimeSeries): Analysis data segment.
        event_time (float): Time of the event.
        ax (matplotlib.axes.Axes): Axes to plot on.
    """
    d = analysis_data.whiten()
    t = event_time - analysis_data.times.value
    ax.plot(t, d.value, label="Analysis Data", color="tab:orange")

    # twin axis for the second y-axis
    if injection is not None:
        # Plot the injection signal if provided
        ax2 = ax.twinx()
        ax2.plot(t, injection, label="Injected Signal", color="tab:blue", alpha=0.5)


    # ax.set_epoch(event_time)
    ax.set_xlim(- 0.1,  + 0.1)
    ax.axvline(0, color='red',  label='Event Time', alpha=0.3,lw=3)
    ax.set_xlabel(f'Time [s] from event')
    ax.set_ylabel('Whitened Strain')
    ax.legend(frameon=False)
