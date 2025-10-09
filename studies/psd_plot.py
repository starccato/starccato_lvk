import matplotlib.pyplot as pp
import pycbc.noise
import pycbc.psd
from pycbc.psd import welch
import numpy as np

# Parameters
flow = 30.0
delta_f = 1.0 / 16   # Frequency resolution in Hz
flen = int(2048 / delta_f) + 1
psd_model = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

# Generate 32 seconds of noise at 4096 Hz
fs = 4096  # Sampling frequency in Hz
delta_t = 1.0 / fs
tsamples = int(32 / delta_t)
N = tsamples  # Number of samples
ts = pycbc.noise.noise_from_psd(tsamples, delta_t, psd_model, seed=127)

# Compute PSD estimate from generated noise
segment_length = int(4 / delta_t)  # e.g., 4-second segments for averaging
psd_est = welch(ts, avg_method='median')

# Make sure we compare over same frequency range
freqs = psd_est.sample_frequencies

fft_vals = np.fft.rfft(ts.data)
psd_periodogram = (np.abs(fft_vals) ** 2) * (2.0 / (fs * N))

# Frequency bins for rfft
freqs = np.fft.rfftfreq(N, delta_t)


# Plot
pp.figure(figsize=(8,5))
pp.loglog(freqs, psd_periodogram, label="Periodogram PSD", linestyle=':')
pp.loglog(psd_est.sample_frequencies.numpy(), psd_est, label="Estimated PSD (Welch)")
pp.loglog(psd_model.sample_frequencies.numpy(), psd_model.data, label="aLIGO Zero Det High Power PSD", linestyle='--')
pp.xlim(flow, fs / 2)

pp.xlabel("Frequency (Hz)")
pp.ylabel("PSD [strain^2/Hz]")
pp.legend()
pp.grid(True, which="both", ls=":")
pp.show()